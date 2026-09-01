import {
  BundleGroup,
  Color,
  DoubleSide,
  Float32BufferAttribute,
  Group,
  IndirectStorageBufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  MeshBasicNodeMaterial,
  NoBlending,
  NoColorSpace,
  RGBAFormat,
  RenderTarget,
  Scene,
  StorageBufferAttribute,
  Uint32BufferAttribute,
  UnsignedByteType,
} from 'three/webgpu';
import {
  Fn,
  float,
  positionGeometry,
  storage,
  uint,
  varyingProperty,
  vec2,
  vec4,
} from 'three/tsl';
import {
  FIRST_INSTANCE_COMMAND_LANES,
  createFirstInstanceCrossoverPacking,
  validateFirstInstanceCommandLane,
  validateFirstInstanceCrossoverPacking,
} from '../culling/first-instance-crossover.js';
import { FIRST_INSTANCE_CROSSOVER_MODE } from '../benchmark/plan.js';
import {
  STORAGE_TRANSFORM_ADDRESS_MODES,
  createStorageTransformMaterial,
  createVisibleIdAddressNode,
} from '../materials/storage-transform.js';
import {
  createMergedIndexedBucketGeometry,
  createSharedGeometryShell,
} from '../render/indexed-bucket-geometry.js';
import { compareMembership } from '../validation/membership.js';
import {
  createMembershipDigestEvidence,
  sha256CanonicalUint32,
} from '../validation/membership-digests.js';
import { createFirstInstanceShaderEvidence } from '../validation/first-instance-shader-evidence.js';

const [PORTABLE_LANE, FEATURE_LANE] = FIRST_INSTANCE_COMMAND_LANES;

const ADDRESS_MODE_BY_LANE = Object.freeze({
  [PORTABLE_LANE]: STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE,
  [FEATURE_LANE]: STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE,
});

function asUint32(buffer) {
  if (buffer instanceof ArrayBuffer) return new Uint32Array(buffer);
  if (ArrayBuffer.isView(buffer)) {
    return new Uint32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
  }
  throw new TypeError('GPU readback must be an ArrayBuffer or ArrayBuffer view.');
}

function freezeStaticTransform(object) {
  object.matrixAutoUpdate = false;
  object.matrixWorldAutoUpdate = false;
}

function exactSequence(left, right) {
  return left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

function exactUint32(left, right) {
  return left instanceof Uint32Array
    && right instanceof Uint32Array
    && exactSequence(left, right);
}

function snapshotCamera(camera) {
  if (!camera?.matrixWorldInverse?.elements || !camera?.projectionMatrix?.elements) {
    throw new TypeError('First-instance crossover requires a matrix camera.');
  }
  camera.updateMatrixWorld();
  return {
    view: Float32Array.from(camera.matrixWorldInverse.elements),
    projection: Float32Array.from(camera.projectionMatrix.elements),
  };
}

function validateInputs({ scenario, sourceGeometries, renderer, camera }) {
  if (!Array.isArray(sourceGeometries)
    || sourceGeometries.length !== scenario?.bucketCount) {
    throw new RangeError('sourceGeometries length must equal scenario.bucketCount.');
  }
  if (!(scenario?.matrices instanceof Float32Array)
    || scenario.matrices.length !== scenario.objectCount * 16) {
    throw new RangeError('scenario.matrices must contain one Float32 mat4 per object.');
  }
  if (renderer?.hasFeature?.('indirect-first-instance') !== true) {
    throw new Error(
      'The frozen first-instance crossover requires the indirect-first-instance feature.',
    );
  }
  snapshotCamera(camera);
  if (scenario.objectCount > 0x00ff_ffff) {
    throw new RangeError('First-instance address pixels require RGB24 object IDs.');
  }
}

function bytesOf(attribute) {
  return new Uint8Array(
    attribute.array.buffer,
    attribute.array.byteOffset,
    attribute.array.byteLength,
  );
}

function toHex(buffer) {
  return [...new Uint8Array(buffer)]
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

async function sha256Bytes(bytes) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error('Web Crypto SHA-256 is unavailable.');
  return toHex(await subtle.digest('SHA-256', bytes));
}

async function sha256Text(value) {
  return sha256Bytes(new TextEncoder().encode(value));
}

function shaderDiagnosticLines(source) {
  return source.split('\n').filter((line) => (
    /@vertex|@fragment|instance_index|interpolate\(flat|var<storage|atomic/i.test(line)
    || /Challenge|vAddress/.test(line)
  ));
}

export function runtimeVertexInputEvidence(renderer, renderObject) {
  const attributeUtils = renderer?.backend?.attributeUtils;
  if (typeof attributeUtils?.createShaderVertexBuffers !== 'function') {
    throw new Error('Pinned r185 WebGPU vertex-layout inspection is unavailable.');
  }
  if (typeof renderObject?.getNodeBuilderState !== 'function'
    || typeof renderObject?.getAttributes !== 'function') {
    throw new TypeError('Runtime vertex evidence requires a compiled r185 render object.');
  }
  const state = renderObject.getNodeBuilderState();
  const attributes = renderObject.getAttributes();
  if (!Array.isArray(state?.nodeAttributes)
    || !Array.isArray(attributes)
    || state.nodeAttributes.length !== attributes.length) {
    throw new Error('Runtime node attributes and render-object attributes do not align.');
  }
  const layouts = attributeUtils.createShaderVertexBuffers(renderObject);
  if (!Array.isArray(layouts)) {
    throw new Error('Pinned r185 did not expose WebGPU vertex buffer layouts.');
  }
  const layoutsByLocation = new Map();
  for (const layout of layouts) {
    if (!Array.isArray(layout?.attributes)
      || !['vertex', 'instance'].includes(layout.stepMode)) {
      throw new Error('Runtime WebGPU vertex layout metadata is malformed.');
    }
    for (const entry of layout.attributes) {
      if (!Number.isInteger(entry?.shaderLocation)
        || typeof entry.format !== 'string'
        || layoutsByLocation.has(entry.shaderLocation)) {
        throw new Error('Runtime WebGPU vertex locations are malformed or duplicated.');
      }
      layoutsByLocation.set(entry.shaderLocation, {
        format: entry.format,
        stepMode: layout.stepMode,
      });
    }
  }
  if (layoutsByLocation.size !== attributes.length) {
    throw new Error('Runtime WebGPU vertex layouts do not cover every shader attribute.');
  }
  return attributes.map((attribute, shaderLocation) => {
    const nodeAttribute = state.nodeAttributes[shaderLocation];
    const layout = layoutsByLocation.get(shaderLocation);
    if (!layout
      || typeof nodeAttribute?.name !== 'string'
      || !ArrayBuffer.isView(attribute?.array)
      || !Number.isInteger(attribute.id)
      || !Number.isInteger(attribute.count)
      || attribute.count <= 0) {
      throw new Error(`Runtime vertex input ${shaderLocation} is not inspectable.`);
    }
    return {
      name: nodeAttribute.name,
      shaderLocation,
      format: layout.format,
      stepMode: layout.stepMode,
      arrayType: attribute.array.constructor.name,
      itemSize: attribute.itemSize,
      normalized: attribute.normalized === true,
      count: attribute.count,
      resourceId: attribute.id,
    };
  });
}

function parseStorageCoordinates(vertexShader, expectedCount = 2) {
  const declarationPattern = /((?:@(binding|group)\s*\(\s*\d+\s*\)\s*){2})var\s*<\s*storage\s*,\s*(read|read_write)\s*>\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)\s*;/g;
  const records = [];
  for (const match of vertexShader.matchAll(declarationPattern)) {
    const coordinates = {};
    for (const annotation of match[1].matchAll(/@(binding|group)\s*\(\s*(\d+)\s*\)/g)) {
      coordinates[annotation[1]] = Number(annotation[2]);
    }
    const structPattern = new RegExp(
      `struct\\s+${match[5]}\\s*\\{([\\s\\S]*?)\\}\\s*;`,
    );
    const structMatch = structPattern.exec(vertexShader);
    const elementMatch = structMatch
      ? /^\s*value\s*:\s*array\s*<([\s\S]+)>\s*,?\s*$/.exec(structMatch[1])
      : null;
    if (!Number.isInteger(coordinates.group)
      || !Number.isInteger(coordinates.binding)
      || match[3] !== 'read'
      || !elementMatch) {
      throw new Error('Unable to join raw WGSL storage declarations to runtime resources.');
    }
    records.push({
      group: coordinates.group,
      binding: coordinates.binding,
      access: match[3],
      elementType: elementMatch[1].replace(/\s+/g, ''),
    });
  }
  if (records.length !== expectedCount) {
    throw new Error(
      `The inspected vertex shader must expose exactly ${expectedCount} storage declarations.`,
    );
  }
  return records;
}

function normalizeRuntimeStorageAccess(access) {
  if (access === 'readOnly') return 'read';
  if (access === 'readWrite') return 'read_write';
  throw new Error(`Unexpected r185 storage binding access ${String(access)}.`);
}

export function runtimeStorageBindingEvidence(vertexShader, renderObject, {
  matrixAttribute,
  visibleIdsAttribute,
}) {
  if (typeof renderObject?.getBindings !== 'function') {
    throw new TypeError('Runtime storage evidence requires a compiled r185 render object.');
  }
  const resources = new Map();
  if (matrixAttribute) {
    resources.set(matrixAttribute, {
      semantic: 'matrix',
      expectedElementType: 'mat4x4<f32>',
    });
  }
  if (visibleIdsAttribute) {
    resources.set(visibleIdsAttribute, {
      semantic: 'visibleIds',
      expectedElementType: 'u32',
    });
  }
  if (resources.size === 0) {
    throw new Error('Runtime storage evidence requires at least one expected resource.');
  }
  const declarations = new Map(parseStorageCoordinates(vertexShader, resources.size).map((record) => [
    `${record.group}:${record.binding}`,
    record,
  ]));
  const groups = renderObject.getBindings();
  if (!Array.isArray(groups)) {
    throw new Error('Pinned r185 did not expose render-object bind groups.');
  }
  const records = [];
  for (let group = 0; group < groups.length; group += 1) {
    const bindings = groups[group]?.bindings;
    if (!Array.isArray(bindings)) {
      throw new Error(`Pinned r185 bind group ${group} is malformed.`);
    }
    for (let bindingIndex = 0; bindingIndex < bindings.length; bindingIndex += 1) {
      const binding = bindings[bindingIndex];
      if (binding?.isStorageBuffer !== true) continue;
      const resource = resources.get(binding.attribute);
      if (!resource) {
        throw new Error('The timed vertex pipeline exposes an unexpected storage resource.');
      }
      if (binding.nodeUniform?.value !== binding.attribute
        || binding.buffer !== binding.attribute.array) {
        throw new Error(`The ${resource.semantic} runtime binding does not own its attribute.`);
      }
      const declaration = declarations.get(`${group}:${bindingIndex}`);
      if (!declaration) {
        throw new Error(
          `The ${resource.semantic} runtime binding has no WGSL declaration at ${group}:${bindingIndex}.`,
        );
      }
      const access = normalizeRuntimeStorageAccess(binding.access);
      if (access !== declaration.access
        || declaration.elementType !== resource.expectedElementType) {
        throw new Error(`The ${resource.semantic} runtime binding differs from its WGSL declaration.`);
      }
      records.push({
        semantic: resource.semantic,
        group,
        binding: bindingIndex,
        access,
        visibility: typeof binding.getVisibility === 'function'
          ? binding.getVisibility()
          : binding.visibility,
        elementType: declaration.elementType,
        count: binding.attribute.count,
        byteLength: binding.attribute.array.byteLength,
        resourceId: binding.attribute.id,
      });
    }
  }
  if (records.length !== resources.size
    || new Set(records.map((record) => record.semantic)).size !== resources.size) {
    throw new Error('Timed runtime storage resources are missing or duplicated.');
  }
  return records;
}

function resolveTimedRenderObject(
  renderer,
  scene,
  camera,
  mesh,
  explicitRenderTarget = undefined,
) {
  const useFrameBufferTarget = renderer.needsFrameBufferTarget
    && renderer._renderTarget === null;
  const renderTarget = explicitRenderTarget === undefined
    ? (useFrameBufferTarget
      ? renderer._getFrameBufferTarget()
      : (renderer._renderTarget || renderer._outputRenderTarget))
    : explicitRenderTarget;
  const renderList = renderer._renderLists?.get(scene, camera);
  const renderContext = renderer._renderContexts?.get(renderTarget, renderer._mrt);
  const material = scene.overrideMaterial || mesh.material;
  if (!renderList?.lightsNode || !renderContext?.clippingContext
    || typeof renderer._objects?.get !== 'function') {
    throw new Error('Pinned r185 render-object inspection is unavailable after shader capture.');
  }
  const renderObject = renderer._objects.get(
    mesh,
    material,
    scene,
    camera,
    renderList.lightsNode,
    renderContext,
    renderContext.clippingContext,
  );
  if (renderObject?.object !== mesh
    || renderObject.material !== material
    || renderObject.geometry !== mesh.geometry) {
    throw new Error('Pinned r185 resolved a different object for timed shader evidence.');
  }
  return renderObject;
}

async function attributeEvidence(name, attribute) {
  return {
    name,
    id: attribute.id ?? null,
    version: attribute.version,
    arrayType: attribute.array.constructor.name,
    itemSize: attribute.itemSize,
    count: attribute.count,
    normalized: attribute.normalized === true,
    gpuType: attribute.gpuType,
    byteLength: attribute.array.byteLength,
    isInstancedBufferAttribute: attribute.isInstancedBufferAttribute === true,
    meshPerAttribute: attribute.meshPerAttribute ?? null,
    sha256: await sha256Bytes(bytesOf(attribute)),
  };
}

const ADDRESS_CHALLENGE_TARGET_WIDTH = 256;

function createAddressChallengeTargetShape(objectCount) {
  return Object.freeze({
    width: ADDRESS_CHALLENGE_TARGET_WIDTH,
    height: Math.ceil(objectCount / ADDRESS_CHALLENGE_TARGET_WIDTH),
    pixelCount: Math.ceil(objectCount / ADDRESS_CHALLENGE_TARGET_WIDTH)
      * ADDRESS_CHALLENGE_TARGET_WIDTH,
  });
}

function createClearAddressChallengePixels(targetShape) {
  return new Uint8Array(targetShape.pixelCount * 4);
}

function createExpectedAddressChallenge(packing, targetShape) {
  const expected = createClearAddressChallengePixels(targetShape);
  for (let bucket = 0; bucket < packing.bucketCount; bucket += 1) {
    const base = packing.bucketBases[bucket];
    const activeEnd = base + packing.visibleCounts[bucket];
    for (let address = base; address < activeEnd; address += 1) {
      const encoded = packing.visibleIds[address] + 1;
      const pixelBase = address * 4;
      expected[pixelBase] = encoded & 0xff;
      expected[pixelBase + 1] = (encoded >>> 8) & 0xff;
      expected[pixelBase + 2] = (encoded >>> 16) & 0xff;
      expected[pixelBase + 3] = 0xff;
    }
  }
  return expected;
}

function addressChallengeTargetEvidence(target, targetShape) {
  const texture = target.texture;
  const viewportExact = target.viewport.x === 0
    && target.viewport.y === 0
    && target.viewport.z === targetShape.width
    && target.viewport.w === targetShape.height;
  const scissorExact = target.scissor.x === 0
    && target.scissor.y === 0
    && target.scissor.z === targetShape.width
    && target.scissor.w === targetShape.height;
  const pass = target.width === targetShape.width
    && target.height === targetShape.height
    && target.depthBuffer === false
    && target.stencilBuffer === false
    && target.samples === 0
    && target.scissorTest === false
    && viewportExact
    && scissorExact
    && texture.format === RGBAFormat
    && texture.type === UnsignedByteType
    && texture.colorSpace === NoColorSpace
    && texture.flipY === false
    && texture.generateMipmaps === false
    && targetShape.width * 4 % 256 === 0;
  if (!pass) {
    throw new Error('Fragment address challenge render-target invariants changed.');
  }
  return Object.freeze({
    ...targetShape,
    pass,
    configuredFormat: 'RGBAFormat/UnsignedByteType',
    readbackArrayType: 'Uint8Array',
    bytesPerPixel: 4,
    bytesPerRow: targetShape.width * 4,
    rowAlignmentBytes: 256,
    origin: 'top-left',
    sampleLocation: 'integer-plus-half',
    samples: target.samples,
    depthBuffer: target.depthBuffer,
    stencilBuffer: target.stencilBuffer,
    scissorTest: target.scissorTest,
    viewportExact,
    scissorExact,
    colorSpace: 'none',
    flipY: texture.flipY,
    generateMipmaps: texture.generateMipmaps,
  });
}

function runtimeAddressChallengeTargetEvidence(renderer, target, targetShape) {
  const configured = addressChallengeTargetEvidence(target, targetShape);
  const backendRecord = typeof renderer?.backend?.get === 'function'
    ? renderer.backend.get(target.texture)
    : null;
  const backendFormat = backendRecord?.textureDescriptorGPU?.format;
  if (backendFormat !== 'rgba8unorm') {
    throw new Error('Fragment address challenge backend target must be rgba8unorm.');
  }
  return Object.freeze({ ...configured, backendFormat });
}

const ADDRESS_CHALLENGE_PIXEL_TRIANGLE = Object.freeze([
  -0.375, -0.375, 0.5,
  0.375, -0.375, 0.5,
  0, 0.375, 0.5,
]);

/**
 * Builds a triangle-list diagnostic over the production command spans.
 * Each bucket's first triangle covers only the center sample of the addressed
 * target pixel; every later triangle is degenerate. Consequently every
 * submitted instance produces one encoded address pixel while preserving the
 * production indexCount and firstIndex words exactly.
 */
export function createFragmentAddressChallengeGeometry({
  sourceGeometries,
  bucketBases,
  bucketCounts,
  firstIndexes,
}) {
  const bucketCount = sourceGeometries?.length ?? 0;
  if (bucketCount === 0
    || !(bucketBases instanceof Uint32Array)
    || bucketBases.length !== bucketCount
    || !(bucketCounts instanceof Uint32Array)
    || bucketCounts.length !== bucketCount
    || !(firstIndexes instanceof Uint32Array)
    || firstIndexes.length !== bucketCount) {
    throw new RangeError('Fragment address challenge inputs must match every bucket.');
  }

  let totalIndexCount = 0;
  let maxInstanceCount = 0;
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const indexCount = sourceGeometries[bucket]?.index?.count;
    if (!Number.isInteger(indexCount) || indexCount < 3 || indexCount % 3 !== 0) {
      throw new Error(
        `Fragment address challenge bucket ${bucket} must be a nonempty triangle list.`,
      );
    }
    if (firstIndexes[bucket] !== totalIndexCount) {
      throw new Error('Fragment address challenge firstIndex spans are not contiguous.');
    }
    totalIndexCount += indexCount;
    if (!Number.isSafeInteger(totalIndexCount) || totalIndexCount > 0xffff_ffff) {
      throw new RangeError('Fragment address challenge index count exceeds uint32 capacity.');
    }
    maxInstanceCount = Math.max(maxInstanceCount, bucketCounts[bucket]);
  }

  const positions = new Float32Array(totalIndexCount * 3);
  const bases = new Uint32Array(totalIndexCount);
  const indexes = new Uint32Array(totalIndexCount);
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const start = firstIndexes[bucket];
    const end = start + sourceGeometries[bucket].index.count;
    positions.set(ADDRESS_CHALLENGE_PIXEL_TRIANGLE, start * 3);
    bases.fill(bucketBases[bucket], start, end);
    for (let index = start; index < end; index += 1) indexes[index] = index;
  }

  const geometry = new InstancedBufferGeometry();
  geometry.name = 'first-instance-fragment-address-challenge-portable';
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geometry.setAttribute('bucketBase', new Uint32BufferAttribute(bases, 1));
  geometry.setIndex(new Uint32BufferAttribute(indexes, 1));
  geometry.instanceCount = maxInstanceCount;
  geometry.userData.addressChallenge = Object.freeze({
    schemaVersion: 1,
    topology: 'triangle-list',
    pixelLocalCoordinates: true,
    indexCount: totalIndexCount,
    addressedTriangleCount: bucketCount,
    degenerateTriangleCount: totalIndexCount / 3 - bucketCount,
    addressedTrianglesPerSubmittedInstance: 1,
  });
  return geometry;
}

function commandSegmentMap(packing) {
  return Object.fromEntries(FIRST_INSTANCE_COMMAND_LANES.map((lane) => [
    lane,
    {
      index: packing.lanes[lane].commandSegmentIndex,
      recordBase: packing.lanes[lane].commandRecordBase,
      byteBase: packing.lanes[lane].commandByteBase,
    },
  ]));
}

function inspectFragmentAddressChallengeGeometry({
  sourceGeometries,
  scenario,
  packing,
  productionGeometries,
  challengeGeometries,
  indirectAttribute,
  targetShape,
}) {
  const portable = challengeGeometries[PORTABLE_LANE];
  const feature = challengeGeometries[FEATURE_LANE];
  const position = portable.getAttribute('position');
  const bucketBase = portable.getAttribute('bucketBase');
  const index = portable.index;
  let positionMismatchCount = 0;
  let bucketBaseMismatchCount = 0;
  let indexMismatchCount = 0;
  let vertexCursor = 0;
  for (let bucket = 0; bucket < sourceGeometries.length; bucket += 1) {
    const count = sourceGeometries[bucket].index.count;
    for (let local = 0; local < count; local += 1) {
      const vertex = vertexCursor + local;
      if (index.array[vertex] !== vertex) indexMismatchCount += 1;
      if (bucketBase.array[vertex] !== scenario.bucketBases[bucket]) {
        bucketBaseMismatchCount += 1;
      }
      for (let component = 0; component < 3; component += 1) {
        const expected = local < 3
          ? ADDRESS_CHALLENGE_PIXEL_TRIANGLE[local * 3 + component]
          : 0;
        if (!Object.is(position.array[vertex * 3 + component], expected)) {
          positionMismatchCount += 1;
        }
      }
    }
    vertexCursor += count;
  }
  const laneOffsetsExact = Object.fromEntries(FIRST_INSTANCE_COMMAND_LANES.map((lane) => [
    lane,
    exactSequence(
      challengeGeometries[lane].indirectOffset,
      packing.lanes[lane].offsets,
    ),
  ]));
  const indirectIdentityExact = FIRST_INSTANCE_COMMAND_LANES.every((lane) => (
    challengeGeometries[lane].indirect === indirectAttribute
    && productionGeometries[lane].indirect === indirectAttribute
  ));
  const expectedInstanceCount = Math.max(...scenario.bucketCounts);
  const attributesExact = exactSequence(
    Object.keys(portable.attributes).sort(),
    ['bucketBase', 'position'],
  ) && exactSequence(Object.keys(feature.attributes).sort(), ['position']);
  const sharedPayloadExact = feature.index === portable.index
    && feature.getAttribute('position') === position
    && feature.getAttribute('bucketBase') === undefined;
  const pass = vertexCursor === index.count
    && position.count === index.count
    && bucketBase.count === index.count
    && positionMismatchCount === 0
    && bucketBaseMismatchCount === 0
    && indexMismatchCount === 0
    && attributesExact
    && sharedPayloadExact
    && indirectIdentityExact
    && Object.values(laneOffsetsExact).every(Boolean)
    && FIRST_INSTANCE_COMMAND_LANES.every(
      (lane) => challengeGeometries[lane].instanceCount === expectedInstanceCount,
    );
  return {
    schemaVersion: 1,
    kind: 'fragment-address-challenge-geometry-evidence',
    pass,
    topology: 'triangle-list',
    pixelLocalCoordinates: true,
    target: { ...targetShape },
    indexCount: index.count,
    addressedTriangleCount: scenario.bucketCount,
    degenerateTriangleCount: index.count / 3 - scenario.bucketCount,
    addressedTrianglesPerSubmittedInstance: 1,
    positionMismatchCount,
    bucketBaseMismatchCount,
    indexMismatchCount,
    attributesExact,
    sharedPayloadExact,
    indirectIdentityExact,
    laneOffsetsExact,
    expectedInstanceCount,
  };
}

export function buildFirstInstanceCrossoverStrategy({
  scenario,
  sourceGeometries,
  renderer,
  camera,
  laneCommandSegmentOrder = FIRST_INSTANCE_COMMAND_LANES,
}) {
  validateInputs({ scenario, sourceGeometries, renderer, camera });
  const frozenCamera = snapshotCamera(camera);
  const merged = createMergedIndexedBucketGeometry(
    sourceGeometries,
    scenario.bucketBases,
    scenario.bucketCounts,
  );
  const portableGeometry = merged.geometry;
  portableGeometry.name = 'first-instance-crossover-portable-geometry';
  const featureGeometry = createSharedGeometryShell(portableGeometry, {
    omitAttributes: ['bucketBase'],
  });
  featureGeometry.name = 'first-instance-crossover-feature-geometry';
  const geometries = {
    [PORTABLE_LANE]: portableGeometry,
    [FEATURE_LANE]: featureGeometry,
  };
  const packing = createFirstInstanceCrossoverPacking({
    scenario,
    sourceGeometries,
    firstIndexes: merged.firstIndexes,
    laneCommandSegmentOrder,
  });
  const matrixAttribute = new StorageBufferAttribute(scenario.matrices, 16);
  const visibleIdsAttribute = new StorageBufferAttribute(packing.visibleIds, 1);
  const indirectAttribute = new IndirectStorageBufferAttribute(packing.commands, 5);
  const challengeTargetShape = createAddressChallengeTargetShape(scenario.objectCount);
  const expectedAddressChallenge = createExpectedAddressChallenge(
    packing,
    challengeTargetShape,
  );
  const expectedResetAddressChallenge = createClearAddressChallengePixels(
    challengeTargetShape,
  );

  const portableChallengeGeometry = createFragmentAddressChallengeGeometry({
    sourceGeometries,
    bucketBases: scenario.bucketBases,
    bucketCounts: scenario.bucketCounts,
    firstIndexes: packing.firstIndexes,
  });
  const featureChallengeGeometry = createSharedGeometryShell(portableChallengeGeometry, {
    omitAttributes: ['bucketBase'],
  });
  featureChallengeGeometry.name = 'first-instance-fragment-address-challenge-feature';
  featureChallengeGeometry.userData.addressChallenge
    = portableChallengeGeometry.userData.addressChallenge;
  const challengeGeometries = {
    [PORTABLE_LANE]: portableChallengeGeometry,
    [FEATURE_LANE]: featureChallengeGeometry,
  };

  for (const lane of FIRST_INSTANCE_COMMAND_LANES) {
    geometries[lane].setIndirect(
      indirectAttribute,
      Array.from(packing.lanes[lane].offsets),
    );
    challengeGeometries[lane].setIndirect(
      indirectAttribute,
      Array.from(packing.lanes[lane].offsets),
    );
  }
  const challengeGeometryEvidence = inspectFragmentAddressChallengeGeometry({
    sourceGeometries,
    scenario,
    packing,
    productionGeometries: geometries,
    challengeGeometries,
    indirectAttribute,
    targetShape: challengeTargetShape,
  });
  if (!challengeGeometryEvidence.pass) {
    throw new Error('Fragment address challenge geometry failed its exact construction audit.');
  }

  const materials = Object.fromEntries(FIRST_INSTANCE_COMMAND_LANES.map((lane) => [
    lane,
    createStorageTransformMaterial({
      matrixAttribute,
      objectCount: scenario.objectCount,
      visibleIdsAttribute,
      addressMode: ADDRESS_MODE_BY_LANE[lane],
    }),
  ]));
  const meshes = {};
  const bundles = {};
  const bundleRecordCounts = Object.fromEntries(
    FIRST_INSTANCE_COMMAND_LANES.map((lane) => [lane, 0]),
  );
  const root = new Group();
  freezeStaticTransform(root);
  root.name = `${FIRST_INSTANCE_CROSSOVER_MODE}-root`;

  for (const lane of packing.laneCommandSegmentOrder) {
    const mesh = new Mesh(geometries[lane], materials[lane]);
    freezeStaticTransform(mesh);
    mesh.frustumCulled = false;
    mesh.name = `${FIRST_INSTANCE_CROSSOVER_MODE}-${lane}-mesh`;
    mesh.onBeforeRender = (activeRenderer) => {
      if (activeRenderer?._currentRenderBundle !== null
        && activeRenderer?._currentRenderBundle !== undefined) {
        bundleRecordCounts[lane] += 1;
      }
    };
    const bundle = new BundleGroup();
    freezeStaticTransform(bundle);
    bundle.name = `${FIRST_INSTANCE_CROSSOVER_MODE}-${lane}-bundle`;
    bundle.add(mesh);
    meshes[lane] = mesh;
    bundles[lane] = bundle;
    root.add(bundle);
  }

  const visibleRead = storage(
    visibleIdsAttribute,
    'uint',
    scenario.objectCount,
  ).toReadOnly();
  const challengeMaterials = {};
  const challengeScenes = {};
  const challengeMeshes = {};
  for (const lane of FIRST_INSTANCE_COMMAND_LANES) {
    const material = new MeshBasicNodeMaterial();
    material.depthWrite = false;
    material.depthTest = false;
    material.colorWrite = true;
    material.toneMapped = false;
    material.blending = NoBlending;
    material.transparent = false;
    material.premultipliedAlpha = false;
    material.fog = false;
    material.side = DoubleSide;
    material.userData.storageTransformAddressMode = ADDRESS_MODE_BY_LANE[lane];
    material.vertexNode = Fn(() => {
      const address = createVisibleIdAddressNode({
        addressMode: ADDRESS_MODE_BY_LANE[lane],
      }).toVar(`${lane}ChallengeAddress`);
      const objectId = visibleRead.element(address).toVar(`${lane}ChallengeObjectId`);
      varyingProperty('uint', 'vAddressChallengeObjectId').assign(objectId);
      const pixelX = float(address.mod(uint(challengeTargetShape.width))).add(0.5);
      const pixelY = float(address.div(uint(challengeTargetShape.width))).add(0.5);
      const pixelCenter = vec2(
        pixelX.mul(2 / challengeTargetShape.width).sub(1),
        float(1).sub(pixelY.mul(2 / challengeTargetShape.height)),
      );
      const pixelScale = vec2(
        2 / challengeTargetShape.width,
        2 / challengeTargetShape.height,
      );
      return vec4(
        pixelCenter.add(positionGeometry.xy.mul(pixelScale)),
        positionGeometry.z,
        1,
      );
    })();
    material.fragmentNode = Fn(() => {
      const fragmentObjectId = varyingProperty('uint', 'vAddressChallengeObjectId');
      const encoded = fragmentObjectId.add(uint(1)).toVar('encodedAddressChallengeObjectId');
      return vec4(
        float(encoded.bitAnd(uint(0xff))).div(255),
        float(encoded.shiftRight(uint(8)).bitAnd(uint(0xff))).div(255),
        float(encoded.shiftRight(uint(16)).bitAnd(uint(0xff))).div(255),
        1,
      );
    })();
    const mesh = new Mesh(challengeGeometries[lane], material);
    freezeStaticTransform(mesh);
    mesh.frustumCulled = false;
    const challengeScene = new Scene();
    challengeScene.add(mesh);
    challengeMaterials[lane] = material;
    challengeScenes[lane] = challengeScene;
    challengeMeshes[lane] = mesh;
  }
  const challengeTarget = new RenderTarget(
    challengeTargetShape.width,
    challengeTargetShape.height,
    {
      depthBuffer: false,
      stencilBuffer: false,
      samples: 0,
      format: RGBAFormat,
      type: UnsignedByteType,
      colorSpace: NoColorSpace,
    },
  );
  addressChallengeTargetEvidence(challengeTarget, challengeTargetShape);

  let activeLane = PORTABLE_LANE;
  let laneSelectionSerial = 0;
  let bundlesPrimed = false;
  let shaderEvidence = null;

  function setActiveLane(lane) {
    validateFirstInstanceCommandLane(lane);
    activeLane = lane;
    for (const candidate of FIRST_INSTANCE_COMMAND_LANES) {
      bundles[candidate].visible = candidate === lane;
    }
    laneSelectionSerial += 1;
    return packing.lanes[lane].commandByteBase;
  }
  setActiveLane(activeLane);

  async function primeBundles({ scene, render }) {
    if (bundlesPrimed) throw new Error('First-instance bundles were already primed.');
    if (typeof render !== 'function') {
      throw new TypeError('primeBundles requires a render callback.');
    }
    const previousLane = activeLane;
    try {
      for (const lane of packing.laneCommandSegmentOrder) {
        setActiveLane(lane);
        await renderer.compileAsync(scene, camera);
        await render(lane);
        if (bundleRecordCounts[lane] !== 1) {
          throw new Error(`The ${lane} bundle did not record exactly once.`);
        }
      }
      bundlesPrimed = true;
    } finally {
      setActiveLane(previousLane);
    }
  }

  async function collectShaderSources(scene) {
    if (!bundlesPrimed) throw new Error('Shader capture requires primed bundles.');
    const previousLane = activeLane;
    const lanes = {};
    try {
      for (const lane of FIRST_INSTANCE_COMMAND_LANES) {
        setActiveLane(lane);
        const sources = await renderer.debug.getShaderAsync(scene, camera, meshes[lane]);
        const renderObject = resolveTimedRenderObject(renderer, scene, camera, meshes[lane]);
        const state = renderObject.getNodeBuilderState();
        if (state.vertexShader !== sources.vertexShader
          || state.fragmentShader !== sources.fragmentShader) {
          throw new Error(`${lane} debug shader sources differ from the timed render object.`);
        }
        lanes[lane] = {
          vertexShader: sources.vertexShader,
          fragmentShader: sources.fragmentShader,
          vertexInputs: runtimeVertexInputEvidence(renderer, renderObject),
          storageBindings: runtimeStorageBindingEvidence(sources.vertexShader, renderObject, {
            matrixAttribute,
            visibleIdsAttribute,
          }),
        };
      }
    } finally {
      setActiveLane(previousLane);
    }
    const validated = await createFirstInstanceShaderEvidence({
      portable: lanes[PORTABLE_LANE],
      feature: lanes[FEATURE_LANE],
    });
    shaderEvidence = {
      ...validated,
      captureApi: 'renderer.debug.getShaderAsync(scene, camera, mesh)',
      rawSources: {
        [PORTABLE_LANE]: {
          vertexShader: lanes[PORTABLE_LANE].vertexShader,
          fragmentShader: lanes[PORTABLE_LANE].fragmentShader,
        },
        [FEATURE_LANE]: {
          vertexShader: lanes[FEATURE_LANE].vertexShader,
          fragmentShader: lanes[FEATURE_LANE].fragmentShader,
        },
      },
    };
    return shaderEvidence;
  }

  async function challengeAddressing(activeRenderer, activeCamera, lane) {
    validateFirstInstanceCommandLane(lane);
    if (typeof activeRenderer.readRenderTargetPixelsAsync !== 'function') {
      throw new Error('Fragment address challenge requires render-target readback.');
    }
    for (const method of [
      'getRenderTarget',
      'getActiveCubeFace',
      'getActiveMipmapLevel',
      'setRenderTarget',
      'getClearColor',
      'getClearAlpha',
      'setClearColor',
      'clear',
      'render',
    ]) {
      if (typeof activeRenderer[method] !== 'function') {
        throw new Error(`Fragment address challenge requires renderer.${method}().`);
      }
    }
    if (typeof activeRenderer.autoClear !== 'boolean') {
      throw new Error('Fragment address challenge requires a boolean renderer.autoClear.');
    }
    addressChallengeTargetEvidence(challengeTarget, challengeTargetShape);
    const previousTarget = activeRenderer.getRenderTarget();
    const previousCubeFace = activeRenderer.getActiveCubeFace();
    const previousMipmapLevel = activeRenderer.getActiveMipmapLevel();
    const previousClearColor = activeRenderer.getClearColor(new Color());
    const previousClearAlpha = activeRenderer.getClearAlpha();
    const previousAutoClear = activeRenderer.autoClear;

    // Prove the target was cleared before rendering. Otherwise a repeated
    // same-lane challenge could pass with stale pixels from a prior render.
    try {
      activeRenderer.setRenderTarget(challengeTarget);
      activeRenderer.setClearColor(0x000000, 0);
      activeRenderer.clear(true, false, false);
    } finally {
      activeRenderer.setClearColor(previousClearColor, previousClearAlpha);
      activeRenderer.setRenderTarget(
        previousTarget,
        previousCubeFace,
        previousMipmapLevel,
      );
    }
    const resetReadback = await activeRenderer.readRenderTargetPixelsAsync(
      challengeTarget,
      0,
      0,
      challengeTargetShape.width,
      challengeTargetShape.height,
    );
    if (!(resetReadback instanceof Uint8Array)
      || resetReadback.byteLength !== expectedResetAddressChallenge.byteLength) {
      throw new Error('Fragment address reset readback must be exact packed Uint8Array data.');
    }
    const resetActual = resetReadback;
    const observedTargetEvidence = runtimeAddressChallengeTargetEvidence(
      activeRenderer,
      challengeTarget,
      challengeTargetShape,
    );
    const [resetSha256, expectedResetSha256] = await Promise.all([
      sha256Bytes(resetActual),
      sha256Bytes(expectedResetAddressChallenge),
    ]);
    const resetPass = exactSequence(resetActual, expectedResetAddressChallenge)
      && resetSha256 === expectedResetSha256;
    let challengeShaderSources = null;
    try {
      activeRenderer.setRenderTarget(challengeTarget);
      activeRenderer.autoClear = false;
      activeRenderer.render(challengeScenes[lane], activeCamera);
      if (activeRenderer?._objects && activeRenderer?._renderContexts) {
        const renderObject = resolveTimedRenderObject(
          activeRenderer,
          challengeScenes[lane],
          activeCamera,
          challengeMeshes[lane],
          challengeTarget,
        );
        const state = renderObject.getNodeBuilderState();
        challengeShaderSources = {
          vertexShader: state.vertexShader,
          fragmentShader: state.fragmentShader,
          vertexInputs: runtimeVertexInputEvidence(activeRenderer, renderObject),
          storageBindings: runtimeStorageBindingEvidence(
            state.vertexShader,
            renderObject,
            { visibleIdsAttribute },
          ),
        };
      }
    } finally {
      activeRenderer.autoClear = previousAutoClear;
      activeRenderer.setRenderTarget(
        previousTarget,
        previousCubeFace,
        previousMipmapLevel,
      );
    }
    const challengeReadback = await activeRenderer.readRenderTargetPixelsAsync(
      challengeTarget,
      0,
      0,
      challengeTargetShape.width,
      challengeTargetShape.height,
    );
    if (!(challengeReadback instanceof Uint8Array)
      || challengeReadback.byteLength !== expectedAddressChallenge.byteLength) {
      throw new Error('Fragment address output readback must be exact packed Uint8Array data.');
    }
    const actual = challengeReadback;
    const shader = challengeShaderSources === null ? null : {
      vertexSha256: await sha256Text(challengeShaderSources.vertexShader),
      fragmentSha256: await sha256Text(challengeShaderSources.fragmentShader),
      vertexLines: shaderDiagnosticLines(challengeShaderSources.vertexShader),
      fragmentLines: shaderDiagnosticLines(challengeShaderSources.fragmentShader),
      rawSources: {
        vertexShader: challengeShaderSources.vertexShader,
        fragmentShader: challengeShaderSources.fragmentShader,
      },
      vertexInputs: challengeShaderSources.vertexInputs,
      storageBindings: challengeShaderSources.storageBindings,
    };
    const [sha256, expectedSha256] = await Promise.all([
      sha256Bytes(actual),
      sha256Bytes(expectedAddressChallenge),
    ]);
    let activeAddressCount = 0;
    let paddingAddressCount = 0;
    let activeMismatchCount = 0;
    let paddingMismatchCount = 0;
    const activeRows = new Set();
    const nonzeroEncodedChannelPixelCounts = { red: 0, green: 0, blue: 0 };
    for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
      const base = scenario.bucketBases[bucket];
      const activeEnd = base + scenario.visibleCounts[bucket];
      const capacityEnd = base + scenario.bucketCounts[bucket];
      for (let address = base; address < activeEnd; address += 1) {
        activeAddressCount += 1;
        activeRows.add(Math.floor(address / challengeTargetShape.width));
        const byteBase = address * 4;
        if (expectedAddressChallenge[byteBase] !== 0) {
          nonzeroEncodedChannelPixelCounts.red += 1;
        }
        if (expectedAddressChallenge[byteBase + 1] !== 0) {
          nonzeroEncodedChannelPixelCounts.green += 1;
        }
        if (expectedAddressChallenge[byteBase + 2] !== 0) {
          nonzeroEncodedChannelPixelCounts.blue += 1;
        }
        if (!actual.subarray(byteBase, byteBase + 4).every(
          (value, index) => value === expectedAddressChallenge[byteBase + index],
        )) activeMismatchCount += 1;
      }
      for (let address = activeEnd; address < capacityEnd; address += 1) {
        paddingAddressCount += 1;
        const byteBase = address * 4;
        if (!actual.subarray(byteBase, byteBase + 4).every(
          (value, index) => value === expectedAddressChallenge[byteBase + index],
        )) paddingMismatchCount += 1;
      }
    }
    let targetPaddingMismatchCount = 0;
    for (let pixel = scenario.objectCount; pixel < challengeTargetShape.pixelCount; pixel += 1) {
      const byteBase = pixel * 4;
      if (!actual.subarray(byteBase, byteBase + 4).every(
        (value, index) => value === expectedAddressChallenge[byteBase + index],
      )) targetPaddingMismatchCount += 1;
    }
    return {
      schemaVersion: 1,
      kind: 'render-target-all-address-challenge',
      pass: resetPass
        && challengeGeometryEvidence.pass
        && exactSequence(actual, expectedAddressChallenge)
        && sha256 === expectedSha256
        && activeMismatchCount === 0
        && paddingMismatchCount === 0
        && targetPaddingMismatchCount === 0,
      lane,
      outputStage: 'fragment',
      addressTransport: 'vertex-address-to-rgba8-pixel',
      encoding: 'rgb24-object-id-plus-one-transparent-zero-background',
      target: { ...observedTargetEvidence },
      shader,
      commandSegment: { ...commandSegmentMap(packing)[lane] },
      reset: {
        pass: resetPass,
        pixelCount: challengeTargetShape.pixelCount,
        addressCount: scenario.objectCount,
        byteLength: resetActual.byteLength,
        sha256: resetSha256,
        expectedSha256: expectedResetSha256,
      },
      pixelCount: challengeTargetShape.pixelCount,
      addressCount: scenario.objectCount,
      byteLength: actual.byteLength,
      sha256,
      expectedSha256,
      exactExpectedBytes: exactSequence(actual, expectedAddressChallenge),
      activeAddressCount,
      paddingAddressCount,
      targetPaddingPixelCount: challengeTargetShape.pixelCount - scenario.objectCount,
      activeMismatchCount,
      paddingMismatchCount,
      targetPaddingMismatchCount,
      coverage: {
        allBucketsActive: [...scenario.visibleCounts].every((count) => count > 0),
        nonzeroBucketBaseCount: [...scenario.bucketBases].filter((base) => base > 0).length,
        activeRowCount: activeRows.size,
        nonzeroEncodedChannelPixelCounts,
        alphaEncodedPixelCount: activeAddressCount,
      },
    };
  }

  async function createGeometryEvidence() {
    const portableNames = Object.keys(portableGeometry.attributes).sort();
    const featureNames = Object.keys(featureGeometry.attributes).sort();
    const expectedFeatureNames = portableNames.filter((name) => name !== 'bucketBase');
    const commonAttributes = {};
    for (const name of expectedFeatureNames) {
      const portable = portableGeometry.getAttribute(name);
      const feature = featureGeometry.getAttribute(name);
      commonAttributes[name] = {
        sameObject: portable === feature,
        portable: await attributeEvidence(name, portable),
        feature: await attributeEvidence(name, feature),
      };
    }
    const bucketBase = portableGeometry.getAttribute('bucketBase');
    const index = {
      sameObject: portableGeometry.index === featureGeometry.index,
      portable: await attributeEvidence('index', portableGeometry.index),
      feature: await attributeEvidence('index', featureGeometry.index),
    };
    const bucketBaseEvidence = await attributeEvidence('bucketBase', bucketBase);
    let vertexCursor = 0;
    let bucketBaseMismatchCount = 0;
    for (let bucket = 0; bucket < sourceGeometries.length; bucket += 1) {
      const vertexCount = sourceGeometries[bucket].getAttribute('position').count;
      for (let vertex = 0; vertex < vertexCount; vertex += 1) {
        if (bucketBase.array[vertexCursor + vertex] !== scenario.bucketBases[bucket]) {
          bucketBaseMismatchCount += 1;
        }
      }
      vertexCursor += vertexCount;
    }
    const pass = featureNames.length === expectedFeatureNames.length
      && featureNames.every((name, indexValue) => name === expectedFeatureNames[indexValue])
      && featureGeometry.getAttribute('bucketBase') === undefined
      && index.sameObject
      && Object.values(commonAttributes).every((record) => record.sameObject)
      && bucketBaseEvidence.arrayType === 'Uint32Array'
      && bucketBaseEvidence.itemSize === 1
      && bucketBaseEvidence.isInstancedBufferAttribute === false
      && bucketBaseMismatchCount === 0
      && [...portableNames, ...featureNames].every((name) => (
        portableGeometry.getAttribute(name)?.isInstancedBufferAttribute !== true
        && featureGeometry.getAttribute(name)?.isInstancedBufferAttribute !== true
      ));
    return {
      schemaVersion: 1,
      kind: 'shared-first-instance-geometry-evidence',
      pass,
      portableAttributeNames: portableNames,
      featureAttributeNames: featureNames,
      sharedIndex: index,
      commonAttributes,
      bucketBase: bucketBaseEvidence,
      bucketBaseMismatchCount,
      noInstanceSteppedAttributes: [...portableNames, ...featureNames].every((name) => (
        portableGeometry.getAttribute(name)?.isInstancedBufferAttribute !== true
        && featureGeometry.getAttribute(name)?.isInstancedBufferAttribute !== true
      )),
    };
  }

  function lifecycleDiagnostics() {
    const commonAttributeNames = Object.keys(featureGeometry.attributes).sort();
    const bundleStaticFlags = Object.fromEntries(FIRST_INSTANCE_COMMAND_LANES.map(
      (lane) => [lane, bundles[lane].static === true],
    ));
    return {
      kind: 'first-instance-crossover-static-resource-lifecycle',
      bundlesPrimed,
      bundleStaticFlags,
      allBundlesStatic: Object.values(bundleStaticFlags).every(Boolean),
      bundleCount: new Set(Object.values(bundles)).size,
      meshCount: new Set(Object.values(meshes)).size,
      activeRenderObjectCount: FIRST_INSTANCE_COMMAND_LANES.filter(
        (lane) => bundles[lane].visible,
      ).length,
      activeLane,
      laneSelectionSerial,
      laneCommandSegmentOrder: [...packing.laneCommandSegmentOrder],
      commandSegments: commandSegmentMap(packing),
      rootUuid: root.uuid,
      rootVersion: root.version ?? null,
      bundleUuids: Object.fromEntries(FIRST_INSTANCE_COMMAND_LANES.map(
        (lane) => [lane, bundles[lane].uuid],
      )),
      bundleVersions: Object.fromEntries(FIRST_INSTANCE_COMMAND_LANES.map(
        (lane) => [lane, bundles[lane].version ?? null],
      )),
      bundleRecordCounts: { ...bundleRecordCounts },
      meshUuids: Object.fromEntries(FIRST_INSTANCE_COMMAND_LANES.map(
        (lane) => [lane, meshes[lane].uuid],
      )),
      geometryUuids: Object.fromEntries(FIRST_INSTANCE_COMMAND_LANES.map(
        (lane) => [lane, geometries[lane].uuid],
      )),
      materialUuids: Object.fromEntries(FIRST_INSTANCE_COMMAND_LANES.map(
        (lane) => [lane, materials[lane].uuid],
      )),
      materialVersions: Object.fromEntries(FIRST_INSTANCE_COMMAND_LANES.map(
        (lane) => [lane, materials[lane].version],
      )),
      commonAttributeIds: Object.fromEntries(commonAttributeNames.map((name) => [
        name,
        geometries[PORTABLE_LANE].getAttribute(name).id ?? null,
      ])),
      commonAttributeVersions: Object.fromEntries(commonAttributeNames.map((name) => [
        name,
        geometries[PORTABLE_LANE].getAttribute(name).version,
      ])),
      indexAttributeId: portableGeometry.index.id ?? null,
      indexAttributeVersion: portableGeometry.index.version,
      bucketBaseAttributeId: portableGeometry.getAttribute('bucketBase').id ?? null,
      bucketBaseAttributeVersion: portableGeometry.getAttribute('bucketBase').version,
      matrixAttributeId: matrixAttribute.id,
      matrixAttributeVersion: matrixAttribute.version,
      visibleIdsAttributeId: visibleIdsAttribute.id,
      visibleIdsAttributeVersion: visibleIdsAttribute.version,
      indirectAttributeId: indirectAttribute.id,
      indirectAttributeVersion: indirectAttribute.version,
      shaderEvidence: shaderEvidence && {
        pass: shaderEvidence.pass,
        portableVertexSha256: shaderEvidence.portable?.raw.vertex.sha256 ?? null,
        featureVertexSha256: shaderEvidence.feature?.raw.vertex.sha256 ?? null,
        normalizedVertexSha256: shaderEvidence.portable?.normalizedVertex.sha256 ?? null,
        fragmentSha256: shaderEvidence.portable?.raw.fragment.sha256 ?? null,
        storageBindings: Object.fromEntries(FIRST_INSTANCE_COMMAND_LANES.map(
          (lane) => [lane, shaderEvidence[lane]?.storageBindings ?? null],
        )),
      },
      configuredComputeDispatches: 0,
      configuredComputeSubmissions: 0,
    };
  }

  function diagnostics() {
    const lifecycle = lifecycleDiagnostics();
    return {
      ...lifecycle,
      kind: 'frozen-first-instance-addressing-crossover',
      objectCount: scenario.objectCount,
      bucketCount: scenario.bucketCount,
      expectedVisibleCount: scenario.expectedVisibleCount,
      visibleIdsCount: packing.visibleIds.length,
      commandRecordCount: packing.commands.length / 5,
      activeRenderObjectCount: FIRST_INSTANCE_COMMAND_LANES.filter(
        (lane) => bundles[lane].visible,
      ).length,
      geometryIdentityCount: new Set(Object.values(geometries)).size,
      materialIdentityCount: new Set(Object.values(materials)).size,
      commonIndexIdentityCount: new Set(
        Object.values(geometries).map((geometry) => geometry.index),
      ).size,
    };
  }

  return {
    id: FIRST_INSTANCE_CROSSOVER_MODE,
    root,
    geometries: [
      ...Object.values(geometries),
      ...Object.values(challengeGeometries),
    ],
    materials: [
      ...Object.values(materials),
      ...Object.values(challengeMaterials),
    ],
    storageAttributes: [
      matrixAttribute,
      visibleIdsAttribute,
      indirectAttribute,
    ],
    computeNodes: [],
    usesCompute: false,
    configuredDrawCommands: scenario.bucketCount,
    configuredRenderObjects: 1,
    configuredComputeDispatches: 0,
    configuredComputeSubmissions: 0,
    configuredSubmittedInstances: scenario.expectedVisibleCount,
    laneIds: [...FIRST_INSTANCE_COMMAND_LANES],
    laneCommandSegmentOrder: [...packing.laneCommandSegmentOrder],
    commandSegments: commandSegmentMap(packing),
    packing,
    addressChallengeGeometryEvidence: challengeGeometryEvidence,
    get activeLane() {
      return activeLane;
    },
    get laneSelectionSerial() {
      return laneSelectionSerial;
    },
    get shaderEvidence() {
      return shaderEvidence;
    },
    setActiveLane,
    primeBundles,
    collectShaderSources,
    challengeAddressing,
    getActiveParityView() {
      return {
        geometry: geometries[activeLane],
        material: materials[activeLane],
        parityResources: {
          matrixAttribute,
          visibleIdsAttribute,
          visibleIdsCount: scenario.objectCount,
          visibleIdOffset: 0,
          objectCount: scenario.objectCount,
          addressMode: ADDRESS_MODE_BY_LANE[activeLane],
        },
      };
    },
    diagnostics,
    lifecycleDiagnostics,
    update(nextCamera) {
      const current = snapshotCamera(nextCamera);
      if (!exactSequence(current.view, frozenCamera.view)
        || !exactSequence(current.projection, frozenCamera.projection)) {
        throw new Error('First-instance crossover camera changed after packing.');
      }
    },
    submitCompute() {
      throw new Error('First-instance crossover has no timed compute submission.');
    },
    async validate(activeRenderer, expectedIds) {
      if (!(expectedIds instanceof Uint32Array)
        || !exactUint32(expectedIds, scenario.expectedVisibleIds)) {
        throw new TypeError(
          'expectedIds must exactly match the frozen scenario expectedVisibleIds.',
        );
      }
      const [visibleBuffer, commandBuffer] = await Promise.all([
        activeRenderer.getArrayBufferAsync(visibleIdsAttribute),
        activeRenderer.getArrayBufferAsync(indirectAttribute),
      ]);
      const observedPacking = {
        ...packing,
        visibleIds: asUint32(visibleBuffer),
        commands: asUint32(commandBuffer),
      };
      const [frozenPacking, membershipDigests, geometry] = await Promise.all([
        validateFirstInstanceCrossoverPacking({
          packing: observedPacking,
          scenario,
          sourceGeometries,
          firstIndexes: merged.firstIndexes,
          expectedLaneCommandSegmentOrder: packing.laneCommandSegmentOrder,
        }),
        createMembershipDigestEvidence({
          expectedIds,
          actualIds: observedPacking.visibleIds,
          actualCounts: scenario.visibleCounts,
          objectBuckets: scenario.objectBuckets,
          bucketBases: scenario.bucketBases,
          capacities: scenario.bucketCounts,
        }),
        createGeometryEvidence(),
      ]);
      // Both diagnostics intentionally reuse one render target, so clear,
      // render, and readback must remain serialized by lane.
      const portableChallenge = await challengeAddressing(
        activeRenderer,
        camera,
        PORTABLE_LANE,
      );
      const featureChallenge = await challengeAddressing(
        activeRenderer,
        camera,
        FEATURE_LANE,
      );
      const membership = compareMembership({
        expectedIds,
        actualIds: observedPacking.visibleIds,
        actualCounts: scenario.visibleCounts,
        objectBuckets: scenario.objectBuckets,
        bucketBases: scenario.bucketBases,
        capacities: scenario.bucketCounts,
        objectCount: scenario.objectCount,
      });
      const lifecycle = lifecycleDiagnostics();
      const challengesByteIdentical = portableChallenge.sha256 === featureChallenge.sha256;
      const lifecyclePass = lifecycle.bundlesPrimed === true
        && lifecycle.allBundlesStatic === true
        && lifecycle.bundleCount === 2
        && lifecycle.meshCount === 2
        && lifecycle.activeRenderObjectCount === 1
        && FIRST_INSTANCE_COMMAND_LANES.every(
          (lane) => lifecycle.bundleRecordCounts[lane] === 1,
        )
        && lifecycle.configuredComputeDispatches === 0
        && lifecycle.configuredComputeSubmissions === 0
        && lifecycle.shaderEvidence?.pass === true;
      return {
        schemaVersion: 1,
        kind: 'first-instance-crossover-exact-paired-snapshots',
        pass: frozenPacking.pass
          && membership.pass
          && membershipDigests.pass
          && geometry.pass
          && portableChallenge.pass
          && featureChallenge.pass
          && challengesByteIdentical
          && lifecyclePass,
        expectedIdsMatchScenario: true,
        frozenPacking,
        membership,
        membershipDigests,
        geometry,
        addressChallenges: {
          pass: portableChallenge.pass
            && featureChallenge.pass
            && challengesByteIdentical
            && challengeGeometryEvidence.pass,
          byteIdentical: challengesByteIdentical,
          geometry: challengeGeometryEvidence,
          lanes: {
            [PORTABLE_LANE]: portableChallenge,
            [FEATURE_LANE]: featureChallenge,
          },
        },
        shaderEvidence,
        lifecycle,
      };
    },
    dispose() {
      challengeTarget.dispose();
      for (const lane of FIRST_INSTANCE_COMMAND_LANES) {
        challengeScenes[lane].clear();
      }
    },
  };
}
