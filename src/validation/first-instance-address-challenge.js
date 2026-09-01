import {
  Color,
  DoubleSide,
  Float32BufferAttribute,
  InstancedBufferGeometry,
  Mesh,
  MeshBasicNodeMaterial,
  NoBlending,
  NoColorSpace,
  RGBAFormat,
  RenderTarget,
  Scene,
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
  STORAGE_TRANSFORM_ADDRESS_MODES,
  createVisibleIdAddressNode,
} from '../materials/storage-transform.js';
import { createSharedGeometryShell } from '../render/indexed-bucket-geometry.js';

const ADDRESS_CHALLENGE_TARGET_WIDTH = 256;
const ADDRESS_CHALLENGE_PIXEL_TRIANGLE = Object.freeze([
  -0.375, -0.375, 0.5,
  0.375, -0.375, 0.5,
  0, 0.375, 0.5,
]);

function exactSequence(left, right) {
  return left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

function bytesOf(values) {
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
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

function requireUint32(value, label, expectedLength = null) {
  if (!(value instanceof Uint32Array)
    || (expectedLength !== null && value.length !== expectedLength)) {
    throw new TypeError(
      `${label} must be a Uint32Array${expectedLength === null ? '' : ` of length ${expectedLength}`}.`,
    );
  }
  return value;
}

function validateScenario(scenario) {
  if (!Number.isSafeInteger(scenario?.objectCount) || scenario.objectCount <= 0
    || !Number.isSafeInteger(scenario?.bucketCount) || scenario.bucketCount <= 0) {
    throw new TypeError('Address challenge requires positive object and bucket counts.');
  }
  if (scenario.objectCount > 0x00ff_ffff) {
    throw new RangeError('First-instance address pixels require RGB24 object IDs.');
  }
  requireUint32(scenario.bucketBases, 'scenario.bucketBases', scenario.bucketCount);
  requireUint32(scenario.bucketCounts, 'scenario.bucketCounts', scenario.bucketCount);
  let cursor = 0;
  for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
    if (scenario.bucketBases[bucket] !== cursor || scenario.bucketCounts[bucket] === 0) {
      throw new Error('Address challenge requires contiguous nonempty bucket capacities.');
    }
    cursor += scenario.bucketCounts[bucket];
  }
  if (cursor !== scenario.objectCount) {
    throw new Error('Address challenge bucket capacities must cover every object exactly once.');
  }
}

export function createFirstInstanceAddressChallengeTargetShape(objectCount) {
  if (!Number.isSafeInteger(objectCount) || objectCount <= 0) {
    throw new RangeError('Address challenge objectCount must be a positive safe integer.');
  }
  const height = Math.ceil(objectCount / ADDRESS_CHALLENGE_TARGET_WIDTH);
  return Object.freeze({
    width: ADDRESS_CHALLENGE_TARGET_WIDTH,
    height,
    pixelCount: height * ADDRESS_CHALLENGE_TARGET_WIDTH,
  });
}

/**
 * Builds lane-local expected RGBA8 bytes from the exact survivor snapshot that
 * is resident for the command buffer under test. Inactive capacity and target
 * padding remain transparent zero by construction.
 */
export function createFirstInstanceAddressChallengeExpected({
  scenario,
  visibleIds,
  activeCounts,
  targetShape = createFirstInstanceAddressChallengeTargetShape(scenario?.objectCount),
}) {
  validateScenario(scenario);
  requireUint32(visibleIds, 'visibleIds');
  requireUint32(activeCounts, 'activeCounts', scenario.bucketCount);
  if (visibleIds.length < scenario.objectCount) {
    throw new RangeError('visibleIds must cover every fixed-slice address.');
  }
  if (targetShape.width !== ADDRESS_CHALLENGE_TARGET_WIDTH
    || targetShape.height !== Math.ceil(scenario.objectCount / ADDRESS_CHALLENGE_TARGET_WIDTH)
    || targetShape.pixelCount !== targetShape.width * targetShape.height) {
    throw new Error('Address challenge target shape differs from the pinned layout.');
  }

  const bytes = new Uint8Array(targetShape.pixelCount * 4);
  let activeAddressCount = 0;
  let paddingAddressCount = 0;
  let outOfRangeIds = 0;
  const activeRows = new Set();
  const nonzeroEncodedChannelPixelCounts = { red: 0, green: 0, blue: 0 };
  for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
    const base = scenario.bucketBases[bucket];
    const capacity = scenario.bucketCounts[bucket];
    const count = activeCounts[bucket];
    if (count > capacity) {
      throw new RangeError(`Address challenge bucket ${bucket} count exceeds its capacity.`);
    }
    activeAddressCount += count;
    paddingAddressCount += capacity - count;
    for (let localIndex = 0; localIndex < count; localIndex += 1) {
      const address = base + localIndex;
      const objectId = visibleIds[address];
      if (objectId >= scenario.objectCount || objectId > 0x00ff_fffe) {
        outOfRangeIds += 1;
        continue;
      }
      const encoded = objectId + 1;
      const byteBase = address * 4;
      bytes[byteBase] = encoded & 0xff;
      bytes[byteBase + 1] = (encoded >>> 8) & 0xff;
      bytes[byteBase + 2] = (encoded >>> 16) & 0xff;
      bytes[byteBase + 3] = 0xff;
      activeRows.add(Math.floor(address / targetShape.width));
      if (bytes[byteBase] !== 0) nonzeroEncodedChannelPixelCounts.red += 1;
      if (bytes[byteBase + 1] !== 0) nonzeroEncodedChannelPixelCounts.green += 1;
      if (bytes[byteBase + 2] !== 0) nonzeroEncodedChannelPixelCounts.blue += 1;
    }
  }
  return {
    bytes,
    activeAddressCount,
    paddingAddressCount,
    targetPaddingPixelCount: targetShape.pixelCount - scenario.objectCount,
    outOfRangeIds,
    coverage: {
      allBucketsActive: [...activeCounts].every((count) => count > 0),
      nonzeroBucketBaseCount: [...scenario.bucketBases].filter((base) => base > 0).length,
      activeRowCount: activeRows.size,
      nonzeroEncodedChannelPixelCounts,
      alphaEncodedPixelCount: activeAddressCount - outOfRangeIds,
    },
  };
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
  if (!pass) throw new Error('Fragment address challenge render-target invariants changed.');
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

/** Build a one-pixel-per-submitted-instance triangle-list diagnostic. */
export function createFragmentAddressChallengeGeometry({
  sourceGeometries,
  bucketBases,
  bucketCounts,
  firstIndexes,
  addressMode = STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE,
  name = 'first-instance-fragment-address-challenge-portable',
}) {
  const bucketCount = sourceGeometries?.length ?? 0;
  const portableAddressing = addressMode === STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE;
  const featureAddressing = addressMode
    === STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE;
  if (bucketCount === 0
    || (!portableAddressing && !featureAddressing)
    || (portableAddressing && (!(bucketBases instanceof Uint32Array)
      || bucketBases.length !== bucketCount))
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
    if (portableAddressing) bases.fill(bucketBases[bucket], start, end);
    for (let index = start; index < end; index += 1) indexes[index] = index;
  }

  const geometry = new InstancedBufferGeometry();
  geometry.name = name;
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
  if (portableAddressing) {
    geometry.setAttribute('bucketBase', new Uint32BufferAttribute(bases, 1));
  }
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
    addressMode,
    bucketBaseAttributePresent: portableAddressing,
  });
  return geometry;
}

function inspectChallengeGeometry({
  scenario,
  sourceGeometries,
  firstIndexes,
  laneIds,
  laneDefinitions,
  challengeGeometries,
  targetShape,
  visibleIdsAttribute,
}) {
  const portableLane = laneIds.find(
    (laneId) => laneDefinitions[laneId].addressMode
      === STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE,
  );
  const featureLane = laneIds.find(
    (laneId) => laneDefinitions[laneId].addressMode
      === STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE,
  );
  if ((laneIds.length === 2 && (!portableLane || !featureLane || portableLane === featureLane))
    || (laneIds.length === 1 && (portableLane === undefined) === (featureLane === undefined))) {
    throw new Error(
      'Address challenge requires one portable or feature lane, or one lane of each when paired.',
    );
  }
  const portable = portableLane === undefined ? null : challengeGeometries[portableLane];
  const feature = featureLane === undefined ? null : challengeGeometries[featureLane];
  const primary = portable ?? feature;
  const position = primary.getAttribute('position');
  const bucketBase = portable?.getAttribute('bucketBase');
  const index = primary.index;
  let positionMismatchCount = 0;
  let bucketBaseMismatchCount = 0;
  let indexMismatchCount = 0;
  let vertexCursor = 0;
  for (let bucket = 0; bucket < sourceGeometries.length; bucket += 1) {
    const count = sourceGeometries[bucket].index.count;
    if (firstIndexes[bucket] !== vertexCursor) {
      throw new Error('Address challenge first-index evidence is not contiguous.');
    }
    for (let local = 0; local < count; local += 1) {
      const vertex = vertexCursor + local;
      if (index.array[vertex] !== vertex) indexMismatchCount += 1;
      if (bucketBase && bucketBase.array[vertex] !== scenario.bucketBases[bucket]) {
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
  const laneOffsetsExact = Object.fromEntries(laneIds.map((laneId) => [
    laneId,
    exactSequence(
      challengeGeometries[laneId].indirectOffset,
      Array.from(laneDefinitions[laneId].indirectOffsets),
    ),
  ]));
  const productionLaneOffsetsExact = Object.fromEntries(laneIds.map((laneId) => [
    laneId,
    exactSequence(
      laneDefinitions[laneId].productionGeometry.indirectOffset,
      Array.from(laneDefinitions[laneId].indirectOffsets),
    ),
  ]));
  const indirectIdentityExact = laneIds.every((laneId) => (
    challengeGeometries[laneId].indirect === laneDefinitions[laneId].indirectAttribute
    && laneDefinitions[laneId].productionGeometry.indirect
      === laneDefinitions[laneId].indirectAttribute
  ));
  const expectedInstanceCount = Math.max(...scenario.bucketCounts);
  const portableAttributesExact = portable === null
    || exactSequence(Object.keys(portable.attributes).sort(), ['bucketBase', 'position']);
  const featureAttributesExact = feature === null
    || exactSequence(Object.keys(feature.attributes).sort(), ['position']);
  const attributesExact = portableAttributesExact && featureAttributesExact;
  const sharedPayloadExact = portable === null || feature === null || (
    feature.index === portable.index
      && feature.getAttribute('position') === position
      && feature.getAttribute('bucketBase') === undefined
  );
  const featureProductionBucketBaseAbsent = featureLane === undefined
    || laneDefinitions[featureLane].productionGeometry.getAttribute('bucketBase') === undefined;
  const featureOnly = laneIds.length === 1 && featureLane !== undefined;
  const featureSurvivorIdentityExact = !featureOnly
    || laneDefinitions[featureLane].visibleIdsAttribute === visibleIdsAttribute;
  const featureCommand = featureOnly ? laneDefinitions[featureLane].commandBuffer : null;
  const featureOffsets = featureOnly
    ? Array.from(laneDefinitions[featureLane].indirectOffsets)
    : [];
  const featureCommandResourceExact = !featureOnly || (
    featureCommand !== null
      && typeof featureCommand === 'object'
      && featureCommand.attributeId === laneDefinitions[featureLane].indirectAttribute.id
      && featureCommand.recordCount === scenario.bucketCount
      && featureCommand.drawCommandCount === scenario.bucketCount
      && featureCommand.firstOffset === featureOffsets[0]
      && exactSequence(featureCommand.allOffsets, featureOffsets)
  );
  const pass = vertexCursor === index.count
    && position.count === index.count
    && (bucketBase === undefined || bucketBase.count === index.count)
    && positionMismatchCount === 0
    && bucketBaseMismatchCount === 0
    && indexMismatchCount === 0
    && attributesExact
    && sharedPayloadExact
    && featureProductionBucketBaseAbsent
    && featureSurvivorIdentityExact
    && featureCommandResourceExact
    && indirectIdentityExact
    && Object.values(laneOffsetsExact).every(Boolean)
    && Object.values(productionLaneOffsetsExact).every(Boolean)
    && laneIds.every(
      (laneId) => challengeGeometries[laneId].instanceCount === expectedInstanceCount,
    );
  return Object.freeze({
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
    ...(laneIds.length === 1 && portableLane !== undefined ? { portableOnly: true } : {}),
    ...(featureOnly ? {
      featureOnly: true,
      addressMode: STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE,
      bucketBaseAttributeAbsent: feature.getAttribute('bucketBase') === undefined,
      productionBucketBaseAttributeAbsent: featureProductionBucketBaseAbsent,
      productionVisibleIdsIdentityExact: featureSurvivorIdentityExact,
      productionCommandResourceExact: featureCommandResourceExact,
    } : {}),
    indirectIdentityExact,
    laneOffsetsExact,
    productionLaneOffsetsExact,
    expectedInstanceCount,
  });
}

function validateFactoryInputs({
  scenario,
  sourceGeometries,
  firstIndexes,
  visibleIdsAttribute,
  laneIds,
  laneDefinitions,
}) {
  validateScenario(scenario);
  if (!Array.isArray(sourceGeometries)
    || sourceGeometries.length !== scenario.bucketCount) {
    throw new RangeError('Address challenge source geometries must match every bucket.');
  }
  requireUint32(firstIndexes, 'firstIndexes', scenario.bucketCount);
  if (!visibleIdsAttribute?.array || visibleIdsAttribute.count < scenario.objectCount) {
    throw new TypeError('Address challenge requires the production visibleIds attribute.');
  }
  if (!Array.isArray(laneIds)
    || ![1, 2].includes(laneIds.length)
    || new Set(laneIds).size !== laneIds.length) {
    throw new RangeError('Address challenge requires one or two distinct lane IDs.');
  }
  if (!laneDefinitions || typeof laneDefinitions !== 'object'
    || !exactSequence(Object.keys(laneDefinitions).sort(), [...laneIds].sort())) {
    throw new Error('Address challenge lane definitions must name exactly the selected lanes.');
  }
  for (const laneId of laneIds) {
    const lane = laneDefinitions?.[laneId];
    if (!lane?.productionGeometry || !lane.indirectAttribute
      || typeof lane.productionGeometry.getAttribute !== 'function'
      || !Array.isArray(Array.from(lane.indirectOffsets ?? []))
      || Array.from(lane.indirectOffsets ?? []).length !== scenario.bucketCount
      || !Object.values(STORAGE_TRANSFORM_ADDRESS_MODES).includes(lane.addressMode)) {
      throw new TypeError(`Address challenge lane ${laneId} is incomplete.`);
    }
    const expectedMode = laneId === 'portable'
      ? STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE
      : laneId === 'feature'
        ? STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE
        : null;
    if (expectedMode === null || lane.addressMode !== expectedMode) {
      throw new Error(`Address challenge lane ${laneId} is mislabeled for its address mode.`);
    }
  }
  if (laneIds.length === 1 && laneIds[0] === 'feature'
    && laneDefinitions.feature.visibleIdsAttribute !== visibleIdsAttribute) {
    throw new Error(
      'Feature-only address challenge must bind the lane production visibleIds attribute.',
    );
  }
}

function freezeStaticTransform(object) {
  object.matrixAutoUpdate = false;
  object.matrixWorldAutoUpdate = false;
}

/**
 * Owns the generalized fragment all-address oracle. Lane definitions bind the
 * diagnostic geometry to each lane's actual production indirect attribute and
 * bind both shaders to the actual production visibleIds attribute.
 */
export function createFirstInstanceAddressChallengeOracle({
  scenario,
  sourceGeometries,
  firstIndexes,
  visibleIdsAttribute,
  laneIds,
  laneDefinitions,
  inspectRenderObject = null,
  inspectVertexInputs = null,
  inspectStorageBindings = null,
  namePrefix = 'first-instance-fragment-address-challenge',
}) {
  validateFactoryInputs({
    scenario,
    sourceGeometries,
    firstIndexes,
    visibleIdsAttribute,
    laneIds,
    laneDefinitions,
  });
  const targetShape = createFirstInstanceAddressChallengeTargetShape(scenario.objectCount);
  const resetExpected = new Uint8Array(targetShape.pixelCount * 4);
  const portableLane = laneIds.find(
    (laneId) => laneDefinitions[laneId].addressMode
      === STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE,
  );
  const featureLane = laneIds.find(
    (laneId) => laneDefinitions[laneId].addressMode
      === STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE,
  );
  if ((laneIds.length === 2 && (!portableLane || !featureLane))
    || (laneIds.length === 1 && (portableLane === undefined) === (featureLane === undefined))) {
    throw new Error(
      'Address challenge requires one portable or feature lane, or one lane of each when paired.',
    );
  }

  const geometries = {};
  let portableGeometry = null;
  if (portableLane !== undefined) {
    portableGeometry = createFragmentAddressChallengeGeometry({
      sourceGeometries,
      bucketBases: scenario.bucketBases,
      bucketCounts: scenario.bucketCounts,
      firstIndexes,
      addressMode: STORAGE_TRANSFORM_ADDRESS_MODES.BUCKET_BASE,
      name: `${namePrefix}-${portableLane}`,
    });
    geometries[portableLane] = portableGeometry;
  }
  if (featureLane !== undefined && portableGeometry !== null) {
    const featureGeometry = createSharedGeometryShell(portableGeometry, {
      omitAttributes: ['bucketBase'],
    });
    featureGeometry.name = `${namePrefix}-${featureLane}`;
    featureGeometry.userData.addressChallenge = portableGeometry.userData.addressChallenge;
    geometries[featureLane] = featureGeometry;
  } else if (featureLane !== undefined) {
    geometries[featureLane] = createFragmentAddressChallengeGeometry({
      sourceGeometries,
      bucketCounts: scenario.bucketCounts,
      firstIndexes,
      addressMode: STORAGE_TRANSFORM_ADDRESS_MODES.INDIRECT_FIRST_INSTANCE,
      name: `${namePrefix}-${featureLane}`,
    });
  }
  for (const laneId of laneIds) {
    geometries[laneId].setIndirect(
      laneDefinitions[laneId].indirectAttribute,
      Array.from(laneDefinitions[laneId].indirectOffsets),
    );
  }
  const geometryEvidence = inspectChallengeGeometry({
    scenario,
    sourceGeometries,
    firstIndexes,
    laneIds,
    laneDefinitions,
    challengeGeometries: geometries,
    targetShape,
    visibleIdsAttribute,
  });
  if (!geometryEvidence.pass) {
    throw new Error('Fragment address challenge geometry failed its exact construction audit.');
  }

  const visibleRead = storage(
    visibleIdsAttribute,
    'uint',
    scenario.objectCount,
  ).toReadOnly();
  const materials = {};
  const scenes = {};
  const meshes = {};
  for (const laneId of laneIds) {
    const addressMode = laneDefinitions[laneId].addressMode;
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
    material.userData.storageTransformAddressMode = addressMode;
    material.vertexNode = Fn(() => {
      const address = createVisibleIdAddressNode({ addressMode })
        .toVar(`${laneId}ChallengeAddress`);
      const objectId = visibleRead.element(address).toVar(`${laneId}ChallengeObjectId`);
      varyingProperty('uint', 'vAddressChallengeObjectId').assign(objectId);
      const pixelX = float(address.mod(uint(targetShape.width))).add(0.5);
      const pixelY = float(address.div(uint(targetShape.width))).add(0.5);
      const pixelCenter = vec2(
        pixelX.mul(2 / targetShape.width).sub(1),
        float(1).sub(pixelY.mul(2 / targetShape.height)),
      );
      const pixelScale = vec2(2 / targetShape.width, 2 / targetShape.height);
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
    const mesh = new Mesh(geometries[laneId], material);
    freezeStaticTransform(mesh);
    mesh.frustumCulled = false;
    const challengeScene = new Scene();
    challengeScene.add(mesh);
    materials[laneId] = material;
    scenes[laneId] = challengeScene;
    meshes[laneId] = mesh;
  }

  const target = new RenderTarget(targetShape.width, targetShape.height, {
    depthBuffer: false,
    stencilBuffer: false,
    samples: 0,
    format: RGBAFormat,
    type: UnsignedByteType,
    colorSpace: NoColorSpace,
  });
  addressChallengeTargetEvidence(target, targetShape);

  async function challengeLane(activeRenderer, activeCamera, laneId, {
    visibleIds,
    activeCounts,
  }) {
    if (!laneIds.includes(laneId)) throw new RangeError(`Unknown address lane ${laneId}.`);
    const expected = createFirstInstanceAddressChallengeExpected({
      scenario,
      visibleIds,
      activeCounts,
      targetShape,
    });
    if (typeof activeRenderer?.readRenderTargetPixelsAsync !== 'function') {
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
    addressChallengeTargetEvidence(target, targetShape);
    const previousTarget = activeRenderer.getRenderTarget();
    const previousCubeFace = activeRenderer.getActiveCubeFace();
    const previousMipmapLevel = activeRenderer.getActiveMipmapLevel();
    const previousClearColor = activeRenderer.getClearColor(new Color());
    const previousClearAlpha = activeRenderer.getClearAlpha();
    const previousAutoClear = activeRenderer.autoClear;

    try {
      activeRenderer.setRenderTarget(target);
      activeRenderer.setClearColor(0x000000, 0);
      activeRenderer.clear(true, false, false);
    } finally {
      activeRenderer.setClearColor(previousClearColor, previousClearAlpha);
      activeRenderer.setRenderTarget(previousTarget, previousCubeFace, previousMipmapLevel);
    }
    const resetActual = await activeRenderer.readRenderTargetPixelsAsync(
      target, 0, 0, targetShape.width, targetShape.height,
    );
    if (!(resetActual instanceof Uint8Array)
      || resetActual.byteLength !== resetExpected.byteLength) {
      throw new Error('Fragment address reset readback must be exact packed Uint8Array data.');
    }
    const observedTargetEvidence = runtimeAddressChallengeTargetEvidence(
      activeRenderer,
      target,
      targetShape,
    );
    const [resetSha256, expectedResetSha256] = await Promise.all([
      sha256Bytes(resetActual),
      sha256Bytes(resetExpected),
    ]);
    const resetPass = exactSequence(resetActual, resetExpected)
      && resetSha256 === expectedResetSha256;

    let challengeShaderSources = null;
    try {
      activeRenderer.setRenderTarget(target);
      activeRenderer.autoClear = false;
      activeRenderer.render(scenes[laneId], activeCamera);
      if (activeRenderer?._objects && activeRenderer?._renderContexts
        && typeof inspectRenderObject === 'function'
        && typeof inspectVertexInputs === 'function'
        && typeof inspectStorageBindings === 'function') {
        const renderObject = inspectRenderObject(
          activeRenderer,
          scenes[laneId],
          activeCamera,
          meshes[laneId],
          target,
        );
        const state = renderObject.getNodeBuilderState();
        challengeShaderSources = {
          vertexShader: state.vertexShader,
          fragmentShader: state.fragmentShader,
          vertexInputs: inspectVertexInputs(activeRenderer, renderObject),
          storageBindings: inspectStorageBindings(
            state.vertexShader,
            renderObject,
            { visibleIdsAttribute },
          ),
        };
      }
    } finally {
      activeRenderer.autoClear = previousAutoClear;
      activeRenderer.setRenderTarget(previousTarget, previousCubeFace, previousMipmapLevel);
    }
    const actual = await activeRenderer.readRenderTargetPixelsAsync(
      target, 0, 0, targetShape.width, targetShape.height,
    );
    if (!(actual instanceof Uint8Array) || actual.byteLength !== expected.bytes.byteLength) {
      throw new Error('Fragment address output readback must be exact packed Uint8Array data.');
    }

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
    const [sha256, expectedSha256, survivorSha256, activeCountsSha256] = await Promise.all([
      sha256Bytes(actual),
      sha256Bytes(expected.bytes),
      sha256Bytes(bytesOf(visibleIds)),
      sha256Bytes(bytesOf(activeCounts)),
    ]);
    let activeMismatchCount = 0;
    let paddingMismatchCount = 0;
    for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
      const base = scenario.bucketBases[bucket];
      const activeEnd = base + activeCounts[bucket];
      const capacityEnd = base + scenario.bucketCounts[bucket];
      for (let address = base; address < activeEnd; address += 1) {
        const byteBase = address * 4;
        if (!actual.subarray(byteBase, byteBase + 4).every(
          (value, index) => value === expected.bytes[byteBase + index],
        )) activeMismatchCount += 1;
      }
      for (let address = activeEnd; address < capacityEnd; address += 1) {
        const byteBase = address * 4;
        if (!actual.subarray(byteBase, byteBase + 4).every(
          (value, index) => value === expected.bytes[byteBase + index],
        )) paddingMismatchCount += 1;
      }
    }
    let targetPaddingMismatchCount = 0;
    for (let pixel = scenario.objectCount; pixel < targetShape.pixelCount; pixel += 1) {
      const byteBase = pixel * 4;
      if (!actual.subarray(byteBase, byteBase + 4).every(
        (value, index) => value === expected.bytes[byteBase + index],
      )) targetPaddingMismatchCount += 1;
    }
    const commandBuffer = laneDefinitions[laneId].commandBuffer
      ? { ...laneDefinitions[laneId].commandBuffer }
      : null;
    const commandSegment = laneDefinitions[laneId].commandSegment
      ? { ...laneDefinitions[laneId].commandSegment }
      : null;
    const exactExpectedBytes = exactSequence(actual, expected.bytes);
    return {
      schemaVersion: 1,
      kind: 'render-target-all-address-challenge',
      pass: resetPass
        && geometryEvidence.pass
        && expected.outOfRangeIds === 0
        && exactExpectedBytes
        && sha256 === expectedSha256
        && activeMismatchCount === 0
        && paddingMismatchCount === 0
        && targetPaddingMismatchCount === 0,
      lane: laneId,
      outputStage: 'fragment',
      addressTransport: 'vertex-address-to-rgba8-pixel',
      encoding: 'rgb24-object-id-plus-one-transparent-zero-background',
      geometryEvidence,
      target: { ...observedTargetEvidence },
      shader,
      commandSegment,
      commandBuffer,
      commandBufferId: commandBuffer?.attributeId ?? null,
      commandByteOffset: commandBuffer?.byteOffset ?? commandSegment?.byteBase ?? null,
      reset: {
        pass: resetPass,
        pixelCount: targetShape.pixelCount,
        addressCount: scenario.objectCount,
        byteLength: resetActual.byteLength,
        sha256: resetSha256,
        expectedSha256: expectedResetSha256,
      },
      pixelCount: targetShape.pixelCount,
      addressCount: scenario.objectCount,
      byteLength: actual.byteLength,
      sha256,
      expectedSha256,
      survivorSha256,
      activeCountsSha256,
      exactExpectedBytes,
      activeAddressCount: expected.activeAddressCount,
      paddingAddressCount: expected.paddingAddressCount,
      targetPaddingPixelCount: expected.targetPaddingPixelCount,
      activeMismatchCount,
      paddingMismatchCount,
      targetPaddingMismatchCount,
      outOfRangeIds: expected.outOfRangeIds,
      coverage: expected.coverage,
    };
  }

  return Object.freeze({
    schemaVersion: 1,
    kind: 'first-instance-all-address-oracle',
    targetShape,
    geometries: Object.freeze({ ...geometries }),
    materials: Object.freeze({ ...materials }),
    geometryEvidence,
    challengeLane,
    dispose() {
      target.dispose();
      for (const laneId of laneIds) scenes[laneId].clear();
    },
  });
}
