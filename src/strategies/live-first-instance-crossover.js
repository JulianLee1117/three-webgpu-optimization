import { Group } from 'three/webgpu';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_LANES,
  FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
} from '../benchmark/plan.js';
import { registerComputeTimestampGroup } from '../benchmark/gpu-timestamps.js';
import { updateFrustumPlaneState } from '../culling/frustum-planes.js';
import {
  createFirstInstanceShaderEvidence,
} from '../validation/first-instance-shader-evidence.js';
import {
  createFirstInstanceAddressChallengeOracle,
} from '../validation/first-instance-address-challenge.js';
import {
  normalizeLiveIndirectCommandComputeShader,
} from '../validation/live-compute-shader-normalization.js';
import {
  runtimeStorageBindingEvidence,
  runtimeVertexInputEvidence,
} from './first-instance-crossover.js';
import {
  FIXED_SLICE_ADDRESS_MODE_BY_LANE,
  createFixedSliceLane,
  createFixedSliceSharedResources,
  readFixedSliceLaneSnapshot,
  validateFixedSliceLane,
  validateFixedSliceLanePhysicalOrder,
  validateFixedSliceLaneSnapshot,
} from './fixed-slice.js';

const [PORTABLE_LANE, FEATURE_LANE] = FIRST_INSTANCE_LIVE_CROSSOVER_LANES;
let nextComputeTimestampContextId = 1;

function allocateComputeTimestampContextId() {
  const contextId = nextComputeTimestampContextId;
  if (!Number.isSafeInteger(contextId) || contextId <= 0) {
    throw new Error('Live compute timestamp context-ID space was exhausted.');
  }
  nextComputeTimestampContextId += 1;
  return contextId;
}
const SERIALIZED_VALIDATION_ORDER = Object.freeze([PORTABLE_LANE, FEATURE_LANE]);
const SHADER_OBSERVATION_REQUEST_KEYS = Object.freeze([
  'schemaVersion',
  'kind',
  'origin',
  'runId',
  'trialId',
  'planIndex',
  'repetitionIndex',
  'phase',
  'role',
  'captureOrdinal',
  'challengeNonce',
]);
const SHADER_OBSERVATION_PHASES = Object.freeze([
  'preflight',
  'timing-start',
  'postflight',
  'interactive',
]);
const SHADER_OBSERVATION_ROLES = Object.freeze(['render-parity', 'main-validation']);
const SHA256_HEX_PATTERN = /^[0-9a-f]{64}$/;
const RENDERER_MEMORY_FIELDS = Object.freeze([
  'attributes',
  'attributesSize',
  'geometries',
  'indexAttributes',
  'indexAttributesSize',
  'indirectStorageAttributes',
  'indirectStorageAttributesSize',
  'programs',
  'programsSize',
  'storageAttributes',
  'storageAttributesSize',
  'uniformBuffers',
  'uniformBuffersSize',
]);

function freezeStaticTransform(object) {
  object.matrixAutoUpdate = false;
  object.matrixWorldAutoUpdate = false;
}

function exactSequence(left, right) {
  return left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

function validateShaderObservationRequest(request, expectedOrdinal) {
  if (!request || typeof request !== 'object' || Array.isArray(request)) {
    throw new TypeError('Fresh shader observation requires an explicit request object.');
  }
  if (!exactSequence(Object.keys(request).sort(), [...SHADER_OBSERVATION_REQUEST_KEYS].sort())) {
    throw new TypeError('Fresh shader observation request fields differ from the frozen schema.');
  }
  if (request.schemaVersion !== 1
    || request.kind !== 'live-first-instance-shader-observation-challenge'
    || !['node-runner', 'page-interactive'].includes(request.origin)
    || !SHADER_OBSERVATION_PHASES.includes(request.phase)
    || !SHADER_OBSERVATION_ROLES.includes(request.role)
    || request.captureOrdinal !== expectedOrdinal
    || !SHA256_HEX_PATTERN.test(request.challengeNonce ?? '')) {
    throw new TypeError('Fresh shader observation request violates the frozen protocol.');
  }
  if (request.origin === 'node-runner') {
    if (typeof request.runId !== 'string' || request.runId.length === 0
      || typeof request.trialId !== 'string' || request.trialId.length === 0
      || !Number.isSafeInteger(request.planIndex) || request.planIndex < 0
      || !Number.isSafeInteger(request.repetitionIndex) || request.repetitionIndex < 0
      || request.phase === 'interactive') {
      throw new TypeError('Node-runner shader observation request identity is invalid.');
    }
  } else if (request.runId !== null
    || request.trialId !== null
    || request.planIndex !== null
    || request.repetitionIndex !== null
    || request.phase !== 'interactive') {
    throw new TypeError('Interactive shader observation request identity is invalid.');
  }
  return structuredClone(request);
}

function exactFloat32Bits(values, frozen) {
  return values.length === frozen.length
    && frozen.every((value, index) => Object.is(Math.fround(values[index]), value));
}

function snapshotCamera(camera) {
  if (!camera?.matrixWorldInverse?.elements || !camera?.projectionMatrix?.elements) {
    throw new TypeError('Live first-instance crossover requires a matrix camera.');
  }
  return Object.freeze({
    cameraUuid: camera.uuid ?? null,
    view: Float32Array.from(camera.matrixWorldInverse.elements),
    projection: Float32Array.from(camera.projectionMatrix.elements),
    reversedDepth: camera.reversedDepth === true,
    near: camera.near,
    far: camera.far,
    fov: camera.fov ?? null,
    aspect: camera.aspect ?? null,
    zoom: camera.zoom ?? null,
  });
}

function assertCameraStable(camera, frozen) {
  if ((camera.uuid ?? null) !== frozen.cameraUuid
    || (camera.reversedDepth === true) !== frozen.reversedDepth
    || !Object.is(camera.near, frozen.near)
    || !Object.is(camera.far, frozen.far)
    || !Object.is(camera.fov ?? null, frozen.fov)
    || !Object.is(camera.aspect ?? null, frozen.aspect)
    || !Object.is(camera.zoom ?? null, frozen.zoom)
    || !exactFloat32Bits(camera.matrixWorldInverse.elements, frozen.view)
    || !exactFloat32Bits(camera.projectionMatrix.elements, frozen.projection)) {
    throw new Error('Live first-instance crossover camera parameters or matrix bits changed.');
  }
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
    throw new Error('Pinned r185 resolved a different timed render object.');
  }
  return renderObject;
}

function resolveTimedBundleRenderResource(renderer, camera, laneRoot, mesh, renderTarget) {
  if (typeof renderer?._renderContexts?.get !== 'function'
    || typeof renderer?._bundles?.get !== 'function'
    || typeof renderer?.backend?.get !== 'function') {
    throw new Error('Pinned r185 render-bundle inspection is unavailable.');
  }
  const renderContext = renderer._renderContexts.get(renderTarget, renderer._mrt);
  const renderBundle = renderer._bundles.get(laneRoot, camera, renderContext);
  const renderBundleData = renderer.backend.get(renderBundle);
  const renderObjects = renderBundleData?.renderObjects;
  if (!Array.isArray(renderObjects)
    || renderObjects.length !== 1
    || renderObjects[0]?.object !== mesh
    || renderObjects[0]?.bundle !== laneRoot
    || renderBundleData.bundleGPU === undefined
    || renderBundleData.version !== laneRoot.version) {
    throw new Error('Pinned r185 timed render bundle is missing or does not own exactly one mesh.');
  }
  return {
    renderObject: renderObjects[0],
    renderBundle,
    renderBundleData,
    bundleGPU: renderBundleData.bundleGPU,
  };
}

function pinnedRendererInspectionState(renderer) {
  const pipelineCache = renderer?._pipelines?.caches;
  const programs = renderer?._pipelines?.programs;
  const memory = renderer?.info?.memory;
  if (!(pipelineCache instanceof Map)
    || !(programs?.vertex instanceof Map)
    || !(programs?.fragment instanceof Map)
    || !(programs?.compute instanceof Map)
    || !memory) {
    throw new Error(
      'Pinned r185 pipeline/program/memory inspection is unavailable for fresh shader evidence.',
    );
  }
  const memoryCounters = Object.fromEntries(RENDERER_MEMORY_FIELDS.map((field) => {
    const value = memory[field];
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new Error(`Pinned r185 renderer.info.memory.${field} is not inspectable.`);
    }
    return [field, value];
  }));
  return {
    totalPipelineCacheEntries: pipelineCache.size,
    programEntries: {
      vertex: programs.vertex.size,
      fragment: programs.fragment.size,
      compute: programs.compute.size,
    },
    memory: memoryCounters,
  };
}

function storageSemanticMap(shared, lane) {
  return new Map([
    [shared.attributes.matrix, 'matrix'],
    [shared.attributes.bounds, 'bounds'],
    [shared.attributes.objectBucket, 'objectBucket'],
    [shared.attributes.bucketBase, 'bucketBase'],
    [shared.attributes.bucketCapacity, 'bucketCapacity'],
    [shared.attributes.cullOrder, 'cullOrder'],
    [shared.attributes.visibleIds, 'visibleIds'],
    [shared.attributes.overflow, 'overflow'],
    [lane.indirectAttribute, 'indirectCommands'],
  ]);
}

function typedArrayValues(value) {
  if (!ArrayBuffer.isView(value)) return null;
  return Array.from(value);
}

function toHex(bytes) {
  return Array.from(new Uint8Array(bytes), (value) => value.toString(16).padStart(2, '0')).join('');
}

async function sha256Text(value) {
  if (!globalThis.crypto?.subtle || typeof globalThis.crypto.subtle.digest !== 'function') {
    throw new Error('Compute shader evidence requires Web Crypto SHA-256 support.');
  }
  return toHex(await globalThis.crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(value),
  ));
}

function normalizedDimensionVector(value, label) {
  if (!Array.isArray(value)
    || value.length < 1
    || value.length > 3
    || value.some((entry) => !Number.isInteger(entry) || entry <= 0)) {
    throw new RangeError(`${label} must contain one to three positive integers.`);
  }
  return [value[0], value[1] ?? 1, value[2] ?? 1];
}

export function createLiveComputeDispatchCommitment(
  computeNode,
  runtimeDispatchSize = null,
) {
  if (computeNode?.isComputeNode !== true
    || !Number.isInteger(computeNode.count)
    || computeNode.count <= 0) {
    throw new TypeError('Live fixed-slice compute nodes require a positive integer count.');
  }
  const workgroupSize = normalizedDimensionVector(
    computeNode.workgroupSize,
    'computeNode.workgroupSize',
  );
  const invocationsPerWorkgroup = workgroupSize.reduce((product, value) => product * value, 1);
  const derivedDispatchSize = [
    Math.ceil(computeNode.count / invocationsPerWorkgroup),
    1,
    1,
  ];
  const observedDispatchSize = runtimeDispatchSize === null
    ? null
    : normalizedDimensionVector(runtimeDispatchSize, 'runtimeDispatchSize');
  return {
    count: computeNode.count,
    workgroupSize,
    invocationsPerWorkgroup,
    derivedDispatchSize,
    runtimeDispatchSize: observedDispatchSize,
    runtimeMatchesDerived: observedDispatchSize === null
      ? null
      : exactSequence(observedDispatchSize, derivedDispatchSize),
  };
}

function captureComputeBindings(renderer, computeNode, semantics) {
  if (typeof renderer?._nodes?.getForCompute !== 'function') {
    throw new Error('Pinned r185 compute node-builder inspection is unavailable.');
  }
  const state = renderer._nodes.getForCompute(computeNode);
  if (typeof state?.computeShader !== 'string' || !Array.isArray(state.bindings)) {
    throw new Error('Pinned r185 compute shader state is malformed.');
  }
  const bindings = [];
  for (let group = 0; group < state.bindings.length; group += 1) {
    const bindGroup = state.bindings[group];
    if (!Array.isArray(bindGroup?.bindings)) {
      throw new Error('Pinned r185 compute bind group is malformed.');
    }
    for (let binding = 0; binding < bindGroup.bindings.length; binding += 1) {
      const entry = bindGroup.bindings[binding];
      const attribute = entry?.isStorageBuffer ? entry.attribute : null;
      bindings.push({
        group,
        binding,
        kind: entry?.isStorageBuffer
          ? 'storage-buffer'
          : entry?.isUniformBuffer
            ? 'uniform-buffer'
            : entry?.constructor?.name ?? 'unknown',
        semantic: attribute ? (semantics.get(attribute) ?? 'unknown-storage') : 'uniforms',
        access: entry?.access ?? null,
        visibility: entry?.visibility ?? null,
        byteLength: entry?.byteLength ?? entry?.buffer?.byteLength ?? null,
        resourceId: attribute?.id ?? null,
        attributeType: attribute?.constructor?.name ?? null,
        arrayType: attribute?.array?.constructor?.name ?? null,
        itemSize: attribute?.itemSize ?? null,
        count: attribute?.count ?? null,
        uniformValues: entry?.isUniformBuffer ? typedArrayValues(entry.buffer) : null,
      });
    }
  }
  const runtimeDispatchSize = typeof renderer.backend?.get === 'function'
    ? renderer.backend.get(computeNode)?.dispatchSize ?? null
    : null;
  return {
    computeShader: state.computeShader,
    bindings,
    execution: createLiveComputeDispatchCommitment(computeNode, runtimeDispatchSize),
  };
}

function bindingShape(binding) {
  const { resourceId, uniformValues, ...shape } = binding;
  return shape;
}

function compareComputeBindings(portable, feature) {
  const shapeEqual = JSON.stringify(portable.map(bindingShape))
    === JSON.stringify(feature.map(bindingShape));
  const uniformValuesEqual = JSON.stringify(portable.map((binding) => binding.uniformValues))
    === JSON.stringify(feature.map((binding) => binding.uniformValues));
  const resourceComparisons = [];
  if (portable.length === feature.length) {
    for (let index = 0; index < portable.length; index += 1) {
      const left = portable[index];
      const right = feature[index];
      if (left.kind !== 'storage-buffer' || right.kind !== 'storage-buffer') continue;
      const shouldShare = left.semantic !== 'indirectCommands';
      resourceComparisons.push({
        semantic: left.semantic,
        group: left.group,
        binding: left.binding,
        portableResourceId: left.resourceId,
        featureResourceId: right.resourceId,
        shouldShare,
        pass: shouldShare
          ? left.resourceId === right.resourceId
          : left.resourceId !== right.resourceId,
      });
    }
  }
  return {
    pass: shapeEqual
      && uniformValuesEqual
      && resourceComparisons.length > 0
      && resourceComparisons.every((entry) => entry.pass),
    shapeEqual,
    uniformValuesEqual,
    resourceComparisons,
  };
}

function workgroupDeclaration(source) {
  return source.match(/@workgroup_size\s*\([^)]*\)/)?.[0] ?? null;
}

function firstInstanceFieldAudit(source) {
  const declarationCount = source.match(/\bfirstInstance\s*:/g)?.length ?? 0;
  const executableAccessCount = source.match(/\.\s*firstInstance\b/g)?.length ?? 0;
  return {
    pass: executableAccessCount === 0,
    declarationCount,
    executableAccessCount,
  };
}

async function collectComputeShaderEvidence(renderer, lanes) {
  const captures = {};
  for (const laneId of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
    const lane = lanes[laneId];
    const semantics = storageSemanticMap(lanes.shared, lane);
    captures[laneId] = {
      reset: captureComputeBindings(renderer, lane.computeNodes[0], semantics),
      cull: captureComputeBindings(renderer, lane.computeNodes[1], semantics),
    };
  }
  const phases = {};
  const fixedExpectations = {
    reset: {
      count: 32,
      workgroupSize: [64, 1, 1],
      derivedDispatchSize: [1, 1, 1],
    },
    cull: {
      count: 65_536,
      workgroupSize: [64, 1, 1],
      derivedDispatchSize: [1_024, 1, 1],
    },
  };
  for (const phase of ['reset', 'cull']) {
    const portable = captures[PORTABLE_LANE][phase];
    const feature = captures[FEATURE_LANE][phase];
    const normalized = {
      [PORTABLE_LANE]: normalizeLiveIndirectCommandComputeShader(
        portable.computeShader,
        portable.bindings,
      ),
      [FEATURE_LANE]: normalizeLiveIndirectCommandComputeShader(
        feature.computeShader,
        feature.bindings,
      ),
    };
    const rawWgslEqual = portable.computeShader === feature.computeShader;
    const normalizedWgslEqual = normalized[PORTABLE_LANE].normalizedShader
      === normalized[FEATURE_LANE].normalizedShader;
    const laneLocalIdentifiersDistinct = normalized[PORTABLE_LANE].audit.generatedVariableIdentifier
      !== normalized[FEATURE_LANE].audit.generatedVariableIdentifier;
    const rawDifferenceRestrictedToLaneLocalIndirectBinding = !rawWgslEqual
      && normalizedWgslEqual
      && laneLocalIdentifiersDistinct;
    const sourceDigests = Object.fromEntries(await Promise.all(
      FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(async (laneId) => [laneId, {
        rawSha256: await sha256Text(captures[laneId][phase].computeShader),
        normalizedSha256: await sha256Text(normalized[laneId].normalizedShader),
      }]),
    ));
    const bindingComparison = compareComputeBindings(portable.bindings, feature.bindings);
    const portableWordFour = firstInstanceFieldAudit(portable.computeShader);
    const featureWordFour = firstInstanceFieldAudit(feature.computeShader);
    const executionEqual = JSON.stringify({
      count: portable.execution.count,
      workgroupSize: portable.execution.workgroupSize,
      derivedDispatchSize: portable.execution.derivedDispatchSize,
    }) === JSON.stringify({
      count: feature.execution.count,
      workgroupSize: feature.execution.workgroupSize,
      derivedDispatchSize: feature.execution.derivedDispatchSize,
    });
    const fixedExpectation = fixedExpectations[phase];
    const fixedWorkloadExact = [portable, feature].every((capture) => (
      capture.execution.count === fixedExpectation.count
      && exactSequence(capture.execution.workgroupSize, fixedExpectation.workgroupSize)
      && exactSequence(
        capture.execution.derivedDispatchSize,
        fixedExpectation.derivedDispatchSize,
      )
      && capture.execution.runtimeMatchesDerived === true
    ));
    phases[phase] = {
      pass: rawDifferenceRestrictedToLaneLocalIndirectBinding
        && workgroupDeclaration(portable.computeShader)
          === workgroupDeclaration(feature.computeShader)
        && bindingComparison.pass
        && portableWordFour.pass
        && featureWordFour.pass
        && executionEqual
        && fixedWorkloadExact,
      rawWgslEqual,
      normalizedWgslEqual,
      rawDifferenceRestrictedToLaneLocalIndirectBinding,
      sourceDigests,
      normalization: {
        [PORTABLE_LANE]: normalized[PORTABLE_LANE].audit,
        [FEATURE_LANE]: normalized[FEATURE_LANE].audit,
      },
      workgroupDeclaration: workgroupDeclaration(portable.computeShader),
      executionEqual,
      fixedWorkloadExact,
      fixedExpectation,
      bindings: bindingComparison,
      wordFour: {
        [PORTABLE_LANE]: portableWordFour,
        [FEATURE_LANE]: featureWordFour,
      },
      lanes: {
        [PORTABLE_LANE]: portable,
        [FEATURE_LANE]: feature,
      },
    };
  }
  const dispatchDimensionsEqual = ['reset', 'cull'].every((phase) => (
    phases[phase].executionEqual
  ));
  const maxStorageBindingCount = Math.max(...FIRST_INSTANCE_LIVE_CROSSOVER_LANES.flatMap(
    (laneId) => ['reset', 'cull'].map((phase) => (
      captures[laneId][phase].bindings.filter((binding) => binding.kind === 'storage-buffer').length
    )),
  ));
  return {
    schemaVersion: 1,
    kind: 'live-first-instance-compute-shader-evidence',
    pass: phases.reset.pass && phases.cull.pass && dispatchDimensionsEqual,
    dispatchDimensionsEqual,
    dispatchDimensions: {
      reset: [...phases.reset.lanes[PORTABLE_LANE].execution.derivedDispatchSize],
      cull: [...phases.cull.lanes[PORTABLE_LANE].execution.derivedDispatchSize],
    },
    fixedWorkloadExact: phases.reset.fixedWorkloadExact && phases.cull.fixedWorkloadExact,
    maxStorageBindingCount,
    phases,
  };
}

function commandCoreEqual(portable, feature, bucketCount) {
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const base = bucket * 5;
    for (let word = 0; word < 4; word += 1) {
      if (portable[base + word] !== feature[base + word]) return false;
    }
  }
  return true;
}

function canonicalMembershipEqual(portable, feature) {
  const left = portable.membershipDigests;
  const right = feature.membershipDigests;
  return left.actual.count === right.actual.count
    && left.actual.sha256 === right.actual.sha256
    && left.perBucket.length === right.perBucket.length
    && left.perBucket.every((bucket, index) => (
      bucket.actual.count === right.perBucket[index].actual.count
      && bucket.actual.sha256 === right.perBucket[index].actual.sha256
    ));
}

function geometryEvidence(lanes) {
  const portable = lanes[PORTABLE_LANE].geometry;
  const feature = lanes[FEATURE_LANE].geometry;
  const portableNames = Object.keys(portable.attributes).sort();
  const featureNames = Object.keys(feature.attributes).sort();
  const commonNames = portableNames.filter((name) => name !== 'bucketBase');
  const commonAttributesShared = Object.fromEntries(commonNames.map((name) => [
    name,
    portable.getAttribute(name) === feature.getAttribute(name),
  ]));
  const bucketBase = portable.getAttribute('bucketBase');
  const noInstanceSteppedAttributes = [...new Set([...portableNames, ...featureNames])].every(
    (name) => portable.getAttribute(name)?.isInstancedBufferAttribute !== true
      && feature.getAttribute(name)?.isInstancedBufferAttribute !== true,
  );
  const pass = bucketBase !== undefined
    && feature.getAttribute('bucketBase') === undefined
    && exactSequence(featureNames, commonNames)
    && portable.index === feature.index
    && Object.values(commonAttributesShared).every(Boolean)
    && noInstanceSteppedAttributes;
  return {
    pass,
    portableAttributeNames: portableNames,
    featureAttributeNames: featureNames,
    portableOnlyAttributeNames: ['bucketBase'],
    sharedIndex: portable.index === feature.index,
    commonAttributesShared,
    noInstanceSteppedAttributes,
  };
}

export function buildFirstInstanceLiveCrossoverStrategy({
  scenario,
  sourceGeometries,
  renderer,
  camera,
  lanePhysicalOrder = FIRST_INSTANCE_LIVE_CROSSOVER_LANES,
}) {
  if (renderer?.hasFeature?.('indirect-first-instance') !== true) {
    throw new Error(
      'The live first-instance crossover requires the indirect-first-instance feature.',
    );
  }
  validateFixedSliceLanePhysicalOrder(lanePhysicalOrder);
  const shared = createFixedSliceSharedResources(
    { scenario, sourceGeometries },
    { addressModes: Object.values(FIXED_SLICE_ADDRESS_MODE_BY_LANE) },
  );
  updateFrustumPlaneState(shared.planeState, camera, renderer);
  const frozenCamera = snapshotCamera(camera);
  const lanes = { shared };
  const laneConstructionOrder = [];
  const root = new Group();
  freezeStaticTransform(root);
  root.name = `${FIRST_INSTANCE_LIVE_CROSSOVER_MODE}-root`;
  for (const laneId of lanePhysicalOrder) {
    const lane = createFixedSliceLane(shared, {
      lane: laneId,
      id: `${FIRST_INSTANCE_LIVE_CROSSOVER_MODE}-${laneId}`,
    });
    lane.root.name = `${FIRST_INSTANCE_LIVE_CROSSOVER_MODE}-${laneId}-bundle`;
    lane.meshes[0].name = `${FIRST_INSTANCE_LIVE_CROSSOVER_MODE}-${laneId}-mesh`;
    lanes[laneId] = lane;
    root.add(lane.root);
    laneConstructionOrder.push(laneId);
  }

  const captureCommandBufferCommitments = () => Object.freeze(Object.fromEntries(
    FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map((laneId) => [
      laneId,
      Object.freeze(lanes[laneId].commandBufferCommitment()),
    ]),
  ));
  const commandBufferCommitments = captureCommandBufferCommitments();
  const addressChallengeOracle = createFirstInstanceAddressChallengeOracle({
    scenario,
    sourceGeometries,
    firstIndexes: shared.firstIndexes,
    visibleIdsAttribute: shared.attributes.visibleIds,
    laneIds: FIRST_INSTANCE_LIVE_CROSSOVER_LANES,
    laneDefinitions: Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
      (laneId) => [laneId, {
        addressMode: lanes[laneId].addressMode,
        productionGeometry: lanes[laneId].geometry,
        indirectAttribute: lanes[laneId].indirectAttribute,
        indirectOffsets: lanes[laneId].commandLayout.offsets,
        commandBuffer: commandBufferCommitments[laneId],
        commandSegment: { index: 0, recordBase: 0, byteBase: 0 },
      }],
    )),
    inspectRenderObject: resolveTimedRenderObject,
    inspectVertexInputs: runtimeVertexInputEvidence,
    inspectStorageBindings: runtimeStorageBindingEvidence,
    namePrefix: 'live-first-instance-fragment-address-challenge',
  });
  const firstComputeUseOrder = [];
  const renderPipelinePrimeOrder = [];
  const preparedSnapshots = {};
  let activeLane = PORTABLE_LANE;
  let laneSelectionSerial = 0;
  let computeCallSerial = 0;
  let prepareSerial = 0;
  let lanesPrimed = false;
  let residentPreparedLane = null;
  let shaderEvidence = null;
  let shaderScene = null;
  let shaderRenderTarget = null;
  let shaderObservationSerial = 0;
  let inspectionIdentitySerial = 0;
  const inspectionIdentities = new WeakMap();
  const timedRenderResources = {};
  const timedComputeResources = new Map();
  const computeTimestampContextIds = Object.freeze(Object.fromEntries(
    FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
      (laneId) => [laneId, allocateComputeTimestampContextId()],
    ),
  ));
  const computeTimestampRegistrations = {};

  function inspectionIdentity(value) {
    if ((typeof value !== 'object' && typeof value !== 'function') || value === null) {
      throw new TypeError('Pinned r185 inspection identities require objects.');
    }
    let identity = inspectionIdentities.get(value);
    if (identity === undefined) {
      inspectionIdentitySerial += 1;
      identity = inspectionIdentitySerial;
      inspectionIdentities.set(value, identity);
    }
    return identity;
  }

  function applyLaneVisibility(laneId) {
    for (const candidate of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
      lanes[candidate].root.visible = candidate === laneId;
    }
  }
  applyLaneVisibility(activeLane);

  function setActiveLane(laneId) {
    validateFixedSliceLane(laneId);
    activeLane = laneId;
    applyLaneVisibility(laneId);
    laneSelectionSerial += 1;
    return commandBufferCommitments[laneId].attributeId;
  }

  function update(nextCamera, nextRenderer = renderer) {
    if (nextCamera !== camera || nextRenderer !== renderer) {
      throw new Error('Live first-instance crossover camera and renderer identities are fixed.');
    }
    updateFrustumPlaneState(shared.planeState, nextCamera, nextRenderer);
    assertCameraStable(nextCamera, frozenCamera);
  }

  function submitCompute(activeRenderer = renderer) {
    if (activeRenderer !== renderer) {
      throw new Error('Live first-instance crossover renderer identity changed.');
    }
    const lane = lanes[activeLane];
    const registration = computeTimestampRegistrations[activeLane];
    if (!registration) {
      throw new Error(`Live ${activeLane} compute timestamp group was not registered.`);
    }
    if (!firstComputeUseOrder.includes(activeLane)) firstComputeUseOrder.push(activeLane);
    activeRenderer.compute(lane.computeNodes);
    computeCallSerial += 1;
    residentPreparedLane = null;
    return Object.freeze({
      schemaVersion: 1,
      kind: 'live-first-instance-compute-group-submission',
      laneId: activeLane,
      timestampContextId: registration.timestampContextId,
      computeGroupIdentity: registration.computeGroupIdentity,
      registrationSerial: registration.registrationSerial,
      backendIdentity: registration.backendIdentity,
      backendWrapperIdentity: registration.backendWrapperIdentity,
      computeNodeIds: [...registration.computeNodeIds],
      computeCallSerial,
    });
  }

  async function primeLanes({ scene, render }) {
    if (lanesPrimed) throw new Error('Live first-instance lanes were already primed.');
    if (typeof renderer.compileAsync !== 'function' || typeof render !== 'function') {
      throw new TypeError('primeLanes requires renderer.compileAsync() and a render callback.');
    }
    for (const laneId of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
      const lane = lanes[laneId];
      const timestampContextId = computeTimestampContextIds[laneId];
      const registered = registerComputeTimestampGroup(
        renderer,
        lane.computeNodes,
        timestampContextId,
      );
      computeTimestampRegistrations[laneId] = Object.freeze({
        schemaVersion: 1,
        kind: 'live-first-instance-compute-timestamp-group-registration',
        laneId,
        timestampContextId: registered.contextId,
        registrationSerial: registered.registrationSerial,
        backendIdentity: registered.backendIdentity,
        backendWrapperIdentity: registered.backendWrapperIdentity,
        computeGroupIdentity: registered.computeGroupIdentity,
        computeNodeIds: Object.freeze(lane.computeNodes.map((node) => node.id ?? null)),
      });
    }
    const previousLane = activeLane;
    try {
      for (const laneId of lanePhysicalOrder) {
        setActiveLane(laneId);
        update(camera, renderer);
        submitCompute(renderer);
        await renderer.compileAsync(scene, camera);
        await render(laneId);
        if (lanes[laneId].bundleRecordCallbackCount !== 1) {
          throw new Error(`The ${laneId} live render bundle did not record exactly once.`);
        }
        renderPipelinePrimeOrder.push(laneId);
      }
      if (!exactSequence(firstComputeUseOrder, lanePhysicalOrder)
        || !exactSequence(renderPipelinePrimeOrder, lanePhysicalOrder)) {
        throw new Error('Live lane first-use order differed from lanePhysicalOrder.');
      }
      lanesPrimed = true;
    } finally {
      setActiveLane(previousLane);
    }
  }

  function requireInitializedRenderResource(laneId, renderObject, { remember = false } = {}) {
    const capturedBundle = timedRenderResources[laneId];
    const renderBundleData = capturedBundle?.renderBundle
      ? renderer.backend?.get(capturedBundle.renderBundle)
      : null;
    const state = renderObject?._nodeBuilderState;
    const bindings = renderObject?._bindings;
    const pipeline = renderer?._pipelines?.data?.get(renderObject)?.pipeline;
    const bindingData = renderer?._bindings?.data?.get(renderObject);
    if (!state
      || !bindings
      || !pipeline
      || bindingData?.initialized !== true
      || renderBundleData !== capturedBundle?.renderBundleData
      || renderBundleData?.bundleGPU !== capturedBundle?.bundleGPU
      || renderBundleData?.version !== lanes[laneId].root.version
      || renderBundleData?.renderObjects?.length !== 1
      || renderBundleData.renderObjects[0] !== renderObject) {
      throw new Error(
        `Pinned r185 ${laneId} render inspection would lazily allocate or compile a timed resource `
        + `(state=${Boolean(state)}, bindings=${Boolean(bindings)}, pipeline=${Boolean(pipeline)}, `
        + `bindingsInitialized=${bindingData?.initialized === true}).`,
      );
    }
    const remembered = timedRenderResources[laneId];
    if (remember) {
      timedRenderResources[laneId] = Object.freeze({
        renderObject,
        renderBundle: capturedBundle.renderBundle,
        renderBundleData,
        bundleGPU: capturedBundle.bundleGPU,
        state,
        bindings,
        pipeline,
      });
    } else if (!remembered
      || remembered.renderObject !== renderObject
      || remembered.state !== state
      || remembered.bindings !== bindings
      || remembered.pipeline !== pipeline
      || renderObject.object !== lanes[laneId].meshes[0]
      || renderObject.material !== lanes[laneId].material
      || renderObject.geometry !== lanes[laneId].geometry
      || renderObject.version !== lanes[laneId].material.version) {
      throw new Error(`Pinned r185 ${laneId} timed render resources changed after priming.`);
    }
    return {
      renderObjectIdentity: inspectionIdentity(renderObject),
      nodeBuilderStateIdentity: inspectionIdentity(state),
      bindingArrayIdentity: inspectionIdentity(bindings),
      pipelineIdentity: inspectionIdentity(pipeline),
    };
  }

  function requireInitializedComputeResource(laneId, phase, computeNode, { remember = false } = {}) {
    const nodeData = renderer?._nodes?.data?.get(computeNode);
    const state = nodeData?.nodeBuilderState;
    const pipeline = renderer?._pipelines?.data?.get(computeNode)?.pipeline;
    const bindingData = renderer?._bindings?.data?.get(computeNode);
    const bindings = bindingData?.bindings;
    if (!state
      || nodeData.version !== computeNode.version
      || !pipeline
      || bindingData?.initialized !== true
      || bindings !== state.bindings) {
      throw new Error(
        `Pinned r185 ${laneId} ${phase} inspection would lazily allocate or compile a timed resource.`,
      );
    }
    const remembered = timedComputeResources.get(computeNode);
    if (remember) {
      timedComputeResources.set(computeNode, Object.freeze({ state, bindings, pipeline }));
    } else if (!remembered
      || remembered.state !== state
      || remembered.bindings !== bindings
      || remembered.pipeline !== pipeline) {
      throw new Error(`Pinned r185 ${laneId} ${phase} timed compute resources changed after priming.`);
    }
    return {
      computeNodeId: computeNode.id ?? null,
      computeNodeVersion: computeNode.version,
      nodeBuilderStateIdentity: inspectionIdentity(state),
      bindingArrayIdentity: inspectionIdentity(bindings),
      pipelineIdentity: inspectionIdentity(pipeline),
    };
  }

  function capturePrimedResourceIdentities({ remember = false } = {}) {
    const render = {};
    const compute = {};
    for (const laneId of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
      const renderObject = timedRenderResources[laneId]?.renderObject;
      if (!renderObject) {
        throw new Error(`Pinned r185 ${laneId} timed render object was not captured.`);
      }
      render[laneId] = requireInitializedRenderResource(laneId, renderObject, { remember });
      compute[laneId] = {};
      for (const [phaseIndex, phase] of ['reset', 'cull'].entries()) {
        const computeNode = lanes[laneId].computeNodes[phaseIndex];
        compute[laneId][phase] = requireInitializedComputeResource(
          laneId,
          phase,
          computeNode,
          { remember },
        );
      }
    }
    return { render, compute };
  }

  function captureTimedRenderCommitment(laneId = activeLane) {
    validateFixedSliceLane(laneId);
    if (laneId !== activeLane) {
      throw new Error('Timed render commitment requires the requested lane to be active.');
    }
    const resources = capturePrimedResourceIdentities();
    const lane = lanes[laneId];
    return {
      schemaVersion: 1,
      kind: 'live-first-instance-timed-bundle-commitment',
      laneId,
      bundleUuid: lane.root.uuid,
      bundleVersion: lane.root.version,
      meshUuid: lane.meshes[0].uuid,
      geometryUuid: lane.geometry.uuid,
      materialUuid: lane.material.uuid,
      materialVersion: lane.material.version,
      bundleRecordCallbackCount: lane.bundleRecordCallbackCount,
      renderResourceIdentities: resources.render[laneId],
      inspectionState: pinnedRendererInspectionState(renderer),
    };
  }

  async function assembleShaderEvidence({
    renderCaptures,
    computeEvidence,
    captureMode,
    resourceIdentitiesAtStart,
    resourceIdentitiesAtEnd,
    inspectionStateBefore,
    inspectionStateAfter,
    observationRequest = null,
    executionCountersAtStart = null,
    executionCountersAtEnd = null,
  }) {
    const renderEvidence = await createFirstInstanceShaderEvidence({
      portable: renderCaptures[PORTABLE_LANE],
      feature: renderCaptures[FEATURE_LANE],
    });
    const evidence = {
      schemaVersion: 1,
      kind: 'live-first-instance-compute-render-shader-evidence',
      pass: renderEvidence.pass && computeEvidence.pass,
      render: {
        ...renderEvidence,
        rawSources: Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map((laneId) => [
          laneId,
          {
            vertexShader: renderCaptures[laneId].vertexShader,
            fragmentShader: renderCaptures[laneId].fragmentShader,
          },
        ])),
      },
      compute: computeEvidence,
    };
    const semanticSha256 = await sha256Text(JSON.stringify(evidence));
    if (captureMode === 'bootstrap-renderer-debug-capture') return evidence;
    const request = validateShaderObservationRequest(
      observationRequest,
      shaderObservationSerial + 1,
    );
    shaderObservationSerial += 1;
    const observationCore = {
      schemaVersion: 1,
      kind: 'live-first-instance-fresh-shader-runtime-observation',
      serial: shaderObservationSerial,
      captureMode,
      request,
      executionCountersAtStart,
      executionCountersAtEnd,
      executionCountersStable: JSON.stringify(executionCountersAtStart)
        === JSON.stringify(executionCountersAtEnd),
      resourcesPreinitialized: true,
      resourceIdentitiesAtStart,
      resourceIdentitiesAtEnd,
      resourceIdentitiesStable: JSON.stringify(resourceIdentitiesAtStart)
        === JSON.stringify(resourceIdentitiesAtEnd),
      inspectionStateBefore,
      inspectionStateAfter,
      inspectionStateStable: JSON.stringify(inspectionStateBefore)
        === JSON.stringify(inspectionStateAfter),
      semanticSha256,
    };
    const observationSha256 = await sha256Text(JSON.stringify(observationCore));
    return {
      ...evidence,
      observation: {
        ...observationCore,
        observationSha256,
      },
    };
  }

  async function observePrimedShaderSources(observationRequest) {
    if (!shaderScene || !shaderRenderTarget || shaderEvidence === null) {
      throw new Error('Fresh shader observation requires bootstrap shader capture.');
    }
    const executionCountersAtStart = {
      laneSelectionSerial,
      computeCallSerial,
      prepareSerial,
    };
    const inspectionStateBefore = pinnedRendererInspectionState(renderer);
    const resourceIdentitiesAtStart = capturePrimedResourceIdentities();
    const renderCaptures = {};
    for (const laneId of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
      const renderObject = timedRenderResources[laneId].renderObject;
      const state = renderObject.getNodeBuilderState();
      if (state !== timedRenderResources[laneId].state
        || typeof state.vertexShader !== 'string'
        || typeof state.fragmentShader !== 'string') {
        throw new Error(`${laneId} timed render shader state changed after priming.`);
      }
      renderCaptures[laneId] = {
        vertexShader: state.vertexShader,
        fragmentShader: state.fragmentShader,
        vertexInputs: runtimeVertexInputEvidence(renderer, renderObject),
        storageBindings: runtimeStorageBindingEvidence(state.vertexShader, renderObject, {
          matrixAttribute: shared.attributes.matrix,
          visibleIdsAttribute: shared.attributes.visibleIds,
        }),
      };
    }
    const computeEvidence = await collectComputeShaderEvidence(renderer, lanes);
    const resourceIdentitiesAtEnd = capturePrimedResourceIdentities();
    const inspectionStateAfter = pinnedRendererInspectionState(renderer);
    const executionCountersAtEnd = {
      laneSelectionSerial,
      computeCallSerial,
      prepareSerial,
    };
    if (JSON.stringify(resourceIdentitiesAtStart) !== JSON.stringify(resourceIdentitiesAtEnd)
      || JSON.stringify(inspectionStateBefore) !== JSON.stringify(inspectionStateAfter)
      || JSON.stringify(executionCountersAtStart) !== JSON.stringify(executionCountersAtEnd)) {
      throw new Error('Fresh shader inspection changed a pinned timed renderer resource.');
    }
    shaderEvidence = await assembleShaderEvidence({
      renderCaptures,
      computeEvidence,
      captureMode: 'live-r185-cache-inspection',
      resourceIdentitiesAtStart,
      resourceIdentitiesAtEnd,
      inspectionStateBefore,
      inspectionStateAfter,
      observationRequest,
      executionCountersAtStart,
      executionCountersAtEnd,
    });
    return shaderEvidence;
  }

  async function collectShaderSources(scene) {
    if (!lanesPrimed) throw new Error('Shader capture requires primed live lanes.');
    if (shaderEvidence !== null || shaderScene !== null) {
      throw new Error('Bootstrap shader capture may run exactly once.');
    }
    if (typeof renderer.debug?.getShaderAsync !== 'function') {
      throw new Error('Pinned r185 render shader capture is unavailable.');
    }
    shaderScene = scene;
    shaderRenderTarget = renderer.getRenderTarget?.() ?? null;
    if (!shaderRenderTarget) {
      throw new Error('Bootstrap shader capture requires the exact non-null timed render target.');
    }
    const previousLane = activeLane;
    const renderCaptures = {};
    try {
      for (const laneId of FIRST_INSTANCE_LIVE_CROSSOVER_LANES) {
        setActiveLane(laneId);
        const mesh = lanes[laneId].meshes[0];
        const sources = await renderer.debug.getShaderAsync(scene, camera, mesh);
        const timedRenderResource = resolveTimedBundleRenderResource(
          renderer,
          camera,
          lanes[laneId].root,
          mesh,
          shaderRenderTarget,
        );
        const { renderObject } = timedRenderResource;
        const state = renderObject.getNodeBuilderState();
        if (state.vertexShader !== sources.vertexShader
          || state.fragmentShader !== sources.fragmentShader) {
          throw new Error(`${laneId} shader capture differs from the timed render object.`);
        }
        timedRenderResources[laneId] = timedRenderResource;
        renderCaptures[laneId] = {
          vertexShader: sources.vertexShader,
          fragmentShader: sources.fragmentShader,
          vertexInputs: runtimeVertexInputEvidence(renderer, renderObject),
          storageBindings: runtimeStorageBindingEvidence(sources.vertexShader, renderObject, {
            matrixAttribute: shared.attributes.matrix,
            visibleIdsAttribute: shared.attributes.visibleIds,
          }),
        };
      }
    } finally {
      setActiveLane(previousLane);
    }
    const computeEvidence = await collectComputeShaderEvidence(renderer, lanes);
    const inspectionStateBefore = pinnedRendererInspectionState(renderer);
    const resourceIdentitiesAtStart = capturePrimedResourceIdentities({ remember: true });
    const resourceIdentitiesAtEnd = capturePrimedResourceIdentities();
    const inspectionStateAfter = pinnedRendererInspectionState(renderer);
    if (JSON.stringify(resourceIdentitiesAtStart) !== JSON.stringify(resourceIdentitiesAtEnd)
      || JSON.stringify(inspectionStateBefore) !== JSON.stringify(inspectionStateAfter)) {
      throw new Error('Bootstrap shader inspection changed a pinned timed renderer resource.');
    }
    shaderEvidence = await assembleShaderEvidence({
      renderCaptures,
      computeEvidence,
      captureMode: 'bootstrap-renderer-debug-capture',
      resourceIdentitiesAtStart,
      resourceIdentitiesAtEnd,
      inspectionStateBefore,
      inspectionStateAfter,
    });
    return shaderEvidence;
  }

  function getLaneValidationResources(laneId) {
    validateFixedSliceLane(laneId);
    const lane = lanes[laneId];
    return {
      lane: laneId,
      addressMode: lane.addressMode,
      matrixAttribute: shared.attributes.matrix,
      visibleIdsAttribute: shared.attributes.visibleIds,
      overflowAttribute: shared.attributes.overflow,
      indirectAttribute: lane.indirectAttribute,
      commandOffsets: Array.from(lane.commandLayout.offsets),
      commandByteOffset: 0,
      commandRecordCount: lane.commandLayout.recordCount,
      firstIndexes: shared.firstIndexes,
      bucketBases: scenario.bucketBases,
      bucketCounts: scenario.bucketCounts,
      geometry: lane.geometry,
      material: lane.material,
      latestSnapshot: preparedSnapshots[laneId] ?? null,
    };
  }

  async function prepareLaneSnapshot(
    activeRenderer,
    activeCamera,
    laneId,
    expectedIds,
  ) {
    if (!lanesPrimed) throw new Error('Lane snapshot preparation requires primed live lanes.');
    if (activeRenderer !== renderer || activeCamera !== camera) {
      throw new Error('Lane snapshot preparation requires the fixed renderer and camera.');
    }
    setActiveLane(laneId);
    update(activeCamera, activeRenderer);
    submitCompute(activeRenderer);
    const snapshot = await readFixedSliceLaneSnapshot(activeRenderer, shared, lanes[laneId]);
    const validation = await validateFixedSliceLaneSnapshot({
      shared,
      lane: lanes[laneId],
      expectedIds,
      snapshot,
    });
    prepareSerial += 1;
    preparedSnapshots[laneId] = snapshot;
    residentPreparedLane = laneId;
    const observedCommandBuffer = Object.freeze(lanes[laneId].commandBufferCommitment());
    return {
      ...validation,
      kind: 'live-first-instance-lane-snapshot-validation',
      prepareSerial,
      laneSelectionSerial,
      computeCallSerial,
      residentAfterValidation: true,
      commandBuffer: observedCommandBuffer,
    };
  }

  async function validateSerialized(
    activeRenderer,
    expectedIds,
    {
      onLanePrepared = null,
      camera: validationCamera = camera,
      shaderObservationRequest = null,
    } = {},
  ) {
    if (onLanePrepared !== null && typeof onLanePrepared !== 'function') {
      throw new TypeError('onLanePrepared must be a function when supplied.');
    }
    if (validationCamera !== camera) {
      throw new Error('Serialized validation requires the fixed live crossover camera.');
    }
    const observedShaderEvidence = shaderEvidence === null
      ? null
      : await observePrimedShaderSources(shaderObservationRequest);
    const validationStartPrepareSerial = prepareSerial;
    const validationStartComputeCallSerial = computeCallSerial;
    const laneValidations = {};
    const laneHookEvidence = {};
    const laneAddressChallenges = {};
    for (const laneId of SERIALIZED_VALIDATION_ORDER) {
      const validation = await prepareLaneSnapshot(
        activeRenderer,
        validationCamera,
        laneId,
        expectedIds,
      );
      laneValidations[laneId] = validation;
      if (residentPreparedLane !== laneId) {
        throw new Error(`${laneId} lane ceased to be resident before its address challenge.`);
      }
      const snapshot = preparedSnapshots[laneId];
      const addressChallenge = await addressChallengeOracle.challengeLane(
        activeRenderer,
        validationCamera,
        laneId,
        {
          visibleIds: snapshot.visibleIds,
          activeCounts: validation.actualCounts,
        },
      );
      laneAddressChallenges[laneId] = addressChallenge;
      let externalEvidence = null;
      if (onLanePrepared) {
        externalEvidence = await onLanePrepared({
          lane: laneId,
          validation,
          resources: getLaneValidationResources(laneId),
          addressChallenge,
        });
      }
      if (residentPreparedLane !== laneId) {
        throw new Error(`${laneId} lane ceased to be resident inside its validation hook.`);
      }
      laneHookEvidence[laneId] = {
        schemaVersion: 1,
        kind: 'live-first-instance-resident-lane-correctness-hook',
        lane: laneId,
        pass: addressChallenge.pass && externalEvidence?.pass !== false,
        addressChallenge,
        external: externalEvidence,
      };
    }
    const portableSnapshot = preparedSnapshots[PORTABLE_LANE];
    const featureSnapshot = preparedSnapshots[FEATURE_LANE];
    const coresEqual = commandCoreEqual(
      portableSnapshot.commands,
      featureSnapshot.commands,
      scenario.bucketCount,
    );
    const membershipsEqual = canonicalMembershipEqual(
      laneValidations[PORTABLE_LANE],
      laneValidations[FEATURE_LANE],
    );
    const hookPass = SERIALIZED_VALIDATION_ORDER.every(
      (laneId) => laneHookEvidence[laneId]?.pass === true,
    );
    const exactlyOneComputePerLane = computeCallSerial
      === validationStartComputeCallSerial + SERIALIZED_VALIDATION_ORDER.length;
    const addressChallengesPass = SERIALIZED_VALIDATION_ORDER.every(
      (laneId) => laneAddressChallenges[laneId]?.pass === true,
    );
    const geometry = geometryEvidence(lanes);
    const lifecycle = lifecycleDiagnostics();
    const commandBuffersDistinct = lifecycle.commandBuffersDistinct === true;
    const commandBuffersZeroOffset = lifecycle.commandBuffersZeroOffset === true;
    const shaderEvidencePass = observedShaderEvidence?.pass === true
      && observedShaderEvidence?.observation?.captureMode === 'live-r185-cache-inspection'
      && observedShaderEvidence?.observation?.resourceIdentitiesStable === true
      && observedShaderEvidence?.observation?.inspectionStateStable === true;
    return {
      schemaVersion: 1,
      kind: 'first-instance-live-crossover-exact-paired-snapshots',
      pass: SERIALIZED_VALIDATION_ORDER.every((laneId) => laneValidations[laneId].pass)
        && coresEqual
        && membershipsEqual
        && hookPass
        && exactlyOneComputePerLane
        && geometry.pass
        && shaderEvidencePass
        && commandBuffersDistinct
        && commandBuffersZeroOffset
        && addressChallengesPass,
      validationOrder: [...SERIALIZED_VALIDATION_ORDER],
      prepareSerialStart: validationStartPrepareSerial,
      prepareSerialEnd: prepareSerial,
      computeCallSerialStart: validationStartComputeCallSerial,
      computeCallSerialEnd: computeCallSerial,
      exactlyOneComputePerLane,
      commandCoresEqual: coresEqual,
      canonicalMembershipEqual: membershipsEqual,
      commandBuffersDistinct,
      commandBuffersZeroOffset,
      shaderEvidencePass,
      rawSurvivorOrderRequiredEqual: false,
      residentPreparedLane,
      commandBufferCommitments: lifecycle.commandBufferCommitments,
      lanes: laneValidations,
      hooks: laneHookEvidence,
      addressChallenges: {
        pass: addressChallengesPass,
        rawAddressBytesRequiredEqual: false,
        rawSurvivorBytesRequiredEqual: false,
        geometry: addressChallengeOracle.geometryEvidence,
        lanes: laneAddressChallenges,
      },
      geometry,
      shaderEvidence: observedShaderEvidence,
      lifecycle,
    };
  }

  function lifecycleDiagnostics() {
    const commonAttributeNames = Object.keys(lanes[FEATURE_LANE].geometry.attributes).sort();
    const observedCommandBufferCommitments = captureCommandBufferCommitments();
    const bundleStaticFlags = Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
      (laneId) => [laneId, lanes[laneId].root.static === true],
    ));
    return {
      kind: 'live-first-instance-crossover-static-resource-lifecycle',
      lanesPrimed,
      lanePhysicalOrder: [...lanePhysicalOrder],
      laneConstructionOrder: [...laneConstructionOrder],
      firstComputeUseOrder: [...firstComputeUseOrder],
      renderPipelinePrimeOrder: [...renderPipelinePrimeOrder],
      activeLane,
      activeCommandBufferId: observedCommandBufferCommitments[activeLane].attributeId,
      residentPreparedLane,
      laneSelectionSerial,
      computeCallSerial,
      prepareSerial,
      bundleStaticFlags,
      allBundlesStatic: Object.values(bundleStaticFlags).every(Boolean),
      bundleRecordCounts: Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
        (laneId) => [laneId, lanes[laneId].bundleRecordCallbackCount],
      )),
      rootUuid: root.uuid,
      rootVersion: root.version ?? null,
      bundleUuids: Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
        (laneId) => [laneId, lanes[laneId].root.uuid],
      )),
      bundleVersions: Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
        (laneId) => [laneId, lanes[laneId].root.version ?? null],
      )),
      meshUuids: Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
        (laneId) => [laneId, lanes[laneId].meshes[0].uuid],
      )),
      geometryUuids: Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
        (laneId) => [laneId, lanes[laneId].geometry.uuid],
      )),
      materialUuids: Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
        (laneId) => [laneId, lanes[laneId].material.uuid],
      )),
      materialVersions: Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
        (laneId) => [laneId, lanes[laneId].material.version],
      )),
      computeNodeIds: Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
        (laneId) => [laneId, lanes[laneId].computeNodes.map((node) => node.id ?? null)],
      )),
      computeTimestampContextIds: { ...computeTimestampContextIds },
      computeTimestampRegistrations: Object.fromEntries(
        FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map((laneId) => [
          laneId,
          computeTimestampRegistrations[laneId]
            ? {
              ...computeTimestampRegistrations[laneId],
              computeNodeIds: [...computeTimestampRegistrations[laneId].computeNodeIds],
            }
            : null,
        ]),
      ),
      sharedAttributeIds: Object.fromEntries(Object.entries(shared.attributes).map(
        ([name, attribute]) => [name, attribute.id],
      )),
      sharedAttributeVersions: Object.fromEntries(Object.entries(shared.attributes).map(
        ([name, attribute]) => [name, attribute.version],
      )),
      commonVertexAttributeIds: Object.fromEntries(commonAttributeNames.map((name) => [
        name,
        lanes[PORTABLE_LANE].geometry.getAttribute(name).id ?? null,
      ])),
      commonVertexAttributeVersions: Object.fromEntries(commonAttributeNames.map((name) => [
        name,
        lanes[PORTABLE_LANE].geometry.getAttribute(name).version,
      ])),
      indexAttributeId: lanes[PORTABLE_LANE].geometry.index.id ?? null,
      indexAttributeVersion: lanes[PORTABLE_LANE].geometry.index.version,
      bucketBaseAttributeId:
        lanes[PORTABLE_LANE].geometry.getAttribute('bucketBase').id ?? null,
      bucketBaseAttributeVersion:
        lanes[PORTABLE_LANE].geometry.getAttribute('bucketBase').version,
      commandBufferCommitments: observedCommandBufferCommitments,
      indirectAttributeVersions: Object.fromEntries(
        FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
          (laneId) => [laneId, lanes[laneId].indirectAttribute.version],
        ),
      ),
      commandBuffersDistinct: observedCommandBufferCommitments[PORTABLE_LANE].attributeId
        !== observedCommandBufferCommitments[FEATURE_LANE].attributeId,
      commandBuffersZeroOffset: FIRST_INSTANCE_LIVE_CROSSOVER_LANES.every(
        (laneId) => observedCommandBufferCommitments[laneId].byteOffset === 0
          && observedCommandBufferCommitments[laneId].firstOffset === 0,
      ),
      geometry: geometryEvidence(lanes),
      shaderEvidencePass: shaderEvidence?.pass ?? null,
      configuredComputeDispatches: 2,
      configuredComputeSubmissions: 1,
    };
  }

  function diagnostics() {
    const lifecycle = lifecycleDiagnostics();
    return {
      ...lifecycle,
      kind: 'live-first-instance-compute-render-crossover',
      objectCount: scenario.objectCount,
      bucketCount: scenario.bucketCount,
      configuredDrawCommands: scenario.bucketCount,
      configuredRenderObjects: 1,
      sharedStorageAttributeCount: shared.storageAttributes.length,
      laneCommandBufferCount: FIRST_INSTANCE_LIVE_CROSSOVER_LANES.length,
      activeRenderObjectCount: FIRST_INSTANCE_LIVE_CROSSOVER_LANES.filter(
        (laneId) => lanes[laneId].root.visible,
      ).length,
    };
  }

  return {
    id: FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
    root,
    geometries: [
      ...shared.ownedGeometries,
      ...FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
        (laneId) => addressChallengeOracle.geometries[laneId],
      ),
    ],
    materials: [
      ...FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map((laneId) => lanes[laneId].material),
      ...FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
        (laneId) => addressChallengeOracle.materials[laneId],
      ),
    ],
    storageAttributes: [
      ...shared.storageAttributes,
      ...FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
        (laneId) => lanes[laneId].indirectAttribute,
      ),
    ],
    computeNodes: FIRST_INSTANCE_LIVE_CROSSOVER_LANES.flatMap(
      (laneId) => lanes[laneId].computeNodes,
    ),
    usesCompute: true,
    configuredDrawCommands: scenario.bucketCount,
    configuredRenderObjects: 1,
    configuredComputeDispatches: 2,
    configuredComputeSubmissions: 1,
    configuredSubmittedInstances: scenario.expectedVisibleCount,
    laneIds: [...FIRST_INSTANCE_LIVE_CROSSOVER_LANES],
    lanePhysicalOrder: [...lanePhysicalOrder],
    commandBufferCommitments,
    addressChallengeGeometryEvidence: addressChallengeOracle.geometryEvidence,
    sharedResources: shared,
    laneStates: Object.freeze({
      [PORTABLE_LANE]: lanes[PORTABLE_LANE],
      [FEATURE_LANE]: lanes[FEATURE_LANE],
    }),
    get activeLane() {
      return activeLane;
    },
    get activeCommandBufferId() {
      return commandBufferCommitments[activeLane].attributeId;
    },
    get laneSelectionSerial() {
      return laneSelectionSerial;
    },
    get computeCallSerial() {
      return computeCallSerial;
    },
    get prepareSerial() {
      return prepareSerial;
    },
    get residentPreparedLane() {
      return residentPreparedLane;
    },
    get shaderEvidence() {
      return shaderEvidence;
    },
    get shaderObservationSerial() {
      return shaderObservationSerial;
    },
    setActiveLane,
    update,
    submitCompute,
    primeLanes,
    primeBundles: primeLanes,
    collectShaderSources,
    captureTimedRenderCommitment,
    prepareLaneSnapshot,
    validateSerialized,
    validate(activeRenderer, expectedIds, options) {
      return validateSerialized(activeRenderer, expectedIds, options);
    },
    getLaneValidationResources,
    getActiveParityView() {
      const lane = lanes[activeLane];
      return {
        geometry: lane.geometry,
        material: lane.material,
        parityResources: {
          matrixAttribute: shared.attributes.matrix,
          visibleIdsAttribute: shared.attributes.visibleIds,
          visibleIdsCount: scenario.objectCount,
          visibleIdOffset: 0,
          objectCount: scenario.objectCount,
          addressMode: lane.addressMode,
        },
      };
    },
    diagnostics,
    lifecycleDiagnostics,
    dispose() {
      addressChallengeOracle.dispose();
    },
  };
}
