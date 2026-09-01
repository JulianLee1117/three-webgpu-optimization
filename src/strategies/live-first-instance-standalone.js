import { registerComputeTimestampGroup } from '../benchmark/gpu-timestamps.js';
import { FIRST_INSTANCE_LIVE_CROSSOVER_LANES } from '../benchmark/plan.js';
import { updateFrustumPlaneState } from '../culling/frustum-planes.js';
import { createFirstInstanceAddressChallengeOracle } from '../validation/first-instance-address-challenge.js';
import { createFirstInstanceLaneShaderEvidence } from '../validation/first-instance-shader-evidence.js';
import {
  runtimeStorageBindingEvidence,
  runtimeVertexInputEvidence,
} from './first-instance-crossover.js';
import {
  collectLiveComputeLaneEvidence,
} from './live-first-instance-crossover.js';
import {
  FIXED_SLICE_ADDRESS_MODE_BY_LANE,
  buildFixedSliceDeploymentStrategy,
  readFixedSliceLaneSnapshot,
  validateFixedSliceLane,
  validateFixedSliceLaneSnapshot,
} from './fixed-slice.js';

const [PORTABLE_LANE, FEATURE_LANE] = FIRST_INSTANCE_LIVE_CROSSOVER_LANES;

export const FIRST_INSTANCE_LIVE_STANDALONE_MODE =
  'first-instance-live-standalone-deployment';

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

let nextStandaloneComputeTimestampContextId = 100_001;

function allocateStandaloneComputeTimestampContextId() {
  const contextId = nextStandaloneComputeTimestampContextId;
  if (!Number.isSafeInteger(contextId) || contextId <= 0) {
    throw new Error('Standalone compute timestamp context-ID space was exhausted.');
  }
  nextStandaloneComputeTimestampContextId += 1;
  return contextId;
}

function exactSequence(left, right) {
  return left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

function exactTypedArray(left, right) {
  return ArrayBuffer.isView(left)
    && ArrayBuffer.isView(right)
    && left.constructor === right.constructor
    && exactSequence(left, right);
}

function validateShaderObservationRequest(request, expectedOrdinal) {
  if (!request || typeof request !== 'object' || Array.isArray(request)) {
    throw new TypeError('Fresh standalone shader observation requires a request object.');
  }
  if (!exactSequence(Object.keys(request).sort(), [...SHADER_OBSERVATION_REQUEST_KEYS].sort())) {
    throw new TypeError('Standalone shader observation request fields differ from the schema.');
  }
  if (request.schemaVersion !== 1
    || request.kind !== 'live-first-instance-standalone-shader-observation-challenge'
    || !['node-runner', 'page-interactive'].includes(request.origin)
    || !SHADER_OBSERVATION_PHASES.includes(request.phase)
    || !SHADER_OBSERVATION_ROLES.includes(request.role)
    || request.captureOrdinal !== expectedOrdinal
    || !SHA256_HEX_PATTERN.test(request.challengeNonce ?? '')) {
    throw new TypeError('Standalone shader observation request violates the frozen protocol.');
  }
  if (request.origin === 'node-runner') {
    if (typeof request.runId !== 'string' || request.runId.length === 0
      || typeof request.trialId !== 'string' || request.trialId.length === 0
      || !Number.isSafeInteger(request.planIndex) || request.planIndex < 0
      || !Number.isSafeInteger(request.repetitionIndex) || request.repetitionIndex < 0
      || request.phase === 'interactive') {
      throw new TypeError('Node-runner standalone shader observation identity is invalid.');
    }
  } else if (request.runId !== null
    || request.trialId !== null
    || request.planIndex !== null
    || request.repetitionIndex !== null
    || request.phase !== 'interactive') {
    throw new TypeError('Interactive standalone shader observation identity is invalid.');
  }
  return structuredClone(request);
}

function resolveTimedRenderObject(renderer, scene, camera, mesh, renderTarget) {
  const renderList = renderer?._renderLists?.get(scene, camera);
  const renderContext = renderer?._renderContexts?.get(renderTarget, renderer._mrt);
  const material = scene.overrideMaterial || mesh.material;
  if (!renderList?.lightsNode || !renderContext?.clippingContext
    || typeof renderer?._objects?.get !== 'function') {
    throw new Error('Pinned r185 standalone render-object inspection is unavailable.');
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
    throw new Error('Pinned r185 resolved a different standalone timed render object.');
  }
  return renderObject;
}

function resolveTimedBundle(renderer, camera, lane, renderTarget) {
  if (typeof renderer?._renderContexts?.get !== 'function'
    || typeof renderer?._bundles?.get !== 'function'
    || typeof renderer?.backend?.get !== 'function') {
    throw new Error('Pinned r185 standalone render-bundle inspection is unavailable.');
  }
  const renderContext = renderer._renderContexts.get(renderTarget, renderer._mrt);
  const renderBundle = renderer._bundles.get(lane.root, camera, renderContext);
  const renderBundleData = renderer.backend.get(renderBundle);
  if (!Array.isArray(renderBundleData?.renderObjects)
    || renderBundleData.renderObjects.length !== 1
    || renderBundleData.renderObjects[0]?.object !== lane.meshes[0]
    || renderBundleData.renderObjects[0]?.bundle !== lane.root
    || renderBundleData.bundleGPU === undefined
    || renderBundleData.version !== lane.root.version) {
    throw new Error('Standalone timed bundle is missing or does not own exactly one mesh.');
  }
  return {
    renderBundle,
    renderBundleData,
    renderObject: renderBundleData.renderObjects[0],
    bundleGPU: renderBundleData.bundleGPU,
  };
}

function immutableScenarioFieldsExact(base, next) {
  return base.objectCount === next.objectCount
    && base.bucketCount === next.bucketCount
    && base.layout === next.layout
    && exactTypedArray(base.objectBuckets, next.objectBuckets)
    && exactTypedArray(base.bucketBases, next.bucketBases)
    && exactTypedArray(base.bucketCounts, next.bucketCounts)
    && exactTypedArray(base.cullOrder, next.cullOrder);
}

export function validateStandaloneScenarioTransition(base, next) {
  if (!base || !next || typeof base !== 'object' || typeof next !== 'object') {
    throw new TypeError('Standalone visibility transition requires two scenarios.');
  }
  if (!immutableScenarioFieldsExact(base, next)) {
    throw new Error('Standalone visibility transition changed immutable workload fields.');
  }
  if (!(next.matrices instanceof Float32Array)
    || next.matrices.length !== base.objectCount * 16
    || !(next.bounds instanceof Float32Array)
    || next.bounds.length !== base.objectCount * 4
    || !(next.expectedVisibleIds instanceof Uint32Array)
    || !Number.isInteger(next.expectedVisibleCount)
    || next.expectedVisibleCount !== next.expectedVisibleIds.length
    || ![0.2, 0.99].includes(next.visibilityFraction)) {
    throw new Error('Standalone visibility transition payload is malformed.');
  }
  return true;
}

function cacheDiagnostics(renderer) {
  const pipelineCache = renderer?._pipelines?.caches;
  const programs = renderer?._pipelines?.programs;
  const memory = renderer?.info?.memory;
  if (!(pipelineCache instanceof Map)
    || !(programs?.vertex instanceof Map)
    || !(programs?.fragment instanceof Map)
    || !(programs?.compute instanceof Map)
    || !memory) {
    throw new Error('Pinned r185 standalone cache inspection is unavailable.');
  }
  return {
    totalPipelineCacheEntries: pipelineCache.size,
    programEntries: {
      vertex: programs.vertex.size,
      fragment: programs.fragment.size,
      compute: programs.compute.size,
    },
    memory: Object.fromEntries([
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
    ].map((field) => [field, memory[field]])),
  };
}

async function sha256CanonicalJson(value) {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest),
    (entry) => entry.toString(16).padStart(2, '0')).join('');
}

export function buildFirstInstanceLiveStandaloneStrategy({
  scenario,
  sourceGeometries,
  renderer,
  camera,
  laneId,
}) {
  validateFixedSliceLane(laneId, 'laneId');
  if (laneId === FEATURE_LANE
    && renderer?.hasFeature?.('indirect-first-instance') !== true) {
    throw new Error('The standalone feature lane requires indirect-first-instance.');
  }
  const addressMode = FIXED_SLICE_ADDRESS_MODE_BY_LANE[laneId];
  const deployment = buildFixedSliceDeploymentStrategy(
    { scenario, sourceGeometries, renderer, camera },
    { featureAvailable: laneId === FEATURE_LANE },
  );
  const shared = deployment.sharedResources;
  const lane = deployment.laneState;
  if (lane?.lane !== laneId || lane?.addressMode !== addressMode) {
    throw new Error('Standalone deployment factory selected the wrong lane.');
  }
  const absentLane = laneId === PORTABLE_LANE ? FEATURE_LANE : PORTABLE_LANE;
  const constructedAddressModes = Object.keys(shared.geometriesByAddressMode ?? {});
  const constructedLaneIds = Object.entries(FIXED_SLICE_ADDRESS_MODE_BY_LANE)
    .filter(([, mode]) => constructedAddressModes.includes(mode))
    .map(([constructedLaneId]) => constructedLaneId);
  const constructedLaneCount = constructedLaneIds.length;
  const absentLaneConstructed = constructedLaneIds.includes(absentLane);
  if (deployment.geometries?.length !== 1
    || shared.ownedGeometries?.length !== 1
    || constructedLaneCount !== 1
    || constructedLaneIds[0] !== laneId
    || constructedAddressModes.length !== 1
    || constructedAddressModes[0] !== addressMode
    || absentLaneConstructed
    || lane.meshes?.length !== 1
    || lane.computeNodes?.length !== 2
    || !lane.material
    || !lane.indirectAttribute) {
    throw new Error('Standalone deployment factory allocated outside the selected lane.');
  }
  const commandBufferCommitment = Object.freeze(lane.commandBufferCommitment());
  const addressOracle = createFirstInstanceAddressChallengeOracle({
    scenario,
    sourceGeometries,
    firstIndexes: shared.firstIndexes,
    visibleIdsAttribute: shared.attributes.visibleIds,
    laneIds: [laneId],
    laneDefinitions: {
      [laneId]: {
        addressMode: lane.addressMode,
        productionGeometry: lane.geometry,
        indirectAttribute: lane.indirectAttribute,
        visibleIdsAttribute: shared.attributes.visibleIds,
        indirectOffsets: lane.commandLayout.offsets,
        commandBuffer: commandBufferCommitment,
        commandSegment: { index: 0, recordBase: 0, byteBase: 0 },
      },
    },
    inspectRenderObject: resolveTimedRenderObject,
    inspectVertexInputs: runtimeVertexInputEvidence,
    inspectStorageBindings: runtimeStorageBindingEvidence,
    namePrefix: `standalone-first-instance-fragment-address-challenge-${laneId}`,
  });
  const computeTimestampContextId = allocateStandaloneComputeTimestampContextId();
  let timestampRegistration = null;
  const lanesConstructed = constructedLaneCount;
  let primed = false;
  let activeScenario = scenario;
  let scenarioSwitchSerial = 0;
  let computeCallSerial = 0;
  let shaderObservationSerial = 0;
  let shaderEvidence = null;
  let computeShaderEvidence = null;
  let timedScene = null;
  let timedRenderTarget = null;
  let timedBundle = null;
  let timedRenderResources = null;
  const timedComputeResources = new Map();
  let inspectionIdentitySerial = 0;
  const inspectionIdentities = new WeakMap();
  const productionResourceLedger = Object.freeze({
    schemaVersion: 1,
    kind: 'first-instance-live-standalone-production-resource-ledger',
    selectedLane: laneId,
    absentLane,
    absentLaneConstructed,
    deploymentStrategyId: deployment.id,
    constructedLaneIds: Object.freeze([...constructedLaneIds]),
    constructedAddressModes: Object.freeze([...constructedAddressModes]),
    constructedLaneCount,
    indirectCommandBufferCount: new Set([lane.indirectAttribute]).size,
    computeNodeCount: lane.computeNodes.length,
    materialCount: new Set(lane.meshes.map((mesh) => mesh.material)).size,
    meshCount: lane.meshes.length,
    bundleCount: lane.root ? 1 : 0,
    addressMode,
    hasBucketBaseVertexAttribute:
      lane.geometry.getAttribute('bucketBase') !== undefined,
  });

  function update(nextCamera = camera, nextRenderer = renderer) {
    if (nextCamera !== camera || nextRenderer !== renderer) {
      throw new Error('Standalone camera and renderer identities are fixed.');
    }
    updateFrustumPlaneState(shared.planeState, nextCamera, nextRenderer);
  }

  function submitCompute(activeRenderer = renderer) {
    if (activeRenderer !== renderer) {
      throw new Error('Standalone compute requires the construction renderer.');
    }
    computeCallSerial += 1;
    activeRenderer.compute(lane.computeNodes);
    return {
      schemaVersion: 1,
      kind: 'first-instance-live-standalone-compute-submission',
      laneId,
      computeCallSerial,
      timestampContextId: computeTimestampContextId,
      timestampRegistration,
      registrationSerial: timestampRegistration?.registrationSerial ?? null,
      backendIdentity: timestampRegistration?.backendIdentity ?? null,
      backendWrapperIdentity: timestampRegistration?.backendWrapperIdentity ?? null,
      computeGroupIdentity: timestampRegistration?.computeGroupIdentity ?? null,
      computeNodeIds: lane.computeNodes.map((node) => node.id ?? null),
      commandBufferId: commandBufferCommitment.attributeId,
    };
  }

  function inspectionIdentity(value) {
    if ((typeof value !== 'object' && typeof value !== 'function') || value === null) {
      throw new TypeError('Standalone pinned-resource inspection requires an object.');
    }
    let identity = inspectionIdentities.get(value);
    if (identity === undefined) {
      inspectionIdentitySerial += 1;
      identity = inspectionIdentitySerial;
      inspectionIdentities.set(value, identity);
    }
    return identity;
  }

  function capturePrimedResourceIdentities({ remember = false } = {}) {
    if (!timedBundle || !timedRenderTarget) {
      throw new Error('Standalone primed resources have not been captured.');
    }
    const observed = resolveTimedBundle(renderer, camera, lane, timedRenderTarget);
    const renderObject = observed.renderObject;
    const state = renderObject?._nodeBuilderState;
    const bindings = renderObject?._bindings;
    const pipeline = renderer?._pipelines?.data?.get(renderObject)?.pipeline;
    const bindingData = renderer?._bindings?.data?.get(renderObject);
    if (!state
      || !bindings
      || !pipeline
      || bindingData?.initialized !== true
      || observed.renderBundle !== timedBundle.renderBundle
      || observed.renderBundleData !== timedBundle.renderBundleData
      || observed.bundleGPU !== timedBundle.bundleGPU
      || renderObject.object !== lane.meshes[0]
      || renderObject.material !== lane.material
      || renderObject.geometry !== lane.geometry
      || renderObject.version !== lane.material.version) {
      throw new Error('Standalone timed render resources are not fully initialized.');
    }
    if (remember) {
      timedRenderResources = Object.freeze({ renderObject, state, bindings, pipeline });
    } else if (!timedRenderResources
      || timedRenderResources.renderObject !== renderObject
      || timedRenderResources.state !== state
      || timedRenderResources.bindings !== bindings
      || timedRenderResources.pipeline !== pipeline) {
      throw new Error('Standalone timed render resources changed after priming.');
    }

    const compute = {};
    for (const [phaseIndex, phase] of ['reset', 'cull'].entries()) {
      const computeNode = lane.computeNodes[phaseIndex];
      const nodeData = renderer?._nodes?.data?.get(computeNode);
      const computeState = nodeData?.nodeBuilderState;
      const computePipeline = renderer?._pipelines?.data?.get(computeNode)?.pipeline;
      const computeBindingData = renderer?._bindings?.data?.get(computeNode);
      const computeBindings = computeBindingData?.bindings;
      if (!computeState
        || nodeData.version !== computeNode.version
        || !computePipeline
        || computeBindingData?.initialized !== true
        || computeBindings !== computeState.bindings) {
        throw new Error(`Standalone ${phase} resources are not fully initialized.`);
      }
      const remembered = timedComputeResources.get(computeNode);
      if (remember) {
        timedComputeResources.set(computeNode, Object.freeze({
          state: computeState,
          bindings: computeBindings,
          pipeline: computePipeline,
        }));
      } else if (!remembered
        || remembered.state !== computeState
        || remembered.bindings !== computeBindings
        || remembered.pipeline !== computePipeline) {
        throw new Error(`Standalone ${phase} resources changed after priming.`);
      }
      compute[phase] = {
        computeNodeId: computeNode.id ?? null,
        computeNodeVersion: computeNode.version,
        nodeBuilderStateIdentity: inspectionIdentity(computeState),
        bindingArrayIdentity: inspectionIdentity(computeBindings),
        pipelineIdentity: inspectionIdentity(computePipeline),
      };
    }
    return {
      render: {
        renderObjectIdentity: inspectionIdentity(renderObject),
        nodeBuilderStateIdentity: inspectionIdentity(state),
        bindingArrayIdentity: inspectionIdentity(bindings),
        pipelineIdentity: inspectionIdentity(pipeline),
      },
      compute,
    };
  }

  function renderCommitment() {
    if (!primed || !timedBundle) {
      throw new Error('Standalone timed resources have not been primed.');
    }
    const observed = resolveTimedBundle(renderer, camera, lane, timedRenderTarget);
    const resourceIdentities = capturePrimedResourceIdentities();
    return {
      schemaVersion: 1,
      kind: 'first-instance-live-standalone-render-commitment',
      laneId,
      bundleUuid: lane.root.uuid,
      bundleVersion: lane.root.version,
      meshUuid: lane.meshes[0].uuid,
      geometryUuid: lane.geometry.uuid,
      materialUuid: lane.material.uuid,
      materialVersion: lane.material.version,
      bundleRecordCallbackCount: lane.bundleRecordCallbackCount,
      renderBundleStable: observed.renderBundle === timedBundle.renderBundle,
      renderBundleDataStable: observed.renderBundleData === timedBundle.renderBundleData,
      bundleGpuStable: observed.bundleGPU === timedBundle.bundleGPU,
      renderObjectStable: observed.renderObject === timedBundle.renderObject,
      nodeBuilderStateStable:
        resourceIdentities.render.nodeBuilderStateIdentity
          === inspectionIdentity(timedRenderResources.state),
      resourceIdentities,
      matrixAttributeId: shared.attributes.matrix.id,
      matrixAttributeVersion: shared.attributes.matrix.version,
      boundsAttributeId: shared.attributes.bounds.id,
      boundsAttributeVersion: shared.attributes.bounds.version,
      visibleIdsAttributeId: shared.attributes.visibleIds.id,
      visibleIdsAttributeVersion: shared.attributes.visibleIds.version,
      indirectAttributeId: lane.indirectAttribute.id,
      indirectAttributeVersion: lane.indirectAttribute.version,
      commandBuffer: { ...lane.commandBufferCommitment() },
      cache: cacheDiagnostics(renderer),
    };
  }

  async function observeShaders(scene, request) {
    if (!primed || scene !== timedScene) {
      throw new Error('Standalone shader observation requires the primed timed scene.');
    }
    const observationRequest = validateShaderObservationRequest(
      request,
      shaderObservationSerial + 1,
    );
    const resourceIdentitiesAtStart = capturePrimedResourceIdentities();
    // The production shader belongs to the render object recorded inside the
    // bundle. A scene-level renderer._objects lookup can resolve a distinct
    // RenderObject for the same mesh and material.
    const renderObject = timedRenderResources.renderObject;
    const state = renderObject.getNodeBuilderState();
    if (renderObject !== timedBundle.renderObject
      || state !== timedRenderResources.state
      || typeof state?.vertexShader !== 'string'
      || typeof state?.fragmentShader !== 'string') {
      throw new Error('Standalone timed shader state changed after priming.');
    }
    computeShaderEvidence = await collectLiveComputeLaneEvidence(renderer, shared, lane);
    const laneRecord = await createFirstInstanceLaneShaderEvidence({
      laneId,
      lane: {
        vertexShader: state.vertexShader,
        fragmentShader: state.fragmentShader,
        vertexInputs: runtimeVertexInputEvidence(renderer, renderObject),
        storageBindings: runtimeStorageBindingEvidence(
          state.vertexShader,
          renderObject,
          {
            matrixAttribute: shared.attributes.matrix,
            visibleIdsAttribute: shared.attributes.visibleIds,
          },
        ),
      },
      compute: Object.fromEntries(['reset', 'cull'].map((phase) => [phase, {
        shader: computeShaderEvidence.phases[phase].capture.computeShader,
        bindings: computeShaderEvidence.phases[phase].capture.bindings,
      }])),
    });
    shaderObservationSerial += 1;
    const commitment = renderCommitment();
    const resourceIdentitiesAtEnd = commitment.resourceIdentities;
    if (JSON.stringify(resourceIdentitiesAtStart)
      !== JSON.stringify(resourceIdentitiesAtEnd)) {
      throw new Error('Standalone shader inspection changed a pinned timed resource.');
    }
    const observationDigest = await sha256CanonicalJson({
      observationRequest,
      laneRecord,
      computeShaderEvidence,
      commitment,
      shaderObservationSerial,
    });
    shaderEvidence = {
      schemaVersion: 1,
      kind: 'first-instance-live-standalone-fresh-shader-observation',
      pass: laneRecord.pass === true
        && computeShaderEvidence.pass === true
        && Object.values({
          renderBundleStable: commitment.renderBundleStable,
          renderBundleDataStable: commitment.renderBundleDataStable,
          bundleGpuStable: commitment.bundleGpuStable,
          renderObjectStable: commitment.renderObjectStable,
          nodeBuilderStateStable: commitment.nodeBuilderStateStable,
        }).every(Boolean),
      laneId,
      observationSerial: shaderObservationSerial,
      observationRequest,
      observationDigest,
      laneRecord,
      computeShaderEvidence,
      rawSources: {
        vertexShader: state.vertexShader,
        fragmentShader: state.fragmentShader,
      },
      resourceIdentitiesAtStart,
      resourceIdentitiesAtEnd,
      commitment,
    };
    return structuredClone(shaderEvidence);
  }

  async function prime({ scene, renderTarget, render }) {
    if (primed) throw new Error('Standalone lane was already primed.');
    if (typeof render !== 'function' || !scene || !renderTarget) {
      throw new TypeError('Standalone priming requires scene, target, and render callback.');
    }
    timestampRegistration = registerComputeTimestampGroup(
      renderer,
      lane.computeNodes,
      computeTimestampContextId,
    );
    update();
    submitCompute();
    await renderer.compileAsync(scene, camera);
    await render();
    if (lane.bundleRecordCallbackCount !== 1) {
      throw new Error('Standalone production bundle did not record exactly once.');
    }
    timedScene = scene;
    timedRenderTarget = renderTarget;
    timedBundle = resolveTimedBundle(renderer, camera, lane, renderTarget);
    capturePrimedResourceIdentities({ remember: true });
    primed = true;
    return lifecycleDiagnostics();
  }

  function loadScenario(nextScenario) {
    if (!primed) throw new Error('Standalone visibility switch requires a primed lane.');
    validateStandaloneScenarioTransition(scenario, nextScenario);
    if (nextScenario === activeScenario) {
      throw new Error('Standalone visibility switch requires a different scenario snapshot.');
    }
    const before = lifecycleDiagnostics();
    shared.attributes.matrix.array.set(nextScenario.matrices);
    shared.attributes.bounds.array.set(nextScenario.bounds);
    shared.attributes.matrix.needsUpdate = true;
    shared.attributes.bounds.needsUpdate = true;
    activeScenario = nextScenario;
    scenarioSwitchSerial += 1;
    const after = lifecycleDiagnostics();
    const stableIdentity = before.rootUuid === after.rootUuid
      && before.meshUuid === after.meshUuid
      && before.geometryUuid === after.geometryUuid
      && before.materialUuid === after.materialUuid
      && before.commandBuffer.attributeId === after.commandBuffer.attributeId
      && before.computeNodeIds.every(
        (nodeId, index) => nodeId === after.computeNodeIds[index],
      );
    if (!stableIdentity) {
      throw new Error('Standalone visibility switch changed a production resource identity.');
    }
    return {
      schemaVersion: 1,
      kind: 'first-instance-live-standalone-visibility-switch',
      pass: stableIdentity,
      laneId,
      scenarioSwitchSerial,
      fromVisibilityFraction: scenarioSwitchSerial === 1
        ? scenario.visibilityFraction
        : before.activeVisibilityFraction,
      toVisibilityFraction: nextScenario.visibilityFraction,
      matrixVersionBefore: before.matrixAttributeVersion,
      matrixVersionAfter: after.matrixAttributeVersion,
      boundsVersionBefore: before.boundsAttributeVersion,
      boundsVersionAfter: after.boundsAttributeVersion,
      stableIdentity,
      before,
      after,
    };
  }

  async function validate(activeRenderer, expectedIds, {
    shaderObservationRequest = null,
  } = {}) {
    if (!primed) throw new Error('Standalone validation requires a primed lane.');
    if (activeRenderer !== renderer) {
      throw new Error('Standalone validation requires the construction renderer.');
    }
    update();
    const computeSubmission = submitCompute();
    const snapshot = await readFixedSliceLaneSnapshot(renderer, shared, lane);
    const correctness = await validateFixedSliceLaneSnapshot({
      shared,
      lane,
      expectedIds,
      snapshot,
    });
    const address = await addressOracle.challengeLane(renderer, camera, laneId, {
      visibleIds: snapshot.visibleIds,
      activeCounts: correctness.actualCounts,
    });
    const observedShaderEvidence = shaderObservationRequest === null
      ? shaderEvidence
      : await observeShaders(timedScene, shaderObservationRequest);
    const lifecycle = lifecycleDiagnostics();
    return {
      schemaVersion: 1,
      kind: 'first-instance-live-standalone-validation',
      pass: correctness.pass === true
        && address.pass === true
        && observedShaderEvidence?.pass === true
        && lifecycle.pass === true,
      laneId,
      visibilityFraction: activeScenario.visibilityFraction,
      computeSubmission,
      correctness,
      address,
      shaderEvidence: observedShaderEvidence,
      lifecycle,
      snapshot: {
        commands: Array.from(snapshot.commands),
        visibleIds: Array.from(snapshot.visibleIds),
        overflow: snapshot.overflow,
      },
    };
  }

  function lifecycleDiagnostics() {
    const observedCommandBuffer = lane.commandBufferCommitment();
    const hasBucketBase = lane.geometry.getAttribute('bucketBase') !== undefined;
    const laneShapeExact = laneId === PORTABLE_LANE ? hasBucketBase : !hasBucketBase;
    return {
      schemaVersion: 1,
      kind: 'first-instance-live-standalone-static-resource-lifecycle',
      pass: lanesConstructed === 1
        && primed
        && laneShapeExact
        && lane.bundleRecordCallbackCount === 1
        && observedCommandBuffer.attributeId === commandBufferCommitment.attributeId
        && observedCommandBuffer.byteOffset === 0
        && observedCommandBuffer.firstOffset === 0
        && timestampRegistration !== null,
      selectedLane: laneId,
      absentLane,
      absentLaneConstructed,
      lanesConstructed,
      primed,
      activeVisibilityFraction: activeScenario.visibilityFraction,
      scenarioSwitchSerial,
      computeCallSerial,
      shaderObservationSerial,
      rootUuid: lane.root.uuid,
      rootVersion: lane.root.version ?? null,
      meshUuid: lane.meshes[0].uuid,
      geometryUuid: lane.geometry.uuid,
      materialUuid: lane.material.uuid,
      materialVersion: lane.material.version,
      bundleRecordCallbackCount: lane.bundleRecordCallbackCount,
      matrixAttributeId: shared.attributes.matrix.id,
      matrixAttributeVersion: shared.attributes.matrix.version,
      boundsAttributeId: shared.attributes.bounds.id,
      boundsAttributeVersion: shared.attributes.bounds.version,
      visibleIdsAttributeId: shared.attributes.visibleIds.id,
      visibleIdsAttributeVersion: shared.attributes.visibleIds.version,
      indirectAttributeId: lane.indirectAttribute.id,
      indirectAttributeVersion: lane.indirectAttribute.version,
      computeNodeIds: lane.computeNodes.map((node) => node.id ?? null),
      commandBuffer: observedCommandBuffer,
      timestampContextId: computeTimestampContextId,
      timestampRegistration,
      productionResourceLedger,
      shaderEvidencePass: shaderEvidence?.pass ?? null,
    };
  }

  function diagnostics() {
    return {
      ...lifecycleDiagnostics(),
      kind: 'first-instance-live-standalone-deployment',
      objectCount: scenario.objectCount,
      bucketCount: scenario.bucketCount,
      configuredDrawCommands: scenario.bucketCount,
      configuredRenderObjects: 1,
      configuredComputeDispatches: 2,
      configuredComputeSubmissions: 1,
      laneCommandBufferCount: 1,
      addressMode,
    };
  }

  return {
    id: FIRST_INSTANCE_LIVE_STANDALONE_MODE,
    laneId,
    root: lane.root,
    geometries: [...shared.ownedGeometries],
    materials: [lane.material],
    parityResources: Object.freeze({
      matrixAttribute: shared.attributes.matrix,
      visibleIdsAttribute: shared.attributes.visibleIds,
      objectCount: scenario.objectCount,
      addressMode,
    }),
    storageAttributes: [...shared.storageAttributes, lane.indirectAttribute],
    computeNodes: lane.computeNodes,
    usesCompute: true,
    configuredDrawCommands: scenario.bucketCount,
    configuredRenderObjects: 1,
    configuredComputeDispatches: 2,
    configuredComputeSubmissions: 1,
    configuredSubmittedInstances: null,
    sharedResources: shared,
    laneState: lane,
    commandBufferCommitment,
    productionResourceLedger,
    get activeVisibilityFraction() {
      return activeScenario.visibilityFraction;
    },
    get expectedVisibleCount() {
      return activeScenario.expectedVisibleCount;
    },
    get computeCallSerial() {
      return computeCallSerial;
    },
    get shaderObservationSerial() {
      return shaderObservationSerial;
    },
    get shaderEvidence() {
      return shaderEvidence;
    },
    get timestampRegistration() {
      return timestampRegistration;
    },
    get computeTimestampContextId() {
      return computeTimestampContextId;
    },
    update,
    submitCompute,
    prime,
    loadScenario,
    observeShaders,
    validate,
    diagnostics,
    lifecycleDiagnostics,
    captureTimedRenderCommitment: renderCommitment,
    dispose() {
      addressOracle.dispose();
    },
  };
}
