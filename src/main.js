import * as THREE from 'three/webgpu';
import WebGPU from 'three/addons/capabilities/WebGPU.js';
import {
  CAMERA,
  DEVELOPMENT_PROTOCOL,
  THREE_REVISION,
  VIEWPORT,
} from './config.js';
import { TrialController } from './benchmark/trial-controller.js';
import {
  FROZEN_CROSSOVER_MEASURED_FRAMES,
  FROZEN_CROSSOVER_WARMUP_FRAMES,
  frozenCrossoverFrame,
} from './benchmark/frozen-crossover-schedule.js';
import {
  FROZEN_DEPTH_CROSSOVER_LANES,
  FROZEN_DEPTH_CROSSOVER_MODE,
} from './benchmark/plan.js';
import { downloadRows } from './lib/csv.js';
import {
  fingerprintFixedSubsetScenario,
  fingerprintGeometryFixtures,
} from './scenes/geometry-fingerprints.js';
import { computeScenarioDepthBinRange } from './scenes/depth-range.js';
import { createIndexedGeometryFixtures } from './scenes/geometry-fixtures.js';
import { createFixedSubsetScenario } from './scenes/fixed-subsets.js';
import { buildDrawAllStrategy } from './strategies/draw-all.js';
import {
  buildDepthBinnedFrontToBackStrategy,
  buildDepthBinnedReverseStrategy,
} from './strategies/depth-binned-fixed-slice.js';
import { buildFrozenDepthCrossoverStrategy } from './strategies/frozen-depth-crossover.js';
import {
  buildFixedSlicePerBucketStrategy,
  buildFixedSliceStrategy,
} from './strategies/fixed-slice.js';
import { disposeStrategyResources } from './strategies/resources.js';
import {
  buildThreeBlocksCoalescedStrategy,
  buildThreeBlocksCurrentStrategy,
} from './strategies/three-blocks-current.js';
import { buildThreeBlocksHistoricalStrategy } from './strategies/three-blocks-historical.js';
import { cpuVisibleIds } from './validation/membership.js';
import { captureExactRenderParity } from './validation/render-parity.js';
import './styles.css';

const elements = Object.fromEntries([
  'strategy', 'objects', 'buckets', 'visibility', 'layout', 'rebuild', 'validate', 'trial', 'export',
  'status', 'backend', 'expected-visible', 'validation', 'gpu-summary', 'cpu-summary',
  'timestamp-quantum', 'details', 'canvas-host',
].map((id) => [id, document.getElementById(id)]));

function setStatus(message) {
  elements.status.textContent = message;
}

function formatMs(value) {
  return Number.isFinite(value) ? `${value.toFixed(4)} ms` : '—';
}

if (!globalThis.isSecureContext) throw new Error('WebGPU requires a secure context.');
if (!WebGPU.isAvailable()) {
  document.body.appendChild(WebGPU.getErrorMessage());
  throw new Error('WebGPU is unavailable in this browser.');
}

const renderer = new THREE.WebGPURenderer({
  antialias: false,
  powerPreference: 'high-performance',
  reversedDepthBuffer: true,
  trackTimestamp: true,
});
renderer.setPixelRatio(VIEWPORT.devicePixelRatio);
renderer.setSize(VIEWPORT.width, VIEWPORT.height, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(0x030711, 1);
renderer.sortObjects = false;
elements['canvas-host'].appendChild(renderer.domElement);
await renderer.init();
const timestampAvailable = renderer.backend?.trackTimestamp === true;
renderer.backend.trackTimestamp = false;

const camera = new THREE.PerspectiveCamera(CAMERA.fov, CAMERA.aspect, CAMERA.near, CAMERA.far);
camera.coordinateSystem = renderer.coordinateSystem;
camera.position.fromArray(CAMERA.position);
camera.lookAt(new THREE.Vector3().fromArray(CAMERA.target));
camera.updateProjectionMatrix();

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x030711);
scene.add(new THREE.HemisphereLight(0x9bc5ff, 0x182038, 2.2));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.2);
keyLight.position.set(20, 35, 60);
scene.add(keyLight);
const frozenCrossoverRenderTarget = new THREE.RenderTarget(
  VIEWPORT.width,
  VIEWPORT.height,
  { depthBuffer: true, samples: 0 },
);

let strategy = null;
let scenario = null;
let expectedCpuVisible = null;
let sourceGeometries = [];
let sourceGeometryManifest = null;
let sourceScenarioManifest = null;
let rebuilding = false;
let validating = false;
let lastRows = [];
let lastValidation = null;
let frozenSelectorWriteSerial = 0;
let frozenCrossoverConfiguration = Object.freeze({
  laneStorageOrder: Object.freeze([...FROZEN_DEPTH_CROSSOVER_LANES]),
  superblockOrientationOffset: 0,
});

const FROZEN_LANE_ORDER_BY_ID = Object.freeze({
  'fixed-slice-depth-front-to-back': 'front-to-back',
  'fixed-slice-depth-reverse': 'reverse',
});

function isFrozenCrossoverStrategy() {
  return strategy?.id === FROZEN_DEPTH_CROSSOVER_MODE;
}

function frozenLaneOrder(laneId) {
  const order = FROZEN_LANE_ORDER_BY_ID[laneId];
  if (!order) throw new RangeError(`Unsupported frozen crossover lane: ${laneId}`);
  return order;
}

function fnv1a64Float32(values) {
  const bytes = new Uint8Array(Float32Array.from(values).buffer);
  let hash = 0xcbf29ce484222325n;
  for (const byte of bytes) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, '0');
}

function renderBenchmarkScene() {
  if (!isFrozenCrossoverStrategy()) {
    renderer.render(scene, camera);
    return;
  }
  const previousTarget = renderer.getRenderTarget();
  const previousCubeFace = renderer.getActiveCubeFace();
  const previousMipmapLevel = renderer.getActiveMipmapLevel();
  try {
    renderer.setRenderTarget(frozenCrossoverRenderTarget);
    renderer.render(scene, camera);
  } finally {
    renderer.setRenderTarget(previousTarget, previousCubeFace, previousMipmapLevel);
  }
}

const DEPTH_STRATEGY_IDS = new Set([
  'fixed-slice-depth-front-to-back',
  'fixed-slice-depth-reverse',
]);
const DEPTH_ORDERING_LAYOUT_IDS = new Set(['high-overlap', 'low-overlap']);

function validateTrialCompletion(context) {
  const diagnostics = strategy?.diagnostics?.() ?? null;
  const lifecycleDiagnostics = strategy?.lifecycleDiagnostics?.() ?? null;
  if (context.modeId === FROZEN_DEPTH_CROSSOVER_MODE) {
    const timedFrameCount = FROZEN_CROSSOVER_WARMUP_FRAMES
      + FROZEN_CROSSOVER_MEASURED_FRAMES;
    const frontOrder = frozenLaneOrder(FROZEN_DEPTH_CROSSOVER_LANES[0]);
    const reverseOrder = frozenLaneOrder(FROZEN_DEPTH_CROSSOVER_LANES[1]);
    const selectorWritesDuringTiming = frozenSelectorWriteSerial
      - context.selectorWriteSerialAtTimingStart;
    const renderCallsDuringTiming = renderer.info.render.calls
      - context.renderCallSerialAtTimingStart;
    const computeCallsDuringTiming = renderer.info.compute.calls
      - context.computeCallSerialAtTimingStart;
    const cacheAtTimingEnd = pinnedRendererCacheDiagnostics();
    const cameraViewFnv64AtTimingEnd = fnv1a64Float32(camera.matrixWorldInverse.elements);
    const cameraProjectionFnv64AtTimingEnd = fnv1a64Float32(camera.projectionMatrix.elements);
    return {
      pass: lifecycleDiagnostics?.kind === 'frozen-depth-crossover-static-bundle-lifecycle'
        && lifecycleDiagnostics.bundleGroupStatic === true
        && lifecycleDiagnostics.bundleRecordCallbackCount === 1
        && lifecycleDiagnostics.bundleRecordCallbackCount
          === context.bundleRecordCallbackCountAtTimingStart
        && lifecycleDiagnostics.meshCount === 1
        && lifecycleDiagnostics.geometryIdentityCount === 1
        && lifecycleDiagnostics.materialIdentityCount === 1
        && lifecycleDiagnostics.configuredComputeDispatches === 0
        && lifecycleDiagnostics.configuredComputeSubmissions === 0
        && lifecycleDiagnostics.laneOffsets?.[frontOrder] === context.frontLaneBase
        && lifecycleDiagnostics.laneOffsets?.[reverseOrder] === context.reverseLaneBase
        && lifecycleDiagnostics.bundleGroupUuid === context.bundleGroupUuidAtTimingStart
        && lifecycleDiagnostics.meshUuid === context.meshUuidAtTimingStart
        && lifecycleDiagnostics.geometryUuid === context.geometryUuidAtTimingStart
        && lifecycleDiagnostics.materialUuid === context.materialUuidAtTimingStart
        && lifecycleDiagnostics.matrixAttributeId === context.matrixAttributeIdAtTimingStart
        && lifecycleDiagnostics.visibleIdsAttributeId
          === context.visibleIdsAttributeIdAtTimingStart
        && lifecycleDiagnostics.indirectAttributeId
          === context.indirectAttributeIdAtTimingStart
        && lifecycleDiagnostics.selectorChallengeAttributeId
          === context.selectorChallengeAttributeIdAtTimingStart
        && lifecycleDiagnostics.bundleGroupVersion === context.bundleGroupVersionAtTimingStart
        && lifecycleDiagnostics.matrixAttributeVersion
          === context.matrixAttributeVersionAtTimingStart
        && lifecycleDiagnostics.visibleIdsAttributeVersion
          === context.visibleIdsAttributeVersionAtTimingStart
        && lifecycleDiagnostics.indirectAttributeVersion
          === context.indirectAttributeVersionAtTimingStart
        && lifecycleDiagnostics.selectorUniformUuid === context.selectorUniformUuidAtTimingStart
        && frozenCrossoverRenderTarget.texture.uuid
          === context.renderTargetTextureUuidAtTimingStart
        && frozenCrossoverRenderTarget.width === context.renderTargetWidthAtTimingStart
        && frozenCrossoverRenderTarget.height === context.renderTargetHeightAtTimingStart
        && frozenCrossoverRenderTarget.samples === context.renderTargetSamplesAtTimingStart
        && frozenCrossoverRenderTarget.depthBuffer
          === context.renderTargetDepthBufferAtTimingStart
        && cameraViewFnv64AtTimingEnd === context.cameraViewFnv64AtTimingStart
        && cameraProjectionFnv64AtTimingEnd === context.cameraProjectionFnv64AtTimingStart
        && selectorWritesDuringTiming === timedFrameCount
        && renderCallsDuringTiming === timedFrameCount
        && computeCallsDuringTiming === 0
        && cacheAtTimingEnd.available === true
        && cacheAtTimingEnd.totalPipelineCacheEntries
          === context.totalPipelineCacheEntriesAtTimingStart
        && cacheAtTimingEnd.computePipelineCacheEntries
          === context.computePipelineCacheEntriesAtTimingStart,
      kind: 'frozen-depth-crossover-static-bundle-invariant',
      bundleGroupStatic: lifecycleDiagnostics?.bundleGroupStatic ?? null,
      bundleRecordCallbackCountAtTimingStart:
        context.bundleRecordCallbackCountAtTimingStart,
      bundleRecordCallbackCountAtTimingEnd:
        lifecycleDiagnostics?.bundleRecordCallbackCount ?? null,
      meshCount: lifecycleDiagnostics?.meshCount ?? null,
      geometryIdentityCount: lifecycleDiagnostics?.geometryIdentityCount ?? null,
      materialIdentityCount: lifecycleDiagnostics?.materialIdentityCount ?? null,
      configuredComputeDispatches:
        lifecycleDiagnostics?.configuredComputeDispatches ?? null,
      configuredComputeSubmissions:
        lifecycleDiagnostics?.configuredComputeSubmissions ?? null,
      frontLaneBase: lifecycleDiagnostics?.laneOffsets?.[frontOrder] ?? null,
      reverseLaneBase: lifecycleDiagnostics?.laneOffsets?.[reverseOrder] ?? null,
      bundleGroupUuidAtTimingStart: context.bundleGroupUuidAtTimingStart,
      bundleGroupUuidAtTimingEnd: lifecycleDiagnostics?.bundleGroupUuid ?? null,
      meshUuidAtTimingStart: context.meshUuidAtTimingStart,
      meshUuidAtTimingEnd: lifecycleDiagnostics?.meshUuid ?? null,
      geometryUuidAtTimingStart: context.geometryUuidAtTimingStart,
      geometryUuidAtTimingEnd: lifecycleDiagnostics?.geometryUuid ?? null,
      materialUuidAtTimingStart: context.materialUuidAtTimingStart,
      materialUuidAtTimingEnd: lifecycleDiagnostics?.materialUuid ?? null,
      matrixAttributeIdAtTimingStart: context.matrixAttributeIdAtTimingStart,
      matrixAttributeIdAtTimingEnd: lifecycleDiagnostics?.matrixAttributeId ?? null,
      visibleIdsAttributeIdAtTimingStart: context.visibleIdsAttributeIdAtTimingStart,
      visibleIdsAttributeIdAtTimingEnd: lifecycleDiagnostics?.visibleIdsAttributeId ?? null,
      indirectAttributeIdAtTimingStart: context.indirectAttributeIdAtTimingStart,
      indirectAttributeIdAtTimingEnd: lifecycleDiagnostics?.indirectAttributeId ?? null,
      selectorChallengeAttributeIdAtTimingStart:
        context.selectorChallengeAttributeIdAtTimingStart,
      selectorChallengeAttributeIdAtTimingEnd:
        lifecycleDiagnostics?.selectorChallengeAttributeId ?? null,
      bundleGroupVersionAtTimingStart: context.bundleGroupVersionAtTimingStart,
      bundleGroupVersionAtTimingEnd: lifecycleDiagnostics?.bundleGroupVersion ?? null,
      matrixAttributeVersionAtTimingStart: context.matrixAttributeVersionAtTimingStart,
      matrixAttributeVersionAtTimingEnd:
        lifecycleDiagnostics?.matrixAttributeVersion ?? null,
      visibleIdsAttributeVersionAtTimingStart:
        context.visibleIdsAttributeVersionAtTimingStart,
      visibleIdsAttributeVersionAtTimingEnd:
        lifecycleDiagnostics?.visibleIdsAttributeVersion ?? null,
      indirectAttributeVersionAtTimingStart: context.indirectAttributeVersionAtTimingStart,
      indirectAttributeVersionAtTimingEnd:
        lifecycleDiagnostics?.indirectAttributeVersion ?? null,
      selectorUniformUuidAtTimingStart: context.selectorUniformUuidAtTimingStart,
      selectorUniformUuidAtTimingEnd: lifecycleDiagnostics?.selectorUniformUuid ?? null,
      renderTargetTextureUuidAtTimingStart: context.renderTargetTextureUuidAtTimingStart,
      renderTargetTextureUuidAtTimingEnd: frozenCrossoverRenderTarget.texture.uuid,
      renderTargetWidthAtTimingStart: context.renderTargetWidthAtTimingStart,
      renderTargetWidthAtTimingEnd: frozenCrossoverRenderTarget.width,
      renderTargetHeightAtTimingStart: context.renderTargetHeightAtTimingStart,
      renderTargetHeightAtTimingEnd: frozenCrossoverRenderTarget.height,
      renderTargetSamplesAtTimingStart: context.renderTargetSamplesAtTimingStart,
      renderTargetSamplesAtTimingEnd: frozenCrossoverRenderTarget.samples,
      renderTargetDepthBufferAtTimingStart: context.renderTargetDepthBufferAtTimingStart,
      renderTargetDepthBufferAtTimingEnd: frozenCrossoverRenderTarget.depthBuffer,
      cameraViewFnv64AtTimingStart: context.cameraViewFnv64AtTimingStart,
      cameraViewFnv64AtTimingEnd,
      cameraProjectionFnv64AtTimingStart: context.cameraProjectionFnv64AtTimingStart,
      cameraProjectionFnv64AtTimingEnd,
      selectorWriteSerialAtTimingStart: context.selectorWriteSerialAtTimingStart,
      selectorWriteSerialAtTimingEnd: frozenSelectorWriteSerial,
      selectorWritesDuringTiming,
      renderCallSerialAtTimingStart: context.renderCallSerialAtTimingStart,
      renderCallSerialAtTimingEnd: renderer.info.render.calls,
      renderCallsDuringTiming,
      computeCallSerialAtTimingStart: context.computeCallSerialAtTimingStart,
      computeCallSerialAtTimingEnd: renderer.info.compute.calls,
      computeCallsDuringTiming,
      totalPipelineCacheEntriesAtTimingStart:
        context.totalPipelineCacheEntriesAtTimingStart,
      totalPipelineCacheEntriesAtTimingEnd:
        cacheAtTimingEnd.totalPipelineCacheEntries ?? null,
      computePipelineCacheEntriesAtTimingStart:
        context.computePipelineCacheEntriesAtTimingStart,
      computePipelineCacheEntriesAtTimingEnd:
        cacheAtTimingEnd.computePipelineCacheEntries ?? null,
      expectedTimedFrameCount: timedFrameCount,
    };
  }
  if (DEPTH_STRATEGY_IDS.has(context.modeId)) {
    const expectedOrder = context.modeId === 'fixed-slice-depth-front-to-back'
      ? 'front-to-back'
      : 'reverse';
    const expectedTraversal = expectedOrder === 'front-to-back'
      ? [0, 1, 2, 3, 4, 5, 6, 7]
      : [7, 6, 5, 4, 3, 2, 1, 0];
    const expectedWorkItems = [
      context.bucketCount * 8,
      context.objectCount,
      context.bucketCount,
      context.bucketCount,
    ];
    return {
      pass: diagnostics?.kind === 'single-merged-geometry-depth-binned-fixed-slice'
        && diagnostics.depthBinCount === 8
        && diagnostics.depthOrder === expectedOrder
        && diagnostics.binTraversal?.length === expectedTraversal.length
        && diagnostics.binTraversal?.every((value, index) => value === expectedTraversal[index])
        && diagnostics.depthBinRange?.near === context.depthBinRangeNear
        && diagnostics.depthBinRange?.far === context.depthBinRangeFar
        && diagnostics.reverseOrderUniformValue === (expectedOrder === 'reverse')
        && diagnostics.meshCount === 1
        && diagnostics.geometryIdentityCount === 1
        && diagnostics.materialIdentityCount === 1
        && diagnostics.commandCount === context.bucketCount
        && diagnostics.zeroFirstInstanceCount === context.bucketCount
        && diagnostics.computeDispatchCount === 4
        && diagnostics.computeDispatchWorkItems?.length === expectedWorkItems.length
        && diagnostics.computeDispatchWorkItems?.every(
          (value, index) => value === expectedWorkItems[index],
        )
        && context.bundleRecordCallbackCountAtTimingStart === 1
        && diagnostics.bundleRecordCallbackCount
          === context.bundleRecordCallbackCountAtTimingStart,
      kind: 'depth-binned-static-bundle-invariant',
      depthBinCount: diagnostics?.depthBinCount ?? null,
      depthOrder: diagnostics?.depthOrder ?? null,
      binTraversal: diagnostics?.binTraversal ?? null,
      depthBinRange: diagnostics?.depthBinRange ?? null,
      reverseOrderUniformValue: diagnostics?.reverseOrderUniformValue ?? null,
      bundleRecordCallbackCountAtTimingStart:
        context.bundleRecordCallbackCountAtTimingStart,
      bundleRecordCallbackCountAtTimingEnd:
        diagnostics?.bundleRecordCallbackCount ?? null,
      meshCount: diagnostics?.meshCount ?? null,
      geometryIdentityCount: diagnostics?.geometryIdentityCount ?? null,
      materialIdentityCount: diagnostics?.materialIdentityCount ?? null,
      commandCount: diagnostics?.commandCount ?? null,
      zeroFirstInstanceCount: diagnostics?.zeroFirstInstanceCount ?? null,
      computeDispatchCount: diagnostics?.computeDispatchCount ?? null,
      computeDispatchWorkItems: diagnostics?.computeDispatchWorkItems ?? null,
    };
  }
  if (context.modeId === 'fixed-slice'
    && DEPTH_ORDERING_LAYOUT_IDS.has(context.scenarioLayout)) {
    return {
      pass: lifecycleDiagnostics?.kind
          === 'single-merged-geometry-atomic-fixed-slice-lifecycle'
        && lifecycleDiagnostics.bundleGroupStatic === true
        && lifecycleDiagnostics.meshCount === 1
        && lifecycleDiagnostics.geometryIdentityCount === 1
        && lifecycleDiagnostics.materialIdentityCount === 1
        && context.bundleRecordCallbackCountAtTimingStart === 1
        && lifecycleDiagnostics.bundleRecordCallbackCount
          === context.bundleRecordCallbackCountAtTimingStart,
      kind: 'atomic-fixed-slice-static-bundle-invariant',
      bundleGroupStatic: lifecycleDiagnostics?.bundleGroupStatic ?? null,
      bundleRecordCallbackCountAtTimingStart:
        context.bundleRecordCallbackCountAtTimingStart,
      bundleRecordCallbackCountAtTimingEnd:
        lifecycleDiagnostics?.bundleRecordCallbackCount ?? null,
      meshCount: lifecycleDiagnostics?.meshCount ?? null,
      geometryIdentityCount: lifecycleDiagnostics?.geometryIdentityCount ?? null,
      materialIdentityCount: lifecycleDiagnostics?.materialIdentityCount ?? null,
    };
  }
  if (context.modeId !== 'fixed-slice-per-bucket') {
    return {
      pass: context.bundleRecordCallbackCountAtTimingStart === null && diagnostics === null,
      kind: 'no-representation-specific-invariant',
    };
  }
  return {
    pass: diagnostics?.kind === 'shared-merged-geometry-per-bucket-render-objects'
      && diagnostics.geometryIdentityCount === 1
      && diagnostics.materialIdentityCount === 1
      && diagnostics.meshCount === context.bucketCount
      && diagnostics.geometryInstanceCount === Math.ceil(context.objectCount / context.bucketCount)
      && context.bundleRecordCallbackCountAtTimingStart === context.bucketCount
      && diagnostics.bundleRecordCallbackCount
        === context.bundleRecordCallbackCountAtTimingStart,
    kind: 'fixed-slice-per-bucket-static-bundle-invariant',
    bundleRecordCallbackCountAtTimingStart: context.bundleRecordCallbackCountAtTimingStart,
    bundleRecordCallbackCountAtTimingEnd: diagnostics?.bundleRecordCallbackCount ?? null,
    geometryIdentityCount: diagnostics?.geometryIdentityCount ?? null,
    materialIdentityCount: diagnostics?.materialIdentityCount ?? null,
    meshCount: diagnostics?.meshCount ?? null,
    geometryInstanceCount: diagnostics?.geometryInstanceCount ?? null,
  };
}

const trial = new TrialController(renderer, {
  warmupFrames: DEVELOPMENT_PROTOCOL.warmupFrames,
  measuredFrames: DEVELOPMENT_PROTOCOL.measuredFrames,
  onStatus: setStatus,
  onComplete: ({ rows, summary }) => {
    lastRows = rows;
    elements.export.disabled = rows.length === 0 || summary.accepted !== true;
    elements['gpu-summary'].textContent = `${formatMs(summary.gpuPassP50Ms)} / ${formatMs(summary.gpuPassP95Ms)}`;
    elements['cpu-summary'].textContent = `${formatMs(summary.cpuSubmitP50Ms)} / ${formatMs(summary.cpuSubmitP95Ms)}`;
    elements['timestamp-quantum'].textContent = summary.quantumNs === null
      ? summary.classification
      : `${summary.quantumNs.toLocaleString()} ns (${summary.classification})`;
    elements.details.textContent = JSON.stringify(summary, null, 2);
    setControlsLocked(false);
  },
  onError: (error) => {
    lastRows = [];
    elements.export.disabled = true;
    elements.details.textContent = error?.stack ?? String(error);
    setControlsLocked(false);
  },
  validateCompletion: validateTrialCompletion,
});

function selectedConfig() {
  const config = {
    strategyId: elements.strategy.value,
    objectCount: Number(elements.objects.value),
    bucketCount: Number(elements.buckets.value),
    visibilityFraction: Number(elements.visibility.value),
    layout: elements.layout.value,
  };
  if (config.strategyId === FROZEN_DEPTH_CROSSOVER_MODE) {
    config.laneStorageOrder = [...frozenCrossoverConfiguration.laneStorageOrder];
    config.superblockOrientationOffset =
      frozenCrossoverConfiguration.superblockOrientationOffset;
  }
  return config;
}

function configureFrozenCrossover({
  laneStorageOrder,
  superblockOrientationOffset,
} = {}) {
  if (rebuilding || validating || trial.active || trial.resolving) {
    throw new Error('Frozen crossover configuration cannot change during active work.');
  }
  if (!Array.isArray(laneStorageOrder)
    || laneStorageOrder.length !== FROZEN_DEPTH_CROSSOVER_LANES.length
    || new Set(laneStorageOrder).size !== FROZEN_DEPTH_CROSSOVER_LANES.length
    || FROZEN_DEPTH_CROSSOVER_LANES.some((lane) => !laneStorageOrder.includes(lane))) {
    throw new RangeError('laneStorageOrder must be the exact front/reverse lane pair.');
  }
  if (superblockOrientationOffset !== 0 && superblockOrientationOffset !== 1) {
    throw new RangeError('superblockOrientationOffset must be zero or one.');
  }
  frozenCrossoverConfiguration = Object.freeze({
    laneStorageOrder: Object.freeze([...laneStorageOrder]),
    superblockOrientationOffset,
  });
  return selectedConfig();
}

async function fingerprintWorkload() {
  if (!scenario || sourceGeometries.length === 0) {
    throw new Error('Cannot fingerprint an unbuilt workload.');
  }
  const [geometryFixtures, scenarioManifest] = await Promise.all([
    fingerprintGeometryFixtures(sourceGeometries, 'medium'),
    fingerprintFixedSubsetScenario(scenario, DEVELOPMENT_PROTOCOL.seed),
  ]);
  return {
    scenarioSeed: DEVELOPMENT_PROTOCOL.seed,
    geometryFixtures,
    scenario: scenarioManifest,
  };
}

function setControlsLocked(locked) {
  for (const id of ['strategy', 'objects', 'buckets', 'visibility', 'layout', 'rebuild', 'validate', 'trial']) {
    elements[id].disabled = locked;
  }
}

async function rebuild() {
  if (rebuilding || trial.active || trial.resolving) return;
  rebuilding = true;
  setControlsLocked(true);
  setStatus('Building indexed controlled scene…');
  try {
    disposeStrategyResources(renderer, strategy);
    sourceGeometries.forEach((geometry) => geometry.dispose());
    sourceGeometries = [];
    sourceGeometryManifest = null;
    sourceScenarioManifest = null;
    lastValidation = null;
    frozenSelectorWriteSerial = 0;
    const config = selectedConfig();
    sourceGeometries = createIndexedGeometryFixtures(config.bucketCount, 'medium');
    sourceGeometryManifest = await fingerprintGeometryFixtures(sourceGeometries, 'medium');
    scenario = createFixedSubsetScenario({
      objectCount: config.objectCount,
      bucketCount: config.bucketCount,
      visibilityFraction: config.visibilityFraction,
      geometrySpheres: sourceGeometries.map((geometry) => geometry.boundingSphere.clone()),
      seed: DEVELOPMENT_PROTOCOL.seed,
      layout: config.layout,
    });
    scenario.depthBinRange = computeScenarioDepthBinRange(scenario, camera);
    sourceScenarioManifest = await fingerprintFixedSubsetScenario(
      scenario,
      DEVELOPMENT_PROTOCOL.seed,
    );
    expectedCpuVisible = cpuVisibleIds(
      scenario,
      camera,
      renderer.coordinateSystem,
      camera.reversedDepth,
    );
    if (expectedCpuVisible.length !== scenario.expectedVisibleIds.length
      || expectedCpuVisible.some((value, index) => value !== scenario.expectedVisibleIds[index])) {
      throw new Error('Predetermined visibility does not match the independent CPU frustum reference.');
    }
    const builders = {
      'draw-all': buildDrawAllStrategy,
      'fixed-slice': buildFixedSliceStrategy,
      'fixed-slice-depth-front-to-back': buildDepthBinnedFrontToBackStrategy,
      'fixed-slice-depth-reverse': buildDepthBinnedReverseStrategy,
      [FROZEN_DEPTH_CROSSOVER_MODE]: buildFrozenDepthCrossoverStrategy,
      'fixed-slice-per-bucket': buildFixedSlicePerBucketStrategy,
      'three-blocks-coalesced': buildThreeBlocksCoalescedStrategy,
      'three-blocks-current': buildThreeBlocksCurrentStrategy,
      'three-blocks-historical': buildThreeBlocksHistoricalStrategy,
    };
    const builder = builders[config.strategyId];
    if (!builder) throw new Error(`Unknown strategy: ${config.strategyId}`);
    strategy = builder({
      scenario,
      sourceGeometries,
      renderer,
      camera,
      laneStorageOrder: config.strategyId === FROZEN_DEPTH_CROSSOVER_MODE
        ? frozenLaneOrder(config.laneStorageOrder[0])
        : undefined,
    });
    scene.add(strategy.root);
    strategy.update(camera, renderer);
    if (strategy.usesCompute) strategy.submitCompute(renderer);
    await renderer.compileAsync(scene, camera);
    elements['expected-visible'].textContent = `${scenario.expectedVisibleCount.toLocaleString()} / ${scenario.objectCount.toLocaleString()}`;
    elements.validation.textContent = 'not run';
    elements.details.textContent = '';
    lastRows = [];
    elements.export.disabled = true;
    setStatus('Ready.');
  } finally {
    rebuilding = false;
    setControlsLocked(false);
  }
}

async function validateCurrent() {
  if (!strategy || validating || trial.active || trial.resolving) return null;
  validating = true;
  setControlsLocked(true);
  setStatus('Reading back survivor and command buffers…');
  try {
    if (isFrozenCrossoverStrategy()) {
      strategy.setActiveLane(frozenLaneOrder(FROZEN_DEPTH_CROSSOVER_LANES[0]));
    }
    strategy.update(camera, renderer);
    if (strategy.usesCompute) strategy.submitCompute(renderer);
    renderBenchmarkScene();
    const result = await strategy.validate(renderer, expectedCpuVisible);
    lastValidation = result;
    elements.validation.textContent = result.pass ? 'PASS' : 'FAIL';
    elements.details.textContent = JSON.stringify(result, null, 2);
    setStatus(result.pass ? 'Validation passed.' : 'Validation failed.');
    return result;
  } finally {
    validating = false;
    setControlsLocked(false);
  }
}

function frozenParityIdentity(parity) {
  return ['color', 'depth', 'objectId'].map((channel) => (
    `${parity?.[channel]?.format}|${parity?.[channel]?.arrayType}`
    + `|${parity?.[channel]?.byteLength}|${parity?.[channel]?.sha256}`
  )).join('||');
}

async function captureRenderParity() {
  if (!strategy || validating || trial.active || trial.resolving) return null;
  validating = true;
  setControlsLocked(true);
  setStatus('Capturing exact color, depth, and object-ID parity…');
  try {
    strategy.update(camera, renderer);
    if (strategy.usesCompute) strategy.submitCompute(renderer);
    const snapshotValidation = await strategy.validate(renderer, expectedCpuVisible);
    lastValidation = snapshotValidation;
    if (snapshotValidation?.pass !== true) {
      throw new Error('Render parity refused because its computed snapshot failed validation.');
    }
    if (isFrozenCrossoverStrategy()) {
      const previousLane = strategy.activeLane;
      const lanes = {};
      try {
        for (const laneId of FROZEN_DEPTH_CROSSOVER_LANES) {
          strategy.setActiveLane(frozenLaneOrder(laneId));
          lanes[laneId] = await captureExactRenderParity({
            renderer,
            camera,
            strategy,
            expectedIds: expectedCpuVisible,
          });
        }
      } finally {
        strategy.setActiveLane(previousLane);
      }
      const identities = FROZEN_DEPTH_CROSSOVER_LANES.map(
        (laneId) => frozenParityIdentity(lanes[laneId]),
      );
      const crossLaneExact = identities[0] === identities[1];
      const result = {
        schemaVersion: 1,
        kind: 'frozen-depth-crossover-exact-render-parity',
        pass: crossLaneExact
          && FROZEN_DEPTH_CROSSOVER_LANES.every((laneId) => lanes[laneId]?.pass === true),
        laneIds: [...FROZEN_DEPTH_CROSSOVER_LANES],
        crossLaneExact,
        lanes,
        snapshotValidation,
      };
      setStatus(result.pass ? 'Render parity captured.' : 'Render parity capture was unstable.');
      return result;
    }
    const parity = await captureExactRenderParity({
      renderer,
      camera,
      strategy,
      expectedIds: expectedCpuVisible,
    });
    const result = { ...parity, snapshotValidation };
    setStatus(result.pass ? 'Render parity captured.' : 'Render parity capture was unstable.');
    return result;
  } finally {
    validating = false;
    setControlsLocked(false);
  }
}

async function startTrial(extraContext = {}) {
  if (!strategy || trial.active || trial.resolving) return;
  const validation = await validateCurrent();
  if (!validation?.pass) throw new Error('Timing refused because validation did not pass.');
  setControlsLocked(true);
  const config = selectedConfig();
  const workload = await fingerprintWorkload();
  const representationDiagnostics = strategy.diagnostics?.() ?? null;
  const timingDiagnostics = DEPTH_ORDERING_LAYOUT_IDS.has(config.layout)
    ? strategy.lifecycleDiagnostics?.() ?? representationDiagnostics
    : representationDiagnostics;
  const frozenCrossover = strategy.id === FROZEN_DEPTH_CROSSOVER_MODE;
  const frontOrder = frozenLaneOrder(FROZEN_DEPTH_CROSSOVER_LANES[0]);
  const reverseOrder = frozenLaneOrder(FROZEN_DEPTH_CROSSOVER_LANES[1]);
  const cacheAtTimingStart = frozenCrossover ? pinnedRendererCacheDiagnostics() : null;
  if (frozenCrossover && cacheAtTimingStart?.available !== true) {
    throw new Error('Frozen crossover requires pinned renderer cache diagnostics.');
  }
  const trialContext = {
    ...extraContext,
    schemaVersion: 2,
    modeId: strategy.id,
    objectCount: scenario.objectCount,
    bucketCount: scenario.bucketCount,
    targetVisibilityFraction: config.visibilityFraction,
    scenarioLayout: scenario.layout,
    depthBinRangeNear: scenario.depthBinRange.near,
    depthBinRangeFar: scenario.depthBinRange.far,
    expectedVisibleCount: scenario.expectedVisibleCount,
    validationKind: validation.kind,
    validationPass: true,
    usesCompute: strategy.usesCompute,
    configuredRenderObjects: strategy.configuredRenderObjects,
    configuredComputeDispatches: strategy.configuredComputeDispatches,
    configuredComputeSubmissions: strategy.configuredComputeSubmissions,
    bundleRecordCallbackCountAtTimingStart:
      timingDiagnostics?.bundleRecordCallbackCount ?? null,
    timestampAvailable,
    expectedRenderTimestampUidCount: frozenCrossover ? 1 : null,
    plannedLaneStorageOrder: frozenCrossover
      ? config.laneStorageOrder.join('|')
      : null,
    superblockOrientationOffset: frozenCrossover
      ? config.superblockOrientationOffset
      : null,
    frontLaneBase: frozenCrossover ? strategy.laneOffsets[frontOrder] : null,
    reverseLaneBase: frozenCrossover ? strategy.laneOffsets[reverseOrder] : null,
    selectorWriteSerialAtTimingStart: frozenCrossover
      ? frozenSelectorWriteSerial
      : null,
    renderCallSerialAtTimingStart: frozenCrossover
      ? renderer.info.render.calls
      : null,
    computeCallSerialAtTimingStart: frozenCrossover
      ? renderer.info.compute.calls
      : null,
    bundleGroupUuidAtTimingStart:
      frozenCrossover ? timingDiagnostics.bundleGroupUuid : null,
    meshUuidAtTimingStart: frozenCrossover ? timingDiagnostics.meshUuid : null,
    geometryUuidAtTimingStart: frozenCrossover ? timingDiagnostics.geometryUuid : null,
    materialUuidAtTimingStart: frozenCrossover ? timingDiagnostics.materialUuid : null,
    matrixAttributeIdAtTimingStart:
      frozenCrossover ? timingDiagnostics.matrixAttributeId : null,
    visibleIdsAttributeIdAtTimingStart:
      frozenCrossover ? timingDiagnostics.visibleIdsAttributeId : null,
    indirectAttributeIdAtTimingStart:
      frozenCrossover ? timingDiagnostics.indirectAttributeId : null,
    selectorChallengeAttributeIdAtTimingStart:
      frozenCrossover ? timingDiagnostics.selectorChallengeAttributeId : null,
    bundleGroupVersionAtTimingStart:
      frozenCrossover ? timingDiagnostics.bundleGroupVersion : null,
    matrixAttributeVersionAtTimingStart:
      frozenCrossover ? timingDiagnostics.matrixAttributeVersion : null,
    visibleIdsAttributeVersionAtTimingStart:
      frozenCrossover ? timingDiagnostics.visibleIdsAttributeVersion : null,
    indirectAttributeVersionAtTimingStart:
      frozenCrossover ? timingDiagnostics.indirectAttributeVersion : null,
    selectorUniformUuidAtTimingStart:
      frozenCrossover ? timingDiagnostics.selectorUniformUuid : null,
    renderTargetTextureUuidAtTimingStart:
      frozenCrossover ? frozenCrossoverRenderTarget.texture.uuid : null,
    renderTargetWidthAtTimingStart:
      frozenCrossover ? frozenCrossoverRenderTarget.width : null,
    renderTargetHeightAtTimingStart:
      frozenCrossover ? frozenCrossoverRenderTarget.height : null,
    renderTargetSamplesAtTimingStart:
      frozenCrossover ? frozenCrossoverRenderTarget.samples : null,
    renderTargetDepthBufferAtTimingStart:
      frozenCrossover ? frozenCrossoverRenderTarget.depthBuffer : null,
    cameraViewFnv64AtTimingStart:
      frozenCrossover ? fnv1a64Float32(camera.matrixWorldInverse.elements) : null,
    cameraProjectionFnv64AtTimingStart:
      frozenCrossover ? fnv1a64Float32(camera.projectionMatrix.elements) : null,
    totalPipelineCacheEntriesAtTimingStart:
      frozenCrossover ? cacheAtTimingStart.totalPipelineCacheEntries : null,
    computePipelineCacheEntriesAtTimingStart:
      frozenCrossover ? cacheAtTimingStart.computePipelineCacheEntries : null,
  };
  await trial.start(
    trialContext,
    frozenCrossover
      ? {
        warmupFrames: FROZEN_CROSSOVER_WARMUP_FRAMES,
        measuredFrames: FROZEN_CROSSOVER_MEASURED_FRAMES,
      }
      : undefined,
  );
  return {
    validation: structuredClone(validation),
    workload,
  };
}

function frame() {
  if (rebuilding || validating || !strategy || trial.resolving) return;
  const frameStart = performance.now();
  if (isFrozenCrossoverStrategy() && trial.active) {
    const descriptor = trial.frameDescriptor;
    if (!descriptor) throw new Error('Frozen crossover lacks an active-frame descriptor.');
    const scheduled = frozenCrossoverFrame(
      descriptor.phaseFrameIndex,
      trial.context.superblockOrientationOffset,
    );
    const commonStart = performance.now();
    const laneBase = strategy.setActiveLane(frozenLaneOrder(scheduled.laneId));
    frozenSelectorWriteSerial += 1;
    const selectorWriteSerial = frozenSelectorWriteSerial;
    const cpuCommonUpdateMs = performance.now() - commonStart;
    const gpuFrameId = renderer.info.frame;
    const renderCallSerialBefore = renderer.info.render.calls;
    const renderStart = performance.now();
    renderBenchmarkScene();
    const cpuRenderSubmitMs = performance.now() - renderStart;
    const renderCallSerial = renderer.info.render.calls;
    if (renderCallSerial !== renderCallSerialBefore + 1) {
      throw new Error('Frozen crossover must issue exactly one top-level render call per frame.');
    }
    const cpuSubmitTotalMs = cpuRenderSubmitMs;
    const cpuFrameBodyMs = performance.now() - frameStart;
    trial.recordFrame({
      gpuFrameId,
      phaseFrameIndex: descriptor.phaseFrameIndex,
      crossoverBlockIndex: scheduled.crossoverBlockIndex,
      withinBlockPosition: scheduled.withinBlockPosition,
      crossoverPattern: scheduled.pattern,
      crossoverPatternIndex: scheduled.patternIndex,
      laneId: scheduled.laneId,
      laneBase,
      selectorWriteSerial,
      renderCallSerial,
      cpuCommonUpdateMs,
      cpuComputeSubmitMs: null,
      cpuRenderSubmitMs,
      cpuSubmitTotalMs,
      cpuFrameBodyMs,
      configuredDrawCommands: strategy.configuredDrawCommands,
      configuredRenderObjects: strategy.configuredRenderObjects,
      configuredComputeDispatches: strategy.configuredComputeDispatches,
      configuredComputeSubmissions: strategy.configuredComputeSubmissions,
      configuredSubmittedInstances: strategy.configuredSubmittedInstances,
    });
    return;
  }
  const commonStart = performance.now();
  strategy.update(camera, renderer);
  const cpuCommonUpdateMs = performance.now() - commonStart;
  const gpuFrameId = renderer.info.frame;
  let cpuComputeSubmitMs = null;
  if (strategy.usesCompute) {
    const computeStart = performance.now();
    strategy.submitCompute(renderer);
    cpuComputeSubmitMs = performance.now() - computeStart;
  }
  const renderStart = performance.now();
  renderBenchmarkScene();
  const cpuRenderSubmitMs = performance.now() - renderStart;
  const cpuSubmitTotalMs = (cpuComputeSubmitMs ?? 0) + cpuRenderSubmitMs;
  const cpuFrameBodyMs = performance.now() - frameStart;
  trial.recordFrame({
    gpuFrameId,
    cpuCommonUpdateMs,
    cpuComputeSubmitMs,
    cpuRenderSubmitMs,
    cpuSubmitTotalMs,
    cpuFrameBodyMs,
    configuredDrawCommands: strategy.configuredDrawCommands,
    configuredRenderObjects: strategy.configuredRenderObjects,
    configuredComputeDispatches: strategy.configuredComputeDispatches,
    configuredComputeSubmissions: strategy.configuredComputeSubmissions,
    configuredSubmittedInstances: strategy.configuredSubmittedInstances,
  });
}

elements.rebuild.addEventListener('click', () => void rebuild());
elements.validate.addEventListener('click', () => void validateCurrent());
elements.trial.addEventListener('click', () => void startTrial());
elements.export.addEventListener('click', () => {
  downloadRows(`frames-${strategy.id}-${Date.now()}.csv`, lastRows);
});

window.addEventListener('error', (event) => {
  setStatus('Fatal error.');
  elements.details.textContent = event.error?.stack ?? event.message;
});
window.addEventListener('unhandledrejection', (event) => {
  setStatus('Fatal error.');
  elements.details.textContent = event.reason?.stack ?? String(event.reason);
});

const adapterInfo = renderer.backend?.device?.adapterInfo ?? renderer.backend?.adapter?.info ?? {};
const adapterInfoSnapshot = Object.fromEntries([
  'vendor',
  'architecture',
  'device',
  'description',
  'backend',
  'type',
  'driver',
  'isFallbackAdapter',
].map((field) => [field, adapterInfo[field] ?? null]));
function detectPerformanceNowQuantum(iterations = 20_000) {
  let previous = performance.now();
  let minimum = Number.POSITIVE_INFINITY;
  for (let index = 0; index < iterations; index += 1) {
    const current = performance.now();
    const delta = current - previous;
    if (delta > 0 && delta < minimum) minimum = delta;
    previous = current;
  }
  return Number.isFinite(minimum) ? minimum : null;
}

function pinnedRendererCacheDiagnostics() {
  const pipelineCache = renderer._pipelines?.caches;
  const computePrograms = renderer._pipelines?.programs?.compute;
  const memory = renderer.info?.memory;
  if (THREE.REVISION !== THREE_REVISION
    || !(pipelineCache instanceof Map)
    || !(computePrograms instanceof Map)
    || !memory) {
    return { available: false };
  }
  return {
    available: true,
    totalPipelineCacheEntries: pipelineCache.size,
    computePipelineCacheEntries: [...pipelineCache.values()]
      .filter((pipeline) => pipeline?.isComputePipeline === true).length,
    computeProgramEntries: computePrograms.size,
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
    ].map((name) => [name, memory[name]])),
  };
}
const environment = Object.freeze({
  threeRevision: THREE.REVISION,
  userAgent: navigator.userAgent,
  adapterInfo: adapterInfoSnapshot,
  rendererBackend: renderer.backend?.constructor?.name ?? null,
  coordinateSystem: renderer.coordinateSystem,
  get reversedDepth() { return camera.reversedDepth; },
  rendererReversedDepthBuffer: renderer.reversedDepthBuffer === true,
  maxStorageBuffersPerShaderStage:
    renderer.backend?.device?.limits?.maxStorageBuffersPerShaderStage ?? null,
  timestampAvailable,
  indirectFirstInstanceAvailable: renderer.hasFeature('indirect-first-instance'),
  crossOriginIsolated: globalThis.crossOriginIsolated === true,
  performanceNowQuantumMs: detectPerformanceNowQuantum(),
  viewport: VIEWPORT,
});

window.__WEBGPU_BENCH__ = {
  get ready() { return Boolean(strategy) && !rebuilding; },
  environment,
  selectedConfig,
  configureFrozenCrossover,
  rebuild,
  validate: validateCurrent,
  captureRenderParity,
  startTrial,
  fingerprintWorkload,
  cacheDiagnostics: pinnedRendererCacheDiagnostics,
  get geometryManifest() { return sourceGeometryManifest; },
  get scenarioManifest() { return sourceScenarioManifest; },
  get lastValidation() { return lastValidation; },
  get phase() { return trial.phase; },
  get trialError() { return trial.error?.message ?? null; },
  get rows() { return lastRows; },
};

elements.backend.textContent = `${adapterInfo.description ?? adapterInfo.device ?? 'WebGPU'} · ${adapterInfo.backend ?? 'unknown backend'}`;
renderer.setAnimationLoop(frame);
await rebuild();
