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
  FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
  firstInstanceCrossoverFrame,
} from './benchmark/first-instance-crossover-schedule.js';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
  firstInstanceLiveCrossoverFrame,
} from './benchmark/first-instance-live-crossover-schedule.js';
import {
  FIRST_INSTANCE_CROSSOVER_LANES,
  FIRST_INSTANCE_CROSSOVER_MODE,
  FIRST_INSTANCE_LIVE_CROSSOVER_LANES,
  FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
  FROZEN_DEPTH_CROSSOVER_LANES,
  FROZEN_DEPTH_CROSSOVER_MODE,
} from './benchmark/plan.js';
import {
  preprimeTimestampPools,
  timestampPoolDiagnostics,
} from './benchmark/gpu-timestamps.js';
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
import { buildFirstInstanceCrossoverStrategy } from './strategies/first-instance-crossover.js';
import { buildFirstInstanceLiveCrossoverStrategy } from './strategies/live-first-instance-crossover.js';
import {
  buildFixedSlicePerBucketStrategy,
  buildFixedSliceStrategy,
  buildIndirectFirstInstanceStrategy,
} from './strategies/fixed-slice.js';
import { disposeStrategyResources } from './strategies/resources.js';
import {
  buildThreeBlocksCoalescedStrategy,
  buildThreeBlocksCurrentStrategy,
} from './strategies/three-blocks-current.js';
import { buildThreeBlocksHistoricalStrategy } from './strategies/three-blocks-historical.js';
import { cpuVisibleIds } from './validation/membership.js';
import {
  captureExactRenderParity,
  renderReadbackRecord,
} from './validation/render-parity.js';
import { createFirstInstanceAddressChallengeOracle } from './validation/first-instance-address-challenge.js';
import { runFirstInstanceLiveForcedFeatureOffGate as runForcedFeatureOffGate } from './validation/first-instance-live-forced-off.js';

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
const webgpuUncapturedErrors = [];
const webgpuDeviceLosses = [];
renderer.backend?.device?.addEventListener?.('uncapturederror', (event) => {
  const gpuError = event?.error;
  webgpuUncapturedErrors.push({
    name: gpuError?.constructor?.name ?? 'GPUError',
    message: gpuError?.message ?? String(gpuError ?? 'unknown WebGPU error'),
  });
});
renderer.backend?.device?.lost?.then((info) => {
  webgpuDeviceLosses.push({
    reason: info?.reason ?? null,
    message: info?.message ?? null,
  });
});

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
const firstInstanceCrossoverRenderTarget = new THREE.RenderTarget(
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
let lastSummary = null;
let lastValidation = null;
let frozenSelectorWriteSerial = 0;
let firstInstanceSelectorWriteSerial = 0;
let firstInstanceLiveSelectorWriteSerial = 0;
let firstInstanceLifecycleAtTimingStart = null;
let firstInstanceLiveLifecycleAtTimingStart = null;
let firstInstanceLiveTimestampPreprime = null;
let firstInstanceLiveTimestampPoolsAtTimingStart = null;
let firstInstanceLivePreviousLaneId = null;
let firstInstanceLivePreviousPreviousLaneId = null;
let firstInstanceLiveWarmupTailSnapshot = null;
let frozenCrossoverConfiguration = Object.freeze({
  laneStorageOrder: Object.freeze([...FROZEN_DEPTH_CROSSOVER_LANES]),
  superblockOrientationOffset: 0,
});
let firstInstanceCrossoverConfiguration = Object.freeze({
  laneCommandSegmentOrder: Object.freeze([...FIRST_INSTANCE_CROSSOVER_LANES]),
  superblockOrientationOffset: 0,
});
let firstInstanceLiveCrossoverConfiguration = Object.freeze({
  lanePhysicalOrder: Object.freeze([...FIRST_INSTANCE_LIVE_CROSSOVER_LANES]),
  superblockOrientationOffset: 0,
});

const FROZEN_LANE_ORDER_BY_ID = Object.freeze({
  'fixed-slice-depth-front-to-back': 'front-to-back',
  'fixed-slice-depth-reverse': 'reverse',
});

function isFrozenCrossoverStrategy() {
  return strategy?.id === FROZEN_DEPTH_CROSSOVER_MODE;
}

function isFirstInstanceCrossoverStrategy() {
  return strategy?.id === FIRST_INSTANCE_CROSSOVER_MODE;
}

function isFirstInstanceLiveCrossoverStrategy() {
  return strategy?.id === FIRST_INSTANCE_LIVE_CROSSOVER_MODE;
}

function isAnyFirstInstanceCrossoverStrategy() {
  return isFirstInstanceCrossoverStrategy() || isFirstInstanceLiveCrossoverStrategy();
}

function isRenderOnlyCrossoverStrategy() {
  return isFrozenCrossoverStrategy() || isFirstInstanceCrossoverStrategy();
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

function fnv1a64Text(value) {
  const bytes = new TextEncoder().encode(value);
  let hash = 0xcbf29ce484222325n;
  for (const byte of bytes) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, '0');
}

function firstInstanceStaticLifecycle(lifecycle) {
  if (!lifecycle || typeof lifecycle !== 'object') return null;
  const {
    activeLane: _activeLane,
    laneSelectionSerial: _laneSelectionSerial,
    ...staticLifecycle
  } = lifecycle;
  return staticLifecycle;
}

function firstInstanceLifecycleCommitment(lifecycle) {
  const staticLifecycle = firstInstanceStaticLifecycle(lifecycle);
  return staticLifecycle === null ? null : fnv1a64Text(JSON.stringify(staticLifecycle));
}

function firstInstanceLiveStaticLifecycle(lifecycle) {
  if (!lifecycle || typeof lifecycle !== 'object') return null;
  const {
    activeLane: _activeLane,
    activeCommandBufferId: _activeCommandBufferId,
    residentPreparedLane: _residentPreparedLane,
    laneSelectionSerial: _laneSelectionSerial,
    computeCallSerial: _computeCallSerial,
    prepareSerial: _prepareSerial,
    ...staticLifecycle
  } = lifecycle;
  return staticLifecycle;
}

function firstInstanceLiveLifecycleCommitment(lifecycle) {
  const staticLifecycle = firstInstanceLiveStaticLifecycle(lifecycle);
  return staticLifecycle === null ? null : fnv1a64Text(JSON.stringify(staticLifecycle));
}

function timestampPoolStaticCommitment(diagnostics) {
  if (!diagnostics || typeof diagnostics !== 'object') return null;
  return Object.fromEntries(['render', 'compute'].map((type) => {
    const pool = diagnostics[type];
    return [type, pool ? {
      poolIdentity: pool.poolIdentity,
      querySetIdentity: pool.querySetIdentity,
      resolveBufferIdentity: pool.resolveBufferIdentity,
      resultBufferIdentity: pool.resultBufferIdentity,
      maxQueries: pool.maxQueries,
      isDisposed: pool.isDisposed,
    } : null];
  }));
}

function rectangleEvidence(value) {
  return {
    x: value.x,
    y: value.y,
    width: value.z,
    height: value.w,
  };
}

function viewportEvidence(value) {
  return {
    ...rectangleEvidence(value),
    minDepth: value.minDepth ?? 0,
    maxDepth: value.maxDepth ?? 1,
  };
}

function bindLiveFirstInstanceViewportState(renderTarget) {
  renderer.setViewport(0, 0, VIEWPORT.width, VIEWPORT.height, 0, 1);
  renderer.setScissor(0, 0, VIEWPORT.width, VIEWPORT.height);
  renderer.setScissorTest(false);
  renderTarget.viewport.set(0, 0, VIEWPORT.width, VIEWPORT.height);
  renderTarget.viewport.minDepth = 0;
  renderTarget.viewport.maxDepth = 1;
  renderTarget.scissor.set(0, 0, VIEWPORT.width, VIEWPORT.height);
  renderTarget.scissorTest = false;
}

function captureLiveFirstInstanceViewportState(renderTarget) {
  const viewport = renderer.getViewport(new THREE.Vector4());
  const scissor = renderer.getScissor(new THREE.Vector4());
  const canvasViewport = renderer.getCanvasTarget?.()?._viewport ?? viewport;
  return {
    schemaVersion: 1,
    kind: 'three-r185-live-first-instance-viewport-state',
    renderer: {
      viewport: viewportEvidence({
        ...viewport,
        minDepth: canvasViewport.minDepth,
        maxDepth: canvasViewport.maxDepth,
      }),
      scissor: rectangleEvidence(scissor),
      scissorTest: renderer.getScissorTest(),
      activeRenderTargetTextureUuid: renderer.getRenderTarget()?.texture?.uuid ?? null,
    },
    renderTarget: {
      textureUuid: renderTarget.texture.uuid,
      width: renderTarget.width,
      height: renderTarget.height,
      viewport: viewportEvidence(renderTarget.viewport),
      scissor: rectangleEvidence(renderTarget.scissor),
      scissorTest: renderTarget.scissorTest === true,
    },
  };
}

function liveFirstInstanceViewportStateIsPinned(state, renderTarget) {
  const exactRectangle = (value) => value?.x === 0
    && value.y === 0
    && value.width === VIEWPORT.width
    && value.height === VIEWPORT.height;
  const exactViewport = (value) => exactRectangle(value)
    && value.minDepth === 0
    && value.maxDepth === 1;
  return state?.schemaVersion === 1
    && state.kind === 'three-r185-live-first-instance-viewport-state'
    && exactViewport(state.renderer?.viewport)
    && exactRectangle(state.renderer?.scissor)
    && state.renderer?.scissorTest === false
    && state.renderer?.activeRenderTargetTextureUuid === null
    && state.renderTarget?.textureUuid === renderTarget.texture.uuid
    && state.renderTarget?.width === VIEWPORT.width
    && state.renderTarget?.height === VIEWPORT.height
    && exactViewport(state.renderTarget?.viewport)
    && exactRectangle(state.renderTarget?.scissor)
    && state.renderTarget?.scissorTest === false;
}

function renderBenchmarkScene() {
  if (!isRenderOnlyCrossoverStrategy() && !isFirstInstanceLiveCrossoverStrategy()) {
    renderer.render(scene, camera);
    return;
  }
  const renderTarget = isAnyFirstInstanceCrossoverStrategy()
    ? firstInstanceCrossoverRenderTarget
    : frozenCrossoverRenderTarget;
  const previousTarget = renderer.getRenderTarget();
  const previousCubeFace = renderer.getActiveCubeFace();
  const previousMipmapLevel = renderer.getActiveMipmapLevel();
  try {
    renderer.setRenderTarget(renderTarget);
    renderer.render(scene, camera);
  } finally {
    renderer.setRenderTarget(previousTarget, previousCubeFace, previousMipmapLevel);
  }
}

function parseTimestampEvidenceJson(value, label) {
  if (typeof value !== 'string') {
    throw new TypeError(`${label} must be canonical JSON text.`);
  }
  const parsed = JSON.parse(value);
  if (!Array.isArray(parsed) || JSON.stringify(parsed) !== value) {
    throw new Error(`${label} is not a canonical JSON Array.`);
  }
  return parsed;
}

function createLiveFirstInstanceWarmupScheduleAudit(context, lifecycleDiagnostics) {
  const sourceRows = trial.warmupRows;
  const events = sourceRows.map((row) => ({
    schemaVersion: 1,
    kind: 'live-first-instance-warmup-frame-event',
    phase: row.phase,
    frameIndex: row.frameIndex,
    warmupFrameIndex: row.warmupFrameIndex,
    phaseFrameIndex: row.phaseFrameIndex,
    crossoverBlockIndex: row.crossoverBlockIndex,
    withinBlockPosition: row.withinBlockPosition,
    crossoverPattern: row.crossoverPattern,
    crossoverPatternIndex: row.crossoverPatternIndex,
    previousPreviousLaneId: row.previousPreviousLaneId,
    previousLaneId: row.previousLaneId,
    laneId: row.laneId,
    commandBufferId: row.commandBufferId,
    submittedComputeLaneId: row.submittedComputeLaneId,
    computeTimestampContextId: row.computeTimestampContextId,
    computeGroupIdentity: row.computeGroupIdentity,
    computeTimestampRegistrationSerial: row.computeTimestampRegistrationSerial,
    computeTimestampBackendIdentity: row.computeTimestampBackendIdentity,
    computeTimestampBackendWrapperIdentity: row.computeTimestampBackendWrapperIdentity,
    submittedComputeNodeIds: parseTimestampEvidenceJson(
      row.submittedComputeNodeIds,
      `warmup frame ${row.frameIndex} compute-node IDs`,
    ),
    gpuFrameId: row.gpuFrameId,
    selectorWriteSerial: row.selectorWriteSerial,
    strategySelectionSerial: row.strategySelectionSerial,
    strategyComputeCallSerial: row.strategyComputeCallSerial,
    computeCallSerial: row.computeCallSerial,
    computeFrameCallIndex: row.computeFrameCallIndex,
    renderCallSerial: row.renderCallSerial,
    renderFrameCallIndex: row.renderFrameCallIndex,
    gpuComputeMs: row.gpuComputeMs,
    gpuComputeTimestampUids: parseTimestampEvidenceJson(
      row.gpuComputeTimestampUids,
      `warmup frame ${row.frameIndex} compute timestamp UIDs`,
    ),
    gpuComputeTimestampRecords: parseTimestampEvidenceJson(
      row.gpuComputeTimestampRecords,
      `warmup frame ${row.frameIndex} compute timestamp records`,
    ),
    gpuRenderMs: row.gpuRenderMs,
    gpuRenderTimestampUids: parseTimestampEvidenceJson(
      row.gpuRenderTimestampUids,
      `warmup frame ${row.frameIndex} render timestamp UIDs`,
    ),
    gpuRenderTimestampRecords: parseTimestampEvidenceJson(
      row.gpuRenderTimestampRecords,
      `warmup frame ${row.frameIndex} render timestamp records`,
    ),
    gpuPassTotalMs: row.gpuPassTotalMs,
  }));
  const computePhase = trial.timestampPhases?.warmup?.pools?.compute;
  const renderPhase = trial.timestampPhases?.warmup?.pools?.render;
  const commandBufferIds = Object.fromEntries(FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
    (laneId) => [laneId, lifecycleDiagnostics?.commandBufferCommitments?.[laneId]?.attributeId],
  ));
  const registrations = lifecycleDiagnostics?.computeTimestampRegistrations;
  const exactUid = (record, type, event) => {
    const prefix = type === 'compute' ? 'c' : 'r';
    const expectedCallIndex = type === 'compute'
      ? event.computeFrameCallIndex
      : event.renderFrameCallIndex;
    return record?.type === type
      && record?.frameId === event.gpuFrameId
      && record?.callIndex === expectedCallIndex
      && record?.durationMs > 0
      && new RegExp(`^${prefix}:[1-9]\\d*:(0|[1-9]\\d*):f(0|[1-9]\\d*)$`)
        .test(record?.uid ?? '');
  };
  const eventsExact = events.length === FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES
    && events.every((event, index) => {
      const scheduled = firstInstanceLiveCrossoverFrame(
        index,
        context.superblockOrientationOffset,
      );
      const expectedPreviousLaneId = index === 0 ? null : events[index - 1]?.laneId;
      const expectedPreviousPreviousLaneId = index < 2 ? null : events[index - 2]?.laneId;
      const registration = registrations?.[scheduled.laneId];
      const computeRecord = event.gpuComputeTimestampRecords[0];
      const renderRecord = event.gpuRenderTimestampRecords[0];
      const ordinal = index + 1;
      return event.phase === 'warmup'
        && event.frameIndex === index
        && event.warmupFrameIndex === index
        && event.phaseFrameIndex === index
        && event.crossoverBlockIndex === scheduled.crossoverBlockIndex
        && event.withinBlockPosition === scheduled.withinBlockPosition
        && event.crossoverPattern === scheduled.pattern
        && event.crossoverPatternIndex === scheduled.patternIndex
        && event.previousPreviousLaneId === expectedPreviousPreviousLaneId
        && event.previousLaneId === expectedPreviousLaneId
        && event.laneId === scheduled.laneId
        && event.commandBufferId === commandBufferIds[scheduled.laneId]
        && event.submittedComputeLaneId === scheduled.laneId
        && event.computeTimestampContextId === registration?.timestampContextId
        && event.computeGroupIdentity === registration?.computeGroupIdentity
        && event.computeTimestampRegistrationSerial === registration?.registrationSerial
        && event.computeTimestampBackendIdentity === registration?.backendIdentity
        && event.computeTimestampBackendWrapperIdentity
          === registration?.backendWrapperIdentity
        && JSON.stringify(event.submittedComputeNodeIds)
          === JSON.stringify(registration?.computeNodeIds)
        && event.selectorWriteSerial === context.selectorWriteSerialAtTimingStart + ordinal
        && event.strategySelectionSerial
          === context.strategySelectionSerialAtTimingStart + ordinal
        && event.strategyComputeCallSerial
          === context.strategyComputeCallSerialAtTimingStart + ordinal
        && event.computeCallSerial === context.computeCallSerialAtTimingStart + ordinal
        && event.renderCallSerial === context.renderCallSerialAtTimingStart + ordinal
        && event.computeFrameCallIndex === 1
        && event.renderFrameCallIndex === 1
        && event.gpuComputeTimestampUids.length === 1
        && event.gpuComputeTimestampRecords.length === 1
        && event.gpuComputeTimestampUids[0] === computeRecord?.uid
        && exactUid(computeRecord, 'compute', event)
        && computeRecord?.contextId === registration?.timestampContextId
        && event.gpuComputeMs === computeRecord?.durationMs
        && event.gpuRenderTimestampUids.length === 1
        && event.gpuRenderTimestampRecords.length === 1
        && event.gpuRenderTimestampUids[0] === renderRecord?.uid
        && exactUid(renderRecord, 'render', event)
        && event.gpuRenderMs === renderRecord?.durationMs
        && event.gpuPassTotalMs === event.gpuComputeMs + event.gpuRenderMs
        && (index === 0 || event.gpuFrameId === events[index - 1].gpuFrameId + 1);
    });
  const timestampPhaseExact = JSON.stringify(computePhase?.frames)
      === JSON.stringify(events.map((event) => event.gpuFrameId))
    && JSON.stringify(renderPhase?.frames)
      === JSON.stringify(events.map((event) => event.gpuFrameId))
    && JSON.stringify(computePhase?.uidRecords)
      === JSON.stringify(events.flatMap((event) => event.gpuComputeTimestampRecords))
    && JSON.stringify(renderPhase?.uidRecords)
      === JSON.stringify(events.flatMap((event) => event.gpuRenderTimestampRecords));
  const tail = firstInstanceLiveWarmupTailSnapshot;
  const last = events.at(-1);
  const penultimate = events.at(-2);
  const postWarmupStateExact = tail?.previousPreviousLaneId === penultimate?.laneId
    && tail?.previousLaneId === last?.laneId
    && tail?.lastWarmupGpuFrameId === last?.gpuFrameId
    && tail?.selectorWriteSerial === last?.selectorWriteSerial
    && tail?.strategySelectionSerial === last?.strategySelectionSerial
    && tail?.strategyComputeCallSerial === last?.strategyComputeCallSerial
    && tail?.computeCallSerial === last?.computeCallSerial
    && tail?.renderCallSerial === last?.renderCallSerial;
  return {
    schemaVersion: 1,
    kind: 'live-first-instance-compact-warmup-schedule-audit',
    pass: eventsExact && timestampPhaseExact && postWarmupStateExact,
    expectedFrameCount: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    actualStartupHistory: {
      previousPreviousLaneId: events[0]?.previousPreviousLaneId ?? null,
      previousLaneId: events[0]?.previousLaneId ?? null,
    },
    eventsExact,
    timestampPhaseExact,
    postWarmupStateExact,
    postWarmupState: tail === null ? null : { ...tail },
    events,
  };
}

const DEPTH_STRATEGY_IDS = new Set([
  'fixed-slice-depth-front-to-back',
  'fixed-slice-depth-reverse',
]);
const DEPTH_ORDERING_LAYOUT_IDS = new Set(['high-overlap', 'low-overlap']);

function validateTrialCompletion(context) {
  const diagnostics = strategy?.diagnostics?.() ?? null;
  const lifecycleDiagnostics = strategy?.lifecycleDiagnostics?.() ?? null;
  if (context.modeId === FIRST_INSTANCE_LIVE_CROSSOVER_MODE) {
    const timedFrameCount = FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES
      + FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES;
    const selectorWritesDuringTiming = firstInstanceLiveSelectorWriteSerial
      - context.selectorWriteSerialAtTimingStart;
    const strategySelectionsDuringTiming = (lifecycleDiagnostics?.laneSelectionSerial ?? -1)
      - context.strategySelectionSerialAtTimingStart;
    const strategyComputeCallsDuringTiming = (lifecycleDiagnostics?.computeCallSerial ?? -1)
      - context.strategyComputeCallSerialAtTimingStart;
    const strategyPreparesDuringTiming = (lifecycleDiagnostics?.prepareSerial ?? -1)
      - context.strategyPrepareSerialAtTimingStart;
    const renderCallsDuringTiming = renderer.info.render.calls
      - context.renderCallSerialAtTimingStart;
    const computeCallsDuringTiming = renderer.info.compute.calls
      - context.computeCallSerialAtTimingStart;
    const cacheAtTimingEnd = pinnedRendererCacheDiagnostics();
    const viewportStateAtTimingEnd = captureLiveFirstInstanceViewportState(
      firstInstanceCrossoverRenderTarget,
    );
    const viewportStateExact = JSON.stringify(viewportStateAtTimingEnd)
      === context.viewportStateAtTimingStart
      && liveFirstInstanceViewportStateIsPinned(
        viewportStateAtTimingEnd,
        firstInstanceCrossoverRenderTarget,
      );
    const computeProgramEntriesExact = cacheAtTimingEnd.computeProgramEntries
      === context.computeProgramEntriesAtTimingStart;
    const rendererMemoryExact = JSON.stringify(cacheAtTimingEnd.memory)
      === context.rendererMemoryAtTimingStart;
    const cameraViewFnv64AtTimingEnd = fnv1a64Float32(camera.matrixWorldInverse.elements);
    const cameraProjectionFnv64AtTimingEnd = fnv1a64Float32(
      camera.projectionMatrix.elements,
    );
    const lifecycleAtTimingEnd = firstInstanceLiveStaticLifecycle(lifecycleDiagnostics);
    const lifecycleCommitmentAtTimingEnd = firstInstanceLiveLifecycleCommitment(
      lifecycleDiagnostics,
    );
    const lifecycleExact = firstInstanceLiveLifecycleAtTimingStart !== null
      && JSON.stringify(lifecycleAtTimingEnd)
        === JSON.stringify(firstInstanceLiveLifecycleAtTimingStart);
    const plannedLanePhysicalOrder = context.plannedLanePhysicalOrder?.split('|') ?? [];
    const configuredLanePhysicalOrderExact = plannedLanePhysicalOrder.length
      === FIRST_INSTANCE_LIVE_CROSSOVER_LANES.length
      && plannedLanePhysicalOrder.every(
        (laneId, index) => lifecycleDiagnostics?.lanePhysicalOrder?.[index] === laneId,
      )
      && plannedLanePhysicalOrder.every(
        (laneId, index) => lifecycleDiagnostics?.laneConstructionOrder?.[index] === laneId,
      )
      && plannedLanePhysicalOrder.every(
        (laneId, index) => lifecycleDiagnostics?.firstComputeUseOrder?.[index] === laneId,
      )
      && plannedLanePhysicalOrder.every(
        (laneId, index) => lifecycleDiagnostics?.renderPipelinePrimeOrder?.[index] === laneId,
      );
    const commandBuffersExact = JSON.stringify(
      lifecycleDiagnostics?.commandBufferCommitments ?? null,
    ) === context.commandBufferCommitmentsAtTimingStart;
    const bundleRecordCountsExact = FIRST_INSTANCE_LIVE_CROSSOVER_LANES.every(
      (laneId) => lifecycleDiagnostics?.bundleRecordCounts?.[laneId] === 1,
    );
    const timestampPoolsAtTimingEnd = timestampPoolDiagnostics(renderer);
    const timestampPoolStaticAtTimingStart = timestampPoolStaticCommitment(
      firstInstanceLiveTimestampPoolsAtTimingStart,
    );
    const timestampPoolStaticAtTimingEnd = timestampPoolStaticCommitment(
      timestampPoolsAtTimingEnd,
    );
    const timestampPoolsStaticExact = timestampPoolStaticAtTimingStart !== null
      && JSON.stringify(timestampPoolStaticAtTimingEnd)
        === JSON.stringify(timestampPoolStaticAtTimingStart);
    const timestampPreprimeExact = firstInstanceLiveTimestampPreprime?.schemaVersion === 1
      && firstInstanceLiveTimestampPreprime?.kind === 'three-r185-timestamp-pool-preprime'
      && firstInstanceLiveTimestampPreprime?.addedTimestampUidCount?.render === 1
      && firstInstanceLiveTimestampPreprime?.addedTimestampUidCount?.compute === 1
      && JSON.stringify(timestampPoolStaticCommitment(firstInstanceLiveTimestampPreprime.after))
        === JSON.stringify(timestampPoolStaticAtTimingStart);
    const timestampPoolsResolvedCleanly = ['render', 'compute'].every((type) => {
      const before = firstInstanceLiveTimestampPoolsAtTimingStart?.[type];
      const after = timestampPoolsAtTimingEnd?.[type];
      return before
        && after
        && after.currentQueryIndex === 0
        && after.queryOffsetCount === 0
        && after.pendingResolve === false
        && after.isDisposed === false
        && after.resultBufferMapState === 'unmapped'
        && after.frameCount === FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES
        && after.timestampUidCount === before.timestampUidCount + timedFrameCount;
    });
    const commandBufferIds = FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
      (laneId) => lifecycleDiagnostics?.commandBufferCommitments?.[laneId]?.attributeId,
    );
    const warmupScheduleAudit = createLiveFirstInstanceWarmupScheduleAudit(
      context,
      lifecycleDiagnostics,
    );
    const webgpuUncapturedErrorsDuringTiming = webgpuUncapturedErrors.length
      - context.webgpuUncapturedErrorCountAtTimingStart;
    const timestampPoolsExact = timestampPoolsStaticExact && timestampPoolsResolvedCleanly;
    return {
      pass: diagnostics?.kind === 'live-first-instance-compute-render-crossover'
        && diagnostics.objectCount === context.objectCount
        && diagnostics.bucketCount === context.bucketCount
        && diagnostics.configuredDrawCommands === context.bucketCount
        && diagnostics.configuredRenderObjects === 1
        && diagnostics.laneCommandBufferCount === 2
        && diagnostics.activeRenderObjectCount === 1
        && lifecycleDiagnostics?.kind
          === 'live-first-instance-crossover-static-resource-lifecycle'
        && lifecycleDiagnostics.lanesPrimed === true
        && lifecycleDiagnostics.allBundlesStatic === true
        && lifecycleDiagnostics.shaderEvidencePass === true
        && lifecycleDiagnostics.geometry?.pass === true
        && lifecycleDiagnostics.commandBuffersDistinct === true
        && lifecycleDiagnostics.commandBuffersZeroOffset === true
        && lifecycleDiagnostics.configuredComputeDispatches === 2
        && lifecycleDiagnostics.configuredComputeSubmissions === 1
        && bundleRecordCountsExact
        && configuredLanePhysicalOrderExact
        && commandBuffersExact
        && new Set(commandBufferIds).size === 2
        && lifecycleExact
        && lifecycleCommitmentAtTimingEnd === context.lifecycleCommitmentAtTimingStart
        && firstInstanceCrossoverRenderTarget.texture.uuid
          === context.renderTargetTextureUuidAtTimingStart
        && firstInstanceCrossoverRenderTarget.width === context.renderTargetWidthAtTimingStart
        && firstInstanceCrossoverRenderTarget.height === context.renderTargetHeightAtTimingStart
        && firstInstanceCrossoverRenderTarget.samples === context.renderTargetSamplesAtTimingStart
        && firstInstanceCrossoverRenderTarget.depthBuffer
          === context.renderTargetDepthBufferAtTimingStart
        && cameraViewFnv64AtTimingEnd === context.cameraViewFnv64AtTimingStart
        && cameraProjectionFnv64AtTimingEnd === context.cameraProjectionFnv64AtTimingStart
        && selectorWritesDuringTiming === timedFrameCount
        && strategySelectionsDuringTiming === timedFrameCount
        && strategyComputeCallsDuringTiming === timedFrameCount
        && strategyPreparesDuringTiming === 0
        && renderCallsDuringTiming === timedFrameCount
        && computeCallsDuringTiming === timedFrameCount
        && timestampPreprimeExact
        && timestampPoolsExact
        && warmupScheduleAudit.pass === true
        && cacheAtTimingEnd.available === true
        && cacheAtTimingEnd.totalPipelineCacheEntries
          === context.totalPipelineCacheEntriesAtTimingStart
        && cacheAtTimingEnd.computePipelineCacheEntries
          === context.computePipelineCacheEntriesAtTimingStart
        && computeProgramEntriesExact
        && rendererMemoryExact
        && viewportStateExact
        && webgpuUncapturedErrorsDuringTiming === 0,
      kind: 'first-instance-live-crossover-static-resource-invariant',
      lanesPrimed: lifecycleDiagnostics?.lanesPrimed ?? null,
      bundlesPrimed: lifecycleDiagnostics?.lanesPrimed ?? null,
      bundleStaticFlags: lifecycleDiagnostics?.bundleStaticFlags ?? null,
      allBundlesStatic: lifecycleDiagnostics?.allBundlesStatic ?? null,
      bundleRecordCounts: lifecycleDiagnostics?.bundleRecordCounts ?? null,
      shaderEvidencePass: lifecycleDiagnostics?.shaderEvidencePass ?? null,
      geometryEvidencePass: lifecycleDiagnostics?.geometry?.pass ?? null,
      plannedLanePhysicalOrder: context.plannedLanePhysicalOrder,
      observedLanePhysicalOrder: lifecycleDiagnostics?.lanePhysicalOrder ?? null,
      observedLaneConstructionOrder:
        lifecycleDiagnostics?.laneConstructionOrder?.join('|') ?? null,
      observedFirstComputeUseOrder:
        lifecycleDiagnostics?.firstComputeUseOrder?.join('|') ?? null,
      observedRenderPipelinePrimeOrder:
        lifecycleDiagnostics?.renderPipelinePrimeOrder?.join('|') ?? null,
      configuredLanePhysicalOrderExact,
      lanePhysicalOrderExact: configuredLanePhysicalOrderExact,
      commandBuffersDistinct: lifecycleDiagnostics?.commandBuffersDistinct ?? null,
      commandBuffersZeroOffset: lifecycleDiagnostics?.commandBuffersZeroOffset ?? null,
      commandBuffersExact,
      commandBufferCommitments: lifecycleDiagnostics?.commandBufferCommitments ?? null,
      lifecycleExact,
      staticLifecycleAtTimingStart: firstInstanceLiveLifecycleAtTimingStart === null
        ? null
        : structuredClone(firstInstanceLiveLifecycleAtTimingStart),
      staticLifecycleAtTimingEnd: lifecycleAtTimingEnd === null
        ? null
        : structuredClone(lifecycleAtTimingEnd),
      lifecycleCommitmentAtTimingStart: context.lifecycleCommitmentAtTimingStart,
      lifecycleCommitmentAtTimingEnd,
      selectorWriteSerialAtTimingStart: context.selectorWriteSerialAtTimingStart,
      selectorWriteSerialAtTimingEnd: firstInstanceLiveSelectorWriteSerial,
      selectorWritesDuringTiming,
      strategySelectionSerialAtTimingStart: context.strategySelectionSerialAtTimingStart,
      strategySelectionSerialAtTimingEnd:
        lifecycleDiagnostics?.laneSelectionSerial ?? null,
      strategySelectionsDuringTiming,
      strategyComputeCallSerialAtTimingStart: context.strategyComputeCallSerialAtTimingStart,
      strategyComputeCallSerialAtTimingEnd:
        lifecycleDiagnostics?.computeCallSerial ?? null,
      strategyComputeCallsDuringTiming,
      strategyPrepareSerialAtTimingStart: context.strategyPrepareSerialAtTimingStart,
      strategyPrepareSerialAtTimingEnd: lifecycleDiagnostics?.prepareSerial ?? null,
      strategyPreparesDuringTiming,
      renderCallSerialAtTimingStart: context.renderCallSerialAtTimingStart,
      renderCallSerialAtTimingEnd: renderer.info.render.calls,
      renderCallsDuringTiming,
      computeCallSerialAtTimingStart: context.computeCallSerialAtTimingStart,
      computeCallSerialAtTimingEnd: renderer.info.compute.calls,
      computeCallsDuringTiming,
      timestampPreprime: firstInstanceLiveTimestampPreprime,
      timestampPoolPreprime: firstInstanceLiveTimestampPreprime,
      timestampPoolsAtTimingStart: firstInstanceLiveTimestampPoolsAtTimingStart,
      timestampPoolsAtTimingEnd,
      timestampPoolsStaticExact,
      timestampPoolsResolvedCleanly,
      timestampPreprimeExact,
      timestampPoolsExact,
      warmupScheduleAudit,
      renderTargetTextureUuidAtTimingStart: context.renderTargetTextureUuidAtTimingStart,
      renderTargetTextureUuidAtTimingEnd: firstInstanceCrossoverRenderTarget.texture.uuid,
      renderTargetWidthAtTimingStart: context.renderTargetWidthAtTimingStart,
      renderTargetWidthAtTimingEnd: firstInstanceCrossoverRenderTarget.width,
      renderTargetHeightAtTimingStart: context.renderTargetHeightAtTimingStart,
      renderTargetHeightAtTimingEnd: firstInstanceCrossoverRenderTarget.height,
      renderTargetSamplesAtTimingStart: context.renderTargetSamplesAtTimingStart,
      renderTargetSamplesAtTimingEnd: firstInstanceCrossoverRenderTarget.samples,
      renderTargetDepthBufferAtTimingStart: context.renderTargetDepthBufferAtTimingStart,
      renderTargetDepthBufferAtTimingEnd: firstInstanceCrossoverRenderTarget.depthBuffer,
      cameraViewFnv64AtTimingStart: context.cameraViewFnv64AtTimingStart,
      cameraViewFnv64AtTimingEnd,
      cameraProjectionFnv64AtTimingStart: context.cameraProjectionFnv64AtTimingStart,
      cameraProjectionFnv64AtTimingEnd,
      totalPipelineCacheEntriesAtTimingStart:
        context.totalPipelineCacheEntriesAtTimingStart,
      totalPipelineCacheEntriesAtTimingEnd:
        cacheAtTimingEnd.totalPipelineCacheEntries ?? null,
      computePipelineCacheEntriesAtTimingStart:
        context.computePipelineCacheEntriesAtTimingStart,
      computePipelineCacheEntriesAtTimingEnd:
        cacheAtTimingEnd.computePipelineCacheEntries ?? null,
      computeProgramEntriesExact,
      computeProgramEntriesAtTimingStart:
        context.computeProgramEntriesAtTimingStart,
      computeProgramEntriesAtTimingEnd:
        cacheAtTimingEnd.computeProgramEntries ?? null,
      rendererMemoryExact,
      rendererMemoryAtTimingStart: JSON.parse(context.rendererMemoryAtTimingStart),
      rendererMemoryAtTimingEnd: cacheAtTimingEnd.memory ?? null,
      viewportStateExact,
      viewportStateAtTimingStart: JSON.parse(context.viewportStateAtTimingStart),
      viewportStateAtTimingEnd,
      webgpuUncapturedErrorCountAtTimingStart:
        context.webgpuUncapturedErrorCountAtTimingStart,
      webgpuUncapturedErrorCountAtTimingEnd: webgpuUncapturedErrors.length,
      webgpuUncapturedErrorsDuringTiming,
      expectedTimedFrameCount: timedFrameCount,
    };
  }
  if (context.modeId === FIRST_INSTANCE_CROSSOVER_MODE) {
    const timedFrameCount = FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES
      + FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES;
    const selectorWritesDuringTiming = firstInstanceSelectorWriteSerial
      - context.selectorWriteSerialAtTimingStart;
    const strategySelectionsDuringTiming = (lifecycleDiagnostics?.laneSelectionSerial ?? -1)
      - context.strategySelectionSerialAtTimingStart;
    const renderCallsDuringTiming = renderer.info.render.calls
      - context.renderCallSerialAtTimingStart;
    const computeCallsDuringTiming = renderer.info.compute.calls
      - context.computeCallSerialAtTimingStart;
    const cacheAtTimingEnd = pinnedRendererCacheDiagnostics();
    const cameraViewFnv64AtTimingEnd = fnv1a64Float32(camera.matrixWorldInverse.elements);
    const cameraProjectionFnv64AtTimingEnd = fnv1a64Float32(camera.projectionMatrix.elements);
    const lifecycleAtTimingEnd = firstInstanceStaticLifecycle(lifecycleDiagnostics);
    const lifecycleCommitmentAtTimingEnd = firstInstanceLifecycleCommitment(
      lifecycleDiagnostics,
    );
    const lifecycleExact = firstInstanceLifecycleAtTimingStart !== null
      && JSON.stringify(lifecycleAtTimingEnd)
        === JSON.stringify(firstInstanceLifecycleAtTimingStart);
    const plannedCommandOrder = context.plannedLaneCommandSegmentOrder?.split('|') ?? [];
    const configuredCommandOrderExact = plannedCommandOrder.length
      === FIRST_INSTANCE_CROSSOVER_LANES.length
      && plannedCommandOrder.every(
        (laneId, index) => lifecycleDiagnostics?.laneCommandSegmentOrder?.[index] === laneId,
      );
    const commandSegmentsExact = JSON.stringify(lifecycleDiagnostics?.commandSegments ?? null)
      === context.plannedCommandSegments;
    const bundleRecordCountsExact = FIRST_INSTANCE_CROSSOVER_LANES.every(
      (laneId) => lifecycleDiagnostics?.bundleRecordCounts?.[laneId] === 1,
    );
    return {
      pass: diagnostics?.kind === 'frozen-first-instance-addressing-crossover'
        && diagnostics.activeRenderObjectCount === 1
        && diagnostics.geometryIdentityCount === 2
        && diagnostics.materialIdentityCount === 2
        && diagnostics.commonIndexIdentityCount === 1
        && diagnostics.commandRecordCount === Math.max(2, context.bucketCount) * 2
        && diagnostics.visibleIdsCount === context.objectCount
        && lifecycleDiagnostics?.kind
          === 'first-instance-crossover-static-resource-lifecycle'
        && lifecycleDiagnostics.bundlesPrimed === true
        && lifecycleDiagnostics.allBundlesStatic === true
        && lifecycleDiagnostics.bundleCount === 2
        && lifecycleDiagnostics.meshCount === 2
        && lifecycleDiagnostics.activeRenderObjectCount === 1
        && lifecycleDiagnostics.shaderEvidence?.pass === true
        && bundleRecordCountsExact
        && lifecycleDiagnostics.configuredComputeDispatches === 0
        && lifecycleDiagnostics.configuredComputeSubmissions === 0
        && configuredCommandOrderExact
        && commandSegmentsExact
        && lifecycleExact
        && lifecycleCommitmentAtTimingEnd === context.lifecycleCommitmentAtTimingStart
        && firstInstanceCrossoverRenderTarget.texture.uuid
          === context.renderTargetTextureUuidAtTimingStart
        && firstInstanceCrossoverRenderTarget.width === context.renderTargetWidthAtTimingStart
        && firstInstanceCrossoverRenderTarget.height === context.renderTargetHeightAtTimingStart
        && firstInstanceCrossoverRenderTarget.samples === context.renderTargetSamplesAtTimingStart
        && firstInstanceCrossoverRenderTarget.depthBuffer
          === context.renderTargetDepthBufferAtTimingStart
        && cameraViewFnv64AtTimingEnd === context.cameraViewFnv64AtTimingStart
        && cameraProjectionFnv64AtTimingEnd === context.cameraProjectionFnv64AtTimingStart
        && selectorWritesDuringTiming === timedFrameCount
        && strategySelectionsDuringTiming === timedFrameCount
        && renderCallsDuringTiming === timedFrameCount
        && computeCallsDuringTiming === 0
        && cacheAtTimingEnd.available === true
        && cacheAtTimingEnd.totalPipelineCacheEntries
          === context.totalPipelineCacheEntriesAtTimingStart
        && cacheAtTimingEnd.computePipelineCacheEntries
          === context.computePipelineCacheEntriesAtTimingStart,
      kind: 'first-instance-crossover-static-resource-invariant',
      bundlesPrimed: lifecycleDiagnostics?.bundlesPrimed ?? null,
      bundleStaticFlags: lifecycleDiagnostics?.bundleStaticFlags ?? null,
      allBundlesStatic: lifecycleDiagnostics?.allBundlesStatic ?? null,
      bundleCount: lifecycleDiagnostics?.bundleCount ?? null,
      meshCount: lifecycleDiagnostics?.meshCount ?? null,
      activeRenderObjectCount: lifecycleDiagnostics?.activeRenderObjectCount ?? null,
      bundleRecordCounts: lifecycleDiagnostics?.bundleRecordCounts ?? null,
      shaderEvidencePass: lifecycleDiagnostics?.shaderEvidence?.pass ?? null,
      plannedLaneCommandSegmentOrder: context.plannedLaneCommandSegmentOrder,
      observedLaneCommandSegmentOrder:
        lifecycleDiagnostics?.laneCommandSegmentOrder?.join('|') ?? null,
      configuredCommandOrderExact,
      plannedCommandSegments: context.plannedCommandSegments,
      observedCommandSegments: lifecycleDiagnostics?.commandSegments ?? null,
      commandSegmentsExact,
      representation: diagnostics ? {
        kind: diagnostics.kind,
        activeRenderObjectCount: diagnostics.activeRenderObjectCount,
        geometryIdentityCount: diagnostics.geometryIdentityCount,
        materialIdentityCount: diagnostics.materialIdentityCount,
        commonIndexIdentityCount: diagnostics.commonIndexIdentityCount,
        commandRecordCount: diagnostics.commandRecordCount,
        visibleIdsCount: diagnostics.visibleIdsCount,
      } : null,
      lifecycleExact,
      staticLifecycleAtTimingStart: firstInstanceLifecycleAtTimingStart === null
        ? null
        : structuredClone(firstInstanceLifecycleAtTimingStart),
      staticLifecycleAtTimingEnd: lifecycleAtTimingEnd === null
        ? null
        : structuredClone(lifecycleAtTimingEnd),
      lifecycleCommitmentAtTimingStart: context.lifecycleCommitmentAtTimingStart,
      lifecycleCommitmentAtTimingEnd,
      selectorWriteSerialAtTimingStart: context.selectorWriteSerialAtTimingStart,
      selectorWriteSerialAtTimingEnd: firstInstanceSelectorWriteSerial,
      selectorWritesDuringTiming,
      strategySelectionSerialAtTimingStart: context.strategySelectionSerialAtTimingStart,
      strategySelectionSerialAtTimingEnd:
        lifecycleDiagnostics?.laneSelectionSerial ?? null,
      strategySelectionsDuringTiming,
      renderCallSerialAtTimingStart: context.renderCallSerialAtTimingStart,
      renderCallSerialAtTimingEnd: renderer.info.render.calls,
      renderCallsDuringTiming,
      computeCallSerialAtTimingStart: context.computeCallSerialAtTimingStart,
      computeCallSerialAtTimingEnd: renderer.info.compute.calls,
      computeCallsDuringTiming,
      renderTargetTextureUuidAtTimingStart: context.renderTargetTextureUuidAtTimingStart,
      renderTargetTextureUuidAtTimingEnd: firstInstanceCrossoverRenderTarget.texture.uuid,
      renderTargetWidthAtTimingStart: context.renderTargetWidthAtTimingStart,
      renderTargetWidthAtTimingEnd: firstInstanceCrossoverRenderTarget.width,
      renderTargetHeightAtTimingStart: context.renderTargetHeightAtTimingStart,
      renderTargetHeightAtTimingEnd: firstInstanceCrossoverRenderTarget.height,
      renderTargetSamplesAtTimingStart: context.renderTargetSamplesAtTimingStart,
      renderTargetSamplesAtTimingEnd: firstInstanceCrossoverRenderTarget.samples,
      renderTargetDepthBufferAtTimingStart: context.renderTargetDepthBufferAtTimingStart,
      renderTargetDepthBufferAtTimingEnd: firstInstanceCrossoverRenderTarget.depthBuffer,
      cameraViewFnv64AtTimingStart: context.cameraViewFnv64AtTimingStart,
      cameraViewFnv64AtTimingEnd,
      cameraProjectionFnv64AtTimingStart: context.cameraProjectionFnv64AtTimingStart,
      cameraProjectionFnv64AtTimingEnd,
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
  if (context.modeId === 'fixed-slice-indirect-first-instance') {
    return {
      pass: diagnostics?.kind === 'single-merged-geometry-indirect-first-instance'
        && diagnostics.addressMode === 'indirect-first-instance'
        && diagnostics.hasBucketBaseAttribute === false
        && diagnostics.nonzeroFirstInstanceCount === Math.max(0, context.bucketCount - 1)
        && diagnostics.meshCount === 1
        && diagnostics.geometryIdentityCount === 1
        && diagnostics.materialIdentityCount === 1
        && context.bundleRecordCallbackCountAtTimingStart === 1
        && diagnostics.bundleRecordCallbackCount
          === context.bundleRecordCallbackCountAtTimingStart,
      kind: 'indirect-first-instance-static-bundle-invariant',
      addressMode: diagnostics?.addressMode ?? null,
      hasBucketBaseAttribute: diagnostics?.hasBucketBaseAttribute ?? null,
      nonzeroFirstInstanceCount: diagnostics?.nonzeroFirstInstanceCount ?? null,
      bundleRecordCallbackCountAtTimingStart:
        context.bundleRecordCallbackCountAtTimingStart,
      bundleRecordCallbackCountAtTimingEnd:
        diagnostics?.bundleRecordCallbackCount ?? null,
      meshCount: diagnostics?.meshCount ?? null,
      geometryIdentityCount: diagnostics?.geometryIdentityCount ?? null,
      materialIdentityCount: diagnostics?.materialIdentityCount ?? null,
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
    lastSummary = summary;
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
    lastSummary = null;
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
  if (config.strategyId === FIRST_INSTANCE_CROSSOVER_MODE) {
    config.laneCommandSegmentOrder = [
      ...firstInstanceCrossoverConfiguration.laneCommandSegmentOrder,
    ];
    config.superblockOrientationOffset =
      firstInstanceCrossoverConfiguration.superblockOrientationOffset;
  }
  if (config.strategyId === FIRST_INSTANCE_LIVE_CROSSOVER_MODE) {
    config.lanePhysicalOrder = [
      ...firstInstanceLiveCrossoverConfiguration.lanePhysicalOrder,
    ];
    config.superblockOrientationOffset =
      firstInstanceLiveCrossoverConfiguration.superblockOrientationOffset;
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

function configureFirstInstanceCrossover({
  laneCommandSegmentOrder,
  superblockOrientationOffset,
} = {}) {
  if (rebuilding || validating || trial.active || trial.resolving) {
    throw new Error('First-instance crossover configuration cannot change during active work.');
  }
  if (!Array.isArray(laneCommandSegmentOrder)
    || laneCommandSegmentOrder.length !== FIRST_INSTANCE_CROSSOVER_LANES.length
    || new Set(laneCommandSegmentOrder).size !== FIRST_INSTANCE_CROSSOVER_LANES.length
    || FIRST_INSTANCE_CROSSOVER_LANES.some(
      (laneId) => !laneCommandSegmentOrder.includes(laneId),
    )) {
    throw new RangeError('laneCommandSegmentOrder must be the exact portable/feature pair.');
  }
  if (superblockOrientationOffset !== 0 && superblockOrientationOffset !== 1) {
    throw new RangeError('superblockOrientationOffset must be zero or one.');
  }
  firstInstanceCrossoverConfiguration = Object.freeze({
    laneCommandSegmentOrder: Object.freeze([...laneCommandSegmentOrder]),
    superblockOrientationOffset,
  });
  return selectedConfig();
}

function configureFirstInstanceLiveCrossover({
  lanePhysicalOrder,
  superblockOrientationOffset,
} = {}) {
  if (rebuilding || validating || trial.active || trial.resolving) {
    throw new Error('Live first-instance crossover configuration cannot change during active work.');
  }
  if (!Array.isArray(lanePhysicalOrder)
    || lanePhysicalOrder.length !== FIRST_INSTANCE_LIVE_CROSSOVER_LANES.length
    || new Set(lanePhysicalOrder).size !== FIRST_INSTANCE_LIVE_CROSSOVER_LANES.length
    || FIRST_INSTANCE_LIVE_CROSSOVER_LANES.some(
      (laneId) => !lanePhysicalOrder.includes(laneId),
    )) {
    throw new RangeError('lanePhysicalOrder must be the exact portable/feature pair.');
  }
  if (superblockOrientationOffset !== 0 && superblockOrientationOffset !== 1) {
    throw new RangeError('superblockOrientationOffset must be zero or one.');
  }
  firstInstanceLiveCrossoverConfiguration = Object.freeze({
    lanePhysicalOrder: Object.freeze([...lanePhysicalOrder]),
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
    firstInstanceSelectorWriteSerial = 0;
    firstInstanceLiveSelectorWriteSerial = 0;
    firstInstanceLifecycleAtTimingStart = null;
    firstInstanceLiveLifecycleAtTimingStart = null;
    firstInstanceLiveTimestampPreprime = null;
    firstInstanceLiveTimestampPoolsAtTimingStart = null;
    firstInstanceLiveWarmupTailSnapshot = null;
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
      'fixed-slice-indirect-first-instance': buildIndirectFirstInstanceStrategy,
      'fixed-slice-depth-front-to-back': buildDepthBinnedFrontToBackStrategy,
      'fixed-slice-depth-reverse': buildDepthBinnedReverseStrategy,
      [FROZEN_DEPTH_CROSSOVER_MODE]: buildFrozenDepthCrossoverStrategy,
      [FIRST_INSTANCE_CROSSOVER_MODE]: buildFirstInstanceCrossoverStrategy,
      [FIRST_INSTANCE_LIVE_CROSSOVER_MODE]: buildFirstInstanceLiveCrossoverStrategy,
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
      laneCommandSegmentOrder: config.strategyId === FIRST_INSTANCE_CROSSOVER_MODE
        ? config.laneCommandSegmentOrder
        : undefined,
      lanePhysicalOrder: config.strategyId === FIRST_INSTANCE_LIVE_CROSSOVER_MODE
        ? config.lanePhysicalOrder
        : undefined,
    });
    scene.add(strategy.root);
    if (!isFirstInstanceLiveCrossoverStrategy()) {
      strategy.update(camera, renderer);
      if (strategy.usesCompute) strategy.submitCompute(renderer);
    }
    if (isFirstInstanceCrossoverStrategy()) {
      const previousTarget = renderer.getRenderTarget();
      const previousCubeFace = renderer.getActiveCubeFace();
      const previousMipmapLevel = renderer.getActiveMipmapLevel();
      try {
        // compileAsync and shader capture both derive their render context from
        // the current target, so hold the exact timed target for the full prime.
        renderer.setRenderTarget(firstInstanceCrossoverRenderTarget);
        await strategy.primeBundles({
          scene,
          render: () => renderer.render(scene, camera),
        });
        await strategy.collectShaderSources(scene);
      } finally {
        renderer.setRenderTarget(
          previousTarget,
          previousCubeFace,
          previousMipmapLevel,
        );
      }
    } else if (isFirstInstanceLiveCrossoverStrategy()) {
      const previousTarget = renderer.getRenderTarget();
      const previousCubeFace = renderer.getActiveCubeFace();
      const previousMipmapLevel = renderer.getActiveMipmapLevel();
      try {
        renderer.setRenderTarget(firstInstanceCrossoverRenderTarget);
        await strategy.primeLanes({
          renderer,
          camera,
          scene,
          render: () => renderer.render(scene, camera),
        });
        await strategy.collectShaderSources(scene);
        strategy.setActiveLane(config.lanePhysicalOrder[0]);
        strategy.update(camera, renderer);
        firstInstanceLiveTimestampPreprime = await preprimeTimestampPools(renderer, {
          submitCompute: () => strategy.submitCompute(renderer),
          submitRender: () => renderer.render(scene, camera),
        });
      } finally {
        renderer.setRenderTarget(
          previousTarget,
          previousCubeFace,
          previousMipmapLevel,
        );
      }
    } else {
      await renderer.compileAsync(scene, camera);
    }
    elements['expected-visible'].textContent = `${scenario.expectedVisibleCount.toLocaleString()} / ${scenario.objectCount.toLocaleString()}`;
    elements.validation.textContent = 'not run';
    elements.details.textContent = '';
    lastRows = [];
    lastSummary = null;
    elements.export.disabled = true;
    setStatus('Ready.');
  } finally {
    rebuilding = false;
    setControlsLocked(false);
  }
}

async function runFirstInstanceLiveForcedFeatureOffGate({
  objectCount,
  bucketCount,
  visibilityFraction,
  scenarioSeed,
} = {}) {
  if (rebuilding || validating || trial.active || trial.resolving) {
    throw new Error('Forced-feature-off gate requires an idle disposable page.');
  }
  if (scenarioSeed !== DEVELOPMENT_PROTOCOL.seed) {
    throw new RangeError('Forced-feature-off gate scenario seed differs from the pinned seed.');
  }
  const selections = {
    strategy: 'draw-all',
    objects: String(objectCount),
    buckets: String(bucketCount),
    visibility: String(visibilityFraction),
    layout: 'baseline',
  };
  for (const [id, value] of Object.entries(selections)) {
    const select = elements[id];
    if (!select || ![...select.options].some((option) => option.value === value)) {
      throw new RangeError(`Forced-feature-off gate does not support ${id}=${value}.`);
    }
    select.value = value;
  }
  await rebuild();
  if (scenario.objectCount !== objectCount
    || scenario.bucketCount !== bucketCount
    || scenario.expectedVisibleIds.length !== expectedCpuVisible.length
    || renderer.backend?.trackTimestamp !== false) {
    throw new Error('Forced-feature-off disposable workload did not reach the pinned state.');
  }

  validating = true;
  setControlsLocked(true);
  const uncapturedErrorCountBefore = webgpuUncapturedErrors.length;
  try {
    const evidence = await runForcedFeatureOffGate({
      scenario,
      sourceGeometries,
      renderer,
      camera,
      expectedIds: expectedCpuVisible,
      addressChallenge: async (context) => {
        const laneId = 'portable';
        const commandBuffer = context.strategy.laneState.commandBufferCommitment();
        const oracle = createFirstInstanceAddressChallengeOracle({
          scenario,
          sourceGeometries,
          firstIndexes: context.resources.firstIndexes,
          visibleIdsAttribute: context.resources.visibleIdsAttribute,
          laneIds: [laneId],
          laneDefinitions: {
            [laneId]: {
              addressMode: context.resources.addressMode,
              productionGeometry: context.resources.geometry,
              indirectAttribute: context.resources.indirectAttribute,
              indirectOffsets: context.resources.commandOffsets,
              commandBuffer,
              commandSegment: { index: 0, recordBase: 0, byteBase: 0 },
            },
          },
          namePrefix: 'forced-off-first-instance-fragment-address-challenge',
        });
        try {
          return await oracle.challengeLane(renderer, camera, laneId, {
            visibleIds: context.snapshot.visibleIds,
            activeCounts: context.correctness.actualCounts,
          });
        } finally {
          oracle.dispose();
        }
      },
      captureOutput: (context) => captureExactRenderParity({
        renderer,
        camera,
        strategy: context.strategy,
        expectedIds: context.expectedIds,
      }),
    });
    const uncapturedErrorsDuringGate = webgpuUncapturedErrors.length
      - uncapturedErrorCountBefore;
    return {
      ...evidence,
      pass: evidence.pass
        && uncapturedErrorsDuringGate === 0
        && renderer.backend?.trackTimestamp === false,
      configuration: {
        objectCount,
        bucketCount,
        visibilityFraction,
        scenarioSeed,
        layout: 'baseline',
      },
      timestampTrackingEnabled: renderer.backend?.trackTimestamp === true,
      uncapturedErrorCountBefore,
      uncapturedErrorCountAfter: webgpuUncapturedErrors.length,
      uncapturedErrorsDuringGate,
    };
  } finally {
    validating = false;
    setControlsLocked(false);
  }
}

function createInteractiveShaderObservationRequest(role) {
  const challengeBytes = crypto.getRandomValues(new Uint8Array(32));
  return {
    schemaVersion: 1,
    kind: 'live-first-instance-shader-observation-challenge',
    origin: 'page-interactive',
    runId: null,
    trialId: null,
    planIndex: null,
    repetitionIndex: null,
    phase: 'interactive',
    role,
    captureOrdinal: strategy.shaderObservationSerial + 1,
    challengeNonce: [...challengeBytes]
      .map((value) => value.toString(16).padStart(2, '0'))
      .join(''),
  };
}

async function validateCurrent(shaderObservationRequest = null) {
  if (!strategy || validating || trial.active || trial.resolving) return null;
  validating = true;
  setControlsLocked(true);
  setStatus('Reading back survivor and command buffers…');
  try {
    if (isFirstInstanceLiveCrossoverStrategy()) {
      strategy.update(camera, renderer);
      const result = await strategy.validateSerialized(
        renderer,
        expectedCpuVisible,
        {
          camera,
          scene,
          shaderObservationRequest: shaderObservationRequest
            ?? createInteractiveShaderObservationRequest('main-validation'),
        },
      );
      lastValidation = result;
      elements.validation.textContent = result.pass ? 'PASS' : 'FAIL';
      elements.details.textContent = JSON.stringify(result, null, 2);
      setStatus(result.pass ? 'Validation passed.' : 'Validation failed.');
      return result;
    }
    if (isFrozenCrossoverStrategy()) {
      strategy.setActiveLane(frozenLaneOrder(FROZEN_DEPTH_CROSSOVER_LANES[0]));
    }
    if (isFirstInstanceCrossoverStrategy()) {
      strategy.setActiveLane(FIRST_INSTANCE_CROSSOVER_LANES[0]);
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

function parityIdentity(parity) {
  return ['color', 'depth', 'objectId'].map((channel) => (
    `${parity?.[channel]?.format}|${parity?.[channel]?.arrayType}`
    + `|${parity?.[channel]?.byteLength}|${parity?.[channel]?.sha256}`
  )).join('||');
}

function exactReadbackIdentity(left, right) {
  return left?.format === right?.format
    && left?.arrayType === right?.arrayType
    && left?.byteLength === right?.byteLength
    && left?.sha256 === right?.sha256;
}

async function captureLiveProductionBundleOutput(laneId, directDiagnosticColor) {
  if (!isFirstInstanceLiveCrossoverStrategy()
    || strategy.activeLane !== laneId
    || renderer.autoClear !== true
    || renderer.backend?.trackTimestamp !== false) {
    throw new Error(
      'Production bundle capture requires one active live lane, autoClear, and disabled timestamps.',
    );
  }
  const target = firstInstanceCrossoverRenderTarget;
  if (target.width !== VIEWPORT.width
    || target.height !== VIEWPORT.height
    || target.samples !== 0
    || target.depthBuffer !== true) {
    throw new Error('Production bundle capture target differs from the timed target contract.');
  }
  const captureOnce = async () => {
    const previousTarget = renderer.getRenderTarget();
    const previousCubeFace = renderer.getActiveCubeFace();
    const previousMipmapLevel = renderer.getActiveMipmapLevel();
    try {
      renderer.setRenderTarget(target);
      renderer.render(scene, camera);
    } finally {
      renderer.setRenderTarget(previousTarget, previousCubeFace, previousMipmapLevel);
    }
    const pixels = await renderer.readRenderTargetPixelsAsync(
      target,
      0,
      0,
      VIEWPORT.width,
      VIEWPORT.height,
    );
    return renderReadbackRecord(
      pixels,
      'rgba8unorm',
      VIEWPORT.width * VIEWPORT.height * 4,
    );
  };
  const executionCounters = () => ({
    laneSelectionSerial: strategy.laneSelectionSerial,
    strategyComputeCallSerial: strategy.computeCallSerial,
    strategyPrepareSerial: strategy.prepareSerial,
    rendererComputeCallSerial: renderer.info.compute.calls,
    rendererRenderCallSerial: renderer.info.render.calls,
  });
  const commitmentBefore = strategy.captureTimedRenderCommitment(laneId);
  const executionBefore = executionCounters();
  const first = await captureOnce();
  const commitmentBetween = strategy.captureTimedRenderCommitment(laneId);
  const executionBetween = executionCounters();
  const second = await captureOnce();
  const commitmentAfter = strategy.captureTimedRenderCommitment(laneId);
  const executionAfter = executionCounters();
  const stable = exactReadbackIdentity(first, second);
  const resourcesStable = JSON.stringify(commitmentBefore)
    === JSON.stringify(commitmentBetween)
    && JSON.stringify(commitmentBefore) === JSON.stringify(commitmentAfter);
  const bundleRecordedExactlyOnce = commitmentBefore.bundleRecordCallbackCount === 1
    && commitmentBetween.bundleRecordCallbackCount === 1
    && commitmentAfter.bundleRecordCallbackCount === 1;
  const executionExact = executionBetween.rendererRenderCallSerial
      === executionBefore.rendererRenderCallSerial + 1
    && executionAfter.rendererRenderCallSerial
      === executionBetween.rendererRenderCallSerial + 1
    && ['laneSelectionSerial', 'strategyComputeCallSerial', 'strategyPrepareSerial',
      'rendererComputeCallSerial'].every(
      (field) => executionBefore[field] === executionBetween[field]
        && executionBefore[field] === executionAfter[field],
    );
  const directDiagnosticExact = exactReadbackIdentity(second, directDiagnosticColor);
  return {
    schemaVersion: 1,
    kind: 'live-first-instance-production-bundle-output',
    pass: stable
      && resourcesStable
      && bundleRecordedExactlyOnce
      && executionExact
      && directDiagnosticExact,
    laneId,
    target: {
      textureUuid: target.texture.uuid,
      width: target.width,
      height: target.height,
      samples: target.samples,
      depthBuffer: target.depthBuffer,
    },
    captures: 2,
    color: second,
    directDiagnosticColor,
    directDiagnosticExact,
    resourcesStable,
    bundleRecordedExactlyOnce,
    executionExact,
    executionBefore,
    executionBetween,
    executionAfter,
    commitmentBefore,
    commitmentBetween,
    commitmentAfter,
    stability: {
      pass: stable,
      firstCapture: first,
      secondCapture: second,
    },
  };
}

async function captureRenderParity(shaderObservationRequest = null) {
  if (!strategy || validating || trial.active || trial.resolving) return null;
  validating = true;
  setControlsLocked(true);
  setStatus('Capturing exact color, depth, and object-ID parity…');
  try {
    if (isFirstInstanceLiveCrossoverStrategy()) {
      const previousLane = strategy.activeLane;
      const lanes = {};
      let snapshotValidation;
      try {
        snapshotValidation = await strategy.validateSerialized(
          renderer,
          expectedCpuVisible,
          {
            shaderObservationRequest: shaderObservationRequest
              ?? createInteractiveShaderObservationRequest('render-parity'),
            onLanePrepared: async ({ lane }) => {
              const parity = await captureExactRenderParity({
                renderer,
                camera,
                strategy,
                expectedIds: expectedCpuVisible,
              });
              const productionBundleOutput = await captureLiveProductionBundleOutput(
                lane,
                parity.color,
              );
              const combined = {
                ...parity,
                pass: parity.pass === true && productionBundleOutput.pass === true,
                productionBundleOutput,
              };
              lanes[lane] = combined;
              return combined;
            },
          },
        );
      } finally {
        strategy.setActiveLane(previousLane);
      }
      lastValidation = snapshotValidation;
      if (snapshotValidation?.pass !== true) {
        throw new Error('Live render parity refused because paired snapshots failed validation.');
      }
      const identities = FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map(
        (laneId) => parityIdentity(lanes[laneId]),
      );
      const crossLaneExact = identities[0] === identities[1];
      const crossLaneProductionExact = exactReadbackIdentity(
        lanes[FIRST_INSTANCE_LIVE_CROSSOVER_LANES[0]]?.productionBundleOutput?.color,
        lanes[FIRST_INSTANCE_LIVE_CROSSOVER_LANES[1]]?.productionBundleOutput?.color,
      );
      const result = {
        schemaVersion: 1,
        kind: 'first-instance-live-crossover-exact-render-parity',
        pass: crossLaneExact
          && crossLaneProductionExact
          && FIRST_INSTANCE_LIVE_CROSSOVER_LANES.every(
            (laneId) => lanes[laneId]?.pass === true,
          ),
        laneIds: [...FIRST_INSTANCE_LIVE_CROSSOVER_LANES],
        crossLaneExact,
        crossLaneProductionExact,
        lanes,
        snapshotValidation,
      };
      setStatus(result.pass ? 'Render parity captured.' : 'Render parity capture was unstable.');
      return result;
    }
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
        (laneId) => parityIdentity(lanes[laneId]),
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
    if (isFirstInstanceCrossoverStrategy()) {
      const previousLane = strategy.activeLane;
      const lanes = {};
      try {
        for (const laneId of FIRST_INSTANCE_CROSSOVER_LANES) {
          strategy.setActiveLane(laneId);
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
      const identities = FIRST_INSTANCE_CROSSOVER_LANES.map(
        (laneId) => parityIdentity(lanes[laneId]),
      );
      const crossLaneExact = identities[0] === identities[1];
      const result = {
        schemaVersion: 1,
        kind: 'first-instance-crossover-exact-render-parity',
        pass: crossLaneExact
          && FIRST_INSTANCE_CROSSOVER_LANES.every(
            (laneId) => lanes[laneId]?.pass === true,
          ),
        laneIds: [...FIRST_INSTANCE_CROSSOVER_LANES],
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

async function startTrial(extraContext = {}, shaderObservationRequest = null) {
  if (!strategy || trial.active || trial.resolving) return;
  const validation = await validateCurrent(shaderObservationRequest);
  if (!validation?.pass) throw new Error('Timing refused because validation did not pass.');
  setControlsLocked(true);
  const config = selectedConfig();
  const workload = await fingerprintWorkload();
  const representationDiagnostics = strategy.diagnostics?.() ?? null;
  const depthCrossover = strategy.id === FROZEN_DEPTH_CROSSOVER_MODE;
  const firstInstanceCrossover = strategy.id === FIRST_INSTANCE_CROSSOVER_MODE;
  const liveFirstInstanceCrossover = strategy.id === FIRST_INSTANCE_LIVE_CROSSOVER_MODE;
  const renderOnlyCrossover = depthCrossover || firstInstanceCrossover;
  const pinnedCrossover = renderOnlyCrossover || liveFirstInstanceCrossover;
  const timingDiagnostics = pinnedCrossover
    ? strategy.lifecycleDiagnostics?.() ?? representationDiagnostics
    : DEPTH_ORDERING_LAYOUT_IDS.has(config.layout)
    ? strategy.lifecycleDiagnostics?.() ?? representationDiagnostics
    : representationDiagnostics;
  const frontOrder = frozenLaneOrder(FROZEN_DEPTH_CROSSOVER_LANES[0]);
  const reverseOrder = frozenLaneOrder(FROZEN_DEPTH_CROSSOVER_LANES[1]);
  if (liveFirstInstanceCrossover) {
    bindLiveFirstInstanceViewportState(firstInstanceCrossoverRenderTarget);
  }
  const liveViewportStateAtTimingStart = liveFirstInstanceCrossover
    ? captureLiveFirstInstanceViewportState(firstInstanceCrossoverRenderTarget)
    : null;
  if (liveFirstInstanceCrossover
    && !liveFirstInstanceViewportStateIsPinned(
      liveViewportStateAtTimingStart,
      firstInstanceCrossoverRenderTarget,
    )) {
    throw new Error('Live crossover viewport/scissor state is not exactly pinned.');
  }
  const cacheAtTimingStart = pinnedCrossover ? pinnedRendererCacheDiagnostics() : null;
  if (pinnedCrossover && cacheAtTimingStart?.available !== true) {
    throw new Error('Pinned crossover requires renderer cache diagnostics.');
  }
  const timingRenderTarget = firstInstanceCrossover || liveFirstInstanceCrossover
    ? firstInstanceCrossoverRenderTarget
    : depthCrossover
      ? frozenCrossoverRenderTarget
      : null;
  firstInstanceLifecycleAtTimingStart = firstInstanceCrossover
    ? structuredClone(firstInstanceStaticLifecycle(timingDiagnostics))
    : null;
  const firstInstanceLifecycleCommitmentAtTimingStart = firstInstanceCrossover
    ? firstInstanceLifecycleCommitment(timingDiagnostics)
    : null;
  firstInstanceLiveLifecycleAtTimingStart = liveFirstInstanceCrossover
    ? structuredClone(firstInstanceLiveStaticLifecycle(timingDiagnostics))
    : null;
  const firstInstanceLiveLifecycleCommitmentAtTimingStart = liveFirstInstanceCrossover
    ? firstInstanceLiveLifecycleCommitment(timingDiagnostics)
    : null;
  firstInstanceLiveTimestampPoolsAtTimingStart = liveFirstInstanceCrossover
    ? structuredClone(timestampPoolDiagnostics(renderer))
    : null;
  if (liveFirstInstanceCrossover) {
    const preprimeAfter = firstInstanceLiveTimestampPreprime?.after;
    const timingPools = firstInstanceLiveTimestampPoolsAtTimingStart;
    const preprimeCommitment = timestampPoolStaticCommitment(preprimeAfter);
    const timingCommitment = timestampPoolStaticCommitment(timingPools);
    const poolsClean = ['render', 'compute'].every((type) => {
      const pool = timingPools?.[type];
      return pool
        && pool.currentQueryIndex === 0
        && pool.queryOffsetCount === 0
        && pool.pendingResolve === false
        && pool.isDisposed === false
        && pool.resultBufferMapState === 'unmapped'
        && pool.maxQueries >= FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES * 2;
    });
    if (firstInstanceLiveTimestampPreprime?.kind !== 'three-r185-timestamp-pool-preprime'
      || firstInstanceLiveTimestampPreprime.addedTimestampUidCount?.render !== 1
      || firstInstanceLiveTimestampPreprime.addedTimestampUidCount?.compute !== 1
      || timingPools?.backendTrackingEnabled !== false
      || JSON.stringify(preprimeCommitment) !== JSON.stringify(timingCommitment)
      || !poolsClean) {
      throw new Error('Live crossover timestamp pools were not cleanly pre-primed.');
    }
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
    configuredDrawCommands: strategy.configuredDrawCommands,
    configuredRenderObjects: strategy.configuredRenderObjects,
    configuredComputeDispatches: strategy.configuredComputeDispatches,
    configuredComputeSubmissions: strategy.configuredComputeSubmissions,
    configuredSubmittedInstances: strategy.configuredSubmittedInstances,
    bundleRecordCallbackCountAtTimingStart:
      timingDiagnostics?.bundleRecordCallbackCount ?? null,
    timestampAvailable,
    expectedComputeTimestampUidCount: liveFirstInstanceCrossover ? 1 : null,
    strictTimestampUidAttribution: liveFirstInstanceCrossover,
    expectedRenderTimestampUidCount: pinnedCrossover ? 1 : null,
    plannedLaneStorageOrder: depthCrossover
      ? config.laneStorageOrder.join('|')
      : null,
    plannedLaneCommandSegmentOrder: firstInstanceCrossover
      ? config.laneCommandSegmentOrder.join('|')
      : null,
    plannedLanePhysicalOrder: liveFirstInstanceCrossover
      ? config.lanePhysicalOrder.join('|')
      : null,
    lanePhysicalOrder: liveFirstInstanceCrossover
      ? config.lanePhysicalOrder.join('|')
      : null,
    plannedCommandSegments: firstInstanceCrossover
      ? JSON.stringify(strategy.commandSegments)
      : null,
    superblockOrientationOffset: pinnedCrossover
      ? config.superblockOrientationOffset
      : null,
    frontLaneBase: depthCrossover ? strategy.laneOffsets[frontOrder] : null,
    reverseLaneBase: depthCrossover ? strategy.laneOffsets[reverseOrder] : null,
    selectorWriteSerialAtTimingStart: depthCrossover
      ? frozenSelectorWriteSerial
      : firstInstanceCrossover
        ? firstInstanceSelectorWriteSerial
        : liveFirstInstanceCrossover
          ? firstInstanceLiveSelectorWriteSerial
        : null,
    strategySelectionSerialAtTimingStart: firstInstanceCrossover
      || liveFirstInstanceCrossover
      ? strategy.laneSelectionSerial
      : null,
    strategyComputeCallSerialAtTimingStart: liveFirstInstanceCrossover
      ? strategy.computeCallSerial
      : null,
    strategyPrepareSerialAtTimingStart: liveFirstInstanceCrossover
      ? strategy.prepareSerial
      : null,
    commandBufferCommitmentsAtTimingStart: liveFirstInstanceCrossover
      ? JSON.stringify(strategy.commandBufferCommitments)
      : null,
    portableCommandBufferIdAtTimingStart: liveFirstInstanceCrossover
      ? strategy.commandBufferCommitments[FIRST_INSTANCE_LIVE_CROSSOVER_LANES[0]].attributeId
      : null,
    featureCommandBufferIdAtTimingStart: liveFirstInstanceCrossover
      ? strategy.commandBufferCommitments[FIRST_INSTANCE_LIVE_CROSSOVER_LANES[1]].attributeId
      : null,
    lifecycleCommitmentAtTimingStart: liveFirstInstanceCrossover
      ? firstInstanceLiveLifecycleCommitmentAtTimingStart
      : firstInstanceLifecycleCommitmentAtTimingStart,
    rootUuidAtTimingStart: firstInstanceCrossover ? timingDiagnostics.rootUuid : null,
    rootVersionAtTimingStart: firstInstanceCrossover ? timingDiagnostics.rootVersion : null,
    bundleUuidsAtTimingStart: firstInstanceCrossover
      ? FIRST_INSTANCE_CROSSOVER_LANES.map(
        (laneId) => timingDiagnostics.bundleUuids[laneId],
      ).join('|')
      : null,
    bundleVersionsAtTimingStart: firstInstanceCrossover
      ? FIRST_INSTANCE_CROSSOVER_LANES.map(
        (laneId) => timingDiagnostics.bundleVersions[laneId],
      ).join('|')
      : null,
    meshUuidsAtTimingStart: firstInstanceCrossover
      ? FIRST_INSTANCE_CROSSOVER_LANES.map(
        (laneId) => timingDiagnostics.meshUuids[laneId],
      ).join('|')
      : null,
    geometryUuidsAtTimingStart: firstInstanceCrossover
      ? FIRST_INSTANCE_CROSSOVER_LANES.map(
        (laneId) => timingDiagnostics.geometryUuids[laneId],
      ).join('|')
      : null,
    materialUuidsAtTimingStart: firstInstanceCrossover
      ? FIRST_INSTANCE_CROSSOVER_LANES.map(
        (laneId) => timingDiagnostics.materialUuids[laneId],
      ).join('|')
      : null,
    commonAttributeIdsAtTimingStart: firstInstanceCrossover
      ? JSON.stringify(timingDiagnostics.commonAttributeIds)
      : null,
    commonAttributeVersionsAtTimingStart: firstInstanceCrossover
      ? JSON.stringify(timingDiagnostics.commonAttributeVersions)
      : null,
    indexAttributeIdAtTimingStart: firstInstanceCrossover
      ? timingDiagnostics.indexAttributeId
      : null,
    indexAttributeVersionAtTimingStart: firstInstanceCrossover
      ? timingDiagnostics.indexAttributeVersion
      : null,
    bucketBaseAttributeIdAtTimingStart: firstInstanceCrossover
      ? timingDiagnostics.bucketBaseAttributeId
      : null,
    bucketBaseAttributeVersionAtTimingStart: firstInstanceCrossover
      ? timingDiagnostics.bucketBaseAttributeVersion
      : null,
    shaderPortableVertexSha256AtTimingStart: firstInstanceCrossover
      ? timingDiagnostics.shaderEvidence?.portableVertexSha256 ?? null
      : null,
    shaderFeatureVertexSha256AtTimingStart: firstInstanceCrossover
      ? timingDiagnostics.shaderEvidence?.featureVertexSha256 ?? null
      : null,
    shaderNormalizedVertexSha256AtTimingStart: firstInstanceCrossover
      ? timingDiagnostics.shaderEvidence?.normalizedVertexSha256 ?? null
      : null,
    shaderFragmentSha256AtTimingStart: firstInstanceCrossover
      ? timingDiagnostics.shaderEvidence?.fragmentSha256 ?? null
      : null,
    renderCallSerialAtTimingStart: pinnedCrossover
      ? renderer.info.render.calls
      : null,
    computeCallSerialAtTimingStart: pinnedCrossover
      ? renderer.info.compute.calls
      : null,
    bundleGroupUuidAtTimingStart:
      depthCrossover ? timingDiagnostics.bundleGroupUuid : null,
    meshUuidAtTimingStart: depthCrossover ? timingDiagnostics.meshUuid : null,
    geometryUuidAtTimingStart: depthCrossover ? timingDiagnostics.geometryUuid : null,
    materialUuidAtTimingStart: depthCrossover ? timingDiagnostics.materialUuid : null,
    matrixAttributeIdAtTimingStart:
      renderOnlyCrossover ? timingDiagnostics.matrixAttributeId : null,
    visibleIdsAttributeIdAtTimingStart:
      renderOnlyCrossover ? timingDiagnostics.visibleIdsAttributeId : null,
    indirectAttributeIdAtTimingStart:
      renderOnlyCrossover ? timingDiagnostics.indirectAttributeId : null,
    selectorChallengeAttributeIdAtTimingStart:
      depthCrossover ? timingDiagnostics.selectorChallengeAttributeId : null,
    bundleGroupVersionAtTimingStart:
      depthCrossover ? timingDiagnostics.bundleGroupVersion : null,
    matrixAttributeVersionAtTimingStart:
      renderOnlyCrossover ? timingDiagnostics.matrixAttributeVersion : null,
    visibleIdsAttributeVersionAtTimingStart:
      renderOnlyCrossover ? timingDiagnostics.visibleIdsAttributeVersion : null,
    indirectAttributeVersionAtTimingStart:
      renderOnlyCrossover ? timingDiagnostics.indirectAttributeVersion : null,
    selectorUniformUuidAtTimingStart:
      depthCrossover ? timingDiagnostics.selectorUniformUuid : null,
    renderTargetTextureUuidAtTimingStart:
      pinnedCrossover ? timingRenderTarget.texture.uuid : null,
    renderTargetWidthAtTimingStart:
      pinnedCrossover ? timingRenderTarget.width : null,
    renderTargetHeightAtTimingStart:
      pinnedCrossover ? timingRenderTarget.height : null,
    renderTargetSamplesAtTimingStart:
      pinnedCrossover ? timingRenderTarget.samples : null,
    renderTargetDepthBufferAtTimingStart:
      pinnedCrossover ? timingRenderTarget.depthBuffer : null,
    cameraViewFnv64AtTimingStart:
      pinnedCrossover ? fnv1a64Float32(camera.matrixWorldInverse.elements) : null,
    cameraProjectionFnv64AtTimingStart:
      pinnedCrossover ? fnv1a64Float32(camera.projectionMatrix.elements) : null,
    totalPipelineCacheEntriesAtTimingStart:
      pinnedCrossover ? cacheAtTimingStart.totalPipelineCacheEntries : null,
    computePipelineCacheEntriesAtTimingStart:
      pinnedCrossover ? cacheAtTimingStart.computePipelineCacheEntries : null,
    computeProgramEntriesAtTimingStart:
      liveFirstInstanceCrossover ? cacheAtTimingStart.computeProgramEntries : null,
    rendererMemoryAtTimingStart: liveFirstInstanceCrossover
      ? JSON.stringify(cacheAtTimingStart.memory)
      : null,
    viewportStateAtTimingStart: liveFirstInstanceCrossover
      ? JSON.stringify(liveViewportStateAtTimingStart)
      : null,
    timestampPoolStaticCommitmentAtTimingStart: liveFirstInstanceCrossover
      ? JSON.stringify(timestampPoolStaticCommitment(firstInstanceLiveTimestampPoolsAtTimingStart))
      : null,
    webgpuUncapturedErrorCountAtTimingStart: liveFirstInstanceCrossover
      ? webgpuUncapturedErrors.length
      : null,
  };
  if (liveFirstInstanceCrossover) {
    firstInstanceLivePreviousLaneId = null;
    firstInstanceLivePreviousPreviousLaneId = null;
    firstInstanceLiveWarmupTailSnapshot = null;
  }
  await trial.start(
    trialContext,
    depthCrossover
      ? {
        warmupFrames: FROZEN_CROSSOVER_WARMUP_FRAMES,
        measuredFrames: FROZEN_CROSSOVER_MEASURED_FRAMES,
      }
      : firstInstanceCrossover
        ? {
          warmupFrames: FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
          measuredFrames: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
        }
        : liveFirstInstanceCrossover
          ? {
            warmupFrames: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
            measuredFrames: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
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
  // The live candidate has explicit priming, validation, and timed submissions.
  // Do not let the interactive idle loop add unrecorded compute/render work
  // between those evidence boundaries.
  if (isFirstInstanceLiveCrossoverStrategy() && !trial.active) return;
  const frameStart = performance.now();
  if (isFirstInstanceLiveCrossoverStrategy() && trial.active) {
    const descriptor = trial.frameDescriptor;
    if (!descriptor) {
      throw new Error('Live first-instance crossover lacks an active-frame descriptor.');
    }
    const scheduled = firstInstanceLiveCrossoverFrame(
      descriptor.phaseFrameIndex,
      trial.context.superblockOrientationOffset,
    );
    const previousLaneId = firstInstanceLivePreviousLaneId;
    const previousPreviousLaneId = firstInstanceLivePreviousPreviousLaneId;
    const historyMustBeCyclic = descriptor.phase === 'measure'
      || descriptor.phaseFrameIndex >= 2;
    if (historyMustBeCyclic
      && (previousLaneId !== scheduled.previousLaneId
        || previousPreviousLaneId !== scheduled.previousPreviousLaneId)) {
      throw new Error(
        'Live crossover actual lane history differs from its cyclic schedule.',
      );
    }
    const selectionSerialBefore = strategy.laneSelectionSerial;
    const commonStart = performance.now();
    const commandBufferId = strategy.setActiveLane(scheduled.laneId);
    firstInstanceLiveSelectorWriteSerial += 1;
    const selectorWriteSerial = firstInstanceLiveSelectorWriteSerial;
    const strategySelectionSerial = strategy.laneSelectionSerial;
    strategy.update(camera, renderer);
    const cpuCommonUpdateMs = performance.now() - commonStart;
    const commandBuffer = strategy.commandBufferCommitments[scheduled.laneId];
    if (strategySelectionSerial !== selectionSerialBefore + 1) {
      throw new Error('Live crossover must perform exactly one lane selection per frame.');
    }
    if (!commandBuffer
      || commandBufferId !== commandBuffer.attributeId
      || strategy.activeCommandBufferId !== commandBuffer.attributeId
      || commandBuffer.byteOffset !== 0
      || commandBuffer.firstOffset !== 0) {
      throw new Error('Live crossover selected an invalid zero-offset command buffer.');
    }
    const gpuFrameId = renderer.info.frame;
    const rendererComputeCallSerialBefore = renderer.info.compute.calls;
    const strategyComputeCallSerialBefore = strategy.computeCallSerial;
    const computeStart = performance.now();
    const computeSubmission = strategy.submitCompute(renderer);
    const cpuComputeSubmitMs = performance.now() - computeStart;
    const computeCallSerial = renderer.info.compute.calls;
    const computeFrameCallIndex = renderer.info.compute.frameCalls;
    const strategyComputeCallSerial = strategy.computeCallSerial;
    if (computeCallSerial !== rendererComputeCallSerialBefore + 1
      || strategyComputeCallSerial !== strategyComputeCallSerialBefore + 1) {
      throw new Error('Live crossover must issue exactly one top-level compute call per frame.');
    }
    const renderCallSerialBefore = renderer.info.render.calls;
    const renderStart = performance.now();
    renderBenchmarkScene();
    const cpuRenderSubmitMs = performance.now() - renderStart;
    const renderCallSerial = renderer.info.render.calls;
    const renderFrameCallIndex = renderer.info.render.frameCalls;
    if (renderCallSerial !== renderCallSerialBefore + 1) {
      throw new Error('Live crossover must issue exactly one top-level render call per frame.');
    }
    const cpuSubmitTotalMs = cpuComputeSubmitMs + cpuRenderSubmitMs;
    const cpuFrameBodyMs = performance.now() - frameStart;
    trial.recordFrame({
      gpuFrameId,
      phaseFrameIndex: descriptor.phaseFrameIndex,
      crossoverBlockIndex: scheduled.crossoverBlockIndex,
      withinBlockPosition: scheduled.withinBlockPosition,
      crossoverPattern: scheduled.pattern,
      crossoverPatternIndex: scheduled.patternIndex,
      previousPreviousLaneId,
      previousLaneId,
      laneId: scheduled.laneId,
      commandBufferId,
      submittedComputeLaneId: computeSubmission.laneId,
      computeTimestampContextId: computeSubmission.timestampContextId,
      computeGroupIdentity: computeSubmission.computeGroupIdentity,
      computeTimestampRegistrationSerial: computeSubmission.registrationSerial,
      computeTimestampBackendIdentity: computeSubmission.backendIdentity,
      computeTimestampBackendWrapperIdentity: computeSubmission.backendWrapperIdentity,
      submittedComputeNodeIds: JSON.stringify(computeSubmission.computeNodeIds),
      commandSegmentIndex: 0,
      commandRecordBase: 0,
      commandByteBase: 0,
      commandByteOffset: 0,
      commandBufferRecordCount: commandBuffer.recordCount,
      commandBufferByteLength: commandBuffer.byteLength,
      selectorWriteSerial,
      strategySelectionSerial,
      strategyComputeCallSerial,
      computeCallSerial,
      computeFrameCallIndex,
      renderCallSerial,
      renderFrameCallIndex,
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
    firstInstanceLivePreviousPreviousLaneId = previousLaneId;
    firstInstanceLivePreviousLaneId = scheduled.laneId;
    if (descriptor.phase === 'warmup'
      && descriptor.phaseFrameIndex === FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES - 1) {
      firstInstanceLiveWarmupTailSnapshot = Object.freeze({
        schemaVersion: 1,
        kind: 'live-first-instance-post-warmup-state',
        previousPreviousLaneId: firstInstanceLivePreviousPreviousLaneId,
        previousLaneId: firstInstanceLivePreviousLaneId,
        lastWarmupGpuFrameId: gpuFrameId,
        selectorWriteSerial,
        strategySelectionSerial,
        strategyComputeCallSerial,
        computeCallSerial,
        renderCallSerial,
      });
    }
    return;
  }
  if (isFirstInstanceCrossoverStrategy() && trial.active) {
    const descriptor = trial.frameDescriptor;
    if (!descriptor) throw new Error('First-instance crossover lacks an active-frame descriptor.');
    const scheduled = firstInstanceCrossoverFrame(
      descriptor.phaseFrameIndex,
      trial.context.superblockOrientationOffset,
    );
    const selectionSerialBefore = strategy.laneSelectionSerial;
    const commonStart = performance.now();
    const commandByteBase = strategy.setActiveLane(scheduled.laneId);
    firstInstanceSelectorWriteSerial += 1;
    const selectorWriteSerial = firstInstanceSelectorWriteSerial;
    const strategySelectionSerial = strategy.laneSelectionSerial;
    const commandSegment = strategy.commandSegments[scheduled.laneId];
    if (strategySelectionSerial !== selectionSerialBefore + 1) {
      throw new Error('First-instance crossover must perform exactly one lane selection per frame.');
    }
    if (!commandSegment || commandByteBase !== commandSegment.byteBase) {
      throw new Error('First-instance crossover selected the wrong physical command segment.');
    }
    const cpuCommonUpdateMs = performance.now() - commonStart;
    const gpuFrameId = renderer.info.frame;
    const renderCallSerialBefore = renderer.info.render.calls;
    const renderStart = performance.now();
    renderBenchmarkScene();
    const cpuRenderSubmitMs = performance.now() - renderStart;
    const renderCallSerial = renderer.info.render.calls;
    if (renderCallSerial !== renderCallSerialBefore + 1) {
      throw new Error(
        'First-instance crossover must issue exactly one top-level render call per frame.',
      );
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
      commandSegmentIndex: commandSegment.index,
      commandRecordBase: commandSegment.recordBase,
      commandByteBase,
      selectorWriteSerial,
      strategySelectionSerial,
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
  get webgpuUncapturedErrorCount() { return webgpuUncapturedErrors.length; },
  get webgpuValidationErrorCount() {
    return webgpuUncapturedErrors.filter((error) => error.name === 'GPUValidationError').length;
  },
  get webgpuDeviceLossCount() { return webgpuDeviceLosses.length; },
  crossOriginIsolated: globalThis.crossOriginIsolated === true,
  performanceNowQuantumMs: detectPerformanceNowQuantum(),
  viewport: VIEWPORT,
});

window.__WEBGPU_BENCH__ = {
  get ready() { return Boolean(strategy) && !rebuilding; },
  environment,
  selectedConfig,
  configureFrozenCrossover,
  configureFirstInstanceCrossover,
  configureFirstInstanceLiveCrossover,
  runFirstInstanceLiveForcedFeatureOffGate,
  rebuild,
  validate: validateCurrent,
  captureRenderParity,
  startTrial,
  fingerprintWorkload,
  cacheDiagnostics: pinnedRendererCacheDiagnostics,
  get geometryManifest() { return sourceGeometryManifest; },
  get scenarioManifest() { return sourceScenarioManifest; },
  get strategyDiagnostics() { return strategy?.diagnostics?.() ?? null; },
  get strategyLifecycle() { return strategy?.lifecycleDiagnostics?.() ?? null; },
  get firstInstanceShaderEvidence() {
    return isAnyFirstInstanceCrossoverStrategy() ? strategy.shaderEvidence : null;
  },
  get timestampPoolDiagnostics() { return timestampPoolDiagnostics(renderer); },
  get timestampPoolPreprime() { return firstInstanceLiveTimestampPreprime; },
  get webgpuUncapturedErrors() { return structuredClone(webgpuUncapturedErrors); },
  get webgpuDeviceLosses() { return structuredClone(webgpuDeviceLosses); },
  get lastValidation() { return lastValidation; },
  get phase() { return trial.phase; },
  get trialError() { return trial.error?.message ?? null; },
  get rows() { return lastRows; },
  get summary() { return lastSummary; },
};

elements.backend.textContent = `${adapterInfo.description ?? adapterInfo.device ?? 'WebGPU'} · ${adapterInfo.backend ?? 'unknown backend'}`;
renderer.setAnimationLoop(frame);
await rebuild();
