import * as THREE from 'three/webgpu';
import WebGPU from 'three/addons/capabilities/WebGPU.js';
import {
  CAMERA,
  DEVELOPMENT_PROTOCOL,
  THREE_REVISION,
  VIEWPORT,
} from './config.js';
import { TrialController } from './benchmark/trial-controller.js';
import { downloadRows } from './lib/csv.js';
import {
  fingerprintFixedSubsetScenario,
  fingerprintGeometryFixtures,
} from './scenes/geometry-fingerprints.js';
import { createIndexedGeometryFixtures } from './scenes/geometry-fixtures.js';
import { createFixedSubsetScenario } from './scenes/fixed-subsets.js';
import { buildDrawAllStrategy } from './strategies/draw-all.js';
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
import './styles.css';

const elements = Object.fromEntries([
  'strategy', 'objects', 'buckets', 'visibility', 'rebuild', 'validate', 'trial', 'export',
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

function validateTrialCompletion(context) {
  const diagnostics = strategy?.diagnostics?.() ?? null;
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
  return {
    strategyId: elements.strategy.value,
    objectCount: Number(elements.objects.value),
    bucketCount: Number(elements.buckets.value),
    visibilityFraction: Number(elements.visibility.value),
  };
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
  for (const id of ['strategy', 'objects', 'buckets', 'visibility', 'rebuild', 'validate', 'trial']) {
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
    const config = selectedConfig();
    sourceGeometries = createIndexedGeometryFixtures(config.bucketCount, 'medium');
    sourceGeometryManifest = await fingerprintGeometryFixtures(sourceGeometries, 'medium');
    scenario = createFixedSubsetScenario({
      objectCount: config.objectCount,
      bucketCount: config.bucketCount,
      visibilityFraction: config.visibilityFraction,
      geometrySpheres: sourceGeometries.map((geometry) => geometry.boundingSphere.clone()),
      seed: DEVELOPMENT_PROTOCOL.seed,
    });
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
      'fixed-slice-per-bucket': buildFixedSlicePerBucketStrategy,
      'three-blocks-coalesced': buildThreeBlocksCoalescedStrategy,
      'three-blocks-current': buildThreeBlocksCurrentStrategy,
      'three-blocks-historical': buildThreeBlocksHistoricalStrategy,
    };
    const builder = builders[config.strategyId];
    if (!builder) throw new Error(`Unknown strategy: ${config.strategyId}`);
    strategy = builder({ scenario, sourceGeometries, renderer });
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
    strategy.update(camera, renderer);
    if (strategy.usesCompute) strategy.submitCompute(renderer);
    renderer.render(scene, camera);
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

async function startTrial(extraContext = {}) {
  if (!strategy || trial.active || trial.resolving) return;
  const validation = await validateCurrent();
  if (!validation?.pass) throw new Error('Timing refused because validation did not pass.');
  setControlsLocked(true);
  const config = selectedConfig();
  const workload = await fingerprintWorkload();
  const representationDiagnostics = strategy.diagnostics?.() ?? null;
  await trial.start({
    ...extraContext,
    schemaVersion: 2,
    modeId: strategy.id,
    objectCount: scenario.objectCount,
    bucketCount: scenario.bucketCount,
    targetVisibilityFraction: config.visibilityFraction,
    expectedVisibleCount: scenario.expectedVisibleCount,
    validationKind: validation.kind,
    validationPass: true,
    usesCompute: strategy.usesCompute,
    configuredRenderObjects: strategy.configuredRenderObjects,
    configuredComputeDispatches: strategy.configuredComputeDispatches,
    configuredComputeSubmissions: strategy.configuredComputeSubmissions,
    bundleRecordCallbackCountAtTimingStart:
      representationDiagnostics?.bundleRecordCallbackCount ?? null,
    timestampAvailable,
  });
  return {
    validation: structuredClone(validation),
    workload,
  };
}

function frame() {
  if (rebuilding || validating || !strategy || trial.resolving) return;
  const frameStart = performance.now();
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
  renderer.render(scene, camera);
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
  reversedDepth: camera.reversedDepth,
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
  rebuild,
  validate: validateCurrent,
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
