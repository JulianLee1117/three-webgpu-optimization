import { createHash, randomBytes } from 'node:crypto';
import {
  access,
  mkdir,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import {
  brotliCompress,
  constants as zlibConstants,
} from 'node:zlib';

import { chromium } from 'playwright-core';
import { createServer } from 'vite';

import {
  summarizeFirstInstanceLiveOrderFactorial,
} from '../analysis/first-instance-live-order-factorial-summary.mjs';
import {
  summarizeLiveFirstInstanceTrialRows,
} from '../analysis/live-first-instance-crossover-summary.mjs';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
} from '../src/benchmark/first-instance-live-crossover-schedule.js';
import {
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_KIND,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_COUNT,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIAL_COUNT,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIALS_PER_SESSION,
  buildFirstInstanceLiveOrderFactorialPlan,
} from '../src/benchmark/first-instance-live-order-factorial-plan.js';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
  buildFirstInstanceLiveCrossoverPlan,
} from '../src/benchmark/plan.js';
import {
  collectExecutionDependencyClosure,
  createCandidateViteRuntimeGuard,
  executionDependencyClosuresMatch,
} from './execution-dependency-closure.mjs';
import {
  liveFirstInstanceCrossoverScheduleSha256,
  validateLiveFirstInstanceForcedFeatureOffGate,
  validateLiveFirstInstanceTrialEvidence,
} from './live-first-instance-evidence-validation.mjs';
import {
  collectSourceProvenance,
  sourceProvenanceMatches,
} from './source-provenance.mjs';

const brotliCompressAsync = promisify(brotliCompress);
const OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;
const VISIBILITY_FRACTION = 0.99;
const LAYOUT = 'baseline';
const SCENARIO_SEED = 0xb1ad_2026;
const MAXIMUM_CPU_TIMER_QUANTUM_MS = 0.01;
const SETUP_PRIME_TOPOLOGY = 'staged-order-factorial-v1';
const COMPATIBILITY_MAPPING = 'candidate-high-visibility-c-orientation-v1';
const BROWSER_OPERATION_TIMEOUT_MS = 180_000;
const BROWSER_CLOSE_TIMEOUT_MS = 60_000;
const SERVER_OPERATION_TIMEOUT_MS = 120_000;
const BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
]);
const ISOLATION_HEADERS = Object.freeze({
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
});
const runnerArguments = process.argv.slice(2);
if (runnerArguments.some((argument) => argument !== '--smoke')
  || runnerArguments.filter((argument) => argument === '--smoke').length > 1) {
  throw new Error('Usage: node scripts/run-first-instance-live-order-factorial.mjs [--smoke]');
}
const smokeMode = runnerArguments[0] === '--smoke';
const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runTimestamp = new Date().toISOString().replaceAll(':', '-').replaceAll('.', '-');
const runId = `first-instance-live-order-factorial${smokeMode ? '-smoke' : ''}`
  + `-${runTimestamp}-${randomBytes(4).toString('hex')}`;
const developmentRoot = path.join(
  projectRoot,
  'results',
  'development',
  'first-instance-live-order-factorial',
);
const runDirectory = path.join(developmentRoot, runId);
const trialDirectory = path.join(runDirectory, 'trials');
let terminationSignal = null;

function terminationError() {
  const error = new Error(`Run termination requested by ${terminationSignal}.`);
  error.code = 'FIRST_INSTANCE_LIVE_ORDER_TERMINATED';
  return error;
}

function requireRunActive() {
  if (terminationSignal !== null) throw terminationError();
}

function requireCondition(condition, message, evidence = null) {
  if (condition) return;
  const serialized = evidence === null ? '' : JSON.stringify(evidence);
  const detail = serialized.length === 0
    ? ''
    : `: ${serialized.length <= 4_000
      ? serialized
      : `${serialized.slice(0, 4_000)}...[${serialized.length} characters]`}`;
  throw new Error(`${message}${detail}`);
}

async function withDeadline(promise, timeoutMs, label) {
  let timeoutId;
  const deadline = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(`${label} exceeded its ${timeoutMs} ms deadline.`));
    }, timeoutMs);
  });
  try {
    return await Promise.race([promise, deadline]);
  } finally {
    clearTimeout(timeoutId);
  }
}

function sha256Bytes(value) {
  return createHash('sha256').update(value).digest('hex');
}

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function relativeArtifactPath(absolutePath) {
  return path.relative(runDirectory, absolutePath).split(path.sep).join('/');
}

async function writeExclusiveJson(
  absolutePath,
  value,
  { allowDuringTermination = false } = {},
) {
  if (!allowDuringTermination) requireRunActive();
  const bytes = jsonBytes(value);
  await writeFile(absolutePath, bytes, { flag: 'wx' });
  if (!allowDuringTermination) requireRunActive();
  return Object.freeze({
    path: relativeArtifactPath(absolutePath),
    encoding: 'json-utf8',
    byteLength: bytes.length,
    sha256: sha256Bytes(bytes),
  });
}

async function writeExclusiveBrotliJson(
  absolutePath,
  value,
  { allowDuringTermination = false } = {},
) {
  if (!allowDuringTermination) requireRunActive();
  const bytes = jsonBytes(value);
  const compressed = await brotliCompressAsync(bytes, {
    params: {
      [zlibConstants.BROTLI_PARAM_QUALITY]: 6,
      [zlibConstants.BROTLI_PARAM_MODE]: zlibConstants.BROTLI_MODE_TEXT,
    },
  });
  if (!allowDuringTermination) requireRunActive();
  await writeFile(absolutePath, compressed, { flag: 'wx' });
  if (!allowDuringTermination) requireRunActive();
  return Object.freeze({
    path: relativeArtifactPath(absolutePath),
    encoding: 'brotli-json-utf8',
    jsonByteLength: bytes.length,
    jsonSha256: sha256Bytes(bytes),
    brotliByteLength: compressed.length,
    brotliSha256: sha256Bytes(compressed),
  });
}

function sameArray(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function sameJson(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function environmentIdentity(environment) {
  return {
    threeRevision: environment?.threeRevision ?? null,
    userAgent: environment?.userAgent ?? null,
    adapterInfo: environment?.adapterInfo ?? null,
    rendererBackend: environment?.rendererBackend ?? null,
    coordinateSystem: environment?.coordinateSystem ?? null,
    reversedDepth: environment?.reversedDepth ?? null,
    rendererReversedDepthBuffer: environment?.rendererReversedDepthBuffer ?? null,
    maxStorageBuffersPerShaderStage:
      environment?.maxStorageBuffersPerShaderStage ?? null,
    timestampAvailable: environment?.timestampAvailable ?? null,
    indirectFirstInstanceAvailable:
      environment?.indirectFirstInstanceAvailable ?? null,
    crossOriginIsolated: environment?.crossOriginIsolated ?? null,
    performanceNowQuantumMs: environment?.performanceNowQuantumMs ?? null,
    viewport: environment?.viewport ?? null,
  };
}

function workloadIdentity(workload) {
  return {
    scenarioSeed: workload?.scenarioSeed ?? null,
    geometrySha256: workload?.geometryFixtures?.sha256 ?? null,
    scenarioSha256: workload?.scenario?.sha256 ?? null,
    expectedVisibleIdsCanonicalSha256:
      workload?.scenario?.expectedVisibleIdsCanonicalSha256 ?? null,
    objectCount: workload?.scenario?.objectCount ?? null,
    bucketCount: workload?.scenario?.bucketCount ?? null,
    visibilityFraction: workload?.scenario?.visibilityFraction ?? null,
    layout: workload?.scenario?.layout ?? null,
  };
}

function requireCleanEnvironment(environment, label) {
  requireCondition(environment?.indirectFirstInstanceAvailable === true,
    `${label} lacks indirect-first-instance`, environment);
  requireCondition(environment?.timestampAvailable === true,
    `${label} lacks timestamp queries`, environment);
  requireCondition(environment?.reversedDepth === true
    && environment?.rendererReversedDepthBuffer === true,
  `${label} lacks pinned reversed depth`, environment);
  requireCondition(environment?.crossOriginIsolated === true,
    `${label} lost cross-origin isolation`, environment);
  requireCondition(Number.isFinite(environment?.performanceNowQuantumMs)
    && environment.performanceNowQuantumMs > 0
    && environment.performanceNowQuantumMs <= MAXIMUM_CPU_TIMER_QUANTUM_MS,
  `${label} has an invalid CPU timer quantum`, environment);
  requireCondition(environment?.webgpuUncapturedErrorCount === 0,
    `${label} observed an uncaptured WebGPU error`, environment);
  requireCondition(environment?.webgpuValidationErrorCount === 0,
    `${label} observed a WebGPU validation error`, environment);
  requireCondition(environment?.webgpuDeviceLossCount === 0,
    `${label} observed WebGPU device loss`, environment);
}

function requireCleanGpuRecords(record, label) {
  requireCondition(Array.isArray(record?.webgpuUncapturedErrors)
    && record.webgpuUncapturedErrors.length === 0,
  `${label} has uncaptured WebGPU errors`, record?.webgpuUncapturedErrors);
  requireCondition(Array.isArray(record?.webgpuDeviceLosses)
    && record.webgpuDeviceLosses.length === 0,
  `${label} has WebGPU device losses`, record?.webgpuDeviceLosses);
  requireCleanEnvironment(record?.environment, `${label} environment`);
}

async function findBrowser() {
  const programFiles = process.env.ProgramFiles ?? 'C:\\Program Files';
  const programFilesX86 = process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)';
  const localAppData = process.env.LOCALAPPDATA;
  const candidates = [
    process.env.BROWSER_PATH,
    path.join(programFiles, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    path.join(programFilesX86, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    path.join(programFiles, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    path.join(programFilesX86, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    localAppData && path.join(localAppData, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
  ].filter(Boolean);
  for (const candidate of [...new Set(candidates)]) {
    try {
      await access(candidate);
      return path.resolve(candidate);
    } catch {
      // Continue through installed system-browser locations.
    }
  }
  throw new Error('No installed Chrome, Chromium, or Edge executable was found.');
}

function attachErrorCapture(page, records) {
  page.on('pageerror', (error) => records.push({
    source: 'pageerror',
    detail: error.stack ?? error.message,
  }));
  page.on('console', (message) => {
    if (message.type() === 'error') {
      records.push({ source: 'console', detail: message.text() });
    }
  });
}

let browserInstanceSerial = 0;
let activeBrowserState = null;
const processLifecycle = {
  gate: null,
  sessions: [],
};

async function launchBrowserProcess(executablePath, role, sessionOrdinal = null) {
  requireRunActive();
  requireCondition(activeBrowserState === null,
    `Refusing overlapping browser launch for ${role}`,
    activeBrowserState?.record ?? null);
  browserInstanceSerial += 1;
  const record = {
    schemaVersion: 1,
    kind: 'first-instance-live-order-factorial-browser-lifecycle',
    browserInstanceSerial,
    role,
    sessionOrdinal,
    launchedAt: new Date().toISOString(),
    contextCreatedAt: null,
    pageCreatedAt: null,
    closedAt: null,
    contextCountBeforeClose: null,
    pageCountBeforeClose: null,
    disconnectedEventCount: 0,
    closedBeforeNextLaunch: false,
  };
  const browser = await chromium.launch({
    executablePath,
    headless: true,
    args: BROWSER_ARGS,
    timeout: BROWSER_OPERATION_TIMEOUT_MS,
  });
  if (terminationSignal !== null) {
    await withDeadline(
      browser.close(),
      BROWSER_CLOSE_TIMEOUT_MS,
      `${role} browser close after termination request`,
    ).catch(() => undefined);
    throw terminationError();
  }
  browser.on('disconnected', () => {
    record.disconnectedEventCount += 1;
  });
  activeBrowserState = {
    browser,
    context: null,
    page: null,
    record,
    closePromise: null,
  };
  requireCondition(browser.contexts().length === 0,
    `${role} browser started with an unexpected context`);
  return activeBrowserState;
}

async function openOnlyPage(state, url, errors) {
  requireRunActive();
  requireCondition(activeBrowserState === state, 'Browser state is no longer active.');
  requireCondition(state.context === null && state.page === null,
    `${state.record.role} attempted to create more than one context/page.`);
  const context = await withDeadline(state.browser.newContext({
    viewport: { width: 1280, height: 900 },
    deviceScaleFactor: 1,
  }), BROWSER_OPERATION_TIMEOUT_MS, `${state.record.role} context creation`);
  state.context = context;
  state.record.contextCreatedAt = new Date().toISOString();
  requireCondition(state.browser.contexts().length === 1,
    `${state.record.role} did not retain exactly one context.`);
  const page = await withDeadline(
    context.newPage(),
    BROWSER_OPERATION_TIMEOUT_MS,
    `${state.record.role} page creation`,
  );
  state.page = page;
  state.record.pageCreatedAt = new Date().toISOString();
  attachErrorCapture(page, errors);
  requireCondition(context.pages().length === 1,
    `${state.record.role} did not retain exactly one page.`);
  await withDeadline(
    page.goto(url, { waitUntil: 'domcontentloaded' }),
    BROWSER_OPERATION_TIMEOUT_MS,
    `${state.record.role} entry navigation`,
  );
  await withDeadline(page.waitForFunction(
    () => window.__WEBGPU_BENCH__?.ready === true,
    null,
    { timeout: 120_000 },
  ), BROWSER_OPERATION_TIMEOUT_MS, `${state.record.role} benchmark readiness`);
  requireRunActive();
  return page;
}

function closeBrowserProcess(state) {
  if (state.closePromise !== null) return state.closePromise;
  requireCondition(activeBrowserState === state,
    `Attempted to close a non-active ${state.record.role} browser.`);
  state.closePromise = (async () => {
    const { browser, context, page, record } = state;
    record.contextCountBeforeClose = browser.contexts().length;
    record.pageCountBeforeClose = context?.pages().length ?? 0;
    requireCondition(record.contextCountBeforeClose === 1
      && record.pageCountBeforeClose === 1,
    `${record.role} violated its one-context/one-page boundary`, record);
    await withDeadline(
      context.close(),
      BROWSER_CLOSE_TIMEOUT_MS,
      `${record.role} context close`,
    );
    requireCondition(page.isClosed() === true && browser.contexts().length === 0,
      `${record.role} context/page did not close cleanly.`);
    await withDeadline(
      browser.close(),
      BROWSER_CLOSE_TIMEOUT_MS,
      `${record.role} browser close`,
    );
    requireCondition(browser.isConnected() === false,
      `${record.role} browser process remained connected after close.`);
    requireCondition(record.disconnectedEventCount === 1,
      `${record.role} browser emitted an unexpected disconnected event count`, record);
    record.closedAt = new Date().toISOString();
    record.closedBeforeNextLaunch = true;
    if (activeBrowserState === state) activeBrowserState = null;
    return Object.freeze({ ...record });
  })();
  return state.closePromise;
}

async function forceCloseActiveBrowser(reason = 'forced-cleanup') {
  const state = activeBrowserState;
  if (state === null) return null;
  if (state.closePromise !== null) {
    const cleanLifecycle = await state.closePromise.catch(() => null);
    if (cleanLifecycle !== null) return cleanLifecycle;
  }
  activeBrowserState = null;
  state.record.abortedAt = new Date().toISOString();
  state.record.abortReason = reason;
  if (state.context) {
    await withDeadline(
      state.context.close(),
      BROWSER_CLOSE_TIMEOUT_MS,
      `${state.record.role} forced context close`,
    ).catch(() => undefined);
  }
  if (state.browser) {
    await withDeadline(
      state.browser.close(),
      BROWSER_CLOSE_TIMEOUT_MS,
      `${state.record.role} forced browser close`,
    ).catch(() => undefined);
  }
  state.record.closedAt ??= new Date().toISOString();
  state.record.closedBeforeNextLaunch = false;
  return structuredClone(state.record);
}

function createShaderObservationChallenges(spec, usedNonces) {
  return Object.freeze([
    ['preflight', 'render-parity'],
    ['preflight', 'main-validation'],
    ['timing-start', 'render-parity'],
    ['timing-start', 'main-validation'],
    ['postflight', 'render-parity'],
    ['postflight', 'main-validation'],
  ].map(([phase, role], index) => {
    const challengeNonce = randomBytes(32).toString('hex');
    requireCondition(!usedNonces.has(challengeNonce),
      'Shader observation challenge nonce repeated.');
    usedNonces.add(challengeNonce);
    return Object.freeze({
      schemaVersion: 1,
      kind: 'live-first-instance-shader-observation-challenge',
      origin: 'node-runner',
      runId,
      trialId: spec.trialId,
      planIndex: spec.planIndex,
      repetitionIndex: spec.repetitionIndex,
      phase,
      role,
      captureOrdinal: index + 1,
      challengeNonce,
    });
  }));
}

function candidateCompatibilitySpec(factorialTrial, compatibilityHighPlan) {
  const matches = compatibilityHighPlan.filter((candidate) => (
    candidate.repetitionIndex < 4
      && candidate.visibilityFraction === VISIBILITY_FRACTION
      && sameArray(candidate.lanePhysicalOrder, factorialTrial.laneConstructionOrder)
      && candidate.superblockOrientationOffset
        === factorialTrial.superblockOrientationOffset
  ));
  requireCondition(matches.length === 1,
    `Factorial trial ${factorialTrial.trialId} has no unique candidate compatibility cell`,
    matches);
  const selected = matches[0];
  return Object.freeze({
    ...selected,
    runId,
    trialId: factorialTrial.trialId,
    lanePhysicalOrder: [...selected.lanePhysicalOrder],
    laneConstructionOrder: [...factorialTrial.laneConstructionOrder],
    firstComputeUseOrder: [...factorialTrial.firstComputeUseOrder],
    renderPipelinePrimeOrder: [...factorialTrial.renderPipelinePrimeOrder],
    setupPrimeTopology: SETUP_PRIME_TOPOLOGY,
    timestampPreprimeLaneId: factorialTrial.timestampPreprimeLaneId,
  });
}

function createAuditContext(spec, factorialTrial) {
  return Object.freeze({
    runId,
    trialId: spec.trialId,
    planIndex: spec.planIndex,
    repetitionIndex: spec.repetitionIndex,
    modeOrderPosition: spec.modeOrderPosition,
    visibilityOrderPosition: spec.visibilityOrderPosition,
    layoutOrderPosition: spec.layoutOrderPosition,
    plannedModeOrder: spec.modeOrder.join('|'),
    plannedVisibilityOrder: spec.visibilityOrder.join('|'),
    plannedLayoutOrder: spec.layoutOrder.join('|'),
    protocolWarmupFrames: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    protocolMeasuredFrames: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    plannedLanePhysicalOrder: spec.lanePhysicalOrder.join('|'),
    plannedLaneConstructionOrder: spec.laneConstructionOrder.join('|'),
    plannedFirstComputeUseOrder: spec.firstComputeUseOrder.join('|'),
    plannedRenderPipelinePrimeOrder: spec.renderPipelinePrimeOrder.join('|'),
    setupPrimeTopology: spec.setupPrimeTopology,
    timestampPreprimeLaneId: spec.timestampPreprimeLaneId,
    plannedTimestampPreprimeLaneId: spec.timestampPreprimeLaneId,
    superblockOrientationOffset: spec.superblockOrientationOffset,
    plannedScheduleSha256: liveFirstInstanceCrossoverScheduleSha256(
      spec.superblockOrientationOffset,
    ),
    factorialExperimentKind: FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_KIND,
    factorialPlanIndex: factorialTrial.planIndex,
    factorialSessionIndex: factorialTrial.sessionIndex,
    factorialSessionTrialIndex: factorialTrial.sessionTrialIndex,
    factorialCellIndex: factorialTrial.factorialCellIndex,
    factorialCellId: factorialTrial.factorialCellId,
    factorialC: factorialTrial.factorLevels.C,
    factorialK: factorialTrial.factorLevels.K,
    factorialR: factorialTrial.factorLevels.R,
    factorialT: factorialTrial.factorLevels.T,
    factorialLaneConstructionOrder: factorialTrial.laneConstructionOrder.join('|'),
    factorialFirstComputeUseOrder: factorialTrial.firstComputeUseOrder.join('|'),
    factorialRenderPipelinePrimeOrder: factorialTrial.renderPipelinePrimeOrder.join('|'),
    factorialTimestampPreprimeLaneId: factorialTrial.timestampPreprimeLaneId,
    factorialCompatibilityMapping: COMPATIBILITY_MAPPING,
    factorialCompatibilityPlanIndex: spec.planIndex,
    factorialCompatibilityRepetitionIndex: spec.repetitionIndex,
  });
}

async function configureTrial(page, factorialTrial) {
  return page.evaluate(async (configuration) => {
    const bench = window.__WEBGPU_BENCH__;
    bench.configureFirstInstanceLiveCrossover({
      lanePhysicalOrder: configuration.laneConstructionOrder,
      plannedFirstComputeUseOrder: configuration.firstComputeUseOrder,
      plannedRenderPipelinePrimeOrder: configuration.renderPipelinePrimeOrder,
      setupPrimeTopology: configuration.setupPrimeTopology,
      timestampPreprimeLaneId: configuration.timestampPreprimeLaneId,
      superblockOrientationOffset: configuration.superblockOrientationOffset,
    });
    for (const [id, value] of Object.entries({
      strategy: configuration.modeId,
      objects: String(configuration.objectCount),
      buckets: String(configuration.bucketCount),
      visibility: String(configuration.visibilityFraction),
      layout: configuration.layout,
    })) {
      const select = document.getElementById(id);
      if (!select || ![...select.options].some((option) => option.value === value)) {
        throw new Error(`Unsupported order-factorial configuration ${id}=${value}.`);
      }
      select.value = value;
    }
    await bench.rebuild();
    return {
      selectedConfig: bench.selectedConfig(),
      environment: bench.environment,
      shaderEvidence: bench.firstInstanceShaderEvidence,
      timestampPoolPreprime: bench.timestampPoolPreprime,
      timestampPoolDiagnostics: bench.timestampPoolDiagnostics,
      webgpuUncapturedErrors: bench.webgpuUncapturedErrors,
      webgpuDeviceLosses: bench.webgpuDeviceLosses,
    };
  }, {
    ...factorialTrial,
    modeId: FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    visibilityFraction: VISIBILITY_FRACTION,
    layout: LAYOUT,
    setupPrimeTopology: SETUP_PRIME_TOPOLOGY,
  });
}

async function captureEvidencePoint(page, observationChallenges) {
  return page.evaluate(async (challenges) => {
    const bench = window.__WEBGPU_BENCH__;
    const renderParity = await bench.captureRenderParity(challenges[0]);
    const validation = await bench.validate(challenges[1]);
    return {
      validation,
      renderParity,
      workload: await bench.fingerprintWorkload(),
      environment: bench.environment,
      webgpuUncapturedErrors: bench.webgpuUncapturedErrors,
      webgpuDeviceLosses: bench.webgpuDeviceLosses,
    };
  }, observationChallenges);
}

function requireConfiguredTrial(configured, factorialTrial, label) {
  const config = configured?.selectedConfig;
  requireCondition(config?.strategyId === FIRST_INSTANCE_LIVE_CROSSOVER_MODE
    && config.objectCount === OBJECT_COUNT
    && config.bucketCount === BUCKET_COUNT
    && config.visibilityFraction === VISIBILITY_FRACTION
    && config.layout === LAYOUT,
  `${label} selected the wrong fixed workload`, config);
  requireCondition(sameArray(config.lanePhysicalOrder,
    factorialTrial.laneConstructionOrder),
  `${label} construction order differs from C`, config);
  requireCondition(sameArray(config.plannedFirstComputeUseOrder,
    factorialTrial.firstComputeUseOrder),
  `${label} compute-use order differs from K`, config);
  requireCondition(sameArray(config.plannedRenderPipelinePrimeOrder,
    factorialTrial.renderPipelinePrimeOrder),
  `${label} render-prime order differs from R`, config);
  requireCondition(config.setupPrimeTopology === SETUP_PRIME_TOPOLOGY,
    `${label} selected the wrong setup topology`, config);
  requireCondition(config.timestampPreprimeLaneId
    === factorialTrial.timestampPreprimeLaneId,
  `${label} timestamp preprime lane differs from T`, config);
  requireCondition(config.superblockOrientationOffset
    === factorialTrial.superblockOrientationOffset,
  `${label} selected the wrong schedule orientation`, config);
  requireCondition(configured.shaderEvidence?.pass === true,
    `${label} shader evidence failed`, configured.shaderEvidence);
  requireCondition(configured.timestampPoolPreprime?.kind
    === 'three-r185-timestamp-pool-preprime',
  `${label} timestamp pools were not pre-primed`, configured.timestampPoolPreprime);
  requireCleanGpuRecords(configured, label);
}

function assertStableIdentity(state, environment, workload, label) {
  const observedEnvironment = environmentIdentity(environment);
  const observedWorkload = workloadIdentity(workload);
  if (state.environment === null) state.environment = observedEnvironment;
  if (state.workload === null) state.workload = observedWorkload;
  requireCondition(sameJson(observedEnvironment, state.environment),
    `${label} environment identity drifted`, {
      expected: state.environment,
      observed: observedEnvironment,
    });
  requireCondition(sameJson(observedWorkload, state.workload),
    `${label} workload identity drifted`, {
      expected: state.workload,
      observed: observedWorkload,
    });
}

async function runFactorialTrial({
  page,
  pageErrors,
  factorialTrial,
  compatibilitySpec,
  identityState,
  usedNonces,
}) {
  const label = `factorial plan ${factorialTrial.planIndex} (${factorialTrial.factorialCellId})`;
  const shaderObservationChallenges = createShaderObservationChallenges(
    compatibilitySpec,
    usedNonces,
  );
  const auditContext = createAuditContext(compatibilitySpec, factorialTrial);
  const configured = await withDeadline(
    configureTrial(page, factorialTrial),
    BROWSER_OPERATION_TIMEOUT_MS,
    `${label} configuration`,
  );
  requireConfiguredTrial(configured, factorialTrial, `${label} construction`);
  requireCondition(pageErrors.length === 0, `${label} emitted page errors`, pageErrors);

  const preflight = await withDeadline(
    captureEvidencePoint(page, shaderObservationChallenges.slice(0, 2)),
    BROWSER_OPERATION_TIMEOUT_MS,
    `${label} preflight evidence`,
  );
  requireCleanGpuRecords(preflight, `${label} preflight`);
  assertStableIdentity(identityState, preflight.environment, preflight.workload,
    `${label} preflight`);

  const timingParity = await withDeadline(page.evaluate(
    (challenge) => window.__WEBGPU_BENCH__.captureRenderParity(challenge),
    shaderObservationChallenges[2],
  ), BROWSER_OPERATION_TIMEOUT_MS, `${label} timing-start parity`);
  const timingStart = await withDeadline(page.evaluate(
    ({ context, challenge }) => window.__WEBGPU_BENCH__.startTrial(context, challenge),
    { context: auditContext, challenge: shaderObservationChallenges[3] },
  ), BROWSER_OPERATION_TIMEOUT_MS, `${label} timing start`);
  requireCondition(timingStart?.validation?.pass === true,
    `${label} timing-start validation failed`, timingStart?.validation);
  assertStableIdentity(identityState, configured.environment, timingStart.workload,
    `${label} timing start`);

  await withDeadline(page.waitForFunction(
    () => ['complete', 'error'].includes(window.__WEBGPU_BENCH__?.phase),
    null,
    { timeout: 180_000 },
  ), BROWSER_OPERATION_TIMEOUT_MS, `${label} timed phase completion`);
  const timing = await withDeadline(page.evaluate(() => ({
    phase: window.__WEBGPU_BENCH__.phase,
    error: window.__WEBGPU_BENCH__.trialError,
    rows: window.__WEBGPU_BENCH__.rows,
    summary: JSON.parse(document.getElementById('details')?.textContent ?? 'null'),
    environment: window.__WEBGPU_BENCH__.environment,
    webgpuUncapturedErrors: window.__WEBGPU_BENCH__.webgpuUncapturedErrors,
    webgpuDeviceLosses: window.__WEBGPU_BENCH__.webgpuDeviceLosses,
  })), BROWSER_OPERATION_TIMEOUT_MS, `${label} timing collection`);
  requireCondition(timing.phase === 'complete', `${label} timing failed`, timing.error);
  requireCondition(Array.isArray(timing.rows)
    && timing.rows.length === FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
  `${label} returned an incomplete measured row set`, timing.rows?.length);
  requireCleanGpuRecords(timing, `${label} timing`);

  const postflight = await withDeadline(
    captureEvidencePoint(page, shaderObservationChallenges.slice(4, 6)),
    BROWSER_OPERATION_TIMEOUT_MS,
    `${label} postflight evidence`,
  );
  requireCleanGpuRecords(postflight, `${label} postflight`);
  assertStableIdentity(identityState, postflight.environment, postflight.workload,
    `${label} postflight`);
  requireCondition(sameJson(workloadIdentity(preflight.workload),
    workloadIdentity(timingStart.workload))
      && sameJson(workloadIdentity(preflight.workload),
        workloadIdentity(postflight.workload)),
  `${label} workload changed across evidence phases`);

  const protocol = {
    schemaVersion: 2,
    warmupFrames: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    measuredFrames: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    plannedScheduleSha256: auditContext.plannedScheduleSha256,
  };
  const strict = await validateLiveFirstInstanceTrialEvidence({
    spec: compatibilitySpec,
    environment: configured.environment,
    preflightValidation: preflight.validation,
    preflightRenderParity: preflight.renderParity,
    validation: timingStart.validation,
    renderParity: timingParity,
    postflightValidation: postflight.validation,
    postflightRenderParity: postflight.renderParity,
    shaderObservationChallenges,
    rows: timing.rows,
    summary: timing.summary,
    protocol,
    scenarioManifest: preflight.workload.scenario,
    geometryManifest: preflight.workload.geometryFixtures,
  });
  requireCondition(pageErrors.length === 0, `${label} emitted page errors`, pageErrors);
  requireCondition(strict.pass === true, `${label} strict evidence failed`, strict);

  const trialSummary = summarizeLiveFirstInstanceTrialRows(
    timing.rows,
    compatibilitySpec,
    runId,
  );
  const completionInvariant = timing.summary?.completionInvariant;
  const timingStartCacheState = {
    totalPipelineCacheEntries:
      completionInvariant?.totalPipelineCacheEntriesAtTimingStart ?? null,
    computePipelineCacheEntries:
      completionInvariant?.computePipelineCacheEntriesAtTimingStart ?? null,
    computeProgramEntries: completionInvariant?.computeProgramEntriesAtTimingStart ?? null,
    rendererMemory: completionInvariant?.rendererMemoryAtTimingStart ?? null,
  };
  requireCondition([
    timingStartCacheState.totalPipelineCacheEntries,
    timingStartCacheState.computePipelineCacheEntries,
    timingStartCacheState.computeProgramEntries,
  ].every((value) => Number.isSafeInteger(value) && value >= 0)
    && timingStartCacheState.rendererMemory !== null
    && typeof timingStartCacheState.rendererMemory === 'object'
    && !Array.isArray(timingStartCacheState.rendererMemory),
  `${label} timing-start cache state is unavailable`, timingStartCacheState);
  const artifactBody = {
    schemaVersion: 1,
    kind: 'first-instance-live-order-factorial-trial-artifact',
    scope: 'development diagnostic; not candidate evidence or a pass claim',
    runId,
    capturedAt: new Date().toISOString(),
    factorialTrial,
    compatibility: {
      mapping: COMPATIBILITY_MAPPING,
      validationSpec: compatibilitySpec,
    },
    auditContext,
    configured,
    shaderObservationChallenges,
    preflight,
    timingStart,
    timing: {
      renderParity: timingParity,
      rows: timing.rows,
      summary: timing.summary,
      environment: timing.environment,
      webgpuUncapturedErrors: timing.webgpuUncapturedErrors,
      webgpuDeviceLosses: timing.webgpuDeviceLosses,
    },
    postflight,
    protocol,
    strictValidation: strict,
    trialSummary,
    pageErrors: [...pageErrors],
  };
  const filename = `trial-${String(factorialTrial.planIndex).padStart(2, '0')}`
    + `-s${factorialTrial.sessionOrdinal}`
    + `-t${String(factorialTrial.sessionTrialOrdinal).padStart(2, '0')}`
    + `-${factorialTrial.factorialCellId}.json.br`;
  const artifact = await writeExclusiveBrotliJson(
    path.join(trialDirectory, filename),
    artifactBody,
  );
  return {
    analysisRecord: {
      ...factorialTrial,
      trialSummary,
    },
    manifestRecord: {
      factorialPlanIndex: factorialTrial.planIndex,
      factorialTrialId: factorialTrial.trialId,
      sessionIndex: factorialTrial.sessionIndex,
      sessionTrialIndex: factorialTrial.sessionTrialIndex,
      factorialCellId: factorialTrial.factorialCellId,
      compatibilityPlanIndex: compatibilitySpec.planIndex,
      compatibilityRepetitionIndex: compatibilitySpec.repetitionIndex,
      strictSemanticSha256: strict.semanticSha256,
      timingStartCacheState,
      artifact,
    },
  };
}

let server = null;
let viteRuntimeGuard = null;
const completedTrialArtifacts = [];
const completedSessionArtifacts = [];
let sourceProvenanceStart = null;
let sourceProvenanceEnd = null;
let executionDependencyClosureStart = null;
let executionDependencyClosureEnd = null;
let executionIdentityStartArtifact = null;
let executionIdentityEndArtifact = null;
let serverClosePromise = null;
let viteRuntimeGuardOperation = null;
let ownedResourceCleanupPromise = null;
let signalBrowserCleanupPromise = null;
let runCompletionCommitted = false;

function closeViteServer(label) {
  if (serverClosePromise !== null) return serverClosePromise;
  const ownedServer = server;
  server = null;
  serverClosePromise = ownedServer === null
    ? Promise.resolve()
    : withDeadline(
      ownedServer.close(),
      SERVER_OPERATION_TIMEOUT_MS,
      label,
    );
  return serverClosePromise;
}

function disposeViteRuntimeGuard(label) {
  if (viteRuntimeGuardOperation !== null) {
    return viteRuntimeGuardOperation.then(() => undefined);
  }
  const ownedGuard = viteRuntimeGuard;
  viteRuntimeGuard = null;
  viteRuntimeGuardOperation = ownedGuard === null
    ? Promise.resolve()
    : withDeadline(
      ownedGuard.dispose(),
      SERVER_OPERATION_TIMEOUT_MS,
      label,
    );
  return viteRuntimeGuardOperation;
}

function finalizeViteRuntimeGuard() {
  requireCondition(viteRuntimeGuardOperation === null && viteRuntimeGuard !== null,
    'Vite runtime guard is unavailable for finalization.');
  const ownedGuard = viteRuntimeGuard;
  viteRuntimeGuard = null;
  viteRuntimeGuardOperation = (async () => {
    try {
      return await withDeadline(
        ownedGuard.finalize(),
        SERVER_OPERATION_TIMEOUT_MS,
        'Vite runtime audit finalization',
      );
    } finally {
      await withDeadline(
        ownedGuard.dispose(),
        SERVER_OPERATION_TIMEOUT_MS,
        'Vite runtime guard disposal after finalization',
      ).catch(() => undefined);
    }
  })();
  return viteRuntimeGuardOperation;
}

function cleanupOwnedResources(reason = 'forced-cleanup') {
  if (ownedResourceCleanupPromise !== null) return ownedResourceCleanupPromise;
  ownedResourceCleanupPromise = (async () => {
    const [abortedBrowserLifecycle] = await Promise.all([
      (signalBrowserCleanupPromise ?? forceCloseActiveBrowser(reason))
        .catch(() => null),
      closeViteServer(`${reason} Vite server close`).catch(() => undefined),
      disposeViteRuntimeGuard(`${reason} Vite runtime guard disposal`)
        .catch(() => undefined),
    ]);
    return abortedBrowserLifecycle;
  })();
  return ownedResourceCleanupPromise;
}

function handleTerminationSignal(signal) {
  if (runCompletionCommitted || terminationSignal !== null) return;
  terminationSignal = signal;
  process.exitCode = signal === 'SIGINT' ? 130 : 143;
  signalBrowserCleanupPromise = forceCloseActiveBrowser(signal);
  void signalBrowserCleanupPromise.catch(() => undefined);
}

process.on('SIGINT', () => handleTerminationSignal('SIGINT'));
process.on('SIGTERM', () => handleTerminationSignal('SIGTERM'));

try {
  await mkdir(developmentRoot, { recursive: true });
  await mkdir(runDirectory, { recursive: false });
  await mkdir(trialDirectory, { recursive: false });

  const factorialPlan = buildFirstInstanceLiveOrderFactorialPlan({ runId });
  requireCondition(factorialPlan.length === FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIAL_COUNT,
    'Frozen factorial plan has the wrong trial count.');
  const candidatePlan = buildFirstInstanceLiveCrossoverPlan({
    runId,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  });
  const compatibilityHighPlan = candidatePlan.filter((trial) => (
    trial.repetitionIndex < 4 && trial.visibilityFraction === VISIBILITY_FRACTION
  ));
  requireCondition(compatibilityHighPlan.length === 4,
    'Candidate compatibility map does not contain the four C/orientation cells.');
  requireCondition(sameArray(
    compatibilityHighPlan.map((trial) => trial.repetitionIndex).sort((left, right) => left - right),
    [0, 1, 2, 3],
  ), 'Candidate compatibility map must use repetitions zero through three exactly.');
  const compatibilitySpecs = factorialPlan.map(
    (trial) => candidateCompatibilitySpec(trial, compatibilityHighPlan),
  );
  const smokeTrial = factorialPlan.find((trial) => (
    trial.sessionIndex === 0
      && !sameArray(trial.firstComputeUseOrder, trial.renderPipelinePrimeOrder)
  ));
  requireCondition(!smokeMode || smokeTrial !== undefined,
    'Smoke mode could not select a staged independent-order factorial trial.');
  const executionPlan = smokeMode ? Object.freeze([smokeTrial]) : factorialPlan;
  const planArtifact = await writeExclusiveJson(path.join(runDirectory, 'plan.json'), {
    schemaVersion: 1,
    kind: 'first-instance-live-order-factorial-execution-plan',
    runId,
    frozenPlan: factorialPlan,
    executionMode: smokeMode ? 'one-trial-smoke-excluded-from-analysis' : 'full-factorial',
    executedFactorialPlanIndexes: executionPlan.map((trial) => trial.planIndex),
    compatibilityMapping: COMPATIBILITY_MAPPING,
    compatibilitySpecs: compatibilitySpecs.map((spec, index) => ({
      factorialPlanIndex: factorialPlan[index].planIndex,
      factorialTrialId: factorialPlan[index].trialId,
      candidatePlanIndex: spec.planIndex,
      candidateRepetitionIndex: spec.repetitionIndex,
      candidateVisibilityOrderPosition: spec.visibilityOrderPosition,
      lanePhysicalOrder: spec.lanePhysicalOrder,
      superblockOrientationOffset: spec.superblockOrientationOffset,
    })),
    executionPolicy: {
      retryCount: 0,
      replacementAllowed: false,
      outlierRemovalAllowed: false,
      efficacyStoppingAllowed: false,
      claim: 'development diagnostic only',
    },
  });

  [sourceProvenanceStart, executionDependencyClosureStart] = await Promise.all([
    collectSourceProvenance(projectRoot),
    collectExecutionDependencyClosure(projectRoot),
  ]);
  requireCondition(sourceProvenanceStart.status === 'available'
    && sourceProvenanceStart.captureStable === true
    && sourceProvenanceStart.dirty === false
    && sourceProvenanceStart.packageLockTracked === true,
  'Order-factorial timing requires clean, stable, available source provenance',
  sourceProvenanceStart);
  requireCondition(executionDependencyClosureStart.schemaVersion === 1
    && executionDependencyClosureStart.kind === 'installed-execution-dependency-closure'
    && executionDependencyClosureStart.fileCount > 0
    && /^[0-9a-f]{64}$/.test(executionDependencyClosureStart.sha256),
  'Installed execution dependency closure is unavailable or invalid',
  executionDependencyClosureStart);
  executionIdentityStartArtifact = await writeExclusiveJson(
    path.join(runDirectory, 'execution-identity-start.json'),
    {
      schemaVersion: 1,
      kind: 'first-instance-live-order-factorial-execution-identity-start',
      runId,
      sourceProvenance: sourceProvenanceStart,
      executionDependencyClosure: executionDependencyClosureStart,
    },
  );

  const executablePath = await findBrowser();
  requireRunActive();
  const createdViteRuntimeGuard = await createCandidateViteRuntimeGuard(projectRoot);
  if (terminationSignal !== null) {
    await withDeadline(
      createdViteRuntimeGuard.dispose(),
      SERVER_OPERATION_TIMEOUT_MS,
      'Vite runtime guard disposal after termination request',
    ).catch(() => undefined);
    throw terminationError();
  }
  viteRuntimeGuard = createdViteRuntimeGuard;
  const createdServer = await createServer({
    root: projectRoot,
    configFile: false,
    resolve: { dedupe: ['three'] },
    server: {
      host: '127.0.0.1',
      port: 0,
      headers: ISOLATION_HEADERS,
    },
    logLevel: 'error',
    ...viteRuntimeGuard.viteConfig,
  });
  if (terminationSignal !== null) {
    await withDeadline(
      createdServer.close(),
      SERVER_OPERATION_TIMEOUT_MS,
      'Vite server close after termination request',
    ).catch(() => undefined);
    throw terminationError();
  }
  server = createdServer;
  await withDeadline(
    server.listen(),
    SERVER_OPERATION_TIMEOUT_MS,
    'Vite server listen',
  );
  requireRunActive();
  const url = server.resolvedUrls?.local?.[0];
  requireCondition(Boolean(url), 'Vite did not expose a local URL.');

  const identityState = { environment: null, workload: null };
  const usedNonces = new Set();
  process.stdout.write(`[gate] launching disposable forced-feature-off browser\n`);
  const gateErrors = [];
  const gateState = await launchBrowserProcess(executablePath, 'forced-feature-off-gate');
  const gatePage = await openOnlyPage(gateState, url, gateErrors);
  const gateCapture = await withDeadline(gatePage.evaluate(async (options) => {
    const bench = window.__WEBGPU_BENCH__;
    const gate = await bench.runFirstInstanceLiveForcedFeatureOffGate(options);
    return {
      gate,
      environment: bench.environment,
      workload: await bench.fingerprintWorkload(),
      webgpuUncapturedErrors: bench.webgpuUncapturedErrors,
      webgpuDeviceLosses: bench.webgpuDeviceLosses,
    };
  }, {
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    visibilityFraction: VISIBILITY_FRACTION,
    scenarioSeed: SCENARIO_SEED,
  }), BROWSER_OPERATION_TIMEOUT_MS, 'forced-feature-off gate');
  const gateRejections = validateLiveFirstInstanceForcedFeatureOffGate(gateCapture.gate);
  requireCondition(gateErrors.length === 0,
    'Disposable forced-feature-off gate emitted page errors', gateErrors);
  requireCleanGpuRecords(gateCapture, 'forced-feature-off gate');
  requireCondition(gateRejections.length === 0,
    'Disposable forced-feature-off gate failed', gateRejections);
  assertStableIdentity(identityState, gateCapture.environment, gateCapture.workload,
    'forced-feature-off gate');
  processLifecycle.gate = await closeBrowserProcess(gateState);
  requireCondition(gateErrors.length === 0,
    'Disposable gate emitted a late page error while closing', gateErrors);
  const gateArtifact = await writeExclusiveBrotliJson(
    path.join(runDirectory, 'forced-feature-off-gate.json.br'),
    {
      schemaVersion: 1,
      kind: 'first-instance-live-order-factorial-forced-feature-off-gate',
      runId,
      capturedAt: new Date().toISOString(),
      capture: gateCapture,
      validationRejections: gateRejections,
      pageErrors: gateErrors,
      browserLifecycle: processLifecycle.gate,
    },
  );
  process.stdout.write(`[gate] complete; disposable browser closed\n`);

  const analysisRecords = [];
  const executedSessionIndexes = [...new Set(
    executionPlan.map((trial) => trial.sessionIndex),
  )];
  for (let executionSessionIndex = 0;
    executionSessionIndex < executedSessionIndexes.length;
    executionSessionIndex += 1) {
    const sessionIndex = executedSessionIndexes[executionSessionIndex];
    requireCondition(activeBrowserState === null,
      `Session ${sessionIndex + 1} would overlap a prior browser.`);
    if (executionSessionIndex > 0) {
      requireCondition(processLifecycle.sessions[executionSessionIndex - 1]
        ?.closedBeforeNextLaunch
        === true,
      `Prior browser was not closed before session ${sessionIndex + 1}.`);
    }
    const sessionTrials = executionPlan.filter((trial) => trial.sessionIndex === sessionIndex);
    requireCondition(sessionTrials.length
      === (smokeMode ? 1 : FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIALS_PER_SESSION),
    `Session ${sessionIndex + 1} has the wrong frozen trial count.`);
    process.stdout.write(
      `[session ${executionSessionIndex + 1}/${executedSessionIndexes.length}] `
        + `launching fresh browser for ${sessionTrials.length} trials\n`,
    );
    const pageErrors = [];
    const sessionState = await launchBrowserProcess(
      executablePath,
      'factorial-session',
      sessionIndex + 1,
    );
    const page = await openOnlyPage(sessionState, url, pageErrors);
    for (let sessionExecutionTrialIndex = 0;
      sessionExecutionTrialIndex < sessionTrials.length;
      sessionExecutionTrialIndex += 1) {
      const factorialTrial = sessionTrials[sessionExecutionTrialIndex];
      const compatibilitySpec = compatibilitySpecs[factorialTrial.planIndex];
      const overallOrdinal = analysisRecords.length + 1;
      process.stdout.write(
        `[session ${sessionIndex + 1} trial `
          + `${String(sessionExecutionTrialIndex + 1).padStart(2, '0')}`
          + `/${sessionTrials.length}; overall `
          + `${String(overallOrdinal).padStart(2, '0')}`
          + `/${executionPlan.length}] `
          + `${factorialTrial.factorialCellId}\n`,
      );
      const result = await runFactorialTrial({
        page,
        pageErrors,
        factorialTrial,
        compatibilitySpec,
        identityState,
        usedNonces,
      });
      analysisRecords.push(result.analysisRecord);
      completedTrialArtifacts.push(result.manifestRecord);
      process.stdout.write(
        `  saved ${result.manifestRecord.artifact.path} `
          + `(${result.manifestRecord.artifact.brotliByteLength} bytes)\n`,
      );
    }
    const sessionEnd = await withDeadline(page.evaluate(() => ({
      environment: window.__WEBGPU_BENCH__.environment,
      webgpuUncapturedErrors: window.__WEBGPU_BENCH__.webgpuUncapturedErrors,
      webgpuDeviceLosses: window.__WEBGPU_BENCH__.webgpuDeviceLosses,
    })), BROWSER_OPERATION_TIMEOUT_MS, `session ${sessionIndex + 1} end capture`);
    requireCondition(pageErrors.length === 0,
      `Session ${sessionIndex + 1} emitted page errors`, pageErrors);
    requireCleanGpuRecords(sessionEnd, `session ${sessionIndex + 1} end`);
    requireCondition(sameJson(environmentIdentity(sessionEnd.environment),
      identityState.environment),
    `Session ${sessionIndex + 1} environment drifted at close`, sessionEnd.environment);
    const lifecycle = await closeBrowserProcess(sessionState);
    requireCondition(pageErrors.length === 0,
      `Session ${sessionIndex + 1} emitted a late page error while closing`, pageErrors);
    processLifecycle.sessions.push(lifecycle);
    const sessionManifest = await writeExclusiveJson(
      path.join(runDirectory, `session-${sessionIndex + 1}.json`),
      {
        schemaVersion: 1,
        kind: 'first-instance-live-order-factorial-session-completion',
        runId,
        sessionIndex,
        sessionOrdinal: sessionIndex + 1,
        browserLifecycle: lifecycle,
        endingEnvironment: sessionEnd.environment,
        trials: completedTrialArtifacts.filter((trial) => trial.sessionIndex === sessionIndex),
      },
    );
    completedSessionArtifacts.push(sessionManifest);
    process.stdout.write(
      `[session ${sessionIndex + 1}] browser closed; ${sessionManifest.path}\n`,
    );
  }

  requireCondition(activeBrowserState === null,
    'A browser remained active after both factorial sessions.');
  requireCondition(processLifecycle.sessions.length
    === (smokeMode ? 1 : FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_COUNT),
    'Execution retained the wrong number of session browser lifecycles.');
  requireCondition(browserInstanceSerial === (smokeMode ? 2 : 3)
    && processLifecycle.gate?.browserInstanceSerial === 1
    && processLifecycle.sessions[0]?.browserInstanceSerial === 2
    && (smokeMode || processLifecycle.sessions[1]?.browserInstanceSerial === 3),
  'Execution used the wrong number or order of browser processes',
  processLifecycle);
  const gateClosed = Date.parse(processLifecycle.gate.closedAt);
  const firstLaunched = Date.parse(processLifecycle.sessions[0].launchedAt);
  requireCondition(Number.isFinite(gateClosed)
    && Number.isFinite(firstLaunched)
    && firstLaunched >= gateClosed,
  'Gate/session browser processes overlapped', processLifecycle);
  if (!smokeMode) {
    const firstClosed = Date.parse(processLifecycle.sessions[0].closedAt);
    const secondLaunched = Date.parse(processLifecycle.sessions[1].launchedAt);
    requireCondition(Number.isFinite(firstClosed)
      && Number.isFinite(secondLaunched)
      && secondLaunched >= firstClosed,
    'Factorial session browser processes overlapped', processLifecycle.sessions);
  }
  requireCondition(completedTrialArtifacts.length
    === (smokeMode ? 1 : FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIAL_COUNT),
  'Factorial execution did not persist all trial artifacts.');
  requireCondition(usedNonces.size
    === (smokeMode ? 6 : FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIAL_COUNT * 6),
  'Shader observation challenges were not globally fresh.');

  [sourceProvenanceEnd, executionDependencyClosureEnd] = await Promise.all([
    collectSourceProvenance(projectRoot),
    collectExecutionDependencyClosure(projectRoot),
  ]);
  requireCondition(sourceProvenanceMatches(sourceProvenanceStart, sourceProvenanceEnd),
    'Source provenance changed during order-factorial execution', {
      start: sourceProvenanceStart,
      end: sourceProvenanceEnd,
    });
  requireCondition(executionDependencyClosuresMatch(
    executionDependencyClosureStart,
    executionDependencyClosureEnd,
  ), 'Installed execution dependency closure changed during order-factorial execution', {
    start: executionDependencyClosureStart,
    end: executionDependencyClosureEnd,
  });
  executionIdentityEndArtifact = await writeExclusiveJson(
    path.join(runDirectory, 'execution-identity-end.json'),
    {
      schemaVersion: 1,
      kind: 'first-instance-live-order-factorial-execution-identity-end',
      runId,
      sourceProvenance: sourceProvenanceEnd,
      executionDependencyClosure: executionDependencyClosureEnd,
      sourceMatchesStart: true,
      executionDependencyClosureMatchesStart: true,
    },
  );

  const diagnosticSummary = smokeMode
    ? {
      schemaVersion: 1,
      kind: 'first-instance-live-order-factorial-smoke-summary',
      decision: 'smoke-only-excluded-from-factorial-diagnostic-analysis',
      analysisInvoked: false,
      runId,
      factorialPlanIndex: analysisRecords[0].planIndex,
      factorialCellId: analysisRecords[0].factorialCellId,
      trialSummary: analysisRecords[0].trialSummary,
    }
    : summarizeFirstInstanceLiveOrderFactorial(analysisRecords, runId);
  const summaryArtifact = await writeExclusiveJson(
    path.join(runDirectory, 'diagnostic-summary.json'),
    diagnosticSummary,
  );

  await closeViteServer('Vite server close');
  requireRunActive();
  const runtimeAudit = await finalizeViteRuntimeGuard();
  requireRunActive();
  requireCondition(runtimeAudit.entryHtml.successfulResponseCount === (smokeMode ? 2 : 3),
    'Runtime served an unexpected number of entry documents', runtimeAudit.entryHtml);
  const runtimeAuditArtifact = await writeExclusiveJson(
    path.join(runDirectory, 'vite-runtime-audit.json'),
    runtimeAudit,
  );
  const finalManifest = {
    schemaVersion: 1,
    kind: smokeMode
      ? 'first-instance-live-order-factorial-smoke-manifest'
      : 'first-instance-live-order-factorial-development-manifest',
    scope: smokeMode
      ? 'one-trial plumbing smoke; explicitly excluded from factorial diagnostic analysis'
      : 'development diagnostic; no retry, replacement, outlier deletion, or pass claim',
    runId,
    completedAt: new Date().toISOString(),
    browser: {
      executable: path.basename(executablePath),
      arguments: [...BROWSER_ARGS],
    },
    fixedWorkload: {
      objectCount: OBJECT_COUNT,
      bucketCount: BUCKET_COUNT,
      visibilityFraction: VISIBILITY_FRACTION,
      layout: LAYOUT,
      scenarioSeed: SCENARIO_SEED,
      warmupFrames: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
      measuredFrames: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
      setupPrimeTopology: SETUP_PRIME_TOPOLOGY,
    },
    executionPolicy: {
      smokeMode,
      sessionCount: smokeMode ? 1 : FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_COUNT,
      trialsPerSession: smokeMode
        ? 1
        : FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIALS_PER_SESSION,
      retryCount: 0,
      replacementAllowed: false,
      outlierRemovalAllowed: false,
      efficacyStoppingAllowed: false,
      artificialTimestampMapDelayMs: 0,
    },
    compatibilityMapping: COMPATIBILITY_MAPPING,
    environmentIdentity: identityState.environment,
    workloadIdentity: identityState.workload,
    executionIdentity: {
      sourceProvenanceStart,
      sourceProvenanceEnd,
      sourceProvenanceMatched: true,
      executionDependencyClosureStart,
      executionDependencyClosureEnd,
      executionDependencyClosureMatched: true,
    },
    processLifecycle,
    artifacts: {
      plan: planArtifact,
      forcedFeatureOffGate: gateArtifact,
      executionIdentityStart: executionIdentityStartArtifact,
      executionIdentityEnd: executionIdentityEndArtifact,
      sessions: completedSessionArtifacts,
      trials: completedTrialArtifacts,
      diagnosticSummary: summaryArtifact,
      viteRuntimeAudit: runtimeAuditArtifact,
    },
    completion: {
      frozenTrialCount: factorialPlan.length,
      executedTrialCount: executionPlan.length,
      persistedTrialCount: completedTrialArtifacts.length,
      freshShaderChallengeCount: usedNonces.size,
      entryDocumentCount: runtimeAudit.entryHtml.successfulResponseCount,
      allBrowsersClosed: activeBrowserState === null,
      sessionOverlapDetected: false,
    },
  };
  const manifestArtifact = await writeExclusiveJson(
    path.join(runDirectory, 'manifest.json'),
    finalManifest,
  );
  runCompletionCommitted = true;
  process.stdout.write(
    `Order-factorial ${smokeMode ? 'staged smoke' : 'diagnostic'} complete.\n`
      + `  directory: ${runDirectory}\n`
      + `  manifest: ${path.join(runDirectory, manifestArtifact.path)}\n`
      + `  summary: ${path.join(runDirectory, summaryArtifact.path)}\n`,
  );
} catch (error) {
  const abortedBrowserLifecycle = await cleanupOwnedResources(
    terminationSignal ?? 'failed-run',
  );
  if (terminationSignal !== null) {
    await writeExclusiveJson(path.join(runDirectory, 'interruption.json'), {
      schemaVersion: 1,
      kind: smokeMode
        ? 'first-instance-live-order-factorial-smoke-interruption'
        : 'first-instance-live-order-factorial-development-interruption',
      scope: smokeMode
        ? 'interrupted one-trial plumbing smoke; excluded from factorial analysis'
        : 'interrupted development diagnostic; never eligible for factorial analysis',
      runId,
      signal: terminationSignal,
      interruptedAt: new Date().toISOString(),
      completedTrialArtifacts,
      completedSessionArtifacts,
      abortedBrowserLifecycle,
      processLifecycle,
    }, { allowDuringTermination: true }).catch(() => undefined);
  } else {
    await writeExclusiveJson(path.join(runDirectory, 'failure.json'), {
      schemaVersion: 1,
      kind: smokeMode
        ? 'first-instance-live-order-factorial-smoke-failure'
        : 'first-instance-live-order-factorial-development-failure',
      scope: smokeMode
        ? 'failed one-trial plumbing smoke; excluded from factorial analysis'
        : 'failed closed; no retry or replacement attempted',
      runId,
      failedAt: new Date().toISOString(),
      error: {
        name: error?.name ?? 'Error',
        message: error?.message ?? String(error),
        stack: error?.stack ?? null,
      },
      completedTrialArtifacts,
      completedSessionArtifacts,
      sourceProvenanceStart,
      sourceProvenanceEnd,
      executionDependencyClosureStart,
      executionDependencyClosureEnd,
      processLifecycle,
    }, { allowDuringTermination: true }).catch(() => undefined);
    throw error;
  }
} finally {
  await cleanupOwnedResources(terminationSignal ?? 'final-cleanup');
}
