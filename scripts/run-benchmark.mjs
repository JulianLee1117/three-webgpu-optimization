import {
  access,
  mkdir,
  open,
  readFile,
  rename,
  unlink,
  writeFile,
} from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import path from 'node:path';
import process from 'node:process';
import { performance } from 'node:perf_hooks';
import { chromium } from 'playwright-core';
import { createServer } from 'vite';
import { NvidiaTelemetryRecorder } from './nvidia-telemetry.mjs';
import {
  BENCHMARK_VISIBILITY_LEVELS,
  DEPTH_ORDERING_LAYOUTS,
  DEPTH_ORDERING_MODES,
  DEPTH_ORDERING_VISIBILITY,
  FIXED_SLICE_REPRESENTATION_MODES,
  FROZEN_DEPTH_CROSSOVER_LANES,
  FROZEN_DEPTH_CROSSOVER_MODE,
  FROZEN_DEPTH_CROSSOVER_REPETITIONS,
  assertBalancedModeOrders,
  buildBenchmarkPlan,
  buildDepthOrderingPlan,
  buildFrozenDepthCrossoverPlan,
  createEcosystemModeOrders,
  createRepresentationModeOrders,
} from '../src/benchmark/plan.js';
import {
  FROZEN_CROSSOVER_BLOCK_SIZE,
  FROZEN_CROSSOVER_MEASURED_BLOCKS,
  FROZEN_CROSSOVER_MEASURED_FRAMES,
  FROZEN_CROSSOVER_PATTERNS,
  FROZEN_CROSSOVER_WARMUP_BLOCKS,
  FROZEN_CROSSOVER_WARMUP_FRAMES,
  frozenCrossoverFrame,
} from '../src/benchmark/frozen-crossover-schedule.js';
import {
  physicalBinSequenceIdentity,
  renderParityIdentity,
  sha256Json,
  validateBenchmarkProtocolEnvironment,
  validateExactValidation,
  validateGeometryFixtureManifest,
  validateRenderParity,
  validateScenarioManifest,
  validateTrialRows,
} from './evidence-validation.mjs';
import {
  collectSourceProvenance,
  sourceProvenanceMatches,
} from './source-provenance.mjs';

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const RESULT_ROOT = path.resolve(
  process.env.BENCHMARK_RESULT_ROOT ?? path.join(PROJECT_ROOT, 'results', 'runs'),
);
const ALLOWED_EVIDENCE_STATUSES = Object.freeze(['development', 'candidate']);
const EVIDENCE_STATUS = validatedStringEnvironmentChoice(
  'BENCHMARK_EVIDENCE_STATUS',
  'development',
  ALLOWED_EVIDENCE_STATUSES,
);
const ENVIRONMENT_NOTE = process.env.BENCHMARK_ENVIRONMENT_NOTE ?? null;
const MAXIMUM_CPU_TIMER_QUANTUM_MS = 0.01;
const ARTIFACT_SCHEMA_VERSION = 2;
const ALLOWED_OBJECT_COUNTS = Object.freeze([4_096, 16_384, 65_536]);
const ALLOWED_BUCKET_COUNTS = Object.freeze([1, 4, 32, 128]);
const ALLOWED_HETEROGENEOUS_COMPARATORS = Object.freeze([
  'coalesced-v11',
  'historical-v10',
]);
const ALLOWED_BENCHMARK_MATRICES = Object.freeze([
  'ecosystem',
  'fixed-slice-representation',
  'depth-ordering',
  'depth-ordering-render-only',
]);
const LEGACY_WARMUP_FRAMES = 300;
const LEGACY_MEASURED_FRAMES = 240;
const SCENARIO_SEED = 0xb1ad_2026;
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

function validatedEnvironmentChoice(name, fallback, allowedValues) {
  const rawValue = process.env[name];
  if (rawValue === undefined) return fallback;
  const value = rawValue.trim();
  const allowedStrings = allowedValues.map(String);
  if (!allowedStrings.includes(value)) {
    throw new Error(`${name} must be one of ${allowedStrings.join(', ')}; received ${JSON.stringify(rawValue)}.`);
  }
  return Number(value);
}

function validatedStringEnvironmentChoice(name, fallback, allowedValues) {
  const rawValue = process.env[name];
  if (rawValue === undefined) return fallback;
  const value = rawValue.trim();
  if (!allowedValues.includes(value)) {
    throw new Error(`${name} must be one of ${allowedValues.join(', ')}; received ${JSON.stringify(rawValue)}.`);
  }
  return value;
}

const BENCHMARK_MATRIX = validatedStringEnvironmentChoice(
  'BENCHMARK_MATRIX',
  'ecosystem',
  ALLOWED_BENCHMARK_MATRICES,
);
const FROZEN_CROSSOVER_MATRIX = BENCHMARK_MATRIX === 'depth-ordering-render-only';
const DEPTH_LAYOUT_MATRIX = BENCHMARK_MATRIX === 'depth-ordering'
  || FROZEN_CROSSOVER_MATRIX;
const WARMUP_FRAMES = FROZEN_CROSSOVER_MATRIX
  ? FROZEN_CROSSOVER_WARMUP_FRAMES
  : LEGACY_WARMUP_FRAMES;
const MEASURED_FRAMES = FROZEN_CROSSOVER_MATRIX
  ? FROZEN_CROSSOVER_MEASURED_FRAMES
  : LEGACY_MEASURED_FRAMES;

const OBJECT_COUNT = validatedEnvironmentChoice(
  'BENCHMARK_OBJECT_COUNT',
  DEPTH_LAYOUT_MATRIX ? 65_536 : 16_384,
  ALLOWED_OBJECT_COUNTS,
);
const BUCKET_COUNT = validatedEnvironmentChoice(
  'BENCHMARK_BUCKET_COUNT',
  32,
  ALLOWED_BUCKET_COUNTS,
);
if (DEPTH_LAYOUT_MATRIX
  && (OBJECT_COUNT !== 65_536 || BUCKET_COUNT !== 32)) {
  throw new Error(
    `The ${BENCHMARK_MATRIX} protocol is fixed at BENCHMARK_OBJECT_COUNT=65536 and `
      + 'BENCHMARK_BUCKET_COUNT=32.',
  );
}
const HETEROGENEOUS_COMPARATOR = validatedStringEnvironmentChoice(
  'BENCHMARK_HETEROGENEOUS_COMPARATOR',
  'coalesced-v11',
  ALLOWED_HETEROGENEOUS_COMPARATORS,
);
if (BENCHMARK_MATRIX !== 'ecosystem'
  && process.env.BENCHMARK_HETEROGENEOUS_COMPARATOR !== undefined) {
  throw new Error(
    `BENCHMARK_HETEROGENEOUS_COMPARATOR does not apply to the ${BENCHMARK_MATRIX} matrix.`,
  );
}
if (BENCHMARK_MATRIX === 'ecosystem'
  && BUCKET_COUNT === 1
  && process.env.BENCHMARK_HETEROGENEOUS_COMPARATOR !== undefined
  && HETEROGENEOUS_COMPARATOR !== 'coalesced-v11') {
  throw new Error(
    'BENCHMARK_HETEROGENEOUS_COMPARATOR applies only when BENCHMARK_BUCKET_COUNT is 4, 32, or 128; '
    + 'single-bucket runs retain the Three Blocks 0.11 public baseline.',
  );
}
const THREE_BLOCKS_MODE = BUCKET_COUNT === 1
  ? 'three-blocks-current'
  : HETEROGENEOUS_COMPARATOR === 'historical-v10'
    ? 'three-blocks-historical'
    : 'three-blocks-coalesced';
const ECOSYSTEM_MODES = Object.freeze(['draw-all', THREE_BLOCKS_MODE, 'fixed-slice']);
const REPRESENTATION_MODES = FIXED_SLICE_REPRESENTATION_MODES;
const MODES = BENCHMARK_MATRIX === 'fixed-slice-representation'
  ? REPRESENTATION_MODES
  : FROZEN_CROSSOVER_MATRIX
    ? Object.freeze([FROZEN_DEPTH_CROSSOVER_MODE])
    : BENCHMARK_MATRIX === 'depth-ordering'
      ? DEPTH_ORDERING_MODES
      : ECOSYSTEM_MODES;
const MODE_ORDERS = BENCHMARK_MATRIX === 'fixed-slice-representation'
  ? createRepresentationModeOrders(REPRESENTATION_MODES)
  : FROZEN_CROSSOVER_MATRIX
    ? Object.freeze(Array.from(
      { length: FROZEN_DEPTH_CROSSOVER_REPETITIONS },
      () => Object.freeze([FROZEN_DEPTH_CROSSOVER_MODE]),
    ))
    : createEcosystemModeOrders(MODES);
const VISIBILITY_LEVELS = DEPTH_LAYOUT_MATRIX
  ? Object.freeze([DEPTH_ORDERING_VISIBILITY])
  : BENCHMARK_VISIBILITY_LEVELS;

assertBalancedModeOrders(MODES, MODE_ORDERS);
const DEPTH_BINNED_MODE_IDS = new Set([
  'fixed-slice-depth-front-to-back',
  'fixed-slice-depth-reverse',
]);
const MATRIX_ID = `${BENCHMARK_MATRIX}-o${OBJECT_COUNT}-b${BUCKET_COUNT}`;

function frozenScheduleCommitmentRecord(orientationOffset) {
  const rows = (frameCount) => Array.from({ length: frameCount }, (_, phaseFrameIndex) => {
    const scheduled = frozenCrossoverFrame(phaseFrameIndex, orientationOffset);
    return {
      phaseFrameIndex,
      crossoverBlockIndex: scheduled.crossoverBlockIndex,
      withinBlockPosition: scheduled.withinBlockPosition,
      patternIndex: scheduled.patternIndex,
      pattern: scheduled.pattern,
      laneId: scheduled.laneId,
    };
  });
  return {
    schemaVersion: 1,
    kind: 'frozen-depth-crossover-frame-schedule',
    blockSize: FROZEN_CROSSOVER_BLOCK_SIZE,
    warmupFrames: FROZEN_CROSSOVER_WARMUP_FRAMES,
    measuredFrames: FROZEN_CROSSOVER_MEASURED_FRAMES,
    orientationOffset,
    warmup: rows(FROZEN_CROSSOVER_WARMUP_FRAMES),
    measured: rows(FROZEN_CROSSOVER_MEASURED_FRAMES),
  };
}

const FROZEN_SCHEDULE_SHA256_BY_ORIENTATION = FROZEN_CROSSOVER_MATRIX
  ? Object.freeze(Object.fromEntries([0, 1].map((orientationOffset) => [
    String(orientationOffset),
    sha256Json(frozenScheduleCommitmentRecord(orientationOffset)),
  ])))
  : null;
const sourceProvenanceStart = await collectSourceProvenance(PROJECT_ROOT, {
  allowUnavailable: EVIDENCE_STATUS === 'development',
});
if (EVIDENCE_STATUS !== 'development'
  && (sourceProvenanceStart.status !== 'available'
    || sourceProvenanceStart.captureStable !== true
    || sourceProvenanceStart.dirty
    || sourceProvenanceStart.packageLockTracked !== true)) {
  throw new Error(
    `Evidence status ${JSON.stringify(EVIDENCE_STATUS)} requires a clean, Git-tracked source tree.`,
  );
}

function compactTimestamp(isoTimestamp) {
  return isoTimestamp.replaceAll(':', '-');
}

function percentile(values, fraction) {
  const finite = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (finite.length === 0) return null;
  const index = Math.min(finite.length - 1, Math.max(0, Math.ceil(finite.length * fraction) - 1));
  return finite[index];
}

function escapeCsv(value) {
  if (value === null || value === undefined) return '';
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function rowsToCsv(rows) {
  if (rows.length === 0) return '';
  const fields = [...new Set(rows.flatMap((row) => Object.keys(row)))].sort();
  return [
    fields.join(','),
    ...rows.map((row) => fields.map((field) => escapeCsv(row[field])).join(',')),
  ].join('\n');
}

function sha256Bytes(value) {
  return createHash('sha256').update(value).digest('hex');
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function artifactSafeText(value) {
  const roots = [PROJECT_ROOT, PROJECT_ROOT.replaceAll('\\', '/')]
    .sort((left, right) => right.length - left.length);
  return roots.reduce(
    (text, root) => text.replace(new RegExp(escapeRegExp(root), 'gi'), '[project-root]'),
    String(value),
  );
}

async function atomicWriteJson(filename, value) {
  const temporary = `${filename}.${process.pid}.tmp`;
  const handle = await open(temporary, 'w');
  let writeError = null;
  try {
    await handle.writeFile(`${JSON.stringify(value, null, 2)}\n`, 'utf8');
    await handle.sync();
  } catch (error) {
    writeError = error;
  } finally {
    await handle.close();
  }
  if (writeError) {
    await unlink(temporary).catch(() => undefined);
    throw writeError;
  }
  try {
    await rename(temporary, filename);
  } catch (error) {
    await unlink(temporary).catch(() => undefined);
    throw error;
  }
}

function standardBrowserCandidates() {
  const programFiles = process.env.ProgramFiles ?? 'C:\\Program Files';
  const programFilesX86 = process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)';
  const localAppData = process.env.LOCALAPPDATA;
  return [
    process.env.BROWSER_PATH,
    path.join(programFiles, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    path.join(programFilesX86, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    path.join(programFiles, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    path.join(programFilesX86, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    localAppData && path.join(localAppData, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    '/usr/bin/google-chrome',
    '/usr/bin/microsoft-edge',
    '/usr/bin/chromium',
  ].filter(Boolean);
}

async function findBrowser() {
  for (const candidate of [...new Set(standardBrowserCandidates())]) {
    try {
      await access(candidate);
      return path.resolve(candidate);
    } catch {
      // Continue through known system-browser locations.
    }
  }
  throw new Error('No installed Chrome or Edge executable found. Set BROWSER_PATH.');
}

function errorRecord(error) {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: artifactSafeText(error.message),
      stack: error.stack ? artifactSafeText(error.stack) : null,
    };
  }
  return { name: 'Error', message: artifactSafeText(error), stack: null };
}

function validateBenchmarkEnvironment(environment) {
  const rejectionReasons = [];
  if (environment?.crossOriginIsolated !== true) {
    rejectionReasons.push('crossOriginIsolated is not true');
  }
  if (!Number.isFinite(environment?.performanceNowQuantumMs)) {
    rejectionReasons.push('performanceNowQuantumMs is missing or nonfinite');
  } else if (environment.performanceNowQuantumMs > MAXIMUM_CPU_TIMER_QUANTUM_MS) {
    rejectionReasons.push(
      `performanceNowQuantumMs ${environment.performanceNowQuantumMs} exceeds ${MAXIMUM_CPU_TIMER_QUANTUM_MS}`,
    );
  }
  if (THREE_BLOCKS_MODE === 'three-blocks-historical'
    && environment?.indirectFirstInstanceAvailable !== true) {
    rejectionReasons.push(
      'the historical Three Blocks comparator requires the WebGPU indirect-first-instance feature',
    );
  }
  rejectionReasons.push(...validateBenchmarkProtocolEnvironment({
    matrixKind: BENCHMARK_MATRIX,
    reversedDepthBuffer: DEPTH_LAYOUT_MATRIX ? true : null,
    minimumStorageBuffersPerShaderStage:
      DEPTH_LAYOUT_MATRIX ? 8 : null,
  }, environment));
  if (FROZEN_CROSSOVER_MATRIX
    && environment?.rendererReversedDepthBuffer !== true) {
    rejectionReasons.push(
      'the frozen depth crossover requires the renderer reversed-depth buffer',
    );
  }
  if (FROZEN_CROSSOVER_MATRIX
    && (!Number.isInteger(environment?.maxStorageBuffersPerShaderStage)
      || environment.maxStorageBuffersPerShaderStage < 8)) {
    rejectionReasons.push(
      'the frozen depth crossover requires at least eight storage buffers per shader stage',
    );
  }
  if (rejectionReasons.length > 0) {
    throw new Error(`Benchmark environment rejected at startup: ${rejectionReasons.join('; ')}.`);
  }
}

async function configureAndValidateTrial(page, spec) {
  return page.evaluate(async (trial) => {
    const bench = window.__WEBGPU_BENCH__;
    if (!bench) throw new Error('Benchmark page API is unavailable.');
    const frozenCrossover = trial.modeId === 'fixed-slice-depth-frozen-crossover';
    if (frozenCrossover) {
      bench.configureFrozenCrossover({
        laneStorageOrder: trial.laneStorageOrder,
        superblockOrientationOffset: trial.superblockOrientationOffset,
      });
    }
    const selections = {
      strategy: trial.modeId,
      objects: String(trial.objectCount),
      buckets: String(trial.bucketCount),
      visibility: String(trial.visibilityFraction),
      layout: trial.layout,
    };
    for (const [id, value] of Object.entries(selections)) {
      const select = document.getElementById(id);
      if (!select) throw new Error(`Missing benchmark control: ${id}`);
      if (![...select.options].some((option) => option.value === value)) {
        throw new Error(`Benchmark control ${id} does not support ${value}.`);
      }
      select.value = value;
    }
    await bench.rebuild();
    const selected = bench.selectedConfig();
    if (selected.strategyId !== trial.modeId
      || selected.objectCount !== trial.objectCount
      || selected.bucketCount !== trial.bucketCount
      || selected.visibilityFraction !== trial.visibilityFraction
      || selected.layout !== trial.layout
      || (frozenCrossover
        && (selected.superblockOrientationOffset !== trial.superblockOrientationOffset
          || !Array.isArray(selected.laneStorageOrder)
          || selected.laneStorageOrder.length !== trial.laneStorageOrder.length
          || selected.laneStorageOrder.some(
            (laneId, index) => laneId !== trial.laneStorageOrder[index],
          )))) {
      throw new Error(`Page configuration mismatch: ${JSON.stringify(selected)}`);
    }
    let validation = null;
    let validationError = null;
    let renderParity = null;
    try {
      validation = await bench.validate();
      if (trial.layout !== 'baseline') {
        renderParity = await bench.captureRenderParity();
        validation = await bench.validate();
      }
    } catch (error) {
      validation = structuredClone(bench.lastValidation);
      validationError = error instanceof Error ? error.message : String(error);
    }
    const workload = await bench.fingerprintWorkload();
    return {
      selectedConfig: selected,
      validation,
      validationError,
      renderParity,
      workload,
    };
  }, spec);
}

async function startConfiguredTrial(page, auditContext, capturePairedRenderParity = false) {
  return page.evaluate(async ({ context, captureParity }) => {
    const bench = window.__WEBGPU_BENCH__;
    if (!bench) throw new Error('Benchmark page API is unavailable.');
    let renderParity = null;
    try {
      if (captureParity) renderParity = await bench.captureRenderParity();
      const evidence = await bench.startTrial(context);
      return {
        evidence: { ...evidence, renderParity },
        phase: bench.phase,
        trialError: bench.trialError,
        startError: null,
      };
    } catch (error) {
      return {
        evidence: {
          validation: structuredClone(bench.lastValidation),
          workload: await bench.fingerprintWorkload(),
          renderParity,
        },
        phase: bench.phase,
        trialError: bench.trialError,
        startError: error instanceof Error ? error.message : String(error),
      };
    }
  }, { context: auditContext, captureParity: capturePairedRenderParity });
}

async function collectPostTrialEvidence(page, capturePairedRenderParity = false) {
  return page.evaluate(async (captureParity) => {
    const bench = window.__WEBGPU_BENCH__;
    if (!bench) throw new Error('Benchmark page API is unavailable.');
    let validation = null;
    let validationError = null;
    let renderParity = null;
    try {
      validation = await bench.validate();
      if (captureParity) {
        renderParity = await bench.captureRenderParity();
        validation = await bench.validate();
      }
    } catch (error) {
      validation = structuredClone(bench.lastValidation);
      validationError = error instanceof Error ? error.message : String(error);
    }
    const workload = await bench.fingerprintWorkload();
    return { validation, validationError, renderParity, workload };
  }, capturePairedRenderParity);
}

const startedAt = new Date().toISOString();
const runId = `${MATRIX_ID}-${compactTimestamp(startedAt)}`;
const runDirectory = path.join(RESULT_ROOT, runId);
const plan = (FROZEN_CROSSOVER_MATRIX
  ? buildFrozenDepthCrossoverPlan({
    runId,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  })
  : BENCHMARK_MATRIX === 'depth-ordering'
    ? buildDepthOrderingPlan({
      runId,
      modeOrders: MODE_ORDERS,
      objectCount: OBJECT_COUNT,
      bucketCount: BUCKET_COUNT,
    })
    : buildBenchmarkPlan({
      runId,
      modeOrders: MODE_ORDERS,
      visibilityLevels: VISIBILITY_LEVELS,
      objectCount: OBJECT_COUNT,
      bucketCount: BUCKET_COUNT,
    })).map((trial) => ({ ...trial, runId }));
const runStartedMonotonic = performance.now();
const frameRows = [];
const trialSummaries = [];
const validationArtifacts = [];
const workloadManifestCatalog = {
  schemaVersion: 1,
  hashAlgorithm: 'sha256',
  geometryFixturesBySha256: {},
  scenariosBySha256: {},
  invalidObservations: [],
};
const pageErrors = [];
let geometryFixtureManifest = null;
let scenarioSeed = null;
const scenarioSha256ByCell = new Map();
const renderParitySha256ByCell = new Map();
const physicalBinSequenceByPair = new Map();
let telemetryContext = Object.freeze({
  phase: 'startup',
  trialId: null,
  planIndex: null,
  repetitionIndex: null,
  modeId: null,
  visibilityFraction: null,
  layout: null,
});
let backend = null;
let pageEnvironment = null;
let browserMetadata = null;
let browser = null;
let page = null;
let server = null;
let runError = null;
let activeValidationArtifact = null;
let preComputeProcesses = null;
let postComputeProcesses = null;
let telemetryReport = null;

await mkdir(RESULT_ROOT, { recursive: true });
await mkdir(runDirectory);

function refreshArtifactDigest(artifact) {
  delete artifact.sha256;
  artifact.sha256 = sha256Json(artifact);
}

async function persistValidationArtifacts() {
  await atomicWriteJson(path.join(runDirectory, 'validation-artifacts.json'), validationArtifacts);
}

async function persistWorkloadManifests() {
  await atomicWriteJson(path.join(runDirectory, 'workload-manifests.json'), workloadManifestCatalog);
}

function inspectEvidenceCapture(spec, evidence) {
  const validation = evidence?.validation ?? null;
  const workload = evidence?.workload ?? null;
  const geometry = workload?.geometryFixtures ?? null;
  const scenario = workload?.scenario ?? null;
  const rejectionReasons = [];
  if (evidence?.validationError) {
    rejectionReasons.push(`validation threw: ${artifactSafeText(evidence.validationError)}`);
  }
  if (workload?.scenarioSeed !== SCENARIO_SEED) {
    rejectionReasons.push(`workload scenario seed is ${JSON.stringify(workload?.scenarioSeed)}; expected ${SCENARIO_SEED}`);
  }
  const geometryRejectionReasons = validateGeometryFixtureManifest(geometry, {
    bucketCount: spec.bucketCount,
    tier: 'medium',
  });
  const scenarioRejectionReasons = validateScenarioManifest(scenario, {
    objectCount: spec.objectCount,
    bucketCount: spec.bucketCount,
    visibilityFraction: spec.visibilityFraction,
    seed: SCENARIO_SEED,
    layout: spec.layout,
  });
  rejectionReasons.push(...geometryRejectionReasons, ...scenarioRejectionReasons);
  const validationCheck = validateExactValidation(validation, {
    modeId: spec.modeId,
    objectCount: spec.objectCount,
    bucketCount: spec.bucketCount,
    expectedVisibleCount: scenario?.expectedVisibleCount,
    expectedVisibleIdsCanonicalSha256: scenario?.expectedVisibleIdsCanonicalSha256,
    geometryManifest: geometry,
    scenarioManifest: scenario,
    laneStorageOrder: spec.laneStorageOrder ?? null,
  });
  rejectionReasons.push(...validationCheck.rejectionReasons);
  return {
    capture: {
      capturedAt: new Date().toISOString(),
      workload: {
        scenarioSeed: workload?.scenarioSeed ?? null,
        geometryFixtureSha256: geometry?.sha256 ?? null,
        scenarioSha256: scenario?.sha256 ?? null,
      },
      validation: {
        payloadSha256: validation ? sha256Json(validation) : null,
        semanticSha256: validationCheck.semanticSha256 ?? null,
        payload: validation,
      },
      renderParity: evidence?.renderParity ?? null,
      accepted: rejectionReasons.length === 0,
      rejectionReasons: [...new Set(rejectionReasons)],
    },
    geometryManifest: geometry,
    scenarioManifest: scenario,
    geometryManifestAccepted: geometryRejectionReasons.length === 0,
    scenarioManifestAccepted: scenarioRejectionReasons.length === 0,
  };
}

async function registerWorkloadManifests(spec, phase, inspection) {
  let changed = false;
  if (inspection.geometryManifestAccepted) {
    const digest = inspection.geometryManifest.sha256;
    if (!(digest in workloadManifestCatalog.geometryFixturesBySha256)) {
      workloadManifestCatalog.geometryFixturesBySha256[digest] = inspection.geometryManifest;
      changed = true;
    }
  }
  if (inspection.scenarioManifestAccepted) {
    const digest = inspection.scenarioManifest.sha256;
    if (!(digest in workloadManifestCatalog.scenariosBySha256)) {
      workloadManifestCatalog.scenariosBySha256[digest] = inspection.scenarioManifest;
      changed = true;
    }
  }
  if (!inspection.geometryManifestAccepted || !inspection.scenarioManifestAccepted) {
    workloadManifestCatalog.invalidObservations.push({
      trialId: spec.trialId,
      planIndex: spec.planIndex,
      phase,
      geometryFixtures: inspection.geometryManifest,
      scenario: inspection.scenarioManifest,
    });
    changed = true;
  }
  if (changed) await persistWorkloadManifests();
}

await Promise.all([persistValidationArtifacts(), persistWorkloadManifests()]);

const telemetry = new NvidiaTelemetryRecorder({
  runId,
  runDirectory,
  runStartedMonotonic,
  getContext: () => telemetryContext,
});
try {
  await telemetry.start();
  preComputeProcesses = await telemetry.captureComputeSnapshot('pre-run');
} catch {
  // Telemetry is environmental evidence, not a technical benchmark prerequisite.
}

let rejectPageError;
const pageErrorSignal = new Promise((resolve, reject) => {
  void resolve;
  rejectPageError = reject;
});
pageErrorSignal.catch(() => undefined);

function capturePageError(source, detail) {
  const record = { source, detail: artifactSafeText(detail), timestamp: new Date().toISOString() };
  pageErrors.push(record);
  if (pageErrors.length === 1) rejectPageError(new Error(`${source}: ${detail}`));
}

function guarded(operation) {
  return Promise.race([operation, pageErrorSignal]);
}

try {
  telemetryContext = Object.freeze({ ...telemetryContext, phase: 'browser-startup' });
  const executablePath = await findBrowser();
  server = await createServer({
    root: PROJECT_ROOT,
    configFile: false,
    resolve: { dedupe: ['three'] },
    server: {
      host: '127.0.0.1',
      port: 0,
      headers: ISOLATION_HEADERS,
    },
    logLevel: 'error',
  });
  await server.listen();
  const url = server.resolvedUrls?.local?.[0];
  if (!url) throw new Error('Vite did not expose a local URL.');

  browser = await chromium.launch({
    executablePath,
    headless: true,
    args: [...BROWSER_ARGS],
  });
  browserMetadata = {
    executable: path.basename(executablePath),
    version: browser.version(),
    headless: true,
    args: [...BROWSER_ARGS],
    viewport: { width: 1280, height: 900, deviceScaleFactor: 1 },
  };
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    deviceScaleFactor: 1,
  });
  page = await context.newPage();
  page.on('pageerror', (error) => capturePageError('pageerror', error.stack ?? error.message));
  page.on('console', (message) => {
    if (message.type() !== 'error') return;
    const location = message.location();
    const suffix = location.url
      ? ` (${location.url}:${location.lineNumber ?? 0}:${location.columnNumber ?? 0})`
      : '';
    capturePageError('console', `${message.text()}${suffix}`);
  });

  await guarded(page.goto(url, { waitUntil: 'domcontentloaded' }));
  await guarded(page.waitForFunction(
    () => window.__WEBGPU_BENCH__?.ready === true,
    null,
    { timeout: 120_000 },
  ));
  backend = await guarded(page.locator('#backend').textContent());
  pageEnvironment = await guarded(page.evaluate(
    () => window.__WEBGPU_BENCH__?.environment ?? null,
  ));
  validateBenchmarkEnvironment(pageEnvironment);

  for (const spec of plan) {
    if (pageErrors.length) throw new Error(`${pageErrors[0].source}: ${pageErrors[0].detail}`);
    const trialStartedAt = new Date().toISOString();
    const trialStartedMonotonic = performance.now();
    const auditContext = {
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
      protocolWarmupFrames: WARMUP_FRAMES,
      protocolMeasuredFrames: MEASURED_FRAMES,
      ...(FROZEN_CROSSOVER_MATRIX ? {
        plannedLaneStorageOrder: spec.laneStorageOrder.join('|'),
        superblockOrientationOffset: spec.superblockOrientationOffset,
        plannedScheduleSha256:
          FROZEN_SCHEDULE_SHA256_BY_ORIENTATION[String(spec.superblockOrientationOffset)],
      } : {}),
    };

    telemetryContext = Object.freeze({
      phase: 'validation',
      trialId: spec.trialId,
      planIndex: spec.planIndex,
      repetitionIndex: spec.repetitionIndex,
      modeId: spec.modeId,
      visibilityFraction: spec.visibilityFraction,
      layout: spec.layout,
    });
    const prepared = await guarded(configureAndValidateTrial(page, spec));
    const selectedConfig = prepared?.selectedConfig ?? null;
    const preflightInspection = inspectEvidenceCapture(spec, prepared);
    await registerWorkloadManifests(spec, 'preflight', preflightInspection);
    const preflight = preflightInspection.capture;
    const fixtureManifest = preflightInspection.geometryManifest;
    const scenarioManifest = preflightInspection.scenarioManifest;
    const preflightRejectionReasons = [...preflight.rejectionReasons];
    if (geometryFixtureManifest !== null
      && geometryFixtureManifest.sha256 !== fixtureManifest?.sha256) {
      preflightRejectionReasons.push('geometry fixture manifest changed between trials');
    }
    if (geometryFixtureManifest === null && preflight.accepted) {
      geometryFixtureManifest = fixtureManifest;
    }
    if (scenarioSeed !== null && scenarioSeed !== preflight.workload?.scenarioSeed) {
      preflightRejectionReasons.push('scenario seed changed between trials');
    }
    if (scenarioSeed === null && preflight.accepted) scenarioSeed = preflight.workload.scenarioSeed;
    const scenarioKey = `${spec.layout}|${spec.visibilityFraction}`;
    const priorScenarioSha256 = scenarioSha256ByCell.get(scenarioKey);
    if (priorScenarioSha256 !== undefined && priorScenarioSha256 !== scenarioManifest?.sha256) {
      preflightRejectionReasons.push('scenario manifest changed within a visibility cell');
    }
    if (priorScenarioSha256 === undefined && preflight.accepted) {
      scenarioSha256ByCell.set(scenarioKey, scenarioManifest.sha256);
    }
    if (DEPTH_LAYOUT_MATRIX) {
      const parityRejectionReasons = validateRenderParity(preflight.renderParity, {
        spec,
        geometryManifest: fixtureManifest,
        scenarioManifest,
      });
      preflightRejectionReasons.push(...parityRejectionReasons);
      if (parityRejectionReasons.length === 0) {
        const paritySha256 = renderParityIdentity(preflight.renderParity);
        const priorParitySha256 = renderParitySha256ByCell.get(scenarioKey);
        if (priorParitySha256 !== undefined && priorParitySha256 !== paritySha256) {
          preflightRejectionReasons.push(
            'exact color/depth/object-ID/material parity changed within a layout cell',
          );
        } else if (priorParitySha256 === undefined) {
          renderParitySha256ByCell.set(scenarioKey, paritySha256);
        }
      }
      if (FROZEN_CROSSOVER_MATRIX
        && (preflight.renderParity?.snapshotValidation === undefined
          || sha256Json(preflight.renderParity.snapshotValidation)
            !== preflight.validation.payloadSha256)) {
        preflightRejectionReasons.push(
          'frozen preflight parity snapshot differs from its exact validation payload',
        );
      }
      if (DEPTH_BINNED_MODE_IDS.has(spec.modeId)) {
        const validationSequenceSha256 = physicalBinSequenceIdentity(
          preflight.validation?.payload,
        );
        const paritySequenceSha256 = physicalBinSequenceIdentity(
          preflight.renderParity?.snapshotValidation,
        );
        if (validationSequenceSha256 === null || paritySequenceSha256 === null) {
          preflightRejectionReasons.push(
            'ordered lane lacks a physical-bin sequence commitment',
          );
        } else if (validationSequenceSha256 !== paritySequenceSha256) {
          preflightRejectionReasons.push(
            'ordered lane physical-bin sequence changed between parity and preflight snapshots',
          );
        } else {
          const pairKey = [
            spec.repetitionIndex,
            spec.layout,
            spec.visibilityFraction,
            scenarioManifest?.sha256,
          ].join('|');
          const prior = physicalBinSequenceByPair.get(pairKey);
          if (prior !== undefined && prior.sha256 !== validationSequenceSha256) {
            preflightRejectionReasons.push(
              'paired front/reverse physical-bin sequences differ after traversal normalization',
            );
          } else if (prior === undefined) {
            physicalBinSequenceByPair.set(pairKey, {
              sha256: validationSequenceSha256,
              modeIds: new Set([spec.modeId]),
            });
          } else {
            prior.modeIds.add(spec.modeId);
          }
        }
      }
      if (FROZEN_CROSSOVER_MATRIX) {
        const validationSequenceSha256 = preflight.validation?.payload
          ?.physicalBinSequenceSha256 ?? null;
        const paritySequenceSha256 = preflight.renderParity?.snapshotValidation
          ?.physicalBinSequenceSha256 ?? null;
        if (!/^[0-9a-f]{64}$/.test(validationSequenceSha256 ?? '')
          || !/^[0-9a-f]{64}$/.test(paritySequenceSha256 ?? '')) {
          preflightRejectionReasons.push(
            'frozen paired lanes lack a physical-bin sequence commitment',
          );
        } else if (validationSequenceSha256 !== paritySequenceSha256) {
          preflightRejectionReasons.push(
            'frozen physical-bin sequence changed between parity and preflight snapshots',
          );
        } else {
          const pairKey = [
            spec.repetitionIndex,
            spec.layout,
            spec.visibilityFraction,
            scenarioManifest?.sha256,
          ].join('|');
          if (physicalBinSequenceByPair.has(pairKey)) {
            preflightRejectionReasons.push(
              'frozen physical-bin sequence pair was observed more than once',
            );
          } else {
            physicalBinSequenceByPair.set(pairKey, {
              sha256: validationSequenceSha256,
              modeIds: new Set([spec.modeId]),
            });
          }
        }
      }
    }
    preflight.accepted = preflightRejectionReasons.length === 0;
    preflight.rejectionReasons = [...new Set(preflightRejectionReasons)];
    const validationArtifact = {
      schemaVersion: 2,
      trialId: spec.trialId,
      planIndex: spec.planIndex,
      repetitionIndex: spec.repetitionIndex,
      modeId: spec.modeId,
      visibilityFraction: spec.visibilityFraction,
      layout: spec.layout,
      objectCount: spec.objectCount,
      bucketCount: spec.bucketCount,
      ...(FROZEN_CROSSOVER_MATRIX ? {
        laneStorageOrder: [...spec.laneStorageOrder],
        superblockOrientationOffset: spec.superblockOrientationOffset,
        plannedScheduleSha256:
          FROZEN_SCHEDULE_SHA256_BY_ORIENTATION[String(spec.superblockOrientationOffset)],
      } : {}),
      selectedConfig,
      status: preflight.accepted ? 'preflight-accepted' : 'rejected',
      rejectionReasons: [...preflight.rejectionReasons],
      pre: preflight,
      timingStart: null,
      post: null,
    };
    refreshArtifactDigest(validationArtifact);
    validationArtifacts.push(validationArtifact);
    activeValidationArtifact = validationArtifact;
    await persistValidationArtifacts();
    if (!preflight.accepted) {
      throw new Error(`Trial ${spec.trialId} failed preflight validation: ${preflight.rejectionReasons.join('; ')}`);
    }

    const started = await guarded(startConfiguredTrial(
      page,
      auditContext,
      FROZEN_CROSSOVER_MATRIX,
    ));
    const timingStartInspection = inspectEvidenceCapture(spec, started?.evidence);
    await registerWorkloadManifests(spec, 'timing-start', timingStartInspection);
    const timingStart = timingStartInspection.capture;
    const startRejectionReasons = [...timingStart.rejectionReasons];
    if (started?.startError) {
      startRejectionReasons.push(`timing start validation failed: ${artifactSafeText(started.startError)}`);
    }
    if (started?.phase === 'error') {
      startRejectionReasons.push(
        `trial failed to start: ${started.trialError ?? 'unknown timestamp-resolution failure'}`,
      );
    } else if (started?.phase !== 'warmup') {
      startRejectionReasons.push(`trial did not enter warmup; current phase is ${started?.phase}`);
    }
    if (timingStart.workload.geometryFixtureSha256 !== preflight.workload.geometryFixtureSha256) {
      startRejectionReasons.push('fresh geometry fixture manifest changed before timing');
    }
    if (timingStart.workload.scenarioSha256 !== preflight.workload.scenarioSha256) {
      startRejectionReasons.push('fresh scenario manifest changed before timing');
    }
    if (timingStart.validation.semanticSha256 !== preflight.validation.semanticSha256) {
      startRejectionReasons.push('validation semantics changed before timing');
    }
    if (spec.modeId !== 'three-blocks-historical'
      && timingStart.validation.payloadSha256 !== preflight.validation.payloadSha256) {
      startRejectionReasons.push('exact validation payload changed before timing');
    }
    if (FROZEN_CROSSOVER_MATRIX) {
      const parityRejectionReasons = validateRenderParity(timingStart.renderParity, {
        spec,
        geometryManifest: fixtureManifest,
        scenarioManifest,
      });
      startRejectionReasons.push(...parityRejectionReasons.map(
        (reason) => `timing-start ${reason}`,
      ));
      if (parityRejectionReasons.length === 0
        && renderParityIdentity(timingStart.renderParity)
          !== renderParityIdentity(preflight.renderParity)) {
        startRejectionReasons.push(
          'paired exact color/depth/object-ID/material parity changed before timing',
        );
      }
      if (timingStart.renderParity?.snapshotValidation === undefined
        || sha256Json(timingStart.renderParity.snapshotValidation)
          !== timingStart.validation.payloadSha256) {
        startRejectionReasons.push(
          'frozen timing-start parity snapshot differs from its exact validation payload',
        );
      }
    }
    timingStart.accepted = startRejectionReasons.length === 0;
    timingStart.rejectionReasons = [...new Set(startRejectionReasons)];
    validationArtifact.timingStart = timingStart;
    validationArtifact.status = timingStart.accepted ? 'timing' : 'rejected';
    validationArtifact.rejectionReasons = [...timingStart.rejectionReasons];
    refreshArtifactDigest(validationArtifact);
    await persistValidationArtifacts();
    if (!timingStart.accepted) {
      throw new Error(`Trial ${spec.trialId} refused timing: ${timingStart.rejectionReasons.join('; ')}`);
    }
    telemetryContext = Object.freeze({ ...telemetryContext, phase: 'warmup' });
    await guarded(page.waitForFunction(
      () => ['measure', 'resolving-measurement', 'complete', 'error']
        .includes(window.__WEBGPU_BENCH__?.phase),
      null,
      { timeout: 180_000 },
    ));
    const observedPhase = await guarded(page.evaluate(() => window.__WEBGPU_BENCH__?.phase ?? null));
    if (observedPhase === 'error') {
      const trialError = await guarded(page.evaluate(
        () => window.__WEBGPU_BENCH__?.trialError ?? 'unknown timestamp-resolution failure',
      ));
      throw new Error(`Trial ${spec.trialId} failed: ${trialError}`);
    }
    telemetryContext = Object.freeze({
      ...telemetryContext,
      phase: observedPhase === 'measure' ? 'measurement' : observedPhase ?? 'unclassified',
    });
    if (observedPhase !== 'complete') {
      await guarded(page.waitForFunction(
        () => ['complete', 'error'].includes(window.__WEBGPU_BENCH__?.phase),
        null,
        { timeout: 180_000 },
      ));
      const terminal = await guarded(page.evaluate(() => ({
        phase: window.__WEBGPU_BENCH__?.phase ?? null,
        error: window.__WEBGPU_BENCH__?.trialError ?? null,
      })));
      if (terminal.phase === 'error') {
        throw new Error(`Trial ${spec.trialId} failed: ${terminal.error ?? 'unknown timestamp-resolution failure'}`);
      }
    }
    telemetryContext = Object.freeze({ ...telemetryContext, phase: 'trial-complete' });
    const result = await guarded(page.evaluate(() => {
      const summaryText = document.getElementById('details')?.textContent ?? '';
      return {
        rows: window.__WEBGPU_BENCH__?.rows ?? [],
        summary: summaryText ? JSON.parse(summaryText) : null,
      };
    }));
    if (pageErrors.length) throw new Error(`${pageErrors[0].source}: ${pageErrors[0].detail}`);

    telemetryContext = Object.freeze({ ...telemetryContext, phase: 'post-validation' });
    const postEvidence = await guarded(collectPostTrialEvidence(
      page,
      FROZEN_CROSSOVER_MATRIX,
    ));
    const postInspection = inspectEvidenceCapture(spec, postEvidence);
    await registerWorkloadManifests(spec, 'post-trial', postInspection);
    const post = postInspection.capture;
    const rejectionReasons = [
      ...post.rejectionReasons,
      ...validateTrialRows(
        spec,
        result.rows,
        result.summary,
        timingStart.validation.payload,
        scenarioManifest,
        {
          schemaVersion: ARTIFACT_SCHEMA_VERSION,
          warmupFrames: WARMUP_FRAMES,
          measuredFrames: MEASURED_FRAMES,
          plannedScheduleSha256: FROZEN_CROSSOVER_MATRIX
            ? FROZEN_SCHEDULE_SHA256_BY_ORIENTATION[
              String(spec.superblockOrientationOffset)
            ]
            : null,
        },
      ),
    ];
    if (post.workload.geometryFixtureSha256 !== timingStart.workload.geometryFixtureSha256) {
      rejectionReasons.push('fresh geometry fixture manifest changed during the trial');
    }
    if (post.workload.scenarioSha256 !== timingStart.workload.scenarioSha256) {
      rejectionReasons.push('fresh scenario manifest changed during the trial');
    }
    if (post.workload?.scenarioSeed !== timingStart.workload.scenarioSeed) {
      rejectionReasons.push('scenario seed changed during the trial');
    }
    if (post.validation.semanticSha256 !== timingStart.validation.semanticSha256) {
      rejectionReasons.push('untimed post-trial validation semantics differ from pre-trial validation');
    }
    if (spec.modeId !== 'three-blocks-historical'
      && post.validation.payloadSha256 !== timingStart.validation.payloadSha256) {
      rejectionReasons.push('untimed post-trial exact-validation payload changed');
    }
    if (FROZEN_CROSSOVER_MATRIX) {
      const parityRejectionReasons = validateRenderParity(post.renderParity, {
        spec,
        geometryManifest: fixtureManifest,
        scenarioManifest,
      });
      rejectionReasons.push(...parityRejectionReasons.map(
        (reason) => `postflight ${reason}`,
      ));
      if (parityRejectionReasons.length === 0
        && renderParityIdentity(post.renderParity)
          !== renderParityIdentity(timingStart.renderParity)) {
        rejectionReasons.push(
          'paired exact color/depth/object-ID/material parity changed during timing',
        );
      }
      if (post.renderParity?.snapshotValidation === undefined
        || sha256Json(post.renderParity.snapshotValidation)
          !== post.validation.payloadSha256) {
        rejectionReasons.push(
          'frozen postflight parity snapshot differs from its exact validation payload',
        );
      }
    }
    const uniqueRejectionReasons = [...new Set(rejectionReasons)];
    post.accepted = uniqueRejectionReasons.length === 0;
    post.rejectionReasons = uniqueRejectionReasons;
    validationArtifact.post = post;
    validationArtifact.status = post.accepted ? 'accepted' : 'rejected';
    validationArtifact.rejectionReasons = uniqueRejectionReasons;
    refreshArtifactDigest(validationArtifact);
    await persistValidationArtifacts();

    const firstRow = result.rows[0] ?? {};
    const trialSummary = {
      trialId: spec.trialId,
      planIndex: spec.planIndex,
      repetitionIndex: spec.repetitionIndex,
      modeId: spec.modeId,
      modeOrder: spec.modeOrder,
      modeOrderPosition: spec.modeOrderPosition,
      visibilityFraction: spec.visibilityFraction,
      visibilityOrder: spec.visibilityOrder,
      visibilityOrderPosition: spec.visibilityOrderPosition,
      layout: spec.layout,
      layoutOrder: spec.layoutOrder,
      layoutOrderPosition: spec.layoutOrderPosition,
      objectCount: spec.objectCount,
      bucketCount: spec.bucketCount,
      ...(FROZEN_CROSSOVER_MATRIX ? {
        laneStorageOrder: [...spec.laneStorageOrder],
        superblockOrientationOffset: spec.superblockOrientationOffset,
        plannedScheduleSha256:
          FROZEN_SCHEDULE_SHA256_BY_ORIENTATION[String(spec.superblockOrientationOffset)],
      } : {}),
      selectedConfig,
      startedAt: trialStartedAt,
      completedAt: new Date().toISOString(),
      elapsedMs: performance.now() - trialStartedMonotonic,
      validation: {
        pass: firstRow.validationPass === true,
        kind: firstRow.validationKind ?? null,
        artifactSha256: validationArtifact.sha256,
      },
      timestamps: {
        accepted: result.summary?.accepted === true,
        available: result.summary?.timestampAvailable === true,
        rowCount: result.summary?.rowCount ?? null,
        missingRenderFrames: result.summary?.missingRenderFrames ?? null,
        missingComputeFrames: result.summary?.missingComputeFrames ?? null,
        ...(FROZEN_CROSSOVER_MATRIX ? {
          expectedRenderTimestampUidCount:
            result.summary?.expectedRenderTimestampUidCount ?? null,
          invalidRenderTimestampUidCountFrames:
            result.summary?.invalidRenderTimestampUidCountFrames ?? null,
        } : {}),
        quantumNs: result.summary?.quantumNs ?? null,
        classification: result.summary?.classification ?? null,
      },
      completionInvariant: structuredClone(result.summary?.completionInvariant ?? null),
      timing: {
        cpuCommonUpdateP50Ms: percentile(result.rows.map((row) => row.cpuCommonUpdateMs), 0.5),
        cpuCommonUpdateP95Ms: percentile(result.rows.map((row) => row.cpuCommonUpdateMs), 0.95),
        cpuFrameBodyP50Ms: percentile(result.rows.map((row) => row.cpuFrameBodyMs), 0.5),
        cpuFrameBodyP95Ms: percentile(result.rows.map((row) => row.cpuFrameBodyMs), 0.95),
        cpuSubmitP50Ms: percentile(result.rows.map((row) => row.cpuSubmitTotalMs), 0.5),
        cpuSubmitP95Ms: percentile(result.rows.map((row) => row.cpuSubmitTotalMs), 0.95),
        gpuComputeP50Ms: percentile(result.rows.map((row) => row.gpuComputeMs), 0.5),
        gpuComputeP95Ms: percentile(result.rows.map((row) => row.gpuComputeMs), 0.95),
        gpuRenderP50Ms: percentile(result.rows.map((row) => row.gpuRenderMs), 0.5),
        gpuRenderP95Ms: percentile(result.rows.map((row) => row.gpuRenderMs), 0.95),
        gpuPassTotalP50Ms: percentile(result.rows.map((row) => row.gpuPassTotalMs), 0.5),
        gpuPassTotalP95Ms: percentile(result.rows.map((row) => row.gpuPassTotalMs), 0.95),
      },
      accepted: uniqueRejectionReasons.length === 0,
      rejectionReasons: uniqueRejectionReasons,
    };
    trialSummaries.push(trialSummary);
    frameRows.push(...result.rows);
    activeValidationArtifact = null;
    if (!trialSummary.accepted) {
      throw new Error(`Trial ${spec.trialId} rejected: ${uniqueRejectionReasons.join('; ')}`);
    }
    process.stdout.write(
      `[${spec.planIndex + 1}/${plan.length}] ${spec.modeId} layout=${spec.layout} `
      + `visibility=${spec.visibilityFraction} `
      + `GPU p50=${trialSummary.timing.gpuPassTotalP50Ms?.toFixed(4)} ms\n`,
    );
  }
  if (BENCHMARK_MATRIX === 'depth-ordering') {
    const expectedPairCount = MODE_ORDERS.length * DEPTH_ORDERING_LAYOUTS.length;
    if (physicalBinSequenceByPair.size !== expectedPairCount
      || [...physicalBinSequenceByPair.values()].some(
        (pair) => pair.modeIds.size !== DEPTH_BINNED_MODE_IDS.size
          || [...DEPTH_BINNED_MODE_IDS].some((modeId) => !pair.modeIds.has(modeId)),
      )) {
      throw new Error('Physical-bin sequence commitments do not cover every ordered-mode pair.');
    }
  }
  if (FROZEN_CROSSOVER_MATRIX && physicalBinSequenceByPair.size !== plan.length) {
    throw new Error(
      'Frozen physical-bin sequence commitments do not cover every paired trial.',
    );
  }
  if (pageErrors.length) throw new Error(`${pageErrors[0].source}: ${pageErrors[0].detail}`);
} catch (error) {
  runError = error;
  if (activeValidationArtifact
    && !['accepted', 'rejected'].includes(activeValidationArtifact.status)) {
    activeValidationArtifact.status = 'failed';
    activeValidationArtifact.failure = errorRecord(error);
    activeValidationArtifact.rejectionReasons = [
      ...new Set([
        ...activeValidationArtifact.rejectionReasons,
        artifactSafeText(error instanceof Error ? error.message : error),
      ]),
    ];
    refreshArtifactDigest(activeValidationArtifact);
    try {
      await persistValidationArtifacts();
    } catch (persistenceError) {
      runError = new AggregateError(
        [error, persistenceError],
        'Benchmark failed and its final validation snapshot could not be persisted.',
      );
    }
  }
  activeValidationArtifact = null;
} finally {
  telemetryContext = Object.freeze({ ...telemetryContext, phase: 'teardown' });
  try {
    await browser?.close();
  } catch (error) {
    runError ??= error;
  }
  try {
    await server?.close();
  } catch (error) {
    runError ??= error;
  }
  try {
    postComputeProcesses = await telemetry.captureComputeSnapshot('post-run');
  } catch {
    // A missing process snapshot does not invalidate otherwise accepted measurements.
  }
  try {
    await telemetry.stop();
  } catch {
    // The benchmark result remains usable when optional telemetry cannot be finalized.
  }
  telemetryReport = telemetry.report({ preComputeProcesses, postComputeProcesses });
}

if (!runError && pageErrors.length) {
  runError = new Error(`${pageErrors[0].source}: ${pageErrors[0].detail}`);
}

const completedAt = new Date().toISOString();
const sourceProvenanceEnd = await collectSourceProvenance(PROJECT_ROOT, {
  allowUnavailable: true,
});
const sourceStable = sourceProvenanceStart.status === 'available'
  && sourceProvenanceEnd.status === 'available'
  ? sourceProvenanceMatches(sourceProvenanceStart, sourceProvenanceEnd)
  : null;
if (sourceStable === false) {
  runError ??= new Error('Tracked source or dependency lock changed while the benchmark was running.');
}
if (sourceProvenanceStart.status === 'available' && sourceStable !== true) {
  runError ??= new Error('Source provenance could not be proven stable through teardown.');
}
if (EVIDENCE_STATUS !== 'development' && sourceStable !== true) {
  runError ??= new Error('Non-development evidence requires stable source provenance through teardown.');
}
const metadata = {
  schemaVersion: ARTIFACT_SCHEMA_VERSION,
  runId,
  status: runError ? 'failed' : 'complete',
  startedAt,
  completedAt,
  elapsedMs: performance.now() - runStartedMonotonic,
  environment: {
    node: process.version,
    platform: process.platform,
    architecture: process.arch,
    browser: browserMetadata,
    backend,
    benchmarkPage: pageEnvironment,
    note: ENVIRONMENT_NOTE,
    gpuTelemetry: telemetryReport,
  },
  evidenceStatus: EVIDENCE_STATUS,
  sourceProvenance: {
    start: sourceProvenanceStart,
    end: sourceProvenanceEnd,
    stable: sourceStable,
  },
  workload: {
    scenarioGenerator: 'createFixedSubsetScenario',
    scenarioSeed,
    manifestArtifact: 'workload-manifests.json',
    geometryFixtureSha256: geometryFixtureManifest?.sha256 ?? null,
    scenarioSha256ByVisibility: DEPTH_LAYOUT_MATRIX
      ? null
      : Object.fromEntries([...scenarioSha256ByCell].map(([cell, digest]) => (
        [cell.slice(cell.indexOf('|') + 1), digest]
      ))),
    scenarioSha256ByCell: Object.fromEntries(scenarioSha256ByCell),
    renderParitySha256ByCell: DEPTH_LAYOUT_MATRIX
      ? Object.fromEntries(renderParitySha256ByCell)
      : null,
    physicalBinSequenceSha256ByPair: DEPTH_LAYOUT_MATRIX
      ? Object.fromEntries([...physicalBinSequenceByPair].map(([key, pair]) => (
        [key, pair.sha256]
      )))
      : null,
  },
  protocol: {
    matrix: MATRIX_ID,
    matrixKind: BENCHMARK_MATRIX,
    representationScaleRole: BENCHMARK_MATRIX === 'fixed-slice-representation'
      ? BUCKET_COUNT === 1
        ? 'negative-control-equal-mesh-render-object-count'
        : 'primary-one-versus-b-mesh-render-object-representation-ablation'
      : null,
    layouts: DEPTH_LAYOUT_MATRIX
      ? [...DEPTH_ORDERING_LAYOUTS]
      : ['baseline'],
    depthBinCount: DEPTH_LAYOUT_MATRIX ? 8 : null,
    reversedDepthBuffer: DEPTH_LAYOUT_MATRIX ? true : null,
    minimumStorageBuffersPerShaderStage: DEPTH_LAYOUT_MATRIX ? 8 : null,
    renderParity: FROZEN_CROSSOVER_MATRIX
      ? 'preflight, timing-start, and postflight paired-lane exact validation plus two stable offscreen captures per lane of rgba8 color, depth32float, and encoded object ID'
      : BENCHMARK_MATRIX === 'depth-ordering'
        ? 'same-snapshot exact validation plus two stable offscreen captures of rgba8 color, depth32float, and encoded object ID'
        : null,
    ...(FROZEN_CROSSOVER_MATRIX ? {
      frozenCrossover: {
        lanes: [...FROZEN_DEPTH_CROSSOVER_LANES],
        blockSize: FROZEN_CROSSOVER_BLOCK_SIZE,
        warmupBlocks: FROZEN_CROSSOVER_WARMUP_BLOCKS,
        measuredBlocks: FROZEN_CROSSOVER_MEASURED_BLOCKS,
        patterns: FROZEN_CROSSOVER_PATTERNS.map((pattern) => pattern.map(
          (laneId) => (laneId === FROZEN_DEPTH_CROSSOVER_LANES[0] ? 'F' : 'R'),
        ).join('')),
        expectedMeasuredRowsPerLane: FROZEN_CROSSOVER_MEASURED_FRAMES / 2,
        expectedRenderCallsPerFrame: 1,
        expectedRenderTimestampUidCount: 1,
        expectedComputeTimestampsPerFrame: 0,
        survivorBufferSegments: 2,
        survivorSegmentLength: OBJECT_COUNT,
        legalLaneBases: [0, OBJECT_COUNT],
        scheduleSha256ByOrientation: { ...FROZEN_SCHEDULE_SHA256_BY_ORIENTATION },
      },
    } : {}),
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    allowedObjectCounts: [...ALLOWED_OBJECT_COUNTS],
    allowedBucketCounts: [...ALLOWED_BUCKET_COUNTS],
    allowedHeterogeneousComparators: [...ALLOWED_HETEROGENEOUS_COMPARATORS],
    heterogeneousComparator: BENCHMARK_MATRIX === 'ecosystem' && BUCKET_COUNT !== 1
      ? HETEROGENEOUS_COMPARATOR
      : null,
    modes: [...MODES],
    visibilityLevels: [...VISIBILITY_LEVELS],
    repetitions: MODE_ORDERS.length,
    warmupFrames: WARMUP_FRAMES,
    measuredFrames: MEASURED_FRAMES,
    maximumCpuTimerQuantumMs: MAXIMUM_CPU_TIMER_QUANTUM_MS,
    ordering: BENCHMARK_MATRIX === 'fixed-slice-representation'
      ? 'six-repetition-balanced-ab-ba-with-rotated-visibility-order'
      : FROZEN_CROSSOVER_MATRIX
        ? 'twelve-repetition-paired-eight-frame-frozen-crossover-with-balanced-layout-storage-base-and-starting-orientation'
        : BENCHMARK_MATRIX === 'depth-ordering'
          ? 'all-six-mode-permutations-with-alternating-high-low-layout-order'
          : 'all-six-mode-permutations-with-rotated-visibility-order',
    threeBlocksScheduling: BENCHMARK_MATRIX !== 'ecosystem'
      ? null
      : BUCKET_COUNT === 1
        ? 'public explicit update'
        : HETEROGENEOUS_COMPARATOR === 'historical-v10'
          ? 'published v0.10 indirect-batching execution path with pinned diagnostic readback'
          : 'version-pinned coalesced compute-node probe',
  },
  plan,
  expectedTrialCount: plan.length,
  completedTrialCount: trialSummaries.length,
  acceptedTrialCount: trialSummaries.filter((trial) => trial.accepted).length,
  frameRowCount: frameRows.length,
  validationArtifactCount: validationArtifacts.length,
  validationArtifactSha256: validationArtifacts.map((artifact) => artifact.sha256),
  pageErrors,
  error: runError ? errorRecord(runError) : null,
};

await atomicWriteJson(
  path.join(runDirectory, 'validation-artifacts.json'),
  validationArtifacts,
);
await persistWorkloadManifests();
await Promise.all([
  writeFile(path.join(runDirectory, 'frames.csv'), rowsToCsv(frameRows), 'utf8'),
  writeFile(path.join(runDirectory, 'metadata.json'), `${JSON.stringify(metadata, null, 2)}\n`, 'utf8'),
  writeFile(path.join(runDirectory, 'trial-summaries.json'), `${JSON.stringify(trialSummaries, null, 2)}\n`, 'utf8'),
  writeFile(
    path.join(runDirectory, 'gpu-telemetry-summary.json'),
    `${JSON.stringify(telemetryReport, null, 2)}\n`,
    'utf8',
  ),
]);

const requiredArtifactNames = Object.freeze([
  'frames.csv',
  'metadata.json',
  'trial-summaries.json',
  'validation-artifacts.json',
  'workload-manifests.json',
  'gpu-telemetry-summary.json',
]);
const optionalArtifactNames = Object.freeze(['gpu-telemetry.csv']);
const artifactRoles = Object.freeze({
  'frames.csv': 'frame-level timing rows',
  'metadata.json': 'run protocol, environment, provenance, and completion state',
  'trial-summaries.json': 'per-trial acceptance and timing summaries',
  'validation-artifacts.json': 'crash-safe pre/post correctness and workload evidence',
  'workload-manifests.json': 'deduplicated geometry and scenario fingerprint manifests',
  'gpu-telemetry-summary.json': 'telemetry availability and process-snapshot summary',
  'gpu-telemetry.csv': 'optional device telemetry samples',
});
const artifactFiles = [];
for (const name of [...requiredArtifactNames, ...optionalArtifactNames]) {
  const required = requiredArtifactNames.includes(name);
  try {
    const contents = await readFile(path.join(runDirectory, name));
    artifactFiles.push({
      name,
      role: artifactRoles[name],
      required,
      present: true,
      bytes: contents.length,
      sha256: sha256Bytes(contents),
      absenceReason: null,
    });
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
    artifactFiles.push({
      name,
      role: artifactRoles[name],
      required,
      present: false,
      bytes: null,
      sha256: null,
      absenceReason: required
        ? 'required artifact was not written'
        : telemetryReport?.reason ?? `telemetry status: ${telemetryReport?.status ?? 'unavailable'}`,
    });
  }
}
const missingRequiredArtifacts = artifactFiles
  .filter((record) => record.required && !record.present)
  .map((record) => record.name);
if (missingRequiredArtifacts.length > 0) {
  runError ??= new Error(`Required artifacts are missing: ${missingRequiredArtifacts.join(', ')}.`);
  metadata.status = 'failed';
  metadata.error = errorRecord(runError);
  await writeFile(
    path.join(runDirectory, 'metadata.json'),
    `${JSON.stringify(metadata, null, 2)}\n`,
    'utf8',
  );
  const metadataContents = await readFile(path.join(runDirectory, 'metadata.json'));
  const metadataRecord = artifactFiles.find((record) => record.name === 'metadata.json');
  Object.assign(metadataRecord, {
    present: true,
    bytes: metadataContents.length,
    sha256: sha256Bytes(metadataContents),
    absenceReason: null,
  });
}
const optionalFiles = optionalArtifactNames.map((name) => {
  const record = artifactFiles.find((candidate) => candidate.name === name);
  const evidenceAvailable = name === 'gpu-telemetry.csv'
    ? telemetryReport?.status === 'available' && record?.present === true
    : record?.present === true;
  return {
    name,
    present: record?.present === true,
    evidenceAvailable,
    absenceReason: evidenceAvailable
      ? null
      : telemetryReport?.reason ?? `telemetry status: ${telemetryReport?.status ?? 'unavailable'}`,
  };
});
await atomicWriteJson(
  path.join(runDirectory, 'artifact-manifest.json'),
  {
    schemaVersion: 2,
    runId,
    hashAlgorithm: 'sha256',
    requiredFiles: [...requiredArtifactNames],
    optionalFiles,
    files: artifactFiles,
  },
);

if (runError) throw runError;

process.stdout.write(`${JSON.stringify({
  status: metadata.status,
  runDirectory,
  trials: metadata.completedTrialCount,
  frames: metadata.frameRowCount,
  elapsedMs: metadata.elapsedMs,
  evidenceStatus: metadata.evidenceStatus,
  gpuTelemetryStatus: telemetryReport?.status ?? 'unavailable',
  browser: browserMetadata,
  backend,
}, null, 2)}\n`);
