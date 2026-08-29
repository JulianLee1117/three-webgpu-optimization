import { access, mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import process from 'node:process';
import { performance } from 'node:perf_hooks';
import { chromium } from 'playwright-core';
import { createServer } from 'vite';
import { NvidiaTelemetryRecorder } from './nvidia-telemetry.mjs';

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const RESULT_ROOT = path.resolve(
  process.env.BENCHMARK_RESULT_ROOT ?? path.join(PROJECT_ROOT, 'results', 'runs'),
);
const EVIDENCE_STATUS = process.env.BENCHMARK_EVIDENCE_STATUS ?? 'development';
const ENVIRONMENT_NOTE = process.env.BENCHMARK_ENVIRONMENT_NOTE ?? null;
const MAXIMUM_CPU_TIMER_QUANTUM_MS = 0.01;
const ALLOWED_OBJECT_COUNTS = Object.freeze([4_096, 16_384, 65_536]);
const ALLOWED_BUCKET_COUNTS = Object.freeze([1, 4, 32, 128]);
const ALLOWED_HETEROGENEOUS_COMPARATORS = Object.freeze([
  'coalesced-v11',
  'historical-v10',
]);
const WARMUP_FRAMES = 300;
const MEASURED_FRAMES = 240;
const VISIBILITY_LEVELS = Object.freeze([0.2, 0.8, 0.99]);
const BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
]);

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

const OBJECT_COUNT = validatedEnvironmentChoice(
  'BENCHMARK_OBJECT_COUNT',
  16_384,
  ALLOWED_OBJECT_COUNTS,
);
const BUCKET_COUNT = validatedEnvironmentChoice(
  'BENCHMARK_BUCKET_COUNT',
  32,
  ALLOWED_BUCKET_COUNTS,
);
const HETEROGENEOUS_COMPARATOR = validatedStringEnvironmentChoice(
  'BENCHMARK_HETEROGENEOUS_COMPARATOR',
  'coalesced-v11',
  ALLOWED_HETEROGENEOUS_COMPARATORS,
);
if (BUCKET_COUNT === 1
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
const MODES = Object.freeze(['draw-all', THREE_BLOCKS_MODE, 'fixed-slice']);
const MODE_PERMUTATIONS = Object.freeze([
  Object.freeze([MODES[0], MODES[1], MODES[2]]),
  Object.freeze([MODES[1], MODES[2], MODES[0]]),
  Object.freeze([MODES[2], MODES[0], MODES[1]]),
  Object.freeze([MODES[2], MODES[1], MODES[0]]),
  Object.freeze([MODES[1], MODES[0], MODES[2]]),
  Object.freeze([MODES[0], MODES[2], MODES[1]]),
]);
const MATRIX_ID = `focused-o${OBJECT_COUNT}-b${BUCKET_COUNT}`;

function compactTimestamp(isoTimestamp) {
  return isoTimestamp.replaceAll(':', '-');
}

function rotate(values, offset) {
  const normalized = offset % values.length;
  return [...values.slice(normalized), ...values.slice(0, normalized)];
}

function buildPlan(runId) {
  const plan = [];
  for (let repetitionIndex = 0; repetitionIndex < MODE_PERMUTATIONS.length; repetitionIndex += 1) {
    const modeOrder = [...MODE_PERMUTATIONS[repetitionIndex]];
    const visibilityOrder = rotate(VISIBILITY_LEVELS, repetitionIndex);
    for (let visibilityOrderPosition = 0; visibilityOrderPosition < visibilityOrder.length; visibilityOrderPosition += 1) {
      const visibilityFraction = visibilityOrder[visibilityOrderPosition];
      for (let modeOrderPosition = 0; modeOrderPosition < modeOrder.length; modeOrderPosition += 1) {
        const modeId = modeOrder[modeOrderPosition];
        const planIndex = plan.length;
        plan.push({
          trialId: `${runId}-t${String(planIndex + 1).padStart(2, '0')}`,
          planIndex,
          repetitionIndex,
          modeId,
          modeOrder,
          modeOrderPosition,
          visibilityFraction,
          visibilityOrder,
          visibilityOrderPosition,
          objectCount: OBJECT_COUNT,
          bucketCount: BUCKET_COUNT,
        });
      }
    }
  }
  return plan;
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
    return { name: error.name, message: error.message, stack: error.stack ?? null };
  }
  return { name: 'Error', message: String(error), stack: null };
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
  if (rejectionReasons.length > 0) {
    throw new Error(`Benchmark environment rejected at startup: ${rejectionReasons.join('; ')}.`);
  }
}

function validateTrialResult(spec, rows, pageSummary) {
  const rejectionReasons = [];
  const expectsCompute = spec.modeId !== 'draw-all';
  const expectedComputeDispatches = spec.modeId === 'draw-all'
    ? 0
    : spec.modeId === 'fixed-slice'
      ? 2
      : spec.modeId === 'three-blocks-historical'
        ? 9
        : spec.bucketCount * 4;
  const expectedComputeSubmissions = spec.modeId === 'draw-all'
    ? 0
    : spec.modeId === 'fixed-slice'
      || spec.modeId === 'three-blocks-coalesced'
      || spec.modeId === 'three-blocks-historical'
      ? 1
      : spec.bucketCount;
  if (rows.length !== MEASURED_FRAMES) {
    rejectionReasons.push(`expected ${MEASURED_FRAMES} rows, received ${rows.length}`);
  }
  if (pageSummary?.accepted !== true) rejectionReasons.push('page timing summary was rejected');
  if (pageSummary?.timestampAvailable !== true) rejectionReasons.push('GPU timestamps were unavailable');
  if (pageSummary?.rowCount !== MEASURED_FRAMES) {
    rejectionReasons.push(`page summary row count was ${pageSummary?.rowCount ?? 'missing'}`);
  }
  if (pageSummary?.missingRenderFrames !== 0) {
    rejectionReasons.push(`${pageSummary?.missingRenderFrames ?? 'unknown'} render timestamp frames missing`);
  }
  if (pageSummary?.missingComputeFrames !== 0) {
    rejectionReasons.push(`${pageSummary?.missingComputeFrames ?? 'unknown'} compute timestamp frames missing`);
  }

  const frameIds = new Set();
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const prefix = `frame ${index}`;
    if (row.runId !== spec.runId || row.trialId !== spec.trialId || row.planIndex !== spec.planIndex) {
      rejectionReasons.push(`${prefix} audit identifiers do not match the plan`);
    }
    if (row.frameIndex !== index) rejectionReasons.push(`${prefix} has frameIndex ${row.frameIndex}`);
    if (row.modeId !== spec.modeId) rejectionReasons.push(`${prefix} has mode ${row.modeId}`);
    if (row.objectCount !== OBJECT_COUNT || row.bucketCount !== BUCKET_COUNT) {
      rejectionReasons.push(`${prefix} has an unexpected object or bucket count`);
    }
    if (row.targetVisibilityFraction !== spec.visibilityFraction) {
      rejectionReasons.push(`${prefix} has visibility ${row.targetVisibilityFraction}`);
    }
    if (row.validationPass !== true || typeof row.validationKind !== 'string') {
      rejectionReasons.push(`${prefix} lacks accepted validation context`);
    }
    if (row.configuredComputeDispatches !== expectedComputeDispatches
      || row.configuredComputeSubmissions !== expectedComputeSubmissions) {
      rejectionReasons.push(`${prefix} has an unexpected compute schedule`);
    }
    if (row.timestampAvailable !== true) rejectionReasons.push(`${prefix} lacks timestamp support`);
    if (!Number.isInteger(row.gpuFrameId)) {
      rejectionReasons.push(`${prefix} has no integer GPU frame ID`);
    } else {
      frameIds.add(row.gpuFrameId);
    }
    if (!Number.isFinite(row.cpuCommonUpdateMs)
      || !Number.isFinite(row.cpuRenderSubmitMs)
      || !Number.isFinite(row.cpuSubmitTotalMs)
      || !Number.isFinite(row.cpuFrameBodyMs)) {
      rejectionReasons.push(`${prefix} has a missing CPU timing`);
    }
    if (expectsCompute) {
      if (row.usesCompute !== true || !Number.isFinite(row.cpuComputeSubmitMs)) {
        rejectionReasons.push(`${prefix} has a missing compute submission timing`);
      }
      if (!Number.isFinite(row.gpuComputeMs)) rejectionReasons.push(`${prefix} has a missing compute timestamp`);
    } else if (row.usesCompute !== false || row.gpuComputeMs !== null) {
      rejectionReasons.push(`${prefix} unexpectedly reports compute work`);
    }
    if (!Number.isFinite(row.gpuRenderMs) || !Number.isFinite(row.gpuPassTotalMs)) {
      rejectionReasons.push(`${prefix} has a missing GPU render or total timestamp`);
    } else {
      const expectedTotal = row.gpuRenderMs + (row.gpuComputeMs ?? 0);
      if (Math.abs(row.gpuPassTotalMs - expectedTotal) > 1e-9) {
        rejectionReasons.push(`${prefix} GPU pass total does not equal its component sum`);
      }
    }
  }
  if (frameIds.size !== rows.length) rejectionReasons.push('GPU frame IDs are missing or duplicated');

  return [...new Set(rejectionReasons)];
}

async function configureAndStartTrial(page, spec, auditContext) {
  return page.evaluate(async ({ trial, context }) => {
    const bench = window.__WEBGPU_BENCH__;
    if (!bench) throw new Error('Benchmark page API is unavailable.');
    const selections = {
      strategy: trial.modeId,
      objects: String(trial.objectCount),
      buckets: String(trial.bucketCount),
      visibility: String(trial.visibilityFraction),
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
      || selected.visibilityFraction !== trial.visibilityFraction) {
      throw new Error(`Page configuration mismatch: ${JSON.stringify(selected)}`);
    }
    await bench.startTrial(context);
    if (bench.phase === 'error') {
      throw new Error(`Trial failed to start: ${bench.trialError ?? 'unknown timestamp-resolution failure'}`);
    }
    if (bench.phase !== 'warmup') {
      throw new Error(`Trial did not enter warmup; current phase is ${bench.phase}.`);
    }
    return selected;
  }, { trial: spec, context: auditContext });
}

const startedAt = new Date().toISOString();
const runId = `${MATRIX_ID}-${compactTimestamp(startedAt)}`;
const runDirectory = path.join(RESULT_ROOT, runId);
const plan = buildPlan(runId).map((trial) => ({ ...trial, runId }));
const runStartedMonotonic = performance.now();
const frameRows = [];
const trialSummaries = [];
const pageErrors = [];
let telemetryContext = Object.freeze({
  phase: 'startup',
  trialId: null,
  planIndex: null,
  repetitionIndex: null,
  modeId: null,
  visibilityFraction: null,
});
let backend = null;
let pageEnvironment = null;
let browserMetadata = null;
let browser = null;
let page = null;
let server = null;
let runError = null;
let preComputeProcesses = null;
let postComputeProcesses = null;
let telemetryReport = null;

await mkdir(RESULT_ROOT, { recursive: true });
await mkdir(runDirectory);

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
  const record = { source, detail, timestamp: new Date().toISOString() };
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
    server: { host: '127.0.0.1', port: 0 },
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
      plannedModeOrder: spec.modeOrder.join('|'),
      plannedVisibilityOrder: spec.visibilityOrder.join('|'),
      protocolWarmupFrames: WARMUP_FRAMES,
      protocolMeasuredFrames: MEASURED_FRAMES,
    };

    telemetryContext = Object.freeze({
      phase: 'validation',
      trialId: spec.trialId,
      planIndex: spec.planIndex,
      repetitionIndex: spec.repetitionIndex,
      modeId: spec.modeId,
      visibilityFraction: spec.visibilityFraction,
    });
    const selectedConfig = await guarded(configureAndStartTrial(page, spec, auditContext));
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

    const rejectionReasons = validateTrialResult(spec, result.rows, result.summary);
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
      objectCount: spec.objectCount,
      bucketCount: spec.bucketCount,
      selectedConfig,
      startedAt: trialStartedAt,
      completedAt: new Date().toISOString(),
      elapsedMs: performance.now() - trialStartedMonotonic,
      validation: {
        pass: firstRow.validationPass === true,
        kind: firstRow.validationKind ?? null,
      },
      timestamps: {
        accepted: result.summary?.accepted === true,
        available: result.summary?.timestampAvailable === true,
        rowCount: result.summary?.rowCount ?? null,
        missingRenderFrames: result.summary?.missingRenderFrames ?? null,
        missingComputeFrames: result.summary?.missingComputeFrames ?? null,
        quantumNs: result.summary?.quantumNs ?? null,
        classification: result.summary?.classification ?? null,
      },
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
      accepted: rejectionReasons.length === 0,
      rejectionReasons,
    };
    trialSummaries.push(trialSummary);
    frameRows.push(...result.rows);
    if (!trialSummary.accepted) {
      throw new Error(`Trial ${spec.trialId} rejected: ${rejectionReasons.join('; ')}`);
    }
    process.stdout.write(
      `[${spec.planIndex + 1}/${plan.length}] ${spec.modeId} visibility=${spec.visibilityFraction} `
      + `GPU p50=${trialSummary.timing.gpuPassTotalP50Ms?.toFixed(4)} ms\n`,
    );
  }
  if (pageErrors.length) throw new Error(`${pageErrors[0].source}: ${pageErrors[0].detail}`);
} catch (error) {
  runError = error;
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
const metadata = {
  schemaVersion: 1,
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
  protocol: {
    matrix: MATRIX_ID,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    allowedObjectCounts: [...ALLOWED_OBJECT_COUNTS],
    allowedBucketCounts: [...ALLOWED_BUCKET_COUNTS],
    allowedHeterogeneousComparators: [...ALLOWED_HETEROGENEOUS_COMPARATORS],
    heterogeneousComparator: BUCKET_COUNT === 1 ? null : HETEROGENEOUS_COMPARATOR,
    modes: [...MODES],
    visibilityLevels: [...VISIBILITY_LEVELS],
    repetitions: MODE_PERMUTATIONS.length,
    warmupFrames: WARMUP_FRAMES,
    measuredFrames: MEASURED_FRAMES,
    maximumCpuTimerQuantumMs: MAXIMUM_CPU_TIMER_QUANTUM_MS,
    ordering: 'all-six-mode-permutations-with-rotated-visibility-order',
    threeBlocksScheduling: BUCKET_COUNT === 1
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
  pageErrors,
  error: runError ? errorRecord(runError) : null,
};

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
