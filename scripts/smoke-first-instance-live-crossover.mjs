import { access } from 'node:fs/promises';
import { randomBytes } from 'node:crypto';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { createServer } from 'vite';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
} from '../src/benchmark/first-instance-live-crossover-schedule.js';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
  buildFirstInstanceLiveCrossoverPlan,
} from '../src/benchmark/plan.js';
import {
  liveFirstInstanceCrossoverScheduleSha256,
  validateLiveFirstInstanceForcedFeatureOffGate,
  validateLiveFirstInstanceTrialEvidence,
} from './live-first-instance-evidence-validation.mjs';
import {
  createCandidateViteRuntimeGuard,
} from './execution-dependency-closure.mjs';

const OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;
const SCENARIO_SEED = 0xb1ad_2026;
const RUN_ID = 'first-instance-live-crossover-smoke';
const BROWSER_ARGS = [
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
];
const ISOLATION_HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};
const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const spec = Object.freeze({
  ...buildFirstInstanceLiveCrossoverPlan({
    runId: RUN_ID,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  })[0],
  runId: RUN_ID,
});
const shaderObservationChallenges = Object.freeze([
  ['preflight', 'render-parity'],
  ['preflight', 'main-validation'],
  ['timing-start', 'render-parity'],
  ['timing-start', 'main-validation'],
  ['postflight', 'render-parity'],
  ['postflight', 'main-validation'],
].map(([phase, role], index) => Object.freeze({
  schemaVersion: 1,
  kind: 'live-first-instance-shader-observation-challenge',
  origin: 'node-runner',
  runId: RUN_ID,
  trialId: spec.trialId,
  planIndex: spec.planIndex,
  repetitionIndex: spec.repetitionIndex,
  phase,
  role,
  captureOrdinal: index + 1,
  challengeNonce: randomBytes(32).toString('hex'),
})));

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

function compactShaderEvidence(evidence) {
  const compute = evidence?.compute;
  const differingShaderLines = (phase) => {
    const portable = compute?.phases?.[phase]?.lanes?.portable?.computeShader?.split('\n') ?? [];
    const feature = compute?.phases?.[phase]?.lanes?.feature?.computeShader?.split('\n') ?? [];
    const differences = [];
    for (let index = 0; index < Math.max(portable.length, feature.length); index += 1) {
      if (portable[index] !== feature[index]) {
        differences.push({ line: index + 1, portable: portable[index], feature: feature[index] });
      }
      if (differences.length === 24) break;
    }
    return differences;
  };
  return {
    pass: evidence?.pass ?? null,
    render: {
      pass: evidence?.render?.pass ?? null,
      reasons: evidence?.render?.reasons ?? null,
    },
    compute: compute && {
      pass: compute.pass,
      dispatchDimensionsEqual: compute.dispatchDimensionsEqual,
      fixedWorkloadExact: compute.fixedWorkloadExact,
      maxStorageBindingCount: compute.maxStorageBindingCount,
      phases: Object.fromEntries(['reset', 'cull'].map((phase) => {
        const item = compute.phases?.[phase];
        return [phase, item && {
          pass: item.pass,
          rawWgslEqual: item.rawWgslEqual,
          workgroupDeclaration: item.workgroupDeclaration,
          executionEqual: item.executionEqual,
          fixedWorkloadExact: item.fixedWorkloadExact,
          bindings: item.bindings,
          wordFour: item.wordFour,
          differingShaderLines: differingShaderLines(phase),
          executions: Object.fromEntries(Object.entries(item.lanes ?? {}).map(
            ([lane, capture]) => [lane, capture.execution],
          )),
        }];
      })),
    },
  };
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

async function openReadyPage(browser, url, errors) {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    deviceScaleFactor: 1,
  });
  const page = await context.newPage();
  attachErrorCapture(page, errors);
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__WEBGPU_BENCH__?.ready === true,
    null,
    { timeout: 120_000 },
  );
  return { context, page };
}

async function configureLiveTrial(page, trial) {
  return page.evaluate(async (configuration) => {
    const bench = window.__WEBGPU_BENCH__;
    bench.configureFirstInstanceLiveCrossover({
      lanePhysicalOrder: configuration.lanePhysicalOrder,
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
        throw new Error(`Unsupported smoke configuration ${id}=${value}.`);
      }
      select.value = value;
    }
    await bench.rebuild();
    return {
      selectedConfig: bench.selectedConfig(),
      environment: bench.environment,
      shaderEvidence: bench.firstInstanceShaderEvidence,
      timestampPoolPreprime: bench.timestampPoolPreprime,
      webgpuUncapturedErrors: bench.webgpuUncapturedErrors,
    };
  }, trial);
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
      webgpuUncapturedErrors: bench.webgpuUncapturedErrors,
    };
  }, observationChallenges);
}

const auditContext = Object.freeze({
  runId: RUN_ID,
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
  superblockOrientationOffset: spec.superblockOrientationOffset,
  plannedScheduleSha256: liveFirstInstanceCrossoverScheduleSha256(
    spec.superblockOrientationOffset,
  ),
});

let server = null;
let disposableBrowser = null;
let candidateBrowser = null;
let candidateViteRuntimeGuard = null;
try {
  const executablePath = await findBrowser();
  candidateViteRuntimeGuard = await createCandidateViteRuntimeGuard(projectRoot);
  server = await createServer({
    root: projectRoot,
    configFile: false,
    resolve: { dedupe: ['three'] },
    server: { host: '127.0.0.1', port: 0, headers: ISOLATION_HEADERS },
    logLevel: 'error',
    ...candidateViteRuntimeGuard.viteConfig,
  });
  await server.listen();
  const url = server.resolvedUrls?.local?.[0];
  requireCondition(Boolean(url), 'Vite did not expose a local URL.');

  const gateErrors = [];
  disposableBrowser = await chromium.launch({
    executablePath,
    headless: true,
    args: BROWSER_ARGS,
  });
  const gatePage = await openReadyPage(disposableBrowser, url, gateErrors);
  const gate = await gatePage.page.evaluate((options) => (
    window.__WEBGPU_BENCH__.runFirstInstanceLiveForcedFeatureOffGate(options)
  ), {
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    visibilityFraction: 0.99,
    scenarioSeed: SCENARIO_SEED,
  });
  const gateRejections = validateLiveFirstInstanceForcedFeatureOffGate(gate);
  requireCondition(gateErrors.length === 0, 'Disposable gate emitted page errors', gateErrors);
  requireCondition(gateRejections.length === 0, 'Disposable gate failed', gateRejections);
  await disposableBrowser.close();
  disposableBrowser = null;

  const pageErrors = [];
  candidateBrowser = await chromium.launch({
    executablePath,
    headless: true,
    args: BROWSER_ARGS,
  });
  const candidate = await openReadyPage(candidateBrowser, url, pageErrors);
  const configured = await configureLiveTrial(candidate.page, spec);
  requireCondition(configured.selectedConfig.strategyId === FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
    'Live smoke selected the wrong strategy', configured.selectedConfig);
  requireCondition(configured.environment.indirectFirstInstanceAvailable === true,
    'indirect-first-instance is unavailable', configured.environment);
  requireCondition(configured.environment.crossOriginIsolated === true,
    'Candidate custom entry did not retain cross-origin isolation', configured.environment);
  requireCondition(configured.environment.webgpuUncapturedErrorCount === 0,
    'WebGPU error occurred during construction', configured.environment);
  requireCondition(configured.shaderEvidence?.pass === true,
    'Compute/render shader evidence failed', compactShaderEvidence(configured.shaderEvidence));
  requireCondition(configured.timestampPoolPreprime?.kind
    === 'three-r185-timestamp-pool-preprime',
  'Timestamp pools were not pre-primed', configured.timestampPoolPreprime);

  const preflight = await captureEvidencePoint(
    candidate.page,
    shaderObservationChallenges.slice(0, 2),
  );
  const timingParity = await candidate.page.evaluate(
    (challenge) => window.__WEBGPU_BENCH__.captureRenderParity(challenge),
    shaderObservationChallenges[2],
  );
  const timingStart = await candidate.page.evaluate(
    ({ context, challenge }) => window.__WEBGPU_BENCH__.startTrial(context, challenge),
    { context: auditContext, challenge: shaderObservationChallenges[3] },
  );
  await candidate.page.waitForFunction(
    () => ['complete', 'error'].includes(window.__WEBGPU_BENCH__?.phase),
    null,
    { timeout: 180_000 },
  );
  const timing = await candidate.page.evaluate(() => ({
    phase: window.__WEBGPU_BENCH__.phase,
    error: window.__WEBGPU_BENCH__.trialError,
    rows: window.__WEBGPU_BENCH__.rows,
    summary: JSON.parse(document.getElementById('details')?.textContent ?? 'null'),
    webgpuUncapturedErrors: window.__WEBGPU_BENCH__.webgpuUncapturedErrors,
  }));
  requireCondition(timing.phase === 'complete', 'Live smoke timing failed', timing.error);
  const postflight = await captureEvidencePoint(
    candidate.page,
    shaderObservationChallenges.slice(4, 6),
  );
  const strict = await validateLiveFirstInstanceTrialEvidence({
    spec,
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
    protocol: {
      schemaVersion: 2,
      warmupFrames: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
      measuredFrames: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
      plannedScheduleSha256: auditContext.plannedScheduleSha256,
    },
    scenarioManifest: preflight.workload.scenario,
    geometryManifest: preflight.workload.geometryFixtures,
  });
  requireCondition(pageErrors.length === 0, 'Candidate page emitted errors', pageErrors);
  requireCondition(timing.webgpuUncapturedErrors.length === 0
    && postflight.webgpuUncapturedErrors.length === 0,
  'Candidate page observed uncaptured WebGPU errors');
  requireCondition(strict.pass === true, 'Strict live trial evidence failed', strict);

  await candidateBrowser.close();
  candidateBrowser = null;
  await server.close();
  server = null;
  const viteRuntimeAudit = await candidateViteRuntimeGuard.finalize();
  requireCondition(viteRuntimeAudit.entryHtml.successfulResponseCount === 2,
    'Candidate smoke did not serve exactly two entry documents', viteRuntimeAudit.entryHtml);

  process.stdout.write(`${JSON.stringify({
    pass: true,
    kind: 'first-instance-live-crossover-browser-smoke',
    browser: path.basename(executablePath),
    gatePass: gate.pass,
    validationPass: timingStart.validation.pass,
    renderParityPass: timingParity.pass,
    rows: timing.rows.length,
    quantumNs: timing.summary.quantumNs,
    gpuComputeP50Ms: timing.rows.map((row) => row.gpuComputeMs)
      .sort((a, b) => a - b)[Math.floor(timing.rows.length / 2)],
    gpuRenderP50Ms: timing.rows.map((row) => row.gpuRenderMs)
      .sort((a, b) => a - b)[Math.floor(timing.rows.length / 2)],
    viteRuntimeAudit: {
      policyId: viteRuntimeAudit.policyId,
      entryHtmlResponseCount: viteRuntimeAudit.entryHtml.successfulResponseCount,
      moduleCount: viteRuntimeAudit.moduleCount,
      dependencyModuleCount: viteRuntimeAudit.dependencyModuleCount,
      modulesSha256: viteRuntimeAudit.modulesSha256,
    },
  }, null, 2)}\n`);
} finally {
  await disposableBrowser?.close().catch(() => undefined);
  await candidateBrowser?.close().catch(() => undefined);
  await server?.close().catch(() => undefined);
  await candidateViteRuntimeGuard?.dispose().catch(() => undefined);
}
