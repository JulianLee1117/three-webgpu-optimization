import { createHash } from 'node:crypto';
import { access, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { chromium } from 'playwright-core';
import { createServer } from 'vite';

const PORTABLE = 'fixed-slice';
const FEATURE = 'fixed-slice-indirect-first-instance';
const DEFAULT_REPETITIONS = 6;
const DEFAULT_OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;
const VISIBILITY_LEVELS = Object.freeze([0.99, 0.2]);

function integerEnvironment(name, fallback, allowed) {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || !allowed.includes(value)) {
    throw new Error(`${name} must be one of ${allowed.join(', ')}.`);
  }
  return value;
}

const repetitions = integerEnvironment(
  'FIRST_INSTANCE_SCREEN_REPETITIONS',
  DEFAULT_REPETITIONS,
  [1, DEFAULT_REPETITIONS],
);
const objectCount = integerEnvironment(
  'FIRST_INSTANCE_SCREEN_OBJECT_COUNT',
  DEFAULT_OBJECT_COUNT,
  [4_096, DEFAULT_OBJECT_COUNT],
);
const protocolConformant = repetitions === DEFAULT_REPETITIONS
  && objectCount === DEFAULT_OBJECT_COUNT;

function percentile(values, fraction) {
  if (values.length === 0 || values.some((value) => !Number.isFinite(value))) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * fraction) - 1))];
}

function median(values) {
  if (values.length === 0 || values.some((value) => !Number.isFinite(value))) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const midpoint = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[midpoint - 1] + sorted[midpoint]) / 2
    : sorted[midpoint];
}

function sha256Json(value) {
  return createHash('sha256').update(`${JSON.stringify(value)}\n`).digest('hex');
}

function compactTimestamp(iso) {
  return iso.replaceAll(':', '-').replaceAll('.', '-');
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
      return candidate;
    } catch {
      // Continue through the pinned system-browser search order.
    }
  }
  throw new Error('No Chrome, Chromium, or Edge executable found. Set BROWSER_PATH.');
}

async function waitForTrial(page, label) {
  await page.waitForFunction(
    () => ['complete', 'error'].includes(window.__WEBGPU_BENCH__?.phase),
    null,
    { timeout: 180_000 },
  );
  const terminal = await page.evaluate(() => ({
    phase: window.__WEBGPU_BENCH__?.phase ?? null,
    error: window.__WEBGPU_BENCH__?.trialError ?? null,
  }));
  if (terminal.phase !== 'complete') {
    throw new Error(`${label} failed: ${terminal.error ?? terminal.phase}`);
  }
}

async function configure(page, { strategy, visibilityFraction }) {
  return page.evaluate(async (selected) => {
    document.querySelector('#objects').value = String(selected.objectCount);
    document.querySelector('#buckets').value = String(selected.bucketCount);
    document.querySelector('#visibility').value = String(selected.visibilityFraction);
    document.querySelector('#layout').value = 'baseline';
    document.querySelector('#strategy').value = selected.strategy;
    await window.__WEBGPU_BENCH__.rebuild();
    const validation = await window.__WEBGPU_BENCH__.validate();
    const parity = await window.__WEBGPU_BENCH__.captureRenderParity();
    return {
      selectedConfig: window.__WEBGPU_BENCH__.selectedConfig(),
      validation,
      parity,
      workload: await window.__WEBGPU_BENCH__.fingerprintWorkload(),
    };
  }, {
    strategy,
    visibilityFraction,
    objectCount,
    bucketCount: BUCKET_COUNT,
  });
}

function channelIdentity(parity) {
  return ['color', 'depth', 'objectId'].map((name) => {
    const channel = parity?.[name];
    return `${channel?.format}|${channel?.arrayType}|${channel?.byteLength}|${channel?.sha256}`;
  }).join('||');
}

function assertLaneEvidence(lane, { strategy, visibilityFraction }) {
  const validation = lane?.validation;
  if (lane?.selectedConfig?.strategyId !== strategy
    || lane.selectedConfig.objectCount !== objectCount
    || lane.selectedConfig.bucketCount !== BUCKET_COUNT
    || lane.selectedConfig.visibilityFraction !== visibilityFraction
    || lane.selectedConfig.layout !== 'baseline'
    || validation?.pass !== true
    || validation.kind !== `${strategy}-exact-membership`
    || validation.membership?.pass !== true
    || validation.membershipDigests?.pass !== true
    || validation.commandValidation?.pass !== true
    || validation.commandValidation?.records?.length !== BUCKET_COUNT
    || validation.overflow !== 0
    || lane.parity?.pass !== true
    || lane.parity.snapshotValidation?.pass !== true
    || lane.workload?.geometryFixtures?.bucketCount !== BUCKET_COUNT
    || lane.workload?.scenario?.visibilityFraction !== visibilityFraction) {
    throw new Error(`Exact ${strategy} evidence failed: ${JSON.stringify(lane)}`);
  }

  const feature = strategy === FEATURE;
  for (let bucket = 0; bucket < BUCKET_COUNT; bucket += 1) {
    const record = validation.commandValidation.records[bucket];
    const firstInstanceShapePass = feature
      ? bucket === 0
        ? record.expected.firstInstance === 0
        : record.expected.firstInstance > 0
      : record.expected.firstInstance === 0;
    if (!firstInstanceShapePass
      || record.actual.firstInstance !== record.expected.firstInstance) {
      throw new Error(`${strategy} bucket ${bucket} has the wrong firstInstance.`);
    }
  }

  if (feature && (
    validation.representation?.addressMode !== 'indirect-first-instance'
    || validation.representation?.hasBucketBaseAttribute !== false
    || validation.representation?.nonzeroFirstInstanceCount !== BUCKET_COUNT - 1
  )) {
    throw new Error(`Feature representation audit failed: ${JSON.stringify(validation.representation)}`);
  }
}

async function runTimedLane(page, {
  strategy,
  visibilityFraction,
  repetitionIndex,
  modeOrderPosition,
  visibilityOrderPosition,
}) {
  const lane = await configure(page, { strategy, visibilityFraction });
  assertLaneEvidence(lane, { strategy, visibilityFraction });
  await page.evaluate((context) => window.__WEBGPU_BENCH__.startTrial(context), {
    firstInstanceAddressingScreen: true,
    repetitionIndex,
    modeOrderPosition,
    visibilityOrderPosition,
    plannedModeOrder: repetitionIndex % 2 === 0
      ? `${PORTABLE}|${FEATURE}`
      : `${FEATURE}|${PORTABLE}`,
    plannedVisibilityOrder: repetitionIndex % 2 === 0 ? '0.99|0.2' : '0.2|0.99',
  });
  await waitForTrial(
    page,
    `repetition ${repetitionIndex}, visibility ${visibilityFraction}, ${strategy}`,
  );
  const timing = await page.evaluate(async () => {
    let summary = null;
    try {
      summary = JSON.parse(document.querySelector('#details').textContent);
    } catch {
      // The fail-closed checks below report malformed completion evidence.
    }
    const postValidation = await window.__WEBGPU_BENCH__.validate();
    return {
      rows: window.__WEBGPU_BENCH__.rows,
      summary,
      postValidation,
    };
  });
  const rows = timing.rows;
  if (timing.summary?.accepted !== true
    || rows.length !== 240
    || rows.some((row, index) => (
      row.frameIndex !== index
      || row.modeId !== strategy
      || row.objectCount !== objectCount
      || row.bucketCount !== BUCKET_COUNT
      || row.targetVisibilityFraction !== visibilityFraction
      || row.scenarioLayout !== 'baseline'
      || row.validationPass !== true
      || row.configuredDrawCommands !== BUCKET_COUNT
      || row.configuredRenderObjects !== 1
      || row.configuredComputeDispatches !== 2
      || row.configuredComputeSubmissions !== 1
      || !Number.isFinite(row.gpuRenderMs)
      || !Number.isFinite(row.gpuComputeMs)
      || !Number.isFinite(row.gpuPassTotalMs)
      || row.gpuRenderTimestampUidCount < 1
    ))
    || timing.postValidation?.pass !== true) {
    throw new Error(`Timed lane failed closed: ${JSON.stringify({
      strategy,
      visibilityFraction,
      summary: timing.summary,
      rowCount: rows.length,
      postValidation: timing.postValidation,
    })}`);
  }
  const renderValues = rows.map((row) => row.gpuRenderMs);
  const computeValues = rows.map((row) => row.gpuComputeMs);
  const totalValues = rows.map((row) => row.gpuPassTotalMs);
  return {
    strategy,
    repetitionIndex,
    modeOrderPosition,
    visibilityOrderPosition,
    visibilityFraction,
    validationSha256: sha256Json(lane.validation),
    workloadSha256: lane.workload.scenario.sha256,
    parityIdentity: channelIdentity(lane.parity),
    parity: lane.parity,
    summary: timing.summary,
    postValidationSha256: sha256Json(timing.postValidation),
    gpuRenderP50Ms: median(renderValues),
    gpuRenderP95Ms: percentile(renderValues, 0.95),
    gpuComputeP50Ms: median(computeValues),
    gpuPassTotalP50Ms: median(totalValues),
    measuredRows: rows.length,
  };
}

function pairedEstimates(trials) {
  const pairs = [];
  for (let repetitionIndex = 0; repetitionIndex < repetitions; repetitionIndex += 1) {
    for (const visibilityFraction of VISIBILITY_LEVELS) {
      const portable = trials.find((trial) => (
        trial.repetitionIndex === repetitionIndex
        && trial.visibilityFraction === visibilityFraction
        && trial.strategy === PORTABLE
      ));
      const feature = trials.find((trial) => (
        trial.repetitionIndex === repetitionIndex
        && trial.visibilityFraction === visibilityFraction
        && trial.strategy === FEATURE
      ));
      if (!portable || !feature || portable.workloadSha256 !== feature.workloadSha256) {
        throw new Error(`Missing or mismatched pair at repetition ${repetitionIndex}.`);
      }
      if (portable.parityIdentity !== feature.parityIdentity) {
        throw new Error(`Exact render parity differs at repetition ${repetitionIndex}.`);
      }
      const renderDeltaMs = feature.gpuRenderP50Ms - portable.gpuRenderP50Ms;
      const totalDeltaMs = feature.gpuPassTotalP50Ms - portable.gpuPassTotalP50Ms;
      pairs.push({
        repetitionIndex,
        visibilityFraction,
        featureOrderPosition: feature.modeOrderPosition,
        renderDeltaMs,
        renderDeltaPercent: (renderDeltaMs / portable.gpuRenderP50Ms) * 100,
        totalDeltaMs,
        totalDeltaPercent: (totalDeltaMs / portable.gpuPassTotalP50Ms) * 100,
        portableRenderP50Ms: portable.gpuRenderP50Ms,
        featureRenderP50Ms: feature.gpuRenderP50Ms,
      });
    }
  }
  return pairs;
}

function summarizeCell(pairs, visibilityFraction) {
  const cell = pairs.filter((pair) => pair.visibilityFraction === visibilityFraction);
  return {
    visibilityFraction,
    repetitionCount: cell.length,
    renderDeltaMedianMs: median(cell.map((pair) => pair.renderDeltaMs)),
    renderDeltaMedianPercent: median(cell.map((pair) => pair.renderDeltaPercent)),
    totalDeltaMedianMs: median(cell.map((pair) => pair.totalDeltaMs)),
    totalDeltaMedianPercent: median(cell.map((pair) => pair.totalDeltaPercent)),
    negativeRenderRepetitions: cell.filter((pair) => pair.renderDeltaMs < 0).length,
    renderDeltaMedianByFeatureOrderPosition: Object.fromEntries([0, 1].map((position) => [
      position,
      median(cell.filter((pair) => pair.featureOrderPosition === position)
        .map((pair) => pair.renderDeltaMs)),
    ])),
  };
}

function classify(high, low, exactPass) {
  const orderMedians = Object.values(high.renderDeltaMedianByFeatureOrderPosition)
    .filter(Number.isFinite);
  const noLowRegression = low.renderDeltaMedianMs <= 0.02
    && low.renderDeltaMedianPercent <= 5;
  const promote = protocolConformant
    && exactPass
    && high.renderDeltaMedianMs <= -0.05
    && high.renderDeltaMedianPercent <= -10
    && high.negativeRenderRepetitions >= 5
    && orderMedians.length === 2
    && orderMedians.every((value) => value < 0)
    && noLowRegression;
  const stop = protocolConformant
    && exactPass
    && (
      high.renderDeltaMedianMs > -0.02
      || high.renderDeltaMedianPercent > -5
      || high.negativeRenderRepetitions < 4
      || !noLowRegression
    );
  return {
    decision: !protocolConformant ? 'non-protocol-smoke'
      : promote ? 'promote-to-render-only-crossover'
        : stop ? 'stop-as-standalone-direction'
          : 'intermediate-repeat-once',
    exactPass,
    promote,
    stop,
    gates: {
      highAbsolute: high.renderDeltaMedianMs <= -0.05,
      highPercent: high.renderDeltaMedianPercent <= -10,
      highDirection: high.negativeRenderRepetitions >= 5,
      bothOrderPositionsNegative: orderMedians.length === 2
        && orderMedians.every((value) => value < 0),
      lowDoseNoMaterialRegression: noLowRegression,
    },
  };
}

const executablePath = await findBrowser();
const startedAt = new Date().toISOString();
const outputDirectory = path.resolve('.test-output');
await mkdir(outputDirectory, { recursive: true });
const server = await createServer({
  server: { host: '127.0.0.1', port: 0 },
  logLevel: 'error',
});
let browser;
const errors = [];

try {
  await server.listen();
  const url = server.resolvedUrls?.local?.[0];
  if (!url) throw new Error('Vite did not expose a local URL.');
  browser = await chromium.launch({
    executablePath,
    headless: true,
    args: [
      '--enable-unsafe-webgpu',
      '--enable-webgpu-developer-features',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
    ],
  });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    deviceScaleFactor: 1,
  });
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(`console: ${message.text()}`);
  });
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.__WEBGPU_BENCH__?.ready === true, null, {
    timeout: 120_000,
  });
  const environment = await page.evaluate(() => window.__WEBGPU_BENCH__.environment);
  if (environment.indirectFirstInstanceAvailable !== true) {
    throw new Error('The adapter does not expose indirect-first-instance.');
  }
  if (environment.timestampAvailable !== true) {
    throw new Error('WebGPU timestamp queries are unavailable.');
  }

  const trials = [];
  for (let repetitionIndex = 0; repetitionIndex < repetitions; repetitionIndex += 1) {
    const modeOrder = repetitionIndex % 2 === 0
      ? [PORTABLE, FEATURE]
      : [FEATURE, PORTABLE];
    const visibilityOrder = repetitionIndex % 2 === 0
      ? [...VISIBILITY_LEVELS]
      : [...VISIBILITY_LEVELS].reverse();
    for (let visibilityOrderPosition = 0;
      visibilityOrderPosition < visibilityOrder.length;
      visibilityOrderPosition += 1) {
      const visibilityFraction = visibilityOrder[visibilityOrderPosition];
      for (let modeOrderPosition = 0;
        modeOrderPosition < modeOrder.length;
        modeOrderPosition += 1) {
        const strategy = modeOrder[modeOrderPosition];
        process.stdout.write(
          `screen r${repetitionIndex + 1}/${repetitions} v${visibilityFraction} ${strategy}\n`,
        );
        trials.push(await runTimedLane(page, {
          strategy,
          visibilityFraction,
          repetitionIndex,
          modeOrderPosition,
          visibilityOrderPosition,
        }));
      }
    }
  }
  if (errors.length > 0) throw new Error(errors.join('\n'));

  const pairs = pairedEstimates(trials);
  const high = summarizeCell(pairs, 0.99);
  const low = summarizeCell(pairs, 0.2);
  const exactPass = trials.every((trial) => trial.parity?.pass === true)
    && pairs.length === repetitions * VISIBILITY_LEVELS.length;
  const decision = classify(high, low, exactPass);
  const completedAt = new Date().toISOString();
  const result = {
    schemaVersion: 1,
    kind: 'indirect-first-instance-development-screen',
    protocolConformant,
    startedAt,
    completedAt,
    browser: {
      executable: path.basename(executablePath),
      version: browser.version(),
    },
    environment,
    design: {
      objectCount,
      bucketCount: BUCKET_COUNT,
      visibilityLevels: [...VISIBILITY_LEVELS],
      repetitions,
      warmupFrames: 300,
      measuredFrames: 240,
      modes: [PORTABLE, FEATURE],
      layout: 'baseline',
    },
    high,
    low,
    decision,
    pairs,
    trials,
    errors,
  };
  result.sha256 = sha256Json(result);
  const suffix = protocolConformant ? 'screen' : 'smoke';
  const outputPath = path.join(
    outputDirectory,
    `indirect-first-instance-${suffix}-${compactTimestamp(startedAt)}.json`,
  );
  await writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({ outputPath, high, low, decision }, null, 2)}\n`);
} finally {
  await browser?.close();
  await server.close();
}
