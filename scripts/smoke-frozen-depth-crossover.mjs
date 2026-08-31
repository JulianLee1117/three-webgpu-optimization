import { access, mkdir } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { createServer } from 'vite';
import {
  FROZEN_CROSSOVER_MEASURED_FRAMES,
  frozenCrossoverFrame,
} from '../src/benchmark/frozen-crossover-schedule.js';
import {
  FROZEN_DEPTH_CROSSOVER_LANES,
  FROZEN_DEPTH_CROSSOVER_MODE,
} from '../src/benchmark/plan.js';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const failureDirectory = path.join(projectRoot, '.test-output');
const browserCandidates = [
  process.env.BROWSER_PATH,
  path.join(process.env.ProgramFiles ?? 'C:\\Program Files', 'Google', 'Chrome', 'Application', 'chrome.exe'),
  path.join(process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)', 'Google', 'Chrome', 'Application', 'chrome.exe'),
  path.join(process.env.ProgramFiles ?? 'C:\\Program Files', 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
  process.env.LOCALAPPDATA
    && path.join(process.env.LOCALAPPDATA, 'Google', 'Chrome', 'Application', 'chrome.exe'),
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);

async function findBrowser() {
  for (const candidate of [...new Set(browserCandidates)]) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Try the next installed browser.
    }
  }
  throw new Error('No installed Chrome, Chromium, or Edge executable found. Set BROWSER_PATH.');
}

function assertValidation(validation, expectedFirstLane) {
  const front = validation?.lanes?.['front-to-back'];
  const reverse = validation?.lanes?.reverse;
  const expectedFirstOrder = expectedFirstLane === FROZEN_DEPTH_CROSSOVER_LANES[0]
    ? 'front-to-back'
    : 'reverse';
  if (validation?.pass !== true
    || validation.kind !== 'frozen-depth-crossover-exact-paired-snapshots'
    || validation.laneStorageOrder !== expectedFirstOrder
    || validation.physicalBinSequenceCommitmentsEqual !== true
    || front?.pass !== true
    || reverse?.pass !== true
    || front.depthBins?.pass !== true
    || reverse.depthBins?.pass !== true
    || front.storageOffset === reverse.storageOffset
    || ![front.storageOffset, reverse.storageOffset].includes(0)
    || ![front.storageOffset, reverse.storageOffset].includes(4096)
    || validation.commandValidation?.pass !== true
    || validation.representation?.bundleRecordCallbackCount !== 1
    || validation.representation?.meshCount !== 1
    || validation.representation?.configuredComputeDispatches !== 0
    || validation.representation?.configuredComputeSubmissions !== 0) {
    throw new Error(`Frozen crossover validation failed: ${JSON.stringify(validation)}`);
  }
}

function assertParity(parity) {
  if (parity?.pass !== true
    || parity.kind !== 'frozen-depth-crossover-exact-render-parity'
    || parity.crossLaneExact !== true
    || parity.snapshotValidation?.pass !== true
    || FROZEN_DEPTH_CROSSOVER_LANES.some((lane) => (
      parity.lanes?.[lane]?.pass !== true
      || parity.lanes[lane].stability?.pass !== true
      || parity.lanes[lane].objectIdValidation?.pass !== true
      || parity.lanes[lane].reversedDepthBuffer !== true
    ))) {
    throw new Error(`Frozen crossover parity failed: ${JSON.stringify(parity)}`);
  }
}

async function configure(page, { layout, laneStorageOrder, orientationOffset }) {
  return page.evaluate(async (configuration) => {
    const bench = window.__WEBGPU_BENCH__;
    bench.configureFrozenCrossover({
      laneStorageOrder: configuration.laneStorageOrder,
      superblockOrientationOffset: configuration.orientationOffset,
    });
    document.querySelector('#objects').value = '4096';
    document.querySelector('#buckets').value = '4';
    document.querySelector('#visibility').value = '0.99';
    document.querySelector('#layout').value = configuration.layout;
    document.querySelector('#strategy').value = configuration.mode;
    await bench.rebuild();
    const validation = await bench.validate();
    const parity = await bench.captureRenderParity();
    const repeatedValidation = await bench.validate();
    return { selectedConfig: bench.selectedConfig(), validation, parity, repeatedValidation };
  }, {
    layout,
    laneStorageOrder,
    orientationOffset,
    mode: FROZEN_DEPTH_CROSSOVER_MODE,
  });
}

await mkdir(failureDirectory, { recursive: true });
const executablePath = await findBrowser();
const server = await createServer({
  root: projectRoot,
  configFile: false,
  resolve: { dedupe: ['three'] },
  server: {
    host: '127.0.0.1',
    port: 0,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  logLevel: 'error',
});
let browser;
let page;
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
  page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.stack ?? error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__WEBGPU_BENCH__?.ready === true,
    null,
    { timeout: 120_000 },
  );
  const pageEnvironment = await page.evaluate(() => window.__WEBGPU_BENCH__.environment);
  if (pageEnvironment.reversedDepth !== true
    || pageEnvironment.rendererReversedDepthBuffer !== true) {
    throw new Error(`Frozen crossover reversed-depth state is invalid: ${JSON.stringify(pageEnvironment)}`);
  }

  const firstOrder = [...FROZEN_DEPTH_CROSSOVER_LANES];
  const pre = await configure(page, {
    layout: 'high-overlap',
    laneStorageOrder: firstOrder,
    orientationOffset: 0,
  });
  assertValidation(pre.validation, firstOrder[0]);
  assertValidation(pre.repeatedValidation, firstOrder[0]);
  assertParity(pre.parity);
  if (JSON.stringify(pre.validation) !== JSON.stringify(pre.repeatedValidation)) {
    throw new Error('Frozen crossover validation changed before timing.');
  }

  await page.evaluate(async () => {
    await window.__WEBGPU_BENCH__.startTrial({ frozenCrossoverSmoke: true });
  });
  await page.waitForFunction(
    () => ['complete', 'error'].includes(window.__WEBGPU_BENCH__?.phase),
    null,
    { timeout: 180_000 },
  );
  const timing = await page.evaluate(() => ({
    phase: window.__WEBGPU_BENCH__.phase,
    error: window.__WEBGPU_BENCH__.trialError,
    rows: window.__WEBGPU_BENCH__.rows,
    summary: JSON.parse(document.querySelector('#details').textContent),
  }));
  if (timing.phase !== 'complete'
    || timing.summary?.accepted !== true
    || timing.rows.length !== FROZEN_CROSSOVER_MEASURED_FRAMES
    || timing.summary.expectedRenderTimestampUidCount !== 1
    || timing.summary.invalidRenderTimestampUidCountFrames !== 0
    || timing.summary.completionInvariant?.pass !== true) {
    throw new Error(`Frozen crossover timing failed: ${JSON.stringify(timing)}`);
  }
  for (let index = 0; index < timing.rows.length; index += 1) {
    const row = timing.rows[index];
    const scheduled = frozenCrossoverFrame(index, 0);
    const expectedBase = scheduled.laneId === firstOrder[0] ? 0 : 4096;
    if (row.frameIndex !== index
      || row.phaseFrameIndex !== index
      || row.crossoverBlockIndex !== scheduled.crossoverBlockIndex
      || row.withinBlockPosition !== scheduled.withinBlockPosition
      || row.crossoverPattern !== scheduled.pattern
      || row.laneId !== scheduled.laneId
      || row.laneBase !== expectedBase
      || row.gpuRenderTimestampUidCount !== 1
      || row.gpuComputeMs !== null
      || row.configuredComputeDispatches !== 0
      || row.configuredComputeSubmissions !== 0
      || (index > 0 && (
        row.selectorWriteSerial !== timing.rows[index - 1].selectorWriteSerial + 1
        || row.renderCallSerial !== timing.rows[index - 1].renderCallSerial + 1
      ))) {
      throw new Error(`Frozen crossover row ${index} violates the schedule.`);
    }
  }

  const post = await page.evaluate(async () => ({
    validation: await window.__WEBGPU_BENCH__.validate(),
    parity: await window.__WEBGPU_BENCH__.captureRenderParity(),
  }));
  assertValidation(post.validation, firstOrder[0]);
  assertParity(post.parity);
  if (JSON.stringify(post.validation) !== JSON.stringify(pre.validation)) {
    throw new Error('Frozen crossover validation changed after timing.');
  }

  const reversedOrder = [...FROZEN_DEPTH_CROSSOVER_LANES].reverse();
  const reversed = await configure(page, {
    layout: 'low-overlap',
    laneStorageOrder: reversedOrder,
    orientationOffset: 1,
  });
  assertValidation(reversed.validation, reversedOrder[0]);
  assertValidation(reversed.repeatedValidation, reversedOrder[0]);
  assertParity(reversed.parity);
  if (errors.length > 0) throw new Error(errors.join('\n'));

  process.stdout.write(`${JSON.stringify({
    browser: { executable: path.basename(executablePath), version: browser.version() },
    backend: await page.locator('#backend').textContent(),
    reversedDepth: {
      camera: pageEnvironment.reversedDepth,
      renderer: pageEnvironment.rendererReversedDepthBuffer,
    },
    timing: {
      rowCount: timing.rows.length,
      timestampQuantumNs: timing.summary.quantumNs,
      completionInvariant: timing.summary.completionInvariant,
    },
    storageOrders: [pre.selectedConfig.laneStorageOrder, reversed.selectedConfig.laneStorageOrder],
    exactParity: true,
  }, null, 2)}\n`);
} catch (error) {
  if (page) {
    await page.screenshot({
      path: path.join(failureDirectory, 'frozen-depth-crossover-smoke-failure.png'),
      fullPage: true,
    });
  }
  throw error;
} finally {
  await browser?.close();
  await server.close();
}
