import { access, mkdir } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { chromium } from 'playwright-core';
import { createServer } from 'vite';

const programFiles = process.env.ProgramFiles ?? 'C:\\Program Files';
const programFilesX86 = process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)';
const localAppData = process.env.LOCALAPPDATA;
const browserCandidates = [
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

async function findBrowser() {
  for (const candidate of [...new Set(browserCandidates)]) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Try the next known system-browser location.
    }
  }
  throw new Error('No installed Chrome, Chromium, or Edge executable found. Set BROWSER_PATH.');
}

async function waitForTrial(page, label) {
  await page.waitForFunction(
    () => ['complete', 'error'].includes(window.__WEBGPU_BENCH__?.phase),
    null,
    { timeout: 120_000 },
  );
  const terminal = await page.evaluate(() => ({
    phase: window.__WEBGPU_BENCH__?.phase ?? null,
    error: window.__WEBGPU_BENCH__?.trialError ?? null,
  }));
  if (terminal.phase === 'error') {
    throw new Error(`${label} failed while resolving timestamps: ${terminal.error ?? 'unknown error'}`);
  }
}

const outputDirectory = path.resolve('.test-output');
await mkdir(outputDirectory, { recursive: true });
const executablePath = await findBrowser();
const server = await createServer({
  server: { host: '127.0.0.1', port: 0 },
  logLevel: 'error',
});

let browser;
let page;
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
  page = await context.newPage();
  page.on('pageerror', (error) => errors.push(`pageerror: ${error.stack ?? error.message}`));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(`console: ${message.text()}`);
  });

  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__WEBGPU_BENCH__?.ready === true, null, { timeout: 120_000 });
  const environment = await page.evaluate(() => window.__WEBGPU_BENCH__.environment);
  if (environment.crossOriginIsolated !== true
    || !Number.isFinite(environment.performanceNowQuantumMs)
    || environment.performanceNowQuantumMs > 0.01) {
    throw new Error(`High-resolution CPU timing is unavailable: ${JSON.stringify(environment)}`);
  }
  if (typeof environment.indirectFirstInstanceAvailable !== 'boolean') {
    throw new Error(
      `Benchmark environment did not report indirect-first-instance support: ${JSON.stringify(environment)}`,
    );
  }

  const v11Lifecycle = await page.evaluate(async () => {
    document.querySelector('#objects').value = '4096';
    document.querySelector('#buckets').value = '4';
    document.querySelector('#visibility').value = '0.2';
    document.querySelector('#strategy').value = 'draw-all';
    await window.__WEBGPU_BENCH__.rebuild();
    const baseline = window.__WEBGPU_BENCH__.cacheDiagnostics();
    const cycles = [];
    for (let cycle = 0; cycle < 3; cycle += 1) {
      document.querySelector('#strategy').value = 'three-blocks-current';
      await window.__WEBGPU_BENCH__.rebuild();
      const validation = await window.__WEBGPU_BENCH__.validate();
      const active = window.__WEBGPU_BENCH__.cacheDiagnostics();
      document.querySelector('#strategy').value = 'draw-all';
      await window.__WEBGPU_BENCH__.rebuild();
      const disposed = window.__WEBGPU_BENCH__.cacheDiagnostics();
      cycles.push({ validation, active, disposed });
    }
    return { baseline, cycles };
  });
  if (v11Lifecycle.baseline?.available !== true
    || v11Lifecycle.cycles.some((cycle) => (
      cycle.validation?.pass !== true
      || cycle.active?.computePipelineCacheEntries <= v11Lifecycle.baseline.computePipelineCacheEntries
      || cycle.disposed?.computePipelineCacheEntries
        !== v11Lifecycle.baseline.computePipelineCacheEntries
      || cycle.disposed?.computeProgramEntries !== v11Lifecycle.baseline.computeProgramEntries
    ))) {
    throw new Error(`Three Blocks 0.11 lifecycle stress failed: ${JSON.stringify(v11Lifecycle)}`);
  }

  const fourBucketControl = await page.evaluate(async () => {
    document.querySelector('#objects').value = '4096';
    document.querySelector('#buckets').value = '4';
    document.querySelector('#visibility').value = '0.2';
    const validations = {};
    for (const strategy of ['draw-all', 'fixed-slice', 'three-blocks-coalesced']) {
      document.querySelector('#strategy').value = strategy;
      await window.__WEBGPU_BENCH__.rebuild();
      validations[strategy] = await window.__WEBGPU_BENCH__.validate();
    }
    return validations;
  });
  if (Object.values(fourBucketControl).some((validation) => validation?.pass !== true)) {
    throw new Error(`Four-bucket family-control validation failed: ${JSON.stringify(fourBucketControl)}`);
  }

  const drawAll = await page.evaluate(async () => {
    document.querySelector('#objects').value = '4096';
    document.querySelector('#buckets').value = '1';
    document.querySelector('#visibility').value = '0.2';
    document.querySelector('#strategy').value = 'draw-all';
    await window.__WEBGPU_BENCH__.rebuild();
    return window.__WEBGPU_BENCH__.validate();
  });
  if (!drawAll?.pass) throw new Error(`Draw-all validation failed: ${JSON.stringify(drawAll)}`);

  const fixedSlice = await page.evaluate(async () => {
    document.querySelector('#strategy').value = 'fixed-slice';
    await window.__WEBGPU_BENCH__.rebuild();
    return window.__WEBGPU_BENCH__.validate();
  });
  if (!fixedSlice?.pass) throw new Error(`Fixed-slice validation failed: ${JSON.stringify(fixedSlice)}`);

  const threeBlocksCurrent = await page.evaluate(async () => {
    document.querySelector('#strategy').value = 'three-blocks-current';
    await window.__WEBGPU_BENCH__.rebuild();
    return window.__WEBGPU_BENCH__.validate();
  });
  if (!threeBlocksCurrent?.pass) {
    throw new Error(`Three Blocks 0.11 validation failed: ${JSON.stringify(threeBlocksCurrent)}`);
  }

  await page.evaluate(() => window.__WEBGPU_BENCH__.startTrial({ smokeRun: true }));
  await waitForTrial(page, 'Three Blocks timing smoke');
  const timing = await page.evaluate(() => {
    const rows = window.__WEBGPU_BENCH__.rows;
    return {
      rowCount: rows.length,
      missingCompute: rows.filter((row) => !Number.isFinite(row.gpuComputeMs)).length,
      missingRender: rows.filter((row) => !Number.isFinite(row.gpuRenderMs)).length,
      badTotals: rows.filter((row) => (
        !Number.isFinite(row.gpuPassTotalMs)
        || Math.abs(row.gpuPassTotalMs - row.gpuComputeMs - row.gpuRenderMs) > 1e-9
      )).length,
    };
  });
  if (timing.rowCount !== 240
    || timing.missingCompute !== 0
    || timing.missingRender !== 0
    || timing.badTotals !== 0) {
    throw new Error(`Three Blocks timing smoke failed: ${JSON.stringify(timing)}`);
  }

  const heterogeneous = await page.evaluate(async () => {
    document.querySelector('#objects').value = '4096';
    document.querySelector('#buckets').value = '32';
    document.querySelector('#visibility').value = '0.2';
    document.querySelector('#strategy').value = 'three-blocks-current';
    await window.__WEBGPU_BENCH__.rebuild();
    const threeBlocks = await window.__WEBGPU_BENCH__.validate();
    document.querySelector('#strategy').value = 'fixed-slice';
    await window.__WEBGPU_BENCH__.rebuild();
    const fixedSlice32 = await window.__WEBGPU_BENCH__.validate();
    document.querySelector('#strategy').value = 'three-blocks-coalesced';
    await window.__WEBGPU_BENCH__.rebuild();
    const coalesced = await window.__WEBGPU_BENCH__.validate();
    return { threeBlocks, fixedSlice: fixedSlice32, coalesced };
  });
  if (!heterogeneous.threeBlocks?.pass
    || !heterogeneous.fixedSlice?.pass
    || !heterogeneous.coalesced?.pass) {
    throw new Error(`32-bucket validation failed: ${JSON.stringify(heterogeneous)}`);
  }

  await page.evaluate(() => window.__WEBGPU_BENCH__.startTrial({ heterogeneousSmokeRun: true }));
  await waitForTrial(page, '32-bucket timing smoke');
  const heterogeneousTiming = await page.evaluate(() => {
    const rows = window.__WEBGPU_BENCH__.rows;
    return {
      rowCount: rows.length,
      missingCompute: rows.filter((row) => !Number.isFinite(row.gpuComputeMs)).length,
      missingRender: rows.filter((row) => !Number.isFinite(row.gpuRenderMs)).length,
      badSchedule: rows.filter((row) => (
        row.configuredComputeDispatches !== 128
        || row.configuredComputeSubmissions !== 1
      )).length,
    };
  });
  if (heterogeneousTiming.rowCount !== 240
    || heterogeneousTiming.missingCompute !== 0
    || heterogeneousTiming.missingRender !== 0
    || heterogeneousTiming.badSchedule !== 0) {
    throw new Error(`32-bucket timing smoke failed: ${JSON.stringify(heterogeneousTiming)}`);
  }

  let historical;
  if (!environment.indirectFirstInstanceAvailable) {
    historical = {
      status: 'skipped',
      reason: 'The WebGPU adapter does not expose the optional indirect-first-instance feature.',
    };
  } else {
    const validation = await page.evaluate(async () => {
      document.querySelector('#objects').value = '4096';
      document.querySelector('#buckets').value = '32';
      document.querySelector('#visibility').value = '0.2';
      document.querySelector('#strategy').value = 'three-blocks-historical';
      await window.__WEBGPU_BENCH__.rebuild();
      return window.__WEBGPU_BENCH__.validate();
    });
    if (validation?.pass !== true
      || validation?.kind !== 'three-blocks-historical-exact-membership') {
      throw new Error(`Three Blocks 0.10 exact validation failed: ${JSON.stringify(validation)}`);
    }

    await page.evaluate(() => window.__WEBGPU_BENCH__.startTrial({ historicalSmokeRun: true }));
    await waitForTrial(page, 'Three Blocks 0.10 timing smoke');
    const timing = await page.evaluate(() => {
      const rows = window.__WEBGPU_BENCH__.rows;
      return {
        rowCount: rows.length,
        missingCompute: rows.filter((row) => !Number.isFinite(row.gpuComputeMs)).length,
        missingRender: rows.filter((row) => !Number.isFinite(row.gpuRenderMs)).length,
        badTotals: rows.filter((row) => (
          !Number.isFinite(row.gpuPassTotalMs)
          || Math.abs(row.gpuPassTotalMs - row.gpuComputeMs - row.gpuRenderMs) > 1e-9
        )).length,
        badSchedule: rows.filter((row) => (
          row.configuredComputeDispatches !== 9
          || row.configuredComputeSubmissions !== 1
        )).length,
        failedValidation: rows.filter((row) => (
          row.validationPass !== true
          || row.validationKind !== 'three-blocks-historical-exact-membership'
        )).length,
      };
    });
    if (timing.rowCount !== 240
      || timing.missingCompute !== 0
      || timing.missingRender !== 0
      || timing.badTotals !== 0
      || timing.badSchedule !== 0
      || timing.failedValidation !== 0) {
      throw new Error(`Three Blocks 0.10 timing smoke failed: ${JSON.stringify(timing)}`);
    }
    const reconstructionCycles = await page.evaluate(async () => {
      const validations = [];
      for (let cycle = 0; cycle < 3; cycle += 1) {
        document.querySelector('#strategy').value = 'draw-all';
        await window.__WEBGPU_BENCH__.rebuild();
        document.querySelector('#strategy').value = 'three-blocks-historical';
        await window.__WEBGPU_BENCH__.rebuild();
        validations.push(await window.__WEBGPU_BENCH__.validate());
      }
      return validations;
    });
    if (reconstructionCycles.some((result) => (
      result?.pass !== true
      || result?.kind !== 'three-blocks-historical-exact-membership'
    ))) {
      throw new Error(`Three Blocks 0.10 reconstruction failed: ${JSON.stringify(reconstructionCycles)}`);
    }
    historical = {
      status: 'passed',
      validation,
      timing,
      reconstruction: reconstructionCycles.at(-1),
      reconstructionCycles,
    };
  }
  if (errors.length) throw new Error(errors.join('\n'));

  const backend = await page.locator('#backend').textContent();
  process.stdout.write(`${JSON.stringify({
    browser: executablePath,
    backend,
    environment,
    v11Lifecycle,
    fourBucketControl,
    drawAll,
    fixedSlice,
    threeBlocksCurrent,
    timing,
    heterogeneous,
    heterogeneousTiming,
    historical,
  }, null, 2)}\n`);
} catch (error) {
  if (page) await page.screenshot({ path: path.join(outputDirectory, 'browser-smoke-failure.png'), fullPage: true });
  throw error;
} finally {
  await browser?.close();
  await server.close();
}
