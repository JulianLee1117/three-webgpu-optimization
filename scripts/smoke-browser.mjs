import { createHash } from 'node:crypto';
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

async function waitForAnimationFrames(page, count = 2) {
  await page.evaluate((frameCount) => new Promise((resolve) => {
    let remaining = frameCount;
    const advance = () => {
      remaining -= 1;
      if (remaining === 0) resolve();
      else requestAnimationFrame(advance);
    };
    requestAnimationFrame(advance);
  }), count);
}

async function comparePngPixels(page, left, right, channelTolerance = 0) {
  const pixelComparison = await page.evaluate(async ({ leftBase64, rightBase64, tolerance }) => {
    const decode = async (base64) => {
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
      }
      return createImageBitmap(new Blob([bytes], { type: 'image/png' }));
    };
    const [leftImage, rightImage] = await Promise.all([
      decode(leftBase64),
      decode(rightBase64),
    ]);
    if (leftImage.width !== rightImage.width || leftImage.height !== rightImage.height) {
      return {
        pass: false,
        dimensionsEqual: false,
        leftWidth: leftImage.width,
        leftHeight: leftImage.height,
        rightWidth: rightImage.width,
        rightHeight: rightImage.height,
        channelTolerance: tolerance,
        differingPixels: null,
        differingChannels: null,
        maxChannelDifference: null,
      };
    }
    const width = leftImage.width;
    const height = leftImage.height;
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext('2d', {
      alpha: true,
      colorSpace: 'srgb',
      willReadFrequently: true,
    });
    const pixels = (image) => {
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0);
      return context.getImageData(0, 0, canvas.width, canvas.height).data;
    };
    const leftPixels = pixels(leftImage);
    const rightPixels = pixels(rightImage);
    leftImage.close();
    rightImage.close();
    let differingPixels = 0;
    let differingChannels = 0;
    let maxChannelDifference = 0;
    for (let offset = 0; offset < leftPixels.length; offset += 4) {
      let pixelDiffers = false;
      for (let channel = 0; channel < 4; channel += 1) {
        const difference = Math.abs(leftPixels[offset + channel] - rightPixels[offset + channel]);
        maxChannelDifference = Math.max(maxChannelDifference, difference);
        if (difference > tolerance) {
          differingChannels += 1;
          pixelDiffers = true;
        }
      }
      if (pixelDiffers) differingPixels += 1;
    }
    return {
      pass: differingPixels === 0,
      dimensionsEqual: true,
      width,
      height,
      channelTolerance: tolerance,
      differingPixels,
      differingChannels,
      maxChannelDifference,
    };
  }, {
    leftBase64: left.toString('base64'),
    rightBase64: right.toString('base64'),
    tolerance: channelTolerance,
  });
  return {
    encodingBytesEqual: left.equals(right),
    leftSha256: createHash('sha256').update(left).digest('hex'),
    rightSha256: createHash('sha256').update(right).digest('hex'),
    ...pixelComparison,
  };
}

async function captureStableCanvas(page, label) {
  const canvas = page.locator('#canvas-host canvas');
  const previousInlineStyle = await canvas.evaluate((element) => {
    const previous = element.getAttribute('style');
    Object.assign(element.style, {
      position: 'fixed',
      left: '0px',
      top: '0px',
      zIndex: '2147483647',
      width: `${element.width}px`,
      height: `${element.height}px`,
    });
    return previous;
  });
  try {
    await waitForAnimationFrames(page);
    const first = await canvas.screenshot({ type: 'png', animations: 'disabled' });
    await waitForAnimationFrames(page);
    const second = await canvas.screenshot({ type: 'png', animations: 'disabled' });
    const stability = await comparePngPixels(page, first, second, 0);
    if (!stability.pass) {
      throw new Error(`${label} decoded-RGBA screenshot was not stable: ${JSON.stringify(stability)}`);
    }
    return { png: second, stability };
  } finally {
    await canvas.evaluate((element, previous) => {
      if (previous === null) element.removeAttribute('style');
      else element.setAttribute('style', previous);
    }, previousInlineStyle);
  }
}

async function configureAndValidate(page, {
  strategy,
  objectCount = 4096,
  bucketCount,
  visibilityFraction = 0.2,
}) {
  return page.evaluate(async (config) => {
    document.querySelector('#objects').value = String(config.objectCount);
    document.querySelector('#buckets').value = String(config.bucketCount);
    document.querySelector('#visibility').value = String(config.visibilityFraction);
    document.querySelector('#strategy').value = config.strategy;
    await window.__WEBGPU_BENCH__.rebuild();
    return window.__WEBGPU_BENCH__.validate();
  }, { strategy, objectCount, bucketCount, visibilityFraction });
}

function assertFixedSliceTiming(timing, { label, bucketCount }) {
  if (timing.rowCount !== 240
    || timing.missingCompute !== 0
    || timing.missingRender !== 0
    || timing.badTotals !== 0
    || timing.badSchedule !== 0
    || timing.failedValidation !== 0
    || timing.badFrameSequence !== 0
    || timing.uniqueGpuFrameIds !== 240
    || timing.nonIncreasingGpuFrameIds !== 0) {
    throw new Error(`${label} failed: ${JSON.stringify(timing)}`);
  }
  if (timing.bucketCounts.length !== 1 || timing.bucketCounts[0] !== bucketCount) {
    throw new Error(`${label} reported the wrong bucket edge: ${JSON.stringify(timing)}`);
  }
}

async function readFixedSliceTiming(page, expectedBucketCount) {
  return page.evaluate((bucketCount) => {
    const rows = window.__WEBGPU_BENCH__.rows;
    return {
      rowCount: rows.length,
      bucketCounts: [...new Set(rows.map((row) => row.bucketCount))],
      missingCompute: rows.filter((row) => !Number.isFinite(row.gpuComputeMs)).length,
      missingRender: rows.filter((row) => !Number.isFinite(row.gpuRenderMs)).length,
      badTotals: rows.filter((row) => (
        !Number.isFinite(row.gpuPassTotalMs)
        || Math.abs(row.gpuPassTotalMs - row.gpuComputeMs - row.gpuRenderMs) > 1e-9
      )).length,
      badSchedule: rows.filter((row) => (
        row.bucketCount !== bucketCount
        || row.configuredDrawCommands !== bucketCount
        || row.configuredRenderObjects !== 1
        || row.configuredComputeDispatches !== 2
        || row.configuredComputeSubmissions !== 1
      )).length,
      failedValidation: rows.filter((row) => (
        row.validationPass !== true
        || row.validationKind !== 'fixed-slice-exact-membership'
      )).length,
      badFrameSequence: rows.filter((row, index) => row.frameIndex !== index).length,
      uniqueGpuFrameIds: new Set(rows.map((row) => row.gpuFrameId)).size,
      nonIncreasingGpuFrameIds: rows.filter((row, index) => (
        index > 0 && row.gpuFrameId <= rows[index - 1].gpuFrameId
      )).length,
    };
  }, expectedBucketCount);
}

function assertTrialEvidence(evidence, { label, bucketCount, validationKind }) {
  if (evidence?.validation?.pass !== true
    || evidence.validation.kind !== validationKind
    || evidence.validation.commandValidation?.commandCount !== bucketCount
    || evidence.validation.commandValidation?.records?.length !== bucketCount
    || evidence.validation.membershipDigests?.pass !== true
    || evidence.validation.membershipDigests?.expected?.sha256
      !== evidence?.workload?.scenario?.expectedVisibleIdsCanonicalSha256
    || evidence?.workload?.geometryFixtures?.bucketCount !== bucketCount
    || !/^[0-9a-f]{64}$/.test(evidence?.workload?.geometryFixtures?.sha256 ?? '')
    || evidence?.workload?.scenario?.bucketCount !== bucketCount
    || !/^[0-9a-f]{64}$/.test(evidence?.workload?.scenario?.sha256 ?? '')
    || !Number.isInteger(evidence?.workload?.scenarioSeed)) {
    throw new Error(`${label} provenance/validation evidence failed: ${JSON.stringify(evidence)}`);
  }
}

function sameTrackedMemory(left, right) {
  const fields = [
    'attributes',
    'attributesSize',
    'geometries',
    'indexAttributes',
    'indexAttributesSize',
    'indirectStorageAttributes',
    'indirectStorageAttributesSize',
    'storageAttributes',
    'storageAttributesSize',
  ];
  return fields.every((field) => left?.[field] === right?.[field]);
}

function summarizeValidation(validation) {
  if (!validation) return validation;
  return {
    pass: validation.pass,
    kind: validation.kind,
    membership: validation.membership && {
      pass: validation.membership.pass,
      expectedCount: validation.membership.expectedCount,
      listedCount: validation.membership.listedCount,
      errors: validation.membership.errors,
    },
    membershipDigest: validation.membershipDigests && {
      pass: validation.membershipDigests.pass,
      expectedCount: validation.membershipDigests.expected?.count,
      expectedSha256: validation.membershipDigests.expected?.sha256,
      actualCount: validation.membershipDigests.actual?.count,
      actualSha256: validation.membershipDigests.actual?.sha256,
    },
    commandValidation: validation.commandValidation && {
      pass: validation.commandValidation.pass,
      commandCount: validation.commandValidation.commandCount,
      totalInstanceCount: validation.commandValidation.totalInstanceCount,
      errorCount: validation.commandValidation.errors?.length ?? 0,
    },
    overflow: validation.overflow,
    readbackErrorCount: validation.readbackErrors?.length ?? 0,
  };
}

function summarizeLifecycle(lifecycle) {
  return {
    baseline: lifecycle.baseline,
    cycles: lifecycle.cycles.map((cycle) => ({
      validation: summarizeValidation(cycle.validation),
      active: cycle.active,
      disposed: cycle.disposed,
    })),
  };
}

function summarizeValidationMap(validations) {
  return Object.fromEntries(
    Object.entries(validations).map(([name, validation]) => [name, summarizeValidation(validation)]),
  );
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

  const colorScreenshotParity = [];
  const fixedSliceReferencePng = new Map();
  for (const bucketCount of [4, 32, 128]) {
    const drawAllValidation = await configureAndValidate(page, {
      strategy: 'draw-all',
      bucketCount,
    });
    if (drawAllValidation?.pass !== true || drawAllValidation?.kind !== 'draw-all-reference') {
      throw new Error(
        `Draw-all B=${bucketCount} render reference failed: ${JSON.stringify(drawAllValidation)}`,
      );
    }
    const drawAllCapture = await captureStableCanvas(page, `Draw-all B=${bucketCount}`);

    const fixedSliceValidation = await configureAndValidate(page, {
      strategy: 'fixed-slice',
      bucketCount,
    });
    if (fixedSliceValidation?.pass !== true
      || fixedSliceValidation?.kind !== 'fixed-slice-exact-membership'
      || fixedSliceValidation?.membershipDigests?.pass !== true
      || fixedSliceValidation?.commandValidation?.commandCount !== bucketCount) {
      throw new Error(
        `Fixed-slice B=${bucketCount} render candidate failed: ${JSON.stringify(fixedSliceValidation)}`,
      );
    }
    const fixedSliceCapture = await captureStableCanvas(page, `Fixed-slice B=${bucketCount}`);
    const comparison = await comparePngPixels(
      page,
      drawAllCapture.png,
      fixedSliceCapture.png,
      0,
    );
    if (!comparison.pass) {
      throw new Error(
        `Draw-all/fixed-slice B=${bucketCount} decoded-RGBA screenshot differs: ${JSON.stringify(comparison)}`,
      );
    }
    fixedSliceReferencePng.set(bucketCount, fixedSliceCapture.png);
    colorScreenshotParity.push({
      scope: 'fixed-camera-static-color-output',
      objectCount: 4096,
      bucketCount,
      visibilityFraction: 0.2,
      channelTolerance: 0,
      drawAllValidationKind: drawAllValidation.kind,
      fixedSliceValidationKind: fixedSliceValidation.kind,
      fixedSliceMembershipDigest: fixedSliceValidation.membershipDigests.expected.sha256,
      drawAllStability: drawAllCapture.stability,
      fixedSliceStability: fixedSliceCapture.stability,
      comparison,
    });
  }

  const fixedSlice128Evidence = await page.evaluate(
    () => window.__WEBGPU_BENCH__.startTrial({ fixedSlice128EdgeSmokeRun: true }),
  );
  assertTrialEvidence(fixedSlice128Evidence, {
    label: 'Fixed-slice B=128 timing edge',
    bucketCount: 128,
    validationKind: 'fixed-slice-exact-membership',
  });
  await waitForTrial(page, 'Fixed-slice B=128 timing edge');
  const fixedSlice128Timing = await readFixedSliceTiming(page, 128);
  assertFixedSliceTiming(fixedSlice128Timing, {
    label: 'Fixed-slice B=128 timing edge',
    bucketCount: 128,
  });
  const fixedSlice128ReplayValidation = await page.evaluate(
    () => window.__WEBGPU_BENCH__.validate(),
  );
  const fixedSlice128ReplayCapture = await captureStableCanvas(page, 'Fixed-slice B=128 replay');
  const fixedSlice128ReplayComparison = await comparePngPixels(
    page,
    fixedSliceReferencePng.get(128),
    fixedSlice128ReplayCapture.png,
    0,
  );
  if (fixedSlice128ReplayValidation?.pass !== true || !fixedSlice128ReplayComparison.pass) {
    throw new Error(`Fixed-slice B=128 replay failed: ${JSON.stringify({
      validation: fixedSlice128ReplayValidation,
      comparison: fixedSlice128ReplayComparison,
    })}`);
  }
  const fixedSlice128Edge = {
    timing: fixedSlice128Timing,
    replayValidationKind: fixedSlice128ReplayValidation.kind,
    replayMembershipDigest: fixedSlice128ReplayValidation.membershipDigests.expected.sha256,
    replayComparison: fixedSlice128ReplayComparison,
  };

  const fixedSlice32Validation = await configureAndValidate(page, {
    strategy: 'fixed-slice',
    bucketCount: 32,
  });
  if (fixedSlice32Validation?.pass !== true) {
    throw new Error(
      `Fixed-slice B=32 sustained validation failed: ${JSON.stringify(fixedSlice32Validation)}`,
    );
  }
  const fixedSlice32Evidence = await page.evaluate(
    () => window.__WEBGPU_BENCH__.startTrial({ fixedSliceSustainedSmokeRun: true }),
  );
  assertTrialEvidence(fixedSlice32Evidence, {
    label: 'Fixed-slice B=32 sustained timing',
    bucketCount: 32,
    validationKind: 'fixed-slice-exact-membership',
  });
  await waitForTrial(page, 'Fixed-slice B=32 sustained timing');
  const fixedSlice32Timing = await readFixedSliceTiming(page, 32);
  assertFixedSliceTiming(fixedSlice32Timing, {
    label: 'Fixed-slice B=32 sustained timing',
    bucketCount: 32,
  });
  const fixedSlice32ReplayValidation = await page.evaluate(
    () => window.__WEBGPU_BENCH__.validate(),
  );
  const fixedSlice32ReplayCapture = await captureStableCanvas(page, 'Fixed-slice B=32 replay');
  const fixedSlice32ReplayComparison = await comparePngPixels(
    page,
    fixedSliceReferencePng.get(32),
    fixedSlice32ReplayCapture.png,
    0,
  );
  if (fixedSlice32ReplayValidation?.pass !== true || !fixedSlice32ReplayComparison.pass) {
    throw new Error(`Fixed-slice B=32 sustained replay failed: ${JSON.stringify({
      validation: fixedSlice32ReplayValidation,
      comparison: fixedSlice32ReplayComparison,
    })}`);
  }
  const fixedSlice32Sustained = {
    timing: fixedSlice32Timing,
    replayValidationKind: fixedSlice32ReplayValidation.kind,
    replayMembershipDigest: fixedSlice32ReplayValidation.membershipDigests.expected.sha256,
    replayComparison: fixedSlice32ReplayComparison,
  };

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
      || !sameTrackedMemory(cycle.disposed?.memory, v11Lifecycle.baseline.memory)
    ))) {
    throw new Error(`Three Blocks 0.11 lifecycle stress failed: ${JSON.stringify(v11Lifecycle)}`);
  }

  const fixedSliceLifecycle = await page.evaluate(async () => {
    document.querySelector('#objects').value = '4096';
    document.querySelector('#buckets').value = '32';
    document.querySelector('#visibility').value = '0.2';
    document.querySelector('#strategy').value = 'draw-all';
    await window.__WEBGPU_BENCH__.rebuild();
    await window.__WEBGPU_BENCH__.validate();
    const baseline = window.__WEBGPU_BENCH__.cacheDiagnostics();
    const cycles = [];
    for (let cycle = 0; cycle < 3; cycle += 1) {
      document.querySelector('#strategy').value = 'fixed-slice';
      await window.__WEBGPU_BENCH__.rebuild();
      const validation = await window.__WEBGPU_BENCH__.validate();
      const active = window.__WEBGPU_BENCH__.cacheDiagnostics();
      document.querySelector('#strategy').value = 'draw-all';
      await window.__WEBGPU_BENCH__.rebuild();
      await window.__WEBGPU_BENCH__.validate();
      const disposed = window.__WEBGPU_BENCH__.cacheDiagnostics();
      cycles.push({ validation, active, disposed });
    }
    return { baseline, cycles };
  });
  if (fixedSliceLifecycle.baseline?.available !== true
    || fixedSliceLifecycle.cycles.some((cycle) => (
      cycle.validation?.pass !== true
      || cycle.active?.memory?.storageAttributes
        <= fixedSliceLifecycle.baseline.memory.storageAttributes
      || cycle.active?.memory?.indirectStorageAttributes
        <= fixedSliceLifecycle.baseline.memory.indirectStorageAttributes
      || !sameTrackedMemory(cycle.disposed?.memory, fixedSliceLifecycle.baseline.memory)
    ))) {
    throw new Error(
      `Fixed-slice storage lifecycle stress failed: ${JSON.stringify(fixedSliceLifecycle)}`,
    );
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

  const currentEvidence = await page.evaluate(
    () => window.__WEBGPU_BENCH__.startTrial({ smokeRun: true }),
  );
  assertTrialEvidence(currentEvidence, {
    label: 'Three Blocks timing smoke',
    bucketCount: 1,
    validationKind: 'three-blocks-current-exact-membership',
  });
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

  const heterogeneousEvidence = await page.evaluate(
    () => window.__WEBGPU_BENCH__.startTrial({ heterogeneousSmokeRun: true }),
  );
  assertTrialEvidence(heterogeneousEvidence, {
    label: '32-bucket timing smoke',
    bucketCount: 32,
    validationKind: 'three-blocks-coalesced-exact-membership',
  });
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

    const historicalEvidence = await page.evaluate(
      () => window.__WEBGPU_BENCH__.startTrial({ historicalSmokeRun: true }),
    );
    assertTrialEvidence(historicalEvidence, {
      label: 'Three Blocks 0.10 timing smoke',
      bucketCount: 32,
      validationKind: 'three-blocks-historical-exact-membership',
    });
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
    const historicalLifecycle = await page.evaluate(async () => {
      document.querySelector('#strategy').value = 'draw-all';
      await window.__WEBGPU_BENCH__.rebuild();
      await window.__WEBGPU_BENCH__.validate();
      const baseline = window.__WEBGPU_BENCH__.cacheDiagnostics();
      const cycles = [];
      for (let cycle = 0; cycle < 3; cycle += 1) {
        document.querySelector('#strategy').value = 'three-blocks-historical';
        await window.__WEBGPU_BENCH__.rebuild();
        const validationResult = await window.__WEBGPU_BENCH__.validate();
        const active = window.__WEBGPU_BENCH__.cacheDiagnostics();
        document.querySelector('#strategy').value = 'draw-all';
        await window.__WEBGPU_BENCH__.rebuild();
        await window.__WEBGPU_BENCH__.validate();
        const disposed = window.__WEBGPU_BENCH__.cacheDiagnostics();
        cycles.push({ validation: validationResult, active, disposed });
      }
      return { baseline, cycles };
    });
    if (historicalLifecycle.baseline?.available !== true
      || historicalLifecycle.cycles.some((cycle) => (
        cycle.validation?.pass !== true
        || cycle.validation?.kind !== 'three-blocks-historical-exact-membership'
        || cycle.active?.computePipelineCacheEntries
          <= historicalLifecycle.baseline.computePipelineCacheEntries
        || cycle.disposed?.computePipelineCacheEntries
          !== historicalLifecycle.baseline.computePipelineCacheEntries
        || cycle.disposed?.computeProgramEntries
          !== historicalLifecycle.baseline.computeProgramEntries
        || !sameTrackedMemory(cycle.disposed?.memory, historicalLifecycle.baseline.memory)
      ))) {
      throw new Error(
        `Three Blocks 0.10 lifecycle/reconstruction failed: ${JSON.stringify(historicalLifecycle)}`,
      );
    }
    historical = {
      status: 'passed',
      validation: summarizeValidation(validation),
      timing,
      lifecycle: summarizeLifecycle(historicalLifecycle),
    };
  }
  if (errors.length) throw new Error(errors.join('\n'));

  const backend = await page.locator('#backend').textContent();
  process.stdout.write(`${JSON.stringify({
    browser: {
      executable: path.basename(executablePath),
      version: browser.version(),
    },
    backend,
    environment,
    colorScreenshotParity,
    fixedSlice128Edge,
    fixedSlice32Sustained,
    v11Lifecycle: summarizeLifecycle(v11Lifecycle),
    fixedSliceLifecycle: summarizeLifecycle(fixedSliceLifecycle),
    fourBucketControl: summarizeValidationMap(fourBucketControl),
    drawAll: summarizeValidation(drawAll),
    fixedSlice: summarizeValidation(fixedSlice),
    threeBlocksCurrent: summarizeValidation(threeBlocksCurrent),
    timing,
    heterogeneous: summarizeValidationMap(heterogeneous),
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
