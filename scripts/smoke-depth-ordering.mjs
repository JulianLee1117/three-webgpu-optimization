import { access, mkdir } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { createServer } from 'vite';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const failureDirectory = path.join(projectRoot, '.test-output');
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

const modes = Object.freeze([
  'fixed-slice',
  'fixed-slice-depth-front-to-back',
  'fixed-slice-depth-reverse',
]);
const layouts = Object.freeze(['high-overlap', 'low-overlap']);

function physicalBinSequenceSha256(validation) {
  return validation?.depthBins?.physicalBinSequenceCommitment?.sha256 ?? null;
}

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

function assertDepthValidation(mode, validation) {
  if (validation?.pass !== true
    || validation.membership?.pass !== true
    || validation.membershipDigests?.pass !== true
    || validation.commandValidation?.pass !== true
    || validation.overflow !== 0) {
    throw new Error(`${mode} exact validation failed: ${JSON.stringify({
      kind: validation?.kind,
      membership: validation?.membership,
      commands: validation?.commandValidation?.records,
      depthCounts: validation?.depthBins?.actualCounts,
      depthErrors: validation?.depthBins?.errors?.slice(0, 12),
      representation: validation?.representation,
    })}`);
  }
  if (mode === 'fixed-slice') {
    if (validation.kind !== 'fixed-slice-exact-membership'
      || validation.representation !== null) {
      throw new Error(`Atomic fixed-slice shape changed: ${JSON.stringify(validation)}`);
    }
    return;
  }
  const expectedOrder = mode.endsWith('front-to-back') ? 'front-to-back' : 'reverse';
  const expectedTraversal = expectedOrder === 'front-to-back'
    ? [0, 1, 2, 3, 4, 5, 6, 7]
    : [7, 6, 5, 4, 3, 2, 1, 0];
  if (validation.kind !== `${mode}-exact-membership-and-depth-order`
    || validation.depthBins?.pass !== true
    || validation.depthBins?.binCount !== 8
    || validation.depthBins?.order !== expectedOrder
    || JSON.stringify(validation.depthBins?.traversal) !== JSON.stringify(expectedTraversal)
    || validation.depthBins?.errors?.length !== 0
    || validation.depthBins?.physicalBinSequenceCommitment?.encoding
      !== 'bucket-major-physical-bin-major-tagged-uint32-little-endian'
    || validation.depthBins?.physicalBinSequenceCommitment?.bucketCount !== 4
    || validation.depthBins?.physicalBinSequenceCommitment?.binCount !== 8
    || validation.depthBins?.physicalBinSequenceCommitment?.recordCount !== 32
    || validation.depthBins?.physicalBinSequenceCommitment?.survivorCount
      !== validation.membership.expectedCount
    || !/^[0-9a-f]{64}$/.test(physicalBinSequenceSha256(validation) ?? '')
    || validation.representation?.kind
      !== 'single-merged-geometry-depth-binned-fixed-slice'
    || validation.representation?.bundleRecordCallbackCount !== 1
    || validation.representation?.meshCount !== 1
    || validation.representation?.geometryIdentityCount !== 1
    || validation.representation?.materialIdentityCount !== 1
    || validation.representation?.commandCount !== 4
    || validation.representation?.zeroFirstInstanceCount !== 4
    || validation.representation?.computeDispatchCount !== 4
    || JSON.stringify(validation.representation?.computeDispatchWorkItems)
      !== JSON.stringify([32, 4096, 4, 4])) {
    throw new Error(`${mode} depth-order validation failed: ${JSON.stringify(validation)}`);
  }
}

async function configureValidateAndCapture(page, mode, layout) {
  return page.evaluate(async ({ selectedMode, selectedLayout }) => {
    document.querySelector('#objects').value = '4096';
    document.querySelector('#buckets').value = '4';
    document.querySelector('#visibility').value = '0.99';
    document.querySelector('#layout').value = selectedLayout;
    document.querySelector('#strategy').value = selectedMode;
    await window.__WEBGPU_BENCH__.rebuild();
    const validation = await window.__WEBGPU_BENCH__.validate();
    if (validation?.pass !== true) {
      return { validation, postParityValidation: validation };
    }
    const parity = await window.__WEBGPU_BENCH__.captureRenderParity();
    const repeatedParity = await window.__WEBGPU_BENCH__.captureRenderParity();
    const postParityValidation = await window.__WEBGPU_BENCH__.validate();
    return {
      selectedConfig: window.__WEBGPU_BENCH__.selectedConfig(),
      validation,
      parity,
      repeatedParity,
      postParityValidation,
    };
  }, { selectedMode: mode, selectedLayout: layout });
}

async function waitForTrial(page, label) {
  await page.waitForFunction(
    () => ['complete', 'error'].includes(window.__WEBGPU_BENCH__?.phase),
    null,
    { timeout: 180_000 },
  );
  const result = await page.evaluate(() => ({
    phase: window.__WEBGPU_BENCH__.phase,
    error: window.__WEBGPU_BENCH__.trialError,
    rows: window.__WEBGPU_BENCH__.rows,
    summary: JSON.parse(document.querySelector('#details').textContent),
  }));
  if (result.phase !== 'complete' || result.summary?.accepted !== true) {
    throw new Error(`${label} timing failed: ${JSON.stringify(result)}`);
  }
  return result;
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
  const environment = await page.evaluate(() => window.__WEBGPU_BENCH__.environment);
  if (environment?.rendererReversedDepthBuffer !== true
    || !Number.isInteger(environment?.maxStorageBuffersPerShaderStage)
    || environment.maxStorageBuffersPerShaderStage < 8) {
    throw new Error(`Depth smoke environment rejected: ${JSON.stringify(environment)}`);
  }

  const cells = [];
  for (const layout of layouts) {
    let reference = null;
    let orderedSequenceReference = null;
    for (const mode of modes) {
      const evidence = await configureValidateAndCapture(page, mode, layout);
      if (errors.length > 0) {
        throw new Error(`Browser errors before ${mode}/${layout} validation: ${errors.join('\n')}`);
      }
      assertDepthValidation(mode, evidence.validation);
      assertDepthValidation(mode, evidence.parity?.snapshotValidation);
      assertDepthValidation(mode, evidence.repeatedParity?.snapshotValidation);
      assertDepthValidation(mode, evidence.postParityValidation);
      let physicalBinSequence = null;
      if (mode !== 'fixed-slice') {
        const sequenceSnapshots = [
          evidence.validation,
          evidence.parity.snapshotValidation,
          evidence.repeatedParity.snapshotValidation,
          evidence.postParityValidation,
        ].map(physicalBinSequenceSha256);
        if (new Set(sequenceSnapshots).size !== 1) {
          throw new Error(`${mode}/${layout} physical-bin sequence changed across snapshots.`);
        }
        [physicalBinSequence] = sequenceSnapshots;
        if (orderedSequenceReference === null) {
          orderedSequenceReference = physicalBinSequence;
        } else if (physicalBinSequence !== orderedSequenceReference) {
          throw new Error(`${layout} front/reverse physical-bin sequences differ.`);
        }
      }
      const parityCaptures = [evidence.parity, evidence.repeatedParity];
      if (parityCaptures.some((parity) => (
        parity?.pass !== true
        || parity?.stability?.pass !== true
        || parity?.objectIdValidation?.pass !== true
        || parity?.objectIdValidation?.coveredPixels <= 0
        || parity?.objectIdValidation?.backgroundPixels
          + parity?.objectIdValidation?.coveredPixels !== 1280 * 720
        || parity?.objectIdValidation?.outOfRangePixels !== 0
        || parity?.objectIdValidation?.nonVisiblePixels !== 0
        || parity?.reversedDepthBuffer !== true
        || parity?.width !== 1280
        || parity?.height !== 720
        || ['color', 'depth', 'objectId'].some(
          (channel) => parity?.[channel]?.byteLength !== 1280 * 720 * 4,
        )
      ))) {
        throw new Error(`${mode}/${layout} parity capture failed: ${JSON.stringify(evidence.parity)}`);
      }
      const identity = {
        membership: evidence.validation.membershipDigests.expected.sha256,
        material: evidence.parity.material,
        color: evidence.parity.color.sha256,
        depth: evidence.parity.depth.sha256,
        objectId: evidence.parity.objectId.sha256,
      };
      const repeatedIdentity = {
        membership: evidence.repeatedParity.snapshotValidation.membershipDigests.expected.sha256,
        material: evidence.repeatedParity.material,
        color: evidence.repeatedParity.color.sha256,
        depth: evidence.repeatedParity.depth.sha256,
        objectId: evidence.repeatedParity.objectId.sha256,
      };
      if (JSON.stringify(repeatedIdentity) !== JSON.stringify(identity)) {
        throw new Error(`${mode}/${layout} recompute parity failed: ${JSON.stringify({
          identity,
          repeatedIdentity,
        })}`);
      }
      const crossModeMatch = reference === null
        || JSON.stringify(identity) === JSON.stringify(reference);
      if (reference === null) reference = identity;
      else if (!crossModeMatch) {
        throw new Error(`${layout} cross-mode render parity failed: ${JSON.stringify({
          reference,
          identity,
        })}`);
      }
      cells.push({
        layout,
        mode,
        validation: evidence.validation.kind,
        physicalBinSequence,
        parity: identity,
        crossModeMatch,
      });
    }
  }

  const timing = [];
  for (const mode of modes) {
    await configureValidateAndCapture(page, mode, 'high-overlap');
    await page.evaluate(async () => {
      await window.__WEBGPU_BENCH__.startTrial({ depthOrderingSmoke: true });
    });
    const result = await waitForTrial(page, mode);
    const expectedDispatches = mode === 'fixed-slice' ? 2 : 4;
    const expectedInvariantKind = mode === 'fixed-slice'
      ? 'atomic-fixed-slice-static-bundle-invariant'
      : 'depth-binned-static-bundle-invariant';
    if (result.rows.length !== 240
      || result.rows.some((row) => (
        row.scenarioLayout !== 'high-overlap'
        || row.configuredComputeDispatches !== expectedDispatches
        || row.configuredComputeSubmissions !== 1
      ))
      || result.summary.completionInvariant?.kind !== expectedInvariantKind
      || result.summary.completionInvariant?.bundleRecordCallbackCountAtTimingStart !== 1
      || result.summary.completionInvariant?.bundleRecordCallbackCountAtTimingEnd !== 1) {
      throw new Error(`${mode} timing invariant failed: ${JSON.stringify(result.summary)}`);
    }
    timing.push({ mode, completionInvariant: result.summary.completionInvariant });
  }
  if (errors.length > 0) throw new Error(errors.join('\n'));

  process.stdout.write(`${JSON.stringify({
    browser: { executable: path.basename(executablePath), version: browser.version() },
    backend: await page.locator('#backend').textContent(),
    environment: {
      rendererReversedDepthBuffer: environment.rendererReversedDepthBuffer,
      maxStorageBuffersPerShaderStage: environment.maxStorageBuffersPerShaderStage,
    },
    cells,
    timing,
  }, null, 2)}\n`);
} catch (error) {
  if (page) {
    await page.screenshot({
      path: path.join(failureDirectory, 'depth-ordering-smoke-failure.png'),
      fullPage: true,
    });
  }
  throw error;
} finally {
  await browser?.close();
  await server.close();
}
