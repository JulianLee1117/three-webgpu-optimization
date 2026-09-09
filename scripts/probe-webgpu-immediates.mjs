import { createReadStream } from 'node:fs';
import {
  access,
  chmod,
  mkdir,
  readFile,
  stat,
  writeFile,
} from 'node:fs/promises';
import { createHash, randomBytes } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

import { chromium } from 'playwright-core';
import { createServer } from 'vite';

import { validateImmediateCanaryResult } from '../src/immediate-canary.js';
import {
  collectSourceProvenance,
  sourceProvenanceMatches,
} from './source-provenance.mjs';

export const WEBGPU_IMMEDIATES_PROBE_KIND =
  'raw-webgpu-immediates-correctness-probe';
export const WEBGPU_IMMEDIATES_DEFAULT_PROBE_KIND =
  'raw-webgpu-immediates-default-exposure-correctness-probe';
export const WEBGPU_IMMEDIATES_EXPOSURE_MODES = Object.freeze({
  DEVELOPER_ENABLED: 'developer-enabled',
  DEFAULT_WEB_PLATFORM: 'default-web-platform',
});
export const WEBGPU_IMMEDIATES_BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
]);
export const WEBGPU_IMMEDIATES_DEFAULT_BROWSER_ARGS = Object.freeze([]);

export function rawWebGpuImmediatesArtifactModeIsReadOnly(metadata) {
  return Number.isInteger(metadata?.mode) && (metadata.mode & 0o222) === 0;
}

export const FORBIDDEN_DEFAULT_EXPOSURE_ARGUMENT_PREFIXES = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--enable-dawn-features',
  '--disable-dawn-features',
  '--use-webgpu-adapter',
  '--force-webgpu-compat',
  '--ignore-gpu-blocklist',
  '--force-high-performance-gpu',
]);

const BROWSER_OPERATION_TIMEOUT_MS = 120_000;
const PAGE_READINESS_TIMEOUT_MS = 120_000;
const SERVER_OPERATION_TIMEOUT_MS = 120_000;
const MAXIMUM_DIAGNOSTIC_EVENTS = 64;
const SHA256_PATTERN = /^[a-f0-9]{64}$/u;
const ISOLATION_HEADERS = Object.freeze({
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
});
const RUNTIME_SOURCE_FILES = Object.freeze([
  'immediate-canary.html',
  'src/immediate-canary.js',
  'scripts/probe-webgpu-immediates.mjs',
  'package.json',
  'package-lock.json',
]);
const PROBE_CONFIGURATIONS = Object.freeze({
  [WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEVELOPER_ENABLED]: Object.freeze({
    exposureMode: WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEVELOPER_ENABLED,
    kind: WEBGPU_IMMEDIATES_PROBE_KIND,
    browserArguments: WEBGPU_IMMEDIATES_BROWSER_ARGS,
    resultDirectoryName: 'webgpu-immediates-canary',
    requireEffectiveArguments: false,
  }),
  [WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEFAULT_WEB_PLATFORM]: Object.freeze({
    exposureMode: WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEFAULT_WEB_PLATFORM,
    kind: WEBGPU_IMMEDIATES_DEFAULT_PROBE_KIND,
    browserArguments: WEBGPU_IMMEDIATES_DEFAULT_BROWSER_ARGS,
    resultDirectoryName: 'webgpu-immediates-default-exposure-canary',
    requireEffectiveArguments: true,
  }),
});

export function findForbiddenDefaultExposureArguments(arguments_) {
  if (!Array.isArray(arguments_)) return ['<effective-arguments-unavailable>'];
  return arguments_.filter((argument) => {
    if (typeof argument !== 'string') return true;
    const normalized = argument.toLowerCase();
    return FORBIDDEN_DEFAULT_EXPOSURE_ARGUMENT_PREFIXES.some(
      (prefix) => normalized === prefix || normalized.startsWith(`${prefix}=`),
    );
  });
}

export function assessDefaultExposureBrowserRecord(browserRecord) {
  const reasons = [];
  if (browserRecord?.exposureMode
    !== WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEFAULT_WEB_PLATFORM) {
    reasons.push('browser exposure mode is not default-web-platform');
  }
  if (!Array.isArray(browserRecord?.launchArguments)
    || browserRecord.launchArguments.length !== 0) {
    reasons.push('default-exposure custom launch arguments are not empty');
  }
  const capture = browserRecord?.effectiveArgumentsCapture;
  if (capture?.api !== 'Browser.getBrowserCommandLine'
    || capture?.available !== true
    || !Array.isArray(capture?.arguments)) {
    reasons.push('effective browser arguments were not captured');
  } else {
    const forbidden = findForbiddenDefaultExposureArguments(capture.arguments);
    if (!sameStringArrays(capture.forbiddenArguments, forbidden)
      || forbidden.length !== 0) {
      reasons.push('effective browser arguments contain a capability-expanding switch');
    }
  }
  return deepFreeze({ pass: reasons.length === 0, reasons });
}

function clone(value) {
  if (value === undefined) return undefined;
  return structuredClone(value);
}

function deepFreeze(value) {
  if (value === null || typeof value !== 'object' || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}

function sameStringArrays(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function requireCondition(condition, message, detail = undefined) {
  if (condition) return;
  const error = new Error(message);
  if (detail !== undefined) error.detail = clone(detail);
  throw error;
}

function boundedValue(value, maximumJsonBytes = 65_536) {
  if (value === undefined) return null;
  let json;
  try {
    json = JSON.stringify(value);
  } catch {
    return { serializationFailed: true };
  }
  const bytes = Buffer.from(json, 'utf8');
  if (bytes.length <= maximumJsonBytes) return clone(value);
  return {
    truncated: true,
    jsonByteLength: bytes.length,
    jsonSha256: createHash('sha256').update(bytes).digest('hex'),
    jsonPrefix: json.slice(0, 16_384),
  };
}

function serializeError(error) {
  return {
    name: String(error?.name ?? 'Error').slice(0, 256),
    message: String(error?.message ?? error).slice(0, 8_192),
    stack: error?.stack === undefined || error?.stack === null
      ? null
      : String(error.stack).slice(0, 32_768),
    detail: boundedValue(error?.detail),
    code: error?.code === undefined || error?.code === null
      ? null
      : String(error.code).slice(0, 512),
  };
}

function withDeadline(promise, timeoutMs, label) {
  let timeout;
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      timeout = setTimeout(() => reject(new Error(`${label} exceeded its deadline.`)), timeoutMs);
    }),
  ]).finally(() => clearTimeout(timeout));
}

async function sha256FileStream(filename) {
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(filename)) digest.update(chunk);
  return digest.digest('hex');
}

async function fileIdentity(filename, displayedPath = filename) {
  const fileStat = await stat(filename);
  requireCondition(fileStat.isFile(), `Identity target is not a file: ${displayedPath}`);
  return {
    path: displayedPath.replaceAll('\\', '/'),
    byteLength: fileStat.size,
    sha256: await sha256FileStream(filename),
  };
}

async function collectRuntimeSourceIdentity(projectRoot) {
  const files = [];
  for (const relativePath of RUNTIME_SOURCE_FILES) {
    files.push(await fileIdentity(path.join(projectRoot, relativePath), relativePath));
  }
  const aggregate = createHash('sha256');
  aggregate.update('webgpu-immediates-runtime-source-v1\0');
  for (const file of files) {
    aggregate.update(`${file.path}\0${file.byteLength}\0${file.sha256}\0`);
  }
  return {
    schemaVersion: 1,
    files,
    aggregateSha256: aggregate.digest('hex'),
  };
}

function runtimeSourceIdentityMatches(left, right) {
  return left?.schemaVersion === 1
    && right?.schemaVersion === 1
    && SHA256_PATTERN.test(left.aggregateSha256)
    && left.aggregateSha256 === right.aggregateSha256
    && JSON.stringify(left.files) === JSON.stringify(right.files);
}

function browserIdentityMatches(left, right) {
  return left?.path === right?.path
    && left?.byteLength === right?.byteLength
    && SHA256_PATTERN.test(left?.sha256 ?? '')
    && left.sha256 === right?.sha256;
}

export async function findWebGpuImmediatesBrowser() {
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
      // Continue through the installed-system-browser locations used by project runners.
    }
  }
  throw new Error('No installed Chrome, Chromium, or Edge executable was found.');
}

function probeConfiguration(exposureMode) {
  const configuration = PROBE_CONFIGURATIONS[exposureMode];
  requireCondition(configuration !== undefined,
    `Unknown WebGPU immediate-data exposure mode: ${exposureMode}`);
  return configuration;
}

async function captureEffectiveBrowserArguments(browser) {
  const session = await withDeadline(
    browser.newBrowserCDPSession(),
    BROWSER_OPERATION_TIMEOUT_MS,
    'browser CDP session creation',
  );
  try {
    const response = await withDeadline(
      session.send('Browser.getBrowserCommandLine'),
      BROWSER_OPERATION_TIMEOUT_MS,
      'effective browser command-line capture',
    );
    requireCondition(Array.isArray(response?.arguments)
      && response.arguments.every((argument) => typeof argument === 'string'),
    'Browser.getBrowserCommandLine did not return a string argument array.', response);
    const arguments_ = [...response.arguments];
    return {
      api: 'Browser.getBrowserCommandLine',
      available: true,
      arguments: arguments_,
      forbiddenArguments: findForbiddenDefaultExposureArguments(arguments_),
    };
  } finally {
    await withDeadline(
      session.detach(),
      BROWSER_OPERATION_TIMEOUT_MS,
      'browser CDP session detach',
    );
  }
}

function createPageDiagnostics() {
  const state = {
    events: [],
    overflowCount: 0,
    shutdownStarted: false,
  };
  const append = (event) => {
    if (state.shutdownStarted) return;
    if (state.events.length < MAXIMUM_DIAGNOSTIC_EVENTS) {
      state.events.push(boundedValue(event, 16_384));
    } else {
      state.overflowCount += 1;
    }
  };
  return { state, append };
}

function attachPageDiagnostics(page, append) {
  page.on('pageerror', (error) => append({
    type: 'pageerror',
    error: serializeError(error),
  }));
  page.on('console', (message) => {
    if (message.type() !== 'error') return;
    append({
      type: 'console-error',
      text: message.text().slice(0, 8_192),
    });
  });
  page.on('requestfailed', (request) => append({
    type: 'request-failed',
    method: request.method(),
    resourceType: request.resourceType(),
    url: request.url().slice(0, 4_096),
    failure: boundedValue(request.failure()),
  }));
  page.on('response', (response) => {
    if (response.status() < 400) return;
    append({
      type: 'http-error',
      status: response.status(),
      url: response.url().slice(0, 4_096),
    });
  });
  page.on('crash', () => append({ type: 'page-crash' }));
}

function pageDiagnosticsPass(diagnostics) {
  return Array.isArray(diagnostics?.events)
    && diagnostics.events.length === 0
    && diagnostics?.overflowCount === 0;
}

function hasNoPerformanceEvidenceBoundary(record) {
  return record?.executionMode === 'technical-canary'
    && record?.phase === 'phase-0-raw-capability-canary'
    && record?.fullPhase0Pass === false
    && record?.integratedCanaryInvoked === false
    && record?.analysisEligible === false
    && record?.efficacyAnalysisAllowed === false
    && record?.numericalDecision === null
    && record?.timingCaptured === false
    && record?.efficacyEvaluated === false;
}

export function validateWebGpuImmediatesProbeReport(report) {
  const reasons = [];
  const configuration = report?.kind === WEBGPU_IMMEDIATES_PROBE_KIND
    ? PROBE_CONFIGURATIONS[WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEVELOPER_ENABLED]
    : report?.kind === WEBGPU_IMMEDIATES_DEFAULT_PROBE_KIND
      ? PROBE_CONFIGURATIONS[WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEFAULT_WEB_PLATFORM]
      : null;
  if (report?.schemaVersion !== 1 || configuration === null) {
    reasons.push('probe report identity is invalid');
  }
  if (configuration?.exposureMode
    === WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEFAULT_WEB_PLATFORM) {
    if (report?.exposureMode !== configuration.exposureMode
      || report?.executionPolicy?.developerFlagFallbackAllowed !== false) {
      reasons.push('default-exposure execution policy is invalid');
    }
  } else if (report?.exposureMode !== undefined
    && report.exposureMode !== WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEVELOPER_ENABLED) {
    reasons.push('developer-enabled exposure mode is invalid');
  }
  if (report?.status !== 'pass') reasons.push('probe did not report pass');
  if (!hasNoPerformanceEvidenceBoundary(report)) {
    reasons.push('correctness-only evidence boundary is invalid');
  }
  if (report?.executionPolicy?.attemptCount !== 1
    || report.executionPolicy.retryCount !== 0
    || report.executionPolicy.replacementAllowed !== false
    || report.executionPolicy.trialExecutionInvoked !== false
    || report.executionPolicy.timingCaptured !== false
    || report.executionPolicy.efficacyEvaluated !== false) {
    reasons.push('one-shot execution policy is invalid');
  }
  if (report?.browser?.launchApi !== 'playwright-core.chromium.launch'
    || report.browser.persistentContext !== false
    || report.browser.profilePolicy !== 'fresh-playwright-temporary-profile'
    || !Array.isArray(report.browser.launchArguments)
    || !sameStringArrays(
      report.browser.launchArguments,
      configuration?.browserArguments ?? [],
    )
    || typeof report.browser.browserVersion !== 'string'
    || report.browser.browserVersion.length === 0
    || report.browser.identityMatchesStart !== true
    || !browserIdentityMatches(report.browser.executableStart, report.browser.executableEnd)) {
    reasons.push('exact browser execution identity is invalid or unstable');
  }
  if (configuration?.requireEffectiveArguments === true) {
    const exposureAssessment = assessDefaultExposureBrowserRecord(report?.browser);
    if (!exposureAssessment.pass) reasons.push(...exposureAssessment.reasons);
  }
  if (report?.sourceIdentity?.provenanceMatchesStart !== true
    || !sourceProvenanceMatches(
      report?.sourceIdentity?.provenanceStart,
      report?.sourceIdentity?.provenanceEnd,
    )
    || report?.sourceIdentity?.runtimeFilesMatchStart !== true
    || !runtimeSourceIdentityMatches(
      report?.sourceIdentity?.runtimeFilesStart,
      report?.sourceIdentity?.runtimeFilesEnd,
    )) {
    reasons.push('source identity is unavailable or changed during the probe');
  }
  const pageValidation = validateImmediateCanaryResult(report?.pageResult);
  if (!pageValidation.pass) {
    reasons.push(...pageValidation.reasons.map((reason) => `page result: ${reason}`));
  }
  if (!pageDiagnosticsPass(report?.pageDiagnostics)) {
    reasons.push('page emitted an error diagnostic');
  }
  if (report?.browser?.singleContext !== true
    || report?.browser?.singlePage !== true
    || report?.browser?.closedCleanly !== true) {
    reasons.push('browser lifecycle did not retain and close one context/page');
  }
  if (report?.server?.localOnly !== true || report?.server?.closedCleanly !== true) {
    reasons.push('local Vite lifecycle is invalid');
  }
  if (report?.failure !== null) reasons.push('probe retained a failure');
  return deepFreeze({
    pass: reasons.length === 0,
    reasons: [...new Set(reasons)],
  });
}

async function closeBrowserState({ browser, context, page, diagnostics, browserRecord }) {
  if (browser === null) return;
  diagnostics.state.shutdownStarted = true;
  if (context !== null) {
    await withDeadline(context.close(), BROWSER_OPERATION_TIMEOUT_MS, 'browser context close');
  }
  await withDeadline(browser.close(), BROWSER_OPERATION_TIMEOUT_MS, 'browser close');
  browserRecord.closedCleanly = (page === null || page.isClosed())
    && browser.isConnected() === false;
}

async function writeExclusiveReport(projectRoot, report, resultDirectoryName) {
  const resultParent = path.join(
    projectRoot,
    'results',
    'development',
    resultDirectoryName,
  );
  await mkdir(resultParent, { recursive: true });
  const runId = `canary-${randomBytes(12).toString('hex')}`;
  const runDirectory = path.join(resultParent, runId);
  await mkdir(runDirectory, { recursive: false });
  const reportPath = path.join(runDirectory, 'report.json');
  const bytes = Buffer.from(`${JSON.stringify(report)}\n`, 'utf8');
  await writeFile(reportPath, bytes, { flag: 'wx' });
  const writtenBytes = await readFile(reportPath);
  if (!writtenBytes.equals(bytes)) {
    throw new Error('Just-written raw immediate-data report failed its reread.');
  }
  await chmod(reportPath, 0o444);
  const [immutableBytes, metadata] = await Promise.all([
    readFile(reportPath),
    stat(reportPath),
  ]);
  if (!immutableBytes.equals(bytes)
    || !rawWebGpuImmediatesArtifactModeIsReadOnly(metadata)) {
    throw new Error('Just-written raw immediate-data report is not exact and read-only.');
  }
  return {
    reportPath,
    relativePath: path.relative(projectRoot, reportPath).replaceAll('\\', '/'),
    byteLength: bytes.length,
    sha256: createHash('sha256').update(bytes).digest('hex'),
  };
}

export async function runWebGpuImmediatesProbe({
  exposureMode = WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEVELOPER_ENABLED,
  cacheDirectory = null,
  emitSummary = true,
  setProcessExitCode = true,
} = {}) {
  const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const resolvedCacheDirectory = cacheDirectory === null
    ? path.join(projectRoot, 'node_modules', '.vite') : path.resolve(cacheDirectory);
  const configuration = probeConfiguration(exposureMode);
  let browser = null;
  let context = null;
  let page = null;
  let server = null;
  let executablePath = null;
  let executableStart = null;
  let executableEnd = null;
  let sourceProvenanceStart = null;
  let sourceProvenanceEnd = null;
  let runtimeFilesStart = null;
  let runtimeFilesEnd = null;
  let pageResult = null;
  let failure = null;
  let browserVersion = null;
  const diagnostics = createPageDiagnostics();
  const browserRecord = {
    launchApi: 'playwright-core.chromium.launch',
    headless: true,
    persistentContext: false,
    profilePolicy: 'fresh-playwright-temporary-profile',
    exposureMode: configuration.exposureMode,
    launchArguments: [...configuration.browserArguments],
    effectiveArgumentsCapture: null,
    browserVersion: null,
    executableStart: null,
    executableEnd: null,
    identityMatchesStart: false,
    singleContext: false,
    singlePage: false,
    closedCleanly: false,
  };
  const serverRecord = {
    implementation: 'vite.createServer',
    configFile: false,
    host: '127.0.0.1',
    dynamicPort: true,
    localOnly: true,
    isolationHeaders: { ...ISOLATION_HEADERS },
    cacheDirectory: path.relative(projectRoot, resolvedCacheDirectory).replaceAll('\\', '/'),
    closedCleanly: false,
  };

  try {
    [sourceProvenanceStart, runtimeFilesStart] = await Promise.all([
      collectSourceProvenance(projectRoot),
      collectRuntimeSourceIdentity(projectRoot),
    ]);
    requireCondition(sourceProvenanceStart?.status === 'available'
      && sourceProvenanceStart.captureStable === true,
    'Stable source provenance is required.', sourceProvenanceStart);

    executablePath = await findWebGpuImmediatesBrowser();
    executableStart = await fileIdentity(executablePath, executablePath);
    browserRecord.executableStart = executableStart;

    server = await createServer({
      root: projectRoot,
      configFile: false,
      appType: 'mpa',
      cacheDir: resolvedCacheDirectory,
      server: {
        host: '127.0.0.1',
        port: 0,
        headers: ISOLATION_HEADERS,
      },
      optimizeDeps: {
        noDiscovery: true,
        include: [],
      },
      logLevel: 'error',
    });
    await withDeadline(server.listen(), SERVER_OPERATION_TIMEOUT_MS, 'Vite server listen');
    const baseUrl = server.resolvedUrls?.local?.[0];
    requireCondition(typeof baseUrl === 'string'
      && baseUrl.startsWith('http://127.0.0.1:'),
    'Vite did not expose the expected loopback URL.', { baseUrl });

    browser = await chromium.launch({
      executablePath,
      headless: true,
      args: configuration.browserArguments,
      timeout: BROWSER_OPERATION_TIMEOUT_MS,
    });
    browserVersion = browser.version();
    browserRecord.browserVersion = browserVersion;
    if (configuration.requireEffectiveArguments) {
      browserRecord.effectiveArgumentsCapture =
        await captureEffectiveBrowserArguments(browser);
      const exposureAssessment = assessDefaultExposureBrowserRecord(browserRecord);
      requireCondition(exposureAssessment.pass,
        'Default-exposure browser command line is not clean.', exposureAssessment);
    }
    requireCondition(browser.contexts().length === 0,
      'Fresh browser unexpectedly began with a context.');
    context = await withDeadline(browser.newContext({
      viewport: { width: 1280, height: 720 },
      deviceScaleFactor: 1,
    }), BROWSER_OPERATION_TIMEOUT_MS, 'browser context creation');
    browserRecord.singleContext = browser.contexts().length === 1;
    requireCondition(browserRecord.singleContext,
      'Browser did not retain exactly one context.');
    page = await withDeadline(context.newPage(), BROWSER_OPERATION_TIMEOUT_MS,
      'page creation');
    browserRecord.singlePage = context.pages().length === 1;
    requireCondition(browserRecord.singlePage, 'Context did not retain exactly one page.');
    attachPageDiagnostics(page, diagnostics.append);
    await withDeadline(page.goto(`${baseUrl}immediate-canary.html`, {
      waitUntil: 'domcontentloaded',
    }), BROWSER_OPERATION_TIMEOUT_MS, 'canary navigation');
    await withDeadline(page.waitForFunction(
      () => window.__WEBGPU_IMMEDIATES_CANARY__?.ready === true,
      null,
      { timeout: PAGE_READINESS_TIMEOUT_MS },
    ), BROWSER_OPERATION_TIMEOUT_MS, 'canary readiness');
    pageResult = await withDeadline(page.evaluate(
      () => structuredClone(window.__WEBGPU_IMMEDIATES_CANARY__.result),
    ), BROWSER_OPERATION_TIMEOUT_MS, 'canary result capture');
    const pageValidation = validateImmediateCanaryResult(pageResult);
    requireCondition(pageValidation.pass, 'Page canary failed closed.', pageValidation);
    requireCondition(pageDiagnosticsPass(diagnostics.state),
      'Page emitted an error diagnostic.', diagnostics.state);

    await closeBrowserState({
      browser,
      context,
      page,
      diagnostics,
      browserRecord,
    });
    browser = null;
    context = null;
    page = null;
    requireCondition(browserRecord.closedCleanly, 'Browser did not close cleanly.');

    await withDeadline(server.close(), SERVER_OPERATION_TIMEOUT_MS, 'Vite server close');
    server = null;
    serverRecord.closedCleanly = true;

    [sourceProvenanceEnd, runtimeFilesEnd, executableEnd] = await Promise.all([
      collectSourceProvenance(projectRoot),
      collectRuntimeSourceIdentity(projectRoot),
      fileIdentity(executablePath, executablePath),
    ]);
    browserRecord.executableEnd = executableEnd;
    browserRecord.identityMatchesStart = browserIdentityMatches(
      executableStart,
      executableEnd,
    );
    requireCondition(browserRecord.identityMatchesStart,
      'Browser executable changed during the canary.');
    requireCondition(sourceProvenanceMatches(sourceProvenanceStart, sourceProvenanceEnd),
      'Source provenance changed during the canary.');
    requireCondition(runtimeSourceIdentityMatches(runtimeFilesStart, runtimeFilesEnd),
      'Runtime source bytes changed during the canary.');
  } catch (error) {
    failure = serializeError(error);
  } finally {
    if (browser !== null) {
      await closeBrowserState({
        browser,
        context,
        page,
        diagnostics,
        browserRecord,
      }).catch((error) => {
        browserRecord.cleanupFailure = serializeError(error);
      });
    }
    if (server !== null) {
      await withDeadline(server.close(), SERVER_OPERATION_TIMEOUT_MS, 'Vite cleanup')
        .then(() => {
          serverRecord.closedCleanly = true;
        })
        .catch((error) => {
          serverRecord.cleanupFailure = serializeError(error);
        });
    }
    if (runtimeFilesEnd === null) {
      runtimeFilesEnd = await collectRuntimeSourceIdentity(projectRoot).catch(() => null);
    }
    if (sourceProvenanceEnd === null) {
      sourceProvenanceEnd = await collectSourceProvenance(projectRoot, {
        allowUnavailable: true,
      }).catch(() => null);
    }
    if (executablePath !== null && executableEnd === null) {
      executableEnd = await fileIdentity(executablePath, executablePath).catch(() => null);
      browserRecord.executableEnd = executableEnd;
    }
    browserRecord.identityMatchesStart = browserIdentityMatches(
      executableStart,
      executableEnd,
    );
  }

  const sourceIdentity = {
    provenanceStart: sourceProvenanceStart,
    provenanceEnd: sourceProvenanceEnd,
    provenanceMatchesStart: sourceProvenanceMatches(
      sourceProvenanceStart,
      sourceProvenanceEnd,
    ),
    runtimeFilesStart,
    runtimeFilesEnd,
    runtimeFilesMatchStart: runtimeSourceIdentityMatches(
      runtimeFilesStart,
      runtimeFilesEnd,
    ),
  };
  let report = {
    schemaVersion: 1,
    kind: configuration.kind,
    status: failure === null ? 'pass' : 'fail',
    exposureMode: configuration.exposureMode,
    scope: 'raw WebGPU immediate-data capability and correctness only',
    executionMode: 'technical-canary',
    phase: 'phase-0-raw-capability-canary',
    fullPhase0Pass: false,
    integratedCanaryInvoked: false,
    analysisEligible: false,
    efficacyAnalysisAllowed: false,
    numericalDecision: null,
    timingCaptured: false,
    efficacyEvaluated: false,
    executionPolicy: {
      attemptCount: 1,
      retryCount: 0,
      replacementAllowed: false,
      developerFlagFallbackAllowed: false,
      trialExecutionInvoked: false,
      timingCaptured: false,
      efficacyEvaluated: false,
    },
    browser: browserRecord,
    server: serverRecord,
    sourceIdentity,
    pageDiagnostics: {
      events: diagnostics.state.events,
      overflowCount: diagnostics.state.overflowCount,
    },
    pageResult,
    failure,
  };
  if (failure === null) {
    const validation = validateWebGpuImmediatesProbeReport(report);
    if (!validation.pass) {
      report.status = 'fail';
      report.failure = {
        name: 'ProbeValidationError',
        message: 'Completed correctness probe failed its independent validator.',
        stack: null,
        detail: validation,
        code: null,
      };
    }
  }
  report = deepFreeze(report);
  const artifact = await writeExclusiveReport(
    projectRoot,
    report,
    configuration.resultDirectoryName,
  );
  if (emitSummary) {
    process.stdout.write(`${JSON.stringify({
      status: report.status,
      artifact,
      browserVersion,
    })}\n`);
  }
  if (setProcessExitCode && report.status !== 'pass') process.exitCode = 1;
  return { report, artifact };
}

export function parseWebGpuImmediatesProbeArguments(arguments_) {
  if (arguments_.length === 0) {
    return {
      exposureMode: WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEVELOPER_ENABLED,
    };
  }
  requireCondition(arguments_.length === 1 && arguments_[0] === '--default-exposure',
    'probe-webgpu-immediates accepts only --default-exposure.');
  return {
    exposureMode: WEBGPU_IMMEDIATES_EXPOSURE_MODES.DEFAULT_WEB_PLATFORM,
  };
}

const invokedPath = process.argv[1] === undefined
  ? null
  : pathToFileURL(path.resolve(process.argv[1])).href;
if (invokedPath === import.meta.url) {
  try {
    const options = parseWebGpuImmediatesProbeArguments(process.argv.slice(2));
    await runWebGpuImmediatesProbe(options);
  } catch (error) {
    process.stderr.write(`${error?.stack ?? error}\n`);
    process.exitCode ||= 1;
  }
}
