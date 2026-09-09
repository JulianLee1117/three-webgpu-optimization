import {
  access,
  mkdir,
  stat,
  writeFile,
} from 'node:fs/promises';
import { writeFileSync } from 'node:fs';
import { createHash, randomBytes } from 'node:crypto';
import { execFile } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { promisify } from 'node:util';
import { setTimeout as delay } from 'node:timers/promises';

import { chromium } from 'playwright-core';
import { createServer } from 'vite';

import {
  buildFirstInstanceStandaloneDeploymentPlan,
  validateFirstInstanceStandaloneDeploymentPlan,
} from '../src/benchmark/first-instance-standalone-deployment-plan.js';
import {
  firstInstanceStandaloneBootSearch,
} from '../src/benchmark/first-instance-standalone-boot.js';
import {
  THREE_REVISION,
  VIEWPORT,
} from '../src/config.js';
import {
  STANDALONE_MODE_ID,
  STANDALONE_POST_DISCONNECT_DELAY_MS,
  validateStandaloneBrowserLifecycleChain,
} from './run-first-instance-standalone-deployment.mjs';
import {
  collectSourceProvenance,
  sourceProvenanceMatches,
} from './source-provenance.mjs';
import {
  collectExecutionDependencyClosure,
  createCandidateViteRuntimeGuard,
  executionDependencyClosuresMatch,
} from './execution-dependency-closure.mjs';
import {
  validateLiveFirstInstanceForcedFeatureOffGate,
} from './live-first-instance-evidence-validation.mjs';

const execFileAsync = promisify(execFile);

export const STARTUP_ENDURANCE_PROBE_KIND =
  'first-instance-standalone-startup-endurance-probe';
export const STARTUP_ENDURANCE_EXECUTION_MODE = 'startup-endurance-excluded';
export const STARTUP_ENDURANCE_QUARTET_COUNT = 8;
export const STARTUP_ENDURANCE_SESSION_COUNT = 32;
export const STARTUP_ENDURANCE_BROWSER_COUNT = STARTUP_ENDURANCE_SESSION_COUNT + 1;
export const STARTUP_ENDURANCE_TCP_QUIET_WINDOW_MS = 10 * 60 * 1_000;
export const STARTUP_ENDURANCE_MINIMUM_FREE_DYNAMIC_PORTS = 4_096;
export const STARTUP_ENDURANCE_MINIMUM_FREE_DYNAMIC_PORT_FRACTION = 0.25;
// BROWSER_ARGS is intentionally private in the frozen candidate runner. Keep
// this exact local copy visible and unit-tested instead of changing that runner.
export const STARTUP_ENDURANCE_BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
]);

const BROWSER_OPERATION_TIMEOUT_MS = 180_000;
const BENCHMARK_READINESS_TIMEOUT_MS = 120_000;
const BROWSER_CLOSE_TIMEOUT_MS = 60_000;
const SERVER_OPERATION_TIMEOUT_MS = 120_000;
const TCP_QUERY_TIMEOUT_MS = 60_000;
const MAXIMUM_PAGE_DIAGNOSTIC_EVENTS = 128;
const OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;
const LAYOUT = 'baseline';
const SCENARIO_SEED = 0xb1ad_2026;
const ISOLATION_HEADERS = Object.freeze({
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
});
const EXPECTED_QUARTET_CODES = Object.freeze([
  'AX', 'AY', 'BX', 'BY', 'AX', 'AY', 'BX', 'BY',
]);
const SHA256_PATTERN = /^[a-f0-9]{64}$/u;
const WINDOWS_TCP_SNAPSHOT_COMMAND = String.raw`
$ErrorActionPreference = 'Stop'
$systemLog = Get-WinEvent -ListLog System -ErrorAction Stop
$systemReadProbe = Get-WinEvent -LogName System -MaxEvents 1 -ErrorAction Stop
$tcpipProvider = Get-WinEvent -ListProvider Tcpip -ErrorAction Stop
$allSettings = @(Get-NetTCPSetting -ErrorAction Stop)
$settings = @($allSettings | Where-Object {
  $null -ne $_.DynamicPortRangeStartPort -and
  $null -ne $_.DynamicPortRangeNumberOfPorts -and
  [int]$_.DynamicPortRangeNumberOfPorts -gt 0
} | ForEach-Object {
  [pscustomobject]@{
    settingName = [string]$_.SettingName
    dynamicPortRangeStartPort = [int]$_.DynamicPortRangeStartPort
    dynamicPortRangeNumberOfPorts = [int]$_.DynamicPortRangeNumberOfPorts
    autoReusePortRangeStartPort = if ($null -eq $_.AutoReusePortRangeStartPort) { $null } else { [int]$_.AutoReusePortRangeStartPort }
    autoReusePortRangeNumberOfPorts = if ($null -eq $_.AutoReusePortRangeNumberOfPorts) { $null } else { [int]$_.AutoReusePortRangeNumberOfPorts }
  }
})
if ($settings.Count -eq 0) { throw 'No configured Windows TCP dynamic-port range was available.' }
$connections = @(Get-NetTCPConnection -ErrorAction Stop)
$ranges = @($settings |
  Group-Object dynamicPortRangeStartPort,dynamicPortRangeNumberOfPorts |
  ForEach-Object {
    $start = [int]$_.Group[0].dynamicPortRangeStartPort
    $count = [int]$_.Group[0].dynamicPortRangeNumberOfPorts
    $end = $start + $count - 1
    $dynamic = @($connections | Where-Object {
      [int]$_.LocalPort -ge $start -and [int]$_.LocalPort -le $end
    })
    $timeWait = @($dynamic | Where-Object { [string]$_.State -eq 'TimeWait' })
    $uniquePorts = @($dynamic | ForEach-Object { [int]$_.LocalPort } | Sort-Object -Unique)
    $uniqueTimeWaitPorts = @($timeWait | ForEach-Object { [int]$_.LocalPort } | Sort-Object -Unique)
    [pscustomobject]@{
      startPort = $start
      numberOfPorts = $count
      endPort = $end
      connectionCount = $dynamic.Count
      uniqueLocalPortCount = $uniquePorts.Count
      timeWaitConnectionCount = $timeWait.Count
      timeWaitUniqueLocalPortCount = $uniqueTimeWaitPorts.Count
      freeUniqueLocalPortMargin = $count - $uniquePorts.Count
    }
  } | Sort-Object startPort,numberOfPorts)
$eventQueryErrors = @()
$events = @(Get-WinEvent -FilterHashtable @{
  LogName = 'System'
  ProviderName = 'Tcpip'
  Id = 4231
} -MaxEvents 1 -ErrorAction SilentlyContinue -ErrorVariable eventQueryErrors)
$unexpectedEventQueryErrors = @($eventQueryErrors | Where-Object {
  [string]$_.FullyQualifiedErrorId -notmatch '^NoMatchingEventsFound'
})
if ($unexpectedEventQueryErrors.Count -gt 0) {
  throw 'The bounded Tcpip Event 4231 query failed unexpectedly.'
}
$latest = $null
if ($events.Count -gt 0) {
  $event = $events[0]
  $latest = [pscustomobject]@{
    providerName = [string]$event.ProviderName
    timeCreatedUtc = $event.TimeCreated.ToUniversalTime().ToString('o')
    recordId = [long]$event.RecordId
    logName = [string]$event.LogName
  }
}
[pscustomobject]@{
  schemaVersion = 1
  kind = 'windows-tcp-dynamic-port-readiness-snapshot'
  capturedAt = (Get-Date).ToUniversalTime().ToString('o')
  provider = 'Get-NetTCPSetting + Get-NetTCPConnection + bounded Get-WinEvent'
  eventQuery = [pscustomobject]@{
    status = 'available'
    logName = [string]$systemLog.LogName
    systemReadProbeRecordId = [long]$systemReadProbe.RecordId
    systemReadProbeTimeCreatedUtc = $systemReadProbe.TimeCreated.ToUniversalTime().ToString('o')
    providerName = [string]$tcpipProvider.Name
    eventId = 4231
    maximumEventsScanned = 1
  }
  unconfiguredSettingNames = @($allSettings | Where-Object {
    $null -eq $_.DynamicPortRangeStartPort -or $null -eq $_.DynamicPortRangeNumberOfPorts
  } | ForEach-Object { [string]$_.SettingName })
  settings = $settings
  totalConnectionCount = $connections.Count
  ranges = $ranges
  latestTcpip4231 = $latest
} | ConvertTo-Json -Depth 7 -Compress
`;

function clone(value) {
  if (value === undefined) return undefined;
  return structuredClone(value);
}

function deepFreeze(value) {
  if (value === null || typeof value !== 'object' || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}

function requireCondition(condition, message, detail = undefined) {
  if (condition) return;
  const error = new Error(message);
  if (detail !== undefined) error.detail = clone(detail);
  throw error;
}

function sameArray(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function sha256Bytes(value) {
  return createHash('sha256').update(value).digest('hex');
}

function canonicalJsonSha256(value) {
  return sha256Bytes(Buffer.from(JSON.stringify(value), 'utf8'));
}

function boundedDiagnosticValue(value, maximumJsonBytes = 65_536) {
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
    jsonSha256: sha256Bytes(bytes),
    jsonPrefix: json.slice(0, 16_384),
  };
}

function serializedError(error) {
  return {
    name: error?.name ?? 'Error',
    message: String(error?.message ?? error).slice(0, 8_192),
    stack: error?.stack === undefined || error?.stack === null
      ? null
      : String(error.stack).slice(0, 32_768),
    detail: boundedDiagnosticValue(error?.detail),
    code: error?.code === undefined || error?.code === null
      ? null
      : String(error.code).slice(0, 512),
  };
}

export function assertStartupEnduranceNotInterrupted(
  terminationSignal,
  boundary = 'unspecified-boundary',
) {
  requireCondition(terminationSignal === null,
    `Startup endurance probe interrupted by ${terminationSignal}.`, {
      terminationSignal,
      boundary,
    });
  return true;
}

export async function claimStartupEnduranceAttemptRoot(resultRoot) {
  requireCondition(typeof resultRoot === 'string' && path.isAbsolute(resultRoot),
    'Startup endurance attempt root must be an absolute path.');
  const parentDirectory = path.dirname(resultRoot);
  let parentMetadata;
  try {
    parentMetadata = await stat(parentDirectory);
  } catch (error) {
    const failure = new Error(
      `Startup endurance parent directory is unavailable: ${parentDirectory}.`,
    );
    failure.cause = error;
    throw failure;
  }
  requireCondition(parentMetadata.isDirectory(),
    `Startup endurance parent path is not a directory: ${parentDirectory}.`);
  try {
    await mkdir(resultRoot, { recursive: false });
  } catch (error) {
    if (error?.code === 'EEXIST') {
      const failure = new Error(
        'Startup endurance attempt root already exists; the sole declared probe attempt '
          + `has already been consumed: ${resultRoot}.`,
      );
      failure.code = 'STARTUP_ENDURANCE_ATTEMPT_ALREADY_CONSUMED';
      failure.cause = error;
      throw failure;
    }
    throw error;
  }
  return deepFreeze({
    schemaVersion: 1,
    kind: 'first-instance-startup-endurance-attempt-root-claim',
    resultRoot,
    claimedAt: new Date().toISOString(),
    priorAttemptAllowed: false,
  });
}

async function withDeadline(promise, timeoutMs, label) {
  let timeoutId;
  const timeout = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(`${label} exceeded its ${timeoutMs} ms deadline.`));
    }, timeoutMs);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(timeoutId);
  }
}

function countBy(values) {
  const counts = {};
  for (const value of values) counts[value] = (counts[value] ?? 0) + 1;
  return counts;
}

export function selectStartupEnduranceSessions(standalonePlan) {
  validateFirstInstanceStandaloneDeploymentPlan(standalonePlan, {
    runId: standalonePlan?.runId,
  });
  const matrix = standalonePlan.matrices[0];
  const quartets = standalonePlan.quartets.filter(
    (quartet) => quartet.matrixIndex === 0
      && quartet.quartetIndex < STARTUP_ENDURANCE_QUARTET_COUNT,
  );
  const sessionById = new Map(
    standalonePlan.sessions.map((session) => [session.sessionId, session]),
  );
  const sessionIds = quartets.flatMap((quartet) => quartet.sessionIds);
  const sessions = sessionIds.map((sessionId) => sessionById.get(sessionId));
  const cellCounts = countBy(sessions.map(
    (session) => `${session.assignedLaneId}:${session.visibilityOrderId}`,
  ));
  const balance = {
    quartetCodes: countBy(quartets.map((quartet) => quartet.quartetCode)),
    lanes: countBy(sessions.map((session) => session.assignedLaneId)),
    visibilityOrders: countBy(sessions.map((session) => session.visibilityOrderId)),
    laneByVisibilityOrder: cellCounts,
  };
  const reasons = [];
  if (matrix?.matrixIndex !== 0 || matrix?.matrixOrdinal !== 1) {
    reasons.push('the source matrix is not frozen matrix 1');
  }
  if (quartets.length !== STARTUP_ENDURANCE_QUARTET_COUNT
    || sessions.length !== STARTUP_ENDURANCE_SESSION_COUNT) {
    reasons.push('selection does not contain exactly eight quartets and 32 sessions');
  }
  if (!sameArray(quartets.map((quartet) => quartet.quartetCode), EXPECTED_QUARTET_CODES)) {
    reasons.push('quartet order is not the frozen first-eight matrix-1 prefix');
  }
  if (sessions.some((session, index) => session === undefined
    || session.matrixIndex !== 0
    || session.matrixSessionIndex !== index
    || session.globalSessionIndex !== index
    || session.quartetIndex !== Math.floor(index / 4))) {
    reasons.push('session order is not the exact canonical first-32 prefix');
  }
  for (const [field, expected] of [
    ['portable', 16],
    ['feature', 16],
  ]) {
    if (balance.lanes[field] !== expected) reasons.push(`lane ${field} is not balanced`);
  }
  for (const field of ['H', 'L']) {
    if (balance.visibilityOrders[field] !== 16) {
      reasons.push(`visibility order ${field} is not balanced`);
    }
  }
  for (const field of ['portable:H', 'portable:L', 'feature:H', 'feature:L']) {
    if (balance.laneByVisibilityOrder[field] !== 8) {
      reasons.push(`lane/visibility cell ${field} is not balanced`);
    }
  }
  for (const field of ['AX', 'AY', 'BX', 'BY']) {
    if (balance.quartetCodes[field] !== 2) {
      reasons.push(`quartet code ${field} is not balanced`);
    }
  }
  requireCondition(reasons.length === 0,
    `Startup endurance selection is invalid: ${reasons.join('; ')}.`, {
      reasons,
      balance,
    });
  return deepFreeze({
    matrix: clone(matrix),
    quartets: clone(quartets),
    sessions: clone(sessions),
    balance,
  });
}

export function buildStartupEnduranceProbePlan({ runId, standalonePlan }) {
  requireCondition(typeof runId === 'string' && runId.length > 0,
    'Startup endurance runId is required.');
  const selection = selectStartupEnduranceSessions(standalonePlan);
  return deepFreeze({
    schemaVersion: 1,
    kind: STARTUP_ENDURANCE_PROBE_KIND,
    runId,
    executionMode: STARTUP_ENDURANCE_EXECUTION_MODE,
    analysisEligible: false,
    scope: 'one forced-off gate plus 32 fresh-browser startup/configuration observations; '
      + 'no trial timing or efficacy',
    sourcePlan: {
      kind: standalonePlan.kind,
      runId: standalonePlan.runId,
      sha256: canonicalJsonSha256(standalonePlan),
      matrixOrdinal: 1,
      quartetPrefixLength: STARTUP_ENDURANCE_QUARTET_COUNT,
    },
    executionPolicy: {
      attemptCount: 1,
      retryCount: 0,
      replacementAllowed: false,
      priorInvocationAllowed: false,
      resultRootMustBeAbsent: true,
      stopOnFirstFailure: true,
      serverInstanceCount: 1,
      serverRestartAllowed: false,
      forcedFeatureOffGateCount: 1,
      browserProcessCount: STARTUP_ENDURANCE_BROWSER_COUNT,
      freshBrowserProcessPerSession: true,
      freshPlaywrightTemporaryProfilePerSession: true,
      postDisconnectDelayMs: STANDALONE_POST_DISCONNECT_DELAY_MS,
    },
    exclusionPolicy: {
      analysisEligible: false,
      startTrialAllowed: false,
      timingCaptureAllowed: false,
      efficacyAnalysisAllowed: false,
      numericalDecision: null,
    },
    networkReadinessPolicy: {
      provider: 'bounded Windows PowerShell aggregate queries',
      tcpipEventId: 4231,
      quietWindowMs: STARTUP_ENDURANCE_TCP_QUIET_WINDOW_MS,
      minimumFreeDynamicPortFormula:
        'max(4096, ceil(0.25 * dynamicPortRangeNumberOfPorts))',
      minimumFreeDynamicPorts: STARTUP_ENDURANCE_MINIMUM_FREE_DYNAMIC_PORTS,
      minimumFreeDynamicPortFraction:
        STARTUP_ENDURANCE_MINIMUM_FREE_DYNAMIC_PORT_FRACTION,
      captureBeforeEveryBrowserLaunch: true,
      captureAfterEveryBrowserDisconnectDelay: true,
      scheduledSnapshotCount: STARTUP_ENDURANCE_BROWSER_COUNT * 2,
      failClosedOnQueryOrParseFailure: true,
      limitation:
        'Get-NetTCPConnection is a bounded aggregate snapshot of global local-port use; '
        + 'Windows allocation may also depend on tuple, address family, and network compartment, '
        + 'and Event 4231 does not identify the exhausting process.',
    },
    browser: {
      launchApi: 'chromium.launch',
      arguments: [...STARTUP_ENDURANCE_BROWSER_ARGS],
      persistentContext: false,
      profilePolicy: 'fresh-playwright-temporary-profile-per-process',
      viewport: { width: 1280, height: 900 },
      deviceScaleFactor: 1,
      benchmarkReadinessTimeoutMs: BENCHMARK_READINESS_TIMEOUT_MS,
    },
    workload: {
      modeId: STANDALONE_MODE_ID,
      objectCount: OBJECT_COUNT,
      bucketCount: BUCKET_COUNT,
      layout: LAYOUT,
      configurationOnly: true,
    },
    schedule: selection,
  });
}

function validInteger(value, minimum = 0) {
  return Number.isSafeInteger(value) && value >= minimum;
}

export function assessStartupTcpReadiness(snapshot, {
  nowMs = Date.parse(snapshot?.capturedAt ?? ''),
} = {}) {
  const reasons = [];
  const rangeAssessments = [];
  if (snapshot?.schemaVersion !== 1
    || snapshot?.kind !== 'windows-tcp-dynamic-port-readiness-snapshot') {
    reasons.push('TCP snapshot envelope is unavailable or invalid');
  }
  if (!Number.isFinite(nowMs)) reasons.push('TCP snapshot capture time is invalid');
  if (snapshot?.eventQuery?.status !== 'available'
    || snapshot?.eventQuery?.eventId !== 4231
    || snapshot?.eventQuery?.logName !== 'System'
    || snapshot?.eventQuery?.providerName !== 'Tcpip'
    || !validInteger(snapshot?.eventQuery?.systemReadProbeRecordId, 1)
    || !Number.isFinite(Date.parse(
      snapshot?.eventQuery?.systemReadProbeTimeCreatedUtc ?? '',
    ))
    || snapshot?.eventQuery?.maximumEventsScanned !== 1) {
    reasons.push('bounded Tcpip Event 4231 query was unavailable');
  }
  if (!Array.isArray(snapshot?.settings) || snapshot.settings.length === 0) {
    reasons.push('configured TCP settings are unavailable');
  }
  if (!Array.isArray(snapshot?.ranges) || snapshot.ranges.length === 0) {
    reasons.push('dynamic TCP ranges are unavailable');
  } else {
    for (const [index, range] of snapshot.ranges.entries()) {
      const valid = validInteger(range?.startPort, 1)
        && validInteger(range?.numberOfPorts, 1)
        && validInteger(range?.endPort, 1)
        && range.endPort === range.startPort + range.numberOfPorts - 1
        && range.endPort <= 65_535
        && validInteger(range?.connectionCount)
        && validInteger(range?.uniqueLocalPortCount)
        && validInteger(range?.timeWaitConnectionCount)
        && validInteger(range?.timeWaitUniqueLocalPortCount)
        && validInteger(range?.freeUniqueLocalPortMargin)
        && range.freeUniqueLocalPortMargin
          === range.numberOfPorts - range.uniqueLocalPortCount
        && range.timeWaitConnectionCount <= range.connectionCount
        && range.timeWaitUniqueLocalPortCount <= range.uniqueLocalPortCount;
      const requiredFreeDynamicPorts = validInteger(range?.numberOfPorts, 1)
        ? Math.max(
          STARTUP_ENDURANCE_MINIMUM_FREE_DYNAMIC_PORTS,
          Math.ceil(
            STARTUP_ENDURANCE_MINIMUM_FREE_DYNAMIC_PORT_FRACTION * range.numberOfPorts,
          ),
        )
        : null;
      const pass = valid
        && range.freeUniqueLocalPortMargin >= requiredFreeDynamicPorts;
      rangeAssessments.push({
        rangeIndex: index,
        startPort: range?.startPort ?? null,
        numberOfPorts: range?.numberOfPorts ?? null,
        freeUniqueLocalPortMargin: range?.freeUniqueLocalPortMargin ?? null,
        requiredFreeDynamicPorts,
        pass,
      });
      if (!valid) reasons.push(`dynamic TCP range ${index} is internally inconsistent`);
      else if (!pass) {
        reasons.push(
          `dynamic TCP range ${index} has only ${range.freeUniqueLocalPortMargin} free `
          + `unique local ports; ${requiredFreeDynamicPorts} are required`,
        );
      }
    }
  }
  let latestTcpip4231 = null;
  if (snapshot?.latestTcpip4231 !== null && snapshot?.latestTcpip4231 !== undefined) {
    const event = snapshot.latestTcpip4231;
    const eventMs = Date.parse(event?.timeCreatedUtc ?? '');
    const ageMs = nowMs - eventMs;
    const valid = typeof event?.providerName === 'string'
      && /tcpip/iu.test(event.providerName)
      && Number.isFinite(eventMs)
      && validInteger(event?.recordId, 1)
      && typeof event?.logName === 'string';
    latestTcpip4231 = {
      providerName: event?.providerName ?? null,
      timeCreatedUtc: event?.timeCreatedUtc ?? null,
      recordId: event?.recordId ?? null,
      ageMs: Number.isFinite(ageMs) ? ageMs : null,
      valid,
      withinQuietWindow: valid
        && ageMs >= 0
        && ageMs <= STARTUP_ENDURANCE_TCP_QUIET_WINDOW_MS,
    };
    if (!valid || ageMs < 0) reasons.push('latest Tcpip Event 4231 record is invalid');
    else if (latestTcpip4231.withinQuietWindow) {
      reasons.push(
        `Tcpip Event 4231 record ${event.recordId} occurred ${Math.round(ageMs)} ms ago`,
      );
    }
  }
  return deepFreeze({
    schemaVersion: 1,
    kind: 'first-instance-startup-tcp-readiness-assessment',
    pass: reasons.length === 0,
    quietWindowMs: STARTUP_ENDURANCE_TCP_QUIET_WINDOW_MS,
    minimumFreeDynamicPortFormula:
      'max(4096, ceil(0.25 * dynamicPortRangeNumberOfPorts))',
    rangeAssessments,
    latestTcpip4231,
    reasons: [...new Set(reasons)],
  });
}

export function validateStartupEnduranceConfiguration(capture, canonicalSession) {
  const reasons = [];
  const boot = capture?.initialPageBoot;
  const pageLifecycle = capture?.pageConstructionLifecycle;
  const config = capture?.selectedConfig;
  const lifecycle = capture?.strategyLifecycle;
  const diagnostics = capture?.strategyDiagnostics;
  const ledger = lifecycle?.productionResourceLedger;
  const lane = canonicalSession?.assignedLaneId;
  const absentLane = canonicalSession?.absentLaneId;
  const expectedAddressMode = lane === 'portable'
    ? 'bucket-base'
    : 'indirect-first-instance';
  if (boot?.schemaVersion !== 1
    || boot?.kind !== 'first-instance-standalone-initial-page-boot'
    || boot?.modeId !== STANDALONE_MODE_ID
    || boot?.laneId !== lane
    || !sameArray(boot?.visibilityOrder, canonicalSession?.visibilityOrder)
    || boot?.objectCount !== OBJECT_COUNT
    || boot?.bucketCount !== BUCKET_COUNT
    || boot?.layout !== LAYOUT
    || boot?.initialRebuildCount !== 1
    || boot?.priorStrategyConstructionCount !== 0) {
    reasons.push('initial standalone boot differs from the canonical session');
  }
  if (pageLifecycle?.schemaVersion !== 1
    || pageLifecycle?.kind !== 'benchmark-page-strategy-construction-lifecycle'
    || pageLifecycle?.rebuildCount !== 1
    || pageLifecycle?.strategyConstructionCount !== 1
    || !sameArray(pageLifecycle?.constructedStrategyIds, [STANDALONE_MODE_ID])
    || pageLifecycle?.selectedStrategyId !== STANDALONE_MODE_ID
    || pageLifecycle?.strictStandaloneBoot !== true) {
    reasons.push('page did not construct exactly one strict-boot standalone strategy');
  }
  if (config?.strategyId !== STANDALONE_MODE_ID
    || config?.objectCount !== OBJECT_COUNT
    || config?.bucketCount !== BUCKET_COUNT
    || config?.visibilityFraction !== canonicalSession?.visibilityOrder?.[0]
    || config?.layout !== LAYOUT
    || config?.laneId !== lane
    || !sameArray(config?.visibilityOrder, canonicalSession?.visibilityOrder)) {
    reasons.push('selected configuration differs from the canonical session');
  }
  if (capture?.environment?.threeRevision !== THREE_REVISION
    || capture?.environment?.rendererBackend !== 'WebGPUBackend'
    || capture?.environment?.reversedDepth !== true
    || capture?.environment?.rendererReversedDepthBuffer !== true
    || capture?.environment?.timestampAvailable !== true
    || capture?.environment?.indirectFirstInstanceAvailable !== true
    || capture?.environment?.crossOriginIsolated !== true
    || capture?.environment?.webgpuUncapturedErrorCount !== 0
    || capture?.environment?.webgpuValidationErrorCount !== 0
    || capture?.environment?.webgpuDeviceLossCount !== 0
    || !Number.isFinite(capture?.environment?.performanceNowQuantumMs)
    || capture.environment.performanceNowQuantumMs <= 0
    || capture.environment.performanceNowQuantumMs > 0.01) {
    reasons.push('WebGPU environment is unavailable, dirty, or differs from the candidate');
  }
  if (!sameArray([
    capture?.environment?.viewport?.width,
    capture?.environment?.viewport?.height,
    capture?.environment?.viewport?.devicePixelRatio,
  ], [VIEWPORT.width, VIEWPORT.height, VIEWPORT.devicePixelRatio])) {
    reasons.push('render viewport differs from the fixed workload');
  }
  if (capture?.shaderEvidence !== null) {
    reasons.push('startup probe unexpectedly performed a shader challenge');
  }
  if (capture?.timestampPoolPreprime?.schemaVersion !== 1
    || capture?.timestampPoolPreprime?.kind !== 'three-r185-timestamp-pool-preprime'
    || capture?.timestampPoolPreprime?.addedTimestampUidCount?.render !== 1
    || capture?.timestampPoolPreprime?.addedTimestampUidCount?.compute !== 1) {
    reasons.push('standalone boot did not perform its exact untimed timestamp pre-prime');
  }
  const workloadScenario = capture?.workload?.scenario;
  if (capture?.workload?.scenarioSeed !== SCENARIO_SEED
    || capture?.workload?.geometryFixtures?.bucketCount !== BUCKET_COUNT
    || workloadScenario?.seed !== SCENARIO_SEED
    || workloadScenario?.objectCount !== OBJECT_COUNT
    || workloadScenario?.bucketCount !== BUCKET_COUNT
    || workloadScenario?.visibilityFraction !== canonicalSession?.visibilityOrder?.[0]
    || workloadScenario?.layout !== LAYOUT
    || JSON.stringify(capture?.geometryManifest)
      !== JSON.stringify(capture?.workload?.geometryFixtures)
    || JSON.stringify(capture?.scenarioManifest)
      !== JSON.stringify(workloadScenario)) {
    reasons.push('fingerprinted workload differs from the fixed startup configuration');
  }
  if (lifecycle?.schemaVersion !== 1
    || lifecycle?.kind !== 'first-instance-live-standalone-static-resource-lifecycle'
    || lifecycle?.pass !== true
    || lifecycle?.selectedLane !== lane
    || lifecycle?.absentLane !== absentLane
    || lifecycle?.absentLaneConstructed !== false
    || lifecycle?.lanesConstructed !== 1
    || lifecycle?.primed !== true
    || lifecycle?.activeVisibilityFraction !== canonicalSession?.visibilityOrder?.[0]
    || lifecycle?.scenarioSwitchSerial !== 0
    || lifecycle?.shaderObservationSerial !== 0
    || lifecycle?.bundleRecordCallbackCount !== 1
    || lifecycle?.commandBuffer?.byteOffset !== 0
    || lifecycle?.commandBuffer?.firstOffset !== 0
    || !validInteger(lifecycle?.commandBuffer?.attributeId, 1)
    || !Array.isArray(lifecycle?.computeNodeIds)
    || lifecycle.computeNodeIds.length !== 2) {
    reasons.push('selected-lane lifecycle is invalid at startup');
  }
  if (ledger?.schemaVersion !== 1
    || ledger?.kind !== 'first-instance-live-standalone-production-resource-ledger'
    || ledger?.selectedLane !== lane
    || ledger?.absentLane !== absentLane
    || ledger?.absentLaneConstructed !== false
    || !sameArray(ledger?.constructedLaneIds, [lane])
    || !sameArray(ledger?.constructedAddressModes, [expectedAddressMode])
    || ledger?.constructedLaneCount !== 1
    || ledger?.indirectCommandBufferCount !== 1
    || ledger?.computeNodeCount !== 2
    || ledger?.materialCount !== 1
    || ledger?.meshCount !== 1
    || ledger?.bundleCount !== 1
    || ledger?.addressMode !== expectedAddressMode
    || ledger?.hasBucketBaseVertexAttribute !== (lane === 'portable')) {
    reasons.push('production resource ledger is not single-lane exact');
  }
  if (diagnostics?.schemaVersion !== 1
    || diagnostics?.kind !== 'first-instance-live-standalone-deployment'
    || diagnostics?.pass !== true
    || diagnostics?.selectedLane !== lane
    || diagnostics?.absentLane !== absentLane
    || diagnostics?.addressMode !== expectedAddressMode
    || diagnostics?.configuredDrawCommands !== BUCKET_COUNT
    || diagnostics?.configuredRenderObjects !== 1
    || diagnostics?.configuredComputeDispatches !== 2
    || diagnostics?.configuredComputeSubmissions !== 1
    || diagnostics?.laneCommandBufferCount !== 1) {
    reasons.push('standalone strategy diagnostics differ from the deployment path');
  }
  if (!Array.isArray(capture?.webgpuUncapturedErrors)
    || capture.webgpuUncapturedErrors.length !== 0
    || !Array.isArray(capture?.webgpuDeviceLosses)
    || capture.webgpuDeviceLosses.length !== 0) {
    reasons.push('startup capture contains a WebGPU error or device loss');
  }
  if (capture?.trialExecutionState?.phase !== 'idle'
    || capture?.trialExecutionState?.trialError !== null
    || capture?.trialExecutionState?.rowCount !== 0
    || capture?.trialExecutionState?.summary !== null) {
    reasons.push('startup probe unexpectedly entered or retained trial timing state');
  }
  return deepFreeze({
    schemaVersion: 1,
    kind: 'first-instance-standalone-startup-configuration-validation',
    pass: reasons.length === 0,
    reasons: [...new Set(reasons)],
  });
}

export async function captureWindowsTcpState(captureLabel) {
  requireCondition(process.platform === 'win32',
    'Startup endurance TCP readiness capture requires Windows.');
  requireCondition(typeof captureLabel === 'string' && captureLabel.length > 0,
    'TCP capture label is required.');
  const systemRoot = process.env.SystemRoot ?? 'C:\\Windows';
  const executable = path.join(
    systemRoot,
    'System32',
    'WindowsPowerShell',
    'v1.0',
    'powershell.exe',
  );
  const { stdout, stderr } = await execFileAsync(executable, [
    '-NoLogo',
    '-NoProfile',
    '-NonInteractive',
    '-ExecutionPolicy',
    'Bypass',
    '-Command',
    WINDOWS_TCP_SNAPSHOT_COMMAND,
  ], {
    windowsHide: true,
    timeout: TCP_QUERY_TIMEOUT_MS,
    maxBuffer: 1024 * 1024,
    encoding: 'utf8',
  });
  requireCondition(stderr.trim() === '', 'Windows TCP readiness query wrote stderr.', {
    stderr: stderr.slice(0, 4_096),
  });
  let parsed;
  try {
    parsed = JSON.parse(stdout.trim());
  } catch (error) {
    throw new Error(`Windows TCP readiness query returned invalid JSON: ${error.message}`);
  }
  const snapshot = {
    ...parsed,
    captureLabel,
    queryPolicy: {
      boundedConnectionEvidence: 'aggregate counts only; no connection rows retained',
      maximumSystemEventsScanned: 1,
      commandFamily:
        'Get-NetTCPSetting/Get-NetTCPConnection/Get-WinEvent via Windows PowerShell',
    },
  };
  const assessment = assessStartupTcpReadiness(snapshot);
  return deepFreeze({ snapshot, assessment });
}

function tcpEvidenceRecord(capture, artifact) {
  return deepFreeze({
    captureLabel: capture.snapshot.captureLabel,
    snapshot: clone(capture.snapshot),
    assessment: clone(capture.assessment),
    artifact,
  });
}

class IncrementalArtifactStore {
  constructor(runDirectory) {
    this.runDirectory = runDirectory;
    this.journalSerial = 0;
    this.journalArtifacts = [];
  }

  async json(relativePath, value) {
    const absolutePath = path.join(this.runDirectory, ...relativePath.split('/'));
    const bytes = jsonBytes(value);
    await writeFile(absolutePath, bytes, { flag: 'wx' });
    return Object.freeze({
      path: relativePath,
      encoding: 'json-utf8',
      byteLength: bytes.length,
      sha256: sha256Bytes(bytes),
    });
  }

  jsonSynchronous(relativePath, value) {
    const absolutePath = path.join(this.runDirectory, ...relativePath.split('/'));
    const bytes = jsonBytes(value);
    // The terminal completion write is deliberately synchronous. Node cannot
    // dispatch a newly arrived signal between the immediately preceding
    // interruption gate and this exclusive write.
    writeFileSync(absolutePath, bytes, { flag: 'wx' });
    return Object.freeze({
      path: relativePath,
      encoding: 'json-utf8',
      byteLength: bytes.length,
      sha256: sha256Bytes(bytes),
    });
  }

  async journal(eventKind, payload) {
    this.journalSerial += 1;
    const relativePath = `journal/${String(this.journalSerial).padStart(4, '0')}`
      + `-${eventKind}.json`;
    const artifact = await this.json(relativePath, {
      schemaVersion: 1,
      kind: 'first-instance-startup-endurance-journal-event',
      journalSerial: this.journalSerial,
      eventKind,
      committedAt: new Date().toISOString(),
      payload,
    });
    this.journalArtifacts.push(artifact);
    return artifact;
  }
}

function createBoundedPageDiagnostics() {
  const state = {
    events: [],
    droppedEventCount: 0,
  };
  const append = (event) => {
    if (state.events.length >= MAXIMUM_PAGE_DIAGNOSTIC_EVENTS) {
      state.droppedEventCount += 1;
      return;
    }
    state.events.push({
      ...event,
      capturedAt: new Date().toISOString(),
    });
  };
  return { state, append };
}

export function validateStartupPageDiagnostics(diagnostics) {
  const reasons = [];
  if (!Array.isArray(diagnostics?.events)) {
    reasons.push('page diagnostic events are unavailable');
  } else if (diagnostics.events.length !== 0) {
    reasons.push(`page retained ${diagnostics.events.length} fatal diagnostic event(s)`);
  }
  if (!validInteger(diagnostics?.droppedEventCount)
    || diagnostics.droppedEventCount !== 0) {
    reasons.push('page diagnostic capture overflowed or is invalid');
  }
  return deepFreeze({
    schemaVersion: 1,
    kind: 'first-instance-startup-page-diagnostics-validation',
    pass: reasons.length === 0,
    reasons,
  });
}

function attachPageDiagnostics(page, append, ignoreExpectedShutdownFailure) {
  page.on('pageerror', (error) => append({
    source: 'pageerror',
    detail: String(error?.stack ?? error?.message ?? error).slice(0, 8_192),
  }));
  page.on('console', (message) => {
    if (message.type() === 'error') {
      append({
        source: 'console',
        detail: message.text().slice(0, 8_192),
      });
    }
  });
  page.on('requestfailed', (request) => {
    if (ignoreExpectedShutdownFailure()) return;
    append({
      source: 'requestfailed',
      url: request.url().slice(0, 2_048),
      resourceType: request.resourceType(),
      failure: String(request.failure()?.errorText ?? 'unknown').slice(0, 2_048),
    });
  });
  page.on('crash', () => append({
    source: 'page-crash',
    detail: 'Playwright page crash event',
  }));
}

async function attemptEmergencyClose(label, target, timeoutMs) {
  if (target === null || target === undefined) {
    return {
      attempted: false,
      succeeded: true,
      error: null,
    };
  }
  try {
    await withDeadline(
      Promise.resolve().then(() => target.close()),
      timeoutMs,
      label,
    );
    return {
      attempted: true,
      succeeded: true,
      error: null,
    };
  } catch (error) {
    return {
      attempted: true,
      succeeded: false,
      error: serializedError(error),
    };
  }
}

function observeEmergencyCloseState(state) {
  const observationErrors = [];
  let pageClosed = null;
  let contextCount = null;
  let browserConnected = null;
  try {
    pageClosed = state.page === null ? null : state.page.isClosed();
  } catch (error) {
    observationErrors.push({ field: 'pageClosed', error: serializedError(error) });
  }
  try {
    contextCount = state.browser.contexts().length;
  } catch (error) {
    observationErrors.push({ field: 'contextCount', error: serializedError(error) });
  }
  try {
    browserConnected = state.browser.isConnected();
  } catch (error) {
    observationErrors.push({ field: 'browserConnected', error: serializedError(error) });
  }
  return {
    pageCreated: state.page !== null,
    pageClosed,
    contextCreated: state.context !== null,
    contextCount,
    browserConnected,
    disconnectedEventCount: state.record.disconnectedEventCount,
    disconnectedAt: state.record.disconnectedAt,
    disconnectedRunElapsedMs: state.record.disconnectedRunElapsedMs,
    observationErrors,
  };
}

export async function closeStartupBrowserStateAfterAbort(state, reason, {
  closeTimeoutMs = BROWSER_CLOSE_TIMEOUT_MS,
  postDisconnectDelayMs = STANDALONE_POST_DISCONNECT_DELAY_MS,
} = {}) {
  requireCondition(state !== null && typeof state === 'object'
    && state.browser !== null && typeof state.browser === 'object'
    && state.record !== null && typeof state.record === 'object',
  'Emergency startup browser cleanup requires a live browser state.');
  if (state.forceClosePromise !== null && state.forceClosePromise !== undefined) {
    return state.forceClosePromise;
  }
  state.forceClosePromise = (async () => {
    state.shutdownStarted = true;
    state.record.abortedAt ??= new Date().toISOString();
    state.record.abortReason ??= String(reason).slice(0, 2_048);
    const cleanup = {
      schemaVersion: 1,
      kind: 'first-instance-startup-emergency-browser-cleanup',
      attemptedAt: new Date().toISOString(),
      completedAt: null,
      contextClose: null,
      browserClose: null,
      observedFinalState: null,
      provenClosed: false,
      postDisconnectDelayObserved: false,
    };
    cleanup.contextClose = await attemptEmergencyClose(
      'aborted startup context close',
      state.context,
      closeTimeoutMs,
    );
    cleanup.browserClose = await attemptEmergencyClose(
      'aborted startup browser close',
      state.browser,
      closeTimeoutMs,
    );
    cleanup.observedFinalState = observeEmergencyCloseState(state);
    cleanup.provenClosed = cleanup.contextClose.succeeded === true
      && cleanup.browserClose.succeeded === true
      && cleanup.observedFinalState.observationErrors.length === 0
      && (cleanup.observedFinalState.pageCreated === false
        || cleanup.observedFinalState.pageClosed === true)
      && cleanup.observedFinalState.contextCount === 0
      && cleanup.observedFinalState.browserConnected === false
      && cleanup.observedFinalState.disconnectedEventCount === 1
      && Number.isFinite(cleanup.observedFinalState.disconnectedRunElapsedMs);
    if (cleanup.provenClosed) {
      state.record.closedAt ??= new Date().toISOString();
      state.record.closedBeforeNextLaunch = true;
    } else {
      // A failed/unknown cleanup is retained as such. In particular, never
      // manufacture closedAt for a browser that might still be orphaned.
      state.record.closedBeforeNextLaunch = false;
    }
    if (cleanup.observedFinalState.disconnectedEventCount === 1
      && Number.isFinite(cleanup.observedFinalState.disconnectedRunElapsedMs)
      && state.record.postDisconnectDelay === null) {
      const startedMonotonic = performance.now();
      await delay(postDisconnectDelayMs);
      const completedMonotonic = performance.now();
      state.record.postDisconnectDelay = {
        requestedMs: postDisconnectDelayMs,
        startedAt: new Date(Date.now() - (completedMonotonic - startedMonotonic)).toISOString(),
        startedRunElapsedMs: null,
        completedAt: new Date().toISOString(),
        completedRunElapsedMs: null,
        elapsedMs: completedMonotonic - startedMonotonic,
        abortedSession: true,
      };
      cleanup.postDisconnectDelayObserved =
        state.record.postDisconnectDelay.elapsedMs >= postDisconnectDelayMs;
    }
    cleanup.completedAt = new Date().toISOString();
    state.record.emergencyCleanup = cleanup;
    return clone(state.record);
  })();
  return state.forceClosePromise;
}

class StartupBrowserManager {
  constructor({ executablePath, runStartedMonotonic }) {
    this.executablePath = executablePath;
    this.runStartedMonotonic = runStartedMonotonic;
    this.records = [];
    this.active = null;
    this.emergencyClosePromise = null;
  }

  elapsed() {
    return performance.now() - this.runStartedMonotonic;
  }

  async launch({ role, matrixOrdinal = 1, session = null }) {
    requireCondition(this.active === null,
      `Refusing overlapping browser launch for ${role}.`);
    requireCondition(typeof role === 'string' && role.length > 0,
      'Startup browser role is required.');
    const prior = this.records.at(-1) ?? null;
    if (prior !== null) {
      requireCondition(prior.disconnectedEventCount === 1
        && prior.postDisconnectDelay?.requestedMs === STANDALONE_POST_DISCONNECT_DELAY_MS
        && prior.postDisconnectDelay?.elapsedMs >= STANDALONE_POST_DISCONNECT_DELAY_MS,
      'Prior startup browser did not complete its fixed disconnect interval.', prior);
    }
    const launchedRunElapsedMs = this.elapsed();
    const record = {
      schemaVersion: 1,
      kind: 'first-instance-standalone-browser-profile-lifecycle',
      browserInstanceSerial: this.records.length + 1,
      role,
      matrixOrdinal,
      sessionId: session?.sessionId ?? null,
      globalSessionIndex: session?.globalSessionIndex ?? null,
      sessionNamespace: session === null
        ? null
        : `${session.sessionId}:startup-browser-${this.records.length + 1}`,
      launchApi: 'chromium.launch',
      persistentContext: false,
      profilePolicy: 'fresh-playwright-temporary-profile-per-process',
      userDataDirectoryExposed: false,
      profileReused: false,
      launchArguments: [...STARTUP_ENDURANCE_BROWSER_ARGS],
      launchedAt: new Date().toISOString(),
      launchedRunElapsedMs,
      priorBrowserInstanceSerial: prior?.browserInstanceSerial ?? null,
      priorBrowserDisconnectedAt: prior?.disconnectedAt ?? null,
      previousDisconnectToLaunchGapMs: prior === null
        ? null
        : launchedRunElapsedMs - prior.disconnectedRunElapsedMs,
      launchSucceeded: false,
      contextCreatedAt: null,
      pageCreatedAt: null,
      navigationCompletedAt: null,
      readinessObservedAt: null,
      closedAt: null,
      disconnectedAt: null,
      disconnectedRunElapsedMs: null,
      contextCountBeforeClose: null,
      pageCountBeforeClose: null,
      disconnectedEventCount: 0,
      closedBeforeNextLaunch: false,
      postDisconnectDelay: null,
    };
    this.records.push(record);
    if (prior !== null) {
      requireCondition(record.previousDisconnectToLaunchGapMs
        >= STANDALONE_POST_DISCONNECT_DELAY_MS,
      'Startup browser launched before its fixed post-disconnect interval.', record);
    }
    const browser = await chromium.launch({
      executablePath: this.executablePath,
      headless: true,
      args: STARTUP_ENDURANCE_BROWSER_ARGS,
      timeout: BROWSER_OPERATION_TIMEOUT_MS,
    });
    record.launchSucceeded = true;
    browser.on('disconnected', () => {
      record.disconnectedEventCount += 1;
      record.disconnectedAt ??= new Date().toISOString();
      record.disconnectedRunElapsedMs ??= this.elapsed();
    });
    const pageDiagnostics = createBoundedPageDiagnostics();
    const state = {
      browser,
      context: null,
      page: null,
      pageDiagnostics,
      record,
      closePromise: null,
      forceClosePromise: null,
      shutdownStarted: false,
    };
    this.active = state;
    requireCondition(browser.contexts().length === 0,
      'Startup browser unexpectedly reused a context.');
    return state;
  }

  async openOnlyPage(state, url) {
    requireCondition(this.active === state && state.context === null && state.page === null,
      'Startup browser attempted more than one context/page.');
    state.context = await withDeadline(state.browser.newContext({
      viewport: { width: 1280, height: 900 },
      deviceScaleFactor: 1,
    }), BROWSER_OPERATION_TIMEOUT_MS, 'startup context creation');
    state.record.contextCreatedAt = new Date().toISOString();
    requireCondition(state.browser.contexts().length === 1,
      'Startup browser did not retain exactly one context.');
    state.page = await withDeadline(
      state.context.newPage(),
      BROWSER_OPERATION_TIMEOUT_MS,
      'startup page creation',
    );
    state.record.pageCreatedAt = new Date().toISOString();
    attachPageDiagnostics(
      state.page,
      state.pageDiagnostics.append,
      () => state.shutdownStarted,
    );
    requireCondition(state.context.pages().length === 1,
      'Startup browser did not retain exactly one page.');
    await withDeadline(
      state.page.goto(url, { waitUntil: 'domcontentloaded' }),
      BROWSER_OPERATION_TIMEOUT_MS,
      'startup entry navigation',
    );
    state.record.navigationCompletedAt = new Date().toISOString();
    await withDeadline(state.page.waitForFunction(
      () => window.__WEBGPU_BENCH__?.ready === true,
      null,
      { timeout: BENCHMARK_READINESS_TIMEOUT_MS },
    ), BROWSER_OPERATION_TIMEOUT_MS, 'startup benchmark readiness');
    state.record.readinessObservedAt = new Date().toISOString();
    return state.page;
  }

  async captureActiveFailureSnapshot() {
    const state = this.active;
    if (state === null) return null;
    let document = null;
    if (state.page !== null && !state.page.isClosed()) {
      document = await withDeadline(state.page.evaluate(() => ({
        url: globalThis.location?.href?.slice(0, 2_048) ?? null,
        documentReadyState: globalThis.document?.readyState ?? null,
        title: globalThis.document?.title?.slice(0, 512) ?? null,
        statusText: globalThis.document?.getElementById?.('status')?.textContent?.slice(0, 4_096)
          ?? null,
        benchPresent: typeof window.__WEBGPU_BENCH__ === 'object',
        benchReady: window.__WEBGPU_BENCH__?.ready ?? null,
      })), 2_000, 'active startup failure snapshot').catch(() => null);
    }
    return {
      lifecycle: clone(state.record),
      pageDiagnostics: clone(state.pageDiagnostics.state),
      document,
    };
  }

  async close(state) {
    if (state.closePromise !== null) return state.closePromise;
    requireCondition(this.active === state, 'Attempted to close a non-active startup browser.');
    state.closePromise = (async () => {
      const { browser, context, page, record } = state;
      record.contextCountBeforeClose = browser.contexts().length;
      record.pageCountBeforeClose = context?.pages().length ?? 0;
      requireCondition(record.contextCountBeforeClose === 1
        && record.pageCountBeforeClose === 1,
      'Startup browser violated its one-context/one-page boundary.', record);
      state.shutdownStarted = true;
      await withDeadline(context.close(), BROWSER_CLOSE_TIMEOUT_MS, 'startup context close');
      requireCondition(page.isClosed() && browser.contexts().length === 0,
        'Startup context/page did not close cleanly.');
      await withDeadline(browser.close(), BROWSER_CLOSE_TIMEOUT_MS, 'startup browser close');
      requireCondition(browser.isConnected() === false,
        'Startup browser remained connected after close.');
      requireCondition(record.disconnectedEventCount === 1
        && Number.isFinite(record.disconnectedRunElapsedMs),
      'Startup browser emitted an unexpected disconnected event count.', record);
      record.closedAt = new Date().toISOString();
      record.closedBeforeNextLaunch = true;
      this.active = null;
      return record;
    })();
    return state.closePromise;
  }

  async waitAfterDisconnect(record) {
    requireCondition(this.active === null
      && record?.disconnectedEventCount === 1
      && record?.postDisconnectDelay === null,
    'Startup post-disconnect delay lacks a newly closed browser.', record);
    const startedMonotonic = performance.now();
    const waitRecord = {
      requestedMs: STANDALONE_POST_DISCONNECT_DELAY_MS,
      startedAt: new Date().toISOString(),
      startedRunElapsedMs: startedMonotonic - this.runStartedMonotonic,
      completedAt: null,
      completedRunElapsedMs: null,
      elapsedMs: null,
    };
    await delay(STANDALONE_POST_DISCONNECT_DELAY_MS);
    const completedMonotonic = performance.now();
    waitRecord.completedAt = new Date().toISOString();
    waitRecord.completedRunElapsedMs = completedMonotonic - this.runStartedMonotonic;
    waitRecord.elapsedMs = completedMonotonic - startedMonotonic;
    requireCondition(waitRecord.elapsedMs >= STANDALONE_POST_DISCONNECT_DELAY_MS,
      'Startup post-disconnect interval completed early.', waitRecord);
    record.postDisconnectDelay = waitRecord;
    return waitRecord;
  }

  async forceClose(reason) {
    const state = this.active;
    if (state === null) return this.emergencyClosePromise;
    this.active = null;
    this.emergencyClosePromise = closeStartupBrowserStateAfterAbort(state, reason);
    return this.emergencyClosePromise;
  }
}

// Browser discovery and the capture body mirror the frozen runner's private
// findBrowser/configureStandaloneSession helpers. They remain local because
// this diagnostic must not modify the candidate runner's public surface.
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
      // Continue through the same installed-system-browser locations as the candidate runner.
    }
  }
  throw new Error('No installed Chrome, Chromium, or Edge executable was found.');
}

export function validateStartupForcedFeatureOffGate(capture) {
  const reasons = [];
  let gateRejections;
  try {
    gateRejections = validateLiveFirstInstanceForcedFeatureOffGate(capture?.gate);
  } catch (error) {
    gateRejections = [`forced-off evidence validator threw: ${error.message}`];
  }
  if (!Array.isArray(gateRejections) || gateRejections.length !== 0) {
    reasons.push(...(Array.isArray(gateRejections)
      ? gateRejections.map((reason) => `forced-off gate: ${reason}`)
      : ['forced-off gate validator returned an invalid result']));
  }
  const boot = capture?.gate?.initialPageBoot;
  const pageLifecycle = capture?.gate?.construction?.pageConstructionLifecycle;
  if (boot?.schemaVersion !== 1
    || boot?.kind !== 'first-instance-standalone-initial-page-boot'
    || boot?.modeId !== STANDALONE_MODE_ID
    || boot?.laneId !== 'portable'
    || !sameArray(boot?.visibilityOrder, [0.99, 0.2])
    || boot?.initialRebuildCount !== 1
    || boot?.priorStrategyConstructionCount !== 0) {
    reasons.push('forced-off gate lacks the strict portable initial boot');
  }
  if (pageLifecycle?.rebuildCount !== 1
    || pageLifecycle?.strategyConstructionCount !== 1
    || !sameArray(pageLifecycle?.constructedStrategyIds, [STANDALONE_MODE_ID])
    || pageLifecycle?.selectedStrategyId !== STANDALONE_MODE_ID
    || pageLifecycle?.strictStandaloneBoot !== true) {
    reasons.push('forced-off gate constructed more than its sole portable strategy');
  }
  if (capture?.environment?.threeRevision !== THREE_REVISION
    || capture?.environment?.rendererBackend !== 'WebGPUBackend'
    || capture?.environment?.reversedDepth !== true
    || capture?.environment?.rendererReversedDepthBuffer !== true
    || capture?.environment?.timestampAvailable !== true
    || capture?.environment?.indirectFirstInstanceAvailable !== true
    || capture?.environment?.crossOriginIsolated !== true
    || capture?.environment?.webgpuUncapturedErrorCount !== 0
    || capture?.environment?.webgpuValidationErrorCount !== 0
    || capture?.environment?.webgpuDeviceLossCount !== 0) {
    reasons.push('forced-off gate WebGPU environment is unavailable, dirty, or changed');
  }
  if (!Array.isArray(capture?.webgpuUncapturedErrors)
    || capture.webgpuUncapturedErrors.length !== 0
    || !Array.isArray(capture?.webgpuDeviceLosses)
    || capture.webgpuDeviceLosses.length !== 0) {
    reasons.push('forced-off gate captured a WebGPU error or device loss');
  }
  if (capture?.workload?.scenarioSeed !== SCENARIO_SEED
    || capture?.workload?.geometryFixtures?.bucketCount !== BUCKET_COUNT
    || capture?.workload?.scenario?.seed !== SCENARIO_SEED
    || capture?.workload?.scenario?.objectCount !== OBJECT_COUNT
    || capture?.workload?.scenario?.bucketCount !== BUCKET_COUNT
    || capture?.workload?.scenario?.visibilityFraction !== 0.99
    || capture?.workload?.scenario?.layout !== LAYOUT) {
    reasons.push('forced-off gate workload differs from the fixed high-visibility cell');
  }
  if (capture?.trialExecutionState?.phase !== 'idle'
    || capture?.trialExecutionState?.trialError !== null
    || capture?.trialExecutionState?.rowCount !== 0
    || capture?.trialExecutionState?.summary !== null) {
    reasons.push('forced-off gate unexpectedly entered or retained trial timing state');
  }
  return deepFreeze({
    schemaVersion: 1,
    kind: 'first-instance-startup-endurance-forced-feature-off-validation',
    pass: reasons.length === 0,
    reasons: [...new Set(reasons)],
  });
}

async function captureForcedFeatureOffGate(page) {
  return page.evaluate(async (options) => {
    const bench = window.__WEBGPU_BENCH__;
    const gate = await bench.runFirstInstanceLiveForcedFeatureOffGate(options);
    return {
      gate,
      environment: bench.environment,
      workload: await bench.fingerprintWorkload(),
      webgpuUncapturedErrors: bench.webgpuUncapturedErrors,
      webgpuDeviceLosses: bench.webgpuDeviceLosses,
      trialExecutionState: {
        phase: bench.phase,
        trialError: bench.trialError,
        rowCount: bench.rows.length,
        summary: bench.summary,
      },
    };
  }, {
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    visibilityFraction: 0.99,
    scenarioSeed: SCENARIO_SEED,
  });
}

async function captureStandaloneConfiguration(page) {
  return page.evaluate(async () => {
    const bench = window.__WEBGPU_BENCH__;
    return {
      initialPageBoot: bench.initialPageBoot,
      pageConstructionLifecycle: bench.pageConstructionLifecycle,
      selectedConfig: bench.selectedConfig(),
      environment: bench.environment,
      shaderEvidence: bench.firstInstanceShaderEvidence,
      timestampPoolPreprime: bench.timestampPoolPreprime,
      timestampPoolDiagnostics: bench.timestampPoolDiagnostics,
      strategyLifecycle: bench.strategyLifecycle,
      strategyDiagnostics: bench.strategyDiagnostics,
      cacheDiagnostics: bench.cacheDiagnostics(),
      geometryManifest: bench.geometryManifest,
      scenarioManifest: bench.scenarioManifest,
      workload: await bench.fingerprintWorkload(),
      webgpuUncapturedErrors: bench.webgpuUncapturedErrors,
      webgpuDeviceLosses: bench.webgpuDeviceLosses,
      trialExecutionState: {
        phase: bench.phase,
        trialError: bench.trialError,
        rowCount: bench.rows.length,
        summary: bench.summary,
      },
    };
  });
}

function validateCrossSessionEnvironment(captures) {
  const environmentIdentities = captures.map((capture) => {
    const { performanceNowQuantumMs: _quantum, ...stableEnvironment } = capture.environment;
    return JSON.stringify(stableEnvironment);
  });
  const geometryIdentities = captures.map(
    (capture) => JSON.stringify(capture.geometryManifest),
  );
  const workloadIdentityByVisibility = new Map();
  for (const capture of captures) {
    const visibility = String(capture.selectedConfig.visibilityFraction);
    const identity = JSON.stringify({
      scenarioManifest: capture.scenarioManifest,
      workload: capture.workload,
    });
    const identities = workloadIdentityByVisibility.get(visibility) ?? [];
    identities.push(identity);
    workloadIdentityByVisibility.set(visibility, identities);
  }
  const workloadStrata = Object.fromEntries([...workloadIdentityByVisibility].map(
    ([visibility, identities]) => [visibility, {
      observationCount: identities.length,
      exactIdentityCount: new Set(identities).size,
      pass: identities.length === 16 && new Set(identities).size === 1,
    }],
  ));
  const pass = captures.length === STARTUP_ENDURANCE_SESSION_COUNT
    && new Set(environmentIdentities).size === 1
    && new Set(geometryIdentities).size === 1
    && sameArray(Object.keys(workloadStrata).sort(), ['0.2', '0.99'])
    && Object.values(workloadStrata).every((stratum) => stratum.pass);
  return deepFreeze({
    schemaVersion: 1,
    kind: 'first-instance-startup-endurance-cross-session-environment',
    observationCount: captures.length,
    pass,
    environmentExactIdentityCount: new Set(environmentIdentities).size,
    geometryExactIdentityCount: new Set(geometryIdentities).size,
    workloadStrata,
  });
}

function parseArguments(arguments_) {
  if (arguments_.length !== 0) {
    throw new Error(
      'Usage: node scripts/probe-first-instance-standalone-startup-endurance.mjs',
    );
  }
}

export async function runFirstInstanceStandaloneStartupEnduranceProbe() {
  const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const runTimestamp = new Date().toISOString().replaceAll(':', '-').replaceAll('.', '-');
  const runId = `first-instance-standalone-startup-endurance-${runTimestamp}-`
    + randomBytes(4).toString('hex');
  const resultRoot = path.join(
    projectRoot,
    'results',
    'development',
    'first-instance-standalone-startup-endurance',
  );
  const runDirectory = path.join(resultRoot, runId);
  const runStartedMonotonic = performance.now();
  const startedAt = new Date().toISOString();
  let server = null;
  let viteRuntimeGuard = null;
  let browserManager = null;
  let artifactStore = null;
  let terminationSignal = null;
  let completionCommitted = false;
  let activeFailureSnapshot = null;
  let abortedBrowserLifecycle = null;
  let failureTcpCapture = null;
  let sourceProvenanceStart = null;
  let dependencyClosureStart = null;
  let attemptRootClaim = null;
  let forcedFeatureOffGateRecord = null;
  const completedSessions = [];
  const configurationCaptures = [];
  const tcpArtifacts = [];
  const tcpReadinessEvidence = [];

  const signalHandler = (signal) => {
    if (completionCommitted || terminationSignal !== null) return;
    terminationSignal = signal;
    process.exitCode = signal === 'SIGINT' ? 130 : 143;
  };
  const onSigint = () => signalHandler('SIGINT');
  const onSigterm = () => signalHandler('SIGTERM');
  process.on('SIGINT', onSigint);
  process.on('SIGTERM', onSigterm);
  const requireActive = (boundary) => assertStartupEnduranceNotInterrupted(
    terminationSignal,
    boundary,
  );

  try {
    attemptRootClaim = await claimStartupEnduranceAttemptRoot(resultRoot);
    await mkdir(runDirectory, { recursive: false });
    for (const relativeDirectory of ['sessions', 'network', 'journal']) {
      await mkdir(path.join(runDirectory, relativeDirectory), { recursive: false });
    }
    artifactStore = new IncrementalArtifactStore(runDirectory);
    const attemptRootClaimArtifact = await artifactStore.json(
      'attempt-root-claim.json',
      attemptRootClaim,
    );
    await artifactStore.journal('attempt-root-claimed', {
      artifact: attemptRootClaimArtifact,
      priorInvocationAllowed: false,
    });
    requireActive('after-attempt-root-claim');

    const standalonePlan = buildFirstInstanceStandaloneDeploymentPlan({ runId });
    validateFirstInstanceStandaloneDeploymentPlan(standalonePlan, { runId });
    const probePlan = buildStartupEnduranceProbePlan({ runId, standalonePlan });
    const planArtifact = await artifactStore.json('plan.json', probePlan);
    await artifactStore.journal('plan-committed', { artifact: planArtifact });

    [sourceProvenanceStart, dependencyClosureStart] = await Promise.all([
      collectSourceProvenance(projectRoot),
      collectExecutionDependencyClosure(projectRoot),
    ]);
    requireCondition(sourceProvenanceStart.status === 'available'
      && sourceProvenanceStart.captureStable === true
      && sourceProvenanceStart.dirty === false
      && sourceProvenanceStart.packageLockTracked === true,
    'Startup endurance probe requires clean, stable source provenance.',
    sourceProvenanceStart);
    requireCondition(dependencyClosureStart?.schemaVersion === 1
      && dependencyClosureStart?.kind === 'installed-execution-dependency-closure'
      && dependencyClosureStart.fileCount > 0
      && SHA256_PATTERN.test(dependencyClosureStart.sha256),
    'Startup endurance dependency closure is unavailable.', dependencyClosureStart);
    const identityStartArtifact = await artifactStore.json('execution-identity-start.json', {
      schemaVersion: 1,
      kind: 'first-instance-startup-endurance-execution-identity-start',
      runId,
      capturedAt: new Date().toISOString(),
      sourceProvenance: sourceProvenanceStart,
      executionDependencyClosure: dependencyClosureStart,
    });
    await artifactStore.journal('execution-identity-start-committed', {
      artifact: identityStartArtifact,
    });

    const executablePath = await findBrowser();
    requireActive('before-vite-start');
    viteRuntimeGuard = await createCandidateViteRuntimeGuard(projectRoot);
    server = await createServer({
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
    await withDeadline(server.listen(), SERVER_OPERATION_TIMEOUT_MS, 'startup Vite listen');
    const url = server.resolvedUrls?.local?.[0];
    requireCondition(Boolean(url), 'Startup Vite server did not expose a local URL.');
    requireActive('after-vite-start');
    browserManager = new StartupBrowserManager({
      executablePath,
      runStartedMonotonic,
    });

    process.stdout.write('[startup forced-feature-off gate] portable 0.99->0.2\n');
    const gateTcpPre = await captureWindowsTcpState('forced-feature-off-gate-pre-launch');
    const gateTcpPreArtifact = await artifactStore.json(
      'network/forced-feature-off-gate-pre-launch.json',
      gateTcpPre,
    );
    tcpArtifacts.push(gateTcpPreArtifact);
    tcpReadinessEvidence.push(tcpEvidenceRecord(gateTcpPre, gateTcpPreArtifact));
    await artifactStore.journal('tcp-pre-launch-committed', {
      role: 'forced-feature-off-gate',
      assessmentPass: gateTcpPre.assessment.pass,
      artifact: gateTcpPreArtifact,
    });
    requireCondition(gateTcpPre.assessment.pass === true,
      'Forced-feature-off TCP readiness gate failed before browser launch.',
      gateTcpPre.assessment);
    requireActive('forced-feature-off-before-browser-launch');
    const gateState = await browserManager.launch({
      role: 'forced-feature-off-gate',
      matrixOrdinal: 1,
    });
    await artifactStore.journal('browser-launched', {
      role: 'forced-feature-off-gate',
      browserInstanceSerial: gateState.record.browserInstanceSerial,
    });
    const gateUrl = `${url}${firstInstanceStandaloneBootSearch({
      laneId: 'portable',
      visibilityOrder: [0.99, 0.2],
    })}`;
    const gatePage = await browserManager.openOnlyPage(gateState, gateUrl);
    const gateCapture = await withDeadline(
      captureForcedFeatureOffGate(gatePage),
      BROWSER_OPERATION_TIMEOUT_MS,
      'startup forced-feature-off gate capture',
    );
    const gateValidation = validateStartupForcedFeatureOffGate(gateCapture);
    requireCondition(gateValidation.pass === true,
      'Startup forced-feature-off gate failed closed.', gateValidation);
    requireCondition(validateStartupPageDiagnostics(
      gateState.pageDiagnostics.state,
    ).pass === true,
    'Startup forced-feature-off gate emitted a page diagnostic error.',
    gateState.pageDiagnostics.state);
    const gateBrowserLifecycle = await browserManager.close(gateState);
    requireCondition(validateStartupPageDiagnostics(
      gateState.pageDiagnostics.state,
    ).pass === true,
    'Startup forced-feature-off gate emitted a late page diagnostic error.',
    gateState.pageDiagnostics.state);
    await browserManager.waitAfterDisconnect(gateBrowserLifecycle);
    const gateTcpPost = await captureWindowsTcpState(
      'forced-feature-off-gate-post-disconnect',
    );
    const gateTcpPostArtifact = await artifactStore.json(
      'network/forced-feature-off-gate-post-disconnect.json',
      gateTcpPost,
    );
    tcpArtifacts.push(gateTcpPostArtifact);
    tcpReadinessEvidence.push(tcpEvidenceRecord(gateTcpPost, gateTcpPostArtifact));
    await artifactStore.journal('tcp-post-disconnect-committed', {
      role: 'forced-feature-off-gate',
      assessmentPass: gateTcpPost.assessment.pass,
      artifact: gateTcpPostArtifact,
    });
    const forcedFeatureOffGateArtifact = await artifactStore.json(
      'forced-feature-off-gate.json',
      {
        schemaVersion: 1,
        kind: 'first-instance-startup-endurance-forced-feature-off-gate',
        executionMode: STARTUP_ENDURANCE_EXECUTION_MODE,
        analysisEligible: false,
        runId,
        browserLifecycle: gateBrowserLifecycle,
        capture: gateCapture,
        validation: gateValidation,
        pageDiagnostics: clone(gateState.pageDiagnostics.state),
        tcpPreLaunch: {
          artifact: gateTcpPreArtifact,
          assessment: gateTcpPre.assessment,
        },
        tcpPostDisconnect: {
          artifact: gateTcpPostArtifact,
          assessment: gateTcpPost.assessment,
        },
        trialExecutionInvoked: false,
        timingCaptured: false,
        efficacyEvaluated: false,
      },
    );
    forcedFeatureOffGateRecord = {
      browserInstanceSerial: gateBrowserLifecycle.browserInstanceSerial,
      validationPass: gateValidation.pass,
      tcpPreLaunchPass: gateTcpPre.assessment.pass,
      tcpPostDisconnectPass: gateTcpPost.assessment.pass,
      artifact: forcedFeatureOffGateArtifact,
    };
    await artifactStore.journal(
      'forced-feature-off-gate-committed',
      forcedFeatureOffGateRecord,
    );
    requireCondition(gateTcpPost.assessment.pass === true,
      'Forced-feature-off post-disconnect TCP readiness gate failed closed.',
      gateTcpPost.assessment);
    requireActive('after-forced-feature-off-gate');

    for (let index = 0; index < probePlan.schedule.sessions.length; index += 1) {
      requireActive(`session-${index + 1}-start`);
      const canonicalSession = probePlan.schedule.sessions[index];
      const ordinal = String(index + 1).padStart(2, '0');
      process.stdout.write(
        `[startup ${index + 1}/${STARTUP_ENDURANCE_SESSION_COUNT}] `
          + `${canonicalSession.assignedLaneId} ${canonicalSession.visibilityOrder.join('->')}\n`,
      );

      const tcpPre = await captureWindowsTcpState(`session-${ordinal}-pre-launch`);
      const tcpPreArtifact = await artifactStore.json(
        `network/session-${ordinal}-pre-launch.json`,
        tcpPre,
      );
      tcpArtifacts.push(tcpPreArtifact);
      tcpReadinessEvidence.push(tcpEvidenceRecord(tcpPre, tcpPreArtifact));
      await artifactStore.journal('tcp-pre-launch-committed', {
        sessionId: canonicalSession.sessionId,
        assessmentPass: tcpPre.assessment.pass,
        artifact: tcpPreArtifact,
      });
      requireCondition(tcpPre.assessment.pass === true,
        `Session ${index + 1} TCP readiness gate failed before browser launch.`,
        tcpPre.assessment);
      requireActive(`session-${index + 1}-before-browser-launch`);

      const state = await browserManager.launch({
        role: 'standalone-startup-endurance-session',
        matrixOrdinal: canonicalSession.matrixOrdinal,
        session: canonicalSession,
      });
      await artifactStore.journal('browser-launched', {
        sessionId: canonicalSession.sessionId,
        browserInstanceSerial: state.record.browserInstanceSerial,
      });
      const sessionUrl = `${url}${firstInstanceStandaloneBootSearch({
        laneId: canonicalSession.assignedLaneId,
        visibilityOrder: canonicalSession.visibilityOrder,
      })}`;
      const page = await browserManager.openOnlyPage(state, sessionUrl);
      const configuration = await withDeadline(
        captureStandaloneConfiguration(page),
        BROWSER_OPERATION_TIMEOUT_MS,
        `startup session ${index + 1} configuration capture`,
      );
      const configurationValidation = validateStartupEnduranceConfiguration(
        configuration,
        canonicalSession,
      );
      requireCondition(configurationValidation.pass === true,
        `Startup session ${index + 1} configuration failed closed.`,
        configurationValidation);
      requireCondition(validateStartupPageDiagnostics(
        state.pageDiagnostics.state,
      ).pass === true,
      `Startup session ${index + 1} emitted a page diagnostic error.`,
      state.pageDiagnostics.state);

      const browserLifecycle = await browserManager.close(state);
      requireCondition(validateStartupPageDiagnostics(
        state.pageDiagnostics.state,
      ).pass === true,
      `Startup session ${index + 1} emitted a late page diagnostic error.`,
      state.pageDiagnostics.state);
      await browserManager.waitAfterDisconnect(browserLifecycle);
      const tcpPost = await captureWindowsTcpState(`session-${ordinal}-post-disconnect`);
      const tcpPostArtifact = await artifactStore.json(
        `network/session-${ordinal}-post-disconnect.json`,
        tcpPost,
      );
      tcpArtifacts.push(tcpPostArtifact);
      tcpReadinessEvidence.push(tcpEvidenceRecord(tcpPost, tcpPostArtifact));
      await artifactStore.journal('tcp-post-disconnect-committed', {
        sessionId: canonicalSession.sessionId,
        assessmentPass: tcpPost.assessment.pass,
        artifact: tcpPostArtifact,
      });

      const sessionArtifact = await artifactStore.json(`sessions/session-${ordinal}.json`, {
        schemaVersion: 1,
        kind: 'first-instance-standalone-startup-endurance-session',
        executionMode: STARTUP_ENDURANCE_EXECUTION_MODE,
        analysisEligible: false,
        runId,
        sessionOrdinal: index + 1,
        canonicalSession,
        browserLifecycle,
        configuration,
        configurationValidation,
        pageDiagnostics: clone(state.pageDiagnostics.state),
        tcpPreLaunch: {
          artifact: tcpPreArtifact,
          assessment: tcpPre.assessment,
        },
        tcpPostDisconnect: {
          artifact: tcpPostArtifact,
          assessment: tcpPost.assessment,
        },
        trialExecutionInvoked: false,
        timingCaptured: false,
        efficacyEvaluated: false,
      });
      const sessionRecord = {
        sessionId: canonicalSession.sessionId,
        globalSessionIndex: canonicalSession.globalSessionIndex,
        browserInstanceSerial: browserLifecycle.browserInstanceSerial,
        configurationPass: configurationValidation.pass,
        tcpPreLaunchPass: tcpPre.assessment.pass,
        tcpPostDisconnectPass: tcpPost.assessment.pass,
        artifact: sessionArtifact,
      };
      completedSessions.push(sessionRecord);
      configurationCaptures.push(configuration);
      await artifactStore.journal('session-committed', sessionRecord);
      requireCondition(tcpPost.assessment.pass === true,
        `Session ${index + 1} post-disconnect TCP readiness gate failed closed.`,
        tcpPost.assessment);
      requireActive(`session-${index + 1}-committed`);
    }

    requireActive('after-session-schedule');
    requireCondition(browserManager.active === null,
      'A startup endurance browser remained active after the schedule.');
    requireCondition(tcpArtifacts.length === STARTUP_ENDURANCE_BROWSER_COUNT * 2
      && tcpReadinessEvidence.length === STARTUP_ENDURANCE_BROWSER_COUNT * 2
      && tcpReadinessEvidence.every((record) => record.assessment.pass === true),
    'Startup endurance probe lacks its exact passing TCP snapshot schedule.', {
      expected: STARTUP_ENDURANCE_BROWSER_COUNT * 2,
      artifactCount: tcpArtifacts.length,
      evidenceCount: tcpReadinessEvidence.length,
    });
    const lifecycleValidation = validateStandaloneBrowserLifecycleChain(
      browserManager.records,
      { requireTerminalDelay: true },
    );
    requireCondition(lifecycleValidation.pass === true
      && browserManager.records.length === STARTUP_ENDURANCE_BROWSER_COUNT,
    'Startup endurance browser lifecycle chain failed.', lifecycleValidation);
    const crossSessionEnvironment = validateCrossSessionEnvironment(configurationCaptures);
    requireCondition(crossSessionEnvironment.pass === true,
      'Startup endurance environment/workload identity drifted across sessions.',
      crossSessionEnvironment);

    requireActive('before-final-identity-capture');
    const [sourceProvenanceEnd, dependencyClosureEnd] = await Promise.all([
      collectSourceProvenance(projectRoot),
      collectExecutionDependencyClosure(projectRoot),
    ]);
    requireActive('after-final-identity-capture');
    const sourceMatched = sourceProvenanceMatches(
      sourceProvenanceStart,
      sourceProvenanceEnd,
    );
    const dependencyMatched = executionDependencyClosuresMatch(
      dependencyClosureStart,
      dependencyClosureEnd,
    );
    requireCondition(sourceMatched === true,
      'Tracked source changed during startup endurance probe.');
    requireCondition(dependencyMatched === true,
      'Installed dependency bytes changed during startup endurance probe.');
    const identityEndArtifact = await artifactStore.json('execution-identity-end.json', {
      schemaVersion: 1,
      kind: 'first-instance-startup-endurance-execution-identity-end',
      runId,
      capturedAt: new Date().toISOString(),
      sourceProvenance: sourceProvenanceEnd,
      executionDependencyClosure: dependencyClosureEnd,
      sourceMatchesStart: sourceMatched,
      executionDependencyClosureMatchesStart: dependencyMatched,
    });
    await artifactStore.journal('execution-identity-end-committed', {
      artifact: identityEndArtifact,
      sourceMatched,
      dependencyMatched,
    });
    requireActive('after-final-identity-commit');

    requireActive('before-vite-close');
    await withDeadline(server.close(), SERVER_OPERATION_TIMEOUT_MS, 'startup Vite close');
    server = null;
    requireActive('after-vite-close');
    let runtimeAudit;
    requireActive('before-runtime-audit-finalize');
    try {
      runtimeAudit = await withDeadline(
        viteRuntimeGuard.finalize(),
        SERVER_OPERATION_TIMEOUT_MS,
        'startup Vite runtime audit',
      );
    } finally {
      await withDeadline(
        viteRuntimeGuard.dispose(),
        SERVER_OPERATION_TIMEOUT_MS,
        'startup Vite runtime guard disposal',
      ).catch(() => undefined);
      viteRuntimeGuard = null;
    }
    requireActive('after-runtime-audit-finalize');
    requireCondition(runtimeAudit?.entryHtml?.successfulResponseCount
      === STARTUP_ENDURANCE_BROWSER_COUNT,
    'Startup probe served an unexpected number of entry documents.', runtimeAudit?.entryHtml);
    requireCondition(runtimeAudit?.modules?.some(
      (record) => record.sourceRelativePath
        === 'src/strategies/live-first-instance-standalone.js',
    ), 'Startup runtime audit lacks the standalone strategy module.');
    const runtimeArtifact = await artifactStore.json('vite-runtime-audit.json', runtimeAudit);
    await artifactStore.journal('vite-runtime-audit-committed', {
      artifact: runtimeArtifact,
      modulesSha256: runtimeAudit.modulesSha256,
    });
    requireActive('after-runtime-audit-commit');

    requireActive('before-completion-manifest-build');
    const manifest = {
      schemaVersion: 1,
      kind: 'first-instance-standalone-startup-endurance-completion',
      executionMode: STARTUP_ENDURANCE_EXECUTION_MODE,
      analysisEligible: false,
      scope: 'one forced-off gate plus startup/configuration endurance only; '
        + 'excluded from every efficacy decision',
      runId,
      startedAt,
      completedAt: new Date().toISOString(),
      elapsedMs: performance.now() - runStartedMonotonic,
      status: 'complete',
      attemptRootClaim,
      result: {
        scheduledForcedFeatureOffGateCount: 1,
        completedForcedFeatureOffGateCount: forcedFeatureOffGateRecord === null ? 0 : 1,
        forcedFeatureOffGatePassed: forcedFeatureOffGateRecord?.validationPass === true,
        expectedBrowserProcessCount: STARTUP_ENDURANCE_BROWSER_COUNT,
        observedBrowserProcessCount: browserManager.records.length,
        scheduledSessionCount: STARTUP_ENDURANCE_SESSION_COUNT,
        completedSessionCount: completedSessions.length,
        allStartupConfigurationsPassed: true,
        allTcpReadinessGatesPassed: true,
        expectedTcpSnapshotCount: STARTUP_ENDURANCE_BROWSER_COUNT * 2,
        observedTcpSnapshotCount: tcpReadinessEvidence.length,
      },
      executionPolicy: probePlan.executionPolicy,
      exclusionPolicy: probePlan.exclusionPolicy,
      networkReadinessPolicy: probePlan.networkReadinessPolicy,
      tcpReadinessEvidence,
      executionIdentity: {
        sourceProvenanceStart,
        sourceProvenanceEnd,
        sourceMatched,
        executionDependencyClosureStart: dependencyClosureStart,
        executionDependencyClosureEnd: dependencyClosureEnd,
        executionDependencyClosureMatched: dependencyMatched,
      },
      lifecycleValidation,
      crossSessionEnvironment,
      browserLifecycles: clone(browserManager.records),
      numericalDecision: null,
      efficacyConclusion: null,
      artifacts: {
        attemptRootClaim: attemptRootClaimArtifact,
        plan: planArtifact,
        executionIdentityStart: identityStartArtifact,
        executionIdentityEnd: identityEndArtifact,
        forcedFeatureOffGate: forcedFeatureOffGateRecord,
        sessions: completedSessions,
        tcpSnapshots: tcpArtifacts,
        viteRuntimeAudit: runtimeArtifact,
        journal: clone(artifactStore.journalArtifacts),
      },
    };
    requireActive('immediately-before-completion-manifest-commit');
    const manifestArtifact = artifactStore.jsonSynchronous('manifest.json', manifest);
    completionCommitted = true;
    process.stdout.write(
      'Standalone startup endurance probe complete (analysis-ineligible).\n'
        + `  directory: ${runDirectory}\n`
        + `  manifest: ${path.join(runDirectory, manifestArtifact.path)}\n`
        + '  trial timing: not invoked\n'
        + '  efficacy analysis: forbidden\n',
    );
    return Object.freeze({ runDirectory, runId, manifestArtifact });
  } catch (error) {
    activeFailureSnapshot = await browserManager?.captureActiveFailureSnapshot()
      .catch(() => null) ?? null;
    abortedBrowserLifecycle = await browserManager?.forceClose(
      terminationSignal ?? 'failed-startup-endurance-probe',
    ).catch(() => null) ?? null;
    if (process.platform === 'win32') {
      try {
        failureTcpCapture = await captureWindowsTcpState('failed-probe-post-cleanup');
        if (artifactStore !== null) {
          const artifact = await artifactStore.json(
            'network/failure-post-cleanup.json',
            failureTcpCapture,
          );
          tcpArtifacts.push(artifact);
          tcpReadinessEvidence.push(tcpEvidenceRecord(failureTcpCapture, artifact));
        }
      } catch (tcpError) {
        failureTcpCapture = {
          captureFailed: true,
          error: serializedError(tcpError),
        };
      }
    }
    if (server !== null) {
      await withDeadline(
        server.close(),
        SERVER_OPERATION_TIMEOUT_MS,
        'failed startup Vite close',
      ).catch(() => undefined);
      server = null;
    }
    if (viteRuntimeGuard !== null) {
      await withDeadline(
        viteRuntimeGuard.dispose(),
        SERVER_OPERATION_TIMEOUT_MS,
        'failed startup Vite runtime guard disposal',
      ).catch(() => undefined);
      viteRuntimeGuard = null;
    }
    if (artifactStore !== null) {
      await artifactStore.json('failure.json', {
        schemaVersion: 1,
        kind: 'first-instance-standalone-startup-endurance-failure',
        executionMode: STARTUP_ENDURANCE_EXECUTION_MODE,
        analysisEligible: false,
        scope: 'failed closed; one attempt; no retry, timing, replacement, or efficacy',
        runId,
        startedAt,
        failedAt: new Date().toISOString(),
        signal: terminationSignal,
        error: serializedError(error),
        completedSessionCount: completedSessions.length,
        completedSessions,
        forcedFeatureOffGate: forcedFeatureOffGateRecord,
        activeFailureSnapshot,
        abortedBrowserLifecycle,
        failureTcpCapture,
        tcpReadinessEvidence,
        browserLifecycles: clone(browserManager?.records ?? []),
        attemptRootClaim,
        sourceProvenanceStart,
        executionDependencyClosureStart: dependencyClosureStart,
        tcpArtifacts,
        journalArtifacts: clone(artifactStore.journalArtifacts),
        executionPolicy: {
          attemptCount: 1,
          retryCount: 0,
          replacementAllowed: false,
          trialExecutionInvoked: false,
          timingCaptured: false,
          efficacyEvaluated: false,
        },
      }).catch(() => undefined);
    }
    throw error;
  } finally {
    process.removeListener('SIGINT', onSigint);
    process.removeListener('SIGTERM', onSigterm);
    if (!completionCommitted) {
      await browserManager?.forceClose('final-startup-probe-cleanup').catch(() => undefined);
    }
  }
}

const invokedPath = process.argv[1] === undefined
  ? null
  : pathToFileURL(path.resolve(process.argv[1])).href;
if (invokedPath === import.meta.url) {
  try {
    parseArguments(process.argv.slice(2));
    await runFirstInstanceStandaloneStartupEnduranceProbe();
  } catch (error) {
    process.stderr.write(`${error?.stack ?? error}\n`);
    process.exitCode ||= 1;
  }
}
