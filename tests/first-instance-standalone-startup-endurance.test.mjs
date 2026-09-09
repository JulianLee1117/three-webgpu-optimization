import assert from 'node:assert/strict';
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  STARTUP_ENDURANCE_BROWSER_ARGS,
  STARTUP_ENDURANCE_BROWSER_COUNT,
  STARTUP_ENDURANCE_EXECUTION_MODE,
  STARTUP_ENDURANCE_MINIMUM_FREE_DYNAMIC_PORTS,
  STARTUP_ENDURANCE_QUARTET_COUNT,
  STARTUP_ENDURANCE_SESSION_COUNT,
  STARTUP_ENDURANCE_TCP_QUIET_WINDOW_MS,
  assessStartupTcpReadiness,
  assertStartupEnduranceNotInterrupted,
  buildStartupEnduranceProbePlan,
  claimStartupEnduranceAttemptRoot,
  closeStartupBrowserStateAfterAbort,
  selectStartupEnduranceSessions,
  validateStartupEnduranceConfiguration,
  validateStartupPageDiagnostics,
} from '../scripts/probe-first-instance-standalone-startup-endurance.mjs';
import {
  buildFirstInstanceStandaloneDeploymentPlan,
} from '../src/benchmark/first-instance-standalone-deployment-plan.js';

const RUN_ID = 'startup-endurance-unit';
const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId: RUN_ID });

function validTcpSnapshot({
  capturedAt = '2026-09-03T12:00:00.000Z',
  latestTcpip4231 = {
    providerName: 'Tcpip',
    timeCreatedUtc: '2026-09-03T11:49:59.999Z',
    recordId: 42,
    logName: 'System',
  },
  uniqueLocalPortCount = 100,
} = {}) {
  const numberOfPorts = 16_384;
  return {
    schemaVersion: 1,
    kind: 'windows-tcp-dynamic-port-readiness-snapshot',
    capturedAt,
    eventQuery: {
      status: 'available',
      logName: 'System',
      providerName: 'Tcpip',
      systemReadProbeRecordId: 99,
      systemReadProbeTimeCreatedUtc: '2026-09-03T12:00:00.000Z',
      eventId: 4231,
      maximumEventsScanned: 1,
    },
    settings: [{
      settingName: 'Internet',
      dynamicPortRangeStartPort: 49_152,
      dynamicPortRangeNumberOfPorts: numberOfPorts,
    }],
    ranges: [{
      startPort: 49_152,
      numberOfPorts,
      endPort: 65_535,
      connectionCount: uniqueLocalPortCount + 10,
      uniqueLocalPortCount,
      timeWaitConnectionCount: uniqueLocalPortCount,
      timeWaitUniqueLocalPortCount: uniqueLocalPortCount,
      freeUniqueLocalPortMargin: numberOfPorts - uniqueLocalPortCount,
    }],
    latestTcpip4231,
  };
}

function validConfiguration(canonicalSession) {
  const lane = canonicalSession.assignedLaneId;
  const absentLane = canonicalSession.absentLaneId;
  const addressMode = lane === 'portable' ? 'bucket-base' : 'indirect-first-instance';
  const ledger = {
    schemaVersion: 1,
    kind: 'first-instance-live-standalone-production-resource-ledger',
    selectedLane: lane,
    absentLane,
    absentLaneConstructed: false,
    constructedLaneIds: [lane],
    constructedAddressModes: [addressMode],
    constructedLaneCount: 1,
    indirectCommandBufferCount: 1,
    computeNodeCount: 2,
    materialCount: 1,
    meshCount: 1,
    bundleCount: 1,
    addressMode,
    hasBucketBaseVertexAttribute: lane === 'portable',
  };
  const geometryManifest = { schemaVersion: 1, bucketCount: 32, sha256: 'a'.repeat(64) };
  const scenarioManifest = {
    schemaVersion: 1,
    seed: 0xb1ad_2026,
    objectCount: 65_536,
    bucketCount: 32,
    visibilityFraction: canonicalSession.visibilityOrder[0],
    layout: 'baseline',
    sha256: 'b'.repeat(64),
  };
  return {
    initialPageBoot: {
      schemaVersion: 1,
      kind: 'first-instance-standalone-initial-page-boot',
      modeId: 'first-instance-live-standalone-deployment',
      laneId: lane,
      visibilityOrder: [...canonicalSession.visibilityOrder],
      objectCount: 65_536,
      bucketCount: 32,
      layout: 'baseline',
      initialRebuildCount: 1,
      priorStrategyConstructionCount: 0,
    },
    pageConstructionLifecycle: {
      schemaVersion: 1,
      kind: 'benchmark-page-strategy-construction-lifecycle',
      rebuildCount: 1,
      strategyConstructionCount: 1,
      constructedStrategyIds: ['first-instance-live-standalone-deployment'],
      selectedStrategyId: 'first-instance-live-standalone-deployment',
      strictStandaloneBoot: true,
    },
    selectedConfig: {
      strategyId: 'first-instance-live-standalone-deployment',
      objectCount: 65_536,
      bucketCount: 32,
      visibilityFraction: canonicalSession.visibilityOrder[0],
      layout: 'baseline',
      laneId: lane,
      visibilityOrder: [...canonicalSession.visibilityOrder],
    },
    environment: {
      threeRevision: '185',
      rendererBackend: 'WebGPUBackend',
      reversedDepth: true,
      rendererReversedDepthBuffer: true,
      timestampAvailable: true,
      indirectFirstInstanceAvailable: true,
      crossOriginIsolated: true,
      webgpuUncapturedErrorCount: 0,
      webgpuValidationErrorCount: 0,
      webgpuDeviceLossCount: 0,
      performanceNowQuantumMs: 0.005,
      viewport: { width: 1280, height: 720, devicePixelRatio: 1 },
    },
    shaderEvidence: null,
    timestampPoolPreprime: {
      schemaVersion: 1,
      kind: 'three-r185-timestamp-pool-preprime',
      addedTimestampUidCount: { render: 1, compute: 1 },
    },
    geometryManifest,
    scenarioManifest,
    workload: {
      scenarioSeed: 0xb1ad_2026,
      geometryFixtures: structuredClone(geometryManifest),
      scenario: structuredClone(scenarioManifest),
    },
    strategyLifecycle: {
      schemaVersion: 1,
      kind: 'first-instance-live-standalone-static-resource-lifecycle',
      pass: true,
      selectedLane: lane,
      absentLane,
      absentLaneConstructed: false,
      lanesConstructed: 1,
      primed: true,
      activeVisibilityFraction: canonicalSession.visibilityOrder[0],
      scenarioSwitchSerial: 0,
      shaderObservationSerial: 0,
      bundleRecordCallbackCount: 1,
      commandBuffer: { attributeId: 1, byteOffset: 0, firstOffset: 0 },
      computeNodeIds: [2, 3],
      productionResourceLedger: ledger,
    },
    strategyDiagnostics: {
      schemaVersion: 1,
      kind: 'first-instance-live-standalone-deployment',
      pass: true,
      selectedLane: lane,
      absentLane,
      addressMode,
      configuredDrawCommands: 32,
      configuredRenderObjects: 1,
      configuredComputeDispatches: 2,
      configuredComputeSubmissions: 1,
      laneCommandBufferCount: 1,
    },
    webgpuUncapturedErrors: [],
    webgpuDeviceLosses: [],
    trialExecutionState: {
      phase: 'idle',
      trialError: null,
      rowCount: 0,
      summary: null,
    },
  };
}

test('startup endurance selects the exact balanced first eight matrix-1 quartets', () => {
  const selection = selectStartupEnduranceSessions(plan);
  assert.equal(selection.quartets.length, STARTUP_ENDURANCE_QUARTET_COUNT);
  assert.equal(selection.sessions.length, STARTUP_ENDURANCE_SESSION_COUNT);
  assert.deepEqual(selection.quartets.map(({ quartetCode }) => quartetCode), [
    'AX', 'AY', 'BX', 'BY', 'AX', 'AY', 'BX', 'BY',
  ]);
  assert.deepEqual(selection.balance.quartetCodes, { AX: 2, AY: 2, BX: 2, BY: 2 });
  assert.deepEqual(selection.balance.lanes, { portable: 16, feature: 16 });
  assert.deepEqual(selection.balance.visibilityOrders, { H: 16, L: 16 });
  assert.deepEqual(selection.balance.laneByVisibilityOrder, {
    'portable:H': 8,
    'feature:H': 8,
    'feature:L': 8,
    'portable:L': 8,
  });
  assert.deepEqual(
    selection.sessions.map(({ globalSessionIndex }) => globalSessionIndex),
    Array.from({ length: 32 }, (_, index) => index),
  );
});

test('startup endurance plan permanently excludes timing, retries, and efficacy', () => {
  const probe = buildStartupEnduranceProbePlan({ runId: RUN_ID, standalonePlan: plan });
  assert.equal(probe.executionMode, STARTUP_ENDURANCE_EXECUTION_MODE);
  assert.equal(probe.analysisEligible, false);
  assert.deepEqual(probe.executionPolicy, {
    attemptCount: 1,
    retryCount: 0,
    replacementAllowed: false,
    priorInvocationAllowed: false,
    resultRootMustBeAbsent: true,
    stopOnFirstFailure: true,
    serverInstanceCount: 1,
    serverRestartAllowed: false,
    forcedFeatureOffGateCount: 1,
    browserProcessCount: 33,
    freshBrowserProcessPerSession: true,
    freshPlaywrightTemporaryProfilePerSession: true,
    postDisconnectDelayMs: 2_000,
  });
  assert.deepEqual(probe.exclusionPolicy, {
    analysisEligible: false,
    startTrialAllowed: false,
    timingCaptureAllowed: false,
    efficacyAnalysisAllowed: false,
    numericalDecision: null,
  });
  assert.deepEqual(STARTUP_ENDURANCE_BROWSER_ARGS, [
    '--enable-unsafe-webgpu',
    '--enable-webgpu-developer-features',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
  ]);
  assert.equal(STARTUP_ENDURANCE_BROWSER_COUNT, 33);
  assert.equal(probe.networkReadinessPolicy.scheduledSnapshotCount, 66);
});

test('TCP readiness freezes the quiet window and conservative free-port margin', () => {
  const accepted = assessStartupTcpReadiness(validTcpSnapshot());
  assert.equal(STARTUP_ENDURANCE_TCP_QUIET_WINDOW_MS, 600_000);
  assert.equal(STARTUP_ENDURANCE_MINIMUM_FREE_DYNAMIC_PORTS, 4_096);
  assert.equal(accepted.pass, true);
  assert.equal(accepted.rangeAssessments[0].requiredFreeDynamicPorts, 4_096);

  const recent = validTcpSnapshot({
    latestTcpip4231: {
      providerName: 'Tcpip',
      timeCreatedUtc: '2026-09-03T11:59:59.999Z',
      recordId: 43,
      logName: 'System',
    },
  });
  const recentAssessment = assessStartupTcpReadiness(recent);
  assert.equal(recentAssessment.pass, false);
  assert.match(recentAssessment.reasons.join('\n'), /record 43 occurred 1 ms ago/u);

  const saturated = validTcpSnapshot({ uniqueLocalPortCount: 12_289 });
  const saturatedAssessment = assessStartupTcpReadiness(saturated);
  assert.equal(saturatedAssessment.pass, false);
  assert.match(saturatedAssessment.reasons.join('\n'), /only 4095 free/u);

  const unavailable = validTcpSnapshot();
  unavailable.eventQuery.status = 'unavailable';
  assert.equal(assessStartupTcpReadiness(unavailable).pass, false);

  const widenedQuery = validTcpSnapshot();
  widenedQuery.eventQuery.maximumEventsScanned = 2;
  assert.equal(assessStartupTcpReadiness(widenedQuery).pass, false);

  const wrongLog = validTcpSnapshot();
  wrongLog.eventQuery.logName = 'Application';
  assert.equal(assessStartupTcpReadiness(wrongLog).pass, false);
});

test('startup configuration validator accepts both lanes and rejects trial execution', () => {
  const portable = plan.sessions.find((session) => session.assignedLaneId === 'portable');
  const feature = plan.sessions.find((session) => session.assignedLaneId === 'feature');
  assert.equal(validateStartupEnduranceConfiguration(
    validConfiguration(portable),
    portable,
  ).pass, true);
  assert.equal(validateStartupEnduranceConfiguration(
    validConfiguration(feature),
    feature,
  ).pass, true);

  const timed = validConfiguration(feature);
  timed.trialExecutionState.phase = 'complete';
  timed.trialExecutionState.rowCount = 480;
  assert.equal(validateStartupEnduranceConfiguration(timed, feature).pass, false);
});

test('page diagnostic validation rejects late fatal events and capture overflow', () => {
  assert.equal(validateStartupPageDiagnostics({
    events: [],
    droppedEventCount: 0,
  }).pass, true);

  const lateConsoleError = validateStartupPageDiagnostics({
    events: [{ source: 'console', detail: 'late close-path error' }],
    droppedEventCount: 0,
  });
  assert.equal(lateConsoleError.pass, false);
  assert.match(lateConsoleError.reasons.join('\n'), /retained 1 fatal diagnostic event/u);

  const overflow = validateStartupPageDiagnostics({
    events: [],
    droppedEventCount: 1,
  });
  assert.equal(overflow.pass, false);
  assert.match(overflow.reasons.join('\n'), /capture overflowed/u);
});

test('latched termination fails every named completion boundary', () => {
  assert.equal(assertStartupEnduranceNotInterrupted(null, 'test-boundary'), true);
  assert.throws(
    () => assertStartupEnduranceNotInterrupted('SIGTERM', 'before-manifest'),
    (error) => error.message === 'Startup endurance probe interrupted by SIGTERM.'
      && error.detail?.terminationSignal === 'SIGTERM'
      && error.detail?.boundary === 'before-manifest',
  );
});

test('the fixed startup endurance result root can be claimed only once', async (t) => {
  const temporaryRoot = await mkdtemp(path.join(tmpdir(), 'startup-endurance-attempt-'));
  t.after(() => rm(temporaryRoot, { recursive: true, force: true }));
  const parentDirectory = path.join(temporaryRoot, 'results', 'development');
  const resultRoot = path.join(
    parentDirectory,
    'first-instance-standalone-startup-endurance',
  );
  await mkdir(parentDirectory, { recursive: true });

  const claim = await claimStartupEnduranceAttemptRoot(resultRoot);
  assert.equal(claim.resultRoot, resultRoot);
  assert.equal(claim.priorAttemptAllowed, false);
  await assert.rejects(
    claimStartupEnduranceAttemptRoot(resultRoot),
    (error) => error.code === 'STARTUP_ENDURANCE_ATTEMPT_ALREADY_CONSUMED'
      && /sole declared probe attempt has already been consumed/u.test(error.message),
  );
});

test('emergency browser cleanup is idempotent and never invents a close', async () => {
  let contextCloseCount = 0;
  let browserCloseCount = 0;
  const record = {
    disconnectedEventCount: 0,
    disconnectedAt: null,
    disconnectedRunElapsedMs: null,
    closedAt: null,
    closedBeforeNextLaunch: false,
    postDisconnectDelay: null,
  };
  const state = {
    browser: {
      async close() {
        browserCloseCount += 1;
        throw new Error('browser remained alive');
      },
      contexts: () => [{}],
      isConnected: () => true,
    },
    context: {
      async close() {
        contextCloseCount += 1;
        throw new Error('context remained alive');
      },
    },
    page: { isClosed: () => false },
    record,
    forceClosePromise: null,
    shutdownStarted: false,
  };

  const first = await closeStartupBrowserStateAfterAbort(state, 'unit failure', {
    closeTimeoutMs: 100,
    postDisconnectDelayMs: 0,
  });
  const second = await closeStartupBrowserStateAfterAbort(state, 'ignored repeat', {
    closeTimeoutMs: 100,
    postDisconnectDelayMs: 0,
  });
  assert.deepEqual(second, first);
  assert.equal(contextCloseCount, 1);
  assert.equal(browserCloseCount, 1);
  assert.equal(first.closedAt, null);
  assert.equal(first.closedBeforeNextLaunch, false);
  assert.equal(first.emergencyCleanup.provenClosed, false);
  assert.equal(first.emergencyCleanup.contextClose.succeeded, false);
  assert.equal(first.emergencyCleanup.contextClose.error.message, 'context remained alive');
  assert.equal(first.emergencyCleanup.browserClose.succeeded, false);
  assert.equal(first.emergencyCleanup.browserClose.error.message, 'browser remained alive');
  assert.deepEqual(first.emergencyCleanup.observedFinalState, {
    pageCreated: true,
    pageClosed: false,
    contextCreated: true,
    contextCount: 1,
    browserConnected: true,
    disconnectedEventCount: 0,
    disconnectedAt: null,
    disconnectedRunElapsedMs: null,
    observationErrors: [],
  });
});

test('emergency browser cleanup labels closed only after exact observed closure', async () => {
  let pageClosed = false;
  let contexts = [{}];
  let browserConnected = true;
  const record = {
    disconnectedEventCount: 0,
    disconnectedAt: null,
    disconnectedRunElapsedMs: null,
    closedAt: null,
    closedBeforeNextLaunch: false,
    postDisconnectDelay: null,
  };
  const state = {
    browser: {
      async close() {
        browserConnected = false;
        record.disconnectedEventCount = 1;
        record.disconnectedAt = new Date().toISOString();
        record.disconnectedRunElapsedMs = 1;
      },
      contexts: () => contexts,
      isConnected: () => browserConnected,
    },
    context: {
      async close() {
        pageClosed = true;
        contexts = [];
      },
    },
    page: { isClosed: () => pageClosed },
    record,
    forceClosePromise: null,
    shutdownStarted: false,
  };

  const cleanup = await closeStartupBrowserStateAfterAbort(state, 'unit cleanup', {
    closeTimeoutMs: 100,
    postDisconnectDelayMs: 0,
  });
  assert.equal(cleanup.emergencyCleanup.provenClosed, true);
  assert.equal(cleanup.closedBeforeNextLaunch, true);
  assert.equal(typeof cleanup.closedAt, 'string');
  assert.equal(cleanup.emergencyCleanup.postDisconnectDelayObserved, true);
});

test('startup probe source contains no benchmark trial or efficacy invocation', async () => {
  const source = await readFile(new URL(
    '../scripts/probe-first-instance-standalone-startup-endurance.mjs',
    import.meta.url,
  ), 'utf8');
  assert.doesNotMatch(source, /bench\.startTrial\s*\(/u);
  assert.doesNotMatch(source, /bench\.captureRenderParity\s*\(/u);
  assert.doesNotMatch(source, /summarizeFirstInstanceStandaloneDeployment\s*\(/u);
  assert.equal(
    source.match(/bench\.runFirstInstanceLiveForcedFeatureOffGate\s*\(/gu)?.length,
    1,
  );
  assert.doesNotMatch(source, /mkdir\(resultRoot,\s*\{\s*recursive:\s*true\s*\}\)/u);
  assert.match(source,
    /attemptRootClaim = await claimStartupEnduranceAttemptRoot\(resultRoot\)/u);
  assert.match(source,
    /requireActive\('after-session-schedule'\)[\s\S]*requireActive\('immediately-before-completion-manifest-commit'\);\s*const manifestArtifact = artifactStore\.jsonSynchronous/u);
});
