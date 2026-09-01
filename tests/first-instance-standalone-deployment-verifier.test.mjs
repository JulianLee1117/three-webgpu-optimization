import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  lstat,
  mkdir,
  mkdtemp,
  rm,
  symlink,
  writeFile,
} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { brotliCompressSync } from 'node:zlib';

import {
  readFirstInstanceStandaloneDeploymentDeclaredArtifacts,
  resolveStandaloneDeploymentDecision,
  verifyStandaloneAddressDigestCommitment,
  verifyStandaloneBrowserLifecycleChain,
  verifyStandaloneContinuousTimestampResolutionForTest,
  verifyStandaloneEnvironmentObservations,
  verifyStandaloneLaneShaderNormalization,
  verifyStandaloneInteractiveChallengeForTest,
  verifyStandaloneRenderCommitmentForTest,
  verifyStandaloneSessionResourceIdentityRecords,
  verifyStandaloneSessionTimestampContinuity,
  verifyStandaloneTimedSerialInterval,
  verifyStandaloneTimestampPhaseBoundary,
  verifyStandaloneTimestampRecordForTest,
} from '../analysis/verify-first-instance-standalone-deployment.mjs';
import {
  createFirstInstanceLaneShaderEvidence,
} from '../src/validation/first-instance-shader-evidence.js';
import {
  allocateBalancedCounts,
  allocateVisibleCounts,
  prefixBases,
} from '../src/scenes/fixed-subsets.js';
import {
  createFirstInstanceAddressChallengeExpected,
  createFirstInstanceAddressChallengeTargetShape,
} from '../src/validation/first-instance-address-challenge.js';

const sha256 = (value) => createHash('sha256').update(value).digest('hex');

function environment(performanceNowQuantumMs) {
  return {
    threeRevision: '185',
    userAgent: 'fixed-agent',
    adapterInfo: { vendor: 'fixed' },
    rendererBackend: 'WebGPUBackend',
    coordinateSystem: 2000,
    reversedDepth: true,
    rendererReversedDepthBuffer: true,
    maxStorageBuffersPerShaderStage: 10,
    timestampAvailable: true,
    indirectFirstInstanceAvailable: true,
    crossOriginIsolated: true,
    viewport: { width: 1280, height: 720 },
    performanceNowQuantumMs,
    webgpuUncapturedErrorCount: 0,
    webgpuValidationErrorCount: 0,
    webgpuDeviceLossCount: 0,
  };
}

test('environment identity excludes only independently thresholded timer quantum', () => {
  const identity = verifyStandaloneEnvironmentObservations([
    environment(0.001),
    environment(0.002),
  ]);
  assert.equal(Object.hasOwn(identity, 'performanceNowQuantumMs'), false);
  assert.throws(
    () => verifyStandaloneEnvironmentObservations([environment(0.011)]),
    /coarse CPU timer quantum/,
  );
});

test('realm-local numeric resource IDs may repeat but must be stable within a session', () => {
  const resourceIdentity = { meshUuid: 'mesh', commandBufferAttributeId: 7 };
  const shaderResourceIdentity = {
    render: { pipelineIdentity: 3 },
    compute: { reset: { pipelineIdentity: 4 }, cull: { pipelineIdentity: 5 } },
  };
  const records = [
    ['session-a', 'session-a:browser-2'],
    ['session-a', 'session-a:browser-2'],
    ['session-b', 'session-b:browser-3'],
    ['session-b', 'session-b:browser-3'],
  ].map(([sessionId, sessionNamespace]) => ({
    sessionId,
    sessionNamespace,
    resourceIdentity: structuredClone(resourceIdentity),
    shaderResourceIdentity: structuredClone(shaderResourceIdentity),
  }));
  assert.equal(verifyStandaloneSessionResourceIdentityRecords(records).sessionCount, 2);
  const changed = structuredClone(records);
  changed[1].shaderResourceIdentity.render.pipelineIdentity += 1;
  assert.throws(
    () => verifyStandaloneSessionResourceIdentityRecords(changed),
    /render\/compute state binding pipeline identity/,
  );
});

function lifecycle(serial, role, matrixOrdinal, session = null, previous = null) {
  const launch = previous === null ? 0 : previous.disconnectedRunElapsedMs + 2_100;
  const disconnected = launch + 100;
  const value = {
    schemaVersion: 1,
    kind: 'first-instance-standalone-browser-profile-lifecycle',
    browserInstanceSerial: serial,
    role,
    matrixOrdinal,
    sessionId: session?.sessionId ?? null,
    globalSessionIndex: session?.globalSessionIndex ?? null,
    sessionNamespace: session === null ? null : `${session.sessionId}:browser-${serial}`,
    launchApi: 'chromium.launch',
    persistentContext: false,
    profilePolicy: 'fresh-playwright-temporary-profile-per-process',
    userDataDirectoryExposed: false,
    profileReused: false,
    launchArguments: [
      '--enable-unsafe-webgpu',
      '--enable-webgpu-developer-features',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
    ],
    launchedAt: new Date(launch + 1_000).toISOString(),
    launchedRunElapsedMs: launch,
    priorBrowserInstanceSerial: previous?.browserInstanceSerial ?? null,
    priorBrowserDisconnectedAt: previous?.disconnectedAt ?? null,
    previousDisconnectToLaunchGapMs: previous === null
      ? null
      : launch - previous.disconnectedRunElapsedMs,
    contextCreatedAt: new Date(launch + 1_010).toISOString(),
    pageCreatedAt: new Date(launch + 1_020).toISOString(),
    closedAt: new Date(disconnected + 1_010).toISOString(),
    disconnectedAt: new Date(disconnected + 1_000).toISOString(),
    disconnectedRunElapsedMs: disconnected,
    contextCountBeforeClose: 1,
    pageCountBeforeClose: 1,
    disconnectedEventCount: 1,
    closedBeforeNextLaunch: true,
    postDisconnectDelay: {
      requestedMs: 2_000,
      startedAt: new Date(disconnected + 1_000).toISOString(),
      startedRunElapsedMs: disconnected,
      completedAt: new Date(disconnected + 3_000).toISOString(),
      completedRunElapsedMs: disconnected + 2_000,
      elapsedMs: 2_000,
    },
  };
  return value;
}

test('browser lifecycle verifier requires sequential disposable processes', () => {
  const session = {
    sessionId: 'session-1',
    globalSessionIndex: 0,
    matrixIndex: 0,
    matrixOrdinal: 1,
  };
  const gate = lifecycle(1, 'forced-feature-off-gate', 1);
  const measured = lifecycle(2, 'standalone-measured-session', 1, session, gate);
  const selection = { matrices: [{ matrixOrdinal: 1, matrixIndex: 0 }], sessions: [session] };
  assert.equal(verifyStandaloneBrowserLifecycleChain([gate, measured], selection).pass, true);
  const overlap = structuredClone(measured);
  overlap.previousDisconnectToLaunchGapMs = 1_999;
  assert.throws(
    () => verifyStandaloneBrowserLifecycleChain([gate, overlap], selection),
    /overlaps or lacks its exact predecessor gap/,
  );
});

test('technical environment/telemetry gate misses cannot become a standalone confirmation', () => {
  assert.equal(resolveStandaloneDeploymentDecision({
    smokeMode: false,
    numericalPass: true,
    technicalGatePass: false,
  }), 'standalone-confirmation-not-met');
  assert.equal(resolveStandaloneDeploymentDecision({
    smokeMode: true,
    numericalPass: true,
    technicalGatePass: true,
  }), null);
});

test('timed serial and strict timestamp UID helpers reject shifted or detached evidence', () => {
  const starts = {
    strategyComputeCallSerial: 10,
    computeCallSerial: 20,
    renderCallSerial: 30,
  };
  const rows = Array.from({ length: 480 }, (_, index) => ({
    strategyComputeCallSerial: starts.strategyComputeCallSerial + 320 + index + 1,
    computeCallSerial: starts.computeCallSerial + 320 + index + 1,
    renderCallSerial: starts.renderCallSerial + 320 + index + 1,
  }));
  assert.equal(verifyStandaloneTimedSerialInterval(rows, starts), true);
  const shifted = structuredClone(rows);
  shifted[0].computeCallSerial += 1;
  assert.throws(
    () => verifyStandaloneTimedSerialInterval(shifted, starts),
    /shifted from its exact interval/,
  );
  const timestamp = {
    uid: 'c:1:100001:f42',
    type: 'compute',
    callIndex: 1,
    contextId: 100001,
    frameId: 42,
    durationMs: 0.25,
  };
  verifyStandaloneTimestampRecordForTest(timestamp, {
    type: 'compute', frameId: 42, callIndex: 1, contextId: 100001, durationMs: 0.25,
  });
  assert.throws(
    () => verifyStandaloneTimestampRecordForTest(
      { ...timestamp, uid: 'c:1:100001:f43' },
      {
        type: 'compute', frameId: 42, callIndex: 1, contextId: 100001, durationMs: 0.25,
      },
    ),
    /strict timestamp UID grammar/,
  );
});

test('timestamp phase boundary requires exact warmup/measurement adjacency', () => {
  const warmup = { compute: [10, 11], render: [10, 11] };
  assert.deepEqual(
    verifyStandaloneTimestampPhaseBoundary(
      warmup,
      { compute: [12, 13], render: [12, 13] },
    ).idleRafCount,
    0,
  );
  assert.throws(
    () => verifyStandaloneTimestampPhaseBoundary(
      warmup,
      { compute: [13, 14], render: [13, 14] },
    ),
    /not exactly adjacent to warmup/,
  );
  assert.throws(
    () => verifyStandaloneTimestampPhaseBoundary(
      warmup,
      { compute: [11, 12], render: [11, 12] },
    ),
    /not exactly adjacent to warmup/,
  );
  assert.throws(
    () => verifyStandaloneTimestampPhaseBoundary(
      warmup,
      { compute: [13, 14], render: [14, 15] },
    ),
    /compute\/render GPU-frame interval/,
  );
  assert.throws(
    () => verifyStandaloneTimestampPhaseBoundary(
      { compute: [10, 12], render: [10, 12] },
      { compute: [13, 14], render: [13, 14] },
    ),
    /not consecutive/,
  );
});

function continuousTimestampEvidence() {
  const warmupFrames = Array.from({ length: 320 }, (_, index) => 10 + index);
  const measurementFrames = Array.from({ length: 480 }, (_, index) => 330 + index);
  const combinedFrames = [...warmupFrames, ...measurementFrames];
  const uid = (type, frame) => `${type === 'compute' ? 'c' : 'r'}:1:${
    type === 'compute' ? 2 : 0
  }:f${frame}`;
  const timedUids = Object.fromEntries(['render', 'compute'].map((type) => [
    type,
    combinedFrames.map((frame) => uid(type, frame)),
  ]));
  const primeUids = { render: ['r:2:0:f1'], compute: ['c:2:2:f1'] };
  const pool = (type, {
    frames,
    timestampUids,
    queryOffsetUids = [],
    currentQueryIndex = 0,
  }) => {
    const base = type === 'render' ? 10 : 20;
    return {
      poolIdentity: base,
      querySetIdentity: base + 1,
      resolveBufferIdentity: base + 2,
      resultBufferIdentity: base + 3,
      maxQueries: 2_048,
      currentQueryIndex,
      queryOffsetCount: queryOffsetUids.length,
      queryOffsetUids,
      frameCount: frames.length,
      frames,
      timestampUidCount: timestampUids.length,
      timestampUids: [...timestampUids].sort(),
      pendingResolve: false,
      isDisposed: false,
      resultBufferMapState: 'unmapped',
    };
  };
  const diagnostics = ({ tracking, stage }) => ({
    schemaVersion: 1,
    kind: 'three-r185-timestamp-pool-diagnostics',
    backendTrackingEnabled: tracking,
    ...Object.fromEntries(['render', 'compute'].map((type) => {
      if (stage === 'start') return [type, pool(type, {
        frames: [1], timestampUids: primeUids[type],
      })];
      if (stage === 'before') return [type, pool(type, {
        frames: [1],
        timestampUids: primeUids[type],
        queryOffsetUids: timedUids[type],
        currentQueryIndex: 1_600,
      })];
      return [type, pool(type, {
        frames: combinedFrames,
        timestampUids: [...primeUids[type], ...timedUids[type]],
      })];
    })),
  });
  const start = diagnostics({ tracking: false, stage: 'start' });
  const before = diagnostics({ tracking: true, stage: 'before' });
  const after = diagnostics({ tracking: true, stage: 'after' });
  return {
    topology: {
      schemaVersion: 1,
      kind: 'three-r185-timestamp-resolution-topology',
      mode: 'single-post-measurement',
      warmupBoundaryResolveBatchCount: 0,
      postMeasurementResolveBatchCount: 1,
      queriesPerTimestampUid: 2,
      requiredQueriesPerType: 1_600,
      resolvedFrameCountByType: { render: 800, compute: 800 },
      firstGpuFrameId: 10,
      lastGpuFrameId: 809,
      intervalContiguous: true,
      poolsAtStart: start,
      poolsBeforePostMeasurementResolve: before,
      poolsAfterPostMeasurementResolve: after,
    },
    completion: {
      timestampPoolsAtTimingStart: start,
      timestampPoolsAtTimingEnd: after,
      timestampPhaseBoundaryExact: true,
      timestampResolvedFrameIntervalExact: true,
      timestampResolutionTopologyExact: true,
    },
    timestampEvidence: {
      warmupFramesByType: { render: warmupFrames, compute: warmupFrames },
      measurementFramesByType: { render: measurementFrames, compute: measurementFrames },
      warmupUidsByType: {
        render: timedUids.render.slice(0, 320),
        compute: timedUids.compute.slice(0, 320),
      },
      measurementUidsByType: {
        render: timedUids.render.slice(320),
        compute: timedUids.compute.slice(320),
      },
    },
  };
}

test('continuous timestamp resolution proves capacity, unresolved state, and all 800 frames', () => {
  const valid = continuousTimestampEvidence();
  assert.equal(verifyStandaloneContinuousTimestampResolutionForTest(valid), true);

  const shortEnd = structuredClone(valid);
  shortEnd.topology.poolsAfterPostMeasurementResolve.render.frameCount = 480;
  shortEnd.completion.timestampPoolsAtTimingEnd.render.frameCount = 480;
  assert.throws(
    () => verifyStandaloneContinuousTimestampResolutionForTest(shortEnd),
    /invalid resolved frame IDs|query capacity or pool state|post-resolve cardinality/,
  );

  const lowCapacity = structuredClone(valid);
  for (const snapshot of [
    lowCapacity.topology.poolsAtStart,
    lowCapacity.topology.poolsBeforePostMeasurementResolve,
    lowCapacity.topology.poolsAfterPostMeasurementResolve,
    lowCapacity.completion.timestampPoolsAtTimingStart,
    lowCapacity.completion.timestampPoolsAtTimingEnd,
  ]) snapshot.compute.maxQueries = 1_599;
  assert.throws(
    () => verifyStandaloneContinuousTimestampResolutionForTest(lowCapacity),
    /must be a safe integer|query capacity or pool state/,
  );

  const nonintegerCapacity = structuredClone(valid);
  for (const snapshot of [
    nonintegerCapacity.topology.poolsAtStart,
    nonintegerCapacity.topology.poolsBeforePostMeasurementResolve,
    nonintegerCapacity.topology.poolsAfterPostMeasurementResolve,
    nonintegerCapacity.completion.timestampPoolsAtTimingStart,
    nonintegerCapacity.completion.timestampPoolsAtTimingEnd,
  ]) snapshot.render.maxQueries = 1_600.5;
  assert.throws(
    () => verifyStandaloneContinuousTimestampResolutionForTest(nonintegerCapacity),
    /must be a safe integer|query capacity or pool state/,
  );

  const earlyResolve = structuredClone(valid);
  earlyResolve.topology.poolsBeforePostMeasurementResolve.compute.currentQueryIndex = 960;
  assert.throws(
    () => verifyStandaloneContinuousTimestampResolutionForTest(earlyResolve),
    /query capacity or pool state/,
  );

  const inconsistentLists = structuredClone(valid);
  inconsistentLists.topology.poolsBeforePostMeasurementResolve.render.timestampUidCount = 2;
  assert.throws(
    () => verifyStandaloneContinuousTimestampResolutionForTest(inconsistentLists),
    /invalid resolved timestamp UIDs|query capacity or pool state/,
  );

  const duplicateLists = structuredClone(valid);
  const duplicateUid = duplicateLists.topology.poolsAtStart.compute.timestampUids[0];
  duplicateLists.topology.poolsAtStart.compute.timestampUids = [duplicateUid, duplicateUid];
  duplicateLists.topology.poolsAtStart.compute.timestampUidCount = 2;
  duplicateLists.topology.poolsBeforePostMeasurementResolve.compute.timestampUids = [
    duplicateUid,
    duplicateUid,
  ];
  duplicateLists.topology.poolsBeforePostMeasurementResolve.compute.timestampUidCount = 2;
  duplicateLists.topology.poolsAfterPostMeasurementResolve.compute.timestampUids = [
    duplicateUid,
    ...duplicateLists.topology.poolsAfterPostMeasurementResolve.compute.timestampUids,
  ].sort();
  duplicateLists.topology.poolsAfterPostMeasurementResolve.compute.timestampUidCount = 802;
  assert.throws(
    () => verifyStandaloneContinuousTimestampResolutionForTest(duplicateLists),
    /invalid resolved timestamp UIDs|query capacity or pool state/,
  );

  const extraBatch = structuredClone(valid);
  extraBatch.topology.postMeasurementResolveBatchCount = 2;
  assert.throws(
    () => verifyStandaloneContinuousTimestampResolutionForTest(extraBatch),
    /one continuous post-measurement resolution/,
  );

  const wrongDiagnosticMetadata = structuredClone(valid);
  for (const diagnostics of new Set([
    wrongDiagnosticMetadata.topology.poolsAtStart,
    wrongDiagnosticMetadata.completion.timestampPoolsAtTimingStart,
  ])) {
    diagnostics.schemaVersion = 999;
    diagnostics.kind = 'forged';
  }
  assert.throws(
    () => verifyStandaloneContinuousTimestampResolutionForTest(wrongDiagnosticMetadata),
    /invalid diagnostic metadata/,
  );

  const forgedDiagnostics = structuredClone(valid);
  for (const diagnostics of new Set([
    forgedDiagnostics.topology.poolsAtStart,
    forgedDiagnostics.topology.poolsBeforePostMeasurementResolve,
    forgedDiagnostics.topology.poolsAfterPostMeasurementResolve,
    forgedDiagnostics.completion.timestampPoolsAtTimingStart,
    forgedDiagnostics.completion.timestampPoolsAtTimingEnd,
  ])) {
    diagnostics.unrecognizedProof = true;
  }
  assert.throws(
    () => verifyStandaloneContinuousTimestampResolutionForTest(forgedDiagnostics),
    /unexpected schema/,
  );

  const identityFreeDiagnostics = structuredClone(valid);
  for (const diagnostics of new Set([
    identityFreeDiagnostics.topology.poolsAtStart,
    identityFreeDiagnostics.topology.poolsBeforePostMeasurementResolve,
    identityFreeDiagnostics.topology.poolsAfterPostMeasurementResolve,
    identityFreeDiagnostics.completion.timestampPoolsAtTimingStart,
    identityFreeDiagnostics.completion.timestampPoolsAtTimingEnd,
  ])) {
    for (const type of ['render', 'compute']) {
      diagnostics[type].poolIdentity = null;
      diagnostics[type].querySetIdentity = null;
      diagnostics[type].resolveBufferIdentity = null;
      diagnostics[type].resultBufferIdentity = null;
    }
  }
  assert.throws(
    () => verifyStandaloneContinuousTimestampResolutionForTest(identityFreeDiagnostics),
    /must be a safe integer/,
  );
});

test('reused session timestamp pools join the first resolved history to trial two', () => {
  const first = continuousTimestampEvidence();
  const secondStart = structuredClone(first.topology.poolsAfterPostMeasurementResolve);
  secondStart.backendTrackingEnabled = false;
  const records = [
    {
      sessionId: 'session-1',
      visibilityOrderPosition: 0,
      timestampPoolsAtStart: first.topology.poolsAtStart,
      timestampPoolsAtEnd: first.topology.poolsAfterPostMeasurementResolve,
    },
    {
      sessionId: 'session-1',
      visibilityOrderPosition: 1,
      timestampPoolsAtStart: secondStart,
      timestampPoolsAtEnd: first.topology.poolsAfterPostMeasurementResolve,
    },
  ];
  assert.deepEqual(verifyStandaloneSessionTimestampContinuity(records), {
    sessionCount: 1,
    transitionCount: 1,
    withinSessionTimestampHistoryExact: true,
  });
  const detached = structuredClone(records);
  detached[1].timestampPoolsAtStart.compute.timestampUids.pop();
  detached[1].timestampPoolsAtStart.compute.timestampUidCount -= 1;
  assert.throws(
    () => verifyStandaloneSessionTimestampContinuity(detached),
    /compute timestamp history transition/,
  );
});

test('address commitment is rebuilt from raw survivors and active command counts', () => {
  const capacities = allocateBalancedCounts(65_536, 32);
  const bases = prefixBases(capacities);
  const activeCounts = allocateVisibleCounts(capacities, 0.99);
  const survivors = new Uint32Array(65_536);
  const commands = new Uint32Array(32 * 5);
  for (let bucket = 0; bucket < 32; bucket += 1) {
    commands[bucket * 5 + 1] = activeCounts[bucket];
    for (let local = 0; local < activeCounts[bucket]; local += 1) {
      survivors[bases[bucket] + local] = bases[bucket] + local;
    }
  }
  const expected = createFirstInstanceAddressChallengeExpected({
    scenario: {
      objectCount: 65_536,
      bucketCount: 32,
      bucketBases: bases,
      bucketCounts: capacities,
    },
    visibleIds: survivors,
    activeCounts,
    targetShape: createFirstInstanceAddressChallengeTargetShape(65_536),
  });
  const address = {
    expectedSha256: sha256(expected.bytes),
    survivorSha256: sha256(Buffer.from(survivors.buffer)),
    activeCountsSha256: sha256(Buffer.from(activeCounts.buffer)),
    activeAddressCount: expected.activeAddressCount,
    paddingAddressCount: expected.paddingAddressCount,
    targetPaddingPixelCount: expected.targetPaddingPixelCount,
    coverage: expected.coverage,
  };
  verifyStandaloneAddressDigestCommitment({
    visibilityFraction: 0.99,
    commands: Array.from(commands),
    survivors: Array.from(survivors),
    address,
  });
  assert.throws(
    () => verifyStandaloneAddressDigestCommitment({
      visibilityFraction: 0.99,
      commands: Array.from(commands),
      survivors: Array.from(survivors),
      address: { ...address, expectedSha256: '0'.repeat(64) },
    }),
    /address digest commitment.expectedSha256/,
  );
});

test('forced-off interactive shader challenges require exact ordinals and unique nonces', () => {
  const challenge = {
    schemaVersion: 1,
    kind: 'live-first-instance-standalone-shader-observation-challenge',
    origin: 'page-interactive',
    runId: null,
    trialId: null,
    planIndex: null,
    repetitionIndex: null,
    phase: 'interactive',
    role: 'render-parity',
    captureOrdinal: 1,
    challengeNonce: 'a'.repeat(64),
  };
  const nonces = new Set();
  verifyStandaloneInteractiveChallengeForTest(challenge, 1, 'render-parity', nonces);
  assert.throws(
    () => verifyStandaloneInteractiveChallengeForTest(challenge, 1, 'render-parity', nonces),
    /invalid or reused/,
  );
  assert.throws(
    () => verifyStandaloneInteractiveChallengeForTest(
      { ...challenge, challengeNonce: 'b'.repeat(64) },
      2,
      'render-parity',
      new Set(),
    ),
    /invalid or reused/,
  );
});

function descriptor(relativePath, bytes) {
  return {
    path: relativePath,
    encoding: 'json-utf8',
    byteLength: bytes.length,
    sha256: sha256(bytes),
  };
}

test('declared-artifact reader verifies exact inventory and lazily hashes Brotli JSON', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'standalone-verifier-'));
  try {
    const bodies = new Map([
      ['plan.json', Buffer.from('{"plan":1}\n')],
      ['execution-identity-start.json', Buffer.from('{"start":1}\n')],
      ['execution-identity-end.json', Buffer.from('{"end":1}\n')],
      ['vite-runtime-audit.json', Buffer.from('{"vite":1}\n')],
    ]);
    const trialJson = Buffer.from('{"trial":1}\n');
    const trialCompressed = brotliCompressSync(trialJson);
    const trialPath = 'trials/trial-001.json.br';
    await mkdir(path.join(directory, 'trials'));
    for (const [relativePath, bytes] of bodies) {
      await writeFile(path.join(directory, relativePath), bytes);
    }
    await writeFile(path.join(directory, trialPath), trialCompressed);
    const trialDescriptor = {
      path: trialPath,
      encoding: 'brotli-json-utf8',
      jsonByteLength: trialJson.length,
      jsonSha256: sha256(trialJson),
      brotliByteLength: trialCompressed.length,
      brotliSha256: sha256(trialCompressed),
    };
    const manifest = {
      artifacts: {
        plan: descriptor('plan.json', bodies.get('plan.json')),
        executionIdentityStart: descriptor(
          'execution-identity-start.json',
          bodies.get('execution-identity-start.json'),
        ),
        executionIdentityEnd: descriptor(
          'execution-identity-end.json',
          bodies.get('execution-identity-end.json'),
        ),
        forcedFeatureOffGates: [],
        trials: [{
          trialId: 'trial-1',
          planIndex: 0,
          matrixIndex: 0,
          sessionId: 'session-1',
          assignedLaneId: 'portable',
          visibilityFraction: 0.99,
          visibilityExposure: 'first',
          sessionNamespace: 'session-1:browser-2',
          shaderCaptureOrdinals: [1, 2, 3, 4, 5, 6],
          artifact: trialDescriptor,
        }],
        sessions: [],
        matrices: [],
        telemetry: [],
        viteRuntimeAudit: descriptor(
          'vite-runtime-audit.json',
          bodies.get('vite-runtime-audit.json'),
        ),
        journal: [],
      },
    };
    await writeFile(path.join(directory, 'manifest.json'), '{}\n');
    const archive = await readFirstInstanceStandaloneDeploymentDeclaredArtifacts(
      directory,
      manifest,
    );
    assert.equal(archive.artifactCount, 5);
    assert.deepEqual(await archive.read(trialDescriptor), { trial: 1 });
    await writeFile(path.join(directory, trialPath), Buffer.from('tampered'));
    await assert.rejects(() => archive.read(trialDescriptor), /Brotli file|brotliByteLength/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

function primedResourceIdentities() {
  return {
    render: {
      renderObjectIdentity: 1,
      nodeBuilderStateIdentity: 2,
      bindingArrayIdentity: 3,
      pipelineIdentity: 4,
    },
    compute: {
      reset: {
        computeNodeId: 5,
        computeNodeVersion: 0,
        nodeBuilderStateIdentity: 6,
        bindingArrayIdentity: 7,
        pipelineIdentity: 8,
      },
      cull: {
        computeNodeId: 9,
        computeNodeVersion: 0,
        nodeBuilderStateIdentity: 10,
        bindingArrayIdentity: 11,
        pipelineIdentity: 12,
      },
    },
  };
}

function renderCommitment() {
  return {
    schemaVersion: 1,
    kind: 'first-instance-live-standalone-render-commitment',
    laneId: 'feature',
    bundleUuid: 'bundle',
    bundleVersion: 0,
    meshUuid: 'mesh',
    geometryUuid: 'geometry',
    materialUuid: 'material',
    materialVersion: 0,
    bundleRecordCallbackCount: 1,
    renderBundleStable: true,
    renderBundleDataStable: true,
    bundleGpuStable: true,
    renderObjectStable: true,
    nodeBuilderStateStable: true,
    resourceIdentities: primedResourceIdentities(),
    matrixAttributeId: 13,
    matrixAttributeVersion: 0,
    boundsAttributeId: 14,
    boundsAttributeVersion: 0,
    visibleIdsAttributeId: 15,
    visibleIdsAttributeVersion: 0,
    indirectAttributeId: 16,
    indirectAttributeVersion: 0,
    commandBuffer: {
      lane: 'feature',
      attributeId: 16,
      byteOffset: 0,
      byteLength: 640,
      recordCount: 32,
      drawCommandCount: 32,
      firstOffset: 0,
      allOffsets: Array.from({ length: 32 }, (_, index) => index * 20),
    },
    cache: { available: true },
  };
}

test('render commitment validates underlying identities, not only stability booleans', () => {
  const commitment = renderCommitment();
  const lifecycleIdentity = {
    rootUuid: 'bundle',
    meshUuid: 'mesh',
    geometryUuid: 'geometry',
    materialUuid: 'material',
    matrixAttributeId: 13,
    boundsAttributeId: 14,
    visibleIdsAttributeId: 15,
    indirectAttributeId: 16,
    commandBufferAttributeId: 16,
    commandBufferByteOffset: 0,
    commandBufferFirstOffset: 0,
  };
  assert.equal(
    verifyStandaloneRenderCommitmentForTest(commitment, 'feature', lifecycleIdentity)
      .resourceIdentities.render.pipelineIdentity,
    4,
  );
  const disjoint = structuredClone(commitment);
  disjoint.bundleUuid = 'different-bundle';
  assert.throws(
    () => verifyStandaloneRenderCommitmentForTest(disjoint, 'feature', lifecycleIdentity),
    /lifecycle identity/,
  );
  const forgedPass = structuredClone(commitment);
  forgedPass.nodeBuilderStateStable = false;
  assert.throws(
    () => verifyStandaloneRenderCommitmentForTest(forgedPass, 'feature', lifecycleIdentity),
    /stable production render bundle/,
  );
});

function vertexShader() {
  return `struct MatrixBuffer {
  value : array< mat4x4<f32> >
};
@binding( 3 ) @group( 1 ) var<storage, read> matrices : MatrixBuffer;
struct VisibleBuffer {
  value : array< u32 >
};
@binding( 4 ) @group( 1 ) var<storage, read> visible : VisibleBuffer;
struct VaryingsStruct { @builtin(position) position : vec4<f32> };
@vertex fn main( @builtin( instance_index ) instanceIndex : u32,
  @location( 0 ) position : vec3<f32>, @location( 1 ) normal : vec3<f32> )
  -> VaryingsStruct {
  var objectMatrix : mat4x4<f32>;
  objectMatrix = matrices.value[ visible.value[ instanceIndex ] ];
  var varyings : VaryingsStruct;
  varyings.position = objectMatrix * vec4<f32>( position + normal * 0.0, 1.0 );
  return varyings;
}`;
}

function computeShader(identifier) {
  return `struct FixedSliceIndexedDraw {
  indexCount : u32,
  instanceCount : atomic<u32>,
  firstIndex : u32,
  baseVertex : i32,
  firstInstance : u32
};
struct ${identifier}Struct {
  value : array<FixedSliceIndexedDraw>
};
@binding( 4 ) @group( 0 )
var<storage, read_write> ${identifier} : ${identifier}Struct;
@compute @workgroup_size( 64, 1, 1 )
fn main() {
  atomicAdd( &${identifier}.value[ instanceIndex ].instanceCount, 1u );
}`;
}

test('shader normalization is independently reconstructed from retained raw WGSL', async () => {
  const computeBinding = [{
    semantic: 'indirectCommands',
    kind: 'storage-buffer',
    group: 0,
    binding: 4,
    access: 'readWrite',
    attributeType: 'IndirectStorageBufferAttribute',
    arrayType: 'Uint32Array',
    itemSize: 5,
    count: 32,
    byteLength: 640,
  }];
  const record = await createFirstInstanceLaneShaderEvidence({
    laneId: 'feature',
    lane: {
      vertexShader: vertexShader(),
      fragmentShader: '@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1); }',
      vertexInputs: [
        {
          name: 'position', shaderLocation: 0, format: 'float32x3', stepMode: 'vertex',
          arrayType: 'Float32Array', itemSize: 3, normalized: false, count: 3,
          resourceId: 1,
        },
        {
          name: 'normal', shaderLocation: 1, format: 'float32x3', stepMode: 'vertex',
          arrayType: 'Float32Array', itemSize: 3, normalized: false, count: 3,
          resourceId: 2,
        },
      ],
      storageBindings: [
        {
          semantic: 'matrix', group: 1, binding: 3, access: 'read', visibility: 1,
          elementType: 'mat4x4<f32>', count: 65_536, byteLength: 65_536 * 64,
          resourceId: 3,
        },
        {
          semantic: 'visibleIds', group: 1, binding: 4, access: 'read', visibility: 1,
          elementType: 'u32', count: 65_536, byteLength: 65_536 * 4, resourceId: 4,
        },
      ],
    },
    compute: {
      reset: { shader: computeShader('NodeBuffer_100'), bindings: computeBinding },
      cull: { shader: computeShader('NodeBuffer_200'), bindings: computeBinding },
    },
  });
  assert.equal(record.pass, true, record.reasons.join('\n'));
  await verifyStandaloneLaneShaderNormalization(record);
  const cotampered = structuredClone(record);
  cotampered.render.raw.vertex.source = cotampered.render.raw.vertex.source
    .replace('visible.value[ instanceIndex ]', 'visible.value[ instanceIndex + 1u ]');
  cotampered.render.raw.vertex.byteLength = Buffer.byteLength(
    cotampered.render.raw.vertex.source,
  );
  cotampered.render.raw.vertex.sha256 = sha256(cotampered.render.raw.vertex.source);
  cotampered.render.normalizedVertex = structuredClone(cotampered.render.raw.vertex);
  cotampered.normalizedSemantics.vertexSha256 = cotampered.render.normalizedVertex.sha256;
  await assert.rejects(
    () => verifyStandaloneLaneShaderNormalization(cotampered),
    /independently reconstructed normalization/,
  );
});

test('artifact inventory rejects undeclared files and symlinks', async (context) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'standalone-inventory-'));
  try {
    const bytes = Buffer.from('{}\n');
    const files = [
      'plan.json',
      'execution-identity-start.json',
      'execution-identity-end.json',
      'vite-runtime-audit.json',
    ];
    await Promise.all(files.map((name) => writeFile(path.join(directory, name), bytes)));
    await writeFile(path.join(directory, 'manifest.json'), bytes);
    const manifest = {
      artifacts: {
        plan: descriptor(files[0], bytes),
        executionIdentityStart: descriptor(files[1], bytes),
        executionIdentityEnd: descriptor(files[2], bytes),
        forcedFeatureOffGates: [],
        trials: [],
        sessions: [],
        matrices: [],
        telemetry: [],
        viteRuntimeAudit: descriptor(files[3], bytes),
        journal: [],
      },
    };
    await writeFile(path.join(directory, 'extra.json'), bytes);
    await assert.rejects(
      () => readFirstInstanceStandaloneDeploymentDeclaredArtifacts(directory, manifest),
      /run file inventory/,
    );
    await rm(path.join(directory, 'extra.json'));
    try {
      await symlink(path.join(directory, files[0]), path.join(directory, 'extra-link'));
    } catch (error) {
      if (['EPERM', 'EACCES'].includes(error?.code)) {
        context.skip('creating symlinks is not permitted on this Windows host');
        return;
      }
      throw error;
    }
    assert.equal((await lstat(path.join(directory, 'extra-link'))).isSymbolicLink(), true);
    await assert.rejects(
      () => readFirstInstanceStandaloneDeploymentDeclaredArtifacts(directory, manifest),
      /is a symlink/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
