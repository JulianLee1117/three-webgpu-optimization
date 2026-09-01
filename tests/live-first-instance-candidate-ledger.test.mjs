import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  mkdir,
  mkdtemp,
  readFile,
  writeFile,
} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  appendCandidateLedgerEvent,
  buildAttemptContentCommitment,
  CANDIDATE_ARTIFACT_PERSISTENCE_FAILURE_PREFIX,
  candidateSeriesReservationRecord,
  openCandidateSeries,
  parseCandidateLedgerText,
  readCandidateLedger,
  reserveCandidateAttempt,
} from '../scripts/live-first-instance-candidate-ledger.mjs';
import {
  classifyUnfinalizedCandidateAttempt,
  verifyCandidateSeries,
} from '../analysis/verify-live-first-instance-candidate-pair.mjs';
import {
  recoverPendingReservation,
  runCandidateSeriesFlow,
} from '../scripts/run-live-first-instance-candidate.mjs';
import {
  evaluateLiveFirstInstanceAdapterTelemetryAssociation,
} from '../scripts/live-first-instance-adapter-telemetry-association.mjs';

const SOURCE = Object.freeze({
  commit: 'a'.repeat(40),
  tree: 'b'.repeat(40),
  trackedFilesSha256: 'c'.repeat(64),
  packageLockSha256: 'd'.repeat(64),
  executionDependencyClosureSha256: '2'.repeat(64),
  executionDependencyFileCount: 2,
  executionDependencyTotalBytes: 3,
});

const CLEAN_PROVENANCE = Object.freeze({
  status: 'available',
  captureStable: true,
  dirty: false,
  stagedChanges: 0,
  unstagedChanges: 0,
  untrackedFiles: 0,
  packageLockTracked: true,
  ...SOURCE,
});

const DEPENDENCY_CLOSURE = Object.freeze({
  schemaVersion: 1,
  kind: 'installed-execution-dependency-closure',
  root: 'node_modules',
  format: 'node-modules-sorted-path-size-content-sha256-v1',
  hashAlgorithm: 'sha256',
  exclusions: ['.bin/**', '.vite*/**'],
  fileCount: SOURCE.executionDependencyFileCount,
  totalBytes: SOURCE.executionDependencyTotalBytes,
  sha256: SOURCE.executionDependencyClosureSha256,
});

const VITE_RUNTIME_AUDIT = Object.freeze({
  schemaVersion: 3,
  kind: 'candidate-vite-runtime-module-audit',
  policyId: 'vite-unoptimized-fresh-cache-runtime-module-audit-v3',
  configuration: {
    configFile: false,
    appType: 'custom',
    optimizeDepsNoDiscovery: true,
    optimizeDepsInclude: [],
    entryHtmlPolicy: 'exact-tracked-index-html-without-vite-client',
    cacheDirectoryPolicy: 'unique-fresh-os-temporary-directory-outside-project',
    cachePreexisted: false,
  },
  cache: { existedAtFinalization: false, entryCount: 0 },
  entryHtml: {
    sourceRelativePath: 'index.html',
    sourceByteCount: 10,
    sourceSha256: '3'.repeat(64),
    successfulResponseCount: 2,
    responseHeaders: {
      crossOriginOpenerPolicy: 'same-origin',
      crossOriginEmbedderPolicy: 'require-corp',
      contentType: 'text/html; charset=utf-8',
      cacheControl: 'no-store',
    },
    responseVariants: [{
      byteCount: 10,
      sha256: '3'.repeat(64),
      responseCount: 2,
    }],
  },
  prohibitedOptimizedArtifactCount: 0,
  requiredModulePaths: [
    'src/main.js',
    'src/strategies/live-first-instance-crossover.js',
  ],
  moduleCount: 3,
  dependencyModuleCount: 1,
  modulesSha256: '4'.repeat(64),
  modules: [
    'node_modules/three/build/three.webgpu.js',
    'src/main.js',
    'src/strategies/live-first-instance-crossover.js',
  ].map((sourceRelativePath, index) => ({
    sourceRelativePath,
    sourceByteCount: 100 + index,
    sourceSha256: String(5 + index).repeat(64),
    successfulResponseCount: 2,
    transformedVariants: [{
      byteCount: 120 + index,
      sha256: ['8', '9', 'a'][index].repeat(64),
      responseCount: 2,
    }],
  })),
});

function artifactPersistenceStderr(code = 'ENOSPC') {
  return `${CANDIDATE_ARTIFACT_PERSISTENCE_FAILURE_PREFIX}${JSON.stringify({
    schemaVersion: 1,
    kind: 'first-instance-live-candidate-artifact-persistence-failure',
    operation: 'persist-run-artifacts',
    code,
  })}\n`;
}

function environmentGate({ pass = true, collectorFailure = false } = {}) {
  if (collectorFailure) {
    return {
      schemaVersion: 2,
      kind: 'first-instance-live-candidate-environment-gate',
      applicable: true,
      status: 'failed-retryable-collector',
      pass: false,
      retryable: true,
      nonReplaceable: false,
      collectorPass: false,
      processIdentityPass: null,
      adapterTelemetryAssociationPass: null,
      failureCodes: ['telemetry-status-unavailable', 'telemetry-coverage-invalid'],
      reasons: [
        'telemetry status is unavailable',
        'telemetry collector coverage audit did not pass',
      ],
    };
  }
  if (!pass) {
    return {
      schemaVersion: 2,
      kind: 'first-instance-live-candidate-environment-gate',
      applicable: true,
      status: 'failed-non-replaceable-process-set-mismatch',
      pass: false,
      retryable: false,
      nonReplaceable: true,
      collectorPass: true,
      processIdentityPass: false,
      adapterTelemetryAssociationPass: true,
      failureCodes: ['compute-process-set-mismatch'],
      reasons: ['pre-run and post-run compute-process identity sets differ'],
    };
  }
  return {
    schemaVersion: 2,
    kind: 'first-instance-live-candidate-environment-gate',
    applicable: true,
    status: 'passed',
    pass: true,
    retryable: false,
    nonReplaceable: false,
    collectorPass: true,
    processIdentityPass: true,
    adapterTelemetryAssociationPass: true,
    failureCodes: [],
    reasons: [],
  };
}

function overallDecision({ numericalPass = true, environmentPass = true } = {}) {
  const pass = numericalPass && environmentPass;
  return {
    schemaVersion: 1,
    kind: 'first-instance-live-overall-evidence-decision',
    applicable: true,
    status: pass ? 'passed' : 'failed-non-replaceable',
    pass,
    retryable: false,
    nonReplaceable: !pass,
    failedGates: [
      ...(environmentPass ? [] : ['candidateEnvironmentGate']),
      ...(numericalPass ? [] : ['preregisteredNumericalDecision']),
    ],
    failureCodes: pass ? [] : ['fixture-failure'],
    reasons: pass ? [] : ['fixture failure'],
  };
}

function metadata({
  runId,
  startedAt,
  completedAt,
  numericalPass = true,
  environmentPass = true,
  collectorFailure = false,
  browserVersion = 'Chrome/140.0.0.0',
} = {}) {
  const adapterInfo = {
    vendor: 'nvidia',
    architecture: 'fixture',
    device: 'fixture-device',
    description: 'NVIDIA RTX fixture',
    backend: 'D3D12',
    type: 'discrete GPU',
    driver: 'fixture-driver',
    isFallbackAdapter: false,
  };
  const gpuTelemetry = {
    summary: {
      gpus: [{ gpuIndex: 0, gpuName: 'NVIDIA RTX fixture', gpuUuid: 'GPU-fixture' }],
    },
  };
  const adapterTelemetryAssociation =
    evaluateLiveFirstInstanceAdapterTelemetryAssociation({
      adapterInfo,
      telemetryReport: gpuTelemetry,
    });
  const gate = environmentGate({ pass: environmentPass, collectorFailure });
  const overall = collectorFailure
    ? {
      schemaVersion: 1,
      kind: 'first-instance-live-overall-evidence-decision',
      applicable: true,
      status: 'failed-retryable-collector',
      pass: false,
      retryable: true,
      nonReplaceable: false,
      failedGates: ['candidateEnvironmentGate'],
      failureCodes: [...gate.failureCodes],
      reasons: [...gate.reasons],
    }
    : overallDecision({ numericalPass, environmentPass });
  return {
    schemaVersion: 2,
    runId,
    status: collectorFailure ? 'failed' : 'complete',
    error: collectorFailure
      ? { message: 'Live first-instance candidate telemetry collector rejected: fixture.' }
      : null,
    evidenceStatus: 'candidate',
    startedAt,
    completedAt,
    sourceProvenance: { start: { ...SOURCE }, end: { ...SOURCE }, stable: true },
    executionDependencyClosure: {
      start: { ...DEPENDENCY_CLOSURE },
      end: { ...DEPENDENCY_CLOSURE },
      stable: true,
    },
    candidateViteRuntimeAudit: structuredClone(VITE_RUNTIME_AUDIT),
    environment: {
      node: 'v22.12.0',
      platform: 'win32',
      architecture: 'x64',
      backend: 'NVIDIA RTX fixture · D3D12',
      browser: {
        executable: 'chrome.exe',
        version: browserVersion,
        headless: true,
        args: ['--enable-unsafe-webgpu'],
        viewport: { width: 1280, height: 900, deviceScaleFactor: 1 },
      },
      gpuTelemetry,
      benchmarkPage: {
        userAgent: browserVersion,
        threeRevision: '185',
        rendererBackend: 'WebGPUBackend',
        coordinateSystem: 2001,
        viewport: { width: 1280, height: 720, devicePixelRatio: 1 },
        reversedDepth: true,
        rendererReversedDepthBuffer: true,
        maxStorageBuffersPerShaderStage: 10,
        timestampAvailable: true,
        indirectFirstInstanceAvailable: true,
        crossOriginIsolated: true,
        webgpuDeviceLossCount: 0,
        adapterInfo,
      },
      benchmarkPageAtEnd: null,
    },
    protocol: {
      matrix: 'first-instance-live-o65536-b32',
      matrixKind: 'first-instance-live',
      objectCount: 65_536,
      bucketCount: 32,
      measuredFrames: 480,
    },
    workload: {
      scenarioGenerator: 'createFixedSubsetScenario',
      scenarioSeed: 2_980_910_118,
      geometryFixtureSha256: 'e'.repeat(64),
      scenarioSha256ByVisibility: { '0.2': 'f'.repeat(64), '0.99': '1'.repeat(64) },
    },
    liveFirstInstanceEnvironmentAudit: {
      schemaVersion: 4,
      kind: 'first-instance-live-crossover-environment-audit',
      telemetryStatus: collectorFailure ? 'unavailable' : 'available',
      telemetryMalformedLineCount: collectorFailure ? null : 0,
      telemetryStderrByteCount: collectorFailure ? null : 0,
      telemetrySampleCount: collectorFailure ? null : 10,
      telemetryCoveragePass: collectorFailure ? null : true,
      adapterTelemetryAssociation,
      computeProcessIdentityComparison: environmentPass
        ? { pass: true, pre: [], post: [], reasons: [] }
        : { pass: false, pre: [], post: [{}], reasons: [
          'pre-run and post-run compute-process identity sets differ',
        ] },
      candidateEnvironmentGate: gate,
      overallEvidenceDecision: overall,
    },
    expectedTrialCount: 24,
    completedTrialCount: 24,
    acceptedTrialCount: 24,
    validationArtifactCount: 24,
    frameRowCount: 11_520,
    pageErrors: [],
    webgpuUncapturedErrors: [],
    liveFirstInstanceAnalysisAudit: { nTrials: 24, nRows: 11_520 },
    fixtureNumericalPass: numericalPass,
  };
}

async function writeConsistentFixtureManifest(runDirectory, runMetadata) {
  const bytes = Buffer.from(`${JSON.stringify(runMetadata, null, 2)}\n`);
  await writeFile(path.join(runDirectory, 'metadata.json'), bytes);
  await writeFile(path.join(runDirectory, 'artifact-manifest.json'), `${JSON.stringify({
    requiredFiles: ['metadata.json'],
    optionalFiles: [],
    files: [{
      name: 'metadata.json',
      required: true,
      present: true,
      bytes: bytes.length,
      sha256: createHash('sha256').update(bytes).digest('hex'),
    }],
  })}\n`);
}

async function fixtureSeries(t) {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'first-instance-ledger-'));
  t.after(async () => {
    const { rm } = await import('node:fs/promises');
    await rm(directory, { recursive: true, force: true });
  });
  const opened = await openCandidateSeries(
    directory,
    SOURCE,
    new Date('2026-09-01T00:00:00.000Z'),
  );
  return { directory, ledgerPath: opened.ledgerPath };
}

function verifierFor(validRunNames) {
  return async (runDirectory) => {
    if (!validRunNames.has(path.basename(runDirectory))) {
      throw new Error('fixture run verification failure');
    }
    return {
      artifactVerification: {
        status: 'consistent',
        evidenceStatus: 'candidate',
      },
    };
  };
}

async function summarizeFixture(runDirectory) {
  const runMetadata = JSON.parse(await readFile(path.join(runDirectory, 'metadata.json'), 'utf8'));
  return {
    artifactVerification: { status: 'consistent', evidenceStatus: 'candidate' },
    preregisteredNumericalDecision: {
      status: 'evaluated',
      pass: runMetadata.fixtureNumericalPass,
      failedGates: runMetadata.fixtureNumericalPass ? [] : ['highVisibilityTotalMaterial'],
    },
    liveFirstInstanceEvidenceDecision: {
      adapterTelemetryAssociation:
        runMetadata.liveFirstInstanceEnvironmentAudit.adapterTelemetryAssociation,
      candidateEnvironmentGate:
        runMetadata.liveFirstInstanceEnvironmentAudit.candidateEnvironmentGate,
      overallEvidenceDecision:
        runMetadata.liveFirstInstanceEnvironmentAudit.overallEvidenceDecision,
    },
  };
}

async function verifyFixtureCandidateSeries(seriesDirectory, options = {}) {
  const {
    registryMaterializedRecordedAt = '2026-09-01T00:00:00.000Z',
    ...verificationOptions
  } = options;
  return verifyCandidateSeries(seriesDirectory, {
    ...verificationOptions,
    candidateSeriesRoot: path.dirname(seriesDirectory),
    registryVerifier: async (_candidateRoot, requestedSeries) => {
      const state = await readCandidateLedger(
        path.join(requestedSeries, 'candidate-attempts.jsonl'),
      );
      return {
        schemaVersion: 1,
        kind: 'first-instance-live-candidate-registry-verification',
        experimentId: 'first-instance-live-first-device-v1',
        studyKey: '7'.repeat(64),
        seriesDirectory: path.basename(requestedSeries),
        sourceIdentity: state.sourceIdentity,
        seriesId: state.seriesId,
        registryFilename: 'candidate-series-registry.jsonl',
        registryEventCount: 2,
        registryFinalEventSha256: '8'.repeat(64),
        registryClaimEventSha256: '9'.repeat(64),
        registryMaterializedEventSha256: '1'.repeat(64),
        registryMaterializedRecordedAt,
        seriesOpeningEventSha256: state.events[0].eventSha256,
        anchor: {
          tagName: 'fixture-anchor',
          messageSha256: '2'.repeat(64),
        },
      };
    },
    anchorTagVerifier: async (_repositoryRoot, registry) => ({
      ...registry.anchor,
      verified: true,
    }),
  });
}

test('candidate CLI auto-runs matrix 2 after a valid matrix 1', async () => {
  const results = [
    {
      classification: 'valid-candidate',
      completedMatrixCount: 1,
      matrixOrdinal: 1,
    },
    {
      classification: 'valid-candidate',
      completedMatrixCount: 2,
      matrixOrdinal: 2,
    },
  ];
  const calls = [];
  const pair = { decision: { pass: false } };
  const flow = await runCandidateSeriesFlow(
    path.join(process.cwd(), 'results', 'candidate-series', 'flow-auto-matrix-two'),
    {
      runAttempt: async (seriesInput) => {
        calls.push(seriesInput);
        return results.shift();
      },
      verifySeries: async () => pair,
      emitPair: () => {},
    },
  );
  assert.equal(calls.length, 2);
  assert.equal(flow.status, 'verified-two-matrix-series');
  assert.equal(flow.attemptCount, 2);
  assert.equal(flow.pair, pair);
});

test('candidate CLI stops after one retryable infrastructure attempt', async () => {
  let attemptCalls = 0;
  let verifyCalls = 0;
  const flow = await runCandidateSeriesFlow(
    path.join(process.cwd(), 'results', 'candidate-series', 'flow-retry-stop'),
    {
      runAttempt: async () => {
        attemptCalls += 1;
        return {
          classification: 'infrastructure-invalid-retryable',
          completedMatrixCount: 0,
          matrixOrdinal: 1,
        };
      },
      verifySeries: async () => {
        verifyCalls += 1;
        return {};
      },
      emitPair: () => {},
    },
  );
  assert.equal(attemptCalls, 1);
  assert.equal(verifyCalls, 0);
  assert.equal(flow.status, 'stopped-retryable-infrastructure');
  assert.equal(flow.pair, null);
});

async function addAttempt({
  directory,
  ledgerPath,
  runMetadata = null,
  runName = null,
  validRunNames,
  stderr = '',
  lifecycleExitCode = undefined,
  lifecycleSignal = null,
  reservedAtOverride = null,
  finalizedAtOverride = null,
  extraAttemptFile = null,
  extraRunName = null,
  mutateReservation = null,
  mutateRunArtifactsBeforeClassification = null,
}) {
  let state = await readCandidateLedger(ledgerPath);
  const attemptOrdinal = state.nextAttemptOrdinal;
  const reservedAt = reservedAtOverride ?? (runMetadata === null
    ? new Date(`2026-09-01T00:${String(attemptOrdinal * 2).padStart(2, '0')}:00.000Z`)
    : new Date(Date.parse(runMetadata.startedAt) - 1_000));
  const reserved = await reserveCandidateAttempt(
    directory,
    ledgerPath,
    state,
    SOURCE,
    reservedAt,
  );
  await writeFile(path.join(reserved.absoluteAttemptDirectory, 'runner-stdout.log'), 'fixture\n');
  await writeFile(path.join(reserved.absoluteAttemptDirectory, 'runner-stderr.log'), stderr);
  await writeFile(
    path.join(reserved.absoluteAttemptDirectory, 'orchestrator-source-after.json'),
    `${JSON.stringify({
      schemaVersion: 1,
      kind: 'first-instance-live-orchestrator-source-after',
      sourceProvenance: CLEAN_PROVENANCE,
      executionDependencyClosure: DEPENDENCY_CLOSURE,
    })}\n`,
  );
  const expectedExitCode = lifecycleSignal === null
    ? lifecycleExitCode ?? (runName !== null && validRunNames.has(runName) ? 0 : 1)
    : null;
  await writeFile(
    path.join(reserved.absoluteAttemptDirectory, 'runner-child-lifecycle.json'),
    `${JSON.stringify({
      schemaVersion: 1,
      kind: 'first-instance-live-candidate-child-lifecycle',
      status: 'closed',
      childPid: 1234,
      spawnedAt: reservedAt.toISOString(),
      closedAt: (runMetadata === null
        ? new Date(reservedAt.getTime() + 1_000)
        : new Date(Date.parse(runMetadata.completedAt) + 500)).toISOString(),
      exitCode: expectedExitCode,
      signal: lifecycleSignal,
    })}\n`,
  );
  if (extraAttemptFile !== null) {
    await writeFile(path.join(reserved.absoluteAttemptDirectory, extraAttemptFile), 'extra');
  }
  if (runMetadata !== null) {
    runMetadata.candidateSeriesReservation = candidateSeriesReservationRecord(reserved.event);
    mutateReservation?.(runMetadata.candidateSeriesReservation);
    const runDirectory = path.join(reserved.absoluteAttemptDirectory, 'runs', runName);
    await mkdir(runDirectory);
    await writeConsistentFixtureManifest(runDirectory, runMetadata);
    await mutateRunArtifactsBeforeClassification?.(runDirectory);
  }
  if (extraRunName !== null) {
    await mkdir(path.join(reserved.absoluteAttemptDirectory, 'runs', extraRunName));
  }
  const classified = await classifyUnfinalizedCandidateAttempt(
    directory,
    reserved.event.attemptDirectory,
    {
      runVerifier: verifierFor(validRunNames),
      expectedSourceIdentity: reserved.event.sourceIdentity,
      expectedReservation: reserved.event,
    },
  );
  const contentCommitment = await buildAttemptContentCommitment(
    reserved.absoluteAttemptDirectory,
  );
  state = await readCandidateLedger(ledgerPath);
  await appendCandidateLedgerEvent(ledgerPath, {
    eventType: 'attempt-finalized',
    seriesId: state.seriesId,
    recordedAt: (finalizedAtOverride ?? (runMetadata === null
      ? new Date(
        `2026-09-01T00:${String(attemptOrdinal * 2 + 1).padStart(2, '0')}:00.000Z`,
      )
      : new Date(Date.parse(runMetadata.completedAt) + 1_000))).toISOString(),
    attemptOrdinal: reserved.event.attemptOrdinal,
    matrixOrdinal: reserved.event.matrixOrdinal,
    attemptDirectory: reserved.event.attemptDirectory,
    reservationEventSha256: reserved.event.eventSha256,
    classification: classified.classification,
    reasonCode: classified.reasonCode,
    runDirectory: classified.runDirectoryRelative,
    execution: { exitCode: expectedExitCode, signal: lifecycleSignal },
    contentCommitment,
    classificationEvidence: classified.classificationEvidence,
  });
  return classified;
}

test('two same-identity passing matrices produce the conjunctive pair pass', async (t) => {
  const fixture = await fixtureSeries(t);
  const valid = new Set(['matrix-1', 'matrix-2']);
  await addAttempt({
    ...fixture,
    runMetadata: metadata({
      runId: 'matrix-1',
      startedAt: '2026-09-01T01:00:00.000Z',
      completedAt: '2026-09-01T01:10:00.000Z',
    }),
    runName: 'matrix-1',
    validRunNames: valid,
  });
  await addAttempt({
    ...fixture,
    runMetadata: metadata({
      runId: 'matrix-2',
      startedAt: '2026-09-01T01:11:00.000Z',
      completedAt: '2026-09-01T01:21:00.000Z',
    }),
    runName: 'matrix-2',
    validRunNames: valid,
  });

  const result = await verifyFixtureCandidateSeries(fixture.directory, {
    runVerifier: verifierFor(valid),
    summarizeRun: summarizeFixture,
  });
  assert.equal(result.attemptCount, 2);
  assert.equal(result.validCandidateCount, 2);
  assert.equal(result.registry.studyKey, '7'.repeat(64));
  assert.equal(result.registry.anchorVerified, true);
  assert.equal(result.pairEligibility.canonicalRegistryAndAnchorBound, true);
  assert.equal(result.pairEligibility.pass, true);
  assert.equal(result.decision.pass, true);
});

test('private pair verification fails closed on absent or late materialization chronology', async (t) => {
  const fixture = await fixtureSeries(t);
  const valid = new Set(['late-materialization']);
  await addAttempt({
    ...fixture,
    runMetadata: metadata({
      runId: 'late-materialization',
      startedAt: '2026-09-01T01:00:00.000Z',
      completedAt: '2026-09-01T01:10:00.000Z',
    }),
    runName: 'late-materialization',
    validRunNames: valid,
  });

  await assert.rejects(
    verifyFixtureCandidateSeries(fixture.directory, {
      runVerifier: verifierFor(valid),
      summarizeRun: summarizeFixture,
      registryMaterializedRecordedAt: '2026-09-01T01:00:00.000Z',
    }),
    /registry materialization timestamp is missing, malformed, or later than the first attempt reservation/,
  );
  await assert.rejects(
    verifyFixtureCandidateSeries(fixture.directory, {
      runVerifier: verifierFor(valid),
      summarizeRun: summarizeFixture,
      registryMaterializedRecordedAt: null,
    }),
    /registry materialization timestamp is missing, malformed, or later than the first attempt reservation/,
  );
});

test('a numerical miss and a stable process mismatch remain consumed candidates', async (t) => {
  const fixture = await fixtureSeries(t);
  const valid = new Set(['numeric-miss', 'process-mismatch']);
  await addAttempt({
    ...fixture,
    runMetadata: metadata({
      runId: 'numeric-miss',
      startedAt: '2026-09-01T02:00:00.000Z',
      completedAt: '2026-09-01T02:10:00.000Z',
      numericalPass: false,
    }),
    runName: 'numeric-miss',
    validRunNames: valid,
  });
  await addAttempt({
    ...fixture,
    runMetadata: metadata({
      runId: 'process-mismatch',
      startedAt: '2026-09-01T02:11:00.000Z',
      completedAt: '2026-09-01T02:21:00.000Z',
      environmentPass: false,
    }),
    runName: 'process-mismatch',
    validRunNames: valid,
  });

  const result = await verifyFixtureCandidateSeries(fixture.directory, {
    runVerifier: verifierFor(valid),
    summarizeRun: summarizeFixture,
  });
  assert.deepEqual(result.attempts.map(({ classification }) => classification), [
    'valid-candidate',
    'valid-candidate',
  ]);
  assert.equal(result.matrices[0].numericalPass, false);
  assert.equal(result.matrices[1].environmentPass, false);
  assert.equal(result.pairEligibility.pass, true);
  assert.equal(result.decision.pass, false);
});

test('only preserved infrastructure failure retries the same matrix ordinal', async (t) => {
  const fixture = await fixtureSeries(t);
  const valid = new Set(['matrix-1', 'matrix-2']);
  const interrupted = await addAttempt({
    ...fixture,
    validRunNames: valid,
    lifecycleSignal: 'SIGTERM',
  });
  assert.equal(interrupted.classification, 'infrastructure-invalid-retryable');
  assert.equal(interrupted.reasonCode, 'run-artifact-incompleteness-or-corruption');
  assert.equal(
    interrupted.classificationEvidence.artifactInfrastructureFailureObserved,
    true,
  );
  await addAttempt({
    ...fixture,
    runMetadata: metadata({
      runId: 'matrix-1',
      startedAt: '2026-09-01T03:00:00.000Z',
      completedAt: '2026-09-01T03:10:00.000Z',
    }),
    runName: 'matrix-1',
    validRunNames: valid,
  });
  await addAttempt({
    ...fixture,
    runMetadata: metadata({
      runId: 'matrix-2',
      startedAt: '2026-09-01T03:11:00.000Z',
      completedAt: '2026-09-01T03:21:00.000Z',
    }),
    runName: 'matrix-2',
    validRunNames: valid,
  });

  const result = await verifyFixtureCandidateSeries(fixture.directory, {
    runVerifier: verifierFor(valid),
    summarizeRun: summarizeFixture,
  });
  assert.deepEqual(result.attempts.map(({ matrixOrdinal }) => matrixOrdinal), [1, 1, 2]);
  assert.equal(result.retryCount, 1);
  assert.equal(result.decision.pass, true);
});

test('collector failure is retryable but a consistent ordinary failure terminates the series', async (t) => {
  const collectorFixture = await fixtureSeries(t);
  const collector = await addAttempt({
    ...collectorFixture,
    runMetadata: metadata({
      runId: 'collector-failure',
      startedAt: '2026-09-01T04:00:00.000Z',
      completedAt: '2026-09-01T04:10:00.000Z',
      collectorFailure: true,
    }),
    runName: 'collector-failure',
    validRunNames: new Set(),
  });
  assert.equal(collector.classification, 'infrastructure-invalid-retryable');
  assert.equal(collector.reasonCode, 'telemetry-collector-failure');

  const failureFixture = await fixtureSeries(t);
  const ordinary = await addAttempt({
    ...failureFixture,
    runMetadata: metadata({
      runId: 'ordinary-failure',
      startedAt: '2026-09-01T05:00:00.000Z',
      completedAt: '2026-09-01T05:10:00.000Z',
    }),
    runName: 'ordinary-failure',
    validRunNames: new Set(),
  });
  assert.equal(ordinary.classification, 'implementation/evidence-failure-nonretryable');
  const state = await readCandidateLedger(failureFixture.ledgerPath);
  assert.notEqual(state.terminalFailure, null);
  await assert.rejects(
    reserveCandidateAttempt(
      failureFixture.directory,
      failureFixture.ledgerPath,
      state,
      SOURCE,
    ),
    /ended with a non-retryable implementation\/evidence failure/,
  );
});

test('artifact incompleteness retries only with preserved infrastructure provenance', async (t) => {
  await t.test('successful completed metadata cannot be discarded by corrupting its manifest', async () => {
    const fixture = await fixtureSeries(t);
    const classified = await addAttempt({
      ...fixture,
      runMetadata: metadata({
        runId: 'completed-then-corrupt',
        startedAt: '2026-09-01T05:11:00.000Z',
        completedAt: '2026-09-01T05:21:00.000Z',
      }),
      runName: 'completed-then-corrupt',
      validRunNames: new Set(),
      lifecycleExitCode: 0,
      async mutateRunArtifactsBeforeClassification(runDirectory) {
        await writeFile(path.join(runDirectory, 'artifact-manifest.json'), '{corrupt');
      },
    });
    assert.equal(classified.classification, 'implementation/evidence-failure-nonretryable');
    assert.equal(classified.classificationEvidence.artifactManifestStatus, 'incomplete-or-corrupt');
    assert.equal(
      classified.classificationEvidence.artifactInfrastructureFailureObserved,
      false,
    );
  });

  await t.test('complete metadata stays consumed even when a later signal resembles device loss', async () => {
    const fixture = await fixtureSeries(t);
    const completedMetadata = metadata({
      runId: 'completed-before-signal',
      startedAt: '2026-09-01T05:21:01.000Z',
      completedAt: '2026-09-01T05:21:02.000Z',
    });
    completedMetadata.pageErrors = [{ source: 'page', detail: 'GPU device lost' }];
    const classified = await addAttempt({
      ...fixture,
      runMetadata: completedMetadata,
      runName: 'completed-before-signal',
      validRunNames: new Set(),
      lifecycleSignal: 'SIGKILL',
      async mutateRunArtifactsBeforeClassification(runDirectory) {
        await writeFile(path.join(runDirectory, 'artifact-manifest.json'), '{corrupt');
      },
    });
    assert.equal(classified.classification, 'implementation/evidence-failure-nonretryable');
    assert.equal(classified.classificationEvidence.browserOrDeviceLossObserved, true);
    assert.equal(classified.classificationEvidence.artifactManifestStatus, 'incomplete-or-corrupt');
  });

  await t.test('shader or validation failure stays nonretryable with an incomplete manifest', async () => {
    const fixture = await fixtureSeries(t);
    const failedMetadata = metadata({
      runId: 'shader-failure',
      startedAt: '2026-09-01T05:22:00.000Z',
      completedAt: '2026-09-01T05:32:00.000Z',
    });
    failedMetadata.status = 'failed';
    failedMetadata.error = {
      name: 'Error',
      message: 'GPUValidationError: shader binding validation failed near ENOSPC',
      stack: null,
    };
    const classified = await addAttempt({
      ...fixture,
      runMetadata: failedMetadata,
      runName: 'shader-failure',
      validRunNames: new Set(),
      stderr: artifactPersistenceStderr(),
      async mutateRunArtifactsBeforeClassification(runDirectory) {
        await writeFile(path.join(runDirectory, 'artifact-manifest.json'), '{corrupt');
      },
    });
    assert.equal(classified.classification, 'implementation/evidence-failure-nonretryable');
    assert.equal(
      classified.classificationEvidence.artifactInfrastructureFailureObserved,
      false,
    );
  });

  await t.test('exact allowlisted artifact I/O marker permits a retry before metadata completion', async () => {
    const fixture = await fixtureSeries(t);
    const classified = await addAttempt({
      ...fixture,
      validRunNames: new Set(),
      stderr: artifactPersistenceStderr(),
    });
    assert.equal(classified.classification, 'infrastructure-invalid-retryable');
    assert.equal(classified.reasonCode, 'run-artifact-incompleteness-or-corruption');
    assert.equal(
      classified.classificationEvidence.artifactInfrastructureFailureObserved,
      true,
    );
  });

  await t.test('failed metadata retries only when it binds the exact marker I/O code', async () => {
    const fixture = await fixtureSeries(t);
    const persistenceMetadata = metadata({
      runId: 'artifact-io-failure',
      startedAt: '2026-09-01T05:32:01.000Z',
      completedAt: '2026-09-01T05:32:02.000Z',
    });
    persistenceMetadata.status = 'failed';
    persistenceMetadata.error = {
      name: 'Error',
      message: 'ENOSPC: no space left on device, write',
      stack: 'Error: ENOSPC: no space left on device, write',
    };
    const classified = await addAttempt({
      ...fixture,
      runMetadata: persistenceMetadata,
      runName: 'artifact-io-failure',
      validRunNames: new Set(),
      stderr: artifactPersistenceStderr(),
    });
    assert.equal(classified.classification, 'infrastructure-invalid-retryable');
    assert.equal(classified.reasonCode, 'run-artifact-incompleteness-or-corruption');
    assert.equal(
      classified.classificationEvidence.artifactInfrastructureFailureObserved,
      true,
    );
  });

  await t.test('verifier-passing artifacts with a nonzero child exit cannot count as valid', async () => {
    const fixture = await fixtureSeries(t);
    const valid = new Set(['nonzero-after-complete']);
    const classified = await addAttempt({
      ...fixture,
      runMetadata: metadata({
        runId: 'nonzero-after-complete',
        startedAt: '2026-09-01T05:33:00.000Z',
        completedAt: '2026-09-01T05:43:00.000Z',
      }),
      runName: 'nonzero-after-complete',
      validRunNames: valid,
      lifecycleExitCode: 1,
    });
    assert.equal(classified.classification, 'implementation/evidence-failure-nonretryable');
  });
});

test('pending recovery refuses to overlap a recorded live child', async (t) => {
  const fixture = await fixtureSeries(t);
  let state = await readCandidateLedger(fixture.ledgerPath);
  const reserved = await reserveCandidateAttempt(
    fixture.directory,
    fixture.ledgerPath,
    state,
    SOURCE,
    new Date('2026-09-01T05:30:00.000Z'),
  );
  await writeFile(
    path.join(reserved.absoluteAttemptDirectory, 'runner-child-lifecycle.json'),
    `${JSON.stringify({
      schemaVersion: 1,
      kind: 'first-instance-live-candidate-child-lifecycle',
      status: 'running',
      childPid: process.pid,
      spawnedAt: '2026-09-01T05:30:00.000Z',
      closedAt: null,
      exitCode: null,
      signal: null,
    })}\n`,
  );
  state = await readCandidateLedger(fixture.ledgerPath);
  await assert.rejects(
    recoverPendingReservation(fixture.directory, fixture.ledgerPath, state),
    /still has a live child process/,
  );
  assert.notEqual((await readCandidateLedger(fixture.ledgerPath)).pendingReservation, null);
});

test('pending recovery fails closed without an exact completion marker', async (t) => {
  const fixture = await fixtureSeries(t);
  let state = await readCandidateLedger(fixture.ledgerPath);
  await reserveCandidateAttempt(
    fixture.directory,
    fixture.ledgerPath,
    state,
    SOURCE,
    new Date('2026-09-01T05:40:00.000Z'),
  );
  state = await readCandidateLedger(fixture.ledgerPath);
  await assert.rejects(
    recoverPendingReservation(fixture.directory, fixture.ledgerPath, state),
    /lacks an unambiguous closed-child marker/,
  );
  assert.notEqual((await readCandidateLedger(fixture.ledgerPath)).pendingReservation, null);
});

test('ledger chain, directory inventory, content commitment, and identity fail closed', async (t) => {
  await t.test('event mutation breaks the hash chain', async () => {
    const fixture = await fixtureSeries(t);
    const text = await readFile(fixture.ledgerPath, 'utf8');
    const changed = text.replace('series-opened', 'series-opener');
    assert.throws(() => parseCandidateLedgerText(changed), /unsupported eventType|digest/);
  });

  await t.test('an unlisted attempt directory is rejected', async () => {
    const fixture = await fixtureSeries(t);
    await mkdir(path.join(fixture.directory, 'attempt-9999'));
    await assert.rejects(
      verifyFixtureCandidateSeries(fixture.directory, {
        runVerifier: verifierFor(new Set()),
        summarizeRun: summarizeFixture,
      }),
      /inventory does not exactly match/,
    );
  });

  await t.test('post-finalization content mutation is rejected', async () => {
    const fixture = await fixtureSeries(t);
    await addAttempt({ ...fixture, validRunNames: new Set() });
    await writeFile(path.join(fixture.directory, 'attempt-0001', 'late-file'), 'late');
    await assert.rejects(
      verifyFixtureCandidateSeries(fixture.directory, {
        runVerifier: verifierFor(new Set()),
        summarizeRun: summarizeFixture,
      }),
      /content changed after finalization/,
    );
  });

  await t.test('alternate evidence outside the exact attempt layout is nonretryable', async () => {
    const fixture = await fixtureSeries(t);
    const classified = await addAttempt({
      ...fixture,
      validRunNames: new Set(),
      extraAttemptFile: 'discarded-result.json',
    });
    assert.equal(classified.classification, 'implementation/evidence-failure-nonretryable');
    assert.equal(classified.classificationEvidence.attemptLayoutExact, false);
  });

  await t.test('a surplus runner output cannot be relabeled as incomplete retryable evidence', async () => {
    const fixture = await fixtureSeries(t);
    const valid = new Set(['kept-run']);
    const classified = await addAttempt({
      ...fixture,
      runMetadata: metadata({
        runId: 'kept-run',
        startedAt: '2026-09-01T07:30:00.000Z',
        completedAt: '2026-09-01T07:40:00.000Z',
      }),
      runName: 'kept-run',
      extraRunName: 'discarded-run',
      validRunNames: valid,
    });
    assert.equal(classified.classification, 'implementation/evidence-failure-nonretryable');
    assert.equal(classified.classificationEvidence.runDirectoryCount, 2);
  });

  await t.test('candidate metadata must bind the exact reservation event', async () => {
    const fixture = await fixtureSeries(t);
    const valid = new Set(['wrong-reservation']);
    const classified = await addAttempt({
      ...fixture,
      runMetadata: metadata({
        runId: 'wrong-reservation',
        startedAt: '2026-09-01T07:41:00.000Z',
        completedAt: '2026-09-01T07:51:00.000Z',
      }),
      runName: 'wrong-reservation',
      validRunNames: valid,
      mutateReservation(reservation) {
        reservation.reservationEventSha256 = '0'.repeat(64);
      },
    });
    assert.equal(classified.classification, 'implementation/evidence-failure-nonretryable');
    assert.equal(
      classified.classificationEvidence.candidateReservationBindingStatus,
      'mismatched',
    );
  });

  await t.test('metadata timing must remain inside reservation and finalization', async () => {
    const fixture = await fixtureSeries(t);
    const valid = new Set(['bad-timeline']);
    await addAttempt({
      ...fixture,
      runMetadata: metadata({
        runId: 'bad-timeline',
        startedAt: '2026-09-01T08:00:00.000Z',
        completedAt: '2026-09-01T08:10:00.000Z',
      }),
      runName: 'bad-timeline',
      validRunNames: valid,
      reservedAtOverride: new Date('2026-09-01T08:01:00.000Z'),
      finalizedAtOverride: new Date('2026-09-01T08:11:00.000Z'),
    });
    await assert.rejects(
      verifyFixtureCandidateSeries(fixture.directory, {
        runVerifier: verifierFor(valid),
        summarizeRun: summarizeFixture,
      }),
      /does not prove reservation before timing/,
    );
  });

  await t.test('an active or stale series lock is not ignored by verification', async () => {
    const fixture = await fixtureSeries(t);
    await writeFile(
      path.join(fixture.directory, '.candidate-series.lock'),
      `${JSON.stringify({ pid: 1, createdAt: '2026-09-01T00:00:00.000Z' })}\n`,
    );
    await assert.rejects(
      verifyFixtureCandidateSeries(fixture.directory, {
        runVerifier: verifierFor(new Set()),
        summarizeRun: summarizeFixture,
      }),
      /unexpected entry ".candidate-series.lock"/,
    );
  });

  await t.test('different browser build fails same-environment pairing', async () => {
    const fixture = await fixtureSeries(t);
    const valid = new Set(['matrix-1', 'matrix-2']);
    await addAttempt({
      ...fixture,
      runMetadata: metadata({
        runId: 'matrix-1',
        startedAt: '2026-09-01T06:00:00.000Z',
        completedAt: '2026-09-01T06:10:00.000Z',
      }),
      runName: 'matrix-1',
      validRunNames: valid,
    });
    await addAttempt({
      ...fixture,
      runMetadata: metadata({
        runId: 'matrix-2',
        startedAt: '2026-09-01T06:11:00.000Z',
        completedAt: '2026-09-01T06:21:00.000Z',
        browserVersion: 'Chrome/141.0.0.0',
      }),
      runName: 'matrix-2',
      validRunNames: valid,
    });
    await assert.rejects(
      verifyFixtureCandidateSeries(fixture.directory, {
        runVerifier: verifierFor(valid),
        summarizeRun: summarizeFixture,
      }),
      /differ in frozen source, dependency, Vite runtime audit, browser, backend, adapter\/driver, adapter-to-telemetry association, or workload/,
    );
  });

  await t.test('adapter-to-telemetry association is independently reconstructed', async () => {
    const fixture = await fixtureSeries(t);
    const valid = new Set(['matrix-1', 'matrix-2']);
    await addAttempt({
      ...fixture,
      runMetadata: metadata({
        runId: 'matrix-1',
        startedAt: '2026-09-01T06:30:00.000Z',
        completedAt: '2026-09-01T06:40:00.000Z',
      }),
      runName: 'matrix-1',
      validRunNames: valid,
    });
    const secondMetadata = metadata({
      runId: 'matrix-2',
      startedAt: '2026-09-01T06:41:00.000Z',
      completedAt: '2026-09-01T06:51:00.000Z',
    });
    secondMetadata.environment.benchmarkPage.adapterInfo.description =
      'NVIDIA RTX substituted';
    await addAttempt({
      ...fixture,
      runMetadata: secondMetadata,
      runName: 'matrix-2',
      validRunNames: valid,
    });
    await assert.rejects(
      verifyFixtureCandidateSeries(fixture.directory, {
        runVerifier: verifierFor(valid),
        summarizeRun: summarizeFixture,
      }),
      /adapter-to-telemetry association differs from the page adapter and telemetry GPU evidence/,
    );
  });

  await t.test('multi-GPU telemetry cannot enter a candidate pair identity', async () => {
    const fixture = await fixtureSeries(t);
    const valid = new Set(['matrix-1', 'matrix-2']);
    await addAttempt({
      ...fixture,
      runMetadata: metadata({
        runId: 'matrix-1',
        startedAt: '2026-09-01T06:55:00.000Z',
        completedAt: '2026-09-01T07:05:00.000Z',
      }),
      runName: 'matrix-1',
      validRunNames: valid,
    });
    const secondMetadata = metadata({
      runId: 'matrix-2',
      startedAt: '2026-09-01T07:06:00.000Z',
      completedAt: '2026-09-01T07:16:00.000Z',
    });
    secondMetadata.environment.gpuTelemetry.summary.gpus.push({
      gpuIndex: 1,
      gpuName: 'NVIDIA RTX fixture',
      gpuUuid: 'GPU-fixture-second',
    });
    await addAttempt({
      ...fixture,
      runMetadata: secondMetadata,
      runName: 'matrix-2',
      validRunNames: valid,
    });
    await assert.rejects(
      verifyFixtureCandidateSeries(fixture.directory, {
        runVerifier: verifierFor(valid),
        summarizeRun: summarizeFixture,
      }),
      /telemetry must identify exactly one concrete physical GPU/,
    );
  });

  await t.test('different canonical Vite runtime graph fails pairing', async () => {
    const fixture = await fixtureSeries(t);
    const valid = new Set(['matrix-1', 'matrix-2']);
    await addAttempt({
      ...fixture,
      runMetadata: metadata({
        runId: 'matrix-1',
        startedAt: '2026-09-01T06:30:00.000Z',
        completedAt: '2026-09-01T06:40:00.000Z',
      }),
      runName: 'matrix-1',
      validRunNames: valid,
    });
    const secondMetadata = metadata({
      runId: 'matrix-2',
      startedAt: '2026-09-01T06:41:00.000Z',
      completedAt: '2026-09-01T06:51:00.000Z',
    });
    secondMetadata.candidateViteRuntimeAudit.modules[0]
      .transformedVariants[0].sha256 = 'f'.repeat(64);
    secondMetadata.candidateViteRuntimeAudit.modulesSha256 = 'f'.repeat(64);
    await addAttempt({
      ...fixture,
      runMetadata: secondMetadata,
      runName: 'matrix-2',
      validRunNames: valid,
    });
    await assert.rejects(
      verifyFixtureCandidateSeries(fixture.directory, {
        runVerifier: verifierFor(valid),
        summarizeRun: summarizeFixture,
      }),
      /differ in frozen source, dependency, Vite runtime audit/,
    );
  });

  await t.test('run provenance must match the pre-timing series reservation', async () => {
    const fixture = await fixtureSeries(t);
    const valid = new Set(['matrix-1', 'matrix-2']);
    const wrongSourceOne = metadata({
      runId: 'matrix-1',
      startedAt: '2026-09-01T07:00:00.000Z',
      completedAt: '2026-09-01T07:10:00.000Z',
    });
    const wrongSourceTwo = metadata({
      runId: 'matrix-2',
      startedAt: '2026-09-01T07:11:00.000Z',
      completedAt: '2026-09-01T07:21:00.000Z',
    });
    for (const runMetadata of [wrongSourceOne, wrongSourceTwo]) {
      runMetadata.sourceProvenance.start.commit = '9'.repeat(40);
      runMetadata.sourceProvenance.end.commit = '9'.repeat(40);
    }
    await addAttempt({
      ...fixture,
      runMetadata: wrongSourceOne,
      runName: 'matrix-1',
      validRunNames: valid,
    });
    await addAttempt({
      ...fixture,
      runMetadata: wrongSourceTwo,
      runName: 'matrix-2',
      validRunNames: valid,
    });
    await assert.rejects(
      verifyFixtureCandidateSeries(fixture.directory, {
        runVerifier: verifierFor(valid),
        summarizeRun: summarizeFixture,
      }),
      /source provenance differs from its pre-timing ledger reservation/,
    );
  });
});
