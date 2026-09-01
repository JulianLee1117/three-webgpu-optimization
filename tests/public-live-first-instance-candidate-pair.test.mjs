import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  symlink,
  writeFile,
} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  verifyPublicLiveCandidatePairBundle,
} from '../analysis/verify-public-live-first-instance-candidate-pair.mjs';
import {
  buildPublicLiveCandidatePairBundle,
  serializePublicLiveCandidatePairBuildResult,
} from '../scripts/build-public-live-first-instance-candidate-pair.mjs';
import {
  appendCandidateLedgerEvent,
  buildAttemptContentCommitment,
  openCandidateSeries,
  readCandidateLedger,
  reserveCandidateAttempt,
  sha256Canonical,
} from '../scripts/live-first-instance-candidate-ledger.mjs';
import {
  CANDIDATE_REGISTRY_LOCK_FILENAME,
  candidateRegistryAnchor,
  claimCandidateStudy,
  materializeCandidateStudy,
  readCandidateRegistry,
} from '../scripts/live-first-instance-candidate-registry.mjs';
import {
  evaluateLiveFirstInstanceAdapterTelemetryAssociation,
} from '../scripts/live-first-instance-adapter-telemetry-association.mjs';
import {
  PUBLIC_LIVE_PAIR_BUNDLE_FILES,
  PUBLIC_LIVE_PAIR_LEDGER_NAME,
  PUBLIC_LIVE_PAIR_MANIFEST_NAME,
  PUBLIC_LIVE_PAIR_RECEIPT_NAME,
  PUBLIC_LIVE_PAIR_REGISTRY_NAME,
} from '../scripts/public-live-first-instance-pair-policy.mjs';

const PRIVATE_PATH_SENTINEL = 'C:\\Users\\private-owner\\candidate-private-app.exe';
const PRIVATE_GPU_SENTINEL = 'GPU-01234567-89ab-cdef-0123-456789abcdef';

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
}

function sourceIdentity() {
  return {
    commit: 'a'.repeat(40),
    tree: 'b'.repeat(40),
    trackedFilesSha256: '1'.repeat(64),
    packageLockSha256: '2'.repeat(64),
    executionDependencyClosureSha256: '3'.repeat(64),
    executionDependencyFileCount: 321,
    executionDependencyTotalBytes: 654321,
  };
}

function classificationEvidence() {
  return {
    artifactManifestStatus: 'consistent',
    artifactInfrastructureFailureObserved: false,
    attemptLayoutExact: true,
    browserOrDeviceLossObserved: false,
    candidateReservationBindingStatus: 'matched',
    childLifecycleClosed: true,
    runVerificationStatus: 'passed',
    runDirectoryCount: 1,
    sourceIdentityStable: true,
    telemetryCollectorFailureObserved: false,
  };
}

function candidateEnvironmentDecision(pass) {
  return {
    schemaVersion: 2,
    kind: 'first-instance-live-candidate-environment-gate',
    applicable: true,
    status: pass ? 'passed' : 'failed-non-replaceable-process-set-mismatch',
    pass,
    retryable: false,
    nonReplaceable: !pass,
    collectorPass: true,
    processIdentityPass: pass,
    failureCodes: pass ? [] : ['compute-process-set-mismatch'],
    reasons: pass ? [] : ['pre-run and post-run compute-process identity sets differ'],
  };
}

function overallDecision(numericalPass, environmentPass) {
  const pass = numericalPass && environmentPass;
  return {
    schemaVersion: 1,
    kind: 'first-instance-live-overall-evidence-decision',
    applicable: true,
    status: pass ? 'passed' : 'failed-non-replaceable',
    pass,
    retryable: false,
    nonReplaceable: !pass,
    failedGates: environmentPass ? [] : ['candidateEnvironmentGate'],
    failureCodes: environmentPass ? [] : ['compute-process-set-mismatch'],
    reasons: environmentPass ? [] : ['pre-run and post-run compute-process identity sets differ'],
  };
}

function matrixDecision({ matrixOrdinal, attemptOrdinal, runId, environmentPass }) {
  const numericalDecision = {
    schemaVersion: 1,
    kind: 'fixture-preregistered-numerical-decision',
    pass: true,
    failureCodes: [],
    reasons: [],
  };
  const environmentDecision = candidateEnvironmentDecision(environmentPass);
  const evidenceDecision = overallDecision(true, environmentPass);
  return {
    matrixOrdinal,
    attemptOrdinal,
    runId,
    numericalPass: true,
    environmentPass,
    evidencePass: evidenceDecision.pass,
    pass: evidenceDecision.pass,
    numericalDecision,
    environmentDecision,
    overallEvidenceDecision: evidenceDecision,
  };
}

function publicIdentity(source) {
  const adapterAndDriver = {
    vendor: 'nvidia',
    architecture: 'public-adapter',
    description: 'Public fixture GPU',
  };
  const physicalGpuSet = [{
    gpuIndex: 0,
    gpuName: 'Public fixture GPU',
    gpuUuid: 'GPU-PUBLIC-DEVICE-0',
  }];
  const adapterTelemetryAssociation =
    evaluateLiveFirstInstanceAdapterTelemetryAssociation({
      adapterInfo: adapterAndDriver,
      telemetryReport: { summary: { gpus: physicalGpuSet } },
    });
  return {
    source,
    dependency: {
      packageLockSha256: source.packageLockSha256,
      threeRevision: '185',
      executionDependencyClosure: {
        schemaVersion: 1,
        kind: 'installed-execution-dependency-closure',
        root: 'node_modules',
        format: 'node-modules-sorted-path-size-content-sha256-v1',
        hashAlgorithm: 'sha256',
        exclusions: ['.bin/**', '.vite*/**'],
        fileCount: source.executionDependencyFileCount,
        totalBytes: source.executionDependencyTotalBytes,
        sha256: source.executionDependencyClosureSha256,
      },
    },
    viteRuntimeAudit: {
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
      entryHtml: { sourceRelativePath: 'index.html', sourceSha256: '5'.repeat(64) },
      prohibitedOptimizedArtifactCount: 0,
      requiredModulePaths: [
        'src/main.js',
        'src/strategies/live-first-instance-crossover.js',
      ],
      moduleCount: 42,
      dependencyModuleCount: 6,
      modulesSha256: '6'.repeat(64),
      modules: [{ sourceRelativePath: 'src/main.js', sourceSha256: '7'.repeat(64) }],
    },
    machine: { platform: 'win32', architecture: 'x64', node: 'v22.12.0' },
    browser: {
      runner: {
        executable: 'chrome.exe',
        version: '151.0.0.0',
        headless: true,
        args: ['--enable-unsafe-webgpu'],
        viewport: { width: 1280, height: 720 },
      },
      userAgent: 'public-fixture-browser/1',
    },
    pageProtocolEnvironment: {
      viewport: { width: 1280, height: 720 },
      reversedDepth: true,
      rendererReversedDepthBuffer: true,
      maxStorageBuffersPerShaderStage: 10,
      timestampAvailable: true,
      indirectFirstInstanceAvailable: true,
      crossOriginIsolated: true,
    },
    backend: {
      runnerBackend: 'webgpu',
      rendererBackend: 'WebGPUBackend',
      coordinateSystem: 'WebGPUCoordinateSystem',
    },
    adapterAndDriver,
    physicalGpuSet,
    adapterTelemetryAssociation,
    workload: {
      protocol: { matrixKind: 'first-instance-live' },
      workload: { objectCount: 65536, bucketCount: 32 },
    },
  };
}

async function temporaryRoot(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'public-live-pair-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

async function addValidAttempt({
  seriesDirectory,
  ledgerPath,
  source,
  runId,
  recordedAt,
}) {
  let state = await readCandidateLedger(ledgerPath);
  const reserved = await reserveCandidateAttempt(
    seriesDirectory,
    ledgerPath,
    state,
    source,
    new Date(recordedAt),
  );
  const runDirectory = path.join(reserved.absoluteAttemptDirectory, 'runs', runId);
  await mkdir(runDirectory);
  const privateManifestBytes = jsonBytes({
    runId,
    privatePath: PRIVATE_PATH_SENTINEL,
    privateGpuUuid: PRIVATE_GPU_SENTINEL,
    privateProcessName: 'candidate-private-app.exe',
    privatePid: 424242,
  });
  await writeFile(path.join(runDirectory, 'artifact-manifest.json'), privateManifestBytes);
  await writeFile(path.join(reserved.absoluteAttemptDirectory, 'runner-stdout.log'), PRIVATE_PATH_SENTINEL);
  await writeFile(path.join(reserved.absoluteAttemptDirectory, 'runner-stderr.log'), '');
  const commitment = await buildAttemptContentCommitment(reserved.absoluteAttemptDirectory);
  const event = await appendCandidateLedgerEvent(ledgerPath, {
    eventType: 'attempt-finalized',
    seriesId: state.seriesId,
    recordedAt: new Date(Date.parse(recordedAt) + 60 * 60_000).toISOString(),
    attemptOrdinal: reserved.event.attemptOrdinal,
    matrixOrdinal: reserved.event.matrixOrdinal,
    attemptDirectory: reserved.event.attemptDirectory,
    reservationEventSha256: reserved.event.eventSha256,
    classification: 'valid-candidate',
    reasonCode: 'completed-candidate',
    runDirectory: `${reserved.event.attemptDirectory}/runs/${runId}`,
    execution: { exitCode: 0, signal: null },
    contentCommitment: commitment,
    classificationEvidence: classificationEvidence(),
  });
  state = await readCandidateLedger(ledgerPath);
  return {
    reservation: reserved.event,
    finalization: event,
    privateManifestBytes,
    state,
  };
}

async function createPublicRun({
  root,
  source,
  seriesId,
  attempt,
  runId,
  startedAt,
  completedAt,
  environmentPass,
}) {
  const directory = path.join(root, `public-matrix-${attempt.reservation.matrixOrdinal}`);
  await mkdir(directory);
  const privateManifestSha256 = sha256(attempt.privateManifestBytes);
  const manifest = {
    runId,
    marker: 'public fixture manifest',
    bundleProvenance: {
      policyId: 'first-instance-live-public-evidence-v2',
      privateArtifactManifest: {
        name: 'artifact-manifest.json',
        sha256: privateManifestSha256,
      },
      candidateSource: { commit: source.commit, tree: source.tree },
    },
  };
  await writeFile(path.join(directory, 'artifact-manifest.json'), jsonBytes(manifest));
  const reservation = {
    schemaVersion: 1,
    kind: 'first-instance-live-candidate-series-reservation',
    seriesId,
    reservationEventSha256: attempt.reservation.eventSha256,
    attemptOrdinal: attempt.reservation.attemptOrdinal,
    matrixOrdinal: attempt.reservation.matrixOrdinal,
    sourceCommit: source.commit,
    sourceTree: source.tree,
    executionDependencyClosureSha256: source.executionDependencyClosureSha256,
  };
  const fixture = {
    metadata: {
      runId,
      startedAt,
      completedAt,
      candidateSeriesReservation: reservation,
    },
    reservation,
    identity: publicIdentity(source),
    matrixDecision: matrixDecision({
      matrixOrdinal: reservation.matrixOrdinal,
      attemptOrdinal: reservation.attemptOrdinal,
      runId,
      environmentPass,
    }),
  };
  await writeFile(path.join(directory, 'public-test-inspection.json'), jsonBytes(fixture));
  return directory;
}

async function inspectPublicRunFixture(directory) {
  const manifestBytes = await readFile(path.join(directory, 'artifact-manifest.json'));
  const manifest = JSON.parse(manifestBytes);
  const fixture = JSON.parse(await readFile(
    path.join(directory, 'public-test-inspection.json'),
    'utf8',
  ));
  return {
    resolvedDirectory: path.resolve(directory),
    manifest,
    manifestBytes,
    manifestSha256: sha256(manifestBytes),
    metadata: fixture.metadata,
    reservation: fixture.reservation,
    identity: fixture.identity,
    matrixDecision: fixture.matrixDecision,
    startedAt: fixture.metadata.startedAt,
    completedAt: fixture.metadata.completedAt,
    privateArtifactManifestSha256:
      manifest.bundleProvenance.privateArtifactManifest.sha256,
    publicBundlePolicyId: manifest.bundleProvenance.policyId,
    candidateCommit: manifest.bundleProvenance.candidateSource.commit,
    candidateTree: manifest.bundleProvenance.candidateSource.tree,
  };
}

async function captureDirectoryContents(directory) {
  const contents = new Map();
  for (const name of (await readdir(directory)).sort()) {
    contents.set(name, Buffer.from(await readFile(path.join(directory, name))));
  }
  return contents;
}

function anchorVerifierFixture(_repositoryRoot, registryVerification) {
  return Promise.resolve({ ...registryVerification.anchor, verified: true });
}

async function createFixture(t) {
  const root = await temporaryRoot(t);
  const candidateRoot = path.join(root, 'candidate-series');
  await mkdir(candidateRoot);
  await writeFile(
    path.join(candidateRoot, CANDIDATE_REGISTRY_LOCK_FILENAME),
    '{"pid":1}\n',
  );
  const source = sourceIdentity();
  const claimed = await claimCandidateStudy(
    candidateRoot,
    source,
    new Date('2026-09-01T00:00:00.000Z'),
  );
  const seriesDirectory = path.join(candidateRoot, claimed.event.seriesDirectory);
  const opened = await openCandidateSeries(
    seriesDirectory,
    source,
    new Date('2026-09-01T00:01:00.000Z'),
  );
  await materializeCandidateStudy(
    claimed.registryPath,
    claimed.event,
    opened.state,
    new Date('2026-09-01T00:02:00.000Z'),
  );
  await rm(path.join(candidateRoot, CANDIDATE_REGISTRY_LOCK_FILENAME));
  const first = await addValidAttempt({
    seriesDirectory,
    ledgerPath: opened.ledgerPath,
    source,
    runId: 'public-pair-matrix-1',
    recordedAt: '2026-09-01T01:00:00.000Z',
  });
  const second = await addValidAttempt({
    seriesDirectory,
    ledgerPath: opened.ledgerPath,
    source,
    runId: 'public-pair-matrix-2',
    recordedAt: '2026-09-01T02:00:00.000Z',
  });
  const ledger = await readCandidateLedger(opened.ledgerPath);
  const registry = await readCandidateRegistry(claimed.registryPath);
  const registryRecord = registry.byStudyKey.get(claimed.event.studyKey);
  const anchor = candidateRegistryAnchor(registryRecord);
  const firstPublic = await createPublicRun({
    root,
    source,
    seriesId: ledger.seriesId,
    attempt: first,
    runId: 'public-pair-matrix-1',
    startedAt: '2026-09-01T01:01:00.000Z',
    completedAt: '2026-09-01T01:40:00.000Z',
    environmentPass: true,
  });
  const secondPublic = await createPublicRun({
    root,
    source,
    seriesId: ledger.seriesId,
    attempt: second,
    runId: 'public-pair-matrix-2',
    startedAt: '2026-09-01T02:01:00.000Z',
    completedAt: '2026-09-01T02:40:00.000Z',
    environmentPass: false,
  });
  const deterministicPublicContents = new Map([
    ['public-pair-matrix-1', await captureDirectoryContents(firstPublic)],
    ['public-pair-matrix-2', await captureDirectoryContents(secondPublic)],
  ]);
  const publicRunDeriver = async (privateRunDirectory, outputDirectory) => {
    const runId = path.basename(privateRunDirectory);
    const contents = deterministicPublicContents.get(runId);
    assert.ok(contents, `missing deterministic public fixture for ${runId}`);
    await mkdir(outputDirectory);
    for (const [name, bytes] of contents) {
      await writeFile(path.join(outputDirectory, name), bytes, { flag: 'wx' });
    }
  };
  const attempts = ledger.attempts.map(({ finalization }) => ({
    attemptOrdinal: finalization.attemptOrdinal,
    matrixOrdinal: finalization.matrixOrdinal,
    attemptDirectory: finalization.attemptDirectory,
    classification: finalization.classification,
    reasonCode: finalization.reasonCode,
    classificationEvidence: finalization.classificationEvidence,
  }));
  const matrices = [
    (JSON.parse(await readFile(path.join(firstPublic, 'public-test-inspection.json')))).matrixDecision,
    (JSON.parse(await readFile(path.join(secondPublic, 'public-test-inspection.json')))).matrixDecision,
  ];
  const pairEligibility = {
    canonicalRegistryAndAnchorBound: true,
    exactlyTwoValidCompletedMatrices: true,
    secondMatrixPresentRegardlessOfFirst: true,
    sameFrozenIdentity: true,
    everyCandidateBoundToSeriesSource: true,
    noLaterSubstitutes: true,
    sourceSeriesFreeOfNonretryableFailure: true,
    pass: true,
  };
  const privatePair = {
    schemaVersion: 1,
    kind: 'first-instance-live-candidate-pair-verification',
    seriesId: ledger.seriesId,
    ledger: {
      filename: PUBLIC_LIVE_PAIR_LEDGER_NAME,
      eventCount: ledger.events.length,
      finalEventSha256: ledger.events.at(-1).eventSha256,
      sourceIdentity: source,
    },
    registry: {
      experimentId: claimed.event.experimentId,
      studyKey: claimed.event.studyKey,
      filename: 'candidate-series-registry.jsonl',
      eventCount: registry.events.length,
      finalEventSha256: registry.finalEventSha256,
      claimEventSha256: registryRecord.claim.eventSha256,
      materializedEventSha256: registryRecord.materialization.eventSha256,
      seriesOpeningEventSha256: registryRecord.materialization.seriesOpeningEventSha256,
      anchorTagName: anchor.tagName,
      anchorMessageSha256: anchor.messageSha256,
      anchorVerified: true,
    },
    attemptCount: attempts.length,
    attempts,
    retryCount: 0,
    completedMatrixCount: 2,
    validCandidateCount: 2,
    nonretryableFailureCount: 0,
    pairIdentitySha256: '4'.repeat(64),
    pairEligibility,
    matrices,
    decision: {
      status: 'confirmation-not-met',
      pass: false,
      rule: 'both matrices independently pass all numerical, environment, and evidence gates',
    },
  };
  return {
    root,
    seriesDirectory,
    publicRuns: [firstPublic, secondPublic],
    privatePair,
    publicRunDeriver,
  };
}

async function buildFixtureBundle(
  fixture,
  outputName = 'public-pair-bundle',
  overrides = {},
) {
  const output = path.join(fixture.root, outputName);
  const result = await buildPublicLiveCandidatePairBundle(
    fixture.seriesDirectory,
    fixture.publicRuns,
    output,
    {
      privatePairVerifier: async () => fixture.privatePair,
      inspectPublicRun: inspectPublicRunFixture,
      anchorTagVerifier: anchorVerifierFixture,
      publicRunDeriver: fixture.publicRunDeriver,
      ...overrides,
    },
  );
  return { output, result };
}

async function verifyFixtureBundle(fixture, output) {
  return verifyPublicLiveCandidatePairBundle(output, fixture.publicRuns, {
    inspectPublicRun: inspectPublicRunFixture,
    anchorTagVerifier: anchorVerifierFixture,
  });
}

async function updateBundleManifestEntry(bundleDirectory, name) {
  const manifestPath = path.join(bundleDirectory, PUBLIC_LIVE_PAIR_MANIFEST_NAME);
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const contents = await readFile(path.join(bundleDirectory, name));
  const entry = manifest.files.find((candidate) => candidate.name === name);
  entry.bytes = contents.length;
  entry.sha256 = sha256(contents);
  await writeFile(manifestPath, jsonBytes(manifest));
}

async function directoryBytes(directory) {
  const entries = (await readdir(directory)).sort();
  return Object.fromEntries(await Promise.all(entries.map(async (name) => [
    name,
    (await readFile(path.join(directory, name))).toString('base64'),
  ])));
}

test('public pair builder is deterministic, privacy-safe, and independently reproducible', async (t) => {
  const fixture = await createFixture(t);
  const first = await buildFixtureBundle(fixture, 'pair-a');
  const second = await buildFixtureBundle(fixture, 'pair-b');
  assert.deepEqual(await directoryBytes(first.output), await directoryBytes(second.output));
  assert.deepEqual((await readdir(first.output)).sort(), [
    PUBLIC_LIVE_PAIR_MANIFEST_NAME,
    ...PUBLIC_LIVE_PAIR_BUNDLE_FILES.map(({ name }) => name),
  ].sort());

  const verification = await verifyFixtureBundle(fixture, first.output);
  assert.equal(verification.status, 'consistent');
  assert.equal(verification.selectionLedgerReplayed, true);
  assert.equal(verification.matrixDecisionsRecomputed, true);
  assert.equal(verification.pairDecisionRecomputed, true);
  assert.equal(verification.decision.pass, false);
  assert.equal(verification.authenticityVerified, false);
  assert.equal(verification.privateOriginalManifestBytesVerified, false);
  assert.equal(verification.privateAttemptContentCommitmentsVerified, false);
  assert.equal(verification.deterministicSanitizerRederivationVerified, false);
  assert.equal(first.result.deterministicSanitizerRederivationVerified, true);

  const allBundleBytes = (await Promise.all((await readdir(first.output)).map(
    (name) => readFile(path.join(first.output, name), 'utf8'),
  ))).join('\n');
  const cliOutput = serializePublicLiveCandidatePairBuildResult(first.result);
  for (const privateValue of [
    fixture.root,
    PRIVATE_PATH_SENTINEL,
    PRIVATE_GPU_SENTINEL,
    'candidate-private-app.exe',
    '424242',
  ]) {
    assert.equal(allBundleBytes.includes(privateValue), false, privateValue);
    assert.equal(cliOutput.includes(privateValue), false, privateValue);
  }
  assert.doesNotMatch(cliOutput, /publicOutputDirectory|privateSeriesDirectory|[A-Za-z]:\\/);
});

test('public pair builder rejects a decision-neutral public rewrite after manifest recomputation', async (t) => {
  const fixture = await createFixture(t);
  const publicDirectory = fixture.publicRuns[0];
  const inspectionPath = path.join(publicDirectory, 'public-test-inspection.json');
  const inspection = JSON.parse(await readFile(inspectionPath, 'utf8'));
  inspection.metadata.decisionNeutralPublicationNote = 'rewritten after sanitization';
  const inspectionBytes = jsonBytes(inspection);
  await writeFile(inspectionPath, inspectionBytes);
  const manifestPath = path.join(publicDirectory, 'artifact-manifest.json');
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  manifest.recomputedInspectionSha256 = sha256(inspectionBytes);
  await writeFile(manifestPath, jsonBytes(manifest));
  assert.match(
    manifest.bundleProvenance.privateArtifactManifest.sha256,
    /^[0-9a-f]{64}$/,
  );
  await assert.rejects(
    buildFixtureBundle(fixture, 'rewritten-public-pair'),
    /not the deterministic sanitizer output/,
  );
});

test('public pair builder detects a public input changed after staged verification', async (t) => {
  const fixture = await createFixture(t);
  await assert.rejects(
    buildFixtureBundle(fixture, 'raced-public-pair', {
      publicBundleVerifier: async (bundleDirectory, publicRunDirectories) => {
        const result = await verifyPublicLiveCandidatePairBundle(
          bundleDirectory,
          publicRunDirectories,
          {
            inspectPublicRun: inspectPublicRunFixture,
            anchorTagVerifier: anchorVerifierFixture,
          },
        );
        await writeFile(
          path.join(publicRunDirectories[0], 'post-verification-race.txt'),
          'changed',
        );
        return result;
      },
    }),
    /public run input 1 changed during pair derivation/,
  );
});

test('public pair verifier enforces reservation-to-finalization run chronology', async (t) => {
  const fixture = await createFixture(t);
  const { output } = await buildFixtureBundle(fixture);
  const inspectionPath = path.join(
    fixture.publicRuns[0],
    'public-test-inspection.json',
  );
  const inspection = JSON.parse(await readFile(inspectionPath, 'utf8'));
  inspection.metadata.startedAt = '2026-09-01T00:59:59.999Z';
  await writeFile(inspectionPath, jsonBytes(inspection));
  await assert.rejects(
    verifyFixtureBundle(fixture, output),
    /reservation <= start <= completion <= finalization chronology/,
  );
});

test('public pair verifier rejects a different disclosed Vite runtime graph', async (t) => {
  const fixture = await createFixture(t);
  const { output } = await buildFixtureBundle(fixture);
  const inspectionPath = path.join(
    fixture.publicRuns[1],
    'public-test-inspection.json',
  );
  const inspection = JSON.parse(await readFile(inspectionPath, 'utf8'));
  inspection.identity.viteRuntimeAudit.modulesSha256 = 'f'.repeat(64);
  inspection.identity.viteRuntimeAudit.modules[0].sourceSha256 = 'f'.repeat(64);
  await writeFile(inspectionPath, jsonBytes(inspection));
  await assert.rejects(
    verifyFixtureBundle(fixture, output),
    /two public candidate runs differ in frozen identity/,
  );
});

test('public pair verifier reconstructs the disclosed adapter-to-telemetry join', async (t) => {
  const fixture = await createFixture(t);
  const { output } = await buildFixtureBundle(fixture);
  const inspectionPath = path.join(
    fixture.publicRuns[1],
    'public-test-inspection.json',
  );
  const inspection = JSON.parse(await readFile(inspectionPath, 'utf8'));
  inspection.identity.adapterAndDriver.description = 'Substituted public GPU';
  await writeFile(inspectionPath, jsonBytes(inspection));
  await assert.rejects(
    verifyFixtureBundle(fixture, output),
    /adapter-to-telemetry association is not reconstructable from its disclosed identity/,
  );
});

test('public pair verifier rejects a disclosed multi-GPU identity', async (t) => {
  const fixture = await createFixture(t);
  const { output } = await buildFixtureBundle(fixture);
  const inspectionPath = path.join(
    fixture.publicRuns[1],
    'public-test-inspection.json',
  );
  const inspection = JSON.parse(await readFile(inspectionPath, 'utf8'));
  inspection.identity.physicalGpuSet.push({
    gpuIndex: 1,
    gpuName: 'Public fixture GPU',
    gpuUuid: 'GPU-PUBLIC-DEVICE-1',
  });
  await writeFile(inspectionPath, jsonBytes(inspection));
  await assert.rejects(
    verifyFixtureBundle(fixture, output),
    /adapter-to-telemetry association is not reconstructable from its disclosed identity/,
  );
});

test('public pair verifier detects a first run changed while inspecting the second', async (t) => {
  const fixture = await createFixture(t);
  const { output } = await buildFixtureBundle(fixture);
  let inspectionCount = 0;
  await assert.rejects(
    verifyPublicLiveCandidatePairBundle(output, fixture.publicRuns, {
      inspectPublicRun: async (directory) => {
        const inspected = await inspectPublicRunFixture(directory);
        inspectionCount += 1;
        if (inspectionCount === 2) {
          await writeFile(
            path.join(fixture.publicRuns[0], 'concurrent-change.txt'),
            'changed',
          );
        }
        return inspected;
      },
      anchorTagVerifier: anchorVerifierFixture,
    }),
    /public run input changed during pair verification/,
  );
});

test('public pair verifier rejects byte tampering and a recomputed manifest cannot hide receipt drift', async (t) => {
  const fixture = await createFixture(t);
  const { output } = await buildFixtureBundle(fixture);
  const receiptPath = path.join(output, PUBLIC_LIVE_PAIR_RECEIPT_NAME);
  const receipt = JSON.parse(await readFile(receiptPath, 'utf8'));
  receipt.publicRuns[0].runId = 'substituted-run';
  await writeFile(receiptPath, jsonBytes(receipt));
  await assert.rejects(
    verifyFixtureBundle(fixture, output),
    /differs from the public pair bundle manifest/,
  );
  await updateBundleManifestEntry(output, PUBLIC_LIVE_PAIR_RECEIPT_NAME);
  await assert.rejects(
    verifyFixtureBundle(fixture, output),
    /manifest bindings differ|runId differs/,
  );
});

test('public pair verifier rejects rewritten ledger bytes even after manifest and receipt rebinding', async (t) => {
  const fixture = await createFixture(t);
  const { output } = await buildFixtureBundle(fixture);
  const ledgerPath = path.join(output, PUBLIC_LIVE_PAIR_LEDGER_NAME);
  const lines = (await readFile(ledgerPath, 'utf8')).trimEnd().split('\n');
  const finalEvent = JSON.parse(lines.at(-1));
  finalEvent.classificationEvidence.runDirectoryCount = 2;
  lines[lines.length - 1] = JSON.stringify(finalEvent);
  const changedLedger = Buffer.from(`${lines.join('\n')}\n`);
  await writeFile(ledgerPath, changedLedger);
  const receiptPath = path.join(output, PUBLIC_LIVE_PAIR_RECEIPT_NAME);
  const receipt = JSON.parse(await readFile(receiptPath, 'utf8'));
  receipt.ledgerBinding.bytes = changedLedger.length;
  receipt.ledgerBinding.sha256 = sha256(changedLedger);
  await writeFile(receiptPath, jsonBytes(receipt));
  await updateBundleManifestEntry(output, PUBLIC_LIVE_PAIR_LEDGER_NAME);
  await updateBundleManifestEntry(output, PUBLIC_LIVE_PAIR_RECEIPT_NAME);
  await assert.rejects(
    verifyFixtureBundle(fixture, output),
    /event .* digest differs from its bytes/,
  );
});

test('public pair verifier rejects rewritten root-registry bytes and an unverified anchor tag', async (t) => {
  await t.test('rewritten registry chain', async (t2) => {
    const fixture = await createFixture(t2);
    const { output } = await buildFixtureBundle(fixture);
    const registryPath = path.join(output, PUBLIC_LIVE_PAIR_REGISTRY_NAME);
    const lines = (await readFile(registryPath, 'utf8')).trimEnd().split('\n');
    const claim = JSON.parse(lines[0]);
    claim.seriesDirectory = 'substituted-series';
    lines[0] = JSON.stringify(claim);
    const changedRegistry = Buffer.from(`${lines.join('\n')}\n`);
    await writeFile(registryPath, changedRegistry);
    const receiptPath = path.join(output, PUBLIC_LIVE_PAIR_RECEIPT_NAME);
    const receipt = JSON.parse(await readFile(receiptPath, 'utf8'));
    receipt.seriesRootRegistryBinding.bytes = changedRegistry.length;
    receipt.seriesRootRegistryBinding.sha256 = sha256(changedRegistry);
    await writeFile(receiptPath, jsonBytes(receipt));
    await updateBundleManifestEntry(output, PUBLIC_LIVE_PAIR_REGISTRY_NAME);
    await updateBundleManifestEntry(output, PUBLIC_LIVE_PAIR_RECEIPT_NAME);
    await assert.rejects(
      verifyFixtureBundle(fixture, output),
      /registry event .* digest differs from its bytes/,
    );
  });
  await t.test('cross-file registry and ledger chronology', async (t2) => {
    const fixture = await createFixture(t2);
    const { output } = await buildFixtureBundle(fixture);
    const registryPath = path.join(output, PUBLIC_LIVE_PAIR_REGISTRY_NAME);
    const events = (await readFile(registryPath, 'utf8')).trimEnd().split('\n').map(JSON.parse);
    events[0].recordedAt = '2026-09-01T00:01:30.000Z';
    delete events[0].eventSha256;
    events[0].eventSha256 = sha256Canonical(events[0]);
    events[1].previousEventSha256 = events[0].eventSha256;
    events[1].claimEventSha256 = events[0].eventSha256;
    delete events[1].eventSha256;
    events[1].eventSha256 = sha256Canonical(events[1]);
    await writeFile(registryPath, `${events.map(JSON.stringify).join('\n')}\n`);
    await updateBundleManifestEntry(output, PUBLIC_LIVE_PAIR_REGISTRY_NAME);
    await assert.rejects(
      verifyFixtureBundle(fixture, output),
      /claim, ledger opening, and materialization chronology is invalid/,
    );
  });
  await t.test('materialization after first reservation', async (t2) => {
    const fixture = await createFixture(t2);
    const { output } = await buildFixtureBundle(fixture);
    const registryPath = path.join(output, PUBLIC_LIVE_PAIR_REGISTRY_NAME);
    const events = (await readFile(registryPath, 'utf8')).trimEnd().split('\n').map(JSON.parse);
    events[1].recordedAt = '2026-09-01T01:00:00.001Z';
    delete events[1].eventSha256;
    events[1].eventSha256 = sha256Canonical(events[1]);
    await writeFile(registryPath, `${events.map(JSON.stringify).join('\n')}\n`);
    await updateBundleManifestEntry(output, PUBLIC_LIVE_PAIR_REGISTRY_NAME);
    await assert.rejects(
      verifyFixtureBundle(fixture, output),
      /registry materialization occurs after the first ledger reservation/,
    );
  });
  await t.test('missing or mismatched tag anchor', async (t2) => {
    const fixture = await createFixture(t2);
    const { output } = await buildFixtureBundle(fixture);
    await assert.rejects(
      verifyPublicLiveCandidatePairBundle(output, fixture.publicRuns, {
        inspectPublicRun: inspectPublicRunFixture,
        anchorTagVerifier: async (_root, registryVerification) => ({
          ...registryVerification.anchor,
          verified: false,
        }),
      }),
      /anchor tag was not independently verified/,
    );
  });
});

test('public pair verifier rejects extra entries and public-manifest substitution', async (t) => {
  await t.test('extra receipt-bundle entry', async (t2) => {
    const fixture = await createFixture(t2);
    const { output } = await buildFixtureBundle(fixture);
    await writeFile(path.join(output, 'undeclared.txt'), 'undeclared');
    await assert.rejects(
      verifyFixtureBundle(fixture, output),
      /undeclared or missing entry/,
    );
  });
  await t.test('changed supplied public manifest', async (t2) => {
    const fixture = await createFixture(t2);
    const { output } = await buildFixtureBundle(fixture);
    const manifestPath = path.join(fixture.publicRuns[0], 'artifact-manifest.json');
    const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
    manifest.marker = 'substituted public manifest';
    await writeFile(manifestPath, jsonBytes(manifest));
    await assert.rejects(
      verifyFixtureBundle(fixture, output),
      /public-run manifest bindings differ/,
    );
  });
});

test('public pair verifier rejects unexpected receipt schema even with a recomputed bundle digest', async (t) => {
  const fixture = await createFixture(t);
  const { output } = await buildFixtureBundle(fixture);
  const receiptPath = path.join(output, PUBLIC_LIVE_PAIR_RECEIPT_NAME);
  const receipt = JSON.parse(await readFile(receiptPath, 'utf8'));
  receipt.localDebugPath = PRIVATE_PATH_SENTINEL;
  await writeFile(receiptPath, jsonBytes(receipt));
  await updateBundleManifestEntry(output, PUBLIC_LIVE_PAIR_RECEIPT_NAME);
  await assert.rejects(
    verifyFixtureBundle(fixture, output),
    /receipt has an unexpected schema/,
  );
});

test('public pair verifier rejects a symlinked declared entry when supported', async (t) => {
  const fixture = await createFixture(t);
  const { output } = await buildFixtureBundle(fixture);
  const receiptPath = path.join(output, PUBLIC_LIVE_PAIR_RECEIPT_NAME);
  const targetPath = path.join(fixture.root, 'receipt-target.json');
  await writeFile(targetPath, await readFile(receiptPath));
  await rm(receiptPath);
  try {
    await symlink(targetPath, receiptPath, 'file');
  } catch (error) {
    if (['EPERM', 'EACCES', 'ENOTSUP'].includes(error?.code)) {
      t.skip(`symlinks unavailable: ${error.code}`);
      return;
    }
    throw error;
  }
  await assert.rejects(
    verifyFixtureBundle(fixture, output),
    /not a regular file/,
  );
});
