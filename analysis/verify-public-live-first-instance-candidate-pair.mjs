import { createHash } from 'node:crypto';
import {
  lstat,
  readFile,
  readdir,
} from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  summarizeCsv,
  verifyRunDirectory,
} from './summarize.mjs';
import {
  canonicalJson,
  parseCandidateLedgerText,
  sourceIdentityFromProvenance,
} from '../scripts/live-first-instance-candidate-ledger.mjs';
import {
  candidateRegistryAnchor,
  parseCandidateRegistryText,
  verifyCandidateRegistryAnchorTag,
} from '../scripts/live-first-instance-candidate-registry.mjs';
import {
  LIVE_PUBLIC_BUNDLE_LABEL,
  LIVE_PUBLIC_EVIDENCE_POLICY_ID,
} from '../scripts/live-evidence-sanitizer-policy.mjs';
import {
  evaluateLiveFirstInstanceAdapterTelemetryAssociation,
} from '../scripts/live-first-instance-adapter-telemetry-association.mjs';
import {
  PUBLIC_LIVE_PAIR_BUNDLE_FILES,
  PUBLIC_LIVE_PAIR_BUNDLE_LABEL,
  PUBLIC_LIVE_PAIR_INTEGRITY_SCOPE,
  PUBLIC_LIVE_PAIR_LEDGER_NAME,
  PUBLIC_LIVE_PAIR_MANIFEST_NAME,
  PUBLIC_LIVE_PAIR_POLICY_ID,
  PUBLIC_LIVE_PAIR_POLICY_SCHEMA_VERSION,
  PUBLIC_LIVE_PAIR_PRIVATE_ATTEMPT_SCOPE,
  PUBLIC_LIVE_PAIR_REGISTRY_NAME,
  PUBLIC_LIVE_PAIR_RECEIPT_NAME,
  projectPrivatePairVerification,
  validatePublicLivePairBundleManifest,
  validatePublicLivePairReceipt,
} from '../scripts/public-live-first-instance-pair-policy.mjs';

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SHA256_PATTERN = /^[0-9a-f]{64}$/;

function reject(message) {
  throw new Error(`Public live candidate-pair verification rejected: ${message}`);
}

function sha256Bytes(value) {
  return createHash('sha256').update(value).digest('hex');
}

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function exactKeys(value, expected, label) {
  if (!isRecord(value)) reject(`${label} must be an object.`);
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (JSON.stringify(actual) !== JSON.stringify(wanted)) {
    reject(`${label} has an unexpected schema.`);
  }
  return value;
}

function nonemptyString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    reject(`${label} must be a non-empty string.`);
  }
  return value;
}

function array(value, label) {
  if (!Array.isArray(value)) reject(`${label} must be an array.`);
  return value;
}

function record(value, label) {
  if (!isRecord(value)) reject(`${label} must be an object.`);
  return value;
}

function sameCanonical(left, right) {
  return canonicalJson(left) === canonicalJson(right);
}

function parseJson(contents, label) {
  try {
    return JSON.parse(contents.toString('utf8'));
  } catch {
    reject(`${label} is not valid JSON.`);
  }
}

async function readRegularFile(filename, label) {
  let stats;
  try {
    stats = await lstat(filename);
  } catch (error) {
    if (error?.code === 'ENOENT') reject(`${label} is missing.`);
    throw error;
  }
  if (!stats.isFile() || stats.isSymbolicLink()) {
    reject(`${label} must be a non-symbolic-link regular file.`);
  }
  return readFile(filename);
}

async function snapshotRegularDirectory(directory, label) {
  const entries = await readdir(directory, { withFileTypes: true });
  const snapshot = [];
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const filename = path.join(directory, entry.name);
    const stats = await lstat(filename);
    if (!entry.isFile() || entry.isSymbolicLink() || !stats.isFile()) {
      reject(`${label} contains a non-regular entry.`);
    }
    const bytes = await readFile(filename);
    snapshot.push({
      name: entry.name,
      bytes: bytes.length,
      sha256: sha256Bytes(bytes),
    });
  }
  return snapshot;
}

function summaryFromVerifiedRun(verified) {
  return {
    ...summarizeCsv(verified.csvText),
    bundleIntegrity: verified.bundleIntegrity,
    liveFirstInstanceEvidenceDecision: verified.liveFirstInstanceEvidenceDecision,
    artifactVerification: verified.artifactVerification,
  };
}

function publicCandidateIdentity(metadata) {
  const provenance = record(metadata?.sourceProvenance?.start, 'candidate source provenance');
  const dependencyClosure = record(
    metadata?.executionDependencyClosure?.start,
    'candidate execution dependency closure',
  );
  const viteRuntimeAudit = record(
    metadata?.candidateViteRuntimeAudit,
    'candidate Vite runtime audit',
  );
  const browser = record(metadata?.environment?.browser, 'candidate browser identity');
  const page = record(metadata?.environment?.benchmarkPage, 'candidate page environment');
  const adapterInfo = record(page.adapterInfo, 'candidate adapter identity');
  const telemetryGpus = array(
    metadata?.environment?.gpuTelemetry?.summary?.gpus,
    'candidate telemetry GPU identities',
  ).map((gpu, index) => {
    const identity = record(gpu, `candidate telemetry GPU ${index}`);
    return {
      gpuIndex: identity.gpuIndex,
      gpuName: identity.gpuName,
      gpuUuid: identity.gpuUuid,
    };
  }).sort((left, right) => String(left.gpuUuid).localeCompare(String(right.gpuUuid)));
  if (telemetryGpus.length !== 1
    || telemetryGpus.some((gpu) => (
      !Number.isSafeInteger(gpu.gpuIndex)
      || typeof gpu.gpuName !== 'string'
      || gpu.gpuName === ''
      || typeof gpu.gpuUuid !== 'string'
      || gpu.gpuUuid === ''
    ))) {
    reject('public candidate telemetry must identify exactly one concrete pseudonymized GPU.');
  }
  const adapterTelemetryAssociation =
    evaluateLiveFirstInstanceAdapterTelemetryAssociation({
      adapterInfo,
      telemetryReport: metadata?.environment?.gpuTelemetry,
    });
  const retainedAdapterTelemetryAssociation = record(
    metadata?.liveFirstInstanceEnvironmentAudit?.adapterTelemetryAssociation,
    'public candidate adapter-to-telemetry association',
  );
  if (adapterTelemetryAssociation.pass !== true
    || !sameCanonical(
      retainedAdapterTelemetryAssociation,
      adapterTelemetryAssociation,
    )) {
    reject(
      'public candidate adapter-to-telemetry association differs from the page adapter and disclosed telemetry GPU evidence.',
    );
  }
  return {
    source: sourceIdentityFromProvenance(provenance, dependencyClosure),
    dependency: {
      packageLockSha256: provenance.packageLockSha256,
      threeRevision: page.threeRevision,
      executionDependencyClosure: dependencyClosure,
    },
    viteRuntimeAudit,
    machine: {
      platform: metadata.environment.platform,
      architecture: metadata.environment.architecture,
      node: metadata.environment.node,
    },
    browser: {
      runner: browser,
      userAgent: page.userAgent,
    },
    pageProtocolEnvironment: {
      viewport: page.viewport,
      reversedDepth: page.reversedDepth,
      rendererReversedDepthBuffer: page.rendererReversedDepthBuffer,
      maxStorageBuffersPerShaderStage: page.maxStorageBuffersPerShaderStage,
      timestampAvailable: page.timestampAvailable,
      indirectFirstInstanceAvailable: page.indirectFirstInstanceAvailable,
      crossOriginIsolated: page.crossOriginIsolated,
    },
    backend: {
      runnerBackend: metadata.environment.backend,
      rendererBackend: page.rendererBackend,
      coordinateSystem: page.coordinateSystem,
    },
    adapterAndDriver: adapterInfo,
    physicalGpuSet: telemetryGpus,
    adapterTelemetryAssociation,
    workload: {
      protocol: metadata.protocol,
      workload: metadata.workload,
    },
  };
}

function requirePublicIdentityAdapterTelemetryAssociation(identity, label) {
  const adapterInfo = record(identity?.adapterAndDriver, `${label} adapter identity`);
  const telemetryGpus = array(identity?.physicalGpuSet, `${label} telemetry GPU set`);
  const retained = record(
    identity?.adapterTelemetryAssociation,
    `${label} adapter-to-telemetry association`,
  );
  const reconstructed = evaluateLiveFirstInstanceAdapterTelemetryAssociation({
    adapterInfo,
    telemetryReport: {
      summary: { gpus: telemetryGpus },
      coverageAudit: { gpuIdentities: telemetryGpus },
    },
  });
  if (reconstructed.pass !== true || !sameCanonical(retained, reconstructed)) {
    reject(
      `${label} adapter-to-telemetry association is not reconstructable from its disclosed identity.`,
    );
  }
}

function matrixDecisionFromPublicRun(metadata, summary) {
  const numerical = record(
    summary.preregisteredNumericalDecision,
    'public run preregistered numerical decision',
  );
  const evidence = exactKeys(
    summary.liveFirstInstanceEvidenceDecision,
    [
      'adapterTelemetryAssociation',
      'candidateEnvironmentGate',
      'overallEvidenceDecision',
    ],
    'public run live evidence decision',
  );
  const environment = record(
    evidence.candidateEnvironmentGate,
    'public run candidate environment decision',
  );
  const overall = record(
    evidence.overallEvidenceDecision,
    'public run overall evidence decision',
  );
  if (typeof numerical.pass !== 'boolean'
    || typeof environment.pass !== 'boolean'
    || typeof overall.pass !== 'boolean') {
    reject('public run decisions must expose boolean pass fields.');
  }
  if (!sameCanonical(
    metadata.liveFirstInstanceEnvironmentAudit?.adapterTelemetryAssociation,
    evidence.adapterTelemetryAssociation,
  ) || !sameCanonical(
    metadata.liveFirstInstanceEnvironmentAudit?.candidateEnvironmentGate,
    environment,
  ) || !sameCanonical(
    metadata.liveFirstInstanceEnvironmentAudit?.overallEvidenceDecision,
    overall,
  )) {
    reject('public run analyzer decisions differ from metadata.');
  }
  return {
    matrixOrdinal: metadata.candidateSeriesReservation.matrixOrdinal,
    attemptOrdinal: metadata.candidateSeriesReservation.attemptOrdinal,
    runId: metadata.runId,
    numericalPass: numerical.pass,
    environmentPass: environment.pass,
    evidencePass: overall.pass,
    pass: overall.pass,
    numericalDecision: numerical,
    environmentDecision: environment,
    overallEvidenceDecision: overall,
  };
}

export async function inspectPublicLiveCandidateRun(
  runDirectory,
  { repositoryRoot = PROJECT_ROOT } = {},
) {
  const resolvedDirectory = path.resolve(runDirectory);
  const directoryStats = await lstat(resolvedDirectory);
  if (!directoryStats.isDirectory() || directoryStats.isSymbolicLink()) {
    reject('a public run input is not a non-symbolic-link directory.');
  }
  const inputSnapshot = await snapshotRegularDirectory(
    resolvedDirectory,
    'public run input',
  );
  const verified = await verifyRunDirectory(resolvedDirectory, { repositoryRoot });
  if (verified?.artifactVerification?.status !== 'consistent'
    || verified.artifactVerification.evidenceStatus !== 'candidate'
    || verified?.bundleIntegrity?.bundleLabel !== LIVE_PUBLIC_BUNDLE_LABEL
    || verified.bundleIntegrity.policyId !== LIVE_PUBLIC_EVIDENCE_POLICY_ID) {
    reject('a public run was not independently accepted as public-derived candidate evidence.');
  }
  const manifestBytes = await readRegularFile(
    path.join(resolvedDirectory, 'artifact-manifest.json'),
    'public artifact-manifest.json',
  );
  const metadataBytes = await readRegularFile(
    path.join(resolvedDirectory, 'metadata.json'),
    'public metadata.json',
  );
  const manifest = parseJson(manifestBytes, 'public artifact-manifest.json');
  const metadata = parseJson(metadataBytes, 'public metadata.json');
  const metadataDeclaration = Array.isArray(manifest.files)
    ? manifest.files.find(({ name }) => name === 'metadata.json')
    : undefined;
  if (metadataDeclaration === undefined
    || metadataDeclaration.bytes !== metadataBytes.length
    || metadataDeclaration.sha256 !== sha256Bytes(metadataBytes)) {
    reject('public metadata bytes differ from the verified public manifest.');
  }
  const summary = summaryFromVerifiedRun(verified);
  const reservation = exactKeys(metadata.candidateSeriesReservation, [
    'schemaVersion',
    'kind',
    'seriesId',
    'reservationEventSha256',
    'attemptOrdinal',
    'matrixOrdinal',
    'sourceCommit',
    'sourceTree',
    'executionDependencyClosureSha256',
  ], 'public metadata candidateSeriesReservation');
  if (reservation.schemaVersion !== 1
    || reservation.kind !== 'first-instance-live-candidate-series-reservation') {
    reject('public metadata candidateSeriesReservation has an unsupported identity.');
  }
  const manifestSha256 = sha256Bytes(manifestBytes);
  if (verified.bundleIntegrity.artifactManifestSha256 !== manifestSha256) {
    reject('strict public verifier manifest digest differs from the public manifest bytes.');
  }
  const finalSnapshot = await snapshotRegularDirectory(
    resolvedDirectory,
    'public run input',
  );
  if (!sameCanonical(inputSnapshot, finalSnapshot)) {
    reject('a public run input changed during strict verification and inspection.');
  }
  return {
    resolvedDirectory,
    manifest,
    manifestBytes,
    manifestSha256,
    metadata,
    reservation,
    identity: publicCandidateIdentity(metadata),
    matrixDecision: matrixDecisionFromPublicRun(metadata, summary),
    startedAt: nonemptyString(metadata.startedAt, 'public candidate startedAt'),
    completedAt: nonemptyString(metadata.completedAt, 'public candidate completedAt'),
    privateArtifactManifestSha256:
      manifest.bundleProvenance.privateArtifactManifest.sha256,
    publicBundlePolicyId: manifest.bundleProvenance.policyId,
    candidateCommit: manifest.bundleProvenance.candidateSource.commit,
    candidateTree: manifest.bundleProvenance.candidateSource.tree,
  };
}

async function readPublicPairBundle(bundleDirectory) {
  const resolvedDirectory = path.resolve(bundleDirectory);
  const stats = await lstat(resolvedDirectory);
  if (!stats.isDirectory() || stats.isSymbolicLink()) {
    reject('public pair bundle input is not a non-symbolic-link directory.');
  }
  const expectedNames = [
    PUBLIC_LIVE_PAIR_MANIFEST_NAME,
    ...PUBLIC_LIVE_PAIR_BUNDLE_FILES.map(({ name }) => name),
  ].sort();
  const entries = await readdir(resolvedDirectory, { withFileTypes: true });
  const actualNames = entries.map(({ name }) => name).sort();
  if (JSON.stringify(actualNames) !== JSON.stringify(expectedNames)) {
    reject('public pair bundle contains an undeclared or missing entry.');
  }
  for (const entry of entries) {
    const entryStats = await lstat(path.join(resolvedDirectory, entry.name));
    if (!entry.isFile() || entry.isSymbolicLink() || !entryStats.isFile()) {
      reject(`public pair bundle entry ${JSON.stringify(entry.name)} is not a regular file.`);
    }
  }
  const manifestBytes = await readRegularFile(
    path.join(resolvedDirectory, PUBLIC_LIVE_PAIR_MANIFEST_NAME),
    PUBLIC_LIVE_PAIR_MANIFEST_NAME,
  );
  const manifest = validatePublicLivePairBundleManifest(parseJson(
    manifestBytes,
    PUBLIC_LIVE_PAIR_MANIFEST_NAME,
  ));
  const contents = new Map();
  for (const declaration of manifest.files) {
    const bytes = await readRegularFile(
      path.join(resolvedDirectory, declaration.name),
      declaration.name,
    );
    if (bytes.length !== declaration.bytes
      || sha256Bytes(bytes) !== declaration.sha256) {
      reject(`${declaration.name} differs from the public pair bundle manifest.`);
    }
    contents.set(declaration.name, bytes);
  }
  const ledgerBytes = contents.get(PUBLIC_LIVE_PAIR_LEDGER_NAME);
  const registryBytes = contents.get(PUBLIC_LIVE_PAIR_REGISTRY_NAME);
  const receiptBytes = contents.get(PUBLIC_LIVE_PAIR_RECEIPT_NAME);
  const receipt = validatePublicLivePairReceipt(parseJson(
    receiptBytes,
    PUBLIC_LIVE_PAIR_RECEIPT_NAME,
  ));
  return {
    resolvedDirectory,
    manifest,
    manifestBytes,
    ledgerBytes,
    registryBytes,
    receipt,
    receiptBytes,
  };
}

function attemptProjectionFromLedger(state) {
  return state.attempts.map(({ finalization }) => ({
    attemptOrdinal: finalization.attemptOrdinal,
    matrixOrdinal: finalization.matrixOrdinal,
    attemptDirectory: finalization.attemptDirectory,
    classification: finalization.classification,
    reasonCode: finalization.reasonCode,
    classificationEvidence: finalization.classificationEvidence,
  }));
}

function ensureChronologicalDistinctSessions(publicRuns) {
  if (publicRuns[0].metadata.runId === publicRuns[1].metadata.runId) {
    reject('the public candidate matrices reuse a runId.');
  }
  const firstStart = Date.parse(publicRuns[0].startedAt);
  const firstEnd = Date.parse(publicRuns[0].completedAt);
  const secondStart = Date.parse(publicRuns[1].startedAt);
  const secondEnd = Date.parse(publicRuns[1].completedAt);
  if (![firstStart, firstEnd, secondStart, secondEnd].every(Number.isFinite)
    || firstEnd < firstStart
    || secondEnd < secondStart
    || secondStart < firstEnd) {
    reject('public candidate metadata does not prove chronological, non-overlapping sessions.');
  }
}

function expectedPairEligibility(state, publicRuns, sameFrozenIdentity) {
  const validAttempts = state.attempts.filter(
    ({ finalization }) => finalization.classification === 'valid-candidate',
  );
  const nonretryableFailures = state.attempts.filter(
    ({ finalization }) => (
      finalization.classification === 'implementation/evidence-failure-nonretryable'
    ),
  );
  const everyCandidateBoundToSeriesSource = publicRuns.every(
    ({ identity }) => sameCanonical(identity.source, state.sourceIdentity),
  );
  const pairEligibility = {
    canonicalRegistryAndAnchorBound: true,
    exactlyTwoValidCompletedMatrices: validAttempts.length === 2
      && publicRuns.length === 2
      && sameFrozenIdentity,
    secondMatrixPresentRegardlessOfFirst: validAttempts.some(
      ({ finalization }) => finalization.matrixOrdinal === 2,
    ),
    sameFrozenIdentity,
    everyCandidateBoundToSeriesSource,
    noLaterSubstitutes: state.completedMatrixCount === validAttempts.length
      && state.completedMatrixCount <= 2
      && state.attempts.every(({ finalization }) => finalization.matrixOrdinal <= 2),
    sourceSeriesFreeOfNonretryableFailure: nonretryableFailures.length === 0,
  };
  pairEligibility.pass = Object.values(pairEligibility).every((value) => value === true);
  return pairEligibility;
}

function bindPublicRunToLedger(publicRun, state) {
  const reservation = publicRun.reservation;
  if (reservation.seriesId !== state.seriesId) {
    reject(`matrix ${reservation.matrixOrdinal} seriesId differs from the disclosed ledger.`);
  }
  const attempt = state.attempts.find(
    ({ reservation: event }) => event.eventSha256 === reservation.reservationEventSha256,
  );
  if (attempt === undefined) {
    reject(`matrix ${reservation.matrixOrdinal} reservation digest is absent from the disclosed ledger.`);
  }
  if (attempt.finalization.classification !== 'valid-candidate'
    || attempt.reservation.attemptOrdinal !== reservation.attemptOrdinal
    || attempt.reservation.matrixOrdinal !== reservation.matrixOrdinal
    || attempt.reservation.seriesId !== reservation.seriesId
    || attempt.reservation.sourceIdentity.commit !== reservation.sourceCommit
    || attempt.reservation.sourceIdentity.tree !== reservation.sourceTree
    || attempt.reservation.sourceIdentity.executionDependencyClosureSha256
      !== reservation.executionDependencyClosureSha256) {
    reject(`matrix ${reservation.matrixOrdinal} public reservation differs from its ledger event.`);
  }
  const declaredRunDirectory = `${attempt.finalization.attemptDirectory}/runs/${publicRun.metadata.runId}`;
  if (attempt.finalization.runDirectory !== declaredRunDirectory) {
    reject(`matrix ${reservation.matrixOrdinal} runId differs from the ledger run directory.`);
  }
  const reservationAt = Date.parse(attempt.reservation.recordedAt);
  const startedAt = Date.parse(publicRun.startedAt);
  const completedAt = Date.parse(publicRun.completedAt);
  const finalizationAt = Date.parse(attempt.finalization.recordedAt);
  if (![reservationAt, startedAt, completedAt, finalizationAt].every(Number.isFinite)
    || reservationAt > startedAt
    || startedAt > completedAt
    || completedAt > finalizationAt) {
    reject(
      `matrix ${reservation.matrixOrdinal} does not satisfy reservation <= start <= completion <= finalization chronology.`,
    );
  }
  return attempt;
}

export async function verifyPublicLiveCandidatePairBundle(
  bundleDirectory,
  publicRunDirectories,
  {
    repositoryRoot = PROJECT_ROOT,
    inspectPublicRun = inspectPublicLiveCandidateRun,
    anchorTagVerifier = verifyCandidateRegistryAnchorTag,
  } = {},
) {
  if (!Array.isArray(publicRunDirectories) || publicRunDirectories.length !== 2) {
    reject('exactly two public-derived run directories are required.');
  }
  const publicInputSnapshots = new Map();
  for (const directory of publicRunDirectories) {
    const resolvedDirectory = path.resolve(directory);
    if (publicInputSnapshots.has(resolvedDirectory)) {
      reject('the two public run inputs must be distinct directories.');
    }
    publicInputSnapshots.set(
      resolvedDirectory,
      await snapshotRegularDirectory(resolvedDirectory, 'public run input'),
    );
  }
  const bundle = await readPublicPairBundle(bundleDirectory);
  const receipt = bundle.receipt;
  const ledgerText = bundle.ledgerBytes.toString('utf8');
  const state = parseCandidateLedgerText(ledgerText);
  const registryState = parseCandidateRegistryText(bundle.registryBytes.toString('utf8'));
  if (state.pendingReservation !== null) reject('the disclosed ledger has a pending reservation.');
  if (state.terminalFailure !== null) reject('the disclosed ledger ends in a nonretryable failure.');
  if (receipt.seriesId !== state.seriesId
    || receipt.privatePairVerification.seriesId !== state.seriesId) {
    reject('receipt seriesId differs from the disclosed ledger.');
  }
  if (registryState.claims.some(({ materialization }) => materialization === null)) {
    reject('the disclosed root registry contains an unmaterialized claim.');
  }
  const registryBinding = receipt.seriesRootRegistryBinding;
  const selectedRegistryRecord = registryState.byStudyKey.get(registryBinding.studyKey);
  if (selectedRegistryRecord === undefined
    || selectedRegistryRecord.materialization === null
    || selectedRegistryRecord.claim.seriesDirectory !== registryBinding.seriesDirectory
    || !sameCanonical(selectedRegistryRecord.claim.sourceIdentity, state.sourceIdentity)
    || selectedRegistryRecord.materialization.seriesId !== state.seriesId
    || selectedRegistryRecord.materialization.seriesOpeningEventSha256
      !== state.events[0].eventSha256) {
    reject('the disclosed root registry does not materialize the disclosed candidate ledger.');
  }
  const claimRecordedAt = Date.parse(selectedRegistryRecord.claim.recordedAt);
  const seriesOpenedAt = Date.parse(state.events[0].recordedAt);
  const materializedAt = Date.parse(selectedRegistryRecord.materialization.recordedAt);
  if (![claimRecordedAt, seriesOpenedAt, materializedAt].every(Number.isFinite)
    || claimRecordedAt > seriesOpenedAt
    || seriesOpenedAt > materializedAt) {
    reject('the disclosed registry claim, ledger opening, and materialization chronology is invalid.');
  }
  const firstReservationAt = Date.parse(state.attempts[0]?.reservation?.recordedAt);
  if (!Number.isFinite(firstReservationAt) || materializedAt > firstReservationAt) {
    reject('the disclosed registry materialization occurs after the first ledger reservation.');
  }
  const expectedRegistryBinding = {
    filename: PUBLIC_LIVE_PAIR_REGISTRY_NAME,
    bytes: bundle.registryBytes.length,
    sha256: sha256Bytes(bundle.registryBytes),
    eventCount: registryState.events.length,
    finalEventSha256: registryState.finalEventSha256,
    claimEventSha256: selectedRegistryRecord.claim.eventSha256,
    materializedEventSha256: selectedRegistryRecord.materialization.eventSha256,
    seriesOpeningEventSha256:
      selectedRegistryRecord.materialization.seriesOpeningEventSha256,
    studyKey: selectedRegistryRecord.claim.studyKey,
    seriesDirectory: selectedRegistryRecord.claim.seriesDirectory,
    anchorTagName: candidateRegistryAnchor(selectedRegistryRecord).tagName,
    anchorMessageSha256: candidateRegistryAnchor(selectedRegistryRecord).messageSha256,
  };
  if (!sameCanonical(registryBinding, expectedRegistryBinding)) {
    reject('receipt root-registry binding differs from the disclosed registry bytes.');
  }
  const anchorVerification = await anchorTagVerifier(repositoryRoot, {
    anchor: candidateRegistryAnchor(selectedRegistryRecord),
  });
  if (anchorVerification?.verified !== true
    || anchorVerification.tagName !== registryBinding.anchorTagName
    || anchorVerification.messageSha256 !== registryBinding.anchorMessageSha256) {
    reject('the disclosed registry anchor tag was not independently verified.');
  }
  const ledgerBinding = {
    filename: PUBLIC_LIVE_PAIR_LEDGER_NAME,
    bytes: bundle.ledgerBytes.length,
    sha256: sha256Bytes(bundle.ledgerBytes),
    eventCount: state.events.length,
    finalEventSha256: state.events.at(-1).eventSha256,
  };
  if (!sameCanonical(receipt.ledgerBinding, ledgerBinding)) {
    reject('receipt ledger binding differs from the disclosed ledger bytes.');
  }
  const pairReceipt = receipt.privatePairVerification;
  const expectedPrivateRegistryReceipt = {
    experimentId: selectedRegistryRecord.claim.experimentId,
    studyKey: selectedRegistryRecord.claim.studyKey,
    filename: PUBLIC_LIVE_PAIR_REGISTRY_NAME,
    eventCount: registryState.events.length,
    finalEventSha256: registryState.finalEventSha256,
    claimEventSha256: selectedRegistryRecord.claim.eventSha256,
    materializedEventSha256: selectedRegistryRecord.materialization.eventSha256,
    seriesOpeningEventSha256:
      selectedRegistryRecord.materialization.seriesOpeningEventSha256,
    anchorTagName: anchorVerification.tagName,
    anchorMessageSha256: anchorVerification.messageSha256,
    anchorVerified: true,
  };
  if (!sameCanonical(pairReceipt.registry, expectedPrivateRegistryReceipt)) {
    reject('private pair-verifier registry receipt differs from the disclosed registry and tag.');
  }
  if (pairReceipt.ledger.eventCount !== state.events.length
    || pairReceipt.ledger.finalEventSha256 !== state.events.at(-1).eventSha256
    || !sameCanonical(pairReceipt.ledger.sourceIdentity, state.sourceIdentity)
    || pairReceipt.attemptCount !== state.attempts.length
    || !sameCanonical(pairReceipt.attempts, attemptProjectionFromLedger(state))) {
    reject('private pair-verifier receipt differs from the disclosed ledger chronology.');
  }
  const retryCount = state.attempts.filter(
    ({ finalization }) => finalization.classification === 'infrastructure-invalid-retryable',
  ).length;
  const validAttempts = state.attempts.filter(
    ({ finalization }) => finalization.classification === 'valid-candidate',
  );
  const nonretryableFailureCount = state.attempts.filter(
    ({ finalization }) => (
      finalization.classification === 'implementation/evidence-failure-nonretryable'
    ),
  ).length;
  if (pairReceipt.retryCount !== retryCount
    || pairReceipt.completedMatrixCount !== validAttempts.length
    || pairReceipt.validCandidateCount !== validAttempts.length
    || pairReceipt.nonretryableFailureCount !== nonretryableFailureCount) {
    reject('private pair-verifier receipt counts differ from the disclosed ledger.');
  }

  const publicRuns = [];
  for (const directory of publicRunDirectories) {
    publicRuns.push(await inspectPublicRun(directory, { repositoryRoot }));
  }
  publicRuns.sort(
    (left, right) => left.reservation.matrixOrdinal - right.reservation.matrixOrdinal,
  );
  if (publicRuns[0].reservation.matrixOrdinal !== 1
    || publicRuns[1].reservation.matrixOrdinal !== 2) {
    reject('the public run inputs do not represent matrices 1 and 2.');
  }
  for (const publicRun of publicRuns) {
    requirePublicIdentityAdapterTelemetryAssociation(
      publicRun.identity,
      `public matrix ${publicRun.reservation.matrixOrdinal}`,
    );
  }
  for (const publicRun of publicRuns) bindPublicRunToLedger(publicRun, state);
  ensureChronologicalDistinctSessions(publicRuns);
  const sameFrozenIdentity = sameCanonical(publicRuns[0].identity, publicRuns[1].identity);
  if (!sameFrozenIdentity) reject('the two public candidate runs differ in frozen identity.');
  const publicPairIdentitySha256 = sha256Bytes(
    Buffer.from(canonicalJson(publicRuns[0].identity), 'utf8'),
  );
  const pairEligibility = expectedPairEligibility(state, publicRuns, sameFrozenIdentity);
  if (!sameCanonical(pairReceipt.pairEligibility, pairEligibility)
    || pairEligibility.pass !== true) {
    reject('publicly reconstructed pair eligibility differs from the private pair receipt.');
  }

  const publicMatrices = publicRuns.map(({ matrixDecision }) => matrixDecision);
  if (!sameCanonical(pairReceipt.matrices, publicMatrices)) {
    reject('publicly reconstructed matrix decisions differ from the private pair receipt.');
  }
  const decisionPass = pairEligibility.pass
    && publicMatrices.every(({ pass }) => pass === true);
  const decision = {
    status: decisionPass ? 'confirmed-first-device-live' : 'confirmation-not-met',
    pass: decisionPass,
    rule: 'both matrices independently pass all numerical, environment, and evidence gates',
  };
  if (!sameCanonical(pairReceipt.decision, decision)) {
    reject('publicly reconstructed pair decision differs from the private pair receipt.');
  }

  const expectedPublicBindings = publicRuns.map((publicRun) => ({
    matrixOrdinal: publicRun.reservation.matrixOrdinal,
    attemptOrdinal: publicRun.reservation.attemptOrdinal,
    portableLabel: `matrix-${publicRun.reservation.matrixOrdinal}-public-run`,
    runId: publicRun.metadata.runId,
    reservationEventSha256: publicRun.reservation.reservationEventSha256,
    privateArtifactManifest: {
      name: 'artifact-manifest.json',
      sha256: publicRun.privateArtifactManifestSha256,
    },
    publicArtifactManifest: {
      name: 'artifact-manifest.json',
      bytes: publicRun.manifestBytes.length,
      sha256: publicRun.manifestSha256,
    },
    publicBundlePolicyId: publicRun.publicBundlePolicyId,
    candidateCommit: publicRun.candidateCommit,
    candidateTree: publicRun.candidateTree,
  }));
  if (!sameCanonical(receipt.publicRuns, expectedPublicBindings)) {
    reject('receipt public-run manifest bindings differ from the supplied public bundles.');
  }
  for (const [directory, initialSnapshot] of publicInputSnapshots) {
    const finalSnapshot = await snapshotRegularDirectory(directory, 'public run input');
    if (!sameCanonical(initialSnapshot, finalSnapshot)) {
      reject('a public run input changed during pair verification.');
    }
  }

  return {
    schemaVersion: PUBLIC_LIVE_PAIR_POLICY_SCHEMA_VERSION,
    kind: 'first-instance-live-public-candidate-pair-verification',
    status: 'consistent',
    policyId: PUBLIC_LIVE_PAIR_POLICY_ID,
    bundleLabel: PUBLIC_LIVE_PAIR_BUNDLE_LABEL,
    seriesId: state.seriesId,
    integrityScope: PUBLIC_LIVE_PAIR_INTEGRITY_SCOPE,
    privateAttemptScope: PUBLIC_LIVE_PAIR_PRIVATE_ATTEMPT_SCOPE,
    authenticityVerified: false,
    privateOriginalManifestBytesVerified: false,
    privateAttemptArtifactBytesDisclosed: false,
    privateAttemptContentCommitmentsVerified: false,
    privateRootInventoryVerified: false,
    deterministicSanitizerRederivationVerified: false,
    registryLedgerChronologyVerified: true,
    selectionLedgerReplayed: true,
    selectedPublicRunArtifactsVerified: true,
    matrixDecisionsRecomputed: true,
    pairDecisionRecomputed: true,
    receiptBundle: {
      manifest: {
        name: PUBLIC_LIVE_PAIR_MANIFEST_NAME,
        bytes: bundle.manifestBytes.length,
        sha256: sha256Bytes(bundle.manifestBytes),
      },
      ledger: receipt.ledgerBinding,
      registry: receipt.seriesRootRegistryBinding,
      receipt: {
        name: PUBLIC_LIVE_PAIR_RECEIPT_NAME,
        bytes: bundle.receiptBytes.length,
        sha256: sha256Bytes(bundle.receiptBytes),
      },
    },
    privatePairIdentitySha256: pairReceipt.pairIdentitySha256,
    publicPairIdentitySha256,
    attemptCount: state.attempts.length,
    retryCount,
    publicRuns: expectedPublicBindings,
    pairEligibility,
    matrices: publicMatrices,
    decision,
  };
}

export function serializePublicLivePairVerification(result) {
  return `${JSON.stringify(result, null, 2)}\n`;
}

async function main() {
  const [bundleDirectory, firstPublicRun, secondPublicRun, ...extra] = process.argv.slice(2);
  if (!bundleDirectory || !firstPublicRun || !secondPublicRun || extra.length > 0) {
    process.stderr.write(
      'Usage: node analysis/verify-public-live-first-instance-candidate-pair.mjs '
        + '<public-pair-receipt-bundle> <public-run-1> <public-run-2>\n',
    );
    process.exitCode = 2;
    return;
  }
  const result = await verifyPublicLiveCandidatePairBundle(
    bundleDirectory,
    [firstPublicRun, secondPublicRun],
  );
  process.stdout.write(serializePublicLivePairVerification(result));
}

const invokedUrl = process.argv[1] === undefined
  ? null
  : pathToFileURL(path.resolve(process.argv[1])).href;

if (import.meta.url === invokedUrl) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}

export {
  jsonBytes as publicLivePairJsonBytes,
  sha256Bytes as publicLivePairSha256Bytes,
};
