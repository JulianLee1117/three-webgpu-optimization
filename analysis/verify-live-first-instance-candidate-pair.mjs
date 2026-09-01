import { createHash } from 'node:crypto';
import {
  lstat,
  readFile,
  readdir,
  stat,
} from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  buildAttemptContentCommitment,
  CANDIDATE_ARTIFACT_PERSISTENCE_ERROR_CODES,
  CANDIDATE_ARTIFACT_PERSISTENCE_FAILURE_PREFIX,
  candidateSeriesReservationRecord,
  CANDIDATE_LEDGER_FILENAME,
  listSeriesAttemptDirectories,
  readCandidateLedger,
  sha256Canonical,
  sourceIdentityFromProvenance,
} from '../scripts/live-first-instance-candidate-ledger.mjs';
import {
  verifyCandidateRegistryAnchorTag,
  verifyCandidateStudyRegistry,
} from '../scripts/live-first-instance-candidate-registry.mjs';
import {
  evaluateLiveFirstInstanceAdapterTelemetryAssociation,
} from '../scripts/live-first-instance-adapter-telemetry-association.mjs';
import {
  summarizeInput,
  verifyRunDirectory,
} from './summarize.mjs';

const LIVE_MATRIX_KIND = 'first-instance-live';
const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CANDIDATE_SERIES_ROOT = path.join(PROJECT_ROOT, 'results', 'candidate-series');
const ENVIRONMENT_AUDIT_KEYS = Object.freeze([
  'schemaVersion',
  'kind',
  'telemetryStatus',
  'telemetryMalformedLineCount',
  'telemetryStderrByteCount',
  'telemetrySampleCount',
  'telemetryCoveragePass',
  'adapterTelemetryAssociation',
  'computeProcessIdentityComparison',
  'candidateEnvironmentGate',
  'overallEvidenceDecision',
]);
const ENVIRONMENT_GATE_KEYS = Object.freeze([
  'schemaVersion',
  'kind',
  'applicable',
  'status',
  'pass',
  'retryable',
  'nonReplaceable',
  'collectorPass',
  'processIdentityPass',
  'adapterTelemetryAssociationPass',
  'failureCodes',
  'reasons',
]);
const OVERALL_DECISION_KEYS = Object.freeze([
  'schemaVersion',
  'kind',
  'applicable',
  'status',
  'pass',
  'retryable',
  'nonReplaceable',
  'failedGates',
  'failureCodes',
  'reasons',
]);
const RETRYABLE_COLLECTOR_CODES = new Set([
  'telemetry-status-unavailable',
  'telemetry-malformed-records',
  'telemetry-stderr-output',
  'telemetry-no-samples',
  'compute-process-snapshot-invalid',
  'telemetry-coverage-invalid',
]);
const ADAPTER_TELEMETRY_ASSOCIATION_FAILURE_CODES = new Set([
  'adapter-vendor-invalid',
  'adapter-vendor-not-nvidia',
  'adapter-description-invalid',
  'telemetry-gpu-count-not-one',
  'telemetry-gpu-index-invalid',
  'telemetry-gpu-name-invalid',
  'adapter-telemetry-name-mismatch',
]);
const DEVICE_LOSS_PATTERNS = Object.freeze([
  /\bdevice\s+(?:was\s+)?lost\b/i,
  /\bGPUDevice[^\r\n]*\blost\b/i,
  /\bdevice-lost\b/i,
  /\bTargetClosedError\b/i,
  /\btarget (?:page, context or browser|page|browser) has been closed\b/i,
  /\bpage crashed\b/i,
  /\bbrowser disconnected\b/i,
  /\bGPU process crashed\b/i,
]);

function reject(message) {
  throw new Error(`Live first-instance candidate-pair verification rejected: ${message}`);
}

function record(value, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    reject(`${label} must be an object.`);
  }
  return value;
}

function array(value, label) {
  if (!Array.isArray(value)) reject(`${label} must be an array.`);
  return value;
}

function exactKeys(value, expected, label) {
  const candidate = record(value, label);
  const keys = Object.keys(candidate).sort();
  const expectedKeys = [...expected].sort();
  if (keys.length !== expectedKeys.length
    || keys.some((key, index) => key !== expectedKeys[index])) {
    reject(`${label} has an unexpected schema.`);
  }
  return candidate;
}

function nonemptyString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') reject(`${label} must be non-empty.`);
  return value;
}

function sameCanonical(left, right) {
  return sha256Canonical(left) === sha256Canonical(right);
}

async function readJsonIfPresent(filename) {
  try {
    return JSON.parse(await readFile(filename, 'utf8'));
  } catch (error) {
    if (error?.code === 'ENOENT' || error instanceof SyntaxError) return null;
    throw error;
  }
}

async function readTextIfPresent(filename) {
  try {
    return await readFile(filename, 'utf8');
  } catch (error) {
    if (error?.code === 'ENOENT') return null;
    throw error;
  }
}

function exactArtifactPersistenceMarker(stderrText) {
  if (typeof stderrText !== 'string') return null;
  const markerLines = stderrText.split(/\r?\n/).filter(
    (line) => line.startsWith(CANDIDATE_ARTIFACT_PERSISTENCE_FAILURE_PREFIX),
  );
  if (markerLines.length !== 1) return null;
  let marker;
  try {
    marker = JSON.parse(markerLines[0].slice(
      CANDIDATE_ARTIFACT_PERSISTENCE_FAILURE_PREFIX.length,
    ));
  } catch {
    return null;
  }
  try {
    exactKeys(
      marker,
      ['schemaVersion', 'kind', 'operation', 'code'],
      'runner artifact-persistence marker',
    );
  } catch {
    return null;
  }
  return marker.schemaVersion === 1
    && marker.kind === 'first-instance-live-candidate-artifact-persistence-failure'
    && marker.operation === 'persist-run-artifacts'
    && CANDIDATE_ARTIFACT_PERSISTENCE_ERROR_CODES.includes(marker.code)
    ? marker
    : null;
}

function metadataRecordsArtifactPersistenceFailure(metadata, marker) {
  if (metadata?.status !== 'failed' || marker === null) return false;
  const errorText = [metadata?.error?.message, metadata?.error?.stack]
    .filter((value) => typeof value === 'string')
    .join('\n');
  if (/\bGPUValidationError\b|\bWGSL\b|\bshader\b|\bWebGPU\s+validation\b/i.test(errorText)) {
    return false;
  }
  return new RegExp(`(?:^|[^A-Z0-9_])${marker.code}(?:$|[^A-Z0-9_])`).test(errorText);
}

function metadataRecordsShaderOrValidationFailure(metadata) {
  if (metadata?.status !== 'failed') return false;
  const errorText = [metadata?.error?.name, metadata?.error?.message, metadata?.error?.stack]
    .filter((value) => typeof value === 'string')
    .join('\n');
  return /\bGPUValidationError\b|\bWGSL\b|\bshader\b|\bWebGPU\s+validation\b/i.test(
    errorText,
  );
}

async function artifactInfrastructureFailureObserved(
  attemptDirectory,
  metadata,
  child,
) {
  // Complete benchmark metadata is always consumed. Failed metadata permits an
  // artifact retry only when both the runner marker and its error record bind
  // the same allowlisted filesystem code. An ordinary shader/validation error
  // therefore cannot borrow a later I/O marker or manifest loss.
  if (metadata?.status === 'complete' || child.pass !== true) return false;
  const lifecycle = child.lifecycle;
  const childUnsuccessful = lifecycle.exitCode !== 0 || lifecycle.signal !== null;
  if (!childUnsuccessful) return false;
  const signal = lifecycle.signal;
  const spawnFailure = typeof signal === 'string' && signal.startsWith('spawn-error:');
  const operatingSystemSignal = typeof signal === 'string' && !spawnFailure;
  const windowsAbnormalStatus = signal === null
    && Number.isSafeInteger(lifecycle.exitCode)
    && lifecycle.exitCode >= 0x80000000;
  const stderrText = await readTextIfPresent(
    path.join(attemptDirectory, 'runner-stderr.log'),
  );
  const persistenceMarker = exactArtifactPersistenceMarker(stderrText);
  const noReadableMetadata = metadata === null;
  return (noReadableMetadata && (spawnFailure
    || operatingSystemSignal
    || windowsAbnormalStatus
    || persistenceMarker !== null))
    || metadataRecordsArtifactPersistenceFailure(metadata, persistenceMarker);
}

async function directRunDirectories(attemptDirectory) {
  const runsDirectory = path.join(attemptDirectory, 'runs');
  try {
    const entries = await readdir(runsDirectory, { withFileTypes: true });
    const names = [];
    for (const entry of entries) {
      if (!entry.isDirectory()) reject(`runner output root contains non-directory ${entry.name}.`);
      const entryStat = await lstat(path.join(runsDirectory, entry.name));
      if (entryStat.isSymbolicLink()) reject(`runner output ${entry.name} is a symlink.`);
      names.push(entry.name);
    }
    return names.sort();
  } catch (error) {
    if (error?.code === 'ENOENT') return [];
    throw error;
  }
}

function manifestFileEntryMap(manifest) {
  if (manifest === null || typeof manifest !== 'object' || Array.isArray(manifest)) return null;
  if (!Array.isArray(manifest.requiredFiles)
    || !Array.isArray(manifest.optionalFiles)
    || !Array.isArray(manifest.files)) return null;
  const entries = new Map();
  for (const entry of manifest.files) {
    if (entry === null || typeof entry !== 'object' || Array.isArray(entry)
      || typeof entry.name !== 'string' || entries.has(entry.name)) return null;
    entries.set(entry.name, entry);
  }
  return entries;
}

async function artifactManifestStatus(runDirectory) {
  if (runDirectory === null) return 'unavailable';
  const manifestPath = path.join(runDirectory, 'artifact-manifest.json');
  const manifest = await readJsonIfPresent(manifestPath);
  const entries = manifestFileEntryMap(manifest);
  if (entries === null) return 'incomplete-or-corrupt';
  const actualEntries = await readdir(runDirectory, { withFileTypes: true });
  if (actualEntries.some((entry) => !entry.isFile())) return 'incomplete-or-corrupt';
  const declared = new Set([
    ...manifest.requiredFiles,
    ...manifest.optionalFiles.map((entry) => entry?.name),
  ]);
  if (declared.has(undefined)
    || declared.size !== entries.size
    || [...declared].some((name) => !entries.has(name))) {
    return 'incomplete-or-corrupt';
  }
  const expectedActualNames = new Set([
    'artifact-manifest.json',
    ...[...entries.entries()]
      .filter(([, entry]) => entry.present === true)
      .map(([name]) => name),
  ]);
  const actualNames = new Set(actualEntries.map((entry) => entry.name));
  if (expectedActualNames.size !== actualNames.size
    || [...expectedActualNames].some((name) => !actualNames.has(name))) {
    return 'incomplete-or-corrupt';
  }
  for (const [name, entry] of entries) {
    if (entry.present !== true) {
      if (entry.required === true) return 'incomplete-or-corrupt';
      try {
        await stat(path.join(runDirectory, name));
        return 'incomplete-or-corrupt';
      } catch (error) {
        if (error?.code !== 'ENOENT') throw error;
      }
      continue;
    }
    try {
      const contents = await readFile(path.join(runDirectory, name));
      if (contents.length !== entry.bytes
        || createHash('sha256').update(contents).digest('hex') !== entry.sha256) {
        return 'incomplete-or-corrupt';
      }
    } catch (error) {
      if (error?.code === 'ENOENT') return 'incomplete-or-corrupt';
      throw error;
    }
  }
  return 'consistent';
}

async function attemptLayoutExact(attemptDirectory) {
  const entries = await readdir(attemptDirectory, { withFileTypes: true });
  const expectedFiles = new Set([
    'orchestrator-source-after.json',
    'runner-child-lifecycle.json',
    'runner-stderr.log',
    'runner-stdout.log',
  ]);
  const expectedDirectory = 'runs';
  if (entries.length !== expectedFiles.size + 1) return false;
  for (const entry of entries) {
    if (expectedFiles.has(entry.name)) {
      if (!entry.isFile()) return false;
    } else if (entry.name === expectedDirectory) {
      if (!entry.isDirectory()) return false;
    } else {
      return false;
    }
    if ((await lstat(path.join(attemptDirectory, entry.name))).isSymbolicLink()) return false;
  }
  return true;
}

async function closedChildLifecycle(attemptDirectory) {
  const lifecycle = await readJsonIfPresent(
    path.join(attemptDirectory, 'runner-child-lifecycle.json'),
  );
  if (lifecycle === null || typeof lifecycle !== 'object' || Array.isArray(lifecycle)) {
    return { pass: false, lifecycle: null };
  }
  const keys = Object.keys(lifecycle).sort();
  const expected = [
    'childPid',
    'closedAt',
    'exitCode',
    'kind',
    'schemaVersion',
    'signal',
    'spawnedAt',
    'status',
  ].sort();
  const pass = keys.length === expected.length
    && keys.every((key, index) => key === expected[index])
    && lifecycle.schemaVersion === 1
    && lifecycle.kind === 'first-instance-live-candidate-child-lifecycle'
    && lifecycle.status === 'closed'
    && (lifecycle.childPid === null
      || (Number.isSafeInteger(lifecycle.childPid) && lifecycle.childPid > 0))
    && Number.isFinite(Date.parse(lifecycle.spawnedAt))
    && Number.isFinite(Date.parse(lifecycle.closedAt))
    && Date.parse(lifecycle.closedAt) >= Date.parse(lifecycle.spawnedAt)
    && (lifecycle.exitCode === null
      || (Number.isSafeInteger(lifecycle.exitCode) && lifecycle.exitCode >= 0))
    && (lifecycle.signal === null
      || (typeof lifecycle.signal === 'string' && lifecycle.signal !== ''))
    && !(lifecycle.exitCode === null && lifecycle.signal === null);
  return { pass, lifecycle };
}

function candidateEnvironmentGate(metadata, { allowFailedMetadata = false } = {}) {
  const audit = exactKeys(
    metadata?.liveFirstInstanceEnvironmentAudit,
    ENVIRONMENT_AUDIT_KEYS,
    'metadata.liveFirstInstanceEnvironmentAudit',
  );
  if (audit.schemaVersion !== 4
    || audit.kind !== 'first-instance-live-crossover-environment-audit') {
    reject('metadata.liveFirstInstanceEnvironmentAudit must use the version-4 protocol schema.');
  }
  const gate = exactKeys(
    audit.candidateEnvironmentGate,
    ENVIRONMENT_GATE_KEYS,
    'metadata candidateEnvironmentGate',
  );
  if (gate.schemaVersion !== 2
    || gate.kind !== 'first-instance-live-candidate-environment-gate'
    || gate.applicable !== true) {
    reject('candidateEnvironmentGate is not the candidate gate declared by the protocol.');
  }
  array(gate.failureCodes, 'candidateEnvironmentGate.failureCodes');
  array(gate.reasons, 'candidateEnvironmentGate.reasons');
  const passed = gate.status === 'passed'
    && gate.pass === true
    && gate.retryable === false
    && gate.nonReplaceable === false
    && gate.collectorPass === true
    && gate.processIdentityPass === true
    && gate.adapterTelemetryAssociationPass === true
    && gate.failureCodes.length === 0
    && gate.reasons.length === 0;
  const nonReplaceableMismatch = gate.status === 'failed-non-replaceable-process-set-mismatch'
    && gate.pass === false
    && gate.retryable === false
    && gate.nonReplaceable === true
    && gate.collectorPass === true
    && gate.processIdentityPass === false
    && typeof gate.adapterTelemetryAssociationPass === 'boolean'
    && gate.failureCodes.length === 1
    && gate.failureCodes[0] === 'compute-process-set-mismatch';
  const nonReplaceableAdapterAssociation =
    gate.status === 'failed-non-replaceable-adapter-telemetry-association'
    && gate.pass === false
    && gate.retryable === false
    && gate.nonReplaceable === true
    && gate.collectorPass === true
    && gate.processIdentityPass === true
    && gate.adapterTelemetryAssociationPass === false
    && gate.failureCodes.length > 0
    && gate.failureCodes.every(
      (code) => ADAPTER_TELEMETRY_ASSOCIATION_FAILURE_CODES.has(code),
    )
    && sameCanonical(gate.failureCodes, audit.adapterTelemetryAssociation?.failureCodes)
    && sameCanonical(gate.reasons, audit.adapterTelemetryAssociation?.reasons);
  const retryableCollector = gate.status === 'failed-retryable-collector'
    && gate.pass === false
    && gate.retryable === true
    && gate.nonReplaceable === false
    && gate.collectorPass === false
    && gate.processIdentityPass === null
    && gate.adapterTelemetryAssociationPass === null
    && gate.failureCodes.length > 0
    && gate.failureCodes.every((code) => RETRYABLE_COLLECTOR_CODES.has(code));
  const coverageFailureDeclared = gate.failureCodes.includes('telemetry-coverage-invalid');
  const coverageDispositionConsistent = audit.telemetryCoveragePass === true
    ? !coverageFailureDeclared
    : coverageFailureDeclared;
  if (!passed
    && !nonReplaceableMismatch
    && !nonReplaceableAdapterAssociation
    && !(allowFailedMetadata && retryableCollector)) {
    reject('candidateEnvironmentGate fields are internally inconsistent.');
  }
  if (!coverageDispositionConsistent
    || (gate.collectorPass === true && audit.telemetryCoveragePass !== true)) {
    reject('candidateEnvironmentGate disagrees with the telemetry coverage audit.');
  }
  return gate;
}

function retryableTelemetryCollectorFailure(metadata) {
  try {
    const gate = candidateEnvironmentGate(metadata, { allowFailedMetadata: true });
    const expectedRows = metadata?.expectedTrialCount * metadata?.protocol?.measuredFrames;
    return gate.status === 'failed-retryable-collector'
      && metadata?.status === 'failed'
      && typeof metadata?.error?.message === 'string'
      && metadata.error.message.startsWith(
        'Live first-instance candidate telemetry collector rejected:',
      )
      && Array.isArray(metadata.pageErrors)
      && metadata.pageErrors.length === 0
      && Array.isArray(metadata.webgpuUncapturedErrors)
      && metadata.webgpuUncapturedErrors.length === 0
      && Number.isSafeInteger(metadata.expectedTrialCount)
      && metadata.expectedTrialCount === 24
      && metadata.completedTrialCount === metadata.expectedTrialCount
      && metadata.acceptedTrialCount === metadata.expectedTrialCount
      && metadata.validationArtifactCount === metadata.expectedTrialCount
      && metadata.frameRowCount === expectedRows
      && metadata.liveFirstInstanceAnalysisAudit?.nTrials === metadata.expectedTrialCount
      && metadata.liveFirstInstanceAnalysisAudit?.nRows === expectedRows;
  } catch {
    return false;
  }
}

function flattenErrorText(value, result = []) {
  if (value === null || value === undefined) return result;
  if (typeof value === 'string') {
    result.push(value);
    return result;
  }
  if (Array.isArray(value)) {
    for (const item of value) flattenErrorText(item, result);
    return result;
  }
  if (typeof value === 'object') {
    for (const [key, item] of Object.entries(value)) {
      if (/error|reason|message|detail|loss/i.test(key)) flattenErrorText(item, result);
    }
  }
  return result;
}

async function browserOrDeviceLossObserved(metadata) {
  const page = metadata?.environment?.benchmarkPage;
  const pageAtEnd = metadata?.environment?.benchmarkPageAtEnd;
  if ((Number.isSafeInteger(page?.webgpuDeviceLossCount) && page.webgpuDeviceLossCount > 0)
    || (Number.isSafeInteger(pageAtEnd?.webgpuDeviceLossCount)
      && pageAtEnd.webgpuDeviceLossCount > 0)) return true;
  const textParts = flattenErrorText({
    error: metadata?.error,
    pageErrors: metadata?.pageErrors,
    uncaptured: metadata?.webgpuUncapturedErrors,
  });
  return DEVICE_LOSS_PATTERNS.some((pattern) => textParts.some((text) => pattern.test(text)));
}

function relativeRunDirectory(seriesDirectory, runDirectory) {
  if (runDirectory === null) return null;
  return path.relative(seriesDirectory, runDirectory).replaceAll('\\', '/');
}

export async function inspectCandidateAttempt(
  seriesDirectory,
  attempt,
  { runVerifier = verifyRunDirectory } = {},
) {
  const finalization = attempt.finalization;
  const attemptDirectory = path.join(seriesDirectory, finalization.attemptDirectory);
  const commitment = await buildAttemptContentCommitment(attemptDirectory);
  if (!sameCanonical(commitment, finalization.contentCommitment)) {
    reject(`${finalization.attemptDirectory} content changed after finalization.`);
  }
  const classified = await classifyUnfinalizedCandidateAttempt(
    seriesDirectory,
    finalization.attemptDirectory,
    {
      runVerifier,
      expectedSourceIdentity: attempt.reservation.sourceIdentity,
      expectedReservation: attempt.reservation,
    },
  );
  const {
    runDirectory,
    metadata,
    artifactVerification,
    verificationError,
    classification,
    reasonCode,
    classificationEvidence,
  } = classified;
  const declaredRunDirectory = relativeRunDirectory(seriesDirectory, runDirectory);
  if (declaredRunDirectory !== finalization.runDirectory) {
    reject(`${finalization.attemptDirectory} run-directory declaration differs from inventory.`);
  }
  if (classification !== finalization.classification
    || reasonCode !== finalization.reasonCode
    || !sameCanonical(classificationEvidence, finalization.classificationEvidence)) {
    reject(`${finalization.attemptDirectory} final classification is not independently reproducible.`);
  }
  if (classified.childLifecycle !== null
    && (classified.childLifecycle.exitCode !== finalization.execution.exitCode
      || classified.childLifecycle.signal !== finalization.execution.signal)) {
    reject(`${finalization.attemptDirectory} child lifecycle differs from final execution.`);
  }
  if (metadata !== null) {
    const reservedAt = Date.parse(attempt.reservation.recordedAt);
    const startedAt = Date.parse(metadata.startedAt);
    const completedAt = Date.parse(metadata.completedAt);
    const finalizedAt = Date.parse(finalization.recordedAt);
    if (![reservedAt, startedAt, completedAt, finalizedAt].every(Number.isFinite)
      || startedAt < reservedAt
      || completedAt < startedAt
      || finalizedAt < completedAt) {
      reject(
        `${finalization.attemptDirectory} does not prove reservation before timing and `
          + 'finalization after completion.',
      );
    }
  }
  return {
    attemptOrdinal: finalization.attemptOrdinal,
    matrixOrdinal: finalization.matrixOrdinal,
    attemptDirectory: finalization.attemptDirectory,
    runDirectory,
    classification,
    reasonCode,
    classificationEvidence,
    metadata,
    artifactVerification,
    verificationError,
  };
}

export async function classifyUnfinalizedCandidateAttempt(
  seriesDirectory,
  attemptDirectoryName,
  {
    runVerifier = verifyRunDirectory,
    expectedSourceIdentity = null,
    expectedReservation = null,
  } = {},
) {
  const attemptDirectory = path.join(seriesDirectory, attemptDirectoryName);
  const runNames = await directRunDirectories(attemptDirectory);
  const runDirectory = runNames.length === 1
    ? path.join(attemptDirectory, 'runs', runNames[0])
    : null;
  const metadata = runDirectory === null
    ? null
    : await readJsonIfPresent(path.join(runDirectory, 'metadata.json'));
  const manifestStatus = runNames.length === 1
    ? await artifactManifestStatus(runDirectory)
    : 'unavailable';
  let verificationStatus = 'failed';
  let artifactVerification = null;
  let verificationError = null;
  if (runDirectory !== null) {
    try {
      const verified = await runVerifier(runDirectory);
      artifactVerification = verified?.artifactVerification ?? verified;
      verificationStatus = 'passed';
    } catch (error) {
      verificationError = error instanceof Error ? error.message : String(error);
    }
  }
  const collectorFailure = retryableTelemetryCollectorFailure(metadata);
  const deviceLoss = await browserOrDeviceLossObserved(metadata);
  const layoutExact = await attemptLayoutExact(attemptDirectory);
  const sourceAfter = await readJsonIfPresent(
    path.join(attemptDirectory, 'orchestrator-source-after.json'),
  );
  let sourceIdentityStable = false;
  try {
    const sourceAfterRecord = exactKeys(
      sourceAfter,
      ['schemaVersion', 'kind', 'sourceProvenance', 'executionDependencyClosure'],
      'orchestrator source-after evidence',
    );
    const dependencyAfter = exactKeys(
      sourceAfterRecord.executionDependencyClosure,
      [
        'schemaVersion',
        'kind',
        'root',
        'format',
        'hashAlgorithm',
        'exclusions',
        'fileCount',
        'totalBytes',
        'sha256',
      ],
      'orchestrator execution dependency closure',
    );
    if (sourceAfterRecord.schemaVersion !== 1
      || sourceAfterRecord.kind !== 'first-instance-live-orchestrator-source-after'
      || dependencyAfter.schemaVersion !== 1
      || dependencyAfter.kind !== 'installed-execution-dependency-closure'
      || dependencyAfter.root !== 'node_modules'
      || dependencyAfter.format !== 'node-modules-sorted-path-size-content-sha256-v1'
      || dependencyAfter.hashAlgorithm !== 'sha256'
      || !sameCanonical(dependencyAfter.exclusions, ['.bin/**', '.vite*/**'])) {
      throw new Error('source-after evidence has an unsupported dependency closure.');
    }
    const afterIdentity = sourceIdentityFromProvenance(
      sourceAfterRecord.sourceProvenance,
      dependencyAfter,
    );
    sourceIdentityStable = expectedSourceIdentity !== null
      && sameCanonical(afterIdentity, expectedSourceIdentity);
  } catch {
    sourceIdentityStable = false;
  }
  const expectedReservationRecord = expectedReservation === null
    ? null
    : candidateSeriesReservationRecord(expectedReservation);
  const reservationBindingStatus = metadata === null
    ? 'unavailable'
    : expectedReservationRecord !== null
      && sameCanonical(metadata.candidateSeriesReservation, expectedReservationRecord)
      ? 'matched'
      : 'mismatched';
  const child = await closedChildLifecycle(attemptDirectory);
  const surplusRunDirectories = runNames.length > 1;
  const childExecutionSuccessful = child.pass === true
    && child.lifecycle.exitCode === 0
    && child.lifecycle.signal === null;
  const artifactInfrastructureFailure = await artifactInfrastructureFailureObserved(
    attemptDirectory,
    metadata,
    child,
  );
  const corruptedCompleteOrImplementationMetadata =
    manifestStatus === 'incomplete-or-corrupt'
    && (metadata?.status === 'complete'
      || metadataRecordsShaderOrValidationFailure(metadata));

  let classification;
  let reasonCode;
  if (!sourceIdentityStable
    || !layoutExact
    || !child.pass
    || reservationBindingStatus === 'mismatched'
    || surplusRunDirectories) {
    classification = 'implementation/evidence-failure-nonretryable';
    reasonCode = 'implementation-or-evidence-failure';
  } else if (verificationStatus === 'passed' && childExecutionSuccessful) {
    classification = 'valid-candidate';
    reasonCode = 'completed-candidate';
  } else if (corruptedCompleteOrImplementationMetadata) {
    classification = 'implementation/evidence-failure-nonretryable';
    reasonCode = 'implementation-or-evidence-failure';
  } else if (deviceLoss && !childExecutionSuccessful) {
    classification = 'infrastructure-invalid-retryable';
    reasonCode = 'browser-or-device-loss';
  } else if (collectorFailure && !childExecutionSuccessful) {
    classification = 'infrastructure-invalid-retryable';
    reasonCode = 'telemetry-collector-failure';
  } else if (artifactInfrastructureFailure) {
    classification = 'infrastructure-invalid-retryable';
    reasonCode = 'run-artifact-incompleteness-or-corruption';
  } else {
    classification = 'implementation/evidence-failure-nonretryable';
    reasonCode = 'implementation-or-evidence-failure';
  }
  const classificationEvidence = {
    artifactManifestStatus: manifestStatus,
    artifactInfrastructureFailureObserved: artifactInfrastructureFailure,
    attemptLayoutExact: layoutExact,
    browserOrDeviceLossObserved: deviceLoss,
    candidateReservationBindingStatus: reservationBindingStatus,
    childLifecycleClosed: child.pass,
    runVerificationStatus: verificationStatus,
    runDirectoryCount: runNames.length,
    sourceIdentityStable,
    telemetryCollectorFailureObserved: collectorFailure,
  };
  return {
    runDirectory,
    runDirectoryRelative: relativeRunDirectory(seriesDirectory, runDirectory),
    classification,
    reasonCode,
    classificationEvidence,
    metadata,
    artifactVerification,
    verificationError,
    childLifecycle: child.lifecycle,
  };
}

function candidateIdentity(metadata) {
  if (metadata?.status !== 'complete'
    || metadata?.error !== null
    || metadata?.evidenceStatus !== 'candidate'
    || metadata?.protocol?.matrixKind !== LIVE_MATRIX_KIND) {
    reject('a valid-candidate matrix lacks complete candidate live-matrix metadata.');
  }
  const provenance = record(metadata.sourceProvenance?.start, 'candidate source provenance');
  const browser = record(metadata.environment?.browser, 'candidate browser identity');
  const page = record(metadata.environment?.benchmarkPage, 'candidate page environment');
  const adapterInfo = record(page.adapterInfo, 'candidate adapter identity');
  const workload = record(metadata.workload, 'candidate workload identity');
  const dependencyClosure = record(
    metadata.executionDependencyClosure?.start,
    'candidate execution dependency closure',
  );
  const viteRuntimeAudit = record(
    metadata.candidateViteRuntimeAudit,
    'candidate Vite runtime audit',
  );
  const telemetryGpus = array(
    metadata.environment?.gpuTelemetry?.summary?.gpus,
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
    reject('candidate telemetry must identify exactly one concrete physical GPU.');
  }
  const adapterTelemetryAssociation =
    evaluateLiveFirstInstanceAdapterTelemetryAssociation({
      adapterInfo,
      telemetryReport: metadata.environment?.gpuTelemetry,
    });
  const retainedAdapterTelemetryAssociation = record(
    metadata.liveFirstInstanceEnvironmentAudit?.adapterTelemetryAssociation,
    'candidate adapter-to-telemetry association',
  );
  if (adapterTelemetryAssociation.pass !== true
    || !sameCanonical(
    retainedAdapterTelemetryAssociation,
    adapterTelemetryAssociation,
  )) {
    reject(
      'candidate adapter-to-telemetry association differs from the page adapter and telemetry GPU evidence.',
    );
  }
  return {
    source: {
      commit: provenance.commit,
      tree: provenance.tree,
      trackedFilesSha256: provenance.trackedFilesSha256,
      packageLockSha256: provenance.packageLockSha256,
      executionDependencyClosureSha256: dependencyClosure.sha256,
      executionDependencyFileCount: dependencyClosure.fileCount,
      executionDependencyTotalBytes: dependencyClosure.totalBytes,
    },
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
      workload,
    },
  };
}

function overallEvidenceDecision(summary) {
  const evidence = exactKeys(
    summary?.liveFirstInstanceEvidenceDecision,
    [
      'adapterTelemetryAssociation',
      'candidateEnvironmentGate',
      'overallEvidenceDecision',
    ],
    'single-matrix liveFirstInstanceEvidenceDecision',
  );
  const decision = exactKeys(
    evidence.overallEvidenceDecision,
    OVERALL_DECISION_KEYS,
    'single-matrix overallEvidenceDecision',
  );
  if (decision.schemaVersion !== 1
    || decision.kind !== 'first-instance-live-overall-evidence-decision'
    || decision.applicable !== true) {
    reject('single-matrix overallEvidenceDecision is not candidate-applicable.');
  }
  array(decision.failedGates, 'overallEvidenceDecision.failedGates');
  array(decision.failureCodes, 'overallEvidenceDecision.failureCodes');
  array(decision.reasons, 'overallEvidenceDecision.reasons');
  if (typeof decision.pass !== 'boolean'
    || typeof decision.retryable !== 'boolean'
    || typeof decision.nonReplaceable !== 'boolean') {
    reject('overallEvidenceDecision disposition fields must be boolean.');
  }
  return decision;
}

async function evaluateValidCandidate(inspection, summarizeRun) {
  const summary = await summarizeRun(inspection.runDirectory);
  if (summary?.artifactVerification?.status !== 'consistent'
    || summary.artifactVerification.evidenceStatus !== 'candidate') {
    reject(`matrix ${inspection.matrixOrdinal} summary is not bound to candidate artifacts.`);
  }
  const numerical = record(
    summary.preregisteredNumericalDecision,
    `matrix ${inspection.matrixOrdinal} numerical decision`,
  );
  if (typeof numerical.pass !== 'boolean') {
    reject(`matrix ${inspection.matrixOrdinal} numerical decision lacks a boolean pass.`);
  }
  const environment = candidateEnvironmentGate(inspection.metadata);
  const overall = overallEvidenceDecision(summary);
  if (!sameCanonical(
    summary.liveFirstInstanceEvidenceDecision.adapterTelemetryAssociation,
    inspection.metadata.liveFirstInstanceEnvironmentAudit.adapterTelemetryAssociation,
  ) || !sameCanonical(
    summary.liveFirstInstanceEvidenceDecision.candidateEnvironmentGate,
    environment,
  ) || !sameCanonical(
    inspection.metadata.liveFirstInstanceEnvironmentAudit.overallEvidenceDecision,
    overall,
  )) {
    reject(`matrix ${inspection.matrixOrdinal} analyzer decisions differ from metadata.`);
  }
  const expectedOverallPass = numerical.pass === true && environment.pass === true;
  if (overall.pass !== expectedOverallPass) {
    reject(`matrix ${inspection.matrixOrdinal} overall decision does not conjunct numerical and environment gates.`);
  }
  if (overall.retryable !== false
    || overall.nonReplaceable !== (expectedOverallPass ? false : true)) {
    reject(`matrix ${inspection.matrixOrdinal} overall disposition is inconsistent.`);
  }
  if (expectedOverallPass && overall.status !== 'passed') {
    reject(`matrix ${inspection.matrixOrdinal} passing overall decision has the wrong status.`);
  }
  if (!expectedOverallPass && overall.status !== 'failed-non-replaceable') {
    reject(`matrix ${inspection.matrixOrdinal} failed overall decision has the wrong status.`);
  }
  const expectedFailedGates = [
    ...(environment.pass ? [] : ['candidateEnvironmentGate']),
    ...(numerical.pass ? [] : ['preregisteredNumericalDecision']),
  ];
  if (!sameCanonical(overall.failedGates, expectedFailedGates)) {
    reject(`matrix ${inspection.matrixOrdinal} overall failedGates are not reconstructable.`);
  }
  return {
    matrixOrdinal: inspection.matrixOrdinal,
    attemptOrdinal: inspection.attemptOrdinal,
    runId: nonemptyString(inspection.metadata.runId, 'candidate runId'),
    startedAt: nonemptyString(inspection.metadata.startedAt, 'candidate startedAt'),
    completedAt: nonemptyString(inspection.metadata.completedAt, 'candidate completedAt'),
    identity: candidateIdentity(inspection.metadata),
    numericalDecision: numerical,
    environmentDecision: environment,
    overallEvidenceDecision: overall,
    pass: overall.pass,
    summary,
  };
}

function verifyPairIdentity(candidates) {
  if (candidates.length !== 2) return { pass: false, sha256: null };
  if (!sameCanonical(candidates[0].identity, candidates[1].identity)) {
    reject('the two completed candidate matrices differ in frozen source, dependency, Vite runtime audit, browser, backend, adapter/driver, adapter-to-telemetry association, or workload identity.');
  }
  if (candidates[0].runId === candidates[1].runId) {
    reject('the two candidate matrices reuse a runId instead of separate sessions.');
  }
  const firstStart = Date.parse(candidates[0].startedAt);
  const firstEnd = Date.parse(candidates[0].completedAt);
  const secondStart = Date.parse(candidates[1].startedAt);
  const secondEnd = Date.parse(candidates[1].completedAt);
  if (![firstStart, firstEnd, secondStart, secondEnd].every(Number.isFinite)
    || firstEnd < firstStart
    || secondEnd < secondStart
    || secondStart < firstEnd) {
    reject('candidate metadata does not prove two chronological, non-overlapping sessions.');
  }
  return { pass: true, sha256: sha256Canonical(candidates[0].identity) };
}

export async function verifyCandidateSeries(
  seriesInput,
  {
    runVerifier = verifyRunDirectory,
    summarizeRun = summarizeInput,
    candidateSeriesRoot = CANDIDATE_SERIES_ROOT,
    registryVerifier = verifyCandidateStudyRegistry,
    anchorTagVerifier = verifyCandidateRegistryAnchorTag,
    repositoryRoot = PROJECT_ROOT,
  } = {},
) {
  const resolvedInput = path.resolve(seriesInput);
  const inputStats = await stat(resolvedInput);
  const seriesDirectory = inputStats.isDirectory() ? resolvedInput : path.dirname(resolvedInput);
  const ledgerPath = inputStats.isDirectory()
    ? path.join(resolvedInput, CANDIDATE_LEDGER_FILENAME)
    : resolvedInput;
  if (!inputStats.isDirectory() && path.basename(ledgerPath) !== CANDIDATE_LEDGER_FILENAME) {
    reject(`ledger filename must be ${CANDIDATE_LEDGER_FILENAME}.`);
  }
  const registryVerification = await registryVerifier(
    candidateSeriesRoot,
    seriesDirectory,
  );
  const anchorVerification = await anchorTagVerifier(
    repositoryRoot,
    registryVerification,
  );
  const state = await readCandidateLedger(ledgerPath);
  if (!sameCanonical(registryVerification.sourceIdentity, state.sourceIdentity)
    || registryVerification.seriesId !== state.seriesId
    || registryVerification.seriesOpeningEventSha256 !== state.events[0].eventSha256) {
    reject('series ledger differs from the canonical root-registry materialization.');
  }
  const firstReservation = state.events.find(
    (event) => event.eventType === 'attempt-reserved',
  );
  if (firstReservation !== undefined) {
    const materializedRecordedAt = registryVerification.registryMaterializedRecordedAt;
    const materializedAt = typeof materializedRecordedAt === 'string'
      && /^\d{4}-\d{2}-\d{2}T/.test(materializedRecordedAt)
      ? Date.parse(materializedRecordedAt)
      : Number.NaN;
    const firstReservationAt = Date.parse(firstReservation.recordedAt);
    if (!Number.isFinite(materializedAt)
      || !Number.isFinite(firstReservationAt)
      || materializedAt > firstReservationAt) {
      reject(
        'canonical registry materialization timestamp is missing, malformed, or later than the first attempt reservation.',
      );
    }
  }
  if (state.pendingReservation !== null) reject('the final reserved attempt is not finalized.');
  const inventory = await listSeriesAttemptDirectories(seriesDirectory);
  const declaredDirectories = state.attempts.map(
    (attempt) => attempt.finalization.attemptDirectory,
  );
  if (!sameCanonical(inventory, declaredDirectories)) {
    reject('series directory inventory does not exactly match every ledger attempt.');
  }

  const inspections = [];
  for (const attempt of state.attempts) {
    inspections.push(await inspectCandidateAttempt(
      seriesDirectory,
      attempt,
      { runVerifier },
    ));
  }
  const validAttempts = inspections.filter(
    (attempt) => attempt.classification === 'valid-candidate',
  );
  const nonretryableFailures = inspections.filter(
    (attempt) => attempt.classification === 'implementation/evidence-failure-nonretryable',
  );
  const candidates = [];
  for (const attempt of validAttempts) {
    candidates.push(await evaluateValidCandidate(attempt, summarizeRun));
  }
  candidates.sort((left, right) => left.matrixOrdinal - right.matrixOrdinal);
  const seriesSourceBound = candidates.every(
    (candidate) => sameCanonical(candidate.identity.source, state.sourceIdentity),
  );
  if (!seriesSourceBound) {
    reject('candidate run source provenance differs from its pre-timing ledger reservation.');
  }
  const pairIdentity = verifyPairIdentity(candidates);
  const exactlyTwoValidCompletedMatrices = validAttempts.length === 2
    && candidates.length === 2
    && pairIdentity.pass;
  const secondMatrixPresentRegardlessOfFirst = validAttempts.some(
    (attempt) => attempt.matrixOrdinal === 2,
  );
  const noLaterSubstitutes = state.completedMatrixCount === validAttempts.length
    && state.completedMatrixCount <= 2
    && inspections.every((attempt) => attempt.matrixOrdinal <= 2);
  const pairEligibility = {
    canonicalRegistryAndAnchorBound: anchorVerification.verified === true,
    exactlyTwoValidCompletedMatrices,
    secondMatrixPresentRegardlessOfFirst,
    sameFrozenIdentity: pairIdentity.pass,
    everyCandidateBoundToSeriesSource: seriesSourceBound,
    noLaterSubstitutes,
    sourceSeriesFreeOfNonretryableFailure: nonretryableFailures.length === 0,
  };
  pairEligibility.pass = Object.values(pairEligibility).every((value) => value === true);
  const combinedPass = pairEligibility.pass
    && candidates.every((candidate) => candidate.pass === true);
  return {
    schemaVersion: 1,
    kind: 'first-instance-live-candidate-pair-verification',
    seriesId: state.seriesId,
    ledger: {
      filename: CANDIDATE_LEDGER_FILENAME,
      eventCount: state.events.length,
      finalEventSha256: state.events.at(-1).eventSha256,
      sourceIdentity: state.sourceIdentity,
    },
    registry: {
      experimentId: registryVerification.experimentId,
      studyKey: registryVerification.studyKey,
      filename: registryVerification.registryFilename,
      eventCount: registryVerification.registryEventCount,
      finalEventSha256: registryVerification.registryFinalEventSha256,
      claimEventSha256: registryVerification.registryClaimEventSha256,
      materializedEventSha256:
        registryVerification.registryMaterializedEventSha256,
      seriesOpeningEventSha256: registryVerification.seriesOpeningEventSha256,
      anchorTagName: anchorVerification.tagName,
      anchorMessageSha256: anchorVerification.messageSha256,
      anchorVerified: anchorVerification.verified,
    },
    attemptCount: inspections.length,
    attempts: inspections.map((attempt) => ({
      attemptOrdinal: attempt.attemptOrdinal,
      matrixOrdinal: attempt.matrixOrdinal,
      attemptDirectory: attempt.attemptDirectory,
      classification: attempt.classification,
      reasonCode: attempt.reasonCode,
      classificationEvidence: attempt.classificationEvidence,
    })),
    retryCount: inspections.filter(
      (attempt) => attempt.classification === 'infrastructure-invalid-retryable',
    ).length,
    completedMatrixCount: validAttempts.length,
    validCandidateCount: validAttempts.length,
    nonretryableFailureCount: nonretryableFailures.length,
    pairIdentitySha256: pairIdentity.sha256,
    pairEligibility,
    matrices: candidates.map((candidate) => ({
      matrixOrdinal: candidate.matrixOrdinal,
      attemptOrdinal: candidate.attemptOrdinal,
      runId: candidate.runId,
      numericalPass: candidate.numericalDecision.pass,
      environmentPass: candidate.environmentDecision.pass,
      evidencePass: candidate.overallEvidenceDecision.pass,
      pass: candidate.pass,
      numericalDecision: candidate.numericalDecision,
      environmentDecision: candidate.environmentDecision,
      overallEvidenceDecision: candidate.overallEvidenceDecision,
    })),
    decision: {
      status: combinedPass ? 'confirmed-first-device-live' : 'confirmation-not-met',
      pass: combinedPass,
      rule: 'both matrices independently pass all numerical, environment, and evidence gates',
    },
  };
}

async function main() {
  const arguments_ = process.argv.slice(2);
  if (arguments_.length !== 1) {
    throw new Error(
      'Usage: node analysis/verify-live-first-instance-candidate-pair.mjs '
        + '<candidate-series-directory-or-candidate-attempts.jsonl>',
    );
  }
  const result = await verifyCandidateSeries(arguments_[0]);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

const invokedUrl = process.argv[1] === undefined
  ? null
  : pathToFileURL(path.resolve(process.argv[1])).href;

if (import.meta.url === invokedUrl) {
  main().catch((error) => {
    process.stderr.write(
      `verify-live-first-instance-candidate-pair: ${
        error instanceof Error ? error.message : String(error)
      }\n`,
    );
    process.exitCode = 1;
  });
}
