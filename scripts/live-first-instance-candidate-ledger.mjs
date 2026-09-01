import { createHash, randomUUID } from 'node:crypto';
import {
  lstat,
  mkdir,
  open,
  readFile,
  readdir,
  unlink,
} from 'node:fs/promises';
import path from 'node:path';

export const CANDIDATE_LEDGER_FILENAME = 'candidate-attempts.jsonl';
export const CANDIDATE_SERIES_LOCK_FILENAME = '.candidate-series.lock';
export const CANDIDATE_LEDGER_SCHEMA_VERSION = 1;
export const CANDIDATE_CLASSIFICATIONS = Object.freeze([
  'valid-candidate',
  'infrastructure-invalid-retryable',
  'implementation/evidence-failure-nonretryable',
]);
export const CANDIDATE_REASON_CODES = Object.freeze({
  'valid-candidate': Object.freeze(['completed-candidate']),
  'infrastructure-invalid-retryable': Object.freeze([
    'browser-or-device-loss',
    'telemetry-collector-failure',
    'run-artifact-incompleteness-or-corruption',
  ]),
  'implementation/evidence-failure-nonretryable': Object.freeze([
    'implementation-or-evidence-failure',
  ]),
});
export const CANDIDATE_ARTIFACT_PERSISTENCE_FAILURE_PREFIX =
  'FIRST_INSTANCE_LIVE_ARTIFACT_PERSISTENCE_FAILURE ';
export const CANDIDATE_ARTIFACT_PERSISTENCE_ERROR_CODES = Object.freeze([
  'EACCES',
  'EBUSY',
  'EDQUOT',
  'EFBIG',
  'EIO',
  'EMFILE',
  'ENFILE',
  'ENOSPC',
  'EPERM',
  'EROFS',
]);

const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const GIT_OBJECT_PATTERN = /^[0-9a-f]{40,64}$/;
const EVENT_COMMON_KEYS = Object.freeze([
  'schemaVersion',
  'eventIndex',
  'eventType',
  'seriesId',
  'recordedAt',
  'previousEventSha256',
  'eventSha256',
]);
const EVENT_KEYS = Object.freeze({
  'series-opened': Object.freeze([...EVENT_COMMON_KEYS, 'sourceIdentity']),
  'attempt-reserved': Object.freeze([
    ...EVENT_COMMON_KEYS,
    'attemptOrdinal',
    'matrixOrdinal',
    'attemptDirectory',
    'sourceIdentity',
  ]),
  'attempt-finalized': Object.freeze([
    ...EVENT_COMMON_KEYS,
    'attemptOrdinal',
    'matrixOrdinal',
    'attemptDirectory',
    'reservationEventSha256',
    'classification',
    'reasonCode',
    'runDirectory',
    'execution',
    'contentCommitment',
    'classificationEvidence',
  ]),
});
const SOURCE_IDENTITY_KEYS = Object.freeze([
  'commit',
  'tree',
  'trackedFilesSha256',
  'packageLockSha256',
  'executionDependencyClosureSha256',
  'executionDependencyFileCount',
  'executionDependencyTotalBytes',
]);

function reject(message) {
  throw new Error(`Live first-instance candidate ledger rejected: ${message}`);
}

function exactKeys(value, expected, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    reject(`${label} must be an object.`);
  }
  const keys = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  if (keys.length !== sortedExpected.length
    || keys.some((key, index) => key !== sortedExpected[index])) {
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

function safeInteger(value, label, { minimum = 0 } = {}) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    reject(`${label} must be an integer no smaller than ${minimum}.`);
  }
  return value;
}

function sha256(value, label) {
  if (typeof value !== 'string' || !SHA256_PATTERN.test(value)) {
    reject(`${label} must be a lowercase SHA-256 digest.`);
  }
  return value;
}

function isoTimestamp(value, label) {
  nonemptyString(value, label);
  if (!/^\d{4}-\d{2}-\d{2}T/.test(value) || !Number.isFinite(Date.parse(value))) {
    reject(`${label} must be an ISO timestamp.`);
  }
  return value;
}

function normalizeCanonical(value) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) reject('canonical JSON cannot contain a non-finite number.');
    return value;
  }
  if (Array.isArray(value)) return value.map(normalizeCanonical);
  if (typeof value !== 'object') reject('canonical JSON contains an unsupported value.');
  if (Object.getPrototypeOf(value) !== Object.prototype
    && Object.getPrototypeOf(value) !== null) {
    reject('canonical JSON objects must use the ordinary object prototype.');
  }
  return Object.fromEntries(
    Object.keys(value).sort().map((key) => {
      if (value[key] === undefined) reject('canonical JSON cannot contain undefined.');
      return [key, normalizeCanonical(value[key])];
    }),
  );
}

export function canonicalJson(value) {
  return JSON.stringify(normalizeCanonical(value));
}

export function sha256Canonical(value) {
  return createHash('sha256').update(canonicalJson(value)).digest('hex');
}

function eventDigest(event) {
  const withoutDigest = { ...event };
  delete withoutDigest.eventSha256;
  return sha256Canonical(withoutDigest);
}

export function sourceIdentityFromProvenance(provenance, executionDependencyClosure) {
  if (provenance?.status !== 'available'
    || provenance.captureStable !== true
    || provenance.dirty !== false
    || provenance.stagedChanges !== 0
    || provenance.unstagedChanges !== 0
    || provenance.untrackedFiles !== 0
    || provenance.packageLockTracked !== true) {
    reject('candidate reservation requires clean, stable, available source provenance.');
  }
  const result = {
    commit: provenance.commit,
    tree: provenance.tree,
    trackedFilesSha256: provenance.trackedFilesSha256,
    packageLockSha256: provenance.packageLockSha256,
    executionDependencyClosureSha256: executionDependencyClosure?.sha256,
    executionDependencyFileCount: executionDependencyClosure?.fileCount,
    executionDependencyTotalBytes: executionDependencyClosure?.totalBytes,
  };
  validateSourceIdentity(result, 'candidate source identity');
  return result;
}

export function validateSourceIdentity(value, label = 'source identity') {
  const identity = exactKeys(value, SOURCE_IDENTITY_KEYS, label);
  for (const field of ['commit', 'tree']) {
    if (typeof identity[field] !== 'string' || !GIT_OBJECT_PATTERN.test(identity[field])) {
      reject(`${label}.${field} must be a Git object ID.`);
    }
  }
  sha256(identity.trackedFilesSha256, `${label}.trackedFilesSha256`);
  sha256(identity.packageLockSha256, `${label}.packageLockSha256`);
  sha256(
    identity.executionDependencyClosureSha256,
    `${label}.executionDependencyClosureSha256`,
  );
  safeInteger(
    identity.executionDependencyFileCount,
    `${label}.executionDependencyFileCount`,
    { minimum: 1 },
  );
  safeInteger(
    identity.executionDependencyTotalBytes,
    `${label}.executionDependencyTotalBytes`,
    { minimum: 1 },
  );
  return identity;
}

function identitiesMatch(left, right) {
  return SOURCE_IDENTITY_KEYS.every((key) => left[key] === right[key]);
}

function validateRelativeBasename(value, expected, label) {
  nonemptyString(value, label);
  if (value !== expected || path.basename(value) !== value || value.includes('/') || value.includes('\\')) {
    reject(`${label} must be exactly ${JSON.stringify(expected)}.`);
  }
  return value;
}

function validateRelativeRunDirectory(value, attemptDirectory, label) {
  if (value === null) return null;
  nonemptyString(value, label);
  const normalized = value.replaceAll('\\', '/');
  const prefix = `${attemptDirectory}/runs/`;
  if (normalized !== value
    || !normalized.startsWith(prefix)
    || normalized.slice(prefix.length).includes('/')
    || path.posix.normalize(normalized) !== normalized) {
    reject(`${label} must name one direct runner output under ${prefix}.`);
  }
  return value;
}

function validateExecution(value, label) {
  const execution = exactKeys(value, ['exitCode', 'signal'], label);
  if (execution.exitCode !== null) safeInteger(execution.exitCode, `${label}.exitCode`);
  if (execution.signal !== null) nonemptyString(execution.signal, `${label}.signal`);
  if (execution.exitCode === null && execution.signal === null) {
    reject(`${label} must record an exit code or signal.`);
  }
  return execution;
}

function validateContentCommitment(value, label) {
  const commitment = exactKeys(
    value,
    ['format', 'fileCount', 'totalBytes', 'sha256'],
    label,
  );
  if (commitment.format !== 'sorted-relative-path-size-sha256-v1') {
    reject(`${label}.format is unsupported.`);
  }
  safeInteger(commitment.fileCount, `${label}.fileCount`);
  safeInteger(commitment.totalBytes, `${label}.totalBytes`);
  sha256(commitment.sha256, `${label}.sha256`);
  return commitment;
}

function validateClassificationEvidence(value, label) {
  const evidence = exactKeys(
    value,
    [
      'artifactManifestStatus',
      'artifactInfrastructureFailureObserved',
      'attemptLayoutExact',
      'browserOrDeviceLossObserved',
      'candidateReservationBindingStatus',
      'childLifecycleClosed',
      'runVerificationStatus',
      'runDirectoryCount',
      'sourceIdentityStable',
      'telemetryCollectorFailureObserved',
    ],
    label,
  );
  const statuses = ['consistent', 'incomplete-or-corrupt', 'unavailable'];
  if (!statuses.includes(evidence.artifactManifestStatus)) {
    reject(`${label}.artifactManifestStatus is unsupported.`);
  }
  if (!['passed', 'failed'].includes(evidence.runVerificationStatus)) {
    reject(`${label}.runVerificationStatus is unsupported.`);
  }
  if (!['matched', 'mismatched', 'unavailable'].includes(
    evidence.candidateReservationBindingStatus,
  )) {
    reject(`${label}.candidateReservationBindingStatus is unsupported.`);
  }
  safeInteger(evidence.runDirectoryCount, `${label}.runDirectoryCount`);
  for (const field of [
    'artifactInfrastructureFailureObserved',
    'attemptLayoutExact',
    'browserOrDeviceLossObserved',
    'childLifecycleClosed',
    'sourceIdentityStable',
    'telemetryCollectorFailureObserved',
  ]) {
    if (typeof evidence[field] !== 'boolean') reject(`${label}.${field} must be boolean.`);
  }
  return evidence;
}

function validateEventShape(event, index) {
  if (event === null || typeof event !== 'object' || Array.isArray(event)) {
    reject(`event ${index} must be an object.`);
  }
  const expectedKeys = EVENT_KEYS[event.eventType];
  if (!expectedKeys) reject(`event ${index} has unsupported eventType.`);
  exactKeys(event, expectedKeys, `event ${index}`);
  if (event.schemaVersion !== CANDIDATE_LEDGER_SCHEMA_VERSION) {
    reject(`event ${index} has unsupported schemaVersion.`);
  }
  if (event.eventIndex !== index) reject(`event ${index} has a non-contiguous eventIndex.`);
  nonemptyString(event.seriesId, `event ${index}.seriesId`);
  isoTimestamp(event.recordedAt, `event ${index}.recordedAt`);
  if (index === 0) {
    if (event.previousEventSha256 !== null) reject('the opening event must have no predecessor.');
  } else {
    sha256(event.previousEventSha256, `event ${index}.previousEventSha256`);
  }
  sha256(event.eventSha256, `event ${index}.eventSha256`);
  if (event.eventSha256 !== eventDigest(event)) reject(`event ${index} digest differs from its bytes.`);
}

export function parseCandidateLedgerText(text) {
  if (typeof text !== 'string' || text === '') reject('the ledger is empty.');
  if (!text.endsWith('\n')) reject('the ledger has an incomplete final record.');
  const lines = text.slice(0, -1).split('\n');
  if (lines.some((line) => line.trim() === '')) reject('the ledger contains a blank record.');
  const events = lines.map((line, index) => {
    try {
      return JSON.parse(line);
    } catch {
      reject(`event ${index} is not valid JSON.`);
    }
  });
  if (events.length === 0) reject('the ledger contains no events.');

  let seriesId = null;
  let sourceIdentity = null;
  let lastTimestamp = -Infinity;
  let pendingReservation = null;
  let nextAttemptOrdinal = 1;
  let completedMatrixCount = 0;
  let terminalFailure = null;
  const attempts = [];

  for (const [index, event] of events.entries()) {
    validateEventShape(event, index);
    if (index > 0 && event.previousEventSha256 !== events[index - 1].eventSha256) {
      reject(`event ${index} does not extend the preceding event digest.`);
    }
    const timestamp = Date.parse(event.recordedAt);
    if (timestamp < lastTimestamp) reject(`event ${index} moves backward in time.`);
    lastTimestamp = timestamp;

    if (index === 0) {
      if (event.eventType !== 'series-opened') reject('the first event must open the series.');
      seriesId = event.seriesId;
      sourceIdentity = validateSourceIdentity(event.sourceIdentity, 'series source identity');
      continue;
    }
    if (event.seriesId !== seriesId) reject(`event ${index} changes seriesId.`);
    if (event.eventType === 'series-opened') reject('the series may be opened only once.');

    if (event.eventType === 'attempt-reserved') {
      if (pendingReservation !== null) reject('a second attempt was reserved before finalization.');
      if (terminalFailure !== null) {
        reject('an attempt was reserved after a non-retryable implementation/evidence failure.');
      }
      if (completedMatrixCount >= 2) reject('an attempt was reserved after both candidate slots closed.');
      safeInteger(event.attemptOrdinal, `event ${index}.attemptOrdinal`, { minimum: 1 });
      safeInteger(event.matrixOrdinal, `event ${index}.matrixOrdinal`, { minimum: 1 });
      if (event.attemptOrdinal !== nextAttemptOrdinal) reject('attempt ordinals are not contiguous.');
      if (event.matrixOrdinal !== completedMatrixCount + 1) {
        reject('matrixOrdinal does not preserve retry and candidate-slot chronology.');
      }
      validateRelativeBasename(
        event.attemptDirectory,
        `attempt-${String(event.attemptOrdinal).padStart(4, '0')}`,
        `event ${index}.attemptDirectory`,
      );
      const attemptSource = validateSourceIdentity(
        event.sourceIdentity,
        `event ${index}.sourceIdentity`,
      );
      if (!identitiesMatch(attemptSource, sourceIdentity)) {
        reject(`event ${index} changes the frozen source identity.`);
      }
      pendingReservation = event;
      nextAttemptOrdinal += 1;
      continue;
    }

    if (event.eventType === 'attempt-finalized') {
      if (pendingReservation === null) reject('an attempt was finalized without a reservation.');
      for (const field of ['attemptOrdinal', 'matrixOrdinal', 'attemptDirectory']) {
        if (event[field] !== pendingReservation[field]) {
          reject(`event ${index}.${field} differs from its reservation.`);
        }
      }
      if (event.reservationEventSha256 !== pendingReservation.eventSha256) {
        reject(`event ${index} is not bound to its reservation digest.`);
      }
      if (!CANDIDATE_CLASSIFICATIONS.includes(event.classification)) {
        reject(`event ${index} has an unsupported classification.`);
      }
      if (!CANDIDATE_REASON_CODES[event.classification].includes(event.reasonCode)) {
        reject(`event ${index} reasonCode is not permitted for its classification.`);
      }
      validateRelativeRunDirectory(
        event.runDirectory,
        event.attemptDirectory,
        `event ${index}.runDirectory`,
      );
      validateExecution(event.execution, `event ${index}.execution`);
      validateContentCommitment(event.contentCommitment, `event ${index}.contentCommitment`);
      validateClassificationEvidence(
        event.classificationEvidence,
        `event ${index}.classificationEvidence`,
      );
      attempts.push({ reservation: pendingReservation, finalization: event });
      if (event.classification === 'valid-candidate') {
        completedMatrixCount += 1;
      } else if (event.classification === 'implementation/evidence-failure-nonretryable') {
        terminalFailure = event;
      }
      pendingReservation = null;
    }
  }

  return {
    events,
    seriesId,
    sourceIdentity,
    attempts,
    pendingReservation,
    completedMatrixCount,
    terminalFailure,
    nextAttemptOrdinal,
  };
}

export async function readCandidateLedger(ledgerPath) {
  return parseCandidateLedgerText(await readFile(ledgerPath, 'utf8'));
}

function buildEvent(priorEvents, payload) {
  const previous = priorEvents.at(-1) ?? null;
  const event = normalizeCanonical({
    schemaVersion: CANDIDATE_LEDGER_SCHEMA_VERSION,
    eventIndex: priorEvents.length,
    ...payload,
    previousEventSha256: previous?.eventSha256 ?? null,
  });
  event.eventSha256 = eventDigest(event);
  return event;
}

export async function appendCandidateLedgerEvent(ledgerPath, payload) {
  let priorEvents = [];
  try {
    priorEvents = (await readCandidateLedger(ledgerPath)).events;
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
  const event = buildEvent(priorEvents, payload);
  const handle = await open(ledgerPath, priorEvents.length === 0 ? 'wx' : 'a');
  try {
    await handle.writeFile(`${JSON.stringify(event)}\n`, 'utf8');
    await handle.sync();
  } finally {
    await handle.close();
  }
  await readCandidateLedger(ledgerPath);
  return event;
}

export async function openCandidateSeries(seriesDirectory, sourceIdentity, now = new Date()) {
  await mkdir(seriesDirectory, { recursive: true });
  const ledgerPath = path.join(seriesDirectory, CANDIDATE_LEDGER_FILENAME);
  try {
    const state = await readCandidateLedger(ledgerPath);
    if (!identitiesMatch(state.sourceIdentity, sourceIdentity)) {
      reject('the current clean source does not match the frozen series source.');
    }
    return { ledgerPath, state };
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
  const seriesId = randomUUID();
  await appendCandidateLedgerEvent(ledgerPath, {
    eventType: 'series-opened',
    seriesId,
    recordedAt: now.toISOString(),
    sourceIdentity,
  });
  return { ledgerPath, state: await readCandidateLedger(ledgerPath) };
}

export async function reserveCandidateAttempt(
  seriesDirectory,
  ledgerPath,
  state,
  sourceIdentity,
  now = new Date(),
) {
  if (state.pendingReservation !== null) reject('the preceding attempt is not finalized.');
  if (state.terminalFailure !== null) {
    reject('the source series ended with a non-retryable implementation/evidence failure.');
  }
  if (state.completedMatrixCount >= 2) reject('both candidate matrix slots are already closed.');
  if (!identitiesMatch(state.sourceIdentity, sourceIdentity)) {
    reject('the candidate attempt does not use the series frozen source.');
  }
  const attemptOrdinal = state.nextAttemptOrdinal;
  const matrixOrdinal = state.completedMatrixCount + 1;
  const attemptDirectory = `attempt-${String(attemptOrdinal).padStart(4, '0')}`;
  const absoluteAttemptDirectory = path.join(seriesDirectory, attemptDirectory);
  await mkdir(absoluteAttemptDirectory);
  await mkdir(path.join(absoluteAttemptDirectory, 'runs'));
  const event = await appendCandidateLedgerEvent(ledgerPath, {
    eventType: 'attempt-reserved',
    seriesId: state.seriesId,
    recordedAt: now.toISOString(),
    attemptOrdinal,
    matrixOrdinal,
    attemptDirectory,
    sourceIdentity,
  });
  return { event, absoluteAttemptDirectory };
}

async function walkFiles(root, relativeDirectory, records) {
  const absoluteDirectory = path.join(root, relativeDirectory);
  const entries = await readdir(absoluteDirectory, { withFileTypes: true });
  entries.sort((left, right) => left.name.localeCompare(right.name));
  for (const entry of entries) {
    const relativePath = path.posix.join(
      relativeDirectory.replaceAll('\\', '/'),
      entry.name,
    ).replace(/^\//, '');
    const absolutePath = path.join(root, ...relativePath.split('/'));
    const stats = await lstat(absolutePath);
    if (stats.isSymbolicLink()) reject(`attempt content contains symlink ${relativePath}.`);
    if (stats.isDirectory()) {
      await walkFiles(root, relativePath, records);
    } else if (stats.isFile()) {
      const contents = await readFile(absolutePath);
      records.push({
        path: relativePath,
        bytes: contents.length,
        sha256: createHash('sha256').update(contents).digest('hex'),
      });
    } else {
      reject(`attempt content contains unsupported filesystem entry ${relativePath}.`);
    }
  }
}

export async function buildAttemptContentCommitment(attemptDirectory) {
  const records = [];
  await walkFiles(attemptDirectory, '', records);
  records.sort((left, right) => left.path.localeCompare(right.path));
  return {
    format: 'sorted-relative-path-size-sha256-v1',
    fileCount: records.length,
    totalBytes: records.reduce((sum, record) => sum + record.bytes, 0),
    sha256: sha256Canonical({
      format: 'sorted-relative-path-size-sha256-v1',
      files: records,
    }),
  };
}

export async function listSeriesAttemptDirectories(seriesDirectory) {
  const entries = await readdir(seriesDirectory, { withFileTypes: true });
  const attempts = [];
  for (const entry of entries) {
    if (entry.name === CANDIDATE_LEDGER_FILENAME) {
      continue;
    }
    if (!entry.isDirectory() || !/^attempt-\d{4}$/.test(entry.name)) {
      reject(`series root contains unexpected entry ${JSON.stringify(entry.name)}.`);
    }
    attempts.push(entry.name);
  }
  return attempts.sort();
}

export async function acquireCandidateSeriesLock(seriesDirectory) {
  await mkdir(seriesDirectory, { recursive: true });
  const lockPath = path.join(seriesDirectory, CANDIDATE_SERIES_LOCK_FILENAME);
  let handle;
  try {
    handle = await open(lockPath, 'wx');
  } catch (error) {
    if (error?.code !== 'EEXIST') throw error;
    let owner = null;
    try {
      owner = JSON.parse(await readFile(lockPath, 'utf8'));
    } catch {
      reject('the candidate-series lock exists but is not readable provenance.');
    }
    let ownerAlive = false;
    if (Number.isSafeInteger(owner?.pid) && owner.pid > 0) {
      try {
        process.kill(owner.pid, 0);
        ownerAlive = true;
      } catch (probeError) {
        if (probeError?.code === 'EPERM') ownerAlive = true;
        else if (probeError?.code !== 'ESRCH') throw probeError;
      }
    }
    if (ownerAlive) reject(`candidate series is locked by live process ${owner.pid}.`);
    await unlink(lockPath);
    handle = await open(lockPath, 'wx');
  }
  await handle.writeFile(`${JSON.stringify({ pid: process.pid, createdAt: new Date().toISOString() })}\n`);
  await handle.sync();
  return {
    lockPath,
    async release() {
      await handle.close();
      await unlink(lockPath);
    },
  };
}

export function permittedReasonForClassification(classification, reasonCode) {
  return CANDIDATE_REASON_CODES[classification]?.includes(reasonCode) === true;
}

export function candidateSeriesReservationRecord(reservationEvent) {
  return {
    schemaVersion: 1,
    kind: 'first-instance-live-candidate-series-reservation',
    seriesId: reservationEvent.seriesId,
    reservationEventSha256: reservationEvent.eventSha256,
    attemptOrdinal: reservationEvent.attemptOrdinal,
    matrixOrdinal: reservationEvent.matrixOrdinal,
    sourceCommit: reservationEvent.sourceIdentity.commit,
    sourceTree: reservationEvent.sourceIdentity.tree,
    executionDependencyClosureSha256:
      reservationEvent.sourceIdentity.executionDependencyClosureSha256,
  };
}
