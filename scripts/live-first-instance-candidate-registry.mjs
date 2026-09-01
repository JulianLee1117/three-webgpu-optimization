import { execFile } from 'node:child_process';
import {
  lstat,
  mkdir,
  open,
  readFile,
  readdir,
  unlink,
} from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import {
  CANDIDATE_LEDGER_FILENAME,
  CANDIDATE_SERIES_LOCK_FILENAME,
  canonicalJson,
  readCandidateLedger,
  sha256Canonical,
  validateSourceIdentity,
} from './live-first-instance-candidate-ledger.mjs';

const execFileAsync = promisify(execFile);

export const LIVE_FIRST_INSTANCE_EXPERIMENT_ID =
  'first-instance-live-first-device-v1';
export const CANDIDATE_REGISTRY_FILENAME = 'candidate-series-registry.jsonl';
export const CANDIDATE_REGISTRY_LOCK_FILENAME = '.candidate-series-root.lock';
export const CANDIDATE_REGISTRY_SCHEMA_VERSION = 1;

const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const EVENT_COMMON_KEYS = Object.freeze([
  'schemaVersion',
  'eventIndex',
  'eventType',
  'experimentId',
  'recordedAt',
  'previousEventSha256',
  'eventSha256',
]);
const EVENT_KEYS = Object.freeze({
  'source-claimed': Object.freeze([
    ...EVENT_COMMON_KEYS,
    'studyKey',
    'seriesDirectory',
    'sourceIdentity',
  ]),
  'series-materialized': Object.freeze([
    ...EVENT_COMMON_KEYS,
    'studyKey',
    'seriesDirectory',
    'claimEventSha256',
    'seriesId',
    'seriesOpeningEventSha256',
  ]),
});

function reject(message) {
  throw new Error(`Live first-instance candidate registry rejected: ${message}`);
}

function exactKeys(value, expected, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    reject(`${label} must be an object.`);
  }
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length
    || actual.some((key, index) => key !== wanted[index])) {
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

function sameCanonical(left, right) {
  return sha256Canonical(left) === sha256Canonical(right);
}

function eventDigest(event) {
  const payload = { ...event };
  delete payload.eventSha256;
  return sha256Canonical(payload);
}

export function candidateStudyIdentity(sourceIdentity) {
  const source = validateSourceIdentity(sourceIdentity, 'candidate registry source identity');
  return {
    experimentId: LIVE_FIRST_INSTANCE_EXPERIMENT_ID,
    tree: source.tree,
    trackedFilesSha256: source.trackedFilesSha256,
    packageLockSha256: source.packageLockSha256,
  };
}

export function candidateStudyKey(sourceIdentity) {
  return sha256Canonical(candidateStudyIdentity(sourceIdentity));
}

export function candidateSeriesBasename(sourceIdentity) {
  return `first-device-live-${candidateStudyKey(sourceIdentity).slice(0, 16)}`;
}

export function candidateSeriesDirectory(candidateRoot, sourceIdentity) {
  return path.join(path.resolve(candidateRoot), candidateSeriesBasename(sourceIdentity));
}

export function validateCandidateSeriesLocation(
  candidateRoot,
  seriesDirectory,
  sourceIdentity,
) {
  const root = path.resolve(candidateRoot);
  const series = path.resolve(seriesDirectory);
  const expected = candidateSeriesDirectory(root, sourceIdentity);
  if (series !== expected || path.dirname(series) !== root) {
    reject(
      `series path must be the canonical ${candidateSeriesBasename(sourceIdentity)} directory.`,
    );
  }
  return series;
}

function validateEventShape(event, index) {
  if (event === null || typeof event !== 'object' || Array.isArray(event)) {
    reject(`registry event ${index} must be an object.`);
  }
  const expected = EVENT_KEYS[event.eventType];
  if (!expected) reject(`registry event ${index} has an unsupported eventType.`);
  exactKeys(event, expected, `registry event ${index}`);
  if (event.schemaVersion !== CANDIDATE_REGISTRY_SCHEMA_VERSION
    || event.eventIndex !== index
    || event.experimentId !== LIVE_FIRST_INSTANCE_EXPERIMENT_ID) {
    reject(`registry event ${index} has an unsupported identity or index.`);
  }
  isoTimestamp(event.recordedAt, `registry event ${index}.recordedAt`);
  if (index === 0) {
    if (event.previousEventSha256 !== null) {
      reject('the opening registry event must have no predecessor.');
    }
  } else {
    sha256(event.previousEventSha256, `registry event ${index}.previousEventSha256`);
  }
  sha256(event.eventSha256, `registry event ${index}.eventSha256`);
  if (event.eventSha256 !== eventDigest(event)) {
    reject(`registry event ${index} digest differs from its bytes.`);
  }
}

export function parseCandidateRegistryText(text) {
  if (typeof text !== 'string' || text === '') reject('the root registry is empty.');
  if (!text.endsWith('\n')) reject('the root registry has an incomplete final record.');
  const lines = text.slice(0, -1).split('\n');
  if (lines.some((line) => line.trim() === '')) {
    reject('the root registry contains a blank record.');
  }
  const events = lines.map((line, index) => {
    try {
      return JSON.parse(line);
    } catch {
      reject(`registry event ${index} is not valid JSON.`);
    }
  });
  const claims = [];
  const byStudyKey = new Map();
  const byDirectory = new Map();
  let lastTimestamp = -Infinity;
  for (const [index, event] of events.entries()) {
    validateEventShape(event, index);
    if (index > 0 && event.previousEventSha256 !== events[index - 1].eventSha256) {
      reject(`registry event ${index} does not extend its predecessor.`);
    }
    const timestamp = Date.parse(event.recordedAt);
    if (timestamp < lastTimestamp) reject(`registry event ${index} moves backward in time.`);
    lastTimestamp = timestamp;
    sha256(event.studyKey, `registry event ${index}.studyKey`);
    nonemptyString(event.seriesDirectory, `registry event ${index}.seriesDirectory`);

    if (event.eventType === 'source-claimed') {
      const sourceIdentity = validateSourceIdentity(
        event.sourceIdentity,
        `registry event ${index}.sourceIdentity`,
      );
      const expectedKey = candidateStudyKey(sourceIdentity);
      const expectedDirectory = candidateSeriesBasename(sourceIdentity);
      if (event.studyKey !== expectedKey || event.seriesDirectory !== expectedDirectory) {
        reject(`registry event ${index} does not use its derived study key and directory.`);
      }
      if (byStudyKey.has(event.studyKey) || byDirectory.has(event.seriesDirectory)) {
        reject(`registry event ${index} duplicates a claimed study or directory.`);
      }
      const claim = { claim: event, materialization: null };
      claims.push(claim);
      byStudyKey.set(event.studyKey, claim);
      byDirectory.set(event.seriesDirectory, claim);
      continue;
    }

    const claim = byStudyKey.get(event.studyKey);
    if (!claim
      || claim.claim.seriesDirectory !== event.seriesDirectory
      || claim.materialization !== null) {
      reject(`registry event ${index} does not materialize one pending claim.`);
    }
    sha256(event.claimEventSha256, `registry event ${index}.claimEventSha256`);
    sha256(
      event.seriesOpeningEventSha256,
      `registry event ${index}.seriesOpeningEventSha256`,
    );
    nonemptyString(event.seriesId, `registry event ${index}.seriesId`);
    if (event.claimEventSha256 !== claim.claim.eventSha256) {
      reject(`registry event ${index} is not bound to its claim.`);
    }
    claim.materialization = event;
  }
  return {
    events,
    claims,
    byStudyKey,
    finalEventSha256: events.at(-1).eventSha256,
  };
}

export async function readCandidateRegistry(registryPath) {
  return parseCandidateRegistryText(await readFile(registryPath, 'utf8'));
}

function buildEvent(priorEvents, payload) {
  const event = {
    schemaVersion: CANDIDATE_REGISTRY_SCHEMA_VERSION,
    eventIndex: priorEvents.length,
    ...payload,
    previousEventSha256: priorEvents.at(-1)?.eventSha256 ?? null,
  };
  event.eventSha256 = eventDigest(event);
  return event;
}

async function appendRegistryEvent(registryPath, payload) {
  let priorEvents = [];
  try {
    priorEvents = (await readCandidateRegistry(registryPath)).events;
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
  const event = buildEvent(priorEvents, payload);
  const handle = await open(registryPath, priorEvents.length === 0 ? 'wx' : 'a');
  try {
    await handle.writeFile(`${JSON.stringify(event)}\n`, 'utf8');
    await handle.sync();
  } finally {
    await handle.close();
  }
  await readCandidateRegistry(registryPath);
  return event;
}

export async function claimCandidateStudy(
  candidateRoot,
  sourceIdentity,
  now = new Date(),
) {
  const root = path.resolve(candidateRoot);
  const source = validateSourceIdentity(sourceIdentity, 'candidate claim source identity');
  const registryPath = path.join(root, CANDIDATE_REGISTRY_FILENAME);
  const studyKey = candidateStudyKey(source);
  let state = null;
  try {
    state = await readCandidateRegistry(registryPath);
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
  await verifyCandidateRegistryInitializationInventory(root, state, studyKey);
  const existing = state?.byStudyKey.get(studyKey) ?? null;
  if (existing !== null) {
    if (!sameCanonical(existing.claim.sourceIdentity, source)) {
      reject(
        'the canonical tracked study is already claimed with a different commit or installed-dependency closure.',
      );
    }
    return { registryPath, state, record: existing, created: false };
  }
  if (state?.claims.some((claim) => claim.materialization === null)) {
    reject('another root-registry claim is pending materialization.');
  }
  const event = await appendRegistryEvent(registryPath, {
    eventType: 'source-claimed',
    experimentId: LIVE_FIRST_INSTANCE_EXPERIMENT_ID,
    recordedAt: now.toISOString(),
    studyKey,
    seriesDirectory: candidateSeriesBasename(source),
    sourceIdentity: source,
  });
  state = await readCandidateRegistry(registryPath);
  return {
    registryPath,
    state,
    record: state.byStudyKey.get(studyKey),
    created: true,
    event,
  };
}

export async function materializeCandidateStudy(
  registryPath,
  claimEvent,
  seriesState,
  now = new Date(),
) {
  let state = await readCandidateRegistry(registryPath);
  const record = state.byStudyKey.get(claimEvent.studyKey);
  if (!record || record.claim.eventSha256 !== claimEvent.eventSha256) {
    reject('series materialization does not match the root claim.');
  }
  const opening = seriesState?.events?.[0];
  if (opening?.eventType !== 'series-opened'
    || seriesState.seriesId !== opening.seriesId
    || !sameCanonical(seriesState.sourceIdentity, claimEvent.sourceIdentity)) {
    reject('series opening does not match the claimed source identity.');
  }
  if (record.materialization !== null) {
    const existing = record.materialization;
    if (existing.seriesId !== seriesState.seriesId
      || existing.seriesOpeningEventSha256 !== opening.eventSha256) {
      reject('the materialized series differs from its registry record.');
    }
    return { state, record, created: false };
  }
  await appendRegistryEvent(registryPath, {
    eventType: 'series-materialized',
    experimentId: LIVE_FIRST_INSTANCE_EXPERIMENT_ID,
    recordedAt: now.toISOString(),
    studyKey: claimEvent.studyKey,
    seriesDirectory: claimEvent.seriesDirectory,
    claimEventSha256: claimEvent.eventSha256,
    seriesId: seriesState.seriesId,
    seriesOpeningEventSha256: opening.eventSha256,
  });
  state = await readCandidateRegistry(registryPath);
  return { state, record: state.byStudyKey.get(claimEvent.studyKey), created: true };
}

export function candidateRegistryAnchor(record) {
  if (!record?.claim || !record?.materialization) {
    reject('a registry anchor requires one materialized claim.');
  }
  const { claim, materialization } = record;
  const tagName = `first-instance-live-candidate-${claim.studyKey.slice(0, 16)}`;
  const payload = {
    schemaVersion: 1,
    kind: 'first-instance-live-candidate-registry-anchor',
    experimentId: LIVE_FIRST_INSTANCE_EXPERIMENT_ID,
    studyKey: claim.studyKey,
    seriesDirectory: claim.seriesDirectory,
    seriesId: materialization.seriesId,
    sourceCommit: claim.sourceIdentity.commit,
    sourceTree: claim.sourceIdentity.tree,
    registryClaimEventSha256: claim.eventSha256,
    seriesOpeningEventSha256: materialization.seriesOpeningEventSha256,
    registryMaterializedEventSha256: materialization.eventSha256,
  };
  const message = canonicalJson(payload);
  return {
    schemaVersion: 1,
    kind: 'first-instance-live-candidate-registry-annotated-tag',
    tagName,
    targetCommit: claim.sourceIdentity.commit,
    message,
    messageSha256: sha256Canonical(payload),
    payload,
  };
}

async function regularDirectory(filename, label) {
  const stats = await lstat(filename);
  if (!stats.isDirectory() || stats.isSymbolicLink()) {
    reject(`${label} must be a non-symbolic-link directory.`);
  }
}

export async function verifyCandidateRegistryInitializationInventory(
  candidateRoot,
  state,
  currentStudyKey,
) {
  const root = path.resolve(candidateRoot);
  await regularDirectory(root, 'candidate-series root');
  const entries = await readdir(root, { withFileTypes: true });
  const allowed = new Set([CANDIDATE_REGISTRY_LOCK_FILENAME]);
  if (state !== null) allowed.add(CANDIDATE_REGISTRY_FILENAME);
  for (const record of state?.claims ?? []) {
    if (record.materialization !== null) {
      allowed.add(record.claim.seriesDirectory);
    } else if (record.claim.studyKey === currentStudyKey) {
      const pending = entries.find((entry) => entry.name === record.claim.seriesDirectory);
      if (pending) allowed.add(record.claim.seriesDirectory);
    } else {
      reject('another root-registry claim is pending materialization.');
    }
  }
  if (entries.length !== allowed.size
    || entries.some((entry) => !allowed.has(entry.name))) {
    reject('candidate-series root contains an undeclared entry during initialization.');
  }
  for (const entry of entries) {
    if (entry.name === CANDIDATE_REGISTRY_LOCK_FILENAME
      || entry.name === CANDIDATE_REGISTRY_FILENAME) {
      if (!entry.isFile()) reject(`candidate root entry ${entry.name} is not a regular file.`);
    } else if (!entry.isDirectory()) {
      reject(`candidate root entry ${entry.name} is not a series directory.`);
    }
    const stats = await lstat(path.join(root, entry.name));
    if (stats.isSymbolicLink()) reject(`candidate root entry ${entry.name} is a symlink.`);
  }
}

export async function verifyCandidateStudyRegistry(
  candidateRoot,
  seriesDirectory,
  { allowRootLock = false, allowSeriesLockDirectory = null } = {},
) {
  const root = path.resolve(candidateRoot);
  const series = path.resolve(seriesDirectory);
  await regularDirectory(root, 'candidate-series root');
  const entries = await readdir(root, { withFileTypes: true });
  if (!allowRootLock && entries.some((entry) => entry.name === CANDIDATE_REGISTRY_LOCK_FILENAME)) {
    reject('candidate-series root is locked during verification.');
  }
  const registryPath = path.join(root, CANDIDATE_REGISTRY_FILENAME);
  const registryStats = await lstat(registryPath);
  if (!registryStats.isFile() || registryStats.isSymbolicLink()) {
    reject('candidate root registry must be a regular non-symbolic-link file.');
  }
  if (allowRootLock) {
    const lockStats = await lstat(path.join(root, CANDIDATE_REGISTRY_LOCK_FILENAME));
    if (!lockStats.isFile() || lockStats.isSymbolicLink()) {
      reject('candidate root lock must be a regular non-symbolic-link file.');
    }
  }
  const state = await readCandidateRegistry(registryPath);
  if (state.claims.some((claim) => claim.materialization === null)) {
    reject('candidate-series registry contains an unmaterialized claim.');
  }
  const expectedNames = new Set([
    CANDIDATE_REGISTRY_FILENAME,
    ...state.claims.map((claim) => claim.claim.seriesDirectory),
    ...(allowRootLock ? [CANDIDATE_REGISTRY_LOCK_FILENAME] : []),
  ]);
  if (entries.length !== expectedNames.size
    || entries.some((entry) => !expectedNames.has(entry.name))) {
    reject('candidate-series root inventory differs from the registry.');
  }

  let selected = null;
  for (const record of state.claims) {
    const candidateDirectory = path.join(root, record.claim.seriesDirectory);
    const entry = entries.find((candidate) => candidate.name === record.claim.seriesDirectory);
    if (!entry?.isDirectory()) reject('a registered candidate series is not a directory.');
    await regularDirectory(candidateDirectory, `series ${record.claim.seriesDirectory}`);
    const seriesEntries = await readdir(candidateDirectory, { withFileTypes: true });
    const seriesLock = seriesEntries.find(
      (candidate) => candidate.name === CANDIDATE_SERIES_LOCK_FILENAME,
    );
    if (seriesLock) {
      const allowedLockDirectory = allowSeriesLockDirectory === null
        ? null
        : path.resolve(allowSeriesLockDirectory);
      if (candidateDirectory !== allowedLockDirectory
        || !seriesLock.isFile()
        || (await lstat(path.join(candidateDirectory, seriesLock.name))).isSymbolicLink()) {
        reject(`series ${record.claim.seriesDirectory} is locked during verification.`);
      }
    }
    const ledgerState = await readCandidateLedger(
      path.join(candidateDirectory, CANDIDATE_LEDGER_FILENAME),
    );
    const declaredSeriesEntries = new Set([
      CANDIDATE_LEDGER_FILENAME,
      ...ledgerState.events
        .filter((event) => event.eventType === 'attempt-reserved')
        .map((event) => event.attemptDirectory),
      ...(seriesLock ? [CANDIDATE_SERIES_LOCK_FILENAME] : []),
    ]);
    if (seriesEntries.length !== declaredSeriesEntries.size
      || seriesEntries.some((candidate) => !declaredSeriesEntries.has(candidate.name))) {
      reject(`series ${record.claim.seriesDirectory} inventory differs from its ledger.`);
    }
    for (const seriesEntry of seriesEntries) {
      const entryPath = path.join(candidateDirectory, seriesEntry.name);
      const entryStats = await lstat(entryPath);
      if (entryStats.isSymbolicLink()) {
        reject(`series ${record.claim.seriesDirectory} contains a symbolic link.`);
      }
      if (seriesEntry.name === CANDIDATE_LEDGER_FILENAME
        || seriesEntry.name === CANDIDATE_SERIES_LOCK_FILENAME) {
        if (!entryStats.isFile()) {
          reject(`series ${record.claim.seriesDirectory} contains a non-file control entry.`);
        }
      } else if (!entryStats.isDirectory()) {
        reject(`series ${record.claim.seriesDirectory} contains a non-directory attempt.`);
      }
    }
    if (!sameCanonical(ledgerState.sourceIdentity, record.claim.sourceIdentity)
      || ledgerState.seriesId !== record.materialization.seriesId
      || ledgerState.events[0].eventSha256
        !== record.materialization.seriesOpeningEventSha256) {
      reject(`series ${record.claim.seriesDirectory} differs from its materialization.`);
    }
    const claimTimestamp = Date.parse(record.claim.recordedAt);
    const openingTimestamp = Date.parse(ledgerState.events[0].recordedAt);
    const materializedTimestamp = Date.parse(record.materialization.recordedAt);
    if (openingTimestamp < claimTimestamp || materializedTimestamp < openingTimestamp) {
      reject(`series ${record.claim.seriesDirectory} chronology crosses its registry events.`);
    }
    const firstReservation = ledgerState.events.find(
      (event) => event.eventType === 'attempt-reserved',
    );
    if (firstReservation !== undefined
      && materializedTimestamp > Date.parse(firstReservation.recordedAt)) {
      reject(
        `series ${record.claim.seriesDirectory} registry materialization occurs after its first attempt reservation.`,
      );
    }
    validateCandidateSeriesLocation(root, candidateDirectory, record.claim.sourceIdentity);
    if (candidateDirectory === series) selected = { record, ledgerState };
  }
  if (selected === null) reject('requested series is not the one declared by the root registry.');
  return {
    schemaVersion: 1,
    kind: 'first-instance-live-candidate-registry-verification',
    experimentId: LIVE_FIRST_INSTANCE_EXPERIMENT_ID,
    studyKey: selected.record.claim.studyKey,
    seriesDirectory: selected.record.claim.seriesDirectory,
    sourceIdentity: selected.record.claim.sourceIdentity,
    seriesId: selected.record.materialization.seriesId,
    registryFilename: CANDIDATE_REGISTRY_FILENAME,
    registryEventCount: state.events.length,
    registryFinalEventSha256: state.finalEventSha256,
    registryClaimEventSha256: selected.record.claim.eventSha256,
    registryMaterializedEventSha256: selected.record.materialization.eventSha256,
    registryMaterializedRecordedAt: selected.record.materialization.recordedAt,
    seriesOpeningEventSha256:
      selected.record.materialization.seriesOpeningEventSha256,
    anchor: candidateRegistryAnchor(selected.record),
  };
}

function parseAnnotatedTag(contents, expectedName) {
  const separator = contents.indexOf('\n\n');
  if (separator < 0 || contents.includes('\r')) reject('candidate anchor tag object is malformed.');
  const headers = Object.fromEntries(contents.slice(0, separator).split('\n').map((line) => {
    const split = line.indexOf(' ');
    return [line.slice(0, split), line.slice(split + 1)];
  }));
  let message = contents.slice(separator + 2);
  if (!message.endsWith('\n')) reject('candidate anchor tag message lacks its final newline.');
  message = message.slice(0, -1);
  if (headers.type !== 'commit' || headers.tag !== expectedName) {
    reject('candidate anchor is not the expected annotated commit tag.');
  }
  return { headers, message };
}

async function git(projectRoot, args) {
  const { stdout } = await execFileAsync('git', args, {
    cwd: projectRoot,
    encoding: 'utf8',
    maxBuffer: 16 * 1024 * 1024,
    windowsHide: true,
  });
  return stdout;
}

export async function verifyCandidateRegistryAnchorTag(
  projectRoot,
  registryVerification,
) {
  const expected = registryVerification.anchor;
  let objectType;
  let resolvedCommit;
  let tagObject;
  try {
    [objectType, resolvedCommit, tagObject] = await Promise.all([
      git(projectRoot, ['cat-file', '-t', `refs/tags/${expected.tagName}`]),
      git(projectRoot, ['rev-parse', '--verify', `refs/tags/${expected.tagName}^{commit}`]),
      git(projectRoot, ['cat-file', '-p', `refs/tags/${expected.tagName}`]),
    ]);
  } catch {
    reject(`required annotated anchor tag ${expected.tagName} is missing.`);
  }
  if (objectType.trim() !== 'tag' || resolvedCommit.trim() !== expected.targetCommit) {
    reject('candidate anchor tag has the wrong object type or target commit.');
  }
  const parsed = parseAnnotatedTag(tagObject, expected.tagName);
  if (parsed.headers.object !== expected.targetCommit || parsed.message !== expected.message) {
    reject('candidate anchor tag does not contain the exact registry commitment.');
  }
  return {
    ...expected,
    verified: true,
  };
}

export async function acquireCandidateRegistryLock(candidateRoot) {
  const root = path.resolve(candidateRoot);
  await mkdir(root, { recursive: true });
  const lockPath = path.join(root, CANDIDATE_REGISTRY_LOCK_FILENAME);
  let handle;
  try {
    handle = await open(lockPath, 'wx');
  } catch (error) {
    if (error?.code !== 'EEXIST') throw error;
    let owner;
    try {
      owner = JSON.parse(await readFile(lockPath, 'utf8'));
    } catch {
      reject('candidate root lock exists without readable ownership provenance.');
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
    if (ownerAlive) reject(`candidate registry is locked by live process ${owner.pid}.`);
    await unlink(lockPath);
    handle = await open(lockPath, 'wx');
  }
  await handle.writeFile(`${JSON.stringify({
    pid: process.pid,
    createdAt: new Date().toISOString(),
  })}\n`);
  await handle.sync();
  return {
    lockPath,
    async release() {
      await handle.close();
      await unlink(lockPath);
    },
  };
}
