import { spawn } from 'node:child_process';
import { createWriteStream } from 'node:fs';
import { readFile, rename, writeFile } from 'node:fs/promises';
import { finished } from 'node:stream/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';
import process from 'node:process';
import {
  acquireCandidateSeriesLock,
  appendCandidateLedgerEvent,
  buildAttemptContentCommitment,
  candidateSeriesReservationRecord,
  openCandidateSeries,
  readCandidateLedger,
  reserveCandidateAttempt,
  sha256Canonical,
  sourceIdentityFromProvenance,
} from './live-first-instance-candidate-ledger.mjs';
import {
  acquireCandidateRegistryLock,
  candidateSeriesDirectory,
  claimCandidateStudy,
  materializeCandidateStudy,
  verifyCandidateRegistryAnchorTag,
  verifyCandidateRegistryInitializationInventory,
  verifyCandidateStudyRegistry,
} from './live-first-instance-candidate-registry.mjs';
import { collectSourceProvenance } from './source-provenance.mjs';
import { collectExecutionDependencyClosure } from './execution-dependency-closure.mjs';
import {
  classifyUnfinalizedCandidateAttempt,
  verifyCandidateSeries,
} from '../analysis/verify-live-first-instance-candidate-pair.mjs';

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CANDIDATE_SERIES_ROOT = path.join(PROJECT_ROOT, 'results', 'candidate-series');

function resolveCandidateSeriesDirectory(seriesInput) {
  const resolved = path.resolve(seriesInput);
  const relative = path.relative(CANDIDATE_SERIES_ROOT, resolved);
  if (relative === ''
    || relative === '..'
    || relative.startsWith(`..${path.sep}`)
    || path.isAbsolute(relative)) {
    throw new Error(
      `Candidate series must be a named directory below ${
        path.relative(PROJECT_ROOT, CANDIDATE_SERIES_ROOT).replaceAll('\\', '/')
      }/.`,
    );
  }
  return resolved;
}

async function collectCandidateSourceIdentity() {
  const [provenance, executionDependencyClosure] = await Promise.all([
    collectSourceProvenance(PROJECT_ROOT),
    collectExecutionDependencyClosure(PROJECT_ROOT),
  ]);
  return sourceIdentityFromProvenance(provenance, executionDependencyClosure);
}

function sameSourceIdentity(left, right) {
  return sha256Canonical(left) === sha256Canonical(right);
}

export async function initializeCurrentCandidateSeries() {
  const sourceBeforeLock = await collectCandidateSourceIdentity();
  const seriesDirectory = candidateSeriesDirectory(
    CANDIDATE_SERIES_ROOT,
    sourceBeforeLock,
  );
  const rootLock = await acquireCandidateRegistryLock(CANDIDATE_SERIES_ROOT);
  try {
    const sourceIdentity = await collectCandidateSourceIdentity();
    if (!sameSourceIdentity(sourceBeforeLock, sourceIdentity)) {
      throw new Error('Candidate source identity changed while acquiring the root registry lock.');
    }
    const claimed = await claimCandidateStudy(
      CANDIDATE_SERIES_ROOT,
      sourceIdentity,
    );
    await verifyCandidateRegistryInitializationInventory(
      CANDIDATE_SERIES_ROOT,
      claimed.state,
      claimed.record.claim.studyKey,
    );
    if (claimed.record.materialization === null) {
      const opened = await openCandidateSeries(seriesDirectory, sourceIdentity);
      const materialized = await materializeCandidateStudy(
        claimed.registryPath,
        claimed.record.claim,
        opened.state,
      );
      await verifyCandidateRegistryInitializationInventory(
        CANDIDATE_SERIES_ROOT,
        materialized.state,
        claimed.record.claim.studyKey,
      );
    }
    const registry = await verifyCandidateStudyRegistry(
      CANDIDATE_SERIES_ROOT,
      seriesDirectory,
      { allowRootLock: true },
    );
    const anchor = registry.anchor;
    return {
      schemaVersion: 1,
      kind: 'first-instance-live-candidate-series-initialization',
      status: 'initialized-awaiting-external-anchor',
      studyKey: registry.studyKey,
      seriesDirectory: path.relative(PROJECT_ROOT, seriesDirectory).replaceAll('\\', '/'),
      registry,
      anchor,
      nextAction:
        'create and publish the exact annotated tag, then run npm run candidate:first-instance-live',
    };
  } finally {
    await rootLock.release();
  }
}

function candidateEnvironment(seriesId, reservation, resultRoot) {
  const environment = { ...process.env };
  const binding = candidateSeriesReservationRecord(reservation);
  delete environment.BENCHMARK_HETEROGENEOUS_COMPARATOR;
  Object.assign(environment, {
    BENCHMARK_MATRIX: 'first-instance-live',
    BENCHMARK_EVIDENCE_STATUS: 'candidate',
    BENCHMARK_OBJECT_COUNT: '65536',
    BENCHMARK_BUCKET_COUNT: '32',
    BENCHMARK_RESULT_ROOT: resultRoot,
    BENCHMARK_ENVIRONMENT_NOTE:
      `first-instance-live candidate series ${seriesId}; `
        + `matrix ${reservation.matrixOrdinal}; attempt ${reservation.attemptOrdinal}`,
    BENCHMARK_CANDIDATE_SERIES_ID: binding.seriesId,
    BENCHMARK_CANDIDATE_RESERVATION_EVENT_SHA256: binding.reservationEventSha256,
    BENCHMARK_CANDIDATE_ATTEMPT_ORDINAL: String(binding.attemptOrdinal),
    BENCHMARK_CANDIDATE_MATRIX_ORDINAL: String(binding.matrixOrdinal),
    BENCHMARK_CANDIDATE_SOURCE_COMMIT: binding.sourceCommit,
    BENCHMARK_CANDIDATE_SOURCE_TREE: binding.sourceTree,
    BENCHMARK_CANDIDATE_DEPENDENCY_CLOSURE_SHA256:
      binding.executionDependencyClosureSha256,
  });
  return environment;
}

async function writeChildLifecycle(filename, record) {
  const temporary = `${filename}.tmp-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(record, null, 2)}\n`, 'utf8');
  await rename(temporary, filename);
}

async function runBenchmarkProcess(seriesId, reservation, absoluteAttemptDirectory) {
  const stdoutPath = path.join(absoluteAttemptDirectory, 'runner-stdout.log');
  const stderrPath = path.join(absoluteAttemptDirectory, 'runner-stderr.log');
  const stdoutLog = createWriteStream(stdoutPath, { flags: 'wx' });
  const stderrLog = createWriteStream(stderrPath, { flags: 'wx' });
  const runnerPath = path.join(PROJECT_ROOT, 'scripts', 'run-benchmark.mjs');
  const resultRoot = path.join(absoluteAttemptDirectory, 'runs');
  const lifecyclePath = path.join(
    absoluteAttemptDirectory,
    'runner-child-lifecycle.json',
  );
  const spawnedAt = new Date().toISOString();
  await writeChildLifecycle(lifecyclePath, {
    schemaVersion: 1,
    kind: 'first-instance-live-candidate-child-lifecycle',
    status: 'pre-spawn',
    childPid: null,
    spawnedAt,
    closedAt: null,
    exitCode: null,
    signal: null,
  });
  const child = spawn(process.execPath, [runnerPath], {
    cwd: PROJECT_ROOT,
    env: candidateEnvironment(seriesId, reservation, resultRoot),
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  child.stdout.on('data', (chunk) => {
    stdoutLog.write(chunk);
    process.stdout.write(chunk);
  });
  child.stderr.on('data', (chunk) => {
    stderrLog.write(chunk);
    process.stderr.write(chunk);
  });
  const executionPromise = new Promise((resolve) => {
    let spawnError = null;
    child.once('error', (error) => {
      spawnError = error;
    });
    child.once('close', (exitCode, signal) => {
      resolve({
        exitCode: spawnError === null ? exitCode : 1,
        signal: spawnError === null ? signal : `spawn-error:${spawnError.code ?? 'unknown'}`,
      });
    });
  });
  await writeChildLifecycle(lifecyclePath, {
    schemaVersion: 1,
    kind: 'first-instance-live-candidate-child-lifecycle',
    status: 'running',
    childPid: child.pid ?? null,
    spawnedAt,
    closedAt: null,
    exitCode: null,
    signal: null,
  });
  const execution = await executionPromise;
  stdoutLog.end();
  stderrLog.end();
  await Promise.all([finished(stdoutLog), finished(stderrLog)]);
  await writeChildLifecycle(lifecyclePath, {
    schemaVersion: 1,
    kind: 'first-instance-live-candidate-child-lifecycle',
    status: 'closed',
    childPid: child.pid ?? null,
    spawnedAt,
    closedAt: new Date().toISOString(),
    exitCode: execution.exitCode,
    signal: execution.signal,
  });
  return execution;
}

async function finalizeReservation(
  seriesDirectory,
  ledgerPath,
  state,
  execution,
) {
  const reservation = state.pendingReservation;
  if (reservation === null) throw new Error('No pending candidate reservation to finalize.');
  const sourceEvidencePath = path.join(
    seriesDirectory,
    reservation.attemptDirectory,
    'orchestrator-source-after.json',
  );
  const sourceAfter = await collectSourceProvenance(
    PROJECT_ROOT,
    { allowUnavailable: true },
  );
  let dependencyAfter = null;
  try {
    dependencyAfter = await collectExecutionDependencyClosure(PROJECT_ROOT);
  } catch {
    // Preserve a closed attempt with an explicit unavailable dependency
    // recapture. The independent classifier treats this as a nonretryable
    // provenance failure instead of leaving a recoverable pending attempt.
  }
  try {
    await writeFile(
      sourceEvidencePath,
      `${JSON.stringify({
        schemaVersion: 1,
        kind: 'first-instance-live-orchestrator-source-after',
        sourceProvenance: sourceAfter,
        executionDependencyClosure: dependencyAfter,
      }, null, 2)}\n`,
      { encoding: 'utf8', flag: 'wx' },
    );
  } catch (error) {
    if (error?.code !== 'EEXIST') throw error;
  }
  const classification = await classifyUnfinalizedCandidateAttempt(
    seriesDirectory,
    reservation.attemptDirectory,
    {
      expectedSourceIdentity: reservation.sourceIdentity,
      expectedReservation: reservation,
    },
  );
  const contentCommitment = await buildAttemptContentCommitment(
    path.join(seriesDirectory, reservation.attemptDirectory),
  );
  return appendCandidateLedgerEvent(ledgerPath, {
    eventType: 'attempt-finalized',
    seriesId: state.seriesId,
    recordedAt: new Date().toISOString(),
    attemptOrdinal: reservation.attemptOrdinal,
    matrixOrdinal: reservation.matrixOrdinal,
    attemptDirectory: reservation.attemptDirectory,
    reservationEventSha256: reservation.eventSha256,
    classification: classification.classification,
    reasonCode: classification.reasonCode,
    runDirectory: classification.runDirectoryRelative,
    execution,
    contentCommitment,
    classificationEvidence: classification.classificationEvidence,
  });
}

export async function recoverPendingReservation(seriesDirectory, ledgerPath, state) {
  if (state.pendingReservation === null) return state;
  const lifecyclePath = path.join(
    seriesDirectory,
    state.pendingReservation.attemptDirectory,
    'runner-child-lifecycle.json',
  );
  let lifecycle = null;
  try {
    lifecycle = JSON.parse(await readFile(lifecyclePath, 'utf8'));
  } catch {
    // Missing or malformed lifecycle is ambiguous and fails closed below.
  }
  if (lifecycle?.status !== 'closed') {
    let childAlive = false;
    if (Number.isSafeInteger(lifecycle?.childPid) && lifecycle.childPid > 0) {
      try {
        process.kill(lifecycle.childPid, 0);
        childAlive = true;
      } catch (error) {
        if (error?.code === 'EPERM') childAlive = true;
        else if (error?.code !== 'ESRCH') throw error;
      }
    }
    throw new Error(
      childAlive
        ? `Pending ${state.pendingReservation.attemptDirectory} still has a live child process.`
        : `Pending ${state.pendingReservation.attemptDirectory} lacks an unambiguous closed-child marker; refusing overlapping recovery.`,
    );
  }
  process.stderr.write(
    `Finalizing closed ${state.pendingReservation.attemptDirectory} before any new attempt.\n`,
  );
  await finalizeReservation(
    seriesDirectory,
    ledgerPath,
    state,
    { exitCode: lifecycle.exitCode, signal: lifecycle.signal },
  );
  return readCandidateLedger(ledgerPath);
}

async function prepareCandidateSeriesForRun(
  seriesInput,
  { allowSelectedSeriesLock = false } = {},
) {
  const requestedDirectory = resolveCandidateSeriesDirectory(seriesInput);
  const sourceIdentity = await collectCandidateSourceIdentity();
  const expectedDirectory = candidateSeriesDirectory(CANDIDATE_SERIES_ROOT, sourceIdentity);
  if (requestedDirectory !== expectedDirectory) {
    throw new Error(
      'Candidate runs refuse aliases: the series directory must be derived from the current frozen study identity.',
    );
  }
  const registry = await verifyCandidateStudyRegistry(
    CANDIDATE_SERIES_ROOT,
    requestedDirectory,
    allowSelectedSeriesLock
      ? { allowSeriesLockDirectory: requestedDirectory }
      : undefined,
  );
  if (!sameSourceIdentity(registry.sourceIdentity, sourceIdentity)) {
    throw new Error(
      'Candidate registry source identity differs from the current clean source/dependency identity.',
    );
  }
  const anchor = await verifyCandidateRegistryAnchorTag(PROJECT_ROOT, registry);
  return { seriesDirectory: requestedDirectory, sourceIdentity, registry, anchor };
}

export async function runNextCandidateAttempt(seriesInput) {
  const preparedBeforeLock = await prepareCandidateSeriesForRun(
    seriesInput,
    { allowSelectedSeriesLock: true },
  );
  const { seriesDirectory } = preparedBeforeLock;
  const lock = await acquireCandidateSeriesLock(seriesDirectory);
  try {
    const prepared = await prepareCandidateSeriesForRun(
      seriesDirectory,
      { allowSelectedSeriesLock: true },
    );
    const { sourceIdentity } = prepared;
    if (!sameSourceIdentity(sourceIdentity, preparedBeforeLock.sourceIdentity)
      || prepared.registry.registryClaimEventSha256
        !== preparedBeforeLock.registry.registryClaimEventSha256
      || prepared.registry.registryMaterializedEventSha256
        !== preparedBeforeLock.registry.registryMaterializedEventSha256
      || prepared.anchor.messageSha256 !== preparedBeforeLock.anchor.messageSha256) {
      throw new Error('Candidate registry/source anchor changed while acquiring the series lock.');
    }
    let { ledgerPath, state } = await openCandidateSeries(
      seriesDirectory,
      sourceIdentity,
    );
    state = await recoverPendingReservation(seriesDirectory, ledgerPath, state);
    if (state.terminalFailure !== null) {
      throw new Error(
        'This frozen-source candidate series ended in a non-retryable '
          + 'implementation/evidence failure; a corrected clean commit requires a new series.',
      );
    }
    const reserved = await reserveCandidateAttempt(
      seriesDirectory,
      ledgerPath,
      state,
      sourceIdentity,
    );
    process.stdout.write(
      `Reserved ${reserved.event.attemptDirectory} for candidate matrix `
        + `${reserved.event.matrixOrdinal}; the reservation is committed before browser timing.\n`,
    );
    const execution = await runBenchmarkProcess(
      state.seriesId,
      reserved.event,
      reserved.absoluteAttemptDirectory,
    );
    state = await readCandidateLedger(ledgerPath);
    const finalized = await finalizeReservation(
      seriesDirectory,
      ledgerPath,
      state,
      execution,
    );
    const finalState = await readCandidateLedger(ledgerPath);
    const result = {
      schemaVersion: 1,
      kind: 'first-instance-live-candidate-attempt-result',
      seriesId: state.seriesId,
      attemptOrdinal: finalized.attemptOrdinal,
      matrixOrdinal: finalized.matrixOrdinal,
      classification: finalized.classification,
      reasonCode: finalized.reasonCode,
      finalEventSha256: finalized.eventSha256,
      studyKey: prepared.registry.studyKey,
      registryClaimEventSha256: prepared.registry.registryClaimEventSha256,
      registryMaterializedEventSha256:
        prepared.registry.registryMaterializedEventSha256,
      anchorTagName: prepared.anchor.tagName,
      completedMatrixCount: finalState.completedMatrixCount,
      nextAction: finalized.classification === 'infrastructure-invalid-retryable'
        ? `retry matrix ${finalized.matrixOrdinal} in a fresh session`
        : finalized.classification === 'implementation/evidence-failure-nonretryable'
          ? 'freeze a corrected clean commit and start a new candidate series'
          : finalState.completedMatrixCount === 1
            ? 'run candidate matrix 2 even if matrix 1 missed a decision gate'
            : 'verify the closed two-matrix candidate series',
    };
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    return result;
  } finally {
    await lock.release();
  }
}

export async function runCandidateSeriesFlow(
  seriesInput,
  {
    runAttempt = runNextCandidateAttempt,
    verifySeries = verifyCandidateSeries,
    emitPair = (pair) => process.stdout.write(`${JSON.stringify(pair, null, 2)}\n`),
  } = {},
) {
  const seriesDirectory = resolveCandidateSeriesDirectory(seriesInput);
  let attemptCount = 0;
  while (true) {
    const result = await runAttempt(seriesInput);
    attemptCount += 1;
    if (result.classification === 'infrastructure-invalid-retryable') {
      return {
        status: 'stopped-retryable-infrastructure',
        attemptCount,
        result,
        pair: null,
      };
    }
    if (result.classification === 'implementation/evidence-failure-nonretryable') {
      const pair = await verifySeries(seriesDirectory);
      emitPair(pair);
      return { status: 'stopped-nonretryable-evidence', attemptCount, result, pair };
    }
    if (result.completedMatrixCount === 2) {
      const pair = await verifySeries(seriesDirectory);
      emitPair(pair);
      return { status: 'verified-two-matrix-series', attemptCount, result, pair };
    }
  }
}

async function main() {
  const arguments_ = process.argv.slice(2);
  if (arguments_.length === 1 && arguments_[0] === '--initialize') {
    const initialized = await initializeCurrentCandidateSeries();
    process.stdout.write(`${JSON.stringify(initialized, null, 2)}\n`);
    return;
  }
  if (arguments_.length !== 0) {
    throw new Error(
      'Usage: node scripts/run-live-first-instance-candidate.mjs [--initialize]',
    );
  }
  const sourceIdentity = await collectCandidateSourceIdentity();
  await runCandidateSeriesFlow(candidateSeriesDirectory(
    CANDIDATE_SERIES_ROOT,
    sourceIdentity,
  ));
}

const invokedUrl = process.argv[1] === undefined
  ? null
  : pathToFileURL(path.resolve(process.argv[1])).href;

if (import.meta.url === invokedUrl) {
  main().catch((error) => {
    process.stderr.write(
      `run-live-first-instance-candidate: ${error instanceof Error ? error.message : String(error)}\n`,
    );
    process.exitCode = 1;
  });
}
