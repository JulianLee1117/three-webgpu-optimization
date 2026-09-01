import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';
import {
  CANDIDATE_LEDGER_FILENAME,
  CANDIDATE_SERIES_LOCK_FILENAME,
  openCandidateSeries,
  readCandidateLedger,
  reserveCandidateAttempt,
} from '../scripts/live-first-instance-candidate-ledger.mjs';
import {
  CANDIDATE_REGISTRY_FILENAME,
  acquireCandidateRegistryLock,
  candidateSeriesDirectory,
  candidateStudyKey,
  claimCandidateStudy,
  materializeCandidateStudy,
  validateCandidateSeriesLocation,
  verifyCandidateRegistryAnchorTag,
  verifyCandidateRegistryInitializationInventory,
  verifyCandidateStudyRegistry,
} from '../scripts/live-first-instance-candidate-registry.mjs';

const execFileAsync = promisify(execFile);

const SOURCE = Object.freeze({
  commit: 'a'.repeat(40),
  tree: 'b'.repeat(40),
  trackedFilesSha256: 'c'.repeat(64),
  packageLockSha256: 'd'.repeat(64),
  executionDependencyClosureSha256: 'e'.repeat(64),
  executionDependencyFileCount: 10,
  executionDependencyTotalBytes: 20,
});

async function temporaryRoot(t, prefix = 'first-instance-registry-') {
  const root = await mkdtemp(path.join(os.tmpdir(), prefix));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

async function materializedFixture(t, sourceIdentity = SOURCE) {
  const root = await temporaryRoot(t);
  const lock = await acquireCandidateRegistryLock(root);
  try {
    const claimed = await claimCandidateStudy(
      root,
      sourceIdentity,
      new Date('2026-09-01T00:00:00.000Z'),
    );
    const seriesDirectory = candidateSeriesDirectory(root, sourceIdentity);
    const opened = await openCandidateSeries(
      seriesDirectory,
      sourceIdentity,
      new Date('2026-09-01T00:00:01.000Z'),
    );
    await materializeCandidateStudy(
      claimed.registryPath,
      claimed.record.claim,
      opened.state,
      new Date('2026-09-01T00:00:02.000Z'),
    );
    const verification = await verifyCandidateStudyRegistry(
      root,
      seriesDirectory,
      { allowRootLock: true },
    );
    return { root, seriesDirectory, verification, lock };
  } catch (error) {
    await lock.release();
    throw error;
  }
}

test('canonical registry claims and materializes exactly one source series', async (t) => {
  const fixture = await materializedFixture(t);
  await fixture.lock.release();
  const verified = await verifyCandidateStudyRegistry(
    fixture.root,
    fixture.seriesDirectory,
  );
  assert.equal(verified.studyKey, candidateStudyKey(SOURCE));
  assert.equal(verified.sourceIdentity.commit, SOURCE.commit);
  assert.equal(verified.registryEventCount, 2);
  assert.equal(verified.registryMaterializedRecordedAt, '2026-09-01T00:00:02.000Z');
  assert.match(verified.seriesDirectory, /^first-device-live-[0-9a-f]{16}$/);
  assert.equal(verified.anchor.payload.registryClaimEventSha256, verified.registryClaimEventSha256);
  assert.equal(
    verified.anchor.payload.seriesOpeningEventSha256,
    verified.seriesOpeningEventSha256,
  );
});

test('study key ignores empty commits but full claim rejects commit or closure substitution', async (t) => {
  const sameTreeNewCommit = { ...SOURCE, commit: 'f'.repeat(40) };
  const changedClosure = {
    ...SOURCE,
    executionDependencyClosureSha256: '1'.repeat(64),
  };
  const changedTree = {
    ...SOURCE,
    tree: '2'.repeat(40),
    trackedFilesSha256: '3'.repeat(64),
  };
  assert.equal(candidateStudyKey(sameTreeNewCommit), candidateStudyKey(SOURCE));
  assert.equal(candidateStudyKey(changedClosure), candidateStudyKey(SOURCE));
  assert.notEqual(candidateStudyKey(changedTree), candidateStudyKey(SOURCE));

  const fixture = await materializedFixture(t);
  await fixture.lock.release();
  const lock = await acquireCandidateRegistryLock(fixture.root);
  try {
    await assert.rejects(
      claimCandidateStudy(fixture.root, sameTreeNewCommit),
      /different commit or installed-dependency closure/,
    );
    await assert.rejects(
      claimCandidateStudy(fixture.root, changedClosure),
      /different commit or installed-dependency closure/,
    );
    const changedClaim = await claimCandidateStudy(
      fixture.root,
      changedTree,
      new Date('2026-09-01T00:00:03.000Z'),
    );
    const changedSeriesDirectory = candidateSeriesDirectory(fixture.root, changedTree);
    const changedOpened = await openCandidateSeries(
      changedSeriesDirectory,
      changedTree,
      new Date('2026-09-01T00:00:04.000Z'),
    );
    await materializeCandidateStudy(
      changedClaim.registryPath,
      changedClaim.record.claim,
      changedOpened.state,
      new Date('2026-09-01T00:00:05.000Z'),
    );
    const changedVerification = await verifyCandidateStudyRegistry(
      fixture.root,
      changedSeriesDirectory,
      { allowRootLock: true },
    );
    assert.equal(changedVerification.studyKey, candidateStudyKey(changedTree));
    assert.equal(changedVerification.registryEventCount, 4);
  } finally {
    await lock.release();
  }
});

test('a crash after claim resumes the one pending canonical series', async (t) => {
  const root = await temporaryRoot(t);
  const lock = await acquireCandidateRegistryLock(root);
  try {
    const first = await claimCandidateStudy(root, SOURCE);
    const repeated = await claimCandidateStudy(root, SOURCE);
    assert.equal(first.record.claim.eventSha256, repeated.record.claim.eventSha256);
    assert.equal(repeated.created, false);
    const seriesDirectory = candidateSeriesDirectory(root, SOURCE);
    const opened = await openCandidateSeries(seriesDirectory, SOURCE);
    const materialized = await materializeCandidateStudy(
      first.registryPath,
      first.record.claim,
      opened.state,
    );
    assert.equal(materialized.created, true);
    const repeatedMaterialization = await materializeCandidateStudy(
      first.registryPath,
      first.record.claim,
      opened.state,
    );
    assert.equal(repeatedMaterialization.created, false);
    await verifyCandidateStudyRegistry(root, seriesDirectory, { allowRootLock: true });
  } finally {
    await lock.release();
  }
});

test('aliases, undeclared siblings, root locks, and missing registries fail closed', async (t) => {
  await t.test('an alias is never a canonical series location', async () => {
    const root = await temporaryRoot(t, 'first-instance-alias-');
    assert.throws(
      () => validateCandidateSeriesLocation(root, path.join(root, 'my-preferred-run'), SOURCE),
      /canonical/,
    );
  });

  await t.test('an undeclared copied sibling invalidates root inventory', async () => {
    const fixture = await materializedFixture(t);
    await fixture.lock.release();
    await mkdir(path.join(fixture.root, 'first-device-live-copy'));
    await assert.rejects(
      verifyCandidateStudyRegistry(fixture.root, fixture.seriesDirectory),
      /inventory differs/,
    );
  });

  await t.test('a root lock blocks independent verification', async () => {
    const fixture = await materializedFixture(t);
    await assert.rejects(
      verifyCandidateStudyRegistry(fixture.root, fixture.seriesDirectory),
      /root is locked/,
    );
    await fixture.lock.release();
  });

  await t.test('a series lock blocks independent verification', async () => {
    const fixture = await materializedFixture(t);
    await fixture.lock.release();
    const lockPath = path.join(
      fixture.seriesDirectory,
      CANDIDATE_SERIES_LOCK_FILENAME,
    );
    await writeFile(lockPath, '{"pid":999999,"createdAt":"2026-09-01T00:00:03.000Z"}\n');
    await assert.rejects(
      verifyCandidateStudyRegistry(fixture.root, fixture.seriesDirectory),
      /series .* is locked/,
    );
    await assert.rejects(
      verifyCandidateStudyRegistry(
        fixture.root,
        fixture.seriesDirectory,
        { allowSeriesLockDirectory: path.join(fixture.root, 'not-the-selected-series') },
      ),
      /series .* is locked/,
    );
    const verifiedWhileOwned = await verifyCandidateStudyRegistry(
      fixture.root,
      fixture.seriesDirectory,
      { allowSeriesLockDirectory: fixture.seriesDirectory },
    );
    assert.equal(verifiedWhileOwned.seriesId, fixture.verification.seriesId);
  });

  await t.test('an undeclared entry inside a materialized series fails closed', async () => {
    const fixture = await materializedFixture(t);
    await fixture.lock.release();
    await writeFile(path.join(fixture.seriesDirectory, 'alternate-ledger.jsonl'), '{}\n');
    await assert.rejects(
      verifyCandidateStudyRegistry(fixture.root, fixture.seriesDirectory),
      /series .* inventory differs/,
    );
  });

  await t.test('a pre-existing series without its registry cannot be claimed', async () => {
    const root = await temporaryRoot(t, 'first-instance-missing-registry-');
    const lock = await acquireCandidateRegistryLock(root);
    try {
      await mkdir(candidateSeriesDirectory(root, SOURCE));
      await assert.rejects(
        claimCandidateStudy(root, SOURCE),
        /undeclared entry/,
      );
    } finally {
      await lock.release();
    }
  });
});

test('claim, series opening, and materialization chronology is cross-checked', async (t) => {
  await t.test('series opening cannot predate the root claim', async () => {
    const root = await temporaryRoot(t, 'first-instance-claim-chronology-');
    const lock = await acquireCandidateRegistryLock(root);
    try {
      const claimed = await claimCandidateStudy(
        root,
        SOURCE,
        new Date('2026-09-01T00:00:02.000Z'),
      );
      const seriesDirectory = candidateSeriesDirectory(root, SOURCE);
      const opened = await openCandidateSeries(
        seriesDirectory,
        SOURCE,
        new Date('2026-09-01T00:00:01.000Z'),
      );
      await materializeCandidateStudy(
        claimed.registryPath,
        claimed.record.claim,
        opened.state,
        new Date('2026-09-01T00:00:03.000Z'),
      );
      await assert.rejects(
        verifyCandidateStudyRegistry(root, seriesDirectory, { allowRootLock: true }),
        /chronology crosses/,
      );
    } finally {
      await lock.release();
    }
  });

  await t.test('materialization cannot predate the series opening', async () => {
    const root = await temporaryRoot(t, 'first-instance-materialization-chronology-');
    const lock = await acquireCandidateRegistryLock(root);
    try {
      const claimed = await claimCandidateStudy(
        root,
        SOURCE,
        new Date('2026-09-01T00:00:00.000Z'),
      );
      const seriesDirectory = candidateSeriesDirectory(root, SOURCE);
      const opened = await openCandidateSeries(
        seriesDirectory,
        SOURCE,
        new Date('2026-09-01T00:00:03.000Z'),
      );
      await materializeCandidateStudy(
        claimed.registryPath,
        claimed.record.claim,
        opened.state,
        new Date('2026-09-01T00:00:02.000Z'),
      );
      await assert.rejects(
        verifyCandidateStudyRegistry(root, seriesDirectory, { allowRootLock: true }),
        /chronology crosses/,
      );
    } finally {
      await lock.release();
    }
  });

  await t.test('the first attempt reservation cannot predate registry materialization', async () => {
    const fixture = await materializedFixture(t);
    try {
      const ledgerPath = path.join(fixture.seriesDirectory, CANDIDATE_LEDGER_FILENAME);
      const state = await readCandidateLedger(ledgerPath);
      await reserveCandidateAttempt(
        fixture.seriesDirectory,
        ledgerPath,
        state,
        SOURCE,
        new Date('2026-09-01T00:00:01.500Z'),
      );
      await assert.rejects(
        verifyCandidateStudyRegistry(
          fixture.root,
          fixture.seriesDirectory,
          { allowRootLock: true },
        ),
        /registry materialization occurs after its first attempt reservation/,
      );
    } finally {
      await fixture.lock.release();
    }
  });
});

test('a materialized series cannot be deleted or transplanted', async (t) => {
  await t.test('deleted series is terminal', async () => {
    const fixture = await materializedFixture(t);
    await fixture.lock.release();
    await rm(fixture.seriesDirectory, { recursive: true, force: true });
    await assert.rejects(
      verifyCandidateStudyRegistry(fixture.root, fixture.seriesDirectory),
      /inventory differs|no such file/i,
    );
  });

  await t.test('mismatched series ledger is terminal', async () => {
    const fixture = await materializedFixture(t);
    await fixture.lock.release();
    const ledgerPath = path.join(fixture.seriesDirectory, CANDIDATE_LEDGER_FILENAME);
    const ledger = await readFile(ledgerPath, 'utf8');
    await writeFile(ledgerPath, ledger.replace('series-opened', 'series-altered'));
    await assert.rejects(
      verifyCandidateStudyRegistry(fixture.root, fixture.seriesDirectory),
      /unsupported eventType|digest/,
    );
  });
});

test('annotated tag exactly anchors claim, opening, and materialization digests', async (t) => {
  const repository = await temporaryRoot(t, 'first-instance-anchor-git-');
  const runGit = async (...args) => (await execFileAsync('git', args, {
    cwd: repository,
    encoding: 'utf8',
    windowsHide: true,
  })).stdout.trim();
  await runGit('init');
  await runGit('config', 'user.name', 'Candidate Registry Test');
  await runGit('config', 'user.email', 'candidate-registry@example.invalid');
  await writeFile(path.join(repository, 'tracked.txt'), 'frozen\n');
  await runGit('add', 'tracked.txt');
  await runGit('commit', '-m', 'frozen source');
  const source = {
    ...SOURCE,
    commit: await runGit('rev-parse', 'HEAD'),
    tree: await runGit('rev-parse', 'HEAD^{tree}'),
  };
  const candidateRoot = path.join(repository, 'results', 'candidate-series');
  const lock = await acquireCandidateRegistryLock(candidateRoot);
  let verification;
  try {
    const claimed = await claimCandidateStudy(candidateRoot, source);
    const seriesDirectory = candidateSeriesDirectory(candidateRoot, source);
    const opened = await openCandidateSeries(seriesDirectory, source);
    await materializeCandidateStudy(
      claimed.registryPath,
      claimed.record.claim,
      opened.state,
    );
    verification = await verifyCandidateStudyRegistry(
      candidateRoot,
      seriesDirectory,
      { allowRootLock: true },
    );
  } finally {
    await lock.release();
  }
  const { anchor } = verification;
  await runGit(
    'tag',
    '-a',
    anchor.tagName,
    anchor.targetCommit,
    '-m',
    anchor.message,
  );
  const accepted = await verifyCandidateRegistryAnchorTag(repository, verification);
  assert.equal(accepted.verified, true);
  assert.equal(accepted.messageSha256, anchor.messageSha256);

  await runGit('tag', '-d', anchor.tagName);
  await runGit('tag', anchor.tagName, anchor.targetCommit);
  await assert.rejects(
    verifyCandidateRegistryAnchorTag(repository, verification),
    /wrong object type|annotated/i,
  );
});

test('registry filename is fixed and cannot be replaced by an alternate root record', () => {
  assert.equal(CANDIDATE_REGISTRY_FILENAME, 'candidate-series-registry.jsonl');
});
