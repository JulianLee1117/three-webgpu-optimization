import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  collectSourceProvenance,
  sha256File,
  sha256TrackedFiles,
  sourceProvenanceMatches,
  summarizePorcelain,
} from '../scripts/source-provenance.mjs';

test('porcelain summary counts every entry without retaining paths', () => {
  assert.deepEqual(summarizePorcelain([
    'M  staged.js',
    ' M unstaged.js',
    'MM both.js',
    '?? untracked.js',
    '',
  ].join('\n')), {
    dirty: true,
    stagedChanges: 2,
    unstagedChanges: 2,
    untrackedFiles: 1,
    entryCount: 4,
  });
});

test('source hashes are deterministic and bind names as well as contents', async (context) => {
  const directory = await mkdtemp(path.join(tmpdir(), 'three-webgpu-provenance-'));
  context.after(async () => {
    const { rm } = await import('node:fs/promises');
    await rm(directory, { recursive: true, force: true });
  });
  await writeFile(path.join(directory, 'a.txt'), 'alpha');
  await writeFile(path.join(directory, 'b.txt'), 'beta');

  const first = await sha256TrackedFiles(directory, ['b.txt', 'a.txt']);
  const second = await sha256TrackedFiles(directory, ['a.txt', 'b.txt', 'a.txt']);
  assert.equal(first, second);
  assert.equal(
    await sha256File(path.join(directory, 'a.txt')),
    createHash('sha256').update('alpha').digest('hex'),
  );

  await writeFile(path.join(directory, 'b.txt'), 'changed');
  assert.notEqual(await sha256TrackedFiles(directory, ['a.txt', 'b.txt']), first);
});

test('repository provenance records a commit and reproducible working-tree hashes', async () => {
  const projectRoot = path.resolve(import.meta.dirname, '..');
  const first = await collectSourceProvenance(projectRoot);
  const second = await collectSourceProvenance(projectRoot);
  assert.equal(first.status, 'available');
  assert.match(first.commit, /^[0-9a-f]{40}$/);
  assert.match(first.tree, /^[0-9a-f]{40}$/);
  assert.match(first.trackedFilesSha256, /^[0-9a-f]{64}$/);
  assert.match(first.packageLockSha256, /^[0-9a-f]{64}$/);
  assert.match(first.porcelainSha256, /^[0-9a-f]{64}$/);
  assert.match(first.trackedListSha256, /^[0-9a-f]{64}$/);
  assert.equal(first.captureStable, true);
  assert.equal(first.captureAttempts >= 1 && first.captureAttempts <= 3, true);
  assert.equal(typeof first.dirty, 'boolean');
  assert.equal(first.packageLockTracked, true);
  assert.equal(typeof first.capturedAt, 'string');
  assert.equal(first.trackedFileCount > 0, true);
  assert.equal(Number.isInteger(first.porcelainEntryCount), true);
  assert.equal(first.porcelainEntryCount >= first.untrackedFiles, true);
  assert.equal(sourceProvenanceMatches(first, second), true);
  assert.equal(sourceProvenanceMatches(first, {
    ...second,
    porcelainSha256: '0'.repeat(64),
  }), false);
  assert.equal(sourceProvenanceMatches(first, { ...second, captureStable: false }), false);
});
