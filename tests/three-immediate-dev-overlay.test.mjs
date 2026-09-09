import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  access,
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';
import {
  buildThreeImmediateDevOverlay,
  DEFAULT_THREE_DEV_SOURCE_ROOT,
  EXPECTED_IMMEDIATE_DRAW_BASE_SYMBOL,
  EXPECTED_THREE_DEV_COMMIT,
  EXPECTED_THREE_DEV_PATCHED_HASHES,
  EXPECTED_THREE_DEV_REVISION,
  EXPECTED_THREE_DEV_TARGET_HASHES,
  EXPECTED_THREE_DEV_TREE,
  EXPECTED_THREE_DEV_VERSION,
  inspectThreeImmediateDevSource,
} from '../scripts/build-three-immediate-dev-overlay.mjs';

const execFileAsync = promisify(execFile);

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

async function makeTemporaryDirectory(t, prefix) {
  const directory = await mkdtemp(path.join(os.tmpdir(), prefix));
  t.after(async () => {
    await rm(directory, { recursive: true, force: true });
  });
  return directory;
}

async function copyPinnedTargetFixture(destination) {
  for (const relativePath of Object.keys(EXPECTED_THREE_DEV_TARGET_HASHES)) {
    const destinationPath = path.join(destination, ...relativePath.split('/'));
    await mkdir(path.dirname(destinationPath), { recursive: true });
    await copyFile(
      path.join(DEFAULT_THREE_DEV_SOURCE_ROOT, ...relativePath.split('/')),
      destinationPath,
    );
  }
}

async function runGit(root, args) {
  const { stdout } = await execFileAsync(
    'git',
    ['-c', `safe.directory=${root}`, '-C', root, ...args],
    { encoding: 'utf8', windowsHide: true },
  );
  return stdout;
}

async function listFiles(root, relativeDirectory = '') {
  const directory = path.join(root, ...relativeDirectory.split('/').filter(Boolean));
  const entries = await readdir(directory, { withFileTypes: true });
  entries.sort((left, right) => left.name.localeCompare(right.name, 'en'));

  const files = [];
  for (const entry of entries) {
    const relativePath = relativeDirectory === ''
      ? entry.name
      : `${relativeDirectory}/${entry.name}`;
    if (entry.isDirectory()) {
      files.push(...await listFiles(root, relativePath));
    } else {
      assert.equal(entry.isFile(), true, `unexpected non-file payload entry ${relativePath}`);
      files.push(relativePath);
    }
  }
  return files;
}

async function outputFingerprint(outputRoot) {
  const fingerprint = {};
  for (const relativePath of await listFiles(outputRoot)) {
    fingerprint[relativePath] = sha256(
      await readFile(path.join(outputRoot, ...relativePath.split('/'))),
    );
  }
  return fingerprint;
}

function sourceSnapshot(inspection) {
  return {
    commit: inspection.commit,
    tree: inspection.tree,
    trackedStatus: inspection.trackedStatus,
    fullStatus: inspection.fullStatus,
    canonicalHashes: inspection.canonicalHashes,
    worktreeHashes: inspection.worktreeHashes,
  };
}

test('dev overlay pins the exact clean upstream identity and all targets', async () => {
  const inspection = await inspectThreeImmediateDevSource();

  assert.equal(inspection.sourceRoot, DEFAULT_THREE_DEV_SOURCE_ROOT);
  assert.equal(inspection.packageMetadata.name, 'three');
  assert.equal(inspection.packageMetadata.version, EXPECTED_THREE_DEV_VERSION);
  assert.equal(inspection.runtimeRevision, EXPECTED_THREE_DEV_REVISION);
  assert.equal(inspection.commit, EXPECTED_THREE_DEV_COMMIT);
  assert.equal(inspection.tree, EXPECTED_THREE_DEV_TREE);
  assert.equal(inspection.trackedStatus, '');
  assert.equal(inspection.fullStatus, '');
  assert.deepEqual(
    inspection.canonicalHashes,
    EXPECTED_THREE_DEV_TARGET_HASHES,
  );
});

test('generation is deterministic, minimal, source-preserving, and uses legal WGSL', {
  timeout: 120_000,
}, async (t) => {
  const temporaryRoot = await makeTemporaryDirectory(
    t,
    'three-immediate-dev-overlay-generation-',
  );
  const outputRoot = path.join(temporaryRoot, 'overlay');
  const beforeSource = sourceSnapshot(await inspectThreeImmediateDevSource());

  const first = await buildThreeImmediateDevOverlay({ outputRoot });
  const firstFingerprint = await outputFingerprint(outputRoot);
  const firstManifestBytes = await readFile(
    path.join(outputRoot, 'THREE_IMMEDIATE_DEV_OVERLAY.json'),
    'utf8',
  );

  const second = await buildThreeImmediateDevOverlay({ outputRoot });
  const secondFingerprint = await outputFingerprint(outputRoot);
  const secondManifestBytes = await readFile(
    path.join(outputRoot, 'THREE_IMMEDIATE_DEV_OVERLAY.json'),
    'utf8',
  );
  const afterSource = sourceSnapshot(await inspectThreeImmediateDevSource());

  assert.equal(first.outputRoot, outputRoot);
  assert.equal(second.outputRoot, outputRoot);
  assert.deepEqual(secondFingerprint, firstFingerprint);
  assert.equal(secondManifestBytes, firstManifestBytes);
  assert.deepEqual(afterSource, beforeSource);

  assert.deepEqual(
    (await readdir(outputRoot)).sort(),
    [
      'LICENSE',
      'README.md',
      'THREE_IMMEDIATE_DEV_OVERLAY.json',
      'build',
      'examples',
      'package.json',
      'src',
    ],
  );
  assert.deepEqual(await readdir(path.join(outputRoot, 'examples')), ['jsm']);
  await assert.rejects(access(path.join(outputRoot, '.git')), /ENOENT/);
  await assert.rejects(access(path.join(outputRoot, 'docs')), /ENOENT/);
  await assert.rejects(access(path.join(outputRoot, 'test')), /ENOENT/);

  const manifest = JSON.parse(firstManifestBytes);
  assert.equal(manifest.source.version, EXPECTED_THREE_DEV_VERSION);
  assert.equal(manifest.source.runtimeRevision, EXPECTED_THREE_DEV_REVISION);
  assert.equal(manifest.source.commit, EXPECTED_THREE_DEV_COMMIT);
  assert.equal(manifest.source.tree, EXPECTED_THREE_DEV_TREE);
  assert.equal(manifest.source.cleanTrackedTree, true);
  assert.equal(manifest.source.cleanFullWorktree, true);
  assert.equal(manifest.source.statusPorcelain, '');
  assert.deepEqual(
    manifest.source.targetCanonicalSha256,
    EXPECTED_THREE_DEV_TARGET_HASHES,
  );
  assert.deepEqual(
    manifest.transform.patchedTargetSha256,
    EXPECTED_THREE_DEV_PATCHED_HASHES,
  );
  assert.equal(manifest.contract.symbol, EXPECTED_IMMEDIATE_DRAW_BASE_SYMBOL);
  assert.deepEqual(manifest.contract.servedEntrypoints, {
    'three/webgpu': 'build/three.webgpu.js',
    'three/tsl': 'build/three.tsl.js',
  });
  assert.deepEqual(
    manifest.contract.unpatchedUnexportedBuilds,
    ['build/three.webgpu.nodes.js'],
  );
  assert.match(EXPECTED_IMMEDIATE_DRAW_BASE_SYMBOL, /^(?!__)[A-Za-z_][A-Za-z0-9_]*$/);
  assert.equal(
    Object.keys(firstFingerprint).length,
    manifest.output.fileCount + 1,
  );
  assert.equal(
    manifest.output.files.find((file) => file.path === 'package.json')?.sha256,
    EXPECTED_THREE_DEV_TARGET_HASHES['package.json'],
    'the output must use canonical Git blob bytes rather than CRLF worktree bytes',
  );

  for (const relativePath of [
    'src/renderers/webgpu/nodes/WGSLNodeBuilder.js',
    'build/three.webgpu.js',
  ]) {
    const source = await readFile(
      path.join(outputRoot, ...relativePath.split('/')),
      'utf8',
    );
    assert.match(source, /requires immediate_address_space;/);
    assert.match(source, /var<immediate> threeImmediateDrawBase : u32;/);
    assert.doesNotMatch(source, /__threeImmediateDrawBase/);
  }
});

test('wrong commit fails closed and preserves an existing output', {
  timeout: 60_000,
}, async (t) => {
  const temporaryRoot = await makeTemporaryDirectory(
    t,
    'three-immediate-dev-overlay-commit-',
  );
  const sourceRoot = path.join(temporaryRoot, 'source');
  const outputRoot = path.join(temporaryRoot, 'output');
  await mkdir(sourceRoot, { recursive: true });
  await mkdir(outputRoot, { recursive: true });
  await copyPinnedTargetFixture(sourceRoot);
  await writeFile(path.join(outputRoot, 'prior-output.txt'), 'keep me');

  await runGit(sourceRoot, ['init', '--quiet']);
  await runGit(sourceRoot, ['add', '--all']);
  await runGit(sourceRoot, [
    '-c',
    'user.name=Overlay Test',
    '-c',
    'user.email=overlay-test@example.invalid',
    'commit',
    '--quiet',
    '-m',
    'wrong pinned commit',
  ]);

  await assert.rejects(
    buildThreeImmediateDevOverlay({ sourceRoot, outputRoot }),
    /commit mismatch/,
  );
  assert.equal(
    await readFile(path.join(outputRoot, 'prior-output.txt'), 'utf8'),
    'keep me',
  );
  assert.deepEqual(await readdir(outputRoot), ['prior-output.txt']);
});

test('wrong target hash fails closed and preserves an existing output', async (t) => {
  const temporaryRoot = await makeTemporaryDirectory(
    t,
    'three-immediate-dev-overlay-hash-',
  );
  const sourceRoot = path.join(temporaryRoot, 'source');
  const outputRoot = path.join(temporaryRoot, 'output');
  await mkdir(sourceRoot, { recursive: true });
  await mkdir(outputRoot, { recursive: true });
  await copyPinnedTargetFixture(sourceRoot);
  await writeFile(path.join(outputRoot, 'prior-output.txt'), 'keep me');

  const changedPath = path.join(sourceRoot, 'src', 'core', 'BufferGeometry.js');
  await writeFile(
    changedPath,
    Buffer.concat([await readFile(changedPath), Buffer.from('\n// changed\n')]),
  );

  await assert.rejects(
    buildThreeImmediateDevOverlay({ sourceRoot, outputRoot }),
    /source hash mismatch for src\/core\/BufferGeometry\.js/,
  );
  assert.equal(
    await readFile(path.join(outputRoot, 'prior-output.txt'), 'utf8'),
    'keep me',
  );
  assert.deepEqual(await readdir(outputRoot), ['prior-output.txt']);
});
