import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  access,
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import {
  buildThreeImmediateOverlay,
  DEFAULT_SOURCE_ROOT,
  EXPECTED_TARGET_HASHES,
  EXPECTED_THREE_VERSION,
  IMMEDIATE_DRAW_BASE_SYMBOL,
  transformThreeImmediateTarget,
} from '../scripts/build-three-immediate-overlay.mjs';

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

async function targetHashes(root) {
  const hashes = {};
  for (const relativePath of Object.keys(EXPECTED_TARGET_HASHES)) {
    hashes[relativePath] = sha256(await readFile(path.join(root, relativePath)));
  }
  return hashes;
}

async function makeTemporaryDirectory(t, prefix) {
  const directory = await mkdtemp(path.join(os.tmpdir(), prefix));
  t.after(async () => {
    await rm(directory, { recursive: true, force: true });
  });
  return directory;
}

async function copyPinnedTargetFixture(destination) {
  for (const relativePath of Object.keys(EXPECTED_TARGET_HASHES)) {
    const destinationPath = path.join(destination, relativePath);
    await mkdir(path.dirname(destinationPath), { recursive: true });
    await copyFile(path.join(DEFAULT_SOURCE_ROOT, relativePath), destinationPath);
  }
}

async function outputFingerprint(outputRoot) {
  const files = [
    ...Object.keys(EXPECTED_TARGET_HASHES),
    'THREE_IMMEDIATE_OVERLAY.json',
  ];
  const fingerprint = {};
  for (const relativePath of files) {
    fingerprint[relativePath] = sha256(
      await readFile(path.join(outputRoot, relativePath)),
    );
  }
  return fingerprint;
}

test('overlay pins the exact installed three package and every patch target', async () => {
  const packageMetadata = JSON.parse(
    await readFile(path.join(DEFAULT_SOURCE_ROOT, 'package.json'), 'utf8'),
  );

  assert.equal(packageMetadata.name, 'three');
  assert.equal(packageMetadata.version, EXPECTED_THREE_VERSION);
  assert.deepEqual(await targetHashes(DEFAULT_SOURCE_ROOT), EXPECTED_TARGET_HASHES);
});

test('every text transformation is idempotent and malformed input fails closed', async () => {
  for (const relativePath of Object.keys(EXPECTED_TARGET_HASHES)) {
    if (relativePath === 'package.json') continue;

    const original = await readFile(path.join(DEFAULT_SOURCE_ROOT, relativePath), 'utf8');
    const patched = transformThreeImmediateTarget(relativePath, original);
    assert.notEqual(patched, original, `${relativePath} should be patched`);
    assert.equal(
      transformThreeImmediateTarget(relativePath, patched),
      patched,
      `${relativePath} should be idempotent`,
    );
  }

  assert.throws(
    () => transformThreeImmediateTarget(
      'src/renderers/webgpu/nodes/WGSLNodeBuilder.js',
      'unrelated source',
    ),
    /found 0 source anchors; expected exactly 1/,
  );

  const wgslPath = 'src/renderers/webgpu/nodes/WGSLNodeBuilder.js';
  const originalWGSL = await readFile(path.join(DEFAULT_SOURCE_ROOT, wgslPath), 'utf8');
  const partialWGSL = transformThreeImmediateTarget(wgslPath, originalWGSL)
    .replace('requires immediate_address_space;', 'requires broken_address_space;');
  assert.throws(
    () => transformThreeImmediateTarget(wgslPath, partialWGSL),
    /only partially present/,
  );
});

test('generation is deterministic, copies the package, and leaves node_modules untouched', async (t) => {
  const temporaryRoot = await makeTemporaryDirectory(
    t,
    'three-immediate-overlay-generation-',
  );
  const outputRoot = path.join(temporaryRoot, 'overlay');
  const beforeSourceHashes = await targetHashes(DEFAULT_SOURCE_ROOT);

  const first = await buildThreeImmediateOverlay({ outputRoot });
  const firstFingerprint = await outputFingerprint(outputRoot);
  const firstManifestBytes = await readFile(
    path.join(outputRoot, 'THREE_IMMEDIATE_OVERLAY.json'),
    'utf8',
  );

  const second = await buildThreeImmediateOverlay({ outputRoot });
  const secondFingerprint = await outputFingerprint(outputRoot);
  const secondManifestBytes = await readFile(
    path.join(outputRoot, 'THREE_IMMEDIATE_OVERLAY.json'),
    'utf8',
  );

  assert.deepEqual(secondFingerprint, firstFingerprint);
  assert.equal(secondManifestBytes, firstManifestBytes);
  assert.deepEqual(await targetHashes(DEFAULT_SOURCE_ROOT), beforeSourceHashes);
  assert.deepEqual(beforeSourceHashes, EXPECTED_TARGET_HASHES);
  assert.equal(first.outputRoot, outputRoot);
  assert.equal(second.outputRoot, outputRoot);

  await access(path.join(outputRoot, 'LICENSE'));
  await access(path.join(outputRoot, 'examples', 'jsm', 'Addons.js'));

  const manifest = JSON.parse(firstManifestBytes);
  assert.equal(manifest.source.version, EXPECTED_THREE_VERSION);
  assert.equal(manifest.contract.symbol, IMMEDIATE_DRAW_BASE_SYMBOL);
  assert.equal(manifest.contract.immediateSizeBytes, 4);
  assert.deepEqual(manifest.source.targetSha256, EXPECTED_TARGET_HASHES);

  const descriptorSource = await readFile(
    path.join(
      outputRoot,
      'src/renderers/webgpu/descriptors/GPUPipelineLayoutDescriptor.js',
    ),
    'utf8',
  );
  assert.match(descriptorSource, /this\.immediateSize = 0;/);
  assert.match(
    descriptorSource,
    /prevent the singleton descriptor from leaking opt-in[\s\S]*this\.immediateSize = 0;/,
  );

  const pipelineSource = await readFile(
    path.join(outputRoot, 'src/renderers/webgpu/utils/WebGPUPipelineUtils.js'),
    'utf8',
  );
  assert.match(
    pipelineSource,
    /immediateSize = vertexProgram\.code\.includes\( 'var<immediate> threeImmediateDrawBase : u32;' \) \? 4 : 0;/,
  );

  const wgslSource = await readFile(
    path.join(outputRoot, 'src/renderers/webgpu/nodes/WGSLNodeBuilder.js'),
    'utf8',
  );
  assert.match(wgslSource, /requires immediate_address_space;/);
  assert.match(wgslSource, /var<immediate> threeImmediateDrawBase : u32;/);

  const backendSource = await readFile(
    path.join(outputRoot, 'src/renderers/webgpu/WebGPUBackend.js'),
    'utf8',
  );
  assert.equal(
    backendSource.match(
      /passEncoderGPU\.setImmediates\( 0, indirectImmediateBases, i, 1 \);/g,
    )?.length,
    2,
  );
  assert.match(
    backendSource,
    /set one indexed-indirect base immediately before its draw[\s\S]*setImmediates\( 0, indirectImmediateBases, i, 1 \);[\s\S]*drawIndexedIndirect\( buffer, indirectOffsets\[ i \] \);/,
  );
  assert.match(
    backendSource,
    /set one non-indexed-indirect base immediately before its draw[\s\S]*setImmediates\( 0, indirectImmediateBases, i, 1 \);[\s\S]*drawIndirect\( buffer, indirectOffsets\[ i \] \);/,
  );

  const descriptorModule = await import(
    `${pathToFileURL(path.join(
      outputRoot,
      'src/renderers/webgpu/descriptors/GPUPipelineLayoutDescriptor.js',
    )).href}?overlay-test=descriptor`
  );
  const descriptor = new descriptorModule.default();
  assert.equal(descriptor.immediateSize, 0);
  descriptor.immediateSize = 4;
  descriptor.reset();
  assert.equal(descriptor.immediateSize, 0);

  const webgpuModule = await import(
    `${pathToFileURL(path.join(outputRoot, 'build', 'three.webgpu.js')).href}?overlay-test=build`
  );
  const geometry = new webgpuModule.BufferGeometry();
  const bases = new Uint32Array([3, 11]);
  geometry.setIndirect(null, [0, 20], bases);
  assert.equal(geometry.getIndirectImmediateBases(), bases);
});

test('wrong Three.js version fails before copying and preserves prior output', async (t) => {
  const temporaryRoot = await makeTemporaryDirectory(
    t,
    'three-immediate-overlay-version-',
  );
  const sourceRoot = path.join(temporaryRoot, 'source');
  const outputRoot = path.join(temporaryRoot, 'output');
  await mkdir(sourceRoot, { recursive: true });
  await mkdir(outputRoot, { recursive: true });
  await writeFile(
    path.join(sourceRoot, 'package.json'),
    JSON.stringify({ name: 'three', version: '0.185.0' }),
  );
  await writeFile(path.join(outputRoot, 'prior-output.txt'), 'keep me');

  await assert.rejects(
    buildThreeImmediateOverlay({ sourceRoot, outputRoot }),
    /requires three@0\.185\.1; found three@0\.185\.0/,
  );
  assert.equal(
    await readFile(path.join(outputRoot, 'prior-output.txt'), 'utf8'),
    'keep me',
  );
});

test('changed pinned source fails hash verification and preserves prior output', async (t) => {
  const temporaryRoot = await makeTemporaryDirectory(
    t,
    'three-immediate-overlay-hash-',
  );
  const sourceRoot = path.join(temporaryRoot, 'source');
  const outputRoot = path.join(temporaryRoot, 'output');
  await copyPinnedTargetFixture(sourceRoot);
  await mkdir(outputRoot, { recursive: true });
  await writeFile(path.join(outputRoot, 'prior-output.txt'), 'keep me');

  const changedPath = path.join(sourceRoot, 'src/core/BufferGeometry.js');
  await writeFile(changedPath, `${await readFile(changedPath, 'utf8')}\n// changed\n`);

  await assert.rejects(
    buildThreeImmediateOverlay({ sourceRoot, outputRoot }),
    /source hash mismatch for src\/core\/BufferGeometry\.js/,
  );
  assert.equal(
    await readFile(path.join(outputRoot, 'prior-output.txt'), 'utf8'),
    'keep me',
  );
});
