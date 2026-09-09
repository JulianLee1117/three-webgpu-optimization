import { execFile } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import {
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  rename,
  rm,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import {
  IMMEDIATE_DRAW_BASE_SYMBOL,
  transformThreeImmediateTarget,
} from './build-three-immediate-overlay.mjs';

const execFileAsync = promisify(execFile);

export const EXPECTED_THREE_DEV_COMMIT =
  '994260a7a59abe466de8e59db1ecc9e30352751d';
export const EXPECTED_THREE_DEV_TREE =
  '7f3d4a048ea19bef1a488b0e2e243e47f6b30002';
export const EXPECTED_THREE_DEV_VERSION = '0.185.0';
export const EXPECTED_THREE_DEV_REVISION = '186dev';
export const EXPECTED_IMMEDIATE_DRAW_BASE_SYMBOL = 'threeImmediateDrawBase';

// These are SHA-256 hashes of the canonical Git blob bytes at the pinned
// commit. Text files in a Windows checkout may have CRLF worktree endings;
// target verification canonicalizes CRLF back to LF before comparing.
export const EXPECTED_THREE_DEV_TARGET_HASHES = Object.freeze({
  'package.json': 'd10b14049185b5a0f07a73273b0885f5e24be5d57e8bf0dd6cba436cdb810670',
  'src/constants.js': '90a3d227f75ef2a9ebb100d793ff1b01f8af429b7313bee48ba7e80b6cdc2615',
  'src/core/BufferGeometry.js': 'a920f442ebafcc78a2749546a6c1ecd8290de47af693c4866a0034ab0c362c1a',
  'src/renderers/common/Geometries.js': '1b21162c0a3d6e8558d35c20e83f51af60d3a951aa9940d265be7abd0fbcdbb7',
  'src/renderers/common/RenderObject.js': 'fd39f216baf77dad588cbc57712a08bf047bd27475f448b47c157b50e43f0838',
  'src/renderers/webgpu/nodes/WGSLNodeBuilder.js': 'f742b3fe5d2a9396c0aadff8c9ba0489a2ce8dc833b199abc95a4b683dd74ed3',
  'src/renderers/webgpu/descriptors/GPUPipelineLayoutDescriptor.js': 'be01afed4209eab3538645cea84bc74c01d1c840f5c224fd9b1e287ae1cc491b',
  'src/renderers/webgpu/utils/WebGPUPipelineUtils.js': 'ed9084df4210ec3f6eb848b0a99cab606821e78319aee0a8757abe57987f85fa',
  'src/renderers/webgpu/WebGPUBackend.js': 'fe2781e3c7c9e390b6c0121e1d4823d929a8a60e89aae618d49dad0abbc1574e',
  'build/three.core.js': 'c6c3642f5dfda0166f04c351dea70d13acaf44f34625450154ce79226c6fa98f',
  'build/three.webgpu.js': '3517e4f3987cd3b032a86832157173d5294773be8aec88c90e4734cde62d96af',
});

// Pinning the transformed bytes prevents a later change to the shared r185
// transform from silently changing this dev overlay.
export const EXPECTED_THREE_DEV_PATCHED_HASHES = Object.freeze({
  'src/core/BufferGeometry.js': '1ec1ae480437500ced22cac31cc377065a6c32b7e033c22c62500d4af7738971',
  'src/renderers/common/Geometries.js': '8a7593bc84c1db4a7232e56946767316a16f3caf3531f3ed374931a7d8d235d6',
  'src/renderers/common/RenderObject.js': 'd64edfebe64113c9c44865a25e0fc253b0bb62351d61f94da53ab3e9d3957680',
  'src/renderers/webgpu/nodes/WGSLNodeBuilder.js': 'c58eeb4145a9425bf2c05b24c54b06e1b902f2e9bfd399d6f801657086f40c4c',
  'src/renderers/webgpu/descriptors/GPUPipelineLayoutDescriptor.js': 'ed4f6d8e8fc5ec906d5ee1618154ba24cddc3a89ae61a8164304da2577391031',
  'src/renderers/webgpu/utils/WebGPUPipelineUtils.js': '949f4fedc739e892f517d1acde343cdc2083ddfbe9355a069cd9d0cb1ab2864f',
  'src/renderers/webgpu/WebGPUBackend.js': 'ef436a8cbbacb9e3188d6d23270a5a765f0afe44cfb5d05d976c1d8331bbeac9',
  'build/three.core.js': '2ed0b684bf494c7a6080a3ccc9cdfaebe10612a022278572bee00b52f682ad99',
  'build/three.webgpu.js': 'd67de0124a4edb1afecdc40fabe0912b81bccada43cd41941d2940e1f86477ba',
});

export const THREE_DEV_RUNTIME_PATHS = Object.freeze([
  'package.json',
  'LICENSE',
  'README.md',
  'build',
  'src',
  'examples/jsm',
]);

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const DEFAULT_THREE_DEV_SOURCE_ROOT = path.resolve(repoRoot, '..', 'upstream-three');
export const DEFAULT_THREE_DEV_OUTPUT_ROOT = path.join(
  repoRoot,
  '.generated',
  'three-immediate-dev-overlay',
);

const PATCH_TARGETS = Object.freeze(Object.keys(EXPECTED_THREE_DEV_PATCHED_HASHES));
const textDecoder = new TextDecoder('utf-8', { fatal: true });

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function canonicalText(bytes, relativePath) {
  let text;
  try {
    text = textDecoder.decode(bytes);
  } catch (error) {
    throw new Error(`Three dev overlay target is not valid UTF-8: ${relativePath}`, {
      cause: error,
    });
  }
  return text.replaceAll('\r\n', '\n');
}

function comparePaths(left, right) {
  const leftResolved = path.resolve(left);
  const rightResolved = path.resolve(right);
  if (process.platform === 'win32') {
    return leftResolved.toLowerCase() === rightResolved.toLowerCase();
  }
  return leftResolved === rightResolved;
}

function isPathInside(candidate, parent) {
  const relative = path.relative(path.resolve(parent), path.resolve(candidate));
  return relative !== ''
    && relative !== '..'
    && !relative.startsWith(`..${path.sep}`)
    && !path.isAbsolute(relative);
}

function assertSafeRoots(sourceRoot, outputRoot) {
  const source = path.resolve(sourceRoot);
  const output = path.resolve(outputRoot);
  if (output === path.parse(output).root) {
    throw new Error('Three dev overlay output cannot be a filesystem root');
  }
  if (comparePaths(source, output)
      || isPathInside(output, source)
      || isPathInside(source, output)) {
    throw new Error('Three dev overlay source and output must be disjoint directories');
  }
}

async function runGit(sourceRoot, args, {
  encoding = 'utf8',
  config = [],
} = {}) {
  const result = await execFileAsync(
    'git',
    [
      '-c',
      `safe.directory=${sourceRoot}`,
      ...config.flatMap((value) => ['-c', value]),
      '-C',
      sourceRoot,
      ...args,
    ],
    {
      encoding,
      maxBuffer: 64 * 1024 * 1024,
      windowsHide: true,
    },
  );
  return result.stdout;
}

async function verifyTargetHashes(sourceRoot) {
  const canonicalHashes = {};
  const worktreeHashes = {};

  for (const [relativePath, expectedHash]
    of Object.entries(EXPECTED_THREE_DEV_TARGET_HASHES)) {
    const bytes = await readFile(path.join(sourceRoot, relativePath));
    const canonicalBytes = Buffer.from(canonicalText(bytes, relativePath), 'utf8');
    const canonicalHash = sha256(canonicalBytes);
    if (canonicalHash !== expectedHash) {
      throw new Error(
        `Three dev overlay source hash mismatch for ${relativePath}: expected ${expectedHash}, received ${canonicalHash}`,
      );
    }
    canonicalHashes[relativePath] = canonicalHash;
    worktreeHashes[relativePath] = sha256(bytes);
  }

  return { canonicalHashes, worktreeHashes };
}

function verifyPackageMetadata(packageMetadata) {
  if (packageMetadata.name !== 'three'
      || packageMetadata.version !== EXPECTED_THREE_DEV_VERSION) {
    throw new Error(
      `Three dev overlay requires three@${EXPECTED_THREE_DEV_VERSION}; found ${packageMetadata.name ?? '<unknown>'}@${packageMetadata.version ?? '<unknown>'}`,
    );
  }

  const packageFiles = [...(packageMetadata.files ?? [])].sort();
  const expectedFiles = [...THREE_DEV_RUNTIME_PATHS].sort();
  if (JSON.stringify(packageFiles) !== JSON.stringify(expectedFiles)) {
    throw new Error(
      `Pinned Three dev package runtime payload changed: expected ${expectedFiles.join(', ')}, received ${packageFiles.join(', ')}`,
    );
  }
}

function assertLegalImmediateSymbol() {
  const legalIdentifier = /^(?!__)[A-Za-z_][A-Za-z0-9_]*$/;
  if (IMMEDIATE_DRAW_BASE_SYMBOL !== EXPECTED_IMMEDIATE_DRAW_BASE_SYMBOL
      || !legalIdentifier.test(IMMEDIATE_DRAW_BASE_SYMBOL)) {
    throw new Error(
      `Three dev overlay requires the legal WGSL symbol ${EXPECTED_IMMEDIATE_DRAW_BASE_SYMBOL}; shared transform exports ${IMMEDIATE_DRAW_BASE_SYMBOL}`,
    );
  }
}

export async function inspectThreeImmediateDevSource(
  sourceRoot = DEFAULT_THREE_DEV_SOURCE_ROOT,
) {
  const resolvedSourceRoot = path.resolve(sourceRoot);
  const targetHashes = await verifyTargetHashes(resolvedSourceRoot);

  const packageBytes = await readFile(path.join(resolvedSourceRoot, 'package.json'));
  let packageMetadata;
  try {
    packageMetadata = JSON.parse(canonicalText(packageBytes, 'package.json'));
  } catch (error) {
    throw new Error(
      `Cannot parse Three dev package metadata at ${path.join(resolvedSourceRoot, 'package.json')}`,
      { cause: error },
    );
  }
  verifyPackageMetadata(packageMetadata);

  const constantsSource = canonicalText(
    await readFile(path.join(resolvedSourceRoot, 'src', 'constants.js')),
    'src/constants.js',
  );
  const revisionMatches = [...constantsSource.matchAll(
    /^export const REVISION = '([^']+)';$/gm,
  )];
  if (revisionMatches.length !== 1
      || revisionMatches[0][1] !== EXPECTED_THREE_DEV_REVISION) {
    throw new Error(
      `Three dev overlay requires runtime revision ${EXPECTED_THREE_DEV_REVISION}; found ${revisionMatches.length === 1 ? revisionMatches[0][1] : '<missing-or-ambiguous>'}`,
    );
  }

  const topLevel = (await runGit(
    resolvedSourceRoot,
    ['rev-parse', '--show-toplevel'],
  )).trim();
  if (!comparePaths(topLevel, resolvedSourceRoot)) {
    throw new Error(
      `Three dev overlay source root must be the Git worktree root: expected ${resolvedSourceRoot}, received ${topLevel}`,
    );
  }

  const commit = (await runGit(
    resolvedSourceRoot,
    ['rev-parse', '--verify', 'HEAD'],
  )).trim();
  if (commit !== EXPECTED_THREE_DEV_COMMIT) {
    throw new Error(
      `Three dev overlay commit mismatch: expected ${EXPECTED_THREE_DEV_COMMIT}, received ${commit}`,
    );
  }

  const tree = (await runGit(
    resolvedSourceRoot,
    ['rev-parse', 'HEAD^{tree}'],
  )).trim();
  if (tree !== EXPECTED_THREE_DEV_TREE) {
    throw new Error(
      `Three dev overlay tree mismatch: expected ${EXPECTED_THREE_DEV_TREE}, received ${tree}`,
    );
  }

  const trackedStatus = await runGit(
    resolvedSourceRoot,
    ['status', '--porcelain=v1', '--untracked-files=no'],
  );
  if (trackedStatus.trim() !== '') {
    throw new Error(
      `Three dev overlay requires a clean tracked tree; status was ${JSON.stringify(trackedStatus.trim())}`,
    );
  }

  const fullStatus = await runGit(
    resolvedSourceRoot,
    ['status', '--porcelain=v1', '--untracked-files=all'],
  );
  if (fullStatus.trim() !== '') {
    throw new Error(
      `Three dev overlay requires a fully clean worktree; status was ${JSON.stringify(fullStatus.trim())}`,
    );
  }

  return {
    sourceRoot: resolvedSourceRoot,
    packageMetadata,
    runtimeRevision: revisionMatches[0][1],
    commit,
    tree,
    trackedStatus,
    fullStatus,
    ...targetHashes,
  };
}

function isAllowedRuntimePath(relativePath) {
  return relativePath === 'package.json'
    || relativePath === 'LICENSE'
    || relativePath === 'README.md'
    || relativePath.startsWith('build/')
    || relativePath.startsWith('src/')
    || relativePath.startsWith('examples/jsm/');
}

async function listRuntimePayload(sourceRoot) {
  const output = await runGit(
    sourceRoot,
    ['ls-files', '-s', '-z', '--', ...THREE_DEV_RUNTIME_PATHS],
  );
  const entries = output
    .split('\0')
    .filter((record) => record.length > 0)
    .map((record) => {
      const tabIndex = record.indexOf('\t');
      const metadata = record.slice(0, tabIndex);
      const relativePath = record.slice(tabIndex + 1);
      const match = /^([0-7]{6}) ([0-9a-f]{40,64}) ([0-3])$/.exec(metadata);
      if (tabIndex === -1 || match === null || match[3] !== '0') {
        throw new Error(`Cannot parse pinned Three dev payload entry ${JSON.stringify(record)}`);
      }
      if (!isAllowedRuntimePath(relativePath)) {
        throw new Error(`Three dev payload escaped the runtime allowlist: ${relativePath}`);
      }
      if (match[1] !== '100644' && match[1] !== '100755') {
        throw new Error(`Three dev payload contains a non-file entry: ${relativePath} (${match[1]})`);
      }
      return {
        path: relativePath,
        gitMode: match[1],
        gitBlob: match[2],
      };
    })
    .sort((left, right) => left.path.localeCompare(right.path, 'en'));

  const paths = new Set(entries.map((entry) => entry.path));
  if (paths.size !== entries.length || !paths.has('package.json')) {
    throw new Error('Pinned Three dev runtime payload is missing package.json or has duplicates');
  }
  for (const prefix of ['build/', 'src/', 'examples/jsm/']) {
    if (!entries.some((entry) => entry.path.startsWith(prefix))) {
      throw new Error(`Pinned Three dev runtime payload has no files beneath ${prefix}`);
    }
  }

  return entries;
}

function payloadIndexHash(entries) {
  const commitment = entries
    .map((entry) => `${entry.gitMode}\0${entry.gitBlob}\0${entry.path}\n`)
    .join('');
  return sha256(Buffer.from(commitment, 'utf8'));
}

async function copyRuntimePayload(sourceRoot, stagingRoot, entries) {
  // Materialize bytes from the pinned index with checkout conversion disabled.
  // A clean Windows worktree can contain CRLF-transformed files even though the
  // exact commit contains LF. Serving the canonical blobs makes the overlay's
  // source identity portable and avoids binding evidence to Git checkout policy.
  const prefix = `${stagingRoot.replaceAll('\\', '/')}/`;
  const batches = [];
  let batch = [];
  let batchLength = 0;
  for (const entry of entries) {
    const addedLength = entry.path.length + 1;
    if (batch.length > 0 && batchLength + addedLength > 12_000) {
      batches.push(batch);
      batch = [];
      batchLength = 0;
    }
    batch.push(entry.path);
    batchLength += addedLength;
  }
  if (batch.length > 0) batches.push(batch);

  for (const paths of batches) {
    await runGit(
      sourceRoot,
      ['checkout-index', '--force', `--prefix=${prefix}`, '--', ...paths],
      { config: ['core.autocrlf=false', 'core.eol=lf'] },
    );
  }

  for (const entry of entries) {
    const outputPath = path.join(stagingRoot, ...entry.path.split('/'));
    const outputStat = await lstat(outputPath);
    if (!outputStat.isFile() || outputStat.isSymbolicLink()) {
      throw new Error(`Three dev payload output is not a regular file: ${entry.path}`);
    }
  }
}

async function patchTargets(stagingRoot) {
  const patchedHashes = {};

  for (const relativePath of PATCH_TARGETS) {
    const targetPath = path.join(stagingRoot, ...relativePath.split('/'));
    const originalBytes = await readFile(targetPath);
    const canonicalSource = canonicalText(originalBytes, relativePath);
    const patched = transformThreeImmediateTarget(relativePath, canonicalSource);
    if (patched === canonicalSource) {
      throw new Error(`Three dev overlay transform made no change to ${relativePath}`);
    }
    if (patched.includes('__threeImmediateDrawBase')) {
      throw new Error(`Three dev overlay produced an illegal reserved WGSL symbol in ${relativePath}`);
    }

    const patchedHash = sha256(Buffer.from(patched, 'utf8'));
    const expectedHash = EXPECTED_THREE_DEV_PATCHED_HASHES[relativePath];
    if (patchedHash !== expectedHash) {
      throw new Error(
        `Three dev overlay patched hash mismatch for ${relativePath}: expected ${expectedHash}, received ${patchedHash}`,
      );
    }

    await writeFile(targetPath, patched, 'utf8');
    patchedHashes[relativePath] = patchedHash;
  }

  const wgslPath = path.join(
    stagingRoot,
    'src',
    'renderers',
    'webgpu',
    'nodes',
    'WGSLNodeBuilder.js',
  );
  const wgslSource = await readFile(wgslPath, 'utf8');
  const exactDeclaration = `var<immediate> ${EXPECTED_IMMEDIATE_DRAW_BASE_SYMBOL} : u32;`;
  if (!wgslSource.includes('requires immediate_address_space;')
      || !wgslSource.includes(exactDeclaration)) {
    throw new Error('Three dev overlay did not produce the exact legal WGSL declaration');
  }

  return patchedHashes;
}

async function inventoryOutput(stagingRoot, payloadEntries) {
  const files = [];
  let totalBytes = 0;

  for (const entry of payloadEntries) {
    const bytes = await readFile(path.join(stagingRoot, ...entry.path.split('/')));
    totalBytes += bytes.length;
    files.push({
      path: entry.path,
      gitMode: entry.gitMode,
      bytes: bytes.length,
      sha256: sha256(bytes),
    });
  }

  const commitment = files
    .map((file) => `${file.gitMode}\0${file.path}\0${file.bytes}\0${file.sha256}\n`)
    .join('');
  return {
    fileCount: files.length,
    totalBytes,
    inventorySha256: sha256(Buffer.from(commitment, 'utf8')),
    files,
  };
}

async function pathExists(targetPath) {
  try {
    await lstat(targetPath);
    return true;
  } catch (error) {
    if (error?.code === 'ENOENT') return false;
    throw error;
  }
}

async function installStagingDirectory(stagingRoot, outputRoot) {
  let backupRoot = null;
  if (await pathExists(outputRoot)) {
    backupRoot = `${outputRoot}.backup-${process.pid}-${randomBytes(6).toString('hex')}`;
    await rename(outputRoot, backupRoot);
  }

  try {
    await rename(stagingRoot, outputRoot);
  } catch (error) {
    if (backupRoot !== null) {
      try {
        await rename(backupRoot, outputRoot);
      } catch (restoreError) {
        throw new AggregateError(
          [error, restoreError],
          `Three dev overlay installation and rollback both failed; preserved prior output is at ${backupRoot}`,
        );
      }
    }
    throw error;
  }

  if (backupRoot !== null) {
    await rm(backupRoot, { recursive: true, force: true });
  }
}

export async function buildThreeImmediateDevOverlay({
  sourceRoot = DEFAULT_THREE_DEV_SOURCE_ROOT,
  outputRoot = DEFAULT_THREE_DEV_OUTPUT_ROOT,
} = {}) {
  const resolvedSourceRoot = path.resolve(sourceRoot);
  const resolvedOutputRoot = path.resolve(outputRoot);
  assertSafeRoots(resolvedSourceRoot, resolvedOutputRoot);
  assertLegalImmediateSymbol();

  // All source and identity verification happens before a staging or output
  // path is touched, preserving an existing overlay on any pin failure.
  const source = await inspectThreeImmediateDevSource(resolvedSourceRoot);
  const payloadEntries = await listRuntimePayload(resolvedSourceRoot);
  const outputParent = path.dirname(resolvedOutputRoot);
  await mkdir(outputParent, { recursive: true });
  const stagingRoot = await mkdtemp(path.join(
    outputParent,
    '.three-immediate-dev-overlay.tmp-',
  ));

  try {
    await copyRuntimePayload(resolvedSourceRoot, stagingRoot, payloadEntries);
    const patchedHashes = await patchTargets(stagingRoot);
    const outputInventory = await inventoryOutput(stagingRoot, payloadEntries);
    const manifest = {
      schemaVersion: 1,
      overlay: 'three-immediate-address-attribution-dev',
      source: {
        package: 'three',
        version: EXPECTED_THREE_DEV_VERSION,
        runtimeRevision: source.runtimeRevision,
        commit: source.commit,
        tree: source.tree,
        cleanTrackedTree: true,
        cleanFullWorktree: true,
        statusPorcelain: source.fullStatus,
        payloadGitIndexSha256: payloadIndexHash(payloadEntries),
        targetCanonicalSha256: source.canonicalHashes,
        targetWorktreeSha256: source.worktreeHashes,
      },
      transform: {
        reusedExport: 'transformThreeImmediateTarget',
        sourceModule: 'scripts/build-three-immediate-overlay.mjs',
        patchedTargetSha256: patchedHashes,
      },
      contract: {
        symbol: EXPECTED_IMMEDIATE_DRAW_BASE_SYMBOL,
        wgslRequirement: 'immediate_address_space',
        immediateSizeBytes: 4,
        geometryProperty: 'indirectImmediateBases',
        setter: 'setIndirect(indirect, indirectOffset, indirectImmediateBases)',
        servedEntrypoints: {
          'three/webgpu': 'build/three.webgpu.js',
          'three/tsl': 'build/three.tsl.js',
        },
        unpatchedUnexportedBuilds: ['build/three.webgpu.nodes.js'],
      },
      output: outputInventory,
    };

    await writeFile(
      path.join(stagingRoot, 'THREE_IMMEDIATE_DEV_OVERLAY.json'),
      `${JSON.stringify(manifest, null, 2)}\n`,
      'utf8',
    );
    await installStagingDirectory(stagingRoot, resolvedOutputRoot);

    return {
      outputRoot: resolvedOutputRoot,
      manifest,
    };
  } catch (error) {
    await rm(stagingRoot, { recursive: true, force: true });
    throw error;
  }
}

function isDirectExecution() {
  return process.argv[1] !== undefined
    && fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);
}

if (isDirectExecution()) {
  buildThreeImmediateDevOverlay()
    .then(({ outputRoot }) => {
      process.stdout.write(
        `Generated pinned Three dev immediate-data overlay at ${outputRoot}\n`,
      );
    })
    .catch((error) => {
      process.stderr.write(`${error.stack ?? error.message}\n`);
      process.exitCode = 1;
    });
}
