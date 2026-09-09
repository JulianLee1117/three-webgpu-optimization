import { execFile } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import {
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rename,
  rm,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const execFileAsync = promisify(execFile);

export const EXPECTED_THREE_DRAW_INDEX_COMMIT =
  '994260a7a59abe466de8e59db1ecc9e30352751d';
export const EXPECTED_THREE_DRAW_INDEX_TREE =
  '7f3d4a048ea19bef1a488b0e2e243e47f6b30002';
export const EXPECTED_THREE_DRAW_INDEX_SRC_TREE =
  'acfdb0cb4b58fcfd37917c45f83406652150efdb';
export const EXPECTED_THREE_DRAW_INDEX_VERSION = '0.185.0';
export const EXPECTED_THREE_DRAW_INDEX_REVISION = '186dev';
export const EXPECTED_THREE_DRAW_INDEX_OBJECT_FORMAT = 'sha1';
export const EXPECTED_THREE_DRAW_INDEX_PATCH_SHA256 =
  '98f469c576ee047ab6fd653f93bcaa1121869759ba5e5e3e63a7b94f26715b25';
export const EXPECTED_THREE_DRAW_INDEX_PATCH_BYTES = 16920;
export const EXPECTED_THREE_DRAW_INDEX_SOURCE_FILE_COUNT = 754;
export const EXPECTED_THREE_DRAW_INDEX_SOURCE_BYTES = 4643565;
export const EXPECTED_THREE_DRAW_INDEX_SOURCE_INDEX_SHA256 =
  'f51261f54f3d52b146ffeabf9107a016523b7deab5efee7b57045f79f84931f2';

export const THREE_DRAW_INDEX_TARGET_PATHS = Object.freeze([
  'src/nodes/accessors/Batch.js',
  'src/nodes/core/IndexNode.js',
  'src/nodes/core/NodeBuilder.js',
  'src/renderers/common/nodes/NodeBuilderState.js',
  'src/renderers/common/nodes/NodeManager.js',
  'src/renderers/webgpu/WebGPUBackend.js',
  'src/renderers/webgpu/descriptors/GPUPipelineLayoutDescriptor.js',
  'src/renderers/webgpu/nodes/WGSLNodeBuilder.js',
  'src/renderers/webgpu/utils/WebGPUPipelineUtils.js',
]);

export const THREE_DRAW_INDEX_RUNTIME_ALIASES = Object.freeze({
  '^three$': 'src/Three.js',
  '^three/webgpu$': 'src/Three.WebGPU.js',
  '^three/tsl$': 'src/Three.TSL.js',
});

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const DEFAULT_THREE_DRAW_INDEX_SOURCE_ROOT = path.resolve(
  repoRoot,
  '..',
  'upstream-three',
);
export const DEFAULT_THREE_DRAW_INDEX_OUTPUT_ROOT = path.join(
  repoRoot,
  '.generated',
  'three-draw-index-overlay',
);
export const THREE_DRAW_INDEX_PIN_PATH = path.join(
  repoRoot,
  'patches',
  'three-webgpu-draw-index-immediates.json',
);
export const THREE_DRAW_INDEX_PATCH_PATH = path.join(
  repoRoot,
  'patches',
  'three-webgpu-draw-index-immediates.patch',
);

const textDecoder = new TextDecoder('utf-8', { fatal: true });

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function gitBlobHash(bytes, objectFormat) {
  const header = Buffer.from(`blob ${bytes.length}\0`, 'utf8');
  return createHash(objectFormat).update(header).update(bytes).digest('hex');
}

function canonicalText(bytes, label) {
  try {
    return textDecoder.decode(bytes).replaceAll('\r\n', '\n');
  } catch (error) {
    throw new Error(`${label} is not valid UTF-8`, { cause: error });
  }
}

function comparePaths(left, right) {
  const leftResolved = path.resolve(left);
  const rightResolved = path.resolve(right);
  return process.platform === 'win32'
    ? leftResolved.toLowerCase() === rightResolved.toLowerCase()
    : leftResolved === rightResolved;
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
    throw new Error('Three draw-index overlay output cannot be a filesystem root');
  }
  if (comparePaths(source, output)
      || isPathInside(output, source)
      || isPathInside(source, output)) {
    throw new Error('Three draw-index overlay source and output must be disjoint');
  }
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected ${expected}, received ${actual}`);
  }
}

function assertExactObject(actual, expected, label) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${label} does not match the frozen contract`);
  }
}

async function runGit(root, args, {
  encoding = 'utf8',
  config = [],
} = {}) {
  const { stdout } = await execFileAsync(
    'git',
    [
      '-c',
      `safe.directory=${root}`,
      ...config.flatMap((value) => ['-c', value]),
      '-C',
      root,
      ...args,
    ],
    {
      encoding,
      maxBuffer: 64 * 1024 * 1024,
      windowsHide: true,
    },
  );
  return stdout;
}

async function initializeStagingRepository(stagingRoot) {
  return execFileAsync(
    'git',
    ['-c', 'init.defaultBranch=detached-overlay', '-C', stagingRoot, 'init', '--quiet'],
    {
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
      windowsHide: true,
    },
  );
}

async function runGitApply(stagingRoot, args) {
  return execFileAsync(
    'git',
    [
      '-c',
      'core.autocrlf=false',
      '-c',
      'core.eol=lf',
      '-C',
      stagingRoot,
      'apply',
      ...args,
      THREE_DRAW_INDEX_PATCH_PATH,
    ],
    {
      encoding: 'utf8',
      maxBuffer: 64 * 1024 * 1024,
      windowsHide: true,
    },
  );
}

async function readPinManifest() {
  const bytes = await readFile(THREE_DRAW_INDEX_PIN_PATH);
  let manifest;
  try {
    manifest = JSON.parse(canonicalText(bytes, 'Three draw-index pin manifest'));
  } catch (error) {
    throw new Error('Cannot parse the Three draw-index pin manifest', { cause: error });
  }

  assertEqual(manifest.schemaVersion, 1, 'pin schema version');
  assertEqual(manifest.experiment, 'three-webgpu-draw-index-immediates', 'pin experiment');
  assertEqual(manifest.upstream?.commit, EXPECTED_THREE_DRAW_INDEX_COMMIT, 'pin commit');
  assertEqual(manifest.upstream?.tree, EXPECTED_THREE_DRAW_INDEX_TREE, 'pin tree');
  assertEqual(manifest.upstream?.srcTree, EXPECTED_THREE_DRAW_INDEX_SRC_TREE, 'pin src tree');
  assertEqual(
    manifest.upstream?.objectFormat,
    EXPECTED_THREE_DRAW_INDEX_OBJECT_FORMAT,
    'pin object format',
  );
  assertEqual(
    manifest.upstream?.packageVersion,
    EXPECTED_THREE_DRAW_INDEX_VERSION,
    'pin package version',
  );
  assertEqual(
    manifest.upstream?.runtimeRevision,
    EXPECTED_THREE_DRAW_INDEX_REVISION,
    'pin runtime revision',
  );
  assertEqual(
    manifest.sourcePayload?.fileCount,
    EXPECTED_THREE_DRAW_INDEX_SOURCE_FILE_COUNT,
    'pin source file count',
  );
  assertEqual(
    manifest.sourcePayload?.totalBytes,
    EXPECTED_THREE_DRAW_INDEX_SOURCE_BYTES,
    'pin source byte count',
  );
  assertEqual(
    manifest.sourcePayload?.gitIndexSha256,
    EXPECTED_THREE_DRAW_INDEX_SOURCE_INDEX_SHA256,
    'pin source index hash',
  );
  assertEqual(
    manifest.patch?.path,
    'patches/three-webgpu-draw-index-immediates.patch',
    'pin patch path',
  );
  assertEqual(manifest.patch?.bytes, EXPECTED_THREE_DRAW_INDEX_PATCH_BYTES, 'pin patch bytes');
  assertEqual(
    manifest.patch?.sha256,
    EXPECTED_THREE_DRAW_INDEX_PATCH_SHA256,
    'pin patch hash',
  );
  assertExactObject(manifest.runtimeAliases, THREE_DRAW_INDEX_RUNTIME_ALIASES, 'runtime aliases');

  const targetPaths = Object.keys(manifest.targets ?? {}).sort();
  assertExactObject(targetPaths, [...THREE_DRAW_INDEX_TARGET_PATHS].sort(), 'pin target paths');

  for (const relativePath of targetPaths) {
    const target = manifest.targets[relativePath];
    for (const [field, length] of [
      ['gitBlob', 40],
      ['afterGitBlob', 40],
      ['beforeSha256', 64],
      ['afterSha256', 64],
    ]) {
      if (typeof target[field] !== 'string'
          || !new RegExp(`^[0-9a-f]{${length}}$`).test(target[field])) {
        throw new Error(`Invalid ${field} for patch target ${relativePath}`);
      }
    }
  }

  return manifest;
}

function inspectPatch(patchBytes, manifest) {
  assertEqual(patchBytes.length, EXPECTED_THREE_DRAW_INDEX_PATCH_BYTES, 'patch byte count');
  assertEqual(sha256(patchBytes), EXPECTED_THREE_DRAW_INDEX_PATCH_SHA256, 'patch SHA-256');

  if (patchBytes.includes(13)) {
    throw new Error('Three draw-index patch must use canonical LF line endings');
  }
  const patchText = canonicalText(patchBytes, 'Three draw-index patch');
  if (/^(new file mode|deleted file mode|rename from|rename to|copy from|copy to|GIT binary patch)/m.test(patchText)) {
    throw new Error('Three draw-index patch may only modify existing text files');
  }

  const headers = [...patchText.matchAll(
    /^diff --git a\/(\S+) b\/(\S+)\nindex ([0-9a-f]{40})\.\.([0-9a-f]{40}) 100644\n--- a\/(\S+)\n\+\+\+ b\/(\S+)$/gm,
  )];
  const paths = [];
  for (const match of headers) {
    const [, left, right, beforeBlob, afterBlob, oldPath, newPath] = match;
    if (left !== right || left !== oldPath || left !== newPath) {
      throw new Error(`Three draw-index patch path mismatch near ${left}`);
    }
    if (paths.includes(left)) {
      throw new Error(`Three draw-index patch repeats target ${left}`);
    }
    const target = manifest.targets[left];
    if (target === undefined) {
      throw new Error(`Three draw-index patch contains undeclared target ${left}`);
    }
    assertEqual(beforeBlob, target.gitBlob, `${left} patch preimage blob`);
    assertEqual(afterBlob, target.afterGitBlob, `${left} patch postimage blob`);
    paths.push(left);
  }

  assertExactObject(paths.sort(), [...THREE_DRAW_INDEX_TARGET_PATHS].sort(), 'patch target paths');
  return patchText;
}

async function listSourcePayload(sourceRoot) {
  const output = await runGit(
    sourceRoot,
    ['ls-files', '-s', '-z', '--', 'src'],
  );
  const entries = output
    .split('\0')
    .filter(Boolean)
    .map((record) => {
      const tabIndex = record.indexOf('\t');
      const metadata = record.slice(0, tabIndex);
      const relativePath = record.slice(tabIndex + 1);
      const match = /^([0-7]{6}) ([0-9a-f]{40}) ([0-3])$/.exec(metadata);
      if (tabIndex === -1 || match === null || match[3] !== '0') {
        throw new Error(`Cannot parse pinned source entry ${JSON.stringify(record)}`);
      }
      if (!relativePath.startsWith('src/')) {
        throw new Error(`Pinned source payload escaped src/: ${relativePath}`);
      }
      if (match[1] !== '100644' && match[1] !== '100755') {
        throw new Error(`Pinned source payload contains a non-file: ${relativePath}`);
      }
      return {
        path: relativePath,
        gitMode: match[1],
        gitBlob: match[2],
      };
    })
    .sort((left, right) => left.path.localeCompare(right.path, 'en'));

  if (new Set(entries.map((entry) => entry.path)).size !== entries.length) {
    throw new Error('Pinned source payload contains duplicate paths');
  }
  assertEqual(entries.length, EXPECTED_THREE_DRAW_INDEX_SOURCE_FILE_COUNT, 'source file count');

  const commitment = entries
    .map((entry) => `${entry.gitMode}\0${entry.gitBlob}\0${entry.path}\n`)
    .join('');
  assertEqual(
    sha256(Buffer.from(commitment, 'utf8')),
    EXPECTED_THREE_DRAW_INDEX_SOURCE_INDEX_SHA256,
    'source Git-index commitment',
  );
  return entries;
}

export async function inspectThreeDrawIndexSource(
  sourceRoot = DEFAULT_THREE_DRAW_INDEX_SOURCE_ROOT,
) {
  const resolvedSourceRoot = path.resolve(sourceRoot);
  const manifest = await readPinManifest();
  const patchBytes = await readFile(THREE_DRAW_INDEX_PATCH_PATH);
  inspectPatch(patchBytes, manifest);

  const topLevel = (await runGit(
    resolvedSourceRoot,
    ['rev-parse', '--show-toplevel'],
  )).trim();
  if (!comparePaths(topLevel, resolvedSourceRoot)) {
    throw new Error(
      `Three draw-index source must be the worktree root: expected ${resolvedSourceRoot}, received ${topLevel}`,
    );
  }

  const commit = (await runGit(
    resolvedSourceRoot,
    ['rev-parse', '--verify', 'HEAD'],
  )).trim();
  assertEqual(commit, EXPECTED_THREE_DRAW_INDEX_COMMIT, 'source commit');

  const tree = (await runGit(
    resolvedSourceRoot,
    ['rev-parse', 'HEAD^{tree}'],
  )).trim();
  assertEqual(tree, EXPECTED_THREE_DRAW_INDEX_TREE, 'source tree');

  const srcTree = (await runGit(
    resolvedSourceRoot,
    ['rev-parse', 'HEAD:src'],
  )).trim();
  assertEqual(srcTree, EXPECTED_THREE_DRAW_INDEX_SRC_TREE, 'source src tree');

  const objectFormat = (await runGit(
    resolvedSourceRoot,
    ['rev-parse', '--show-object-format'],
  )).trim();
  assertEqual(objectFormat, EXPECTED_THREE_DRAW_INDEX_OBJECT_FORMAT, 'source object format');

  const trackedStatus = await runGit(
    resolvedSourceRoot,
    ['status', '--porcelain=v1', '--untracked-files=no'],
  );
  if (trackedStatus.trim() !== '') {
    throw new Error(`Three draw-index source has tracked changes: ${trackedStatus.trim()}`);
  }

  const fullStatus = await runGit(
    resolvedSourceRoot,
    ['status', '--porcelain=v1', '--untracked-files=all'],
  );
  if (fullStatus.trim() !== '') {
    throw new Error(`Three draw-index source is not fully clean: ${fullStatus.trim()}`);
  }

  const packageBytes = await runGit(
    resolvedSourceRoot,
    ['show', 'HEAD:package.json'],
    { encoding: null },
  );
  const packageMetadata = JSON.parse(canonicalText(packageBytes, 'pinned package.json'));
  assertEqual(packageMetadata.name, 'three', 'source package name');
  assertEqual(packageMetadata.version, EXPECTED_THREE_DRAW_INDEX_VERSION, 'source package version');

  const constantsBytes = await runGit(
    resolvedSourceRoot,
    ['show', 'HEAD:src/constants.js'],
    { encoding: null },
  );
  const constantsSource = canonicalText(constantsBytes, 'pinned src/constants.js');
  const revisions = [...constantsSource.matchAll(/^export const REVISION = '([^']+)';$/gm)];
  if (revisions.length !== 1) {
    throw new Error('Pinned Three source has an ambiguous runtime revision');
  }
  assertEqual(revisions[0][1], EXPECTED_THREE_DRAW_INDEX_REVISION, 'source runtime revision');

  for (const relativePath of THREE_DRAW_INDEX_TARGET_PATHS) {
    const target = manifest.targets[relativePath];
    const blob = (await runGit(
      resolvedSourceRoot,
      ['rev-parse', `HEAD:${relativePath}`],
    )).trim();
    assertEqual(blob, target.gitBlob, `${relativePath} source blob`);
    const bytes = await runGit(
      resolvedSourceRoot,
      ['show', `HEAD:${relativePath}`],
      { encoding: null },
    );
    assertEqual(sha256(bytes), target.beforeSha256, `${relativePath} source SHA-256`);
  }

  const payloadEntries = await listSourcePayload(resolvedSourceRoot);
  return {
    sourceRoot: resolvedSourceRoot,
    commit,
    tree,
    srcTree,
    objectFormat,
    trackedStatus,
    fullStatus,
    packageMetadata,
    runtimeRevision: revisions[0][1],
    manifest,
    patchBytes,
    payloadEntries,
  };
}

async function materializeSource(sourceRoot, stagingRoot, entries) {
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
      if (!entry.isFile() || entry.isSymbolicLink()) {
        throw new Error(`Generated source contains a non-file entry: ${relativePath}`);
      }
      files.push(relativePath);
    }
  }
  return files;
}

async function verifyPreimage(stagingRoot, source, expectedPaths) {
  let totalBytes = 0;
  const entriesByPath = new Map(source.payloadEntries.map((entry) => [entry.path, entry]));
  const actualPaths = (await listFiles(stagingRoot))
    .sort((left, right) => left.localeCompare(right, 'en'));
  assertExactObject(actualPaths, expectedPaths, 'materialized source path set');

  for (const relativePath of actualPaths) {
    const absolutePath = path.join(stagingRoot, ...relativePath.split('/'));
    const stat = await lstat(absolutePath);
    if (!stat.isFile() || stat.isSymbolicLink()) {
      throw new Error(`Materialized source is not a regular file: ${relativePath}`);
    }
    const bytes = await readFile(absolutePath);
    totalBytes += bytes.length;
    const entry = entriesByPath.get(relativePath);
    assertEqual(
      gitBlobHash(bytes, source.objectFormat),
      entry.gitBlob,
      `${relativePath} materialized blob`,
    );
    const target = source.manifest.targets[relativePath];
    if (target !== undefined) {
      assertEqual(sha256(bytes), target.beforeSha256, `${relativePath} materialized preimage`);
    }
  }
  assertEqual(totalBytes, EXPECTED_THREE_DRAW_INDEX_SOURCE_BYTES, 'materialized source bytes');
}

async function verifyPostimageAndInventory(stagingRoot, source, expectedPaths) {
  const files = [];
  let totalBytes = 0;
  const entriesByPath = new Map(source.payloadEntries.map((entry) => [entry.path, entry]));
  const actualPaths = (await listFiles(stagingRoot))
    .sort((left, right) => left.localeCompare(right, 'en'));
  assertExactObject(actualPaths, expectedPaths, 'patched source path set');

  for (const relativePath of actualPaths) {
    const bytes = await readFile(path.join(stagingRoot, ...relativePath.split('/')));
    const sha = sha256(bytes);
    const entry = entriesByPath.get(relativePath);
    const target = source.manifest.targets[relativePath];
    if (target === undefined) {
      assertEqual(
        gitBlobHash(bytes, source.objectFormat),
        entry.gitBlob,
        `${relativePath} unchanged blob`,
      );
    } else {
      assertEqual(sha, target.afterSha256, `${relativePath} patched SHA-256`);
      assertEqual(
        gitBlobHash(bytes, source.objectFormat),
        target.afterGitBlob,
        `${relativePath} patched blob`,
      );
    }
    totalBytes += bytes.length;
    files.push({
      path: relativePath,
      gitMode: entry.gitMode,
      bytes: bytes.length,
      sha256: sha,
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

async function assertReplaceableOutput(outputRoot) {
  let outputStat;
  try {
    outputStat = await lstat(outputRoot);
  } catch (error) {
    if (error?.code === 'ENOENT') return;
    throw error;
  }

  if (!outputStat.isDirectory() || outputStat.isSymbolicLink()) {
    throw new Error(
      'Existing Three draw-index overlay output is not a recognized generated directory',
    );
  }

  const markerPath = path.join(outputRoot, 'THREE_DRAW_INDEX_OVERLAY.json');
  let markerStat;
  let markerBytes;
  try {
    markerStat = await lstat(markerPath);
    markerBytes = await readFile(markerPath);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      throw new Error(
        'Existing Three draw-index overlay output lacks its generated marker',
      );
    }
    throw error;
  }

  if (!markerStat.isFile() || markerStat.isSymbolicLink()) {
    throw new Error(
      'Existing Three draw-index overlay marker is not a regular file',
    );
  }

  let marker;
  try {
    marker = JSON.parse(canonicalText(markerBytes, 'existing overlay marker'));
  } catch (error) {
    throw new Error('Existing Three draw-index overlay marker is invalid', {
      cause: error,
    });
  }

  if (marker.schemaVersion !== 1
      || marker.overlay !== 'three-webgpu-draw-index-immediates'
      || marker.source?.package !== 'three') {
    throw new Error(
      'Existing Three draw-index overlay marker does not identify this generator',
    );
  }
}

async function installStagingDirectory(stagingRoot, outputRoot) {
  let backupRoot = null;
  if (await pathExists(outputRoot)) {
    await assertReplaceableOutput(outputRoot);
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
          `Overlay installation and rollback failed; prior output is at ${backupRoot}`,
        );
      }
    }
    throw error;
  }

  if (backupRoot !== null) {
    await rm(backupRoot, { recursive: true, force: true });
  }
}

export async function buildThreeDrawIndexOverlay({
  sourceRoot = DEFAULT_THREE_DRAW_INDEX_SOURCE_ROOT,
  outputRoot = DEFAULT_THREE_DRAW_INDEX_OUTPUT_ROOT,
} = {}) {
  const resolvedSourceRoot = path.resolve(sourceRoot);
  const resolvedOutputRoot = path.resolve(outputRoot);
  assertSafeRoots(resolvedSourceRoot, resolvedOutputRoot);

  // Verify every immutable input before creating staging or touching an
  // existing output directory.
  const source = await inspectThreeDrawIndexSource(resolvedSourceRoot);
  await assertReplaceableOutput(resolvedOutputRoot);
  const outputParent = path.dirname(resolvedOutputRoot);
  await mkdir(outputParent, { recursive: true });
  const stagingRoot = await mkdtemp(path.join(
    outputParent,
    '.three-draw-index-overlay.tmp-',
  ));

  try {
    const expectedPaths = source.payloadEntries.map((entry) => entry.path);
    await materializeSource(resolvedSourceRoot, stagingRoot, source.payloadEntries);
    await verifyPreimage(stagingRoot, source, expectedPaths);

    // The nested temporary repository prevents Git from discovering this
    // research repository (or the immutable source checkout) as the worktree.
    await initializeStagingRepository(stagingRoot);
    await runGitApply(stagingRoot, ['--check', '--whitespace=error-all']);
    await runGitApply(stagingRoot, ['--whitespace=error-all']);
    await runGitApply(stagingRoot, ['--check', '--reverse', '--whitespace=error-all']);
    await rm(path.join(stagingRoot, '.git'), { recursive: true, force: true });

    const outputInventory = await verifyPostimageAndInventory(
      stagingRoot,
      source,
      expectedPaths,
    );
    const generatedManifest = {
      schemaVersion: 1,
      overlay: 'three-webgpu-draw-index-immediates',
      source: {
        package: 'three',
        version: source.packageMetadata.version,
        runtimeRevision: source.runtimeRevision,
        commit: source.commit,
        tree: source.tree,
        srcTree: source.srcTree,
        objectFormat: source.objectFormat,
        cleanTrackedTree: true,
        cleanFullWorktree: true,
        statusPorcelain: source.fullStatus,
        payloadGitIndexSha256: EXPECTED_THREE_DRAW_INDEX_SOURCE_INDEX_SHA256,
      },
      patch: {
        path: 'patches/three-webgpu-draw-index-immediates.patch',
        bytes: source.patchBytes.length,
        sha256: sha256(source.patchBytes),
        changedPaths: [...THREE_DRAW_INDEX_TARGET_PATHS],
      },
      contract: {
        wgslRequirement: 'immediate_address_space',
        immediateSizeBytes: 4,
        shaderSymbol: 'nodeDrawIndex',
        runtimeAliases: THREE_DRAW_INDEX_RUNTIME_ALIASES,
      },
      output: outputInventory,
    };

    await writeFile(
      path.join(stagingRoot, 'THREE_DRAW_INDEX_OVERLAY.json'),
      `${JSON.stringify(generatedManifest, null, 2)}\n`,
      'utf8',
    );
    await installStagingDirectory(stagingRoot, resolvedOutputRoot);

    return {
      outputRoot: resolvedOutputRoot,
      manifest: generatedManifest,
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
  buildThreeDrawIndexOverlay()
    .then(({ outputRoot }) => {
      process.stdout.write(`Generated pinned Three draw-index overlay at ${outputRoot}\n`);
    })
    .catch((error) => {
      process.stderr.write(`${error.stack ?? error.message}\n`);
      process.exitCode = 1;
    });
}
