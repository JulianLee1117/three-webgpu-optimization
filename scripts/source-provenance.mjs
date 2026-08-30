import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

async function runGit(projectRoot, args) {
  const { stdout } = await execFileAsync('git', args, {
    cwd: projectRoot,
    encoding: 'utf8',
    maxBuffer: 4 * 1024 * 1024,
    windowsHide: true,
  });
  return stdout;
}

async function runGitBytes(projectRoot, args) {
  const { stdout } = await execFileAsync('git', args, {
    cwd: projectRoot,
    encoding: 'buffer',
    maxBuffer: 4 * 1024 * 1024,
    windowsHide: true,
  });
  return Buffer.isBuffer(stdout) ? stdout : Buffer.from(stdout);
}

export async function sha256File(filename) {
  return createHash('sha256').update(await readFile(filename)).digest('hex');
}

export async function sha256TrackedFiles(projectRoot, relativePaths) {
  const digest = createHash('sha256');
  digest.update('three-webgpu-tracked-files-v1\0');
  const normalizedPaths = [...new Set(relativePaths.map((filename) => filename.replaceAll('\\', '/')))]
    .sort();

  for (const relativePath of normalizedPaths) {
    const absolutePath = path.resolve(projectRoot, relativePath);
    const projectRelative = path.relative(projectRoot, absolutePath);
    if (projectRelative === '..'
      || projectRelative.startsWith(`..${path.sep}`)
      || path.isAbsolute(projectRelative)) {
      throw new Error(`Tracked path escapes the project root: ${relativePath}`);
    }
    digest.update(`${Buffer.byteLength(relativePath)}:`);
    digest.update(relativePath);
    digest.update('\0');
    try {
      const contents = await readFile(absolutePath);
      digest.update(`${contents.length}:`);
      digest.update(contents);
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
      digest.update('missing');
    }
    digest.update('\0');
  }
  return digest.digest('hex');
}

export function summarizePorcelain(statusText) {
  let stagedChanges = 0;
  let unstagedChanges = 0;
  let untrackedFiles = 0;
  const entries = statusText.split(/\r?\n/).filter(Boolean);
  for (const line of entries) {
    if (line.startsWith('??')) {
      untrackedFiles += 1;
      continue;
    }
    if (line[0] && line[0] !== ' ') stagedChanges += 1;
    if (line[1] && line[1] !== ' ') unstagedChanges += 1;
  }
  return {
    dirty: stagedChanges + unstagedChanges + untrackedFiles > 0,
    stagedChanges,
    unstagedChanges,
    untrackedFiles,
    entryCount: entries.length,
  };
}

function redactProjectRoot(value, projectRoot) {
  const text = String(value);
  const candidates = [path.resolve(projectRoot), path.resolve(projectRoot).replaceAll('\\', '/')]
    .filter(Boolean)
    .sort((left, right) => right.length - left.length);
  return candidates.reduce(
    (redacted, candidate) => redacted.replace(
      new RegExp(candidate.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'),
      '[project-root]',
    ),
    text,
  );
}

function unavailableRecord(error, projectRoot) {
  return {
    schemaVersion: 1,
    status: 'unavailable',
    capturedAt: new Date().toISOString(),
    error: redactProjectRoot(error instanceof Error ? error.message : String(error), projectRoot),
  };
}

async function captureGitState(projectRoot) {
  const [commit, tree, ref, statusBytes, trackedBytes] = await Promise.all([
    runGit(projectRoot, ['rev-parse', '--verify', 'HEAD']),
    runGit(projectRoot, ['rev-parse', '--verify', 'HEAD^{tree}']),
    runGit(projectRoot, ['rev-parse', '--abbrev-ref', 'HEAD']),
    runGitBytes(projectRoot, ['status', '--porcelain=v1', '--untracked-files=all']),
    runGitBytes(projectRoot, ['ls-files', '-z']),
  ]);
  return {
    commit: commit.trim(),
    tree: tree.trim(),
    ref: ref.trim() === 'HEAD' ? null : ref.trim(),
    statusBytes,
    trackedBytes,
    porcelainSha256: createHash('sha256').update(statusBytes).digest('hex'),
    trackedListSha256: createHash('sha256').update(trackedBytes).digest('hex'),
  };
}

function gitStateMatches(left, right) {
  return left.commit === right.commit
    && left.tree === right.tree
    && left.ref === right.ref
    && left.statusBytes.equals(right.statusBytes)
    && left.trackedBytes.equals(right.trackedBytes);
}

export async function collectSourceProvenance(projectRoot, { allowUnavailable = false } = {}) {
  try {
    const topLevel = await runGit(projectRoot, ['rev-parse', '--show-toplevel']);
    if (path.resolve(topLevel.trim()) !== path.resolve(projectRoot)) {
      throw new Error('Git top level does not match the benchmark project root.');
    }
    const captureWindowStartedAt = new Date().toISOString();
    const maximumAttempts = 3;
    for (let captureAttempt = 1; captureAttempt <= maximumAttempts; captureAttempt += 1) {
      const before = await captureGitState(projectRoot);
      const trackedFiles = before.trackedBytes.toString('utf8').split('\0').filter(Boolean);
      const packageLockPath = path.join(projectRoot, 'package-lock.json');
      const [trackedFilesSha256, packageLockSha256] = await Promise.all([
        sha256TrackedFiles(projectRoot, trackedFiles),
        sha256File(packageLockPath),
      ]);
      const after = await captureGitState(projectRoot);
      const [verifiedTrackedFilesSha256, verifiedPackageLockSha256] = await Promise.all([
        sha256TrackedFiles(projectRoot, trackedFiles),
        sha256File(packageLockPath),
      ]);
      const verified = await captureGitState(projectRoot);
      if (!gitStateMatches(before, after)
        || !gitStateMatches(after, verified)
        || trackedFilesSha256 !== verifiedTrackedFilesSha256
        || packageLockSha256 !== verifiedPackageLockSha256) {
        continue;
      }

      const statusText = before.statusBytes.toString('utf8');
      const porcelain = summarizePorcelain(statusText);
      return {
        schemaVersion: 1,
        status: 'available',
        capturedAt: new Date().toISOString(),
        captureWindowStartedAt,
        captureStable: true,
        captureAttempts: captureAttempt,
        hashAlgorithm: 'sha256',
        trackedManifestFormat: 'sorted-path-and-working-bytes-v1',
        commit: before.commit,
        tree: before.tree,
        ref: before.ref,
        dirty: porcelain.dirty,
        stagedChanges: porcelain.stagedChanges,
        unstagedChanges: porcelain.unstagedChanges,
        untrackedFiles: porcelain.untrackedFiles,
        porcelainEntryCount: porcelain.entryCount,
        porcelainFormat: 'v1-with-untracked-files-all',
        porcelainByteCount: before.statusBytes.length,
        porcelainSha256: before.porcelainSha256,
        trackedListFormat: 'git-ls-files-z',
        trackedListByteCount: before.trackedBytes.length,
        trackedListSha256: before.trackedListSha256,
        trackedFileCount: trackedFiles.length,
        trackedFilesSha256,
        packageLockTracked: trackedFiles.includes('package-lock.json'),
        packageLockSha256,
      };
    }
    throw new Error('Git state or tracked working-tree bytes changed during three provenance capture attempts.');
  } catch (error) {
    if (allowUnavailable) return unavailableRecord(error, projectRoot);
    throw new Error(`Source provenance is unavailable: ${error instanceof Error ? error.message : String(error)}`);
  }
}

export function sourceProvenanceMatches(left, right) {
  if (left?.status !== 'available' || right?.status !== 'available') return false;
  return left.captureStable === true
    && right.captureStable === true
    && left.commit === right.commit
    && left.tree === right.tree
    && left.ref === right.ref
    && left.dirty === right.dirty
    && left.stagedChanges === right.stagedChanges
    && left.unstagedChanges === right.unstagedChanges
    && left.untrackedFiles === right.untrackedFiles
    && left.porcelainEntryCount === right.porcelainEntryCount
    && left.porcelainByteCount === right.porcelainByteCount
    && left.porcelainSha256 === right.porcelainSha256
    && left.trackedListByteCount === right.trackedListByteCount
    && left.trackedListSha256 === right.trackedListSha256
    && left.trackedFileCount === right.trackedFileCount
    && left.trackedFilesSha256 === right.trackedFilesSha256
    && left.packageLockTracked === right.packageLockTracked
    && left.packageLockSha256 === right.packageLockSha256;
}
