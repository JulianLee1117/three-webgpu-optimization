import { createHash } from 'node:crypto';
import {
  lstat,
  mkdir,
  open,
  readdir,
  realpath,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  verifyPublicLiveCandidatePairBundle,
} from '../analysis/verify-public-live-first-instance-candidate-pair.mjs';
import {
  PUBLIC_LIVE_PAIR_MANIFEST_NAME,
  auditPublicLivePairValue,
} from './public-live-first-instance-pair-policy.mjs';
import {
  PUBLIC_LIVE_PACKAGE_MANIFEST_NAME,
  PUBLIC_LIVE_PACKAGE_KIND,
  PUBLIC_LIVE_PACKAGE_LABEL,
  PUBLIC_LIVE_PACKAGE_MAX_ARCHIVE_BYTES,
  PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES,
  PUBLIC_LIVE_PACKAGE_PAIR_PREFIX,
  PUBLIC_LIVE_PACKAGE_POLICY_ID,
  PUBLIC_LIVE_PACKAGE_RUN_PREFIX,
  PUBLIC_LIVE_PACKAGE_SCHEMA_VERSION,
  compressDeterministicPublicLiveTar,
  encodeDeterministicPublicLiveTar,
  publicLivePackageArchiveFormat,
  publicLivePackageEncoderIdentity,
  publicLivePackageJsonBytes,
  publicLivePackageSha256,
  validatePublicLivePackageEntryDescriptors,
  validatePublicLivePackageEntries,
  validatePublicLivePackageManifest,
} from './public-live-first-instance-package-policy.mjs';

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function fail(message) {
  throw new Error(`Public live candidate-pair package build failed: ${message}`);
}

function compareCodePoints(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function pathIsWithin(parent, child) {
  const relative = path.relative(parent, child);
  return relative === ''
    || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative));
}

async function resolveProspectiveOutputPath(filename) {
  const missingSegments = [path.basename(filename)];
  let existingAncestor = path.dirname(filename);
  while (true) {
    try {
      const stats = await lstat(existingAncestor);
      if (!stats.isDirectory()) fail('output parent ancestor is not a directory.');
      const resolvedAncestor = await realpath(existingAncestor);
      return path.join(resolvedAncestor, ...missingSegments.reverse());
    } catch (error) {
      if (error?.code !== 'ENOENT') throw error;
      const parent = path.dirname(existingAncestor);
      if (parent === existingAncestor) fail('output path has no existing directory ancestor.');
      missingSegments.push(path.basename(existingAncestor));
      existingAncestor = parent;
    }
  }
}

async function assertNewPath(filename) {
  try {
    await lstat(filename);
  } catch (error) {
    if (error?.code === 'ENOENT') return;
    throw error;
  }
  fail('output archive already exists.');
}

async function resolveRequiredDirectory(directory, label) {
  const stats = await lstat(directory);
  if (!stats.isDirectory() || stats.isSymbolicLink()) {
    fail(`${label} must be a non-symbolic-link directory.`);
  }
  return realpath(directory);
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino;
}

async function readStableRegularFile(filename, label) {
  const before = await lstat(filename);
  if (!before.isFile() || before.isSymbolicLink()) {
    fail(`${label} is not a non-symbolic-link regular file.`);
  }
  const handle = await open(filename, 'r');
  try {
    const opened = await handle.stat();
    const pathAfterOpen = await lstat(filename);
    if (!opened.isFile()
      || !pathAfterOpen.isFile()
      || pathAfterOpen.isSymbolicLink()
      || !sameFileIdentity(before, opened)
      || !sameFileIdentity(opened, pathAfterOpen)) {
      fail(`${label} changed identity while it was opened.`);
    }
    if (opened.size > PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES) {
      fail(`${label} exceeds the frozen individual-file package limit.`);
    }
    const contents = await handle.readFile();
    const afterRead = await handle.stat();
    const pathAfterRead = await lstat(filename);
    if (!pathAfterRead.isFile()
      || pathAfterRead.isSymbolicLink()
      || !sameFileIdentity(opened, afterRead)
      || !sameFileIdentity(afterRead, pathAfterRead)
      || opened.size !== afterRead.size
      || opened.mtimeMs !== afterRead.mtimeMs
      || contents.length !== afterRead.size) {
      fail(`${label} changed while it was read.`);
    }
    return contents;
  } finally {
    await handle.close();
  }
}

async function hashStableRegularFile(filename, label) {
  const before = await lstat(filename);
  if (!before.isFile() || before.isSymbolicLink()) {
    fail(`${label} is not a non-symbolic-link regular file.`);
  }
  if (before.size > PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES) {
    fail(`${label} exceeds the frozen individual-file package limit.`);
  }
  const handle = await open(filename, 'r');
  try {
    const opened = await handle.stat();
    const pathAfterOpen = await lstat(filename);
    if (!opened.isFile()
      || !pathAfterOpen.isFile()
      || pathAfterOpen.isSymbolicLink()
      || !sameFileIdentity(before, opened)
      || !sameFileIdentity(opened, pathAfterOpen)
      || opened.size !== before.size) {
      fail(`${label} changed identity while it was opened.`);
    }
    const hash = createHash('sha256');
    const chunk = Buffer.allocUnsafe(Math.min(1024 * 1024, Math.max(1, opened.size)));
    let position = 0;
    while (position < opened.size) {
      const length = Math.min(chunk.length, opened.size - position);
      const { bytesRead } = await handle.read(chunk, 0, length, position);
      if (bytesRead !== length) fail(`${label} became truncated while it was hashed.`);
      hash.update(chunk.subarray(0, bytesRead));
      position += bytesRead;
    }
    const afterRead = await handle.stat();
    const pathAfterRead = await lstat(filename);
    if (!pathAfterRead.isFile()
      || pathAfterRead.isSymbolicLink()
      || !sameFileIdentity(opened, afterRead)
      || !sameFileIdentity(afterRead, pathAfterRead)
      || opened.size !== afterRead.size
      || opened.mtimeMs !== afterRead.mtimeMs) {
      fail(`${label} changed while it was hashed.`);
    }
    return { bytes: opened.size, sha256: hash.digest('hex') };
  } finally {
    await handle.close();
  }
}

async function snapshotFlatDirectory(directory, label, { retainContents = true } = {}) {
  const records = [];
  const entries = await readdir(directory, { withFileTypes: true });
  entries.sort((left, right) => compareCodePoints(left.name, right.name));
  for (const entry of entries) {
    if (!entry.isFile() || entry.isSymbolicLink()
      || entry.name === '.' || entry.name === '..'
      || entry.name.includes('/') || entry.name.includes('\\')) {
      fail(`${label} contains a non-regular or unsafe entry.`);
    }
    const filename = path.join(directory, entry.name);
    const stable = retainContents
      ? await readStableRegularFile(filename, `${label}/${entry.name}`)
      : await hashStableRegularFile(filename, `${label}/${entry.name}`);
    const contents = retainContents ? stable : null;
    records.push({
      name: entry.name,
      bytes: retainContents ? contents.length : stable.bytes,
      sha256: retainContents ? publicLivePackageSha256(contents) : stable.sha256,
      ...(retainContents ? { contents } : {}),
    });
  }
  return records;
}

async function preflightInputDirectory(directory, label, archivePrefix) {
  const descriptors = [];
  const entries = await readdir(directory, { withFileTypes: true });
  entries.sort((left, right) => compareCodePoints(left.name, right.name));
  for (const entry of entries) {
    if (!entry.isFile() || entry.isSymbolicLink()
      || entry.name === '.' || entry.name === '..'
      || entry.name.includes('/') || entry.name.includes('\\')) {
      fail(`${label} contains a non-regular or unsafe entry.`);
    }
    const filename = path.join(directory, entry.name);
    const stats = await lstat(filename);
    if (!stats.isFile() || stats.isSymbolicLink()) {
      fail(`${label} contains a non-regular or symbolic-link entry.`);
    }
    if (stats.size > PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES) {
      fail(`${label}/${entry.name} exceeds the frozen individual-file package limit.`);
    }
    descriptors.push({ name: `${archivePrefix}/${entry.name}`, bytes: stats.size });
  }
  return descriptors;
}

async function preflightPackageInputs(pairDirectory, runDirectories) {
  const descriptors = [
    { name: PUBLIC_LIVE_PACKAGE_MANIFEST_NAME, bytes: 0 },
    ...await preflightInputDirectory(
      pairDirectory,
      'public pair bundle',
      PUBLIC_LIVE_PACKAGE_PAIR_PREFIX,
    ),
  ];
  for (const [index, directory] of runDirectories.entries()) {
    descriptors.push(...await preflightInputDirectory(
      directory,
      `public run ${index + 1}`,
      `${PUBLIC_LIVE_PACKAGE_RUN_PREFIX}/matrix-${index + 1}`,
    ));
  }
  descriptors.sort((left, right) => compareCodePoints(left.name, right.name));
  validatePublicLivePackageEntryDescriptors(descriptors);
}

function snapshotIdentity(snapshot) {
  return snapshot.map(({ name, bytes, sha256 }) => ({ name, bytes, sha256 }));
}

function sameSnapshot(left, right) {
  return JSON.stringify(snapshotIdentity(left)) === JSON.stringify(snapshotIdentity(right));
}

function recordByName(snapshot, name, label) {
  const record = snapshot.find((candidate) => candidate.name === name);
  if (record === undefined) fail(`${label} lacks ${name}.`);
  return record;
}

function parseJson(record, label) {
  try {
    return JSON.parse(record.contents.toString('utf8'));
  } catch {
    fail(`${label} is not valid JSON.`);
  }
}

function verifiedRunByOrdinal(verification, matrixOrdinal) {
  const result = verification.publicRuns?.find(
    (run) => run.matrixOrdinal === matrixOrdinal,
  );
  if (result === undefined) fail(`pair verification lacks matrix ${matrixOrdinal}.`);
  return result;
}

export async function buildPublicLiveCandidatePairPackage(
  pairBundleDirectory,
  publicRunDirectories,
  outputArchive,
  {
    repositoryRoot = PROJECT_ROOT,
    pairVerifier = verifyPublicLiveCandidatePairBundle,
    maximumArchiveBytes = PUBLIC_LIVE_PACKAGE_MAX_ARCHIVE_BYTES,
  } = {},
) {
  if (!Array.isArray(publicRunDirectories) || publicRunDirectories.length !== 2) {
    fail('exactly two public run directories are required.');
  }
  if (!Number.isSafeInteger(maximumArchiveBytes)
    || maximumArchiveBytes <= 0
    || maximumArchiveBytes > PUBLIC_LIVE_PACKAGE_MAX_ARCHIVE_BYTES) {
    fail('maximumArchiveBytes exceeds the frozen publication limit.');
  }
  const requestedPairDirectory = path.resolve(pairBundleDirectory);
  const requestedRunDirectories = publicRunDirectories.map(
    (directory) => path.resolve(directory),
  );
  const requestedArchivePath = path.resolve(outputArchive);
  if (!requestedArchivePath.toLowerCase().endsWith('.tar.br')) {
    fail('output archive must use the .tar.br extension.');
  }
  const pairDirectory = await resolveRequiredDirectory(
    requestedPairDirectory,
    'public pair bundle',
  );
  const runDirectories = await Promise.all(requestedRunDirectories.map(
    (directory, index) => resolveRequiredDirectory(directory, `public run ${index + 1}`),
  ));
  if (new Set(runDirectories).size !== 2) fail('public run directories must be distinct.');
  const prospectiveArchivePath = await resolveProspectiveOutputPath(requestedArchivePath);
  for (const directory of [pairDirectory, ...runDirectories]) {
    if (pathIsWithin(directory, prospectiveArchivePath)) {
      fail('output archive cannot be an input or an input descendant.');
    }
  }
  const requestedOutputParent = path.dirname(requestedArchivePath);
  await mkdir(requestedOutputParent, { recursive: true });
  const outputParent = await realpath(requestedOutputParent);
  const archivePath = path.join(outputParent, path.basename(requestedArchivePath));
  for (const directory of [pairDirectory, ...runDirectories]) {
    if (pathIsWithin(directory, archivePath)) {
      fail('output archive cannot be an input or an input descendant.');
    }
  }
  await assertNewPath(archivePath);
  await preflightPackageInputs(pairDirectory, runDirectories);

  const initialPairSnapshot = await snapshotFlatDirectory(
    pairDirectory,
    'public pair bundle',
    { retainContents: false },
  );
  const initialRunSnapshots = await Promise.all(runDirectories.map(
    (directory, index) => snapshotFlatDirectory(
      directory,
      `public run ${index + 1}`,
      { retainContents: false },
    ),
  ));

  // No archive bytes are constructed until the established public-pair verifier succeeds.
  const verification = await pairVerifier(pairDirectory, runDirectories, { repositoryRoot });
  if (verification?.status !== 'consistent'
    || verification?.kind !== 'first-instance-live-public-candidate-pair-verification'
    || !Array.isArray(verification.publicRuns)
    || verification.publicRuns.length !== 2) {
    fail('existing public-pair verifier did not return a consistent two-run result.');
  }

  const pairSnapshot = await snapshotFlatDirectory(pairDirectory, 'public pair bundle');
  const runSnapshots = await Promise.all(runDirectories.map(
    (directory, index) => snapshotFlatDirectory(directory, `public run ${index + 1}`),
  ));
  if (!sameSnapshot(initialPairSnapshot, pairSnapshot)
    || runSnapshots.some((snapshot, index) => !sameSnapshot(
      initialRunSnapshots[index],
      snapshot,
    ))) {
    fail('an input changed during established public-pair verification.');
  }

  const runsByOrdinal = new Map();
  const encoderIdentity = publicLivePackageEncoderIdentity();
  for (const [index, snapshot] of runSnapshots.entries()) {
    const metadata = parseJson(recordByName(snapshot, 'metadata.json', `public run ${index + 1}`),
      `public run ${index + 1} metadata.json`);
    const matrixOrdinal = metadata?.candidateSeriesReservation?.matrixOrdinal;
    const runId = metadata?.runId;
    if ((matrixOrdinal !== 1 && matrixOrdinal !== 2)
      || typeof runId !== 'string' || runId === ''
      || runsByOrdinal.has(matrixOrdinal)) {
      fail('verified public runs do not expose unique matrix ordinals 1 and 2.');
    }
    if (metadata?.environment?.node !== encoderIdentity.node) {
      fail(
        `matrix ${matrixOrdinal} Node identity differs from the package encoder runtime.`,
      );
    }
    const verified = verifiedRunByOrdinal(verification, matrixOrdinal);
    if (verified.runId !== runId) {
      fail(`matrix ${matrixOrdinal} metadata differs from the pair-verifier run binding.`);
    }
    runsByOrdinal.set(matrixOrdinal, { snapshot, runId, verified });
  }
  if (runsByOrdinal.size !== 2) fail('verified public runs do not cover both matrices.');

  const archiveEntries = new Map();
  const sourceFiles = [];
  for (const record of pairSnapshot) {
    const archiveEntryPath = `${PUBLIC_LIVE_PACKAGE_PAIR_PREFIX}/${record.name}`;
    archiveEntries.set(archiveEntryPath, record.contents);
    sourceFiles.push({
      archivePath: archiveEntryPath,
      role: 'verified public candidate-pair receipt bundle',
      bytes: record.bytes,
      sha256: record.sha256,
    });
  }
  const publicRuns = [];
  for (const matrixOrdinal of [1, 2]) {
    const { snapshot, runId, verified } = runsByOrdinal.get(matrixOrdinal);
    const prefix = `${PUBLIC_LIVE_PACKAGE_RUN_PREFIX}/matrix-${matrixOrdinal}`;
    for (const record of snapshot) {
      const archiveEntryPath = `${prefix}/${record.name}`;
      archiveEntries.set(archiveEntryPath, record.contents);
      sourceFiles.push({
        archivePath: archiveEntryPath,
        role: `verified public-derived matrix-${matrixOrdinal} run artifact`,
        bytes: record.bytes,
        sha256: record.sha256,
      });
    }
    const artifactManifest = recordByName(
      snapshot,
      'artifact-manifest.json',
      `public matrix ${matrixOrdinal}`,
    );
    if (verified.publicArtifactManifest?.bytes !== artifactManifest.bytes
      || verified.publicArtifactManifest?.sha256 !== artifactManifest.sha256) {
      fail(`matrix ${matrixOrdinal} artifact manifest differs from pair verification.`);
    }
    publicRuns.push({
      matrixOrdinal,
      runId,
      directoryLabel: `matrix-${matrixOrdinal}`,
      artifactManifest: {
        archivePath: `${prefix}/artifact-manifest.json`,
        bytes: artifactManifest.bytes,
        sha256: artifactManifest.sha256,
      },
    });
  }
  sourceFiles.sort((left, right) => compareCodePoints(left.archivePath, right.archivePath));
  const pairManifestRecord = recordByName(
    pairSnapshot,
    PUBLIC_LIVE_PAIR_MANIFEST_NAME,
    'public pair bundle',
  );
  const packageManifest = validatePublicLivePackageManifest({
    schemaVersion: PUBLIC_LIVE_PACKAGE_SCHEMA_VERSION,
    kind: PUBLIC_LIVE_PACKAGE_KIND,
    policyId: PUBLIC_LIVE_PACKAGE_POLICY_ID,
    packageLabel: PUBLIC_LIVE_PACKAGE_LABEL,
    hashAlgorithm: 'sha256',
    archiveFormat: publicLivePackageArchiveFormat(),
    encoderIdentity,
    pairVerification: {
      schemaVersion: verification.schemaVersion,
      kind: verification.kind,
      status: verification.status,
      policyId: verification.policyId,
      seriesId: verification.seriesId,
      decision: structuredClone(verification.decision),
      receiptBundleManifest: {
        archivePath: `${PUBLIC_LIVE_PACKAGE_PAIR_PREFIX}/${PUBLIC_LIVE_PAIR_MANIFEST_NAME}`,
        bytes: pairManifestRecord.bytes,
        sha256: pairManifestRecord.sha256,
      },
    },
    publicRuns,
    files: sourceFiles,
  });
  auditPublicLivePairValue(packageManifest);
  const packageManifestBytes = publicLivePackageJsonBytes(packageManifest);
  archiveEntries.set(PUBLIC_LIVE_PACKAGE_MANIFEST_NAME, packageManifestBytes);
  const packageSize = validatePublicLivePackageEntries(archiveEntries);
  const tarBytes = encodeDeterministicPublicLiveTar(archiveEntries);
  if (tarBytes.length !== packageSize.tarBytes) {
    fail('canonical tar byte length differs from the pre-encoding package-size audit.');
  }
  const archiveBytes = compressDeterministicPublicLiveTar(tarBytes);
  if (archiveBytes.length >= maximumArchiveBytes) {
    fail(
      `compressed archive is ${archiveBytes.length} bytes and does not fit the `
      + `${maximumArchiveBytes}-byte single-file publication limit.`,
    );
  }

  const finalPairSnapshot = await snapshotFlatDirectory(
    pairDirectory,
    'public pair bundle',
    { retainContents: false },
  );
  const finalRunSnapshots = await Promise.all(runDirectories.map(
    (directory, index) => snapshotFlatDirectory(
      directory,
      `public run ${index + 1}`,
      { retainContents: false },
    ),
  ));
  if (!sameSnapshot(pairSnapshot, finalPairSnapshot)
    || finalRunSnapshots.some((snapshot, index) => !sameSnapshot(runSnapshots[index], snapshot))) {
    fail('an input changed during deterministic package construction.');
  }
  await assertNewPath(archivePath);
  await writeFile(archivePath, archiveBytes, { flag: 'wx' });

  return Object.freeze({
    schemaVersion: PUBLIC_LIVE_PACKAGE_SCHEMA_VERSION,
    kind: 'first-instance-live-public-candidate-pair-package-build-result',
    policyId: PUBLIC_LIVE_PACKAGE_POLICY_ID,
    packageLabel: PUBLIC_LIVE_PACKAGE_LABEL,
    status: 'created',
    seriesId: verification.seriesId,
    decision: structuredClone(verification.decision),
    sourceFileCount: sourceFiles.length,
    uncompressedTarBytes: tarBytes.length,
    archiveBytes: archiveBytes.length,
    archiveSha256: publicLivePackageSha256(archiveBytes),
    encoderIdentity,
    singleFilePublicationLimitBytes: maximumArchiveBytes,
    singleFilePublicationLimitSatisfied: true,
    outputLabel: 'public-candidate-pair.tar.br',
  });
}

export function serializePublicLiveCandidatePairPackageBuild(result) {
  return `${JSON.stringify(result, null, 2)}\n`;
}

async function main() {
  const [pairDirectory, firstPublicRun, secondPublicRun, outputArchive, ...extra] =
    process.argv.slice(2);
  if (!pairDirectory || !firstPublicRun || !secondPublicRun || !outputArchive
    || extra.length > 0) {
    process.stderr.write(
      'Usage: node scripts/build-public-live-first-instance-package.mjs '
      + '<public-pair-receipt-bundle> <public-run-1> <public-run-2> <new-output.tar.br>\n',
    );
    process.exitCode = 2;
    return;
  }
  const result = await buildPublicLiveCandidatePairPackage(
    pairDirectory,
    [firstPublicRun, secondPublicRun],
    outputArchive,
  );
  process.stdout.write(serializePublicLiveCandidatePairPackageBuild(result));
}

const invokedUrl = process.argv[1] === undefined
  ? null
  : pathToFileURL(path.resolve(process.argv[1])).href;

if (import.meta.url === invokedUrl) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
