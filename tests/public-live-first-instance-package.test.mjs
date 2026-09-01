import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import {
  lstat,
  mkdir,
  mkdtemp,
  open,
  readFile,
  rm,
  symlink,
  writeFile,
} from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  brotliCompressSync,
  constants as zlibConstants,
} from 'node:zlib';

import {
  serializePublicLiveCandidatePairPackageVerification,
  verifyPublicLiveCandidatePairPackage,
} from '../analysis/verify-public-live-first-instance-package.mjs';
import {
  buildPublicLiveCandidatePairPackage,
  serializePublicLiveCandidatePairPackageBuild,
} from '../scripts/build-public-live-first-instance-package.mjs';
import {
  LIVE_PUBLIC_BUNDLE_LABEL,
  LIVE_PUBLIC_EVIDENCE_POLICY_ID,
} from '../scripts/live-evidence-sanitizer-policy.mjs';
import {
  PUBLIC_LIVE_PACKAGE_MANIFEST_NAME,
  PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES,
  PUBLIC_LIVE_PACKAGE_MAX_ENTRY_COUNT,
  PUBLIC_LIVE_PACKAGE_MAX_UNCOMPRESSED_BYTES,
  compressDeterministicPublicLiveTar,
  decompressDeterministicPublicLiveTar,
  encodeDeterministicPublicLiveTar,
  parseDeterministicPublicLiveTar,
  publicLivePackageSha256,
  requireSafePublicLivePackagePath,
  validatePublicLivePackageEntryDescriptors,
} from '../scripts/public-live-first-instance-package-policy.mjs';
import {
  PUBLIC_LIVE_PAIR_BUNDLE_FILES,
  PUBLIC_LIVE_PAIR_BUNDLE_LABEL,
  PUBLIC_LIVE_PAIR_LEDGER_NAME,
  PUBLIC_LIVE_PAIR_MANIFEST_NAME,
  PUBLIC_LIVE_PAIR_POLICY_ID,
  PUBLIC_LIVE_PAIR_POLICY_SCHEMA_VERSION,
  PUBLIC_LIVE_PAIR_RECEIPT_NAME,
  PUBLIC_LIVE_PAIR_REGISTRY_NAME,
} from '../scripts/public-live-first-instance-pair-policy.mjs';

const SERIES_ID = 'public-package-fixture-series';
const CANDIDATE_COMMIT = 'a'.repeat(40);
const CANDIDATE_TREE = 'b'.repeat(40);
const LEDGER_FINAL = 'c'.repeat(64);
const REGISTRY_FINAL = 'd'.repeat(64);
const PRIVATE_SENTINEL = 'C:\\Users\\private-owner\\secret-candidate.json';
const DECISION = Object.freeze({
  status: 'confirmation-not-met',
  pass: false,
  rule: 'both matrices independently pass all numerical, environment, and evidence gates',
});

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

async function temporaryRoot(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'public-live-package-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

async function writePublicRun(root, matrixOrdinal) {
  const directory = path.join(root, `public-run-${matrixOrdinal}`);
  await mkdir(directory);
  const runId = `public-fixture-run-${matrixOrdinal}`;
  const reservationEventSha256 = String(matrixOrdinal).repeat(64);
  const metadataBytes = jsonBytes({
    runId,
    environment: { node: process.version },
    candidateSeriesReservation: {
      matrixOrdinal,
      attemptOrdinal: matrixOrdinal,
      reservationEventSha256,
    },
  });
  const payloadBytes = jsonBytes({
    kind: 'public-fixture-payload',
    matrixOrdinal,
    visibility: 'public-derived-only',
  });
  await writeFile(path.join(directory, 'metadata.json'), metadataBytes);
  await writeFile(path.join(directory, 'payload.json'), payloadBytes);
  const files = [
    ['metadata.json', 'public run metadata', metadataBytes],
    ['payload.json', 'public fixture payload', payloadBytes],
  ].map(([name, role, bytes]) => ({
    name,
    role,
    required: true,
    present: true,
    bytes: bytes.length,
    sha256: sha256(bytes),
    absenceReason: null,
  }));
  const manifestBytes = jsonBytes({
    schemaVersion: 2,
    runId,
    hashAlgorithm: 'sha256',
    requiredFiles: ['metadata.json', 'payload.json'],
    optionalFiles: [],
    files,
    bundleProvenance: {
      schemaVersion: 2,
      policyId: LIVE_PUBLIC_EVIDENCE_POLICY_ID,
      bundleLabel: LIVE_PUBLIC_BUNDLE_LABEL,
      candidateSource: {
        commit: CANDIDATE_COMMIT,
        tree: CANDIDATE_TREE,
      },
    },
  });
  await writeFile(path.join(directory, 'artifact-manifest.json'), manifestBytes);
  return {
    directory,
    matrixOrdinal,
    attemptOrdinal: matrixOrdinal,
    runId,
    reservationEventSha256,
    publicArtifactManifest: {
      name: 'artifact-manifest.json',
      bytes: manifestBytes.length,
      sha256: sha256(manifestBytes),
    },
    publicBundlePolicyId: LIVE_PUBLIC_EVIDENCE_POLICY_ID,
    candidateCommit: CANDIDATE_COMMIT,
    candidateTree: CANDIDATE_TREE,
  };
}

async function writePairBundle(root, publicRuns) {
  const directory = path.join(root, 'public-pair');
  await mkdir(directory);
  const ledgerBytes = Buffer.from('{"kind":"fixture-ledger"}\n', 'utf8');
  const registryBytes = Buffer.from('{"kind":"fixture-registry"}\n', 'utf8');
  const receipt = {
    seriesId: SERIES_ID,
    ledgerBinding: {
      filename: PUBLIC_LIVE_PAIR_LEDGER_NAME,
      bytes: ledgerBytes.length,
      sha256: sha256(ledgerBytes),
      eventCount: 1,
      finalEventSha256: LEDGER_FINAL,
    },
    seriesRootRegistryBinding: {
      filename: PUBLIC_LIVE_PAIR_REGISTRY_NAME,
      bytes: registryBytes.length,
      sha256: sha256(registryBytes),
      eventCount: 1,
      finalEventSha256: REGISTRY_FINAL,
    },
    privatePairVerification: { decision: DECISION },
    publicRuns: publicRuns.map((run) => ({
      matrixOrdinal: run.matrixOrdinal,
      attemptOrdinal: run.attemptOrdinal,
      runId: run.runId,
      reservationEventSha256: run.reservationEventSha256,
      publicArtifactManifest: run.publicArtifactManifest,
      publicBundlePolicyId: run.publicBundlePolicyId,
      candidateCommit: run.candidateCommit,
      candidateTree: run.candidateTree,
    })),
  };
  const receiptBytes = jsonBytes(receipt);
  const sourceFiles = new Map([
    [PUBLIC_LIVE_PAIR_REGISTRY_NAME, registryBytes],
    [PUBLIC_LIVE_PAIR_LEDGER_NAME, ledgerBytes],
    [PUBLIC_LIVE_PAIR_RECEIPT_NAME, receiptBytes],
  ]);
  for (const [name, bytes] of sourceFiles) {
    await writeFile(path.join(directory, name), bytes);
  }
  const manifestBytes = jsonBytes({
    schemaVersion: PUBLIC_LIVE_PAIR_POLICY_SCHEMA_VERSION,
    kind: 'first-instance-live-public-candidate-pair-bundle-manifest',
    policyId: PUBLIC_LIVE_PAIR_POLICY_ID,
    bundleLabel: PUBLIC_LIVE_PAIR_BUNDLE_LABEL,
    hashAlgorithm: 'sha256',
    files: PUBLIC_LIVE_PAIR_BUNDLE_FILES.map(({ name, role }) => {
      const bytes = sourceFiles.get(name);
      return { name, role, bytes: bytes.length, sha256: sha256(bytes) };
    }),
  });
  await writeFile(path.join(directory, PUBLIC_LIVE_PAIR_MANIFEST_NAME), manifestBytes);
  return { directory, receipt, manifestBytes };
}

function fixturePairVerification(publicRuns, pairBundle) {
  return {
    schemaVersion: PUBLIC_LIVE_PAIR_POLICY_SCHEMA_VERSION,
    kind: 'first-instance-live-public-candidate-pair-verification',
    status: 'consistent',
    policyId: PUBLIC_LIVE_PAIR_POLICY_ID,
    seriesId: SERIES_ID,
    decision: DECISION,
    receiptBundle: {
      manifest: {
        name: PUBLIC_LIVE_PAIR_MANIFEST_NAME,
        bytes: pairBundle.manifestBytes.length,
        sha256: sha256(pairBundle.manifestBytes),
      },
    },
    publicRuns: publicRuns.map((run) => ({
      matrixOrdinal: run.matrixOrdinal,
      attemptOrdinal: run.attemptOrdinal,
      runId: run.runId,
      reservationEventSha256: run.reservationEventSha256,
      publicArtifactManifest: run.publicArtifactManifest,
      publicBundlePolicyId: run.publicBundlePolicyId,
      candidateCommit: run.candidateCommit,
      candidateTree: run.candidateTree,
    })),
  };
}

async function fixture(t) {
  const root = await temporaryRoot(t);
  const publicRuns = await Promise.all([1, 2].map(
    (matrixOrdinal) => writePublicRun(root, matrixOrdinal),
  ));
  const pairBundle = await writePairBundle(root, publicRuns);
  const pairVerification = fixturePairVerification(publicRuns, pairBundle);
  return { root, publicRuns, pairBundle, pairVerification };
}

function verifierDependencies() {
  return {
    receiptValidator: (value) => value,
    ledgerParser: () => ({
      seriesId: SERIES_ID,
      events: [{ eventSha256: LEDGER_FINAL }],
    }),
    registryParser: () => ({
      events: [{}],
      finalEventSha256: REGISTRY_FINAL,
    }),
    provenanceValidator: () => [],
  };
}

async function buildFixturePackage(t, outputName = 'candidate-pair.tar.br') {
  const value = await fixture(t);
  const output = path.join(value.root, outputName);
  const pairVerifier = async () => value.pairVerification;
  const result = await buildPublicLiveCandidatePairPackage(
    value.pairBundle.directory,
    value.publicRuns.map(({ directory }) => directory),
    output,
    { pairVerifier },
  );
  return { ...value, output, result, pairVerifier };
}

async function writeArchiveFromEntries(filename, entries) {
  const tarBytes = encodeDeterministicPublicLiveTar(entries);
  await writeFile(filename, compressDeterministicPublicLiveTar(tarBytes));
}

function rewriteTarChecksum(tarBytes, headerOffset = 0) {
  tarBytes.fill(0x20, headerOffset + 148, headerOffset + 156);
  const header = tarBytes.subarray(headerOffset, headerOffset + 512);
  const checksum = header.reduce((sum, byte) => sum + byte, 0);
  const encoded = `${checksum.toString(8).padStart(6, '0')}\0 `;
  Buffer.from(encoded, 'ascii').copy(tarBytes, headerOffset + 148);
}

test('builds byte-identical public pair packages and verifies them without extraction', async (t) => {
  const value = await fixture(t);
  const first = path.join(value.root, 'first.tar.br');
  const second = path.join(value.root, 'second.tar.br');
  const pairVerifier = async () => value.pairVerification;
  const directories = value.publicRuns.map(({ directory }) => directory);
  const firstResult = await buildPublicLiveCandidatePairPackage(
    value.pairBundle.directory,
    directories,
    first,
    { pairVerifier },
  );
  const secondResult = await buildPublicLiveCandidatePairPackage(
    value.pairBundle.directory,
    directories,
    second,
    { pairVerifier },
  );
  const [firstBytes, secondBytes] = await Promise.all([readFile(first), readFile(second)]);
  assert.deepEqual(firstBytes, secondBytes);
  assert.equal(firstResult.archiveSha256, publicLivePackageSha256(firstBytes));
  assert.equal(secondResult.archiveSha256, firstResult.archiveSha256);
  assert.equal(firstResult.singleFilePublicationLimitSatisfied, true);
  assert.equal(firstResult.outputLabel, 'public-candidate-pair.tar.br');
  assert.equal(serializePublicLiveCandidatePairPackageBuild(firstResult).includes(value.root), false);

  const verified = await verifyPublicLiveCandidatePairPackage(first, verifierDependencies());
  assert.equal(verified.status, 'consistent');
  assert.equal(verified.pathSafetyVerified, true);
  assert.equal(verified.canonicalTarMetadataVerified, true);
  assert.deepEqual(verified.recordedEncoderIdentity, {
    node: process.version,
    brotli: process.versions.brotli,
    zlib: process.versions.zlib,
  });
  assert.equal(verified.brotliStreamDecodedWithinLimits, true);
  assert.equal(
    verified.compressedByteDeterminismScope,
    'conditional-on-recorded-encoder-identity',
  );
  assert.equal(verified.sameEncoderCompressedByteReproductionChecked, false);
  assert.equal(verified.recordedEncoderMatchesPublicRunNodeIdentities, true);
  assert.equal(verified.embeddedReceiptBundleIntegrityVerified, true);
  assert.equal(verified.embeddedPublicRunIntegrityVerified, true);
  assert.equal(verified.existingPublicPairScientificDecisionReplayedFromArchive, false);
  assert.equal(verified.authenticityVerified, false);
  assert.equal(
    serializePublicLiveCandidatePairPackageVerification(verified).includes(value.root),
    false,
  );

  const tarText = decompressDeterministicPublicLiveTar(firstBytes).toString('utf8');
  assert.equal(tarText.includes(value.root), false);
  assert.equal(tarText.includes(PRIVATE_SENTINEL), false);
});

test('builder fails closed before writing when established pair verification fails', async (t) => {
  const value = await fixture(t);
  const output = path.join(value.root, 'rejected.tar.br');
  await assert.rejects(
    buildPublicLiveCandidatePairPackage(
      value.pairBundle.directory,
      value.publicRuns.map(({ directory }) => directory),
      output,
      { pairVerifier: async () => { throw new Error('fixture pair rejection'); } },
    ),
    /fixture pair rejection/,
  );
  await assert.rejects(lstat(output), { code: 'ENOENT' });
});

test('builder rejects a nested output without creating its missing parent', async (t) => {
  const value = await fixture(t);
  const forbiddenParent = path.join(value.publicRuns[0].directory, 'new-package-directory');
  const output = path.join(forbiddenParent, 'nested.tar.br');
  await assert.rejects(
    buildPublicLiveCandidatePairPackage(
      value.pairBundle.directory,
      value.publicRuns.map(({ directory }) => directory),
      output,
      { pairVerifier: async () => value.pairVerification },
    ),
    /output archive cannot be an input or an input descendant/,
  );
  await assert.rejects(lstat(forbiddenParent), { code: 'ENOENT' });
});

test('builder rejects an oversized source before pair verification or retained reads', async (t) => {
  const value = await fixture(t);
  const oversized = path.join(value.publicRuns[0].directory, 'oversized-public-artifact.bin');
  const handle = await open(oversized, 'w');
  try {
    await handle.truncate(PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES + 1);
  } finally {
    await handle.close();
  }
  let verifierCalled = false;
  const output = path.join(value.root, 'oversized-rejected.tar.br');
  await assert.rejects(
    buildPublicLiveCandidatePairPackage(
      value.pairBundle.directory,
      value.publicRuns.map(({ directory }) => directory),
      output,
      {
        pairVerifier: async () => {
          verifierCalled = true;
          return value.pairVerification;
        },
      },
    ),
    /exceeds the frozen individual-file package limit/,
  );
  assert.equal(verifierCalled, false);
  await assert.rejects(lstat(output), { code: 'ENOENT' });
});

test('builder preflight rejects excessive entry count before pair work', async (t) => {
  const value = await fixture(t);
  const existingSourceEntryCount = 10;
  const extrasNeeded = PUBLIC_LIVE_PACKAGE_MAX_ENTRY_COUNT - existingSourceEntryCount;
  for (let index = 0; index < extrasNeeded; index += 1) {
    await writeFile(
      path.join(value.publicRuns[0].directory, `extra-${String(index).padStart(2, '0')}.txt`),
      'x',
    );
  }
  let verifierCalled = false;
  const output = path.join(value.root, 'entry-count-rejected.tar.br');
  await assert.rejects(
    buildPublicLiveCandidatePairPackage(
      value.pairBundle.directory,
      value.publicRuns.map(({ directory }) => directory),
      output,
      {
        pairVerifier: async () => {
          verifierCalled = true;
          return value.pairVerification;
        },
      },
    ),
    /entry count/,
  );
  assert.equal(verifierCalled, false);
  await assert.rejects(lstat(output), { code: 'ENOENT' });
});

test('builder preflight rejects aggregate tar size before pair work', async (t) => {
  const value = await fixture(t);
  for (const [index, run] of value.publicRuns.entries()) {
    const filename = path.join(run.directory, `aggregate-limit-${index + 1}.bin`);
    const handle = await open(filename, 'w');
    try {
      await handle.truncate(PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES);
    } finally {
      await handle.close();
    }
  }
  let verifierCalled = false;
  const output = path.join(value.root, 'aggregate-rejected.tar.br');
  await assert.rejects(
    buildPublicLiveCandidatePairPackage(
      value.pairBundle.directory,
      value.publicRuns.map(({ directory }) => directory),
      output,
      {
        pairVerifier: async () => {
          verifierCalled = true;
          return value.pairVerification;
        },
      },
    ),
    /canonical tar size exceeds/,
  );
  assert.equal(verifierCalled, false);
  await assert.rejects(lstat(output), { code: 'ENOENT' });
});

test('builder rejects input mutation across established pair verification', async (t) => {
  const value = await fixture(t);
  const output = path.join(value.root, 'raced.tar.br');
  await assert.rejects(
    buildPublicLiveCandidatePairPackage(
      value.pairBundle.directory,
      value.publicRuns.map(({ directory }) => directory),
      output,
      {
        pairVerifier: async () => {
          await writeFile(path.join(value.publicRuns[0].directory, 'raced.txt'), 'raced\n');
          return value.pairVerification;
        },
      },
    ),
    /input changed during established public-pair verification/,
  );
  await assert.rejects(lstat(output), { code: 'ENOENT' });
});

test('builder rejects a public-run Node identity unlike its encoder', async (t) => {
  const value = await fixture(t);
  const metadataFilename = path.join(value.publicRuns[0].directory, 'metadata.json');
  const metadata = JSON.parse(await readFile(metadataFilename, 'utf8'));
  metadata.environment.node = 'v0.0.0';
  await writeFile(metadataFilename, jsonBytes(metadata));
  const output = path.join(value.root, 'node-mismatch.tar.br');
  await assert.rejects(
    buildPublicLiveCandidatePairPackage(
      value.pairBundle.directory,
      value.publicRuns.map(({ directory }) => directory),
      output,
      { pairVerifier: async () => value.pairVerification },
    ),
    /Node identity differs from the package encoder runtime/,
  );
  await assert.rejects(lstat(output), { code: 'ENOENT' });
});

test('archive verifier rejects changed bytes and undeclared files', async (t) => {
  const value = await buildFixturePackage(t);
  const archiveBytes = await readFile(value.output);
  const entries = parseDeterministicPublicLiveTar(
    decompressDeterministicPublicLiveTar(archiveBytes),
  );

  const tampered = new Map(entries);
  tampered.set('runs/matrix-1/payload.json', Buffer.from('changed\n', 'utf8'));
  const tamperedFilename = path.join(value.root, 'tampered.tar.br');
  await writeArchiveFromEntries(tamperedFilename, tampered);
  await assert.rejects(
    verifyPublicLiveCandidatePairPackage(tamperedFilename, verifierDependencies()),
    /differs from its recorded byte commitment/,
  );

  const extra = new Map(entries);
  extra.set('runs/matrix-1/undeclared.txt', Buffer.from('undeclared\n', 'utf8'));
  const extraFilename = path.join(value.root, 'extra.tar.br');
  await writeArchiveFromEntries(extraFilename, extra);
  await assert.rejects(
    verifyPublicLiveCandidatePairPackage(extraFilename, verifierDependencies()),
    /archive entries do not exactly match the package manifest/,
  );
});

test('archive verifier rejects unsafe paths and noncanonical tar metadata', async (t) => {
  const value = await buildFixturePackage(t);
  const archiveBytes = await readFile(value.output);
  const canonicalTar = decompressDeterministicPublicLiveTar(archiveBytes);

  const unsafeTar = Buffer.from(canonicalTar);
  unsafeTar.fill(0, 0, 100);
  Buffer.from('../escape', 'ascii').copy(unsafeTar, 0);
  rewriteTarChecksum(unsafeTar);
  const unsafeFilename = path.join(value.root, 'unsafe.tar.br');
  await writeFile(unsafeFilename, compressDeterministicPublicLiveTar(unsafeTar));
  await assert.rejects(
    verifyPublicLiveCandidatePairPackage(unsafeFilename, verifierDependencies()),
    /portable, normalized relative path/,
  );

  const noncanonicalTar = Buffer.from(canonicalTar);
  Buffer.from('0000600\0', 'ascii').copy(noncanonicalTar, 100);
  rewriteTarChecksum(noncanonicalTar);
  const noncanonicalFilename = path.join(value.root, 'noncanonical.tar.br');
  await writeFile(noncanonicalFilename, compressDeterministicPublicLiveTar(noncanonicalTar));
  await assert.rejects(
    verifyPublicLiveCandidatePairPackage(noncanonicalFilename, verifierDependencies()),
    /header metadata is not canonical/,
  );

  const symlinkTar = Buffer.from(canonicalTar);
  symlinkTar[156] = 0x32;
  rewriteTarChecksum(symlinkTar);
  const symlinkFilename = path.join(value.root, 'archive-symlink.tar.br');
  await writeFile(symlinkFilename, compressDeterministicPublicLiveTar(symlinkTar));
  await assert.rejects(
    verifyPublicLiveCandidatePairPackage(symlinkFilename, verifierDependencies()),
    /non-regular-file entry/,
  );

  assert.throws(() => requireSafePublicLivePackagePath('..\\private'),
    /portable, normalized relative path/);
});

test('builder rejects symbolic-link input entries', async (t) => {
  const value = await fixture(t);
  const target = path.join(value.root, 'link-target.txt');
  const link = path.join(value.publicRuns[0].directory, 'linked.txt');
  await writeFile(target, 'public fixture target\n');
  try {
    await symlink(target, link, 'file');
  } catch (error) {
    if (error?.code === 'EPERM') {
      t.skip('symbolic-link creation is unavailable on this Windows host');
      return;
    }
    throw error;
  }
  const output = path.join(value.root, 'symlink-rejected.tar.br');
  await assert.rejects(
    buildPublicLiveCandidatePairPackage(
      value.pairBundle.directory,
      value.publicRuns.map(({ directory }) => directory),
      output,
      { pairVerifier: async () => value.pairVerification },
    ),
    /non-regular or unsafe entry/,
  );
  await assert.rejects(lstat(output), { code: 'ENOENT' });
});

test('package manifest cannot be moved into an unrecognized namespace', async (t) => {
  const value = await buildFixturePackage(t);
  const entries = parseDeterministicPublicLiveTar(
    decompressDeterministicPublicLiveTar(await readFile(value.output)),
  );
  const manifest = JSON.parse(entries.get(PUBLIC_LIVE_PACKAGE_MANIFEST_NAME).toString('utf8'));
  manifest.files[0].archivePath = 'misc/candidate-attempts.jsonl';
  manifest.files.sort((left, right) => left.archivePath < right.archivePath ? -1 : 1);
  entries.set(PUBLIC_LIVE_PACKAGE_MANIFEST_NAME, jsonBytes(manifest));
  const filename = path.join(value.root, 'namespace.tar.br');
  await writeArchiveFromEntries(filename, entries);
  await assert.rejects(
    verifyPublicLiveCandidatePairPackage(filename, verifierDependencies()),
    /outside the frozen package namespaces|exact public-pair receipt-bundle files/,
  );
});

test('shared package policy rejects entry-count, individual-file, and total-size excess', () => {
  assert.throws(
    () => validatePublicLivePackageEntryDescriptors(Array.from(
      { length: PUBLIC_LIVE_PACKAGE_MAX_ENTRY_COUNT + 1 },
      (_, index) => ({ name: `entry-${String(index).padStart(2, '0')}`, bytes: 1 }),
    )),
    /entry count/,
  );
  assert.throws(
    () => validatePublicLivePackageEntryDescriptors([{
      name: 'oversized-file',
      bytes: PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES + 1,
    }]),
    /file limit/,
  );
  assert.throws(
    () => validatePublicLivePackageEntryDescriptors([
      { name: 'first-max-file', bytes: PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES },
      { name: 'second-max-file', bytes: PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES },
    ]),
    new RegExp(`canonical tar size exceeds the ${PUBLIC_LIVE_PACKAGE_MAX_UNCOMPRESSED_BYTES}`),
  );
});

test('encoder identity is bound to both runs without cross-runtime recompression', async (t) => {
  const value = await buildFixturePackage(t);
  const canonicalTar = decompressDeterministicPublicLiveTar(await readFile(value.output));
  const entries = parseDeterministicPublicLiveTar(canonicalTar);
  const manifest = JSON.parse(entries.get(PUBLIC_LIVE_PACKAGE_MANIFEST_NAME).toString('utf8'));
  manifest.encoderIdentity.node = 'v0.0.0';
  entries.set(PUBLIC_LIVE_PACKAGE_MANIFEST_NAME, jsonBytes(manifest));
  const mismatchedFilename = path.join(value.root, 'encoder-mismatch.tar.br');
  await writeArchiveFromEntries(mismatchedFilename, entries);
  await assert.rejects(
    verifyPublicLiveCandidatePairPackage(mismatchedFilename, verifierDependencies()),
    /Node identity differs from the recorded encoder/,
  );

  const differentlyCompressed = path.join(value.root, 'different-brotli-runtime.tar.br');
  await writeFile(differentlyCompressed, brotliCompressSync(canonicalTar, {
    params: {
      [zlibConstants.BROTLI_PARAM_MODE]: zlibConstants.BROTLI_MODE_GENERIC,
      [zlibConstants.BROTLI_PARAM_QUALITY]: 1,
      [zlibConstants.BROTLI_PARAM_LGWIN]: 18,
    },
  }));
  const verified = await verifyPublicLiveCandidatePairPackage(
    differentlyCompressed,
    verifierDependencies(),
  );
  assert.equal(verified.status, 'consistent');
  assert.equal(verified.sameEncoderCompressedByteReproductionChecked, false);
  assert.equal(
    verified.compressedByteDeterminismScope,
    'conditional-on-recorded-encoder-identity',
  );
});
