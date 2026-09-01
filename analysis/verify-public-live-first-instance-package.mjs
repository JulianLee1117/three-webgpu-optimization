import { createHash } from 'node:crypto';
import { lstat, open } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  parseCandidateLedgerText,
} from '../scripts/live-first-instance-candidate-ledger.mjs';
import {
  parseCandidateRegistryText,
} from '../scripts/live-first-instance-candidate-registry.mjs';
import {
  LIVE_PUBLIC_BUNDLE_LABEL,
  LIVE_PUBLIC_EVIDENCE_POLICY_ID,
  validateLivePublicBundleProvenanceShape,
} from '../scripts/live-evidence-sanitizer-policy.mjs';
import {
  PUBLIC_LIVE_PAIR_BUNDLE_FILES,
  PUBLIC_LIVE_PAIR_LEDGER_NAME,
  PUBLIC_LIVE_PAIR_MANIFEST_NAME,
  PUBLIC_LIVE_PAIR_RECEIPT_NAME,
  PUBLIC_LIVE_PAIR_REGISTRY_NAME,
  auditPublicLivePairValue,
  validatePublicLivePairBundleManifest,
  validatePublicLivePairReceipt,
} from '../scripts/public-live-first-instance-pair-policy.mjs';
import {
  PUBLIC_LIVE_PACKAGE_MANIFEST_NAME,
  PUBLIC_LIVE_PACKAGE_LABEL,
  PUBLIC_LIVE_PACKAGE_MAX_ARCHIVE_BYTES,
  PUBLIC_LIVE_PACKAGE_PAIR_PREFIX,
  PUBLIC_LIVE_PACKAGE_POLICY_ID,
  PUBLIC_LIVE_PACKAGE_RUN_PREFIX,
  PUBLIC_LIVE_PACKAGE_SCHEMA_VERSION,
  decompressDeterministicPublicLiveTar,
  parseDeterministicPublicLiveTar,
  publicLivePackageSha256,
  validatePublicLivePackageManifest,
} from '../scripts/public-live-first-instance-package-policy.mjs';

function reject(message) {
  throw new Error(`Public live candidate-pair package verification rejected: ${message}`);
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function sameFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino;
}

async function openStableArchive(filename) {
  const before = await lstat(filename);
  if (!before.isFile() || before.isSymbolicLink()) {
    reject('archive input must be a non-symbolic-link regular file.');
  }
  if (before.size <= 0 || before.size >= PUBLIC_LIVE_PACKAGE_MAX_ARCHIVE_BYTES) {
    reject('archive byte length is empty or exceeds the single-file publication limit.');
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
      reject('archive changed identity while it was opened.');
    }
    const archiveBytes = await handle.readFile();
    const afterRead = await handle.stat();
    if (!sameFileIdentity(opened, afterRead)
      || opened.size !== afterRead.size
      || opened.mtimeMs !== afterRead.mtimeMs
      || archiveBytes.length !== opened.size) {
      reject('archive changed while it was read.');
    }
    return { handle, opened, archiveBytes };
  } catch (error) {
    await handle.close();
    throw error;
  }
}

async function sha256OpenFile(handle, byteCount) {
  const hash = createHash('sha256');
  const chunk = Buffer.allocUnsafe(Math.min(1024 * 1024, byteCount));
  let position = 0;
  while (position < byteCount) {
    const length = Math.min(chunk.length, byteCount - position);
    const { bytesRead } = await handle.read(chunk, 0, length, position);
    if (bytesRead !== length) reject('archive became truncated during final verification.');
    hash.update(chunk.subarray(0, bytesRead));
    position += bytesRead;
  }
  return hash.digest('hex');
}

function parseJson(bytes, label) {
  try {
    return JSON.parse(bytes.toString('utf8'));
  } catch {
    reject(`${label} is not valid JSON.`);
  }
}

function archiveEntry(entries, name, label = name) {
  const bytes = entries.get(name);
  if (bytes === undefined) reject(`${label} is missing from the archive.`);
  return bytes;
}

function verifyBoundBytes(binding, bytes, label) {
  if (binding?.bytes !== bytes.length
    || binding?.sha256 !== publicLivePackageSha256(bytes)) {
    reject(`${label} differs from its recorded byte commitment.`);
  }
}

function verifyPackageFiles(entries, manifest) {
  const expected = new Set([
    PUBLIC_LIVE_PACKAGE_MANIFEST_NAME,
    ...manifest.files.map(({ archivePath }) => archivePath),
  ]);
  if (entries.size !== expected.size
    || [...entries.keys()].some((name) => !expected.has(name))) {
    reject('archive entries do not exactly match the package manifest.');
  }
  for (const file of manifest.files) {
    verifyBoundBytes(file, archiveEntry(entries, file.archivePath), file.archivePath);
  }
}

function verifyPairBundle(entries, packageManifest, {
  receiptValidator,
  ledgerParser,
  registryParser,
}) {
  const pairPath = (name) => `${PUBLIC_LIVE_PACKAGE_PAIR_PREFIX}/${name}`;
  const pairManifestBytes = archiveEntry(entries, pairPath(PUBLIC_LIVE_PAIR_MANIFEST_NAME));
  verifyBoundBytes(
    packageManifest.pairVerification.receiptBundleManifest,
    pairManifestBytes,
    'embedded public-pair bundle manifest',
  );
  const pairManifest = validatePublicLivePairBundleManifest(parseJson(
    pairManifestBytes,
    'embedded public-pair bundle manifest',
  ));
  for (const expected of PUBLIC_LIVE_PAIR_BUNDLE_FILES) {
    const binding = pairManifest.files.find(({ name }) => name === expected.name);
    if (binding === undefined) reject(`embedded public-pair manifest lacks ${expected.name}.`);
    verifyBoundBytes(
      binding,
      archiveEntry(entries, pairPath(expected.name)),
      `embedded public-pair ${expected.name}`,
    );
  }
  const expectedPairPaths = new Set([
    pairPath(PUBLIC_LIVE_PAIR_MANIFEST_NAME),
    ...PUBLIC_LIVE_PAIR_BUNDLE_FILES.map(({ name }) => pairPath(name)),
  ]);
  const actualPairPaths = [...entries.keys()].filter(
    (name) => name.startsWith(`${PUBLIC_LIVE_PACKAGE_PAIR_PREFIX}/`),
  );
  if (actualPairPaths.length !== expectedPairPaths.size
    || actualPairPaths.some((name) => !expectedPairPaths.has(name))) {
    reject('embedded public-pair receipt bundle contains an undeclared file.');
  }
  const receiptBytes = archiveEntry(entries, pairPath(PUBLIC_LIVE_PAIR_RECEIPT_NAME));
  const ledgerBytes = archiveEntry(entries, pairPath(PUBLIC_LIVE_PAIR_LEDGER_NAME));
  const registryBytes = archiveEntry(entries, pairPath(PUBLIC_LIVE_PAIR_REGISTRY_NAME));
  const receipt = receiptValidator(parseJson(receiptBytes, 'embedded public-pair receipt'));
  const ledger = ledgerParser(ledgerBytes.toString('utf8'));
  const registry = registryParser(registryBytes.toString('utf8'));
  if (receipt.seriesId !== packageManifest.pairVerification.seriesId
    || receipt.privatePairVerification?.decision?.status
      !== packageManifest.pairVerification.decision.status
    || receipt.privatePairVerification?.decision?.pass
      !== packageManifest.pairVerification.decision.pass
    || receipt.privatePairVerification?.decision?.rule
      !== packageManifest.pairVerification.decision.rule) {
    reject('embedded receipt differs from the package pair-verification decision.');
  }
  verifyBoundBytes(receipt.ledgerBinding, ledgerBytes, 'embedded candidate ledger');
  verifyBoundBytes(
    receipt.seriesRootRegistryBinding,
    registryBytes,
    'embedded candidate root registry',
  );
  if (ledger.seriesId !== receipt.seriesId
    || ledger.events?.length !== receipt.ledgerBinding.eventCount
    || ledger.finalEventSha256 !== receipt.ledgerBinding.finalEventSha256
    || registry.events?.length !== receipt.seriesRootRegistryBinding.eventCount
    || registry.finalEventSha256 !== receipt.seriesRootRegistryBinding.finalEventSha256) {
    reject('embedded registry or ledger hash chain differs from its receipt binding.');
  }
  return { receipt, ledger, registry, pairManifestBytes };
}

function verifyArtifactManifest(entries, run, receiptBinding, provenanceValidator) {
  const prefix = `${PUBLIC_LIVE_PACKAGE_RUN_PREFIX}/${run.directoryLabel}`;
  const manifestBytes = archiveEntry(entries, run.artifactManifest.archivePath);
  verifyBoundBytes(run.artifactManifest, manifestBytes,
    `embedded matrix ${run.matrixOrdinal} artifact manifest`);
  if (receiptBinding?.publicArtifactManifest?.bytes !== manifestBytes.length
    || receiptBinding?.publicArtifactManifest?.sha256
      !== publicLivePackageSha256(manifestBytes)) {
    reject(`embedded matrix ${run.matrixOrdinal} artifact manifest differs from the receipt.`);
  }
  const manifest = parseJson(
    manifestBytes,
    `embedded matrix ${run.matrixOrdinal} artifact manifest`,
  );
  if (!isRecord(manifest)
    || manifest.schemaVersion !== 2
    || manifest.hashAlgorithm !== 'sha256'
    || !Array.isArray(manifest.files)
    || !isRecord(manifest.bundleProvenance)
    || manifest.bundleProvenance.bundleLabel !== LIVE_PUBLIC_BUNDLE_LABEL
    || manifest.bundleProvenance.policyId !== LIVE_PUBLIC_EVIDENCE_POLICY_ID) {
    reject(`embedded matrix ${run.matrixOrdinal} is not a public-derived run manifest.`);
  }
  const provenanceReasons = provenanceValidator(manifest.bundleProvenance);
  if (!Array.isArray(provenanceReasons) || provenanceReasons.length !== 0) {
    reject(
      `embedded matrix ${run.matrixOrdinal} public provenance is invalid: `
      + `${Array.isArray(provenanceReasons) ? provenanceReasons.join('; ') : 'invalid result'}.`,
    );
  }
  const declared = new Set(['artifact-manifest.json']);
  const manifestNames = new Set();
  for (const [index, file] of manifest.files.entries()) {
    if (!isRecord(file)
      || typeof file.name !== 'string'
      || file.name === ''
      || file.name.includes('/')
      || file.name.includes('\\')
      || file.name === 'artifact-manifest.json'
      || manifestNames.has(file.name)
      || typeof file.present !== 'boolean') {
      reject(`embedded matrix ${run.matrixOrdinal} manifest file ${index} is unsafe.`);
    }
    manifestNames.add(file.name);
    if (!file.present) continue;
    declared.add(file.name);
    const bytes = archiveEntry(entries, `${prefix}/${file.name}`);
    verifyBoundBytes(file, bytes, `embedded matrix ${run.matrixOrdinal} ${file.name}`);
  }
  const actual = [...entries.keys()].filter((name) => name.startsWith(`${prefix}/`));
  const expected = [...declared].map((name) => `${prefix}/${name}`).sort();
  actual.sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    reject(`embedded matrix ${run.matrixOrdinal} files differ from its artifact manifest.`);
  }
  const metadata = parseJson(
    archiveEntry(entries, `${prefix}/metadata.json`),
    `embedded matrix ${run.matrixOrdinal} metadata`,
  );
  const reservation = metadata?.candidateSeriesReservation;
  if (manifest.runId !== run.runId
    || metadata?.runId !== run.runId
    || reservation?.matrixOrdinal !== run.matrixOrdinal
    || receiptBinding?.matrixOrdinal !== run.matrixOrdinal
    || receiptBinding?.attemptOrdinal !== reservation?.attemptOrdinal
    || receiptBinding?.reservationEventSha256 !== reservation?.reservationEventSha256
    || receiptBinding?.runId !== run.runId
    || receiptBinding?.publicBundlePolicyId !== LIVE_PUBLIC_EVIDENCE_POLICY_ID
    || receiptBinding?.candidateCommit
      !== manifest.bundleProvenance.candidateSource?.commit
    || receiptBinding?.candidateTree
      !== manifest.bundleProvenance.candidateSource?.tree) {
    reject(`embedded matrix ${run.matrixOrdinal} reservation or source binding differs from receipt.`);
  }
  return { manifest, metadata };
}

export async function verifyPublicLiveCandidatePairPackage(archiveFilename, {
  receiptValidator = validatePublicLivePairReceipt,
  ledgerParser = parseCandidateLedgerText,
  registryParser = parseCandidateRegistryText,
  provenanceValidator = validateLivePublicBundleProvenanceShape,
} = {}) {
  const filename = path.resolve(archiveFilename);
  const { handle, opened, archiveBytes } = await openStableArchive(filename);
  try {
    const archiveSha256 = publicLivePackageSha256(archiveBytes);
    const tarBytes = decompressDeterministicPublicLiveTar(archiveBytes);
    const entries = parseDeterministicPublicLiveTar(tarBytes);
    const manifestBytes = archiveEntry(entries, PUBLIC_LIVE_PACKAGE_MANIFEST_NAME);
    const packageManifest = validatePublicLivePackageManifest(parseJson(
      manifestBytes,
      PUBLIC_LIVE_PACKAGE_MANIFEST_NAME,
    ));
    auditPublicLivePairValue(packageManifest);
    if (packageManifest.policyId !== PUBLIC_LIVE_PACKAGE_POLICY_ID) {
      reject('package policy ID is unsupported.');
    }
    verifyPackageFiles(entries, packageManifest);
    const pair = verifyPairBundle(entries, packageManifest, {
      receiptValidator,
      ledgerParser,
      registryParser,
    });
    const receiptRuns = pair.receipt.publicRuns;
    if (!Array.isArray(receiptRuns) || receiptRuns.length !== 2) {
      reject('embedded receipt does not bind exactly two public runs.');
    }
    for (const run of packageManifest.publicRuns) {
      const receiptBinding = receiptRuns.find(
        (candidate) => candidate.matrixOrdinal === run.matrixOrdinal,
      );
      if (receiptBinding === undefined) {
        reject(`embedded receipt lacks matrix ${run.matrixOrdinal}.`);
      }
      const { metadata } = verifyArtifactManifest(
        entries,
        run,
        receiptBinding,
        provenanceValidator,
      );
      if (metadata?.environment?.node !== packageManifest.encoderIdentity.node) {
        reject(
          `embedded matrix ${run.matrixOrdinal} Node identity differs from the recorded encoder.`,
        );
      }
    }
    const finalDigest = await sha256OpenFile(handle, opened.size);
    const finalHandleStats = await handle.stat();
    const finalPathStats = await lstat(filename);
    if (!finalPathStats.isFile()
      || finalPathStats.isSymbolicLink()
      || !sameFileIdentity(opened, finalHandleStats)
      || !sameFileIdentity(finalHandleStats, finalPathStats)
      || finalHandleStats.size !== opened.size
      || finalHandleStats.mtimeMs !== opened.mtimeMs
      || finalDigest !== archiveSha256) {
      reject('archive changed during verification.');
    }
    return Object.freeze({
      schemaVersion: PUBLIC_LIVE_PACKAGE_SCHEMA_VERSION,
      kind: 'first-instance-live-public-candidate-pair-package-verification',
      policyId: PUBLIC_LIVE_PACKAGE_POLICY_ID,
      status: 'consistent',
      packageLabel: PUBLIC_LIVE_PACKAGE_LABEL,
      archiveBytes: archiveBytes.length,
      archiveSha256,
      uncompressedTarBytes: tarBytes.length,
      sourceFileCount: packageManifest.files.length,
      seriesId: packageManifest.pairVerification.seriesId,
      decision: structuredClone(packageManifest.pairVerification.decision),
      recordedEncoderIdentity: structuredClone(packageManifest.encoderIdentity),
      pathSafetyVerified: true,
      canonicalTarMetadataVerified: true,
      brotliStreamDecodedWithinLimits: true,
      compressedByteDeterminismScope: 'conditional-on-recorded-encoder-identity',
      sameEncoderCompressedByteReproductionChecked: false,
      recordedEncoderMatchesPublicRunNodeIdentities: true,
      sourceFileCommitmentsVerified: true,
      embeddedReceiptBundleIntegrityVerified: true,
      embeddedPublicRunIntegrityVerified: true,
      existingPublicPairVerifierRequiredBeforeBuild: true,
      existingPublicPairScientificDecisionReplayedFromArchive: false,
      authenticityVerified: false,
    });
  } finally {
    await handle.close();
  }
}

export function serializePublicLiveCandidatePairPackageVerification(result) {
  return `${JSON.stringify(result, null, 2)}\n`;
}

async function main() {
  const [archiveFilename, ...extra] = process.argv.slice(2);
  if (!archiveFilename || extra.length > 0) {
    process.stderr.write(
      'Usage: node analysis/verify-public-live-first-instance-package.mjs <package.tar.br>\n',
    );
    process.exitCode = 2;
    return;
  }
  const result = await verifyPublicLiveCandidatePairPackage(archiveFilename);
  process.stdout.write(serializePublicLiveCandidatePairPackageVerification(result));
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
