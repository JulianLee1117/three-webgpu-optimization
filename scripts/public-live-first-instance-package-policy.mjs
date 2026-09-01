import { createHash } from 'node:crypto';
import {
  brotliCompressSync,
  brotliDecompressSync,
  constants as zlibConstants,
} from 'node:zlib';

import {
  PUBLIC_LIVE_PAIR_BUNDLE_FILES,
  PUBLIC_LIVE_PAIR_MANIFEST_NAME,
  PUBLIC_LIVE_PAIR_POLICY_ID,
  PUBLIC_LIVE_PAIR_POLICY_SCHEMA_VERSION,
} from './public-live-first-instance-pair-policy.mjs';

export const PUBLIC_LIVE_PACKAGE_SCHEMA_VERSION = 1;
export const PUBLIC_LIVE_PACKAGE_POLICY_ID =
  'first-instance-live-public-candidate-pair-tar-br-v1';
export const PUBLIC_LIVE_PACKAGE_KIND =
  'first-instance-live-public-candidate-pair-package-manifest';
export const PUBLIC_LIVE_PACKAGE_LABEL = 'public-candidate-pair-lossless-package';
export const PUBLIC_LIVE_PACKAGE_MANIFEST_NAME =
  'public-candidate-pair-package-manifest.json';
export const PUBLIC_LIVE_PACKAGE_PAIR_PREFIX = 'pair';
export const PUBLIC_LIVE_PACKAGE_RUN_PREFIX = 'runs';
export const PUBLIC_LIVE_PACKAGE_MAX_ARCHIVE_BYTES = 100_000_000;
export const PUBLIC_LIVE_PACKAGE_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024;
export const PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES = 256 * 1024 * 1024;
export const PUBLIC_LIVE_PACKAGE_MAX_ENTRY_COUNT = 64;
export const PUBLIC_LIVE_PACKAGE_BROTLI_QUALITY = 9;
export const PUBLIC_LIVE_PACKAGE_BROTLI_LGWIN = 22;

const BLOCK_BYTES = 512;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const SAFE_ARCHIVE_PATH_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._/-]*$/;

function compareCodePoints(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function reject(message) {
  throw new Error(`Public live candidate-pair package policy rejected: ${message}`);
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function exactKeys(value, expected, label) {
  if (!isRecord(value)) reject(`${label} must be an object.`);
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (JSON.stringify(actual) !== JSON.stringify(wanted)) {
    reject(`${label} has an unexpected schema.`);
  }
  return value;
}

function nonemptyString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    reject(`${label} must be a non-empty string.`);
  }
  return value;
}

function safeInteger(value, label, minimum = 0) {
  if (!Number.isSafeInteger(value) || value < minimum) {
    reject(`${label} must be a safe integer no smaller than ${minimum}.`);
  }
  return value;
}

function sha256(value, label) {
  if (typeof value !== 'string' || !SHA256_PATTERN.test(value)) {
    reject(`${label} must be a lowercase SHA-256 digest.`);
  }
  return value;
}

export function publicLivePackageSha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

export function publicLivePackageJsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

export function publicLivePackageEncoderIdentity() {
  return Object.freeze({
    node: process.version,
    brotli: process.versions.brotli,
    zlib: process.versions.zlib,
  });
}

export function requireSafePublicLivePackagePath(value, label = 'archive path') {
  nonemptyString(value, label);
  if (!SAFE_ARCHIVE_PATH_PATTERN.test(value)
    || value.startsWith('/')
    || value.includes('\\')
    || value.split('/').some((part) => part === '' || part === '.' || part === '..')) {
    reject(`${label} must be a portable, normalized relative path.`);
  }
  if (Buffer.byteLength(value, 'utf8') > 100) {
    reject(`${label} exceeds the frozen ustar name field.`);
  }
  return value;
}

const ARCHIVE_FORMAT = Object.freeze({
  format: 'tar.br',
  tarFormat: 'ustar',
  entryOrder: 'unicode-code-point-ascending',
  regularFileMode: '0644',
  uid: 0,
  gid: 0,
  mtime: 0,
  compression: Object.freeze({
    algorithm: 'brotli',
    mode: 'generic',
    quality: PUBLIC_LIVE_PACKAGE_BROTLI_QUALITY,
    lgwin: PUBLIC_LIVE_PACKAGE_BROTLI_LGWIN,
  }),
});

export function publicLivePackageArchiveFormat() {
  return structuredClone(ARCHIVE_FORMAT);
}

export function validatePublicLivePackageManifest(value) {
  const manifest = exactKeys(value, [
    'schemaVersion',
    'kind',
    'policyId',
    'packageLabel',
    'hashAlgorithm',
    'archiveFormat',
    'encoderIdentity',
    'pairVerification',
    'publicRuns',
    'files',
  ], 'package manifest');
  if (manifest.schemaVersion !== PUBLIC_LIVE_PACKAGE_SCHEMA_VERSION
    || manifest.kind !== PUBLIC_LIVE_PACKAGE_KIND
    || manifest.policyId !== PUBLIC_LIVE_PACKAGE_POLICY_ID
    || manifest.packageLabel !== PUBLIC_LIVE_PACKAGE_LABEL
    || manifest.hashAlgorithm !== 'sha256') {
    reject('package manifest identity differs from the frozen policy.');
  }
  if (JSON.stringify(manifest.archiveFormat) !== JSON.stringify(ARCHIVE_FORMAT)) {
    reject('package manifest archive format differs from the frozen policy.');
  }
  const encoderIdentity = exactKeys(
    manifest.encoderIdentity,
    ['node', 'brotli', 'zlib'],
    'package manifest encoderIdentity',
  );
  if (!/^v[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$/.test(
    encoderIdentity.node ?? '',
  )) {
    reject('package manifest encoderIdentity.node is not a concrete Node version.');
  }
  nonemptyString(encoderIdentity.brotli, 'package manifest encoderIdentity.brotli');
  nonemptyString(encoderIdentity.zlib, 'package manifest encoderIdentity.zlib');
  const pair = exactKeys(manifest.pairVerification, [
    'schemaVersion', 'kind', 'status', 'policyId', 'seriesId', 'decision',
    'receiptBundleManifest',
  ], 'package manifest pairVerification');
  if (pair.schemaVersion !== PUBLIC_LIVE_PAIR_POLICY_SCHEMA_VERSION
    || pair.kind !== 'first-instance-live-public-candidate-pair-verification'
    || pair.status !== 'consistent'
    || pair.policyId !== PUBLIC_LIVE_PAIR_POLICY_ID) {
    reject('package manifest does not record the frozen public-pair verification identity.');
  }
  nonemptyString(pair.seriesId, 'package manifest pairVerification.seriesId');
  const decision = exactKeys(
    pair.decision,
    ['status', 'pass', 'rule'],
    'package manifest pairVerification.decision',
  );
  nonemptyString(decision.status, 'package manifest pairVerification.decision.status');
  if (typeof decision.pass !== 'boolean') {
    reject('package manifest pairVerification.decision.pass must be boolean.');
  }
  nonemptyString(decision.rule, 'package manifest pairVerification.decision.rule');
  const pairManifest = exactKeys(pair.receiptBundleManifest, [
    'archivePath', 'bytes', 'sha256',
  ], 'package manifest pairVerification.receiptBundleManifest');
  requireSafePublicLivePackagePath(
    pairManifest.archivePath,
    'package manifest pairVerification.receiptBundleManifest.archivePath',
  );
  if (pairManifest.archivePath
    !== `${PUBLIC_LIVE_PACKAGE_PAIR_PREFIX}/${PUBLIC_LIVE_PAIR_MANIFEST_NAME}`) {
    reject('package manifest public-pair manifest path differs from the frozen layout.');
  }
  safeInteger(pairManifest.bytes,
    'package manifest pairVerification.receiptBundleManifest.bytes', 1);
  sha256(pairManifest.sha256,
    'package manifest pairVerification.receiptBundleManifest.sha256');

  if (!Array.isArray(manifest.publicRuns) || manifest.publicRuns.length !== 2) {
    reject('package manifest must contain exactly two public runs.');
  }
  const matrixOrdinals = [];
  const runIds = new Set();
  for (const [index, run] of manifest.publicRuns.entries()) {
    const label = `package manifest publicRuns[${index}]`;
    exactKeys(run, [
      'matrixOrdinal', 'runId', 'directoryLabel', 'artifactManifest',
    ], label);
    safeInteger(run.matrixOrdinal, `${label}.matrixOrdinal`, 1);
    matrixOrdinals.push(run.matrixOrdinal);
    nonemptyString(run.runId, `${label}.runId`);
    if (runIds.has(run.runId)) reject(`${label}.runId is duplicated.`);
    runIds.add(run.runId);
    if (run.directoryLabel !== `matrix-${run.matrixOrdinal}`) {
      reject(`${label}.directoryLabel differs from its matrix ordinal.`);
    }
    const artifactManifest = exactKeys(
      run.artifactManifest,
      ['archivePath', 'bytes', 'sha256'],
      `${label}.artifactManifest`,
    );
    requireSafePublicLivePackagePath(
      artifactManifest.archivePath,
      `${label}.artifactManifest.archivePath`,
    );
    if (artifactManifest.archivePath
      !== `${PUBLIC_LIVE_PACKAGE_RUN_PREFIX}/${run.directoryLabel}/artifact-manifest.json`) {
      reject(`${label}.artifactManifest.archivePath differs from the frozen layout.`);
    }
    safeInteger(artifactManifest.bytes, `${label}.artifactManifest.bytes`, 1);
    sha256(artifactManifest.sha256, `${label}.artifactManifest.sha256`);
  }
  if (JSON.stringify(matrixOrdinals) !== JSON.stringify([1, 2])) {
    reject('package manifest public runs are not in exact matrix order 1, 2.');
  }

  if (!Array.isArray(manifest.files) || manifest.files.length === 0) {
    reject('package manifest files must be a non-empty array.');
  }
  const priorPaths = new Set();
  let priorPath = null;
  for (const [index, file] of manifest.files.entries()) {
    const label = `package manifest files[${index}]`;
    exactKeys(file, ['archivePath', 'role', 'bytes', 'sha256'], label);
    requireSafePublicLivePackagePath(file.archivePath, `${label}.archivePath`);
    if (file.archivePath === PUBLIC_LIVE_PACKAGE_MANIFEST_NAME) {
      reject('package manifest cannot list itself as a source file.');
    }
    if (priorPaths.has(file.archivePath)) reject(`${label}.archivePath is duplicated.`);
    if (priorPath !== null && compareCodePoints(priorPath, file.archivePath) >= 0) {
      reject('package manifest files are not in strict archive-path order.');
    }
    priorPaths.add(file.archivePath);
    priorPath = file.archivePath;
    nonemptyString(file.role, `${label}.role`);
    safeInteger(file.bytes, `${label}.bytes`);
    sha256(file.sha256, `${label}.sha256`);

    const pairPrefix = `${PUBLIC_LIVE_PACKAGE_PAIR_PREFIX}/`;
    const matrixOnePrefix = `${PUBLIC_LIVE_PACKAGE_RUN_PREFIX}/matrix-1/`;
    const matrixTwoPrefix = `${PUBLIC_LIVE_PACKAGE_RUN_PREFIX}/matrix-2/`;
    if (file.archivePath.startsWith(pairPrefix)) {
      if (file.role !== 'verified public candidate-pair receipt bundle') {
        reject(`${label}.role differs from the frozen pair-bundle role.`);
      }
    } else if (file.archivePath.startsWith(matrixOnePrefix)) {
      if (file.role !== 'verified public-derived matrix-1 run artifact') {
        reject(`${label}.role differs from the frozen matrix-1 role.`);
      }
    } else if (file.archivePath.startsWith(matrixTwoPrefix)) {
      if (file.role !== 'verified public-derived matrix-2 run artifact') {
        reject(`${label}.role differs from the frozen matrix-2 role.`);
      }
    } else {
      reject(`${label}.archivePath lies outside the frozen package namespaces.`);
    }
  }

  const expectedPairPaths = new Set([
    `${PUBLIC_LIVE_PACKAGE_PAIR_PREFIX}/${PUBLIC_LIVE_PAIR_MANIFEST_NAME}`,
    ...PUBLIC_LIVE_PAIR_BUNDLE_FILES.map(
      ({ name }) => `${PUBLIC_LIVE_PACKAGE_PAIR_PREFIX}/${name}`,
    ),
  ]);
  const actualPairPaths = new Set(manifest.files
    .map(({ archivePath }) => archivePath)
    .filter((archivePath) => archivePath.startsWith(`${PUBLIC_LIVE_PACKAGE_PAIR_PREFIX}/`)));
  if (actualPairPaths.size !== expectedPairPaths.size
    || [...actualPairPaths].some((archivePath) => !expectedPairPaths.has(archivePath))) {
    reject('package manifest does not contain the exact public-pair receipt-bundle files.');
  }
  for (const run of manifest.publicRuns) {
    if (!priorPaths.has(run.artifactManifest.archivePath)) {
      reject(`package manifest does not list matrix ${run.matrixOrdinal} artifact manifest.`);
    }
  }
  return manifest;
}

function writeAscii(target, offset, length, value, label) {
  const bytes = Buffer.from(value, 'ascii');
  if (bytes.length > length) reject(`${label} exceeds its ustar field.`);
  bytes.copy(target, offset);
}

function octal(value, digits, label) {
  safeInteger(value, label);
  const encoded = value.toString(8);
  if (encoded.length > digits) reject(`${label} exceeds its ustar octal field.`);
  return encoded.padStart(digits, '0');
}

function tarHeader(name, size) {
  const header = Buffer.alloc(BLOCK_BYTES);
  writeAscii(header, 0, 100, name, 'tar entry name');
  writeAscii(header, 100, 8, `${octal(0o644, 7, 'tar mode')}\0`, 'tar mode');
  writeAscii(header, 108, 8, `${octal(0, 7, 'tar uid')}\0`, 'tar uid');
  writeAscii(header, 116, 8, `${octal(0, 7, 'tar gid')}\0`, 'tar gid');
  writeAscii(header, 124, 12, `${octal(size, 11, 'tar size')}\0`, 'tar size');
  writeAscii(header, 136, 12, `${octal(0, 11, 'tar mtime')}\0`, 'tar mtime');
  header.fill(0x20, 148, 156);
  header[156] = 0x30;
  writeAscii(header, 257, 6, 'ustar\0', 'tar magic');
  writeAscii(header, 263, 2, '00', 'tar version');
  writeAscii(header, 329, 8, `${octal(0, 7, 'tar devmajor')}\0`, 'tar devmajor');
  writeAscii(header, 337, 8, `${octal(0, 7, 'tar devminor')}\0`, 'tar devminor');
  const checksum = header.reduce((sum, byte) => sum + byte, 0);
  writeAscii(header, 148, 8, `${octal(checksum, 6, 'tar checksum')}\0 `, 'tar checksum');
  return header;
}

function normalizeEntries(entries) {
  const normalized = [...entries].map(([name, value]) => {
    requireSafePublicLivePackagePath(name, 'tar entry path');
    const bytes = Buffer.isBuffer(value) ? value : Buffer.from(value);
    return [name, bytes];
  }).sort(([left], [right]) => compareCodePoints(left, right));
  for (let index = 1; index < normalized.length; index += 1) {
    if (normalized[index - 1][0] === normalized[index][0]) {
      reject(`tar entry path ${JSON.stringify(normalized[index][0])} is duplicated.`);
    }
  }
  return normalized;
}

export function validatePublicLivePackageEntryDescriptors(descriptors) {
  if (!Array.isArray(descriptors)) reject('package entry descriptors must be an array.');
  if (descriptors.length === 0 || descriptors.length > PUBLIC_LIVE_PACKAGE_MAX_ENTRY_COUNT) {
    reject(
      `package entry count must be between 1 and ${PUBLIC_LIVE_PACKAGE_MAX_ENTRY_COUNT}.`,
    );
  }
  let priorName = null;
  let contentBytes = 0;
  let tarBytes = BLOCK_BYTES * 2;
  for (const [index, descriptor] of descriptors.entries()) {
    const label = `package entry descriptors[${index}]`;
    exactKeys(descriptor, ['name', 'bytes'], label);
    requireSafePublicLivePackagePath(descriptor.name, `${label}.name`);
    safeInteger(descriptor.bytes, `${label}.bytes`);
    if (priorName !== null && compareCodePoints(priorName, descriptor.name) >= 0) {
      reject('package entry descriptors are not in strict archive-path order.');
    }
    if (descriptor.bytes > PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES) {
      reject(
        `${label}.bytes exceeds the ${PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES}-byte file limit.`,
      );
    }
    const padding = (BLOCK_BYTES - (descriptor.bytes % BLOCK_BYTES)) % BLOCK_BYTES;
    contentBytes += descriptor.bytes;
    tarBytes += BLOCK_BYTES + descriptor.bytes + padding;
    priorName = descriptor.name;
  }
  if (tarBytes > PUBLIC_LIVE_PACKAGE_MAX_UNCOMPRESSED_BYTES) {
    reject(
      `canonical tar size exceeds the ${PUBLIC_LIVE_PACKAGE_MAX_UNCOMPRESSED_BYTES}-byte limit.`,
    );
  }
  return Object.freeze({
    entryCount: descriptors.length,
    contentBytes,
    tarBytes,
  });
}

export function validatePublicLivePackageEntries(entries) {
  const normalized = normalizeEntries(entries);
  return validatePublicLivePackageEntryDescriptors(normalized.map(([name, bytes]) => ({
    name,
    bytes: bytes.length,
  })));
}

export function encodeDeterministicPublicLiveTar(entries) {
  validatePublicLivePackageEntries(entries);
  const chunks = [];
  for (const [name, bytes] of normalizeEntries(entries)) {
    chunks.push(tarHeader(name, bytes.length), bytes);
    const padding = (BLOCK_BYTES - (bytes.length % BLOCK_BYTES)) % BLOCK_BYTES;
    if (padding > 0) chunks.push(Buffer.alloc(padding));
  }
  chunks.push(Buffer.alloc(BLOCK_BYTES * 2));
  return Buffer.concat(chunks);
}

function zeroBlock(bytes) {
  for (const byte of bytes) if (byte !== 0) return false;
  return true;
}

function readAsciiField(header, offset, length, label) {
  const field = header.subarray(offset, offset + length);
  const nul = field.indexOf(0);
  const content = nul === -1 ? field : field.subarray(0, nul);
  for (const byte of content) {
    if (byte < 0x20 || byte > 0x7e) reject(`${label} is not printable ASCII.`);
  }
  return content.toString('ascii');
}

function readOctalField(header, offset, length, label) {
  const text = readAsciiField(header, offset, length, label).trim();
  if (!/^[0-7]+$/.test(text)) reject(`${label} is not canonical octal.`);
  const value = Number.parseInt(text, 8);
  if (!Number.isSafeInteger(value)) reject(`${label} exceeds safe integers.`);
  return value;
}

export function parseDeterministicPublicLiveTar(tarBytes) {
  if (!Buffer.isBuffer(tarBytes)) reject('tar input must be a Buffer.');
  if (tarBytes.length < BLOCK_BYTES * 2 || tarBytes.length % BLOCK_BYTES !== 0) {
    reject('tar byte length is not a complete ustar stream.');
  }
  if (tarBytes.length > PUBLIC_LIVE_PACKAGE_MAX_UNCOMPRESSED_BYTES) {
    reject('tar byte length exceeds the uncompressed package limit.');
  }
  const entries = new Map();
  let offset = 0;
  let priorName = null;
  while (offset < tarBytes.length) {
    const header = tarBytes.subarray(offset, offset + BLOCK_BYTES);
    if (zeroBlock(header)) {
      const second = tarBytes.subarray(offset + BLOCK_BYTES, offset + BLOCK_BYTES * 2);
      if (offset + BLOCK_BYTES * 2 !== tarBytes.length || !zeroBlock(second)) {
        reject('tar terminator is not exactly two zero blocks.');
      }
      break;
    }
    const checksum = readOctalField(header, 148, 8, 'tar checksum');
    const checksumHeader = Buffer.from(header);
    checksumHeader.fill(0x20, 148, 156);
    if (checksumHeader.reduce((sum, byte) => sum + byte, 0) !== checksum) {
      reject('tar header checksum is invalid.');
    }
    const name = readAsciiField(header, 0, 100, 'tar entry name');
    requireSafePublicLivePackagePath(name, 'tar entry name');
    if (entries.has(name)) reject(`tar entry ${JSON.stringify(name)} is duplicated.`);
    if (priorName !== null && compareCodePoints(priorName, name) >= 0) {
      reject('tar entries are not in strict archive-path order.');
    }
    if (entries.size >= PUBLIC_LIVE_PACKAGE_MAX_ENTRY_COUNT) {
      reject(`tar entry count exceeds ${PUBLIC_LIVE_PACKAGE_MAX_ENTRY_COUNT}.`);
    }
    if (header[156] !== 0x30) reject('tar contains a non-regular-file entry.');
    const size = readOctalField(header, 124, 12, 'tar entry size');
    if (size > PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES) {
      reject(`tar entry exceeds the ${PUBLIC_LIVE_PACKAGE_MAX_ENTRY_BYTES}-byte file limit.`);
    }
    if (!header.equals(tarHeader(name, size))) {
      reject('tar header metadata is not canonical.');
    }
    const contentStart = offset + BLOCK_BYTES;
    const contentEnd = contentStart + size;
    const padding = (BLOCK_BYTES - (size % BLOCK_BYTES)) % BLOCK_BYTES;
    const next = contentEnd + padding;
    if (next > tarBytes.length) reject('tar entry is truncated.');
    if (!zeroBlock(tarBytes.subarray(contentEnd, next))) {
      reject('tar entry padding is nonzero.');
    }
    entries.set(name, tarBytes.subarray(contentStart, contentEnd));
    priorName = name;
    offset = next;
  }
  if (offset >= tarBytes.length) reject('tar terminator is missing.');
  validatePublicLivePackageEntries(entries);
  return entries;
}

export function compressDeterministicPublicLiveTar(tarBytes) {
  if (!Buffer.isBuffer(tarBytes)
    || tarBytes.length === 0
    || tarBytes.length > PUBLIC_LIVE_PACKAGE_MAX_UNCOMPRESSED_BYTES) {
    reject('tar input is empty or exceeds the uncompressed package limit.');
  }
  return brotliCompressSync(tarBytes, {
    params: {
      [zlibConstants.BROTLI_PARAM_MODE]: zlibConstants.BROTLI_MODE_GENERIC,
      [zlibConstants.BROTLI_PARAM_QUALITY]: PUBLIC_LIVE_PACKAGE_BROTLI_QUALITY,
      [zlibConstants.BROTLI_PARAM_LGWIN]: PUBLIC_LIVE_PACKAGE_BROTLI_LGWIN,
      [zlibConstants.BROTLI_PARAM_SIZE_HINT]: tarBytes.length,
    },
  });
}

export function decompressDeterministicPublicLiveTar(archiveBytes) {
  if (!Buffer.isBuffer(archiveBytes)) reject('archive input must be a Buffer.');
  if (archiveBytes.length === 0
    || archiveBytes.length >= PUBLIC_LIVE_PACKAGE_MAX_ARCHIVE_BYTES) {
    reject('archive byte length is empty or exceeds the single-file publication limit.');
  }
  let tarBytes;
  try {
    tarBytes = brotliDecompressSync(archiveBytes, {
      maxOutputLength: PUBLIC_LIVE_PACKAGE_MAX_UNCOMPRESSED_BYTES,
    });
  } catch (error) {
    reject(`Brotli decompression failed: ${error instanceof Error ? error.message : String(error)}.`);
  }
  return tarBytes;
}
