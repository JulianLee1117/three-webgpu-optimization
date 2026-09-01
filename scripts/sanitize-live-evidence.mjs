import { createHash } from 'node:crypto';
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
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { verifyRunDirectory } from '../analysis/summarize.mjs';
import {
  compareComputeProcessIdentitySets,
  parseCsvRecord,
  TELEMETRY_CSV_FIELDS,
} from './nvidia-telemetry.mjs';
import {
  ENVIRONMENT_NOTE_REPLACEMENT,
  LIVE_OPTIONAL_ARTIFACTS,
  LIVE_PRIVATE_BUNDLE_LABEL,
  LIVE_PRIVATE_MANIFEST_NAME,
  LIVE_PUBLIC_INTEGRITY_BYTE_SCOPE,
  LIVE_PUBLIC_BUNDLE_LABEL,
  LIVE_PUBLIC_EVIDENCE_CSV_REDACTION_ALLOWLIST,
  LIVE_PUBLIC_EVIDENCE_JSON_REDACTION_ALLOWLIST,
  LIVE_PUBLIC_EVIDENCE_POLICY_ID,
  LIVE_PUBLIC_EVIDENCE_POLICY_SCHEMA_VERSION,
  LIVE_PUBLIC_PRIVATE_BYTE_SCOPE,
  LIVE_REQUIRED_ARTIFACTS,
  resolveLiveSanitizerImplementationAtCommit,
} from './live-evidence-sanitizer-policy.mjs';
import {
  collectSourceProvenance,
  sourceProvenanceMatches,
} from './source-provenance.mjs';

const SANITIZER_SOURCE_PATH = fileURLToPath(import.meta.url);
const PROJECT_ROOT = path.resolve(path.dirname(SANITIZER_SOURCE_PATH), '..');
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const PRIVATE_MANIFEST_NAME = LIVE_PRIVATE_MANIFEST_NAME;
const POLICY_ID = LIVE_PUBLIC_EVIDENCE_POLICY_ID;
const REQUIRED_LIVE_ARTIFACTS = LIVE_REQUIRED_ARTIFACTS;
const OPTIONAL_LIVE_ARTIFACTS = LIVE_OPTIONAL_ARTIFACTS;

export const PUBLIC_EVIDENCE_JSON_REDACTION_ALLOWLIST =
  LIVE_PUBLIC_EVIDENCE_JSON_REDACTION_ALLOWLIST;
export const PUBLIC_EVIDENCE_CSV_REDACTION_ALLOWLIST =
  LIVE_PUBLIC_EVIDENCE_CSV_REDACTION_ALLOWLIST;

const JSON_ALLOWLIST_KEYS = new Set(PUBLIC_EVIDENCE_JSON_REDACTION_ALLOWLIST.map(
  (entry) => `${entry.file}:${entry.path}`,
));
const CSV_ALLOWLIST_KEYS = new Set(PUBLIC_EVIDENCE_CSV_REDACTION_ALLOWLIST.map(
  (entry) => `${entry.file}:${entry.column}`,
));
const SENSITIVE_KEYS = new Set([
  'gpuUuid',
  'pid',
  'processName',
  'note',
  'hostname',
  'hostName',
  'username',
  'userName',
  'homeDir',
  'homedir',
  'cwd',
  'userDataDir',
  'profilePath',
  'absolutePath',
  'localPath',
]);
const LOCAL_PATH_PATTERNS = Object.freeze([
  /(?:^|["'\s])(?:[a-zA-Z]:[\\/])[^\r\n]*/,
  /(?:^|["'\s])(?:\/home\/|\/Users\/)[^/\s]+(?:\/[^\s]*)?/,
  /file:\/\/\/(?:[a-zA-Z]:\/|home\/|Users\/)/i,
  /^\\\\[^\\/]+[\\/][^\\/]+/,
]);

function fail(message) {
  throw new Error(`Live evidence sanitization failed: ${message}`);
}

function sha256Bytes(value) {
  return createHash('sha256').update(value).digest('hex');
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function exactKeys(value, expected, label) {
  if (!isRecord(value)) fail(`${label} must be an object.`);
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (JSON.stringify(actual) !== JSON.stringify(wanted)) {
    fail(`${label} keys differ from the frozen schema.`);
  }
  return value;
}

function exactArray(value, expected, label) {
  if (!Array.isArray(value) || JSON.stringify(value) !== JSON.stringify(expected)) {
    fail(`${label} differs from the frozen live-artifact list.`);
  }
}

function safeArtifactName(value, label) {
  if (typeof value !== 'string'
    || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(value)
    || path.basename(value) !== value
    || value === PRIVATE_MANIFEST_NAME) {
    fail(`${label} is not a safe manifest-bound artifact name.`);
  }
  return value;
}

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function parseJson(contents, label) {
  try {
    return JSON.parse(contents.toString('utf8'));
  } catch {
    fail(`${label} is not valid JSON.`);
  }
}

function normalizeJsonPath(pathSegments) {
  return `$${pathSegments.map((segment) => (
    typeof segment === 'number' ? '[*]' : `.${segment}`
  )).join('')}`;
}

function looksLikeLocalPath(value) {
  return typeof value === 'string' && LOCAL_PATH_PATTERNS.some((pattern) => pattern.test(value));
}

function auditSensitiveJsonShape(value, file, pathSegments = []) {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => auditSensitiveJsonShape(entry, file, [...pathSegments, index]));
    return;
  }
  if (!isRecord(value)) {
    if (looksLikeLocalPath(value)) {
      const normalizedPath = normalizeJsonPath(pathSegments);
      if (!JSON_ALLOWLIST_KEYS.has(`${file}:${normalizedPath}`)) {
        fail(`${file} contains a machine-local path at unlisted JSON path ${normalizedPath}.`);
      }
    }
    return;
  }
  for (const [key, child] of Object.entries(value)) {
    const childPath = [...pathSegments, key];
    const normalizedPath = normalizeJsonPath(childPath);
    if (SENSITIVE_KEYS.has(key) && !JSON_ALLOWLIST_KEYS.has(`${file}:${normalizedPath}`)) {
      fail(`${file} contains sensitive key ${JSON.stringify(key)} at unlisted JSON path ${normalizedPath}.`);
    }
    auditSensitiveJsonShape(child, file, childPath);
  }
}

function auditPrivateGpuUuidLocations(value, file, privateGpuUuids, pathSegments = []) {
  if (Array.isArray(value)) {
    value.forEach((entry, index) => auditPrivateGpuUuidLocations(
      entry,
      file,
      privateGpuUuids,
      [...pathSegments, index],
    ));
    return;
  }
  if (isRecord(value)) {
    for (const [key, child] of Object.entries(value)) {
      auditPrivateGpuUuidLocations(child, file, privateGpuUuids, [...pathSegments, key]);
    }
    return;
  }
  if (typeof value !== 'string') return;
  const normalizedPath = normalizeJsonPath(pathSegments);
  if (!JSON_ALLOWLIST_KEYS.has(`${file}:${normalizedPath}`)
    && privateGpuUuids.some((gpuUuid) => value.includes(gpuUuid))) {
    fail(`${file} contains a private GPU UUID at unlisted JSON path ${normalizedPath}.`);
  }
}

function assertOnlyAllowlistedJsonChanges(original, derived, file, pathSegments = []) {
  if (Array.isArray(original) || Array.isArray(derived)) {
    if (!Array.isArray(original) || !Array.isArray(derived) || original.length !== derived.length) {
      fail(`${file} changed an array shape during sanitization.`);
    }
    for (let index = 0; index < original.length; index += 1) {
      assertOnlyAllowlistedJsonChanges(
        original[index],
        derived[index],
        file,
        [...pathSegments, index],
      );
    }
    return;
  }
  if (isRecord(original) || isRecord(derived)) {
    if (!isRecord(original) || !isRecord(derived)) {
      fail(`${file} changed a value type during sanitization.`);
    }
    const originalKeys = Object.keys(original).sort();
    const derivedKeys = Object.keys(derived).sort();
    if (JSON.stringify(originalKeys) !== JSON.stringify(derivedKeys)) {
      fail(`${file} changed object keys during sanitization.`);
    }
    for (const key of originalKeys) {
      assertOnlyAllowlistedJsonChanges(
        original[key],
        derived[key],
        file,
        [...pathSegments, key],
      );
    }
    return;
  }
  if (!Object.is(original, derived)) {
    const normalizedPath = normalizeJsonPath(pathSegments);
    if (!JSON_ALLOWLIST_KEYS.has(`${file}:${normalizedPath}`)) {
      fail(`${file} changed unlisted JSON path ${normalizedPath}.`);
    }
  }
}

function tupleKey(record, label) {
  if (!isRecord(record)
    || typeof record.gpuUuid !== 'string' || record.gpuUuid.length === 0
    || !Number.isSafeInteger(record.pid) || record.pid < 0
    || typeof record.processName !== 'string' || record.processName.length === 0) {
    fail(`${label} has an invalid GPU process identity tuple.`);
  }
  return JSON.stringify([record.gpuUuid, record.pid, record.processName]);
}

function compareText(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function sortedTupleKeys(processes, label) {
  if (!Array.isArray(processes)) fail(`${label} must be an array.`);
  const records = processes.map((record, index) => {
    tupleKey(record, `${label}[${index}]`);
    return record;
  }).sort((left, right) => (
    compareText(left.gpuUuid, right.gpuUuid)
      || left.pid - right.pid
      || compareText(left.processName, right.processName)
  ));
  const keys = records.map((record, index) => tupleKey(record, `${label}[${index}]`));
  if (new Set(keys).size !== keys.length) fail(`${label} contains duplicate identity tuples.`);
  return keys;
}

function requireAvailableProcessSnapshot(report, phase) {
  const snapshot = report?.computeProcesses?.[phase];
  exactKeys(
    snapshot,
    [
      'label',
      'status',
      'capturedAtIso',
      'runElapsedMs',
      'reason',
      'rawNonemptyLineCount',
      'parsedRecordCount',
      'malformedLineCount',
      'stdoutByteCount',
      'stdoutTruncated',
      'stderrByteCount',
      'processes',
    ],
    `gpu-telemetry-summary.json computeProcesses.${phase}`,
  );
  if (snapshot.status !== 'available'
    || snapshot.reason !== null
    || !Array.isArray(snapshot.processes)
    || snapshot.rawNonemptyLineCount !== snapshot.processes.length
    || snapshot.parsedRecordCount !== snapshot.processes.length
    || snapshot.malformedLineCount !== 0
    || !Number.isSafeInteger(snapshot.stdoutByteCount)
    || snapshot.stdoutByteCount < 0
    || snapshot.stdoutTruncated !== false
    || snapshot.stderrByteCount !== 0) {
    fail(`gpu-telemetry-summary.json computeProcesses.${phase} is not available.`);
  }
  for (const [index, record] of snapshot.processes.entries()) {
    exactKeys(
      record,
      ['gpuUuid', 'pid', 'processName', 'usedMemoryMiB'],
      `gpu-telemetry-summary.json computeProcesses.${phase}.processes[${index}]`,
    );
  }
  return snapshot;
}

function buildReplacementMaps(report) {
  if (!isRecord(report) || report.status !== 'available') {
    fail('gpu-telemetry-summary.json must contain available candidate telemetry.');
  }
  const gpus = report?.summary?.gpus;
  if (!Array.isArray(gpus) || gpus.length === 0) {
    fail('gpu-telemetry-summary.json summary.gpus must be nonempty.');
  }
  const gpuUuids = gpus.map((gpu, index) => {
    if (!isRecord(gpu) || typeof gpu.gpuUuid !== 'string' || gpu.gpuUuid.length === 0) {
      fail(`gpu-telemetry-summary.json summary.gpus[${index}].gpuUuid is invalid.`);
    }
    return gpu.gpuUuid;
  }).sort(compareText);
  if (new Set(gpuUuids).size !== gpuUuids.length) {
    fail('gpu-telemetry-summary.json contains duplicate GPU UUIDs.');
  }
  const gpuMap = new Map(gpuUuids.map((gpuUuid, index) => (
    [gpuUuid, `GPU-PUBLIC-DEVICE-${index}`]
  )));

  const pre = requireAvailableProcessSnapshot(report, 'pre');
  const post = requireAvailableProcessSnapshot(report, 'post');
  const preKeys = sortedTupleKeys(pre.processes, 'pre-run process records');
  const postKeys = sortedTupleKeys(post.processes, 'post-run process records');
  const unionKeys = [...new Set([...preKeys, ...postKeys])].sort((leftKey, rightKey) => {
    const [leftGpu, leftPid, leftName] = JSON.parse(leftKey);
    const [rightGpu, rightPid, rightName] = JSON.parse(rightKey);
    return compareText(leftGpu, rightGpu)
      || leftPid - rightPid
      || compareText(leftName, rightName);
  });
  const processMap = new Map(unionKeys.map((key, index) => [key, Object.freeze({
    gpuUuid: gpuMap.get(JSON.parse(key)[0]),
    pid: 100_000 + index + 1,
    processName: `resident-gpu-process-${index + 1}.redacted`,
  })]));
  for (const [key, replacement] of processMap) {
    if (replacement.gpuUuid === undefined) {
      fail(`GPU process tuple ${key} refers to a UUID absent from summary.gpus.`);
    }
  }
  return {
    gpuMap,
    processMap,
    preKeys,
    postKeys,
    privateGpuUuids: gpuUuids,
  };
}

function replaceGpuUuid(value, maps, label) {
  const replacement = maps.gpuMap.get(value);
  if (replacement === undefined) fail(`${label} refers to an unregistered GPU UUID.`);
  return replacement;
}

function redactProcessRecord(record, maps, label) {
  const replacement = maps.processMap.get(tupleKey(record, label));
  if (!replacement) fail(`${label} is absent from the canonical private process set.`);
  record.gpuUuid = replacement.gpuUuid;
  record.pid = replacement.pid;
  record.processName = replacement.processName;
}

function redactTelemetryReport(report, maps, label) {
  if (!Array.isArray(report?.summary?.gpus)) fail(`${label}.summary.gpus is missing.`);
  for (const [index, gpu] of report.summary.gpus.entries()) {
    gpu.gpuUuid = replaceGpuUuid(gpu.gpuUuid, maps, `${label}.summary.gpus[${index}].gpuUuid`);
  }
  if (!Array.isArray(report?.coverageAudit?.gpuIdentities)) {
    fail(`${label}.coverageAudit.gpuIdentities is missing.`);
  }
  for (const [index, gpu] of report.coverageAudit.gpuIdentities.entries()) {
    gpu.gpuUuid = replaceGpuUuid(
      gpu.gpuUuid,
      maps,
      `${label}.coverageAudit.gpuIdentities[${index}].gpuUuid`,
    );
  }
  for (const phase of ['pre', 'post']) {
    const processes = report?.computeProcesses?.[phase]?.processes;
    if (!Array.isArray(processes)) fail(`${label}.computeProcesses.${phase}.processes is missing.`);
    processes.forEach((record, index) => redactProcessRecord(
      record,
      maps,
      `${label}.computeProcesses.${phase}.processes[${index}]`,
    ));
  }
}

function redactLiveAudit(metadata, maps, privateTelemetryReport) {
  const comparison = metadata?.liveFirstInstanceEnvironmentAudit
    ?.computeProcessIdentityComparison;
  exactKeys(
    comparison,
    ['schemaVersion', 'kind', 'identityFields', 'ignoredFields', 'pass', 'pre', 'post', 'reasons'],
    'metadata.json live process identity comparison',
  );
  const expectedPrivateComparison = compareComputeProcessIdentitySets(
    privateTelemetryReport?.computeProcesses?.pre,
    privateTelemetryReport?.computeProcesses?.post,
  );
  if (JSON.stringify(comparison) !== JSON.stringify(expectedPrivateComparison)) {
    fail('metadata.json live process identity comparison does not reconstruct from telemetry.');
  }
  for (const phase of ['pre', 'post']) {
    const keys = sortedTupleKeys(comparison[phase], `metadata live audit ${phase}`);
    if (JSON.stringify(keys) !== JSON.stringify(maps[`${phase}Keys`])) {
      fail(`metadata.json live audit ${phase} differs from the telemetry process set.`);
    }
    comparison[phase].forEach((record, index) => {
      exactKeys(
        record,
        ['gpuUuid', 'pid', 'processName'],
        `metadata.json live audit ${phase}[${index}]`,
      );
      redactProcessRecord(record, maps, `metadata.json live audit ${phase}[${index}]`);
    });
  }
}

function parseCsvLines(contents, file) {
  const text = contents.toString('utf8');
  if (text.includes('\0')) fail(`${file} contains a NUL byte.`);
  const lines = text.split(/\r?\n/);
  if (lines.at(-1) === '') lines.pop();
  if (lines.length === 0) fail(`${file} is empty.`);
  return lines.map((line, index) => {
    const record = parseCsvRecord(line);
    if (!record) fail(`${file} line ${index + 1} is malformed CSV.`);
    return record;
  });
}

function escapeCsv(value) {
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function auditCsvShape(contents, file, privateGpuUuids = []) {
  const records = parseCsvLines(contents, file);
  const headers = records[0];
  for (const header of headers) {
    if (SENSITIVE_KEYS.has(header) && !CSV_ALLOWLIST_KEYS.has(`${file}:${header}`)) {
      fail(`${file} contains sensitive column ${JSON.stringify(header)} outside the CSV allowlist.`);
    }
  }
  for (let rowIndex = 1; rowIndex < records.length; rowIndex += 1) {
    if (records[rowIndex].length !== headers.length) {
      fail(`${file} row ${rowIndex + 1} differs from its header width.`);
    }
    for (let columnIndex = 0; columnIndex < headers.length; columnIndex += 1) {
      if (looksLikeLocalPath(records[rowIndex][columnIndex])
        && !CSV_ALLOWLIST_KEYS.has(`${file}:${headers[columnIndex]}`)) {
        fail(`${file} contains a machine-local path in unlisted column ${headers[columnIndex]}.`);
      }
      if (!CSV_ALLOWLIST_KEYS.has(`${file}:${headers[columnIndex]}`)
        && privateGpuUuids.some((gpuUuid) => records[rowIndex][columnIndex].includes(gpuUuid))) {
        fail(`${file} contains a private GPU UUID in unlisted column ${headers[columnIndex]}.`);
      }
    }
  }
  return records;
}

function redactTelemetryCsv(contents, maps) {
  const records = auditCsvShape(contents, 'gpu-telemetry.csv', maps.privateGpuUuids);
  const headers = records[0];
  if (JSON.stringify(headers) !== JSON.stringify(TELEMETRY_CSV_FIELDS)) {
    fail('gpu-telemetry.csv headers differ from the frozen telemetry schema.');
  }
  const uuidIndex = headers.indexOf('gpuUuid');
  const originalRecords = records.map((record) => [...record]);
  for (let rowIndex = 1; rowIndex < records.length; rowIndex += 1) {
    records[rowIndex][uuidIndex] = replaceGpuUuid(
      records[rowIndex][uuidIndex],
      maps,
      `gpu-telemetry.csv row ${rowIndex + 1} gpuUuid`,
    );
  }
  for (let rowIndex = 0; rowIndex < records.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < headers.length; columnIndex += 1) {
      if (rowIndex === 0 || columnIndex !== uuidIndex) {
        if (records[rowIndex][columnIndex] !== originalRecords[rowIndex][columnIndex]) {
          fail(
            `gpu-telemetry.csv changed unlisted column ${headers[columnIndex]} at row ${rowIndex + 1}.`,
          );
        }
      }
    }
  }
  return Buffer.from(`${records.map((record) => record.map(escapeCsv).join(',')).join('\n')}\n`, 'utf8');
}

function validateManifestShape(manifest) {
  exactKeys(
    manifest,
    ['schemaVersion', 'runId', 'hashAlgorithm', 'requiredFiles', 'optionalFiles', 'files'],
    PRIVATE_MANIFEST_NAME,
  );
  if (manifest.schemaVersion !== 2 || manifest.hashAlgorithm !== 'sha256') {
    fail(`${PRIVATE_MANIFEST_NAME} has an unsupported schema or hash algorithm.`);
  }
  if (typeof manifest.runId !== 'string' || manifest.runId.length === 0) {
    fail(`${PRIVATE_MANIFEST_NAME} runId is invalid.`);
  }
  exactArray(manifest.requiredFiles, REQUIRED_LIVE_ARTIFACTS, 'artifact-manifest.json requiredFiles');
  if (!Array.isArray(manifest.optionalFiles) || manifest.optionalFiles.length !== 1) {
    fail('artifact-manifest.json optionalFiles must contain only gpu-telemetry.csv.');
  }
  exactKeys(
    manifest.optionalFiles[0],
    ['name', 'present', 'evidenceAvailable', 'absenceReason'],
    'artifact-manifest.json optionalFiles[0]',
  );
  if (manifest.optionalFiles[0].name !== OPTIONAL_LIVE_ARTIFACTS[0]
    || manifest.optionalFiles[0].present !== true
    || manifest.optionalFiles[0].evidenceAvailable !== true
    || manifest.optionalFiles[0].absenceReason !== null) {
    fail('live candidate public evidence requires present and available gpu-telemetry.csv.');
  }
  if (!Array.isArray(manifest.files)) fail('artifact-manifest.json files must be an array.');
  const declared = [...REQUIRED_LIVE_ARTIFACTS, ...OPTIONAL_LIVE_ARTIFACTS];
  if (manifest.files.length !== declared.length) {
    fail('artifact-manifest.json files do not exactly cover the live artifact set.');
  }
  const byName = new Map();
  for (const [index, entry] of manifest.files.entries()) {
    exactKeys(
      entry,
      ['name', 'role', 'required', 'present', 'bytes', 'sha256', 'absenceReason'],
      `artifact-manifest.json files[${index}]`,
    );
    const name = safeArtifactName(entry.name, `artifact-manifest.json files[${index}].name`);
    if (byName.has(name)) fail(`artifact-manifest.json repeats ${name}.`);
    const required = REQUIRED_LIVE_ARTIFACTS.includes(name);
    if (!declared.includes(name)
      || entry.required !== required
      || entry.present !== true
      || !Number.isSafeInteger(entry.bytes) || entry.bytes < 0
      || !SHA256_PATTERN.test(entry.sha256 ?? '')
      || entry.absenceReason !== null
      || typeof entry.role !== 'string' || entry.role.length === 0) {
      fail(`artifact-manifest.json entry ${name} is inconsistent with a finalized live candidate.`);
    }
    byName.set(name, entry);
  }
  if (declared.some((name) => !byName.has(name))) {
    fail('artifact-manifest.json omits a live artifact.');
  }
  return byName;
}

async function readRegularFile(filename, label) {
  let fileStat;
  try {
    fileStat = await lstat(filename);
  } catch (error) {
    if (error?.code === 'ENOENT') fail(`${label} is missing.`);
    throw error;
  }
  if (!fileStat.isFile() || fileStat.isSymbolicLink()) fail(`${label} must be a regular file.`);
  return readFile(filename);
}

async function assertPathAbsent(filename, label) {
  try {
    await lstat(filename);
    fail(`${label} already exists.`);
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
}

function pathIsWithin(parent, child) {
  const relative = path.relative(parent, child);
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative));
}

async function readAndVerifyPrivateBundle(privateDirectory) {
  const manifestPath = path.join(privateDirectory, PRIVATE_MANIFEST_NAME);
  const manifestBytes = await readRegularFile(manifestPath, PRIVATE_MANIFEST_NAME);
  const manifest = parseJson(manifestBytes, PRIVATE_MANIFEST_NAME);
  const entries = validateManifestShape(manifest);
  const expectedDirectoryNames = [PRIVATE_MANIFEST_NAME, ...entries.keys()].sort();
  const directoryNames = (await readdir(privateDirectory)).sort();
  if (JSON.stringify(directoryNames) !== JSON.stringify(expectedDirectoryNames)) {
    fail('private run directory contains undeclared or non-artifact entries.');
  }
  const contentsByName = new Map();
  for (const [name, entry] of entries) {
    const contents = await readRegularFile(path.join(privateDirectory, name), name);
    if (contents.length !== entry.bytes || sha256Bytes(contents) !== entry.sha256) {
      fail(`${name} does not match the private artifact manifest.`);
    }
    contentsByName.set(name, contents);
  }
  return { manifest, manifestBytes, contentsByName, directoryNames };
}

async function assertPrivateBundleUnchanged(privateDirectory, original) {
  const directoryNames = (await readdir(privateDirectory)).sort();
  if (JSON.stringify(directoryNames) !== JSON.stringify(original.directoryNames)) {
    fail('private run directory entries changed while the public bundle was being derived.');
  }
  const currentManifest = await readRegularFile(
    path.join(privateDirectory, PRIVATE_MANIFEST_NAME),
    PRIVATE_MANIFEST_NAME,
  );
  if (!currentManifest.equals(original.manifestBytes)) {
    fail('the private artifact manifest changed while the public bundle was being derived.');
  }
  for (const [name, contents] of original.contentsByName) {
    const current = await readRegularFile(path.join(privateDirectory, name), name);
    if (!current.equals(contents)) {
      fail(`${name} changed in the private bundle while the public bundle was being derived.`);
    }
  }
}

function sanitizeArtifactContents(privateBundle) {
  const parsedJsonByName = new Map();
  for (const [name, contents] of privateBundle.contentsByName) {
    if (name.endsWith('.json')) {
      const parsed = parseJson(contents, name);
      auditSensitiveJsonShape(parsed, name);
      if (name === 'metadata.json' || name === 'gpu-telemetry-summary.json') {
        parsedJsonByName.set(name, parsed);
      }
    } else if (name.endsWith('.csv')) {
      auditCsvShape(contents, name);
    }
  }

  const metadata = parsedJsonByName.get('metadata.json');
  const telemetryReport = parsedJsonByName.get('gpu-telemetry-summary.json');
  if (metadata?.status !== 'complete'
    || metadata?.evidenceStatus !== 'candidate'
    || metadata?.protocol?.matrixKind !== 'first-instance-live'
    || metadata?.runId !== privateBundle.manifest.runId) {
    fail('metadata.json is not a finalized first-instance-live candidate run.');
  }
  if (JSON.stringify(metadata?.environment?.gpuTelemetry) !== JSON.stringify(telemetryReport)) {
    fail('metadata.json telemetry differs from gpu-telemetry-summary.json before sanitization.');
  }

  const maps = buildReplacementMaps(telemetryReport);
  for (const [name, contents] of privateBundle.contentsByName) {
    if (name.endsWith('.json')) {
      const parsed = parsedJsonByName.get(name) ?? parseJson(contents, name);
      auditPrivateGpuUuidLocations(parsed, name, maps.privateGpuUuids);
    } else if (name.endsWith('.csv')) {
      auditCsvShape(contents, name, maps.privateGpuUuids);
    }
  }
  const derived = new Map(privateBundle.contentsByName);
  const derivedMetadata = structuredClone(metadata);
  const derivedTelemetryReport = structuredClone(telemetryReport);
  derivedMetadata.environment.note = ENVIRONMENT_NOTE_REPLACEMENT;
  redactTelemetryReport(derivedTelemetryReport, maps, 'gpu-telemetry-summary.json');
  redactTelemetryReport(
    derivedMetadata.environment.gpuTelemetry,
    maps,
    'metadata.json environment.gpuTelemetry',
  );
  redactLiveAudit(derivedMetadata, maps, telemetryReport);
  assertOnlyAllowlistedJsonChanges(metadata, derivedMetadata, 'metadata.json');
  assertOnlyAllowlistedJsonChanges(
    telemetryReport,
    derivedTelemetryReport,
    'gpu-telemetry-summary.json',
  );
  derived.set('metadata.json', jsonBytes(derivedMetadata));
  derived.set('gpu-telemetry-summary.json', jsonBytes(derivedTelemetryReport));
  derived.set(
    'gpu-telemetry.csv',
    redactTelemetryCsv(privateBundle.contentsByName.get('gpu-telemetry.csv'), maps),
  );
  return derived;
}

function candidateSourceBinding(metadata, sanitizerSourceProvenance) {
  const provenance = metadata?.sourceProvenance;
  const start = provenance?.start;
  const end = provenance?.end;
  if (provenance?.stable !== true
    || start?.status !== 'available'
    || end?.status !== 'available'
    || start.dirty !== false
    || end.dirty !== false
    || start.commit !== end.commit
    || start.tree !== end.tree
    || start.trackedFilesSha256 !== end.trackedFilesSha256
    || start.packageLockSha256 !== end.packageLockSha256) {
    fail('candidate source provenance is not clean and stable through teardown.');
  }
  if (sanitizerSourceProvenance?.status !== 'available'
    || sanitizerSourceProvenance.captureStable !== true
    || sanitizerSourceProvenance.dirty !== false
    || sanitizerSourceProvenance.packageLockTracked !== true
    || sanitizerSourceProvenance.commit !== start.commit
    || sanitizerSourceProvenance.tree !== start.tree
    || sanitizerSourceProvenance.trackedFilesSha256 !== start.trackedFilesSha256
    || sanitizerSourceProvenance.packageLockSha256 !== start.packageLockSha256) {
    fail('sanitizer source is not the clean recorded candidate source tree.');
  }
  return Object.freeze({
    commit: start.commit,
    tree: start.tree,
    clean: true,
    stable: true,
    trackedFilesSha256: start.trackedFilesSha256,
    packageLockSha256: start.packageLockSha256,
  });
}

async function sanitizerImplementationBinding(repositoryRoot, candidateSource) {
  let committed;
  try {
    committed = await resolveLiveSanitizerImplementationAtCommit(
      repositoryRoot,
      candidateSource,
    );
  } catch (error) {
    fail(`candidate Git objects do not contain the recorded sanitizer implementation: ${
      error instanceof Error ? error.message : String(error)
    }.`);
  }
  for (const entry of committed.files) {
    const absolutePath = path.resolve(PROJECT_ROOT, entry.repositoryPath);
    if (!pathIsWithin(PROJECT_ROOT, absolutePath)) {
      fail(`sanitizer implementation path escapes the project root: ${entry.repositoryPath}.`);
    }
    const contents = await readRegularFile(
      absolutePath,
      `executing sanitizer implementation ${entry.repositoryPath}`,
    );
    if (sha256Bytes(contents) !== entry.sha256) {
      fail(
        `executing sanitizer implementation ${entry.repositoryPath} differs from candidate commit ${candidateSource.commit}.`,
      );
    }
  }
  return committed.files;
}

function createPublicManifest(
  privateBundle,
  derivedContents,
  implementationFiles,
  candidateSource,
) {
  const files = privateBundle.manifest.files.map((entry) => {
    const contents = derivedContents.get(entry.name);
    if (!Buffer.isBuffer(contents)) fail(`derived bytes are missing for ${entry.name}.`);
    return {
      ...entry,
      bytes: contents.length,
      sha256: sha256Bytes(contents),
    };
  });
  return {
    ...privateBundle.manifest,
    files,
    bundleProvenance: {
      schemaVersion: LIVE_PUBLIC_EVIDENCE_POLICY_SCHEMA_VERSION,
      policyId: POLICY_ID,
      bundleLabel: LIVE_PUBLIC_BUNDLE_LABEL,
      sourceBundleLabel: LIVE_PRIVATE_BUNDLE_LABEL,
      candidateSource,
      privateArtifactManifest: {
        name: PRIVATE_MANIFEST_NAME,
        sha256: sha256Bytes(privateBundle.manifestBytes),
      },
      sanitizer: {
        implementationKind: 'exact-source-file-closure-v1',
        candidateCommit: candidateSource.commit,
        candidateTree: candidateSource.tree,
        files: implementationFiles,
      },
      redactionPolicy: {
        jsonPathAllowlist: PUBLIC_EVIDENCE_JSON_REDACTION_ALLOWLIST,
        csvColumnAllowlist: PUBLIC_EVIDENCE_CSV_REDACTION_ALLOWLIST,
        unlistedFieldChangesPermitted: false,
      },
      integrityByteScope: LIVE_PUBLIC_INTEGRITY_BYTE_SCOPE,
      privateByteScope: LIVE_PUBLIC_PRIVATE_BYTE_SCOPE,
    },
  };
}

async function verifyWrittenPublicBundle(directory, manifest, derivedContents) {
  for (const [name, expected] of derivedContents) {
    const contents = await readRegularFile(path.join(directory, name), `public ${name}`);
    if (!contents.equals(expected)) fail(`public ${name} differs from its derived bytes.`);
  }
  const manifestBytes = await readRegularFile(
    path.join(directory, PRIVATE_MANIFEST_NAME),
    `public ${PRIVATE_MANIFEST_NAME}`,
  );
  if (!manifestBytes.equals(jsonBytes(manifest))) {
    fail('written public artifact manifest differs from the derived manifest.');
  }
}

export async function sanitizeLiveEvidence(
  privateRunDirectory,
  publicOutputDirectory,
  { repositoryRoot = PROJECT_ROOT } = {},
) {
  const privateDirectory = path.resolve(privateRunDirectory);
  const outputDirectory = path.resolve(publicOutputDirectory);
  const candidateRepositoryRoot = path.resolve(repositoryRoot);
  let privateStat;
  try {
    privateStat = await lstat(privateDirectory);
  } catch (error) {
    if (error?.code === 'ENOENT') fail('private run directory does not exist.');
    throw error;
  }
  if (!privateStat.isDirectory() || privateStat.isSymbolicLink()) {
    fail('private run input is not a non-symbolic-link directory.');
  }
  if (pathIsWithin(privateDirectory, outputDirectory)) {
    fail('public output directory must not be the private directory or one of its descendants.');
  }
  await assertPathAbsent(outputDirectory, 'public output directory');

  const sanitizerSourceStart = await collectSourceProvenance(candidateRepositoryRoot);
  const privateBundle = await readAndVerifyPrivateBundle(privateDirectory);
  const derivedContents = sanitizeArtifactContents(privateBundle);
  const privateVerification = await verifyRunDirectory(
    privateDirectory,
    { repositoryRoot: candidateRepositoryRoot },
  );
  if (privateVerification?.artifactVerification?.evidenceStatus !== 'candidate') {
    fail('the independent verifier did not accept the private directory as candidate evidence.');
  }
  const privateMetadata = parseJson(
    privateBundle.contentsByName.get('metadata.json'),
    'metadata.json',
  );
  const candidateSource = candidateSourceBinding(privateMetadata, sanitizerSourceStart);
  const implementationFiles = await sanitizerImplementationBinding(
    candidateRepositoryRoot,
    candidateSource,
  );
  const publicManifest = createPublicManifest(
    privateBundle,
    derivedContents,
    implementationFiles,
    candidateSource,
  );
  const publicManifestBytes = jsonBytes(publicManifest);
  const privateManifestSha256 = sha256Bytes(privateBundle.manifestBytes);
  const publicManifestSha256 = sha256Bytes(publicManifestBytes);
  if (publicManifestSha256 === privateManifestSha256) {
    fail('public and private artifact manifests are not distinct.');
  }

  const parentDirectory = path.dirname(outputDirectory);
  await mkdir(parentDirectory, { recursive: true });
  await assertPathAbsent(outputDirectory, 'public output directory');
  const stagingDirectory = await mkdtemp(path.join(
    parentDirectory,
    `.${path.basename(outputDirectory)}.staging-`,
  ));
  let committed = false;
  try {
    await Promise.all([...derivedContents].map(([name, contents]) => (
      writeFile(path.join(stagingDirectory, name), contents, { flag: 'wx' })
    )));
    await writeFile(
      path.join(stagingDirectory, PRIVATE_MANIFEST_NAME),
      publicManifestBytes,
      { flag: 'wx' },
    );
    await verifyWrittenPublicBundle(stagingDirectory, publicManifest, derivedContents);
    await assertPrivateBundleUnchanged(privateDirectory, privateBundle);
    const publicVerification = await verifyRunDirectory(
      stagingDirectory,
      { repositoryRoot: candidateRepositoryRoot },
    );
    if (publicVerification?.artifactVerification?.evidenceStatus !== 'candidate'
      || publicVerification?.bundleIntegrity?.bundleLabel !== LIVE_PUBLIC_BUNDLE_LABEL) {
      fail('the independent verifier did not accept the staged public-derived bundle.');
    }
    await assertPrivateBundleUnchanged(privateDirectory, privateBundle);
    const sanitizerSourceEnd = await collectSourceProvenance(candidateRepositoryRoot);
    if (!sourceProvenanceMatches(sanitizerSourceStart, sanitizerSourceEnd)) {
      fail('sanitizer source provenance changed while the public bundle was being derived.');
    }
    candidateSourceBinding(privateMetadata, sanitizerSourceEnd);
    const implementationAtCommit = await sanitizerImplementationBinding(
      candidateRepositoryRoot,
      candidateSource,
    );
    if (JSON.stringify(implementationAtCommit) !== JSON.stringify(implementationFiles)) {
      fail('sanitizer implementation changed while the public bundle was being derived.');
    }
    await assertPathAbsent(outputDirectory, 'public output directory');
    await rename(stagingDirectory, outputDirectory);
    committed = true;
  } finally {
    if (!committed) await rm(stagingDirectory, { recursive: true, force: true });
  }

  return Object.freeze({
    schemaVersion: 2,
    policyId: POLICY_ID,
    runId: privateBundle.manifest.runId,
    privateBundleLabel: LIVE_PRIVATE_BUNDLE_LABEL,
    publicBundleLabel: LIVE_PUBLIC_BUNDLE_LABEL,
    privateArtifactManifestSha256: privateManifestSha256,
    publicArtifactManifestSha256: publicManifestSha256,
    candidateSource,
    sanitizerImplementation: implementationFiles,
    copiedArtifactCount: derivedContents.size,
  });
}

export function serializeLiveEvidenceSanitizerResult(result) {
  return `${JSON.stringify(result, null, 2)}\n`;
}

async function main() {
  const [privateRunDirectory, publicOutputDirectory, ...extra] = process.argv.slice(2);
  if (!privateRunDirectory || !publicOutputDirectory || extra.length > 0) {
    process.stderr.write(
      'Usage: node scripts/sanitize-live-evidence.mjs <private-run-directory> <new-public-output-directory>\n',
    );
    process.exitCode = 2;
    return;
  }
  const result = await sanitizeLiveEvidence(privateRunDirectory, publicOutputDirectory);
  process.stdout.write(serializeLiveEvidenceSanitizerResult(result));
}

if (process.argv[1]
  && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
