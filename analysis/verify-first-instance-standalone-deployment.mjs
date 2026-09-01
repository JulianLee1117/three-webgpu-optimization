import { createHash } from 'node:crypto';
import {
  lstat,
  readFile,
  readdir,
} from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { isDeepStrictEqual, promisify } from 'node:util';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { brotliDecompress } from 'node:zlib';

import {
  summarizeFirstInstanceStandaloneDeployment,
} from './first-instance-standalone-deployment-summary.mjs';
import {
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_KIND,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_COUNT,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSION_COUNT,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSIONS_PER_MATRIX,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIAL_COUNT,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIALS_PER_SESSION,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES,
  buildFirstInstanceStandaloneDeploymentPlan,
  validateFirstInstanceStandaloneDeploymentPlan,
} from '../src/benchmark/first-instance-standalone-deployment-plan.js';
import {
  executionDependencyClosuresMatch,
  validateCandidateViteRuntimeAudit,
} from '../scripts/execution-dependency-closure.mjs';
import {
  evaluateLiveFirstInstanceAdapterTelemetryAssociation,
} from '../scripts/live-first-instance-adapter-telemetry-association.mjs';
import {
  TELEMETRY_CSV_FIELDS,
  compareComputeProcessIdentitySets,
  createNvidiaTelemetryCoverageAudit,
  parseCsvRecord,
  summarizeTelemetryRows,
} from '../scripts/nvidia-telemetry.mjs';
import {
  sourceProvenanceMatches,
} from '../scripts/source-provenance.mjs';
import {
  validateLiveFirstInstanceForcedFeatureOffGate,
} from '../scripts/live-first-instance-evidence-validation.mjs';
import {
  allocateBalancedCounts,
  allocateVisibleCounts,
  prefixBases,
} from '../src/scenes/fixed-subsets.js';
import {
  createFirstInstanceLaneShaderEvidence,
} from '../src/validation/first-instance-shader-evidence.js';
import {
  createFirstInstanceAddressChallengeExpected,
  createFirstInstanceAddressChallengeTargetShape,
} from '../src/validation/first-instance-address-challenge.js';

const brotliDecompressAsync = promisify(brotliDecompress);
const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;
const LAYOUT = 'baseline';
const SCENARIO_SEED = 0xb1ad_2026;
const MAXIMUM_CPU_TIMER_QUANTUM_MS = 0.01;
const MAXIMUM_TIMESTAMP_QUANTUM_NS = 1_000;
const POST_DISCONNECT_DELAY_MS = 2_000;
const FULL_EXECUTION_MODE = 'full-candidate';
const SMOKE_EXECUTION_MODE = 'smoke-excluded';
const MAX_MANIFEST_BYTES = 16 * 1024 * 1024;
const MAX_JSON_ARTIFACT_BYTES = 512 * 1024 * 1024;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const UTF8_DECODER = new TextDecoder('utf-8', { fatal: true });
const BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
]);

function reject(message) {
  throw new Error(`Standalone deployment verification rejected: ${message}`);
}

function record(value, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    reject(`${label} must be an object.`);
  }
  return value;
}

function array(value, label) {
  if (!Array.isArray(value)) reject(`${label} must be an array.`);
  return value;
}

function exactKeys(value, expectedKeys, label) {
  const candidate = record(value, label);
  const actual = Object.keys(candidate).sort();
  const expected = [...expectedKeys].sort();
  if (actual.length !== expected.length
    || actual.some((key, index) => key !== expected[index])) {
    reject(`${label} has an unexpected schema.`);
  }
  return candidate;
}

function requireCondition(condition, message) {
  if (!condition) reject(message);
}

function requireSame(actual, expected, label) {
  if (!isDeepStrictEqual(actual, expected)) reject(`${label} differs from its commitment.`);
}

function nonemptyString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') reject(`${label} must be nonempty.`);
  return value;
}

function safeInteger(value, label, { minimum = 0, maximum = Number.MAX_SAFE_INTEGER } = {}) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    reject(`${label} must be a safe integer in [${minimum}, ${maximum}].`);
  }
  return value;
}

function finiteNumber(value, label, { minimum = -Infinity, exclusive = false } = {}) {
  if (!Number.isFinite(value) || (exclusive ? value <= minimum : value < minimum)) {
    reject(`${label} must be finite and ${exclusive ? '>' : '>='} ${minimum}.`);
  }
  return value;
}

function timestamp(value, label) {
  const parsed = typeof value === 'string' ? Date.parse(value) : NaN;
  if (!Number.isFinite(parsed)) reject(`${label} must be a valid timestamp.`);
  return parsed;
}

function sha256Bytes(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function sha256Text(value) {
  return sha256Bytes(Buffer.from(value, 'utf8'));
}

function canonicalRelativePath(value, label) {
  nonemptyString(value, label);
  if (value.includes('\\') || path.posix.isAbsolute(value)
    || path.posix.normalize(value) !== value
    || value === '.' || value.startsWith('../') || value.includes('/../')) {
    reject(`${label} is not a canonical safe relative path.`);
  }
  return value;
}

function descriptorPath(runDirectory, relativePath) {
  const resolved = path.resolve(runDirectory, ...relativePath.split('/'));
  const relative = path.relative(runDirectory, resolved);
  if (relative === '' || relative.startsWith('..') || path.isAbsolute(relative)) {
    reject(`artifact path ${JSON.stringify(relativePath)} escapes the run directory.`);
  }
  return resolved;
}

async function readStableRegularFile(filename, maximumBytes, label) {
  const before = await lstat(filename).catch((error) => {
    reject(`${label} is unavailable: ${error?.code ?? error?.message ?? String(error)}.`);
  });
  if (!before.isFile() || before.isSymbolicLink()) reject(`${label} must be a regular file.`);
  if (before.size > maximumBytes) reject(`${label} exceeds its byte limit.`);
  const bytes = await readFile(filename);
  const after = await lstat(filename);
  if (!after.isFile() || after.isSymbolicLink()
    || before.dev !== after.dev || before.ino !== after.ino
    || before.size !== after.size || before.mtimeMs !== after.mtimeMs
    || bytes.length !== before.size) {
    reject(`${label} changed while it was read.`);
  }
  return bytes;
}

function parseJsonBytes(bytes, label) {
  let text;
  try {
    text = UTF8_DECODER.decode(bytes);
  } catch {
    reject(`${label} is not valid UTF-8.`);
  }
  try {
    return JSON.parse(text);
  } catch {
    reject(`${label} is not valid JSON.`);
  }
}

function validateJsonDescriptor(value, label) {
  const descriptor = exactKeys(value, ['path', 'encoding', 'byteLength', 'sha256'], label);
  canonicalRelativePath(descriptor.path, `${label}.path`);
  requireCondition(descriptor.encoding === 'json-utf8', `${label}.encoding is invalid.`);
  safeInteger(descriptor.byteLength, `${label}.byteLength`, {
    minimum: 1,
    maximum: MAX_JSON_ARTIFACT_BYTES,
  });
  requireCondition(SHA256_PATTERN.test(descriptor.sha256), `${label}.sha256 is invalid.`);
  return descriptor;
}

function validateBrotliDescriptor(value, label) {
  const descriptor = exactKeys(value, [
    'path', 'encoding', 'jsonByteLength', 'jsonSha256',
    'brotliByteLength', 'brotliSha256',
  ], label);
  canonicalRelativePath(descriptor.path, `${label}.path`);
  requireCondition(descriptor.encoding === 'brotli-json-utf8',
    `${label}.encoding is invalid.`);
  safeInteger(descriptor.jsonByteLength, `${label}.jsonByteLength`, {
    minimum: 1,
    maximum: MAX_JSON_ARTIFACT_BYTES,
  });
  safeInteger(descriptor.brotliByteLength, `${label}.brotliByteLength`, {
    minimum: 1,
    maximum: MAX_JSON_ARTIFACT_BYTES,
  });
  requireCondition(SHA256_PATTERN.test(descriptor.jsonSha256),
    `${label}.jsonSha256 is invalid.`);
  requireCondition(SHA256_PATTERN.test(descriptor.brotliSha256),
    `${label}.brotliSha256 is invalid.`);
  return descriptor;
}

function validateRawDescriptor(value, label, encoding) {
  const descriptor = exactKeys(value, ['path', 'encoding', 'byteLength', 'sha256'], label);
  canonicalRelativePath(descriptor.path, `${label}.path`);
  requireCondition(descriptor.encoding === encoding, `${label}.encoding is invalid.`);
  safeInteger(descriptor.byteLength, `${label}.byteLength`, {
    minimum: 1,
    maximum: MAX_JSON_ARTIFACT_BYTES,
  });
  requireCondition(SHA256_PATTERN.test(descriptor.sha256), `${label}.sha256 is invalid.`);
  return descriptor;
}

function collectDescriptors(manifest) {
  const artifacts = exactKeys(manifest.artifacts, [
    'plan', 'executionIdentityStart', 'executionIdentityEnd',
    'forcedFeatureOffGates', 'trials', 'sessions', 'matrices', 'telemetry',
    'viteRuntimeAudit', 'journal',
  ], 'manifest.artifacts');
  const descriptors = [];
  const add = (descriptor, encoding, label) => {
    let validated;
    if (encoding === 'json') validated = validateJsonDescriptor(descriptor, label);
    else if (encoding === 'brotli') validated = validateBrotliDescriptor(descriptor, label);
    else validated = validateRawDescriptor(descriptor, label, encoding);
    descriptors.push({ descriptor: validated, encoding, label });
  };
  add(artifacts.plan, 'json', 'manifest.artifacts.plan');
  add(artifacts.executionIdentityStart, 'json',
    'manifest.artifacts.executionIdentityStart');
  add(artifacts.executionIdentityEnd, 'json',
    'manifest.artifacts.executionIdentityEnd');
  array(artifacts.forcedFeatureOffGates,
    'manifest.artifacts.forcedFeatureOffGates').forEach((entry, index) => {
    exactKeys(entry, ['matrixId', 'artifact', 'browserLifecycle'],
      `manifest.artifacts.forcedFeatureOffGates[${index}]`);
    add(entry.artifact, 'brotli',
      `manifest.artifacts.forcedFeatureOffGates[${index}].artifact`);
  });
  array(artifacts.trials, 'manifest.artifacts.trials').forEach((entry, index) => {
    exactKeys(entry, [
      'trialId', 'planIndex', 'matrixIndex', 'sessionId', 'assignedLaneId',
      'visibilityFraction', 'visibilityExposure', 'sessionNamespace',
      'shaderCaptureOrdinals', 'artifact',
    ], `manifest.artifacts.trials[${index}]`);
    add(entry.artifact, 'brotli', `manifest.artifacts.trials[${index}].artifact`);
  });
  array(artifacts.sessions, 'manifest.artifacts.sessions').forEach((entry, index) => {
    exactKeys(entry, [
      'sessionId', 'globalSessionIndex', 'matrixSessionIndex', 'sessionNamespace',
      'browserInstanceSerial', 'artifact',
    ], `manifest.artifacts.sessions[${index}]`);
    add(entry.artifact, 'json', `manifest.artifacts.sessions[${index}].artifact`);
  });
  array(artifacts.matrices, 'manifest.artifacts.matrices').forEach((entry, index) => {
    exactKeys(entry, [
      'matrixId', 'matrixIndex', 'matrixOrdinal', 'sessionCount', 'trialCount',
      'telemetry', 'identity', 'artifact',
    ], `manifest.artifacts.matrices[${index}]`);
    add(entry.artifact, 'json', `manifest.artifacts.matrices[${index}].artifact`);
  });
  array(artifacts.telemetry, 'manifest.artifacts.telemetry').forEach((entry, index) => {
    const label = `manifest.artifacts.telemetry[${index}]`;
    if (entry?.status === 'excluded') {
      exactKeys(entry, ['status', 'artifact'], label);
      add(entry.artifact, 'json', `${label}.artifact`);
    } else {
      exactKeys(entry, ['status', 'eligibility', 'artifact', 'csvArtifact'], label);
      requireCondition(entry.status === 'captured', `${label}.status is invalid.`);
      add(entry.artifact, 'json', `${label}.artifact`);
      add(entry.csvArtifact, 'csv-utf8', `${label}.csvArtifact`);
    }
  });
  add(artifacts.viteRuntimeAudit, 'json', 'manifest.artifacts.viteRuntimeAudit');
  array(artifacts.journal, 'manifest.artifacts.journal').forEach(
    (descriptor, index) => add(descriptor, 'json', `manifest.artifacts.journal[${index}]`),
  );

  const paths = new Set();
  for (const { descriptor, label } of descriptors) {
    if (descriptor.path === 'manifest.json') reject(`${label} cannot declare manifest.json.`);
    if (paths.has(descriptor.path)) reject(`artifact path ${descriptor.path} is declared twice.`);
    paths.add(descriptor.path);
  }
  return descriptors;
}

async function exactInventory(runDirectory, declaredPaths) {
  const rootStats = await lstat(runDirectory).catch((error) => {
    reject(`run directory is unavailable: ${error?.code ?? error?.message ?? String(error)}.`);
  });
  if (!rootStats.isDirectory() || rootStats.isSymbolicLink()) {
    reject('run directory must be a real directory, not a symlink.');
  }
  const actualFiles = new Set();
  const actualDirectories = new Set();
  const visit = async (directory, relativeDirectory = '') => {
    const entries = await readdir(directory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) {
      const relativePath = relativeDirectory.length === 0
        ? entry.name
        : `${relativeDirectory}/${entry.name}`;
      const absolutePath = path.join(directory, entry.name);
      const stats = await lstat(absolutePath);
      if (stats.isSymbolicLink()) reject(`inventory entry ${relativePath} is a symlink.`);
      if (stats.isDirectory()) {
        actualDirectories.add(relativePath);
        await visit(absolutePath, relativePath);
      } else if (stats.isFile()) actualFiles.add(relativePath);
      else reject(`inventory entry ${relativePath} is not a regular file or directory.`);
    }
  };
  await visit(runDirectory);

  const expectedFiles = new Set(['manifest.json', ...declaredPaths]);
  const expectedDirectories = new Set();
  for (const relativePath of declaredPaths) {
    let parent = path.posix.dirname(relativePath);
    while (parent !== '.') {
      expectedDirectories.add(parent);
      parent = path.posix.dirname(parent);
    }
  }
  requireSame([...actualFiles].sort(), [...expectedFiles].sort(), 'run file inventory');
  requireSame([...actualDirectories].sort(), [...expectedDirectories].sort(),
    'run directory inventory');
}

async function readFinalManifest(runDirectory) {
  const filename = path.join(path.resolve(runDirectory), 'manifest.json');
  const bytes = await readStableRegularFile(filename, MAX_MANIFEST_BYTES, 'final manifest');
  return parseJsonBytes(bytes, 'final manifest');
}

async function readDeclaredDescriptor(runDirectory, descriptorRecord) {
  const { descriptor, encoding, label } = descriptorRecord;
  const filename = descriptorPath(runDirectory, descriptor.path);
  if (encoding === 'brotli') {
    const compressed = await readStableRegularFile(
      filename,
      descriptor.brotliByteLength,
      `${label} Brotli file`,
    );
    requireCondition(compressed.length === descriptor.brotliByteLength,
      `${label}.brotliByteLength differs from the file.`);
    requireCondition(sha256Bytes(compressed) === descriptor.brotliSha256,
      `${label}.brotliSha256 differs from the file.`);
    let jsonBytes;
    try {
      jsonBytes = await brotliDecompressAsync(compressed, {
        maxOutputLength: descriptor.jsonByteLength,
      });
    } catch (error) {
      reject(`${label} Brotli payload cannot be decoded: ${error?.code ?? error?.message}.`);
    }
    requireCondition(jsonBytes.length === descriptor.jsonByteLength,
      `${label}.jsonByteLength differs after decompression.`);
    requireCondition(sha256Bytes(jsonBytes) === descriptor.jsonSha256,
      `${label}.jsonSha256 differs after decompression.`);
    return parseJsonBytes(jsonBytes, `${label} JSON payload`);
  }
  const bytes = await readStableRegularFile(filename, descriptor.byteLength, `${label} file`);
  requireCondition(bytes.length === descriptor.byteLength,
    `${label}.byteLength differs from the file.`);
  requireCondition(sha256Bytes(bytes) === descriptor.sha256,
    `${label}.sha256 differs from the file.`);
  return encoding === 'json'
    ? parseJsonBytes(bytes, `${label} file`)
    : UTF8_DECODER.decode(bytes);
}

/**
 * Verify the exact inventory up front and expose hash-checking lazy reads. A
 * complete run can contain gigabytes of logical JSON, so callers must not
 * materialize all trial bodies concurrently.
 */
export async function readFirstInstanceStandaloneDeploymentDeclaredArtifacts(
  runDirectory,
  manifest,
) {
  const resolvedDirectory = path.resolve(runDirectory);
  const descriptorRecords = collectDescriptors(manifest);
  await exactInventory(
    resolvedDirectory,
    descriptorRecords.map(({ descriptor }) => descriptor.path),
  );
  const byPath = new Map(descriptorRecords.map((entry) => [entry.descriptor.path, entry]));
  return Object.freeze({
    runDirectory: resolvedDirectory,
    artifactCount: descriptorRecords.length,
    descriptorRecords: Object.freeze([...descriptorRecords]),
    async read(descriptor, label = descriptor?.path ?? 'artifact') {
      const declared = byPath.get(descriptor?.path);
      requireCondition(declared !== undefined, `${label} is not declared by the manifest.`);
      requireSame(descriptor, declared.descriptor, `${label} descriptor`);
      return readDeclaredDescriptor(resolvedDirectory, declared);
    },
  });
}

function artifactValue(values, descriptor, label) {
  if (!values.has(descriptor.path)) reject(`${label} was not decoded.`);
  return values.get(descriptor.path);
}

function validateManifestEnvelope(manifest) {
  exactKeys(manifest, [
    'schemaVersion', 'kind', 'executionMode', 'analysisEligible', 'scope',
    'runId', 'planSha256', 'startedAt', 'completedAt', 'elapsedMs', 'browser',
    'fixedWorkload', 'analysis', 'executionPolicy', 'identity',
    'browserLifecycles', 'lifecycleValidation', 'telemetryPolicy', 'artifacts',
    'completion',
  ], 'final manifest');
  requireCondition(manifest.schemaVersion === 1, 'manifest.schemaVersion is invalid.');
  const smokeMode = manifest.executionMode === SMOKE_EXECUTION_MODE;
  requireCondition(smokeMode || manifest.executionMode === FULL_EXECUTION_MODE,
    'manifest.executionMode is invalid.');
  requireCondition(manifest.kind === (smokeMode
    ? 'first-instance-standalone-deployment-smoke-capture-manifest'
    : 'first-instance-standalone-deployment-full-capture-manifest')
    && manifest.analysisEligible === !smokeMode
    && manifest.scope === (smokeMode
      ? 'two fresh P/F sessions with both visibility trials; excluded from analysis and decision'
      : 'complete two-matrix candidate capture awaiting independent analysis and verification'),
  'manifest mode/scope is invalid.');
  nonemptyString(manifest.runId, 'manifest.runId');
  requireCondition(SHA256_PATTERN.test(manifest.planSha256),
    'manifest.planSha256 is invalid.');
  const startedAt = timestamp(manifest.startedAt, 'manifest.startedAt');
  const completedAt = timestamp(manifest.completedAt, 'manifest.completedAt');
  finiteNumber(manifest.elapsedMs, 'manifest.elapsedMs', { minimum: 0, exclusive: true });
  requireCondition(completedAt >= startedAt, 'manifest completion precedes its start.');
  exactKeys(manifest.browser, [
    'executable', 'arguments', 'launchApi', 'profilePolicy', 'persistentContext',
  ], 'manifest.browser');
  nonemptyString(manifest.browser.executable, 'manifest.browser.executable');
  requireSame(manifest.browser.arguments, [...BROWSER_ARGS], 'manifest browser arguments');
  requireCondition(manifest.browser.launchApi === 'chromium.launch'
    && manifest.browser.profilePolicy === 'fresh-playwright-temporary-profile-per-process'
    && manifest.browser.persistentContext === false,
  'manifest browser/profile policy is invalid.');
  requireSame(manifest.fixedWorkload, {
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    layout: LAYOUT,
    visibilityLevels: [0.99, 0.2],
    scenarioSeed: SCENARIO_SEED,
    warmupFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES,
    measuredFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
    measuredBlockSize: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE,
    measuredBlockCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS,
  }, 'manifest fixed workload');
  requireSame(manifest.analysis, {
    invokedByCaptureRunner: false,
    summaryArtifact: null,
    numericalDecision: null,
    nextStep: smokeMode
      ? 'none; smoke artifacts may only be structurally verified'
      : 'run the independent standalone analyzer and verifier',
  }, 'manifest capture-runner analysis boundary');
  requireSame(manifest.executionPolicy, {
    retryCount: 0,
    replacementAllowed: false,
    outlierRemovalAllowed: false,
    efficacyStoppingAllowed: false,
    postDisconnectDelayMs: POST_DISCONNECT_DELAY_MS,
    oppositeLaneConstructedInMeasuredSessions: false,
  }, 'manifest execution policy');
  const expected = smokeMode
    ? { matrices: 1, sessions: 2, trials: 4, browsers: 3, entries: 3 }
    : {
      matrices: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_COUNT,
      sessions: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSION_COUNT,
      trials: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIAL_COUNT,
      browsers: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSION_COUNT
        + FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_COUNT,
      entries: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSION_COUNT
        + FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_COUNT,
    };
  requireSame(manifest.completion, {
    frozenMatrixCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_COUNT,
    frozenSessionCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSION_COUNT,
    frozenTrialCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIAL_COUNT,
    executedMatrixCount: expected.matrices,
    executedSessionCount: expected.sessions,
    executedTrialCount: expected.trials,
    persistedMatrixCount: expected.matrices,
    persistedSessionCount: expected.sessions,
    persistedTrialCount: expected.trials,
    shaderChallengeCount: expected.trials * 6,
    browserProcessCount: expected.browsers,
    entryDocumentCount: expected.entries,
    allBrowsersClosed: true,
    overlapDetected: false,
  }, 'manifest completion');
  if (smokeMode) {
    requireSame(manifest.telemetryPolicy,
      { status: 'excluded-by-frozen-smoke-policy', analysisEligible: false },
      'manifest telemetry policy');
  } else {
    exactKeys(manifest.telemetryPolicy, [
      'status', 'provider', 'intervalMs', 'processSnapshotPolicy',
      'allMatrixGatesPassed',
    ], 'manifest telemetry policy');
    requireCondition(manifest.telemetryPolicy.status === 'captured-per-matrix'
      && manifest.telemetryPolicy.provider === 'nvidia-smi'
      && manifest.telemetryPolicy.intervalMs === 250
      && manifest.telemetryPolicy.processSnapshotPolicy
        === 'pre-first-session-and-after-every-browser-disconnect'
      && typeof manifest.telemetryPolicy.allMatrixGatesPassed === 'boolean',
    'manifest telemetry policy is invalid.');
  }
  return { smokeMode, expected };
}

function selectedExecution(plan, smokeMode) {
  if (!smokeMode) return {
    matrices: [...plan.matrices],
    sessions: [...plan.sessions],
    trials: [...plan.trials],
  };
  const sessions = plan.sessions.slice(0, 2);
  const ids = new Set(sessions.map(({ sessionId }) => sessionId));
  return {
    matrices: [plan.matrices[0]],
    sessions,
    trials: plan.trials.filter((trial) => ids.has(trial.sessionId)),
  };
}

function validatePlanArtifact(manifest, values, expectedPlan, selection, mode) {
  requireCondition(manifest.artifacts.plan.path === 'plan.json',
    'plan descriptor path is invalid.');
  const artifact = artifactValue(values, manifest.artifacts.plan, 'plan artifact');
  exactKeys(artifact, [
    'schemaVersion', 'kind', 'runId', 'executionMode', 'analysisEligible',
    'planSha256', 'frozenPlan', 'executedMatrixIds', 'executedSessionIds',
    'executedTrialIds', 'smokeBoundary', 'executionPolicy',
  ], 'plan artifact');
  requireCondition(artifact.schemaVersion === 1
    && artifact.kind === 'first-instance-standalone-deployment-execution-plan'
    && artifact.runId === manifest.runId
    && artifact.executionMode === manifest.executionMode
    && artifact.analysisEligible === !mode.smokeMode
    && artifact.planSha256 === manifest.planSha256,
  'plan artifact identity is invalid.');
  requireSame(artifact.frozenPlan, expectedPlan, 'retained frozen plan');
  requireSame(artifact.executedMatrixIds,
    selection.matrices.map(({ matrixId }) => matrixId), 'executed matrix IDs');
  requireSame(artifact.executedSessionIds,
    selection.sessions.map(({ sessionId }) => sessionId), 'executed session IDs');
  requireSame(artifact.executedTrialIds,
    selection.trials.map(({ trialId }) => trialId), 'executed trial IDs');
  requireCondition(artifact.smokeBoundary === (mode.smokeMode
    ? 'first two fresh sessions and both ordered visibility trials in each'
    : null),
  'plan smoke boundary is invalid.');
  requireSame(artifact.executionPolicy, {
    retryCount: 0,
    replacementAllowed: false,
    outlierRemovalAllowed: false,
    efficacyStoppingAllowed: false,
    matrixTwoRunsRegardlessOfMatrixOneNumericalOrStableEnvironmentOutcome: true,
    postDisconnectDelayMs: POST_DISCONNECT_DELAY_MS,
    browserProfilePolicy: 'fresh-playwright-temporary-profile-per-process',
  }, 'plan execution policy');
  return artifact;
}

function validateExecutionIdentity(manifest, values) {
  exactKeys(manifest.identity, [
    'planSha256', 'executionIdentity', 'sourceProvenanceStart',
    'sourceProvenanceEnd', 'sourceMatched', 'executionDependencyClosureStart',
    'executionDependencyClosureEnd', 'executionDependencyClosureMatched',
    'viteRuntimePolicyId', 'viteRuntimeModulesSha256',
    'globalEnvironmentAndWorkload',
  ], 'manifest.identity');
  const startDescriptor = manifest.artifacts.executionIdentityStart;
  const endDescriptor = manifest.artifacts.executionIdentityEnd;
  requireCondition(startDescriptor.path === 'execution-identity-start.json'
    && endDescriptor.path === 'execution-identity-end.json',
  'execution identity descriptor paths are invalid.');
  const start = artifactValue(values, startDescriptor, 'execution identity start');
  const end = artifactValue(values, endDescriptor, 'execution identity end');
  exactKeys(start, [
    'schemaVersion', 'kind', 'runId', 'executionMode', 'planSha256',
    'executionIdentity', 'sourceProvenance', 'executionDependencyClosure',
  ], 'execution identity start');
  exactKeys(end, [
    'schemaVersion', 'kind', 'runId', 'executionMode', 'planSha256',
    'sourceProvenance', 'executionDependencyClosure', 'sourceMatchesStart',
    'executionDependencyClosureMatchesStart',
  ], 'execution identity end');
  requireCondition(start.schemaVersion === 1
    && start.kind === 'first-instance-standalone-execution-identity-start'
    && end.schemaVersion === 1
    && end.kind === 'first-instance-standalone-execution-identity-end'
    && start.runId === manifest.runId && end.runId === manifest.runId
    && start.executionMode === manifest.executionMode
    && end.executionMode === manifest.executionMode
    && start.planSha256 === manifest.planSha256
    && end.planSha256 === manifest.planSha256,
  'execution identity artifact identity is invalid.');
  requireSame(manifest.identity.planSha256, manifest.planSha256,
    'manifest identity plan SHA-256');
  requireSame(manifest.identity.executionIdentity, start.executionIdentity,
    'manifest/start execution identity');
  requireSame(manifest.identity.sourceProvenanceStart, start.sourceProvenance,
    'manifest/start source provenance');
  requireSame(manifest.identity.sourceProvenanceEnd, end.sourceProvenance,
    'manifest/end source provenance');
  requireSame(manifest.identity.executionDependencyClosureStart,
    start.executionDependencyClosure, 'manifest/start dependency closure');
  requireSame(manifest.identity.executionDependencyClosureEnd,
    end.executionDependencyClosure, 'manifest/end dependency closure');
  requireCondition(manifest.identity.sourceMatched === true
    && manifest.identity.executionDependencyClosureMatched === true
    && end.sourceMatchesStart === true
    && end.executionDependencyClosureMatchesStart === true
    && sourceProvenanceMatches(start.sourceProvenance, end.sourceProvenance)
    && executionDependencyClosuresMatch(
      start.executionDependencyClosure,
      end.executionDependencyClosure,
    ),
  'source/dependency closure changed during capture.');
  for (const [label, provenance] of [
    ['start', start.sourceProvenance],
    ['end', end.sourceProvenance],
  ]) {
    requireCondition(provenance?.status === 'available'
      && provenance?.captureStable === true
      && provenance?.dirty === false
      && provenance?.packageLockTracked === true,
    `${label} source provenance is not clean, tracked, and stable.`);
  }
  requireCondition(start.executionIdentity?.sourceCommit === start.sourceProvenance.commit
    && start.executionIdentity?.sourceTree === start.sourceProvenance.tree
    && start.executionIdentity?.trackedFilesSha256
      === start.sourceProvenance.trackedFilesSha256
    && start.executionIdentity?.packageLockSha256
      === start.sourceProvenance.packageLockSha256
    && start.executionIdentity?.executionDependencyClosureSha256
      === start.executionDependencyClosure.sha256
    && start.executionIdentity?.viteRuntimePolicyId
      === manifest.identity.viteRuntimePolicyId
    && start.executionIdentity?.viteRuntimeAuditBinding === 'final-capture-manifest'
    && SHA256_PATTERN.test(manifest.identity.viteRuntimeModulesSha256),
  'execution identity reference is internally inconsistent.');
  return start.executionIdentity;
}

function validateBrowserLifecycleRecord(value, expected, previous, label) {
  exactKeys(value, [
    'schemaVersion', 'kind', 'browserInstanceSerial', 'role', 'matrixOrdinal',
    'sessionId', 'globalSessionIndex', 'sessionNamespace', 'launchApi',
    'persistentContext', 'profilePolicy', 'userDataDirectoryExposed',
    'profileReused', 'launchArguments', 'launchedAt', 'launchedRunElapsedMs',
    'priorBrowserInstanceSerial', 'priorBrowserDisconnectedAt',
    'previousDisconnectToLaunchGapMs', 'contextCreatedAt', 'pageCreatedAt',
    'closedAt', 'disconnectedAt', 'disconnectedRunElapsedMs',
    'contextCountBeforeClose', 'pageCountBeforeClose', 'disconnectedEventCount',
    'closedBeforeNextLaunch', 'postDisconnectDelay',
  ], label);
  requireCondition(value.schemaVersion === 1
    && value.kind === 'first-instance-standalone-browser-profile-lifecycle'
    && value.browserInstanceSerial === expected.browserInstanceSerial
    && value.role === expected.role
    && value.matrixOrdinal === expected.matrixOrdinal
    && value.sessionId === (expected.session?.sessionId ?? null)
    && value.globalSessionIndex === (expected.session?.globalSessionIndex ?? null)
    && value.sessionNamespace === (expected.session === null
      ? null
      : `${expected.session.sessionId}:browser-${expected.browserInstanceSerial}`)
    && value.launchApi === 'chromium.launch'
    && value.persistentContext === false
    && value.profilePolicy === 'fresh-playwright-temporary-profile-per-process'
    && value.userDataDirectoryExposed === false
    && value.profileReused === false
    && isDeepStrictEqual(value.launchArguments, [...BROWSER_ARGS])
    && value.contextCountBeforeClose === 1
    && value.pageCountBeforeClose === 1
    && value.disconnectedEventCount === 1
    && value.closedBeforeNextLaunch === true,
  `${label} does not prove a fresh one-page browser lifecycle.`);
  const launchedAt = timestamp(value.launchedAt, `${label}.launchedAt`);
  const contextCreatedAt = timestamp(value.contextCreatedAt, `${label}.contextCreatedAt`);
  const pageCreatedAt = timestamp(value.pageCreatedAt, `${label}.pageCreatedAt`);
  const disconnectedAt = timestamp(value.disconnectedAt, `${label}.disconnectedAt`);
  const closedAt = timestamp(value.closedAt, `${label}.closedAt`);
  requireCondition(launchedAt <= contextCreatedAt
    && contextCreatedAt <= pageCreatedAt
    && pageCreatedAt <= disconnectedAt
    && disconnectedAt <= closedAt,
  `${label} wall-clock chronology is invalid.`);
  for (const field of ['launchedRunElapsedMs', 'disconnectedRunElapsedMs']) {
    finiteNumber(value[field], `${label}.${field}`, { minimum: 0 });
  }
  requireCondition(value.disconnectedRunElapsedMs >= value.launchedRunElapsedMs,
    `${label} monotonic chronology is invalid.`);
  exactKeys(value.postDisconnectDelay, [
    'requestedMs', 'startedAt', 'startedRunElapsedMs', 'completedAt',
    'completedRunElapsedMs', 'elapsedMs',
  ], `${label}.postDisconnectDelay`);
  timestamp(value.postDisconnectDelay.startedAt, `${label}.postDisconnectDelay.startedAt`);
  timestamp(value.postDisconnectDelay.completedAt, `${label}.postDisconnectDelay.completedAt`);
  requireCondition(value.postDisconnectDelay.requestedMs === POST_DISCONNECT_DELAY_MS
    && value.postDisconnectDelay.startedRunElapsedMs >= value.disconnectedRunElapsedMs
    && value.postDisconnectDelay.completedRunElapsedMs
      >= value.postDisconnectDelay.startedRunElapsedMs
    && value.postDisconnectDelay.elapsedMs >= POST_DISCONNECT_DELAY_MS,
  `${label} post-disconnect interval is invalid.`);
  if (previous === null) {
    requireCondition(value.priorBrowserInstanceSerial === null
      && value.priorBrowserDisconnectedAt === null
      && value.previousDisconnectToLaunchGapMs === null,
    `${label} unexpectedly declares a predecessor.`);
  } else {
    requireCondition(value.priorBrowserInstanceSerial === previous.browserInstanceSerial
      && value.priorBrowserDisconnectedAt === previous.disconnectedAt
      && value.previousDisconnectToLaunchGapMs
        === value.launchedRunElapsedMs - previous.disconnectedRunElapsedMs
      && value.previousDisconnectToLaunchGapMs >= POST_DISCONNECT_DELAY_MS
      && value.launchedRunElapsedMs
        >= previous.postDisconnectDelay.completedRunElapsedMs,
    `${label} overlaps or lacks its exact predecessor gap.`);
  }
  return value;
}

export function verifyStandaloneBrowserLifecycleChain(records, selection) {
  array(records, 'browser lifecycle records');
  record(selection, 'browser lifecycle selection');
  const expected = [];
  let serial = 1;
  for (const matrix of selection.matrices) {
    expected.push({
      browserInstanceSerial: serial,
      role: 'forced-feature-off-gate',
      matrixOrdinal: matrix.matrixOrdinal,
      session: null,
    });
    serial += 1;
    for (const session of selection.sessions.filter(
      (candidate) => candidate.matrixIndex === matrix.matrixIndex,
    )) {
      expected.push({
        browserInstanceSerial: serial,
        role: 'standalone-measured-session',
        matrixOrdinal: matrix.matrixOrdinal,
        session,
      });
      serial += 1;
    }
  }
  requireCondition(records.length === expected.length,
    `browser lifecycle count ${records.length} differs from ${expected.length}.`);
  const namespaces = new Set();
  records.forEach((value, index) => {
    validateBrowserLifecycleRecord(
      value,
      expected[index],
      index === 0 ? null : records[index - 1],
      `browser lifecycle ${index}`,
    );
    if (value.sessionNamespace !== null) {
      requireCondition(!namespaces.has(value.sessionNamespace),
        `browser lifecycle reuses session namespace ${value.sessionNamespace}.`);
      namespaces.add(value.sessionNamespace);
    }
  });
  return {
    schemaVersion: 1,
    kind: 'first-instance-standalone-browser-lifecycle-chain-validation',
    pass: true,
    reasons: [],
  };
}

function validateInteractiveChallenge(value, ordinal, role, usedNonces, label) {
  exactKeys(value, [
    'schemaVersion', 'kind', 'origin', 'runId', 'trialId', 'planIndex',
    'repetitionIndex', 'phase', 'role', 'captureOrdinal', 'challengeNonce',
  ], label);
  requireCondition(value.schemaVersion === 1
    && value.kind === 'live-first-instance-standalone-shader-observation-challenge'
    && value.origin === 'page-interactive'
    && value.runId === null
    && value.trialId === null
    && value.planIndex === null
    && value.repetitionIndex === null
    && value.phase === 'interactive'
    && value.role === role
    && value.captureOrdinal === ordinal
    && SHA256_PATTERN.test(value.challengeNonce)
    && !usedNonces.has(value.challengeNonce),
  `${label} interactive shader challenge is invalid or reused.`);
  usedNonces.add(value.challengeNonce);
  return value;
}

export function verifyStandaloneInteractiveChallengeForTest(
  value,
  ordinal,
  role,
  usedNonces = new Set(),
) {
  return validateInteractiveChallenge(
    value,
    ordinal,
    role,
    usedNonces,
    'interactive challenge',
  );
}

async function validateForcedFeatureOffGates(
  manifest,
  values,
  selection,
  lifecycles,
  usedNonces,
) {
  const entries = manifest.artifacts.forcedFeatureOffGates;
  requireCondition(entries.length === selection.matrices.length,
    'forced-feature-off gate count differs from the executed matrix count.');
  const results = [];
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    const matrix = selection.matrices[index];
    const expectedPath = `matrices/matrix-${String(matrix.matrixOrdinal).padStart(2, '0')}`
      + '/forced-feature-off-gate.json.br';
    requireCondition(entry.matrixId === matrix.matrixId
      && entry.artifact.path === expectedPath,
    `matrix ${matrix.matrixOrdinal} forced-off gate record is invalid.`);
    const lifecycle = lifecycles.find((candidate) => (
      candidate.role === 'forced-feature-off-gate'
        && candidate.matrixOrdinal === matrix.matrixOrdinal
    ));
    requireSame(entry.browserLifecycle, lifecycle,
      `matrix ${matrix.matrixOrdinal} gate lifecycle`);
    const artifact = artifactValue(values, entry.artifact,
      `matrix ${matrix.matrixOrdinal} forced-off gate`);
    exactKeys(artifact, [
      'schemaVersion', 'kind', 'executionMode', 'analysisEligible', 'runId',
      'matrixId', 'matrixIndex', 'matrixOrdinal', 'planSha256', 'capturedAt',
      'browserInstanceSerial', 'profilePolicy', 'capture',
      'validationRejections', 'pageErrors',
    ], `matrix ${matrix.matrixOrdinal} forced-off gate artifact`);
    requireCondition(artifact.schemaVersion === 1
      && artifact.kind === 'first-instance-standalone-matrix-forced-feature-off-gate'
      && artifact.executionMode === manifest.executionMode
      && artifact.analysisEligible === manifest.analysisEligible
      && artifact.runId === manifest.runId
      && artifact.matrixId === matrix.matrixId
      && artifact.matrixIndex === matrix.matrixIndex
      && artifact.matrixOrdinal === matrix.matrixOrdinal
      && artifact.planSha256 === manifest.planSha256
      && artifact.browserInstanceSerial === lifecycle.browserInstanceSerial
      && artifact.profilePolicy === lifecycle.profilePolicy,
    `matrix ${matrix.matrixOrdinal} forced-off gate identity is invalid.`);
    timestamp(artifact.capturedAt, `matrix ${matrix.matrixOrdinal} gate capturedAt`);
    requireSame(artifact.pageErrors, [], `matrix ${matrix.matrixOrdinal} gate page errors`);
    requireCleanGpuRecord(artifact.capture,
      `matrix ${matrix.matrixOrdinal} forced-off gate capture`);
    const recomputed = validateLiveFirstInstanceForcedFeatureOffGate(artifact.capture.gate);
    requireSame(artifact.validationRejections, recomputed,
      `matrix ${matrix.matrixOrdinal} gate validation rejections`);
    requireCondition(recomputed.length === 0,
      `matrix ${matrix.matrixOrdinal} forced-feature-off gate failed.`);
    const gate = artifact.capture.gate;
    const boot = gate.initialPageBoot;
    const pageLifecycle = gate.construction?.pageConstructionLifecycle;
    requireCondition(boot?.schemaVersion === 1
      && boot?.kind === 'first-instance-standalone-initial-page-boot'
      && boot?.bootId === 'first-instance-live-standalone-deployment-v1'
      && boot?.modeId === 'first-instance-live-standalone-deployment'
      && boot?.laneId === 'portable'
      && isDeepStrictEqual(boot?.visibilityOrder, [0.99, 0.2])
      && boot?.objectCount === OBJECT_COUNT
      && boot?.bucketCount === BUCKET_COUNT
      && boot?.layout === LAYOUT
      && boot?.initialRebuildCount === 1
      && boot?.priorStrategyConstructionCount === 0
      && pageLifecycle?.schemaVersion === 1
      && pageLifecycle?.kind === 'benchmark-page-strategy-construction-lifecycle'
      && pageLifecycle?.rebuildCount === 1
      && pageLifecycle?.strategyConstructionCount === 1
      && isDeepStrictEqual(pageLifecycle?.constructedStrategyIds,
        ['first-instance-live-standalone-deployment'])
      && pageLifecycle?.selectedStrategyId === 'first-instance-live-standalone-deployment'
      && pageLifecycle?.strictStandaloneBoot === true,
    `matrix ${matrix.matrixOrdinal} gate did not use strict standalone first construction.`);
    requireCondition(gate.pass === true
      && gate.passBeforeDisposal === true
      && gate.actualFeatureAvailable === true
      && gate.forcedFeatureAvailable === false
      && gate.separateDisposableRendererRequired === true
      && gate.timingContaminationBoundary === 'caller-owned-disposable-renderer-device'
      && gate.selection?.lane === 'portable'
      && gate.selection?.strategyId === 'fixed-slice'
      && gate.selection?.featureAvailable === false
      && gate.configuration?.objectCount === OBJECT_COUNT
      && gate.configuration?.bucketCount === BUCKET_COUNT
      && gate.configuration?.visibilityFraction === 0.99
      && gate.configuration?.scenarioSeed === SCENARIO_SEED
      && gate.configuration?.layout === LAYOUT
      && gate.timestampTrackingEnabled === false
      && gate.uncapturedErrorsDuringGate === 0
      && gate.uncapturedErrorCountAfter === gate.uncapturedErrorCountBefore
      && gate.disposal?.pass === true
      && gate.disposal?.attempted === true
      && gate.disposal?.rootDetached === true
      && gate.disposal?.indirectDetached === true,
    `matrix ${matrix.matrixOrdinal} forced-off gate deployment/disposal boundary is invalid.`);
    const expectedTrial = {
      assignedLaneId: 'portable',
      absentLaneId: 'feature',
      visibilityFraction: 0.99,
      visibilityOrderPosition: 0,
    };
    const outputValidation = validateCommandAndMembership(
      gate.output?.snapshotValidation,
      expectedTrial,
      `matrix ${matrix.matrixOrdinal} gate.output.snapshotValidation`,
    );
    const output = renderOutputIdentity(
      gate.output,
      `matrix ${matrix.matrixOrdinal} gate.output`,
    );
    requireSame(gate.correctness, gate.output.snapshotValidation.correctness,
      `matrix ${matrix.matrixOrdinal} gate correctness observations`);
    requireSame(gate.address, gate.output.snapshotValidation.address,
      `matrix ${matrix.matrixOrdinal} gate address observations`);
    const commandWords = gate.output.snapshotValidation.snapshot.commands;
    requireCondition(gate.commands?.pass === true
      && gate.commands?.commandArrayType === 'Uint32Array'
      && gate.commands?.commandWordCount === BUCKET_COUNT * 5
      && gate.commands?.commandRecordCount === BUCKET_COUNT
      && gate.commands?.drawCommandCount === BUCKET_COUNT
      && gate.commands?.commandByteOffset === 0
      && gate.commands?.fifthCommandWordsAllZero === true
      && gate.commands?.nonzeroFirstInstanceCount === 0
      && isDeepStrictEqual(gate.commands?.firstInstanceWords,
        Array.from({ length: BUCKET_COUNT }, () => 0))
      && gate.commands.firstInstanceWords.every(
        (value, bucket) => value === commandWords[bucket * 5 + 4]),
    `matrix ${matrix.matrixOrdinal} forced-off commands are not exact portable commands.`);
    const parityCapture = gate.output.snapshotValidation.shaderEvidence;
    const mainCapture = gate.shaderEvidence;
    const parityChallenge = validateInteractiveChallenge(
      parityCapture?.observationRequest,
      1,
      'render-parity',
      usedNonces,
      `matrix ${matrix.matrixOrdinal} gate parity challenge`,
    );
    const mainChallenge = validateInteractiveChallenge(
      mainCapture?.observationRequest,
      2,
      'main-validation',
      usedNonces,
      `matrix ${matrix.matrixOrdinal} gate main challenge`,
    );
    const parityShader = await validateLaneShaderRecord(
      parityCapture,
      expectedTrial,
      parityChallenge,
      `matrix ${matrix.matrixOrdinal} gate parity shader`,
    );
    const mainShader = await validateLaneShaderRecord(
      mainCapture,
      expectedTrial,
      mainChallenge,
      `matrix ${matrix.matrixOrdinal} gate main shader`,
    );
    requireSame(semanticIdentity(parityShader), semanticIdentity(mainShader),
      `matrix ${matrix.matrixOrdinal} gate shader semantics`);
    requireSame(parityShader.primedResourceIdentities, mainShader.primedResourceIdentities,
      `matrix ${matrix.matrixOrdinal} gate shader resource continuity`);
    requireSame(output.productionResourceIdentities, parityShader.primedResourceIdentities,
      `matrix ${matrix.matrixOrdinal} gate production/shader resource identity`);
    for (const identity of [
      parityShader.commitmentStaticIdentity,
      mainShader.commitmentStaticIdentity,
      output.productionStaticIdentity,
    ]) {
      for (const [key, value] of Object.entries(identity)) {
        requireSame(value, outputValidation.resourceIdentity[key],
          `matrix ${matrix.matrixOrdinal} gate ${key} lifecycle identity`);
      }
    }
    const workload = workloadIdentity(artifact.capture.workload);
    const expectedScenario = expectedStandaloneScenario(0.99);
    requireCondition(workload.scenarioSeed === SCENARIO_SEED
      && workload.objectCount === OBJECT_COUNT
      && workload.bucketCount === BUCKET_COUNT
      && workload.visibilityFraction === 0.99
      && workload.layout === LAYOUT
      && workload.expectedVisibleIdsCanonicalSha256
        === expectedScenario.expectedVisibleIdsCanonicalSha256,
    `matrix ${matrix.matrixOrdinal} gate workload is invalid.`);
    results.push({
      matrixId: matrix.matrixId,
      environment: environmentIdentity(artifact.capture.environment),
      workload,
      resourceIdentity: outputValidation.resourceIdentity,
      shaderResourceIdentity: parityShader.primedResourceIdentities,
    });
  }
  return results;
}

function validateVisibilitySwitch(value, session, baseIdentity, label) {
  exactKeys(value, [
    'switchEvidence', 'selectedConfig', 'environment', 'workload',
    'strategyLifecycle', 'strategyDiagnostics', 'cacheDiagnostics',
    'webgpuUncapturedErrors', 'webgpuDeviceLosses',
  ], label);
  requireCleanGpuRecord(value, label);
  const evidence = value.switchEvidence;
  requireCondition(evidence?.schemaVersion === 1
    && evidence?.kind === 'first-instance-live-standalone-visibility-switch'
    && evidence?.pass === true
    && evidence?.stableIdentity === true
    && evidence?.laneId === session.assignedLaneId
    && evidence?.scenarioSwitchSerial === 1
    && evidence?.fromVisibilityFraction === session.visibilityOrder[0]
    && evidence?.toVisibilityFraction === session.visibilityOrder[1]
    && evidence?.matrixVersionAfter === evidence?.matrixVersionBefore + 1
    && evidence?.boundsVersionAfter === evidence?.boundsVersionBefore + 1,
  `${label} is not the sole frozen visibility transition.`);
  requireCondition(value.selectedConfig?.strategyId
    === 'first-instance-live-standalone-deployment'
    && value.selectedConfig?.laneId === session.assignedLaneId
    && value.selectedConfig?.visibilityFraction === session.visibilityOrder[1]
    && isDeepStrictEqual(value.selectedConfig?.visibilityOrder, session.visibilityOrder),
  `${label} selected configuration changed outside visibility.`);
  const expectedTrial = {
    assignedLaneId: session.assignedLaneId,
    absentLaneId: session.absentLaneId,
    visibilityOrderPosition: 1,
  };
  for (const [phase, lifecycle] of [
    ['before', evidence.before],
    ['after', evidence.after],
    ['result', evidence.lifecycle],
    ['page', value.strategyLifecycle],
  ]) {
    const observed = validateStandaloneLifecycle(
      lifecycle,
      { ...expectedTrial, visibilityOrderPosition: phase === 'before' ? 0 : 1 },
      `${label}.${phase}`,
    );
    requireSame(observed, baseIdentity, `${label}.${phase} static identity`);
  }
  const workload = workloadIdentity(value.workload);
  requireCondition(workload.visibilityFraction === session.visibilityOrder[1]
    && value.workload?.scenario?.expectedVisibleCount === evidence.expectedVisibleCount,
  `${label} workload differs from the second visibility.`);
  return { workload, environment: environmentIdentity(value.environment) };
}

function validateSessionEnd(value, session, baseIdentity, label) {
  exactKeys(value, [
    'selectedConfig', 'environment', 'strategyLifecycle', 'strategyDiagnostics',
    'cacheDiagnostics', 'workload', 'webgpuUncapturedErrors', 'webgpuDeviceLosses',
  ], label);
  requireCleanGpuRecord(value, label);
  requireCondition(value.selectedConfig?.strategyId
    === 'first-instance-live-standalone-deployment'
    && value.selectedConfig?.laneId === session.assignedLaneId
    && value.selectedConfig?.visibilityFraction === session.visibilityOrder[1]
    && isDeepStrictEqual(value.selectedConfig?.visibilityOrder, session.visibilityOrder),
  `${label} selected configuration is invalid.`);
  const observed = validateStandaloneLifecycle(value.strategyLifecycle, {
    assignedLaneId: session.assignedLaneId,
    absentLaneId: session.absentLaneId,
    visibilityOrderPosition: 1,
  }, `${label}.strategyLifecycle`);
  requireSame(observed, baseIdentity, `${label} static resource identity`);
  requireCondition(value.strategyLifecycle.shaderObservationSerial === 12,
    `${label} did not retain exactly twelve challenged shader observations.`);
  return {
    workload: workloadIdentity(value.workload),
    environment: environmentIdentity(value.environment),
  };
}

function validateSessionArtifacts(manifest, values, selection, lifecycles) {
  const records = manifest.artifacts.sessions;
  requireCondition(records.length === selection.sessions.length,
    'session artifact count differs from the selected plan.');
  return records.map((entry, index) => {
    const session = selection.sessions[index];
    const label = `session ${session.globalSessionIndex}`;
    const expectedLifecycle = lifecycles.find((candidate) => (
      candidate.sessionId === session.sessionId
    ));
    requireCondition(entry.sessionId === session.sessionId
      && entry.globalSessionIndex === session.globalSessionIndex
      && entry.matrixSessionIndex === session.matrixSessionIndex
      && entry.sessionNamespace === expectedLifecycle?.sessionNamespace
      && entry.browserInstanceSerial === expectedLifecycle?.browserInstanceSerial
      && entry.artifact.path
        === `matrices/matrix-${String(session.matrixOrdinal).padStart(2, '0')}`
          + `/session-${String(session.matrixSessionOrdinal).padStart(2, '0')}.json`,
    `${label} manifest record is invalid.`);
    const artifact = artifactValue(values, entry.artifact, `${label} artifact`);
    exactKeys(artifact, [
      'schemaVersion', 'kind', 'executionMode', 'analysisEligible', 'runId',
      'planSha256', 'capturedAt', 'canonicalSession', 'sessionNamespace',
      'browserInstanceSerial', 'browserLifecycle', 'sessionConstruction',
      'baseResourceIdentity', 'visibilitySwitch', 'endingEvidence',
      'computeProcessEvidence', 'trials', 'pageErrors',
    ], `${label} artifact`);
    requireCondition(artifact.schemaVersion === 1
      && artifact.kind === 'first-instance-standalone-deployment-session-completion'
      && artifact.executionMode === manifest.executionMode
      && artifact.analysisEligible === manifest.analysisEligible
      && artifact.runId === manifest.runId
      && artifact.planSha256 === manifest.planSha256
      && artifact.sessionNamespace === entry.sessionNamespace
      && artifact.browserInstanceSerial === entry.browserInstanceSerial,
    `${label} artifact identity is invalid.`);
    timestamp(artifact.capturedAt, `${label}.capturedAt`);
    requireSame(artifact.canonicalSession, session, `${label}.canonicalSession`);
    requireSame(artifact.browserLifecycle, expectedLifecycle, `${label}.browserLifecycle`);
    requireSame(artifact.pageErrors, [], `${label}.pageErrors`);
    const firstTrial = selection.trials.find((trial) => trial.sessionId === session.sessionId);
    const construction = validateConfigured(
      artifact.sessionConstruction,
      session,
      firstTrial,
      `${label}.sessionConstruction`,
    );
    requireSame(artifact.baseResourceIdentity, construction.identity,
      `${label}.baseResourceIdentity`);
    requireCondition(artifact.visibilitySwitch !== null,
      `${label} lacks its required two-visibility switch.`);
    const visibilitySwitch = validateVisibilitySwitch(
      artifact.visibilitySwitch,
      session,
      construction.identity,
      `${label}.visibilitySwitch`,
    );
    const ending = validateSessionEnd(
      artifact.endingEvidence,
      session,
      construction.identity,
      `${label}.endingEvidence`,
    );
    requireSame(visibilitySwitch.workload, ending.workload,
      `${label} switch/end workload`);
    requireSame(construction.environment, visibilitySwitch.environment,
      `${label} construction/switch environment`);
    requireSame(construction.environment, ending.environment,
      `${label} construction/end environment`);
    const expectedTrialRecords = manifest.artifacts.trials.filter(
      (trial) => trial.sessionId === session.sessionId,
    );
    requireCondition(expectedTrialRecords.length === FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIALS_PER_SESSION,
      `${label} must retain exactly two trial records.`);
    requireSame(artifact.trials, expectedTrialRecords, `${label}.trials`);
    if (manifest.analysisEligible) {
      record(artifact.computeProcessEvidence, `${label}.computeProcessEvidence`);
      record(artifact.computeProcessEvidence.snapshot,
        `${label}.computeProcessEvidence.snapshot`);
      record(artifact.computeProcessEvidence.comparison,
        `${label}.computeProcessEvidence.comparison`);
    } else {
      requireCondition(artifact.computeProcessEvidence === null,
        `${label} smoke unexpectedly contains telemetry process evidence.`);
    }
    return {
      session,
      record: entry,
      artifact,
      lifecycle: expectedLifecycle,
      resourceIdentity: construction.identity,
      environment: construction.environment,
      constructionWorkload: construction.workload,
    };
  });
}

function parseTelemetryCsv(text, label) {
  requireCondition(typeof text === 'string' && text.endsWith('\n'),
    `${label} must be newline-terminated UTF-8 CSV.`);
  const lines = text.slice(0, -1).split('\n');
  requireCondition(lines.length >= 1, `${label} lacks its telemetry header.`);
  const header = parseCsvRecord(lines[0].replace(/\r$/, ''));
  requireSame(header, [...TELEMETRY_CSV_FIELDS], `${label} header`);
  const numericFields = new Set([
    'runElapsedMs', 'planIndex', 'repetitionIndex', 'visibilityFraction', 'gpuIndex',
    'graphicsClockMHz', 'memoryClockMHz', 'gpuUtilizationPercent',
    'memoryUtilizationPercent', 'memoryUsedMiB', 'memoryTotalMiB',
    'temperatureC', 'powerDrawW',
  ]);
  const integerFields = new Set(['planIndex', 'repetitionIndex', 'gpuIndex']);
  return lines.slice(1).map((line, index) => {
    const fields = parseCsvRecord(line.replace(/\r$/, ''));
    requireCondition(Array.isArray(fields) && fields.length === TELEMETRY_CSV_FIELDS.length,
      `${label} row ${index} is malformed.`);
    return Object.fromEntries(TELEMETRY_CSV_FIELDS.map((field, fieldIndex) => {
      const raw = fields[fieldIndex];
      if (!numericFields.has(field)) return [field, raw === '' ? null : raw];
      if (raw === '') return [field, null];
      const parsed = Number(raw);
      requireCondition(Number.isFinite(parsed)
        && (!integerFields.has(field) || Number.isSafeInteger(parsed)),
      `${label} row ${index}.${field} is invalid.`);
      return [field, parsed];
    }));
  });
}

function recomputeTelemetryEligibility({
  report,
  preComputeProcesses,
  postComputeProcesses,
  processComparisons,
  adapterAssociation,
}) {
  const reasons = [];
  if (report?.status !== 'available') reasons.push('telemetry collector status is not available');
  if (report?.coverageAudit?.pass !== true) reasons.push('telemetry coverage audit failed');
  if (report?.sampling?.malformedLineCount !== 0) reasons.push('telemetry has malformed rows');
  if (report?.sampling?.stderrByteCount !== 0) reasons.push('telemetry collector wrote stderr');
  if (!Array.isArray(report?.summary?.gpus) || report.summary.gpus.length !== 1) {
    reasons.push('telemetry did not retain exactly one GPU identity');
  }
  if (preComputeProcesses?.status !== 'available') {
    reasons.push('pre-matrix compute-process snapshot is unavailable');
  }
  if (postComputeProcesses?.status !== 'available') {
    reasons.push('post-matrix compute-process snapshot is unavailable');
  }
  if (!Array.isArray(processComparisons)
    || processComparisons.length !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSIONS_PER_MATRIX) {
    reasons.push('matrix lacks one process-set comparison per session');
  } else if (processComparisons.some((entry) => entry?.comparison?.pass !== true)) {
    reasons.push('external GPU compute-process identity set changed during matrix');
  }
  if (adapterAssociation?.pass !== true) {
    reasons.push('page adapter and telemetry GPU association failed');
  }
  return {
    schemaVersion: 1,
    kind: 'first-instance-standalone-matrix-telemetry-eligibility',
    pass: reasons.length === 0,
    reasons,
  };
}

function validateFullMatrixTelemetry({
  manifest,
  values,
  matrix,
  entry,
  sessions,
}) {
  const label = `matrix ${matrix.matrixOrdinal} telemetry`;
  const directory = `matrices/matrix-${String(matrix.matrixOrdinal).padStart(2, '0')}`;
  requireCondition(entry.status === 'captured'
    && entry.artifact.path === `${directory}/telemetry.json`
    && entry.csvArtifact.path === `${directory}/gpu-telemetry.csv`,
  `${label} manifest paths/status are invalid.`);
  const artifact = artifactValue(values, entry.artifact, label);
  exactKeys(artifact, [
    'schemaVersion', 'kind', 'executionMode', 'analysisEligible', 'runId',
    'matrixId', 'matrixIndex', 'matrixOrdinal', 'planSha256', 'report',
    'preComputeProcesses', 'postComputeProcesses', 'processComparisons',
    'adapterAssociation', 'eligibility',
  ], `${label} artifact`);
  requireCondition(artifact.schemaVersion === 1
    && artifact.kind === 'first-instance-standalone-matrix-telemetry-evidence'
    && artifact.executionMode === FULL_EXECUTION_MODE
    && artifact.analysisEligible === true
    && artifact.runId === manifest.runId
    && artifact.matrixId === matrix.matrixId
    && artifact.matrixIndex === matrix.matrixIndex
    && artifact.matrixOrdinal === matrix.matrixOrdinal
    && artifact.planSha256 === manifest.planSha256,
  `${label} artifact identity is invalid.`);
  requireSame(entry.eligibility, artifact.eligibility,
    `${label} manifest/artifact eligibility`);
  const rows = parseTelemetryCsv(
    artifactValue(values, entry.csvArtifact, `${label} CSV`),
    `${label} CSV`,
  );
  requireCondition(rows.every((row) => row.runId === manifest.runId
    && row.modeId === 'first-instance-live-standalone-deployment'
    && row.layout === LAYOUT
    && row.runElapsedMs >= 0
    && Number.isFinite(Date.parse(row.observedAtIso))),
  `${label} CSV contains an invalid run/workload context.`);
  const report = artifact.report;
  requireCondition(report?.provider === 'nvidia-smi'
    && ['available', 'unavailable'].includes(report?.status)
    && report?.sampling?.processModel === 'one-long-lived-process'
    && report?.sampling?.requestedIntervalMs === 250
    && report?.sampling?.outputFile === 'gpu-telemetry.csv'
    && Number.isSafeInteger(report?.sampling?.malformedLineCount)
    && report.sampling.malformedLineCount >= 0
    && Number.isSafeInteger(report?.sampling?.stderrByteCount)
    && report.sampling.stderrByteCount >= 0
    && report?.summary?.sampleCount === rows.length,
  `${label} collector report is invalid.`);
  requireSame(report.summary, summarizeTelemetryRows(rows), `${label} recomputed summary`);
  const recomputedCoverage = createNvidiaTelemetryCoverageAudit(rows, {
    collectorStartedRunElapsedMs: report.sampling.collectorStartedRunElapsedMs,
    collectorStopRequestedRunElapsedMs: report.sampling.collectorStopRequestedRunElapsedMs,
    requestedIntervalMs: report.sampling.requestedIntervalMs,
  });
  requireSame(report.coverageAudit, recomputedCoverage, `${label} recomputed coverage audit`);
  requireSame(report.computeProcesses, {
    pre: artifact.preComputeProcesses,
    post: artifact.postComputeProcesses,
  }, `${label} report process snapshots`);
  record(artifact.preComputeProcesses, `${label}.preComputeProcesses`);
  record(artifact.postComputeProcesses, `${label}.postComputeProcesses`);
  requireCondition(Array.isArray(artifact.processComparisons)
    && artifact.processComparisons.length === FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSIONS_PER_MATRIX,
  `${label} lacks one process comparison per session.`);
  artifact.processComparisons.forEach((comparisonRecord, index) => {
    const session = sessions[index];
    exactKeys(comparisonRecord, ['sessionId', 'snapshot', 'comparison'],
      `${label}.processComparisons[${index}]`);
    requireCondition(comparisonRecord.sessionId === session.session.sessionId,
    `${label} process comparison ${index} identity is invalid.`);
    const recomputed = compareComputeProcessIdentitySets(
      artifact.preComputeProcesses,
      comparisonRecord.snapshot,
    );
    requireSame(comparisonRecord.comparison, recomputed,
      `${label} process comparison ${index}`);
    requireSame(session.artifact.computeProcessEvidence, {
      snapshot: comparisonRecord.snapshot,
      comparison: comparisonRecord.comparison,
    }, `${label} session ${session.session.sessionId} process evidence`);
  });
  requireSame(artifact.postComputeProcesses,
    artifact.processComparisons.at(-1).snapshot, `${label} final process snapshot`);
  const recomputedAssociation = evaluateLiveFirstInstanceAdapterTelemetryAssociation({
    adapterInfo: sessions[0].environment.adapterInfo,
    telemetryReport: report,
  });
  requireSame(artifact.adapterAssociation, recomputedAssociation,
    `${label} adapter/telemetry association`);
  const recomputedEligibility = recomputeTelemetryEligibility({
    report,
    preComputeProcesses: artifact.preComputeProcesses,
    postComputeProcesses: artifact.postComputeProcesses,
    processComparisons: artifact.processComparisons,
    adapterAssociation: artifact.adapterAssociation,
  });
  requireSame(artifact.eligibility, recomputedEligibility,
    `${label} recomputed eligibility`);
  return { artifact, rows, technicalGatePass: recomputedEligibility.pass };
}

function validateSmokeTelemetry(manifest, values, matrix, entry) {
  const directory = `matrices/matrix-${String(matrix.matrixOrdinal).padStart(2, '0')}`;
  requireCondition(entry.status === 'excluded'
    && entry.artifact.path === `${directory}/telemetry-exclusion.json`,
  'smoke telemetry exclusion path/status is invalid.');
  const artifact = artifactValue(values, entry.artifact, 'smoke telemetry exclusion');
  requireSame(artifact, {
    schemaVersion: 1,
    kind: 'first-instance-standalone-smoke-telemetry-exclusion',
    executionMode: SMOKE_EXECUTION_MODE,
    analysisEligible: false,
    runId: manifest.runId,
    matrixId: matrix.matrixId,
    planSha256: manifest.planSha256,
    status: 'excluded-by-frozen-smoke-policy',
    reason: 'Smoke validates plumbing only and cannot enter analysis or a decision.',
  }, 'smoke telemetry exclusion');
  return { artifact, rows: [], technicalGatePass: false };
}

function validateTelemetryArtifacts(manifest, values, selection, sessions) {
  requireCondition(manifest.artifacts.telemetry.length === selection.matrices.length,
    'telemetry artifact count differs from the selected matrices.');
  return selection.matrices.map((matrix, index) => {
    const matrixSessions = sessions.filter(
      (session) => session.session.matrixIndex === matrix.matrixIndex,
    );
    return manifest.analysisEligible
      ? validateFullMatrixTelemetry({
        manifest,
        values,
        matrix,
        entry: manifest.artifacts.telemetry[index],
        sessions: matrixSessions,
      })
      : validateSmokeTelemetry(
        manifest,
        values,
        matrix,
        manifest.artifacts.telemetry[index],
      );
  });
}

function validateTrialManifestRecord(entry, expectedTrial, session, index) {
  const label = `manifest.artifacts.trials[${index}]`;
  const expectedPath = `trials/trial-${String(expectedTrial.planIndex + 1).padStart(3, '0')}`
    + `-m${expectedTrial.matrixOrdinal}`
    + `-s${String(expectedTrial.matrixSessionIndex + 1).padStart(2, '0')}`
    + `-t${expectedTrial.visibilityOrderPosition + 1}`
    + `-${expectedTrial.assignedLaneId}`
    + `-v${Math.round(expectedTrial.visibilityFraction * 100)}.json.br`;
  requireCondition(entry.trialId === expectedTrial.trialId
    && entry.planIndex === expectedTrial.planIndex
    && entry.matrixIndex === expectedTrial.matrixIndex
    && entry.sessionId === expectedTrial.sessionId
    && entry.assignedLaneId === expectedTrial.assignedLaneId
    && entry.visibilityFraction === expectedTrial.visibilityFraction
    && entry.visibilityExposure === expectedTrial.visibilityExposure
    && entry.sessionNamespace === session.record.sessionNamespace
    && isDeepStrictEqual(entry.shaderCaptureOrdinals,
      Array.from({ length: 6 }, (_, offset) => (
        expectedTrial.visibilityOrderPosition * 6 + offset + 1
      )))
    && entry.artifact.path === expectedPath,
  `${label} differs from the frozen trial/path/session namespace.`);
}

async function validateTrialArtifacts({
  manifest,
  archive,
  selection,
  sessions,
  executionIdentity,
  usedNonces,
}) {
  requireCondition(manifest.artifacts.trials.length === selection.trials.length,
    'trial artifact count differs from the selected plan.');
  const initialNonceCount = usedNonces.size;
  const results = [];
  for (let index = 0; index < selection.trials.length; index += 1) {
    const expectedTrial = selection.trials[index];
    const entry = manifest.artifacts.trials[index];
    const session = sessions.find(
      (candidate) => candidate.session.sessionId === expectedTrial.sessionId,
    );
    validateTrialManifestRecord(entry, expectedTrial, session, index);
    const artifact = await archive.read(entry.artifact,
      `trial ${expectedTrial.planIndex} artifact`);
    const result = await validateTrialArtifact({
      artifact,
      expectedTrial,
      expectedSession: session.session,
      runId: manifest.runId,
      executionMode: manifest.executionMode,
      planSha256: manifest.planSha256,
      sessionNamespace: session.record.sessionNamespace,
      browserInstanceSerial: session.record.browserInstanceSerial,
      executionIdentity,
      usedNonces,
    });
    requireSame(artifact.configured, session.artifact.sessionConstruction,
      `trial ${expectedTrial.planIndex}/session construction`);
    requireSame(result.resourceIdentity, session.resourceIdentity,
      `trial ${expectedTrial.planIndex}/session resource identity`);
    results.push(result);
  }
  requireCondition(usedNonces.size === initialNonceCount + selection.trials.length * 6,
    'global shader challenge nonce count is invalid.');
  const resourceAudit = verifyStandaloneSessionResourceIdentityRecords(results.map(
    ({ sessionId, sessionNamespace, resourceIdentity, shaderResourceIdentity }) => ({
      sessionId,
      sessionNamespace,
      resourceIdentity,
      shaderResourceIdentity,
    }),
  ));
  requireCondition(resourceAudit.sessionCount === selection.sessions.length
    && resourceAudit.sessionNamespaceCount === selection.sessions.length,
  'trial records do not prove one unique namespace per selected session.');
  return { results, usedNonces, resourceAudit };
}

function expectedIdentityWorkloads(trialResults) {
  const byVisibility = new Map();
  for (const trial of trialResults) {
    const key = String(trial.visibilityFraction);
    byVisibility.set(key, trial.workload);
  }
  return Object.fromEntries([...byVisibility.entries()].sort(
    ([left], [right]) => Number(left) - Number(right),
  ));
}

function validateIdentityReport(identity, sessions, trials, label, expectedObservationCount) {
  exactKeys(identity, [
    'baselineEnvironment', 'environmentObservationCount', 'environmentIdentityExact',
    'environmentDrift', 'geometrySha256', 'workloadByVisibility',
  ], label);
  requireCondition(identity.environmentObservationCount === expectedObservationCount
    && identity.environmentIdentityExact === (identity.environmentDrift.length === 0),
  `${label} environment gate/report is internally inconsistent.`);
  for (const [index, drift] of identity.environmentDrift.entries()) {
    requireCondition(typeof drift?.label === 'string'
      && drift.label.length > 0
      && isDeepStrictEqual(drift?.expected, identity.baselineEnvironment)
      && !isDeepStrictEqual(drift?.observed, identity.baselineEnvironment),
    `${label}.environmentDrift[${index}] is not a genuine baseline mismatch.`);
  }
  if (identity.environmentIdentityExact) {
    for (const session of sessions) {
      requireSame(session.environment, identity.baselineEnvironment,
        `${label} session ${session.session.sessionId} environment`);
    }
  }
  requireCondition(SHA256_PATTERN.test(identity.geometrySha256)
    && trials.every((trial) => trial.workload.geometrySha256 === identity.geometrySha256),
  `${label} geometry identity drifted.`);
  requireSame(identity.workloadByVisibility, expectedIdentityWorkloads(trials),
    `${label}.workloadByVisibility`);
  return { ...identity, technicalGatePass: identity.environmentIdentityExact };
}

function validateMatrixArtifacts({
  manifest,
  values,
  selection,
  sessions,
  trialResults,
  gateResults,
  telemetryResults,
}) {
  requireCondition(manifest.artifacts.matrices.length === selection.matrices.length,
    'matrix artifact count differs from the selected plan.');
  return selection.matrices.map((matrix, index) => {
    const entry = manifest.artifacts.matrices[index];
    const label = `matrix ${matrix.matrixOrdinal}`;
    const matrixSessions = sessions.filter(
      (session) => session.session.matrixIndex === matrix.matrixIndex,
    );
    const matrixTrials = trialResults.filter(
      (trial) => trial.matrixIndex === matrix.matrixIndex,
    );
    const matrixTrialRecords = manifest.artifacts.trials.filter(
      (trial) => trial.matrixIndex === matrix.matrixIndex,
    );
    const matrixSessionRecords = manifest.artifacts.sessions.filter(
      (session) => selection.sessions[session.globalSessionIndex]?.matrixIndex
        === matrix.matrixIndex,
    );
    const telemetryRecord = manifest.artifacts.telemetry[index];
    requireCondition(entry.matrixId === matrix.matrixId
      && entry.matrixIndex === matrix.matrixIndex
      && entry.matrixOrdinal === matrix.matrixOrdinal
      && entry.sessionCount === matrixSessions.length
      && entry.trialCount === matrixTrials.length
      && isDeepStrictEqual(entry.telemetry, telemetryRecord)
      && entry.artifact.path
        === `matrices/matrix-${String(matrix.matrixOrdinal).padStart(2, '0')}`
          + '/matrix-completion.json',
    `${label} manifest record is invalid.`);
    const identityResult = validateIdentityReport(entry.identity, matrixSessions, matrixTrials,
      `${label} manifest identity`, 1 + matrixSessions.length * 9);
    requireSame(gateResults[index].environment, entry.identity.baselineEnvironment,
      `${label} gate/environment identity`);
    requireCondition(gateResults[index].workload.geometrySha256
      === entry.identity.geometrySha256
      && isDeepStrictEqual(
        gateResults[index].workload,
        entry.identity.workloadByVisibility['0.99'],
      ),
    `${label} gate workload differs from the measured high-visibility cell.`);
    const artifact = artifactValue(values, entry.artifact, `${label} completion`);
    exactKeys(artifact, [
      'schemaVersion', 'kind', 'executionMode', 'analysisEligible', 'runId',
      'planSha256', 'canonicalMatrix', 'capturedAt', 'forcedFeatureOffGate',
      'sessions', 'trials', 'telemetry', 'identity', 'numericalAnalysisInvoked',
      'numericalDecision',
    ], `${label} completion`);
    requireCondition(artifact.schemaVersion === 1
      && artifact.kind === 'first-instance-standalone-deployment-matrix-completion'
      && artifact.executionMode === manifest.executionMode
      && artifact.analysisEligible === manifest.analysisEligible
      && artifact.runId === manifest.runId
      && artifact.planSha256 === manifest.planSha256
      && artifact.numericalAnalysisInvoked === false
      && artifact.numericalDecision === null,
    `${label} completion identity/analysis boundary is invalid.`);
    timestamp(artifact.capturedAt, `${label}.capturedAt`);
    requireSame(artifact.canonicalMatrix, matrix, `${label}.canonicalMatrix`);
    requireSame(artifact.forcedFeatureOffGate,
      manifest.artifacts.forcedFeatureOffGates[index], `${label}.forcedFeatureOffGate`);
    requireSame(artifact.sessions, matrixSessionRecords, `${label}.sessions`);
    requireSame(artifact.trials, matrixTrialRecords, `${label}.trials`);
    requireSame(artifact.telemetry, telemetryRecord, `${label}.telemetry`);
    requireSame(artifact.identity, entry.identity, `${label}.identity`);
    requireSame(gateResults[index].matrixId, matrix.matrixId, `${label} gate binding`);
    requireCondition(telemetryResults[index] !== undefined,
      `${label} telemetry validation is missing.`);
    return {
      entry,
      artifact,
      technicalGatePass: identityResult.technicalGatePass
        && telemetryResults[index].technicalGatePass,
    };
  });
}

function expectedJournalEvents(manifest, selection) {
  const events = [
    {
      eventKind: 'plan-committed',
      payload: { planSha256: manifest.planSha256, artifact: manifest.artifacts.plan },
    },
    {
      eventKind: 'execution-identity-start-committed',
      payload: { artifact: manifest.artifacts.executionIdentityStart },
    },
  ];
  for (let matrixIndex = 0; matrixIndex < selection.matrices.length; matrixIndex += 1) {
    const matrix = selection.matrices[matrixIndex];
    const gate = manifest.artifacts.forcedFeatureOffGates[matrixIndex];
    events.push({
      eventKind: 'forced-feature-off-gate-committed',
      payload: { matrixId: matrix.matrixId, artifact: gate.artifact },
    });
    const sessions = manifest.artifacts.sessions.filter(
      (session) => selection.sessions[session.globalSessionIndex]?.matrixIndex
        === matrix.matrixIndex,
    );
    for (const session of sessions) {
      for (const trial of manifest.artifacts.trials.filter(
        (candidate) => candidate.sessionId === session.sessionId,
      )) {
        events.push({
          eventKind: 'trial-committed',
          payload: {
            trialId: trial.trialId,
            planIndex: trial.planIndex,
            sessionId: trial.sessionId,
            artifact: trial.artifact,
          },
        });
      }
      events.push({ eventKind: 'session-committed', payload: session });
    }
    const telemetry = manifest.artifacts.telemetry[matrixIndex];
    if (telemetry.status === 'captured') {
      events.push({
        eventKind: 'matrix-telemetry-committed',
        payload: {
          matrixId: matrix.matrixId,
          artifact: telemetry.artifact,
          csvArtifact: telemetry.csvArtifact,
          eligibility: telemetry.eligibility,
        },
      });
    }
    events.push({
      eventKind: 'matrix-committed',
      payload: manifest.artifacts.matrices[matrixIndex],
    });
  }
  events.push({
    eventKind: 'execution-identity-end-committed',
    payload: {
      artifact: manifest.artifacts.executionIdentityEnd,
      sourceMatched: true,
      dependencyMatched: true,
    },
  });
  events.push({
    eventKind: 'vite-runtime-audit-committed',
    payload: {
      artifact: manifest.artifacts.viteRuntimeAudit,
      policyId: manifest.identity.viteRuntimePolicyId,
      modulesSha256: manifest.identity.viteRuntimeModulesSha256,
    },
  });
  return events;
}

function validateJournal(manifest, values, selection) {
  const expected = expectedJournalEvents(manifest, selection);
  requireCondition(manifest.artifacts.journal.length === expected.length,
    `journal event count differs from ${expected.length}.`);
  manifest.artifacts.journal.forEach((descriptor, index) => {
    const serial = index + 1;
    const event = expected[index];
    const expectedPath = `journal/${String(serial).padStart(4, '0')}`
      + `-${event.eventKind}.json`;
    requireCondition(descriptor.path === expectedPath,
      `journal descriptor ${index} path is invalid.`);
    const artifact = artifactValue(values, descriptor, `journal event ${serial}`);
    requireSame(artifact, {
      schemaVersion: 1,
      kind: 'first-instance-standalone-incremental-artifact-journal-event',
      journalSerial: serial,
      eventKind: event.eventKind,
      committedAt: artifact.committedAt,
      payload: event.payload,
    }, `journal event ${serial}`);
    timestamp(artifact.committedAt, `journal event ${serial}.committedAt`);
  });
  return { eventCount: expected.length };
}

async function validateViteRuntime(manifest, values, projectRoot, mode) {
  requireCondition(manifest.artifacts.viteRuntimeAudit.path === 'vite-runtime-audit.json',
    'Vite runtime audit descriptor path is invalid.');
  const audit = artifactValue(values, manifest.artifacts.viteRuntimeAudit,
    'Vite runtime audit');
  requireCondition(audit?.policyId === manifest.identity.viteRuntimePolicyId
    && audit?.modulesSha256 === manifest.identity.viteRuntimeModulesSha256
    && audit?.entryHtml?.successfulResponseCount === mode.expected.entries
    && audit?.modules?.some((entry) => entry.sourceRelativePath
      === 'src/strategies/live-first-instance-standalone.js'),
  'Vite runtime audit is not bound to the manifest/standalone strategy.');
  const validation = await validateCandidateViteRuntimeAudit(audit, path.resolve(projectRoot));
  requireCondition(validation.pass === true,
    `Vite runtime audit failed: ${validation.reasons.join('; ')}`);
  return validation;
}

function validateGlobalIdentity(manifest, sessions, trials, gateResults) {
  const identity = validateIdentityReport(
    manifest.identity.globalEnvironmentAndWorkload,
    sessions,
    trials,
    'manifest global environment/workload identity',
    gateResults.length + sessions.length * 9,
  );
  for (const gate of gateResults) {
    if (identity.environmentIdentityExact) {
      requireSame(gate.environment, identity.baselineEnvironment,
        `gate ${gate.matrixId} global environment identity`);
    }
    requireCondition(gate.workload.geometrySha256 === identity.geometrySha256
      && isDeepStrictEqual(gate.workload, identity.workloadByVisibility['0.99']),
    `gate ${gate.matrixId} global workload identity differs.`);
  }
  return identity;
}

function environmentIdentity(environment) {
  return {
    threeRevision: environment?.threeRevision ?? null,
    userAgent: environment?.userAgent ?? null,
    adapterInfo: environment?.adapterInfo ?? null,
    rendererBackend: environment?.rendererBackend ?? null,
    coordinateSystem: environment?.coordinateSystem ?? null,
    reversedDepth: environment?.reversedDepth ?? null,
    rendererReversedDepthBuffer: environment?.rendererReversedDepthBuffer ?? null,
    maxStorageBuffersPerShaderStage: environment?.maxStorageBuffersPerShaderStage ?? null,
    timestampAvailable: environment?.timestampAvailable ?? null,
    indirectFirstInstanceAvailable: environment?.indirectFirstInstanceAvailable ?? null,
    crossOriginIsolated: environment?.crossOriginIsolated ?? null,
    viewport: environment?.viewport ?? null,
  };
}

function workloadIdentity(workload) {
  return {
    scenarioSeed: workload?.scenarioSeed ?? null,
    geometrySha256: workload?.geometryFixtures?.sha256 ?? null,
    scenarioSha256: workload?.scenario?.sha256 ?? null,
    expectedVisibleIdsCanonicalSha256:
      workload?.scenario?.expectedVisibleIdsCanonicalSha256 ?? null,
    objectCount: workload?.scenario?.objectCount ?? null,
    bucketCount: workload?.scenario?.bucketCount ?? null,
    visibilityFraction: workload?.scenario?.visibilityFraction ?? null,
    layout: workload?.scenario?.layout ?? null,
  };
}

function requireCleanEnvironment(environment, label) {
  requireCondition(environment?.indirectFirstInstanceAvailable === true,
    `${label} lacks indirect-first-instance.`);
  requireCondition(environment?.timestampAvailable === true,
    `${label} lacks timestamp queries.`);
  requireCondition(environment?.reversedDepth === true
    && environment?.rendererReversedDepthBuffer === true,
  `${label} lacks pinned reversed depth.`);
  requireCondition(environment?.crossOriginIsolated === true,
    `${label} lost cross-origin isolation.`);
  finiteNumber(environment?.performanceNowQuantumMs, `${label}.performanceNowQuantumMs`, {
    minimum: 0,
    exclusive: true,
  });
  requireCondition(environment.performanceNowQuantumMs <= MAXIMUM_CPU_TIMER_QUANTUM_MS,
    `${label} has a coarse CPU timer quantum.`);
  requireCondition(environment?.webgpuUncapturedErrorCount === 0
    && environment?.webgpuValidationErrorCount === 0
    && environment?.webgpuDeviceLossCount === 0,
  `${label} has a WebGPU error or device loss.`);
}

function requireCleanGpuRecord(value, label) {
  requireCondition(Array.isArray(value?.webgpuUncapturedErrors)
    && value.webgpuUncapturedErrors.length === 0,
  `${label} has uncaptured WebGPU errors.`);
  requireCondition(Array.isArray(value?.webgpuDeviceLosses)
    && value.webgpuDeviceLosses.length === 0,
  `${label} has WebGPU device losses.`);
  requireCleanEnvironment(value?.environment, `${label}.environment`);
}

function staticResourceIdentity(lifecycle) {
  return {
    selectedLane: lifecycle?.selectedLane ?? null,
    absentLane: lifecycle?.absentLane ?? null,
    rootUuid: lifecycle?.rootUuid ?? null,
    meshUuid: lifecycle?.meshUuid ?? null,
    geometryUuid: lifecycle?.geometryUuid ?? null,
    materialUuid: lifecycle?.materialUuid ?? null,
    matrixAttributeId: lifecycle?.matrixAttributeId ?? null,
    boundsAttributeId: lifecycle?.boundsAttributeId ?? null,
    visibleIdsAttributeId: lifecycle?.visibleIdsAttributeId ?? null,
    indirectAttributeId: lifecycle?.indirectAttributeId ?? null,
    computeNodeIds: lifecycle?.computeNodeIds ?? null,
    commandBufferAttributeId: lifecycle?.commandBuffer?.attributeId ?? null,
    commandBufferByteOffset: lifecycle?.commandBuffer?.byteOffset ?? null,
    commandBufferFirstOffset: lifecycle?.commandBuffer?.firstOffset ?? null,
    timestampContextId: lifecycle?.timestampContextId ?? null,
    timestampRegistrationSerial:
      lifecycle?.timestampRegistration?.registrationSerial ?? null,
    timestampBackendIdentity:
      lifecycle?.timestampRegistration?.backendIdentity ?? null,
    timestampBackendWrapperIdentity:
      lifecycle?.timestampRegistration?.backendWrapperIdentity ?? null,
    timestampComputeGroupIdentity:
      lifecycle?.timestampRegistration?.computeGroupIdentity ?? null,
    productionResourceLedger: lifecycle?.productionResourceLedger ?? null,
  };
}

function canonicalUint32Sha256(values) {
  const bytes = Buffer.allocUnsafe(values.length * Uint32Array.BYTES_PER_ELEMENT);
  for (let index = 0; index < values.length; index += 1) {
    bytes.writeUInt32LE(values[index], index * Uint32Array.BYTES_PER_ELEMENT);
  }
  return sha256Bytes(bytes);
}

export function verifyStandaloneAddressDigestCommitment({
  visibilityFraction,
  commands,
  survivors,
  address,
}) {
  requireCondition(Array.isArray(commands) && commands.length === BUCKET_COUNT * 5,
    'address digest commands are incomplete.');
  requireCondition(Array.isArray(survivors) && survivors.length === OBJECT_COUNT,
    'address digest survivors are incomplete.');
  const expected = expectedStandaloneScenario(visibilityFraction);
  const activeCounts = Uint32Array.from(
    Array.from({ length: BUCKET_COUNT }, (_, bucket) => commands[bucket * 5 + 1]),
  );
  const targetShape = createFirstInstanceAddressChallengeTargetShape(OBJECT_COUNT);
  const rebuilt = createFirstInstanceAddressChallengeExpected({
    scenario: {
      objectCount: OBJECT_COUNT,
      bucketCount: BUCKET_COUNT,
      bucketBases: Uint32Array.from(expected.bases),
      bucketCounts: Uint32Array.from(expected.capacities),
    },
    visibleIds: Uint32Array.from(survivors),
    activeCounts,
    targetShape,
  });
  const identity = {
    expectedSha256: sha256Bytes(rebuilt.bytes),
    survivorSha256: canonicalUint32Sha256(survivors),
    activeCountsSha256: canonicalUint32Sha256(activeCounts),
    activeAddressCount: rebuilt.activeAddressCount,
    paddingAddressCount: rebuilt.paddingAddressCount,
    targetPaddingPixelCount: rebuilt.targetPaddingPixelCount,
    coverage: rebuilt.coverage,
  };
  for (const [key, value] of Object.entries(identity)) {
    requireSame(address?.[key], value, `address digest commitment.${key}`);
  }
  return identity;
}

const EXPECTED_SCENARIOS = new Map();

function expectedStandaloneScenario(visibilityFraction) {
  requireCondition([0.99, 0.2].includes(visibilityFraction),
    `unsupported visibility fraction ${visibilityFraction}.`);
  if (EXPECTED_SCENARIOS.has(visibilityFraction)) {
    return EXPECTED_SCENARIOS.get(visibilityFraction);
  }
  const capacities = allocateBalancedCounts(OBJECT_COUNT, BUCKET_COUNT);
  const bases = prefixBases(capacities);
  const visibleCounts = allocateVisibleCounts(capacities, visibilityFraction);
  const expectedVisibleIds = [];
  for (let bucket = 0; bucket < BUCKET_COUNT; bucket += 1) {
    for (let local = 0; local < visibleCounts[bucket]; local += 1) {
      expectedVisibleIds.push(bases[bucket] + local);
    }
  }
  const result = {
    capacities: Array.from(capacities),
    bases: Array.from(bases),
    visibleCounts: Array.from(visibleCounts),
    expectedVisibleIds,
    expectedVisibleIdsCanonicalSha256: canonicalUint32Sha256(expectedVisibleIds),
  };
  EXPECTED_SCENARIOS.set(visibilityFraction, result);
  return result;
}

function requireReadback(value, expected, label) {
  record(value, label);
  exactKeys(value, ['format', 'arrayType', 'byteLength', 'sha256'], label);
  requireCondition(value.format === expected.format
    && value.arrayType === expected.arrayType
    && value.byteLength === expected.byteLength
    && SHA256_PATTERN.test(value.sha256),
  `${label} is not the frozen readback commitment.`);
  return {
    format: value.format,
    arrayType: value.arrayType,
    byteLength: value.byteLength,
    sha256: value.sha256,
  };
}

function renderCommitmentStaticIdentity(commitment) {
  return {
    rootUuid: commitment.bundleUuid,
    meshUuid: commitment.meshUuid,
    geometryUuid: commitment.geometryUuid,
    materialUuid: commitment.materialUuid,
    matrixAttributeId: commitment.matrixAttributeId,
    boundsAttributeId: commitment.boundsAttributeId,
    visibleIdsAttributeId: commitment.visibleIdsAttributeId,
    indirectAttributeId: commitment.indirectAttributeId,
    commandBufferAttributeId: commitment.commandBuffer.attributeId,
    commandBufferByteOffset: commitment.commandBuffer.byteOffset,
    commandBufferFirstOffset: commitment.commandBuffer.firstOffset,
  };
}

function validatePrimedResourceIdentities(value, label) {
  record(value, label);
  exactKeys(value, ['render', 'compute'], label);
  exactKeys(value.render, [
    'renderObjectIdentity', 'nodeBuilderStateIdentity', 'bindingArrayIdentity',
    'pipelineIdentity',
  ], `${label}.render`);
  requireCondition(Object.values(value.render).every(Number.isSafeInteger),
    `${label}.render contains an invalid identity.`);
  exactKeys(value.compute, ['reset', 'cull'], `${label}.compute`);
  for (const phase of ['reset', 'cull']) {
    const identity = value.compute[phase];
    exactKeys(identity, [
      'computeNodeId', 'computeNodeVersion', 'nodeBuilderStateIdentity',
      'bindingArrayIdentity', 'pipelineIdentity',
    ], `${label}.compute.${phase}`);
    requireCondition(Number.isFinite(identity.computeNodeId)
      && Number.isFinite(identity.computeNodeVersion)
      && Number.isSafeInteger(identity.nodeBuilderStateIdentity)
      && Number.isSafeInteger(identity.bindingArrayIdentity)
      && Number.isSafeInteger(identity.pipelineIdentity),
    `${label}.compute.${phase} contains an invalid identity.`);
  }
  return value;
}

function validateRenderCommitment(commitment, laneId, label, lifecycleIdentity = null) {
  record(commitment, label);
  exactKeys(commitment, [
    'schemaVersion', 'kind', 'laneId', 'bundleUuid', 'bundleVersion', 'meshUuid',
    'geometryUuid', 'materialUuid', 'materialVersion', 'bundleRecordCallbackCount',
    'renderBundleStable', 'renderBundleDataStable', 'bundleGpuStable',
    'renderObjectStable', 'nodeBuilderStateStable', 'resourceIdentities',
    'matrixAttributeId', 'matrixAttributeVersion', 'boundsAttributeId',
    'boundsAttributeVersion', 'visibleIdsAttributeId', 'visibleIdsAttributeVersion',
    'indirectAttributeId', 'indirectAttributeVersion', 'commandBuffer', 'cache',
  ], label);
  requireCondition(commitment.schemaVersion === 1
    && commitment.kind === 'first-instance-live-standalone-render-commitment'
    && commitment.laneId === laneId
    && commitment.bundleRecordCallbackCount === 1
    && commitment.renderBundleStable === true
    && commitment.renderBundleDataStable === true
    && commitment.bundleGpuStable === true
    && commitment.renderObjectStable === true
    && commitment.nodeBuilderStateStable === true,
  `${label} does not prove a stable production render bundle.`);
  for (const key of ['bundleUuid', 'meshUuid', 'geometryUuid', 'materialUuid']) {
    nonemptyString(commitment[key], `${label}.${key}`);
  }
  for (const key of [
    'bundleVersion', 'materialVersion', 'matrixAttributeId', 'matrixAttributeVersion',
    'boundsAttributeId', 'boundsAttributeVersion', 'visibleIdsAttributeId',
    'visibleIdsAttributeVersion', 'indirectAttributeId', 'indirectAttributeVersion',
  ]) {
    requireCondition(Number.isSafeInteger(commitment[key]) && commitment[key] >= 0,
      `${label}.${key} is invalid.`);
  }
  const command = commitment.commandBuffer;
  requireCondition(command?.lane === laneId
    && Number.isSafeInteger(command?.attributeId)
    && command?.byteOffset === 0
    && command?.byteLength === BUCKET_COUNT * 5 * Uint32Array.BYTES_PER_ELEMENT
    && command?.recordCount === BUCKET_COUNT
    && command?.drawCommandCount === BUCKET_COUNT
    && command?.firstOffset === 0
    && isDeepStrictEqual(command?.allOffsets,
      Array.from({ length: BUCKET_COUNT }, (_, bucket) => bucket * 20)),
  `${label}.commandBuffer differs from the frozen production buffer.`);
  record(commitment.cache, `${label}.cache`);
  const resourceIdentities = validatePrimedResourceIdentities(
    commitment.resourceIdentities,
    `${label}.resourceIdentities`,
  );
  const staticIdentity = renderCommitmentStaticIdentity(commitment);
  if (lifecycleIdentity !== null) {
    for (const [key, value] of Object.entries(staticIdentity)) {
      requireSame(value, lifecycleIdentity[key], `${label}/${key} lifecycle identity`);
    }
  }
  return { staticIdentity, resourceIdentities };
}

function renderOutputIdentity(renderParity, label) {
  requireCondition(renderParity?.schemaVersion === 1
    && renderParity?.kind === 'first-instance-live-standalone-exact-render-parity'
    && renderParity?.pass === true
    && renderParity?.width === 1280
    && renderParity?.height === 720
    && renderParity?.captures === 2
    && renderParity?.reversedDepthBuffer === true
    && renderParity?.stability?.pass === true,
  `${label} exact render parity failed.`);
  const color = requireReadback(renderParity.color, {
    format: 'rgba8unorm',
    arrayType: 'Uint8Array',
    byteLength: 1280 * 720 * 4,
  }, `${label}.color`);
  const depth = requireReadback(renderParity.depth, {
    format: 'depth32float',
    arrayType: 'Float32Array',
    byteLength: 1280 * 720 * 4,
  }, `${label}.depth`);
  const objectId = requireReadback(renderParity.objectId, {
    format: 'rgba8unorm-object-id-plus-one',
    arrayType: 'Uint8Array',
    byteLength: 1280 * 720 * 4,
  }, `${label}.objectId`);
  for (const [channel, identity] of Object.entries({ color, depth, objectId })) {
    const first = renderParity.stability?.firstCapture?.[channel];
    requireSame(first, identity, `${label} first/second ${channel} readback`);
  }
  requireCondition(renderParity.objectIdValidation?.pass === true
    && renderParity.objectIdValidation?.coveredPixels > 0
    && renderParity.objectIdValidation?.outOfRangePixels === 0
    && renderParity.objectIdValidation?.nonVisiblePixels === 0,
  `${label} object-ID coverage is invalid.`);
  const production = renderParity.productionBundleOutput;
  requireCondition(production?.schemaVersion === 1
    && production?.kind === 'first-instance-live-standalone-production-bundle-output'
    && production?.pass === true
    && production?.captures === 2
    && production?.resourcesStable === true
    && production?.executionExact === true
    && production?.directDiagnosticExact === true
    && production?.stability?.pass === true,
  `${label} production bundle output failed.`);
  requireSame(production.color, color, `${label} production/color commitment`);
  requireSame(production.directDiagnosticColor, color,
    `${label} production/direct diagnostic commitment`);
  requireSame(production.stability.firstCapture, color,
    `${label} production first capture commitment`);
  requireSame(production.stability.secondCapture, color,
    `${label} production second capture commitment`);
  requireSame(production.commitmentBetween, production.commitmentBefore,
    `${label} production commitment before/between`);
  requireSame(production.commitmentAfter, production.commitmentBefore,
    `${label} production commitment before/after`);
  const commitment = validateRenderCommitment(
    production.commitmentBefore,
    renderParity.laneId,
    `${label}.production.commitmentBefore`,
  );
  requireCondition(production.executionBetween?.rendererRenderCallSerial
    === production.executionBefore?.rendererRenderCallSerial + 1
    && production.executionAfter?.rendererRenderCallSerial
      === production.executionBetween?.rendererRenderCallSerial + 1
    && ['strategyComputeCallSerial', 'rendererComputeCallSerial'].every(
      (field) => production.executionBefore?.[field]
        === production.executionBetween?.[field]
        && production.executionBefore?.[field] === production.executionAfter?.[field],
    ),
  `${label} production capture execution counters are invalid.`);
  return {
    color,
    depth,
    objectId,
    productionResourceIdentities: commitment.resourceIdentities,
    productionStaticIdentity: commitment.staticIdentity,
  };
}

function outputSemanticIdentity(output) {
  return {
    color: output.color,
    depth: output.depth,
    objectId: output.objectId,
  };
}

function validateProductionLedger(ledger, expectedTrial, label) {
  const lane = expectedTrial.assignedLaneId;
  requireCondition(ledger?.schemaVersion === 1
    && ledger?.kind === 'first-instance-live-standalone-production-resource-ledger'
    && ledger?.selectedLane === lane
    && ledger?.absentLane === expectedTrial.absentLaneId
    && ledger?.absentLaneConstructed === false
    && isDeepStrictEqual(ledger?.constructedLaneIds, [lane])
    && isDeepStrictEqual(ledger?.constructedAddressModes, [lane === 'portable'
      ? 'bucket-base'
      : 'indirect-first-instance'])
    && ledger?.constructedLaneCount === 1
    && ledger?.indirectCommandBufferCount === 1
    && ledger?.computeNodeCount === 2
    && ledger?.materialCount === 1
    && ledger?.meshCount === 1
    && ledger?.bundleCount === 1
    && ledger?.addressMode === (lane === 'portable'
      ? 'bucket-base'
      : 'indirect-first-instance')
    && ledger?.hasBucketBaseVertexAttribute === (lane === 'portable'),
  `${label} allocation ledger is not exactly one selected lane and zero absent lane.`);
}

function validateStandaloneLifecycle(lifecycle, expectedTrial, label) {
  requireCondition(lifecycle?.schemaVersion === 1
    && lifecycle?.kind === 'first-instance-live-standalone-static-resource-lifecycle'
    && lifecycle?.pass === true
    && lifecycle?.selectedLane === expectedTrial.assignedLaneId
    && lifecycle?.absentLane === expectedTrial.absentLaneId
    && lifecycle?.absentLaneConstructed === false
    && lifecycle?.lanesConstructed === 1
    && lifecycle?.primed === true
    && lifecycle?.scenarioSwitchSerial === expectedTrial.visibilityOrderPosition
    && lifecycle?.bundleRecordCallbackCount === 1,
  `${label} violates the selected-only static lifecycle.`);
  validateProductionLedger(lifecycle.productionResourceLedger, expectedTrial, label);
  const command = lifecycle.commandBuffer;
  requireCondition(command?.lane === expectedTrial.assignedLaneId
    && Number.isSafeInteger(command?.attributeId)
    && command?.byteOffset === 0
    && command?.byteLength === BUCKET_COUNT * 5 * Uint32Array.BYTES_PER_ELEMENT
    && command?.recordCount === BUCKET_COUNT
    && command?.drawCommandCount === BUCKET_COUNT
    && command?.firstOffset === 0
    && isDeepStrictEqual(command?.allOffsets,
      Array.from({ length: BUCKET_COUNT }, (_, bucket) => bucket * 20)),
  `${label} command-buffer commitment is invalid.`);
  requireCondition(Array.isArray(lifecycle.computeNodeIds)
    && lifecycle.computeNodeIds.length === 2
    && lifecycle.computeNodeIds.every((value) => Number.isFinite(value)),
  `${label} compute-node identity is invalid.`);
  requireCondition(Number.isSafeInteger(lifecycle.timestampContextId)
    && lifecycle.timestampContextId > 0
    && Number.isSafeInteger(lifecycle.timestampRegistration?.registrationSerial)
    && lifecycle.timestampRegistration.registrationSerial > 0,
  `${label} timestamp registration is invalid.`);
  return staticResourceIdentity(lifecycle);
}

function validateCommandAndMembership(validation, expectedTrial, label) {
  requireCondition(validation?.schemaVersion === 1
    && validation?.kind === 'first-instance-live-standalone-validation'
    && validation?.pass === true
    && validation?.laneId === expectedTrial.assignedLaneId
    && validation?.visibilityFraction === expectedTrial.visibilityFraction,
  `${label} selected-lane validation failed.`);
  const expected = expectedStandaloneScenario(expectedTrial.visibilityFraction);
  const correctness = validation.correctness;
  requireCondition(correctness?.pass === true
    && correctness?.lane === expectedTrial.assignedLaneId
    && correctness?.overflow === 0
    && correctness?.membership?.pass === true
    && correctness?.membership?.errors === 0
    && correctness?.membership?.expectedCount === expected.expectedVisibleIds.length
    && correctness?.membership?.listedCount === expected.expectedVisibleIds.length
    && correctness?.membershipDigests?.pass === true
    && correctness?.membershipDigests?.expected?.sha256
      === expected.expectedVisibleIdsCanonicalSha256
    && correctness?.membershipDigests?.actual?.sha256
      === expected.expectedVisibleIdsCanonicalSha256
    && correctness?.commandValidation?.pass === true
    && correctness?.commandValidation?.errors?.length === 0
    && correctness?.commandValidation?.commandCount === BUCKET_COUNT
    && correctness?.commandValidation?.totalInstanceCount
      === expected.expectedVisibleIds.length
    && SHA256_PATTERN.test(correctness?.commandSha256)
    && SHA256_PATTERN.test(correctness?.survivorSha256),
  `${label} command/membership evidence failed.`);
  const commands = validation.snapshot?.commands;
  const survivors = validation.snapshot?.visibleIds;
  requireCondition(Array.isArray(commands) && commands.length === BUCKET_COUNT * 5
    && Array.isArray(survivors) && survivors.length === OBJECT_COUNT
    && validation.snapshot?.overflow === 0,
  `${label} raw production snapshot is incomplete.`);
  requireCondition(canonicalUint32Sha256(commands) === correctness.commandSha256
    && canonicalUint32Sha256(survivors) === correctness.survivorSha256,
  `${label} raw production snapshot differs from its digest.`);
  const records = correctness.commandValidation.records;
  requireCondition(Array.isArray(records) && records.length === BUCKET_COUNT,
    `${label} command validation records are incomplete.`);
  for (let bucket = 0; bucket < BUCKET_COUNT; bucket += 1) {
    const base = bucket * 5;
    const recordValue = records[bucket];
    const expectedFirstInstance = expectedTrial.assignedLaneId === 'feature'
      ? expected.bases[bucket]
      : 0;
    requireCondition(recordValue?.bucket === bucket
      && recordValue?.actual?.indexCount === commands[base]
      && recordValue?.actual?.instanceCount === commands[base + 1]
      && recordValue?.actual?.firstIndex === commands[base + 2]
      && recordValue?.actual?.baseVertex === (commands[base + 3] | 0)
      && recordValue?.actual?.firstInstance === commands[base + 4]
      && recordValue?.expected?.instanceCount === expected.visibleCounts[bucket]
      && recordValue?.expected?.baseVertex === 0
      && recordValue?.expected?.firstInstance === expectedFirstInstance
      && commands[base + 1] === expected.visibleCounts[bucket]
      && commands[base + 3] === 0
      && commands[base + 4] === expectedFirstInstance,
    `${label} bucket ${bucket} command fields differ from the frozen address mode.`);
    const active = survivors.slice(expected.bases[bucket],
      expected.bases[bucket] + expected.visibleCounts[bucket]).sort((a, b) => a - b);
    const expectedIds = Array.from({ length: expected.visibleCounts[bucket] },
      (_, local) => expected.bases[bucket] + local);
    requireSame(active, expectedIds, `${label} bucket ${bucket} survivor membership`);
  }

  const address = validation.address;
  const scenarioForAddress = {
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    bucketBases: Uint32Array.from(expected.bases),
    bucketCounts: Uint32Array.from(expected.capacities),
  };
  const activeCounts = Uint32Array.from(
    Array.from({ length: BUCKET_COUNT }, (_, bucket) => commands[bucket * 5 + 1]),
  );
  const visibleIds = Uint32Array.from(survivors);
  const targetShape = createFirstInstanceAddressChallengeTargetShape(OBJECT_COUNT);
  const rebuiltAddress = createFirstInstanceAddressChallengeExpected({
    scenario: scenarioForAddress,
    visibleIds,
    activeCounts,
    targetShape,
  });
  const rebuiltExpectedSha256 = sha256Bytes(rebuiltAddress.bytes);
  const rebuiltResetSha256 = sha256Bytes(new Uint8Array(targetShape.pixelCount * 4));
  verifyStandaloneAddressDigestCommitment({
    visibilityFraction: expectedTrial.visibilityFraction,
    commands,
    survivors,
    address,
  });
  requireCondition(address?.schemaVersion === 1
    && address?.kind === 'render-target-all-address-challenge'
    && address?.pass === true
    && address?.lane === expectedTrial.assignedLaneId
    && address?.commandBufferId === validation.lifecycle?.commandBuffer?.attributeId
    && address?.commandByteOffset === 0
    && address?.reset?.pass === true
    && address?.exactExpectedBytes === true
    && address?.activeMismatchCount === 0
    && address?.paddingMismatchCount === 0
    && address?.targetPaddingMismatchCount === 0
    && address?.outOfRangeIds === 0
    && address?.survivorSha256 === correctness.survivorSha256
    && address?.activeCountsSha256 === canonicalUint32Sha256(activeCounts)
    && SHA256_PATTERN.test(address?.sha256)
    && address?.sha256 === address?.expectedSha256
    && address?.expectedSha256 === rebuiltExpectedSha256
    && address?.activeAddressCount === rebuiltAddress.activeAddressCount
    && address?.paddingAddressCount === rebuiltAddress.paddingAddressCount
    && address?.targetPaddingPixelCount === rebuiltAddress.targetPaddingPixelCount
    && isDeepStrictEqual(address?.coverage, rebuiltAddress.coverage)
    && address?.pixelCount === targetShape.pixelCount
    && address?.addressCount === OBJECT_COUNT
    && address?.byteLength === targetShape.pixelCount * 4
    && address?.reset?.pixelCount === targetShape.pixelCount
    && address?.reset?.addressCount === OBJECT_COUNT
    && address?.reset?.byteLength === targetShape.pixelCount * 4
    && address?.reset?.sha256 === rebuiltResetSha256
    && address?.reset?.expectedSha256 === rebuiltResetSha256,
  `${label} address oracle failed or is detached from production resources.`);
  requireCondition(address.outputStage === 'fragment'
    && address.addressTransport === 'vertex-address-to-rgba8-pixel'
    && address.encoding === 'rgb24-object-id-plus-one-transparent-zero-background'
    && address.target?.pass === true
    && address.target?.width === targetShape.width
    && address.target?.height === targetShape.height
    && address.target?.pixelCount === targetShape.pixelCount
    && address.target?.backendFormat === 'rgba8unorm'
    && address.target?.readbackArrayType === 'Uint8Array'
    && address.target?.bytesPerPixel === 4
    && address.target?.bytesPerRow === targetShape.width * 4
    && address.target?.rowAlignmentBytes === 256
    && address.target?.configuredFormat === 'RGBAFormat/UnsignedByteType'
    && address.target?.origin === 'top-left'
    && address.target?.sampleLocation === 'integer-plus-half'
    && address.target?.samples === 0
    && address.target?.depthBuffer === false
    && address.target?.stencilBuffer === false
    && address.target?.scissorTest === false
    && address.target?.viewportExact === true
    && address.target?.scissorExact === true
    && address.target?.colorSpace === 'none'
    && address.target?.flipY === false
    && address.target?.generateMipmaps === false,
  `${label} address target is not the exact single-sample rgba8 diagnostic.`);
  requireSame(address.commandBuffer, validation.lifecycle.commandBuffer,
    `${label} address/production command buffer`);
  requireSame(address.commandSegment,
    { index: 0, recordBase: 0, byteBase: 0 },
    `${label} address command segment`);
  const addressShader = address.shader;
  const addressBindings = addressShader?.storageBindings;
  const visibleIdsBinding = Array.isArray(addressBindings) ? addressBindings[0] : null;
  requireCondition(typeof addressShader?.rawSources?.vertexShader === 'string'
    && typeof addressShader?.rawSources?.fragmentShader === 'string'
    && addressShader.vertexSha256 === sha256Text(addressShader.rawSources.vertexShader)
    && addressShader.fragmentSha256 === sha256Text(addressShader.rawSources.fragmentShader)
    && Array.isArray(addressShader.vertexInputs)
    && addressShader.vertexInputs.some((input) => input?.name === 'bucketBase')
      === (expectedTrial.assignedLaneId === 'portable')
    && !/\bbucketBase\b/.test(addressShader.rawSources.vertexShader)
      === (expectedTrial.assignedLaneId === 'feature')
    && addressBindings?.length === 1
    && visibleIdsBinding?.semantic === 'visibleIds'
    && visibleIdsBinding?.access === 'read'
    && visibleIdsBinding?.elementType === 'u32'
    && visibleIdsBinding?.count === OBJECT_COUNT
    && visibleIdsBinding?.byteLength === OBJECT_COUNT * Uint32Array.BYTES_PER_ELEMENT
    && visibleIdsBinding?.resourceId === validation.lifecycle.visibleIdsAttributeId
    && Number.isSafeInteger(visibleIdsBinding?.group)
    && visibleIdsBinding.group >= 0
    && Number.isSafeInteger(visibleIdsBinding?.binding)
    && visibleIdsBinding.binding >= 0
    && Number.isSafeInteger(visibleIdsBinding?.visibility)
    && visibleIdsBinding.visibility > 0,
  `${label} address shader/binding evidence is invalid.`);
  const geometry = address.geometryEvidence;
  requireCondition(geometry?.schemaVersion === 1
    && geometry?.kind === 'fragment-address-challenge-geometry-evidence'
    && geometry?.pass === true
    && geometry?.topology === 'triangle-list'
    && geometry?.pixelLocalCoordinates === true
    && isDeepStrictEqual(geometry?.target, targetShape)
    && geometry?.addressedTriangleCount === BUCKET_COUNT
    && geometry?.addressedTrianglesPerSubmittedInstance === 1
    && geometry?.positionMismatchCount === 0
    && geometry?.bucketBaseMismatchCount === 0
    && geometry?.indexMismatchCount === 0
    && geometry?.attributesExact === true
    && geometry?.sharedPayloadExact === true
    && geometry?.indirectIdentityExact === true
    && geometry?.laneOffsetsExact?.[expectedTrial.assignedLaneId] === true
    && geometry?.productionLaneOffsetsExact?.[expectedTrial.assignedLaneId] === true
    && geometry?.expectedInstanceCount === Math.max(...expected.capacities)
    && Number.isSafeInteger(geometry?.indexCount)
    && geometry.indexCount >= BUCKET_COUNT * 3
    && geometry.indexCount % 3 === 0
    && geometry.degenerateTriangleCount === geometry.indexCount / 3 - BUCKET_COUNT,
  `${label} address geometry is not bound to the production survivor/command resources.`);
  if (expectedTrial.assignedLaneId === 'feature') {
    requireCondition(address?.commandBuffer?.lane === 'feature'
      && validation.lifecycle?.productionResourceLedger?.hasBucketBaseVertexAttribute === false
      && geometry.featureOnly === true
      && geometry.portableOnly === undefined
      && geometry.addressMode === 'indirect-first-instance'
      && geometry.bucketBaseAttributeAbsent === true
      && geometry.productionBucketBaseAttributeAbsent === true
      && geometry.productionVisibleIdsIdentityExact === true
      && geometry.productionCommandResourceExact === true,
    `${label} feature evidence contains a portable bucketBase resource.`);
  } else {
    requireCondition(geometry.portableOnly === true
      && geometry.featureOnly === undefined,
    `${label} portable address geometry is mislabeled or contains feature-only evidence.`);
  }
  return {
    resourceIdentity: validateStandaloneLifecycle(
      validation.lifecycle,
      expectedTrial,
      `${label}.lifecycle`,
    ),
    commandSha256: correctness.commandSha256,
    survivorSha256: correctness.survivorSha256,
    addressSha256: address.sha256,
  };
}

function parseCanonicalJsonArray(value, label) {
  if (typeof value !== 'string') reject(`${label} must be canonical JSON text.`);
  let parsed;
  try {
    parsed = JSON.parse(value);
  } catch {
    reject(`${label} is not valid JSON.`);
  }
  requireCondition(Array.isArray(parsed) && JSON.stringify(parsed) === value,
    `${label} is not a canonical JSON array.`);
  return parsed;
}

function parseCanonicalJsonObject(value, label) {
  if (typeof value !== 'string') reject(`${label} must be canonical JSON text.`);
  let parsed;
  try {
    parsed = JSON.parse(value);
  } catch {
    reject(`${label} is not valid JSON.`);
  }
  requireCondition(parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)
    && JSON.stringify(parsed) === value,
  `${label} is not a canonical JSON object.`);
  return parsed;
}

function validateSourceDigest(value, label) {
  record(value, label);
  nonemptyString(value.source, `${label}.source`);
  requireCondition(value.byteLength === Buffer.byteLength(value.source, 'utf8')
    && value.sha256 === sha256Text(value.source),
  `${label} source digest is invalid.`);
  return value.source;
}

function bindingSemanticShape(value) {
  if (Array.isArray(value)) return value.map(bindingSemanticShape);
  if (value === null || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !/(?:^|_)(?:resource)?id$|identity$|uuid$/i.test(key))
    .map(([key, entry]) => [key, bindingSemanticShape(entry)]));
}

export async function verifyStandaloneLaneShaderNormalization(laneRecord) {
  record(laneRecord, 'lane shader record');
  const laneId = laneRecord.laneId;
  requireCondition(['portable', 'feature'].includes(laneId),
    'lane shader record laneId is invalid.');
  const render = record(laneRecord.render, 'lane shader record.render');
  const recomputed = await createFirstInstanceLaneShaderEvidence({
    laneId,
    lane: {
      vertexShader: render.raw?.vertex?.source,
      fragmentShader: render.raw?.fragment?.source,
      vertexInputs: render.vertexInputs,
      storageBindings: ['matrix', 'visibleIds'].map(
        (semantic) => render.storageBindings?.[semantic],
      ),
    },
    compute: Object.fromEntries(['reset', 'cull'].map((phase) => [phase, {
      shader: laneRecord.compute?.[phase]?.raw?.source,
      bindings: laneRecord.compute?.[phase]?.bindings,
    }])),
  });
  requireSame(laneRecord, recomputed,
    'lane shader record independently reconstructed normalization');
  return recomputed;
}

async function validateLaneShaderRecord(capture, expectedTrial, challenge, label) {
  exactKeys(capture, [
    'schemaVersion', 'kind', 'pass', 'laneId', 'observationSerial',
    'observationRequest', 'observationDigest', 'laneRecord',
    'computeShaderEvidence', 'rawSources', 'resourceIdentitiesAtStart',
    'resourceIdentitiesAtEnd', 'commitment',
  ], label);
  requireCondition(capture?.schemaVersion === 1
    && capture?.kind === 'first-instance-live-standalone-fresh-shader-observation'
    && capture?.pass === true
    && capture?.laneId === expectedTrial.assignedLaneId
    && capture?.observationSerial === challenge.captureOrdinal
    && isDeepStrictEqual(capture?.observationRequest, challenge)
    && capture?.laneRecord?.schemaVersion === 1
    && capture?.laneRecord?.kind === 'indirect-first-instance-lane-shader-evidence'
    && capture?.laneRecord?.pass === true
    && capture?.laneRecord?.reasons?.length === 0
    && capture?.laneRecord?.laneId === expectedTrial.assignedLaneId
    && capture?.computeShaderEvidence?.schemaVersion === 1
    && capture?.computeShaderEvidence?.kind
      === 'live-first-instance-standalone-compute-shader-evidence'
    && capture?.computeShaderEvidence?.pass === true
    && capture?.computeShaderEvidence?.laneId === expectedTrial.assignedLaneId,
  `${label} is not valid fresh lane-local shader evidence.`);
  const laneRecord = capture.laneRecord;
  requireSame(capture.resourceIdentitiesAtStart, capture.resourceIdentitiesAtEnd,
    `${label} shader-inspection resource boundary`);
  requireSame(capture.resourceIdentitiesAtEnd, capture.commitment?.resourceIdentities,
    `${label} shader/commitment resource identity`);
  const primedResourceIdentities = validatePrimedResourceIdentities(
    capture.resourceIdentitiesAtStart,
    `${label}.resourceIdentitiesAtStart`,
  );
  const commitment = validateRenderCommitment(
    capture.commitment,
    expectedTrial.assignedLaneId,
    `${label}.commitment`,
  );
  requireSame(commitment.resourceIdentities, primedResourceIdentities,
    `${label} commitment/observation resource identity`);
  const render = laneRecord.render;
  requireCondition(render?.laneId === expectedTrial.assignedLaneId
    && render?.addressMode === (expectedTrial.assignedLaneId === 'portable'
      ? 'bucket-base'
      : 'indirect-first-instance'),
  `${label} render shader uses the wrong lane semantics.`);
  const rawVertex = validateSourceDigest(render?.raw?.vertex, `${label}.raw.vertex`);
  const rawFragment = validateSourceDigest(render?.raw?.fragment, `${label}.raw.fragment`);
  const normalizedVertex = validateSourceDigest(
    render?.normalizedVertex,
    `${label}.normalizedVertex`,
  );
  const normalizedFragment = validateSourceDigest(
    render?.normalizedFragment,
    `${label}.normalizedFragment`,
  );
  requireCondition(normalizedFragment === rawFragment
    && render?.fragmentNormalization === 'identity-no-approved-lane-contrast'
    && capture?.rawSources?.vertexShader === rawVertex
    && capture?.rawSources?.fragmentShader === rawFragment,
  `${label} raw/normalized render sources are inconsistent.`);
  const portable = expectedTrial.assignedLaneId === 'portable';
  requireCondition(render?.occurrenceCounts?.instanceIndex === 2
    && render?.occurrenceCounts?.bucketBase === (portable ? 2 : 0)
    && (rawVertex.match(/\bbucketBase\b/g)?.length ?? 0) === (portable ? 2 : 0)
    && !normalizedVertex.includes('bucketBase'),
  `${label} does not prove the selected addressing expression.`);
  requireCondition(Array.isArray(render?.vertexInputs)
    && render.vertexInputs.some((input) => input?.name === 'bucketBase') === portable
    && Object.keys(render?.storageBindings ?? {}).sort().join('|') === 'matrix|visibleIds',
  `${label} render input/binding roles are invalid.`);
  const computeSemantics = {};
  const computeInputs = {};
  for (const phaseName of ['reset', 'cull']) {
    const lanePhase = laneRecord.compute?.[phaseName];
    const livePhase = capture.computeShaderEvidence.phases?.[phaseName];
    const rawCompute = validateSourceDigest(lanePhase?.raw,
      `${label}.compute.${phaseName}.raw`);
    const normalizedCompute = validateSourceDigest(lanePhase?.normalized,
      `${label}.compute.${phaseName}.normalized`);
    requireCondition(lanePhase?.executableFirstInstanceReferences === 0
      && !/\.\s*firstInstance\b/.test(rawCompute)
      && livePhase?.pass === true
      && livePhase?.laneId === expectedTrial.assignedLaneId
      && livePhase?.wordFour?.pass === true
      && livePhase?.wordFour?.executableAccessCount === 0
      && livePhase?.fixedWorkloadExact === true
      && livePhase?.storageBindingsKnown === true
      && livePhase?.rawSha256 === lanePhase.raw.sha256
      && livePhase?.normalizedSha256 === lanePhase.normalized.sha256
      && livePhase?.normalizedShader === normalizedCompute
      && livePhase?.capture?.computeShader === rawCompute
      && isDeepStrictEqual(livePhase?.capture?.bindings, lanePhase?.bindings),
    `${label} ${phaseName} compute shader evidence is inconsistent.`);
    const expectedCount = phaseName === 'reset' ? BUCKET_COUNT : OBJECT_COUNT;
    const expectedDispatch = phaseName === 'reset' ? [1, 1, 1] : [1024, 1, 1];
    requireCondition(livePhase?.fixedExpectation?.count === expectedCount
      && isDeepStrictEqual(livePhase?.fixedExpectation?.workgroupSize, [64, 1, 1])
      && isDeepStrictEqual(livePhase?.fixedExpectation?.derivedDispatchSize,
        expectedDispatch)
      && isDeepStrictEqual(livePhase?.capture?.execution?.derivedDispatchSize,
        expectedDispatch)
      && livePhase?.capture?.execution?.runtimeMatchesDerived === true,
    `${label} ${phaseName} dispatch evidence differs from the frozen workload.`);
    computeSemantics[phaseName] = {
      normalizedSource: normalizedCompute,
      normalizedSha256: lanePhase.normalized.sha256,
      bindings: bindingSemanticShape(lanePhase.bindings),
    };
    computeInputs[phaseName] = {
      shader: rawCompute,
      bindings: lanePhase.bindings,
    };
  }
  requireCondition(laneRecord?.normalizedSemantics?.vertexSha256
    === render.normalizedVertex.sha256
    && laneRecord?.normalizedSemantics?.fragmentSha256
      === render.normalizedFragment.sha256
    && laneRecord?.normalizedSemantics?.computeSha256ByPhase?.reset
      === laneRecord.compute.reset.normalized.sha256
    && laneRecord?.normalizedSemantics?.computeSha256ByPhase?.cull
      === laneRecord.compute.cull.normalized.sha256,
  `${label} normalized semantic commitments are invalid.`);
  const expectedObservationDigest = sha256Text(JSON.stringify({
    observationRequest: capture.observationRequest,
    laneRecord: capture.laneRecord,
    computeShaderEvidence: capture.computeShaderEvidence,
    commitment: capture.commitment,
    shaderObservationSerial: capture.observationSerial,
  }));
  requireCondition(capture.observationDigest === expectedObservationDigest,
    `${label} observation digest is invalid.`);
  void computeInputs;
  await verifyStandaloneLaneShaderNormalization(capture.laneRecord);
  return {
    laneId: expectedTrial.assignedLaneId,
    rawVertex,
    normalizedVertex,
    normalizedVertexSha256: render.normalizedVertex.sha256,
    rawFragment,
    fragmentSha256: render.raw.fragment.sha256,
    renderBindingShape: bindingSemanticShape({
      vertexInputs: render.vertexInputs,
      storageBindings: render.storageBindings,
    }),
    compute: computeSemantics,
    primedResourceIdentities,
    commitmentStaticIdentity: commitment.staticIdentity,
  };
}

function expectedChallenge(expectedTrial, runId, ordinal, phase, role) {
  return {
    schemaVersion: 1,
    kind: 'live-first-instance-standalone-shader-observation-challenge',
    origin: 'node-runner',
    runId,
    trialId: expectedTrial.trialId,
    planIndex: expectedTrial.planIndex,
    repetitionIndex: expectedTrial.globalQuartetIndex,
    phase,
    role,
    captureOrdinal: expectedTrial.visibilityOrderPosition * 6 + ordinal,
  };
}

function shaderCaptures(artifact) {
  return [
    artifact.preflight?.renderParity?.snapshotValidation?.shaderEvidence,
    artifact.preflight?.validation?.shaderEvidence,
    artifact.timing?.renderParity?.snapshotValidation?.shaderEvidence,
    artifact.timingStart?.validation?.shaderEvidence,
    artifact.postflight?.renderParity?.snapshotValidation?.shaderEvidence,
    artifact.postflight?.validation?.shaderEvidence,
  ];
}

async function validateShaderChallenges(artifact, expectedTrial, runId, usedNonces, label) {
  const phaseRoles = [
    ['preflight', 'render-parity'],
    ['preflight', 'main-validation'],
    ['timing-start', 'render-parity'],
    ['timing-start', 'main-validation'],
    ['postflight', 'render-parity'],
    ['postflight', 'main-validation'],
  ];
  const challenges = array(artifact.shaderObservationChallenges,
    `${label}.shaderObservationChallenges`);
  requireCondition(challenges.length === phaseRoles.length,
    `${label} must retain exactly six shader challenges.`);
  const captures = shaderCaptures(artifact);
  const results = [];
  for (let index = 0; index < challenges.length; index += 1) {
    const challenge = challenges[index];
    const [phase, role] = phaseRoles[index];
    const expected = expectedChallenge(expectedTrial, runId, index + 1, phase, role);
    exactKeys(challenge, [
      'schemaVersion', 'kind', 'origin', 'runId', 'trialId', 'planIndex',
      'repetitionIndex', 'phase', 'role', 'captureOrdinal', 'challengeNonce',
    ], `${label} challenge ${index}`);
    for (const [key, value] of Object.entries(expected)) {
      requireSame(challenge?.[key], value, `${label} challenge ${index}.${key}`);
    }
    requireCondition(SHA256_PATTERN.test(challenge?.challengeNonce)
      && !usedNonces.has(challenge.challengeNonce),
    `${label} challenge ${index} nonce is invalid or reused.`);
    usedNonces.add(challenge.challengeNonce);
    results.push(await validateLaneShaderRecord(
      captures[index],
      expectedTrial,
      challenge,
      `${label} shader capture ${index}`,
    ));
  }
  return results;
}

function expectedAuditContext(expectedTrial, runId, executionMode, planSha256,
  browserInstanceSerial, sessionNamespace, profilePolicy) {
  return {
    runId,
    trialId: expectedTrial.trialId,
    planIndex: expectedTrial.planIndex,
    matrixIndex: expectedTrial.matrixIndex,
    matrixOrdinal: expectedTrial.matrixOrdinal,
    matrixId: expectedTrial.matrixId,
    globalQuartetIndex: expectedTrial.globalQuartetIndex,
    quartetIndex: expectedTrial.quartetIndex,
    quartetId: expectedTrial.quartetId,
    quartetCode: expectedTrial.quartetCode,
    globalSessionIndex: expectedTrial.globalSessionIndex,
    matrixSessionIndex: expectedTrial.matrixSessionIndex,
    sessionPosition: expectedTrial.sessionPosition,
    sessionId: expectedTrial.sessionId,
    assignedLaneId: expectedTrial.assignedLaneId,
    absentLaneId: expectedTrial.absentLaneId,
    visibilityOrderId: expectedTrial.visibilityOrderId,
    visibilityOrderPosition: expectedTrial.visibilityOrderPosition,
    visibilityExposure: expectedTrial.visibilityExposure,
    visibilityFraction: expectedTrial.visibilityFraction,
    executionMode,
    planSha256,
    browserInstanceSerial,
    sessionNamespace,
    profilePolicy,
    protocolWarmupFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES,
    protocolMeasuredFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
    protocolMeasuredBlockSize: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE,
    protocolMeasuredBlockCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS,
  };
}

const TIMESTAMP_RECORD_KEYS = Object.freeze([
  'uid', 'type', 'callIndex', 'contextId', 'frameId', 'durationMs',
]);

function validateTimestampRecord(value, {
  type,
  frameId,
  callIndex,
  contextId = undefined,
  durationMs,
}, label) {
  exactKeys(value, TIMESTAMP_RECORD_KEYS, label);
  requireCondition(value.type === type
    && value.frameId === frameId
    && value.callIndex === callIndex
    && Number.isSafeInteger(value.callIndex)
    && value.callIndex > 0
    && Number.isSafeInteger(value.contextId)
    && value.contextId >= 0
    && (contextId === undefined || value.contextId === contextId)
    && value.durationMs === durationMs
    && Number.isFinite(value.durationMs)
    && value.durationMs > 0,
  `${label} fields are not exactly bound to the frame.`);
  const prefix = type === 'compute' ? 'c' : 'r';
  requireCondition(value.uid === `${prefix}:${callIndex}:${value.contextId}:f${frameId}`,
    `${label} violates the strict timestamp UID grammar.`);
  return value;
}

export function verifyStandaloneTimestampRecordForTest(value, expected) {
  return validateTimestampRecord(value, expected, 'timestamp record');
}

export function verifyStandaloneTimedSerialInterval(rows, timingStartSerials) {
  const records = array(rows, 'timed serial rows');
  requireCondition(records.length === FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
    'timed serial interval must contain exactly 480 measured rows.');
  const warmup = FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES;
  for (let index = 0; index < records.length; index += 1) {
    for (const key of [
      'strategyComputeCallSerial', 'computeCallSerial', 'renderCallSerial',
    ]) {
      requireCondition(Number.isSafeInteger(timingStartSerials?.[key])
        && records[index]?.[key] === timingStartSerials[key] + warmup + index + 1,
      `timed serial row ${index}.${key} is shifted from its exact interval.`);
    }
  }
  return true;
}

function validateTimestampPhase(phase, {
  phaseName,
  expectedCount,
  expectedFrames = null,
  expectedRecords = null,
}, label) {
  exactKeys(phase, [
    'schemaVersion', 'kind', 'phase', 'includedTypes', 'strictUidGrammar', 'pools',
  ], label);
  requireCondition(phase.schemaVersion === 1
    && phase.kind === 'three-r185-timestamp-phase-result'
    && phase.phase === phaseName
    && isDeepStrictEqual(phase.includedTypes, ['render', 'compute'])
    && phase.strictUidGrammar === true,
  `${label} timestamp phase identity is invalid.`);
  exactKeys(phase.pools, ['render', 'compute'], `${label}.pools`);
  const result = {};
  for (const type of ['render', 'compute']) {
    const pool = phase.pools[type];
    exactKeys(pool, ['type', 'included', 'frames', 'uidRecords', 'resolution'],
      `${label}.pools.${type}`);
    requireCondition(pool.type === type
      && pool.included === true
      && pool.frames?.length === expectedCount
      && pool.uidRecords?.length === expectedCount,
    `${label}.${type} pool cardinality is invalid.`);
    if (expectedFrames !== null) requireSame(pool.frames, expectedFrames,
      `${label}.${type} frames`);
    if (expectedRecords !== null) requireSame(pool.uidRecords, expectedRecords[type],
      `${label}.${type} records`);
    exactKeys(pool.resolution, [
      'quantumNs', 'classification', 'recordCount', 'positiveDurationCount',
      'nonpositiveDurationCount',
    ], `${label}.${type}.resolution`);
    requireCondition(pool.resolution.classification === 'fine'
      && pool.resolution.recordCount === expectedCount
      && pool.resolution.positiveDurationCount === expectedCount
      && pool.resolution.nonpositiveDurationCount === 0
      && Number.isFinite(pool.resolution.quantumNs)
      && pool.resolution.quantumNs > 0
      && pool.resolution.quantumNs <= MAXIMUM_TIMESTAMP_QUANTUM_NS,
    `${label}.${type} timestamp resolution is invalid.`);
    pool.uidRecords.forEach((timestampRecord, index) => validateTimestampRecord(
      timestampRecord,
      {
        type,
        frameId: pool.frames[index],
        callIndex: 1,
        durationMs: timestampRecord?.durationMs,
      },
      `${label}.${type}.uidRecords[${index}]`,
    ));
    result[type] = pool;
  }
  return result;
}

function validateTimingRows(artifact, expectedTrial, baseResourceIdentity, summary, label) {
  const rows = array(artifact.rows, `${label}.rows`);
  requireCondition(rows.length === FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
    `${label} must retain exactly 480 rows.`);
  const auditContext = expectedAuditContext(
    expectedTrial,
    artifact.runId,
    artifact.executionMode,
    artifact.planSha256,
    artifact.browserInstanceSerial,
    artifact.sessionNamespace,
    'fresh-playwright-temporary-profile-per-process',
  );
  const expectedSubmittedInstances = expectedStandaloneScenario(
    expectedTrial.visibilityFraction,
  ).expectedVisibleIds.length;
  let firstSerials = null;
  let timingStartSerials = null;
  let previousGpuFrameId = null;
  let timingContext = null;
  const measurementFrames = [];
  const measurementRecords = { compute: [], render: [] };
  const analysisRows = [];
  for (let index = 0; index < rows.length; index += 1) {
    const row = record(rows[index], `${label}.rows[${index}]`);
    for (const [key, value] of Object.entries(expectedTrial)) {
      requireSame(row[key], value, `${label}.rows[${index}].${key}`);
    }
    for (const [key, value] of Object.entries(auditContext)) {
      requireSame(row[key], value, `${label}.rows[${index}].${key}`);
    }
    requireCondition(row.frameIndex === index
      && row.phaseFrameIndex === index
      && row.measuredBlockIndex === Math.floor(index / FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE)
      && row.withinBlockPosition === index % FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE,
    `${label}.rows[${index}] frame/block identity is invalid.`);
    requireCondition(row.laneId === expectedTrial.assignedLaneId
      && row.submittedComputeLaneId === expectedTrial.assignedLaneId
      && row.targetVisibilityFraction === expectedTrial.visibilityFraction
      && row.commandBufferId === baseResourceIdentity.commandBufferAttributeId
      && row.commandSegmentIndex === 0
      && row.commandRecordBase === 0
      && row.commandByteBase === 0
      && row.commandByteOffset === 0
      && row.commandBufferRecordCount === BUCKET_COUNT
      && row.commandBufferByteLength === BUCKET_COUNT * 5 * 4
      && row.configuredDrawCommands === BUCKET_COUNT
      && row.configuredRenderObjects === 1
      && row.configuredComputeDispatches === 2
      && row.configuredComputeSubmissions === 1
      && row.configuredSubmittedInstances === expectedSubmittedInstances,
    `${label}.rows[${index}] selected resource/workload identity is invalid.`);
    requireCondition(row.computeTimestampContextId === baseResourceIdentity.timestampContextId
      && row.computeTimestampRegistrationSerial
        === baseResourceIdentity.timestampRegistrationSerial
      && row.computeTimestampBackendIdentity === baseResourceIdentity.timestampBackendIdentity
      && row.computeTimestampBackendWrapperIdentity
        === baseResourceIdentity.timestampBackendWrapperIdentity
      && row.computeGroupIdentity === baseResourceIdentity.timestampComputeGroupIdentity
      && isDeepStrictEqual(parseCanonicalJsonArray(row.submittedComputeNodeIds,
        `${label}.rows[${index}].submittedComputeNodeIds`),
      baseResourceIdentity.computeNodeIds),
    `${label}.rows[${index}] compute/timestamp identity is invalid.`);
    for (const field of ['gpuComputeMs', 'gpuRenderMs', 'gpuPassTotalMs']) {
      finiteNumber(row[field], `${label}.rows[${index}].${field}`, {
        minimum: 0,
        exclusive: true,
      });
    }
    for (const field of ['cpuCommonUpdateMs', 'cpuComputeSubmitMs', 'cpuRenderSubmitMs',
      'cpuSubmitTotalMs', 'cpuFrameBodyMs']) {
      finiteNumber(row[field], `${label}.rows[${index}].${field}`, { minimum: 0 });
    }
    requireCondition(row.gpuPassTotalMs === row.gpuComputeMs + row.gpuRenderMs
      && row.cpuSubmitTotalMs === row.cpuComputeSubmitMs + row.cpuRenderSubmitMs,
    `${label}.rows[${index}] timing totals are invalid.`);
    const computeUids = parseCanonicalJsonArray(row.gpuComputeTimestampUids,
      `${label}.rows[${index}].gpuComputeTimestampUids`);
    const renderUids = parseCanonicalJsonArray(row.gpuRenderTimestampUids,
      `${label}.rows[${index}].gpuRenderTimestampUids`);
    const computeRecords = parseCanonicalJsonArray(row.gpuComputeTimestampRecords,
      `${label}.rows[${index}].gpuComputeTimestampRecords`);
    const renderRecords = parseCanonicalJsonArray(row.gpuRenderTimestampRecords,
      `${label}.rows[${index}].gpuRenderTimestampRecords`);
    requireCondition(row.gpuComputeTimestampUidCount === 1
      && row.gpuRenderTimestampUidCount === 1
      && row.gpuComputeTimestampDurationValid === true
      && row.gpuRenderTimestampDurationValid === true
      && computeUids.length === 1 && renderUids.length === 1
      && computeRecords.length === 1 && renderRecords.length === 1
      && computeRecords[0]?.uid === computeUids[0]
      && renderRecords[0]?.uid === renderUids[0]
      && computeRecords[0]?.durationMs === row.gpuComputeMs
      && renderRecords[0]?.durationMs === row.gpuRenderMs,
    `${label}.rows[${index}] timestamp attribution is invalid.`);
    requireCondition(row.computeFrameCallIndex === 1
      && row.renderFrameCallIndex === 1
      && row.strictTimestampUidAttribution === true
      && row.expectedComputeTimestampUidCount === 1
      && row.expectedRenderTimestampUidCount === 1,
    `${label}.rows[${index}] timestamp call/UID policy is invalid.`);
    validateTimestampRecord(computeRecords[0], {
      type: 'compute',
      frameId: row.gpuFrameId,
      callIndex: row.computeFrameCallIndex,
      contextId: row.computeTimestampContextId,
      durationMs: row.gpuComputeMs,
    }, `${label}.rows[${index}] compute timestamp`);
    validateTimestampRecord(renderRecords[0], {
      type: 'render',
      frameId: row.gpuFrameId,
      callIndex: row.renderFrameCallIndex,
      durationMs: row.gpuRenderMs,
    }, `${label}.rows[${index}] render timestamp`);
    requireCondition(Number.isSafeInteger(row.gpuFrameId)
      && row.gpuFrameId >= 0
      && (previousGpuFrameId === null || row.gpuFrameId === previousGpuFrameId + 1),
    `${label}.rows[${index}] GPU frame ID is not consecutive.`);
    previousGpuFrameId = row.gpuFrameId;
    measurementFrames.push(row.gpuFrameId);
    measurementRecords.compute.push(computeRecords[0]);
    measurementRecords.render.push(renderRecords[0]);
    const serials = {
      gpuFrameId: row.gpuFrameId,
      strategyComputeCallSerial: row.strategyComputeCallSerial,
      computeCallSerial: row.computeCallSerial,
      renderCallSerial: row.renderCallSerial,
    };
    requireCondition(Object.values(serials).every(Number.isSafeInteger),
      `${label}.rows[${index}] serials are invalid.`);
    firstSerials ??= serials;
    const starts = {
      strategyComputeCallSerial: row.strategyComputeCallSerialAtTimingStart,
      computeCallSerial: row.computeCallSerialAtTimingStart,
      renderCallSerial: row.renderCallSerialAtTimingStart,
    };
    requireCondition(Object.values(starts).every(Number.isSafeInteger),
      `${label}.rows[${index}] timing-start serials are invalid.`);
    timingStartSerials ??= starts;
    requireSame(starts, timingStartSerials,
      `${label}.rows[${index}] timing-start serial commitment`);
    const rowTimingContext = {
      lifecycleCommitmentAtTimingStart: row.lifecycleCommitmentAtTimingStart,
      standaloneLaneId: row.standaloneLaneId,
      standaloneAbsentLaneId: row.standaloneAbsentLaneId,
      standaloneVisibilityOrder: row.standaloneVisibilityOrder,
      standaloneScenarioSwitchSerialAtTimingStart:
        row.standaloneScenarioSwitchSerialAtTimingStart,
      standaloneProductionResourceLedger: row.standaloneProductionResourceLedger,
      standaloneCommandBufferAtTimingStart: row.standaloneCommandBufferAtTimingStart,
      standaloneRenderCommitmentAtTimingStart: row.standaloneRenderCommitmentAtTimingStart,
      renderTargetTextureUuidAtTimingStart: row.renderTargetTextureUuidAtTimingStart,
      renderTargetWidthAtTimingStart: row.renderTargetWidthAtTimingStart,
      renderTargetHeightAtTimingStart: row.renderTargetHeightAtTimingStart,
      renderTargetSamplesAtTimingStart: row.renderTargetSamplesAtTimingStart,
      renderTargetDepthBufferAtTimingStart: row.renderTargetDepthBufferAtTimingStart,
      totalPipelineCacheEntriesAtTimingStart: row.totalPipelineCacheEntriesAtTimingStart,
      computePipelineCacheEntriesAtTimingStart: row.computePipelineCacheEntriesAtTimingStart,
      computeProgramEntriesAtTimingStart: row.computeProgramEntriesAtTimingStart,
      rendererMemoryAtTimingStart: row.rendererMemoryAtTimingStart,
      viewportStateAtTimingStart: row.viewportStateAtTimingStart,
      timestampPoolStaticCommitmentAtTimingStart:
        row.timestampPoolStaticCommitmentAtTimingStart,
      webgpuUncapturedErrorCountAtTimingStart:
        row.webgpuUncapturedErrorCountAtTimingStart,
    };
    timingContext ??= rowTimingContext;
    requireSame(rowTimingContext, timingContext,
      `${label}.rows[${index}] timed-body context commitment`);
    for (const [key, start] of Object.entries(firstSerials)) {
      requireCondition(serials[key] === start + index,
        `${label}.rows[${index}].${key} is not consecutive.`);
    }
    for (const key of ['strategyComputeCallSerial', 'computeCallSerial', 'renderCallSerial']) {
      requireCondition(serials[key]
        === timingStartSerials[key] + FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES
          + index + 1,
      `${label}.rows[${index}].${key} is shifted from the exact timed interval.`);
    }
    analysisRows.push({
      runId: artifact.runId,
      ...Object.fromEntries(Object.keys(expectedTrial).map((key) => [key, row[key]])),
      frameIndex: row.frameIndex,
      phaseFrameIndex: row.phaseFrameIndex,
      gpuPassTotalMs: row.gpuPassTotalMs,
      gpuRenderMs: row.gpuRenderMs,
      gpuComputeMs: row.gpuComputeMs,
      cpuCommonUpdateMs: row.cpuCommonUpdateMs,
      cpuComputeSubmitMs: row.cpuComputeSubmitMs,
      cpuRenderSubmitMs: row.cpuRenderSubmitMs,
      cpuSubmitTotalMs: row.cpuSubmitTotalMs,
      cpuFrameBodyMs: row.cpuFrameBodyMs,
    });
  }
  const measurementPools = validateTimestampPhase(summary.timestampPhases.measurement, {
    phaseName: 'measurement',
    expectedCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
    expectedFrames: measurementFrames,
    expectedRecords: measurementRecords,
  }, `${label}.summary.timestampPhases.measurement`);
  const warmupPools = validateTimestampPhase(summary.timestampPhases.warmup, {
    phaseName: 'warmup',
    expectedCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES,
  }, `${label}.summary.timestampPhases.warmup`);
  for (const type of ['compute', 'render']) {
    const frames = warmupPools[type].frames;
    requireCondition(frames.every((frameId, index) => Number.isSafeInteger(frameId)
      && frameId >= 0
      && (index === 0 || frameId === frames[index - 1] + 1))
      && frames.at(-1) + 1 === measurementPools[type].frames[0],
    `${label} ${type} warmup/measurement GPU-frame interval is not contiguous.`);
    if (type === 'compute') {
      requireCondition(warmupPools[type].uidRecords.every(
        (entry) => entry.contextId === baseResourceIdentity.timestampContextId,
      ), `${label} warmup compute timestamps use the wrong context.`);
    }
  }
  verifyStandaloneTimedSerialInterval(rows, timingStartSerials);
  return { analysisRows, timingContext, timingStartSerials };
}

function validateTimestampSummary(summary, expectedTrial, label) {
  const zeroFields = [
    'missingWarmupRenderFrames',
    'invalidWarmupRenderTimestampUidCountFrames',
    'invalidWarmupRenderTimestampDurationFrames',
    'missingWarmupComputeFrames',
    'invalidWarmupComputeTimestampUidCountFrames',
    'invalidWarmupComputeTimestampDurationFrames',
    'missingRenderFrames',
    'invalidRenderTimestampUidCountFrames',
    'invalidRenderTimestampDurationFrames',
    'missingComputeFrames',
    'invalidComputeTimestampUidCountFrames',
    'invalidComputeTimestampDurationFrames',
  ];
  requireCondition(summary?.rowCount === FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES
    && summary?.warmupRowCount === FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES
    && summary?.expectedRenderTimestampUidCount === 1
    && summary?.expectedComputeTimestampUidCount === 1
    && summary?.timestampAvailable === true
    && summary?.accepted === true
    && summary?.completionInvariant?.pass === true
    && summary?.renderTimestampPoolQualityValid === true
    && summary?.computeTimestampPoolQualityValid === true
    && summary?.warmupRenderTimestampPoolQualityValid === true
    && summary?.warmupComputeTimestampPoolQualityValid === true
    && summary?.warmupTimestampFrameCountValid === true
    && summary?.measurementTimestampFrameCountValid === true
    && Number.isFinite(summary?.quantumNs)
    && summary.quantumNs > 0
    && summary.quantumNs <= MAXIMUM_TIMESTAMP_QUANTUM_NS
    && zeroFields.every((field) => summary?.[field] === 0),
  `${label} timestamp summary/completion evidence is invalid.`);
  for (const [phaseName, expectedCount] of [
    ['warmup', FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES],
    ['measurement', FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES],
  ]) {
    for (const type of ['compute', 'render']) {
      const pool = summary.timestampPhases?.[phaseName]?.pools?.[type];
      requireCondition(pool?.included === true
        && pool?.frames?.length === expectedCount
        && pool?.uidRecords?.length === expectedCount
        && Number.isFinite(pool?.resolution?.quantumNs)
        && pool.resolution.quantumNs > 0
        && pool.resolution.quantumNs <= MAXIMUM_TIMESTAMP_QUANTUM_NS,
      `${label} ${phaseName}/${type} timestamp pool is invalid.`);
    }
  }
  requireCondition(summary.completionInvariant?.laneId === expectedTrial.assignedLaneId
    && summary.completionInvariant?.absentLaneId === expectedTrial.absentLaneId
    && summary.completionInvariant?.resourceLedgerExact === true
    && summary.completionInvariant?.commandBufferExact === true
    && summary.completionInvariant?.lifecycleExact === true
    && summary.completionInvariant?.renderCommitmentExact === true
    && summary.completionInvariant?.timestampPoolsStaticExact === true
    && summary.completionInvariant?.timestampPoolsResolvedCleanly === true
    && summary.completionInvariant?.timestampPreprimeExact === true
    && summary.completionInvariant?.viewportStateExact === true
    && summary.completionInvariant?.webgpuUncapturedErrorsDuringTiming === 0,
  `${label} timing completion invariant is invalid.`);
  const timedFrameCount = FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES
    + FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES;
  requireCondition(summary.completionInvariant.strategyComputeCallsDuringTiming
      === timedFrameCount
    && summary.completionInvariant.computeCallsDuringTiming === timedFrameCount
    && summary.completionInvariant.renderCallsDuringTiming === timedFrameCount,
  `${label} completion counters do not prove exactly one compute/render call per timed frame.`);
  return summary.completionInvariant;
}

function timestampPoolStaticIdentity(pool) {
  return {
    poolIdentity: pool?.poolIdentity ?? null,
    querySetIdentity: pool?.querySetIdentity ?? null,
    resolveBufferIdentity: pool?.resolveBufferIdentity ?? null,
    resultBufferIdentity: pool?.resultBufferIdentity ?? null,
    maxQueries: pool?.maxQueries ?? null,
    isDisposed: pool?.isDisposed ?? null,
  };
}

function validatePinnedViewportState(state, textureUuid, label) {
  const exactRectangle = (value) => value?.x === 0
    && value?.y === 0
    && value?.width === 1280
    && value?.height === 720;
  const exactViewport = (value) => exactRectangle(value)
    && value?.minDepth === 0
    && value?.maxDepth === 1;
  requireCondition(state?.schemaVersion === 1
    && state?.kind === 'three-r185-live-first-instance-viewport-state'
    && exactViewport(state?.renderer?.viewport)
    && exactRectangle(state?.renderer?.scissor)
    && state?.renderer?.scissorTest === false
    && state?.renderer?.activeRenderTargetTextureUuid === null
    && state?.renderTarget?.textureUuid === textureUuid
    && state?.renderTarget?.width === 1280
    && state?.renderTarget?.height === 720
    && exactViewport(state?.renderTarget?.viewport)
    && exactRectangle(state?.renderTarget?.scissor)
    && state?.renderTarget?.scissorTest === false,
  `${label} is not the frozen viewport/render-target state.`);
}

function validateTimedBodyCommitments({
  timingContext,
  completion,
  expectedTrial,
  configured,
  baseResourceIdentity,
  shaderResourceIdentity,
  label,
}) {
  record(timingContext, `${label}.timingContext`);
  requireCondition(timingContext.standaloneLaneId === expectedTrial.assignedLaneId
    && timingContext.standaloneAbsentLaneId === expectedTrial.absentLaneId
    && isDeepStrictEqual(
      parseCanonicalJsonArray(
        timingContext.standaloneVisibilityOrder,
        `${label}.standaloneVisibilityOrder`,
      ),
      expectedTrial.visibilityOrder,
    )
    && timingContext.standaloneScenarioSwitchSerialAtTimingStart
      === expectedTrial.visibilityOrderPosition,
  `${label} timed-body lane/visibility context is invalid.`);
  const ledger = parseCanonicalJsonObject(
    timingContext.standaloneProductionResourceLedger,
    `${label}.standaloneProductionResourceLedger`,
  );
  validateProductionLedger(ledger, expectedTrial, `${label}.timing ledger`);
  requireSame(ledger, configured.strategyLifecycle.productionResourceLedger,
    `${label} configured/timing production ledger`);
  requireSame(completion.productionResourceLedger, ledger,
    `${label} completion/timing production ledger`);
  const command = parseCanonicalJsonObject(
    timingContext.standaloneCommandBufferAtTimingStart,
    `${label}.standaloneCommandBufferAtTimingStart`,
  );
  requireSame(command, configured.strategyLifecycle.commandBuffer,
    `${label} configured/timing command buffer`);
  requireSame(command.attributeId, baseResourceIdentity.commandBufferAttributeId,
    `${label} timed command-buffer identity`);
  const renderCommitment = parseCanonicalJsonObject(
    timingContext.standaloneRenderCommitmentAtTimingStart,
    `${label}.standaloneRenderCommitmentAtTimingStart`,
  );
  const validatedRenderCommitment = validateRenderCommitment(
    renderCommitment,
    expectedTrial.assignedLaneId,
    `${label}.standaloneRenderCommitmentAtTimingStart`,
    baseResourceIdentity,
  );
  requireSame(completion.renderCommitmentAtTimingEnd, renderCommitment,
    `${label} render commitment timing start/end`);
  requireSame(validatedRenderCommitment.resourceIdentities, shaderResourceIdentity,
    `${label} timed render/shader resource identity`);
  requireCondition(typeof timingContext.lifecycleCommitmentAtTimingStart === 'string'
    && timingContext.lifecycleCommitmentAtTimingStart.length === 16
    && completion.lifecycleCommitmentAtTimingStart
      === timingContext.lifecycleCommitmentAtTimingStart
    && completion.lifecycleCommitmentAtTimingEnd
      === timingContext.lifecycleCommitmentAtTimingStart,
  `${label} static lifecycle commitment changed during timing.`);
  requireSame(completion.timestampPreprime, configured.timestampPoolPreprime,
    `${label} timestamp preprime evidence`);
  nonemptyString(
    timingContext.renderTargetTextureUuidAtTimingStart,
    `${label}.renderTargetTextureUuidAtTimingStart`,
  );
  requireCondition(timingContext.renderTargetWidthAtTimingStart === 1280
    && timingContext.renderTargetHeightAtTimingStart === 720
    && timingContext.renderTargetSamplesAtTimingStart === 0
    && timingContext.renderTargetDepthBufferAtTimingStart === true
    && Number.isSafeInteger(timingContext.totalPipelineCacheEntriesAtTimingStart)
    && timingContext.totalPipelineCacheEntriesAtTimingStart >= 0
    && Number.isSafeInteger(timingContext.computePipelineCacheEntriesAtTimingStart)
    && timingContext.computePipelineCacheEntriesAtTimingStart >= 0
    && Number.isSafeInteger(timingContext.computeProgramEntriesAtTimingStart)
    && timingContext.computeProgramEntriesAtTimingStart >= 0
    && timingContext.webgpuUncapturedErrorCountAtTimingStart === 0,
  `${label} render-target/cache/error timing-start commitment is invalid.`);
  const memoryAtStart = parseCanonicalJsonObject(
    timingContext.rendererMemoryAtTimingStart,
    `${label}.rendererMemoryAtTimingStart`,
  );
  const viewportAtStart = parseCanonicalJsonObject(
    timingContext.viewportStateAtTimingStart,
    `${label}.viewportStateAtTimingStart`,
  );
  const poolStaticAtStart = parseCanonicalJsonObject(
    timingContext.timestampPoolStaticCommitmentAtTimingStart,
    `${label}.timestampPoolStaticCommitmentAtTimingStart`,
  );
  validatePinnedViewportState(
    viewportAtStart,
    timingContext.renderTargetTextureUuidAtTimingStart,
    `${label}.viewportStateAtTimingStart`,
  );
  requireSame(completion.viewportStateAtTimingEnd, viewportAtStart,
    `${label} viewport timing start/end`);
  requireCondition(completion.cacheAtTimingEnd?.totalPipelineCacheEntries
      === timingContext.totalPipelineCacheEntriesAtTimingStart
    && completion.cacheAtTimingEnd?.computePipelineCacheEntries
      === timingContext.computePipelineCacheEntriesAtTimingStart
    && completion.cacheAtTimingEnd?.computeProgramEntries
      === timingContext.computeProgramEntriesAtTimingStart,
  `${label} pipeline/program caches changed during timing.`);
  requireSame(completion.cacheAtTimingEnd?.memory, memoryAtStart,
    `${label} renderer memory timing start/end`);
  for (const type of ['render', 'compute']) {
    const start = completion.timestampPoolsAtTimingStart?.[type];
    const end = completion.timestampPoolsAtTimingEnd?.[type];
    requireSame(timestampPoolStaticIdentity(end), timestampPoolStaticIdentity(start),
      `${label} ${type} timestamp-pool static identity`);
    requireSame(timestampPoolStaticIdentity(start), poolStaticAtStart[type],
      `${label} ${type} row/pool static identity`);
    requireCondition(start && end
      && Number.isSafeInteger(start.timestampUidCount)
      && end.timestampUidCount === start.timestampUidCount
        + FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES
        + FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES
      && end.frameCount === FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES
      && end.currentQueryIndex === 0
      && end.queryOffsetCount === 0
      && end.pendingResolve === false
      && end.isDisposed === false
      && end.resultBufferMapState === 'unmapped',
    `${label} ${type} timestamp pool did not resolve the exact timed interval.`);
  }
  return validatedRenderCommitment.resourceIdentities;
}

function validateConfigured(configured, expectedSession, expectedTrial, label) {
  exactKeys(configured, [
    'initialPageBoot', 'pageConstructionLifecycle', 'selectedConfig', 'environment',
    'shaderEvidence', 'timestampPoolPreprime', 'timestampPoolDiagnostics',
    'strategyLifecycle', 'strategyDiagnostics', 'cacheDiagnostics',
    'geometryManifest', 'scenarioManifest', 'workload', 'webgpuUncapturedErrors',
    'webgpuDeviceLosses',
  ], label);
  const boot = configured.initialPageBoot;
  requireCondition(boot?.schemaVersion === 1
    && boot?.kind === 'first-instance-standalone-initial-page-boot'
    && boot?.bootId === 'first-instance-live-standalone-deployment-v1'
    && boot?.modeId === 'first-instance-live-standalone-deployment'
    && boot?.laneId === expectedSession.assignedLaneId
    && isDeepStrictEqual(boot?.visibilityOrder, expectedSession.visibilityOrder)
    && boot?.objectCount === OBJECT_COUNT
    && boot?.bucketCount === BUCKET_COUNT
    && boot?.layout === LAYOUT
    && boot?.initialRebuildCount === 1
    && boot?.priorStrategyConstructionCount === 0,
  `${label} initial page boot differs from the frozen session.`);
  const pageLifecycle = configured.pageConstructionLifecycle;
  requireCondition(pageLifecycle?.schemaVersion === 1
    && pageLifecycle?.kind === 'benchmark-page-strategy-construction-lifecycle'
    && pageLifecycle?.rebuildCount === 1
    && pageLifecycle?.strategyConstructionCount === 1
    && isDeepStrictEqual(pageLifecycle?.constructedStrategyIds,
      ['first-instance-live-standalone-deployment'])
    && pageLifecycle?.selectedStrategyId === 'first-instance-live-standalone-deployment'
    && pageLifecycle?.strictStandaloneBoot === true,
  `${label} constructed a prior/default or opposite strategy.`);
  const config = configured.selectedConfig;
  requireCondition(config?.strategyId === 'first-instance-live-standalone-deployment'
    && config?.objectCount === OBJECT_COUNT
    && config?.bucketCount === BUCKET_COUNT
    && config?.visibilityFraction === expectedSession.visibilityOrder[0]
    && config?.layout === LAYOUT
    && config?.laneId === expectedSession.assignedLaneId
    && isDeepStrictEqual(config?.visibilityOrder, expectedSession.visibilityOrder),
  `${label} selected configuration differs from the frozen session.`);
  requireCondition(configured.shaderEvidence === null,
    `${label} compiled a shader before the first runner challenge.`);
  requireCondition(configured.timestampPoolPreprime?.schemaVersion === 1
    && configured.timestampPoolPreprime?.kind === 'three-r185-timestamp-pool-preprime'
    && configured.timestampPoolPreprime?.addedTimestampUidCount?.render === 1
    && configured.timestampPoolPreprime?.addedTimestampUidCount?.compute === 1,
  `${label} timestamp pools were not exactly pre-primed.`);
  requireCondition(configured.strategyDiagnostics?.kind
    === 'first-instance-live-standalone-deployment'
    && configured.strategyDiagnostics?.laneCommandBufferCount === 1
    && configured.strategyDiagnostics?.configuredDrawCommands === BUCKET_COUNT
    && configured.strategyDiagnostics?.configuredRenderObjects === 1
    && configured.strategyDiagnostics?.configuredComputeDispatches === 2
    && configured.strategyDiagnostics?.configuredComputeSubmissions === 1,
  `${label} strategy diagnostics are invalid.`);
  requireCleanGpuRecord(configured, label);
  const identity = validateStandaloneLifecycle(
    configured.strategyLifecycle,
    { ...expectedTrial, visibilityOrderPosition: 0 },
    `${label}.strategyLifecycle`,
  );
  return {
    identity,
    environment: environmentIdentity(configured.environment),
    workload: workloadIdentity(configured.workload),
  };
}

function validateEvidencePoint(point, expectedTrial, label) {
  exactKeys(point, [
    'validation', 'renderParity', 'workload', 'environment', 'strategyLifecycle',
    'strategyDiagnostics', 'cacheDiagnostics', 'webgpuUncapturedErrors',
    'webgpuDeviceLosses',
  ], label);
  requireCleanGpuRecord(point, label);
  const mainValidation = validateCommandAndMembership(
    point.validation,
    expectedTrial,
    `${label}.validation`,
  );
  const parityValidation = validateCommandAndMembership(
    point.renderParity?.snapshotValidation,
    expectedTrial,
    `${label}.renderParity.snapshotValidation`,
  );
  requireSame(mainValidation.resourceIdentity, parityValidation.resourceIdentity,
    `${label} validation/parity resource identity`);
  requireSame(staticResourceIdentity(point.strategyLifecycle), mainValidation.resourceIdentity,
    `${label} reported/validated resource identity`);
  const output = renderOutputIdentity(point.renderParity, `${label}.renderParity`);
  const workload = workloadIdentity(point.workload);
  const expectedScenario = expectedStandaloneScenario(expectedTrial.visibilityFraction);
  requireCondition(workload.scenarioSeed === SCENARIO_SEED
    && workload.objectCount === OBJECT_COUNT
    && workload.bucketCount === BUCKET_COUNT
    && workload.visibilityFraction === expectedTrial.visibilityFraction
    && workload.layout === LAYOUT
    && workload.expectedVisibleIdsCanonicalSha256
      === expectedScenario.expectedVisibleIdsCanonicalSha256
    && SHA256_PATTERN.test(workload.geometrySha256)
    && SHA256_PATTERN.test(workload.scenarioSha256),
  `${label} workload commitment is invalid.`);
  return {
    resourceIdentity: mainValidation.resourceIdentity,
    commandSha256: mainValidation.commandSha256,
    survivorSha256: mainValidation.survivorSha256,
    addressSha256: mainValidation.addressSha256,
    output,
    workload,
    environment: environmentIdentity(point.environment),
  };
}

function validateTimingStart(timingStart, expectedTrial, label) {
  exactKeys(timingStart, ['validation', 'workload'], label);
  const validation = validateCommandAndMembership(
    timingStart.validation,
    expectedTrial,
    `${label}.validation`,
  );
  return {
    ...validation,
    workload: workloadIdentity(timingStart.workload),
  };
}

function validateProtocol(protocol, expectedTrial, label) {
  requireSame(protocol, {
    schemaVersion: 1,
    kind: 'first-instance-standalone-deployment-timing-protocol',
    warmupFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES,
    measuredFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
    measuredBlockSize: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE,
    measuredBlockCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS,
    selectedLaneOnly: true,
    absentLaneConstructionAllowed: false,
    visibilityExposure: expectedTrial.visibilityExposure,
    estimator: 'median-of-60-eight-frame-block-means',
  }, label);
}

function expectedTrialArtifactKeys(expectedTrial) {
  return [
    ...Object.keys(expectedTrial),
    'rows',
    'artifactSchemaVersion',
    'artifactKind',
    'executionMode',
    'analysisEligible',
    'planSha256',
    'canonicalTrial',
    'scope',
    'runId',
    'capturedAt',
    'sessionNamespace',
    'browserInstanceSerial',
    'executionIdentity',
    'configured',
    'shaderObservationChallenges',
    'preflight',
    'timingStart',
    'timing',
    'postflight',
    'protocol',
    'runnerValidation',
    'pageErrors',
  ];
}

function normalizeRenderBindingShape(shape) {
  const vertexInputs = (shape?.vertexInputs ?? [])
    .filter((input) => input?.name !== 'bucketBase')
    .map((input) => ({
      ...input,
      ...(input.name === 'normal' ? { shaderLocation: 1 } : {}),
    }));
  return bindingSemanticShape({
    vertexInputs,
    storageBindings: shape?.storageBindings,
  });
}

function semanticIdentity(recordValue) {
  return {
    normalizedVertex: recordValue.normalizedVertex,
    normalizedVertexSha256: recordValue.normalizedVertexSha256,
    rawFragment: recordValue.rawFragment,
    fragmentSha256: recordValue.fragmentSha256,
    normalizedRenderBindingShape: normalizeRenderBindingShape(
      recordValue.renderBindingShape,
    ),
    compute: recordValue.compute,
  };
}

function sameTrialObservationEvidence(observations, label) {
  requireCondition(observations.length === 6, `${label} lacks six shader observations.`);
  const baseline = semanticIdentity(observations[0]);
  const primedResourceIdentities = observations[0].primedResourceIdentities;
  const commitmentStaticIdentity = observations[0].commitmentStaticIdentity;
  for (let index = 1; index < observations.length; index += 1) {
    requireSame(semanticIdentity(observations[index]), baseline,
      `${label} shader semantics observation ${index}`);
    requireSame(observations[index].primedResourceIdentities,
      primedResourceIdentities,
      `${label} shader primed resource identity observation ${index}`);
    requireSame(observations[index].commitmentStaticIdentity,
      commitmentStaticIdentity,
      `${label} shader commitment static identity observation ${index}`);
  }
  return {
    ...baseline,
    laneRenderBindingShape: observations[0].renderBindingShape,
    primedResourceIdentities,
    commitmentStaticIdentity,
  };
}

async function validateTrialArtifact({
  artifact,
  expectedTrial,
  expectedSession,
  runId,
  executionMode,
  planSha256,
  sessionNamespace,
  browserInstanceSerial,
  executionIdentity,
  usedNonces,
}) {
  const label = `trial ${expectedTrial.planIndex}`;
  exactKeys(artifact, expectedTrialArtifactKeys(expectedTrial), `${label} artifact`);
  for (const [key, value] of Object.entries(expectedTrial)) {
    requireSame(artifact[key], value, `${label}.${key}`);
  }
  requireCondition(artifact.artifactSchemaVersion === 1
    && artifact.artifactKind === 'first-instance-standalone-deployment-trial-artifact'
    && artifact.executionMode === executionMode
    && artifact.analysisEligible === (executionMode === FULL_EXECUTION_MODE)
    && artifact.planSha256 === planSha256
    && artifact.runId === runId
    && artifact.sessionNamespace === sessionNamespace
    && artifact.browserInstanceSerial === browserInstanceSerial,
  `${label} artifact identity is invalid.`);
  requireSame(artifact.canonicalTrial, expectedTrial, `${label}.canonicalTrial`);
  requireSame(artifact.executionIdentity, executionIdentity, `${label}.executionIdentity`);
  timestamp(artifact.capturedAt, `${label}.capturedAt`);
  requireCondition(artifact.scope === (executionMode === SMOKE_EXECUTION_MODE
    ? 'plumbing smoke; explicitly excluded from analysis and deployment decision'
    : 'full candidate capture; decision deferred to independent analysis and verification'),
  `${label}.scope is invalid.`);
  requireSame(artifact.pageErrors, [], `${label}.pageErrors`);
  const configured = validateConfigured(
    artifact.configured,
    expectedSession,
    expectedTrial,
    `${label}.configured`,
  );
  const preflight = validateEvidencePoint(artifact.preflight, expectedTrial,
    `${label}.preflight`);
  const timingStart = validateTimingStart(artifact.timingStart, expectedTrial,
    `${label}.timingStart`);
  exactKeys(artifact.timing, [
    'renderParity', 'summary', 'environment', 'strategyLifecycle',
    'strategyDiagnostics', 'cacheDiagnostics', 'shaderEvidence',
    'webgpuUncapturedErrors', 'webgpuDeviceLosses',
  ], `${label}.timing`);
  requireCleanGpuRecord(artifact.timing, `${label}.timing`);
  const timingParityValidation = validateCommandAndMembership(
    artifact.timing.renderParity?.snapshotValidation,
    expectedTrial,
    `${label}.timing.renderParity.snapshotValidation`,
  );
  const timingOutput = renderOutputIdentity(
    artifact.timing.renderParity,
    `${label}.timing.renderParity`,
  );
  const completionInvariant = validateTimestampSummary(
    artifact.timing.summary,
    expectedTrial,
    `${label}.timing.summary`,
  );
  const timingLifecycle = validateStandaloneLifecycle(
    artifact.timing.strategyLifecycle,
    expectedTrial,
    `${label}.timing.strategyLifecycle`,
  );
  const postflight = validateEvidencePoint(artifact.postflight, expectedTrial,
    `${label}.postflight`);
  validateProtocol(artifact.protocol, expectedTrial, `${label}.protocol`);
  requireSame(artifact.runnerValidation, {
    schemaVersion: 1,
    kind: 'first-instance-standalone-runner-timing-validation',
    pass: true,
    reasons: [],
  }, `${label}.runnerValidation`);

  const baseResourceIdentity = configured.identity;
  for (const [phase, identity] of [
    ['preflight', preflight.resourceIdentity],
    ['timing start', timingStart.resourceIdentity],
    ['timing parity', timingParityValidation.resourceIdentity],
    ['timing', timingLifecycle],
    ['postflight', postflight.resourceIdentity],
  ]) requireSame(identity, baseResourceIdentity, `${label} ${phase} static resource identity`);
  requireSame(preflight.workload, timingStart.workload, `${label} preflight/timing workload`);
  requireSame(preflight.workload, postflight.workload, `${label} preflight/postflight workload`);
  requireSame(preflight.output, timingOutput, `${label} preflight/timing output`);
  requireSame(preflight.output, postflight.output, `${label} preflight/postflight output`);
  requireSame(preflight.environment, configured.environment,
    `${label} configured/preflight environment`);
  requireSame(postflight.environment, configured.environment,
    `${label} configured/postflight environment`);
  requireSame(environmentIdentity(artifact.timing.environment), configured.environment,
    `${label} configured/timing environment`);

  const observations = await validateShaderChallenges(
    artifact,
    expectedTrial,
    runId,
    usedNonces,
    label,
  );
  const semantics = sameTrialObservationEvidence(observations, label);
  for (const [key, value] of Object.entries(semantics.commitmentStaticIdentity)) {
    requireSame(value, baseResourceIdentity[key], `${label} shader/${key} lifecycle identity`);
  }
  for (const [phase, output] of [
    ['preflight', preflight.output],
    ['timing', timingOutput],
    ['postflight', postflight.output],
  ]) {
    for (const [key, value] of Object.entries(output.productionStaticIdentity)) {
      requireSame(value, baseResourceIdentity[key],
        `${label} ${phase} production/${key} lifecycle identity`);
    }
    requireSame(output.productionResourceIdentities,
      semantics.primedResourceIdentities,
      `${label} ${phase} production/shader resource identity`);
  }
  const timingRows = validateTimingRows(
    artifact,
    expectedTrial,
    baseResourceIdentity,
    artifact.timing.summary,
    label,
  );
  validateTimedBodyCommitments({
    timingContext: timingRows.timingContext,
    completion: completionInvariant,
    expectedTrial,
    configured: artifact.configured,
    baseResourceIdentity,
    shaderResourceIdentity: semantics.primedResourceIdentities,
    label: `${label}.timedBody`,
  });
  return {
    planIndex: expectedTrial.planIndex,
    matrixIndex: expectedTrial.matrixIndex,
    sessionId: expectedTrial.sessionId,
    sessionNamespace,
    laneId: expectedTrial.assignedLaneId,
    visibilityFraction: expectedTrial.visibilityFraction,
    visibilityExposure: expectedTrial.visibilityExposure,
    resourceIdentity: baseResourceIdentity,
    workload: preflight.workload,
    output: outputSemanticIdentity(preflight.output),
    semantics,
    shaderResourceIdentity: semantics.primedResourceIdentities,
    analysisRecord: { ...expectedTrial, rows: timingRows.analysisRows },
  };
}

/**
 * Enforce within-session resource reuse without treating realm-local numeric
 * identity reuse in a later, fully disconnected browser as a leak.
 */
export function verifyStandaloneSessionResourceIdentityRecords(records) {
  const bySession = new Map();
  const namespaces = new Set();
  for (const recordValue of array(records, 'session resource records')) {
    const item = record(recordValue, 'session resource record');
    nonemptyString(item.sessionId, 'session resource record.sessionId');
    nonemptyString(item.sessionNamespace, 'session resource record.sessionNamespace');
    record(item.resourceIdentity, 'session resource record.resourceIdentity');
    record(item.shaderResourceIdentity,
      'session resource record.shaderResourceIdentity');
    const prior = bySession.get(item.sessionId);
    if (prior === undefined) {
      requireCondition(!namespaces.has(item.sessionNamespace),
        `session namespace ${item.sessionNamespace} is reused.`);
      namespaces.add(item.sessionNamespace);
      bySession.set(item.sessionId, item);
    } else {
      requireCondition(prior.sessionNamespace === item.sessionNamespace,
        `session ${item.sessionId} changed namespace.`);
      requireSame(item.resourceIdentity, prior.resourceIdentity,
        `session ${item.sessionId} resource identity`);
      requireSame(item.shaderResourceIdentity, prior.shaderResourceIdentity,
        `session ${item.sessionId} render/compute state binding pipeline identity`);
    }
  }
  return {
    sessionCount: bySession.size,
    sessionNamespaceCount: namespaces.size,
    withinSessionResourceIdentityExact: true,
  };
}

function validateCrossSessionSemantics(records) {
  const commonIdentity = (semantics) => ({
    normalizedVertex: semantics.normalizedVertex,
    normalizedVertexSha256: semantics.normalizedVertexSha256,
    rawFragment: semantics.rawFragment,
    fragmentSha256: semantics.fragmentSha256,
    normalizedRenderBindingShape: semantics.normalizedRenderBindingShape,
    compute: semantics.compute,
  });
  const baseline = commonIdentity(records[0].semantics);
  const laneBindings = new Map();
  for (const recordValue of records) {
    requireSame(commonIdentity(recordValue.semantics), baseline,
      `trial ${recordValue.planIndex} normalized cross-session shader/compute semantics`);
    const prior = laneBindings.get(recordValue.laneId);
    const current = recordValue.semantics.laneRenderBindingShape;
    if (prior === undefined) laneBindings.set(recordValue.laneId, current);
    else requireSame(current, prior,
      `lane ${recordValue.laneId} runtime binding shape`);
  }
  return {
    normalizedVertexSha256: baseline.normalizedVertexSha256,
    fragmentSha256: baseline.fragmentSha256,
    computeNormalizedSha256: Object.fromEntries(
      Object.entries(baseline.compute).map(([phase, value]) => [phase, value.normalizedSha256]),
    ),
    crossSessionSemanticEquality: true,
  };
}

function validateCrossSessionWorkloadAndOutput(records) {
  const byVisibility = new Map();
  for (const recordValue of records) {
    const key = String(recordValue.visibilityFraction);
    const identity = {
      workload: recordValue.workload,
      output: recordValue.output,
    };
    const prior = byVisibility.get(key);
    if (prior === undefined) byVisibility.set(key, identity);
    else requireSame(identity, prior,
      `visibility ${key} cross-session workload/output identity`);
  }
  return Object.fromEntries([...byVisibility.entries()].sort(
    ([left], [right]) => Number(left) - Number(right),
  ));
}

export function verifyStandaloneEnvironmentObservations(observations) {
  const values = array(observations, 'environment observations');
  requireCondition(values.length > 0, 'environment observations must not be empty.');
  const identities = values.map((value, index) => {
    requireCleanEnvironment(value, `environment observations[${index}]`);
    return environmentIdentity(value);
  });
  for (let index = 1; index < identities.length; index += 1) {
    requireSame(identities[index], identities[0],
      `environment observations[${index}] exact identity`);
  }
  return identities[0];
}

export function verifyStandaloneRenderCommitmentForTest(
  commitment,
  laneId,
  lifecycleIdentity = null,
) {
  return validateRenderCommitment(
    commitment,
    laneId,
    'standalone render commitment',
    lifecycleIdentity,
  );
}

export function resolveStandaloneDeploymentDecision({
  smokeMode,
  numericalPass,
  technicalGatePass,
}) {
  if (smokeMode) return null;
  requireCondition(typeof numericalPass === 'boolean'
    && typeof technicalGatePass === 'boolean',
  'full standalone decision inputs must be booleans.');
  return numericalPass && technicalGatePass
    ? 'standalone-confirmed'
    : 'standalone-confirmation-not-met';
}

export async function verifyFirstInstanceStandaloneDeploymentRunDirectory(
  runDirectory,
  { projectRoot = PROJECT_ROOT } = {},
) {
  const resolvedRunDirectory = path.resolve(runDirectory);
  const manifest = await readFinalManifest(resolvedRunDirectory);
  const mode = validateManifestEnvelope(manifest);
  const archive = await readFirstInstanceStandaloneDeploymentDeclaredArtifacts(
    resolvedRunDirectory,
    manifest,
  );
  const trialPaths = new Set(
    manifest.artifacts.trials.map((entry) => entry.artifact.path),
  );
  const values = new Map();
  for (const descriptorRecord of archive.descriptorRecords) {
    if (!trialPaths.has(descriptorRecord.descriptor.path)) {
      values.set(
        descriptorRecord.descriptor.path,
        await archive.read(descriptorRecord.descriptor, descriptorRecord.label),
      );
    }
  }

  const expectedPlan = buildFirstInstanceStandaloneDeploymentPlan({ runId: manifest.runId });
  validateFirstInstanceStandaloneDeploymentPlan(expectedPlan, { runId: manifest.runId });
  requireCondition(manifest.planSha256 === sha256Text(JSON.stringify(expectedPlan)),
    'manifest plan SHA-256 differs from the independently rebuilt frozen plan.');
  const selection = selectedExecution(expectedPlan, mode.smokeMode);
  validatePlanArtifact(manifest, values, expectedPlan, selection, mode);
  const executionIdentity = validateExecutionIdentity(manifest, values);
  const viteRuntime = await validateViteRuntime(manifest, values, projectRoot, mode);
  const lifecycleValidation = verifyStandaloneBrowserLifecycleChain(
    manifest.browserLifecycles,
    selection,
  );
  requireSame(manifest.lifecycleValidation, lifecycleValidation,
    'manifest independently recomputed browser lifecycle validation');

  const usedNonces = new Set();
  const gateResults = await validateForcedFeatureOffGates(
    manifest,
    values,
    selection,
    manifest.browserLifecycles,
    usedNonces,
  );
  const sessions = validateSessionArtifacts(
    manifest,
    values,
    selection,
    manifest.browserLifecycles,
  );
  const telemetryResults = validateTelemetryArtifacts(
    manifest,
    values,
    selection,
    sessions,
  );
  const trialValidation = await validateTrialArtifacts({
    manifest,
    archive,
    selection,
    sessions,
    executionIdentity,
    usedNonces,
  });
  requireCondition(usedNonces.size
    === selection.trials.length * 6 + selection.matrices.length * 2,
  'global runner/interactive shader nonce cardinality is invalid.');
  const shaderSemantics = validateCrossSessionSemantics(trialValidation.results);
  const workloadAndOutput = validateCrossSessionWorkloadAndOutput(
    trialValidation.results,
  );
  const matrices = validateMatrixArtifacts({
    manifest,
    values,
    selection,
    sessions,
    trialResults: trialValidation.results,
    gateResults,
    telemetryResults,
  });
  const globalIdentity = validateGlobalIdentity(
    manifest,
    sessions,
    trialValidation.results,
    gateResults,
  );
  const journal = validateJournal(manifest, values, selection);
  const analysisMode = mode.smokeMode ? 'smoke' : 'candidate';
  const summary = summarizeFirstInstanceStandaloneDeployment(
    expectedPlan,
    trialValidation.results.map((result) => result.analysisRecord),
    { mode: analysisMode },
  );

  if (mode.smokeMode) {
    requireCondition(summary.decisionEligibility.eligible === false
      && summary.preregisteredNumericalDecision.status === 'excluded'
      && summary.preregisteredNumericalDecision.pass === null,
    'smoke analysis unexpectedly became decision-eligible.');
    return {
      schemaVersion: 1,
      kind: 'first-instance-standalone-deployment-smoke-independent-consistency',
      status: 'consistent',
      runId: manifest.runId,
      runDirectory: resolvedRunDirectory,
      executionMode: manifest.executionMode,
      analysisMode,
      decision: null,
      numericalDecision: null,
      authenticityVerified: false,
      analysisEligible: false,
      artifactCount: archive.artifactCount,
      matrixCount: selection.matrices.length,
      sessionCount: selection.sessions.length,
      trialCount: selection.trials.length,
      rowCount: summary.nRows,
      shaderObservationCount: usedNonces.size,
      lifecycleValidation,
      resourceAudit: trialValidation.resourceAudit,
      shaderSemantics,
      workloadAndOutput,
      viteRuntime,
      journal,
      summary,
    };
  }

  const technicalGatePass = globalIdentity.technicalGatePass
    && matrices.every((matrix) => matrix.technicalGatePass);
  requireCondition(manifest.telemetryPolicy.allMatrixGatesPassed === technicalGatePass,
    'manifest allMatrixGatesPassed differs from independently recomputed gates.');
  const numericalDecision = summary.preregisteredNumericalDecision;
  requireCondition(numericalDecision.status === 'evaluated'
    && numericalDecision.eligible === true
    && typeof numericalDecision.pass === 'boolean',
  'full candidate numerical decision was not independently evaluated.');
  const decision = resolveStandaloneDeploymentDecision({
    smokeMode: false,
    numericalPass: numericalDecision.pass,
    technicalGatePass,
  });
  return {
    schemaVersion: 1,
    kind: 'first-instance-standalone-deployment-independent-verification',
    status: 'consistent',
    runId: manifest.runId,
    runDirectory: resolvedRunDirectory,
    executionMode: manifest.executionMode,
    analysisMode,
    decision,
    numericalDecision,
    technicalGatePass,
    authenticityVerified: false,
    analysisEligible: true,
    artifactCount: archive.artifactCount,
    matrixCount: selection.matrices.length,
    sessionCount: selection.sessions.length,
    trialCount: selection.trials.length,
    rowCount: summary.nRows,
    shaderObservationCount: usedNonces.size,
    lifecycleValidation,
    resourceAudit: trialValidation.resourceAudit,
    shaderSemantics,
    workloadAndOutput,
    globalIdentity,
    matrices: matrices.map(({ entry, technicalGatePass: pass }) => ({
      matrixId: entry.matrixId,
      technicalGatePass: pass,
    })),
    viteRuntime,
    journal,
    summary,
  };
}

function parseCliArguments(arguments_) {
  if (arguments_.length !== 1 || arguments_[0].startsWith('-')) {
    throw new Error(
      'Usage: node analysis/verify-first-instance-standalone-deployment.mjs <run-directory>',
    );
  }
  return arguments_[0];
}

async function main() {
  const runDirectory = parseCliArguments(process.argv.slice(2));
  const result = await verifyFirstInstanceStandaloneDeploymentRunDirectory(runDirectory);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
}

const invokedPath = process.argv[1] === undefined ? null : path.resolve(process.argv[1]);
if (invokedPath !== null && pathToFileURL(invokedPath).href === import.meta.url) {
  main().catch((error) => {
    process.stderr.write(`${error?.stack ?? error}\n`);
    process.exitCode = 1;
  });
}
