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
  summarizeFirstInstanceLiveOrderFactorial,
} from './first-instance-live-order-factorial-summary.mjs';
import {
  summarizeLiveFirstInstanceTrialRows,
} from './live-first-instance-crossover-summary.mjs';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
} from '../src/benchmark/first-instance-live-crossover-schedule.js';
import {
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_KIND,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_COUNT,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIAL_COUNT,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIALS_PER_SESSION,
  buildFirstInstanceLiveOrderFactorialPlan,
} from '../src/benchmark/first-instance-live-order-factorial-plan.js';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
  buildFirstInstanceLiveCrossoverPlan,
} from '../src/benchmark/plan.js';
import {
  executionDependencyClosuresMatch,
  validateCandidateViteRuntimeAudit,
} from '../scripts/execution-dependency-closure.mjs';
import {
  liveFirstInstanceCrossoverScheduleSha256,
  validateLiveFirstInstanceForcedFeatureOffGate,
  validateLiveFirstInstanceTrialEvidence,
} from '../scripts/live-first-instance-evidence-validation.mjs';
import {
  sourceProvenanceMatches,
} from '../scripts/source-provenance.mjs';

const brotliDecompressAsync = promisify(brotliDecompress);
const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;
const VISIBILITY_FRACTION = 0.99;
const LAYOUT = 'baseline';
const SCENARIO_SEED = 0xb1ad_2026;
const SETUP_PRIME_TOPOLOGY = 'staged-order-factorial-v1';
const COMPATIBILITY_MAPPING = 'candidate-high-visibility-c-orientation-v1';
const MAXIMUM_CPU_TIMER_QUANTUM_MS = 0.01;
const MANIFEST_KIND = 'first-instance-live-order-factorial-development-manifest';
const SMOKE_MANIFEST_KIND = 'first-instance-live-order-factorial-smoke-manifest';
const SMOKE_ONLY_DECISION = 'smoke-only-excluded-from-factorial-diagnostic-analysis';
const MAX_MANIFEST_BYTES = 16 * 1024 * 1024;
const MAX_JSON_ARTIFACT_BYTES = 256 * 1024 * 1024;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
]);
const UTF8_DECODER = new TextDecoder('utf-8', { fatal: true });

function reject(message) {
  throw new Error(`Live first-instance order-factorial verification rejected: ${message}`);
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

function sha256Bytes(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function timestamp(value, label) {
  const parsed = typeof value === 'string' ? Date.parse(value) : NaN;
  if (!Number.isFinite(parsed)) reject(`${label} must be a valid timestamp.`);
  return parsed;
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
  const descriptor = exactKeys(
    value,
    ['path', 'encoding', 'byteLength', 'sha256'],
    label,
  );
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
  const descriptor = exactKeys(
    value,
    [
      'path', 'encoding', 'jsonByteLength', 'jsonSha256',
      'brotliByteLength', 'brotliSha256',
    ],
    label,
  );
  canonicalRelativePath(descriptor.path, `${label}.path`);
  requireCondition(
    descriptor.encoding === 'brotli-json-utf8',
    `${label}.encoding is invalid.`,
  );
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

function collectDescriptors(manifest) {
  const artifacts = exactKeys(manifest.artifacts, [
    'plan',
    'forcedFeatureOffGate',
    'executionIdentityStart',
    'executionIdentityEnd',
    'sessions',
    'trials',
    'diagnosticSummary',
    'viteRuntimeAudit',
  ], 'manifest.artifacts');
  const descriptors = [];
  const add = (descriptor, encoding, label) => {
    const validated = encoding === 'json'
      ? validateJsonDescriptor(descriptor, label)
      : validateBrotliDescriptor(descriptor, label);
    descriptors.push({ descriptor: validated, encoding, label });
  };
  add(artifacts.plan, 'json', 'manifest.artifacts.plan');
  add(artifacts.forcedFeatureOffGate, 'brotli', 'manifest.artifacts.forcedFeatureOffGate');
  add(artifacts.executionIdentityStart, 'json',
    'manifest.artifacts.executionIdentityStart');
  add(artifacts.executionIdentityEnd, 'json', 'manifest.artifacts.executionIdentityEnd');
  array(artifacts.sessions, 'manifest.artifacts.sessions').forEach(
    (descriptor, index) => add(descriptor, 'json', `manifest.artifacts.sessions[${index}]`),
  );
  array(artifacts.trials, 'manifest.artifacts.trials').forEach((trial, index) => {
    exactKeys(trial, [
      'factorialPlanIndex',
      'factorialTrialId',
      'sessionIndex',
      'sessionTrialIndex',
      'factorialCellId',
      'compatibilityPlanIndex',
      'compatibilityRepetitionIndex',
      'strictSemanticSha256',
      'timingStartCacheState',
      'artifact',
    ], `manifest.artifacts.trials[${index}]`);
    add(trial.artifact, 'brotli', `manifest.artifacts.trials[${index}].artifact`);
  });
  add(artifacts.diagnosticSummary, 'json', 'manifest.artifacts.diagnosticSummary');
  add(artifacts.viteRuntimeAudit, 'json', 'manifest.artifacts.viteRuntimeAudit');

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
      } else if (stats.isFile()) {
        actualFiles.add(relativePath);
      } else {
        reject(`inventory entry ${relativePath} is not a regular file or directory.`);
      }
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
  requireSame(
    [...actualDirectories].sort(),
    [...expectedDirectories].sort(),
    'run directory inventory',
  );
}

/**
 * Verify the byte-level declared artifact container. This does not perform the
 * factorial semantic verification and never returns a scientific decision.
 */
export async function readFirstInstanceLiveOrderFactorialDeclaredArtifacts(
  runDirectory,
  manifest,
) {
  const resolvedDirectory = path.resolve(runDirectory);
  const descriptorRecords = collectDescriptors(manifest);
  await exactInventory(
    resolvedDirectory,
    descriptorRecords.map(({ descriptor }) => descriptor.path),
  );

  const values = new Map();
  for (const { descriptor, encoding, label } of descriptorRecords) {
    const filename = descriptorPath(resolvedDirectory, descriptor.path);
    if (encoding === 'json') {
      const bytes = await readStableRegularFile(
        filename,
        descriptor.byteLength,
        `${label} file`,
      );
      requireCondition(bytes.length === descriptor.byteLength,
        `${label} byteLength differs from the file.`);
      requireCondition(sha256Bytes(bytes) === descriptor.sha256,
        `${label} sha256 differs from the file.`);
      values.set(descriptor.path, parseJsonBytes(bytes, `${label} file`));
      continue;
    }

    const compressed = await readStableRegularFile(
      filename,
      descriptor.brotliByteLength,
      `${label} Brotli file`,
    );
    requireCondition(compressed.length === descriptor.brotliByteLength,
      `${label} brotliByteLength differs from the file.`);
    requireCondition(sha256Bytes(compressed) === descriptor.brotliSha256,
      `${label} brotliSha256 differs from the file.`);
    let jsonBytes;
    try {
      jsonBytes = await brotliDecompressAsync(compressed, {
        maxOutputLength: descriptor.jsonByteLength,
      });
    } catch (error) {
      reject(`${label} Brotli payload cannot be decoded: ${error?.code ?? error?.message}.`);
    }
    requireCondition(jsonBytes.length === descriptor.jsonByteLength,
      `${label} jsonByteLength differs after decompression.`);
    requireCondition(sha256Bytes(jsonBytes) === descriptor.jsonSha256,
      `${label} jsonSha256 differs after decompression.`);
    values.set(descriptor.path, parseJsonBytes(jsonBytes, `${label} JSON payload`));
  }
  return values;
}

async function readFinalManifest(runDirectory) {
  const filename = path.join(path.resolve(runDirectory), 'manifest.json');
  const bytes = await readStableRegularFile(
    filename,
    MAX_MANIFEST_BYTES,
    'final manifest',
  );
  return parseJsonBytes(bytes, 'final manifest');
}

function sameArray(left, right) {
  return Array.isArray(left) && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

export function deriveFirstInstanceLiveOrderEnvironmentIdentity(environment) {
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
    performanceNowQuantumMs: environment?.performanceNowQuantumMs ?? null,
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

function requireValidCpuTimerQuantum(environment, label) {
  requireCondition(Number.isFinite(environment?.performanceNowQuantumMs)
    && environment.performanceNowQuantumMs > 0
    && environment.performanceNowQuantumMs <= MAXIMUM_CPU_TIMER_QUANTUM_MS,
  `${label} has an invalid CPU timer quantum.`);
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
  requireValidCpuTimerQuantum(environment, label);
  requireCondition(environment?.webgpuUncapturedErrorCount === 0,
    `${label} has an uncaptured WebGPU error.`);
  requireCondition(environment?.webgpuValidationErrorCount === 0,
    `${label} has a WebGPU validation error.`);
  requireCondition(environment?.webgpuDeviceLossCount === 0,
    `${label} has WebGPU device loss.`);
}

function requireCleanGpuRecord(value, label) {
  requireCondition(Array.isArray(value?.webgpuUncapturedErrors)
    && value.webgpuUncapturedErrors.length === 0,
  `${label} has uncaptured WebGPU errors.`);
  requireCondition(Array.isArray(value?.webgpuDeviceLosses)
    && value.webgpuDeviceLosses.length === 0,
  `${label} has WebGPU device losses.`);
  requireCleanEnvironment(value?.environment, `${label} environment`);
}

function candidateCompatibilitySpec(factorialTrial, compatibilityHighPlan, runId) {
  const matches = compatibilityHighPlan.filter((candidate) => (
    candidate.repetitionIndex < 4
      && candidate.visibilityFraction === VISIBILITY_FRACTION
      && sameArray(candidate.lanePhysicalOrder, factorialTrial.laneConstructionOrder)
      && candidate.superblockOrientationOffset === factorialTrial.superblockOrientationOffset
  ));
  requireCondition(matches.length === 1,
    `factorial plan ${factorialTrial.planIndex} has no unique compatibility cell.`);
  const selected = matches[0];
  return {
    ...selected,
    runId,
    trialId: factorialTrial.trialId,
    lanePhysicalOrder: [...selected.lanePhysicalOrder],
    laneConstructionOrder: [...factorialTrial.laneConstructionOrder],
    firstComputeUseOrder: [...factorialTrial.firstComputeUseOrder],
    renderPipelinePrimeOrder: [...factorialTrial.renderPipelinePrimeOrder],
    setupPrimeTopology: SETUP_PRIME_TOPOLOGY,
    timestampPreprimeLaneId: factorialTrial.timestampPreprimeLaneId,
  };
}

function expectedAuditContext(spec, factorialTrial, runId) {
  return {
    runId,
    trialId: spec.trialId,
    planIndex: spec.planIndex,
    repetitionIndex: spec.repetitionIndex,
    modeOrderPosition: spec.modeOrderPosition,
    visibilityOrderPosition: spec.visibilityOrderPosition,
    layoutOrderPosition: spec.layoutOrderPosition,
    plannedModeOrder: spec.modeOrder.join('|'),
    plannedVisibilityOrder: spec.visibilityOrder.join('|'),
    plannedLayoutOrder: spec.layoutOrder.join('|'),
    protocolWarmupFrames: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    protocolMeasuredFrames: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    plannedLanePhysicalOrder: spec.lanePhysicalOrder.join('|'),
    plannedLaneConstructionOrder: spec.laneConstructionOrder.join('|'),
    plannedFirstComputeUseOrder: spec.firstComputeUseOrder.join('|'),
    plannedRenderPipelinePrimeOrder: spec.renderPipelinePrimeOrder.join('|'),
    setupPrimeTopology: spec.setupPrimeTopology,
    timestampPreprimeLaneId: spec.timestampPreprimeLaneId,
    plannedTimestampPreprimeLaneId: spec.timestampPreprimeLaneId,
    superblockOrientationOffset: spec.superblockOrientationOffset,
    plannedScheduleSha256: liveFirstInstanceCrossoverScheduleSha256(
      spec.superblockOrientationOffset,
    ),
    factorialExperimentKind: FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_KIND,
    factorialPlanIndex: factorialTrial.planIndex,
    factorialSessionIndex: factorialTrial.sessionIndex,
    factorialSessionTrialIndex: factorialTrial.sessionTrialIndex,
    factorialCellIndex: factorialTrial.factorialCellIndex,
    factorialCellId: factorialTrial.factorialCellId,
    factorialC: factorialTrial.factorLevels.C,
    factorialK: factorialTrial.factorLevels.K,
    factorialR: factorialTrial.factorLevels.R,
    factorialT: factorialTrial.factorLevels.T,
    factorialLaneConstructionOrder: factorialTrial.laneConstructionOrder.join('|'),
    factorialFirstComputeUseOrder: factorialTrial.firstComputeUseOrder.join('|'),
    factorialRenderPipelinePrimeOrder: factorialTrial.renderPipelinePrimeOrder.join('|'),
    factorialTimestampPreprimeLaneId: factorialTrial.timestampPreprimeLaneId,
    factorialCompatibilityMapping: COMPATIBILITY_MAPPING,
    factorialCompatibilityPlanIndex: spec.planIndex,
    factorialCompatibilityRepetitionIndex: spec.repetitionIndex,
  };
}

function validateManifestIdentity(manifest) {
  exactKeys(manifest, [
    'schemaVersion',
    'kind',
    'scope',
    'runId',
    'completedAt',
    'browser',
    'fixedWorkload',
    'executionPolicy',
    'compatibilityMapping',
    'environmentIdentity',
    'workloadIdentity',
    'executionIdentity',
    'processLifecycle',
    'artifacts',
    'completion',
  ], 'final manifest');
  requireCondition(manifest.schemaVersion === 1, 'manifest.schemaVersion is invalid.');
  requireCondition([MANIFEST_KIND, SMOKE_MANIFEST_KIND].includes(manifest.kind),
    'manifest.kind is invalid.');
  const smokeMode = manifest.kind === SMOKE_MANIFEST_KIND;
  const sessionCount = smokeMode ? 1 : FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_COUNT;
  const executedTrialCount = smokeMode ? 1 : FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIAL_COUNT;
  const trialsPerSession = smokeMode
    ? 1
    : FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIALS_PER_SESSION;
  const entryDocumentCount = smokeMode ? 2 : 3;
  const expectedScope = smokeMode
    ? 'one-trial plumbing smoke; explicitly excluded from factorial diagnostic analysis'
    : 'development diagnostic; no retry, replacement, outlier deletion, or pass claim';
  requireCondition(manifest.scope === expectedScope, 'manifest.scope is invalid.');
  nonemptyString(manifest.runId, 'manifest.runId');
  timestamp(manifest.completedAt, 'manifest.completedAt');
  exactKeys(manifest.browser, ['executable', 'arguments'], 'manifest.browser');
  nonemptyString(manifest.browser.executable, 'manifest.browser.executable');
  requireSame(manifest.browser.arguments, [...BROWSER_ARGS], 'manifest browser arguments');
  requireSame(manifest.fixedWorkload, {
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    visibilityFraction: VISIBILITY_FRACTION,
    layout: LAYOUT,
    scenarioSeed: SCENARIO_SEED,
    warmupFrames: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    measuredFrames: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    setupPrimeTopology: SETUP_PRIME_TOPOLOGY,
  }, 'manifest fixed workload');
  requireSame(manifest.executionPolicy, {
    smokeMode,
    sessionCount,
    trialsPerSession,
    retryCount: 0,
    replacementAllowed: false,
    outlierRemovalAllowed: false,
    efficacyStoppingAllowed: false,
    artificialTimestampMapDelayMs: 0,
  }, 'manifest execution policy');
  requireCondition(manifest.compatibilityMapping === COMPATIBILITY_MAPPING,
    'manifest compatibility mapping is invalid.');
  requireValidCpuTimerQuantum(manifest.environmentIdentity, 'manifest environment identity');
  requireSame(manifest.completion, {
    frozenTrialCount: FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIAL_COUNT,
    executedTrialCount,
    persistedTrialCount: executedTrialCount,
    freshShaderChallengeCount: executedTrialCount * 6,
    entryDocumentCount,
    allBrowsersClosed: true,
    sessionOverlapDetected: false,
  }, 'manifest completion');
  return {
    smokeMode,
    sessionCount,
    executedTrialCount,
    trialsPerSession,
    entryDocumentCount,
  };
}

/** Validate and classify only the final manifest envelope, not its artifacts. */
export function classifyFirstInstanceLiveOrderManifestEnvelope(manifest) {
  return { ...validateManifestIdentity(manifest) };
}

function artifactValue(values, descriptor, label) {
  if (!values.has(descriptor.path)) reject(`${label} was not decoded.`);
  return values.get(descriptor.path);
}

function expectedCompatibilityPlanRecord(factorialTrial, compatibilitySpec) {
  return {
    factorialPlanIndex: factorialTrial.planIndex,
    factorialTrialId: factorialTrial.trialId,
    candidatePlanIndex: compatibilitySpec.planIndex,
    candidateRepetitionIndex: compatibilitySpec.repetitionIndex,
    candidateVisibilityOrderPosition: compatibilitySpec.visibilityOrderPosition,
    lanePhysicalOrder: compatibilitySpec.lanePhysicalOrder,
    superblockOrientationOffset: compatibilitySpec.superblockOrientationOffset,
  };
}

function validatePlanArtifact(
  planArtifact,
  expectedPlan,
  executionPlan,
  compatibilitySpecs,
  runId,
  smokeMode,
) {
  exactKeys(planArtifact, [
    'schemaVersion',
    'kind',
    'runId',
    'frozenPlan',
    'executionMode',
    'executedFactorialPlanIndexes',
    'compatibilityMapping',
    'compatibilitySpecs',
    'executionPolicy',
  ], 'plan artifact');
  requireCondition(planArtifact.schemaVersion === 1
    && planArtifact.kind === 'first-instance-live-order-factorial-execution-plan'
    && planArtifact.runId === runId,
  'plan artifact identity is invalid.');
  requireSame(planArtifact.frozenPlan, expectedPlan, 'retained frozen plan');
  requireCondition(
    planArtifact.executionMode === (smokeMode
      ? 'one-trial-smoke-excluded-from-analysis'
      : 'full-factorial'),
    'plan artifact execution mode is invalid.',
  );
  requireSame(
    planArtifact.executedFactorialPlanIndexes,
    executionPlan.map(({ planIndex }) => planIndex),
    'executed factorial plan indexes',
  );
  requireCondition(planArtifact.compatibilityMapping === COMPATIBILITY_MAPPING,
    'plan compatibility mapping is invalid.');
  requireSame(
    planArtifact.compatibilitySpecs,
    expectedPlan.map((trial, index) => expectedCompatibilityPlanRecord(
      trial,
      compatibilitySpecs[index],
    )),
    'plan compatibility records',
  );
  requireSame(planArtifact.executionPolicy, {
    retryCount: 0,
    replacementAllowed: false,
    outlierRemovalAllowed: false,
    efficacyStoppingAllowed: false,
    claim: 'development diagnostic only',
  }, 'plan execution policy');
}

function validateExecutionIdentity(manifest, values, runId) {
  const declaration = exactKeys(manifest.executionIdentity, [
    'sourceProvenanceStart',
    'sourceProvenanceEnd',
    'sourceProvenanceMatched',
    'executionDependencyClosureStart',
    'executionDependencyClosureEnd',
    'executionDependencyClosureMatched',
  ], 'manifest.executionIdentity');
  const start = artifactValue(
    values,
    manifest.artifacts.executionIdentityStart,
    'execution identity start',
  );
  const end = artifactValue(
    values,
    manifest.artifacts.executionIdentityEnd,
    'execution identity end',
  );
  exactKeys(start, [
    'schemaVersion', 'kind', 'runId', 'sourceProvenance',
    'executionDependencyClosure',
  ], 'execution identity start');
  exactKeys(end, [
    'schemaVersion', 'kind', 'runId', 'sourceProvenance',
    'executionDependencyClosure', 'sourceMatchesStart',
    'executionDependencyClosureMatchesStart',
  ], 'execution identity end');
  requireCondition(start.schemaVersion === 1
    && start.kind === 'first-instance-live-order-factorial-execution-identity-start'
    && start.runId === runId,
  'execution identity start has the wrong identity.');
  requireCondition(end.schemaVersion === 1
    && end.kind === 'first-instance-live-order-factorial-execution-identity-end'
    && end.runId === runId,
  'execution identity end has the wrong identity.');
  requireSame(start.sourceProvenance, declaration.sourceProvenanceStart,
    'declared start source provenance');
  requireSame(end.sourceProvenance, declaration.sourceProvenanceEnd,
    'declared end source provenance');
  requireSame(start.executionDependencyClosure,
    declaration.executionDependencyClosureStart,
    'declared start dependency closure');
  requireSame(end.executionDependencyClosure,
    declaration.executionDependencyClosureEnd,
    'declared end dependency closure');
  requireCondition(declaration.sourceProvenanceMatched === true
    && end.sourceMatchesStart === true
    && sourceProvenanceMatches(start.sourceProvenance, end.sourceProvenance),
  'source provenance start/end equality is invalid.');
  requireCondition(declaration.executionDependencyClosureMatched === true
    && end.executionDependencyClosureMatchesStart === true
    && executionDependencyClosuresMatch(
      start.executionDependencyClosure,
      end.executionDependencyClosure,
    ),
  'execution dependency closure start/end equality is invalid.');
  for (const [label, provenance] of [
    ['start', start.sourceProvenance],
    ['end', end.sourceProvenance],
  ]) {
    requireCondition(provenance?.status === 'available'
      && provenance.captureStable === true
      && provenance.dirty === false
      && provenance.packageLockTracked === true,
    `${label} source provenance is not clean and stable.`);
  }
  return { start, end };
}

function validateLifecycle(value, {
  role,
  sessionOrdinal,
  serial,
  label,
}) {
  exactKeys(value, [
    'schemaVersion',
    'kind',
    'browserInstanceSerial',
    'role',
    'sessionOrdinal',
    'launchedAt',
    'contextCreatedAt',
    'pageCreatedAt',
    'closedAt',
    'contextCountBeforeClose',
    'pageCountBeforeClose',
    'disconnectedEventCount',
    'closedBeforeNextLaunch',
  ], label);
  requireCondition(value.schemaVersion === 1
    && value.kind === 'first-instance-live-order-factorial-browser-lifecycle'
    && value.role === role
    && value.sessionOrdinal === sessionOrdinal
    && value.browserInstanceSerial === serial,
  `${label} identity is invalid.`);
  const launchedAt = timestamp(value.launchedAt, `${label}.launchedAt`);
  const contextCreatedAt = timestamp(value.contextCreatedAt, `${label}.contextCreatedAt`);
  const pageCreatedAt = timestamp(value.pageCreatedAt, `${label}.pageCreatedAt`);
  const closedAt = timestamp(value.closedAt, `${label}.closedAt`);
  requireCondition(launchedAt <= contextCreatedAt
    && contextCreatedAt <= pageCreatedAt
    && pageCreatedAt <= closedAt,
  `${label} chronology is invalid.`);
  requireCondition(value.contextCountBeforeClose === 1
    && value.pageCountBeforeClose === 1
    && value.disconnectedEventCount === 1
    && value.closedBeforeNextLaunch === true,
  `${label} does not prove a clean browser close.`);
  return { launchedAt, closedAt };
}

function validateGate(manifest, values, runId) {
  const gateArtifact = artifactValue(
    values,
    manifest.artifacts.forcedFeatureOffGate,
    'forced-feature-off gate',
  );
  exactKeys(gateArtifact, [
    'schemaVersion',
    'kind',
    'runId',
    'capturedAt',
    'capture',
    'validationRejections',
    'pageErrors',
    'browserLifecycle',
  ], 'forced-feature-off gate artifact');
  requireCondition(gateArtifact.schemaVersion === 1
    && gateArtifact.kind === 'first-instance-live-order-factorial-forced-feature-off-gate'
    && gateArtifact.runId === runId,
  'forced-feature-off gate artifact identity is invalid.');
  timestamp(gateArtifact.capturedAt, 'forced-feature-off gate capturedAt');
  requireSame(gateArtifact.pageErrors, [], 'forced-feature-off gate page errors');
  requireCleanGpuRecord(gateArtifact.capture, 'forced-feature-off gate capture');
  const recomputedRejections = validateLiveFirstInstanceForcedFeatureOffGate(
    gateArtifact.capture.gate,
  );
  requireSame(
    gateArtifact.validationRejections,
    recomputedRejections,
    'forced-feature-off gate validation rejections',
  );
  requireCondition(recomputedRejections.length === 0,
    'forced-feature-off gate validation failed.');
  const chronology = validateLifecycle(gateArtifact.browserLifecycle, {
    role: 'forced-feature-off-gate',
    sessionOrdinal: null,
    serial: 1,
    label: 'forced-feature-off gate browser lifecycle',
  });
  return { artifact: gateArtifact, chronology };
}

function validateShaderChallenges(challenges, spec, runId, usedNonces, label) {
  const expectedPhases = [
    ['preflight', 'render-parity'],
    ['preflight', 'main-validation'],
    ['timing-start', 'render-parity'],
    ['timing-start', 'main-validation'],
    ['postflight', 'render-parity'],
    ['postflight', 'main-validation'],
  ];
  requireCondition(Array.isArray(challenges) && challenges.length === expectedPhases.length,
    `${label} must contain six shader challenges.`);
  challenges.forEach((challenge, index) => {
    exactKeys(challenge, [
      'schemaVersion', 'kind', 'origin', 'runId', 'trialId', 'planIndex',
      'repetitionIndex', 'phase', 'role', 'captureOrdinal', 'challengeNonce',
    ], `${label}[${index}]`);
    const [phase, role] = expectedPhases[index];
    requireCondition(challenge.schemaVersion === 1
      && challenge.kind === 'live-first-instance-shader-observation-challenge'
      && challenge.origin === 'node-runner'
      && challenge.runId === runId
      && challenge.trialId === spec.trialId
      && challenge.planIndex === spec.planIndex
      && challenge.repetitionIndex === spec.repetitionIndex
      && challenge.phase === phase
      && challenge.role === role
      && challenge.captureOrdinal === index + 1
      && typeof challenge.challengeNonce === 'string'
      && /^[0-9a-f]{64}$/.test(challenge.challengeNonce),
    `${label}[${index}] identity is invalid.`);
    requireCondition(!usedNonces.has(challenge.challengeNonce),
      `${label}[${index}] reuses a challenge nonce.`);
    usedNonces.add(challenge.challengeNonce);
  });
}

function requireStableTrialIdentity(value, manifest, label, { workload = true } = {}) {
  requireSame(
    deriveFirstInstanceLiveOrderEnvironmentIdentity(value?.environment),
    manifest.environmentIdentity,
    `${label} environment identity`);
  requireCleanGpuRecord(value, label);
  if (workload) {
    requireSame(workloadIdentity(value?.workload), manifest.workloadIdentity,
      `${label} workload identity`);
  }
}

async function recomputeTrialArtifact({
  artifact,
  manifestRecord,
  expectedTrial,
  compatibilitySpec,
  manifest,
  usedNonces,
}) {
  const label = `factorial trial ${expectedTrial.planIndex}`;
  exactKeys(artifact, [
    'schemaVersion',
    'kind',
    'scope',
    'runId',
    'capturedAt',
    'factorialTrial',
    'compatibility',
    'auditContext',
    'configured',
    'shaderObservationChallenges',
    'preflight',
    'timingStart',
    'timing',
    'postflight',
    'protocol',
    'strictValidation',
    'trialSummary',
    'pageErrors',
  ], `${label} artifact`);
  requireCondition(artifact.schemaVersion === 1
    && artifact.kind === 'first-instance-live-order-factorial-trial-artifact'
    && artifact.runId === manifest.runId,
  `${label} artifact identity is invalid.`);
  requireCondition(
    artifact.scope === 'development diagnostic; not candidate evidence or a pass claim',
    `${label} scope is invalid.`,
  );
  timestamp(artifact.capturedAt, `${label} capturedAt`);
  requireSame(artifact.factorialTrial, expectedTrial, `${label} factorial plan record`);
  exactKeys(artifact.compatibility, ['mapping', 'validationSpec'],
    `${label} compatibility`);
  requireCondition(artifact.compatibility.mapping === COMPATIBILITY_MAPPING,
    `${label} compatibility mapping is invalid.`);
  requireSame(
    artifact.compatibility.validationSpec,
    compatibilitySpec,
    `${label} validation spec`,
  );
  const auditContext = expectedAuditContext(
    compatibilitySpec,
    expectedTrial,
    manifest.runId,
  );
  requireSame(artifact.auditContext, auditContext, `${label} audit context`);
  requireSame(artifact.pageErrors, [], `${label} page errors`);
  validateShaderChallenges(
    artifact.shaderObservationChallenges,
    compatibilitySpec,
    manifest.runId,
    usedNonces,
    `${label} shader challenges`,
  );

  const configured = artifact.configured;
  requireStableTrialIdentity(configured, manifest, `${label} configured`, {
    workload: false,
  });
  const config = configured?.selectedConfig;
  requireCondition(config?.strategyId === FIRST_INSTANCE_LIVE_CROSSOVER_MODE
    && config.objectCount === OBJECT_COUNT
    && config.bucketCount === BUCKET_COUNT
    && config.visibilityFraction === VISIBILITY_FRACTION
    && config.layout === LAYOUT
    && sameArray(config.lanePhysicalOrder, expectedTrial.laneConstructionOrder)
    && sameArray(config.plannedFirstComputeUseOrder, expectedTrial.firstComputeUseOrder)
    && sameArray(config.plannedRenderPipelinePrimeOrder,
      expectedTrial.renderPipelinePrimeOrder)
    && config.setupPrimeTopology === SETUP_PRIME_TOPOLOGY
    && config.timestampPreprimeLaneId === expectedTrial.timestampPreprimeLaneId
    && config.superblockOrientationOffset === expectedTrial.superblockOrientationOffset,
  `${label} configured setup differs from the factorial cell.`);
  requireCondition(configured.shaderEvidence?.pass === true,
    `${label} configured shader evidence failed.`);
  requireCondition(configured.timestampPoolPreprime?.kind
    === 'three-r185-timestamp-pool-preprime',
  `${label} lacks timestamp-pool preprime evidence.`);

  requireStableTrialIdentity(artifact.preflight, manifest, `${label} preflight`);
  requireSame(
    workloadIdentity(artifact.timingStart?.workload),
    manifest.workloadIdentity,
    `${label} timing-start workload identity`,
  );
  requireStableTrialIdentity(artifact.postflight, manifest, `${label} postflight`);
  requireStableTrialIdentity(artifact.timing, manifest, `${label} timing`, {
    workload: false,
  });
  requireSame(
    workloadIdentity(artifact.timingStart.workload),
    workloadIdentity(artifact.preflight.workload),
    `${label} preflight/timing workload`,
  );
  requireSame(
    workloadIdentity(artifact.postflight.workload),
    workloadIdentity(artifact.preflight.workload),
    `${label} preflight/postflight workload`,
  );
  requireCondition(Array.isArray(artifact.timing.rows)
    && artifact.timing.rows.length === FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
  `${label} does not retain exactly 480 rows.`);
  for (const [rowIndex, row] of artifact.timing.rows.entries()) {
    for (const [key, expectedValue] of Object.entries(auditContext)) {
      if (!isDeepStrictEqual(row?.[key], expectedValue)) {
        reject(`${label} row ${rowIndex}.${key} differs from the audit context.`);
      }
    }
  }
  requireSame(artifact.protocol, {
    schemaVersion: 2,
    warmupFrames: FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    measuredFrames: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    plannedScheduleSha256: liveFirstInstanceCrossoverScheduleSha256(
      compatibilitySpec.superblockOrientationOffset,
    ),
  }, `${label} protocol`);

  const recomputedStrictValidation = await validateLiveFirstInstanceTrialEvidence({
    spec: compatibilitySpec,
    environment: configured.environment,
    preflightValidation: artifact.preflight.validation,
    preflightRenderParity: artifact.preflight.renderParity,
    validation: artifact.timingStart.validation,
    renderParity: artifact.timing.renderParity,
    postflightValidation: artifact.postflight.validation,
    postflightRenderParity: artifact.postflight.renderParity,
    shaderObservationChallenges: artifact.shaderObservationChallenges,
    rows: artifact.timing.rows,
    summary: artifact.timing.summary,
    protocol: artifact.protocol,
    scenarioManifest: artifact.preflight.workload.scenario,
    geometryManifest: artifact.preflight.workload.geometryFixtures,
  });
  requireSame(
    artifact.strictValidation,
    recomputedStrictValidation,
    `${label} retained strict validation`,
  );
  requireCondition(recomputedStrictValidation.pass === true,
    `${label} strict evidence validation failed.`);

  const recomputedTrialSummary = summarizeLiveFirstInstanceTrialRows(
    artifact.timing.rows,
    compatibilitySpec,
    manifest.runId,
  );
  requireSame(
    artifact.trialSummary,
    recomputedTrialSummary,
    `${label} retained trial summary`,
  );
  requireCondition(manifestRecord.strictSemanticSha256
    === recomputedStrictValidation.semanticSha256,
  `${label} manifest strict semantic digest is invalid.`);
  const completionInvariant = artifact.timing.summary?.completionInvariant;
  const expectedTimingStartCacheState = {
    totalPipelineCacheEntries:
      completionInvariant?.totalPipelineCacheEntriesAtTimingStart ?? null,
    computePipelineCacheEntries:
      completionInvariant?.computePipelineCacheEntriesAtTimingStart ?? null,
    computeProgramEntries:
      completionInvariant?.computeProgramEntriesAtTimingStart ?? null,
    rendererMemory: completionInvariant?.rendererMemoryAtTimingStart ?? null,
  };
  exactKeys(manifestRecord.timingStartCacheState, [
    'totalPipelineCacheEntries',
    'computePipelineCacheEntries',
    'computeProgramEntries',
    'rendererMemory',
  ], `${label} manifest timing-start cache state`);
  requireCondition([
    manifestRecord.timingStartCacheState.totalPipelineCacheEntries,
    manifestRecord.timingStartCacheState.computePipelineCacheEntries,
    manifestRecord.timingStartCacheState.computeProgramEntries,
  ].every((value) => Number.isSafeInteger(value) && value >= 0)
    && manifestRecord.timingStartCacheState.rendererMemory !== null
    && typeof manifestRecord.timingStartCacheState.rendererMemory === 'object'
    && !Array.isArray(manifestRecord.timingStartCacheState.rendererMemory),
  `${label} manifest timing-start cache state is invalid.`);
  requireSame(
    manifestRecord.timingStartCacheState,
    expectedTimingStartCacheState,
    `${label} timing-start cache state`,
  );
  return {
    ...expectedTrial,
    trialSummary: recomputedTrialSummary,
  };
}

export function summarizeFirstInstanceLiveOrderTimingStartCacheStates(
  manifestTrials,
  executionPlan,
) {
  const sessionIndexes = [...new Set(executionPlan.map(({ sessionIndex }) => sessionIndex))];
  return sessionIndexes.map((sessionIndex) => {
    const records = manifestTrials.filter((trial) => trial.sessionIndex === sessionIndex);
    const orderedStates = records.map((trial) => ({
      factorialPlanIndex: trial.factorialPlanIndex,
      factorialTrialId: trial.factorialTrialId,
      sessionTrialIndex: trial.sessionTrialIndex,
      factorialCellId: trial.factorialCellId,
      timingStartCacheState: trial.timingStartCacheState,
    }));
    const uniqueStates = [];
    for (const record of orderedStates) {
      let group = uniqueStates.find((candidate) => isDeepStrictEqual(
        candidate.timingStartCacheState,
        record.timingStartCacheState,
      ));
      if (group === undefined) {
        group = {
          timingStartCacheState: record.timingStartCacheState,
          occurrenceCount: 0,
          factorialPlanIndexes: [],
        };
        uniqueStates.push(group);
      }
      group.occurrenceCount += 1;
      group.factorialPlanIndexes.push(record.factorialPlanIndex);
    }
    return {
      sessionIndex,
      trialCount: orderedStates.length,
      uniqueStateCount: uniqueStates.length,
      orderedStates,
      uniqueStates,
    };
  });
}

function validateManifestTrialRecord(recordValue, expectedTrial, compatibilitySpec, index) {
  const label = `manifest.artifacts.trials[${index}]`;
  const record = recordValue;
  requireCondition(record.factorialPlanIndex === expectedTrial.planIndex
    && record.factorialTrialId === expectedTrial.trialId
    && record.sessionIndex === expectedTrial.sessionIndex
    && record.sessionTrialIndex === expectedTrial.sessionTrialIndex
    && record.factorialCellId === expectedTrial.factorialCellId
    && record.compatibilityPlanIndex === compatibilitySpec.planIndex
    && record.compatibilityRepetitionIndex === compatibilitySpec.repetitionIndex
    && SHA256_PATTERN.test(record.strictSemanticSha256),
  `${label} identity is invalid.`);
  const expectedFilename = `trials/trial-${String(expectedTrial.planIndex).padStart(2, '0')}`
    + `-s${expectedTrial.sessionOrdinal}`
    + `-t${String(expectedTrial.sessionTrialOrdinal).padStart(2, '0')}`
    + `-${expectedTrial.factorialCellId}.json.br`;
  requireCondition(record.artifact.path === expectedFilename,
    `${label} path differs from the frozen filename.`);
}

function validateSessionArtifacts(manifest, values, executionPlan, mode) {
  requireCondition(manifest.artifacts.sessions.length
    === mode.sessionCount,
  `manifest must declare exactly ${mode.sessionCount} session artifacts.`);
  const expectedSessionIndexes = [...new Set(
    executionPlan.map(({ sessionIndex }) => sessionIndex),
  )];
  requireCondition(expectedSessionIndexes.length === mode.sessionCount,
    'execution plan session count is invalid.');
  const sessions = manifest.artifacts.sessions.map((descriptor, executionSessionIndex) => {
    const sessionIndex = expectedSessionIndexes[executionSessionIndex];
    requireCondition(descriptor.path === `session-${sessionIndex + 1}.json`,
      `session ${sessionIndex} descriptor path is invalid.`);
    const session = artifactValue(values, descriptor, `session ${sessionIndex}`);
    exactKeys(session, [
      'schemaVersion', 'kind', 'runId', 'sessionIndex', 'sessionOrdinal',
      'browserLifecycle', 'endingEnvironment', 'trials',
    ], `session ${sessionIndex} artifact`);
    requireCondition(session.schemaVersion === 1
      && session.kind === 'first-instance-live-order-factorial-session-completion'
      && session.runId === manifest.runId
      && session.sessionIndex === sessionIndex
      && session.sessionOrdinal === sessionIndex + 1,
    `session ${sessionIndex} identity is invalid.`);
    const expectedTrials = manifest.artifacts.trials.filter(
      (trial) => trial.sessionIndex === sessionIndex,
    );
    requireCondition(expectedTrials.length === mode.trialsPerSession,
    `session ${sessionIndex} manifest trial count is invalid.`);
    requireSame(session.trials, expectedTrials, `session ${sessionIndex} trial records`);
    requireSame(
      expectedTrials.map(({ factorialPlanIndex }) => factorialPlanIndex),
      executionPlan.filter((trial) => trial.sessionIndex === sessionIndex)
        .map(({ planIndex }) => planIndex),
      `session ${sessionIndex} trial chronology`,
    );
    requireSame(deriveFirstInstanceLiveOrderEnvironmentIdentity(session.endingEnvironment),
      manifest.environmentIdentity,
      `session ${sessionIndex} ending environment identity`);
    requireCleanEnvironment(session.endingEnvironment,
      `session ${sessionIndex} ending environment`);
    const chronology = validateLifecycle(session.browserLifecycle, {
      role: 'factorial-session',
      sessionOrdinal: sessionIndex + 1,
      serial: executionSessionIndex + 2,
      label: `session ${sessionIndex} browser lifecycle`,
    });
    return { artifact: session, chronology };
  });
  return sessions;
}

function validateProcessChronology(manifest, gate, sessions) {
  exactKeys(manifest.processLifecycle, ['gate', 'sessions'], 'manifest.processLifecycle');
  requireSame(manifest.processLifecycle.gate, gate.artifact.browserLifecycle,
    'manifest gate lifecycle');
  requireSame(
    manifest.processLifecycle.sessions,
    sessions.map(({ artifact }) => artifact.browserLifecycle),
    'manifest session lifecycles',
  );
  requireCondition(gate.chronology.closedAt <= sessions[0].chronology.launchedAt,
    'gate browser overlaps factorial session 1.');
  for (let index = 1; index < sessions.length; index += 1) {
    requireCondition(
      sessions[index - 1].chronology.closedAt <= sessions[index].chronology.launchedAt,
      'factorial session browsers overlap.',
    );
  }
}

/** Recompute and compare the retained descriptive factorial summary. */
export function verifyRetainedFirstInstanceLiveOrderFactorialSummary(
  analysisRecords,
  retainedSummary,
  runId,
) {
  const recomputed = summarizeFirstInstanceLiveOrderFactorial(analysisRecords, runId);
  requireSame(retainedSummary, recomputed, 'retained diagnostic summary');
  return recomputed;
}

export function verifyRetainedFirstInstanceLiveOrderSmokeSummary(
  analysisRecord,
  retainedSummary,
  runId,
) {
  const expected = {
    schemaVersion: 1,
    kind: 'first-instance-live-order-factorial-smoke-summary',
    decision: SMOKE_ONLY_DECISION,
    analysisInvoked: false,
    runId,
    factorialPlanIndex: analysisRecord.planIndex,
    factorialCellId: analysisRecord.factorialCellId,
    trialSummary: analysisRecord.trialSummary,
  };
  requireSame(retainedSummary, expected, 'retained smoke summary');
  return expected;
}

/**
 * Independently verify one complete retained order-factorial development run.
 */
export async function verifyFirstInstanceLiveOrderFactorial(
  runDirectory,
  { projectRoot = PROJECT_ROOT } = {},
) {
  nonemptyString(runDirectory, 'runDirectory');
  const resolvedDirectory = path.resolve(runDirectory);
  const manifest = await readFinalManifest(resolvedDirectory);
  const mode = classifyFirstInstanceLiveOrderManifestEnvelope(manifest);
  const values = await readFirstInstanceLiveOrderFactorialDeclaredArtifacts(
    resolvedDirectory,
    manifest,
  );
  const runId = manifest.runId;
  const expectedPlan = buildFirstInstanceLiveOrderFactorialPlan({ runId });
  const candidatePlan = buildFirstInstanceLiveCrossoverPlan({
    runId,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  });
  const compatibilityHighPlan = candidatePlan.filter((trial) => (
    trial.repetitionIndex < 4 && trial.visibilityFraction === VISIBILITY_FRACTION
  ));
  requireCondition(compatibilityHighPlan.length === 4,
    'candidate compatibility basis is not the frozen four-cell map.');
  const compatibilitySpecs = expectedPlan.map((trial) => (
    candidateCompatibilitySpec(trial, compatibilityHighPlan, runId)
  ));
  const smokeTrial = expectedPlan.find((trial) => (
    trial.sessionIndex === 0
      && !sameArray(trial.firstComputeUseOrder, trial.renderPipelinePrimeOrder)
  ));
  requireCondition(smokeTrial !== undefined, 'frozen smoke trial is unavailable.');
  const executionPlan = mode.smokeMode ? [smokeTrial] : expectedPlan;

  requireCondition(manifest.artifacts.plan.path === 'plan.json',
    'plan descriptor path is invalid.');
  requireCondition(manifest.artifacts.forcedFeatureOffGate.path
    === 'forced-feature-off-gate.json.br',
  'forced-feature-off gate descriptor path is invalid.');
  requireCondition(manifest.artifacts.executionIdentityStart.path
    === 'execution-identity-start.json',
  'execution identity start descriptor path is invalid.');
  requireCondition(manifest.artifacts.executionIdentityEnd.path
    === 'execution-identity-end.json',
  'execution identity end descriptor path is invalid.');
  requireCondition(manifest.artifacts.diagnosticSummary.path === 'diagnostic-summary.json',
    'diagnostic summary descriptor path is invalid.');
  requireCondition(manifest.artifacts.viteRuntimeAudit.path === 'vite-runtime-audit.json',
    'Vite audit descriptor path is invalid.');
  const planArtifact = artifactValue(values, manifest.artifacts.plan, 'plan artifact');
  validatePlanArtifact(
    planArtifact,
    expectedPlan,
    executionPlan,
    compatibilitySpecs,
    runId,
    mode.smokeMode,
  );
  validateExecutionIdentity(manifest, values, runId);

  const gate = validateGate(manifest, values, runId);
  requireSame(deriveFirstInstanceLiveOrderEnvironmentIdentity(
    gate.artifact.capture.environment,
  ),
    manifest.environmentIdentity,
    'gate environment identity');
  requireSame(workloadIdentity(gate.artifact.capture.workload),
    manifest.workloadIdentity,
    'gate workload identity');
  const sessions = validateSessionArtifacts(manifest, values, executionPlan, mode);
  validateProcessChronology(manifest, gate, sessions);

  requireCondition(manifest.artifacts.trials.length === mode.executedTrialCount,
    `manifest must declare exactly ${mode.executedTrialCount} executed trials.`);
  const usedNonces = new Set();
  const analysisRecords = [];
  for (let index = 0; index < executionPlan.length; index += 1) {
    const expectedTrial = executionPlan[index];
    const compatibilitySpec = compatibilitySpecs[expectedTrial.planIndex];
    const manifestRecord = manifest.artifacts.trials[index];
    validateManifestTrialRecord(manifestRecord, expectedTrial, compatibilitySpec, index);
    const artifact = artifactValue(
      values,
      manifestRecord.artifact,
      `factorial trial ${expectedTrial.planIndex}`,
    );
    analysisRecords.push(await recomputeTrialArtifact({
      artifact,
      manifestRecord,
      expectedTrial,
      compatibilitySpec,
      manifest,
      usedNonces,
    }));
  }
  requireCondition(usedNonces.size === mode.executedTrialCount * 6,
    'global shader challenge nonce count is invalid.');

  const retainedSummary = artifactValue(
    values,
    manifest.artifacts.diagnosticSummary,
    'diagnostic summary',
  );
  const retainedAnalysis = mode.smokeMode
    ? verifyRetainedFirstInstanceLiveOrderSmokeSummary(
      analysisRecords[0],
      retainedSummary,
      runId,
    )
    : verifyRetainedFirstInstanceLiveOrderFactorialSummary(
      analysisRecords,
      retainedSummary,
      runId,
    );
  const viteAudit = artifactValue(
    values,
    manifest.artifacts.viteRuntimeAudit,
    'Vite runtime audit',
  );
  const viteValidation = await validateCandidateViteRuntimeAudit(
    viteAudit,
    path.resolve(projectRoot),
  );
  requireCondition(viteValidation.pass === true,
    `Vite runtime audit failed: ${viteValidation.reasons.join('; ')}`);
  requireCondition(viteAudit.entryHtml?.successfulResponseCount === mode.entryDocumentCount,
    'Vite runtime audit has the wrong entry-document count.');

  const timingStartCacheStateBySession = summarizeFirstInstanceLiveOrderTimingStartCacheStates(
    manifest.artifacts.trials,
    executionPlan,
  );
  if (mode.smokeMode) {
    return {
      schemaVersion: 1,
      kind: 'first-instance-live-order-smoke-independent-consistency',
      status: 'consistent',
      scope: SMOKE_ONLY_DECISION,
      decision: SMOKE_ONLY_DECISION,
      runDirectory: resolvedDirectory,
      runId,
      artifactCount: values.size,
      sessionCount: sessions.length,
      trialCount: analysisRecords.length,
      shaderChallengeCount: usedNonces.size,
      artifactIntegrityVerified: true,
      strictTrialEvidenceRecomputed: true,
      trialSummaryRecomputed: true,
      smokeSummaryRecomputed: true,
      factorialAnalysisInvoked: retainedAnalysis.analysisInvoked,
      factorialDiagnosticVerified: false,
      sourceProvenanceMatched: true,
      executionDependencyClosureMatched: true,
      viteRuntimeAuditVerified: true,
      timingStartCacheStateBySession,
      authenticityVerified: false,
    };
  }

  return {
    schemaVersion: 1,
    kind: 'first-instance-live-order-factorial-independent-verification',
    status: 'consistent',
    scope: 'complete development diagnostic; no candidate or deployment claim',
    decision: retainedAnalysis.decision,
    runDirectory: resolvedDirectory,
    runId,
    artifactCount: values.size,
    sessionCount: sessions.length,
    trialCount: analysisRecords.length,
    shaderChallengeCount: usedNonces.size,
    artifactIntegrityVerified: true,
    strictTrialEvidenceRecomputed: true,
    diagnosticSummaryRecomputed: true,
    sourceProvenanceMatched: true,
    executionDependencyClosureMatched: true,
    viteRuntimeAuditVerified: true,
    timingStartCacheStateBySession,
    authenticityVerified: false,
  };
}

export function serializeFirstInstanceLiveOrderFactorialVerification(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

export const verifyFirstInstanceLiveOrderFactorialRun =
  verifyFirstInstanceLiveOrderFactorial;

const invokedPath = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : null;
if (invokedPath === import.meta.url) {
  const args = process.argv.slice(2);
  if (args.length !== 1) {
    process.stderr.write(
      'Usage: node analysis/verify-first-instance-live-order-factorial.mjs <run-directory>\n',
    );
    process.exitCode = 2;
  } else {
    try {
      const result = await verifyFirstInstanceLiveOrderFactorial(args[0]);
      process.stdout.write(serializeFirstInstanceLiveOrderFactorialVerification(result));
    } catch (error) {
      process.stderr.write(`${error?.stack ?? error?.message ?? String(error)}\n`);
      process.exitCode = 1;
    }
  }
}
