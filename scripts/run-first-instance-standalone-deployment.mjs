import { createHash, randomBytes } from 'node:crypto';
import {
  access,
  mkdir,
  readFile,
  writeFile,
} from 'node:fs/promises';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import process from 'node:process';
import { setTimeout as delay } from 'node:timers/promises';
import { promisify } from 'node:util';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  brotliCompress,
  constants as zlibConstants,
} from 'node:zlib';

import { chromium } from 'playwright-core';
import { createServer } from 'vite';

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
  firstInstanceStandaloneBootSearch,
} from '../src/benchmark/first-instance-standalone-boot.js';
import {
  CANDIDATE_VITE_RUNTIME_POLICY_ID,
  collectExecutionDependencyClosure,
  createCandidateViteRuntimeGuard,
  executionDependencyClosuresMatch,
} from './execution-dependency-closure.mjs';
import {
  validateLiveFirstInstanceForcedFeatureOffGate,
} from './live-first-instance-evidence-validation.mjs';
import {
  evaluateLiveFirstInstanceAdapterTelemetryAssociation,
} from './live-first-instance-adapter-telemetry-association.mjs';
import {
  NvidiaTelemetryRecorder,
  compareComputeProcessIdentitySets,
} from './nvidia-telemetry.mjs';
import {
  collectSourceProvenance,
  sourceProvenanceMatches,
} from './source-provenance.mjs';

const brotliCompressAsync = promisify(brotliCompress);

export const STANDALONE_EXECUTION_MODES = Object.freeze({
  FULL: 'full-candidate',
  SMOKE: 'smoke-excluded',
});
export const STANDALONE_MODE_ID = 'first-instance-live-standalone-deployment';
export const STANDALONE_SHADER_CHALLENGE_KIND =
  'live-first-instance-standalone-shader-observation-challenge';
export const STANDALONE_POST_DISCONNECT_DELAY_MS = 2_000;

const OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;
const LAYOUT = 'baseline';
const SCENARIO_SEED = 0xb1ad_2026;
const MAXIMUM_CPU_TIMER_QUANTUM_MS = 0.01;
const MAXIMUM_TIMESTAMP_QUANTUM_NS = 1_000;
const BROWSER_OPERATION_TIMEOUT_MS = 180_000;
const BROWSER_CLOSE_TIMEOUT_MS = 60_000;
const SERVER_OPERATION_TIMEOUT_MS = 120_000;
const BROWSER_ARGS = Object.freeze([
  '--enable-unsafe-webgpu',
  '--enable-webgpu-developer-features',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
]);
const ISOLATION_HEADERS = Object.freeze({
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
});
const SHADER_CAPTURE_ROLES = Object.freeze([
  Object.freeze(['preflight', 'render-parity']),
  Object.freeze(['preflight', 'main-validation']),
  Object.freeze(['timing-start', 'render-parity']),
  Object.freeze(['timing-start', 'main-validation']),
  Object.freeze(['postflight', 'render-parity']),
  Object.freeze(['postflight', 'main-validation']),
]);
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const NONCE_PATTERN = /^[0-9a-f]{64}$/;

function sameJson(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function sameArray(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

function sha256Bytes(value) {
  return createHash('sha256').update(value).digest('hex');
}

function canonicalJsonSha256(value) {
  return sha256Bytes(Buffer.from(JSON.stringify(value), 'utf8'));
}

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function requireCondition(condition, message, evidence = null) {
  if (condition) return;
  const serialized = evidence === null ? '' : JSON.stringify(evidence);
  const detail = serialized.length === 0
    ? ''
    : `: ${serialized.length <= 4_000
      ? serialized
      : `${serialized.slice(0, 4_000)}...[${serialized.length} characters]`}`;
  throw new Error(`${message}${detail}`);
}

async function withDeadline(promise, timeoutMs, label) {
  let timeoutId;
  const deadline = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(new Error(`${label} exceeded its ${timeoutMs} ms deadline.`));
    }, timeoutMs);
  });
  try {
    return await Promise.race([promise, deadline]);
  } finally {
    clearTimeout(timeoutId);
  }
}

function clone(value) {
  return structuredClone(value);
}

function frozenSelectionRecord(plan, matrixIndex, sessionIds, trialIds) {
  return Object.freeze({
    matrix: plan.matrices[matrixIndex],
    sessions: Object.freeze(sessionIds.map(
      (sessionId) => plan.sessions.find((session) => session.sessionId === sessionId),
    )),
    trials: Object.freeze(trialIds.map(
      (trialId) => plan.trials.find((trial) => trial.trialId === trialId),
    )),
  });
}

/**
 * Select the exact capture body without changing the frozen plan. Smoke mode
 * retains both visibility trials in the first portable and feature sessions.
 */
export function selectStandaloneDeploymentExecution(plan, executionMode) {
  if (executionMode !== STANDALONE_EXECUTION_MODES.FULL
    && executionMode !== STANDALONE_EXECUTION_MODES.SMOKE) {
    throw new RangeError('Unknown standalone deployment execution mode.');
  }
  validateFirstInstanceStandaloneDeploymentPlan(plan, { runId: plan?.runId });
  if (executionMode === STANDALONE_EXECUTION_MODES.SMOKE) {
    const sessions = plan.sessions.slice(0, 2);
    const sessionIds = new Set(sessions.map((session) => session.sessionId));
    const trials = plan.trials.filter((trial) => sessionIds.has(trial.sessionId));
    return Object.freeze({
      executionMode,
      analysisEligible: false,
      matrices: Object.freeze([
        frozenSelectionRecord(
          plan,
          0,
          sessions.map((session) => session.sessionId),
          trials.map((trial) => trial.trialId),
        ),
      ]),
      sessions: Object.freeze(sessions),
      trials: Object.freeze(trials),
    });
  }
  return Object.freeze({
    executionMode,
    analysisEligible: true,
    matrices: Object.freeze(plan.matrices.map((matrix) => frozenSelectionRecord(
      plan,
      matrix.matrixIndex,
      matrix.sessionIds,
      matrix.trialIds,
    ))),
    sessions: Object.freeze([...plan.sessions]),
    trials: Object.freeze([...plan.trials]),
  });
}

function defaultNonceFactory() {
  return randomBytes(32).toString('hex');
}

/**
 * A strategy survives the two visibility trials, so capture ordinals are 1-6
 * for trial one and 7-12 for trial two. The caller owns the global nonce set.
 */
export function createStandaloneShaderObservationChallenges({
  canonicalTrial,
  runId,
  firstCaptureOrdinal,
  usedNonces,
  nonceFactory = defaultNonceFactory,
}) {
  if (!canonicalTrial || canonicalTrial.kind !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_KIND) {
    throw new TypeError('A canonical standalone deployment trial is required.');
  }
  if (runId !== canonicalTrialRunId(canonicalTrial)) {
    throw new Error('Shader challenge runId differs from the canonical trial namespace.');
  }
  if (!Number.isSafeInteger(firstCaptureOrdinal) || firstCaptureOrdinal < 1) {
    throw new RangeError('firstCaptureOrdinal must be a positive safe integer.');
  }
  if (!(usedNonces instanceof Set)) {
    throw new TypeError('usedNonces must be a Set.');
  }
  return Object.freeze(SHADER_CAPTURE_ROLES.map(([phase, role], index) => {
    const challengeNonce = nonceFactory();
    if (!NONCE_PATTERN.test(challengeNonce) || usedNonces.has(challengeNonce)) {
      throw new Error('Shader challenge nonce must be fresh lowercase 256-bit hex.');
    }
    usedNonces.add(challengeNonce);
    return Object.freeze({
      schemaVersion: 1,
      kind: STANDALONE_SHADER_CHALLENGE_KIND,
      origin: 'node-runner',
      runId,
      trialId: canonicalTrial.trialId,
      planIndex: canonicalTrial.planIndex,
      repetitionIndex: canonicalTrial.globalQuartetIndex,
      phase,
      role,
      captureOrdinal: firstCaptureOrdinal + index,
      challengeNonce,
    });
  }));
}

function parseCanonicalArray(value, label, reasons) {
  if (typeof value !== 'string') {
    reasons.push(`${label} is not JSON text`);
    return [];
  }
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed) || JSON.stringify(parsed) !== value) {
      reasons.push(`${label} is not a canonical JSON array`);
      return [];
    }
    return parsed;
  } catch {
    reasons.push(`${label} is not valid JSON`);
    return [];
  }
}

/**
 * Runner-local fail-closed timing validation. Cross-session shader and output
 * comparison intentionally belongs to the independent verifier.
 */
export function validateStandaloneTimingRows(rows, canonicalTrial, summary = null) {
  const reasons = [];
  if (!canonicalTrial || canonicalTrial.kind !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_KIND) {
    return Object.freeze({ pass: false, reasons: Object.freeze(['canonical trial is invalid']) });
  }
  if (!Array.isArray(rows)) {
    return Object.freeze({ pass: false, reasons: Object.freeze(['rows are missing']) });
  }
  if (rows.length !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES) {
    reasons.push('measured row count differs from 480');
  }
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const label = `row ${index}`;
    const exactContext = row?.runId === canonicalTrialRunId(canonicalTrial)
      && Object.entries(canonicalTrial).every(
        ([field, expected]) => sameJson(row?.[field], expected),
      );
    if (!exactContext) reasons.push(`${label} canonical context differs`);
    if (row?.frameIndex !== index || row?.phaseFrameIndex !== index) {
      reasons.push(`${label} frame index differs`);
    }
    if (row?.measuredBlockIndex !== Math.floor(index / STANDALONE_BLOCK_SIZE)
      || row?.withinBlockPosition !== index % STANDALONE_BLOCK_SIZE) {
      reasons.push(`${label} block position differs`);
    }
    if (row?.laneId !== canonicalTrial.assignedLaneId
      || row?.submittedComputeLaneId !== canonicalTrial.assignedLaneId
      || row?.targetVisibilityFraction !== canonicalTrial.visibilityFraction) {
      reasons.push(`${label} selected lane or visibility differs`);
    }
    if (row?.commandSegmentIndex !== 0
      || row?.commandRecordBase !== 0
      || row?.commandByteBase !== 0
      || row?.commandByteOffset !== 0
      || row?.commandBufferRecordCount !== BUCKET_COUNT) {
      reasons.push(`${label} command buffer is not the fixed zero-offset layout`);
    }
    if (row?.configuredDrawCommands !== BUCKET_COUNT
      || row?.configuredRenderObjects !== 1
      || row?.configuredComputeDispatches !== 2
      || row?.configuredComputeSubmissions !== 1
      || row?.objectCount !== OBJECT_COUNT
      || row?.bucketCount !== BUCKET_COUNT) {
      reasons.push(`${label} workload cardinality differs`);
    }
    if (![row?.gpuComputeMs, row?.gpuRenderMs, row?.gpuPassTotalMs]
      .every((value) => Number.isFinite(value) && value > 0)
      || row.gpuPassTotalMs !== row.gpuComputeMs + row.gpuRenderMs) {
      reasons.push(`${label} GPU duration is invalid`);
    }
    if (![row?.cpuCommonUpdateMs, row?.cpuComputeSubmitMs, row?.cpuRenderSubmitMs,
      row?.cpuSubmitTotalMs, row?.cpuFrameBodyMs]
      .every((value) => Number.isFinite(value) && value >= 0)
      || row.cpuSubmitTotalMs !== row.cpuComputeSubmitMs + row.cpuRenderSubmitMs) {
      reasons.push(`${label} CPU timing is invalid`);
    }
    const computeUids = parseCanonicalArray(
      row?.gpuComputeTimestampUids,
      `${label} compute timestamp UIDs`,
      reasons,
    );
    const renderUids = parseCanonicalArray(
      row?.gpuRenderTimestampUids,
      `${label} render timestamp UIDs`,
      reasons,
    );
    const computeRecords = parseCanonicalArray(
      row?.gpuComputeTimestampRecords,
      `${label} compute timestamp records`,
      reasons,
    );
    const renderRecords = parseCanonicalArray(
      row?.gpuRenderTimestampRecords,
      `${label} render timestamp records`,
      reasons,
    );
    if (row?.gpuComputeTimestampUidCount !== 1
      || row?.gpuRenderTimestampUidCount !== 1
      || row?.gpuComputeTimestampDurationValid !== true
      || row?.gpuRenderTimestampDurationValid !== true
      || computeUids.length !== 1
      || renderUids.length !== 1
      || computeRecords.length !== 1
      || renderRecords.length !== 1
      || computeRecords[0]?.uid !== computeUids[0]
      || renderRecords[0]?.uid !== renderUids[0]
      || computeRecords[0]?.durationMs !== row?.gpuComputeMs
      || renderRecords[0]?.durationMs !== row?.gpuRenderMs) {
      reasons.push(`${label} timestamp attribution is invalid`);
    }
  }
  if (summary !== null) {
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
    if (summary?.rowCount !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES
      || summary?.warmupRowCount !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES
      || summary?.expectedRenderTimestampUidCount !== 1
      || summary?.expectedComputeTimestampUidCount !== 1
      || summary?.timestampAvailable !== true
      || summary?.accepted !== true
      || summary?.completionInvariant?.pass !== true
      || summary?.renderTimestampPoolQualityValid !== true
      || summary?.computeTimestampPoolQualityValid !== true
      || summary?.warmupRenderTimestampPoolQualityValid !== true
      || summary?.warmupComputeTimestampPoolQualityValid !== true
      || summary?.warmupTimestampFrameCountValid !== true
      || summary?.measurementTimestampFrameCountValid !== true
      || !Number.isFinite(summary?.quantumNs)
      || summary.quantumNs <= 0
      || summary.quantumNs > MAXIMUM_TIMESTAMP_QUANTUM_NS
      || zeroFields.some((field) => summary?.[field] !== 0)) {
      reasons.push('trial summary or completion invariant is invalid');
    }
    for (const phase of ['warmup', 'measurement']) {
      const expectedCount = phase === 'warmup'
        ? FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES
        : FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES;
      for (const type of ['compute', 'render']) {
        const pool = summary?.timestampPhases?.[phase]?.pools?.[type];
        if (pool?.included !== true
          || pool?.frames?.length !== expectedCount
          || pool?.uidRecords?.length !== expectedCount
          || !Number.isFinite(pool?.resolution?.quantumNs)
          || pool.resolution.quantumNs <= 0
          || pool.resolution.quantumNs > MAXIMUM_TIMESTAMP_QUANTUM_NS) {
          reasons.push(`${phase} ${type} timestamp phase is invalid`);
        }
      }
    }
  }
  return Object.freeze({
    schemaVersion: 1,
    kind: 'first-instance-standalone-runner-timing-validation',
    pass: reasons.length === 0,
    reasons: Object.freeze([...new Set(reasons)]),
  });
}

const STANDALONE_BLOCK_SIZE = FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE;

export function createStandaloneTrialArtifact({
  canonicalTrial,
  rows,
  executionMode,
  planSha256,
  body,
}) {
  if (!canonicalTrial || canonicalTrial.kind !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_KIND) {
    throw new TypeError('A canonical standalone trial is required.');
  }
  if (!Array.isArray(rows)) throw new TypeError('Standalone artifact rows must be an array.');
  if (!Object.values(STANDALONE_EXECUTION_MODES).includes(executionMode)) {
    throw new RangeError('Standalone artifact execution mode is invalid.');
  }
  if (!SHA256_PATTERN.test(planSha256)) {
    throw new RangeError('Standalone artifact plan SHA-256 is invalid.');
  }
  if (body === null || typeof body !== 'object' || Array.isArray(body)) {
    throw new TypeError('Standalone artifact body must be an object.');
  }
  const reservedKeys = new Set([
    ...Object.keys(canonicalTrial),
    'rows',
    'artifactSchemaVersion',
    'artifactKind',
    'executionMode',
    'analysisEligible',
    'planSha256',
    'canonicalTrial',
  ]);
  const collision = Object.keys(body).find((key) => reservedKeys.has(key));
  if (collision !== undefined) {
    throw new Error(`Standalone artifact body may not replace reserved field ${collision}.`);
  }
  return {
    ...canonicalTrial,
    rows,
    artifactSchemaVersion: 1,
    artifactKind: 'first-instance-standalone-deployment-trial-artifact',
    executionMode,
    analysisEligible: executionMode === STANDALONE_EXECUTION_MODES.FULL,
    planSha256,
    canonicalTrial: clone(canonicalTrial),
    ...body,
  };
}

export function validateStandaloneBrowserLifecycleChain(records, {
  requireTerminalDelay = false,
} = {}) {
  const reasons = [];
  if (!Array.isArray(records) || records.length === 0) {
    return Object.freeze({ pass: false, reasons: Object.freeze(['browser records are missing']) });
  }
  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (record?.browserInstanceSerial !== index + 1
      || record?.launchApi !== 'chromium.launch'
      || record?.persistentContext === true
      || record?.profilePolicy !== 'fresh-playwright-temporary-profile-per-process'
      || record?.contextCountBeforeClose !== 1
      || record?.pageCountBeforeClose !== 1
      || record?.disconnectedEventCount !== 1
      || !Number.isFinite(record?.disconnectedRunElapsedMs)) {
      reasons.push(`browser record ${index} violates fresh-process/profile lifecycle`);
    }
    if (index > 0) {
      const previous = records[index - 1];
      if (record?.priorBrowserInstanceSerial !== previous?.browserInstanceSerial
        || record?.priorBrowserDisconnectedAt !== previous?.disconnectedAt
        || !Number.isFinite(record?.previousDisconnectToLaunchGapMs)
        || record.previousDisconnectToLaunchGapMs < STANDALONE_POST_DISCONNECT_DELAY_MS
        || previous?.postDisconnectDelay?.requestedMs
          !== STANDALONE_POST_DISCONNECT_DELAY_MS
        || previous?.postDisconnectDelay?.elapsedMs
          < STANDALONE_POST_DISCONNECT_DELAY_MS) {
        reasons.push(`browser record ${index} lacks the fixed post-disconnect gap`);
      }
    }
  }
  if (requireTerminalDelay) {
    const terminal = records.at(-1)?.postDisconnectDelay;
    if (terminal?.requestedMs !== STANDALONE_POST_DISCONNECT_DELAY_MS
      || terminal?.elapsedMs < STANDALONE_POST_DISCONNECT_DELAY_MS) {
      reasons.push('terminal browser lacks its recorded post-disconnect delay');
    }
  }
  return Object.freeze({
    schemaVersion: 1,
    kind: 'first-instance-standalone-browser-lifecycle-chain-validation',
    pass: reasons.length === 0,
    reasons: Object.freeze(reasons),
  });
}

/**
 * A complete full capture remains analyzable when an observed environment,
 * telemetry, or process-set gate fails. Keep that outcome as evidence for the
 * independent verifier instead of converting it into an incomplete run.
 */
export function standaloneFullEnvironmentGatesPassed(globalIdentity, matrixRecords) {
  if (globalIdentity === null
    || typeof globalIdentity !== 'object'
    || Array.isArray(globalIdentity)) {
    throw new TypeError('Standalone global identity report must be an object.');
  }
  if (typeof globalIdentity.environmentIdentityExact !== 'boolean') {
    throw new TypeError('Standalone global environment gate must be boolean.');
  }
  if (!Array.isArray(matrixRecords)) {
    throw new TypeError('Standalone matrix records must be an array.');
  }
  if (matrixRecords.length !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_COUNT) {
    throw new RangeError('Standalone full evidence requires exactly two matrix records.');
  }
  for (const [index, record] of matrixRecords.entries()) {
    if (record === null
      || typeof record !== 'object'
      || Array.isArray(record)
      || typeof record?.identity?.environmentIdentityExact !== 'boolean'
      || typeof record?.telemetry?.eligibility?.pass !== 'boolean') {
      throw new TypeError(`Standalone matrix record ${index} lacks a boolean evidence gate.`);
    }
  }
  return globalIdentity.environmentIdentityExact === true
    && matrixRecords.every(
      (record) => record?.identity?.environmentIdentityExact === true
        && record?.telemetry?.eligibility?.pass === true,
    );
}

function canonicalTrialRunId(canonicalTrial) {
  const suffix = `-m${canonicalTrial?.matrixOrdinal}`;
  const matrixId = canonicalTrial?.matrixId;
  if (typeof matrixId !== 'string' || !matrixId.endsWith(suffix)) return null;
  return matrixId.slice(0, -suffix.length);
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
    maxStorageBuffersPerShaderStage:
      environment?.maxStorageBuffersPerShaderStage ?? null,
    timestampAvailable: environment?.timestampAvailable ?? null,
    indirectFirstInstanceAvailable:
      environment?.indirectFirstInstanceAvailable ?? null,
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
    `${label} lacks indirect-first-instance`, environment);
  requireCondition(environment?.timestampAvailable === true,
    `${label} lacks timestamp queries`, environment);
  requireCondition(environment?.reversedDepth === true
    && environment?.rendererReversedDepthBuffer === true,
  `${label} lacks pinned reversed depth`, environment);
  requireCondition(environment?.crossOriginIsolated === true,
    `${label} lost cross-origin isolation`, environment);
  requireCondition(Number.isFinite(environment?.performanceNowQuantumMs)
    && environment.performanceNowQuantumMs > 0
    && environment.performanceNowQuantumMs <= MAXIMUM_CPU_TIMER_QUANTUM_MS,
  `${label} has an invalid CPU timer quantum`, environment);
  requireCondition(environment?.webgpuUncapturedErrorCount === 0,
    `${label} observed an uncaptured WebGPU error`, environment);
  requireCondition(environment?.webgpuValidationErrorCount === 0,
    `${label} observed a WebGPU validation error`, environment);
  requireCondition(environment?.webgpuDeviceLossCount === 0,
    `${label} observed WebGPU device loss`, environment);
}

function requireCleanGpuRecords(record, label) {
  requireCondition(Array.isArray(record?.webgpuUncapturedErrors)
    && record.webgpuUncapturedErrors.length === 0,
  `${label} has uncaptured WebGPU errors`, record?.webgpuUncapturedErrors);
  requireCondition(Array.isArray(record?.webgpuDeviceLosses)
    && record.webgpuDeviceLosses.length === 0,
  `${label} has WebGPU device losses`, record?.webgpuDeviceLosses);
  requireCleanEnvironment(record?.environment, `${label} environment`);
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

function requireStandaloneLifecycle(lifecycle, canonicalSession, expectedSwitchSerial, label) {
  const ledger = lifecycle?.productionResourceLedger;
  const lane = canonicalSession.assignedLaneId;
  requireCondition(lifecycle?.schemaVersion === 1
    && lifecycle?.kind === 'first-instance-live-standalone-static-resource-lifecycle'
    && lifecycle?.pass === true
    && lifecycle?.selectedLane === lane
    && lifecycle?.absentLane === canonicalSession.absentLaneId
    && lifecycle?.absentLaneConstructed === false
    && lifecycle?.lanesConstructed === 1
    && lifecycle?.primed === true
    && lifecycle?.scenarioSwitchSerial === expectedSwitchSerial
    && lifecycle?.bundleRecordCallbackCount === 1,
  `${label} violates the single-lane lifecycle`, lifecycle);
  requireCondition(ledger?.schemaVersion === 1
    && ledger?.kind === 'first-instance-live-standalone-production-resource-ledger'
    && ledger?.selectedLane === lane
    && ledger?.absentLane === canonicalSession.absentLaneId
    && ledger?.absentLaneConstructed === false
    && sameArray(ledger?.constructedLaneIds, [lane])
    && sameArray(
      ledger?.constructedAddressModes,
      [lane === 'portable' ? 'bucket-base' : 'indirect-first-instance'],
    )
    && ledger?.constructedLaneCount === 1
    && ledger?.indirectCommandBufferCount === 1
    && ledger?.computeNodeCount === 2
    && ledger?.materialCount === 1
    && ledger?.meshCount === 1
    && ledger?.bundleCount === 1
    && ledger?.hasBucketBaseVertexAttribute === (lane === 'portable'),
  `${label} has an invalid production resource ledger`, ledger);
  const identity = staticResourceIdentity(lifecycle);
  requireCondition(identity.commandBufferByteOffset === 0
    && identity.commandBufferFirstOffset === 0
    && Number.isSafeInteger(identity.commandBufferAttributeId)
    && Array.isArray(identity.computeNodeIds)
    && identity.computeNodeIds.length === 2,
  `${label} lacks its fixed command/timestamp identity`, identity);
  return identity;
}

function requireConfiguredSession(configured, canonicalSession, label) {
  const config = configured?.selectedConfig;
  const boot = configured?.initialPageBoot;
  const pageLifecycle = configured?.pageConstructionLifecycle;
  requireCondition(boot?.schemaVersion === 1
    && boot?.kind === 'first-instance-standalone-initial-page-boot'
    && boot?.modeId === STANDALONE_MODE_ID
    && boot?.laneId === canonicalSession.assignedLaneId
    && sameArray(boot?.visibilityOrder, canonicalSession.visibilityOrder)
    && boot?.objectCount === OBJECT_COUNT
    && boot?.bucketCount === BUCKET_COUNT
    && boot?.layout === LAYOUT
    && boot?.initialRebuildCount === 1
    && boot?.priorStrategyConstructionCount === 0,
  `${label} initial page boot differs from the frozen session`, boot);
  requireCondition(pageLifecycle?.schemaVersion === 1
    && pageLifecycle?.kind === 'benchmark-page-strategy-construction-lifecycle'
    && pageLifecycle?.rebuildCount === 1
    && pageLifecycle?.strategyConstructionCount === 1
    && sameArray(pageLifecycle?.constructedStrategyIds, [STANDALONE_MODE_ID])
    && pageLifecycle?.selectedStrategyId === STANDALONE_MODE_ID
    && pageLifecycle?.strictStandaloneBoot === true,
  `${label} constructed more than its sole boot strategy`, pageLifecycle);
  requireCondition(config?.strategyId === STANDALONE_MODE_ID
    && config?.objectCount === OBJECT_COUNT
    && config?.bucketCount === BUCKET_COUNT
    && config?.visibilityFraction === canonicalSession.visibilityOrder[0]
    && config?.layout === LAYOUT
    && config?.laneId === canonicalSession.assignedLaneId
    && sameArray(config?.visibilityOrder, canonicalSession.visibilityOrder),
  `${label} selected the wrong fixed standalone workload`, config);
  requireCondition(configured?.shaderEvidence === null,
    `${label} observed a shader before a runner challenge`, configured?.shaderEvidence);
  requireCondition(configured?.timestampPoolPreprime?.schemaVersion === 1
    && configured.timestampPoolPreprime.kind === 'three-r185-timestamp-pool-preprime'
    && configured.timestampPoolPreprime.addedTimestampUidCount?.render === 1
    && configured.timestampPoolPreprime.addedTimestampUidCount?.compute === 1,
  `${label} timestamp pools were not exactly pre-primed`, configured?.timestampPoolPreprime);
  const identity = requireStandaloneLifecycle(
    configured?.strategyLifecycle,
    canonicalSession,
    0,
    `${label} construction`,
  );
  requireCondition(configured?.strategyDiagnostics?.kind
    === 'first-instance-live-standalone-deployment'
    && configured.strategyDiagnostics.laneCommandBufferCount === 1
    && configured.strategyDiagnostics.configuredDrawCommands === BUCKET_COUNT
    && configured.strategyDiagnostics.configuredRenderObjects === 1
    && configured.strategyDiagnostics.configuredComputeDispatches === 2
    && configured.strategyDiagnostics.configuredComputeSubmissions === 1,
  `${label} strategy diagnostics differ from the frozen deployment`,
  configured?.strategyDiagnostics);
  requireCleanGpuRecords(configured, label);
  return identity;
}

function requireEvidencePoint(record, canonicalTrial, canonicalSession, label) {
  requireCleanGpuRecords(record, label);
  requireCondition(record?.renderParity?.schemaVersion === 1
    && record?.renderParity?.kind
      === 'first-instance-live-standalone-exact-render-parity'
    && record?.renderParity?.pass === true
    && record?.renderParity?.laneId === canonicalTrial.assignedLaneId
    && record?.renderParity?.productionBundleOutput?.pass === true
    && record?.renderParity?.snapshotValidation?.pass === true,
  `${label} render parity failed`, record?.renderParity);
  requireCondition(record?.validation?.schemaVersion === 1
    && record?.validation?.kind === 'first-instance-live-standalone-validation'
    && record?.validation?.pass === true
    && record?.validation?.laneId === canonicalTrial.assignedLaneId
    && record?.validation?.visibilityFraction === canonicalTrial.visibilityFraction,
  `${label} lane-local validation failed`, record?.validation);
  const expectedSwitch = canonicalTrial.visibilityOrderPosition;
  requireStandaloneLifecycle(
    record.validation.lifecycle,
    canonicalSession,
    expectedSwitch,
    `${label} validation`,
  );
  requireCondition(record?.workload?.scenario?.visibilityFraction
    === canonicalTrial.visibilityFraction,
  `${label} workload visibility differs`, record?.workload);
}

function shaderCapturesFromTrial({ preflight, timingParity, timingStart, postflight }) {
  return [
    preflight?.renderParity?.snapshotValidation?.shaderEvidence,
    preflight?.validation?.shaderEvidence,
    timingParity?.snapshotValidation?.shaderEvidence,
    timingStart?.validation?.shaderEvidence,
    postflight?.renderParity?.snapshotValidation?.shaderEvidence,
    postflight?.validation?.shaderEvidence,
  ];
}

function requireShaderCaptures(captures, challenges, canonicalTrial, label) {
  requireCondition(Array.isArray(captures) && captures.length === challenges.length,
    `${label} shader capture count differs`, captures);
  for (let index = 0; index < challenges.length; index += 1) {
    const capture = captures[index];
    const challenge = challenges[index];
    requireCondition(capture?.schemaVersion === 1
      && capture?.kind === 'first-instance-live-standalone-fresh-shader-observation'
      && capture?.pass === true
      && capture?.laneId === canonicalTrial.assignedLaneId
      && capture?.observationSerial === challenge.captureOrdinal
      && sameJson(capture?.observationRequest, challenge)
      && SHA256_PATTERN.test(capture?.observationDigest)
      && capture?.laneRecord?.pass === true
      && capture?.computeShaderEvidence?.pass === true
      && typeof capture?.rawSources?.vertexShader === 'string'
      && capture.rawSources.vertexShader.length > 0
      && typeof capture?.rawSources?.fragmentShader === 'string'
      && capture.rawSources.fragmentShader.length > 0,
    `${label} shader capture ${index + 1} failed its lane-local audit`, capture);
  }
}

function createIdentityTracker() {
  return {
    baselineEnvironment: null,
    environmentObservations: [],
    environmentDrift: [],
    geometrySha256: null,
    workloadByVisibility: new Map(),
    observe(environment, workload, label) {
      const observedEnvironment = environmentIdentity(environment);
      this.baselineEnvironment ??= observedEnvironment;
      this.environmentObservations.push({ label, identity: observedEnvironment });
      if (!sameJson(this.baselineEnvironment, observedEnvironment)) {
        this.environmentDrift.push({
          label,
          expected: this.baselineEnvironment,
          observed: observedEnvironment,
        });
      }
      const observedWorkload = workloadIdentity(workload);
      requireCondition(observedWorkload.scenarioSeed === SCENARIO_SEED
        && observedWorkload.objectCount === OBJECT_COUNT
        && observedWorkload.bucketCount === BUCKET_COUNT
        && observedWorkload.layout === LAYOUT
        && SHA256_PATTERN.test(observedWorkload.geometrySha256)
        && SHA256_PATTERN.test(observedWorkload.scenarioSha256)
        && SHA256_PATTERN.test(observedWorkload.expectedVisibleIdsCanonicalSha256),
      `${label} workload commitment is invalid`, observedWorkload);
      this.geometrySha256 ??= observedWorkload.geometrySha256;
      requireCondition(observedWorkload.geometrySha256 === this.geometrySha256,
        `${label} geometry fixture commitment drifted`, observedWorkload);
      const key = String(observedWorkload.visibilityFraction);
      const prior = this.workloadByVisibility.get(key);
      if (prior === undefined) this.workloadByVisibility.set(key, observedWorkload);
      else requireCondition(sameJson(prior, observedWorkload),
        `${label} workload commitment drifted within its visibility cell`, {
          expected: prior,
          observed: observedWorkload,
        });
    },
    report() {
      return {
        baselineEnvironment: clone(this.baselineEnvironment),
        environmentObservationCount: this.environmentObservations.length,
        environmentIdentityExact: this.environmentDrift.length === 0,
        environmentDrift: clone(this.environmentDrift),
        geometrySha256: this.geometrySha256,
        workloadByVisibility: Object.fromEntries(
          [...this.workloadByVisibility.entries()].sort(([left], [right]) => (
            Number(left) - Number(right)
          )),
        ),
      };
    },
  };
}

async function findBrowser() {
  const programFiles = process.env.ProgramFiles ?? 'C:\\Program Files';
  const programFilesX86 = process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)';
  const localAppData = process.env.LOCALAPPDATA;
  const candidates = [
    process.env.BROWSER_PATH,
    path.join(programFiles, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    path.join(programFilesX86, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    path.join(programFiles, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    path.join(programFilesX86, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    localAppData && path.join(localAppData, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
  ].filter(Boolean);
  for (const candidate of [...new Set(candidates)]) {
    try {
      await access(candidate);
      return path.resolve(candidate);
    } catch {
      // Continue through installed system-browser locations.
    }
  }
  throw new Error('No installed Chrome, Chromium, or Edge executable was found.');
}

function attachErrorCapture(page, records) {
  page.on('pageerror', (error) => records.push({
    source: 'pageerror',
    detail: error.stack ?? error.message,
    capturedAt: new Date().toISOString(),
  }));
  page.on('console', (message) => {
    if (message.type() === 'error') {
      records.push({
        source: 'console',
        detail: message.text(),
        capturedAt: new Date().toISOString(),
      });
    }
  });
}

class BrowserLifecycleManager {
  constructor({ executablePath, runStartedMonotonic, requireRunActive }) {
    this.executablePath = executablePath;
    this.runStartedMonotonic = runStartedMonotonic;
    this.requireRunActive = requireRunActive;
    this.records = [];
    this.active = null;
  }

  elapsed() {
    return performance.now() - this.runStartedMonotonic;
  }

  async launch({ role, matrixOrdinal = null, session = null }) {
    this.requireRunActive();
    requireCondition(this.active === null, `Refusing overlapping browser launch for ${role}.`);
    const prior = this.records.at(-1) ?? null;
    if (prior !== null) {
      requireCondition(prior.disconnectedEventCount === 1
        && prior.postDisconnectDelay?.requestedMs === STANDALONE_POST_DISCONNECT_DELAY_MS
        && prior.postDisconnectDelay?.elapsedMs >= STANDALONE_POST_DISCONNECT_DELAY_MS,
      `Prior browser did not complete its fixed disconnect interval before ${role}.`, prior);
    }
    const launchedRunElapsedMs = this.elapsed();
    const record = {
      schemaVersion: 1,
      kind: 'first-instance-standalone-browser-profile-lifecycle',
      browserInstanceSerial: this.records.length + 1,
      role,
      matrixOrdinal,
      sessionId: session?.sessionId ?? null,
      globalSessionIndex: session?.globalSessionIndex ?? null,
      sessionNamespace: session === null
        ? null
        : `${session.sessionId}:browser-${this.records.length + 1}`,
      launchApi: 'chromium.launch',
      persistentContext: false,
      profilePolicy: 'fresh-playwright-temporary-profile-per-process',
      userDataDirectoryExposed: false,
      profileReused: false,
      launchArguments: [...BROWSER_ARGS],
      launchedAt: new Date().toISOString(),
      launchedRunElapsedMs,
      priorBrowserInstanceSerial: prior?.browserInstanceSerial ?? null,
      priorBrowserDisconnectedAt: prior?.disconnectedAt ?? null,
      previousDisconnectToLaunchGapMs: prior === null
        ? null
        : launchedRunElapsedMs - prior.disconnectedRunElapsedMs,
      contextCreatedAt: null,
      pageCreatedAt: null,
      closedAt: null,
      disconnectedAt: null,
      disconnectedRunElapsedMs: null,
      contextCountBeforeClose: null,
      pageCountBeforeClose: null,
      disconnectedEventCount: 0,
      closedBeforeNextLaunch: false,
      postDisconnectDelay: null,
    };
    if (prior !== null) {
      requireCondition(record.previousDisconnectToLaunchGapMs
        >= STANDALONE_POST_DISCONNECT_DELAY_MS,
      `${role} launched before the fixed post-disconnect interval elapsed.`, record);
    }
    const browser = await chromium.launch({
      executablePath: this.executablePath,
      headless: true,
      args: BROWSER_ARGS,
      timeout: BROWSER_OPERATION_TIMEOUT_MS,
    });
    browser.on('disconnected', () => {
      record.disconnectedEventCount += 1;
      record.disconnectedAt ??= new Date().toISOString();
      record.disconnectedRunElapsedMs ??= this.elapsed();
    });
    requireCondition(browser.contexts().length === 0,
      `${role} browser started with an unexpected context.`);
    const state = {
      browser,
      context: null,
      page: null,
      errors: [],
      record,
      closePromise: null,
    };
    this.records.push(record);
    this.active = state;
    this.requireRunActive();
    return state;
  }

  async openOnlyPage(state, url) {
    this.requireRunActive();
    requireCondition(this.active === state && state.context === null && state.page === null,
      `${state.record.role} attempted more than one context/page.`);
    state.context = await withDeadline(state.browser.newContext({
      viewport: { width: 1280, height: 900 },
      deviceScaleFactor: 1,
    }), BROWSER_OPERATION_TIMEOUT_MS, `${state.record.role} context creation`);
    state.record.contextCreatedAt = new Date().toISOString();
    requireCondition(state.browser.contexts().length === 1,
      `${state.record.role} did not retain exactly one context.`);
    state.page = await withDeadline(
      state.context.newPage(),
      BROWSER_OPERATION_TIMEOUT_MS,
      `${state.record.role} page creation`,
    );
    state.record.pageCreatedAt = new Date().toISOString();
    attachErrorCapture(state.page, state.errors);
    requireCondition(state.context.pages().length === 1,
      `${state.record.role} did not retain exactly one page.`);
    await withDeadline(
      state.page.goto(url, { waitUntil: 'domcontentloaded' }),
      BROWSER_OPERATION_TIMEOUT_MS,
      `${state.record.role} entry navigation`,
    );
    await withDeadline(state.page.waitForFunction(
      () => window.__WEBGPU_BENCH__?.ready === true,
      null,
      { timeout: 120_000 },
    ), BROWSER_OPERATION_TIMEOUT_MS, `${state.record.role} benchmark readiness`);
    this.requireRunActive();
    return state.page;
  }

  async close(state) {
    if (state.closePromise !== null) return state.closePromise;
    requireCondition(this.active === state,
      `Attempted to close a non-active ${state.record.role} browser.`);
    state.closePromise = (async () => {
      const { browser, context, page, record } = state;
      record.contextCountBeforeClose = browser.contexts().length;
      record.pageCountBeforeClose = context?.pages().length ?? 0;
      requireCondition(record.contextCountBeforeClose === 1
        && record.pageCountBeforeClose === 1,
      `${record.role} violated its one-context/one-page boundary.`, record);
      await withDeadline(
        context.close(),
        BROWSER_CLOSE_TIMEOUT_MS,
        `${record.role} context close`,
      );
      requireCondition(page.isClosed() === true && browser.contexts().length === 0,
        `${record.role} context/page did not close cleanly.`);
      await withDeadline(
        browser.close(),
        BROWSER_CLOSE_TIMEOUT_MS,
        `${record.role} browser close`,
      );
      requireCondition(browser.isConnected() === false,
        `${record.role} browser process remained connected after close.`);
      requireCondition(record.disconnectedEventCount === 1
        && Number.isFinite(record.disconnectedRunElapsedMs),
      `${record.role} emitted an unexpected disconnected event count.`, record);
      record.closedAt = new Date().toISOString();
      record.closedBeforeNextLaunch = true;
      this.active = null;
      return record;
    })();
    return state.closePromise;
  }

  async waitAfterDisconnect(record) {
    requireCondition(this.active === null
      && record?.disconnectedEventCount === 1
      && record?.postDisconnectDelay === null,
    'Post-disconnect delay requires one newly closed browser.', record);
    const startedMonotonic = performance.now();
    const waitRecord = {
      requestedMs: STANDALONE_POST_DISCONNECT_DELAY_MS,
      startedAt: new Date().toISOString(),
      startedRunElapsedMs: startedMonotonic - this.runStartedMonotonic,
      completedAt: null,
      completedRunElapsedMs: null,
      elapsedMs: null,
    };
    await delay(STANDALONE_POST_DISCONNECT_DELAY_MS);
    const completedMonotonic = performance.now();
    waitRecord.completedAt = new Date().toISOString();
    waitRecord.completedRunElapsedMs = completedMonotonic - this.runStartedMonotonic;
    waitRecord.elapsedMs = completedMonotonic - startedMonotonic;
    requireCondition(waitRecord.elapsedMs >= STANDALONE_POST_DISCONNECT_DELAY_MS,
      'Post-disconnect interval completed early.', waitRecord);
    record.postDisconnectDelay = waitRecord;
    this.requireRunActive();
    return waitRecord;
  }

  async forceClose(reason) {
    const state = this.active;
    if (state === null) return null;
    this.active = null;
    state.record.abortedAt = new Date().toISOString();
    state.record.abortReason = reason;
    await withDeadline(
      state.context?.close() ?? Promise.resolve(),
      BROWSER_CLOSE_TIMEOUT_MS,
      `${state.record.role} forced context close`,
    ).catch(() => undefined);
    await withDeadline(
      state.browser?.close() ?? Promise.resolve(),
      BROWSER_CLOSE_TIMEOUT_MS,
      `${state.record.role} forced browser close`,
    ).catch(() => undefined);
    state.record.closedAt ??= new Date().toISOString();
    return clone(state.record);
  }
}

class ArtifactStore {
  constructor({ runDirectory, requireRunActive }) {
    this.runDirectory = runDirectory;
    this.requireRunActive = requireRunActive;
    this.journalSerial = 0;
    this.journalArtifacts = [];
  }

  relative(absolutePath) {
    return path.relative(this.runDirectory, absolutePath).split(path.sep).join('/');
  }

  async json(relativePath, value, { allowDuringTermination = false } = {}) {
    if (!allowDuringTermination) this.requireRunActive();
    const absolutePath = path.join(this.runDirectory, ...relativePath.split('/'));
    const bytes = jsonBytes(value);
    await writeFile(absolutePath, bytes, { flag: 'wx' });
    if (!allowDuringTermination) this.requireRunActive();
    return Object.freeze({
      path: this.relative(absolutePath),
      encoding: 'json-utf8',
      byteLength: bytes.length,
      sha256: sha256Bytes(bytes),
    });
  }

  async brotliJson(relativePath, value, { allowDuringTermination = false } = {}) {
    if (!allowDuringTermination) this.requireRunActive();
    const absolutePath = path.join(this.runDirectory, ...relativePath.split('/'));
    const bytes = jsonBytes(value);
    const compressed = await brotliCompressAsync(bytes, {
      params: {
        [zlibConstants.BROTLI_PARAM_QUALITY]: 6,
        [zlibConstants.BROTLI_PARAM_MODE]: zlibConstants.BROTLI_MODE_TEXT,
      },
    });
    if (!allowDuringTermination) this.requireRunActive();
    await writeFile(absolutePath, compressed, { flag: 'wx' });
    if (!allowDuringTermination) this.requireRunActive();
    return Object.freeze({
      path: this.relative(absolutePath),
      encoding: 'brotli-json-utf8',
      jsonByteLength: bytes.length,
      jsonSha256: sha256Bytes(bytes),
      brotliByteLength: compressed.length,
      brotliSha256: sha256Bytes(compressed),
    });
  }

  async existing(relativePath, encoding) {
    this.requireRunActive();
    const absolutePath = path.join(this.runDirectory, ...relativePath.split('/'));
    const bytes = await readFile(absolutePath);
    return Object.freeze({
      path: this.relative(absolutePath),
      encoding,
      byteLength: bytes.length,
      sha256: sha256Bytes(bytes),
    });
  }

  async journal(eventKind, payload) {
    this.journalSerial += 1;
    const filename = `journal/${String(this.journalSerial).padStart(4, '0')}`
      + `-${eventKind}.json`;
    const artifact = await this.json(filename, {
      schemaVersion: 1,
      kind: 'first-instance-standalone-incremental-artifact-journal-event',
      journalSerial: this.journalSerial,
      eventKind,
      committedAt: new Date().toISOString(),
      payload,
    });
    this.journalArtifacts.push(artifact);
    return artifact;
  }
}

async function configureStandaloneSession(page, canonicalSession) {
  return page.evaluate(async () => {
    const bench = window.__WEBGPU_BENCH__;
    return {
      initialPageBoot: bench.initialPageBoot,
      pageConstructionLifecycle: bench.pageConstructionLifecycle,
      selectedConfig: bench.selectedConfig(),
      environment: bench.environment,
      shaderEvidence: bench.firstInstanceShaderEvidence,
      timestampPoolPreprime: bench.timestampPoolPreprime,
      timestampPoolDiagnostics: bench.timestampPoolDiagnostics,
      strategyLifecycle: bench.strategyLifecycle,
      strategyDiagnostics: bench.strategyDiagnostics,
      cacheDiagnostics: bench.cacheDiagnostics(),
      geometryManifest: bench.geometryManifest,
      scenarioManifest: bench.scenarioManifest,
      workload: await bench.fingerprintWorkload(),
      webgpuUncapturedErrors: bench.webgpuUncapturedErrors,
      webgpuDeviceLosses: bench.webgpuDeviceLosses,
    };
  });
}

async function captureEvidencePoint(page, observationChallenges) {
  return page.evaluate(async (challenges) => {
    const bench = window.__WEBGPU_BENCH__;
    const renderParity = await bench.captureRenderParity(challenges[0]);
    const validation = await bench.validate(challenges[1]);
    return {
      validation,
      renderParity,
      workload: await bench.fingerprintWorkload(),
      environment: bench.environment,
      strategyLifecycle: bench.strategyLifecycle,
      strategyDiagnostics: bench.strategyDiagnostics,
      cacheDiagnostics: bench.cacheDiagnostics(),
      webgpuUncapturedErrors: bench.webgpuUncapturedErrors,
      webgpuDeviceLosses: bench.webgpuDeviceLosses,
    };
  }, observationChallenges);
}

async function switchStandaloneVisibility(page, visibilityFraction) {
  return page.evaluate(async (nextVisibilityFraction) => {
    const bench = window.__WEBGPU_BENCH__;
    const switchEvidence = await bench.switchFirstInstanceLiveStandaloneVisibility({
      visibilityFraction: nextVisibilityFraction,
    });
    return {
      switchEvidence,
      selectedConfig: bench.selectedConfig(),
      environment: bench.environment,
      workload: await bench.fingerprintWorkload(),
      strategyLifecycle: bench.strategyLifecycle,
      strategyDiagnostics: bench.strategyDiagnostics,
      cacheDiagnostics: bench.cacheDiagnostics(),
      webgpuUncapturedErrors: bench.webgpuUncapturedErrors,
      webgpuDeviceLosses: bench.webgpuDeviceLosses,
    };
  }, visibilityFraction);
}

export function createStandaloneAuditContext({
  runId,
  canonicalTrial,
  executionMode,
  planSha256,
  browserRecord,
}) {
  return Object.freeze({
    ...clone(canonicalTrial),
    runId,
    executionMode,
    planSha256,
    browserInstanceSerial: browserRecord.browserInstanceSerial,
    sessionNamespace: browserRecord.sessionNamespace,
    profilePolicy: browserRecord.profilePolicy,
    protocolWarmupFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES,
    protocolMeasuredFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
    protocolMeasuredBlockSize: STANDALONE_BLOCK_SIZE,
    protocolMeasuredBlockCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS,
  });
}

function requireWithinSessionIdentity(baseIdentity, lifecycle, label) {
  const observed = staticResourceIdentity(lifecycle);
  requireCondition(sameJson(baseIdentity, observed),
    `${label} changed a selected-lane static resource identity`, {
      expected: baseIdentity,
      observed,
    });
}

function requireVisibilitySwitch(record, canonicalSession, baseIdentity, label) {
  requireCleanGpuRecords(record, label);
  const evidence = record?.switchEvidence;
  requireCondition(evidence?.schemaVersion === 1
    && evidence?.kind === 'first-instance-live-standalone-visibility-switch'
    && evidence?.pass === true
    && evidence?.stableIdentity === true
    && evidence?.laneId === canonicalSession.assignedLaneId
    && evidence?.scenarioSwitchSerial === 1
    && evidence?.fromVisibilityFraction === canonicalSession.visibilityOrder[0]
    && evidence?.toVisibilityFraction === canonicalSession.visibilityOrder[1]
    && evidence?.matrixVersionAfter === evidence?.matrixVersionBefore + 1
    && evidence?.boundsVersionAfter === evidence?.boundsVersionBefore + 1,
  `${label} visibility switch differs from the sole frozen transition`, evidence);
  requireCondition(record?.selectedConfig?.strategyId === STANDALONE_MODE_ID
    && record?.selectedConfig?.laneId === canonicalSession.assignedLaneId
    && record?.selectedConfig?.visibilityFraction === canonicalSession.visibilityOrder[1]
    && sameArray(record?.selectedConfig?.visibilityOrder, canonicalSession.visibilityOrder),
  `${label} selected configuration changed outside visibility`, record?.selectedConfig);
  for (const [suffix, lifecycle] of [
    ['before', evidence.before],
    ['after', evidence.after],
    ['reported lifecycle', evidence.lifecycle],
    ['page lifecycle', record.strategyLifecycle],
  ]) requireWithinSessionIdentity(baseIdentity, lifecycle, `${label} ${suffix}`);
  requireStandaloneLifecycle(evidence.before, canonicalSession, 0, `${label} before`);
  requireStandaloneLifecycle(evidence.after, canonicalSession, 1, `${label} after`);
  requireStandaloneLifecycle(evidence.lifecycle, canonicalSession, 1, `${label} result`);
  requireCondition(record?.workload?.scenario?.visibilityFraction
    === canonicalSession.visibilityOrder[1]
    && record?.workload?.scenario?.expectedVisibleCount
      === evidence.expectedVisibleCount,
  `${label} switched workload commitment differs`, record?.workload);
}

function executionIdentityReference({
  sourceProvenance,
  dependencyClosure,
}) {
  return Object.freeze({
    sourceCommit: sourceProvenance.commit,
    sourceTree: sourceProvenance.tree,
    trackedFilesSha256: sourceProvenance.trackedFilesSha256,
    packageLockSha256: sourceProvenance.packageLockSha256,
    executionDependencyClosureSha256: dependencyClosure.sha256,
    viteRuntimePolicyId: CANDIDATE_VITE_RUNTIME_POLICY_ID,
    viteRuntimeAuditBinding: 'final-capture-manifest',
  });
}

async function runStandaloneTrial({
  page,
  pageErrors,
  canonicalTrial,
  canonicalSession,
  configured,
  baseResourceIdentity,
  identityTracker,
  usedNonces,
  runId,
  executionMode,
  planSha256,
  browserRecord,
  executionIdentity,
  artifactStore,
}) {
  const label = `standalone plan ${canonicalTrial.planIndex} (${canonicalTrial.trialId})`;
  const firstCaptureOrdinal = canonicalTrial.visibilityOrderPosition * 6 + 1;
  const shaderObservationChallenges = createStandaloneShaderObservationChallenges({
    canonicalTrial,
    runId,
    firstCaptureOrdinal,
    usedNonces,
  });
  const auditContext = createStandaloneAuditContext({
    runId,
    canonicalTrial,
    executionMode,
    planSha256,
    browserRecord,
  });
  const preflight = await withDeadline(
    captureEvidencePoint(page, shaderObservationChallenges.slice(0, 2)),
    BROWSER_OPERATION_TIMEOUT_MS,
    `${label} preflight evidence`,
  );
  requireEvidencePoint(preflight, canonicalTrial, canonicalSession, `${label} preflight`);
  identityTracker.observe(preflight.environment, preflight.workload, `${label} preflight`);

  const timingParity = await withDeadline(page.evaluate(
    (challenge) => window.__WEBGPU_BENCH__.captureRenderParity(challenge),
    shaderObservationChallenges[2],
  ), BROWSER_OPERATION_TIMEOUT_MS, `${label} timing-start parity`);
  requireCondition(timingParity?.pass === true
    && timingParity?.kind === 'first-instance-live-standalone-exact-render-parity'
    && timingParity?.laneId === canonicalTrial.assignedLaneId,
  `${label} timing-start render parity failed`, timingParity);
  const timingStart = await withDeadline(page.evaluate(
    ({ context, challenge }) => window.__WEBGPU_BENCH__.startTrial(context, challenge),
    { context: auditContext, challenge: shaderObservationChallenges[3] },
  ), BROWSER_OPERATION_TIMEOUT_MS, `${label} timing start`);
  requireCondition(timingStart?.validation?.pass === true
    && timingStart?.validation?.laneId === canonicalTrial.assignedLaneId
    && timingStart?.validation?.visibilityFraction === canonicalTrial.visibilityFraction,
  `${label} timing-start validation failed`, timingStart?.validation);
  identityTracker.observe(configured.environment, timingStart.workload, `${label} timing start`);

  await withDeadline(page.waitForFunction(
    () => ['complete', 'error'].includes(window.__WEBGPU_BENCH__?.phase),
    null,
    { timeout: BROWSER_OPERATION_TIMEOUT_MS },
  ), BROWSER_OPERATION_TIMEOUT_MS, `${label} timed phase completion`);
  const timing = await withDeadline(page.evaluate(() => ({
    phase: window.__WEBGPU_BENCH__.phase,
    error: window.__WEBGPU_BENCH__.trialError,
    rows: window.__WEBGPU_BENCH__.rows,
    summary: window.__WEBGPU_BENCH__.summary,
    environment: window.__WEBGPU_BENCH__.environment,
    strategyLifecycle: window.__WEBGPU_BENCH__.strategyLifecycle,
    strategyDiagnostics: window.__WEBGPU_BENCH__.strategyDiagnostics,
    cacheDiagnostics: window.__WEBGPU_BENCH__.cacheDiagnostics(),
    shaderEvidence: window.__WEBGPU_BENCH__.firstInstanceShaderEvidence,
    webgpuUncapturedErrors: window.__WEBGPU_BENCH__.webgpuUncapturedErrors,
    webgpuDeviceLosses: window.__WEBGPU_BENCH__.webgpuDeviceLosses,
  })), BROWSER_OPERATION_TIMEOUT_MS, `${label} timing collection`);
  requireCondition(timing.phase === 'complete', `${label} timing failed`, timing.error);
  requireCleanGpuRecords(timing, `${label} timing`);
  const runnerValidation = validateStandaloneTimingRows(
    timing.rows,
    canonicalTrial,
    timing.summary,
  );
  requireCondition(runnerValidation.pass === true,
    `${label} runner timing validation failed`, runnerValidation);

  const postflight = await withDeadline(
    captureEvidencePoint(page, shaderObservationChallenges.slice(4, 6)),
    BROWSER_OPERATION_TIMEOUT_MS,
    `${label} postflight evidence`,
  );
  requireEvidencePoint(postflight, canonicalTrial, canonicalSession, `${label} postflight`);
  identityTracker.observe(postflight.environment, postflight.workload, `${label} postflight`);
  requireCondition(sameJson(workloadIdentity(preflight.workload),
    workloadIdentity(timingStart.workload))
      && sameJson(workloadIdentity(preflight.workload),
        workloadIdentity(postflight.workload)),
  `${label} workload changed across evidence phases`);
  requireCondition(pageErrors.length === 0, `${label} emitted page errors`, pageErrors);

  const captures = shaderCapturesFromTrial({
    preflight,
    timingParity,
    timingStart,
    postflight,
  });
  requireShaderCaptures(captures, shaderObservationChallenges, canonicalTrial, label);
  for (const lifecycle of [
    preflight.renderParity.snapshotValidation.lifecycle,
    preflight.validation.lifecycle,
    timingParity.snapshotValidation.lifecycle,
    timingStart.validation.lifecycle,
    timing.strategyLifecycle,
    timing.summary.completionInvariant?.renderCommitmentAtTimingEnd,
    postflight.renderParity.snapshotValidation.lifecycle,
    postflight.validation.lifecycle,
  ]) {
    if (lifecycle?.kind === 'first-instance-live-standalone-static-resource-lifecycle') {
      requireWithinSessionIdentity(baseResourceIdentity, lifecycle, label);
    }
  }
  requireWithinSessionIdentity(baseResourceIdentity, timing.strategyLifecycle, `${label} timing`);
  requireStandaloneLifecycle(
    timing.strategyLifecycle,
    canonicalSession,
    canonicalTrial.visibilityOrderPosition,
    `${label} timing`,
  );

  const protocol = Object.freeze({
    schemaVersion: 1,
    kind: 'first-instance-standalone-deployment-timing-protocol',
    warmupFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES,
    measuredFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
    measuredBlockSize: STANDALONE_BLOCK_SIZE,
    measuredBlockCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS,
    selectedLaneOnly: true,
    absentLaneConstructionAllowed: false,
    visibilityExposure: canonicalTrial.visibilityExposure,
    estimator: 'median-of-60-eight-frame-block-means',
  });
  const artifactBody = createStandaloneTrialArtifact({
    canonicalTrial,
    rows: timing.rows,
    executionMode,
    planSha256,
    body: {
      scope: executionMode === STANDALONE_EXECUTION_MODES.SMOKE
        ? 'plumbing smoke; explicitly excluded from analysis and deployment decision'
        : 'full candidate capture; decision deferred to independent analysis and verification',
      runId,
      capturedAt: new Date().toISOString(),
      sessionNamespace: browserRecord.sessionNamespace,
      browserInstanceSerial: browserRecord.browserInstanceSerial,
      executionIdentity,
      configured,
      shaderObservationChallenges,
      preflight,
      timingStart,
      timing: {
        renderParity: timingParity,
        summary: timing.summary,
        environment: timing.environment,
        strategyLifecycle: timing.strategyLifecycle,
        strategyDiagnostics: timing.strategyDiagnostics,
        cacheDiagnostics: timing.cacheDiagnostics,
        shaderEvidence: timing.shaderEvidence,
        webgpuUncapturedErrors: timing.webgpuUncapturedErrors,
        webgpuDeviceLosses: timing.webgpuDeviceLosses,
      },
      postflight,
      protocol,
      runnerValidation,
      pageErrors: clone(pageErrors),
    },
  });
  const filename = `trials/trial-${String(canonicalTrial.planIndex + 1).padStart(3, '0')}`
    + `-m${canonicalTrial.matrixOrdinal}`
    + `-s${String(canonicalTrial.matrixSessionIndex + 1).padStart(2, '0')}`
    + `-t${canonicalTrial.visibilityOrderPosition + 1}`
    + `-${canonicalTrial.assignedLaneId}`
    + `-v${Math.round(canonicalTrial.visibilityFraction * 100)}.json.br`;
  const artifact = await artifactStore.brotliJson(filename, artifactBody);
  await artifactStore.journal('trial-committed', {
    trialId: canonicalTrial.trialId,
    planIndex: canonicalTrial.planIndex,
    sessionId: canonicalTrial.sessionId,
    artifact,
  });
  return Object.freeze({
    trialId: canonicalTrial.trialId,
    planIndex: canonicalTrial.planIndex,
    matrixIndex: canonicalTrial.matrixIndex,
    sessionId: canonicalTrial.sessionId,
    assignedLaneId: canonicalTrial.assignedLaneId,
    visibilityFraction: canonicalTrial.visibilityFraction,
    visibilityExposure: canonicalTrial.visibilityExposure,
    sessionNamespace: browserRecord.sessionNamespace,
    shaderCaptureOrdinals: shaderObservationChallenges.map(
      (challenge) => challenge.captureOrdinal,
    ),
    artifact,
  });
}

async function runDisposableForcedFeatureOffGate({
  page,
  pageErrors,
  matrix,
  identityTracker,
  runId,
  executionMode,
  planSha256,
  browserRecord,
  artifactStore,
}) {
  const capture = await withDeadline(page.evaluate(async (options) => {
    const bench = window.__WEBGPU_BENCH__;
    const gate = await bench.runFirstInstanceLiveForcedFeatureOffGate(options);
    return {
      gate,
      environment: bench.environment,
      workload: await bench.fingerprintWorkload(),
      webgpuUncapturedErrors: bench.webgpuUncapturedErrors,
      webgpuDeviceLosses: bench.webgpuDeviceLosses,
    };
  }, {
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    visibilityFraction: 0.99,
    scenarioSeed: SCENARIO_SEED,
  }), BROWSER_OPERATION_TIMEOUT_MS, `matrix ${matrix.matrixOrdinal} forced-feature-off gate`);
  const rejectionReasons = validateLiveFirstInstanceForcedFeatureOffGate(capture.gate);
  requireCondition(capture.gate?.initialPageBoot?.schemaVersion === 1
    && capture.gate?.initialPageBoot?.kind
      === 'first-instance-standalone-initial-page-boot'
    && capture.gate?.initialPageBoot?.modeId === STANDALONE_MODE_ID
    && capture.gate?.initialPageBoot?.laneId === 'portable'
    && sameArray(capture.gate?.initialPageBoot?.visibilityOrder, [0.99, 0.2])
    && capture.gate?.initialPageBoot?.initialRebuildCount === 1
    && capture.gate?.initialPageBoot?.priorStrategyConstructionCount === 0,
  `Matrix ${matrix.matrixOrdinal} gate lacks strict portable initial boot evidence.`,
  capture.gate?.initialPageBoot);
  requireCondition(capture.gate?.construction?.pageConstructionLifecycle?.rebuildCount === 1
    && capture.gate.construction.pageConstructionLifecycle.strategyConstructionCount === 1
    && sameArray(
      capture.gate.construction.pageConstructionLifecycle.constructedStrategyIds,
      [STANDALONE_MODE_ID],
    )
    && capture.gate.construction.pageConstructionLifecycle.selectedStrategyId
      === STANDALONE_MODE_ID
    && capture.gate.construction.pageConstructionLifecycle.strictStandaloneBoot === true,
  `Matrix ${matrix.matrixOrdinal} gate constructed more than its sole portable strategy.`,
  capture.gate?.construction?.pageConstructionLifecycle);
  requireCondition(pageErrors.length === 0,
    `Matrix ${matrix.matrixOrdinal} forced-feature-off gate emitted page errors`, pageErrors);
  requireCleanGpuRecords(capture, `matrix ${matrix.matrixOrdinal} forced-feature-off gate`);
  requireCondition(rejectionReasons.length === 0,
    `Matrix ${matrix.matrixOrdinal} forced-feature-off gate failed`, rejectionReasons);
  identityTracker.observe(
    capture.environment,
    capture.workload,
    `matrix ${matrix.matrixOrdinal} forced-feature-off gate`,
  );
  const body = {
    schemaVersion: 1,
    kind: 'first-instance-standalone-matrix-forced-feature-off-gate',
    executionMode,
    analysisEligible: executionMode === STANDALONE_EXECUTION_MODES.FULL,
    runId,
    matrixId: matrix.matrixId,
    matrixIndex: matrix.matrixIndex,
    matrixOrdinal: matrix.matrixOrdinal,
    planSha256,
    capturedAt: new Date().toISOString(),
    browserInstanceSerial: browserRecord.browserInstanceSerial,
    profilePolicy: browserRecord.profilePolicy,
    capture,
    validationRejections: rejectionReasons,
    pageErrors: clone(pageErrors),
  };
  const relativePath = `matrices/matrix-${String(matrix.matrixOrdinal).padStart(2, '0')}`
    + '/forced-feature-off-gate.json.br';
  const artifact = await artifactStore.brotliJson(relativePath, body);
  await artifactStore.journal('forced-feature-off-gate-committed', {
    matrixId: matrix.matrixId,
    artifact,
  });
  return { capture, artifact };
}

function compositeIdentityTracker(...trackers) {
  return Object.freeze({
    observe(environment, workload, label) {
      for (const tracker of trackers) tracker.observe(environment, workload, label);
    },
  });
}

function telemetryMatrixEligibility({
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
  } else if (processComparisons.some((record) => record?.comparison?.pass !== true)) {
    reasons.push('external GPU compute-process identity set changed during matrix');
  }
  if (adapterAssociation?.pass !== true) {
    reasons.push('page adapter and telemetry GPU association failed');
  }
  return Object.freeze({
    schemaVersion: 1,
    kind: 'first-instance-standalone-matrix-telemetry-eligibility',
    pass: reasons.length === 0,
    reasons: Object.freeze(reasons),
  });
}

function parseRunnerArguments(arguments_) {
  if (arguments_.some((argument) => argument !== '--smoke')
    || arguments_.filter((argument) => argument === '--smoke').length > 1) {
    throw new Error(
      'Usage: node scripts/run-first-instance-standalone-deployment.mjs [--smoke]',
    );
  }
  return arguments_[0] === '--smoke'
    ? STANDALONE_EXECUTION_MODES.SMOKE
    : STANDALONE_EXECUTION_MODES.FULL;
}

export async function runFirstInstanceStandaloneDeployment({
  executionMode = STANDALONE_EXECUTION_MODES.FULL,
} = {}) {
  if (!Object.values(STANDALONE_EXECUTION_MODES).includes(executionMode)) {
    throw new RangeError('Unknown standalone deployment execution mode.');
  }
  const smokeMode = executionMode === STANDALONE_EXECUTION_MODES.SMOKE;
  const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const runTimestamp = new Date().toISOString().replaceAll(':', '-').replaceAll('.', '-');
  const runId = `first-instance-standalone-deployment${smokeMode ? '-smoke' : ''}`
    + `-${runTimestamp}-${randomBytes(4).toString('hex')}`;
  const resultRoot = smokeMode
    ? path.join(
      projectRoot,
      'results',
      'development',
      'first-instance-standalone-deployment-smoke',
    )
    : path.join(projectRoot, 'results', 'candidate-standalone-deployment');
  const runDirectory = path.join(resultRoot, runId);
  const runStartedMonotonic = performance.now();
  const startedAt = new Date().toISOString();
  let terminationSignal = null;
  let runCompletionCommitted = false;
  let server = null;
  let viteRuntimeGuard = null;
  let browserManager = null;
  let activeTelemetry = null;
  let artifactStore = null;
  let sourceProvenanceStart = null;
  let sourceProvenanceEnd = null;
  let dependencyClosureStart = null;
  let dependencyClosureEnd = null;
  let runtimeAudit = null;
  const completedTrialArtifacts = [];
  const completedSessionArtifacts = [];
  const completedMatrixArtifacts = [];
  const gateArtifacts = [];
  const telemetryArtifacts = [];

  const terminationError = () => {
    const error = new Error(`Run termination requested by ${terminationSignal}.`);
    error.code = 'FIRST_INSTANCE_STANDALONE_TERMINATED';
    return error;
  };
  const requireRunActive = () => {
    if (terminationSignal !== null) throw terminationError();
  };
  const signalHandler = (signal) => {
    if (runCompletionCommitted || terminationSignal !== null) return;
    terminationSignal = signal;
    process.exitCode = signal === 'SIGINT' ? 130 : 143;
    void browserManager?.forceClose(signal).catch(() => undefined);
  };
  const onSigint = () => signalHandler('SIGINT');
  const onSigterm = () => signalHandler('SIGTERM');
  process.on('SIGINT', onSigint);
  process.on('SIGTERM', onSigterm);

  try {
    await mkdir(resultRoot, { recursive: true });
    await mkdir(runDirectory, { recursive: false });
    for (const relativeDirectory of ['trials', 'matrices', 'journal']) {
      await mkdir(path.join(runDirectory, relativeDirectory), { recursive: false });
    }
    artifactStore = new ArtifactStore({ runDirectory, requireRunActive });

    const plan = buildFirstInstanceStandaloneDeploymentPlan({ runId });
    validateFirstInstanceStandaloneDeploymentPlan(plan, { runId });
    const planSha256 = canonicalJsonSha256(plan);
    const selection = selectStandaloneDeploymentExecution(plan, executionMode);
    requireCondition(selection.sessions.length
      === (smokeMode ? 2 : FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSION_COUNT)
      && selection.trials.length
        === (smokeMode ? 4 : FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIAL_COUNT)
      && selection.matrices.length
        === (smokeMode ? 1 : FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_COUNT),
    'Standalone execution selection differs from the frozen scope.', selection);
    const planArtifact = await artifactStore.json('plan.json', {
      schemaVersion: 1,
      kind: 'first-instance-standalone-deployment-execution-plan',
      runId,
      executionMode,
      analysisEligible: selection.analysisEligible,
      planSha256,
      frozenPlan: plan,
      executedMatrixIds: selection.matrices.map(({ matrix }) => matrix.matrixId),
      executedSessionIds: selection.sessions.map((session) => session.sessionId),
      executedTrialIds: selection.trials.map((trial) => trial.trialId),
      smokeBoundary: smokeMode
        ? 'first two fresh sessions and both ordered visibility trials in each'
        : null,
      executionPolicy: {
        retryCount: 0,
        replacementAllowed: false,
        outlierRemovalAllowed: false,
        efficacyStoppingAllowed: false,
        matrixTwoRunsRegardlessOfMatrixOneNumericalOrStableEnvironmentOutcome: true,
        postDisconnectDelayMs: STANDALONE_POST_DISCONNECT_DELAY_MS,
        browserProfilePolicy: 'fresh-playwright-temporary-profile-per-process',
      },
    });
    await artifactStore.journal('plan-committed', { planSha256, artifact: planArtifact });

    [sourceProvenanceStart, dependencyClosureStart] = await Promise.all([
      collectSourceProvenance(projectRoot),
      collectExecutionDependencyClosure(projectRoot),
    ]);
    requireCondition(sourceProvenanceStart.status === 'available'
      && sourceProvenanceStart.captureStable === true
      && sourceProvenanceStart.dirty === false
      && sourceProvenanceStart.packageLockTracked === true,
    'Standalone timing requires clean, stable, available source provenance.',
    sourceProvenanceStart);
    requireCondition(dependencyClosureStart?.schemaVersion === 1
      && dependencyClosureStart?.kind === 'installed-execution-dependency-closure'
      && dependencyClosureStart.fileCount > 0
      && SHA256_PATTERN.test(dependencyClosureStart.sha256),
    'Installed execution-dependency closure is unavailable or invalid.',
    dependencyClosureStart);
    const executionIdentity = executionIdentityReference({
      sourceProvenance: sourceProvenanceStart,
      dependencyClosure: dependencyClosureStart,
    });
    const executionIdentityStartArtifact = await artifactStore.json(
      'execution-identity-start.json',
      {
        schemaVersion: 1,
        kind: 'first-instance-standalone-execution-identity-start',
        runId,
        executionMode,
        planSha256,
        executionIdentity,
        sourceProvenance: sourceProvenanceStart,
        executionDependencyClosure: dependencyClosureStart,
      },
    );
    await artifactStore.journal('execution-identity-start-committed', {
      artifact: executionIdentityStartArtifact,
    });

    const executablePath = await findBrowser();
    requireRunActive();
    viteRuntimeGuard = await createCandidateViteRuntimeGuard(projectRoot);
    server = await createServer({
      root: projectRoot,
      configFile: false,
      resolve: { dedupe: ['three'] },
      server: {
        host: '127.0.0.1',
        port: 0,
        headers: ISOLATION_HEADERS,
      },
      logLevel: 'error',
      ...viteRuntimeGuard.viteConfig,
    });
    await withDeadline(server.listen(), SERVER_OPERATION_TIMEOUT_MS, 'Vite server listen');
    requireRunActive();
    const url = server.resolvedUrls?.local?.[0];
    requireCondition(Boolean(url), 'Vite did not expose a local URL.');
    browserManager = new BrowserLifecycleManager({
      executablePath,
      runStartedMonotonic,
      requireRunActive,
    });

    const globalIdentityTracker = createIdentityTracker();
    const usedNonces = new Set();
    let telemetryContext = Object.freeze({
      trialId: null,
      planIndex: null,
      repetitionIndex: null,
      modeId: STANDALONE_MODE_ID,
      visibilityFraction: null,
      layout: LAYOUT,
      phase: 'not-started',
    });

    for (const matrixSelection of selection.matrices) {
      const { matrix } = matrixSelection;
      const matrixDirectoryName = `matrix-${String(matrix.matrixOrdinal).padStart(2, '0')}`;
      const matrixDirectory = path.join(runDirectory, 'matrices', matrixDirectoryName);
      await mkdir(matrixDirectory, { recursive: false });
      const matrixIdentityTracker = createIdentityTracker();
      const identityTracker = compositeIdentityTracker(
        globalIdentityTracker,
        matrixIdentityTracker,
      );
      const matrixSessionArtifacts = [];
      const matrixTrialArtifacts = [];
      const processComparisons = [];

      process.stdout.write(`[matrix ${matrix.matrixOrdinal}] forced-feature-off gate\n`);
      const gateState = await browserManager.launch({
        role: 'forced-feature-off-gate',
        matrixOrdinal: matrix.matrixOrdinal,
      });
      const gateUrl = `${url}${firstInstanceStandaloneBootSearch({
        laneId: 'portable',
        visibilityOrder: [0.99, 0.2],
      })}`;
      const gatePage = await browserManager.openOnlyPage(gateState, gateUrl);
      const gateResult = await runDisposableForcedFeatureOffGate({
        page: gatePage,
        pageErrors: gateState.errors,
        matrix,
        identityTracker,
        runId,
        executionMode,
        planSha256,
        browserRecord: gateState.record,
        artifactStore,
      });
      const gateBrowserLifecycle = await browserManager.close(gateState);
      requireCondition(gateState.errors.length === 0,
        `Matrix ${matrix.matrixOrdinal} gate emitted a late page error.`, gateState.errors);
      await browserManager.waitAfterDisconnect(gateBrowserLifecycle);
      gateArtifacts.push({
        matrixId: matrix.matrixId,
        artifact: gateResult.artifact,
        browserLifecycle: clone(gateBrowserLifecycle),
      });

      let telemetryRecorder = null;
      let preComputeProcesses = null;
      let postComputeProcesses = null;
      if (!smokeMode) {
        telemetryContext = Object.freeze({
          ...telemetryContext,
          phase: `matrix-${matrix.matrixOrdinal}-telemetry-startup`,
        });
        telemetryRecorder = new NvidiaTelemetryRecorder({
          runId,
          runDirectory: matrixDirectory,
          runStartedMonotonic,
          getContext: () => telemetryContext,
        });
        activeTelemetry = { recorder: telemetryRecorder, matrixId: matrix.matrixId };
        const telemetryStatus = await telemetryRecorder.start();
        requireCondition(telemetryStatus === 'active',
          `Matrix ${matrix.matrixOrdinal} requires an active Nvidia telemetry collector.`, {
            status: telemetryStatus,
            reason: telemetryRecorder.reason,
          });
        preComputeProcesses = await telemetryRecorder.captureComputeSnapshot(
          `matrix-${matrix.matrixOrdinal}-pre-first-session`,
        );
        requireCondition(preComputeProcesses.status === 'available',
          `Matrix ${matrix.matrixOrdinal} pre-session process snapshot is unavailable.`,
          preComputeProcesses);
      }

      for (let sessionExecutionIndex = 0;
        sessionExecutionIndex < matrixSelection.sessions.length;
        sessionExecutionIndex += 1) {
        const canonicalSession = matrixSelection.sessions[sessionExecutionIndex];
        const canonicalTrials = matrixSelection.trials.filter(
          (trial) => trial.sessionId === canonicalSession.sessionId,
        );
        requireCondition(canonicalTrials.length
          === FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIALS_PER_SESSION,
        `Session ${canonicalSession.sessionId} has the wrong execution trial count.`);
        telemetryContext = Object.freeze({
          ...telemetryContext,
          trialId: null,
          planIndex: null,
          repetitionIndex: canonicalSession.globalQuartetIndex,
          visibilityFraction: canonicalSession.visibilityOrder[0],
          phase: 'session-configuration',
        });
        process.stdout.write(
          `[matrix ${matrix.matrixOrdinal} session `
            + `${sessionExecutionIndex + 1}/${matrixSelection.sessions.length}] `
            + `${canonicalSession.assignedLaneId} `
            + `${canonicalSession.visibilityOrder.join('->')}\n`,
        );
        const sessionState = await browserManager.launch({
          role: 'standalone-measured-session',
          matrixOrdinal: matrix.matrixOrdinal,
          session: canonicalSession,
        });
        const sessionUrl = `${url}${firstInstanceStandaloneBootSearch({
          laneId: canonicalSession.assignedLaneId,
          visibilityOrder: canonicalSession.visibilityOrder,
        })}`;
        const page = await browserManager.openOnlyPage(sessionState, sessionUrl);
        const configured = await withDeadline(
          configureStandaloneSession(page, canonicalSession),
          BROWSER_OPERATION_TIMEOUT_MS,
          `${canonicalSession.sessionId} configuration`,
        );
        const baseResourceIdentity = requireConfiguredSession(
          configured,
          canonicalSession,
          canonicalSession.sessionId,
        );
        identityTracker.observe(
          configured.environment,
          configured.workload,
          `${canonicalSession.sessionId} construction`,
        );
        requireCondition(sessionState.errors.length === 0,
          `${canonicalSession.sessionId} emitted construction errors.`, sessionState.errors);

        let visibilitySwitch = null;
        const sessionTrialArtifacts = [];
        for (let trialExecutionIndex = 0;
          trialExecutionIndex < canonicalTrials.length;
          trialExecutionIndex += 1) {
          const canonicalTrial = canonicalTrials[trialExecutionIndex];
          requireCondition(canonicalTrial.visibilityOrderPosition === trialExecutionIndex,
            `${canonicalTrial.trialId} is out of frozen session order.`);
          if (trialExecutionIndex === 1) {
            telemetryContext = Object.freeze({
              ...telemetryContext,
              visibilityFraction: canonicalTrial.visibilityFraction,
              phase: 'untimed-visibility-switch',
            });
            visibilitySwitch = await withDeadline(
              switchStandaloneVisibility(page, canonicalTrial.visibilityFraction),
              BROWSER_OPERATION_TIMEOUT_MS,
              `${canonicalSession.sessionId} visibility switch`,
            );
            requireVisibilitySwitch(
              visibilitySwitch,
              canonicalSession,
              baseResourceIdentity,
              `${canonicalSession.sessionId} visibility switch`,
            );
            identityTracker.observe(
              visibilitySwitch.environment,
              visibilitySwitch.workload,
              `${canonicalSession.sessionId} visibility switch`,
            );
          }
          telemetryContext = Object.freeze({
            ...telemetryContext,
            trialId: canonicalTrial.trialId,
            planIndex: canonicalTrial.planIndex,
            repetitionIndex: canonicalTrial.globalQuartetIndex,
            visibilityFraction: canonicalTrial.visibilityFraction,
            phase: `trial-${canonicalTrial.visibilityExposure}`,
          });
          const trialArtifact = await runStandaloneTrial({
            page,
            pageErrors: sessionState.errors,
            canonicalTrial,
            canonicalSession,
            configured,
            baseResourceIdentity,
            identityTracker,
            usedNonces,
            runId,
            executionMode,
            planSha256,
            browserRecord: sessionState.record,
            executionIdentity,
            artifactStore,
          });
          sessionTrialArtifacts.push(trialArtifact);
          matrixTrialArtifacts.push(trialArtifact);
          completedTrialArtifacts.push(trialArtifact);
          process.stdout.write(`  saved ${trialArtifact.artifact.path}\n`);
        }

        const sessionEnd = await withDeadline(page.evaluate(async () => ({
          selectedConfig: window.__WEBGPU_BENCH__.selectedConfig(),
          environment: window.__WEBGPU_BENCH__.environment,
          strategyLifecycle: window.__WEBGPU_BENCH__.strategyLifecycle,
          strategyDiagnostics: window.__WEBGPU_BENCH__.strategyDiagnostics,
          cacheDiagnostics: window.__WEBGPU_BENCH__.cacheDiagnostics(),
          workload: await window.__WEBGPU_BENCH__.fingerprintWorkload(),
          webgpuUncapturedErrors: window.__WEBGPU_BENCH__.webgpuUncapturedErrors,
          webgpuDeviceLosses: window.__WEBGPU_BENCH__.webgpuDeviceLosses,
        })), BROWSER_OPERATION_TIMEOUT_MS, `${canonicalSession.sessionId} end capture`);
        requireCleanGpuRecords(sessionEnd, `${canonicalSession.sessionId} end`);
        requireCondition(sessionState.errors.length === 0,
          `${canonicalSession.sessionId} emitted page errors.`, sessionState.errors);
        requireWithinSessionIdentity(
          baseResourceIdentity,
          sessionEnd.strategyLifecycle,
          `${canonicalSession.sessionId} end`,
        );
        requireStandaloneLifecycle(
          sessionEnd.strategyLifecycle,
          canonicalSession,
          canonicalTrials.length - 1,
          `${canonicalSession.sessionId} end`,
        );
        requireCondition(sessionEnd.strategyLifecycle.shaderObservationSerial
          === canonicalTrials.length * SHADER_CAPTURE_ROLES.length,
        `${canonicalSession.sessionId} shader serial differs at close.`,
        sessionEnd.strategyLifecycle);
        identityTracker.observe(
          sessionEnd.environment,
          sessionEnd.workload,
          `${canonicalSession.sessionId} end`,
        );
        const browserLifecycle = await browserManager.close(sessionState);
        requireCondition(sessionState.errors.length === 0,
          `${canonicalSession.sessionId} emitted a late page error.`, sessionState.errors);
        await browserManager.waitAfterDisconnect(browserLifecycle);

        let computeProcessEvidence = null;
        if (!smokeMode) {
          telemetryContext = Object.freeze({
            ...telemetryContext,
            trialId: null,
            planIndex: null,
            visibilityFraction: null,
            phase: 'post-session-quiescent-process-snapshot',
          });
          const snapshot = await telemetryRecorder.captureComputeSnapshot(
            `matrix-${matrix.matrixOrdinal}-after-session-`
              + `${canonicalSession.matrixSessionOrdinal}`,
          );
          const comparison = compareComputeProcessIdentitySets(
            preComputeProcesses,
            snapshot,
          );
          computeProcessEvidence = { snapshot, comparison };
          processComparisons.push({
            sessionId: canonicalSession.sessionId,
            snapshot,
            comparison,
          });
          postComputeProcesses = snapshot;
        }

        const sessionBody = {
          schemaVersion: 1,
          kind: 'first-instance-standalone-deployment-session-completion',
          executionMode,
          analysisEligible: !smokeMode,
          runId,
          planSha256,
          capturedAt: new Date().toISOString(),
          canonicalSession,
          sessionNamespace: browserLifecycle.sessionNamespace,
          browserInstanceSerial: browserLifecycle.browserInstanceSerial,
          browserLifecycle,
          sessionConstruction: configured,
          baseResourceIdentity,
          visibilitySwitch,
          endingEvidence: sessionEnd,
          computeProcessEvidence,
          trials: sessionTrialArtifacts,
          pageErrors: clone(sessionState.errors),
        };
        const sessionRelativePath = `matrices/${matrixDirectoryName}/session-`
          + `${String(canonicalSession.matrixSessionOrdinal).padStart(2, '0')}.json`;
        const sessionArtifact = await artifactStore.json(sessionRelativePath, sessionBody);
        const sessionRecord = {
          sessionId: canonicalSession.sessionId,
          globalSessionIndex: canonicalSession.globalSessionIndex,
          matrixSessionIndex: canonicalSession.matrixSessionIndex,
          sessionNamespace: browserLifecycle.sessionNamespace,
          browserInstanceSerial: browserLifecycle.browserInstanceSerial,
          artifact: sessionArtifact,
        };
        matrixSessionArtifacts.push(sessionRecord);
        completedSessionArtifacts.push(sessionRecord);
        await artifactStore.journal('session-committed', sessionRecord);
      }

      let matrixTelemetryRecord;
      if (smokeMode) {
        const exclusion = {
          schemaVersion: 1,
          kind: 'first-instance-standalone-smoke-telemetry-exclusion',
          executionMode,
          analysisEligible: false,
          runId,
          matrixId: matrix.matrixId,
          planSha256,
          status: 'excluded-by-frozen-smoke-policy',
          reason: 'Smoke validates plumbing only and cannot enter analysis or a decision.',
        };
        const artifact = await artifactStore.json(
          `matrices/${matrixDirectoryName}/telemetry-exclusion.json`,
          exclusion,
        );
        matrixTelemetryRecord = { status: 'excluded', artifact };
        telemetryArtifacts.push(matrixTelemetryRecord);
      } else {
        requireCondition(postComputeProcesses !== null,
          `Matrix ${matrix.matrixOrdinal} lacks a post-session process snapshot.`);
        telemetryContext = Object.freeze({
          ...telemetryContext,
          phase: `matrix-${matrix.matrixOrdinal}-telemetry-shutdown`,
        });
        await telemetryRecorder.stop();
        const telemetryReport = telemetryRecorder.report({
          preComputeProcesses,
          postComputeProcesses,
        });
        activeTelemetry = null;
        const adapterAssociation = evaluateLiveFirstInstanceAdapterTelemetryAssociation({
          adapterInfo: matrixIdentityTracker.baselineEnvironment?.adapterInfo,
          telemetryReport,
        });
        const eligibility = telemetryMatrixEligibility({
          report: telemetryReport,
          preComputeProcesses,
          postComputeProcesses,
          processComparisons,
          adapterAssociation,
        });
        const telemetryBody = {
          schemaVersion: 1,
          kind: 'first-instance-standalone-matrix-telemetry-evidence',
          executionMode,
          analysisEligible: true,
          runId,
          matrixId: matrix.matrixId,
          matrixIndex: matrix.matrixIndex,
          matrixOrdinal: matrix.matrixOrdinal,
          planSha256,
          report: telemetryReport,
          preComputeProcesses,
          postComputeProcesses,
          processComparisons,
          adapterAssociation,
          eligibility,
        };
        const artifact = await artifactStore.json(
          `matrices/${matrixDirectoryName}/telemetry.json`,
          telemetryBody,
        );
        const csvArtifact = await artifactStore.existing(
          `matrices/${matrixDirectoryName}/gpu-telemetry.csv`,
          'csv-utf8',
        );
        matrixTelemetryRecord = {
          status: 'captured',
          eligibility,
          artifact,
          csvArtifact,
        };
        telemetryArtifacts.push(matrixTelemetryRecord);
        await artifactStore.journal('matrix-telemetry-committed', {
          matrixId: matrix.matrixId,
          artifact,
          csvArtifact,
          eligibility,
        });
      }

      const matrixIdentity = matrixIdentityTracker.report();
      const matrixCompletionBody = {
        schemaVersion: 1,
        kind: 'first-instance-standalone-deployment-matrix-completion',
        executionMode,
        analysisEligible: !smokeMode,
        runId,
        planSha256,
        canonicalMatrix: matrix,
        capturedAt: new Date().toISOString(),
        forcedFeatureOffGate: gateArtifacts.at(-1),
        sessions: matrixSessionArtifacts,
        trials: matrixTrialArtifacts,
        telemetry: matrixTelemetryRecord,
        identity: matrixIdentity,
        numericalAnalysisInvoked: false,
        numericalDecision: null,
      };
      const matrixArtifact = await artifactStore.json(
        `matrices/${matrixDirectoryName}/matrix-completion.json`,
        matrixCompletionBody,
      );
      const matrixRecord = {
        matrixId: matrix.matrixId,
        matrixIndex: matrix.matrixIndex,
        matrixOrdinal: matrix.matrixOrdinal,
        sessionCount: matrixSessionArtifacts.length,
        trialCount: matrixTrialArtifacts.length,
        telemetry: matrixTelemetryRecord,
        identity: matrixIdentity,
        artifact: matrixArtifact,
      };
      completedMatrixArtifacts.push(matrixRecord);
      await artifactStore.journal('matrix-committed', matrixRecord);
    }

    requireCondition(browserManager.active === null,
      'A browser remained active after standalone capture.');
    requireCondition(completedTrialArtifacts.length === selection.trials.length
      && completedSessionArtifacts.length === selection.sessions.length
      && completedMatrixArtifacts.length === selection.matrices.length,
    'Standalone capture did not persist its exact selected plan.', {
      trials: completedTrialArtifacts.length,
      sessions: completedSessionArtifacts.length,
      matrices: completedMatrixArtifacts.length,
    });
    requireCondition(usedNonces.size === selection.trials.length * SHADER_CAPTURE_ROLES.length,
      'Standalone shader challenges were not globally fresh.', usedNonces.size);

    [sourceProvenanceEnd, dependencyClosureEnd] = await Promise.all([
      collectSourceProvenance(projectRoot),
      collectExecutionDependencyClosure(projectRoot),
    ]);
    const sourceMatched = sourceProvenanceMatches(
      sourceProvenanceStart,
      sourceProvenanceEnd,
    );
    const dependencyMatched = executionDependencyClosuresMatch(
      dependencyClosureStart,
      dependencyClosureEnd,
    );
    const executionIdentityEndArtifact = await artifactStore.json(
      'execution-identity-end.json',
      {
        schemaVersion: 1,
        kind: 'first-instance-standalone-execution-identity-end',
        runId,
        executionMode,
        planSha256,
        sourceProvenance: sourceProvenanceEnd,
        executionDependencyClosure: dependencyClosureEnd,
        sourceMatchesStart: sourceMatched,
        executionDependencyClosureMatchesStart: dependencyMatched,
      },
    );
    await artifactStore.journal('execution-identity-end-committed', {
      artifact: executionIdentityEndArtifact,
      sourceMatched,
      dependencyMatched,
    });

    await withDeadline(server.close(), SERVER_OPERATION_TIMEOUT_MS, 'Vite server close');
    server = null;
    try {
      runtimeAudit = await withDeadline(
        viteRuntimeGuard.finalize(),
        SERVER_OPERATION_TIMEOUT_MS,
        'Vite runtime audit finalization',
      );
    } finally {
      await withDeadline(
        viteRuntimeGuard.dispose(),
        SERVER_OPERATION_TIMEOUT_MS,
        'Vite runtime guard disposal',
      ).catch(() => undefined);
      viteRuntimeGuard = null;
    }
    const expectedEntryDocuments = smokeMode ? 3 : 98;
    requireCondition(runtimeAudit?.entryHtml?.successfulResponseCount
      === expectedEntryDocuments,
    'Runtime served an unexpected number of entry documents.', runtimeAudit?.entryHtml);
    requireCondition(runtimeAudit?.policyId === CANDIDATE_VITE_RUNTIME_POLICY_ID
      && runtimeAudit?.modules?.some(
        (record) => record.sourceRelativePath
          === 'src/strategies/live-first-instance-standalone.js',
      ),
    'Runtime audit lacks the standalone strategy module identity.', runtimeAudit);
    const runtimeAuditArtifact = await artifactStore.json(
      'vite-runtime-audit.json',
      runtimeAudit,
    );
    await artifactStore.journal('vite-runtime-audit-committed', {
      artifact: runtimeAuditArtifact,
      policyId: runtimeAudit.policyId,
      modulesSha256: runtimeAudit.modulesSha256,
    });

    const lifecycleValidation = validateStandaloneBrowserLifecycleChain(
      browserManager.records,
      { requireTerminalDelay: true },
    );
    const expectedBrowserCount = smokeMode ? 3 : 98;
    requireCondition(lifecycleValidation.pass === true
      && browserManager.records.length === expectedBrowserCount,
    'Standalone browser/profile lifecycle chain failed.', {
      expectedBrowserCount,
      observedBrowserCount: browserManager.records.length,
      lifecycleValidation,
    });
    requireCondition(sourceMatched === true,
      'Tracked source changed during standalone capture.', {
        start: sourceProvenanceStart,
        end: sourceProvenanceEnd,
      });
    requireCondition(dependencyMatched === true,
      'Installed dependency bytes changed during standalone capture.', {
        start: dependencyClosureStart,
        end: dependencyClosureEnd,
      });
    const globalIdentity = globalIdentityTracker.report();
    const allMatrixGatesPassed = smokeMode
      ? null
      : standaloneFullEnvironmentGatesPassed(globalIdentity, completedMatrixArtifacts);

    const finalManifest = {
      schemaVersion: 1,
      kind: smokeMode
        ? 'first-instance-standalone-deployment-smoke-capture-manifest'
        : 'first-instance-standalone-deployment-full-capture-manifest',
      executionMode,
      analysisEligible: !smokeMode,
      scope: smokeMode
        ? 'two fresh P/F sessions with both visibility trials; excluded from analysis and decision'
        : 'complete two-matrix candidate capture awaiting independent analysis and verification',
      runId,
      planSha256,
      startedAt,
      completedAt: new Date().toISOString(),
      elapsedMs: performance.now() - runStartedMonotonic,
      browser: {
        executable: path.basename(executablePath),
        arguments: [...BROWSER_ARGS],
        launchApi: 'chromium.launch',
        profilePolicy: 'fresh-playwright-temporary-profile-per-process',
        persistentContext: false,
      },
      fixedWorkload: {
        objectCount: OBJECT_COUNT,
        bucketCount: BUCKET_COUNT,
        layout: LAYOUT,
        visibilityLevels: [0.99, 0.2],
        scenarioSeed: SCENARIO_SEED,
        warmupFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES,
        measuredFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
        measuredBlockSize: STANDALONE_BLOCK_SIZE,
        measuredBlockCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS,
      },
      analysis: {
        invokedByCaptureRunner: false,
        summaryArtifact: null,
        numericalDecision: null,
        nextStep: smokeMode
          ? 'none; smoke artifacts may only be structurally verified'
          : 'run the independent standalone analyzer and verifier',
      },
      executionPolicy: {
        retryCount: 0,
        replacementAllowed: false,
        outlierRemovalAllowed: false,
        efficacyStoppingAllowed: false,
        postDisconnectDelayMs: STANDALONE_POST_DISCONNECT_DELAY_MS,
        oppositeLaneConstructedInMeasuredSessions: false,
      },
      identity: {
        planSha256,
        executionIdentity,
        sourceProvenanceStart,
        sourceProvenanceEnd,
        sourceMatched,
        executionDependencyClosureStart: dependencyClosureStart,
        executionDependencyClosureEnd: dependencyClosureEnd,
        executionDependencyClosureMatched: dependencyMatched,
        viteRuntimePolicyId: runtimeAudit.policyId,
        viteRuntimeModulesSha256: runtimeAudit.modulesSha256,
        globalEnvironmentAndWorkload: globalIdentity,
      },
      browserLifecycles: clone(browserManager.records),
      lifecycleValidation,
      telemetryPolicy: smokeMode
        ? {
          status: 'excluded-by-frozen-smoke-policy',
          analysisEligible: false,
        }
        : {
          status: 'captured-per-matrix',
          provider: 'nvidia-smi',
          intervalMs: 250,
          processSnapshotPolicy: 'pre-first-session-and-after-every-browser-disconnect',
          allMatrixGatesPassed,
        },
      artifacts: {
        plan: planArtifact,
        executionIdentityStart: executionIdentityStartArtifact,
        executionIdentityEnd: executionIdentityEndArtifact,
        forcedFeatureOffGates: gateArtifacts,
        trials: completedTrialArtifacts,
        sessions: completedSessionArtifacts,
        matrices: completedMatrixArtifacts,
        telemetry: telemetryArtifacts,
        viteRuntimeAudit: runtimeAuditArtifact,
        journal: clone(artifactStore.journalArtifacts),
      },
      completion: {
        frozenMatrixCount: plan.matrixCount,
        frozenSessionCount: plan.sessionCount,
        frozenTrialCount: plan.trialCount,
        executedMatrixCount: selection.matrices.length,
        executedSessionCount: selection.sessions.length,
        executedTrialCount: selection.trials.length,
        persistedMatrixCount: completedMatrixArtifacts.length,
        persistedSessionCount: completedSessionArtifacts.length,
        persistedTrialCount: completedTrialArtifacts.length,
        shaderChallengeCount: usedNonces.size,
        browserProcessCount: browserManager.records.length,
        entryDocumentCount: runtimeAudit.entryHtml.successfulResponseCount,
        allBrowsersClosed: browserManager.active === null,
        overlapDetected: false,
      },
    };
    const manifestArtifact = await artifactStore.json('manifest.json', finalManifest);
    runCompletionCommitted = true;
    process.stdout.write(
      `Standalone ${smokeMode ? 'smoke' : 'full capture'} complete.\n`
        + `  directory: ${runDirectory}\n`
        + `  manifest: ${path.join(runDirectory, manifestArtifact.path)}\n`
        + '  numerical analysis: not invoked by the capture runner\n',
    );
    return Object.freeze({ runDirectory, runId, manifestArtifact });
  } catch (error) {
    const abortedBrowserLifecycle = await browserManager?.forceClose(
      terminationSignal ?? 'failed-run',
    ).catch(() => null) ?? null;
    let partialTelemetry = null;
    if (activeTelemetry?.recorder) {
      try {
        const report = await activeTelemetry.recorder.stop();
        partialTelemetry = {
          matrixId: activeTelemetry.matrixId,
          report,
        };
      } catch {
        partialTelemetry = {
          matrixId: activeTelemetry.matrixId,
          report: null,
        };
      }
      activeTelemetry = null;
    }
    if (server !== null) {
      await withDeadline(
        server.close(),
        SERVER_OPERATION_TIMEOUT_MS,
        'failed-run Vite server close',
      ).catch(() => undefined);
      server = null;
    }
    if (viteRuntimeGuard !== null) {
      await withDeadline(
        viteRuntimeGuard.dispose(),
        SERVER_OPERATION_TIMEOUT_MS,
        'failed-run Vite runtime guard disposal',
      ).catch(() => undefined);
      viteRuntimeGuard = null;
    }
    if (artifactStore !== null) {
      const filename = terminationSignal === null ? 'failure.json' : 'interruption.json';
      await artifactStore.json(filename, {
        schemaVersion: 1,
        kind: terminationSignal === null
          ? 'first-instance-standalone-deployment-capture-failure'
          : 'first-instance-standalone-deployment-capture-interruption',
        executionMode,
        analysisEligible: false,
        scope: smokeMode
          ? 'failed smoke; excluded from analysis and decision'
          : 'failed closed; no retry or replacement attempted',
        runId,
        failedAt: new Date().toISOString(),
        signal: terminationSignal,
        error: {
          name: error?.name ?? 'Error',
          message: error?.message ?? String(error),
          stack: error?.stack ?? null,
        },
        completedTrialArtifacts,
        completedSessionArtifacts,
        completedMatrixArtifacts,
        gateArtifacts,
        telemetryArtifacts,
        partialTelemetry,
        abortedBrowserLifecycle,
        browserLifecycles: clone(browserManager?.records ?? []),
        sourceProvenanceStart,
        sourceProvenanceEnd,
        executionDependencyClosureStart: dependencyClosureStart,
        executionDependencyClosureEnd: dependencyClosureEnd,
        viteRuntimeAudit: runtimeAudit,
        journalArtifacts: clone(artifactStore.journalArtifacts),
      }, { allowDuringTermination: true }).catch(() => undefined);
    }
    if (terminationSignal === null) throw error;
    return null;
  } finally {
    process.removeListener('SIGINT', onSigint);
    process.removeListener('SIGTERM', onSigterm);
    if (!runCompletionCommitted) {
      await browserManager?.forceClose('final-cleanup').catch(() => undefined);
    }
  }
}

const invokedPath = process.argv[1] === undefined
  ? null
  : pathToFileURL(path.resolve(process.argv[1])).href;
if (invokedPath === import.meta.url) {
  const executionMode = parseRunnerArguments(process.argv.slice(2));
  await runFirstInstanceStandaloneDeployment({ executionMode });
}
