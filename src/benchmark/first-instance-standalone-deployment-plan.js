import { FIRST_INSTANCE_LIVE_CROSSOVER_LANES } from './plan.js';

export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_KIND =
  'first-instance-standalone-deployment-candidate';
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_COUNT = 2;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_QUARTETS_PER_MATRIX = 12;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSIONS_PER_QUARTET = 4;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSIONS_PER_MATRIX = 48;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIALS_PER_SESSION = 2;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIALS_PER_QUARTET = 8;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIALS_PER_MATRIX = 96;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSION_COUNT = 96;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIAL_COUNT = 192;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES = 320;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES = 480;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE = 8;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS = 60;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MAXIMUM_TIMESTAMP_QUANTUM_NS = 1_000;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMESTAMP_RESOLUTION_MODE =
  'single-post-measurement';
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_BOUNDARY_RESOLVE_BATCH_COUNT = 0;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_POST_MEASUREMENT_RESOLVE_BATCH_COUNT = 1;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_QUERIES_PER_TIMESTAMP_UID = 2;
export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_REQUIRED_QUERIES_PER_TYPE =
  (FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES
    + FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES)
  * FIRST_INSTANCE_STANDALONE_DEPLOYMENT_QUERIES_PER_TIMESTAMP_UID;

if (FIRST_INSTANCE_STANDALONE_DEPLOYMENT_REQUIRED_QUERIES_PER_TYPE !== 1_600
  || FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_BOUNDARY_RESOLVE_BATCH_COUNT !== 0
  || FIRST_INSTANCE_STANDALONE_DEPLOYMENT_POST_MEASUREMENT_RESOLVE_BATCH_COUNT !== 1) {
  throw new Error('Standalone timestamp-resolution constants violate the frozen protocol.');
}

const [PORTABLE, FEATURE] = FIRST_INSTANCE_LIVE_CROSSOVER_LANES;
const HIGH_VISIBILITY = 0.99;
const LOW_VISIBILITY = 0.2;

export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS = Object.freeze([
  HIGH_VISIBILITY,
  LOW_VISIBILITY,
]);

export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_LANE_ORDERS = Object.freeze({
  A: Object.freeze([PORTABLE, FEATURE, FEATURE, PORTABLE]),
  B: Object.freeze([FEATURE, PORTABLE, PORTABLE, FEATURE]),
});

export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_MASKS = Object.freeze({
  X: Object.freeze(['H', 'H', 'L', 'L']),
  Y: Object.freeze(['L', 'L', 'H', 'H']),
});

const MATRIX_ONE_QUARTET_CODES = Object.freeze([
  'AX', 'AY', 'BX', 'BY',
  'AX', 'AY', 'BX', 'BY',
  'AX', 'AY', 'BX', 'BY',
]);

export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_QUARTET_CODES = Object.freeze([
  MATRIX_ONE_QUARTET_CODES,
  Object.freeze([...MATRIX_ONE_QUARTET_CODES].reverse()),
]);

function deepFreeze(value) {
  if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const nested of Object.values(value)) deepFreeze(nested);
    Object.freeze(value);
  }
  return value;
}

export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS = deepFreeze({
  highVisibilityGpuPassTotal: {
    maximumDeltaMs: -0.10,
    maximumDeltaPercent: -5,
    minimumNegativeQuartets: 10,
    quartetCount: 12,
  },
  highVisibilityGpuRender: {
    maximumDeltaMs: -0.10,
    maximumDeltaPercent: -5,
    minimumNegativeQuartets: 10,
    quartetCount: 12,
  },
  lowVisibilityGpuPassTotal: {
    strictUpperDeltaMs: 0.02,
    strictUpperDeltaPercent: 5,
    minimumQuartetsBelowEachUpperBound: 10,
    quartetCount: 12,
  },
  pairedHighMinusLowGpuPassTotal: {
    maximumDeltaMs: -0.05,
  },
  nuisanceInteraction: {
    strictMaximumAbsoluteDeltaMs: 0.10,
    strictMaximumAbsoluteDeltaPercentagePoints: 5,
  },
  conditionBlindDrift: {
    strictMaximumAbsoluteDeltaMs: 0.10,
    strictMaximumAbsoluteDeltaPercent: 5,
  },
  maximumTimestampQuantumNs: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MAXIMUM_TIMESTAMP_QUANTUM_NS,
});

const VISIBILITY_ORDERS = Object.freeze({
  H: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS,
  L: Object.freeze([...FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS].reverse()),
});

function validateRunId(runId) {
  if (typeof runId !== 'string' || runId.trim() === '') {
    throw new TypeError('Standalone deployment runId must be a nonempty string.');
  }
  if (runId.length > 160 || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(runId)) {
    throw new RangeError(
      'Standalone deployment runId must be at most 160 portable identifier characters.',
    );
  }
}

function validateBuildOptions(options) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('Standalone deployment plan options must be an object.');
  }
  const keys = Object.keys(options);
  if (keys.length !== 1 || keys[0] !== 'runId') {
    throw new Error('Standalone deployment plan accepts exactly the runId option.');
  }
  validateRunId(options.runId);
}

function pad(value, width) {
  return String(value).padStart(width, '0');
}

function visibilityId(visibilityFraction) {
  return visibilityFraction === HIGH_VISIBILITY ? 'v99' : 'v20';
}

function createCanonicalPlan(runId) {
  const matrices = [];
  const quartets = [];
  const sessions = [];
  const trials = [];

  for (let matrixIndex = 0;
    matrixIndex < FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_COUNT;
    matrixIndex += 1) {
    const matrixOrdinal = matrixIndex + 1;
    const matrixId = `${runId}-m${matrixOrdinal}`;
    const quartetCodes = FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_QUARTET_CODES[
      matrixIndex
    ];
    const matrixQuartetIds = [];
    const matrixSessionIds = [];
    const matrixTrialIds = [];

    for (let quartetIndex = 0;
      quartetIndex < FIRST_INSTANCE_STANDALONE_DEPLOYMENT_QUARTETS_PER_MATRIX;
      quartetIndex += 1) {
      const quartetOrdinal = quartetIndex + 1;
      const quartetCode = quartetCodes[quartetIndex];
      const laneOrderId = quartetCode[0];
      const visibilityMaskId = quartetCode[1];
      const laneOrder = FIRST_INSTANCE_STANDALONE_DEPLOYMENT_LANE_ORDERS[laneOrderId];
      const visibilityOrderMask =
        FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_MASKS[visibilityMaskId];
      const globalQuartetIndex = (
        matrixIndex * FIRST_INSTANCE_STANDALONE_DEPLOYMENT_QUARTETS_PER_MATRIX
      ) + quartetIndex;
      const quartetId = `${matrixId}-q${pad(quartetOrdinal, 2)}-${quartetCode}`;
      const quartetSessionIds = [];
      const quartetTrialIds = [];

      for (let sessionPosition = 0;
        sessionPosition < FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSIONS_PER_QUARTET;
        sessionPosition += 1) {
        const quartetSessionOrdinal = sessionPosition + 1;
        const matrixSessionIndex = (
          quartetIndex * FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSIONS_PER_QUARTET
        ) + sessionPosition;
        const globalSessionIndex = (
          matrixIndex * FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSIONS_PER_MATRIX
        ) + matrixSessionIndex;
        const sessionId = `${quartetId}-s${quartetSessionOrdinal}`;
        const assignedLaneId = laneOrder[sessionPosition];
        const absentLaneId = assignedLaneId === PORTABLE ? FEATURE : PORTABLE;
        const visibilityOrderId = visibilityOrderMask[sessionPosition];
        const visibilityOrder = VISIBILITY_ORDERS[visibilityOrderId];
        const sessionTrialIds = [];

        for (let sessionTrialIndex = 0;
          sessionTrialIndex < FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIALS_PER_SESSION;
          sessionTrialIndex += 1) {
          const visibilityFraction = visibilityOrder[sessionTrialIndex];
          const matrixTrialIndex = (
            matrixSessionIndex * FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIALS_PER_SESSION
          ) + sessionTrialIndex;
          const planIndex = (
            matrixIndex * FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIALS_PER_MATRIX
          ) + matrixTrialIndex;
          const trialId = `${sessionId}-t${sessionTrialIndex + 1}`
            + `-${visibilityId(visibilityFraction)}`;
          sessionTrialIds.push(trialId);
          quartetTrialIds.push(trialId);
          matrixTrialIds.push(trialId);
          trials.push({
            schemaVersion: 1,
            kind: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_KIND,
            trialId,
            planIndex,
            matrixIndex,
            matrixOrdinal,
            matrixId,
            matrixTrialIndex,
            matrixTrialOrdinal: matrixTrialIndex + 1,
            globalQuartetIndex,
            quartetIndex,
            quartetOrdinal,
            quartetId,
            quartetCode,
            quartetTrialIndex: (
              sessionPosition * FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIALS_PER_SESSION
            ) + sessionTrialIndex,
            globalSessionIndex,
            matrixSessionIndex,
            sessionPosition,
            sessionId,
            assignedLaneId,
            absentLaneId,
            visibilityOrderId,
            visibilityOrder: [...visibilityOrder],
            visibilityOrderPosition: sessionTrialIndex,
            visibilityExposure: sessionTrialIndex === 0 ? 'first' : 'second',
            visibilityFraction,
            warmupFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES,
            measuredFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
            measuredBlockSize: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE,
            measuredBlockCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS,
          });
        }

        quartetSessionIds.push(sessionId);
        matrixSessionIds.push(sessionId);
        sessions.push({
          schemaVersion: 1,
          kind: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_KIND,
          sessionId,
          globalSessionIndex,
          globalSessionOrdinal: globalSessionIndex + 1,
          matrixIndex,
          matrixOrdinal,
          matrixId,
          matrixSessionIndex,
          matrixSessionOrdinal: matrixSessionIndex + 1,
          globalQuartetIndex,
          quartetIndex,
          quartetOrdinal,
          quartetId,
          quartetCode,
          sessionPosition,
          quartetSessionOrdinal,
          assignedLaneId,
          absentLaneId,
          visibilityOrderId,
          visibilityOrder: [...visibilityOrder],
          trialIds: sessionTrialIds,
          freshBrowserProcessRequired: true,
          freshPageContextRendererAdapterDeviceRequired: true,
          previousBrowserDisconnectRequired: true,
          selectedLaneConstructionCount: 1,
          absentLaneConstructionCount: 0,
          reuseSelectedLaneAcrossVisibilityTrialsRequired: true,
        });
      }

      matrixQuartetIds.push(quartetId);
      quartets.push({
        schemaVersion: 1,
        kind: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_KIND,
        quartetId,
        globalQuartetIndex,
        globalQuartetOrdinal: globalQuartetIndex + 1,
        matrixIndex,
        matrixOrdinal,
        matrixId,
        quartetIndex,
        quartetOrdinal,
        quartetCode,
        laneOrderId,
        laneOrder: [...laneOrder],
        visibilityMaskId,
        visibilityOrderMask: [...visibilityOrderMask],
        sessionIds: quartetSessionIds,
        trialIds: quartetTrialIds,
      });
    }

    matrices.push({
      schemaVersion: 1,
      kind: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_KIND,
      matrixId,
      matrixIndex,
      matrixOrdinal,
      quartetCodes: [...quartetCodes],
      quartetIds: matrixQuartetIds,
      sessionIds: matrixSessionIds,
      trialIds: matrixTrialIds,
    });
  }

  return {
    schemaVersion: 1,
    kind: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_KIND,
    runId,
    matrixCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_COUNT,
    quartetCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MATRIX_COUNT
      * FIRST_INSTANCE_STANDALONE_DEPLOYMENT_QUARTETS_PER_MATRIX,
    sessionCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_SESSION_COUNT,
    trialCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIAL_COUNT,
    visibilityLevels: [...FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS],
    timing: {
      warmupFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_FRAMES,
      measuredFrames: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
      measuredBlockSize: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE,
      measuredBlockCount: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS,
      timestampResolution: {
        mode: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMESTAMP_RESOLUTION_MODE,
        warmupBoundaryResolveBatchCount:
          FIRST_INSTANCE_STANDALONE_DEPLOYMENT_WARMUP_BOUNDARY_RESOLVE_BATCH_COUNT,
        postMeasurementResolveBatchCount:
          FIRST_INSTANCE_STANDALONE_DEPLOYMENT_POST_MEASUREMENT_RESOLVE_BATCH_COUNT,
        queriesPerTimestampUid:
          FIRST_INSTANCE_STANDALONE_DEPLOYMENT_QUERIES_PER_TIMESTAMP_UID,
        requiredQueriesPerType:
          FIRST_INSTANCE_STANDALONE_DEPLOYMENT_REQUIRED_QUERIES_PER_TYPE,
      },
    },
    thresholds: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_THRESHOLDS,
    matrices,
    quartets,
    sessions,
    trials,
  };
}

function assertExactValue(actual, expected, path) {
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || actual.length !== expected.length) {
      throw new Error(`Standalone deployment plan differs at ${path}.`);
    }
    for (let index = 0; index < expected.length; index += 1) {
      assertExactValue(actual[index], expected[index], `${path}[${index}]`);
    }
    return;
  }
  if (expected !== null && typeof expected === 'object') {
    if (actual === null || typeof actual !== 'object' || Array.isArray(actual)) {
      throw new Error(`Standalone deployment plan differs at ${path}.`);
    }
    const actualKeys = Object.keys(actual);
    const expectedKeys = Object.keys(expected);
    if (actualKeys.length !== expectedKeys.length
      || expectedKeys.some((key, index) => actualKeys[index] !== key)) {
      throw new Error(`Standalone deployment plan differs at ${path} keys.`);
    }
    for (const key of expectedKeys) {
      assertExactValue(actual[key], expected[key], `${path}.${key}`);
    }
    return;
  }
  if (!Object.is(actual, expected)) {
    throw new Error(`Standalone deployment plan differs at ${path}.`);
  }
}

export function validateFirstInstanceStandaloneDeploymentPlan(plan, options) {
  validateBuildOptions(options);
  if (plan === null || typeof plan !== 'object' || Array.isArray(plan)) {
    throw new TypeError('Standalone deployment plan must be an object.');
  }
  const expected = createCanonicalPlan(options.runId);
  assertExactValue(plan, expected, 'plan');
  return true;
}

export function buildFirstInstanceStandaloneDeploymentPlan(options) {
  validateBuildOptions(options);
  const plan = createCanonicalPlan(options.runId);
  validateFirstInstanceStandaloneDeploymentPlan(plan, options);
  return deepFreeze(plan);
}
