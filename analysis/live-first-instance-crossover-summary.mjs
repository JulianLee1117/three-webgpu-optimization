import {
  FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE,
  FIRST_INSTANCE_LIVE_CROSSOVER_HISTORY_TRIPLE_KEYS,
  FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_BLOCKS,
  FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_LIVE_CROSSOVER_TRANSITION_KEYS,
  FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
  firstInstanceLiveCrossoverFrame,
} from '../src/benchmark/first-instance-live-crossover-schedule.js';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_LANES,
  FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
  FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS,
  FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS,
  buildFirstInstanceLiveCrossoverPlan,
} from '../src/benchmark/plan.js';

const [PORTABLE, FEATURE] = FIRST_INSTANCE_LIVE_CROSSOVER_LANES;
const [HIGH_VISIBILITY, LOW_VISIBILITY] =
  FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS;
const OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;
const HALF_BLOCKS = FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_BLOCKS / 2;
const QUARTER_BLOCKS = FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_BLOCKS / 4;
const MATERIAL_MS = 0.10;
const MATERIAL_PERCENT = 5;
const LOW_REGRESSION_MS = 0.02;
const LOW_REGRESSION_PERCENT = 5;
const PAIRED_MATERIAL_MS = 0.05;
const DRIFT_MS = 0.10;
const DRIFT_PERCENT = 5;
const REQUIRED_SIGN_COUNT = 10;
const TOTAL_EQUALITY_TOLERANCE_MS = 1e-9;
const VALIDATION_KIND = 'first-instance-live-crossover-exact-paired-snapshots';
const EXPECTED_TRIALS = FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS
  * FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS.length;
const EXPECTED_ROWS = EXPECTED_TRIALS * FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES;
const EXPECTED_TIMED_FRAMES = FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES
  + FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES;
const MAX_SERIAL_BASE = Number.MAX_SAFE_INTEGER - EXPECTED_TIMED_FRAMES;

const METRICS = Object.freeze({
  gpuPassTotal: Object.freeze({ field: 'gpuPassTotalMs', zeroDenominatorAllowed: false }),
  gpuRender: Object.freeze({ field: 'gpuRenderMs', zeroDenominatorAllowed: false }),
  gpuCompute: Object.freeze({ field: 'gpuComputeMs', zeroDenominatorAllowed: false }),
});

function reject(message) {
  throw new Error(`Live first-instance crossover analysis rejected: ${message}`);
}

function requireRecord(value, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    reject(`${label} must be an object.`);
  }
  return value;
}

function finiteNumber(value, label, {
  minimum = -Infinity,
  maximum = Infinity,
  exclusiveMinimum = false,
} = {}) {
  if (!Number.isFinite(value)
    || (exclusiveMinimum ? value <= minimum : value < minimum)
    || value > maximum) {
    reject(
      `${label} must be a finite number ${exclusiveMinimum ? '>' : '>='} ${minimum}`
        + ` and <= ${maximum}.`,
    );
  }
  return value;
}

function integer(value, label, { minimum = -Infinity, maximum = Infinity } = {}) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    reject(`${label} must be a safe integer in [${minimum}, ${maximum}].`);
  }
  return value;
}

function nonemptyString(value, label) {
  if (typeof value !== 'string' || value.length === 0) {
    reject(`${label} must be a nonempty string.`);
  }
  return value;
}

function scalarIdentity(value, label) {
  if (typeof value === 'string' && value.length > 0) return value;
  if (Number.isSafeInteger(value) && value >= 0) return value;
  reject(`${label} must be a nonempty string or nonnegative safe integer.`);
}

function requireInvariant(actual, expected, label) {
  if (actual !== expected) {
    reject(`${label} is ${JSON.stringify(actual)}; expected ${JSON.stringify(expected)}.`);
  }
}

function finiteDerived(value, label) {
  if (!Number.isFinite(value)) reject(`derived ${label} is non-finite.`);
  return value;
}

function mean(values, label = 'sample') {
  if (values.length === 0) reject(`cannot compute ${label} mean from an empty sample.`);
  const result = values.reduce((sum, value) => sum + value, 0) / values.length;
  if (!Number.isFinite(result)) reject(`derived ${label} mean is non-finite.`);
  return result;
}

function median(values, label = 'sample') {
  if (values.length === 0) reject(`cannot compute ${label} median from an empty sample.`);
  if (values.some((value) => !Number.isFinite(value))) {
    reject(`cannot compute ${label} median containing a non-finite value.`);
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const result = sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
  if (!Number.isFinite(result)) reject(`derived ${label} median is non-finite.`);
  return result;
}

function sameArray(left, right) {
  return left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function parseLaneOrder(value, label) {
  const order = Array.isArray(value)
    ? [...value]
    : typeof value === 'string'
      ? value.split('|')
      : null;
  if (order === null
    || order.length !== FIRST_INSTANCE_LIVE_CROSSOVER_LANES.length
    || new Set(order).size !== FIRST_INSTANCE_LIVE_CROSSOVER_LANES.length
    || FIRST_INSTANCE_LIVE_CROSSOVER_LANES.some((laneId) => !order.includes(laneId))) {
    reject(`${label} must be the exact portable/feature lane permutation.`);
  }
  return order;
}

function lanePhysicalOrderForRow(row, label) {
  if (row.plannedLanePhysicalOrder === undefined
    || row.plannedLanePhysicalOrder === null) {
    reject(`${label}.plannedLanePhysicalOrder is required.`);
  }
  const planned = parseLaneOrder(
    row.plannedLanePhysicalOrder,
    `${label}.plannedLanePhysicalOrder`,
  );
  if (row.lanePhysicalOrder !== undefined && row.lanePhysicalOrder !== null) {
    const observed = parseLaneOrder(row.lanePhysicalOrder, `${label}.lanePhysicalOrder`);
    if (!sameArray(planned, observed)) {
      reject(`${label} has inconsistent lanePhysicalOrder fields.`);
    }
  }
  return planned;
}

function visibilityForRow(row, label) {
  const values = [row.visibilityFraction, row.targetVisibilityFraction]
    .filter((value) => value !== undefined && value !== null);
  if (values.length === 0) {
    reject(`${label} must provide visibilityFraction or targetVisibilityFraction.`);
  }
  for (const value of values) finiteNumber(value, `${label}.visibilityFraction`, {
    minimum: 0,
    maximum: 1,
    exclusiveMinimum: true,
  });
  if (values.some((value) => value !== values[0])) {
    reject(`${label} has inconsistent visibility fields.`);
  }
  if (!FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS.includes(values[0])) {
    reject(`${label}.visibilityFraction must be exactly 0.99 or 0.2.`);
  }
  return values[0];
}

function parseVisibilityOrder(value, label) {
  const order = Array.isArray(value)
    ? [...value]
    : typeof value === 'string'
      ? value.split('|').map(Number)
      : null;
  if (order === null
    || order.length !== FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS.length
    || order.some(
      (visibility) => !FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS.includes(visibility),
    )
    || new Set(order).size !== FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS.length) {
    reject(`${label} must be the exact 0.99/0.2 visibility permutation.`);
  }
  return order;
}

function visibilityOrderForRow(row, label) {
  const entries = [
    ['visibilityOrder', row.visibilityOrder],
    ['plannedVisibilityOrder', row.plannedVisibilityOrder],
  ].filter(([, value]) => value !== undefined && value !== null);
  if (entries.length === 0) reject(`${label} must provide a visibility order.`);
  const orders = entries.map(([name, value]) => parseVisibilityOrder(value, `${label}.${name}`));
  if (orders.some((order) => !sameArray(order, orders[0]))) {
    reject(`${label} has inconsistent visibility-order fields.`);
  }
  return orders[0];
}

function singletonOrder(value, expected, label) {
  const order = Array.isArray(value)
    ? [...value]
    : typeof value === 'string'
      ? value.split('|')
      : null;
  if (order === null || order.length !== 1 || order[0] !== expected) {
    reject(`${label} must contain exactly ${JSON.stringify(expected)}.`);
  }
}

function normalizeRow(row, globalIndex, seenGpuFrameIds) {
  const label = `row ${globalIndex}`;
  requireRecord(row, label);
  const planIndex = integer(row.planIndex, `${label}.planIndex`, {
    minimum: 0,
    maximum: EXPECTED_TRIALS - 1,
  });
  const repetitionIndex = integer(row.repetitionIndex, `${label}.repetitionIndex`, {
    minimum: 0,
    maximum: FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS - 1,
  });
  const frameIndex = integer(row.frameIndex, `${label}.frameIndex`, {
    minimum: 0,
    maximum: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES - 1,
  });
  const phaseFrameIndex = integer(row.phaseFrameIndex, `${label}.phaseFrameIndex`, {
    minimum: 0,
    maximum: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES - 1,
  });
  const visibilityFraction = visibilityForRow(row, label);
  const visibilityOrder = visibilityOrderForRow(row, label);
  const lanePhysicalOrder = lanePhysicalOrderForRow(row, label);
  const gpuFrameId = integer(row.gpuFrameId, `${label}.gpuFrameId`, { minimum: 0 });
  if (seenGpuFrameIds.has(gpuFrameId)) reject(`${label} duplicates gpuFrameId ${gpuFrameId}.`);
  seenGpuFrameIds.add(gpuFrameId);

  const runId = nonemptyString(row.runId, `${label}.runId`);
  requireInvariant(row.schemaVersion, 2, `${label}.schemaVersion`);
  requireInvariant(row.modeId, FIRST_INSTANCE_LIVE_CROSSOVER_MODE, `${label}.modeId`);
  requireInvariant(row.modeOrderPosition, 0, `${label}.modeOrderPosition`);
  const modeOrders = [
    ['modeOrder', row.modeOrder],
    ['plannedModeOrder', row.plannedModeOrder],
  ].filter(([, value]) => value !== undefined && value !== null);
  if (modeOrders.length === 0) reject(`${label} must provide a mode order.`);
  for (const [name, value] of modeOrders) {
    singletonOrder(value, FIRST_INSTANCE_LIVE_CROSSOVER_MODE, `${label}.${name}`);
  }
  requireInvariant(row.scenarioLayout ?? row.layout, 'baseline', `${label}.scenarioLayout`);
  if (row.scenarioLayout !== undefined && row.layout !== undefined) {
    requireInvariant(row.layout, row.scenarioLayout, `${label}.layout`);
  }
  requireInvariant(row.layoutOrderPosition, 0, `${label}.layoutOrderPosition`);
  const layoutOrders = [
    ['layoutOrder', row.layoutOrder],
    ['plannedLayoutOrder', row.plannedLayoutOrder],
  ].filter(([, value]) => value !== undefined && value !== null);
  if (layoutOrders.length === 0) reject(`${label} must provide a layout order.`);
  for (const [name, value] of layoutOrders) {
    singletonOrder(value, 'baseline', `${label}.${name}`);
  }
  requireInvariant(row.objectCount, OBJECT_COUNT, `${label}.objectCount`);
  requireInvariant(row.bucketCount, BUCKET_COUNT, `${label}.bucketCount`);
  requireInvariant(row.protocolWarmupFrames, FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES,
    `${label}.protocolWarmupFrames`);
  requireInvariant(row.protocolMeasuredFrames, FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    `${label}.protocolMeasuredFrames`);
  requireInvariant(row.usesCompute, true, `${label}.usesCompute`);
  requireInvariant(row.configuredDrawCommands, BUCKET_COUNT, `${label}.configuredDrawCommands`);
  requireInvariant(row.configuredRenderObjects, 1, `${label}.configuredRenderObjects`);
  requireInvariant(row.configuredComputeDispatches, 2,
    `${label}.configuredComputeDispatches`);
  requireInvariant(row.configuredComputeSubmissions, 1,
    `${label}.configuredComputeSubmissions`);
  requireInvariant(row.validationPass, true, `${label}.validationPass`);
  requireInvariant(row.validationKind, VALIDATION_KIND, `${label}.validationKind`);
  requireInvariant(row.timestampAvailable, true, `${label}.timestampAvailable`);
  requireInvariant(row.gpuComputeTimestampUidCount, 1,
    `${label}.gpuComputeTimestampUidCount`);
  requireInvariant(row.expectedComputeTimestampUidCount, 1,
    `${label}.expectedComputeTimestampUidCount`);
  requireInvariant(row.gpuRenderTimestampUidCount, 1,
    `${label}.gpuRenderTimestampUidCount`);
  requireInvariant(row.expectedRenderTimestampUidCount, 1,
    `${label}.expectedRenderTimestampUidCount`);

  const expectedVisibleCount = Math.round(OBJECT_COUNT * visibilityFraction);
  requireInvariant(row.expectedVisibleCount, expectedVisibleCount, `${label}.expectedVisibleCount`);
  requireInvariant(row.configuredSubmittedInstances, expectedVisibleCount,
    `${label}.configuredSubmittedInstances`);

  const gpuComputeMs = finiteNumber(row.gpuComputeMs, `${label}.gpuComputeMs`, {
    minimum: 0,
    exclusiveMinimum: true,
  });
  const gpuRenderMs = finiteNumber(row.gpuRenderMs, `${label}.gpuRenderMs`, {
    minimum: 0,
    exclusiveMinimum: true,
  });
  const gpuPassTotalMs = finiteNumber(row.gpuPassTotalMs, `${label}.gpuPassTotalMs`, {
    minimum: 0,
    exclusiveMinimum: true,
  });
  const reconstructedTotal = gpuComputeMs + gpuRenderMs;
  if (!Number.isFinite(reconstructedTotal)) {
    reject(`${label} compute-plus-render total is non-finite.`);
  }
  if (Math.abs(gpuPassTotalMs - reconstructedTotal) > TOTAL_EQUALITY_TOLERANCE_MS) {
    reject(
      `${label}.gpuPassTotalMs differs from gpuComputeMs + gpuRenderMs by more than `
        + `${TOTAL_EQUALITY_TOLERANCE_MS} ms.`,
    );
  }

  const commandBufferId = scalarIdentity(row.commandBufferId, `${label}.commandBufferId`);
  const portableCommandBufferIdAtTimingStart = scalarIdentity(
    row.portableCommandBufferIdAtTimingStart,
    `${label}.portableCommandBufferIdAtTimingStart`,
  );
  const featureCommandBufferIdAtTimingStart = scalarIdentity(
    row.featureCommandBufferIdAtTimingStart,
    `${label}.featureCommandBufferIdAtTimingStart`,
  );
  if (portableCommandBufferIdAtTimingStart === featureCommandBufferIdAtTimingStart) {
    reject(`${label} portable and feature command-buffer identities must be distinct.`);
  }
  for (const [field, value] of [
    ['commandRecordBase', row.commandRecordBase],
    ['commandByteBase', row.commandByteBase],
    ['commandSegmentIndex', row.commandSegmentIndex],
  ]) {
    if (value !== undefined && value !== null) {
      requireInvariant(integer(value, `${label}.${field}`, { minimum: 0 }), 0,
        `${label}.${field}`);
    }
  }

  return {
    ...row,
    runId,
    trialId: nonemptyString(row.trialId, `${label}.trialId`),
    planIndex,
    repetitionIndex,
    frameIndex,
    phaseFrameIndex,
    visibilityFraction,
    visibilityOrder,
    visibilityOrderPosition: integer(
      row.visibilityOrderPosition,
      `${label}.visibilityOrderPosition`,
      { minimum: 0, maximum: 1 },
    ),
    lanePhysicalOrder,
    superblockOrientationOffset: integer(
      row.superblockOrientationOffset,
      `${label}.superblockOrientationOffset`,
      { minimum: 0, maximum: 1 },
    ),
    crossoverBlockIndex: integer(
      row.crossoverBlockIndex,
      `${label}.crossoverBlockIndex`,
      { minimum: 0, maximum: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_BLOCKS - 1 },
    ),
    withinBlockPosition: integer(
      row.withinBlockPosition,
      `${label}.withinBlockPosition`,
      { minimum: 0, maximum: FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE - 1 },
    ),
    crossoverPatternIndex: integer(
      row.crossoverPatternIndex,
      `${label}.crossoverPatternIndex`,
      { minimum: 0, maximum: 1 },
    ),
    previousPreviousLaneId: nonemptyString(
      row.previousPreviousLaneId,
      `${label}.previousPreviousLaneId`,
    ),
    previousLaneId: nonemptyString(row.previousLaneId, `${label}.previousLaneId`),
    selectorWriteSerialAtTimingStart: integer(
      row.selectorWriteSerialAtTimingStart,
      `${label}.selectorWriteSerialAtTimingStart`,
      { minimum: 0, maximum: MAX_SERIAL_BASE },
    ),
    strategySelectionSerialAtTimingStart: integer(
      row.strategySelectionSerialAtTimingStart,
      `${label}.strategySelectionSerialAtTimingStart`,
      { minimum: 0, maximum: MAX_SERIAL_BASE },
    ),
    computeCallSerialAtTimingStart: integer(
      row.computeCallSerialAtTimingStart,
      `${label}.computeCallSerialAtTimingStart`,
      { minimum: 0, maximum: MAX_SERIAL_BASE },
    ),
    renderCallSerialAtTimingStart: integer(
      row.renderCallSerialAtTimingStart,
      `${label}.renderCallSerialAtTimingStart`,
      { minimum: 0, maximum: MAX_SERIAL_BASE },
    ),
    selectorWriteSerial: integer(row.selectorWriteSerial, `${label}.selectorWriteSerial`, {
      minimum: 1,
    }),
    strategySelectionSerial: integer(
      row.strategySelectionSerial,
      `${label}.strategySelectionSerial`,
      { minimum: 1 },
    ),
    computeCallSerial: integer(row.computeCallSerial, `${label}.computeCallSerial`, {
      minimum: 1,
    }),
    renderCallSerial: integer(row.renderCallSerial, `${label}.renderCallSerial`, {
      minimum: 1,
    }),
    commandBufferId,
    portableCommandBufferIdAtTimingStart,
    featureCommandBufferIdAtTimingStart,
    gpuFrameId,
    gpuComputeMs,
    gpuRenderMs,
    gpuPassTotalMs,
  };
}

function percentage(deltaMs, portableMeanMs, label, zeroDenominatorAllowed) {
  if (portableMeanMs === 0) {
    if (zeroDenominatorAllowed) return null;
    reject(`${label} portable mean is zero, so its percentage is undefined.`);
  }
  const result = (deltaMs / portableMeanMs) * 100;
  if (!Number.isFinite(result)) reject(`derived ${label} percentage is non-finite.`);
  return result;
}

function summarizeHistoryBalance(records, label) {
  const transitionCounts = Object.fromEntries(
    FIRST_INSTANCE_LIVE_CROSSOVER_TRANSITION_KEYS.map((key) => [key, 0]),
  );
  const historyTripleCounts = Object.fromEntries(
    FIRST_INSTANCE_LIVE_CROSSOVER_HISTORY_TRIPLE_KEYS.map((key) => [key, 0]),
  );
  for (const row of records) {
    const transitionKey = `${row.previousLaneId}->${row.laneId}`;
    const historyTripleKey =
      `${row.previousPreviousLaneId}->${row.previousLaneId}->${row.laneId}`;
    if (!Object.hasOwn(transitionCounts, transitionKey)) {
      reject(`${label} has an invalid transition ${JSON.stringify(transitionKey)}.`);
    }
    if (!Object.hasOwn(historyTripleCounts, historyTripleKey)) {
      reject(`${label} has an invalid history triple ${JSON.stringify(historyTripleKey)}.`);
    }
    transitionCounts[transitionKey] += 1;
    historyTripleCounts[historyTripleKey] += 1;
  }
  const expectedTransitionCountPerCell = records.length / Object.keys(transitionCounts).length;
  const expectedHistoryTripleCountPerCell =
    records.length / Object.keys(historyTripleCounts).length;
  if (!Number.isInteger(expectedTransitionCountPerCell)
    || !Object.values(transitionCounts).every(
      (count) => count === expectedTransitionCountPerCell,
    )) {
    reject(`${label} is not exactly first-order transition-balanced.`);
  }
  if (!Number.isInteger(expectedHistoryTripleCountPerCell)
    || !Object.values(historyTripleCounts).every(
      (count) => count === expectedHistoryTripleCountPerCell,
    )) {
    reject(`${label} is not exactly two-frame-history balanced.`);
  }
  return {
    schemaVersion: 1,
    kind: 'live-first-instance-crossover-history-balance',
    frameCount: records.length,
    expectedTransitionCountPerCell,
    transitionCounts,
    expectedHistoryTripleCountPerCell,
    historyTripleCounts,
  };
}

function summarizeBlockMetric(records, metricKey, label) {
  const definition = METRICS[metricKey];
  const previousLaneStrata = Object.fromEntries(
    FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map((previousLaneId) => {
      const selected = records.filter((row) => row.previousLaneId === previousLaneId);
      const portable = selected
        .filter((row) => row.laneId === PORTABLE)
        .map((row) => row[definition.field]);
      const feature = selected
        .filter((row) => row.laneId === FEATURE)
        .map((row) => row[definition.field]);
      if (portable.length !== 2 || feature.length !== 2) {
        reject(
          `${label} previous-lane stratum ${previousLaneId} is not exactly balanced `
            + `for ${metricKey}.`,
        );
      }
      const portableMeanMs = mean(
        portable,
        `${label} previous=${previousLaneId} portable ${metricKey}`,
      );
      const featureMeanMs = mean(
        feature,
        `${label} previous=${previousLaneId} feature ${metricKey}`,
      );
      return [previousLaneId, {
        previousLaneId,
        portableCount: portable.length,
        featureCount: feature.length,
        portableMeanMs,
        featureMeanMs,
        deltaMs: finiteDerived(
          featureMeanMs - portableMeanMs,
          `${label} previous=${previousLaneId} ${metricKey} delta`,
        ),
        deltaPercent: percentage(
          featureMeanMs - portableMeanMs,
          portableMeanMs,
          `${label} previous=${previousLaneId} ${metricKey}`,
          definition.zeroDenominatorAllowed,
        ),
      }];
    }),
  );
  const portableMeanMs = mean(
    Object.values(previousLaneStrata).map((stratum) => stratum.portableMeanMs),
    `${label} predecessor-standardized portable ${metricKey}`,
  );
  const featureMeanMs = mean(
    Object.values(previousLaneStrata).map((stratum) => stratum.featureMeanMs),
    `${label} predecessor-standardized feature ${metricKey}`,
  );
  const deltaMs = mean(
    Object.values(previousLaneStrata).map((stratum) => stratum.deltaMs),
    `${label} previous-lane-stratified ${metricKey} delta`,
  );
  if (!Number.isFinite(deltaMs)) reject(`derived ${label} ${metricKey} delta is non-finite.`);
  return {
    estimator:
      'equal-weight mean of within-previous-lane feature-minus-portable contrasts',
    previousLaneStrata,
    portableMeanMs,
    featureMeanMs,
    pooledMeanMs: mean(records.map((row) => row[definition.field]), `${label} ${metricKey}`),
    deltaMs,
    deltaPercent: percentage(
      deltaMs,
      portableMeanMs,
      `${label} ${metricKey}`,
      definition.zeroDenominatorAllowed,
    ),
  };
}

function summarizeMetricBlocks(blocks, metricKey, label) {
  const deltaPercent = blocks.map((block) => block.metrics[metricKey].deltaPercent);
  return {
    deltaMs: median(
      blocks.map((block) => block.metrics[metricKey].deltaMs),
      `${label} ${metricKey} delta`,
    ),
    deltaPercent: deltaPercent.every(Number.isFinite)
      ? median(deltaPercent, `${label} ${metricKey} percentage`)
      : null,
    previousLaneStrata: Object.fromEntries(
      FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map((previousLaneId) => {
        const strata = blocks.map(
          (block) => block.metrics[metricKey].previousLaneStrata[previousLaneId],
        );
        const stratumPercentages = strata.map((stratum) => stratum.deltaPercent);
        return [previousLaneId, {
          previousLaneId,
          nBlocks: strata.length,
          deltaMs: median(
            strata.map((stratum) => stratum.deltaMs),
            `${label} previous=${previousLaneId} ${metricKey} delta`,
          ),
          deltaPercent: stratumPercentages.every(Number.isFinite)
            ? median(
              stratumPercentages,
              `${label} previous=${previousLaneId} ${metricKey} percentage`,
            )
            : null,
        }];
      }),
    ),
  };
}

function summarizeMetricSet(blocks, label) {
  return Object.fromEntries(
    Object.keys(METRICS).map((metricKey) => [
      metricKey,
      summarizeMetricBlocks(blocks, metricKey, label),
    ]),
  );
}

function summarizeTrial(rows, expectedTrial, expectedRunId) {
  if (rows.length !== FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES) {
    reject(
      `plan index ${expectedTrial.planIndex} has ${rows.length} rows; expected `
        + `${FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES}.`,
    );
  }
  const first = rows[0];
  const blockRows = Array.from(
    { length: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_BLOCKS },
    () => [],
  );
  requireInvariant(first.trialId, expectedTrial.trialId,
    `plan ${expectedTrial.planIndex}.trialId`);

  for (let frameIndex = 0; frameIndex < rows.length; frameIndex += 1) {
    const row = rows[frameIndex];
    const label = `${first.trialId}[${frameIndex}]`;
    requireInvariant(row.runId, expectedRunId, `${label}.runId`);
    requireInvariant(row.trialId, first.trialId, `${label}.trialId`);
    requireInvariant(row.planIndex, expectedTrial.planIndex, `${label}.planIndex`);
    requireInvariant(row.repetitionIndex, expectedTrial.repetitionIndex,
      `${label}.repetitionIndex`);
    requireInvariant(row.frameIndex, frameIndex, `${label}.frameIndex`);
    requireInvariant(row.phaseFrameIndex, frameIndex, `${label}.phaseFrameIndex`);
    requireInvariant(row.visibilityFraction, expectedTrial.visibilityFraction,
      `${label}.visibilityFraction`);
    requireInvariant(row.visibilityOrderPosition, expectedTrial.visibilityOrderPosition,
      `${label}.visibilityOrderPosition`);
    if (!sameArray(row.visibilityOrder, expectedTrial.visibilityOrder)) {
      reject(`${label}.visibilityOrder differs from the committed plan.`);
    }
    if (!sameArray(row.lanePhysicalOrder, expectedTrial.lanePhysicalOrder)) {
      reject(`${label}.lanePhysicalOrder differs from the committed plan.`);
    }
    requireInvariant(row.superblockOrientationOffset,
      expectedTrial.superblockOrientationOffset,
      `${label}.superblockOrientationOffset`);

    const scheduled = firstInstanceLiveCrossoverFrame(
      frameIndex,
      expectedTrial.superblockOrientationOffset,
    );
    requireInvariant(row.crossoverBlockIndex, scheduled.crossoverBlockIndex,
      `${label}.crossoverBlockIndex`);
    requireInvariant(row.withinBlockPosition, scheduled.withinBlockPosition,
      `${label}.withinBlockPosition`);
    requireInvariant(row.crossoverPatternIndex, scheduled.patternIndex,
      `${label}.crossoverPatternIndex`);
    requireInvariant(row.crossoverPattern, scheduled.pattern, `${label}.crossoverPattern`);
    requireInvariant(row.previousPreviousLaneId, scheduled.previousPreviousLaneId,
      `${label}.previousPreviousLaneId`);
    requireInvariant(row.previousLaneId, scheduled.previousLaneId,
      `${label}.previousLaneId`);
    requireInvariant(row.laneId, scheduled.laneId, `${label}.laneId`);

    requireInvariant(
      row.portableCommandBufferIdAtTimingStart,
      first.portableCommandBufferIdAtTimingStart,
      `${label}.portableCommandBufferIdAtTimingStart`,
    );
    requireInvariant(
      row.featureCommandBufferIdAtTimingStart,
      first.featureCommandBufferIdAtTimingStart,
      `${label}.featureCommandBufferIdAtTimingStart`,
    );
    const expectedCommandBufferId = scheduled.laneId === PORTABLE
      ? first.portableCommandBufferIdAtTimingStart
      : first.featureCommandBufferIdAtTimingStart;
    requireInvariant(row.commandBufferId, expectedCommandBufferId,
      `${label}.commandBufferId`);

    for (const field of [
      'selectorWriteSerialAtTimingStart',
      'strategySelectionSerialAtTimingStart',
      'computeCallSerialAtTimingStart',
      'renderCallSerialAtTimingStart',
    ]) {
      requireInvariant(row[field], first[field], `${label}.${field}`);
    }
    const serialOffset = FIRST_INSTANCE_LIVE_CROSSOVER_WARMUP_FRAMES + frameIndex + 1;
    for (const field of [
      'selectorWriteSerial',
      'strategySelectionSerial',
      'computeCallSerial',
      'renderCallSerial',
    ]) {
      const startField = `${field}AtTimingStart`;
      requireInvariant(row[field], row[startField] + serialOffset, `${label}.${field}`);
    }
    if (frameIndex > 0) {
      for (const field of [
        'selectorWriteSerial',
        'strategySelectionSerial',
        'computeCallSerial',
        'renderCallSerial',
        'gpuFrameId',
      ]) {
        requireInvariant(row[field], rows[frameIndex - 1][field] + 1, `${label}.${field}`);
      }
    }
    blockRows[scheduled.crossoverBlockIndex].push(row);
  }

  const blocks = blockRows.map((records, crossoverBlockIndex) => {
    if (records.length !== FIRST_INSTANCE_LIVE_CROSSOVER_BLOCK_SIZE) {
      reject(`trial ${JSON.stringify(first.trialId)} block ${crossoverBlockIndex} is incomplete.`);
    }
    const label = `trial ${JSON.stringify(first.trialId)} block ${crossoverBlockIndex}`;
    return {
      crossoverBlockIndex,
      pattern: records[0].crossoverPattern,
      historyBalance: summarizeHistoryBalance(records, label),
      metrics: Object.fromEntries(
        Object.keys(METRICS).map((metricKey) => [
          metricKey,
          summarizeBlockMetric(records, metricKey, label),
        ]),
      ),
    };
  });

  const firstQuarterMeanMs = mean(
    blocks.slice(0, QUARTER_BLOCKS).map(
      (block) => block.metrics.gpuPassTotal.pooledMeanMs,
    ),
    `${first.trialId} first-quarter GPU-pass total`,
  );
  const lastQuarterMeanMs = mean(
    blocks.slice(-QUARTER_BLOCKS).map(
      (block) => block.metrics.gpuPassTotal.pooledMeanMs,
    ),
    `${first.trialId} last-quarter GPU-pass total`,
  );
  const driftDeltaMs = lastQuarterMeanMs - firstQuarterMeanMs;
  if (!Number.isFinite(driftDeltaMs)) reject(`${first.trialId} drift delta is non-finite.`);

  return {
    trialId: first.trialId,
    planIndex: expectedTrial.planIndex,
    repetitionIndex: expectedTrial.repetitionIndex,
    visibilityFraction: expectedTrial.visibilityFraction,
    visibilityOrder: [...expectedTrial.visibilityOrder],
    visibilityOrderPosition: expectedTrial.visibilityOrderPosition,
    lanePhysicalOrder: [...expectedTrial.lanePhysicalOrder],
    portableLanePhysicalOrderPosition: expectedTrial.lanePhysicalOrder.indexOf(PORTABLE),
    featureLanePhysicalOrderPosition: expectedTrial.lanePhysicalOrder.indexOf(FEATURE),
    superblockOrientationOffset: expectedTrial.superblockOrientationOffset,
    commandBufferIds: {
      portable: first.portableCommandBufferIdAtTimingStart,
      feature: first.featureCommandBufferIdAtTimingStart,
    },
    nRows: rows.length,
    nBlocks: blocks.length,
    historyBalance: summarizeHistoryBalance(rows, `trial ${JSON.stringify(first.trialId)}`),
    blocks,
    estimates: summarizeMetricSet(blocks, `${first.trialId} trial`),
    halves: {
      first: summarizeMetricSet(blocks.slice(0, HALF_BLOCKS), `${first.trialId} first half`),
      second: summarizeMetricSet(blocks.slice(HALF_BLOCKS), `${first.trialId} second half`),
    },
    drift: {
      firstQuarterBlocks: QUARTER_BLOCKS,
      lastQuarterBlocks: QUARTER_BLOCKS,
      firstQuarterMeanMs,
      lastQuarterMeanMs,
      deltaMs: driftDeltaMs,
      percent: percentage(
        driftDeltaMs,
        firstQuarterMeanMs,
        `${first.trialId} drift`,
        false,
      ),
    },
  };
}

function summarizeEstimates(estimates, label) {
  const percentages = estimates.map((estimate) => estimate.deltaPercent);
  return {
    n: estimates.length,
    median: {
      deltaMs: median(estimates.map((estimate) => estimate.deltaMs), `${label} delta`),
      deltaPercent: percentages.every(Number.isFinite)
        ? median(percentages, `${label} percentage`)
        : null,
    },
    negativeCount: estimates.filter((estimate) => estimate.deltaMs < 0).length,
  };
}

function binaryStratum(trials, field, levels) {
  const strata = levels.map((level) => {
    const selected = trials.filter((trial) => trial[field] === level);
    if (selected.length !== FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS / 2) {
      reject(`${field}=${level} has ${selected.length} high-visibility trials; expected 6.`);
    }
    return {
      level,
      ...summarizeEstimates(
        selected.map((trial) => trial.estimates.gpuPassTotal),
        `${field}=${level}`,
      ),
    };
  });
  return {
    field,
    strata,
    interaction: {
      deltaMs: finiteDerived(
        strata[1].median.deltaMs - strata[0].median.deltaMs,
        `${field} interaction delta`,
      ),
      deltaPercent: finiteDerived(
        strata[1].median.deltaPercent - strata[0].median.deltaPercent,
        `${field} interaction percentage-point difference`,
      ),
    },
  };
}

function halfStratum(trials) {
  const strata = ['first', 'second'].map((half) => ({
    level: half,
    ...summarizeEstimates(
      trials.map((trial) => trial.halves[half].gpuPassTotal),
      `measurementHalf=${half}`,
    ),
  }));
  return {
    field: 'measurementHalf',
    strata,
    interaction: {
      deltaMs: finiteDerived(
        strata[1].median.deltaMs - strata[0].median.deltaMs,
        'measurement-half interaction delta',
      ),
      deltaPercent: finiteDerived(
        strata[1].median.deltaPercent - strata[0].median.deltaPercent,
        'measurement-half interaction percentage-point difference',
      ),
    },
  };
}

function previousLaneStratum(trials) {
  const strata = FIRST_INSTANCE_LIVE_CROSSOVER_LANES.map((previousLaneId) => ({
    level: previousLaneId,
    ...summarizeEstimates(
      trials.map(
        (trial) => trial.estimates.gpuPassTotal.previousLaneStrata[previousLaneId],
      ),
      `previousLaneId=${previousLaneId}`,
    ),
  }));
  return {
    field: 'previousLaneId',
    strata,
    interaction: {
      deltaMs: finiteDerived(
        strata[1].median.deltaMs - strata[0].median.deltaMs,
        'previous-lane interaction delta',
      ),
      deltaPercent: finiteDerived(
        strata[1].median.deltaPercent - strata[0].median.deltaPercent,
        'previous-lane interaction percentage-point difference',
      ),
    },
  };
}

function visibilityContrast(trials) {
  return {
    gpuPassTotal: {
      ...summarizeEstimates(
        trials.map((trial) => trial.estimates.gpuPassTotal),
        'GPU-pass total',
      ),
      strata: {
        lanePhysicalOrder: binaryStratum(
          trials,
          'portableLanePhysicalOrderPosition',
          [0, 1],
        ),
        startingOrientation: binaryStratum(
          trials,
          'superblockOrientationOffset',
          [0, 1],
        ),
        visibilityOrderPosition: binaryStratum(
          trials,
          'visibilityOrderPosition',
          [0, 1],
        ),
        measurementHalf: halfStratum(trials),
      },
      previousLane: previousLaneStratum(trials),
    },
    gpuRender: summarizeEstimates(
      trials.map((trial) => trial.estimates.gpuRender),
      'GPU render',
    ),
    gpuCompute: summarizeEstimates(
      trials.map((trial) => trial.estimates.gpuCompute),
      'GPU compute',
    ),
  };
}

function summarizeDrift(trials) {
  return {
    n: trials.length,
    median: {
      deltaMs: median(trials.map((trial) => trial.drift.deltaMs), 'drift delta'),
      percent: median(trials.map((trial) => trial.drift.percent), 'drift percentage'),
    },
  };
}

function stratumGate(stratum) {
  const levelsNegative = stratum.strata.every((record) => record.median.deltaMs < 0);
  const interactionWithinBounds = Math.abs(stratum.interaction.deltaMs) < MATERIAL_MS
    && Math.abs(stratum.interaction.deltaPercent) < MATERIAL_PERCENT;
  return {
    pass: levelsNegative && interactionWithinBounds,
    levelsNegative,
    interactionWithinBounds,
    bounds: { absoluteMs: MATERIAL_MS, percentagePoints: MATERIAL_PERCENT },
    ...stratum,
  };
}

function driftGate(summary) {
  return {
    pass: Math.abs(summary.median.deltaMs) < DRIFT_MS
      && Math.abs(summary.median.percent) < DRIFT_PERCENT,
    bounds: { absoluteMs: DRIFT_MS, percent: DRIFT_PERCENT },
    observed: summary.median,
    n: summary.n,
  };
}

function evaluateNumericalGates(contrasts, trialsByVisibility) {
  const high = contrasts.visibilities[String(HIGH_VISIBILITY)];
  const low = contrasts.visibilities[String(LOW_VISIBILITY)];
  const highVisibilityTotalMaterial = {
    pass: high.gpuPassTotal.median.deltaMs <= -MATERIAL_MS
      && high.gpuPassTotal.median.deltaPercent <= -MATERIAL_PERCENT,
    threshold: { absoluteMs: -MATERIAL_MS, percent: -MATERIAL_PERCENT },
    conjunction: 'both thresholds are required',
    observed: high.gpuPassTotal.median,
  };
  const highVisibilityTotalDirection = {
    pass: high.gpuPassTotal.negativeCount >= REQUIRED_SIGN_COUNT,
    requiredNegative: REQUIRED_SIGN_COUNT,
    observedNegative: high.gpuPassTotal.negativeCount,
    total: high.gpuPassTotal.n,
  };
  const highVisibilityRender = {
    pass: high.gpuRender.median.deltaMs <= -MATERIAL_MS
      && high.gpuRender.median.deltaPercent <= -MATERIAL_PERCENT
      && high.gpuRender.negativeCount >= REQUIRED_SIGN_COUNT,
    threshold: { absoluteMs: -MATERIAL_MS, percent: -MATERIAL_PERCENT },
    requiredNegative: REQUIRED_SIGN_COUNT,
    observed: high.gpuRender.median,
    observedNegative: high.gpuRender.negativeCount,
    total: high.gpuRender.n,
    conjunction: 'both material thresholds and the direction count are required',
  };
  const highStrata = Object.fromEntries(
    Object.entries(high.gpuPassTotal.strata).map(
      ([name, value]) => [name, stratumGate(value)],
    ),
  );
  const highVisibilityCarryover = stratumGate(high.gpuPassTotal.previousLane);
  const lowEstimates = trialsByVisibility[String(LOW_VISIBILITY)].map(
    (trial) => trial.estimates.gpuPassTotal,
  );
  const lowCounts = {
    belowMsUpperBound: lowEstimates.filter(
      (estimate) => estimate.deltaMs < LOW_REGRESSION_MS,
    ).length,
    belowPercentUpperBound: lowEstimates.filter(
      (estimate) => estimate.deltaPercent < LOW_REGRESSION_PERCENT,
    ).length,
  };
  const lowMedianBelow = low.gpuPassTotal.median.deltaMs < LOW_REGRESSION_MS
    && low.gpuPassTotal.median.deltaPercent < LOW_REGRESSION_PERCENT;
  const lowVisibilityTotalRegression = {
    pass: lowMedianBelow
      && Object.values(lowCounts).every((count) => count >= REQUIRED_SIGN_COUNT),
    medianBelowBothBounds: lowMedianBelow,
    requiredBelowEachBound: REQUIRED_SIGN_COUNT,
    bounds: { absoluteMs: LOW_REGRESSION_MS, percent: LOW_REGRESSION_PERCENT },
    counts: lowCounts,
    observed: low.gpuPassTotal.median,
  };
  const pairedHighMinusLowTotal = {
    pass: contrasts.pairedHighMinusLow.median.deltaMs <= -PAIRED_MATERIAL_MS,
    thresholdMs: -PAIRED_MATERIAL_MS,
    observed: contrasts.pairedHighMinusLow.median,
    nPairs: contrasts.pairedHighMinusLow.nPairs,
  };
  const drift = {
    overall: driftGate(contrasts.drift.overall),
    visibilities: Object.fromEntries(
      FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS.map((visibility) => [
        String(visibility),
        driftGate(contrasts.drift.visibilities[String(visibility)]),
      ]),
    ),
  };
  drift.pass = drift.overall.pass
    && Object.values(drift.visibilities).every((gate) => gate.pass);

  const gates = {
    highVisibilityTotalMaterial,
    highVisibilityTotalDirection,
    highVisibilityRender,
    highVisibilityTotalStrata: {
      pass: Object.values(highStrata).every((gate) => gate.pass),
      factors: highStrata,
    },
    highVisibilityCarryover,
    lowVisibilityTotalRegression,
    pairedHighMinusLowTotal,
    drift,
  };
  const failedGates = Object.entries(gates)
    .filter(([, gate]) => gate.pass !== true)
    .map(([name]) => name);
  return {
    status: 'evaluated',
    scope: 'preregistered numerical gates only; evidence-integrity gates are external',
    pass: failedGates.length === 0,
    failedGates,
    gates,
  };
}

/**
 * Normalize and reconstruct one exact 480-row live first-instance trial.
 * This is a narrow diagnostic entry point; the candidate analyzer below keeps
 * its existing whole-matrix chronology and numerical-gate behavior.
 */
export function summarizeLiveFirstInstanceTrialRows(
  rows,
  expectedTrial,
  expectedRunId,
) {
  if (!Array.isArray(rows)) reject('trial input rows must be an array.');
  if (rows.length !== FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES) {
    reject(
      `trial input has ${rows.length} rows; expected exactly `
        + `${FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES}.`,
    );
  }
  requireRecord(expectedTrial, 'expectedTrial');
  nonemptyString(expectedRunId, 'expectedRunId');

  const seenGpuFrameIds = new Set();
  const normalized = rows.map(
    (row, index) => normalizeRow(row, index, seenGpuFrameIds),
  );
  if (normalized.some((row) => row.runId !== expectedRunId)) {
    reject(`trial input rows must all use runId ${JSON.stringify(expectedRunId)}.`);
  }
  return summarizeTrial(normalized, expectedTrial, expectedRunId);
}

/**
 * Analyze measured rows from one preregistered live indirect-firstInstance
 * candidate matrix. The function fails closed on any non-exact plan, schedule,
 * timestamp shape, command-buffer selection, or serial stream. The caller is
 * responsible for validation-payload, environment, provenance, telemetry, and
 * artifact-manifest gates before treating the numerical decision as a run pass.
 */
export function summarizeLiveFirstInstanceCrossoverRows(rows) {
  if (!Array.isArray(rows)) reject('input rows must be an array.');
  if (rows.length !== EXPECTED_ROWS) {
    reject(`input has ${rows.length} rows; expected exactly ${EXPECTED_ROWS}.`);
  }

  const seenGpuFrameIds = new Set();
  const normalized = rows.map((row, index) => normalizeRow(row, index, seenGpuFrameIds));
  const runId = normalized[0].runId;
  if (normalized.some((row) => row.runId !== runId)) {
    reject('input rows must contain exactly one runId.');
  }
  const expectedPlan = buildFirstInstanceLiveCrossoverPlan({
    runId,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  });
  const trials = [];
  const seenTrialIds = new Set();
  let previousTrialLastGpuFrameId = null;
  let previousTrialLastComputeCallSerial = null;
  let previousTrialLastRenderCallSerial = null;
  for (const expectedTrial of expectedPlan) {
    const start = expectedTrial.planIndex * FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES;
    const trialRows = normalized.slice(
      start,
      start + FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
    );
    if (trialRows.some((row) => row.planIndex !== expectedTrial.planIndex)) {
      reject(
        `input rows are not contiguous in committed plan-index order at plan `
          + `${expectedTrial.planIndex}.`,
      );
    }
    const trialId = trialRows[0].trialId;
    if (seenTrialIds.has(trialId)) reject(`trialId ${JSON.stringify(trialId)} is reused.`);
    seenTrialIds.add(trialId);
    if (previousTrialLastGpuFrameId !== null
      && trialRows[0].gpuFrameId <= previousTrialLastGpuFrameId) {
      reject(`trial ${expectedTrial.planIndex} violates global GPU frame chronology.`);
    }
    if (previousTrialLastComputeCallSerial !== null
      && trialRows[0].computeCallSerialAtTimingStart <= previousTrialLastComputeCallSerial) {
      reject(`trial ${expectedTrial.planIndex} violates global compute-call chronology.`);
    }
    if (previousTrialLastRenderCallSerial !== null
      && trialRows[0].renderCallSerialAtTimingStart <= previousTrialLastRenderCallSerial) {
      reject(`trial ${expectedTrial.planIndex} violates global render-call chronology.`);
    }
    previousTrialLastGpuFrameId = trialRows.at(-1).gpuFrameId;
    previousTrialLastComputeCallSerial = trialRows.at(-1).computeCallSerial;
    previousTrialLastRenderCallSerial = trialRows.at(-1).renderCallSerial;
    trials.push(summarizeTrial(trialRows, expectedTrial, runId));
  }
  if (seenTrialIds.size !== EXPECTED_TRIALS) {
    reject(`input has ${seenTrialIds.size} trials; expected exactly ${EXPECTED_TRIALS}.`);
  }

  const trialsByVisibility = Object.fromEntries(
    FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS.map((visibility) => [
      String(visibility),
      trials.filter((trial) => trial.visibilityFraction === visibility),
    ]),
  );
  for (const visibility of FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS) {
    const selected = trialsByVisibility[String(visibility)];
    if (selected.length !== FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS) {
      reject(`${visibility} visibility has ${selected.length} trials; expected exactly 12.`);
    }
    const repetitions = new Set(selected.map((trial) => trial.repetitionIndex));
    if (repetitions.size !== FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS
      || Array.from(
        { length: FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS },
        (_, index) => index,
      ).some((index) => !repetitions.has(index))) {
      reject(`${visibility} visibility does not contain exactly repetitions 0 through 11.`);
    }
  }

  const pairedTrials = [];
  for (let repetitionIndex = 0;
    repetitionIndex < FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS;
    repetitionIndex += 1) {
    const high = trialsByVisibility[String(HIGH_VISIBILITY)].find(
      (trial) => trial.repetitionIndex === repetitionIndex,
    );
    const low = trialsByVisibility[String(LOW_VISIBILITY)].find(
      (trial) => trial.repetitionIndex === repetitionIndex,
    );
    if (high.visibilityOrderPosition === low.visibilityOrderPosition) {
      reject(`repetition ${repetitionIndex} does not use complementary visibility positions.`);
    }
    if (!sameArray(high.lanePhysicalOrder, low.lanePhysicalOrder)
      || high.superblockOrientationOffset !== low.superblockOrientationOffset) {
      reject(`repetition ${repetitionIndex} changes crossover factors between visibilities.`);
    }
    pairedTrials.push({
      repetitionIndex,
      highTrialId: high.trialId,
      lowTrialId: low.trialId,
      high: high.estimates.gpuPassTotal,
      low: low.estimates.gpuPassTotal,
      delta: {
        deltaMs: finiteDerived(
          high.estimates.gpuPassTotal.deltaMs - low.estimates.gpuPassTotal.deltaMs,
          `repetition ${repetitionIndex} paired dose delta`,
        ),
        deltaPercent: finiteDerived(
          high.estimates.gpuPassTotal.deltaPercent
            - low.estimates.gpuPassTotal.deltaPercent,
          `repetition ${repetitionIndex} paired dose percentage-point difference`,
        ),
      },
    });
  }

  const contrasts = {
    visibilities: Object.fromEntries(
      FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS.map((visibility) => [
        String(visibility),
        visibilityContrast(trialsByVisibility[String(visibility)]),
      ]),
    ),
    pairedHighMinusLow: {
      nPairs: pairedTrials.length,
      pairs: pairedTrials,
      median: {
        deltaMs: median(pairedTrials.map((pair) => pair.delta.deltaMs), 'paired dose delta'),
        deltaPercent: median(
          pairedTrials.map((pair) => pair.delta.deltaPercent),
          'paired dose percentage-point difference',
        ),
      },
    },
    drift: {
      overall: summarizeDrift(trials),
      visibilities: Object.fromEntries(
        FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS.map((visibility) => [
          String(visibility),
          summarizeDrift(trialsByVisibility[String(visibility)]),
        ]),
      ),
    },
  };
  const preregisteredDecision = evaluateNumericalGates(contrasts, trialsByVisibility);

  return {
    schemaVersion: 2,
    kind: 'indirect-first-instance-live-crossover-summary',
    deltaConvention: 'feature minus portable; negative values favor indirect firstInstance',
    primaryEndpoint: 'gpuPassTotalMs = gpuComputeMs + gpuRenderMs',
    protocol: {
      objectCount: OBJECT_COUNT,
      bucketCount: BUCKET_COUNT,
      repetitions: FIRST_INSTANCE_LIVE_CROSSOVER_REPETITIONS,
      visibilityLevels: [...FIRST_INSTANCE_LIVE_CROSSOVER_VISIBILITY_LEVELS],
      measuredBlocksPerTrial: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_BLOCKS,
      measuredFramesPerTrial: FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES,
      totalEqualityToleranceMs: TOTAL_EQUALITY_TOLERANCE_MS,
      inferentialUnit: 'repetition-level trial estimate',
      estimator:
        'equal-weight mean of within-previous-lane feature-minus-portable block contrasts',
      expectedMeasuredTransitionCountPerCell:
        FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES / 4,
      expectedMeasuredHistoryTripleCountPerCell:
        FIRST_INSTANCE_LIVE_CROSSOVER_MEASURED_FRAMES / 8,
    },
    nRows: rows.length,
    nTrials: trials.length,
    trials,
    contrasts,
    preregisteredDecision,
    preregisteredNumericalDecision: preregisteredDecision,
  };
}
