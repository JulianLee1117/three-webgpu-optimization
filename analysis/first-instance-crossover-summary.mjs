import {
  FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE,
  FIRST_INSTANCE_CROSSOVER_MEASURED_BLOCKS,
  FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
  firstInstanceCrossoverFrame,
} from '../src/benchmark/first-instance-crossover-schedule.js';
import {
  FIRST_INSTANCE_CROSSOVER_COMMAND_SEGMENT_ORDERS,
  FIRST_INSTANCE_CROSSOVER_LANES,
  FIRST_INSTANCE_CROSSOVER_MODE,
  FIRST_INSTANCE_CROSSOVER_ORIENTATION_OFFSETS,
  FIRST_INSTANCE_CROSSOVER_REPETITIONS,
  FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS,
} from '../src/benchmark/plan.js';

const [PORTABLE, FEATURE] = FIRST_INSTANCE_CROSSOVER_LANES;
const [HIGH_VISIBILITY, LOW_VISIBILITY] = FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS;
const OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;
const INDEXED_INDIRECT_RECORD_BYTES = 5 * Uint32Array.BYTES_PER_ELEMENT;
const HALF_BLOCKS = FIRST_INSTANCE_CROSSOVER_MEASURED_BLOCKS / 2;
const QUARTER_BLOCKS = FIRST_INSTANCE_CROSSOVER_MEASURED_BLOCKS / 4;
const MATERIAL_MS = 0.10;
const MATERIAL_PERCENT = 5;
const LOW_REGRESSION_MS = 0.02;
const LOW_REGRESSION_PERCENT = 5;
const PAIRED_MATERIAL_MS = 0.05;
const DRIFT_MS = 0.10;
const DRIFT_PERCENT = 5;
const REQUIRED_SIGN_COUNT = 10;
const VALIDATION_KIND = 'first-instance-crossover-exact-paired-snapshots';
const EXPECTED_TRIALS = FIRST_INSTANCE_CROSSOVER_REPETITIONS
  * FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS.length;
const EXPECTED_ROWS = EXPECTED_TRIALS * FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES;

function reject(message) {
  throw new Error(`First-instance crossover analysis rejected: ${message}`);
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

function requireInvariant(actual, expected, label) {
  if (actual !== expected) {
    reject(`${label} is ${JSON.stringify(actual)}; expected ${JSON.stringify(expected)}.`);
  }
}

function mean(values) {
  if (values.length === 0) reject('cannot compute a mean of an empty sample.');
  const result = values.reduce((sum, value) => sum + value, 0) / values.length;
  if (!Number.isFinite(result)) reject('derived mean is non-finite.');
  return result;
}

function median(values) {
  if (values.length === 0) reject('cannot compute a median of an empty sample.');
  if (values.some((value) => !Number.isFinite(value))) {
    reject('cannot compute a median containing a non-finite value.');
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function sameArray(left, right) {
  return left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function parseLaneOrderValue(value, label) {
  const order = Array.isArray(value)
    ? [...value]
    : typeof value === 'string'
      ? value.split('|')
      : null;
  if (order === null
    || order.length !== FIRST_INSTANCE_CROSSOVER_LANES.length
    || new Set(order).size !== FIRST_INSTANCE_CROSSOVER_LANES.length
    || FIRST_INSTANCE_CROSSOVER_LANES.some((laneId) => !order.includes(laneId))) {
    reject(`${label} must be the exact portable/feature lane permutation.`);
  }
  return order;
}

function laneOrderForRow(row, label) {
  const values = [
    ['laneCommandSegmentOrder', row.laneCommandSegmentOrder],
    ['plannedLaneCommandSegmentOrder', row.plannedLaneCommandSegmentOrder],
  ].filter(([, value]) => value !== undefined && value !== null);
  if (values.length === 0) {
    reject(`${label} must provide laneCommandSegmentOrder or plannedLaneCommandSegmentOrder.`);
  }
  const parsed = values.map(([name, value]) => parseLaneOrderValue(value, `${label}.${name}`));
  if (parsed.some((order) => !sameArray(order, parsed[0]))) {
    reject(`${label} has inconsistent lane command-segment order fields.`);
  }
  return parsed[0];
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
  if (!FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS.includes(values[0])) {
    reject(`${label}.visibilityFraction must be exactly 0.99 or 0.2.`);
  }
  return values[0];
}

function visibilityOrderForRow(row, label) {
  const value = row.visibilityOrder ?? row.plannedVisibilityOrder;
  const order = Array.isArray(value)
    ? [...value]
    : typeof value === 'string'
      ? value.split('|').map(Number)
      : null;
  if (order === null
    || order.length !== FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS.length
    || order.some((visibility) => !FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS.includes(visibility))
    || new Set(order).size !== FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS.length) {
    reject(`${label} must provide the exact 0.99/0.2 visibility order.`);
  }
  if (row.visibilityOrder !== undefined
    && row.plannedVisibilityOrder !== undefined
    && !sameArray(
      Array.isArray(row.visibilityOrder)
        ? row.visibilityOrder
        : row.visibilityOrder.split('|').map(Number),
      Array.isArray(row.plannedVisibilityOrder)
        ? row.plannedVisibilityOrder
        : row.plannedVisibilityOrder.split('|').map(Number),
    )) {
    reject(`${label} has inconsistent visibility-order fields.`);
  }
  return order;
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
  return order;
}

function expectedVisibilityOrder(repetitionIndex) {
  return repetitionIndex % 2 === 0
    ? [...FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS]
    : [...FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS].reverse();
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
    maximum: FIRST_INSTANCE_CROSSOVER_REPETITIONS - 1,
  });
  const frameIndex = integer(row.frameIndex, `${label}.frameIndex`, {
    minimum: 0,
    maximum: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES - 1,
  });
  const phaseFrameIndex = integer(row.phaseFrameIndex, `${label}.phaseFrameIndex`, {
    minimum: 0,
    maximum: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES - 1,
  });
  const visibilityFraction = visibilityForRow(row, label);
  const visibilityOrder = visibilityOrderForRow(row, `${label}.visibilityOrder`);
  const laneCommandSegmentOrder = laneOrderForRow(row, label);
  const gpuFrameId = integer(row.gpuFrameId, `${label}.gpuFrameId`, { minimum: 0 });
  if (seenGpuFrameIds.has(gpuFrameId)) reject(`${label} duplicates gpuFrameId ${gpuFrameId}.`);
  seenGpuFrameIds.add(gpuFrameId);

  const runId = nonemptyString(row.runId, `${label}.runId`);
  requireInvariant(row.schemaVersion, 2, `${label}.schemaVersion`);
  requireInvariant(row.modeId, FIRST_INSTANCE_CROSSOVER_MODE, `${label}.modeId`);
  requireInvariant(row.modeOrderPosition, 0, `${label}.modeOrderPosition`);
  const modeOrders = [
    ['modeOrder', row.modeOrder],
    ['plannedModeOrder', row.plannedModeOrder],
  ].filter(([, value]) => value !== undefined && value !== null);
  if (modeOrders.length === 0) reject(`${label} must provide a mode order.`);
  for (const [name, value] of modeOrders) {
    singletonOrder(value, FIRST_INSTANCE_CROSSOVER_MODE, `${label}.${name}`);
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
  requireInvariant(row.protocolWarmupFrames, FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
    `${label}.protocolWarmupFrames`);
  requireInvariant(row.protocolMeasuredFrames, FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
    `${label}.protocolMeasuredFrames`);
  requireInvariant(row.usesCompute, false, `${label}.usesCompute`);
  requireInvariant(row.configuredDrawCommands, BUCKET_COUNT, `${label}.configuredDrawCommands`);
  requireInvariant(row.configuredRenderObjects, 1, `${label}.configuredRenderObjects`);
  requireInvariant(row.configuredComputeDispatches, 0,
    `${label}.configuredComputeDispatches`);
  requireInvariant(row.configuredComputeSubmissions, 0,
    `${label}.configuredComputeSubmissions`);
  requireInvariant(row.validationPass, true, `${label}.validationPass`);
  requireInvariant(row.validationKind, VALIDATION_KIND, `${label}.validationKind`);
  requireInvariant(row.timestampAvailable, true, `${label}.timestampAvailable`);
  requireInvariant(row.gpuRenderTimestampUidCount, 1,
    `${label}.gpuRenderTimestampUidCount`);
  requireInvariant(row.expectedRenderTimestampUidCount, 1,
    `${label}.expectedRenderTimestampUidCount`);
  requireInvariant(row.gpuComputeMs, null, `${label}.gpuComputeMs`);
  requireInvariant(row.cpuComputeSubmitMs, null, `${label}.cpuComputeSubmitMs`);

  const expectedVisibleCount = Math.round(OBJECT_COUNT * visibilityFraction);
  requireInvariant(row.expectedVisibleCount, expectedVisibleCount, `${label}.expectedVisibleCount`);
  requireInvariant(row.configuredSubmittedInstances, expectedVisibleCount,
    `${label}.configuredSubmittedInstances`);
  const gpuRenderMs = finiteNumber(row.gpuRenderMs, `${label}.gpuRenderMs`, {
    minimum: 0,
    exclusiveMinimum: true,
  });
  requireInvariant(row.gpuPassTotalMs, gpuRenderMs, `${label}.gpuPassTotalMs`);

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
    laneCommandSegmentOrder,
    superblockOrientationOffset: integer(
      row.superblockOrientationOffset,
      `${label}.superblockOrientationOffset`,
      { minimum: 0, maximum: 1 },
    ),
    crossoverBlockIndex: integer(
      row.crossoverBlockIndex,
      `${label}.crossoverBlockIndex`,
      { minimum: 0, maximum: FIRST_INSTANCE_CROSSOVER_MEASURED_BLOCKS - 1 },
    ),
    withinBlockPosition: integer(
      row.withinBlockPosition,
      `${label}.withinBlockPosition`,
      { minimum: 0, maximum: FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE - 1 },
    ),
    crossoverPatternIndex: integer(
      row.crossoverPatternIndex,
      `${label}.crossoverPatternIndex`,
      { minimum: 0, maximum: 1 },
    ),
    commandSegmentIndex: integer(
      row.commandSegmentIndex,
      `${label}.commandSegmentIndex`,
      { minimum: 0, maximum: 1 },
    ),
    commandRecordBase: integer(row.commandRecordBase, `${label}.commandRecordBase`, {
      minimum: 0,
      maximum: BUCKET_COUNT,
    }),
    commandByteBase: integer(row.commandByteBase, `${label}.commandByteBase`, {
      minimum: 0,
      maximum: BUCKET_COUNT * INDEXED_INDIRECT_RECORD_BYTES,
    }),
    selectorWriteSerialAtTimingStart: integer(
      row.selectorWriteSerialAtTimingStart,
      `${label}.selectorWriteSerialAtTimingStart`,
      { minimum: 0 },
    ),
    strategySelectionSerialAtTimingStart: integer(
      row.strategySelectionSerialAtTimingStart,
      `${label}.strategySelectionSerialAtTimingStart`,
      { minimum: 0 },
    ),
    renderCallSerialAtTimingStart: integer(
      row.renderCallSerialAtTimingStart,
      `${label}.renderCallSerialAtTimingStart`,
      { minimum: 0 },
    ),
    selectorWriteSerial: integer(row.selectorWriteSerial, `${label}.selectorWriteSerial`, {
      minimum: 1,
    }),
    strategySelectionSerial: integer(
      row.strategySelectionSerial,
      `${label}.strategySelectionSerial`,
      { minimum: 1 },
    ),
    renderCallSerial: integer(row.renderCallSerial, `${label}.renderCallSerial`, {
      minimum: 1,
    }),
    gpuFrameId,
    gpuRenderMs,
  };
}

function summarizeBlocks(blocks) {
  return {
    deltaMs: median(blocks.map((block) => block.deltaMs)),
    deltaPercent: median(blocks.map((block) => block.deltaPercent)),
  };
}

function summarizeTrial(rows, expectedPlanIndex, expectedRunId) {
  if (rows.length !== FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES) {
    reject(
      `plan index ${expectedPlanIndex} has ${rows.length} rows; expected `
        + `${FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES}.`,
    );
  }
  const first = rows[0];
  const expectedRepetition = Math.floor(expectedPlanIndex / 2);
  const expectedPosition = expectedPlanIndex % 2;
  const committedVisibilityOrder = expectedVisibilityOrder(expectedRepetition);
  const committedLaneOrder = FIRST_INSTANCE_CROSSOVER_COMMAND_SEGMENT_ORDERS[
    expectedRepetition
  ];
  const committedOrientation = FIRST_INSTANCE_CROSSOVER_ORIENTATION_OFFSETS[
    expectedRepetition
  ];
  const expectedVisibility = committedVisibilityOrder[expectedPosition];
  const portableSegmentIndex = committedLaneOrder.indexOf(PORTABLE);
  const featureSegmentIndex = committedLaneOrder.indexOf(FEATURE);
  const blockRows = Array.from(
    { length: FIRST_INSTANCE_CROSSOVER_MEASURED_BLOCKS },
    () => [],
  );
  requireInvariant(
    first.trialId,
    `${expectedRunId}-t${String(expectedPlanIndex + 1).padStart(2, '0')}`,
    `plan ${expectedPlanIndex}.trialId`,
  );

  for (let frameIndex = 0; frameIndex < rows.length; frameIndex += 1) {
    const row = rows[frameIndex];
    const label = `${first.trialId}[${frameIndex}]`;
    requireInvariant(row.runId, expectedRunId, `${label}.runId`);
    requireInvariant(row.trialId, first.trialId, `${label}.trialId`);
    requireInvariant(row.planIndex, expectedPlanIndex, `${label}.planIndex`);
    requireInvariant(row.repetitionIndex, expectedRepetition, `${label}.repetitionIndex`);
    requireInvariant(row.frameIndex, frameIndex, `${label}.frameIndex`);
    requireInvariant(row.phaseFrameIndex, frameIndex, `${label}.phaseFrameIndex`);
    requireInvariant(row.visibilityFraction, expectedVisibility, `${label}.visibilityFraction`);
    requireInvariant(row.visibilityOrderPosition, expectedPosition,
      `${label}.visibilityOrderPosition`);
    if (!sameArray(row.visibilityOrder, committedVisibilityOrder)) {
      reject(`${label}.visibilityOrder differs from the committed plan.`);
    }
    if (!sameArray(row.laneCommandSegmentOrder, committedLaneOrder)) {
      reject(`${label}.laneCommandSegmentOrder differs from the committed plan.`);
    }
    requireInvariant(row.superblockOrientationOffset, committedOrientation,
      `${label}.superblockOrientationOffset`);

    const scheduled = firstInstanceCrossoverFrame(frameIndex, committedOrientation);
    requireInvariant(row.crossoverBlockIndex, scheduled.crossoverBlockIndex,
      `${label}.crossoverBlockIndex`);
    requireInvariant(row.withinBlockPosition, scheduled.withinBlockPosition,
      `${label}.withinBlockPosition`);
    requireInvariant(row.crossoverPatternIndex, scheduled.patternIndex,
      `${label}.crossoverPatternIndex`);
    requireInvariant(row.crossoverPattern, scheduled.pattern, `${label}.crossoverPattern`);
    requireInvariant(row.laneId, scheduled.laneId, `${label}.laneId`);

    const expectedSegmentIndex = committedLaneOrder.indexOf(scheduled.laneId);
    const expectedRecordBase = expectedSegmentIndex * BUCKET_COUNT;
    const expectedByteBase = expectedRecordBase * INDEXED_INDIRECT_RECORD_BYTES;
    requireInvariant(row.commandSegmentIndex, expectedSegmentIndex,
      `${label}.commandSegmentIndex`);
    requireInvariant(row.commandRecordBase, expectedRecordBase,
      `${label}.commandRecordBase`);
    requireInvariant(row.commandByteBase, expectedByteBase, `${label}.commandByteBase`);

    requireInvariant(
      row.selectorWriteSerialAtTimingStart,
      first.selectorWriteSerialAtTimingStart,
      `${label}.selectorWriteSerialAtTimingStart`,
    );
    requireInvariant(
      row.strategySelectionSerialAtTimingStart,
      first.strategySelectionSerialAtTimingStart,
      `${label}.strategySelectionSerialAtTimingStart`,
    );
    requireInvariant(
      row.renderCallSerialAtTimingStart,
      first.renderCallSerialAtTimingStart,
      `${label}.renderCallSerialAtTimingStart`,
    );
    const serialOffset = FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES + frameIndex + 1;
    requireInvariant(
      row.selectorWriteSerial,
      row.selectorWriteSerialAtTimingStart + serialOffset,
      `${label}.selectorWriteSerial`,
    );
    requireInvariant(
      row.strategySelectionSerial,
      row.strategySelectionSerialAtTimingStart + serialOffset,
      `${label}.strategySelectionSerial`,
    );
    requireInvariant(
      row.renderCallSerial,
      row.renderCallSerialAtTimingStart + serialOffset,
      `${label}.renderCallSerial`,
    );
    if (frameIndex > 0) {
      requireInvariant(row.gpuFrameId, rows[frameIndex - 1].gpuFrameId + 1,
        `${label}.gpuFrameId`);
    }
    blockRows[scheduled.crossoverBlockIndex].push(row);
  }

  const blocks = blockRows.map((records, crossoverBlockIndex) => {
    if (records.length !== FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE) {
      reject(`trial ${JSON.stringify(first.trialId)} block ${crossoverBlockIndex} is incomplete.`);
    }
    const portable = records
      .filter((row) => row.laneId === PORTABLE)
      .map((row) => row.gpuRenderMs);
    const feature = records
      .filter((row) => row.laneId === FEATURE)
      .map((row) => row.gpuRenderMs);
    if (portable.length !== 4 || feature.length !== 4) {
      reject(
        `trial ${JSON.stringify(first.trialId)} block ${crossoverBlockIndex} `
          + 'is not lane-balanced.',
      );
    }
    const portableMeanMs = mean(portable);
    const featureMeanMs = mean(feature);
    const deltaMs = featureMeanMs - portableMeanMs;
    return {
      crossoverBlockIndex,
      pattern: records[0].crossoverPattern,
      portableMeanMs,
      featureMeanMs,
      pooledMeanMs: mean(records.map((row) => row.gpuRenderMs)),
      deltaMs,
      deltaPercent: (deltaMs / portableMeanMs) * 100,
    };
  });

  const quarterFrameCount = QUARTER_BLOCKS * FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE;
  const firstQuarterMeanMs = mean(rows.slice(0, quarterFrameCount).map(
    (row) => row.gpuRenderMs,
  ));
  const lastQuarterMeanMs = mean(rows.slice(-quarterFrameCount).map(
    (row) => row.gpuRenderMs,
  ));
  const driftDeltaMs = lastQuarterMeanMs - firstQuarterMeanMs;

  return {
    trialId: first.trialId,
    planIndex: expectedPlanIndex,
    repetitionIndex: expectedRepetition,
    visibilityFraction: expectedVisibility,
    visibilityOrder: [...committedVisibilityOrder],
    visibilityOrderPosition: expectedPosition,
    laneCommandSegmentOrder: [...committedLaneOrder],
    portableCommandSegmentIndex: portableSegmentIndex,
    featureCommandSegmentIndex: featureSegmentIndex,
    superblockOrientationOffset: committedOrientation,
    nRows: rows.length,
    nBlocks: blocks.length,
    blocks,
    estimate: summarizeBlocks(blocks),
    halves: {
      first: summarizeBlocks(blocks.slice(0, HALF_BLOCKS)),
      second: summarizeBlocks(blocks.slice(HALF_BLOCKS)),
    },
    drift: {
      firstQuarterBlocks: QUARTER_BLOCKS,
      lastQuarterBlocks: QUARTER_BLOCKS,
      firstQuarterMeanMs,
      lastQuarterMeanMs,
      deltaMs: driftDeltaMs,
      percent: (driftDeltaMs / firstQuarterMeanMs) * 100,
    },
  };
}

function summarizeEstimates(estimates) {
  return {
    n: estimates.length,
    median: {
      deltaMs: median(estimates.map((estimate) => estimate.deltaMs)),
      deltaPercent: median(estimates.map((estimate) => estimate.deltaPercent)),
    },
    negativeCount: estimates.filter((estimate) => estimate.deltaMs < 0).length,
  };
}

function binaryStratum(trials, field, levels) {
  const strata = levels.map((level) => {
    const selected = trials.filter((trial) => trial[field] === level);
    if (selected.length !== FIRST_INSTANCE_CROSSOVER_REPETITIONS / 2) {
      reject(`${field}=${level} has ${selected.length} high-visibility trials; expected 6.`);
    }
    return {
      level,
      ...summarizeEstimates(selected.map((trial) => trial.estimate)),
    };
  });
  return {
    field,
    strata,
    interaction: {
      deltaMs: strata[1].median.deltaMs - strata[0].median.deltaMs,
      deltaPercent: strata[1].median.deltaPercent - strata[0].median.deltaPercent,
    },
  };
}

function halfStratum(trials) {
  const strata = ['first', 'second'].map((half) => ({
    level: half,
    ...summarizeEstimates(trials.map((trial) => trial.halves[half])),
  }));
  return {
    field: 'measurementHalf',
    strata,
    interaction: {
      deltaMs: strata[1].median.deltaMs - strata[0].median.deltaMs,
      deltaPercent: strata[1].median.deltaPercent - strata[0].median.deltaPercent,
    },
  };
}

function visibilityContrast(trials) {
  return {
    ...summarizeEstimates(trials.map((trial) => trial.estimate)),
    strata: {
      physicalCommandSegment: binaryStratum(
        trials,
        'portableCommandSegmentIndex',
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
  };
}

function summarizeDrift(trials) {
  return {
    n: trials.length,
    median: {
      deltaMs: median(trials.map((trial) => trial.drift.deltaMs)),
      percent: median(trials.map((trial) => trial.drift.percent)),
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
    bounds: { absoluteMs: MATERIAL_MS, percent: MATERIAL_PERCENT },
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
  const highVisibilityMaterial = {
    pass: high.median.deltaMs <= -MATERIAL_MS
      && high.median.deltaPercent <= -MATERIAL_PERCENT,
    threshold: { absoluteMs: -MATERIAL_MS, percent: -MATERIAL_PERCENT },
    conjunction: 'both thresholds are required',
    observed: high.median,
  };
  const highVisibilityDirection = {
    pass: high.negativeCount >= REQUIRED_SIGN_COUNT,
    requiredNegative: REQUIRED_SIGN_COUNT,
    observedNegative: high.negativeCount,
    total: high.n,
  };
  const highStrata = Object.fromEntries(
    Object.entries(high.strata).map(([name, value]) => [name, stratumGate(value)]),
  );
  const lowEstimates = trialsByVisibility[String(LOW_VISIBILITY)].map(
    (trial) => trial.estimate,
  );
  const lowCounts = {
    belowMsUpperBound: lowEstimates.filter(
      (estimate) => estimate.deltaMs < LOW_REGRESSION_MS,
    ).length,
    belowPercentUpperBound: lowEstimates.filter(
      (estimate) => estimate.deltaPercent < LOW_REGRESSION_PERCENT,
    ).length,
  };
  const lowMedianBelow = low.median.deltaMs < LOW_REGRESSION_MS
    && low.median.deltaPercent < LOW_REGRESSION_PERCENT;
  const lowVisibilityRegression = {
    pass: lowMedianBelow
      && Object.values(lowCounts).every((count) => count >= REQUIRED_SIGN_COUNT),
    medianBelowBothBounds: lowMedianBelow,
    requiredBelowEachBound: REQUIRED_SIGN_COUNT,
    bounds: { absoluteMs: LOW_REGRESSION_MS, percent: LOW_REGRESSION_PERCENT },
    counts: lowCounts,
    observed: low.median,
  };
  const pairedHighMinusLow = {
    pass: contrasts.pairedHighMinusLow.median.deltaMs <= -PAIRED_MATERIAL_MS,
    thresholdMs: -PAIRED_MATERIAL_MS,
    observed: contrasts.pairedHighMinusLow.median,
    nPairs: contrasts.pairedHighMinusLow.nPairs,
  };
  const drift = {
    overall: driftGate(contrasts.drift.overall),
    visibilities: Object.fromEntries(
      FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS.map((visibility) => [
        String(visibility),
        driftGate(contrasts.drift.visibilities[String(visibility)]),
      ]),
    ),
  };
  drift.pass = drift.overall.pass
    && Object.values(drift.visibilities).every((gate) => gate.pass);

  const gates = {
    highVisibilityMaterial,
    highVisibilityDirection,
    highVisibilityStrata: {
      pass: Object.values(highStrata).every((gate) => gate.pass),
      factors: highStrata,
    },
    lowVisibilityRegression,
    pairedHighMinusLow,
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
 * Analyze normalized measured rows from the preregistered indirect-firstInstance
 * crossover. This function rejects any non-exact 24-trial plan, schedule,
 * command-segment mapping, timestamp shape, or selection/render serial stream.
 * Artifact, provenance, environment, and validation-payload integrity remain
 * the responsibility of the run-directory verifier that invokes this API.
 */
export function summarizeFirstInstanceCrossoverRows(rows) {
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
  const trials = [];
  const seenTrialIds = new Set();
  let previousTrialLastGpuFrameId = null;
  let previousTrialLastRenderCallSerial = null;
  for (let planIndex = 0; planIndex < EXPECTED_TRIALS; planIndex += 1) {
    const start = planIndex * FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES;
    const trialRows = normalized.slice(start, start + FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES);
    if (trialRows.some((row) => row.planIndex !== planIndex)) {
      reject(`input rows are not contiguous in committed plan-index order at plan ${planIndex}.`);
    }
    const trialId = trialRows[0].trialId;
    if (seenTrialIds.has(trialId)) reject(`trialId ${JSON.stringify(trialId)} is reused.`);
    seenTrialIds.add(trialId);
    if (previousTrialLastGpuFrameId !== null
      && trialRows[0].gpuFrameId <= previousTrialLastGpuFrameId) {
      reject(`trial ${planIndex} violates global GPU frame chronology.`);
    }
    if (previousTrialLastRenderCallSerial !== null
      && trialRows[0].renderCallSerialAtTimingStart <= previousTrialLastRenderCallSerial) {
      reject(`trial ${planIndex} violates global render-call chronology.`);
    }
    previousTrialLastGpuFrameId = trialRows.at(-1).gpuFrameId;
    previousTrialLastRenderCallSerial = trialRows.at(-1).renderCallSerial;
    trials.push(summarizeTrial(trialRows, planIndex, runId));
  }
  if (seenTrialIds.size !== EXPECTED_TRIALS) {
    reject(`input has ${seenTrialIds.size} trials; expected exactly ${EXPECTED_TRIALS}.`);
  }

  const trialsByVisibility = Object.fromEntries(
    FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS.map((visibility) => [
      String(visibility),
      trials.filter((trial) => trial.visibilityFraction === visibility),
    ]),
  );
  for (const visibility of FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS) {
    const selected = trialsByVisibility[String(visibility)];
    if (selected.length !== FIRST_INSTANCE_CROSSOVER_REPETITIONS) {
      reject(`${visibility} visibility has ${selected.length} trials; expected exactly 12.`);
    }
    const repetitions = new Set(selected.map((trial) => trial.repetitionIndex));
    if (repetitions.size !== FIRST_INSTANCE_CROSSOVER_REPETITIONS
      || Array.from({ length: FIRST_INSTANCE_CROSSOVER_REPETITIONS }, (_, index) => index)
        .some((index) => !repetitions.has(index))) {
      reject(`${visibility} visibility does not contain exactly repetitions 0 through 11.`);
    }
  }

  const pairedTrials = [];
  for (let repetitionIndex = 0;
    repetitionIndex < FIRST_INSTANCE_CROSSOVER_REPETITIONS;
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
    if (!sameArray(high.laneCommandSegmentOrder, low.laneCommandSegmentOrder)
      || high.superblockOrientationOffset !== low.superblockOrientationOffset) {
      reject(`repetition ${repetitionIndex} changes crossover factors between visibilities.`);
    }
    pairedTrials.push({
      repetitionIndex,
      highTrialId: high.trialId,
      lowTrialId: low.trialId,
      high: high.estimate,
      low: low.estimate,
      delta: {
        deltaMs: high.estimate.deltaMs - low.estimate.deltaMs,
        deltaPercent: high.estimate.deltaPercent - low.estimate.deltaPercent,
      },
    });
  }

  const contrasts = {
    visibilities: Object.fromEntries(
      FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS.map((visibility) => [
        String(visibility),
        visibilityContrast(trialsByVisibility[String(visibility)]),
      ]),
    ),
    pairedHighMinusLow: {
      nPairs: pairedTrials.length,
      pairs: pairedTrials,
      median: {
        deltaMs: median(pairedTrials.map((pair) => pair.delta.deltaMs)),
        deltaPercent: median(pairedTrials.map((pair) => pair.delta.deltaPercent)),
      },
    },
    drift: {
      overall: summarizeDrift(trials),
      visibilities: Object.fromEntries(
        FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS.map((visibility) => [
          String(visibility),
          summarizeDrift(trialsByVisibility[String(visibility)]),
        ]),
      ),
    },
  };
  const preregisteredDecision = evaluateNumericalGates(contrasts, trialsByVisibility);

  return {
    schemaVersion: 1,
    kind: 'indirect-first-instance-crossover-summary',
    deltaConvention: 'feature minus portable; negative values favor indirect firstInstance',
    protocol: {
      objectCount: OBJECT_COUNT,
      bucketCount: BUCKET_COUNT,
      repetitions: FIRST_INSTANCE_CROSSOVER_REPETITIONS,
      visibilityLevels: [...FIRST_INSTANCE_CROSSOVER_VISIBILITY_LEVELS],
      measuredBlocksPerTrial: FIRST_INSTANCE_CROSSOVER_MEASURED_BLOCKS,
      measuredFramesPerTrial: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
      inferentialUnit: 'repetition-level trial estimate',
    },
    nRows: rows.length,
    nTrials: trials.length,
    trials,
    contrasts,
    preregisteredDecision,
    preregisteredNumericalDecision: preregisteredDecision,
  };
}
