import {
  FROZEN_CROSSOVER_MEASURED_BLOCKS,
  FROZEN_CROSSOVER_MEASURED_FRAMES,
  frozenCrossoverFrame,
} from '../src/benchmark/frozen-crossover-schedule.js';
import {
  FROZEN_DEPTH_CROSSOVER_LANES,
  FROZEN_DEPTH_CROSSOVER_REPETITIONS,
} from '../src/benchmark/plan.js';

const [FRONT_TO_BACK, REVERSE] = FROZEN_DEPTH_CROSSOVER_LANES;
const LAYOUTS = Object.freeze(['high-overlap', 'low-overlap']);
const OBJECT_COUNT = 65_536;
const BLOCK_SIZE = FROZEN_CROSSOVER_MEASURED_FRAMES
  / FROZEN_CROSSOVER_MEASURED_BLOCKS;
const HALF_BLOCKS = FROZEN_CROSSOVER_MEASURED_BLOCKS / 2;
const QUARTER_BLOCKS = FROZEN_CROSSOVER_MEASURED_BLOCKS / 4;
const MATERIAL_MS = 0.10;
const MATERIAL_PERCENT = 10;
const DRIFT_PERCENT = 5;
const REQUIRED_SIGN_COUNT = 10;

function reject(message) {
  throw new Error(`Frozen crossover analysis rejected: ${message}`);
}

function finiteNumber(value, label, { minimum = -Infinity, exclusiveMinimum = false } = {}) {
  if (!Number.isFinite(value)
    || (exclusiveMinimum ? value <= minimum : value < minimum)) {
    reject(`${label} must be a finite number ${exclusiveMinimum ? '>' : '>='} ${minimum}.`);
  }
  return value;
}

function integer(value, label, { minimum = -Infinity, maximum = Infinity } = {}) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    reject(`${label} must be an integer in [${minimum}, ${maximum}].`);
  }
  return value;
}

function nonemptyString(value, label) {
  if (typeof value !== 'string' || value.length === 0) {
    reject(`${label} must be a nonempty string.`);
  }
  return value;
}

function mean(values) {
  if (values.length === 0) reject('cannot compute a mean of an empty sample.');
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values) {
  if (values.length === 0) reject('cannot compute a median of an empty sample.');
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

function parseStorageOrder(value, label) {
  if (value === undefined || value === null) return null;
  if (typeof value !== 'string') {
    reject(`${label} must be a pipe-delimited string when present.`);
  }
  const order = value.split('|');
  if (order.length !== 2
    || new Set(order).size !== 2
    || FROZEN_DEPTH_CROSSOVER_LANES.some((lane) => !order.includes(lane))) {
    reject(`${label} must be the two frozen lane IDs separated by "|".`);
  }
  return order;
}

function mappingForRow(row, label) {
  const storageOrder = parseStorageOrder(row.laneStorageOrder, `${label}.laneStorageOrder`);
  const hasFrontLaneBase = row.frontLaneBase !== undefined && row.frontLaneBase !== null;
  const declaredFrontLaneBase = hasFrontLaneBase
    ? integer(row.frontLaneBase, `${label}.frontLaneBase`, {
      minimum: 0,
      maximum: OBJECT_COUNT,
    })
    : null;
  if (storageOrder === null && declaredFrontLaneBase === null) {
    reject(`${label} must provide laneStorageOrder or frontLaneBase.`);
  }

  const orderFrontLaneBase = storageOrder === null
    ? null
    : storageOrder.indexOf(FRONT_TO_BACK) * OBJECT_COUNT;
  if (orderFrontLaneBase !== null
    && declaredFrontLaneBase !== null
    && orderFrontLaneBase !== declaredFrontLaneBase) {
    reject(`${label} has inconsistent laneStorageOrder and frontLaneBase.`);
  }
  const frontLaneBase = declaredFrontLaneBase ?? orderFrontLaneBase;
  if (frontLaneBase !== 0 && frontLaneBase !== OBJECT_COUNT) {
    reject(`${label}.frontLaneBase must be exactly 0 or ${OBJECT_COUNT}.`);
  }
  const normalizedOrder = storageOrder ?? (frontLaneBase === 0
    ? [FRONT_TO_BACK, REVERSE]
    : [REVERSE, FRONT_TO_BACK]);
  return {
    laneStorageOrder: normalizedOrder,
    frontLaneBase,
    reverseLaneBase: frontLaneBase === 0 ? OBJECT_COUNT : 0,
  };
}

function requireInvariant(actual, expected, label) {
  if (actual !== expected) {
    reject(`${label} is ${JSON.stringify(actual)}; expected ${JSON.stringify(expected)}.`);
  }
}

function summarizeBlocks(blocks) {
  return {
    deltaMs: median(blocks.map((block) => block.deltaMs)),
    deltaPercent: median(blocks.map((block) => block.deltaPercent)),
  };
}

function summarizeTrial(rows, trialId) {
  if (rows.length !== FROZEN_CROSSOVER_MEASURED_FRAMES) {
    reject(
      `trial ${JSON.stringify(trialId)} has ${rows.length} rows; expected ${FROZEN_CROSSOVER_MEASURED_FRAMES}.`,
    );
  }
  const first = rows[0];
  const repetitionIndex = integer(first.repetitionIndex, `${trialId}.repetitionIndex`, {
    minimum: 0,
    maximum: FROZEN_DEPTH_CROSSOVER_REPETITIONS - 1,
  });
  if (!LAYOUTS.includes(first.layout)) {
    reject(`${trialId}.layout must be high-overlap or low-overlap.`);
  }
  const layout = first.layout;
  const layoutOrderPosition = integer(
    first.layoutOrderPosition,
    `${trialId}.layoutOrderPosition`,
    { minimum: 0, maximum: 1 },
  );
  const superblockOrientationOffset = integer(
    first.superblockOrientationOffset,
    `${trialId}.superblockOrientationOffset`,
    { minimum: 0, maximum: 1 },
  );
  const mapping = mappingForRow(first, `${trialId}[0]`);
  const blockRows = Array.from(
    { length: FROZEN_CROSSOVER_MEASURED_BLOCKS },
    () => [],
  );

  for (let frameIndex = 0; frameIndex < rows.length; frameIndex += 1) {
    const row = rows[frameIndex];
    const label = `${trialId}[${frameIndex}]`;
    if (row === null || typeof row !== 'object' || Array.isArray(row)) {
      reject(`${label} must be an object.`);
    }
    requireInvariant(row.trialId, trialId, `${label}.trialId`);
    requireInvariant(row.repetitionIndex, repetitionIndex, `${label}.repetitionIndex`);
    requireInvariant(row.layout, layout, `${label}.layout`);
    requireInvariant(
      row.layoutOrderPosition,
      layoutOrderPosition,
      `${label}.layoutOrderPosition`,
    );
    requireInvariant(
      row.superblockOrientationOffset,
      superblockOrientationOffset,
      `${label}.superblockOrientationOffset`,
    );
    const rowMapping = mappingForRow(row, label);
    if (!sameArray(rowMapping.laneStorageOrder, mapping.laneStorageOrder)
      || rowMapping.frontLaneBase !== mapping.frontLaneBase) {
      reject(`${label} changes the trial lane/base mapping.`);
    }

    const expected = frozenCrossoverFrame(frameIndex, superblockOrientationOffset);
    requireInvariant(
      row.crossoverBlockIndex,
      expected.crossoverBlockIndex,
      `${label}.crossoverBlockIndex`,
    );
    requireInvariant(
      row.withinBlockPosition,
      expected.withinBlockPosition,
      `${label}.withinBlockPosition`,
    );
    requireInvariant(row.crossoverPattern, expected.pattern, `${label}.crossoverPattern`);
    requireInvariant(row.laneId, expected.laneId, `${label}.laneId`);
    const expectedLaneBase = row.laneId === FRONT_TO_BACK
      ? mapping.frontLaneBase
      : mapping.reverseLaneBase;
    requireInvariant(row.laneBase, expectedLaneBase, `${label}.laneBase`);
    finiteNumber(row.gpuRenderMs, `${label}.gpuRenderMs`, {
      minimum: 0,
      exclusiveMinimum: true,
    });
    blockRows[expected.crossoverBlockIndex].push(row);
  }

  const blocks = blockRows.map((records, crossoverBlockIndex) => {
    if (records.length !== BLOCK_SIZE) {
      reject(
        `trial ${JSON.stringify(trialId)} block ${crossoverBlockIndex} is incomplete.`,
      );
    }
    const front = records
      .filter((row) => row.laneId === FRONT_TO_BACK)
      .map((row) => row.gpuRenderMs);
    const reverse = records
      .filter((row) => row.laneId === REVERSE)
      .map((row) => row.gpuRenderMs);
    if (front.length !== 4 || reverse.length !== 4) {
      reject(
        `trial ${JSON.stringify(trialId)} block ${crossoverBlockIndex} is not lane-balanced.`,
      );
    }
    const frontMeanMs = mean(front);
    const reverseMeanMs = mean(reverse);
    const deltaMs = frontMeanMs - reverseMeanMs;
    return {
      crossoverBlockIndex,
      pattern: records[0].crossoverPattern,
      frontMeanMs,
      reverseMeanMs,
      pooledMeanMs: mean(records.map((row) => row.gpuRenderMs)),
      deltaMs,
      deltaPercent: (deltaMs / reverseMeanMs) * 100,
    };
  });

  const firstQuarterRows = rows.slice(0, QUARTER_BLOCKS * BLOCK_SIZE);
  const lastQuarterRows = rows.slice(-QUARTER_BLOCKS * BLOCK_SIZE);
  const firstQuarterMeanMs = mean(firstQuarterRows.map((row) => row.gpuRenderMs));
  const lastQuarterMeanMs = mean(lastQuarterRows.map((row) => row.gpuRenderMs));
  const driftMs = lastQuarterMeanMs - firstQuarterMeanMs;

  return {
    trialId,
    repetitionIndex,
    layout,
    layoutOrderPosition,
    laneStorageOrder: [...mapping.laneStorageOrder],
    laneStorageOrderSignature: mapping.laneStorageOrder.join('|'),
    frontLaneBase: mapping.frontLaneBase,
    reverseLaneBase: mapping.reverseLaneBase,
    superblockOrientationOffset,
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
      deltaMs: driftMs,
      percent: (driftMs / firstQuarterMeanMs) * 100,
    },
  };
}

function requireFactorBalance(trials, layout) {
  const factorDefinitions = [
    ['frontLaneBase', [0, OBJECT_COUNT]],
    ['superblockOrientationOffset', [0, 1]],
    ['layoutOrderPosition', [0, 1]],
  ];
  for (const [field, levels] of factorDefinitions) {
    for (const level of levels) {
      const count = trials.filter((trial) => trial[field] === level).length;
      if (count !== FROZEN_DEPTH_CROSSOVER_REPETITIONS / 2) {
        reject(`${layout} ${field}=${level} has ${count} trials; expected 6.`);
      }
    }
  }
  for (let leftIndex = 0; leftIndex < factorDefinitions.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1;
      rightIndex < factorDefinitions.length;
      rightIndex += 1) {
      const [leftField, leftLevels] = factorDefinitions[leftIndex];
      const [rightField, rightLevels] = factorDefinitions[rightIndex];
      for (const left of leftLevels) {
        for (const right of rightLevels) {
          const count = trials.filter(
            (trial) => trial[leftField] === left && trial[rightField] === right,
          ).length;
          if (count !== FROZEN_DEPTH_CROSSOVER_REPETITIONS / 4) {
            reject(
              `${layout} ${leftField}=${left}/${rightField}=${right} has ${count} trials; expected 3.`,
            );
          }
        }
      }
    }
  }
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
      deltaPercent:
        strata[1].median.deltaPercent - strata[0].median.deltaPercent,
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
      deltaPercent:
        strata[1].median.deltaPercent - strata[0].median.deltaPercent,
    },
  };
}

function layoutContrast(trials) {
  return {
    ...summarizeEstimates(trials.map((trial) => trial.estimate)),
    strata: {
      physicalBase: binaryStratum(trials, 'frontLaneBase', [0, OBJECT_COUNT]),
      startingOrientation: binaryStratum(
        trials,
        'superblockOrientationOffset',
        [0, 1],
      ),
      layoutOrderPosition: binaryStratum(trials, 'layoutOrderPosition', [0, 1]),
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
  const levelsNegative = stratum.strata.every(
    (record) => record.median.deltaMs < 0,
  );
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
    pass: Math.abs(summary.median.deltaMs) < MATERIAL_MS
      && Math.abs(summary.median.percent) < DRIFT_PERCENT,
    bounds: { absoluteMs: MATERIAL_MS, percent: DRIFT_PERCENT },
    observed: summary.median,
    n: summary.n,
  };
}

function evaluateGates(contrasts, trialsByLayout) {
  const high = contrasts.layouts['high-overlap'];
  const low = contrasts.layouts['low-overlap'];
  const highMaterial = {
    pass: high.median.deltaMs <= -MATERIAL_MS
      || high.median.deltaPercent <= -MATERIAL_PERCENT,
    threshold: { absoluteMs: -MATERIAL_MS, percent: -MATERIAL_PERCENT },
    observed: high.median,
  };
  const highDirection = {
    pass: high.negativeCount >= REQUIRED_SIGN_COUNT,
    requiredNegative: REQUIRED_SIGN_COUNT,
    observedNegative: high.negativeCount,
    total: high.n,
  };
  const highStrata = Object.fromEntries(
    Object.entries(high.strata).map(([name, value]) => [name, stratumGate(value)]),
  );
  const lowEstimates = trialsByLayout['low-overlap'].map((trial) => trial.estimate);
  const lowCounts = {
    msAboveLower: lowEstimates.filter((estimate) => estimate.deltaMs > -MATERIAL_MS).length,
    msBelowUpper: lowEstimates.filter((estimate) => estimate.deltaMs < MATERIAL_MS).length,
    percentAboveLower: lowEstimates.filter(
      (estimate) => estimate.deltaPercent > -MATERIAL_PERCENT,
    ).length,
    percentBelowUpper: lowEstimates.filter(
      (estimate) => estimate.deltaPercent < MATERIAL_PERCENT,
    ).length,
  };
  const lowMedianWithin = low.median.deltaMs > -MATERIAL_MS
    && low.median.deltaMs < MATERIAL_MS
    && low.median.deltaPercent > -MATERIAL_PERCENT
    && low.median.deltaPercent < MATERIAL_PERCENT;
  const lowOverlapEquivalence = {
    pass: lowMedianWithin
      && Object.values(lowCounts).every((count) => count >= REQUIRED_SIGN_COUNT),
    medianWithinBounds: lowMedianWithin,
    requiredInsideEachSide: REQUIRED_SIGN_COUNT,
    bounds: {
      absoluteMs: [-MATERIAL_MS, MATERIAL_MS],
      percent: [-MATERIAL_PERCENT, MATERIAL_PERCENT],
    },
    counts: lowCounts,
    observed: low.median,
  };
  const pairedHighMinusLow = {
    pass: contrasts.pairedHighMinusLow.median.deltaMs <= -MATERIAL_MS,
    thresholdMs: -MATERIAL_MS,
    observed: contrasts.pairedHighMinusLow.median,
    nPairs: contrasts.pairedHighMinusLow.nPairs,
  };
  const drift = {
    overall: driftGate(contrasts.drift.overall),
    layouts: Object.fromEntries(
      LAYOUTS.map((layout) => [layout, driftGate(contrasts.drift.layouts[layout])]),
    ),
  };
  drift.pass = drift.overall.pass
    && Object.values(drift.layouts).every((gate) => gate.pass);

  const gates = {
    highOverlapMaterial: highMaterial,
    highOverlapDirection: highDirection,
    highOverlapStrata: {
      pass: Object.values(highStrata).every((gate) => gate.pass),
      factors: highStrata,
    },
    lowOverlapEquivalence,
    pairedHighMinusLow,
    drift,
  };
  const failedGates = Object.entries(gates)
    .filter(([, gate]) => gate.pass !== true)
    .map(([name]) => name);
  return {
    status: 'evaluated',
    pass: failedGates.length === 0,
    failedGates,
    gates,
  };
}

/**
 * Analyze normalized measured rows from the preregistered frozen depth
 * crossover. Invalid schedules or matrix shapes throw instead of yielding a
 * partial result.
 */
export function summarizeFrozenCrossoverRows(rows) {
  if (!Array.isArray(rows)) reject('input rows must be an array.');
  const expectedRows = FROZEN_DEPTH_CROSSOVER_REPETITIONS
    * LAYOUTS.length
    * FROZEN_CROSSOVER_MEASURED_FRAMES;
  if (rows.length !== expectedRows) {
    reject(`input has ${rows.length} rows; expected exactly ${expectedRows}.`);
  }

  const grouped = new Map();
  for (const [index, row] of rows.entries()) {
    if (row === null || typeof row !== 'object' || Array.isArray(row)) {
      reject(`row ${index} must be an object.`);
    }
    const trialId = nonemptyString(row.trialId, `row ${index}.trialId`);
    let group = grouped.get(trialId);
    if (group === undefined) {
      group = [];
      grouped.set(trialId, group);
    }
    group.push(row);
  }
  if (grouped.size !== FROZEN_DEPTH_CROSSOVER_REPETITIONS * LAYOUTS.length) {
    reject(`input has ${grouped.size} trials; expected exactly 24.`);
  }

  const trials = [...grouped].map(([trialId, trialRows]) => (
    summarizeTrial(trialRows, trialId)
  )).sort((left, right) => (
    left.repetitionIndex - right.repetitionIndex
    || left.layoutOrderPosition - right.layoutOrderPosition
    || left.trialId.localeCompare(right.trialId)
  ));

  const trialsByLayout = Object.fromEntries(
    LAYOUTS.map((layout) => [layout, trials.filter((trial) => trial.layout === layout)]),
  );
  for (const layout of LAYOUTS) {
    const selected = trialsByLayout[layout];
    if (selected.length !== FROZEN_DEPTH_CROSSOVER_REPETITIONS) {
      reject(`${layout} has ${selected.length} trials; expected exactly 12.`);
    }
    const repetitions = new Set(selected.map((trial) => trial.repetitionIndex));
    if (repetitions.size !== FROZEN_DEPTH_CROSSOVER_REPETITIONS
      || Array.from({ length: FROZEN_DEPTH_CROSSOVER_REPETITIONS }, (_, index) => index)
        .some((index) => !repetitions.has(index))) {
      reject(`${layout} does not contain exactly repetitions 0 through 11.`);
    }
    requireFactorBalance(selected, layout);
  }

  const pairedTrials = [];
  for (let repetitionIndex = 0;
    repetitionIndex < FROZEN_DEPTH_CROSSOVER_REPETITIONS;
    repetitionIndex += 1) {
    const high = trialsByLayout['high-overlap'].find(
      (trial) => trial.repetitionIndex === repetitionIndex,
    );
    const low = trialsByLayout['low-overlap'].find(
      (trial) => trial.repetitionIndex === repetitionIndex,
    );
    if (high.layoutOrderPosition === low.layoutOrderPosition) {
      reject(`repetition ${repetitionIndex} does not use complementary layout positions.`);
    }
    if (high.frontLaneBase !== low.frontLaneBase
      || high.superblockOrientationOffset !== low.superblockOrientationOffset) {
      reject(`repetition ${repetitionIndex} changes crossover factors between layouts.`);
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
    layouts: Object.fromEntries(
      LAYOUTS.map((layout) => [layout, layoutContrast(trialsByLayout[layout])]),
    ),
    pairedHighMinusLow: {
      nPairs: pairedTrials.length,
      pairs: pairedTrials,
      median: {
        deltaMs: median(pairedTrials.map((pair) => pair.delta.deltaMs)),
        deltaPercent: median(
          pairedTrials.map((pair) => pair.delta.deltaPercent),
        ),
      },
    },
    drift: {
      overall: summarizeDrift(trials),
      layouts: Object.fromEntries(
        LAYOUTS.map((layout) => [layout, summarizeDrift(trialsByLayout[layout])]),
      ),
    },
  };

  return {
    schemaVersion: 1,
    kind: 'frozen-depth-crossover-summary',
    deltaConvention: 'front-to-back minus reverse; negative values favor front-to-back',
    protocol: {
      objectCount: OBJECT_COUNT,
      repetitions: FROZEN_DEPTH_CROSSOVER_REPETITIONS,
      layouts: [...LAYOUTS],
      measuredBlocksPerTrial: FROZEN_CROSSOVER_MEASURED_BLOCKS,
      measuredFramesPerTrial: FROZEN_CROSSOVER_MEASURED_FRAMES,
      inferentialUnit: 'repetition-level trial estimate',
    },
    nRows: rows.length,
    nTrials: trials.length,
    trials,
    contrasts,
    preregisteredDecision: evaluateGates(contrasts, trialsByLayout),
  };
}
