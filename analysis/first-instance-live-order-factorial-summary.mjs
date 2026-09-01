import {
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELL_COUNT,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_FACTOR_KEYS,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_COUNT,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIAL_COUNT,
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIALS_PER_SESSION,
  buildFirstInstanceLiveOrderFactorialPlan,
} from '../src/benchmark/first-instance-live-order-factorial-plan.js';

const METRIC_KEYS = Object.freeze([
  'gpuPassTotal',
  'gpuRender',
  'gpuCompute',
]);
const RESPONSE_KEYS = Object.freeze(['deltaMs', 'deltaPercent']);
const CONTRAST_FACTOR_SETS = Object.freeze([
  Object.freeze(['C']),
  Object.freeze(['K']),
  Object.freeze(['R']),
  Object.freeze(['T']),
  Object.freeze(['C', 'K']),
  Object.freeze(['C', 'R']),
  Object.freeze(['C', 'T']),
  Object.freeze(['K', 'R']),
  Object.freeze(['K', 'T']),
  Object.freeze(['R', 'T']),
  Object.freeze(['C', 'K', 'R']),
  Object.freeze(['C', 'K', 'T']),
  Object.freeze(['C', 'R', 'T']),
  Object.freeze(['K', 'R', 'T']),
  Object.freeze(['C', 'K', 'R', 'T']),
]);

export const FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CONTRASTS = Object.freeze(
  CONTRAST_FACTOR_SETS.map((factors) => factors.join('')),
);

function reject(message) {
  throw new Error(`Live first-instance order-factorial analysis rejected: ${message}`);
}

function requireRecord(value, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    reject(`${label} must be an object.`);
  }
  return value;
}

function finiteNumber(value, label) {
  if (!Number.isFinite(value)) reject(`${label} must be finite.`);
  return value;
}

function mean(values, label) {
  if (values.length === 0 || values.some((value) => !Number.isFinite(value))) {
    reject(`${label} must be a nonempty finite sample.`);
  }
  const result = values.reduce((sum, value) => sum + value, 0) / values.length;
  if (!Number.isFinite(result)) reject(`derived ${label} mean is non-finite.`);
  return result;
}

function sameValue(left, right) {
  if (Object.is(left, right)) return true;
  if (Array.isArray(left) || Array.isArray(right)) {
    return Array.isArray(left)
      && Array.isArray(right)
      && left.length === right.length
      && left.every((value, index) => sameValue(value, right[index]));
  }
  if (left === null || right === null
    || typeof left !== 'object' || typeof right !== 'object') {
    return false;
  }
  const leftKeys = Object.keys(left).sort();
  const rightKeys = Object.keys(right).sort();
  return leftKeys.length === rightKeys.length
    && leftKeys.every((key, index) => (
      key === rightKeys[index] && sameValue(left[key], right[key])
    ));
}

function copyFactorLevels(levels) {
  return Object.fromEntries(
    FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_FACTOR_KEYS.map((key) => [key, levels[key]]),
  );
}

function copyMetrics(metrics) {
  return Object.fromEntries(METRIC_KEYS.map((metricKey) => [metricKey, {
    deltaMs: metrics[metricKey].deltaMs,
    deltaPercent: metrics[metricKey].deltaPercent,
  }]));
}

function meanMetrics(samples, label) {
  return Object.fromEntries(METRIC_KEYS.map((metricKey) => [metricKey,
    Object.fromEntries(RESPONSE_KEYS.map((responseKey) => [responseKey, mean(
      samples.map((sample) => sample[metricKey][responseKey]),
      `${label} ${metricKey}.${responseKey}`,
    )])),
  ]));
}

function validateRecord(record, expectedTrial, recordIndex) {
  const label = `record ${recordIndex}`;
  requireRecord(record, label);
  for (const [key, expectedValue] of Object.entries(expectedTrial)) {
    if (!Object.hasOwn(record, key) || !sameValue(record[key], expectedValue)) {
      reject(
        `${label}.${key} differs from generated plan index ${expectedTrial.planIndex}.`,
      );
    }
  }

  const trialSummary = requireRecord(record.trialSummary, `${label}.trialSummary`);
  if (trialSummary.trialId !== expectedTrial.trialId) {
    reject(`${label}.trialSummary.trialId differs from the generated plan.`);
  }
  if (trialSummary.visibilityFraction !== expectedTrial.visibilityFraction) {
    reject(`${label}.trialSummary.visibilityFraction differs from the generated plan.`);
  }
  if (trialSummary.superblockOrientationOffset
    !== expectedTrial.superblockOrientationOffset) {
    reject(
      `${label}.trialSummary.superblockOrientationOffset differs from the generated plan.`,
    );
  }
  if (trialSummary.nRows !== 480 || trialSummary.nBlocks !== 60) {
    reject(`${label}.trialSummary must represent exactly 480 rows and 60 blocks.`);
  }

  const estimates = requireRecord(
    trialSummary.estimates,
    `${label}.trialSummary.estimates`,
  );
  const metrics = Object.fromEntries(METRIC_KEYS.map((metricKey) => {
    const estimate = requireRecord(
      estimates[metricKey],
      `${label}.trialSummary.estimates.${metricKey}`,
    );
    return [metricKey, {
      deltaMs: finiteNumber(
        estimate.deltaMs,
        `${label}.trialSummary.estimates.${metricKey}.deltaMs`,
      ),
      deltaPercent: finiteNumber(
        estimate.deltaPercent,
        `${label}.trialSummary.estimates.${metricKey}.deltaPercent`,
      ),
    }];
  }));

  return { plan: expectedTrial, trialSummary, metrics };
}

function validateSessionCellOccurrences(validatedRecords, sessionIndex, cellIndex) {
  const occurrences = validatedRecords.filter(({ plan }) => (
    plan.sessionIndex === sessionIndex && plan.factorialCellIndex === cellIndex
  ));
  const label = `session ${sessionIndex} cell ${cellIndex}`;
  if (occurrences.length !== 2) {
    reject(`${label} has ${occurrences.length} occurrences; expected exactly 2.`);
  }
  const ordered = [...occurrences].sort(
    (left, right) => left.plan.sessionCellOccurrenceIndex
      - right.plan.sessionCellOccurrenceIndex,
  );
  if (ordered[0].plan.sessionCellOccurrenceIndex !== 0
    || ordered[1].plan.sessionCellOccurrenceIndex !== 1
    || ordered[0].plan.permutationDirection !== 'forward'
    || ordered[1].plan.permutationDirection !== 'reverse'
    || ordered[0].plan.permutationPosition + ordered[1].plan.permutationPosition !== 15
    || ordered[0].plan.sessionTrialIndex + ordered[1].plan.sessionTrialIndex !== 31) {
    reject(`${label} is not an exact forward/reverse position pair.`);
  }
  if (new Set(ordered.map(({ plan }) => plan.superblockOrientationOffset)).size !== 2) {
    reject(`${label} does not contain both crossover orientations.`);
  }
  return ordered;
}

function sessionCellTable(validatedRecords, sessionIndex) {
  return Array.from(
    { length: FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELL_COUNT },
    (_, factorialCellIndex) => {
      const occurrences = validateSessionCellOccurrences(
        validatedRecords,
        sessionIndex,
        factorialCellIndex,
      );
      const { plan } = occurrences[0];
      return {
        sessionIndex,
        factorialCellIndex,
        factorialCellId: plan.factorialCellId,
        factorLevels: copyFactorLevels(plan.factorLevels),
        nOccurrences: occurrences.length,
        occurrences: occurrences.map((occurrence) => ({
          trialId: occurrence.plan.trialId,
          planIndex: occurrence.plan.planIndex,
          sessionTrialIndex: occurrence.plan.sessionTrialIndex,
          permutationBlockIndex: occurrence.plan.permutationBlockIndex,
          permutationDirection: occurrence.plan.permutationDirection,
          permutationPosition: occurrence.plan.permutationPosition,
          superblockOrientationOffset: occurrence.plan.superblockOrientationOffset,
          metrics: copyMetrics(occurrence.metrics),
        })),
        metrics: meanMetrics(
          occurrences.map(({ metrics }) => metrics),
          `${labelForCell(sessionIndex, factorialCellIndex)} reverse-pair response`,
        ),
      };
    },
  );
}

function labelForCell(sessionIndex, cellIndex) {
  return `session ${sessionIndex} cell ${cellIndex}`;
}

function contrastSign(factorLevels, factors) {
  return factors.reduce(
    (product, factorKey) => product * (factorLevels[factorKey] === 1 ? 1 : -1),
    1,
  );
}

function summarizeContrasts(cellTable, label) {
  return Object.fromEntries(CONTRAST_FACTOR_SETS.map((factors) => {
    const contrastId = factors.join('');
    const level1 = cellTable.filter(
      (cell) => contrastSign(cell.factorLevels, factors) === 1,
    );
    const level0 = cellTable.filter(
      (cell) => contrastSign(cell.factorLevels, factors) === -1,
    );
    if (level1.length !== 8 || level0.length !== 8) {
      reject(`${label} contrast ${contrastId} is not balanced 8 versus 8.`);
    }
    return [contrastId, {
      contrastId,
      factors: [...factors],
      level0CellCount: level0.length,
      level1CellCount: level1.length,
      metrics: Object.fromEntries(METRIC_KEYS.map((metricKey) => [metricKey,
        Object.fromEntries(RESPONSE_KEYS.map((responseKey) => [responseKey,
          finiteNumber(
            mean(
              level1.map((cell) => cell.metrics[metricKey][responseKey]),
              `${label} ${contrastId} level 1 ${metricKey}.${responseKey}`,
            ) - mean(
              level0.map((cell) => cell.metrics[metricKey][responseKey]),
              `${label} ${contrastId} level 0 ${metricKey}.${responseKey}`,
            ),
            `${label} ${contrastId} ${metricKey}.${responseKey} contrast`,
          ),
        ])),
      ])),
    }];
  }));
}

function pooledCellTable(sessionTables) {
  return Array.from(
    { length: FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELL_COUNT },
    (_, factorialCellIndex) => {
      const cells = sessionTables.map(
        ({ cellTable }) => cellTable[factorialCellIndex],
      );
      const first = cells[0];
      return {
        factorialCellIndex,
        factorialCellId: first.factorialCellId,
        factorLevels: copyFactorLevels(first.factorLevels),
        nSessions: cells.length,
        nOccurrences: cells.reduce((sum, cell) => sum + cell.nOccurrences, 0),
        sessionResponses: cells.map((cell) => ({
          sessionIndex: cell.sessionIndex,
          metrics: copyMetrics(cell.metrics),
        })),
        metrics: meanMetrics(
          cells.map(({ metrics }) => metrics),
          `pooled cell ${factorialCellIndex} session response`,
        ),
      };
    },
  );
}

function sign(value) {
  if (value > 0) return 1;
  if (value < 0) return -1;
  return 0;
}

function sessionSignConcordance(sessions) {
  return Object.fromEntries(FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CONTRASTS.map(
    (contrastId) => [contrastId, {
      contrastId,
      metrics: Object.fromEntries(METRIC_KEYS.map((metricKey) => [metricKey,
        Object.fromEntries(RESPONSE_KEYS.map((responseKey) => {
          const sessionValues = sessions.map(
            (session) => session.contrasts[contrastId].metrics[metricKey][responseKey],
          );
          const sessionSigns = sessionValues.map(sign);
          return [responseKey, {
            sessionValues,
            sessionSigns,
            concordant: sessionSigns.every((value) => value === sessionSigns[0]),
            sameNonzeroSign: sessionSigns[0] !== 0
              && sessionSigns.every((value) => value === sessionSigns[0]),
          }];
        })),
      ])),
    }],
  ));
}

function responseStats(values, label) {
  if (values.length === 0 || values.some((value) => !Number.isFinite(value))) {
    reject(`${label} must be a nonempty finite response sample.`);
  }
  return {
    grandMean: mean(values, `${label} grand`),
    min: Math.min(...values),
    max: Math.max(...values),
    negativeCount: values.filter((value) => value < 0).length,
    zeroCount: values.filter((value) => value === 0).length,
    positiveCount: values.filter((value) => value > 0).length,
  };
}

function summarizeResponseOverview(cellTable, label) {
  return {
    nCellResponses: cellTable.length,
    metrics: Object.fromEntries(METRIC_KEYS.map((metricKey) => [metricKey,
      Object.fromEntries(RESPONSE_KEYS.map((responseKey) => [responseKey, responseStats(
        cellTable.map((cell) => cell.metrics[metricKey][responseKey]),
        `${label} ${metricKey}.${responseKey}`,
      )])),
    ])),
  };
}

/**
 * Descriptively summarize the frozen two-session 2^4 development factorial.
 * No record is removed, replaced, or used to adapt the schedule, and this
 * diagnostic deliberately does not produce a candidate decision.
 */
export function summarizeFirstInstanceLiveOrderFactorial(records, expectedRunId) {
  if (!Array.isArray(records)) reject('input records must be an array.');
  if (records.length !== FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIAL_COUNT) {
    reject(
      `input has ${records.length} records; expected exactly `
        + `${FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIAL_COUNT}.`,
    );
  }
  if (typeof expectedRunId !== 'string' || expectedRunId.trim() === '') {
    reject('expectedRunId must be a nonempty string.');
  }

  const expectedPlan = buildFirstInstanceLiveOrderFactorialPlan({ runId: expectedRunId });
  const validatedRecords = records.map(
    (record, index) => validateRecord(record, expectedPlan[index], index),
  );

  const sessions = Array.from(
    { length: FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_SESSION_COUNT },
    (_, sessionIndex) => {
      const cellTable = sessionCellTable(validatedRecords, sessionIndex);
      return {
        sessionIndex,
        nRecords: validatedRecords.filter(
          ({ plan }) => plan.sessionIndex === sessionIndex,
        ).length,
        nCells: cellTable.length,
        cellTable,
        contrasts: summarizeContrasts(cellTable, `session ${sessionIndex}`),
      };
    },
  );
  if (sessions.some(
    (session) => session.nRecords !== FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_TRIALS_PER_SESSION,
  )) {
    reject('sessions do not each contain the frozen 32-record schedule.');
  }

  const pooledCells = pooledCellTable(sessions);
  return {
    schemaVersion: 1,
    kind: 'first-instance-live-order-factorial-diagnostic-summary',
    decision: 'diagnostic-only-no-candidate-pass',
    analysisScope:
      'descriptive bounded-development factorial; no significance or candidate claim',
    effectConvention: 'level1 mean minus level0 mean',
    interactionCoding:
      'standard product coding with factor level 0 = -1 and factor level 1 = +1',
    runId: expectedRunId,
    metricKeys: [...METRIC_KEYS],
    contrastOrder: [...FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CONTRASTS],
    nRecords: validatedRecords.length,
    nSessions: sessions.length,
    nCells: FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CELL_COUNT,
    sessions,
    pooled: {
      nSessions: sessions.length,
      nCells: pooledCells.length,
      cellTable: pooledCells,
      contrasts: summarizeContrasts(pooledCells, 'pooled sessions'),
    },
    responseOverview: {
      sessions: sessions.map((session) => ({
        sessionIndex: session.sessionIndex,
        ...summarizeResponseOverview(
          session.cellTable,
          `session ${session.sessionIndex} response overview`,
        ),
      })),
      pooled: summarizeResponseOverview(pooledCells, 'pooled response overview'),
    },
    sessionSignConcordance: sessionSignConcordance(sessions),
  };
}
