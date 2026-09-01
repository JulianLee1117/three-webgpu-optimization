import assert from 'node:assert/strict';
import test from 'node:test';

import {
  FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CONTRASTS,
  summarizeFirstInstanceLiveOrderFactorial,
} from '../analysis/first-instance-live-order-factorial-summary.mjs';
import {
  buildFirstInstanceLiveOrderFactorialPlan,
} from '../src/benchmark/first-instance-live-order-factorial-plan.js';

const RUN_ID = 'order-factorial-summary-test';
const METRIC_SCALES = {
  gpuPassTotal: 1,
  gpuRender: 2,
  gpuCompute: 0.25,
};

function closeTo(actual, expected, tolerance = 1e-10) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${actual} is not within ${tolerance} of ${expected}`,
  );
}

function contrastSign(factorLevels, contrastId) {
  return [...contrastId].reduce(
    (product, factorKey) => product * (factorLevels[factorKey] === 1 ? 1 : -1),
    1,
  );
}

function sessionCellResponse(planTrial, responseScale) {
  return -100 + FIRST_INSTANCE_LIVE_ORDER_FACTORIAL_CONTRASTS.reduce(
    (value, contrastId, index) => value
      + (((index + 1) * responseScale) / 2)
        * contrastSign(planTrial.factorLevels, contrastId),
    0,
  );
}

function createRecords() {
  return buildFirstInstanceLiveOrderFactorialPlan({ runId: RUN_ID }).map((trial) => {
    const sessionScale = trial.sessionIndex + 1;
    const reversePairNoise = trial.sessionCellOccurrenceIndex === 0 ? 0.125 : -0.125;
    const estimates = Object.fromEntries(Object.entries(METRIC_SCALES).map(
      ([metricKey, metricScale]) => {
        const response = sessionCellResponse(trial, sessionScale) * metricScale;
        return [metricKey, {
          deltaMs: response + reversePairNoise,
          deltaPercent: (response * 10) + reversePairNoise,
        }];
      },
    ));
    return {
      ...trial,
      trialSummary: {
        trialId: trial.trialId,
        visibilityFraction: trial.visibilityFraction,
        superblockOrientationOffset: trial.superblockOrientationOffset,
        nRows: 480,
        nBlocks: 60,
        estimates,
      },
    };
  });
}

function createCellIndexResponseRecords() {
  return buildFirstInstanceLiveOrderFactorialPlan({ runId: RUN_ID }).map((trial) => {
    const reversePairNoise = trial.sessionCellOccurrenceIndex === 0 ? 0.25 : -0.25;
    const response = trial.factorialCellIndex - 8;
    return {
      ...trial,
      trialSummary: {
        trialId: trial.trialId,
        visibilityFraction: trial.visibilityFraction,
        superblockOrientationOffset: trial.superblockOrientationOffset,
        nRows: 480,
        nBlocks: 60,
        estimates: Object.fromEntries(Object.keys(METRIC_SCALES).map((metricKey) => [
          metricKey,
          {
            deltaMs: response + reversePairNoise,
            deltaPercent: response + reversePairNoise,
          },
        ])),
      },
    };
  });
}

test('factorial analyzer reports all session and pooled standard contrasts descriptively', () => {
  const summary = summarizeFirstInstanceLiveOrderFactorial(createRecords(), RUN_ID);

  assert.equal(summary.kind, 'first-instance-live-order-factorial-diagnostic-summary');
  assert.equal(summary.decision, 'diagnostic-only-no-candidate-pass');
  assert.equal(summary.effectConvention, 'level1 mean minus level0 mean');
  assert.equal(summary.nRecords, 64);
  assert.equal(summary.nSessions, 2);
  assert.equal(summary.nCells, 16);
  assert.deepEqual(summary.contrastOrder, [
    'C', 'K', 'R', 'T',
    'CK', 'CR', 'CT', 'KR', 'KT', 'RT',
    'CKR', 'CKT', 'CRT', 'KRT',
    'CKRT',
  ]);
  assert.equal(summary.sessions.length, 2);
  assert.ok(summary.sessions.every((session) => (
    session.nRecords === 32
    && session.nCells === 16
    && session.cellTable.every((cell) => (
      cell.nOccurrences === 2
      && cell.occurrences[0].sessionTrialIndex
        + cell.occurrences[1].sessionTrialIndex === 31
      && cell.occurrences[0].permutationPosition
        + cell.occurrences[1].permutationPosition === 15
    ))
    && Object.keys(session.contrasts).length === 15
  )));
  assert.equal(summary.pooled.cellTable.length, 16);
  assert.ok(summary.pooled.cellTable.every((cell) => (
    cell.nSessions === 2 && cell.nOccurrences === 4
  )));
  assert.equal(Object.keys(summary.pooled.contrasts).length, 15);

  for (const [contrastIndex, contrastId] of summary.contrastOrder.entries()) {
    for (const [metricKey, metricScale] of Object.entries(METRIC_SCALES)) {
      const session0 = summary.sessions[0].contrasts[contrastId].metrics[metricKey];
      const session1 = summary.sessions[1].contrasts[contrastId].metrics[metricKey];
      const pooled = summary.pooled.contrasts[contrastId].metrics[metricKey];
      const effect = (contrastIndex + 1) * metricScale;
      closeTo(session0.deltaMs, effect);
      closeTo(session1.deltaMs, effect * 2);
      closeTo(pooled.deltaMs, effect * 1.5);
      closeTo(session0.deltaPercent, effect * 10);
      closeTo(session1.deltaPercent, effect * 20);
      closeTo(pooled.deltaPercent, effect * 15);

      const concordance = summary.sessionSignConcordance[contrastId].metrics[metricKey];
      assert.deepEqual(concordance.deltaMs.sessionSigns, [1, 1]);
      assert.equal(concordance.deltaMs.concordant, true);
      assert.equal(concordance.deltaMs.sameNonzeroSign, true);
      assert.deepEqual(concordance.deltaPercent.sessionSigns, [1, 1]);
    }
  }

  const firstPlanTrial = buildFirstInstanceLiveOrderFactorialPlan({ runId: RUN_ID })[0];
  closeTo(
    summary.sessions[0].cellTable[0].metrics.gpuPassTotal.deltaMs,
    sessionCellResponse(firstPlanTrial, 1),
  );
  assert.equal(Object.hasOwn(summary, 'preregisteredDecision'), false);
  assert.equal(Object.hasOwn(summary, 'pass'), false);
});

test('factorial analyzer fails closed on count, order, plan, summary, and finite-value errors', async (t) => {
  const pristine = createRecords();
  const cases = [
    ['missing record', (records) => records.pop(), /expected exactly 64/],
    [
      'wrong plan order',
      (records) => { [records[0], records[1]] = [records[1], records[0]]; },
      /differs from generated plan index/,
    ],
    [
      'tampered factor',
      (records) => { records[0].factorLevels.C = 1; },
      /factorLevels differs from generated plan/,
    ],
    [
      'wrong trial summary identity',
      (records) => { records[0].trialSummary.trialId = 'other'; },
      /trialSummary\.trialId differs/,
    ],
    [
      'wrong trial row count',
      (records) => { records[0].trialSummary.nRows = 479; },
      /exactly 480 rows and 60 blocks/,
    ],
    [
      'non-finite total delta',
      (records) => { records[0].trialSummary.estimates.gpuPassTotal.deltaMs = Infinity; },
      /gpuPassTotal\.deltaMs must be finite/,
    ],
    [
      'non-finite compute percentage',
      (records) => { records[0].trialSummary.estimates.gpuCompute.deltaPercent = NaN; },
      /gpuCompute\.deltaPercent must be finite/,
    ],
  ];

  for (const [name, mutate, pattern] of cases) {
    await t.test(name, () => {
      const records = structuredClone(pristine);
      mutate(records);
      assert.throws(
        () => summarizeFirstInstanceLiveOrderFactorial(records, RUN_ID),
        pattern,
      );
    });
  }
  assert.throws(
    () => summarizeFirstInstanceLiveOrderFactorial(pristine, ''),
    /expectedRunId must be a nonempty string/,
  );
});

test('response overview describes all paired cell responses without a gate', () => {
  const summary = summarizeFirstInstanceLiveOrderFactorial(
    createCellIndexResponseRecords(),
    RUN_ID,
  );
  assert.equal(summary.responseOverview.sessions.length, 2);
  const overviews = [
    ...summary.responseOverview.sessions,
    summary.responseOverview.pooled,
  ];
  for (const overview of overviews) {
    assert.equal(overview.nCellResponses, 16);
    for (const metric of Object.values(overview.metrics)) {
      for (const stats of Object.values(metric)) {
        closeTo(stats.grandMean, -0.5);
        assert.equal(stats.min, -8);
        assert.equal(stats.max, 7);
        assert.equal(stats.negativeCount, 8);
        assert.equal(stats.zeroCount, 1);
        assert.equal(stats.positiveCount, 7);
        assert.equal(
          stats.negativeCount + stats.zeroCount + stats.positiveCount,
          overview.nCellResponses,
        );
        assert.equal(Object.hasOwn(stats, 'pass'), false);
        assert.equal(Object.hasOwn(stats, 'significance'), false);
      }
    }
  }
});
