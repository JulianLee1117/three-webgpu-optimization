import {
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_KIND,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_QUARTETS_PER_MATRIX,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIAL_COUNT,
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS,
  validateFirstInstanceStandaloneDeploymentPlan,
} from '../src/benchmark/first-instance-standalone-deployment-plan.js';

const [HIGH_VISIBILITY, LOW_VISIBILITY] =
  FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS;
const PORTABLE = 'portable';
const FEATURE = 'feature';
const TOTAL_EQUALITY_TOLERANCE_MS = 1e-9;
const EARLY_LATE_BLOCK_COUNT = 15;

export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_ANALYSIS_MODES = Object.freeze([
  'candidate',
  'smoke',
  'partial',
]);

export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_GPU_TIMING_FIELDS = Object.freeze([
  'gpuPassTotalMs',
  'gpuRenderMs',
  'gpuComputeMs',
]);

export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_CPU_TIMING_FIELDS = Object.freeze([
  'cpuCommonUpdateMs',
  'cpuComputeSubmitMs',
  'cpuRenderSubmitMs',
  'cpuSubmitTotalMs',
  'cpuFrameBodyMs',
]);

export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS = Object.freeze([
  ...FIRST_INSTANCE_STANDALONE_DEPLOYMENT_GPU_TIMING_FIELDS,
  ...FIRST_INSTANCE_STANDALONE_DEPLOYMENT_CPU_TIMING_FIELDS,
]);

export const FIRST_INSTANCE_STANDALONE_DEPLOYMENT_ROW_IDENTITY_FIELDS = Object.freeze([
  'schemaVersion',
  'kind',
  'trialId',
  'planIndex',
  'matrixIndex',
  'matrixOrdinal',
  'matrixId',
  'matrixTrialIndex',
  'matrixTrialOrdinal',
  'globalQuartetIndex',
  'quartetIndex',
  'quartetOrdinal',
  'quartetId',
  'quartetCode',
  'quartetTrialIndex',
  'globalSessionIndex',
  'matrixSessionIndex',
  'sessionPosition',
  'sessionId',
  'assignedLaneId',
  'absentLaneId',
  'visibilityOrderId',
  'visibilityOrder',
  'visibilityOrderPosition',
  'visibilityExposure',
  'visibilityFraction',
  'warmupFrames',
  'measuredFrames',
  'measuredBlockSize',
  'measuredBlockCount',
]);

const PRIMARY_FIELD = 'gpuPassTotalMs';
const RENDER_FIELD = 'gpuRenderMs';
const SMOKE_PLAN_INDICES = Object.freeze([0, 1, 2, 3]);

function reject(message) {
  throw new Error(`Standalone deployment analysis rejected: ${message}`);
}

function requireRecord(value, label) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    reject(`${label} must be an object.`);
  }
  return value;
}

function requireArray(value, label) {
  if (!Array.isArray(value)) reject(`${label} must be an array.`);
  return value;
}

function finiteNumber(value, label, {
  minimum = -Infinity,
  exclusiveMinimum = false,
} = {}) {
  if (!Number.isFinite(value)
    || (exclusiveMinimum ? value <= minimum : value < minimum)) {
    reject(
      `${label} must be a finite number ${exclusiveMinimum ? '>' : '>='} ${minimum}.`,
    );
  }
  return value;
}

function safeInteger(value, label, { minimum = -Infinity, maximum = Infinity } = {}) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    reject(`${label} must be a safe integer in [${minimum}, ${maximum}].`);
  }
  return value;
}

function finiteDerived(value, label) {
  if (!Number.isFinite(value)) reject(`derived ${label} is non-finite.`);
  return value;
}

function mean(values, label = 'sample') {
  if (!Array.isArray(values) || values.length === 0) {
    reject(`cannot compute ${label} mean from an empty sample.`);
  }
  const result = values.reduce((sum, value) => sum + value, 0) / values.length;
  return finiteDerived(result, `${label} mean`);
}

function median(values, label = 'sample') {
  if (!Array.isArray(values) || values.length === 0) {
    reject(`cannot compute ${label} median from an empty sample.`);
  }
  if (values.some((value) => !Number.isFinite(value))) {
    reject(`cannot compute ${label} median containing a non-finite value.`);
  }
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return finiteDerived(
    ordered.length % 2 === 0
      ? (ordered[middle - 1] + ordered[middle]) / 2
      : ordered[middle],
    `${label} median`,
  );
}

function exactValue(actual, expected, label) {
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || actual.length !== expected.length) {
      reject(`${label} differs from the frozen plan.`);
    }
    expected.forEach((value, index) => exactValue(actual[index], value, `${label}[${index}]`));
    return;
  }
  if (expected !== null && typeof expected === 'object') {
    if (actual === null || typeof actual !== 'object' || Array.isArray(actual)) {
      reject(`${label} differs from the frozen plan.`);
    }
    const expectedKeys = Object.keys(expected);
    const actualKeys = Object.keys(actual);
    if (actualKeys.length !== expectedKeys.length
      || expectedKeys.some((key, index) => key !== actualKeys[index])) {
      reject(`${label} keys differ from the frozen plan.`);
    }
    expectedKeys.forEach((key) => exactValue(actual[key], expected[key], `${label}.${key}`));
    return;
  }
  if (!Object.is(actual, expected)) {
    reject(
      `${label} is ${JSON.stringify(actual)}; expected ${JSON.stringify(expected)}.`,
    );
  }
}

function percentage(deltaMs, referenceMs, label, { allowZero = false } = {}) {
  if (referenceMs === 0) {
    if (allowZero) return null;
    reject(`${label} has a zero reference, so its percentage is undefined.`);
  }
  return finiteDerived((deltaMs / referenceMs) * 100, `${label} percentage`);
}

function validatePlan(plan) {
  requireRecord(plan, 'plan');
  if (typeof plan.runId !== 'string' || plan.runId.length === 0) {
    reject('plan.runId must be a nonempty string.');
  }
  try {
    validateFirstInstanceStandaloneDeploymentPlan(plan, { runId: plan.runId });
  } catch (error) {
    reject(`plan is not the exact frozen plan: ${error.message}`);
  }
  const trialKeys = Object.keys(plan.trials[0]);
  if (trialKeys.length !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_ROW_IDENTITY_FIELDS.length
    || trialKeys.some(
      (key, index) => key !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_ROW_IDENTITY_FIELDS[index],
    )) {
    reject('the analyzer row-identity contract is out of sync with the frozen plan.');
  }
}

function validateOptions(options) {
  if (options === undefined) return { mode: 'candidate' };
  requireRecord(options, 'options');
  const keys = Object.keys(options);
  if (keys.some((key) => key !== 'mode')) {
    reject('options accepts only the mode field.');
  }
  const mode = options.mode ?? 'candidate';
  if (!FIRST_INSTANCE_STANDALONE_DEPLOYMENT_ANALYSIS_MODES.includes(mode)) {
    reject(`options.mode must be one of ${FIRST_INSTANCE_STANDALONE_DEPLOYMENT_ANALYSIS_MODES.join(', ')}.`);
  }
  return { mode };
}

function validateTrialRecordIdentity(record, expectedTrial, label) {
  requireRecord(record, label);
  for (const field of FIRST_INSTANCE_STANDALONE_DEPLOYMENT_ROW_IDENTITY_FIELDS) {
    if (!Object.hasOwn(record, field)) reject(`${label}.${field} is required.`);
    exactValue(record[field], expectedTrial[field], `${label}.${field}`);
  }
  if (!Object.hasOwn(record, 'rows')) reject(`${label}.rows is required.`);
}

function validateRowIdentity(row, expectedTrial, frameIndex, runId, label) {
  requireRecord(row, label);
  if (!Object.hasOwn(row, 'runId')) reject(`${label}.runId is required.`);
  exactValue(row.runId, runId, `${label}.runId`);
  for (const field of FIRST_INSTANCE_STANDALONE_DEPLOYMENT_ROW_IDENTITY_FIELDS) {
    if (!Object.hasOwn(row, field)) reject(`${label}.${field} is required.`);
    exactValue(row[field], expectedTrial[field], `${label}.${field}`);
  }
  if (!Object.hasOwn(row, 'frameIndex')) reject(`${label}.frameIndex is required.`);
  if (!Object.hasOwn(row, 'phaseFrameIndex')) {
    reject(`${label}.phaseFrameIndex is required.`);
  }
  exactValue(row.frameIndex, frameIndex, `${label}.frameIndex`);
  exactValue(row.phaseFrameIndex, frameIndex, `${label}.phaseFrameIndex`);
}

function normalizeTimingRow(row, expectedTrial, frameIndex, runId, label) {
  validateRowIdentity(row, expectedTrial, frameIndex, runId, label);
  const timing = {};
  for (const field of FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS) {
    if (!Object.hasOwn(row, field)) reject(`${label}.${field} is required.`);
    timing[field] = finiteNumber(row[field], `${label}.${field}`, {
      minimum: 0,
      exclusiveMinimum: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_GPU_TIMING_FIELDS
        .includes(field),
    });
  }

  const reconstructedGpuTotal = timing.gpuComputeMs + timing.gpuRenderMs;
  if (!Number.isFinite(reconstructedGpuTotal)
    || Math.abs(timing.gpuPassTotalMs - reconstructedGpuTotal)
      > TOTAL_EQUALITY_TOLERANCE_MS) {
    reject(
      `${label}.gpuPassTotalMs must equal gpuComputeMs + gpuRenderMs within `
        + `${TOTAL_EQUALITY_TOLERANCE_MS} ms.`,
    );
  }
  const reconstructedCpuSubmit = timing.cpuComputeSubmitMs + timing.cpuRenderSubmitMs;
  if (!Number.isFinite(reconstructedCpuSubmit)
    || Math.abs(timing.cpuSubmitTotalMs - reconstructedCpuSubmit)
      > TOTAL_EQUALITY_TOLERANCE_MS) {
    reject(
      `${label}.cpuSubmitTotalMs must equal cpuComputeSubmitMs + cpuRenderSubmitMs within `
        + `${TOTAL_EQUALITY_TOLERANCE_MS} ms.`,
    );
  }
  return timing;
}

function metricMeans(rows, label) {
  return Object.fromEntries(FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS.map((field) => [
    field,
    mean(rows.map((row) => row[field]), `${label}.${field}`),
  ]));
}

function trialIdentity(expectedTrial) {
  return Object.fromEntries(
    FIRST_INSTANCE_STANDALONE_DEPLOYMENT_ROW_IDENTITY_FIELDS.map(
      (field) => [field, Array.isArray(expectedTrial[field])
        ? [...expectedTrial[field]]
        : expectedTrial[field]],
    ),
  );
}

function summarizeTrialRecordUnchecked(record, expectedTrial, runId, label) {
  validateTrialRecordIdentity(record, expectedTrial, label);
  const rows = requireArray(record.rows, `${label}.rows`);
  if (rows.length !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES) {
    reject(
      `${label}.rows has ${rows.length} rows; expected exactly `
        + `${FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES}.`,
    );
  }
  const normalizedRows = rows.map((row, frameIndex) => normalizeTimingRow(
    row,
    expectedTrial,
    frameIndex,
    runId,
    `${label}.rows[${frameIndex}]`,
  ));
  const blocks = [];
  for (let blockIndex = 0;
    blockIndex < FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS;
    blockIndex += 1) {
    const startFrameIndex = blockIndex * FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE;
    const blockRows = normalizedRows.slice(
      startFrameIndex,
      startFrameIndex + FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE,
    );
    if (blockRows.length !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE) {
      reject(`${label} block ${blockIndex} is incomplete.`);
    }
    blocks.push({
      blockIndex,
      startFrameIndex,
      endFrameIndex: startFrameIndex + FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE - 1,
      means: metricMeans(blockRows, `${label}.blocks[${blockIndex}]`),
    });
  }
  const estimates = Object.fromEntries(
    FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS.map((field) => [
      field,
      median(blocks.map((block) => block.means[field]), `${label}.${field} block means`),
    ]),
  );
  const firstBlocks = blocks.slice(0, EARLY_LATE_BLOCK_COUNT);
  const finalBlocks = blocks.slice(-EARLY_LATE_BLOCK_COUNT);
  const drift = Object.fromEntries(
    FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS.map((field) => {
      const first15BlockMeanMs = mean(
        firstBlocks.map((block) => block.means[field]),
        `${label}.${field} first 15 blocks`,
      );
      const final15BlockMeanMs = mean(
        finalBlocks.map((block) => block.means[field]),
        `${label}.${field} final 15 blocks`,
      );
      const deltaMs = finiteDerived(
        final15BlockMeanMs - first15BlockMeanMs,
        `${label}.${field} within-trial drift`,
      );
      return [field, {
        first15BlockMeanMs,
        final15BlockMeanMs,
        deltaMs,
        deltaPercent: percentage(
          deltaMs,
          first15BlockMeanMs,
          `${label}.${field} within-trial drift`,
          { allowZero: !FIRST_INSTANCE_STANDALONE_DEPLOYMENT_GPU_TIMING_FIELDS.includes(field) },
        ),
      }];
    }),
  );

  return {
    ...trialIdentity(expectedTrial),
    nRows: rows.length,
    nBlocks: blocks.length,
    blocks,
    estimates,
    drift,
  };
}

function contrast(featureValues, portableValues, field, label) {
  if (featureValues.length === 0 || featureValues.length !== portableValues.length) {
    reject(`${label} must contain equal nonzero feature and portable samples.`);
  }
  const featureMeanMs = mean(featureValues, `${label} feature`);
  const portableMeanMs = mean(portableValues, `${label} portable`);
  const deltaMs = finiteDerived(featureMeanMs - portableMeanMs, `${label} delta`);
  return {
    field,
    nFeature: featureValues.length,
    nPortable: portableValues.length,
    featureMeanMs,
    portableMeanMs,
    deltaMs,
    deltaPercent: percentage(deltaMs, portableMeanMs, label, {
      allowZero: !FIRST_INSTANCE_STANDALONE_DEPLOYMENT_GPU_TIMING_FIELDS.includes(field),
    }),
  };
}

function contrastFromTrials(trials, field, label, expectedPerLane) {
  const feature = trials.filter((trial) => trial.assignedLaneId === FEATURE);
  const portable = trials.filter((trial) => trial.assignedLaneId === PORTABLE);
  if (feature.length !== expectedPerLane || portable.length !== expectedPerLane) {
    reject(
      `${label} has ${feature.length} feature and ${portable.length} portable trials; `
        + `expected ${expectedPerLane} of each.`,
    );
  }
  return contrast(
    feature.map((trial) => trial.estimates[field]),
    portable.map((trial) => trial.estimates[field]),
    field,
    label,
  );
}

function sessionValue(session, visibilityKey, field) {
  const value = session.estimates[visibilityKey]?.[field];
  if (!Number.isFinite(value)) {
    reject(`${session.sessionId} lacks a finite ${visibilityKey}/${field} estimate.`);
  }
  return value;
}

function buildSessions(plan, trials) {
  const trialsBySession = new Map();
  for (const trial of trials) {
    const selected = trialsBySession.get(trial.sessionId) ?? [];
    selected.push(trial);
    trialsBySession.set(trial.sessionId, selected);
  }
  return plan.sessions.map((expectedSession) => {
    const selected = trialsBySession.get(expectedSession.sessionId) ?? [];
    if (selected.length !== 2) {
      reject(`${expectedSession.sessionId} has ${selected.length} trials; expected exactly 2.`);
    }
    if (selected.some((trial, index) => trial.trialId !== expectedSession.trialIds[index])) {
      reject(`${expectedSession.sessionId} trials are not in frozen visibility order.`);
    }
    const estimates = {};
    const withinTrialDrift = {};
    for (const trial of selected) {
      const key = String(trial.visibilityFraction);
      if (Object.hasOwn(estimates, key)) {
        reject(`${expectedSession.sessionId} repeats visibility ${key}.`);
      }
      estimates[key] = { ...trial.estimates };
      withinTrialDrift[key] = Object.fromEntries(
        FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS.map(
          (field) => [field, { ...trial.drift[field] }],
        ),
      );
    }
    for (const visibility of FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS) {
      if (!Object.hasOwn(estimates, String(visibility))) {
        reject(`${expectedSession.sessionId} lacks visibility ${visibility}.`);
      }
    }
    return {
      ...expectedSession,
      trialIds: [...expectedSession.trialIds],
      nTrials: selected.length,
      estimates,
      withinTrialDrift,
    };
  });
}

function sessionSequenceDrift(sessions, visibilityKey, field, label) {
  const early = sessions.filter((session) => session.sessionPosition < 2);
  const late = sessions.filter((session) => session.sessionPosition >= 2);
  if (early.length !== 2 || late.length !== 2) {
    reject(`${label} must contain exactly two early and two late sessions.`);
  }
  const value = (session) => visibilityKey === null
    ? mean(
      FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS.map(
        (visibility) => sessionValue(session, String(visibility), field),
      ),
      `${label} ${session.sessionId} visibility mean`,
    )
    : sessionValue(session, visibilityKey, field);
  const earlyMeanMs = mean(early.map(value), `${label} early sessions`);
  const lateMeanMs = mean(late.map(value), `${label} late sessions`);
  const deltaMs = finiteDerived(lateMeanMs - earlyMeanMs, `${label} delta`);
  return {
    field,
    nEarlySessions: 2,
    nLateSessions: 2,
    earlyMeanMs,
    lateMeanMs,
    deltaMs,
    deltaPercent: percentage(deltaMs, earlyMeanMs, label, {
      allowZero: !FIRST_INSTANCE_STANDALONE_DEPLOYMENT_GPU_TIMING_FIELDS.includes(field),
    }),
  };
}

function aggregateWithinTrialDrift(trials, visibilityKey, field, label) {
  const selected = visibilityKey === null
    ? trials
    : trials.filter((trial) => String(trial.visibilityFraction) === visibilityKey);
  const expected = visibilityKey === null ? 8 : 4;
  if (selected.length !== expected) {
    reject(`${label} has ${selected.length} trials; expected ${expected}.`);
  }
  const firstMeanMs = mean(
    selected.map((trial) => trial.drift[field].first15BlockMeanMs),
    `${label} first-block means`,
  );
  const finalMeanMs = mean(
    selected.map((trial) => trial.drift[field].final15BlockMeanMs),
    `${label} final-block means`,
  );
  const deltaMs = mean(
    selected.map((trial) => trial.drift[field].deltaMs),
    `${label} trial drift deltas`,
  );
  const trialPercentages = selected.map((trial) => trial.drift[field].deltaPercent);
  return {
    field,
    n: selected.length,
    firstMeanMs,
    finalMeanMs,
    deltaMs,
    deltaPercent: trialPercentages.every(Number.isFinite)
      ? mean(trialPercentages, `${label} trial drift percentages`)
      : null,
    aggregation:
      'equal-weight mean of within-trial deltas and, separately, within-trial percentages',
  };
}

function sameLaneDrift(sessions, positions, visibilityKey, field, label) {
  const [firstPosition, secondPosition] = positions;
  const first = sessions.find((session) => session.sessionPosition === firstPosition);
  const second = sessions.find((session) => session.sessionPosition === secondPosition);
  if (!first || !second || first.assignedLaneId !== second.assignedLaneId) {
    reject(`${label} does not compare the same lane.`);
  }
  const value = (session) => visibilityKey === null
    ? mean(
      FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS.map(
        (visibility) => sessionValue(session, String(visibility), field),
      ),
      `${label} ${session.sessionId} visibility mean`,
    )
    : sessionValue(session, visibilityKey, field);
  const firstMeanMs = value(first);
  const finalMeanMs = value(second);
  const deltaMs = finiteDerived(finalMeanMs - firstMeanMs, `${label} delta`);
  return {
    field,
    laneId: first.assignedLaneId,
    firstSessionPosition: firstPosition,
    finalSessionPosition: secondPosition,
    firstMeanMs,
    finalMeanMs,
    deltaMs,
    deltaPercent: percentage(deltaMs, firstMeanMs, label, {
      allowZero: !FIRST_INSTANCE_STANDALONE_DEPLOYMENT_GPU_TIMING_FIELDS.includes(field),
    }),
  };
}

function buildQuartets(plan, sessions, trials) {
  const sessionsByQuartet = new Map();
  const trialsByQuartet = new Map();
  for (const session of sessions) {
    const selected = sessionsByQuartet.get(session.quartetId) ?? [];
    selected.push(session);
    sessionsByQuartet.set(session.quartetId, selected);
  }
  for (const trial of trials) {
    const selected = trialsByQuartet.get(trial.quartetId) ?? [];
    selected.push(trial);
    trialsByQuartet.set(trial.quartetId, selected);
  }

  return plan.quartets.map((expectedQuartet) => {
    const selectedSessions = sessionsByQuartet.get(expectedQuartet.quartetId) ?? [];
    const selectedTrials = trialsByQuartet.get(expectedQuartet.quartetId) ?? [];
    if (selectedSessions.length !== 4 || selectedTrials.length !== 8) {
      reject(
        `${expectedQuartet.quartetId} has ${selectedSessions.length} sessions and `
          + `${selectedTrials.length} trials; expected 4 and 8.`,
      );
    }
    if (selectedSessions.some(
      (session, index) => session.sessionId !== expectedQuartet.sessionIds[index],
    ) || selectedTrials.some(
      (trial, index) => trial.trialId !== expectedQuartet.trialIds[index],
    )) {
      reject(`${expectedQuartet.quartetId} is not in frozen session/trial order.`);
    }

    const effects = {};
    const highVisibilityStrata = {
      visibilityExposure: { first: {}, second: {} },
      adjacentPairPosition: { early: {}, late: {} },
    };
    for (const visibility of FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS) {
      const visibilityKey = String(visibility);
      const visibilityTrials = selectedTrials.filter(
        (trial) => trial.visibilityFraction === visibility,
      );
      if (visibilityTrials.length !== 4) {
        reject(`${expectedQuartet.quartetId}/${visibilityKey} must contain four trials.`);
      }
      effects[visibilityKey] = Object.fromEntries(
        FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS.map((field) => [
          field,
          contrastFromTrials(
            visibilityTrials,
            field,
            `${expectedQuartet.quartetId}/${visibilityKey}/${field}`,
            2,
          ),
        ]),
      );
      if (visibility === HIGH_VISIBILITY) {
        for (const exposure of ['first', 'second']) {
          const exposedTrials = visibilityTrials.filter(
            (trial) => trial.visibilityExposure === exposure,
          );
          highVisibilityStrata.visibilityExposure[exposure] = Object.fromEntries(
            FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS.map((field) => [
              field,
              contrastFromTrials(
                exposedTrials,
                field,
                `${expectedQuartet.quartetId}/exposure=${exposure}/${field}`,
                1,
              ),
            ]),
          );
        }
        for (const [position, positions] of [
          ['early', [0, 1]],
          ['late', [2, 3]],
        ]) {
          const positionedTrials = visibilityTrials.filter(
            (trial) => positions.includes(trial.sessionPosition),
          );
          highVisibilityStrata.adjacentPairPosition[position] = Object.fromEntries(
            FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS.map((field) => [
              field,
              contrastFromTrials(
                positionedTrials,
                field,
                `${expectedQuartet.quartetId}/pair=${position}/${field}`,
                1,
              ),
            ]),
          );
        }
      }
    }

    const sessionSequence = { overall: {}, visibilities: {} };
    const withinTrial = { overall: {}, visibilities: {} };
    const sameLane = { overall: {}, visibilities: {} };
    for (const field of FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS) {
      sessionSequence.overall[field] = sessionSequenceDrift(
        selectedSessions,
        null,
        field,
        `${expectedQuartet.quartetId}/session-sequence/overall/${field}`,
      );
      withinTrial.overall[field] = aggregateWithinTrialDrift(
        selectedTrials,
        null,
        field,
        `${expectedQuartet.quartetId}/within-trial/overall/${field}`,
      );
      sameLane.overall[field] = {
        outer: sameLaneDrift(
          selectedSessions,
          [0, 3],
          null,
          field,
          `${expectedQuartet.quartetId}/same-lane/outer/overall/${field}`,
        ),
        inner: sameLaneDrift(
          selectedSessions,
          [1, 2],
          null,
          field,
          `${expectedQuartet.quartetId}/same-lane/inner/overall/${field}`,
        ),
      };
    }
    for (const visibility of FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS) {
      const visibilityKey = String(visibility);
      sessionSequence.visibilities[visibilityKey] = {};
      withinTrial.visibilities[visibilityKey] = {};
      sameLane.visibilities[visibilityKey] = {};
      for (const field of FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS) {
        sessionSequence.visibilities[visibilityKey][field] = sessionSequenceDrift(
          selectedSessions,
          visibilityKey,
          field,
          `${expectedQuartet.quartetId}/session-sequence/${visibilityKey}/${field}`,
        );
        withinTrial.visibilities[visibilityKey][field] = aggregateWithinTrialDrift(
          selectedTrials,
          visibilityKey,
          field,
          `${expectedQuartet.quartetId}/within-trial/${visibilityKey}/${field}`,
        );
        sameLane.visibilities[visibilityKey][field] = {
          outer: sameLaneDrift(
            selectedSessions,
            [0, 3],
            visibilityKey,
            field,
            `${expectedQuartet.quartetId}/same-lane/outer/${visibilityKey}/${field}`,
          ),
          inner: sameLaneDrift(
            selectedSessions,
            [1, 2],
            visibilityKey,
            field,
            `${expectedQuartet.quartetId}/same-lane/inner/${visibilityKey}/${field}`,
          ),
        };
      }
    }

    return {
      ...expectedQuartet,
      laneOrder: [...expectedQuartet.laneOrder],
      visibilityOrderMask: [...expectedQuartet.visibilityOrderMask],
      sessionIds: [...expectedQuartet.sessionIds],
      trialIds: [...expectedQuartet.trialIds],
      effects,
      highVisibilityStrata,
      drift: {
        conditionBlindSessionSequence: sessionSequence,
        conditionBlindWithinTrial: withinTrial,
        sameLaneSession: sameLane,
      },
    };
  });
}

function summarizeEffectSet(effects, label) {
  if (effects.length === 0) reject(`${label} has no quartet effects.`);
  const percentages = effects.map((effect) => effect.deltaPercent);
  return {
    nQuartets: effects.length,
    median: {
      deltaMs: median(effects.map((effect) => effect.deltaMs), `${label} delta`),
      deltaPercent: percentages.every(Number.isFinite)
        ? median(percentages, `${label} percentage`)
        : null,
    },
    negativeCount: effects.filter((effect) => effect.deltaMs < 0).length,
    negativePercentCount: effects.filter(
      (effect) => Number.isFinite(effect.deltaPercent) && effect.deltaPercent < 0,
    ).length,
  };
}

function summarizeDriftSet(drifts, label) {
  if (drifts.length === 0) reject(`${label} has no quartet drifts.`);
  const percentages = drifts.map((drift) => drift.deltaPercent);
  return {
    nQuartets: drifts.length,
    median: {
      deltaMs: median(drifts.map((drift) => drift.deltaMs), `${label} delta`),
      deltaPercent: percentages.every(Number.isFinite)
        ? median(percentages, `${label} percentage`)
        : null,
    },
  };
}

function binaryFactorSummary({
  quartets,
  factor,
  levels,
  effectFor,
}) {
  const strata = levels.map((level) => {
    const effects = effectFor(level);
    return {
      level,
      ...summarizeEffectSet(effects, `${factor}=${level}`),
    };
  });
  if (strata.some((stratum) => stratum.nQuartets === 0)) {
    reject(`${factor} has an empty level.`);
  }
  const interaction = {
    deltaMs: finiteDerived(
      strata[1].median.deltaMs - strata[0].median.deltaMs,
      `${factor} interaction delta`,
    ),
    deltaPercentagePoints: finiteDerived(
      strata[1].median.deltaPercent - strata[0].median.deltaPercent,
      `${factor} interaction percentage points`,
    ),
  };
  return { factor, levels: [...levels], strata, interaction, nMatrixQuartets: quartets.length };
}

function matrixFactors(quartets) {
  const highKey = String(HIGH_VISIBILITY);
  return {
    sequence: binaryFactorSummary({
      quartets,
      factor: 'quartetLaneOrder',
      levels: ['A', 'B'],
      effectFor: (level) => quartets
        .filter((quartet) => quartet.laneOrderId === level)
        .map((quartet) => quartet.effects[highKey][PRIMARY_FIELD]),
    }),
    visibilityExposure: binaryFactorSummary({
      quartets,
      factor: 'visibilityExposure',
      levels: ['first', 'second'],
      effectFor: (level) => quartets.map(
        (quartet) => quartet.highVisibilityStrata.visibilityExposure[level][PRIMARY_FIELD],
      ),
    }),
    adjacentPairPosition: binaryFactorSummary({
      quartets,
      factor: 'adjacentPairPosition',
      levels: ['early', 'late'],
      effectFor: (level) => quartets.map(
        (quartet) => quartet.highVisibilityStrata.adjacentPairPosition[level][PRIMARY_FIELD],
      ),
    }),
    matrixHalf: binaryFactorSummary({
      quartets,
      factor: 'matrixHalf',
      levels: ['first', 'last'],
      effectFor: (level) => quartets
        .filter((quartet) => (
          level === 'first' ? quartet.quartetIndex < 6 : quartet.quartetIndex >= 6
        ))
        .map((quartet) => quartet.effects[highKey][PRIMARY_FIELD]),
    }),
  };
}

function summarizeMatrixDrift(quartets, driftName) {
  const result = { overall: {}, visibilities: {} };
  for (const field of FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS) {
    result.overall[field] = summarizeDriftSet(
      quartets.map((quartet) => quartet.drift[driftName].overall[field]),
      `${driftName}/overall/${field}`,
    );
  }
  for (const visibility of FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS) {
    const visibilityKey = String(visibility);
    result.visibilities[visibilityKey] = {};
    for (const field of FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS) {
      result.visibilities[visibilityKey][field] = summarizeDriftSet(
        quartets.map(
          (quartet) => quartet.drift[driftName].visibilities[visibilityKey][field],
        ),
        `${driftName}/${visibilityKey}/${field}`,
      );
    }
  }
  return result;
}

function summarizeSameLaneDiagnostic(quartets) {
  const result = { overall: {}, visibilities: {} };
  for (const field of FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS) {
    result.overall[field] = Object.fromEntries(['outer', 'inner'].map((position) => [
      position,
      summarizeDriftSet(
        quartets.map(
          (quartet) => quartet.drift.sameLaneSession.overall[field][position],
        ),
        `same-lane/${position}/overall/${field}`,
      ),
    ]));
  }
  for (const visibility of FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS) {
    const visibilityKey = String(visibility);
    result.visibilities[visibilityKey] = {};
    for (const field of FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS) {
      result.visibilities[visibilityKey][field] = Object.fromEntries(
        ['outer', 'inner'].map((position) => [
          position,
          summarizeDriftSet(
            quartets.map(
              (quartet) => quartet.drift.sameLaneSession
                .visibilities[visibilityKey][field][position],
            ),
            `same-lane/${position}/${visibilityKey}/${field}`,
          ),
        ]),
      );
    }
  }
  return result;
}

function buildMatrices(plan, quartets) {
  return plan.matrices.map((expectedMatrix) => {
    const selected = quartets.filter(
      (quartet) => quartet.matrixIndex === expectedMatrix.matrixIndex,
    );
    if (selected.length !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_QUARTETS_PER_MATRIX
      || selected.some(
        (quartet, index) => quartet.quartetId !== expectedMatrix.quartetIds[index],
      )) {
      reject(`${expectedMatrix.matrixId} does not contain its exact 12 ordered quartets.`);
    }
    const estimates = Object.fromEntries(
      FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS.map((visibility) => {
        const visibilityKey = String(visibility);
        return [visibilityKey, Object.fromEntries(
          FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TIMING_FIELDS.map((field) => [
            field,
            summarizeEffectSet(
              selected.map((quartet) => quartet.effects[visibilityKey][field]),
              `${expectedMatrix.matrixId}/${visibilityKey}/${field}`,
            ),
          ]),
        )];
      }),
    );
    const pairedDoseQuartets = selected.map((quartet) => {
      const high = quartet.effects[String(HIGH_VISIBILITY)][PRIMARY_FIELD];
      const low = quartet.effects[String(LOW_VISIBILITY)][PRIMARY_FIELD];
      return {
        quartetId: quartet.quartetId,
        highDeltaMs: high.deltaMs,
        lowDeltaMs: low.deltaMs,
        deltaMs: finiteDerived(
          high.deltaMs - low.deltaMs,
          `${quartet.quartetId} high-minus-low dose`,
        ),
        deltaPercentagePoints: finiteDerived(
          high.deltaPercent - low.deltaPercent,
          `${quartet.quartetId} high-minus-low dose percentage points`,
        ),
      };
    });
    const pairedHighMinusLow = {
      nQuartets: pairedDoseQuartets.length,
      quartets: pairedDoseQuartets,
      median: {
        deltaMs: median(
          pairedDoseQuartets.map((record) => record.deltaMs),
          `${expectedMatrix.matrixId} paired dose`,
        ),
        deltaPercentagePoints: median(
          pairedDoseQuartets.map((record) => record.deltaPercentagePoints),
          `${expectedMatrix.matrixId} paired dose percentage points`,
        ),
      },
    };
    return {
      ...expectedMatrix,
      quartetCodes: [...expectedMatrix.quartetCodes],
      quartetIds: [...expectedMatrix.quartetIds],
      sessionIds: [...expectedMatrix.sessionIds],
      trialIds: [...expectedMatrix.trialIds],
      nQuartets: selected.length,
      estimates,
      pairedHighMinusLow,
      highVisibilityFactors: matrixFactors(selected),
      drift: {
        conditionBlindSessionSequence: summarizeMatrixDrift(
          selected,
          'conditionBlindSessionSequence',
        ),
        conditionBlindWithinTrial: summarizeMatrixDrift(
          selected,
          'conditionBlindWithinTrial',
        ),
        sameLaneSession: summarizeSameLaneDiagnostic(selected),
      },
    };
  });
}

function factorGate(factor, thresholds) {
  const levelsNegative = factor.strata.every(
    (stratum) => stratum.median.deltaMs < 0,
  );
  const interactionWithinBounds = Math.abs(factor.interaction.deltaMs)
      < thresholds.strictMaximumAbsoluteDeltaMs
    && Math.abs(factor.interaction.deltaPercentagePoints)
      < thresholds.strictMaximumAbsoluteDeltaPercentagePoints;
  return {
    pass: levelsNegative && interactionWithinBounds,
    levelsNegative,
    interactionWithinBounds,
    bounds: {
      strictAbsoluteDeltaMs: thresholds.strictMaximumAbsoluteDeltaMs,
      strictAbsoluteDeltaPercentagePoints:
        thresholds.strictMaximumAbsoluteDeltaPercentagePoints,
    },
    observed: factor,
  };
}

function driftGate(summary, thresholds) {
  const percentFinite = Number.isFinite(summary.median.deltaPercent);
  return {
    pass: percentFinite
      && Math.abs(summary.median.deltaMs) < thresholds.strictMaximumAbsoluteDeltaMs
      && Math.abs(summary.median.deltaPercent)
        < thresholds.strictMaximumAbsoluteDeltaPercent,
    bounds: {
      strictAbsoluteDeltaMs: thresholds.strictMaximumAbsoluteDeltaMs,
      strictAbsoluteDeltaPercent: thresholds.strictMaximumAbsoluteDeltaPercent,
    },
    observed: summary.median,
    nQuartets: summary.nQuartets,
  };
}

function driftFamilyGate(drift, thresholds) {
  const overall = driftGate(drift.overall[PRIMARY_FIELD], thresholds);
  const visibilities = Object.fromEntries(
    FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS.map((visibility) => {
      const key = String(visibility);
      return [key, driftGate(drift.visibilities[key][PRIMARY_FIELD], thresholds)];
    }),
  );
  return {
    pass: overall.pass && Object.values(visibilities).every((gate) => gate.pass),
    overall,
    visibilities,
  };
}

function evaluateMatrix(matrix, matrixQuartets, thresholds) {
  const high = matrix.estimates[String(HIGH_VISIBILITY)];
  const low = matrix.estimates[String(LOW_VISIBILITY)];
  const highTotalThreshold = thresholds.highVisibilityGpuPassTotal;
  const highRenderThreshold = thresholds.highVisibilityGpuRender;
  const lowThreshold = thresholds.lowVisibilityGpuPassTotal;
  const pairedThreshold = thresholds.pairedHighMinusLowGpuPassTotal;

  const highVisibilityGpuPassTotal = {
    pass: high[PRIMARY_FIELD].median.deltaMs <= highTotalThreshold.maximumDeltaMs
      && high[PRIMARY_FIELD].median.deltaPercent <= highTotalThreshold.maximumDeltaPercent
      && high[PRIMARY_FIELD].negativeCount >= highTotalThreshold.minimumNegativeQuartets,
    thresholds: { ...highTotalThreshold },
    observed: high[PRIMARY_FIELD],
  };
  const highVisibilityGpuRender = {
    pass: high[RENDER_FIELD].median.deltaMs <= highRenderThreshold.maximumDeltaMs
      && high[RENDER_FIELD].median.deltaPercent <= highRenderThreshold.maximumDeltaPercent
      && high[RENDER_FIELD].negativeCount >= highRenderThreshold.minimumNegativeQuartets,
    thresholds: { ...highRenderThreshold },
    observed: high[RENDER_FIELD],
  };
  if (matrixQuartets.length !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_QUARTETS_PER_MATRIX) {
    reject(`${matrix.matrixId} decision lacks its exact quartet set.`);
  }
  const lowEffects = matrixQuartets.map((quartet) => ({
    quartetId: quartet.quartetId,
    ...quartet.effects[String(LOW_VISIBILITY)][PRIMARY_FIELD],
  }));
  const belowDeltaCount = lowEffects.filter(
    (effect) => effect.deltaMs < lowThreshold.strictUpperDeltaMs,
  ).length;
  const belowPercentCount = lowEffects.filter(
    (effect) => effect.deltaPercent < lowThreshold.strictUpperDeltaPercent,
  ).length;
  const lowVisibilityGpuPassTotal = {
    pass: low[PRIMARY_FIELD].median.deltaMs < lowThreshold.strictUpperDeltaMs
      && low[PRIMARY_FIELD].median.deltaPercent < lowThreshold.strictUpperDeltaPercent
      && belowDeltaCount >= lowThreshold.minimumQuartetsBelowEachUpperBound
      && belowPercentCount >= lowThreshold.minimumQuartetsBelowEachUpperBound,
    thresholds: { ...lowThreshold },
    observed: low[PRIMARY_FIELD],
    belowDeltaCount,
    belowPercentCount,
  };
  const pairedHighMinusLowGpuPassTotal = {
    pass: matrix.pairedHighMinusLow.median.deltaMs <= pairedThreshold.maximumDeltaMs,
    thresholds: { ...pairedThreshold },
    observed: matrix.pairedHighMinusLow,
  };
  const factorGates = Object.fromEntries(
    Object.entries(matrix.highVisibilityFactors).map(([name, factor]) => [
      name,
      factorGate(factor, thresholds.nuisanceInteraction),
    ]),
  );
  const highVisibilityNuisanceFactors = {
    pass: Object.values(factorGates).every((gate) => gate.pass),
    factors: factorGates,
  };
  const conditionBlindSessionSequenceDrift = driftFamilyGate(
    matrix.drift.conditionBlindSessionSequence,
    thresholds.conditionBlindDrift,
  );
  const conditionBlindWithinTrialDrift = driftFamilyGate(
    matrix.drift.conditionBlindWithinTrial,
    thresholds.conditionBlindDrift,
  );
  const gates = {
    highVisibilityGpuPassTotal,
    highVisibilityGpuRender,
    lowVisibilityGpuPassTotal,
    pairedHighMinusLowGpuPassTotal,
    highVisibilityNuisanceFactors,
    conditionBlindSessionSequenceDrift,
    conditionBlindWithinTrialDrift,
  };
  const failedGates = Object.entries(gates)
    .filter(([, gate]) => gate.pass !== true)
    .map(([name]) => name);
  return {
    matrixId: matrix.matrixId,
    matrixIndex: matrix.matrixIndex,
    pass: failedGates.length === 0,
    failedGates,
    gates,
  };
}

function decisionForMatrices(matrices, quartets, thresholds) {
  const matrixDecisions = matrices.map((matrix) => evaluateMatrix(
    matrix,
    quartets.filter((quartet) => quartet.matrixIndex === matrix.matrixIndex),
    thresholds,
  ));
  const pass = matrixDecisions.every((decision) => decision.pass);
  return {
    status: 'evaluated',
    eligible: true,
    scope: 'preregistered numerical gates only; evidence-integrity gates are external',
    pass,
    numericalVerdict: pass
      ? 'preregistered-numerical-gates-met'
      : 'preregistered-numerical-gates-not-met',
    standaloneProtocolVerdict: pass ? null : 'standalone-confirmation-not-met',
    standaloneProtocolVerdictReason: pass
      ? 'standalone-confirmed additionally requires verifier-owned evidence gates'
      : 'at least one matrix failed a preregistered numerical gate',
    matrixDecisions,
  };
}

function excludedDecision(mode) {
  return {
    status: 'excluded',
    eligible: false,
    pass: null,
    numericalVerdict: null,
    standaloneProtocolVerdict: null,
    matrixDecisions: [],
    reason: `${mode} analysis is never eligible for a preregistered decision`,
  };
}

function coverageFor(plan, records, mode) {
  const present = new Set(records.map((record) => record.planIndex));
  const missingPlanIndices = plan.trials
    .filter((trial) => !present.has(trial.planIndex))
    .map((trial) => trial.planIndex);
  return {
    mode,
    complete: missingPlanIndices.length === 0,
    presentTrialCount: records.length,
    expectedTrialCount: plan.trialCount,
    missingPlanIndices,
  };
}

/**
 * Validate and reduce one canonical 480-row standalone trial. This diagnostic
 * entry point cannot produce a candidate decision.
 */
export function summarizeFirstInstanceStandaloneDeploymentTrialRecord(plan, record) {
  validatePlan(plan);
  requireRecord(record, 'trialRecord');
  const planIndex = safeInteger(record.planIndex, 'trialRecord.planIndex', {
    minimum: 0,
    maximum: plan.trials.length - 1,
  });
  return summarizeTrialRecordUnchecked(
    record,
    plan.trials[planIndex],
    plan.runId,
    'trialRecord',
  );
}

/**
 * Analyze canonical standalone deployment trial artifacts. Candidate mode
 * requires all 192 trials in exact plan order. Smoke mode requires the exact
 * frozen four-trial prefix. Partial mode accepts any nonempty ordered subset.
 * Smoke and partial inputs are unconditionally excluded from every decision.
 */
export function summarizeFirstInstanceStandaloneDeployment(
  plan,
  trialRecords,
  options,
) {
  validatePlan(plan);
  const { mode } = validateOptions(options);
  const records = requireArray(trialRecords, 'trialRecords');
  if (records.length === 0) reject('trialRecords must not be empty.');
  if (mode === 'candidate' && records.length !== FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIAL_COUNT) {
    reject(
      `candidate input has ${records.length} trial records; expected exactly `
        + `${FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIAL_COUNT}.`,
    );
  }
  if (records.length > FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIAL_COUNT) {
    reject(
      `input has ${records.length} trial records; the frozen plan contains only `
        + `${FIRST_INSTANCE_STANDALONE_DEPLOYMENT_TRIAL_COUNT}.`,
    );
  }
  if (mode === 'smoke' && (
    records.length !== SMOKE_PLAN_INDICES.length
    || records.some((record, index) => record?.planIndex !== SMOKE_PLAN_INDICES[index])
  )) {
    reject(
      `smoke input must contain exactly frozen plan indices ${SMOKE_PLAN_INDICES.join(', ')}.`,
    );
  }

  const trials = [];
  let previousPlanIndex = -1;
  for (let recordIndex = 0; recordIndex < records.length; recordIndex += 1) {
    const record = requireRecord(records[recordIndex], `trialRecords[${recordIndex}]`);
    const planIndex = safeInteger(record.planIndex, `trialRecords[${recordIndex}].planIndex`, {
      minimum: 0,
      maximum: plan.trials.length - 1,
    });
    if (mode === 'candidate' && planIndex !== recordIndex) {
      reject(
        `trialRecords[${recordIndex}].planIndex is ${planIndex}; candidate input must use `
          + 'the complete exact frozen order.',
      );
    }
    if (planIndex <= previousPlanIndex) {
      reject('trialRecords must be strictly ordered by unique frozen planIndex.');
    }
    previousPlanIndex = planIndex;
    trials.push(summarizeTrialRecordUnchecked(
      record,
      plan.trials[planIndex],
      plan.runId,
      `trialRecords[${recordIndex}]`,
    ));
  }

  const coverage = coverageFor(plan, records, mode);
  const decisionEligibility = {
    eligible: mode === 'candidate' && coverage.complete,
    reason: mode === 'candidate'
      ? 'complete candidate input'
      : `${mode} mode is explicitly excluded from preregistered decisions`,
  };
  let sessions = [];
  let quartets = [];
  let matrices = [];
  let preregisteredNumericalDecision = excludedDecision(mode);
  if (decisionEligibility.eligible) {
    sessions = buildSessions(plan, trials);
    quartets = buildQuartets(plan, sessions, trials);
    matrices = buildMatrices(plan, quartets);
    preregisteredNumericalDecision = decisionForMatrices(
      matrices,
      quartets,
      plan.thresholds,
    );
  }

  return {
    schemaVersion: 1,
    kind: 'first-instance-standalone-deployment-summary',
    runId: plan.runId,
    sourceKind: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_KIND,
    analysisMode: mode,
    deltaConvention: 'feature minus portable; negative values favor indirect firstInstance',
    primaryEndpoint: 'gpuPassTotalMs = gpuComputeMs + gpuRenderMs',
    protocol: {
      visibilityLevels: [...FIRST_INSTANCE_STANDALONE_DEPLOYMENT_VISIBILITY_LEVELS],
      measuredFramesPerTrial: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
      measuredBlocksPerTrial: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_BLOCKS,
      measuredFramesPerBlock: FIRST_INSTANCE_STANDALONE_DEPLOYMENT_BLOCK_SIZE,
      earlyLateDriftBlockCount: EARLY_LATE_BLOCK_COUNT,
      totalEqualityToleranceMs: TOTAL_EQUALITY_TOLERANCE_MS,
      inferentialUnit: 'quartet',
      trialEstimator: 'median of 60 consecutive eight-frame block means',
      matrixEstimator: 'conventional median of 12 quartet effects',
      framesSessionsAndTrialsAreInferentialUnits: false,
    },
    nTrialRecords: records.length,
    nRows: records.length * FIRST_INSTANCE_STANDALONE_DEPLOYMENT_MEASURED_FRAMES,
    coverage,
    decisionEligibility,
    trials,
    sessions,
    quartets,
    matrices,
    preregisteredNumericalDecision,
    preregisteredDecision: preregisteredNumericalDecision,
  };
}
