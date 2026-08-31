import { createHash } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

const REQUIRED_RUN_ARTIFACTS = Object.freeze([
  'frames.csv',
  'metadata.json',
  'trial-summaries.json',
  'validation-artifacts.json',
  'workload-manifests.json',
  'gpu-telemetry-summary.json',
]);

const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const ALLOWED_EVIDENCE_STATUSES = new Set(['development', 'candidate']);
const FIXED_SLICE_REPRESENTATION_MATRIX = 'fixed-slice-representation';
const FIXED_SLICE_REPRESENTATION_MODES = Object.freeze([
  'fixed-slice-per-bucket',
  'fixed-slice',
]);
const FIXED_SLICE_REPRESENTATION_VISIBILITIES = Object.freeze([0.2, 0.8, 0.99]);
const FIXED_SLICE_REPRESENTATION_REPETITIONS = 6;
const FIXED_SLICE_REPRESENTATION_WARMUP_FRAMES = 300;
const FIXED_SLICE_REPRESENTATION_MEASURED_FRAMES = 240;
const FIXED_SLICE_REPRESENTATION_OBJECT_COUNTS = Object.freeze([4_096, 16_384, 65_536]);
const FIXED_SLICE_REPRESENTATION_BUCKET_COUNTS = Object.freeze([1, 4, 32, 128]);
const FIXED_SLICE_REPRESENTATION_ORDERING =
  'six-repetition-balanced-ab-ba-with-rotated-visibility-order';
const PROVENANCE_STABILITY_FIELDS = Object.freeze([
  'commit',
  'tree',
  'ref',
  'dirty',
  'stagedChanges',
  'unstagedChanges',
  'untrackedFiles',
  'porcelainEntryCount',
  'porcelainByteCount',
  'porcelainSha256',
  'trackedFileCount',
  'trackedFilesSha256',
  'packageLockTracked',
  'packageLockSha256',
]);

const REQUIRED_COLUMNS = Object.freeze([
  'modeId',
  'targetVisibilityFraction',
  'gpuPassTotalMs',
  'gpuComputeMs',
  'gpuRenderMs',
  'cpuCommonUpdateMs',
  'cpuComputeSubmitMs',
  'cpuRenderSubmitMs',
  'cpuFrameBodyMs',
  'cpuSubmitTotalMs',
]);

const METRICS = Object.freeze([
  'gpuPassTotalMs',
  'gpuComputeMs',
  'gpuRenderMs',
  'cpuCommonUpdateMs',
  'cpuComputeSubmitMs',
  'cpuRenderSubmitMs',
  'cpuFrameBodyMs',
  'cpuSubmitTotalMs',
  'accountedCpuSubmitPlusGpuPassMs',
]);

const REPETITION_COLUMNS = Object.freeze([
  'repetitionIndex',
  'repetition',
  'suiteRepeat',
  'repeat',
  'trialId',
  'runId',
]);

function csvError(message, line) {
  const suffix = line === undefined ? '' : ` at CSV line ${line}`;
  return new Error(`${message}${suffix}`);
}

/**
 * Parse RFC-4180-style CSV, including quoted commas, newlines, and doubled quotes.
 */
export function parseCsv(text) {
  if (typeof text !== 'string') throw new TypeError('CSV input must be a string.');

  const source = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  let closedQuote = false;
  let line = 1;

  const pushField = () => {
    row.push(field);
    field = '';
    closedQuote = false;
  };
  const pushRow = () => {
    pushField();
    rows.push(row);
    row = [];
  };

  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];

    if (quoted) {
      if (character === '"') {
        if (source[index + 1] === '"') {
          field += '"';
          index += 1;
        } else {
          quoted = false;
          closedQuote = true;
        }
      } else {
        field += character;
        if (character === '\n') line += 1;
        if (character === '\r' && source[index + 1] !== '\n') line += 1;
      }
      continue;
    }

    if (closedQuote) {
      if (character === ',') {
        pushField();
      } else if (character === '\n' || character === '\r') {
        if (character === '\r' && source[index + 1] === '\n') index += 1;
        pushRow();
        line += 1;
      } else {
        throw csvError('Unexpected character after a closing quote', line);
      }
      continue;
    }

    if (character === '"') {
      if (field.length !== 0) throw csvError('Quote encountered inside an unquoted field', line);
      quoted = true;
    } else if (character === ',') {
      pushField();
    } else if (character === '\n' || character === '\r') {
      if (character === '\r' && source[index + 1] === '\n') index += 1;
      pushRow();
      line += 1;
    } else {
      field += character;
    }
  }

  if (quoted) throw csvError('Unterminated quoted field', line);
  if (closedQuote || field.length > 0 || row.length > 0) pushRow();

  const nonBlankRows = rows.filter((values) => !(values.length === 1 && values[0] === ''));
  if (nonBlankRows.length === 0) throw new Error('CSV input is empty.');

  const headers = nonBlankRows[0].map((header) => header.trim());
  if (headers.some((header) => header.length === 0)) {
    throw new Error('CSV header contains an empty column name.');
  }
  if (new Set(headers).size !== headers.length) {
    throw new Error('CSV header contains duplicate column names.');
  }

  const records = nonBlankRows.slice(1).map((values, recordIndex) => {
    if (values.length !== headers.length) {
      throw new Error(
        `CSV record ${recordIndex + 2} has ${values.length} fields; expected ${headers.length}.`,
      );
    }
    return Object.fromEntries(headers.map((header, columnIndex) => [header, values[columnIndex]]));
  });

  if (records.length === 0) throw new Error('CSV input contains a header but no frame rows.');
  return { headers, records };
}

function requireColumns(headers) {
  const available = new Set(headers);
  const missing = REQUIRED_COLUMNS.filter((column) => !available.has(column));
  if (missing.length > 0) throw new Error(`CSV is missing required columns: ${missing.join(', ')}.`);
}

function finiteNumber(value, field, recordNumber, { minimum = -Infinity, maximum = Infinity } = {}) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`Record ${recordNumber} has no ${field}.`);
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error(`Record ${recordNumber} has non-finite ${field}: ${JSON.stringify(value)}.`);
  }
  if (number < minimum || number > maximum) {
    throw new Error(
      `Record ${recordNumber} has out-of-range ${field}: ${number}; expected ${minimum}..${maximum}.`,
    );
  }
  return number;
}

function optionalBoolean(value, field, recordNumber) {
  if (value === undefined || value.trim() === '') return null;
  const normalized = value.trim().toLowerCase();
  if (normalized === 'true' || normalized === '1') return true;
  if (normalized === 'false' || normalized === '0') return false;
  throw new Error(`Record ${recordNumber} has invalid ${field}: ${JSON.stringify(value)}.`);
}

function optionalNonnegativeInteger(value, field, recordNumber) {
  if (value === undefined || value.trim() === '') return null;
  const number = Number(value);
  if (!Number.isInteger(number) || number < 0) {
    throw new Error(`Record ${recordNumber} has invalid ${field}: ${JSON.stringify(value)}.`);
  }
  return number;
}

function optionalNonemptyString(value, field, recordNumber) {
  if (value === undefined || value.trim() === '') return null;
  const normalized = value.trim();
  if (normalized.length === 0) {
    throw new Error(`Record ${recordNumber} has invalid ${field}: ${JSON.stringify(value)}.`);
  }
  return normalized;
}

function normalizedRepetition(value, recordNumber) {
  const text = value.trim();
  if (text.length === 0) throw new Error(`Record ${recordNumber} has no repetition identifier.`);
  const numeric = Number(text);
  return Number.isFinite(numeric) ? numeric : text;
}

function repetitionKey(value) {
  return `${typeof value}:${String(value)}`;
}

function nearestRank(values, fraction = 0.5) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(sorted.length * fraction) - 1),
  );
  return sorted[index];
}

function median(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const midpoint = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[midpoint - 1] + sorted[midpoint]) / 2
    : sorted[midpoint];
}

function compareRepetitions(left, right) {
  if (typeof left === 'number' && typeof right === 'number') return left - right;
  return String(left).localeCompare(String(right), undefined, { numeric: true });
}

function parseFrameRecords(parsed) {
  requireColumns(parsed.headers);
  const headerSet = new Set(parsed.headers);
  const repetitionColumn = REPETITION_COLUMNS.find((column) => headerSet.has(column)) ?? null;
  const hasModeOrderPosition = headerSet.has('modeOrderPosition');
  const hasPlannedModeOrder = headerSet.has('plannedModeOrder');

  const frames = parsed.records.map((record, index) => {
    const recordNumber = index + 2;
    const modeId = record.modeId.trim();
    if (modeId.length === 0) throw new Error(`Record ${recordNumber} has no modeId.`);

    const targetVisibilityFraction = finiteNumber(
      record.targetVisibilityFraction,
      'targetVisibilityFraction',
      recordNumber,
      { minimum: 0, maximum: 1 },
    );
    const usesComputeValue = optionalBoolean(record.usesCompute, 'usesCompute', recordNumber);
    const usesCompute = usesComputeValue ?? modeId !== 'draw-all';
    const computeText = record.gpuComputeMs.trim();
    let gpuComputeMs;
    if (computeText === '') {
      if (usesCompute) {
        throw new Error(`Record ${recordNumber} is a compute mode but has no gpuComputeMs.`);
      }
      gpuComputeMs = 0;
    } else {
      gpuComputeMs = finiteNumber(computeText, 'gpuComputeMs', recordNumber, { minimum: 0 });
    }
    const cpuComputeText = record.cpuComputeSubmitMs.trim();
    let cpuComputeSubmitMs;
    if (cpuComputeText === '') {
      if (usesCompute) {
        throw new Error(`Record ${recordNumber} is a compute mode but has no cpuComputeSubmitMs.`);
      }
      cpuComputeSubmitMs = 0;
    } else {
      cpuComputeSubmitMs = finiteNumber(
        cpuComputeText,
        'cpuComputeSubmitMs',
        recordNumber,
        { minimum: 0 },
      );
    }

    const repetition = repetitionColumn === null
      ? 1
      : normalizedRepetition(record[repetitionColumn], recordNumber);

    const gpuPassTotalMs = finiteNumber(
      record.gpuPassTotalMs,
      'gpuPassTotalMs',
      recordNumber,
      { minimum: 0 },
    );
    const cpuSubmitTotalMs = finiteNumber(
      record.cpuSubmitTotalMs,
      'cpuSubmitTotalMs',
      recordNumber,
      { minimum: 0 },
    );

    return {
      modeId,
      targetVisibilityFraction,
      repetition,
      modeOrderPosition: hasModeOrderPosition
        ? optionalNonnegativeInteger(record.modeOrderPosition, 'modeOrderPosition', recordNumber)
        : null,
      plannedModeOrder: hasPlannedModeOrder
        ? optionalNonemptyString(record.plannedModeOrder, 'plannedModeOrder', recordNumber)
        : null,
      usesCompute,
      gpuPassTotalMs,
      gpuComputeMs,
      gpuRenderMs: finiteNumber(record.gpuRenderMs, 'gpuRenderMs', recordNumber, { minimum: 0 }),
      cpuCommonUpdateMs: finiteNumber(
        record.cpuCommonUpdateMs,
        'cpuCommonUpdateMs',
        recordNumber,
        { minimum: 0 },
      ),
      cpuComputeSubmitMs,
      cpuRenderSubmitMs: finiteNumber(
        record.cpuRenderSubmitMs,
        'cpuRenderSubmitMs',
        recordNumber,
        { minimum: 0 },
      ),
      cpuFrameBodyMs: finiteNumber(
        record.cpuFrameBodyMs,
        'cpuFrameBodyMs',
        recordNumber,
        { minimum: 0 },
      ),
      cpuSubmitTotalMs,
      accountedCpuSubmitPlusGpuPassMs: cpuSubmitTotalMs + gpuPassTotalMs,
    };
  });

  return { frames, repetitionColumn };
}

function metricP50(frames) {
  return Object.fromEntries(METRICS.map((metric) => [
    metric,
    nearestRank(frames.map((frame) => frame[metric])),
  ]));
}

function groupFrames(frames) {
  const groups = new Map();
  for (const frame of frames) {
    const groupKey = `${frame.modeId}\u0000${frame.targetVisibilityFraction}`;
    let group = groups.get(groupKey);
    if (group === undefined) {
      group = {
        modeId: frame.modeId,
        targetVisibilityFraction: frame.targetVisibilityFraction,
        frames: [],
        trials: new Map(),
      };
      groups.set(groupKey, group);
    }
    group.frames.push(frame);

    const key = repetitionKey(frame.repetition);
    let trial = group.trials.get(key);
    if (trial === undefined) {
      trial = {
        repetition: frame.repetition,
        usesCompute: frame.usesCompute,
        modeOrderPosition: frame.modeOrderPosition,
        plannedModeOrder: frame.plannedModeOrder,
        frames: [],
      };
      group.trials.set(key, trial);
    } else if (trial.usesCompute !== frame.usesCompute) {
      throw new Error(
        `Mode ${frame.modeId}, visibility ${frame.targetVisibilityFraction}, repetition ${frame.repetition} mixes usesCompute values.`,
      );
    } else if (trial.modeOrderPosition !== frame.modeOrderPosition
      || trial.plannedModeOrder !== frame.plannedModeOrder) {
      throw new Error(
        `Mode ${frame.modeId}, visibility ${frame.targetVisibilityFraction}, repetition ${frame.repetition} mixes order audit values.`,
      );
    }
    trial.frames.push(frame);
  }

  return [...groups.values()]
    .sort((left, right) => (
      left.modeId.localeCompare(right.modeId)
      || left.targetVisibilityFraction - right.targetVisibilityFraction
    ))
    .map((group) => {
      const perTrialP50 = [...group.trials.values()]
        .sort((left, right) => compareRepetitions(left.repetition, right.repetition))
        .map((trial) => ({
          repetition: trial.repetition,
          modeOrderPosition: trial.modeOrderPosition,
          plannedModeOrder: trial.plannedModeOrder,
          nFrames: trial.frames.length,
          p50: metricP50(trial.frames),
        }));
      const medianAcrossTrials = Object.fromEntries(METRICS.map((metric) => [
        metric,
        median(perTrialP50.map((trial) => trial.p50[metric])),
      ]));

      return {
        modeId: group.modeId,
        targetVisibilityFraction: group.targetVisibilityFraction,
        nTrials: perTrialP50.length,
        nFrames: group.frames.length,
        perTrialP50,
        medianAcrossTrials,
      };
    });
}

function metricDelta(fixedValue, baselineValue) {
  const absoluteMs = fixedValue - baselineValue;
  let percent = null;
  if (baselineValue !== 0) percent = (absoluteMs / baselineValue) * 100;
  else if (fixedValue === 0) percent = 0;
  return { absoluteMs, percent };
}

function pairedDelta(fixedP50, baselineP50) {
  return Object.fromEntries(METRICS.map((metric) => [
    metric,
    metricDelta(fixedP50[metric], baselineP50[metric]),
  ]));
}

function summarizePairedDeltas(pairs) {
  return Object.fromEntries(METRICS.map((metric) => {
    const absoluteValues = pairs.map((pair) => pair.delta[metric].absoluteMs);
    const percentValues = pairs
      .map((pair) => pair.delta[metric].percent)
      .filter(Number.isFinite);
    return [metric, {
      absoluteMs: median(absoluteValues),
      percent: median(percentValues),
    }];
  }));
}

function resolvedModeOrderPosition(trial, modeId) {
  let position = trial.modeOrderPosition;
  if (trial.plannedModeOrder !== null) {
    const order = trial.plannedModeOrder.split('|');
    const plannedPosition = order.indexOf(modeId);
    if (plannedPosition < 0 || new Set(order).size !== order.length) {
      throw new Error(
        `Mode ${modeId}, repetition ${trial.repetition} has an invalid plannedModeOrder.`,
      );
    }
    if (position !== null && position !== plannedPosition) {
      throw new Error(
        `Mode ${modeId}, repetition ${trial.repetition} has inconsistent order audit values.`,
      );
    }
    position = plannedPosition;
  }
  return position;
}

function pairedOrderAudit(leftTrial, rightTrial, leftModeId, rightModeId) {
  if (leftTrial.plannedModeOrder !== null
    && rightTrial.plannedModeOrder !== null
    && leftTrial.plannedModeOrder !== rightTrial.plannedModeOrder) {
    throw new Error(
      `Modes ${leftModeId} and ${rightModeId}, repetition ${leftTrial.repetition} have different plannedModeOrder values.`,
    );
  }
  const leftModeOrderPosition = resolvedModeOrderPosition(leftTrial, leftModeId);
  const rightModeOrderPosition = resolvedModeOrderPosition(rightTrial, rightModeId);
  if (leftModeOrderPosition === null || rightModeOrderPosition === null) {
    return {
      status: 'unavailable',
      orderStratum: null,
      leftModeOrderPosition,
      rightModeOrderPosition,
      plannedModeOrder: leftTrial.plannedModeOrder ?? rightTrial.plannedModeOrder,
    };
  }
  if (leftModeOrderPosition === rightModeOrderPosition) {
    throw new Error(
      `Modes ${leftModeId} and ${rightModeId}, repetition ${leftTrial.repetition} claim the same mode-order position.`,
    );
  }
  return {
    status: 'classified',
    orderStratum: leftModeOrderPosition < rightModeOrderPosition ? 'left-first' : 'right-first',
    leftModeOrderPosition,
    rightModeOrderPosition,
    plannedModeOrder: leftTrial.plannedModeOrder ?? rightTrial.plannedModeOrder,
  };
}

function summarizeOrderStratification(pairs) {
  const classified = pairs.filter((pair) => pair.orderAudit.status === 'classified');
  const strata = ['left-first', 'right-first'].map((orderStratum) => {
    const stratumPairs = classified.filter((pair) => pair.orderAudit.orderStratum === orderStratum);
    return {
      orderStratum,
      nPairs: stratumPairs.length,
      medianPairedDelta: summarizePairedDeltas(stratumPairs),
    };
  });
  return {
    status: pairs.length === 0 || classified.length === 0
      ? 'unavailable'
      : classified.length === pairs.length
        ? 'complete'
        : 'partial',
    classifiedPairs: classified.length,
    unclassifiedPairs: pairs.length - classified.length,
    strata,
  };
}

function pairedContrasts(groups, leftModeId, rightModeId) {
  const leftGroups = groups.filter((group) => group.modeId === leftModeId);
  const rightByVisibility = new Map(
    groups
      .filter((group) => group.modeId === rightModeId)
      .map((group) => [group.targetVisibilityFraction, group]),
  );

  return leftGroups
    .map((leftGroup) => {
      const rightGroup = rightByVisibility.get(leftGroup.targetVisibilityFraction);
      if (rightGroup === undefined) return null;
      const rightTrials = new Map(
        rightGroup.perTrialP50.map((trial) => [repetitionKey(trial.repetition), trial]),
      );
      const leftTrials = new Map(
        leftGroup.perTrialP50.map((trial) => [repetitionKey(trial.repetition), trial]),
      );
      const pairs = leftGroup.perTrialP50
        .map((leftTrial) => {
          const rightTrial = rightTrials.get(repetitionKey(leftTrial.repetition));
          if (rightTrial === undefined) return null;
          const orderAudit = pairedOrderAudit(
            leftTrial,
            rightTrial,
            leftModeId,
            rightModeId,
          );
          return {
            repetition: leftTrial.repetition,
            leftP50: leftTrial.p50,
            rightP50: rightTrial.p50,
            delta: pairedDelta(leftTrial.p50, rightTrial.p50),
            orderAudit,
          };
        })
        .filter(Boolean);
      const unmatchedLeftRepetitions = leftGroup.perTrialP50
        .filter((trial) => !rightTrials.has(repetitionKey(trial.repetition)))
        .map((trial) => trial.repetition);
      const unmatchedRightRepetitions = rightGroup.perTrialP50
        .filter((trial) => !leftTrials.has(repetitionKey(trial.repetition)))
        .map((trial) => trial.repetition);

      return {
        targetVisibilityFraction: leftGroup.targetVisibilityFraction,
        leftModeId,
        rightModeId,
        nPairs: pairs.length,
        pairs,
        medianPairedDelta: summarizePairedDeltas(pairs),
        orderStratification: summarizeOrderStratification(pairs),
        unmatchedLeftRepetitions,
        unmatchedRightRepetitions,
      };
    })
    .filter(Boolean)
    .sort((left, right) => left.targetVisibilityFraction - right.targetVisibilityFraction);
}

function comparisonsAgainst(groups, baselineModeId) {
  return pairedContrasts(groups, 'fixed-slice', baselineModeId).map((contrast) => ({
    targetVisibilityFraction: contrast.targetVisibilityFraction,
    baselineModeId,
    nPairs: contrast.nPairs,
    pairs: contrast.pairs.map((pair) => ({
      repetition: pair.repetition,
      fixedSliceP50: pair.leftP50,
      baselineP50: pair.rightP50,
      delta: pair.delta,
    })),
    medianPairedDelta: contrast.medianPairedDelta,
    unmatchedFixedSliceRepetitions: contrast.unmatchedLeftRepetitions,
    unmatchedBaselineRepetitions: contrast.unmatchedRightRepetitions,
  }));
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function failVerification(message) {
  throw new Error(`Run verification failed: ${message}`);
}

function requireRecord(value, label) {
  if (!isRecord(value)) failVerification(`${label} must be an object.`);
  return value;
}

function requireArray(value, label) {
  if (!Array.isArray(value)) failVerification(`${label} must be an array.`);
  return value;
}

function requireNonemptyString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    failVerification(`${label} must be a non-empty string.`);
  }
  return value;
}

function requireInteger(value, label, { minimum = 0 } = {}) {
  if (!Number.isInteger(value) || value < minimum) {
    failVerification(`${label} must be an integer greater than or equal to ${minimum}.`);
  }
  return value;
}

function requireFiniteNumber(value, label, { minimum = -Infinity } = {}) {
  if (!Number.isFinite(value) || value < minimum) {
    failVerification(`${label} must be a finite number greater than or equal to ${minimum}.`);
  }
  return value;
}

function requireSha256(value, label) {
  if (!SHA256_PATTERN.test(value ?? '')) {
    failVerification(`${label} must be a lowercase SHA-256 digest.`);
  }
  return value;
}

function requireIsoTimestamp(value, label) {
  requireNonemptyString(value, label);
  if (!Number.isFinite(Date.parse(value))) failVerification(`${label} must be an ISO timestamp.`);
  return value;
}

function sha256Bytes(value) {
  return createHash('sha256').update(value).digest('hex');
}

function sha256Json(value) {
  return sha256Bytes(JSON.stringify(value));
}

function parseJsonArtifact(contents, name) {
  try {
    return JSON.parse(contents.toString('utf8'));
  } catch (error) {
    failVerification(
      `${name} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

function sortedUniqueStrings(value, label) {
  const values = requireArray(value, label).map((entry, index) => (
    requireNonemptyString(entry, `${label}[${index}]`)
  ));
  if (new Set(values).size !== values.length) failVerification(`${label} contains duplicates.`);
  return [...values].sort();
}

function requireSafeArtifactName(name, label) {
  requireNonemptyString(name, label);
  if (name === '.' || name === '..' || name.includes('/') || name.includes('\\')) {
    failVerification(`${label} is not a safe run-directory filename.`);
  }
  return name;
}

async function readOptionalFile(filename) {
  try {
    return await readFile(filename);
  } catch (error) {
    if (error?.code === 'ENOENT') return null;
    throw error;
  }
}

async function loadVerifiedArtifactContents(runDirectory) {
  const manifestPath = path.join(runDirectory, 'artifact-manifest.json');
  const manifestBytes = await readOptionalFile(manifestPath);
  if (manifestBytes === null) failVerification('artifact-manifest.json is missing.');
  const manifest = requireRecord(
    parseJsonArtifact(manifestBytes, 'artifact-manifest.json'),
    'artifact-manifest.json',
  );
  if (manifest.schemaVersion !== 2) {
    failVerification(`artifact-manifest.json schemaVersion must be 2; received ${JSON.stringify(manifest.schemaVersion)}.`);
  }
  requireNonemptyString(manifest.runId, 'artifact-manifest.json runId');
  if (manifest.hashAlgorithm !== 'sha256') {
    failVerification('artifact-manifest.json hashAlgorithm must be "sha256".');
  }

  const requiredNames = sortedUniqueStrings(
    manifest.requiredFiles,
    'artifact-manifest.json requiredFiles',
  );
  const expectedNames = [...REQUIRED_RUN_ARTIFACTS].sort();
  if (requiredNames.length !== expectedNames.length
    || requiredNames.some((name, index) => name !== expectedNames[index])) {
    failVerification(
      `artifact-manifest.json requiredFiles must be exactly ${REQUIRED_RUN_ARTIFACTS.join(', ')}.`,
    );
  }

  const optionalRecords = requireArray(
    manifest.optionalFiles,
    'artifact-manifest.json optionalFiles',
  );
  const optionalByName = new Map();
  for (const [index, optional] of optionalRecords.entries()) {
    requireRecord(optional, `artifact-manifest.json optionalFiles[${index}]`);
    const name = requireSafeArtifactName(
      optional.name,
      `artifact-manifest.json optionalFiles[${index}].name`,
    );
    if (optionalByName.has(name) || requiredNames.includes(name)) {
      failVerification(`artifact-manifest.json declares ${JSON.stringify(name)} more than once.`);
    }
    if (typeof optional.present !== 'boolean') {
      failVerification(`artifact-manifest.json optional file ${JSON.stringify(name)} lacks a boolean present flag.`);
    }
    if (typeof optional.evidenceAvailable !== 'boolean') {
      failVerification(
        `artifact-manifest.json optional file ${JSON.stringify(name)} lacks a boolean evidenceAvailable flag.`,
      );
    }
    if (optional.evidenceAvailable && !optional.present) {
      failVerification(
        `artifact-manifest.json optional file ${JSON.stringify(name)} cannot provide absent evidence.`,
      );
    }
    if (!optional.evidenceAvailable) {
      requireNonemptyString(
        optional.absenceReason,
        `artifact-manifest.json optional file ${JSON.stringify(name)} absence reason`,
      );
    }
    optionalByName.set(name, optional);
  }

  const entries = requireArray(manifest.files, 'artifact-manifest.json files');
  const entriesByName = new Map();
  for (const [index, entry] of entries.entries()) {
    requireRecord(entry, `artifact-manifest.json files[${index}]`);
    const name = requireSafeArtifactName(entry.name, `artifact-manifest.json files[${index}].name`);
    if (entriesByName.has(name)) {
      failVerification(`artifact-manifest.json has duplicate file entry ${JSON.stringify(name)}.`);
    }
    requireNonemptyString(entry.role, `artifact-manifest.json file ${JSON.stringify(name)} role`);
    if (typeof entry.required !== 'boolean' || typeof entry.present !== 'boolean') {
      failVerification(
        `artifact-manifest.json file ${JSON.stringify(name)} must declare boolean required and present flags.`,
      );
    }
    const declaredRequired = requiredNames.includes(name);
    const declaredOptional = optionalByName.has(name);
    if (!declaredRequired && !declaredOptional) {
      failVerification(`artifact-manifest.json file ${JSON.stringify(name)} is not declared required or optional.`);
    }
    if (entry.required !== declaredRequired) {
      failVerification(`artifact-manifest.json file ${JSON.stringify(name)} has an inconsistent required flag.`);
    }
    if (declaredRequired && !entry.present) {
      failVerification(`required artifact ${name} is marked absent.`);
    }
    if (declaredOptional && entry.present !== optionalByName.get(name).present) {
      failVerification(`optional artifact ${name} has inconsistent present flags.`);
    }
    if (!entry.present) {
      requireNonemptyString(
        entry.absenceReason,
        `artifact-manifest.json file ${JSON.stringify(name)} absenceReason`,
      );
    }
    entriesByName.set(name, entry);
  }

  const declaredNames = [...requiredNames, ...optionalByName.keys()].sort();
  const entryNames = [...entriesByName.keys()].sort();
  if (declaredNames.length !== entryNames.length
    || declaredNames.some((name, index) => name !== entryNames[index])) {
    failVerification('artifact-manifest.json files do not exactly cover its required and optional declarations.');
  }

  const contentsByName = new Map();
  for (const [name, entry] of entriesByName) {
    const contents = await readOptionalFile(path.join(runDirectory, name));
    if (!entry.present) {
      if (contents !== null) {
        failVerification(`artifact ${name} exists but is marked absent in artifact-manifest.json.`);
      }
      continue;
    }
    if (contents === null) failVerification(`artifact ${name} is missing.`);
    requireInteger(entry.bytes, `artifact-manifest.json file ${JSON.stringify(name)} bytes`);
    requireSha256(entry.sha256, `artifact-manifest.json file ${JSON.stringify(name)} sha256`);
    if (contents.length !== entry.bytes) {
      failVerification(
        `artifact ${name} byte size is ${contents.length}; manifest records ${entry.bytes}.`,
      );
    }
    const digest = sha256Bytes(contents);
    if (digest !== entry.sha256) {
      failVerification(`artifact ${name} SHA-256 does not match artifact-manifest.json.`);
    }
    contentsByName.set(name, contents);
  }

  return { manifest, contentsByName };
}

function validateCandidateProvenance(metadata) {
  if (metadata.evidenceStatus !== 'candidate') return;
  const provenance = requireRecord(metadata.sourceProvenance, 'metadata sourceProvenance');
  if (provenance.stable !== true) {
    failVerification('candidate evidence does not declare stable source provenance.');
  }
  const start = requireRecord(provenance.start, 'metadata sourceProvenance.start');
  const end = requireRecord(provenance.end, 'metadata sourceProvenance.end');
  for (const [label, record] of [['start', start], ['end', end]]) {
    if (record.status !== 'available') {
      failVerification(`candidate source provenance ${label} capture is unavailable.`);
    }
    if (record.dirty !== false
      || record.stagedChanges !== 0
      || record.unstagedChanges !== 0
      || record.untrackedFiles !== 0) {
      failVerification(`candidate source provenance ${label} capture is not clean.`);
    }
    if (record.packageLockTracked !== true) {
      failVerification(`candidate source provenance ${label} capture lacks a tracked package lock.`);
    }
    if (!/^[0-9a-f]{40,64}$/.test(record.commit ?? '')
      || !/^[0-9a-f]{40,64}$/.test(record.tree ?? '')) {
      failVerification(`candidate source provenance ${label} capture lacks Git object IDs.`);
    }
    if (record.ref !== null && (typeof record.ref !== 'string' || record.ref === '')) {
      failVerification(`candidate source provenance ${label} capture has an invalid ref.`);
    }
    for (const field of [
      'stagedChanges',
      'unstagedChanges',
      'untrackedFiles',
      'porcelainEntryCount',
      'porcelainByteCount',
      'trackedFileCount',
    ]) {
      requireInteger(record[field], `candidate provenance ${label} ${field}`);
    }
    requireSha256(record.porcelainSha256, `candidate provenance ${label} porcelainSha256`);
    requireSha256(record.trackedFilesSha256, `candidate provenance ${label} trackedFilesSha256`);
    requireSha256(record.packageLockSha256, `candidate provenance ${label} packageLockSha256`);
  }
  for (const field of PROVENANCE_STABILITY_FIELDS) {
    if (start[field] !== end[field]) {
      failVerification(`candidate source provenance changed at field ${field}.`);
    }
  }
}

function orderedValuesMatch(actual, expected) {
  return Array.isArray(actual)
    && actual.length === expected.length
    && actual.every((value, index) => value === expected[index]);
}

export function validateProtocolMatrix(protocol) {
  requireRecord(protocol, 'metadata.json protocol');
  if (protocol.matrixKind !== FIXED_SLICE_REPRESENTATION_MATRIX) return;
  if (!orderedValuesMatch(protocol.modes, FIXED_SLICE_REPRESENTATION_MODES)) {
    failVerification(
      'fixed-slice-representation protocol modes must be exactly fixed-slice-per-bucket, fixed-slice.',
    );
  }
  if (protocol.repetitions !== FIXED_SLICE_REPRESENTATION_REPETITIONS) {
    failVerification('fixed-slice-representation protocol must use exactly six repetitions.');
  }
  if (!orderedValuesMatch(
    protocol.visibilityLevels,
    FIXED_SLICE_REPRESENTATION_VISIBILITIES,
  )) {
    failVerification(
      'fixed-slice-representation visibility levels must be exactly 0.2, 0.8, 0.99.',
    );
  }
  if (protocol.heterogeneousComparator !== null) {
    failVerification('fixed-slice-representation heterogeneousComparator must be null.');
  }
  if (protocol.ordering !== FIXED_SLICE_REPRESENTATION_ORDERING) {
    failVerification(
      `fixed-slice-representation ordering must be ${JSON.stringify(FIXED_SLICE_REPRESENTATION_ORDERING)}.`,
    );
  }
  if (protocol.warmupFrames !== FIXED_SLICE_REPRESENTATION_WARMUP_FRAMES
    || protocol.measuredFrames !== FIXED_SLICE_REPRESENTATION_MEASURED_FRAMES) {
    failVerification(
      'fixed-slice-representation protocol must use 300 warmup and 240 measured frames.',
    );
  }
  if (!FIXED_SLICE_REPRESENTATION_OBJECT_COUNTS.includes(protocol.objectCount)
    || !FIXED_SLICE_REPRESENTATION_BUCKET_COUNTS.includes(protocol.bucketCount)) {
    failVerification('fixed-slice-representation protocol has an unsupported workload size.');
  }
  if (protocol.matrix
    !== `${FIXED_SLICE_REPRESENTATION_MATRIX}-o${protocol.objectCount}-b${protocol.bucketCount}`) {
    failVerification('fixed-slice-representation matrix identifier does not match its workload.');
  }
  const expectedScaleRole = protocol.bucketCount === 1
    ? 'negative-control-equal-mesh-render-object-count'
    : 'primary-one-versus-b-mesh-render-object-representation-ablation';
  if (protocol.representationScaleRole !== expectedScaleRole) {
    failVerification('fixed-slice-representation scale role does not match its bucket count.');
  }
}

function requireMetadataCompleteness(metadata, manifest) {
  requireRecord(metadata, 'metadata.json');
  if (metadata.schemaVersion !== 2) {
    failVerification(`metadata.json schemaVersion must be 2; received ${JSON.stringify(metadata.schemaVersion)}.`);
  }
  const runId = requireNonemptyString(metadata.runId, 'metadata.json runId');
  if (manifest.runId !== runId) {
    failVerification('artifact-manifest.json and metadata.json runId values differ.');
  }
  if (metadata.status !== 'complete') {
    failVerification(`metadata.json status must be "complete"; received ${JSON.stringify(metadata.status)}.`);
  }
  if (metadata.error !== null) failVerification('metadata.json contains a run error.');
  const pageErrors = requireArray(metadata.pageErrors, 'metadata.json pageErrors');
  if (pageErrors.length !== 0) failVerification('metadata.json contains page errors.');
  if (!ALLOWED_EVIDENCE_STATUSES.has(metadata.evidenceStatus)) {
    failVerification(`metadata.json has unsupported evidenceStatus ${JSON.stringify(metadata.evidenceStatus)}.`);
  }
  const startedAt = requireIsoTimestamp(metadata.startedAt, 'metadata.json startedAt');
  const completedAt = requireIsoTimestamp(metadata.completedAt, 'metadata.json completedAt');
  if (Date.parse(completedAt) < Date.parse(startedAt)) {
    failVerification('metadata.json completedAt precedes startedAt.');
  }
  requireFiniteNumber(metadata.elapsedMs, 'metadata.json elapsedMs', { minimum: 0 });
  requireRecord(metadata.environment, 'metadata.json environment');
  const sourceProvenance = requireRecord(
    metadata.sourceProvenance,
    'metadata.json sourceProvenance',
  );
  requireRecord(sourceProvenance.start, 'metadata.json sourceProvenance.start');
  requireRecord(sourceProvenance.end, 'metadata.json sourceProvenance.end');
  if (sourceProvenance.stable !== true
    && sourceProvenance.stable !== false
    && sourceProvenance.stable !== null) {
    failVerification('metadata.json sourceProvenance.stable must be true, false, or null.');
  }
  requireRecord(metadata.workload, 'metadata.json workload');
  const protocol = requireRecord(metadata.protocol, 'metadata.json protocol');
  requireNonemptyString(protocol.matrix, 'metadata.json protocol.matrix');
  requireInteger(protocol.objectCount, 'metadata.json protocol.objectCount', { minimum: 1 });
  requireInteger(protocol.bucketCount, 'metadata.json protocol.bucketCount', { minimum: 1 });
  requireInteger(protocol.repetitions, 'metadata.json protocol.repetitions', { minimum: 1 });
  requireInteger(protocol.warmupFrames, 'metadata.json protocol.warmupFrames', { minimum: 1 });
  requireInteger(protocol.measuredFrames, 'metadata.json protocol.measuredFrames', { minimum: 1 });
  const modes = requireArray(protocol.modes, 'metadata.json protocol.modes');
  if (modes.length === 0 || modes.some((mode) => typeof mode !== 'string' || mode === '')) {
    failVerification('metadata.json protocol.modes must contain mode identifiers.');
  }
  const visibilityLevels = requireArray(
    protocol.visibilityLevels,
    'metadata.json protocol.visibilityLevels',
  );
  if (visibilityLevels.length === 0 || visibilityLevels.some((value) => !Number.isFinite(value))) {
    failVerification('metadata.json protocol.visibilityLevels must contain finite values.');
  }
  validateProtocolMatrix(protocol);
  const plan = requireArray(metadata.plan, 'metadata.json plan');
  if (plan.length === 0) failVerification('metadata.json plan is empty.');
  requireInteger(metadata.expectedTrialCount, 'metadata.json expectedTrialCount', { minimum: 1 });
  requireInteger(metadata.completedTrialCount, 'metadata.json completedTrialCount');
  requireInteger(metadata.acceptedTrialCount, 'metadata.json acceptedTrialCount');
  requireInteger(metadata.frameRowCount, 'metadata.json frameRowCount', { minimum: 1 });
  requireInteger(metadata.validationArtifactCount, 'metadata.json validationArtifactCount', { minimum: 1 });
  requireArray(metadata.validationArtifactSha256, 'metadata.json validationArtifactSha256');
  if (metadata.expectedTrialCount !== plan.length) {
    failVerification('metadata.json expectedTrialCount does not equal plan length.');
  }
  const protocolTrialCount = protocol.repetitions * modes.length * visibilityLevels.length;
  if (metadata.expectedTrialCount !== protocolTrialCount) {
    failVerification('metadata.json expectedTrialCount is inconsistent with the protocol matrix.');
  }
  if (metadata.completedTrialCount !== metadata.expectedTrialCount
    || metadata.acceptedTrialCount !== metadata.expectedTrialCount) {
    failVerification('metadata.json does not report every expected trial as completed and accepted.');
  }
  validateCandidateProvenance(metadata);
  return { plan, protocol };
}

function requireMatchingIdentity(actual, expected, label) {
  for (const field of [
    'trialId',
    'planIndex',
    'repetitionIndex',
    'modeId',
    'visibilityFraction',
  ]) {
    if (actual[field] !== expected[field]) {
      failVerification(`${label} ${field} does not match metadata.json plan.`);
    }
  }
}

function requireExactPermutation(value, expectedValues, label) {
  const order = requireArray(value, label);
  if (order.length !== expectedValues.length
    || new Set(order).size !== order.length
    || expectedValues.some((expected) => !order.includes(expected))) {
    failVerification(`${label} must be an exact permutation of the protocol values.`);
  }
  return order;
}

export function validateBenchmarkPlan(plan, metadata) {
  validateProtocolMatrix(metadata.protocol);
  const byTrialId = new Map();
  const byPlanIndex = new Map();
  const matrixCells = new Set();
  const repetitionOrders = new Map();
  const modes = metadata.protocol.modes;
  const visibilityLevels = metadata.protocol.visibilityLevels;
  for (const [arrayIndex, item] of plan.entries()) {
    requireRecord(item, `metadata.json plan[${arrayIndex}]`);
    const trialId = requireNonemptyString(item.trialId, `metadata.json plan[${arrayIndex}].trialId`);
    requireInteger(item.planIndex, `metadata.json plan[${arrayIndex}].planIndex`);
    requireInteger(item.repetitionIndex, `metadata.json plan[${arrayIndex}].repetitionIndex`);
    requireNonemptyString(item.modeId, `metadata.json plan[${arrayIndex}].modeId`);
    const modeOrder = requireExactPermutation(
      item.modeOrder,
      modes,
      `metadata.json plan[${arrayIndex}].modeOrder`,
    );
    requireInteger(
      item.modeOrderPosition,
      `metadata.json plan[${arrayIndex}].modeOrderPosition`,
    );
    if (modeOrder[item.modeOrderPosition] !== item.modeId) {
      failVerification(`metadata.json plan[${arrayIndex}] mode position is inconsistent.`);
    }
    requireFiniteNumber(
      item.visibilityFraction,
      `metadata.json plan[${arrayIndex}].visibilityFraction`,
      { minimum: 0 },
    );
    const visibilityOrder = requireExactPermutation(
      item.visibilityOrder,
      visibilityLevels,
      `metadata.json plan[${arrayIndex}].visibilityOrder`,
    );
    requireInteger(
      item.visibilityOrderPosition,
      `metadata.json plan[${arrayIndex}].visibilityOrderPosition`,
    );
    if (visibilityOrder[item.visibilityOrderPosition] !== item.visibilityFraction) {
      failVerification(`metadata.json plan[${arrayIndex}] visibility position is inconsistent.`);
    }
    if (item.runId !== metadata.runId) failVerification(`metadata.json plan[${arrayIndex}] has the wrong runId.`);
    requireInteger(item.objectCount, `metadata.json plan[${arrayIndex}].objectCount`, { minimum: 1 });
    requireInteger(item.bucketCount, `metadata.json plan[${arrayIndex}].bucketCount`, { minimum: 1 });
    if (!metadata.protocol.modes.includes(item.modeId)
      || !metadata.protocol.visibilityLevels.includes(item.visibilityFraction)
      || item.repetitionIndex >= metadata.protocol.repetitions
      || item.objectCount !== metadata.protocol.objectCount
      || item.bucketCount !== metadata.protocol.bucketCount) {
      failVerification(`metadata.json plan[${arrayIndex}] is inconsistent with the protocol.`);
    }
    if (item.planIndex !== arrayIndex) {
      failVerification('metadata.json plan indexes are not contiguous and ordered from zero.');
    }
    if (byTrialId.has(trialId) || byPlanIndex.has(item.planIndex)) {
      failVerification('metadata.json plan has duplicate trial identities.');
    }
    const cellKey = JSON.stringify([
      item.repetitionIndex,
      item.modeId,
      item.visibilityFraction,
    ]);
    if (matrixCells.has(cellKey)) {
      failVerification('metadata.json plan duplicates a repetition/mode/visibility cell.');
    }
    matrixCells.add(cellKey);

    const orderRecord = repetitionOrders.get(item.repetitionIndex);
    const modeOrderSignature = JSON.stringify(modeOrder);
    const visibilityOrderSignature = JSON.stringify(visibilityOrder);
    if (orderRecord === undefined) {
      repetitionOrders.set(item.repetitionIndex, {
        modeOrder: [...modeOrder],
        visibilityOrder: [...visibilityOrder],
        modeOrderSignature,
        visibilityOrderSignature,
      });
    } else if (orderRecord.modeOrderSignature !== modeOrderSignature
      || orderRecord.visibilityOrderSignature !== visibilityOrderSignature) {
      failVerification('metadata.json plan changes an order within one repetition.');
    }
    byTrialId.set(trialId, item);
    byPlanIndex.set(item.planIndex, item);
  }

  for (let repetition = 0; repetition < metadata.protocol.repetitions; repetition += 1) {
    if (!repetitionOrders.has(repetition)) {
      failVerification(`metadata.json plan omits repetition ${repetition}.`);
    }
    for (const mode of modes) {
      for (const visibility of visibilityLevels) {
        const cellKey = JSON.stringify([repetition, mode, visibility]);
        if (!matrixCells.has(cellKey)) {
          failVerification('metadata.json plan omits a repetition/mode/visibility cell.');
        }
      }
    }
  }

  let executionIndex = 0;
  for (let repetition = 0; repetition < metadata.protocol.repetitions; repetition += 1) {
    const orderRecord = repetitionOrders.get(repetition);
    if (metadata.protocol.matrixKind === FIXED_SLICE_REPRESENTATION_MATRIX) {
      const expectedModeOrder = repetition % 2 === 0
        ? [...FIXED_SLICE_REPRESENTATION_MODES]
        : [...FIXED_SLICE_REPRESENTATION_MODES].reverse();
      const visibilityOffset = repetition % FIXED_SLICE_REPRESENTATION_VISIBILITIES.length;
      const expectedVisibilityOrder = [
        ...FIXED_SLICE_REPRESENTATION_VISIBILITIES.slice(visibilityOffset),
        ...FIXED_SLICE_REPRESENTATION_VISIBILITIES.slice(0, visibilityOffset),
      ];
      if (!orderedValuesMatch(orderRecord.modeOrder, expectedModeOrder)) {
        failVerification(
          'fixed-slice-representation mode orders must alternate AB/BA by repetition.',
        );
      }
      if (!orderedValuesMatch(orderRecord.visibilityOrder, expectedVisibilityOrder)) {
        failVerification(
          'fixed-slice-representation visibility orders must rotate by repetition.',
        );
      }
    }
    for (let visibilityPosition = 0;
      visibilityPosition < orderRecord.visibilityOrder.length;
      visibilityPosition += 1) {
      for (let modePosition = 0;
        modePosition < orderRecord.modeOrder.length;
        modePosition += 1) {
        const item = plan[executionIndex];
        if (item.repetitionIndex !== repetition
          || item.visibilityOrderPosition !== visibilityPosition
          || item.visibilityFraction !== orderRecord.visibilityOrder[visibilityPosition]
          || item.modeOrderPosition !== modePosition
          || item.modeId !== orderRecord.modeOrder[modePosition]) {
          failVerification(
            'metadata.json plan execution order must be repetition-contiguous with visibility outer and mode inner.',
          );
        }
        executionIndex += 1;
      }
    }
  }

  if (metadata.protocol.repetitions % modes.length === 0) {
    const expectedPerPosition = metadata.protocol.repetitions / modes.length;
    for (const mode of modes) {
      for (let position = 0; position < modes.length; position += 1) {
        const count = [...repetitionOrders.values()]
          .filter((record) => record.modeOrder[position] === mode).length;
        if (count !== expectedPerPosition) {
          failVerification('metadata.json mode ordering is not position-balanced.');
        }
      }
    }
  }
  if (metadata.protocol.repetitions % visibilityLevels.length === 0) {
    const expectedPerPosition = metadata.protocol.repetitions / visibilityLevels.length;
    for (const visibility of visibilityLevels) {
      for (let position = 0; position < visibilityLevels.length; position += 1) {
        const count = [...repetitionOrders.values()]
          .filter((record) => record.visibilityOrder[position] === visibility).length;
        if (count !== expectedPerPosition) {
          failVerification('metadata.json visibility ordering is not position-balanced.');
        }
      }
    }
  }
  return { byTrialId, byPlanIndex };
}

function validateTrialSummaries(trialSummaries, metadata, planIndex) {
  requireArray(trialSummaries, 'trial-summaries.json');
  if (trialSummaries.length !== metadata.expectedTrialCount) {
    failVerification('trial-summaries.json count does not equal metadata expectedTrialCount.');
  }
  const byTrialId = new Map();
  for (const [index, summary] of trialSummaries.entries()) {
    requireRecord(summary, `trial-summaries.json[${index}]`);
    const trialId = requireNonemptyString(summary.trialId, `trial-summaries.json[${index}].trialId`);
    if (byTrialId.has(trialId)) failVerification('trial-summaries.json has duplicate trial IDs.');
    const planned = planIndex.byTrialId.get(trialId);
    if (planned === undefined) {
      failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} is absent from the plan.`);
    }
    requireMatchingIdentity(summary, planned, `trial-summaries.json trial ${JSON.stringify(trialId)}`);
    if (summary.objectCount !== planned.objectCount || summary.bucketCount !== planned.bucketCount) {
      failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} workload size differs from the plan.`);
    }
    if (summary.accepted !== true) {
      failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} was not accepted.`);
    }
    const rejectionReasons = requireArray(
      summary.rejectionReasons,
      `trial-summaries.json trial ${JSON.stringify(trialId)} rejectionReasons`,
    );
    if (rejectionReasons.length !== 0) {
      failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} has rejection reasons.`);
    }
    requireIsoTimestamp(
      summary.startedAt,
      `trial-summaries.json trial ${JSON.stringify(trialId)} startedAt`,
    );
    requireIsoTimestamp(
      summary.completedAt,
      `trial-summaries.json trial ${JSON.stringify(trialId)} completedAt`,
    );
    requireFiniteNumber(
      summary.elapsedMs,
      `trial-summaries.json trial ${JSON.stringify(trialId)} elapsedMs`,
      { minimum: 0 },
    );
    const validation = requireRecord(
      summary.validation,
      `trial-summaries.json trial ${JSON.stringify(trialId)} validation`,
    );
    if (validation.pass !== true) {
      failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} lacks passing validation.`);
    }
    requireSha256(
      validation.artifactSha256,
      `trial-summaries.json trial ${JSON.stringify(trialId)} validation artifactSha256`,
    );
    const timestamps = requireRecord(
      summary.timestamps,
      `trial-summaries.json trial ${JSON.stringify(trialId)} timestamps`,
    );
    if (timestamps.accepted !== true
      || timestamps.available !== true
      || timestamps.rowCount !== metadata.protocol.measuredFrames
      || timestamps.missingRenderFrames !== 0
      || timestamps.missingComputeFrames !== 0) {
      failVerification(`trial-summaries.json trial ${JSON.stringify(trialId)} lacks complete accepted timestamps.`);
    }
    byTrialId.set(trialId, summary);
  }
  return byTrialId;
}

function bodyWithoutSha256(record) {
  return Object.fromEntries(Object.entries(record).filter(([key]) => key !== 'sha256'));
}

function validateEvidenceCapture(capture, label) {
  requireRecord(capture, label);
  requireIsoTimestamp(capture.capturedAt, `${label}.capturedAt`);
  if (capture.accepted !== true) failVerification(`${label} was not accepted.`);
  const rejectionReasons = requireArray(capture.rejectionReasons, `${label}.rejectionReasons`);
  if (rejectionReasons.length !== 0) failVerification(`${label} has rejection reasons.`);
  const workload = requireRecord(capture.workload, `${label}.workload`);
  requireInteger(workload.scenarioSeed, `${label}.workload.scenarioSeed`);
  requireSha256(workload.geometryFixtureSha256, `${label}.workload.geometryFixtureSha256`);
  requireSha256(workload.scenarioSha256, `${label}.workload.scenarioSha256`);
  const validation = requireRecord(capture.validation, `${label}.validation`);
  requireSha256(validation.payloadSha256, `${label}.validation.payloadSha256`);
  requireSha256(validation.semanticSha256, `${label}.validation.semanticSha256`);
  requireRecord(validation.payload, `${label}.validation.payload`);
  if (sha256Json(validation.payload) !== validation.payloadSha256) {
    failVerification(`${label}.validation payload SHA-256 is inconsistent.`);
  }
  return {
    scenarioSeed: workload.scenarioSeed,
    geometrySha256: workload.geometryFixtureSha256,
    scenarioSha256: workload.scenarioSha256,
    payloadSha256: validation.payloadSha256,
    semanticSha256: validation.semanticSha256,
  };
}

function validateWorkloadManifests(catalog, metadata) {
  requireRecord(catalog, 'workload-manifests.json');
  if (catalog.schemaVersion !== 1 || catalog.hashAlgorithm !== 'sha256') {
    failVerification('workload-manifests.json has an unsupported schema or hash algorithm.');
  }
  const geometries = requireRecord(
    catalog.geometryFixturesBySha256,
    'workload-manifests.json geometryFixturesBySha256',
  );
  const scenarios = requireRecord(
    catalog.scenariosBySha256,
    'workload-manifests.json scenariosBySha256',
  );
  const invalid = requireArray(
    catalog.invalidObservations,
    'workload-manifests.json invalidObservations',
  );
  if (invalid.length !== 0) {
    failVerification('workload-manifests.json contains invalid workload observations.');
  }
  for (const [digest, record] of Object.entries(geometries)) {
    requireSha256(digest, 'workload-manifests.json geometry key');
    requireRecord(record, `workload-manifests.json geometry ${digest}`);
    if (record.sha256 !== digest) {
      failVerification(`workload-manifests.json geometry ${digest} has an inconsistent digest.`);
    }
  }
  for (const [digest, record] of Object.entries(scenarios)) {
    requireSha256(digest, 'workload-manifests.json scenario key');
    requireRecord(record, `workload-manifests.json scenario ${digest}`);
    if (record.sha256 !== digest) {
      failVerification(`workload-manifests.json scenario ${digest} has an inconsistent digest.`);
    }
  }
  if (metadata.workload.manifestArtifact !== 'workload-manifests.json') {
    failVerification('metadata workload does not link workload-manifests.json.');
  }
  const metadataGeometrySha256 = requireSha256(
    metadata.workload.geometryFixtureSha256,
    'metadata.json workload.geometryFixtureSha256',
  );
  if (geometries[metadataGeometrySha256] === undefined) {
    failVerification('metadata geometry digest is absent from workload-manifests.json.');
  }
  const scenarioLinks = requireRecord(
    metadata.workload.scenarioSha256ByVisibility,
    'metadata.json workload.scenarioSha256ByVisibility',
  );
  for (const [visibility, digest] of Object.entries(scenarioLinks)) {
    requireSha256(digest, `metadata scenario digest for visibility ${visibility}`);
    if (scenarios[digest] === undefined) {
      failVerification(`metadata scenario digest for visibility ${visibility} is absent from the catalog.`);
    }
  }
  return { geometries, scenarios };
}

function validateValidationArtifacts(
  validationArtifacts,
  metadata,
  planIndex,
  summariesByTrialId,
  workloadCatalog,
) {
  requireArray(validationArtifacts, 'validation-artifacts.json');
  if (validationArtifacts.length !== metadata.validationArtifactCount
    || validationArtifacts.length !== metadata.expectedTrialCount) {
    failVerification('validation-artifacts.json count is inconsistent with metadata trial counts.');
  }
  if (metadata.validationArtifactSha256.length !== validationArtifacts.length) {
    failVerification('metadata validationArtifactSha256 count is inconsistent.');
  }

  const seenTrialIds = new Set();
  for (const [index, artifact] of validationArtifacts.entries()) {
    requireRecord(artifact, `validation-artifacts.json[${index}]`);
    if (artifact.schemaVersion !== 2) {
      failVerification(`validation-artifacts.json[${index}] schemaVersion must be 2.`);
    }
    const trialId = requireNonemptyString(
      artifact.trialId,
      `validation-artifacts.json[${index}].trialId`,
    );
    if (seenTrialIds.has(trialId)) failVerification('validation-artifacts.json has duplicate trial IDs.');
    seenTrialIds.add(trialId);
    const planned = planIndex.byTrialId.get(trialId);
    if (planned === undefined) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} is absent from the plan.`);
    }
    requireMatchingIdentity(artifact, planned, `validation artifact ${JSON.stringify(trialId)}`);
    if (artifact.objectCount !== planned.objectCount || artifact.bucketCount !== planned.bucketCount) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} workload size differs from the plan.`);
    }
    if (artifact.status !== 'accepted') {
      failVerification(`validation artifact ${JSON.stringify(trialId)} status is not "accepted".`);
    }
    const rejectionReasons = requireArray(
      artifact.rejectionReasons,
      `validation artifact ${JSON.stringify(trialId)} rejectionReasons`,
    );
    if (rejectionReasons.length !== 0) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} has rejection reasons.`);
    }
    requireRecord(
      artifact.selectedConfig,
      `validation artifact ${JSON.stringify(trialId)} selectedConfig`,
    );
    requireSha256(artifact.sha256, `validation artifact ${JSON.stringify(trialId)} sha256`);
    if (sha256Json(bodyWithoutSha256(artifact)) !== artifact.sha256) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} record SHA-256 is inconsistent.`);
    }
    if (metadata.validationArtifactSha256[index] !== artifact.sha256) {
      failVerification(`metadata validation digest at index ${index} does not match its artifact.`);
    }
    const summary = summariesByTrialId.get(trialId);
    if (summary?.validation?.artifactSha256 !== artifact.sha256) {
      failVerification(`trial summary and validation artifact digests differ for ${JSON.stringify(trialId)}.`);
    }

    const captures = ['pre', 'timingStart', 'post'].map((name) => validateEvidenceCapture(
      artifact[name],
      `validation artifact ${JSON.stringify(trialId)} ${name}`,
    ));
    for (const field of [
      'scenarioSeed',
      'geometrySha256',
      'scenarioSha256',
      'semanticSha256',
    ]) {
      if (captures.some((capture) => capture[field] !== captures[0][field])) {
        failVerification(
          `validation artifact ${JSON.stringify(trialId)} changed ${field} across captures.`,
        );
      }
    }
    if (captures[0].scenarioSeed !== metadata.workload.scenarioSeed) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} scenario seed differs from metadata.`);
    }
    if (captures[0].geometrySha256 !== metadata.workload.geometryFixtureSha256) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} geometry digest differs from metadata.`);
    }
    const expectedScenarioSha256 = metadata.workload.scenarioSha256ByVisibility?.[
      String(artifact.visibilityFraction)
    ];
    if (captures[0].scenarioSha256 !== expectedScenarioSha256) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} scenario digest differs from metadata.`);
    }
    if (workloadCatalog.geometries[captures[0].geometrySha256] === undefined
      || workloadCatalog.scenarios[captures[0].scenarioSha256] === undefined) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} links an unknown workload manifest.`);
    }
    if (artifact.modeId !== 'three-blocks-historical'
      && captures.some((capture) => capture.payloadSha256 !== captures[0].payloadSha256)) {
      failVerification(`validation artifact ${JSON.stringify(trialId)} changed its exact payload.`);
    }
  }
}

function exactCsvInteger(value, label) {
  if (typeof value !== 'string' || value.trim() === '') failVerification(`${label} is missing.`);
  const number = Number(value);
  if (!Number.isInteger(number)) failVerification(`${label} must be an integer.`);
  return number;
}

function exactCsvNumber(value, label) {
  if (typeof value !== 'string' || value.trim() === '') failVerification(`${label} is missing.`);
  const number = Number(value);
  if (!Number.isFinite(number)) failVerification(`${label} must be finite.`);
  return number;
}

function validateVerifiedFrames(parsed, metadata, planIndex, summariesByTrialId) {
  const requiredAuditColumns = [
    'runId',
    'trialId',
    'planIndex',
    'repetitionIndex',
    'frameIndex',
    'objectCount',
    'bucketCount',
    'validationPass',
    'timestampAvailable',
  ];
  if (metadata.protocol.matrixKind === FIXED_SLICE_REPRESENTATION_MATRIX) {
    requiredAuditColumns.push(
      'modeOrderPosition',
      'visibilityOrderPosition',
      'plannedModeOrder',
      'plannedVisibilityOrder',
    );
  }
  const headers = new Set(parsed.headers);
  const missing = requiredAuditColumns.filter((column) => !headers.has(column));
  if (missing.length > 0) {
    failVerification(`frames.csv lacks audit columns: ${missing.join(', ')}.`);
  }
  if (parsed.records.length !== metadata.frameRowCount) {
    failVerification('frames.csv row count does not equal metadata frameRowCount.');
  }
  const expectedFrameRows = metadata.expectedTrialCount * metadata.protocol.measuredFrames;
  if (metadata.frameRowCount !== expectedFrameRows) {
    failVerification('metadata frameRowCount does not equal trials multiplied by measured frames.');
  }

  const frameIndexesByTrial = new Map();
  for (const [index, record] of parsed.records.entries()) {
    const label = `frames.csv record ${index + 2}`;
    if (record.runId !== metadata.runId) failVerification(`${label} has the wrong runId.`);
    const planned = planIndex.byTrialId.get(record.trialId);
    if (planned === undefined || !summariesByTrialId.has(record.trialId)) {
      failVerification(`${label} has an unknown trialId.`);
    }
    if (exactCsvInteger(record.planIndex, `${label} planIndex`) !== planned.planIndex
      || exactCsvInteger(record.repetitionIndex, `${label} repetitionIndex`) !== planned.repetitionIndex
      || record.modeId !== planned.modeId
      || exactCsvNumber(record.targetVisibilityFraction, `${label} targetVisibilityFraction`)
        !== planned.visibilityFraction
      || exactCsvInteger(record.objectCount, `${label} objectCount`) !== metadata.protocol.objectCount
      || exactCsvInteger(record.bucketCount, `${label} bucketCount`) !== metadata.protocol.bucketCount) {
      failVerification(`${label} does not match its planned trial.`);
    }
    if (metadata.protocol.matrixKind === FIXED_SLICE_REPRESENTATION_MATRIX
      && (exactCsvInteger(record.modeOrderPosition, `${label} modeOrderPosition`)
          !== planned.modeOrderPosition
        || exactCsvInteger(record.visibilityOrderPosition, `${label} visibilityOrderPosition`)
          !== planned.visibilityOrderPosition
        || record.plannedModeOrder !== planned.modeOrder.join('|')
        || record.plannedVisibilityOrder !== planned.visibilityOrder.join('|'))) {
      failVerification(`${label} order audit fields do not match its planned trial.`);
    }
    if (record.validationPass !== 'true' || record.timestampAvailable !== 'true') {
      failVerification(`${label} lacks accepted validation or GPU timestamps.`);
    }
    const frameIndex = exactCsvInteger(record.frameIndex, `${label} frameIndex`);
    let indexes = frameIndexesByTrial.get(record.trialId);
    if (indexes === undefined) {
      indexes = new Set();
      frameIndexesByTrial.set(record.trialId, indexes);
    }
    if (indexes.has(frameIndex)) failVerification(`${label} duplicates a frame index.`);
    indexes.add(frameIndex);
  }
  for (const trialId of planIndex.byTrialId.keys()) {
    const indexes = frameIndexesByTrial.get(trialId);
    if (indexes?.size !== metadata.protocol.measuredFrames) {
      failVerification(`frames.csv has an incomplete row set for trial ${JSON.stringify(trialId)}.`);
    }
    for (let index = 0; index < metadata.protocol.measuredFrames; index += 1) {
      if (!indexes.has(index)) {
        failVerification(`frames.csv trial ${JSON.stringify(trialId)} has non-contiguous frame indexes.`);
      }
    }
  }
}

export async function verifyRunDirectory(runDirectory) {
  const absoluteDirectory = path.resolve(runDirectory);
  const inputStat = await stat(absoluteDirectory);
  if (!inputStat.isDirectory()) failVerification('run-directory input is not a directory.');
  const { manifest, contentsByName } = await loadVerifiedArtifactContents(absoluteDirectory);
  const metadata = parseJsonArtifact(contentsByName.get('metadata.json'), 'metadata.json');
  const trialSummaries = parseJsonArtifact(
    contentsByName.get('trial-summaries.json'),
    'trial-summaries.json',
  );
  const validationArtifacts = parseJsonArtifact(
    contentsByName.get('validation-artifacts.json'),
    'validation-artifacts.json',
  );
  const workloadManifests = parseJsonArtifact(
    contentsByName.get('workload-manifests.json'),
    'workload-manifests.json',
  );
  const telemetrySummary = requireRecord(
    parseJsonArtifact(
      contentsByName.get('gpu-telemetry-summary.json'),
      'gpu-telemetry-summary.json',
    ),
    'gpu-telemetry-summary.json',
  );
  const { plan } = requireMetadataCompleteness(metadata, manifest);
  const planIndex = validateBenchmarkPlan(plan, metadata);
  const summariesByTrialId = validateTrialSummaries(
    trialSummaries,
    metadata,
    planIndex,
  );
  const workloadCatalog = validateWorkloadManifests(workloadManifests, metadata);
  validateValidationArtifacts(
    validationArtifacts,
    metadata,
    planIndex,
    summariesByTrialId,
    workloadCatalog,
  );
  if (JSON.stringify(metadata.environment.gpuTelemetry) !== JSON.stringify(telemetrySummary)) {
    failVerification('metadata GPU telemetry summary differs from gpu-telemetry-summary.json.');
  }
  const csvText = contentsByName.get('frames.csv').toString('utf8');
  const parsed = parseCsv(csvText);
  parseFrameRecords(parsed);
  validateVerifiedFrames(parsed, metadata, planIndex, summariesByTrialId);

  return {
    csvText,
    artifactVerification: {
      status: 'consistent',
      scope: 'artifact-integrity-and-schema-only',
      authenticityVerified: false,
      inputKind: 'run-directory',
      runId: metadata.runId,
      evidenceStatus: metadata.evidenceStatus,
      manifestSchemaVersion: manifest.schemaVersion,
      verifiedArtifactCount: [...contentsByName.keys()].length,
      requiredArtifactCount: REQUIRED_RUN_ARTIFACTS.length,
      completedTrialCount: metadata.completedTrialCount,
      acceptedTrialCount: metadata.acceptedTrialCount,
      sourceProvenanceStable: metadata.sourceProvenance.stable,
    },
  };
}

export function summarizeCsv(text) {
  const parsed = parseCsv(text);
  const { frames, repetitionColumn } = parseFrameRecords(parsed);
  const groups = groupFrames(frames);
  return {
    schemaVersion: 2,
    artifactVerification: {
      status: 'unverified',
      scope: 'artifact-integrity-and-schema-only',
      authenticityVerified: false,
      inputKind: 'raw-csv-content',
      evidenceStatus: null,
      reason: 'Raw CSV is not bound to a consistent run artifact manifest.',
    },
    percentileMethod: {
      frameP50: 'nearest-rank',
      acrossTrials: 'arithmetic midpoint for even sample counts',
    },
    repetitionColumn: repetitionColumn ?? '(implicit single trial)',
    deltaConvention: 'left mode minus right mode; negative values mean the left mode is faster',
    nFrames: frames.length,
    groups,
    comparisons: {
      fixedSliceVsDrawAll: comparisonsAgainst(groups, 'draw-all'),
      fixedSliceVsThreeBlocksCurrent: comparisonsAgainst(groups, 'three-blocks-current'),
      fixedSliceVsThreeBlocksCoalesced: comparisonsAgainst(groups, 'three-blocks-coalesced'),
      fixedSliceVsThreeBlocksHistorical: comparisonsAgainst(groups, 'three-blocks-historical'),
    },
    causalContrasts: {
      mergedFixedSliceVsPerBucketRepresentation: pairedContrasts(
        groups,
        'fixed-slice',
        'fixed-slice-per-bucket',
      ),
    },
  };
}

export async function summarizeInput(inputPath) {
  const absoluteInput = path.resolve(inputPath);
  const inputStat = await stat(absoluteInput);
  if (inputStat.isDirectory()) {
    const verified = await verifyRunDirectory(absoluteInput);
    return {
      ...summarizeCsv(verified.csvText),
      artifactVerification: verified.artifactVerification,
    };
  }
  if (!inputStat.isFile()) {
    throw new Error('Analyzer input must be a run directory or a raw frames.csv file.');
  }
  const csv = await readFile(absoluteInput, 'utf8');
  return {
    ...summarizeCsv(csv),
    artifactVerification: {
      status: 'unverified',
      scope: 'artifact-integrity-and-schema-only',
      authenticityVerified: false,
      inputKind: 'raw-frames-csv',
      evidenceStatus: null,
      reason: 'Standalone CSV input is not bound to a consistent run artifact manifest.',
    },
  };
}

async function main() {
  const arguments_ = process.argv.slice(2);
  if (arguments_.length !== 1) {
    throw new Error('Usage: node analysis/summarize.mjs <run-directory-or-frames.csv>');
  }
  const summary = await summarizeInput(arguments_[0]);
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

const invokedUrl = process.argv[1] === undefined
  ? null
  : pathToFileURL(path.resolve(process.argv[1])).href;

if (import.meta.url === invokedUrl) {
  main().catch((error) => {
    process.stderr.write(`summarize: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
