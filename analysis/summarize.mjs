import { readFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

const REQUIRED_COLUMNS = Object.freeze([
  'modeId',
  'targetVisibilityFraction',
  'gpuPassTotalMs',
  'gpuComputeMs',
  'gpuRenderMs',
  'cpuCommonUpdateMs',
  'cpuFrameBodyMs',
  'cpuSubmitTotalMs',
]);

const METRICS = Object.freeze([
  'gpuPassTotalMs',
  'gpuComputeMs',
  'gpuRenderMs',
  'cpuCommonUpdateMs',
  'cpuFrameBodyMs',
  'cpuSubmitTotalMs',
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

    const repetition = repetitionColumn === null
      ? 1
      : normalizedRepetition(record[repetitionColumn], recordNumber);

    return {
      modeId,
      targetVisibilityFraction,
      repetition,
      usesCompute,
      gpuPassTotalMs: finiteNumber(
        record.gpuPassTotalMs,
        'gpuPassTotalMs',
        recordNumber,
        { minimum: 0 },
      ),
      gpuComputeMs,
      gpuRenderMs: finiteNumber(record.gpuRenderMs, 'gpuRenderMs', recordNumber, { minimum: 0 }),
      cpuCommonUpdateMs: finiteNumber(
        record.cpuCommonUpdateMs,
        'cpuCommonUpdateMs',
        recordNumber,
        { minimum: 0 },
      ),
      cpuFrameBodyMs: finiteNumber(
        record.cpuFrameBodyMs,
        'cpuFrameBodyMs',
        recordNumber,
        { minimum: 0 },
      ),
      cpuSubmitTotalMs: finiteNumber(
        record.cpuSubmitTotalMs,
        'cpuSubmitTotalMs',
        recordNumber,
        { minimum: 0 },
      ),
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
      trial = { repetition: frame.repetition, usesCompute: frame.usesCompute, frames: [] };
      group.trials.set(key, trial);
    } else if (trial.usesCompute !== frame.usesCompute) {
      throw new Error(
        `Mode ${frame.modeId}, visibility ${frame.targetVisibilityFraction}, repetition ${frame.repetition} mixes usesCompute values.`,
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

function comparisonsAgainst(groups, baselineModeId) {
  const fixedGroups = groups.filter((group) => group.modeId === 'fixed-slice');
  const baselineByVisibility = new Map(
    groups
      .filter((group) => group.modeId === baselineModeId)
      .map((group) => [group.targetVisibilityFraction, group]),
  );

  return fixedGroups
    .map((fixedGroup) => {
      const baselineGroup = baselineByVisibility.get(fixedGroup.targetVisibilityFraction);
      if (baselineGroup === undefined) return null;
      const baselineTrials = new Map(
        baselineGroup.perTrialP50.map((trial) => [repetitionKey(trial.repetition), trial]),
      );
      const fixedTrials = new Map(
        fixedGroup.perTrialP50.map((trial) => [repetitionKey(trial.repetition), trial]),
      );
      const pairs = fixedGroup.perTrialP50
        .map((fixedTrial) => {
          const baselineTrial = baselineTrials.get(repetitionKey(fixedTrial.repetition));
          if (baselineTrial === undefined) return null;
          return {
            repetition: fixedTrial.repetition,
            fixedSliceP50: fixedTrial.p50,
            baselineP50: baselineTrial.p50,
            delta: pairedDelta(fixedTrial.p50, baselineTrial.p50),
          };
        })
        .filter(Boolean);
      const unmatchedFixedSliceRepetitions = fixedGroup.perTrialP50
        .filter((trial) => !baselineTrials.has(repetitionKey(trial.repetition)))
        .map((trial) => trial.repetition);
      const unmatchedBaselineRepetitions = baselineGroup.perTrialP50
        .filter((trial) => !fixedTrials.has(repetitionKey(trial.repetition)))
        .map((trial) => trial.repetition);

      return {
        targetVisibilityFraction: fixedGroup.targetVisibilityFraction,
        baselineModeId,
        nPairs: pairs.length,
        pairs,
        medianPairedDelta: summarizePairedDeltas(pairs),
        unmatchedFixedSliceRepetitions,
        unmatchedBaselineRepetitions,
      };
    })
    .filter(Boolean)
    .sort((left, right) => left.targetVisibilityFraction - right.targetVisibilityFraction);
}

export function summarizeCsv(text) {
  const parsed = parseCsv(text);
  const { frames, repetitionColumn } = parseFrameRecords(parsed);
  const groups = groupFrames(frames);
  return {
    schemaVersion: 1,
    percentileMethod: {
      frameP50: 'nearest-rank',
      acrossTrials: 'arithmetic midpoint for even sample counts',
    },
    repetitionColumn: repetitionColumn ?? '(implicit single trial)',
    deltaConvention: 'fixed-slice minus baseline; negative values mean fixed-slice is faster',
    nFrames: frames.length,
    groups,
    comparisons: {
      fixedSliceVsDrawAll: comparisonsAgainst(groups, 'draw-all'),
      fixedSliceVsThreeBlocksCurrent: comparisonsAgainst(groups, 'three-blocks-current'),
      fixedSliceVsThreeBlocksCoalesced: comparisonsAgainst(groups, 'three-blocks-coalesced'),
      fixedSliceVsThreeBlocksHistorical: comparisonsAgainst(groups, 'three-blocks-historical'),
    },
  };
}

async function main() {
  const arguments_ = process.argv.slice(2);
  if (arguments_.length !== 1) {
    throw new Error('Usage: node analysis/summarize.mjs <frame-level.csv>');
  }
  const csv = await readFile(arguments_[0], 'utf8');
  const summary = summarizeCsv(csv);
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
