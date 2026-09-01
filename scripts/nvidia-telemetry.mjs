import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import process from 'node:process';
import readline from 'node:readline';
import { setTimeout as delay } from 'node:timers/promises';

export const NVIDIA_QUERY_FIELDS = Object.freeze([
  'index',
  'name',
  'uuid',
  'pstate',
  'clocks.current.graphics',
  'clocks.current.memory',
  'utilization.gpu',
  'utilization.memory',
  'memory.used',
  'memory.total',
  'temperature.gpu',
  'power.draw',
]);

export const NVIDIA_TELEMETRY_INTERVAL_MS = 250;
export const NVIDIA_TELEMETRY_LIVENESS_TOLERANCE_MULTIPLIER = 8;

export const TELEMETRY_CSV_FIELDS = Object.freeze([
  'observedAtIso',
  'runElapsedMs',
  'runId',
  'trialId',
  'planIndex',
  'repetitionIndex',
  'modeId',
  'visibilityFraction',
  'layout',
  'phase',
  'gpuIndex',
  'gpuName',
  'gpuUuid',
  'pstate',
  'graphicsClockMHz',
  'memoryClockMHz',
  'gpuUtilizationPercent',
  'memoryUtilizationPercent',
  'memoryUsedMiB',
  'memoryTotalMiB',
  'temperatureC',
  'powerDrawW',
]);

const NULL_VALUES = new Set(['', 'N/A', '[N/A]', 'Not Supported', 'Unknown Error']);
const COMPUTE_PROCESS_FIELDS = Object.freeze([
  'gpu_uuid',
  'pid',
  'process_name',
  'used_gpu_memory',
]);

export function parseCsvRecord(line) {
  const fields = [];
  let field = '';
  let state = 'field-start';
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (state === 'quoted') {
      if (character === '"' && line[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        state = 'after-quote';
      } else {
        field += character;
      }
      continue;
    }
    if (state === 'after-quote') {
      if (character === ',') {
        fields.push(field.trim());
        field = '';
        state = 'field-start';
      } else if (!/\s/.test(character)) {
        return null;
      }
      continue;
    }
    if (character === ',') {
      fields.push(field.trim());
      field = '';
      state = 'field-start';
    } else if (character === '"') {
      if (state !== 'field-start' || field.trim() !== '') return null;
      field = '';
      state = 'quoted';
    } else {
      field += character;
      if (!/\s/.test(character)) state = 'unquoted';
    }
  }
  if (state === 'quoted') return null;
  fields.push(field.trim());
  return fields;
}

function nullableText(value) {
  const normalized = value?.trim() ?? '';
  return NULL_VALUES.has(normalized) ? null : normalized;
}

function nullableNumber(value) {
  const normalized = nullableText(value);
  if (normalized === null) return null;
  const numeric = Number(normalized);
  return Number.isFinite(numeric) ? numeric : null;
}

function nullableInteger(value) {
  const numeric = nullableNumber(value);
  return Number.isSafeInteger(numeric) ? numeric : null;
}

export function parseNvidiaSmiLine(line) {
  const fields = parseCsvRecord(line.trim());
  if (!fields || fields.length !== NVIDIA_QUERY_FIELDS.length) return null;
  const [
    gpuIndex,
    gpuName,
    gpuUuid,
    pstate,
    graphicsClockMHz,
    memoryClockMHz,
    gpuUtilizationPercent,
    memoryUtilizationPercent,
    memoryUsedMiB,
    memoryTotalMiB,
    temperatureC,
    powerDrawW,
  ] = fields;
  const parsedIndex = nullableInteger(gpuIndex);
  const parsedUuid = nullableText(gpuUuid);
  if (parsedIndex === null && parsedUuid === null) return null;
  return {
    gpuIndex: parsedIndex,
    gpuName: nullableText(gpuName),
    gpuUuid: parsedUuid,
    pstate: nullableText(pstate),
    graphicsClockMHz: nullableNumber(graphicsClockMHz),
    memoryClockMHz: nullableNumber(memoryClockMHz),
    gpuUtilizationPercent: nullableNumber(gpuUtilizationPercent),
    memoryUtilizationPercent: nullableNumber(memoryUtilizationPercent),
    memoryUsedMiB: nullableNumber(memoryUsedMiB),
    memoryTotalMiB: nullableNumber(memoryTotalMiB),
    temperatureC: nullableNumber(temperatureC),
    powerDrawW: nullableNumber(powerDrawW),
  };
}

function portableBasename(value) {
  const normalized = nullableText(value);
  if (normalized === null) return null;
  return normalized.replaceAll('\\', '/').split('/').at(-1) || null;
}

function strictNullableNumber(value) {
  const normalized = value?.trim() ?? '';
  if (NULL_VALUES.has(normalized)) return { valid: true, value: null };
  const numeric = Number(normalized);
  return Number.isFinite(numeric)
    ? { valid: true, value: numeric }
    : { valid: false, value: null };
}

export function parseComputeProcessCsv(output) {
  const processes = [];
  let rawNonemptyLineCount = 0;
  let malformedLineCount = 0;
  for (const line of output.split(/\r?\n/)) {
    if (!line.trim()) continue;
    rawNonemptyLineCount += 1;
    const fields = parseCsvRecord(line);
    if (!fields || fields.length !== COMPUTE_PROCESS_FIELDS.length) {
      malformedLineCount += 1;
      continue;
    }
    const [gpuUuid, pid, processName, usedMemoryMiB] = fields;
    const parsedGpuUuid = nullableText(gpuUuid);
    const parsedPid = nullableInteger(pid);
    const parsedProcessName = portableBasename(processName);
    const parsedMemory = strictNullableNumber(usedMemoryMiB);
    if (parsedGpuUuid === null
      || parsedPid === null
      || parsedPid <= 0
      || parsedProcessName === null
      || !parsedMemory.valid
      || (parsedMemory.value !== null && parsedMemory.value < 0)) {
      malformedLineCount += 1;
      continue;
    }
    processes.push({
      gpuUuid: parsedGpuUuid,
      pid: parsedPid,
      processName: parsedProcessName,
      usedMemoryMiB: parsedMemory.value,
    });
  }
  return {
    processes,
    rawNonemptyLineCount,
    parsedRecordCount: processes.length,
    malformedLineCount,
  };
}

function computeProcessSnapshotDiagnosticsPass(snapshot) {
  return Number.isSafeInteger(snapshot?.rawNonemptyLineCount)
    && snapshot.rawNonemptyLineCount >= 0
    && Number.isSafeInteger(snapshot?.parsedRecordCount)
    && snapshot.parsedRecordCount >= 0
    && Number.isSafeInteger(snapshot?.malformedLineCount)
    && snapshot.malformedLineCount >= 0
    && Number.isSafeInteger(snapshot?.stdoutByteCount)
    && snapshot.stdoutByteCount >= 0
    && snapshot.stdoutByteCount <= 1_000_000
    && snapshot.stdoutTruncated === false
    && Number.isSafeInteger(snapshot?.stderrByteCount)
    && snapshot.stderrByteCount === 0
    && snapshot.malformedLineCount === 0
    && snapshot.parsedRecordCount + snapshot.malformedLineCount
      === snapshot.rawNonemptyLineCount
    && snapshot.parsedRecordCount === snapshot.rawNonemptyLineCount
    && snapshot.parsedRecordCount === snapshot.processes.length
    && (snapshot.parsedRecordCount === 0 || snapshot.stdoutByteCount > 0);
}

function computeProcessIdentityTuples(snapshot, label, reasons) {
  if (!snapshot
    || snapshot.status !== 'available'
    || !Array.isArray(snapshot.processes)
    || !computeProcessSnapshotDiagnosticsPass(snapshot)) {
    reasons.push(`${label} compute-process snapshot is unavailable`);
    return [];
  }
  const tuples = [];
  for (const [index, processRecord] of snapshot.processes.entries()) {
    const gpuUuid = processRecord?.gpuUuid;
    const pid = processRecord?.pid;
    const processName = portableBasename(processRecord?.processName ?? '');
    if (typeof gpuUuid !== 'string' || gpuUuid.length === 0
      || !Number.isSafeInteger(pid) || pid <= 0
      || typeof processName !== 'string' || processName.length === 0) {
      reasons.push(`${label} compute-process record ${index} has an invalid identity tuple`);
      continue;
    }
    tuples.push(Object.freeze({ gpuUuid, pid, processName }));
  }
  tuples.sort((left, right) => (
    left.gpuUuid.localeCompare(right.gpuUuid)
      || left.pid - right.pid
      || left.processName.localeCompare(right.processName)
  ));
  if (new Set(tuples.map((tuple) => JSON.stringify(tuple))).size !== tuples.length) {
    reasons.push(`${label} compute-process snapshot contains duplicate identity tuples`);
  }
  return tuples;
}

function telemetryGpuIdentity(row) {
  if (!Number.isSafeInteger(row?.gpuIndex)
    || row.gpuIndex < 0
    || typeof row?.gpuName !== 'string'
    || row.gpuName.trim() === ''
    || typeof row?.gpuUuid !== 'string'
    || row.gpuUuid.trim() === '') return null;
  return Object.freeze({
    gpuIndex: row.gpuIndex,
    gpuName: row.gpuName,
    gpuUuid: row.gpuUuid,
  });
}

function identityKey(identity) {
  return JSON.stringify([identity.gpuIndex, identity.gpuName, identity.gpuUuid]);
}

function sameSortedStrings(left, right) {
  return left.length === right.length
    && left.every((value, index) => value === right[index]);
}

export function createNvidiaTelemetryCoverageAudit(rows, {
  collectorStartedRunElapsedMs,
  collectorStopRequestedRunElapsedMs,
  requestedIntervalMs = NVIDIA_TELEMETRY_INTERVAL_MS,
} = {}) {
  const failureCodes = [];
  const reasons = [];
  const addFailure = (code, reason) => {
    if (!failureCodes.includes(code)) failureCodes.push(code);
    if (!reasons.includes(reason)) reasons.push(reason);
  };
  const intervalValid = Number.isFinite(requestedIntervalMs) && requestedIntervalMs > 0;
  const livenessToleranceMs = intervalValid
    ? requestedIntervalMs * NVIDIA_TELEMETRY_LIVENESS_TOLERANCE_MULTIPLIER
    : null;
  const sampleGroupingGapMs = intervalValid ? requestedIntervalMs / 2 : null;
  const boundsValid = Number.isFinite(collectorStartedRunElapsedMs)
    && collectorStartedRunElapsedMs >= 0
    && Number.isFinite(collectorStopRequestedRunElapsedMs)
    && collectorStopRequestedRunElapsedMs >= collectorStartedRunElapsedMs;
  if (!intervalValid || !boundsValid) {
    addFailure(
      'collector-bounds-invalid',
      'collector interval or active elapsed-time bounds are unavailable or invalid',
    );
  }

  const orderedRows = rows.map((row, sourceIndex) => ({ row, sourceIndex }))
    .sort((left, right) => (
      Number(left.row.runElapsedMs) - Number(right.row.runElapsedMs)
        || left.sourceIndex - right.sourceIndex
    ));
  if (orderedRows.length === 0) {
    addFailure('telemetry-no-samples', 'collector produced no valid GPU samples');
  }

  const cycles = [];
  let currentCycle = [];
  let currentCycleIdentityKeys = new Set();
  let precedingElapsedMs = null;
  const finishCurrentCycle = () => {
    if (currentCycle.length === 0) return;
    cycles.push(currentCycle);
    currentCycle = [];
    currentCycleIdentityKeys = new Set();
  };
  for (const entry of orderedRows) {
    const elapsedMs = entry.row.runElapsedMs;
    if (!Number.isFinite(elapsedMs) || elapsedMs < 0) {
      addFailure('telemetry-sample-time-invalid', 'a GPU sample has invalid elapsed time');
      continue;
    }
    if (boundsValid
      && (elapsedMs < collectorStartedRunElapsedMs
        || elapsedMs > collectorStopRequestedRunElapsedMs)) {
      addFailure(
        'telemetry-sample-outside-collector-bounds',
        'a GPU sample falls outside the recorded collector active bounds',
      );
    }
    const identity = telemetryGpuIdentity(entry.row);
    const key = identity === null ? null : identityKey(identity);
    const timingGapStartsCycle = currentCycle.length > 0
      && elapsedMs - precedingElapsedMs > sampleGroupingGapMs;
    // A loop iteration emits each GPU identity once, but separate iterations
    // can be drained from stdout together after the Node event loop was busy.
    const identityRecurrenceStartsCycle = currentCycle.length > 0
      && key !== null
      && currentCycleIdentityKeys.has(key);
    if (timingGapStartsCycle || identityRecurrenceStartsCycle) finishCurrentCycle();
    currentCycle.push(entry.row);
    if (key !== null) currentCycleIdentityKeys.add(key);
    precedingElapsedMs = elapsedMs;
  }
  finishCurrentCycle();

  const cycleIdentityKeys = [];
  const identityByKey = new Map();
  const identityKeyByIndex = new Map();
  const identityKeyByUuid = new Map();
  let identitiesValid = true;
  let identityMappingsValid = true;
  let duplicateIdentityInCycle = false;
  for (const cycle of cycles) {
    const keys = [];
    for (const row of cycle) {
      const identity = telemetryGpuIdentity(row);
      if (identity === null) {
        identitiesValid = false;
        continue;
      }
      const key = identityKey(identity);
      const existingIndexKey = identityKeyByIndex.get(identity.gpuIndex);
      const existingUuidKey = identityKeyByUuid.get(identity.gpuUuid);
      if ((existingIndexKey !== undefined && existingIndexKey !== key)
        || (existingUuidKey !== undefined && existingUuidKey !== key)) {
        identityMappingsValid = false;
      }
      identityKeyByIndex.set(identity.gpuIndex, key);
      identityKeyByUuid.set(identity.gpuUuid, key);
      identityByKey.set(key, identity);
      keys.push(key);
    }
    keys.sort();
    if (new Set(keys).size !== keys.length) duplicateIdentityInCycle = true;
    cycleIdentityKeys.push(keys);
  }
  if (!identitiesValid) {
    addFailure(
      'telemetry-gpu-identity-invalid',
      'a GPU sample lacks a concrete index, name, or UUID identity',
    );
  }
  if (!identityMappingsValid) {
    addFailure(
      'telemetry-gpu-identity-mapping-inconsistent',
      'a GPU index or UUID maps to more than one complete identity tuple',
    );
  }
  const expectedIdentityKeys = cycleIdentityKeys[0] ?? [];
  const constantGpuIdentitySet = identitiesValid
    && identityMappingsValid
    && !duplicateIdentityInCycle
    && expectedIdentityKeys.length > 0
    && cycleIdentityKeys.every((keys) => sameSortedStrings(keys, expectedIdentityKeys));
  if (!constantGpuIdentitySet) {
    addFailure(
      'telemetry-gpu-identity-set-changed',
      'per-sample GPU identity set is empty, duplicated, or changed during collection',
    );
  }

  let initialMaximumGapMs = null;
  let internalMaximumGapMs = null;
  let finalMaximumGapMs = null;
  if (boundsValid && expectedIdentityKeys.length > 0) {
    const rowsByIdentity = new Map(expectedIdentityKeys.map((key) => [key, []]));
    for (const { row } of orderedRows) {
      const identity = telemetryGpuIdentity(row);
      if (identity === null) continue;
      const key = identityKey(identity);
      if (!rowsByIdentity.has(key)) rowsByIdentity.set(key, []);
      rowsByIdentity.get(key).push(row.runElapsedMs);
    }
    for (const key of expectedIdentityKeys) {
      const timestamps = rowsByIdentity.get(key) ?? [];
      if (timestamps.length === 0) continue;
      initialMaximumGapMs = Math.max(
        initialMaximumGapMs ?? 0,
        timestamps[0] - collectorStartedRunElapsedMs,
      );
      finalMaximumGapMs = Math.max(
        finalMaximumGapMs ?? 0,
        Math.max(0, collectorStopRequestedRunElapsedMs - timestamps.at(-1)),
      );
      for (let index = 1; index < timestamps.length; index += 1) {
        internalMaximumGapMs = Math.max(
          internalMaximumGapMs ?? 0,
          timestamps[index] - timestamps[index - 1],
        );
      }
    }
    if ((initialMaximumGapMs ?? Infinity) > livenessToleranceMs) {
      addFailure(
        'telemetry-initial-gap-exceeded',
        `initial GPU sample gap exceeds ${livenessToleranceMs} ms`,
      );
    }
    if ((internalMaximumGapMs ?? 0) > livenessToleranceMs) {
      addFailure(
        'telemetry-internal-gap-exceeded',
        `internal GPU sample gap exceeds ${livenessToleranceMs} ms`,
      );
    }
    if ((finalMaximumGapMs ?? Infinity) > livenessToleranceMs) {
      addFailure(
        'telemetry-final-gap-exceeded',
        `final GPU sample gap exceeds ${livenessToleranceMs} ms`,
      );
    }
  }

  return Object.freeze({
    schemaVersion: 1,
    kind: 'nvidia-telemetry-collector-coverage',
    requestedIntervalMs,
    livenessToleranceMultiplier: NVIDIA_TELEMETRY_LIVENESS_TOLERANCE_MULTIPLIER,
    livenessToleranceMs,
    sampleGroupingGapMs,
    collectorStartedRunElapsedMs: Number.isFinite(collectorStartedRunElapsedMs)
      ? collectorStartedRunElapsedMs
      : null,
    collectorStopRequestedRunElapsedMs: Number.isFinite(collectorStopRequestedRunElapsedMs)
      ? collectorStopRequestedRunElapsedMs
      : null,
    activeDurationMs: boundsValid
      ? collectorStopRequestedRunElapsedMs - collectorStartedRunElapsedMs
      : null,
    sampleCount: rows.length,
    sampleCycleCount: cycles.length,
    gpuIdentities: Object.freeze(expectedIdentityKeys.map((key) => identityByKey.get(key))),
    constantGpuIdentitySet,
    initialMaximumGapMs,
    internalMaximumGapMs,
    finalMaximumGapMs,
    pass: failureCodes.length === 0,
    failureCodes: Object.freeze(failureCodes),
    reasons: Object.freeze(reasons),
  });
}

export function compareComputeProcessIdentitySets(preSnapshot, postSnapshot) {
  const reasons = [];
  const pre = computeProcessIdentityTuples(preSnapshot, 'pre-run', reasons);
  const post = computeProcessIdentityTuples(postSnapshot, 'post-run', reasons);
  if (JSON.stringify(pre) !== JSON.stringify(post)) {
    reasons.push('pre-run and post-run compute-process identity sets differ');
  }
  return Object.freeze({
    schemaVersion: 1,
    kind: 'nvidia-compute-process-identity-set-comparison',
    identityFields: Object.freeze(['gpuUuid', 'pid', 'processName']),
    ignoredFields: Object.freeze(['usedMemoryMiB']),
    pass: reasons.length === 0,
    pre: Object.freeze(pre),
    post: Object.freeze(post),
    reasons: Object.freeze([...reasons]),
  });
}

function median(values) {
  const finite = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (finite.length === 0) return null;
  const midpoint = Math.floor(finite.length / 2);
  return finite.length % 2 === 0
    ? (finite[midpoint - 1] + finite[midpoint]) / 2
    : finite[midpoint];
}

function numericSummary(rows, field) {
  const values = rows.map((row) => row[field]).filter(Number.isFinite);
  if (values.length === 0) return null;
  return {
    minimum: Math.min(...values),
    median: median(values),
    maximum: Math.max(...values),
  };
}

function counts(rows, field) {
  return Object.fromEntries(
    [...rows.reduce((result, row) => {
      const value = row[field] ?? 'unavailable';
      result.set(value, (result.get(value) ?? 0) + 1);
      return result;
    }, new Map()).entries()].sort(([left], [right]) => String(left).localeCompare(String(right))),
  );
}

function maximumGapMs(rows) {
  const timestamps = rows.map((row) => row.runElapsedMs).filter(Number.isFinite).sort((a, b) => a - b);
  let maximum = null;
  for (let index = 1; index < timestamps.length; index += 1) {
    maximum = Math.max(maximum ?? 0, timestamps[index] - timestamps[index - 1]);
  }
  return maximum;
}

export function summarizeTelemetryRows(rows) {
  const groups = new Map();
  for (const row of rows) {
    const key = row.gpuUuid ?? `index:${row.gpuIndex ?? 'unknown'}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  const gpus = [...groups.values()].map((gpuRows) => ({
    gpuIndex: gpuRows[0].gpuIndex,
    gpuName: gpuRows[0].gpuName,
    gpuUuid: gpuRows[0].gpuUuid,
    sampleCount: gpuRows.length,
    firstObservedAtIso: gpuRows[0].observedAtIso,
    lastObservedAtIso: gpuRows.at(-1).observedAtIso,
    maximumSampleGapMs: maximumGapMs(gpuRows),
    phaseSampleCounts: counts(gpuRows, 'phase'),
    pstateSampleCounts: counts(gpuRows, 'pstate'),
    graphicsClockMHz: numericSummary(gpuRows, 'graphicsClockMHz'),
    memoryClockMHz: numericSummary(gpuRows, 'memoryClockMHz'),
    gpuUtilizationPercent: numericSummary(gpuRows, 'gpuUtilizationPercent'),
    memoryUtilizationPercent: numericSummary(gpuRows, 'memoryUtilizationPercent'),
    memoryUsedMiB: numericSummary(gpuRows, 'memoryUsedMiB'),
    memoryTotalMiB: numericSummary(gpuRows, 'memoryTotalMiB'),
    temperatureC: numericSummary(gpuRows, 'temperatureC'),
    powerDrawW: numericSummary(gpuRows, 'powerDrawW'),
  })).sort((left, right) => (left.gpuIndex ?? Number.MAX_SAFE_INTEGER)
    - (right.gpuIndex ?? Number.MAX_SAFE_INTEGER));

  return {
    sampleCount: rows.length,
    gpuCount: gpus.length,
    firstObservedAtIso: rows[0]?.observedAtIso ?? null,
    lastObservedAtIso: rows.at(-1)?.observedAtIso ?? null,
    gpus,
  };
}

function escapeCsv(value) {
  if (value === null || value === undefined) return '';
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function telemetryRowsToCsv(rows) {
  return [
    TELEMETRY_CSV_FIELDS.join(','),
    ...rows.map((row) => TELEMETRY_CSV_FIELDS.map((field) => escapeCsv(row[field])).join(',')),
  ].join('\n');
}

function commandFailureReason(error) {
  if (error?.code === 'ENOENT') return 'command-not-found';
  if (typeof error?.code === 'string' && /^[A-Z0-9_]+$/i.test(error.code)) {
    return `spawn-error-${error.code.toLowerCase()}`;
  }
  return 'spawn-error';
}

async function runQuery(command, arguments_, timeoutMs = 5_000) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(command, arguments_, {
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
    } catch (error) {
      resolve({
        ok: false,
        reason: commandFailureReason(error),
        stdout: '',
        stdoutByteCount: 0,
        stdoutTruncated: false,
        stderrByteCount: 0,
      });
      return;
    }

    const chunks = [];
    let retainedByteCount = 0;
    let stdoutByteCount = 0;
    let stderrByteCount = 0;
    let timedOut = false;
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGTERM');
    }, timeoutMs);
    timer.unref?.();

    child.stdout.on('data', (chunk) => {
      stdoutByteCount += chunk.length;
      if (retainedByteCount >= 1_000_000) return;
      const retained = chunk.subarray(0, Math.max(0, 1_000_000 - retainedByteCount));
      chunks.push(retained);
      retainedByteCount += retained.length;
    });
    child.stderr.on('data', (chunk) => {
      stderrByteCount += chunk.length;
    });
    child.on('error', (error) => finish({
      ok: false,
      reason: commandFailureReason(error),
      stdout: '',
      stdoutByteCount,
      stdoutTruncated: stdoutByteCount > retainedByteCount,
      stderrByteCount,
    }));
    child.on('close', (code) => finish({
      ok: code === 0 && !timedOut,
      reason: timedOut ? 'query-timeout' : code === 0 ? null : `query-exit-${code ?? 'unknown'}`,
      stdout: Buffer.concat(chunks).toString('utf8'),
      stdoutByteCount,
      stdoutTruncated: stdoutByteCount > retainedByteCount,
      stderrByteCount,
    }));
  });
}

export class NvidiaTelemetryRecorder {
  constructor({
    runId,
    runDirectory,
    runStartedMonotonic,
    getContext,
    command = process.env.BENCHMARK_NVIDIA_SMI_PATH ?? 'nvidia-smi',
    intervalMs = NVIDIA_TELEMETRY_INTERVAL_MS,
    monotonicNow,
    wallClockNow,
  }) {
    this.runId = runId;
    this.runDirectory = runDirectory;
    this.runStartedMonotonic = runStartedMonotonic;
    this.getContext = getContext;
    this.command = command;
    this.intervalMs = intervalMs;
    this.monotonicNow = monotonicNow ?? (() => performance.now());
    this.wallClockNow = wallClockNow ?? (() => new Date());
    this.outputFile = 'gpu-telemetry.csv';
    this.rows = [];
    this.malformedLineCount = 0;
    this.stderrByteCount = 0;
    this.status = 'not-started';
    this.reason = null;
    this.commandSpawned = false;
    this.stopRequested = false;
    this.child = null;
    this.lineReader = null;
    this.exit = null;
    this.exitHandler = null;
    this.collectorStartedRunElapsedMs = null;
    this.collectorStopRequestedRunElapsedMs = null;
  }

  async start() {
    if (this.status !== 'not-started') return this.status;
    this.status = 'starting';
    const arguments_ = [
      `--query-gpu=${NVIDIA_QUERY_FIELDS.join(',')}`,
      '--format=csv,noheader,nounits',
      `--loop-ms=${this.intervalMs}`,
    ];

    await new Promise((resolve) => {
      let settled = false;
      const settle = () => {
        if (settled) return;
        settled = true;
        resolve();
      };
      try {
        this.child = spawn(this.command, arguments_, {
          windowsHide: true,
          stdio: ['ignore', 'pipe', 'pipe'],
        });
      } catch (error) {
        this.status = 'unavailable';
        this.reason = commandFailureReason(error);
        settle();
        return;
      }

      this.lineReader = readline.createInterface({ input: this.child.stdout });
      this.lineReader.on('line', (line) => this.consumeLine(line));
      this.child.stderr.on('data', (chunk) => {
        this.stderrByteCount += chunk.length;
      });
      this.child.on('spawn', () => {
        this.commandSpawned = true;
        this.collectorStartedRunElapsedMs = this.monotonicNow() - this.runStartedMonotonic;
        this.status = 'active';
        settle();
      });
      this.child.on('error', (error) => {
        this.status = 'unavailable';
        this.reason = commandFailureReason(error);
        settle();
      });
      this.child.on('close', (code, signal) => {
        this.exit = { code, signal };
        if (!this.stopRequested) {
          this.status = this.rows.length > 0 ? 'interrupted' : 'unavailable';
          this.reason = `sampling-exit-${code ?? signal ?? 'unknown'}`;
        }
        settle();
      });
    });

    if (this.child) {
      this.exitHandler = () => {
        if (this.child?.exitCode === null && this.child?.signalCode === null) {
          this.child.kill('SIGTERM');
        }
      };
      process.once('exit', this.exitHandler);
    }
    return this.status;
  }

  consumeLine(line) {
    if (this.stopRequested) return;
    const sample = parseNvidiaSmiLine(line);
    if (!sample) {
      if (line.trim()) this.malformedLineCount += 1;
      return;
    }
    const context = this.getContext?.() ?? {};
    this.rows.push({
      observedAtIso: this.wallClockNow().toISOString(),
      runElapsedMs: this.monotonicNow() - this.runStartedMonotonic,
      runId: this.runId,
      trialId: context.trialId ?? null,
      planIndex: context.planIndex ?? null,
      repetitionIndex: context.repetitionIndex ?? null,
      modeId: context.modeId ?? null,
      visibilityFraction: context.visibilityFraction ?? null,
      layout: context.layout ?? null,
      phase: context.phase ?? 'unclassified',
      ...sample,
    });
  }

  async captureComputeSnapshot(label) {
    const capturedAtIso = this.wallClockNow().toISOString();
    const runElapsedMs = this.monotonicNow() - this.runStartedMonotonic;
    if (!this.commandSpawned) {
      return {
        label,
        status: 'unavailable',
        capturedAtIso,
        runElapsedMs,
        reason: this.reason ?? 'sampling-command-unavailable',
        rawNonemptyLineCount: 0,
        parsedRecordCount: 0,
        malformedLineCount: 0,
        stdoutByteCount: 0,
        stdoutTruncated: false,
        stderrByteCount: 0,
        processes: [],
      };
    }
    const result = await runQuery(this.command, [
      `--query-compute-apps=${COMPUTE_PROCESS_FIELDS.join(',')}`,
      '--format=csv,noheader,nounits',
    ]);
    const parsed = parseComputeProcessCsv(result.stdout);
    const available = result.ok
      && result.stdoutTruncated === false
      && result.stderrByteCount === 0
      && parsed.malformedLineCount === 0
      && parsed.parsedRecordCount === parsed.rawNonemptyLineCount;
    const reason = !result.ok
      ? result.reason
      : result.stdoutTruncated
        ? 'compute-process-output-truncated'
        : result.stderrByteCount !== 0
          ? 'compute-process-query-stderr'
          : parsed.malformedLineCount !== 0
            ? 'compute-process-output-malformed'
            : null;
    return {
      label,
      status: available ? 'available' : 'unavailable',
      capturedAtIso,
      runElapsedMs,
      reason,
      rawNonemptyLineCount: parsed.rawNonemptyLineCount,
      parsedRecordCount: parsed.parsedRecordCount,
      malformedLineCount: parsed.malformedLineCount,
      stdoutByteCount: result.stdoutByteCount,
      stdoutTruncated: result.stdoutTruncated,
      stderrByteCount: result.stderrByteCount,
      processes: available ? parsed.processes : [],
    };
  }

  async stop() {
    if (!this.stopRequested) {
      this.stopRequested = true;
      this.collectorStopRequestedRunElapsedMs = this.monotonicNow() - this.runStartedMonotonic;
      if (this.child?.exitCode === null && this.child?.signalCode === null) {
        const closed = once(this.child, 'close').catch(() => undefined);
        this.child.kill('SIGTERM');
        await Promise.race([closed, delay(2_000)]);
        if (this.child.exitCode === null && this.child.signalCode === null) {
          this.child.kill('SIGKILL');
          await Promise.race([closed, delay(500)]);
        }
      }
      this.lineReader?.close();
      if (this.exitHandler) process.removeListener('exit', this.exitHandler);
      if (this.rows.length > 0) {
        this.status = this.status === 'interrupted' ? 'interrupted' : 'available';
      } else {
        this.status = 'unavailable';
        this.reason ??= 'no-valid-samples';
      }
    }

    try {
      await writeFile(
        path.join(this.runDirectory, this.outputFile),
        `${telemetryRowsToCsv(this.rows)}\n`,
        'utf8',
      );
    } catch {
      this.status = this.rows.length > 0 ? 'recorded-not-written' : 'unavailable';
      this.reason = 'telemetry-output-write-failed';
    }
    return this.report();
  }

  report({ preComputeProcesses = null, postComputeProcesses = null } = {}) {
    const coverageAudit = createNvidiaTelemetryCoverageAudit(this.rows, {
      collectorStartedRunElapsedMs: this.collectorStartedRunElapsedMs,
      collectorStopRequestedRunElapsedMs: this.collectorStopRequestedRunElapsedMs,
      requestedIntervalMs: this.intervalMs,
    });
    return {
      provider: 'nvidia-smi',
      status: this.status,
      reason: this.reason,
      command: portableBasename(this.command),
      sampling: {
        processModel: 'one-long-lived-process',
        requestedIntervalMs: this.intervalMs,
        queryFields: [...NVIDIA_QUERY_FIELDS],
        outputFile: this.outputFile,
        collectorStartedRunElapsedMs: this.collectorStartedRunElapsedMs,
        collectorStopRequestedRunElapsedMs: this.collectorStopRequestedRunElapsedMs,
        malformedLineCount: this.malformedLineCount,
        stderrByteCount: this.stderrByteCount,
        exit: this.exit,
      },
      summary: summarizeTelemetryRows(this.rows),
      coverageAudit,
      computeProcesses: {
        pre: preComputeProcesses,
        post: postComputeProcesses,
      },
      acceptanceBoundary: {
        affectsTechnicalRunAcceptance: false,
        candidateEnvironmentReviewRequired: true,
        automaticPstateRejectionThreshold: null,
      },
    };
  }
}
