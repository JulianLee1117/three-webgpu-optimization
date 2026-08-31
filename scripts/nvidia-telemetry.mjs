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
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (quoted && line[index + 1] === '"') {
        field += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === ',' && !quoted) {
      fields.push(field.trim());
      field = '';
    } else {
      field += character;
    }
  }
  if (quoted) return null;
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
  return Number.isInteger(numeric) ? numeric : null;
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

export function parseComputeProcessCsv(output) {
  const processes = [];
  for (const line of output.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const fields = parseCsvRecord(line);
    if (!fields || fields.length !== COMPUTE_PROCESS_FIELDS.length) continue;
    const [gpuUuid, pid, processName, usedMemoryMiB] = fields;
    processes.push({
      gpuUuid: nullableText(gpuUuid),
      pid: nullableInteger(pid),
      processName: portableBasename(processName),
      usedMemoryMiB: nullableNumber(usedMemoryMiB),
    });
  }
  return processes;
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
      resolve({ ok: false, reason: commandFailureReason(error), stdout: '' });
      return;
    }

    const chunks = [];
    let byteCount = 0;
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
      if (byteCount >= 1_000_000) return;
      const retained = chunk.subarray(0, Math.max(0, 1_000_000 - byteCount));
      chunks.push(retained);
      byteCount += retained.length;
    });
    child.stderr.resume();
    child.on('error', (error) => finish({
      ok: false,
      reason: commandFailureReason(error),
      stdout: '',
    }));
    child.on('close', (code) => finish({
      ok: code === 0 && !timedOut,
      reason: timedOut ? 'query-timeout' : code === 0 ? null : `query-exit-${code ?? 'unknown'}`,
      stdout: Buffer.concat(chunks).toString('utf8'),
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
    intervalMs = 250,
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
        processes: [],
      };
    }
    const result = await runQuery(this.command, [
      `--query-compute-apps=${COMPUTE_PROCESS_FIELDS.join(',')}`,
      '--format=csv,noheader,nounits',
    ]);
    return {
      label,
      status: result.ok ? 'available' : 'unavailable',
      capturedAtIso,
      runElapsedMs,
      reason: result.reason,
      processes: result.ok ? parseComputeProcessCsv(result.stdout) : [],
    };
  }

  async stop() {
    if (!this.stopRequested) {
      this.stopRequested = true;
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
        malformedLineCount: this.malformedLineCount,
        stderrByteCount: this.stderrByteCount,
        exit: this.exit,
      },
      summary: summarizeTelemetryRows(this.rows),
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
