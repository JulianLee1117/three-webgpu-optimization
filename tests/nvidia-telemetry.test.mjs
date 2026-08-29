import assert from 'node:assert/strict';
import test from 'node:test';
import {
  parseComputeProcessCsv,
  parseNvidiaSmiLine,
  summarizeTelemetryRows,
  telemetryRowsToCsv,
} from '../scripts/nvidia-telemetry.mjs';

test('NVIDIA telemetry parsing supports quoted GPU names and unavailable values', () => {
  const sample = parseNvidiaSmiLine(
    '0, "NVIDIA Prototype, 24 GB", GPU-abc, P2, 2535, 10501, 87, 42, 9441, 16303, 63, [N/A]',
  );
  assert.deepEqual(sample, {
    gpuIndex: 0,
    gpuName: 'NVIDIA Prototype, 24 GB',
    gpuUuid: 'GPU-abc',
    pstate: 'P2',
    graphicsClockMHz: 2535,
    memoryClockMHz: 10501,
    gpuUtilizationPercent: 87,
    memoryUtilizationPercent: 42,
    memoryUsedMiB: 9441,
    memoryTotalMiB: 16303,
    temperatureC: 63,
    powerDrawW: null,
  });
  assert.equal(parseNvidiaSmiLine('malformed event line'), null);
});

test('compute-process snapshots retain basenames but discard private paths', () => {
  const output = [
    'GPU-abc, 1234, "C:\\Users\\person\\tools\\python.exe", 8192',
    'GPU-def, 5678, /opt/workloads/render, [N/A]',
  ].join('\n');
  assert.deepEqual(parseComputeProcessCsv(output), [
    { gpuUuid: 'GPU-abc', pid: 1234, processName: 'python.exe', usedMemoryMiB: 8192 },
    { gpuUuid: 'GPU-def', pid: 5678, processName: 'render', usedMemoryMiB: null },
  ]);
});

test('telemetry summary reports per-GPU ranges, medians, states, phases, and gaps', () => {
  const common = {
    gpuIndex: 0,
    gpuName: 'GPU A',
    gpuUuid: 'GPU-a',
    memoryClockMHz: 1000,
    gpuUtilizationPercent: 50,
    memoryUtilizationPercent: 10,
    memoryUsedMiB: 200,
    memoryTotalMiB: 1000,
    temperatureC: 60,
    powerDrawW: 100,
  };
  const rows = [
    {
      ...common,
      observedAtIso: '2026-08-29T20:00:00.000Z',
      runElapsedMs: 0,
      phase: 'warmup',
      pstate: 'P8',
      graphicsClockMHz: 300,
    },
    {
      ...common,
      observedAtIso: '2026-08-29T20:00:00.250Z',
      runElapsedMs: 250,
      phase: 'measurement',
      pstate: 'P2',
      graphicsClockMHz: 1800,
      gpuUtilizationPercent: 90,
    },
    {
      ...common,
      observedAtIso: '2026-08-29T20:00:00.500Z',
      runElapsedMs: 500,
      phase: 'measurement',
      pstate: 'P2',
      graphicsClockMHz: 2100,
      gpuUtilizationPercent: 100,
    },
  ];

  const summary = summarizeTelemetryRows(rows);
  assert.equal(summary.sampleCount, 3);
  assert.equal(summary.gpuCount, 1);
  assert.deepEqual(summary.gpus[0].graphicsClockMHz, {
    minimum: 300,
    median: 1800,
    maximum: 2100,
  });
  assert.deepEqual(summary.gpus[0].gpuUtilizationPercent, {
    minimum: 50,
    median: 90,
    maximum: 100,
  });
  assert.deepEqual(summary.gpus[0].pstateSampleCounts, { P2: 2, P8: 1 });
  assert.deepEqual(summary.gpus[0].phaseSampleCounts, { measurement: 2, warmup: 1 });
  assert.equal(summary.gpus[0].maximumSampleGapMs, 250);
});

test('telemetry CSV uses stable fields and escapes contextual values', () => {
  const csv = telemetryRowsToCsv([{
    observedAtIso: '2026-08-29T20:00:00.000Z',
    runElapsedMs: 1.5,
    runId: 'run-1',
    phase: 'measurement',
    gpuIndex: 0,
    gpuName: 'GPU, A',
    gpuUuid: 'GPU-a',
  }]);
  const [header, row] = csv.split('\n');
  assert.match(header, /^observedAtIso,runElapsedMs,runId,trialId,/);
  assert.match(row, /,"GPU, A",GPU-a,/);
});
