import assert from 'node:assert/strict';
import test from 'node:test';
import {
  compareComputeProcessIdentitySets,
  createNvidiaTelemetryCoverageAudit,
  parseComputeProcessCsv,
  parseNvidiaSmiLine,
  summarizeTelemetryRows,
  telemetryRowsToCsv,
} from '../scripts/nvidia-telemetry.mjs';

function availableComputeSnapshot(processes, overrides = {}) {
  return {
    status: 'available',
    rawNonemptyLineCount: processes.length,
    parsedRecordCount: processes.length,
    malformedLineCount: 0,
    stdoutByteCount: processes.length === 0 ? 0 : 1,
    stdoutTruncated: false,
    stderrByteCount: 0,
    processes,
    ...overrides,
  };
}

function coverageRow(runElapsedMs, {
  gpuIndex = 0,
  gpuName = 'Fixture GPU',
  gpuUuid = 'GPU-fixture',
  pstate = 'P8',
  gpuUtilizationPercent = 0,
} = {}) {
  return {
    runElapsedMs,
    gpuIndex,
    gpuName,
    gpuUuid,
    pstate,
    gpuUtilizationPercent,
  };
}

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

test('compute-process identity comparison ignores memory but binds GPU, PID, and basename', () => {
  const pre = availableComputeSnapshot([
      { gpuUuid: 'GPU-b', pid: 9, processName: 'python.exe', usedMemoryMiB: 100 },
      { gpuUuid: 'GPU-a', pid: 4, processName: 'chrome.exe', usedMemoryMiB: 200 },
  ]);
  const post = availableComputeSnapshot([
      { gpuUuid: 'GPU-a', pid: 4, processName: 'chrome.exe', usedMemoryMiB: 500 },
      { gpuUuid: 'GPU-b', pid: 9, processName: 'python.exe', usedMemoryMiB: 300 },
  ]);
  const equal = compareComputeProcessIdentitySets(pre, post);
  assert.equal(equal.pass, true);
  assert.deepEqual(equal.identityFields, ['gpuUuid', 'pid', 'processName']);
  const changedProcesses = [...post.processes, {
      gpuUuid: 'GPU-a', pid: 10, processName: 'worker.exe', usedMemoryMiB: 1,
  }];
  const changed = compareComputeProcessIdentitySets(
    pre,
    availableComputeSnapshot(changedProcesses),
  );
  assert.equal(changed.pass, false);
  assert.match(changed.reasons.join('; '), /identity sets differ/);
});

test('compute-process snapshots retain basenames but discard private paths', () => {
  const output = [
    'GPU-abc, 1234, "C:\\Users\\person\\tools\\python.exe", 8192',
    'GPU-def, 5678, /opt/workloads/render, [N/A]',
  ].join('\n');
  assert.deepEqual(parseComputeProcessCsv(output), {
    processes: [
      { gpuUuid: 'GPU-abc', pid: 1234, processName: 'python.exe', usedMemoryMiB: 8192 },
      { gpuUuid: 'GPU-def', pid: 5678, processName: 'render', usedMemoryMiB: null },
    ],
    rawNonemptyLineCount: 2,
    parsedRecordCount: 2,
    malformedLineCount: 0,
  });
});

test('compute-process parser distinguishes a valid empty set from malformed output', () => {
  assert.deepEqual(parseComputeProcessCsv('\r\n  \n'), {
    processes: [],
    rawNonemptyLineCount: 0,
    parsedRecordCount: 0,
    malformedLineCount: 0,
  });
  const parsed = parseComputeProcessCsv([
    'GPU-a, 10, chrome.exe, 100',
    'GPU-b, not-a-pid, worker.exe, 20',
    'GPU-c, 12, "unterminated, 30',
    ', 13, missing-gpu.exe, 40',
    'GPU-d, 14, missing-memory.exe, invalid',
    'GPU-e, 0, zero-pid.exe, 50',
    'GPU-f, 15, bad"quote.exe, 60',
  ].join('\n'));
  assert.deepEqual(parsed, {
    processes: [
      { gpuUuid: 'GPU-a', pid: 10, processName: 'chrome.exe', usedMemoryMiB: 100 },
    ],
    rawNonemptyLineCount: 7,
    parsedRecordCount: 1,
    malformedLineCount: 6,
  });
});

test('compute-process identity comparison rejects malformed or truncated snapshot diagnostics', () => {
  const empty = availableComputeSnapshot([]);
  const malformed = availableComputeSnapshot([], {
    rawNonemptyLineCount: 1,
    malformedLineCount: 1,
  });
  const truncated = availableComputeSnapshot([], {
    stdoutByteCount: 1_048_577,
    stdoutTruncated: true,
  });
  assert.equal(compareComputeProcessIdentitySets(malformed, malformed).pass, false);
  assert.equal(compareComputeProcessIdentitySets(truncated, empty).pass, false);
});

test('telemetry coverage allows jitter within the frozen 2000 ms liveness tolerance', () => {
  const audit = createNvidiaTelemetryCoverageAudit([
    coverageRow(100),
    coverageRow(350),
    coverageRow(900),
  ], {
    collectorStartedRunElapsedMs: 0,
    collectorStopRequestedRunElapsedMs: 1_000,
    requestedIntervalMs: 250,
  });
  assert.equal(audit.pass, true);
  assert.equal(audit.livenessToleranceMs, 2_000);
  assert.equal(audit.initialMaximumGapMs, 100);
  assert.equal(audit.internalMaximumGapMs, 550);
  assert.equal(audit.finalMaximumGapMs, 100);
});

test('telemetry coverage rejects sparse collection and internal liveness gaps', () => {
  const delayedStart = createNvidiaTelemetryCoverageAudit([coverageRow(2_001)], {
    collectorStartedRunElapsedMs: 0,
    collectorStopRequestedRunElapsedMs: 2_100,
  });
  assert.equal(delayedStart.pass, false);
  assert.ok(delayedStart.failureCodes.includes('telemetry-initial-gap-exceeded'));

  const single = createNvidiaTelemetryCoverageAudit([coverageRow(100)], {
    collectorStartedRunElapsedMs: 0,
    collectorStopRequestedRunElapsedMs: 120_000,
  });
  assert.equal(single.pass, false);
  assert.ok(single.failureCodes.includes('telemetry-final-gap-exceeded'));

  const stalled = createNvidiaTelemetryCoverageAudit([
    coverageRow(100),
    coverageRow(2_101),
  ], {
    collectorStartedRunElapsedMs: 0,
    collectorStopRequestedRunElapsedMs: 2_200,
  });
  assert.equal(stalled.pass, false);
  assert.ok(stalled.failureCodes.includes('telemetry-internal-gap-exceeded'));
});

test('telemetry coverage rejects samples outside the recorded collector bounds', () => {
  const audit = createNvidiaTelemetryCoverageAudit([
    coverageRow(99),
    coverageRow(301),
  ], {
    collectorStartedRunElapsedMs: 100,
    collectorStopRequestedRunElapsedMs: 300,
  });
  assert.equal(audit.pass, false);
  assert.ok(audit.failureCodes.includes('telemetry-sample-outside-collector-bounds'));
});

test('telemetry coverage requires a constant GPU identity set per sampling cycle', () => {
  const rows = [
    coverageRow(100, { gpuIndex: 0, gpuName: 'GPU A', gpuUuid: 'GPU-a' }),
    coverageRow(101, { gpuIndex: 1, gpuName: 'GPU B', gpuUuid: 'GPU-b' }),
    coverageRow(350, { gpuIndex: 0, gpuName: 'GPU A', gpuUuid: 'GPU-a' }),
  ];
  const audit = createNvidiaTelemetryCoverageAudit(rows, {
    collectorStartedRunElapsedMs: 0,
    collectorStopRequestedRunElapsedMs: 400,
  });
  assert.equal(audit.pass, false);
  assert.equal(audit.constantGpuIdentitySet, false);
  assert.ok(audit.failureCodes.includes('telemetry-gpu-identity-set-changed'));
});

test('telemetry coverage treats buffered one-GPU identity recurrence as a new cycle', () => {
  const audit = createNvidiaTelemetryCoverageAudit([
    coverageRow(100),
    coverageRow(100.01),
    coverageRow(100.02),
  ], {
    collectorStartedRunElapsedMs: 0,
    collectorStopRequestedRunElapsedMs: 200,
  });
  assert.equal(audit.pass, true);
  assert.equal(audit.sampleCycleCount, 3);
  assert.equal(audit.constantGpuIdentitySet, true);
  assert.deepEqual(audit.failureCodes, []);
});

test('telemetry coverage reconstructs buffered multi-GPU cycles by identity recurrence', () => {
  const gpuA = { gpuIndex: 0, gpuName: 'GPU A', gpuUuid: 'GPU-a' };
  const gpuB = { gpuIndex: 1, gpuName: 'GPU B', gpuUuid: 'GPU-b' };
  for (const [name, identities] of [
    ['stable order', [gpuA, gpuB, gpuA, gpuB]],
    ['reordered next cycle', [gpuA, gpuB, gpuB, gpuA]],
  ]) {
    const audit = createNvidiaTelemetryCoverageAudit(
      identities.map((identity, index) => coverageRow(100 + index * 0.01, identity)),
      {
        collectorStartedRunElapsedMs: 0,
        collectorStopRequestedRunElapsedMs: 200,
      },
    );
    assert.equal(audit.pass, true, name);
    assert.equal(audit.sampleCycleCount, 2, name);
    assert.equal(audit.constantGpuIdentitySet, true, name);
  }
});

test('telemetry coverage rejects a buffered cycle missing a GPU identity', () => {
  const gpuA = { gpuIndex: 0, gpuName: 'GPU A', gpuUuid: 'GPU-a' };
  const gpuB = { gpuIndex: 1, gpuName: 'GPU B', gpuUuid: 'GPU-b' };
  const audit = createNvidiaTelemetryCoverageAudit([
    coverageRow(100, gpuA),
    coverageRow(100.01, gpuB),
    coverageRow(100.02, gpuA),
  ], {
    collectorStartedRunElapsedMs: 0,
    collectorStopRequestedRunElapsedMs: 200,
  });
  assert.equal(audit.pass, false);
  assert.equal(audit.sampleCycleCount, 2);
  assert.equal(audit.constantGpuIdentitySet, false);
  assert.ok(audit.failureCodes.includes('telemetry-gpu-identity-set-changed'));
});

test('telemetry coverage rejects a buffered cycle with a changed GPU identity', () => {
  const gpuA = { gpuIndex: 0, gpuName: 'GPU A', gpuUuid: 'GPU-a' };
  const gpuB = { gpuIndex: 1, gpuName: 'GPU B', gpuUuid: 'GPU-b' };
  const changedGpuB = { gpuIndex: 1, gpuName: 'GPU B', gpuUuid: 'GPU-b-replaced' };
  const audit = createNvidiaTelemetryCoverageAudit([
    coverageRow(100, gpuA),
    coverageRow(100.01, gpuB),
    coverageRow(100.02, gpuA),
    coverageRow(100.03, changedGpuB),
  ], {
    collectorStartedRunElapsedMs: 0,
    collectorStopRequestedRunElapsedMs: 200,
  });
  assert.equal(audit.pass, false);
  assert.equal(audit.sampleCycleCount, 2);
  assert.equal(audit.constantGpuIdentitySet, false);
  assert.ok(audit.failureCodes.includes('telemetry-gpu-identity-set-changed'));
});

test('telemetry coverage rejects inconsistent UUID and index identity mappings', () => {
  const gpuA = { gpuIndex: 0, gpuName: 'GPU A', gpuUuid: 'GPU-shared' };
  const sameUuidDifferentTuple = {
    gpuIndex: 1,
    gpuName: 'GPU B',
    gpuUuid: 'GPU-shared',
  };
  const sameIndexDifferentTuple = {
    gpuIndex: 0,
    gpuName: 'GPU A',
    gpuUuid: 'GPU-replaced',
  };
  for (const [name, gpuB] of [
    ['shared UUID', sameUuidDifferentTuple],
    ['shared index', sameIndexDifferentTuple],
  ]) {
    const audit = createNvidiaTelemetryCoverageAudit(
      [gpuA, gpuB, gpuA, gpuB].map(
        (identity, index) => coverageRow(100 + index * 0.01, identity),
      ),
      {
        collectorStartedRunElapsedMs: 0,
        collectorStopRequestedRunElapsedMs: 200,
      },
    );
    assert.equal(audit.pass, false, name);
    assert.equal(audit.constantGpuIdentitySet, false, name);
    assert.ok(
      audit.failureCodes.includes('telemetry-gpu-identity-mapping-inconsistent'),
      name,
    );
  }
});

test('telemetry coverage fails closed for invalid identities and no samples', () => {
  const invalid = createNvidiaTelemetryCoverageAudit([
    coverageRow(100),
    coverageRow(100.01, { gpuName: null }),
    coverageRow(100.02),
  ], {
    collectorStartedRunElapsedMs: 0,
    collectorStopRequestedRunElapsedMs: 200,
  });
  assert.equal(invalid.pass, false);
  assert.ok(invalid.failureCodes.includes('telemetry-gpu-identity-invalid'));
  assert.ok(invalid.failureCodes.includes('telemetry-gpu-identity-set-changed'));

  const empty = createNvidiaTelemetryCoverageAudit([], {
    collectorStartedRunElapsedMs: 0,
    collectorStopRequestedRunElapsedMs: 200,
  });
  assert.equal(empty.pass, false);
  assert.equal(empty.sampleCycleCount, 0);
  assert.ok(empty.failureCodes.includes('telemetry-no-samples'));
  assert.ok(empty.failureCodes.includes('telemetry-gpu-identity-set-changed'));
});

test('telemetry coverage does not gate observed performance or power-state values', () => {
  const audit = createNvidiaTelemetryCoverageAudit([
    coverageRow(100, { pstate: 'P8', gpuUtilizationPercent: 0 }),
    coverageRow(350, { pstate: 'P0', gpuUtilizationPercent: 100 }),
  ], {
    collectorStartedRunElapsedMs: 0,
    collectorStopRequestedRunElapsedMs: 400,
  });
  assert.equal(audit.pass, true);
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
