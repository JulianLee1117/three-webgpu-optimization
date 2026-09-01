import assert from 'node:assert/strict';
import test from 'node:test';
import { TrialController } from '../src/benchmark/trial-controller.js';

function deferred() {
  let resolve;
  const promise = new Promise((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

function fakeRenderer() {
  const renderer = {
    backend: {
      trackTimestamp: false,
      timestampQueryPool: {
        render: { frames: [], timestamps: new Map() },
        compute: { frames: [], timestamps: new Map() },
      },
    },
    rejection: null,
    manualResolve: false,
    legacyComputeUids: false,
    resolveCounts: { render: 0, compute: 0 },
    durationOverrides: { render: new Map(), compute: new Map() },
    async resolveTimestampsAsync(type) {
      if (this.rejection !== null) throw this.rejection;
      if (this.manualResolve) return;
      const frame = this.resolveCounts[type];
      this.resolveCounts[type] += 1;
      const contextId = type === 'render' ? 1 : 2;
      const prefix = type === 'render' ? 'r' : 'c';
      const durationMs = this.durationOverrides[type].get(frame)
        ?? (type === 'render' ? 0.0005 : 0.0004);
      const pool = this.backend.timestampQueryPool[type];
      pool.frames = [frame];
      const uid = type === 'compute' && this.legacyComputeUids
        ? `r:0:undefined:f${frame}`
        : `${prefix}:1:${contextId}:f${frame}`;
      pool.timestamps.set(uid, durationMs);
    },
  };
  return renderer;
}

async function startedController(overrides = {}) {
  const renderer = fakeRenderer();
  const controller = new TrialController(renderer, {
    warmupFrames: 1,
    measuredFrames: 1,
    ...overrides,
  });
  await controller.start({ usesCompute: true, strictTimestampUidAttribution: true });
  return { controller, renderer };
}

test('initial timestamp failure enters a surfaced error state and can be retried', async () => {
  const renderer = fakeRenderer();
  const failure = new Error('initial timestamp reset failed');
  renderer.rejection = failure;
  const errors = [];
  const controller = new TrialController(renderer, {
    warmupFrames: 1,
    measuredFrames: 1,
    onError: (error) => errors.push(error),
  });

  await controller.start({ usesCompute: true });
  assert.equal(controller.phase, 'error');
  assert.equal(controller.error, failure);
  assert.deepEqual(errors, [failure]);
  assert.equal(renderer.backend.trackTimestamp, false);

  renderer.rejection = null;
  await controller.start({ usesCompute: true });
  assert.equal(controller.phase, 'warmup');
  assert.equal(controller.error, null);
  assert.equal(renderer.backend.trackTimestamp, true);
});

test('generic compute trials retain legacy r185 Array UID compatibility', async () => {
  const renderer = fakeRenderer();
  renderer.legacyComputeUids = true;
  const completed = deferred();
  const controller = new TrialController(renderer, {
    warmupFrames: 1,
    measuredFrames: 1,
    onComplete: (result) => completed.resolve(result),
  });
  await controller.start({
    usesCompute: true,
    expectedComputeTimestampUidCount: 1,
    expectedRenderTimestampUidCount: 1,
  });
  controller.recordFrame({ gpuFrameId: 1 });
  while (controller.phase !== 'measure') await new Promise((resolve) => setTimeout(resolve, 0));
  controller.recordFrame({ gpuFrameId: 2, cpuSubmitTotalMs: 0.2 });
  const result = await completed.promise;

  assert.equal(result.summary.accepted, true);
  assert.equal(result.summary.timestampPhases.measurement.strictUidGrammar, false);
  assert.deepEqual(JSON.parse(result.rows[0].gpuComputeTimestampUids), [
    'r:0:undefined:f2',
  ]);
  assert.equal(
    result.summary.timestampPhases.measurement.pools.compute.uidRecords[0].contextId,
    null,
  );
});

test('exact-attribution trials reject the legacy r185 Array compute UID', async () => {
  const renderer = fakeRenderer();
  renderer.legacyComputeUids = true;
  const controller = new TrialController(renderer, {
    warmupFrames: 1,
    measuredFrames: 1,
  });

  await controller.start({
    usesCompute: true,
    strictTimestampUidAttribution: true,
  });
  assert.equal(controller.phase, 'error');
  assert.match(controller.error.message, /Malformed Three\.js r185 compute timestamp UID/);
  assert.equal(renderer.backend.trackTimestamp, false);
});

test('per-start frame-count overrides expose an immutable active-frame descriptor', async () => {
  const renderer = fakeRenderer();
  const controller = new TrialController(renderer, {
    warmupFrames: 3,
    measuredFrames: 4,
  });

  assert.equal(controller.frameDescriptor, null);
  await controller.start(
    { usesCompute: false },
    { warmupFrames: 2, measuredFrames: 3 },
  );
  assert.deepEqual(controller.frameDescriptor, {
    phase: 'warmup',
    phaseFrameIndex: 0,
    phaseFrameCount: 2,
  });
  assert.equal(Object.isFrozen(controller.frameDescriptor), true);
  assert.throws(
    () => { controller.frameDescriptor.phaseFrameIndex = 99; },
    /read only|Cannot assign/,
  );

  controller.recordFrame({ gpuFrameId: 0 });
  assert.deepEqual(controller.frameDescriptor, {
    phase: 'warmup',
    phaseFrameIndex: 1,
    phaseFrameCount: 2,
  });
  controller.recordFrame({ gpuFrameId: 1 });
  while (controller.phase !== 'measure') await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(controller.frameDescriptor, {
    phase: 'measure',
    phaseFrameIndex: 0,
    phaseFrameCount: 3,
  });
  controller.recordFrame({ gpuFrameId: 2, cpuSubmitTotalMs: 0.1 });
  assert.deepEqual(controller.frameDescriptor, {
    phase: 'measure',
    phaseFrameIndex: 1,
    phaseFrameCount: 3,
  });
});

test('per-start frame counts fall back to constructor defaults on later trials', async () => {
  const renderer = fakeRenderer();
  const controller = new TrialController(renderer, {
    warmupFrames: 2,
    measuredFrames: 3,
  });

  await controller.start(
    { usesCompute: false },
    { warmupFrames: 1, measuredFrames: 1 },
  );
  assert.equal(controller.frameDescriptor.phaseFrameCount, 1);
  controller.fail(new Error('end override trial'));

  await controller.start({ usesCompute: false });
  assert.deepEqual(controller.frameDescriptor, {
    phase: 'warmup',
    phaseFrameIndex: 0,
    phaseFrameCount: 2,
  });
});

test('invalid per-start frame-count overrides fail before timestamp tracking starts', async () => {
  const renderer = fakeRenderer();
  const controller = new TrialController(renderer, {
    warmupFrames: 1,
    measuredFrames: 1,
  });

  await assert.rejects(
    controller.start({ usesCompute: false }, { measuredFrames: 0 }),
    /measuredFrames must be a positive integer/,
  );
  assert.equal(controller.phase, 'idle');
  assert.equal(renderer.backend.trackTimestamp, false);
});

test('warmup timestamp failure terminates the trial and reports the error', async () => {
  const reported = deferred();
  let completed = false;
  const statuses = [];
  const { controller, renderer } = await startedController({
    onStatus: (status) => statuses.push(status),
    onComplete: () => { completed = true; },
    onError: (error) => reported.resolve(error),
  });
  const failure = new Error('warmup timestamp map failed');
  renderer.rejection = failure;

  controller.recordFrame({ gpuFrameId: 1 });
  assert.equal(await reported.promise, failure);

  assert.equal(controller.phase, 'error');
  assert.equal(controller.active, false);
  assert.equal(controller.resolving, false);
  assert.equal(renderer.backend.trackTimestamp, false);
  assert.equal(completed, false);
  assert.match(statuses.at(-1), /Trial failed: warmup timestamp map failed/);
});

test('measurement timestamp failure terminates the trial without completing it', async () => {
  const measuring = deferred();
  const reported = deferred();
  let completed = false;
  const { controller, renderer } = await startedController({
    onStatus: (status) => {
      if (status.startsWith('Measuring')) measuring.resolve();
    },
    onComplete: () => { completed = true; },
    onError: (error) => reported.resolve(error),
  });

  controller.recordFrame({ gpuFrameId: 1 });
  await measuring.promise;
  const failure = new Error('measurement timestamp map failed');
  renderer.rejection = failure;
  controller.recordFrame({ gpuFrameId: 2, cpuSubmitTotalMs: 1 });
  assert.equal(await reported.promise, failure);

  assert.equal(controller.phase, 'error');
  assert.equal(renderer.backend.trackTimestamp, false);
  assert.equal(completed, false);
});

for (const invariantPass of [true, false]) {
  test(`completion invariant ${invariantPass ? 'accepts' : 'rejects'} an otherwise complete trial`, async () => {
    const completed = deferred();
    const contexts = [];
    const statuses = [];
    const { controller, renderer } = await startedController({
      validateCompletion: (context) => {
        contexts.push(context);
        return { pass: invariantPass, kind: 'test-invariant' };
      },
      onStatus: (status) => statuses.push(status),
      onComplete: (result) => completed.resolve(result),
    });

    controller.recordFrame({ gpuFrameId: 1 });
    while (controller.phase !== 'measure') await new Promise((resolve) => setTimeout(resolve, 0));
    controller.recordFrame({
      gpuFrameId: 2,
      cpuSubmitTotalMs: 0.2,
    });
    const result = await completed.promise;

    assert.equal(result.summary.accepted, invariantPass);
    assert.equal(result.summary.invalidRenderTimestampUidCountFrames, 0);
    assert.equal(result.summary.invalidComputeTimestampUidCountFrames, 0);
    assert.equal(result.rows[0].gpuRenderTimestampUidCount, 1);
    assert.equal(result.rows[0].gpuComputeTimestampUidCount, 1);
    assert.deepEqual(result.summary.completionInvariant, {
      pass: invariantPass,
      kind: 'test-invariant',
    });
    assert.equal(contexts.length, 1);
    assert.equal(renderer.backend.trackTimestamp, false);
    assert.match(
      statuses.at(-1),
      invariantPass ? /Trial timing complete/ : /failed completion invariant/,
    );
  });
}

test('expected compute timestamp UID count is exposed on rows and summary', async () => {
  const completed = deferred();
  const { controller } = await startedController({
    onComplete: (result) => completed.resolve(result),
  });
  controller.context.expectedComputeTimestampUidCount = 1;

  controller.recordFrame({ gpuFrameId: 1 });
  while (controller.phase !== 'measure') await new Promise((resolve) => setTimeout(resolve, 0));
  controller.recordFrame({ gpuFrameId: 2, cpuSubmitTotalMs: 0.2 });
  const result = await completed.promise;

  assert.equal(result.rows[0].expectedComputeTimestampUidCount, 1);
  assert.equal(result.rows[0].gpuComputeTimestampUidCount, 1);
  assert.equal(result.rows[0].gpuComputeMs, 0.0004);
  assert.equal(result.summary.expectedComputeTimestampUidCount, 1);
  assert.equal(result.summary.invalidComputeTimestampUidCountFrames, 0);
  assert.equal(result.summary.missingComputeFrames, 0);
  assert.equal(result.summary.accepted, true);
});

test('warmup and measurement expose exact phase-scoped UID-duration records', async () => {
  const completed = deferred();
  const { controller, renderer } = await startedController({
    onComplete: (result) => completed.resolve(result),
  });
  controller.context.expectedComputeTimestampUidCount = 1;
  controller.context.expectedRenderTimestampUidCount = 1;

  controller.recordFrame({ gpuFrameId: 1 });
  while (controller.phase !== 'measure') await new Promise((resolve) => setTimeout(resolve, 0));

  renderer.durationOverrides.render.set(2, 0.00051);
  renderer.durationOverrides.compute.set(2, 0.00041);
  controller.recordFrame({ gpuFrameId: 2, cpuSubmitTotalMs: 0.2 });
  const result = await completed.promise;

  assert.equal(result.timestampPhases, result.summary.timestampPhases);
  assert.deepEqual(result.timestampPhases.warmup.pools.compute.frames, [1]);
  assert.deepEqual(result.timestampPhases.warmup.pools.compute.uidRecords, [{
    uid: 'c:1:2:f1',
    type: 'compute',
    callIndex: 1,
    contextId: 2,
    frameId: 1,
    durationMs: 0.0004,
  }]);
  assert.deepEqual(result.timestampPhases.measurement.pools.render.frames, [2]);
  assert.deepEqual(result.timestampPhases.measurement.pools.render.uidRecords, [{
    uid: 'r:1:1:f2',
    type: 'render',
    callIndex: 1,
    contextId: 1,
    frameId: 2,
    durationMs: 0.00051,
  }]);
  assert.deepEqual(JSON.parse(result.rows[0].gpuComputeTimestampUids), ['c:1:2:f2']);
  assert.deepEqual(JSON.parse(result.rows[0].gpuRenderTimestampUids), ['r:1:1:f2']);
  assert.deepEqual(JSON.parse(result.rows[0].gpuComputeTimestampRecords), [{
    uid: 'c:1:2:f2',
    type: 'compute',
    callIndex: 1,
    contextId: 2,
    frameId: 2,
    durationMs: 0.00041,
  }]);
  assert.equal(result.summary.renderTimestampPoolQualityValid, true);
  assert.equal(result.summary.computeTimestampPoolQualityValid, true);
  assert.equal(result.summary.accepted, true);
});

test('warmup rows preserve caller lane history, join exact UIDs, and freeze after resolution', async () => {
  const completed = deferred();
  const { controller, renderer } = await startedController({
    warmupFrames: 2,
    onComplete: (result) => completed.resolve(result),
  });
  controller.context.expectedComputeTimestampUidCount = 1;
  controller.context.expectedRenderTimestampUidCount = 1;
  renderer.manualResolve = true;
  renderer.backend.timestampQueryPool.render.frames = [10, 11];
  renderer.backend.timestampQueryPool.render.timestamps = new Map([
    ['r:1:1:f10', 0.0005],
    ['r:1:1:f11', 0.00051],
  ]);
  renderer.backend.timestampQueryPool.compute.frames = [10, 11];
  renderer.backend.timestampQueryPool.compute.timestamps = new Map([
    ['c:1:2:f10', 0.0004],
    ['c:1:2:f11', 0.00041],
  ]);

  controller.recordFrame({
    gpuFrameId: 10,
    laneId: 'portable',
    previousPreviousLaneId: null,
    previousLaneId: null,
  });
  controller.recordFrame({
    gpuFrameId: 11,
    laneId: 'feature',
    previousPreviousLaneId: null,
    previousLaneId: 'portable',
  });
  while (controller.phase !== 'measure') await new Promise((resolve) => setTimeout(resolve, 0));

  renderer.backend.timestampQueryPool.render.frames = [12];
  renderer.backend.timestampQueryPool.render.timestamps.set('r:1:1:f12', 0.00052);
  renderer.backend.timestampQueryPool.compute.frames = [12];
  renderer.backend.timestampQueryPool.compute.timestamps.set('c:1:2:f12', 0.00042);
  controller.recordFrame({ gpuFrameId: 12, cpuSubmitTotalMs: 0.2 });
  const result = await completed.promise;

  assert.equal(result.warmupRows, controller.warmupRows);
  assert.equal(Object.isFrozen(result.warmupRows), true);
  assert.equal(Object.isFrozen(result.warmupRows[0]), true);
  assert.equal(Object.hasOwn(result.warmupRows[0], 'usesCompute'), false);
  assert.equal(result.warmupRows[0].phase, 'warmup');
  assert.equal(result.warmupRows[0].frameIndex, 0);
  assert.equal(result.warmupRows[0].warmupFrameIndex, 0);
  assert.equal(result.warmupRows[0].previousPreviousLaneId, null);
  assert.equal(result.warmupRows[0].previousLaneId, null);
  assert.equal(result.warmupRows[1].previousPreviousLaneId, null);
  assert.equal(result.warmupRows[1].previousLaneId, 'portable');
  assert.deepEqual(JSON.parse(result.warmupRows[0].gpuComputeTimestampUids), [
    'c:1:2:f10',
  ]);
  assert.deepEqual(JSON.parse(result.warmupRows[1].gpuRenderTimestampRecords), [{
    uid: 'r:1:1:f11',
    type: 'render',
    callIndex: 1,
    contextId: 1,
    frameId: 11,
    durationMs: 0.00051,
  }]);
  assert.equal(result.warmupRows[0].gpuComputeMs, 0.0004);
  assert.equal(result.warmupRows[1].gpuRenderMs, 0.00051);
});

test('zero compute duration rejects independently of fine positive render timestamps', async () => {
  const completed = deferred();
  const { controller, renderer } = await startedController({
    onComplete: (result) => completed.resolve(result),
  });
  controller.context.expectedComputeTimestampUidCount = 1;
  controller.context.expectedRenderTimestampUidCount = 1;

  renderer.durationOverrides.render.set(1, 0.0005);
  renderer.durationOverrides.compute.set(1, 0.0004);
  controller.recordFrame({ gpuFrameId: 1 });
  while (controller.phase !== 'measure') await new Promise((resolve) => setTimeout(resolve, 0));

  renderer.durationOverrides.render.set(2, 0.0005);
  renderer.durationOverrides.compute.set(2, 0);
  controller.recordFrame({ gpuFrameId: 2, cpuSubmitTotalMs: 0.2 });
  const result = await completed.promise;

  assert.equal(result.rows[0].gpuComputeTimestampUidCount, 1);
  assert.equal(result.rows[0].gpuComputeTimestampDurationValid, false);
  assert.equal(result.rows[0].gpuComputeMs, null);
  assert.equal(result.rows[0].gpuPassTotalMs, null);
  assert.equal(result.summary.invalidComputeTimestampDurationFrames, 1);
  assert.equal(result.summary.computeTimestampPoolQualityValid, false);
  assert.equal(result.summary.renderTimestampPoolQualityValid, true);
  assert.deepEqual(result.summary.timestampResolutions.compute, {
    quantumNs: null,
    classification: 'invalid',
    recordCount: 1,
    positiveDurationCount: 0,
    nonpositiveDurationCount: 1,
  });
  assert.equal(result.summary.quantumNs, null);
  assert.equal(result.summary.classification, 'invalid');
  assert.equal(result.summary.accepted, false);
});

for (const type of ['render', 'compute']) {
  for (const quantumNs of [5_000, 10_000]) {
    test(`${type} pool rejects an independently coarse ${quantumNs} ns resolution`, async () => {
      const completed = deferred();
      const { controller, renderer } = await startedController({
        onComplete: (result) => completed.resolve(result),
      });
      controller.context.expectedComputeTimestampUidCount = 1;
      controller.context.expectedRenderTimestampUidCount = 1;
      renderer.durationOverrides[type].set(2, quantumNs / 1e6);

      controller.recordFrame({ gpuFrameId: 1 });
      while (controller.phase !== 'measure') {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      controller.recordFrame({ gpuFrameId: 2, cpuSubmitTotalMs: 0.2 });
      const result = await completed.promise;

      assert.equal(result.summary[`${type}TimestampPoolQualityValid`], false);
      assert.equal(result.summary.timestampResolutions[type].quantumNs, quantumNs);
      assert.equal(
        result.summary.timestampResolutions[type].classification,
        quantumNs >= 10_000 ? 'quantized' : 'fine',
      );
      assert.equal(result.summary.accepted, false);
    });
  }
}

test('nonpositive warmup timestamps remain auditable and reject final acceptance', async () => {
  const completed = deferred();
  const { controller, renderer } = await startedController({
    onComplete: (result) => completed.resolve(result),
  });
  controller.context.expectedComputeTimestampUidCount = 1;
  controller.context.expectedRenderTimestampUidCount = 1;
  renderer.durationOverrides.compute.set(1, 0);

  controller.recordFrame({ gpuFrameId: 1 });
  while (controller.phase !== 'measure') await new Promise((resolve) => setTimeout(resolve, 0));
  controller.recordFrame({ gpuFrameId: 2, cpuSubmitTotalMs: 0.2 });
  const result = await completed.promise;

  assert.equal(result.warmupRows[0].gpuComputeTimestampDurationValid, false);
  assert.equal(result.warmupRows[0].gpuComputeMs, null);
  assert.equal(result.summary.invalidWarmupComputeTimestampDurationFrames, 1);
  assert.equal(result.summary.warmupComputeTimestampPoolQualityValid, false);
  assert.equal(result.summary.computeTimestampPoolQualityValid, true);
  assert.equal(result.summary.accepted, false);
});

test('multiple compute timestamp UIDs for one measured frame reject the trial', async () => {
  const completed = deferred();
  const { controller, renderer } = await startedController({
    onComplete: (result) => completed.resolve(result),
  });
  controller.context.expectedComputeTimestampUidCount = 1;
  renderer.backend.timestampQueryPool.compute.timestamps.set('c:2:3:f2', 0.3);

  controller.recordFrame({ gpuFrameId: 1 });
  while (controller.phase !== 'measure') await new Promise((resolve) => setTimeout(resolve, 0));
  controller.recordFrame({ gpuFrameId: 2, cpuSubmitTotalMs: 0.2 });
  const result = await completed.promise;

  assert.equal(result.rows[0].gpuComputeTimestampUidCount, 2);
  assert.equal(result.rows[0].gpuComputeMs, null);
  assert.equal(result.rows[0].gpuPassTotalMs, null);
  assert.equal(result.summary.invalidComputeTimestampUidCountFrames, 1);
  assert.equal(result.summary.missingComputeFrames, 1);
  assert.equal(result.summary.accepted, false);
});

test('multiple render timestamp UIDs for one measured frame reject the trial', async () => {
  const completed = deferred();
  const { controller, renderer } = await startedController({
    onComplete: (result) => completed.resolve(result),
  });
  controller.context.expectedRenderTimestampUidCount = 1;
  renderer.backend.timestampQueryPool.render.timestamps.set('r:2:4:f2', 0.3);

  controller.recordFrame({ gpuFrameId: 1 });
  while (controller.phase !== 'measure') await new Promise((resolve) => setTimeout(resolve, 0));
  controller.recordFrame({ gpuFrameId: 2, cpuSubmitTotalMs: 0.2 });
  const result = await completed.promise;

  assert.equal(result.rows[0].gpuRenderTimestampUidCount, 2);
  assert.equal(result.rows[0].gpuRenderMs, null);
  assert.equal(result.rows[0].gpuPassTotalMs, null);
  assert.equal(result.summary.invalidRenderTimestampUidCountFrames, 1);
  assert.equal(result.summary.missingRenderFrames, 1);
  assert.equal(result.summary.accepted, false);
});

for (const invalidResult of [null, undefined]) {
  test(`invalid ${String(invalidResult)} completion result fails closed`, async () => {
    const completed = deferred();
    const { controller } = await startedController({
      validateCompletion: () => invalidResult,
      onComplete: (result) => completed.resolve(result),
    });

    controller.recordFrame({ gpuFrameId: 1 });
    while (controller.phase !== 'measure') await new Promise((resolve) => setTimeout(resolve, 0));
    controller.recordFrame({ gpuFrameId: 2, cpuSubmitTotalMs: 0.2 });
    const result = await completed.promise;
    assert.equal(result.summary.accepted, false);
    assert.deepEqual(result.summary.completionInvariant, {
      pass: false,
      kind: 'invalid-completion-invariant-result',
    });
  });
}

test('throwing completion invariant enters the generic surfaced error state', async () => {
  const reported = deferred();
  const statuses = [];
  const failure = new Error('completion invariant crashed');
  const { controller, renderer } = await startedController({
    validateCompletion: () => { throw failure; },
    onStatus: (status) => statuses.push(status),
    onError: (error) => reported.resolve(error),
  });

  controller.recordFrame({ gpuFrameId: 1 });
  while (controller.phase !== 'measure') await new Promise((resolve) => setTimeout(resolve, 0));
  controller.recordFrame({ gpuFrameId: 2, cpuSubmitTotalMs: 0.2 });
  assert.equal(await reported.promise, failure);
  assert.equal(controller.phase, 'error');
  assert.equal(renderer.backend.trackTimestamp, false);
  assert.match(statuses.at(-1), /Trial failed: completion invariant crashed/);
});
