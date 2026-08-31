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
  return {
    backend: {
      trackTimestamp: false,
      timestampQueryPool: {
        render: { frames: [2], timestamps: new Map([['render:f2', 0.2]]) },
        compute: { frames: [2], timestamps: new Map([['compute:f2', 0.1]]) },
      },
    },
    rejection: null,
    async resolveTimestampsAsync() {
      if (this.rejection !== null) throw this.rejection;
    },
  };
}

async function startedController(overrides = {}) {
  const renderer = fakeRenderer();
  const controller = new TrialController(renderer, {
    warmupFrames: 1,
    measuredFrames: 1,
    ...overrides,
  });
  await controller.start({ usesCompute: true });
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
