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
    backend: { trackTimestamp: false },
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
  assert.match(statuses.at(-1), /Trial failed while resolving timestamps: warmup timestamp map failed/);
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
