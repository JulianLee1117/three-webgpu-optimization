import assert from 'node:assert/strict';
import test from 'node:test';

import {
  FIRST_INSTANCE_LIVE_STANDALONE_MODE,
  validateStandaloneScenarioTransition,
} from '../src/strategies/live-first-instance-standalone.js';

function scenario(visibilityFraction = 0.99) {
  return {
    objectCount: 4,
    bucketCount: 2,
    layout: 'baseline',
    visibilityFraction,
    objectBuckets: new Uint32Array([0, 0, 1, 1]),
    bucketBases: new Uint32Array([0, 2]),
    bucketCounts: new Uint32Array([2, 2]),
    cullOrder: new Uint32Array([0, 1, 2, 3]),
    matrices: new Float32Array(4 * 16),
    bounds: new Float32Array(4 * 4),
    expectedVisibleIds: visibilityFraction === 0.99
      ? new Uint32Array([0, 1, 2, 3])
      : new Uint32Array([0]),
    expectedVisibleCount: visibilityFraction === 0.99 ? 4 : 1,
  };
}

test('standalone strategy mode is a distinct public harness identifier', () => {
  assert.equal(
    FIRST_INSTANCE_LIVE_STANDALONE_MODE,
    'first-instance-live-standalone-deployment',
  );
});

test('standalone visibility transition permits only mutable matrices and bounds', () => {
  const high = scenario(0.99);
  const low = scenario(0.2);
  low.matrices[0] = 7;
  low.bounds[0] = -3;
  assert.equal(validateStandaloneScenarioTransition(high, low), true);
});

test('standalone visibility transition rejects every immutable workload change', () => {
  for (const mutate of [
    (value) => { value.objectCount = 5; },
    (value) => { value.bucketCount = 3; },
    (value) => { value.layout = 'high-overlap'; },
    (value) => { value.objectBuckets[0] = 1; },
    (value) => { value.bucketBases[1] = 1; },
    (value) => { value.bucketCounts[0] = 1; },
    (value) => { value.cullOrder[0] = 3; },
  ]) {
    const high = scenario(0.99);
    const low = scenario(0.2);
    mutate(low);
    assert.throws(
      () => validateStandaloneScenarioTransition(high, low),
      /immutable workload fields/,
    );
  }
});

test('standalone visibility transition rejects malformed mutable payloads', () => {
  const high = scenario(0.99);
  const malformed = [
    { field: 'matrices', value: new Float32Array(3) },
    { field: 'bounds', value: new Float32Array(3) },
    { field: 'expectedVisibleIds', value: [0] },
    { field: 'expectedVisibleCount', value: 2 },
    { field: 'visibilityFraction', value: 0.8 },
  ];
  for (const { field, value } of malformed) {
    const low = scenario(0.2);
    low[field] = value;
    assert.throws(
      () => validateStandaloneScenarioTransition(high, low),
      /payload is malformed/,
    );
  }
});
