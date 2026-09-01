import assert from 'node:assert/strict';
import test from 'node:test';
import {
  createFirstInstanceAddressChallengeExpected,
} from '../src/validation/first-instance-address-challenge.js';

const scenario = Object.freeze({
  objectCount: 5,
  bucketCount: 2,
  bucketBases: Uint32Array.of(0, 2),
  bucketCounts: Uint32Array.of(2, 3),
});

test('lane-local address expectation encodes active IDs and leaves all padding zero', () => {
  const expected = createFirstInstanceAddressChallengeExpected({
    scenario,
    visibleIds: Uint32Array.of(4, 99, 2, 0, 77),
    activeCounts: Uint32Array.of(1, 2),
  });
  assert.equal(expected.activeAddressCount, 3);
  assert.equal(expected.paddingAddressCount, 2);
  assert.equal(expected.targetPaddingPixelCount, 251);
  assert.equal(expected.outOfRangeIds, 0);
  assert.deepEqual([...expected.bytes.slice(0, 20)], [
    5, 0, 0, 255,
    0, 0, 0, 0,
    3, 0, 0, 255,
    1, 0, 0, 255,
    0, 0, 0, 0,
  ]);
  assert.equal(expected.bytes.slice(20).every((value) => value === 0), true);
});

test('lane-local expectation rejects command counts beyond fixed capacity', () => {
  assert.throws(
    () => createFirstInstanceAddressChallengeExpected({
      scenario,
      visibleIds: new Uint32Array(5),
      activeCounts: Uint32Array.of(3, 0),
    }),
    /exceeds its capacity/,
  );
});

test('lane-local expectation records an out-of-range active survivor as a hard failure input', () => {
  const expected = createFirstInstanceAddressChallengeExpected({
    scenario,
    visibleIds: Uint32Array.of(5, 0, 0, 0, 0),
    activeCounts: Uint32Array.of(1, 0),
  });
  assert.equal(expected.outOfRangeIds, 1);
  assert.deepEqual([...expected.bytes.slice(0, 4)], [0, 0, 0, 0]);
});
