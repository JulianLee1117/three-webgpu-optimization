import assert from 'node:assert/strict';
import test from 'node:test';

import {
  FIRST_INSTANCE_STANDALONE_BOOT_ID,
  firstInstanceStandaloneBootSearch,
  parseFirstInstanceStandaloneBoot,
} from '../src/benchmark/first-instance-standalone-boot.js';

test('standalone boot query selects the first and only initial deployment lane', () => {
  for (const laneId of ['portable', 'feature']) {
    for (const visibilityOrder of [[0.99, 0.2], [0.2, 0.99]]) {
      const search = firstInstanceStandaloneBootSearch({ laneId, visibilityOrder });
      const parsed = parseFirstInstanceStandaloneBoot(search);
      assert.equal(parsed.bootId, FIRST_INSTANCE_STANDALONE_BOOT_ID);
      assert.equal(parsed.modeId, 'first-instance-live-standalone-deployment');
      assert.equal(parsed.laneId, laneId);
      assert.deepEqual(parsed.visibilityOrder, visibilityOrder);
      assert.equal(parsed.objectCount, 65_536);
      assert.equal(parsed.bucketCount, 32);
      assert.equal(parsed.initialRebuildCount, 1);
      assert.equal(parsed.priorStrategyConstructionCount, 0);
    }
  }
});

test('ordinary page loads have no standalone boot contract', () => {
  assert.equal(parseFirstInstanceStandaloneBoot(''), null);
  assert.equal(parseFirstInstanceStandaloneBoot('?unrelated=value'), null);
});

test('standalone boot query rejects aliases, duplicates, extras, and malformed factors', () => {
  const valid = firstInstanceStandaloneBootSearch({
    laneId: 'portable',
    visibilityOrder: [0.99, 0.2],
  });
  for (const search of [
    '?benchmarkBoot=standalone&standaloneLane=portable&standaloneVisibilityOrder=0.99%2C0.2',
    `${valid}&standaloneLane=portable`,
    `${valid}&extra=1`,
    valid.replace('portable', 'feature-alias'),
    valid.replace('0.99%2C0.2', '0.99%2C0.5'),
  ]) {
    assert.throws(() => parseFirstInstanceStandaloneBoot(search));
  }
  assert.throws(() => firstInstanceStandaloneBootSearch({
    laneId: 'portable',
    visibilityOrder: [0.99, 0.5],
  }));
});
