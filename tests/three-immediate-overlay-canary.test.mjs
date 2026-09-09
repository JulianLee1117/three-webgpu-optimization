import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import {
  CANONICAL_IMMEDIATE_BASES,
  DRAW_COUNT,
  IMMEDIATE_DRAW_BASE_EXPRESSION,
  INDIRECT_WORD_COUNT,
  MUTATED_IMMEDIATE_BASES,
  TARGET_WIDTH,
  VISIBLE_IDS,
  createIndirectCommands,
  expectedOutputForBases,
} from '../src/three-immediate-overlay-canary.js';

test('integration commands bind draw ordinal through distinct firstIndex words', () => {
  const commands = createIndirectCommands();
  assert.equal(commands.length, DRAW_COUNT * INDIRECT_WORD_COUNT);
  for (let draw = 0; draw < DRAW_COUNT; draw += 1) {
    const record = Array.from(commands.subarray(
      draw * INDIRECT_WORD_COUNT,
      (draw + 1) * INDIRECT_WORD_COUNT,
    ));
    assert.deepEqual(record, [3, 2, draw * 3, 0, 0]);
  }
});

test('bundle source mutation changes every draw-ordinal output pixel', () => {
  const canonical = expectedOutputForBases(CANONICAL_IMMEDIATE_BASES);
  const mutated = expectedOutputForBases(MUTATED_IMMEDIATE_BASES);
  assert.equal(canonical.length, TARGET_WIDTH * 4);
  assert.notDeepEqual(mutated, canonical);
  for (let draw = 0; draw < DRAW_COUNT; draw += 1) {
    for (let instance = 0; instance < 2; instance += 1) {
      const pixel = draw * 2 + instance;
      const address = CANONICAL_IMMEDIATE_BASES[draw] + instance;
      assert.deepEqual(
        Array.from(canonical.subarray(pixel * 4, pixel * 4 + 4)),
        [VISIBLE_IDS[address] + 1, 0, 0, 255],
      );
      assert.notEqual(mutated[pixel * 4], canonical[pixel * 4]);
    }
  }
});

function permutations(values) {
  if (values.length === 0) return [[]];
  return values.flatMap((value, index) => permutations([
    ...values.slice(0, index),
    ...values.slice(index + 1),
  ]).map((tail) => [value, ...tail]));
}

test('draw-specific oracle rejects every base swap, omission, and legal single mutation', () => {
  const canonical = expectedOutputForBases(CANONICAL_IMMEDIATE_BASES);
  for (const permutation of permutations([...CANONICAL_IMMEDIATE_BASES])) {
    if (permutation.every((value, index) => value === CANONICAL_IMMEDIATE_BASES[index])) {
      continue;
    }
    assert.notDeepEqual(expectedOutputForBases(permutation), canonical);
  }
  for (let draw = 0; draw < DRAW_COUNT; draw += 1) {
    const omitted = [...CANONICAL_IMMEDIATE_BASES];
    omitted[draw] = 0xffff_ffff;
    assert.notDeepEqual(expectedOutputForBases(omitted), canonical);
    for (let replacement = 0; replacement <= VISIBLE_IDS.length - 2; replacement += 1) {
      if (replacement === CANONICAL_IMMEDIATE_BASES[draw]) continue;
      const mutated = [...CANONICAL_IMMEDIATE_BASES];
      mutated[draw] = replacement;
      assert.notDeepEqual(expectedOutputForBases(mutated), canonical);
    }
  }
});

test('canary uses a legal WGSL symbol and freezes the timing-free claim boundary', async () => {
  assert.equal(IMMEDIATE_DRAW_BASE_EXPRESSION, 'threeImmediateDrawBase');
  assert.doesNotMatch(IMMEDIATE_DRAW_BASE_EXPRESSION, /^__/);
  const [pageSource, runnerSource] = await Promise.all([
    readFile('src/three-immediate-overlay-canary.js', 'utf8'),
    readFile('scripts/probe-three-immediate-overlay.mjs', 'utf8'),
  ]);
  for (const source of [pageSource, runnerSource]) {
    assert.doesNotMatch(source, /performance\.now\s*\(/);
    assert.doesNotMatch(source, /trackTimestamp\s*:\s*true/);
    assert.match(source, /timingCaptured:\s*false/);
    assert.match(source, /fullPhaseZeroPass:\s*false/);
  }
  assert.match(pageSource, /setImmediates/);
  assert.match(pageSource, /drawIndexedIndirect/);
  assert.match(pageSource, /after-native-finish-returned-before-bundle-execution/);
  assert.match(pageSource, /after-first-render-executed-before-second-cached-render/);
  assert.match(pageSource, /createCommandEncoder/);
});
