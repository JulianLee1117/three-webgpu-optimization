import assert from 'node:assert/strict';
import test from 'node:test';
import {
  resolveTimestampMaps,
  timestampResolution,
} from '../src/benchmark/gpu-timestamps.js';

test('timestamp maps retain summed durations and count UIDs per frame', async () => {
  const resolvedTypes = [];
  const renderer = {
    backend: {
      trackTimestamp: true,
      timestampQueryPool: {
        render: {
          frames: [7, 8],
          timestamps: new Map([
            ['r:1:3:f6', 0.9],
            ['r:1:3:f7', 0.125],
            ['r:2:3:f7', 0.125],
            ['r:1:3:f8', 0.25],
          ]),
        },
      },
    },
    async resolveTimestampsAsync(type) {
      resolvedTypes.push(type);
    },
  };

  const maps = await resolveTimestampMaps(renderer, {
    includeCompute: false,
    collect: true,
  });

  assert.deepEqual(resolvedTypes, ['render']);
  assert.equal(maps.render.get(7), 0.25);
  assert.equal(maps.render.get(8), 0.25);
  assert.equal(maps.render.has(6), false);
  assert.equal(maps.uidCounts.render.get(7), 2);
  assert.equal(maps.uidCounts.render.get(8), 1);
  assert.deepEqual(timestampResolution(maps), {
    quantumNs: 250_000,
    classification: 'quantized',
  });
});
