import assert from 'node:assert/strict';
import test from 'node:test';
import { disposeStrategyResources } from '../src/strategies/resources.js';

test('strategy disposal explicitly deletes deduplicated storage and indirect attributes', () => {
  const shared = { name: 'shared' };
  const indirect = { name: 'indirect' };
  const deleted = [];
  let detached = 0;
  let strategyDisposed = 0;
  let geometryDisposed = 0;

  const renderer = {
    _attributes: {
      delete(attribute) {
        deleted.push(attribute);
      },
    },
  };
  const geometry = {
    indirect,
    setIndirect(value) {
      this.indirect = value;
    },
    dispose() {
      geometryDisposed += 1;
    },
  };
  disposeStrategyResources(renderer, {
    root: { removeFromParent() { detached += 1; } },
    dispose() { strategyDisposed += 1; },
    geometries: [geometry],
    materials: [],
    computeNodes: [],
    storageAttributes: [shared, shared, indirect],
  });

  assert.equal(detached, 1);
  assert.equal(strategyDisposed, 1);
  assert.equal(geometryDisposed, 1);
  assert.equal(geometry.indirect, null);
  assert.deepEqual(deleted, [shared, indirect]);
});

test('strategy disposal fails closed when pinned renderer cleanup is unavailable', () => {
  assert.throws(
    () => disposeStrategyResources({}, {
      geometries: [],
      storageAttributes: [{}],
    }),
    /attribute cleanup is unavailable/,
  );
});
