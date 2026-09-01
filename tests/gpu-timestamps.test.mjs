import assert from 'node:assert/strict';
import test from 'node:test';
import {
  preprimeTimestampPools,
  registerComputeTimestampGroup,
  resolveTimestampMaps,
  timestampPoolDiagnostics,
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
    strictUidGrammar: true,
  });

  assert.deepEqual(resolvedTypes, ['render']);
  assert.equal(maps.render.get(7), 0.25);
  assert.equal(maps.render.get(8), 0.25);
  assert.equal(maps.render.has(6), false);
  assert.equal(maps.uidCounts.render.get(7), 2);
  assert.equal(maps.uidCounts.render.get(8), 1);
  assert.deepEqual(maps.frames.render, [7, 8]);
  assert.deepEqual(maps.uidsByFrame.render.get(7), ['r:1:3:f7', 'r:2:3:f7']);
  assert.equal(maps.uidDurations.render.get('r:1:3:f7'), 0.125);
  assert.deepEqual(maps.uidRecords.render, [
    {
      uid: 'r:1:3:f7',
      type: 'render',
      callIndex: 1,
      contextId: 3,
      frameId: 7,
      durationMs: 0.125,
    },
    {
      uid: 'r:2:3:f7',
      type: 'render',
      callIndex: 2,
      contextId: 3,
      frameId: 7,
      durationMs: 0.125,
    },
    {
      uid: 'r:1:3:f8',
      type: 'render',
      callIndex: 1,
      contextId: 3,
      frameId: 8,
      durationMs: 0.25,
    },
  ]);
  assert.deepEqual(maps.resolutions.render, {
    quantumNs: 125_000,
    classification: 'quantized',
    recordCount: 3,
    positiveDurationCount: 3,
    nonpositiveDurationCount: 0,
  });
  assert.deepEqual(timestampResolution(maps), {
    quantumNs: 125_000,
    classification: 'quantized',
  });
});

test('registered compute Array identities receive valid compute UIDs without mutation', () => {
  const data = new WeakMap();
  const delegated = [];
  const backend = {
    renderer: null,
    get(context) {
      if (!data.has(context)) data.set(context, {});
      return data.get(context);
    },
    updateTimeStampUID(context) {
      assert.equal(this, backend);
      delegated.push(context);
      this.get(context).timestampUID = 'delegated';
    },
  };
  const renderer = {
    backend,
    info: { frame: 9, compute: { frameCalls: 1 } },
  };
  backend.renderer = renderer;
  const computeGroup = [{ id: 1 }, { id: 2 }];
  const secondGroup = [{ id: 3 }];
  const unregistered = {};

  const registration = registerComputeTimestampGroup(renderer, computeGroup, 17);
  const installedWrapper = backend.updateTimeStampUID;
  assert.equal(registration.schemaVersion, 1);
  assert.equal(registration.kind, 'three-r185-compute-timestamp-group-registration');
  assert.equal(registration.contextId, 17);
  assert.equal(registration.registrationSerial, 1);
  assert.equal(Number.isInteger(registration.backendIdentity), true);
  assert.equal(Number.isInteger(registration.backendWrapperIdentity), true);
  assert.equal(Number.isInteger(registration.computeGroupIdentity), true);
  assert.equal(Object.isFrozen(registration), true);
  assert.equal(Object.hasOwn(computeGroup, 'isComputeNode'), false);
  assert.equal(Object.hasOwn(computeGroup, 'id'), false);

  backend.updateTimeStampUID(computeGroup);
  assert.equal(data.get(computeGroup).timestampUID, 'c:1:17:f9');
  assert.deepEqual(delegated, []);

  backend.updateTimeStampUID(unregistered);
  assert.equal(data.get(unregistered).timestampUID, 'delegated');
  assert.deepEqual(delegated, [unregistered]);

  const secondRegistration = registerComputeTimestampGroup(renderer, secondGroup, 18);
  assert.equal(backend.updateTimeStampUID, installedWrapper);
  assert.equal(secondRegistration.registrationSerial, 2);
  assert.equal(secondRegistration.backendIdentity, registration.backendIdentity);
  assert.equal(
    secondRegistration.backendWrapperIdentity,
    registration.backendWrapperIdentity,
  );
  assert.notEqual(secondRegistration.computeGroupIdentity, registration.computeGroupIdentity);
  renderer.info.frame = 10;
  renderer.info.compute.frameCalls = 2;
  backend.updateTimeStampUID(secondGroup);
  assert.equal(data.get(secondGroup).timestampUID, 'c:2:18:f10');

  assert.throws(
    () => registerComputeTimestampGroup(renderer, computeGroup, 19),
    /Array identity is already registered/,
  );
  assert.throws(
    () => registerComputeTimestampGroup(renderer, [{ id: 4 }], 18),
    /contextId 18 is already registered/,
  );
  for (const invalidContextId of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(
      () => registerComputeTimestampGroup(renderer, [{ id: 5 }], invalidContextId),
      /positive safe integer/,
    );
  }
});

test('timestamp resolution rejects malformed or pool-mismatched raw UIDs even without collection', async () => {
  for (const invalidComputeUid of ['r:1:undefined:f4', 'r:1:2:f4']) {
    const renderer = {
      backend: {
        trackTimestamp: true,
        timestampQueryPool: {
          render: {
            frames: [4],
            timestamps: new Map([['r:1:1:f4', 0.2]]),
          },
          compute: {
            frames: [4],
            timestamps: new Map([[invalidComputeUid, 0.1]]),
          },
        },
      },
      async resolveTimestampsAsync() {},
    };
    await assert.rejects(
      resolveTimestampMaps(renderer, {
        includeCompute: true,
        collect: false,
        strictUidGrammar: true,
      }),
      /Malformed Three\.js r185 compute timestamp UID|belongs to render, not compute/,
    );
  }
});

test('legacy array-compute UIDs remain frame-joinable outside exact-attribution mode', async () => {
  const renderer = {
    backend: {
      trackTimestamp: true,
      timestampQueryPool: {
        render: {
          frames: [6],
          timestamps: new Map([['r:1:1:f6', 0.0005]]),
        },
        compute: {
          frames: [6],
          timestamps: new Map([['r:0:undefined:f6', 0.0004]]),
        },
      },
    },
    async resolveTimestampsAsync() {},
  };
  const maps = await resolveTimestampMaps(renderer, {
    includeCompute: true,
    collect: true,
  });

  assert.equal(maps.strictUidGrammar, false);
  assert.equal(maps.compute.get(6), 0.0004);
  assert.deepEqual(maps.uidRecords.compute, [{
    uid: 'r:0:undefined:f6',
    type: 'compute',
    callIndex: null,
    contextId: null,
    frameId: 6,
    durationMs: 0.0004,
  }]);
});

test('compute pool quality cannot be masked by fine render timestamps', async () => {
  const renderer = {
    backend: {
      trackTimestamp: true,
      timestampQueryPool: {
        render: {
          frames: [5],
          timestamps: new Map([['r:1:1:f5', 0.0005]]),
        },
        compute: {
          frames: [5],
          timestamps: new Map([['c:1:2:f5', 0]]),
        },
      },
    },
    async resolveTimestampsAsync() {},
  };
  const maps = await resolveTimestampMaps(renderer, {
    includeCompute: true,
    collect: true,
    strictUidGrammar: true,
  });

  assert.deepEqual(maps.resolutions.render, {
    quantumNs: 500,
    classification: 'fine',
    recordCount: 1,
    positiveDurationCount: 1,
    nonpositiveDurationCount: 0,
  });
  assert.deepEqual(maps.resolutions.compute, {
    quantumNs: null,
    classification: 'invalid',
    recordCount: 1,
    positiveDurationCount: 0,
    nonpositiveDurationCount: 1,
  });
  assert.deepEqual(timestampResolution(maps), {
    quantumNs: null,
    classification: 'invalid',
  });
});

function fakeTimestampPool(prefix) {
  return {
    querySet: {},
    resolveBuffer: {},
    resultBuffer: { mapState: 'unmapped' },
    maxQueries: 2_048,
    currentQueryIndex: 2,
    queryOffsets: new Map([[`${prefix}:1:1:f3`, 0]]),
    frames: [],
    timestamps: new Map(),
    pendingResolve: false,
    isDisposed: false,
  };
}

test('timestamp diagnostics bind pool resources and clean query state', () => {
  const render = fakeTimestampPool('r');
  render.currentQueryIndex = 0;
  render.queryOffsets.clear();
  const renderer = {
    backend: {
      trackTimestamp: false,
      timestampQueryPool: { render, compute: null },
    },
  };
  const first = timestampPoolDiagnostics(renderer);
  const second = timestampPoolDiagnostics(renderer);
  assert.equal(first.render.poolIdentity, second.render.poolIdentity);
  assert.equal(first.render.querySetIdentity, second.render.querySetIdentity);
  assert.equal(first.render.maxQueries, 2_048);
  assert.equal(first.render.currentQueryIndex, 0);
  assert.equal(first.render.queryOffsetCount, 0);
  assert.equal(first.compute, null);
});

test('timestamp pre-prime allocates and resolves both pools before timing', async () => {
  const renderer = {
    backend: {
      trackTimestamp: false,
      timestampQueryPool: { render: null, compute: null },
    },
    async resolveTimestampsAsync(type) {
      const pool = this.backend.timestampQueryPool[type];
      pool.timestamps.set([...pool.queryOffsets.keys()][0], 0.0005);
      pool.frames = [3];
      pool.currentQueryIndex = 0;
      pool.queryOffsets.clear();
    },
  };
  const allocate = (type, prefix) => {
    const pool = fakeTimestampPool(prefix);
    renderer.backend.timestampQueryPool[type] = pool;
  };
  const diagnostics = await preprimeTimestampPools(renderer, {
    submitCompute: () => allocate('compute', 'c'),
    submitRender: () => allocate('render', 'r'),
  });
  assert.equal(renderer.backend.trackTimestamp, false);
  assert.equal(diagnostics.after.compute.currentQueryIndex, 0);
  assert.equal(diagnostics.after.render.currentQueryIndex, 0);
  assert.equal(diagnostics.after.compute.queryOffsetCount, 0);
  assert.equal(diagnostics.after.render.queryOffsetCount, 0);
  assert.equal(diagnostics.after.compute.timestampUidCount, 1);
  assert.equal(diagnostics.after.render.timestampUidCount, 1);
  assert.deepEqual(diagnostics.addedTimestampUidCount, { render: 1, compute: 1 });
});
