import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { PerspectiveCamera, WebGPUCoordinateSystem } from 'three/webgpu';
import {
  IMMEDIATE_AIF_PHASE0_LANES,
  IMMEDIATE_AIF_PHASE0_LANE_ORDERS,
  IMMEDIATE_AIF_PHASE0_SCHEDULES,
  IMMEDIATE_AIF_UNUSED_ADDRESS,
  createImmediateAifAddressOracle,
  createImmediateAifCommandOracle,
  createImmediateAifPhase0Scenarios,
} from '../src/phase0/immediate-aif-plan.js';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import {
  IMMEDIATE_AIF_PHASE0_CORRECTNESS_STATES,
  buildImmediateAifPhase0Strategy,
  createImmediateAifPhase0CorrectnessStateMachine,
} from '../src/strategies/immediate-aif-phase0.js';

const [A, I, F] = IMMEDIATE_AIF_PHASE0_LANES;
const STATE = IMMEDIATE_AIF_PHASE0_CORRECTNESS_STATES;

function completeLiveValidation(machine, order) {
  machine.beginLiveValidation(order);
  for (const lane of order) {
    machine.beginLaneSubmission(lane);
    machine.finishLaneSubmission(lane, true);
  }
  machine.freezeVisibleIds(true);
}

function completePrime(machine, order = [A, I, F]) {
  machine.beginPriming(order);
  for (const lane of order) {
    machine.beginBundlePrime(lane);
    machine.finishBundlePrime(lane, true);
  }
}

function completeRerecord(machine) {
  machine.beginImmediateRerecord();
  machine.finishImmediateRerecord(true);
}

test('Phase 0 correctness state machine rejects unsafe ordering and covers all six permutations', () => {
  const machine = createImmediateAifPhase0CorrectnessStateMachine();
  assert.equal(machine.snapshot().phase, STATE.CONSTRUCTED);
  assert.throws(() => machine.beginLiveValidation([A, I, F]), /scenario-loaded/);
  assert.throws(() => machine.loadScenario('v80'), /v99 or v20/);

  machine.loadScenario('v99');
  machine.beginLiveValidation([I, F, A]);
  assert.throws(() => machine.beginLaneSubmission(A), /lane I must run next/);
  machine.beginLaneSubmission(I);
  assert.throws(() => machine.beginLaneSubmission(F), /already in flight/);
  machine.finishLaneSubmission(I, true);
  for (const lane of [F, A]) {
    machine.beginLaneSubmission(lane);
    machine.finishLaneSubmission(lane, true);
  }
  assert.equal(machine.snapshot().phase, STATE.LIVE_VALIDATED);
  machine.freezeVisibleIds(true);
  assert.equal(machine.snapshot().phase, STATE.VISIBLE_IDS_FROZEN);
  completePrime(machine, [F, A, I]);
  assert.equal(machine.snapshot().phase, STATE.READY);
  assert.deepEqual(machine.snapshot().bundlePrimed, { A: true, I: true, F: true });
  assert.equal(machine.snapshot().immediateRecordedScheduleId, 'canonical');

  machine.transitionSchedule('S1');
  assert.equal(machine.snapshot().phase, STATE.IMMEDIATE_RERECORD_REQUIRED);
  assert.throws(() => machine.beginLaneOrder([A, I, F]), /requires state ready/);
  completeRerecord(machine);
  assert.equal(machine.snapshot().immediateRecordedScheduleId, 'S1');

  for (const order of IMMEDIATE_AIF_PHASE0_LANE_ORDERS) {
    machine.beginLaneOrder(order);
    assert.throws(
      () => machine.beginLaneSelection(order[1]),
      new RegExp(`lane ${order[0]} must run next`),
    );
    for (const lane of order) {
      machine.beginLaneSelection(lane);
      machine.finishLaneSelection(lane, true);
    }
  }
  assert.equal(Object.keys(machine.snapshot().laneOrderCoverage).length, 6);

  machine.transitionSchedule('canonical');
  completeRerecord(machine);
  machine.loadScenario('v20');
  completeLiveValidation(machine, [A, F, I]);
  assert.equal(
    machine.snapshot().phase,
    STATE.READY,
    'the same three bundle identities remain primed for the second visibility',
  );

  const failed = createImmediateAifPhase0CorrectnessStateMachine();
  failed.loadScenario('v99');
  failed.beginLiveValidation([A, I, F]);
  failed.beginLaneSubmission(A);
  failed.finishLaneSubmission(A, false);
  assert.equal(failed.snapshot().phase, STATE.FAILED);
  assert.throws(() => failed.beginLaneSubmission(I), /unavailable after Phase 0 failed/);
});

test('Phase 0 state machine closes only the exact 36-cell ordered coverage', () => {
  const machine = createImmediateAifPhase0CorrectnessStateMachine();
  for (const [scenarioIndex, scenarioId] of ['v99', 'v20'].entries()) {
    machine.loadScenario(scenarioId);
    completeLiveValidation(machine, [A, I, F]);
    if (scenarioIndex === 0) completePrime(machine);
    for (const scheduleId of ['canonical', 'S1', 'S2']) {
      if (scheduleId !== machine.snapshot().activeScheduleId) {
        machine.transitionSchedule(scheduleId);
        completeRerecord(machine);
      }
      for (const order of IMMEDIATE_AIF_PHASE0_LANE_ORDERS) {
        machine.beginLaneOrder(order);
        for (const lane of order) {
          machine.beginLaneSelection(lane);
          machine.finishLaneSelection(lane, true);
        }
      }
    }
    machine.transitionSchedule('canonical');
    completeRerecord(machine);
  }
  const complete = machine.snapshot();
  assert.deepEqual(complete.scenarioLoadSequence, ['v99', 'v20']);
  assert.equal(complete.expectedLaneOrderCoverageCount, 36);
  assert.equal(Object.keys(complete.laneOrderCoverage).length, 36);
  assert.equal(complete.scenarioSequenceExact, true);
  assert.equal(complete.laneOrderCoverageExact, true);
  assert.equal(complete.runtimeCoverageComplete, true);

  machine.beginLaneOrder([A, I, F]);
  for (const lane of [A, I, F]) {
    machine.beginLaneSelection(lane);
    machine.finishLaneSelection(lane, true);
  }
  assert.equal(machine.snapshot().laneOrderCoverageExact, false);
  assert.equal(machine.snapshot().runtimeCoverageComplete, false);
});

function buildFixtureRuntime() {
  const sourceGeometries = createIndexedGeometryFixtures(32, 'medium');
  const scenarios = createImmediateAifPhase0Scenarios({
    geometrySpheres: sourceGeometries.map((geometry) => geometry.boundingSphere),
  });
  const renderer = {
    coordinateSystem: WebGPUCoordinateSystem,
    compute() {
      throw new Error('mock compute was not connected to the strategy');
    },
    async getArrayBufferAsync(attribute) {
      return attribute.array.slice().buffer;
    },
  };
  const camera = new PerspectiveCamera(60, 16 / 9, 0.1, 1_000);
  camera.position.set(0, 0, 8);
  camera.updateProjectionMatrix();
  const strategy = buildImmediateAifPhase0Strategy({
    renderer,
    camera,
    sourceGeometries,
    scenarios,
  });
  return { camera, renderer, scenarios, sourceGeometries, strategy };
}

function disposeFixtureRuntime({ sourceGeometries, strategy }) {
  strategy.dispose();
  for (const node of strategy.computeNodes) node.dispose?.();
  for (const material of strategy.materials) material.dispose();
  for (const geometry of strategy.geometries) {
    geometry.setIndirect?.(null);
    geometry.dispose();
  }
  for (const geometry of sourceGeometries) geometry.dispose();
}

function bundleRecordCallback(context) {
  context.mesh.onBeforeRender({ _currentRenderBundle: {} });
  return { pass: true };
}

function permutePackedVisibleIds(packed, scenario, lane) {
  const result = packed.slice();
  const laneShift = IMMEDIATE_AIF_PHASE0_LANES.indexOf(lane);
  if (laneShift === 0) return result;
  for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
    const base = scenario.bucketBases[bucket];
    const count = scenario.visibleCounts[bucket];
    const active = result.slice(base, base + count);
    const shift = laneShift % count;
    for (let slot = 0; slot < count; slot += 1) {
      result[base + slot] = active[(slot + shift) % count];
    }
  }
  return result;
}

test('Phase 0 runtime owns exact A/I/F topology and loads a full sentinel-reset scenario', () => {
  const fixture = buildFixtureRuntime();
  const { scenarios, strategy } = fixture;
  try {
    const { laneStates, sharedResources } = strategy;
    assert.equal(strategy.root.children.length, 3);
    assert.equal(new Set(Object.values(laneStates).map((lane) => lane.root)).size, 3);
    assert.equal(new Set(Object.values(laneStates).map((lane) => lane.geometry)).size, 3);
    assert.equal(
      new Set(Object.values(laneStates).map((lane) => lane.indirectAttribute)).size,
      3,
    );
    assert.notEqual(laneStates.I.geometry, laneStates.F.geometry);
    assert.equal(laneStates.A.geometry.getAttribute('bucketBase') !== undefined, true);
    assert.equal(laneStates.I.geometry.getAttribute('bucketBase'), undefined);
    assert.equal(laneStates.F.geometry.getAttribute('bucketBase'), undefined);
    assert.equal(laneStates.I.geometry.index, laneStates.F.geometry.index);
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      assert.equal(laneStates[lane].root.static, true);
      assert.equal(laneStates[lane].meshes.length, 1);
      assert.equal(laneStates[lane].root.children.length, 1);
      assert.equal(laneStates[lane].indirectAttribute.array.byteLength, 640);
    }

    const versionBefore = sharedResources.attributes.visibleIds.version;
    const load = strategy.loadScenario('v99');
    assert.equal(load.visibleReset.valueCount, 65_536);
    assert.equal(load.visibleReset.byteLength, 262_144);
    assert.equal(sharedResources.attributes.visibleIds.version, versionBefore + 1);
    assert.equal(
      sharedResources.attributes.visibleIds.array.every(
        (value) => value === IMMEDIATE_AIF_UNUSED_ADDRESS,
      ),
      true,
    );
    assert.deepEqual(sharedResources.attributes.matrix.array, scenarios.v99.matrices);
    assert.deepEqual(sharedResources.attributes.bounds.array, scenarios.v99.bounds);
    assert.deepEqual(sharedResources.attributes.objectBucket.array, scenarios.v99.objectBuckets);
    assert.deepEqual(sharedResources.attributes.cullOrder.array, scenarios.v99.cullOrder);
    assert.equal(sharedResources.attributes.overflow.array[0], IMMEDIATE_AIF_UNUSED_ADDRESS);
    for (let draw = 0; draw < 32; draw += 1) {
      assert.equal(laneStates.A.indirectAttribute.array[draw * 5 + 4], 0);
      assert.equal(laneStates.I.indirectAttribute.array[draw * 5 + 4], 0);
      assert.equal(
        laneStates.F.indirectAttribute.array[draw * 5 + 4],
        scenarios.v99.bucketBases[draw],
      );
    }
    const diagnostics = strategy.diagnostics();
    assert.equal(diagnostics.pass, true);
    assert.deepEqual(diagnostics.commandBufferByteLengths, { A: 640, I: 640, F: 640 });
    assert.deepEqual(diagnostics.bucketBaseVertexStreams, { A: true, I: false, F: false });
    assert.equal(diagnostics.analysisEligible, false);
    assert.equal(diagnostics.numericalDecision, null);
    strategy.laneStates.A.root.visible = true;
    strategy.laneStates.I.root.visible = true;
    strategy.selectActiveLane(F);
    const selected = strategy.diagnostics();
    assert.equal(selected.activeLane, F);
    assert.equal(selected.activeRootCount, 1);
    assert.deepEqual(
      IMMEDIATE_AIF_PHASE0_LANES.filter(
        (lane) => strategy.laneStates[lane].root.visible,
      ),
      [F],
    );
  } finally {
    disposeFixtureRuntime(fixture);
  }
});

test('Phase 0 runtime isolates later loads and schedule oracles from caller/public mutation', async () => {
  const fixture = buildFixtureRuntime();
  const { camera, renderer, scenarios, sourceGeometries, strategy } = fixture;
  const originalIndexCount = sourceGeometries[0].index.count;
  const originalMatrixWord = scenarios.v99.matrices[0];
  const commands = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
    lane,
    createImmediateAifCommandOracle({
      lane,
      scenario: scenarios.v99,
      sourceGeometries,
    }),
  ]));
  const addresses = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
    lane,
    createImmediateAifAddressOracle({ lane, scenario: scenarios.v99 }),
  ]));
  try {
    sourceGeometries[0].index.count = 1;
    scenarios.v99.matrices[0] = originalMatrixWord + 123;
    const exposed = strategy.scenarios;
    exposed.v99.matrices[0] = originalMatrixWord + 456;
    exposed.v99.bucketBases[0] = 99;
    strategy.sharedResources.firstIndexes[0] = 31337;

    const freshPublicCopy = strategy.scenarios;
    assert.equal(freshPublicCopy.v99.matrices[0], originalMatrixWord);
    assert.equal(freshPublicCopy.v99.bucketBases[0], 0);

    const load = strategy.loadScenario('v99');
    assert.equal(load.scenarioId, 'v99');
    assert.equal(strategy.laneStates.A.indirectAttribute.array[0], originalIndexCount);
    assert.equal(strategy.sharedResources.attributes.matrix.array[0], originalMatrixWord);

    renderer.compute = (computeNodes) => {
      const lane = IMMEDIATE_AIF_PHASE0_LANES.find(
        (candidate) => strategy.laneStates[candidate].computeNodes === computeNodes,
      );
      strategy.laneStates[lane].indirectAttribute.array.set(commands[lane].postCullCommands);
      strategy.sharedResources.attributes.visibleIds.array.set(
        addresses[lane].canonicalVisibleIds,
      );
      strategy.sharedResources.attributes.overflow.array[0] = 0;
    };
    strategy.beginLiveValidation([A, I, F]);
    for (const lane of [A, I, F]) {
      assert.equal((await strategy.submitAndValidateLane(lane, { renderer, camera })).pass, true);
    }
    assert.equal(strategy.freezeValidatedVisibleIds().pass, true);
    assert.equal((await strategy.primeBundles({ renderLane: bundleRecordCallback })).pass, true);

    const transition = strategy.transitionSchedule('S1');
    assert.equal(transition.pass, true);
    assert.equal(
      strategy.getLaneValidationResources(F).commandOracle.postCullCommands[0],
      originalIndexCount,
    );
    assert.equal(strategy.laneStates.F.indirectAttribute.array[0], originalIndexCount);
  } finally {
    disposeFixtureRuntime(fixture);
  }
});

test('Phase 0 runtime serializes live captures, freezes survivors, and transitions A/I/F schedules', async () => {
  const fixture = buildFixtureRuntime();
  const {
    camera,
    renderer,
    scenarios,
    sourceGeometries,
    strategy,
  } = fixture;
  const preCompute = [];
  try {
    renderer.compute = (computeNodes) => {
      const lane = IMMEDIATE_AIF_PHASE0_LANES.find(
        (candidate) => strategy.laneStates[candidate].computeNodes === computeNodes,
      );
      assert.ok(lane, 'compute submission must use one known private lane command buffer');
      const scenario = scenarios[strategy.activeScenarioId];
      const command = createImmediateAifCommandOracle({
        lane,
        scenario,
        sourceGeometries,
        firstIndexes: strategy.sharedResources.firstIndexes,
      });
      const address = createImmediateAifAddressOracle({ lane, scenario });
      const indirect = strategy.laneStates[lane].indirectAttribute;
      const visible = strategy.sharedResources.attributes.visibleIds;
      const overflow = strategy.sharedResources.attributes.overflow;
      preCompute.push({
        lane,
        fullVisibleSentinelReset: visible.array.every(
          (value) => value === IMMEDIATE_AIF_UNUSED_ADDRESS,
        ),
        overflowSentinelReset: overflow.array[0] === IMMEDIATE_AIF_UNUSED_ADDRESS,
        initialCommandsExact: assert.deepEqual(indirect.array, command.initialCommands) === undefined,
      });
      indirect.array.set(command.postCullCommands);
      visible.array.set(permutePackedVisibleIds(
        address.canonicalVisibleIds,
        scenario,
        lane,
      ));
      overflow.array[0] = 0;
    };

    strategy.loadScenario('v99');
    const validationOrder = [I, F, A];
    strategy.beginLiveValidation(validationOrder);
    for (const lane of validationOrder) {
      const evidence = await strategy.submitAndValidateLane(lane, { renderer, camera });
      assert.equal(evidence.pass, true);
      assert.equal(evidence.commandExact, true);
      assert.equal(evidence.observedCanonicalOrder, lane === A);
      assert.equal(evidence.paddingSentinelsExact, true);
      assert.equal(evidence.correctness.membership.pass, true);
    }
    assert.deepEqual(preCompute.map((entry) => entry.lane), validationOrder);
    assert.equal(preCompute.every((entry) => entry.fullVisibleSentinelReset), true);
    assert.equal(preCompute.every((entry) => entry.overflowSentinelReset), true);
    assert.equal(preCompute.every((entry) => entry.initialCommandsExact), true);

    const frozen = strategy.freezeValidatedVisibleIds();
    assert.equal(frozen.pass, true);
    assert.equal(frozen.allLiveMembershipValidationsPass, true);
    assert.equal(frozen.canonicalPackingExact, true);
    assert.equal(frozen.observedLiveOrdersEqual, false);
    assert.equal(frozen.source, 'deterministic-cpu-canonical-packing');
    assert.equal(frozen.valueCount, 65_536);
    assert.deepEqual(strategy.frozenVisibleIds, frozen.visibleIds);
    assert.deepEqual(
      frozen.visibleIds,
      createImmediateAifAddressOracle({ lane: A, scenario: scenarios.v99 }).canonicalVisibleIds,
    );

    const bundleIdentities = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map(
      (lane) => [lane, strategy.laneStates[lane].root.uuid],
    ));
    const prime = await strategy.primeBundles({
      order: [F, I, A],
      renderLane: bundleRecordCallback,
    });
    assert.equal(prime.pass, true);
    assert.deepEqual(strategy.diagnostics().bundleRecordCounts, { A: 1, I: 1, F: 1 });
    assert.throws(
      () => strategy.mutateAndDetachImmediateBaseSource(
        Uint32Array.from(IMMEDIATE_AIF_PHASE0_SCHEDULES.S1.sourceBaseByDraw),
      ),
      /only from the native I bundle finish hook/,
    );

    const canonicalOrders = await strategy.runAllLaneOrders({
      renderLane(context) {
        assert.equal(strategy.activeLane, context.lane);
        assert.equal(
          IMMEDIATE_AIF_PHASE0_LANES.filter(
            (lane) => strategy.laneStates[lane].root.visible,
          ).length,
          1,
        );
        assert.equal(context.scheduleId, 'canonical');
        return { pass: true, outputCaptured: true };
      },
    });
    assert.equal(canonicalOrders.pass, true);
    assert.equal(Object.keys(strategy.state.laneOrderCoverage).length, 6);
    assert.deepEqual(strategy.diagnostics().bundleRecordCounts, { A: 1, I: 1, F: 1 });

    for (const scheduleId of ['S1', 'S2', 'canonical']) {
      const beforeCounts = strategy.diagnostics().bundleRecordCounts;
      const beforeIUuid = strategy.laneStates.I.root.uuid;
      const transition = strategy.transitionSchedule(scheduleId);
      assert.equal(transition.pass, true);
      assert.equal(transition.immediateSourceWasDetached, scheduleId === 'S2');
      assert.equal(transition.immediateSourceReplaced, scheduleId === 'S2');
      assert.equal(transition.immediateRerecordRequired, true);
      assert.equal(strategy.state.phase, STATE.IMMEDIATE_RERECORD_REQUIRED);
      await assert.rejects(
        strategy.runLaneOrder([A, I, F], { renderLane() {} }),
        /requires state ready/,
      );
      const expectedBases = IMMEDIATE_AIF_PHASE0_SCHEDULES[scheduleId].sourceBaseByDraw;
      assert.deepEqual([...strategy.laneStates.I.indirectImmediateBases], expectedBases);
      for (let draw = 0; draw < 32; draw += 1) {
        assert.equal(strategy.laneStates.A.indirectAttribute.array[draw * 5 + 4], 0);
        assert.equal(strategy.laneStates.I.indirectAttribute.array[draw * 5 + 4], 0);
        assert.equal(
          strategy.laneStates.F.indirectAttribute.array[draw * 5 + 4],
          expectedBases[draw],
        );
      }
      let vertexCursor = 0;
      const aBaseAttribute = strategy.laneStates.A.geometry.getAttribute('bucketBase');
      for (let draw = 0; draw < 32; draw += 1) {
        assert.equal(aBaseAttribute.array[vertexCursor], expectedBases[draw]);
        vertexCursor += sourceGeometries[draw].getAttribute('position').count;
      }
      assert.equal(vertexCursor, aBaseAttribute.array.length);

      let detach = null;
      let detachedSource = null;
      const rerecord = await strategy.rerecordImmediateBundle({
        renderLane(context) {
          const callback = bundleRecordCallback(context);
          if (scheduleId === 'S1') {
            detachedSource = strategy.laneStates.I.indirectImmediateBases;
            detach = strategy.mutateAndDetachImmediateBaseSource(
              Uint32Array.from(IMMEDIATE_AIF_PHASE0_SCHEDULES.canonical.sourceBaseByDraw),
            );
          }
          return callback;
        },
      });
      assert.equal(rerecord.pass, true);
      assert.equal(rerecord.sameBundleIdentity, true);
      assert.equal(rerecord.sameBundleVersion, true);
      assert.equal(strategy.laneStates.I.root.uuid, beforeIUuid);
      assert.equal(rerecord.after.I, beforeCounts.I + 1);
      assert.equal(rerecord.after.A, beforeCounts.A);
      assert.equal(rerecord.after.F, beforeCounts.F);
      if (scheduleId === 'S1') {
        assert.equal(detach.pass, true);
        assert.equal(detach.detachedDuringImmediateRerecord, true);
        assert.deepEqual([...detach.recordedBases], expectedBases);
        assert.deepEqual(
          [...detach.mutatedBases],
          IMMEDIATE_AIF_PHASE0_SCHEDULES.canonical.sourceBaseByDraw,
        );
        assert.equal(detach.detachedByteLength, 0);
        assert.equal(detach.detachedLength, 0);
        assert.equal(detachedSource.byteLength, 0);
        assert.equal(detachedSource.length, 0);
      }
      const oneOrder = await strategy.runLaneOrder([A, F, I], {
        renderLane(context) {
          assert.equal(context.scheduleId, scheduleId);
          assert.deepEqual([...context.commandOracle.sourceBaseByDraw], expectedBases);
          return { pass: true };
        },
      });
      assert.equal(oneOrder.pass, true);
    }
    assert.equal(strategy.state.activeScheduleId, 'canonical');
    assert.deepEqual(
      Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map(
        (lane) => [lane, strategy.laneStates[lane].root.uuid],
      )),
      bundleIdentities,
    );

    preCompute.length = 0;
    strategy.loadScenario('v20');
    strategy.beginLiveValidation([A, F, I]);
    for (const lane of [A, F, I]) {
      assert.equal((await strategy.submitAndValidateLane(lane)).pass, true);
    }
    assert.equal(strategy.freezeValidatedVisibleIds().pass, true);
    assert.equal(strategy.state.phase, STATE.READY);
    assert.deepEqual(strategy.diagnostics().bundleRecordCounts, { A: 1, I: 4, F: 1 });
    assert.equal(strategy.diagnostics().visibleResetSerial, 8);
  } finally {
    disposeFixtureRuntime(fixture);
  }
});

test('Phase 0 runtime rejects absent, primitive, and malformed callback evidence', async () => {
  const malformedValues = [undefined, null, false, {}, { pass: 0 }];
  for (const malformed of malformedValues) {
    const fixture = buildFixtureRuntime();
    const { renderer, scenarios, sourceGeometries, strategy } = fixture;
    try {
      renderer.compute = (computeNodes) => {
        const lane = IMMEDIATE_AIF_PHASE0_LANES.find(
          (candidate) => strategy.laneStates[candidate].computeNodes === computeNodes,
        );
        const scenario = scenarios.v99;
        const command = createImmediateAifCommandOracle({
          lane,
          scenario,
          sourceGeometries,
          firstIndexes: strategy.sharedResources.firstIndexes,
        });
        const address = createImmediateAifAddressOracle({ lane, scenario });
        strategy.laneStates[lane].indirectAttribute.array.set(command.postCullCommands);
        strategy.sharedResources.attributes.visibleIds.array.set(address.canonicalVisibleIds);
        strategy.sharedResources.attributes.overflow.array[0] = 0;
      };
      strategy.loadScenario('v99');
      strategy.beginLiveValidation([A, I, F]);
      for (const lane of [A, I, F]) {
        assert.equal((await strategy.submitAndValidateLane(lane)).pass, true);
      }
      assert.equal(strategy.freezeValidatedVisibleIds().pass, true);
      const prime = await strategy.primeBundles({
        renderLane(context) {
          context.mesh.onBeforeRender({ _currentRenderBundle: {} });
          return malformed;
        },
      });
      assert.equal(prime.pass, false);
      assert.equal(strategy.state.phase, STATE.FAILED);
    } finally {
      disposeFixtureRuntime(fixture);
    }
  }
});

test('Phase 0 strategy has no timing seam and delegates reset/cull without word-four access', async () => {
  const strategyFilename = fileURLToPath(
    new URL('../src/strategies/immediate-aif-phase0.js', import.meta.url),
  );
  const fixedSliceFilename = fileURLToPath(
    new URL('../src/strategies/fixed-slice.js', import.meta.url),
  );
  const [strategySource, fixedSliceSource] = await Promise.all([
    readFile(strategyFilename, 'utf8'),
    readFile(fixedSliceFilename, 'utf8'),
  ]);
  assert.doesNotMatch(
    strategySource,
    /registerComputeTimestamp|preprimeTimestamp|gpu-timestamps|performance\.now|Date\.now|\?\.pass\s*!==\s*false/,
  );
  const computeStart = fixedSliceSource.indexOf('function createFixedSliceComputeNodes');
  const computeEnd = fixedSliceSource.indexOf('function validateImmediateBases', computeStart);
  assert.ok(computeStart >= 0 && computeEnd > computeStart);
  const computeSource = fixedSliceSource.slice(computeStart, computeEnd);
  assert.doesNotMatch(computeSource, /firstInstance/);
  assert.match(computeSource, /get\('instanceCount'\)/);
});
