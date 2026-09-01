import assert from 'node:assert/strict';
import test from 'node:test';
import {
  Color,
  PerspectiveCamera,
  Scene,
  WebGPUCoordinateSystem,
} from 'three/webgpu';
import {
  FIRST_INSTANCE_LIVE_CROSSOVER_MODE,
} from '../src/benchmark/plan.js';
import { registerComputeTimestampGroup } from '../src/benchmark/gpu-timestamps.js';
import { createFixedSubsetScenario } from '../src/scenes/fixed-subsets.js';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import {
  buildFixedSliceDeploymentStrategy,
  selectFixedSliceDeployment,
} from '../src/strategies/fixed-slice.js';
import {
  buildFirstInstanceLiveCrossoverStrategy,
  createLiveComputeDispatchCommitment,
} from '../src/strategies/live-first-instance-crossover.js';
import { disposeStrategyResources } from '../src/strategies/resources.js';
import {
  createFirstInstanceAddressChallengeExpected,
} from '../src/validation/first-instance-address-challenge.js';

const PORTABLE = 'portable';
const FEATURE = 'feature';

test('live compute commitments pin count, workgroup, and derived dispatch dimensions', () => {
  assert.deepEqual(
    createLiveComputeDispatchCommitment(
      { isComputeNode: true, count: 32, workgroupSize: [64, 1, 1] },
      [1, 1, 1],
    ),
    {
      count: 32,
      workgroupSize: [64, 1, 1],
      invocationsPerWorkgroup: 64,
      derivedDispatchSize: [1, 1, 1],
      runtimeDispatchSize: [1, 1, 1],
      runtimeMatchesDerived: true,
    },
  );
  assert.deepEqual(
    createLiveComputeDispatchCommitment(
      { isComputeNode: true, count: 65_536, workgroupSize: [64] },
      [1_024, 1, 1],
    ).derivedDispatchSize,
    [1_024, 1, 1],
  );
});

function createFixture(lanePhysicalOrder = [FEATURE, PORTABLE]) {
  const sourceGeometries = createIndexedGeometryFixtures(4, 'low');
  const scenario = createFixedSubsetScenario({
    objectCount: 512,
    bucketCount: 4,
    visibilityFraction: 0.2,
    geometrySpheres: sourceGeometries.map((geometry) => geometry.boundingSphere),
    seed: 0xb1ad_2026,
  });
  const camera = new PerspectiveCamera(50, 16 / 9, 0.1, 2_000);
  camera.position.z = 600;
  camera.updateProjectionMatrix();
  const renderer = {
    coordinateSystem: WebGPUCoordinateSystem,
    info: {
      compute: { frameCalls: 0 },
      frame: 0,
    },
    hasFeature(name) {
      return name === 'indirect-first-instance';
    },
    async compileAsync() {},
  };
  const backendRecords = new WeakMap();
  const originalUpdateTimeStampUID = function updateTimeStampUID(context) {
    const contextData = this.get(context);
    contextData.timestampUID = `r:${renderer.info.compute.frameCalls}:0:f${renderer.info.frame}`;
  };
  renderer.backend = {
    renderer,
    updateTimeStampUID: originalUpdateTimeStampUID,
    get(context) {
      if (!backendRecords.has(context)) {
        backendRecords.set(context, {
          textureDescriptorGPU: { format: 'rgba8unorm' },
        });
      }
      return backendRecords.get(context);
    },
  };
  const strategy = buildFirstInstanceLiveCrossoverStrategy({
    scenario,
    sourceGeometries,
    renderer,
    camera,
    lanePhysicalOrder,
  });
  return {
    camera,
    lanePhysicalOrder,
    originalUpdateTimeStampUID,
    renderer,
    scenario,
    sourceGeometries,
    strategy,
  };
}

function disposeFixture(fixture) {
  disposeStrategyResources({ _attributes: { delete() {} } }, fixture.strategy);
  fixture.sourceGeometries.forEach((geometry) => geometry.dispose());
}

function installCpuReadbackSimulation(fixture, events, {
  reverseFeatureSurvivors = false,
} = {}) {
  const { renderer, scenario, strategy } = fixture;
  const laneForNodes = new Map([
    [strategy.laneStates[PORTABLE].computeNodes, PORTABLE],
    [strategy.laneStates[FEATURE].computeNodes, FEATURE],
  ]);
  renderer.compute = (nodes) => {
    const lane = laneForNodes.get(nodes);
    assert.ok(lane, 'compute submitted an unknown node pair');
    renderer.info.compute.frameCalls += 1;
    renderer.backend.updateTimeStampUID(nodes);
    events.push(`compute:${lane}`);
    const command = strategy.laneStates[lane].indirectAttribute.array;
    for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
      command[bucket * 5 + 1] = scenario.visibleCounts[bucket];
    }
    const visible = strategy.sharedResources.attributes.visibleIds.array;
    visible.fill(0);
    for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
      const ids = [...scenario.expectedVisibleIds].filter(
        (objectId) => scenario.objectBuckets[objectId] === bucket,
      );
      if (lane === FEATURE && reverseFeatureSurvivors) ids.reverse();
      for (const [localIndex, objectId] of ids.entries()) {
        visible[scenario.bucketBases[bucket] + localIndex] = objectId;
      }
    }
    strategy.sharedResources.attributes.overflow.array[0] = 0;
  };
  renderer.getArrayBufferAsync = async (attribute) => attribute.array.slice().buffer;
  renderer.autoClear = true;
  renderer.challengeTarget = null;
  renderer.challengeClearColor = new Color(0x123456);
  renderer.challengeClearAlpha = 0.25;
  renderer.challengePixels = new Uint8Array(
    256 * Math.ceil(scenario.objectCount / 256) * 4,
  );
  renderer.getRenderTarget = () => renderer.challengeTarget;
  renderer.getActiveCubeFace = () => 0;
  renderer.getActiveMipmapLevel = () => 0;
  renderer.setRenderTarget = (target) => {
    renderer.challengeTarget = target;
  };
  renderer.getClearColor = (target) => target.copy(renderer.challengeClearColor);
  renderer.getClearAlpha = () => renderer.challengeClearAlpha;
  renderer.setClearColor = (color, alpha) => {
    renderer.challengeClearColor.set(color);
    renderer.challengeClearAlpha = alpha;
  };
  renderer.clear = () => {
    renderer.challengePixels.fill(0);
  };
  renderer.render = () => {
    const lane = strategy.activeLane;
    const commands = strategy.laneStates[lane].indirectAttribute.array;
    const activeCounts = Uint32Array.from(
      { length: scenario.bucketCount },
      (_, bucket) => commands[bucket * 5 + 1],
    );
    renderer.challengePixels = createFirstInstanceAddressChallengeExpected({
      scenario,
      visibleIds: strategy.sharedResources.attributes.visibleIds.array,
      activeCounts,
    }).bytes;
  };
  renderer.readRenderTargetPixelsAsync = async () => renderer.challengePixels.slice();
}

async function primeFixture(fixture, events) {
  const scene = new Scene();
  scene.add(fixture.strategy.root);
  await fixture.strategy.primeLanes({
    scene,
    render: async (lane) => {
      events.push(`render:${lane}`);
      fixture.strategy.laneStates[lane].meshes[0]
        .onBeforeRender({ _currentRenderBundle: {} });
    },
  });
}

test('live crossover shares cull/output resources and payload but owns two zero-offset commands', () => {
  const fixture = createFixture([FEATURE, PORTABLE]);
  const { scenario, strategy } = fixture;
  try {
    assert.equal(strategy.id, FIRST_INSTANCE_LIVE_CROSSOVER_MODE);
    assert.deepEqual(strategy.lanePhysicalOrder, [FEATURE, PORTABLE]);
    assert.deepEqual(
      strategy.root.children.map((bundle) => bundle.name),
      [
        `${FIRST_INSTANCE_LIVE_CROSSOVER_MODE}-${FEATURE}-bundle`,
        `${FIRST_INSTANCE_LIVE_CROSSOVER_MODE}-${PORTABLE}-bundle`,
      ],
    );
    assert.equal(strategy.storageAttributes.length, 10);
    assert.equal(strategy.computeNodes.length, 4);
    assert.equal(strategy.configuredDrawCommands, scenario.bucketCount);
    assert.equal(strategy.configuredRenderObjects, 1);
    assert.equal(strategy.configuredComputeDispatches, 2);
    assert.equal(strategy.configuredComputeSubmissions, 1);
    assert.equal(strategy.configuredSubmittedInstances, scenario.expectedVisibleCount);

    const portable = strategy.laneStates[PORTABLE];
    const feature = strategy.laneStates[FEATURE];
    assert.notEqual(portable.indirectAttribute, feature.indirectAttribute);
    assert.notEqual(
      strategy.commandBufferCommitments[PORTABLE].attributeId,
      strategy.commandBufferCommitments[FEATURE].attributeId,
    );
    for (const lane of [portable, feature]) {
      assert.equal(lane.commandLayout.offsets[0], 0);
      assert.deepEqual(
        [...lane.commandLayout.offsets],
        Array.from({ length: scenario.bucketCount }, (_, bucket) => bucket * 20),
      );
    }
    for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
      assert.equal(portable.indirectAttribute.array[bucket * 5 + 4], 0);
      assert.equal(
        feature.indirectAttribute.array[bucket * 5 + 4],
        scenario.bucketBases[bucket],
      );
    }
    assert.equal(portable.geometry.index, feature.geometry.index);
    assert.equal(feature.geometry.getAttribute('bucketBase'), undefined);
    for (const name of Object.keys(feature.geometry.attributes)) {
      assert.equal(portable.geometry.getAttribute(name), feature.geometry.getAttribute(name));
    }
    assert.equal(strategy.diagnostics().geometry.pass, true);
    assert.equal(strategy.lifecycleDiagnostics().commandBuffersDistinct, true);
    assert.equal(strategy.lifecycleDiagnostics().commandBuffersZeroOffset, true);
  } finally {
    disposeFixture(fixture);
  }
});

test('physical order controls construction and first compute/render use', async () => {
  const fixture = createFixture([FEATURE, PORTABLE]);
  const events = [];
  installCpuReadbackSimulation(fixture, events);
  try {
    await primeFixture(fixture, events);
    assert.deepEqual(events, [
      `compute:${FEATURE}`,
      `render:${FEATURE}`,
      `compute:${PORTABLE}`,
      `render:${PORTABLE}`,
    ]);
    assert.deepEqual(fixture.strategy.lifecycleDiagnostics().laneConstructionOrder, [
      FEATURE,
      PORTABLE,
    ]);
    assert.deepEqual(fixture.strategy.lifecycleDiagnostics().firstComputeUseOrder, [
      FEATURE,
      PORTABLE,
    ]);
    assert.deepEqual(fixture.strategy.lifecycleDiagnostics().renderPipelinePrimeOrder, [
      FEATURE,
      PORTABLE,
    ]);
    assert.deepEqual(fixture.strategy.lifecycleDiagnostics().bundleRecordCounts, {
      portable: 1,
      feature: 1,
    });
    const lifecycle = fixture.strategy.lifecycleDiagnostics();
    const portableRegistration = lifecycle.computeTimestampRegistrations[PORTABLE];
    const featureRegistration = lifecycle.computeTimestampRegistrations[FEATURE];
    assert.ok(Number.isSafeInteger(lifecycle.computeTimestampContextIds[PORTABLE]));
    assert.ok(Number.isSafeInteger(lifecycle.computeTimestampContextIds[FEATURE]));
    assert.notEqual(
      lifecycle.computeTimestampContextIds[PORTABLE],
      lifecycle.computeTimestampContextIds[FEATURE],
    );
    assert.notEqual(portableRegistration.computeGroupIdentity, featureRegistration.computeGroupIdentity);
    assert.equal(portableRegistration.backendIdentity, featureRegistration.backendIdentity);
    assert.equal(
      portableRegistration.backendWrapperIdentity,
      featureRegistration.backendWrapperIdentity,
    );
    for (const lane of [PORTABLE, FEATURE]) {
      const registration = lifecycle.computeTimestampRegistrations[lane];
      assert.equal(registration.schemaVersion, 1);
      assert.equal(
        registration.kind,
        'live-first-instance-compute-timestamp-group-registration',
      );
      assert.equal(registration.laneId, lane);
      assert.equal(
        registration.timestampContextId,
        lifecycle.computeTimestampContextIds[lane],
      );
      assert.deepEqual(registration.computeNodeIds, lifecycle.computeNodeIds[lane]);
    }
    const timestampUidWrapper = fixture.renderer.backend.updateTimeStampUID;
    assert.notEqual(timestampUidWrapper, fixture.originalUpdateTimeStampUID);
    const before = fixture.strategy.computeCallSerial;
    fixture.strategy.setActiveLane(PORTABLE);
    const submission = fixture.strategy.submitCompute(fixture.renderer);
    assert.deepEqual(submission, {
      schemaVersion: 1,
      kind: 'live-first-instance-compute-group-submission',
      laneId: PORTABLE,
      timestampContextId: portableRegistration.timestampContextId,
      computeGroupIdentity: portableRegistration.computeGroupIdentity,
      registrationSerial: portableRegistration.registrationSerial,
      backendIdentity: portableRegistration.backendIdentity,
      backendWrapperIdentity: portableRegistration.backendWrapperIdentity,
      computeNodeIds: lifecycle.computeNodeIds[PORTABLE],
      computeCallSerial: before + 1,
    });
    assert.equal(fixture.strategy.computeCallSerial, before + 1);
    assert.equal(fixture.renderer.backend.updateTimeStampUID, timestampUidWrapper);
    assert.equal(
      fixture.renderer.backend.get(fixture.strategy.laneStates[PORTABLE].computeNodes).timestampUID,
      `c:${fixture.renderer.info.compute.frameCalls}`
        + `:${portableRegistration.timestampContextId}:f${fixture.renderer.info.frame}`,
    );
    assert.equal(events.at(-1), `compute:${PORTABLE}`);
  } finally {
    disposeFixture(fixture);
  }
});

test('compute timestamp registration rejects a replaced backend wrapper', async () => {
  const fixture = createFixture([PORTABLE, FEATURE]);
  const events = [];
  installCpuReadbackSimulation(fixture, events);
  let timestampUidWrapper = null;
  try {
    await primeFixture(fixture, events);
    timestampUidWrapper = fixture.renderer.backend.updateTimeStampUID;
    fixture.renderer.backend.updateTimeStampUID = function replacedUpdateTimeStampUID() {};
    assert.throws(
      () => registerComputeTimestampGroup(
        fixture.renderer,
        [],
        Number.MAX_SAFE_INTEGER,
      ),
      /backend wrapper was replaced/,
    );
  } finally {
    if (timestampUidWrapper) {
      fixture.renderer.backend.updateTimeStampUID = timestampUidWrapper;
    }
    disposeFixture(fixture);
  }
});

test('serialized validation preserves each lane and fails closed before shader capture', async () => {
  const fixture = createFixture([PORTABLE, FEATURE]);
  const events = [];
  installCpuReadbackSimulation(fixture, events);
  try {
    await primeFixture(fixture, events);
    events.length = 0;
    const computeStart = fixture.strategy.computeCallSerial;
    const validation = await fixture.strategy.validateSerialized(
      fixture.renderer,
      fixture.scenario.expectedVisibleIds,
      {
        onLanePrepared({ lane, resources }) {
          events.push(`capture:${lane}`);
          assert.equal(resources.latestSnapshot instanceof Object, true);
          assert.equal(fixture.strategy.residentPreparedLane, lane);
          return { pass: true, lane };
        },
      },
    );
    assert.equal(validation.pass, false);
    assert.equal(validation.shaderEvidencePass, false);
    assert.equal(validation.kind, 'first-instance-live-crossover-exact-paired-snapshots');
    assert.deepEqual(validation.validationOrder, [PORTABLE, FEATURE]);
    assert.equal(validation.computeCallSerialStart, computeStart);
    assert.equal(validation.computeCallSerialEnd, computeStart + 2);
    assert.equal(validation.exactlyOneComputePerLane, true);
    assert.equal(validation.commandCoresEqual, true);
    assert.equal(validation.canonicalMembershipEqual, true);
    assert.equal(validation.rawSurvivorOrderRequiredEqual, false);
    assert.deepEqual(events, [
      `compute:${PORTABLE}`,
      `capture:${PORTABLE}`,
      `compute:${FEATURE}`,
      `capture:${FEATURE}`,
    ]);
    assert.equal(validation.lanes[PORTABLE].commandValidation.pass, true);
    assert.equal(validation.lanes[FEATURE].commandValidation.pass, true);
    assert.equal(validation.lanes[PORTABLE].commandValidation.records[0].actual.firstInstance, 0);
    assert.equal(
      validation.lanes[FEATURE].commandValidation.records[1].actual.firstInstance,
      fixture.scenario.bucketBases[1],
    );
    assert.equal(fixture.strategy.residentPreparedLane, FEATURE);
  } finally {
    disposeFixture(fixture);
  }
});

test('address oracle accepts lane-local survivor order while the shader gate remains closed', async () => {
  const fixture = createFixture([PORTABLE, FEATURE]);
  const events = [];
  installCpuReadbackSimulation(fixture, events, { reverseFeatureSurvivors: true });
  try {
    await primeFixture(fixture, events);
    const validation = await fixture.strategy.validateSerialized(
      fixture.renderer,
      fixture.scenario.expectedVisibleIds,
      { camera: fixture.camera },
    );
    assert.equal(validation.pass, false);
    assert.equal(validation.shaderEvidencePass, false);
    assert.equal(validation.addressChallenges.pass, true);
    assert.equal(validation.addressChallenges.rawAddressBytesRequiredEqual, false);
    assert.equal(validation.addressChallenges.rawSurvivorBytesRequiredEqual, false);
    const portable = validation.hooks[PORTABLE].addressChallenge;
    const feature = validation.hooks[FEATURE].addressChallenge;
    assert.equal(portable.pass, true);
    assert.equal(feature.pass, true);
    assert.equal(portable.reset.pass, true);
    assert.equal(feature.reset.pass, true);
    assert.notEqual(portable.survivorSha256, feature.survivorSha256);
    assert.notEqual(portable.sha256, feature.sha256);
    for (const [lane, challenge] of [[PORTABLE, portable], [FEATURE, feature]]) {
      assert.equal(challenge.lane, lane);
      assert.equal(challenge.commandByteOffset, 0);
      assert.equal(
        challenge.commandBufferId,
        fixture.strategy.commandBufferCommitments[lane].attributeId,
      );
      assert.equal(challenge.exactExpectedBytes, true);
      assert.equal(challenge.activeAddressCount, fixture.scenario.expectedVisibleCount);
      assert.equal(
        challenge.paddingAddressCount,
        fixture.scenario.objectCount - fixture.scenario.expectedVisibleCount,
      );
      assert.equal(challenge.activeMismatchCount, 0);
      assert.equal(challenge.paddingMismatchCount, 0);
      assert.equal(challenge.targetPaddingMismatchCount, 0);
      assert.equal(challenge.outOfRangeIds, 0);
    }
  } finally {
    disposeFixture(fixture);
  }
});

test('live address oracle cannot pass a stale target when the explicit clear is ineffective', async () => {
  const fixture = createFixture([PORTABLE, FEATURE]);
  const events = [];
  installCpuReadbackSimulation(fixture, events);
  try {
    await primeFixture(fixture, events);
    fixture.renderer.challengePixels.fill(0x7f);
    fixture.renderer.clear = () => {};
    const validation = await fixture.strategy.validateSerialized(
      fixture.renderer,
      fixture.scenario.expectedVisibleIds,
      { camera: fixture.camera },
    );
    assert.equal(validation.pass, false);
    assert.equal(validation.hooks[PORTABLE].addressChallenge.reset.pass, false);
    assert.equal(validation.hooks[PORTABLE].addressChallenge.exactExpectedBytes, true);
    assert.equal(validation.hooks[FEATURE].addressChallenge.reset.pass, false);
  } finally {
    disposeFixture(fixture);
  }
});

test('forced feature-off deployment selection constructs portable state only', () => {
  const sourceGeometries = createIndexedGeometryFixtures(4, 'low');
  const scenario = createFixedSubsetScenario({
    objectCount: 512,
    bucketCount: 4,
    visibilityFraction: 0.2,
    geometrySpheres: sourceGeometries.map((geometry) => geometry.boundingSphere),
    seed: 0xb1ad_2026,
  });
  const renderer = { hasFeature: () => true };
  let strategy;
  try {
    assert.deepEqual(
      selectFixedSliceDeployment({ renderer, featureAvailable: false }),
      { lane: PORTABLE, strategyId: 'fixed-slice', featureAvailable: false },
    );
    strategy = buildFixedSliceDeploymentStrategy(
      { scenario, sourceGeometries, renderer },
      { featureAvailable: false },
    );
    assert.equal(strategy.id, 'fixed-slice');
    assert.equal(strategy.laneState.lane, PORTABLE);
    assert.equal(strategy.geometries.length, 1);
    assert.ok(strategy.geometries[0].getAttribute('bucketBase'));
    assert.equal(
      Object.prototype.hasOwnProperty.call(
        strategy.sharedResources.geometriesByAddressMode,
        'indirect-first-instance',
      ),
      false,
    );
    for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
      assert.equal(strategy.laneState.indirectAttribute.array[bucket * 5 + 4], 0);
    }
  } finally {
    if (strategy) disposeStrategyResources({ _attributes: { delete() {} } }, strategy);
    sourceGeometries.forEach((geometry) => geometry.dispose());
  }
});
