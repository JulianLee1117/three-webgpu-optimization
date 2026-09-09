import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import { createFixedSubsetScenario } from '../src/scenes/fixed-subsets.js';
import { STORAGE_TRANSFORM_ADDRESS_MODES } from '../src/materials/storage-transform.js';
import {
  IMMEDIATE_AIF_PHASE0_COUNT_CLASS_GROUPS,
  IMMEDIATE_AIF_PHASE0_LANES,
  IMMEDIATE_AIF_PHASE0_LANE_ORDERS,
  IMMEDIATE_AIF_PHASE0_PLAN,
  IMMEDIATE_AIF_PHASE0_SCHEDULES,
  IMMEDIATE_AIF_PHASE0_WORKLOAD,
  IMMEDIATE_AIF_UNUSED_ADDRESS,
  createImmediateAifAddressOracle,
  createImmediateAifCommandOracle,
  createImmediateAifPhase0Scenarios,
} from '../src/phase0/immediate-aif-plan.js';
import {
  cloneImmediateAifPhase0ScenarioSet,
  validateExactImmediateAifPhase0ScenarioSet,
  validateExactImmediateAifPhase0Workload,
} from '../src/phase0/immediate-aif-workload-validation.js';
import {
  createFixedSliceAddressLane,
  createFixedSliceLane,
  createFixedSliceSharedResources,
} from '../src/strategies/fixed-slice.js';

const [A, I, F] = IMMEDIATE_AIF_PHASE0_LANES;
const {
  BUCKET_BASE,
  IMMEDIATE_BASE,
  INDIRECT_FIRST_INSTANCE,
} = STORAGE_TRANSFORM_ADDRESS_MODES;

function commandWords(commands, bucket) {
  return commands.subarray(bucket * 5, bucket * 5 + 5);
}

function countDescendingEdges(values) {
  let count = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] < values[index - 1]) count += 1;
  }
  return count;
}

test('Phase 0 plan freezes the exact A/I/F workload, orders, and safe schedules', () => {
  assert.equal(IMMEDIATE_AIF_PHASE0_PLAN.executionMode, 'technical-canary');
  assert.equal(IMMEDIATE_AIF_PHASE0_PLAN.analysisEligible, false);
  assert.equal(IMMEDIATE_AIF_PHASE0_PLAN.efficacyAnalysisAllowed, false);
  assert.equal(IMMEDIATE_AIF_PHASE0_PLAN.numericalDecision, null);
  assert.equal(Object.isFrozen(IMMEDIATE_AIF_PHASE0_PLAN), true);
  assert.deepEqual(IMMEDIATE_AIF_PHASE0_LANES, ['A', 'I', 'F']);
  assert.deepEqual(IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.viewport,
    { width: 1_280, height: 720, devicePixelRatio: 1 });
  assert.deepEqual(IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.camera, {
    type: 'PerspectiveCamera',
    fov: 50,
    aspect: 16 / 9,
    near: 0.1,
    far: 1_000,
    zoom: 1,
    position: [0, 0, 140],
    target: [0, 0, 0],
    targetDistance: 140,
    targetDirection: [0, 0, -1],
    worldDirection: [-0, -0, -1],
    up: [0, 1, 0],
    quaternion: [0, 0, 0, 1],
    scale: [1, 1, 1],
    coordinateSystem: 2_001,
    reversedDepth: true,
    matrixWorld: [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 140, 1,
    ],
    matrixWorldInverse: [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, -140, 1,
    ],
    projectionMatrix: [
      1.2062851427866268, 0, 0, 0,
      0, 2.1445069205095586, 0, 0,
      0, 0, 0.00010001000100010001, -1,
      0, 0, 0.10001000100010002, 0,
    ],
    projectionMatrixInverse: [
      0.8289913922755531, 0, 0, 0,
      0, 0.46630765815499864, 0, 0,
      0, 0, 0, 9.998999999999999,
      0, 0, -1, 0.001,
    ],
  });
  assert.equal(
    IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.renderer.reversedDepthBuffer,
    true,
  );
  assert.deepEqual({
    antialias: IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.renderer.antialias,
    samples: IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.renderer.samples,
    powerPreference:
      IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.renderer.powerPreference,
    trackTimestamp:
      IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.renderer.trackTimestamp,
    autoClear: IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.renderer.autoClear,
    toneMapping: IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.renderer.toneMapping,
  }, {
    antialias: false,
    samples: 0,
    powerPreference: 'high-performance',
    trackTimestamp: false,
    autoClear: true,
    toneMapping: 0,
  });
  assert.equal(
    IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.targets.production.depthFormat,
    'depth32float',
  );
  assert.equal(
    IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.targets.objectId.depthFormat,
    'depth32float',
  );
  const materialConfiguration = IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.material;
  assert.deepEqual({
    type: materialConfiguration.type,
    color: materialConfiguration.color,
    roughness: materialConfiguration.roughness,
    metalness: materialConfiguration.metalness,
    emissive: materialConfiguration.emissive,
    opacity: materialConfiguration.opacity,
    transparent: materialConfiguration.transparent,
    depthTest: materialConfiguration.depthTest,
    depthWrite: materialConfiguration.depthWrite,
    depthFunc: materialConfiguration.depthFunc,
    blending: materialConfiguration.blending,
    blendSrc: materialConfiguration.blendSrc,
    blendDst: materialConfiguration.blendDst,
    blendEquation: materialConfiguration.blendEquation,
    alphaHash: materialConfiguration.alphaHash,
    alphaToCoverage: materialConfiguration.alphaToCoverage,
    stencilWrite: materialConfiguration.stencilWrite,
    fog: materialConfiguration.fog,
    lights: materialConfiguration.lights,
    vertexColors: materialConfiguration.vertexColors,
    toneMapped: materialConfiguration.toneMapped,
    wireframe: materialConfiguration.wireframe,
    alphaTest: materialConfiguration.alphaTest,
    forceSinglePass: materialConfiguration.forceSinglePass,
    visible: materialConfiguration.visible,
  }, {
    type: 'MeshStandardNodeMaterial',
    color: 0x68a7f2,
    roughness: 0.68,
    metalness: 0.08,
    emissive: 0,
    opacity: 1,
    transparent: false,
    depthTest: true,
    depthWrite: true,
    depthFunc: 3,
    blending: 1,
    blendSrc: 204,
    blendDst: 205,
    blendEquation: 100,
    alphaHash: false,
    alphaToCoverage: false,
    stencilWrite: false,
    fog: true,
    lights: true,
    vertexColors: false,
    toneMapped: true,
    wireframe: false,
    alphaTest: 0,
    forceSinglePass: false,
    visible: true,
  });
  assert.equal(Object.values(materialConfiguration.maps).every((value) => value === null), true);
  assert.deepEqual(IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.lights.hemisphere, {
    type: 'HemisphereLight',
    name: 'phase0-hemisphere-light',
    skyColor: 0x9bc5ff,
    groundColor: 0x182038,
    intensity: 2.2,
    position: [0, 1, 0],
    worldPosition: [0, 1, 0],
    quaternion: [0, 0, 0, 1],
    scale: [1, 1, 1],
    matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    matrixWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    matrixAutoUpdate: false,
    matrixWorldAutoUpdate: false,
    visible: true,
    layersMask: 1,
    castShadow: false,
    receiveShadow: false,
    frustumCulled: true,
    renderOrder: 0,
    effectiveNormalizedDirection: [0, 1, 0],
  });
  const directional = IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.lights.directional;
  assert.deepEqual(directional.position, [20, 35, 60]);
  assert.deepEqual(directional.worldPosition, [20, 35, 60]);
  assert.deepEqual(directional.target.position, [0, 0, 0]);
  assert.deepEqual(directional.target.worldPosition, [0, 0, 0]);
  assert.equal(directional.target.addedToProductionScene, true);
  assert.deepEqual(directional.targetDirection, [
    -0.27668578554642986,
    -0.48420012470625223,
    -0.8300573566392895,
  ]);
  assert.deepEqual(directional.effectiveNormalizedDirection, [
    0.27668578554642986,
    0.48420012470625223,
    0.8300573566392895,
  ]);
  assert.equal(directional.matrixAutoUpdate, false);
  assert.equal(directional.matrixWorldAutoUpdate, false);
  assert.equal(directional.layersMask, 1);
  assert.equal(directional.castShadow, false);
  assert.equal(directional.target.matrixAutoUpdate, false);
  assert.equal(directional.target.matrixWorldAutoUpdate, false);
  assert.equal(directional.target.layersMask, 1);
  for (const target of Object.values(
    IMMEDIATE_AIF_PHASE0_PLAN.renderConfiguration.targets,
  )) {
    assert.equal(target.resolveDepthBuffer, true);
    assert.equal(target.resolveStencilBuffer, true);
    assert.equal(target.gpuColorDescriptor.usage, 23);
    assert.equal(target.gpuColorDescriptor.sampleCount, 1);
  }

  const encodedOrders = IMMEDIATE_AIF_PHASE0_LANE_ORDERS.map((order) => order.join(''));
  assert.deepEqual(encodedOrders, ['AIF', 'AFI', 'FIA', 'IAF', 'IFA', 'FAI']);
  assert.equal(new Set(encodedOrders).size, 6);
  for (const order of IMMEDIATE_AIF_PHASE0_LANE_ORDERS) {
    assert.deepEqual([...order].sort(), ['A', 'F', 'I']);
    assert.equal(Object.isFrozen(order), true);
  }

  assert.deepEqual(IMMEDIATE_AIF_PHASE0_WORKLOAD, {
    objectCount: 65_536,
    bucketCount: 32,
    geometryTier: 'medium',
    layout: 'baseline',
    seed: 2_980_913_190,
    bucketCapacity: 2_048,
    bucketCounts: Array.from({ length: 32 }, () => 2_048),
    bucketBases: Array.from({ length: 32 }, (_, bucket) => bucket * 2_048),
    drawCount: 32,
    indirectStrideUints: 5,
    indirectStrideBytes: 20,
    commandUint32Length: 160,
    commandByteLength: 640,
    indirectOffsets: Array.from({ length: 32 }, (_, bucket) => bucket * 20),
    visibilityLevels: [
      {
        id: 'v99',
        fraction: 0.99,
        expectedVisibleCount: 64_881,
        visibleCounts: [
          ...Array.from({ length: 17 }, () => 2_028),
          ...Array.from({ length: 15 }, () => 2_027),
        ],
      },
      {
        id: 'v20',
        fraction: 0.2,
        expectedVisibleCount: 13_107,
        visibleCounts: [
          ...Array.from({ length: 19 }, () => 410),
          ...Array.from({ length: 13 }, () => 409),
        ],
      },
    ],
  });
  assert.equal(IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketBases.at(-1), 63_488);
  assert.equal(IMMEDIATE_AIF_PHASE0_WORKLOAD.indirectOffsets.at(-1), 620);
  assert.deepEqual(IMMEDIATE_AIF_PHASE0_COUNT_CLASS_GROUPS, [
    Array.from({ length: 17 }, (_, bucket) => bucket),
    [17, 18],
    Array.from({ length: 13 }, (_, index) => index + 19),
  ]);

  assert.deepEqual(
    IMMEDIATE_AIF_PHASE0_SCHEDULES.S1.sourceBucketByDraw,
    [
      16,
      ...Array.from({ length: 16 }, (_, index) => index),
      18,
      17,
      31,
      ...Array.from({ length: 12 }, (_, index) => index + 19),
    ],
  );
  assert.deepEqual(
    IMMEDIATE_AIF_PHASE0_SCHEDULES.S2.sourceBucketByDraw,
    [
      ...Array.from({ length: 16 }, (_, index) => index + 1),
      0,
      18,
      17,
      ...Array.from({ length: 12 }, (_, index) => index + 20),
      19,
    ],
  );
  for (const schedule of [
    IMMEDIATE_AIF_PHASE0_SCHEDULES.S1,
    IMMEDIATE_AIF_PHASE0_SCHEDULES.S2,
  ]) {
    assert.equal(
      schedule.sourceBucketByDraw.every((sourceBucket, draw) => sourceBucket !== draw),
      true,
      `${schedule.id} must be fully deranged`,
    );
    assert.ok(countDescendingEdges(schedule.sourceBaseByDraw) > 0);
  }
  assert.notDeepEqual(
    IMMEDIATE_AIF_PHASE0_SCHEDULES.S1.sourceBucketByDraw,
    IMMEDIATE_AIF_PHASE0_SCHEDULES.S2.sourceBucketByDraw,
  );

  for (const level of IMMEDIATE_AIF_PHASE0_WORKLOAD.visibilityLevels) {
    for (const schedule of Object.values(IMMEDIATE_AIF_PHASE0_SCHEDULES)) {
      assert.deepEqual([...schedule.sourceBucketByDraw].sort((left, right) => left - right),
        Array.from({ length: 32 }, (_, bucket) => bucket));
      for (let draw = 0; draw < 32; draw += 1) {
        assert.equal(
          level.visibleCounts[schedule.sourceBucketByDraw[draw]],
          level.visibleCounts[draw],
          `${schedule.id} changes the ${level.id} count class for draw ${draw}`,
        );
      }
    }
  }
});

test('Phase 0 scenario, command, and address oracles preserve the frozen invariants', () => {
  const sourceGeometries = createIndexedGeometryFixtures(32, 'medium');
  try {
    const scenarios = createImmediateAifPhase0Scenarios({
      geometrySpheres: sourceGeometries.map((geometry) => geometry.boundingSphere),
    });
    assert.deepEqual(Object.keys(scenarios), ['v99', 'v20']);

    for (const level of IMMEDIATE_AIF_PHASE0_WORKLOAD.visibilityLevels) {
      const scenario = scenarios[level.id];
      assert.equal(scenario.objectCount, 65_536);
      assert.equal(scenario.bucketCount, 32);
      assert.equal(scenario.layout, 'baseline');
      assert.deepEqual([...scenario.bucketCounts], IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCounts);
      assert.deepEqual([...scenario.bucketBases], IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketBases);
      assert.deepEqual([...scenario.visibleCounts], level.visibleCounts);
      assert.equal(scenario.expectedVisibleCount, level.expectedVisibleCount);

      const canonicalAddresses = createImmediateAifAddressOracle({
        lane: A,
        scenario,
      });
      for (const scheduleId of ['canonical', 'S1', 'S2']) {
        const commandsByLane = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
          lane,
          createImmediateAifCommandOracle({
            lane,
            scenario,
            sourceGeometries,
            scheduleId,
          }),
        ]));
        for (let bucket = 0; bucket < 32; bucket += 1) {
          const aInitial = commandWords(commandsByLane.A.initialCommands, bucket);
          const iInitial = commandWords(commandsByLane.I.initialCommands, bucket);
          const fInitial = commandWords(commandsByLane.F.initialCommands, bucket);
          const aPostCull = commandWords(commandsByLane.A.postCullCommands, bucket);
          const iPostCull = commandWords(commandsByLane.I.postCullCommands, bucket);
          const fPostCull = commandWords(commandsByLane.F.postCullCommands, bucket);
          assert.deepEqual([...aInitial.subarray(0, 4)], [...iInitial.subarray(0, 4)]);
          assert.deepEqual([...aInitial.subarray(0, 4)], [...fInitial.subarray(0, 4)]);
          assert.equal(aInitial[1], 0);
          assert.equal(iInitial[1], 0);
          assert.equal(fInitial[1], 0);
          assert.deepEqual([...aPostCull.subarray(0, 4)], [...iPostCull.subarray(0, 4)]);
          assert.deepEqual([...aPostCull.subarray(0, 4)], [...fPostCull.subarray(0, 4)]);
          assert.equal(aPostCull[1], scenario.visibleCounts[bucket]);
          assert.equal(aInitial[4], 0);
          assert.equal(iInitial[4], 0);
          assert.equal(aPostCull[4], 0);
          assert.equal(iPostCull[4], 0);
          assert.equal(fInitial[4], commandsByLane.F.sourceBaseByDraw[bucket]);
          assert.equal(fPostCull[4], commandsByLane.F.sourceBaseByDraw[bucket]);
        }
        assert.equal(commandsByLane.A.initialCommands.byteLength, 640);
        assert.equal(commandsByLane.I.postCullCommands.byteLength, 640);
        assert.deepEqual([...commandsByLane.F.offsets],
          IMMEDIATE_AIF_PHASE0_WORKLOAD.indirectOffsets);
        assert.equal(commandsByLane.A.bucketBaseAttributeByDraw,
          commandsByLane.A.sourceBaseByDraw);
        assert.equal(commandsByLane.I.indirectImmediateBases,
          commandsByLane.I.sourceBaseByDraw);
        assert.equal(commandsByLane.F.indirectFirstInstances,
          commandsByLane.F.sourceBaseByDraw);
        assert.equal(commandsByLane.A.indirectImmediateBases, null);
        assert.equal(commandsByLane.F.indirectImmediateBases, null);

        const addressesByLane = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [
          lane,
          createImmediateAifAddressOracle({ lane, scenario, scheduleId }),
        ]));
        for (const lane of [I, F]) {
          assert.deepEqual(addressesByLane[lane].activeTargetAddresses,
            addressesByLane.A.activeTargetAddresses);
          assert.deepEqual(addressesByLane[lane].activeSourceAddresses,
            addressesByLane.A.activeSourceAddresses);
          assert.deepEqual(addressesByLane[lane].objectIds, addressesByLane.A.objectIds);
          assert.deepEqual(addressesByLane[lane].expectedOutputObjectIds,
            addressesByLane.A.expectedOutputObjectIds);
        }
        assert.deepEqual(
          addressesByLane.A.activeTargetAddresses,
          canonicalAddresses.activeTargetAddresses,
          'the output oracle must remain in canonical draw slots',
        );
        assert.equal(new Set(addressesByLane.A.activeSourceAddresses).size,
          level.expectedVisibleCount);
        assert.equal(
          addressesByLane.A.expectedOutputObjectIds.filter(
            (value) => value !== IMMEDIATE_AIF_UNUSED_ADDRESS,
          ).length,
          level.expectedVisibleCount,
        );
        for (let invocation = 0;
          invocation < addressesByLane.A.activeSourceAddresses.length;
          invocation += 1) {
          assert.equal(
            addressesByLane.A.builtinInstanceIndices[invocation]
              + addressesByLane.A.baseOperands[invocation],
            addressesByLane.A.activeSourceAddresses[invocation],
          );
          assert.equal(
            addressesByLane.F.builtinInstanceIndices[invocation],
            addressesByLane.F.activeSourceAddresses[invocation],
          );
          assert.equal(addressesByLane.F.baseOperands[invocation], 0);
        }
        if (scheduleId !== 'canonical') {
          assert.notDeepEqual(addressesByLane.A.activeSourceAddresses,
            canonicalAddresses.activeSourceAddresses);
          assert.notDeepEqual(addressesByLane.A.expectedOutputObjectIds,
            canonicalAddresses.expectedOutputObjectIds);
        }
      }
    }
  } finally {
    sourceGeometries.forEach((geometry) => geometry.dispose());
  }
});

test('Phase 0 exact workload gate rejects tier drift and every mutable workload channel', () => {
  const sourceGeometries = createIndexedGeometryFixtures(32, 'medium');
  const lowGeometries = createIndexedGeometryFixtures(32, 'low');
  try {
    const canonicalScenarios = createImmediateAifPhase0Scenarios({
      geometrySpheres: sourceGeometries.map((geometry) => geometry.boundingSphere),
    });
    const accepted = validateExactImmediateAifPhase0Workload({
      scenarios: canonicalScenarios,
      sourceGeometries,
    });
    assert.equal(accepted.pass, true);
    accepted.ownedGeometries.forEach((geometry) => geometry.dispose());

    assert.throws(
      () => validateExactImmediateAifPhase0Workload({
        scenarios: canonicalScenarios,
        sourceGeometries: lowGeometries,
      }),
      /canonical medium Phase 0 fixture/,
    );

    for (const name of [
      'bucketCounts',
      'bucketBases',
      'visibleCounts',
      'objectBuckets',
      'cullOrder',
      'matrices',
      'bounds',
      'expectedVisibleIds',
    ]) {
      const changed = cloneImmediateAifPhase0ScenarioSet(canonicalScenarios);
      changed.v99[name][0] = changed.v99[name][0] + 1;
      assert.throws(
        () => validateExactImmediateAifPhase0ScenarioSet(changed, canonicalScenarios),
      );
    }

    for (const name of ['index', 'position', 'normal', 'uv']) {
      const attribute = name === 'index'
        ? sourceGeometries[0].index
        : sourceGeometries[0].getAttribute(name);
      const original = attribute.array[0];
      attribute.array[0] = original + 1;
      assert.throws(
        () => validateExactImmediateAifPhase0Workload({
          scenarios: canonicalScenarios,
          sourceGeometries,
        }),
        /canonical medium Phase 0 fixture/,
        `${name} byte drift must fail closed`,
      );
      attribute.array[0] = original;
    }

    const originalName = sourceGeometries[0].name;
    sourceGeometries[0].name = `${originalName}-spoof`;
    assert.throws(
      () => validateExactImmediateAifPhase0Workload({
        scenarios: canonicalScenarios,
        sourceGeometries,
      }),
      /canonical medium Phase 0 fixture/,
    );
    sourceGeometries[0].name = originalName;
  } finally {
    sourceGeometries.forEach((geometry) => geometry.dispose());
    lowGeometries.forEach((geometry) => geometry.dispose());
  }
});

test('generic fixed-slice address lanes isolate A/I/F geometry and initialize word four', () => {
  const sourceGeometries = createIndexedGeometryFixtures(4, 'low');
  const scenario = createFixedSubsetScenario({
    objectCount: 512,
    bucketCount: 4,
    visibilityFraction: 0.2,
    geometrySpheres: sourceGeometries.map((geometry) => geometry.boundingSphere),
    seed: 0xb1ad_2026,
  });
  const shared = createFixedSliceSharedResources(
    { scenario, sourceGeometries },
    { addressModes: [BUCKET_BASE, IMMEDIATE_BASE, INDIRECT_FIRST_INSTANCE] },
  );
  const immediateBases = scenario.bucketBases.slice();
  const lanes = [];
  try {
    const attribute = createFixedSliceAddressLane(shared, { addressMode: BUCKET_BASE });
    const immediate = createFixedSliceAddressLane(shared, {
      addressMode: IMMEDIATE_BASE,
      indirectImmediateBases: immediateBases,
    });
    const firstInstance = createFixedSliceAddressLane(shared, {
      addressMode: INDIRECT_FIRST_INSTANCE,
    });
    lanes.push(attribute, immediate, firstInstance);

    assert.equal(shared.ownedGeometries.length, 3);
    assert.equal(new Set(shared.ownedGeometries).size, 3);
    assert.notEqual(attribute.geometry, immediate.geometry);
    assert.notEqual(immediate.geometry, firstInstance.geometry);
    assert.notEqual(attribute.geometry, firstInstance.geometry);
    assert.ok(attribute.geometry.getAttribute('bucketBase'));
    assert.equal(immediate.geometry.getAttribute('bucketBase'), undefined);
    assert.equal(firstInstance.geometry.getAttribute('bucketBase'), undefined);
    assert.equal(attribute.geometry.index, immediate.geometry.index);
    assert.equal(immediate.geometry.index, firstInstance.geometry.index);
    for (const name of Object.keys(immediate.geometry.attributes)) {
      assert.equal(immediate.geometry.getAttribute(name), firstInstance.geometry.getAttribute(name));
    }

    assert.equal(immediate.indirectImmediateBases, immediateBases);
    assert.equal(immediate.indirectImmediateBases.length, 4);
    assert.equal(Object.hasOwn(attribute, 'indirectImmediateBases'), false);
    assert.equal(Object.hasOwn(firstInstance, 'indirectImmediateBases'), false);
    assert.deepEqual(immediate.geometry.indirectOffset, [0, 20, 40, 60]);
    for (let bucket = 0; bucket < 4; bucket += 1) {
      assert.equal(attribute.commandLayout.commands[bucket * 5 + 4], 0);
      assert.equal(immediate.commandLayout.commands[bucket * 5 + 4], 0);
      assert.equal(
        firstInstance.commandLayout.commands[bucket * 5 + 4],
        scenario.bucketBases[bucket],
      );
    }
    assert.throws(
      () => createFixedSliceAddressLane(shared, {
        addressMode: BUCKET_BASE,
        indirectImmediateBases: immediateBases,
      }),
      /valid only for immediate-base/,
    );
    assert.throws(
      () => createFixedSliceAddressLane(shared, {
        addressMode: IMMEDIATE_BASE,
        indirectImmediateBases: new Uint32Array(3),
      }),
      /Uint32Array of length 4/,
    );
    assert.throws(
      () => createFixedSliceAddressLane(shared, {
        addressMode: IMMEDIATE_BASE,
        indirectImmediateBases: immediateBases,
        perBucketRenderObjects: true,
      }),
      /one-render-object indirect loop/,
    );
  } finally {
    for (const lane of lanes) {
      lane.root.removeFromParent();
      lane.material.dispose();
    }
    shared.ownedGeometries.forEach((geometry) => geometry.dispose());
    sourceGeometries.forEach((geometry) => geometry.dispose());
  }
});

test('legacy fixed-slice lane wrapper remains portable/feature compatible', () => {
  const sourceGeometries = createIndexedGeometryFixtures(4, 'low');
  const scenario = createFixedSubsetScenario({
    objectCount: 512,
    bucketCount: 4,
    visibilityFraction: 0.99,
    geometrySpheres: sourceGeometries.map((geometry) => geometry.boundingSphere),
    seed: 0xb1ad_2026,
  });
  const shared = createFixedSliceSharedResources(
    { scenario, sourceGeometries },
    { addressModes: [BUCKET_BASE, INDIRECT_FIRST_INSTANCE] },
  );
  const lanes = [];
  try {
    const portable = createFixedSliceLane(shared, { lane: 'portable' });
    const feature = createFixedSliceLane(shared, { lane: 'feature' });
    lanes.push(portable, feature);
    assert.equal(portable.id, 'fixed-slice');
    assert.equal(portable.lane, 'portable');
    assert.equal(portable.addressMode, BUCKET_BASE);
    assert.equal(feature.id, 'fixed-slice-indirect-first-instance');
    assert.equal(feature.lane, 'feature');
    assert.equal(feature.addressMode, INDIRECT_FIRST_INSTANCE);
    assert.equal(Object.hasOwn(portable, 'indirectImmediateBases'), false);
    assert.equal(Object.hasOwn(feature, 'indirectImmediateBases'), false);
    assert.deepEqual(portable.commandBufferCommitment().allOffsets, [0, 20, 40, 60]);
    assert.equal(portable.commandBufferCommitment().lane, 'portable');
    assert.equal(feature.commandBufferCommitment().lane, 'feature');
    for (let bucket = 0; bucket < 4; bucket += 1) {
      assert.equal(portable.commandLayout.commands[bucket * 5 + 4], 0);
      assert.equal(feature.commandLayout.commands[bucket * 5 + 4], scenario.bucketBases[bucket]);
    }
  } finally {
    for (const lane of lanes) lane.material.dispose();
    shared.ownedGeometries.forEach((geometry) => geometry.dispose());
    sourceGeometries.forEach((geometry) => geometry.dispose());
  }
});

test('fixed-slice reset and cull nodes never read or write firstInstance', async () => {
  const filename = fileURLToPath(new URL('../src/strategies/fixed-slice.js', import.meta.url));
  const source = await readFile(filename, 'utf8');
  const start = source.indexOf('function createFixedSliceComputeNodes');
  const end = source.indexOf('function validateImmediateBases', start);
  assert.ok(start >= 0 && end > start);
  const computeSource = source.slice(start, end);
  assert.doesNotMatch(computeSource, /firstInstance/);
  assert.match(computeSource, /get\('instanceCount'\)/);
});
