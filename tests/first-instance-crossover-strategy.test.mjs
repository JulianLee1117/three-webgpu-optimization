import assert from 'node:assert/strict';
import test from 'node:test';
import { Color, PerspectiveCamera, Scene } from 'three/webgpu';
import { FIRST_INSTANCE_COMMAND_LANES } from '../src/culling/first-instance-crossover.js';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import { buildFirstInstanceCrossoverStrategy } from '../src/strategies/first-instance-crossover.js';
import { disposeStrategyResources } from '../src/strategies/resources.js';

const [PORTABLE, FEATURE] = FIRST_INSTANCE_COMMAND_LANES;

function scenarioFixture() {
  const objectCount = 9;
  const matrices = new Float32Array(objectCount * 16);
  for (let objectId = 0; objectId < objectCount; objectId += 1) {
    const base = objectId * 16;
    matrices[base] = 1;
    matrices[base + 5] = 1;
    matrices[base + 10] = 1;
    matrices[base + 15] = 1;
  }
  return {
    objectCount,
    bucketCount: 3,
    bucketBases: Uint32Array.of(0, 3, 5),
    bucketCounts: Uint32Array.of(3, 2, 4),
    visibleCounts: Uint32Array.of(2, 1, 3),
    objectBuckets: Uint32Array.of(0, 0, 0, 1, 1, 2, 2, 2, 2),
    expectedVisibleIds: Uint32Array.of(0, 2, 3, 5, 7, 8),
    expectedVisibleCount: 6,
    matrices,
  };
}

function cameraFixture() {
  const camera = new PerspectiveCamera(50, 1, 0.1, 100);
  camera.position.z = 10;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  return camera;
}

function buildFixture(laneCommandSegmentOrder = [FEATURE, PORTABLE]) {
  const scenario = scenarioFixture();
  const sourceGeometries = createIndexedGeometryFixtures(scenario.bucketCount, 'low');
  const renderer = {
    hasFeature(name) {
      return name === 'indirect-first-instance';
    },
    async compileAsync() {},
  };
  const strategy = buildFirstInstanceCrossoverStrategy({
    scenario,
    sourceGeometries,
    renderer,
    camera: cameraFixture(),
    laneCommandSegmentOrder,
  });
  return { renderer, scenario, sourceGeometries, strategy };
}

function disposeFixture(fixture) {
  const deletedAttributes = [];
  disposeStrategyResources({
    _attributes: {
      delete(attribute) {
        deletedAttributes.push(attribute);
      },
    },
  }, fixture.strategy);
  fixture.sourceGeometries.forEach((geometry) => geometry.dispose());
  return deletedAttributes;
}

function geometryByLane(strategy, lane) {
  return strategy.geometries.find((geometry) => geometry.name.includes(`-${lane}-geometry`));
}

function expectedChallenge(strategy) {
  const width = 256;
  const pixelCount = width * Math.ceil(strategy.packing.objectCount / width);
  const expected = new Uint8Array(pixelCount * 4);
  for (let bucket = 0; bucket < strategy.packing.bucketCount; bucket += 1) {
    const base = strategy.packing.bucketBases[bucket];
    const end = base + strategy.packing.visibleCounts[bucket];
    for (let address = base; address < end; address += 1) {
      const encoded = strategy.packing.visibleIds[address] + 1;
      const pixelBase = address * 4;
      expected[pixelBase] = encoded & 0xff;
      expected[pixelBase + 1] = (encoded >>> 8) & 0xff;
      expected[pixelBase + 2] = (encoded >>> 16) & 0xff;
      expected[pixelBase + 3] = 0xff;
    }
  }
  return expected;
}

function expectedChallengeReset(strategy) {
  return new Uint8Array(256 * Math.ceil(strategy.packing.objectCount / 256) * 4);
}

test('strategy binds each logical lane to its counterbalanced physical command segment', async () => {
  const fixture = buildFixture([FEATURE, PORTABLE]);
  const { strategy } = fixture;
  try {
    const portableGeometry = geometryByLane(strategy, PORTABLE);
    const featureGeometry = geometryByLane(strategy, FEATURE);
    assert.ok(portableGeometry);
    assert.ok(featureGeometry);
    assert.equal(portableGeometry.indirect, featureGeometry.indirect);
    assert.deepEqual(
      portableGeometry.indirectOffset,
      [...strategy.packing.lanes[PORTABLE].offsets],
    );
    assert.deepEqual(
      featureGeometry.indirectOffset,
      [...strategy.packing.lanes[FEATURE].offsets],
    );
    assert.equal(strategy.commandSegments[FEATURE].index, 0);
    assert.equal(strategy.commandSegments[PORTABLE].index, 1);
    assert.deepEqual(
      strategy.root.children.map((bundle) => bundle.name),
      [
        `${strategy.id}-${FEATURE}-bundle`,
        `${strategy.id}-${PORTABLE}-bundle`,
      ],
    );
    assert.equal(featureGeometry.getAttribute('bucketBase'), undefined);
    assert.equal(portableGeometry.index, featureGeometry.index);
    for (const name of Object.keys(featureGeometry.attributes)) {
      assert.equal(featureGeometry.getAttribute(name), portableGeometry.getAttribute(name));
    }

    const scene = new Scene();
    scene.add(strategy.root);
    const renderedLanes = [];
    await strategy.primeBundles({
      scene,
      render: async (lane) => {
        await Promise.resolve();
        renderedLanes.push(lane);
        const activeBundle = strategy.root.children.find((bundle) => bundle.visible);
        activeBundle.children[0].onBeforeRender({ _currentRenderBundle: {} });
      },
    });
    assert.deepEqual(renderedLanes, [FEATURE, PORTABLE]);
    assert.equal(strategy.activeLane, PORTABLE);
    const lifecycle = strategy.lifecycleDiagnostics();
    assert.equal(lifecycle.bundlesPrimed, true);
    assert.deepEqual(lifecycle.bundleRecordCounts, { portable: 1, feature: 1 });
    assert.deepEqual(lifecycle.bundleStaticFlags, { portable: true, feature: true });
    assert.equal(lifecycle.allBundlesStatic, true);
    assert.equal(lifecycle.bundleCount, 2);
    assert.equal(lifecycle.meshCount, 2);
    assert.equal(lifecycle.activeRenderObjectCount, 1);
    assert.deepEqual(Object.keys(lifecycle.materialVersions).sort(), [FEATURE, PORTABLE]);
    const diagnostics = strategy.diagnostics();
    assert.equal(diagnostics.kind, 'frozen-first-instance-addressing-crossover');
    assert.equal(diagnostics.activeRenderObjectCount, 1);
    assert.equal(diagnostics.geometryIdentityCount, 2);
    assert.equal(diagnostics.materialIdentityCount, 2);
    assert.equal(diagnostics.commonIndexIdentityCount, 1);
    assert.equal(diagnostics.commandRecordCount, fixture.scenario.bucketCount * 2);
    assert.equal(diagnostics.visibleIdsCount, fixture.scenario.objectCount);

    const challengeGeometries = strategy.geometries.filter(
      (geometry) => geometry.name.includes('fragment-address-challenge'),
    );
    assert.equal(challengeGeometries.length, 2);
    const portableChallengeGeometry = challengeGeometries.find(
      (geometry) => geometry.name.endsWith('-portable'),
    );
    const featureChallengeGeometry = challengeGeometries.find(
      (geometry) => geometry.name.endsWith('-feature'),
    );
    assert.ok(portableChallengeGeometry);
    assert.ok(featureChallengeGeometry);
    assert.equal(portableChallengeGeometry.indirect, portableGeometry.indirect);
    assert.equal(featureChallengeGeometry.indirect, featureGeometry.indirect);
    assert.deepEqual(
      portableChallengeGeometry.indirectOffset,
      portableGeometry.indirectOffset,
    );
    assert.deepEqual(
      featureChallengeGeometry.indirectOffset,
      featureGeometry.indirectOffset,
    );
    assert.equal(featureChallengeGeometry.index, portableChallengeGeometry.index);
    assert.equal(
      featureChallengeGeometry.getAttribute('position'),
      portableChallengeGeometry.getAttribute('position'),
    );
    assert.equal(featureChallengeGeometry.getAttribute('bucketBase'), undefined);
    assert.equal(strategy.addressChallengeGeometryEvidence.pass, true);
    assert.equal(strategy.addressChallengeGeometryEvidence.positionMismatchCount, 0);
    assert.equal(strategy.addressChallengeGeometryEvidence.bucketBaseMismatchCount, 0);
    assert.equal(strategy.addressChallengeGeometryEvidence.indexMismatchCount, 0);
    assert.deepEqual(
      strategy.addressChallengeGeometryEvidence.laneOffsetsExact,
      { portable: true, feature: true },
    );
  } finally {
    const storageAttributes = new Set(strategy.storageAttributes);
    const deletedAttributes = disposeFixture(fixture);
    assert.equal(strategy.root.parent, null);
    assert.equal(deletedAttributes.length, storageAttributes.size);
    assert.equal(deletedAttributes.every((attribute) => storageAttributes.has(attribute)), true);
    assert.equal(strategy.geometries.every((geometry) => geometry.indirect === null), true);
  }
});

test('address challenge proves reset completion and cannot pass stale prior output', async () => {
  const fixture = buildFixture();
  const { strategy } = fixture;
  const expected = expectedChallenge(strategy);
  const expectedReset = expectedChallengeReset(strategy);
  const events = [];
  let pixels = expectedReset.slice();
  const renderer = {
    autoClear: true,
    clearColor: new Color(0x123456),
    clearAlpha: 0.375,
    backend: {
      get() {
        return { textureDescriptorGPU: { format: 'rgba8unorm' } };
      },
    },
    getRenderTarget() {
      return null;
    },
    getActiveCubeFace() {
      return 0;
    },
    getActiveMipmapLevel() {
      return 0;
    },
    setRenderTarget(target) {
      events.push(target === null ? 'restore-target' : 'set-target');
    },
    getClearColor(target) {
      return target.copy(this.clearColor);
    },
    getClearAlpha() {
      return this.clearAlpha;
    },
    setClearColor(color, alpha) {
      this.clearColor.set(color);
      this.clearAlpha = alpha;
      events.push(alpha === 0 ? 'set-clear-zero' : 'restore-clear');
    },
    clear() {
      events.push('clear-reset');
      pixels = expectedReset.slice();
    },
    render() {
      events.push('render-challenge');
      pixels = expected.slice();
    },
    async readRenderTargetPixelsAsync() {
      events.push('readback');
      return pixels.slice();
    },
  };
  try {
    const passed = await strategy.challengeAddressing(renderer, cameraFixture(), PORTABLE);
    assert.equal(passed.pass, true, JSON.stringify(passed, null, 2));
    assert.equal(passed.kind, 'render-target-all-address-challenge');
    assert.equal(passed.outputStage, 'fragment');
    assert.equal(passed.addressTransport, 'vertex-address-to-rgba8-pixel');
    assert.equal(passed.reset.pass, true);
    assert.deepEqual(events, [
      'set-target',
      'set-clear-zero',
      'clear-reset',
      'restore-clear',
      'restore-target',
      'readback',
      'set-target',
      'render-challenge',
      'restore-target',
      'readback',
    ]);
    assert.equal(renderer.autoClear, true);
    assert.equal(renderer.clearAlpha, 0.375);
    assert.equal(renderer.clearColor.getHex(), 0x123456);

    events.length = 0;
    pixels = expected.slice();
    renderer.clear = () => {
      events.push('clear-reset-no-op');
    };
    renderer.render = () => {
      events.push('render-challenge-no-op');
    };
    const stale = await strategy.challengeAddressing(renderer, cameraFixture(), PORTABLE);
    assert.equal(stale.reset.pass, false);
    assert.equal(stale.sha256, stale.expectedSha256);
    assert.equal(stale.pass, false);
    assert.deepEqual(events, [
      'set-target',
      'set-clear-zero',
      'clear-reset-no-op',
      'restore-clear',
      'restore-target',
      'readback',
      'set-target',
      'render-challenge-no-op',
      'restore-target',
      'readback',
    ]);
    assert.equal(renderer.autoClear, true);
    assert.equal(renderer.clearAlpha, 0.375);
    assert.equal(renderer.clearColor.getHex(), 0x123456);
  } finally {
    disposeFixture(fixture);
  }
});
