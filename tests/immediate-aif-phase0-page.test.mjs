import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import {
  DirectionalLight,
  HemisphereLight,
  PerspectiveCamera,
  Scene,
  Vector3,
  WebGPUCoordinateSystem,
} from 'three';

import {
  IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET,
  IMMEDIATE_AIF_PHASE0_CHAIN_FIELD_NAMES,
  IMMEDIATE_AIF_PHASE0_ERROR_SCOPE_FILTERS,
  IMMEDIATE_AIF_PHASE0_OBSERVATION_ENCODING,
  IMMEDIATE_AIF_PHASE0_OBSERVATION_FIELD_NAMES,
  IMMEDIATE_AIF_PHASE0_PAGE_GLOBAL,
  IMMEDIATE_AIF_PHASE0_PAGE_KIND,
  IMMEDIATE_AIF_PHASE0_SNAPSHOT_PLAN,
  compareImmediateAifPackedAddressBytes,
  commitImmediateAifObservationFields,
  createImmediateAifObservationChain,
  createImmediateAifObservationChallenge,
  createImmediateAifPackedAddressBytes,
  drainImmediateAifErrorScopes,
  immediateAifPhase0PlanEvidence,
  immediateAifBufferBindingLayoutSnapshot,
  immediateAifExpectedDiagnosticVertexInputs,
  immediateAifExpectedBindingVisibility,
  immediateAifQueueWriteBufferSourceSelection,
  normalizeImmediateAifCommonRenderStateProjection,
  realizeAndFreezeImmediateAifTransform,
  realizeImmediateAifReversedDepthCamera,
  serializeImmediateAifPhase0Error,
  validateImmediateAifOrderedProductionChallengeEvidence,
  validateImmediateAifScheduleInstallationEvidence,
  validateImmediateAifPhase0PageResultShape,
  validateImmediateAifPhase0TargetConfiguration,
} from '../src/phase0/immediate-aif-page-contract.js';
import {
  runImmediateAifPhase0ExecutionLifecycle,
} from '../src/phase0/immediate-aif-page.js';
import {
  IMMEDIATE_AIF_PHASE0_LANES,
  IMMEDIATE_AIF_PHASE0_LANE_ORDERS,
  IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION,
  IMMEDIATE_AIF_PHASE0_WORKLOAD,
  IMMEDIATE_AIF_UNUSED_ADDRESS,
} from '../src/phase0/immediate-aif-plan.js';

test('common render-state normalization removes only the intentional lane selection', () => {
  const base = {
    renderer: { outputColorSpace: 'srgb', sortObjects: false },
    scene: {
      activeLane: 'A',
      laneRoots: {
        A: { uuid: 'root-a', visible: true, matrix: [1] },
        I: { uuid: 'root-i', visible: false, matrix: [1] },
        F: { uuid: 'root-f', visible: false, matrix: [1] },
      },
      directChildren: [
        { uuid: 'light', visible: true },
        { uuid: 'root-a', visible: true },
      ],
    },
  };
  const selectedI = structuredClone(base);
  selectedI.scene.activeLane = 'I';
  selectedI.scene.laneRoots.A.visible = false;
  selectedI.scene.laneRoots.I.visible = true;
  selectedI.scene.directChildren[1].visible = false;
  assert.deepEqual(
    normalizeImmediateAifCommonRenderStateProjection(base),
    normalizeImmediateAifCommonRenderStateProjection(selectedI),
  );
  const mutatedRenderer = structuredClone(selectedI);
  mutatedRenderer.renderer.sortObjects = true;
  assert.notDeepEqual(
    normalizeImmediateAifCommonRenderStateProjection(base),
    normalizeImmediateAifCommonRenderStateProjection(mutatedRenderer),
  );
});

test('queue.writeBuffer source selection converts typed-array elements to exact bytes', () => {
  const backing = Uint32Array.of(0x01020304, 0x11121314, 0x21222324, 0x31323334);
  const view = backing.subarray(1, 4);
  const selection = immediateAifQueueWriteBufferSourceSelection(view, 1, 1);
  assert.equal(selection.elementByteSize, 4);
  assert.equal(selection.elementCount, 3);
  assert.equal(selection.elementOffset, 1);
  assert.equal(selection.elementLength, 1);
  assert.equal(selection.byteOffset, 4);
  assert.equal(selection.byteLength, 4);
  assert.deepEqual(
    Array.from(selection.bytes),
    Array.from(new Uint8Array(backing.buffer, 2 * Uint32Array.BYTES_PER_ELEMENT, 4)),
  );
  assert.throws(
    () => immediateAifQueueWriteBufferSourceSelection(view, 3, 1),
    /out of range/,
  );
});

test('error-scope drainage is exact LIFO for normal completion and failure teardown',
  async () => {
  assert.deepEqual(
    IMMEDIATE_AIF_PHASE0_ERROR_SCOPE_FILTERS,
    ['validation', 'internal', 'out-of-memory'],
  );
  const pushedFilters = [...IMMEDIATE_AIF_PHASE0_ERROR_SCOPE_FILTERS];
  const normalCalls = [];
  const normal = await drainImmediateAifErrorScopes({
    device: {
      async popErrorScope() {
        normalCalls.push(true);
        return null;
      },
    },
    pushedFilters,
    trigger: 'normal-completion',
  });
  assert.deepEqual(pushedFilters, ['validation', 'internal', 'out-of-memory']);
  assert.equal(normalCalls.length, 3);
  assert.equal(normal.pass, true);
  assert.equal(normal.applicable, true);
  assert.deepEqual(normal.pushOrder, ['validation', 'internal', 'out-of-memory']);
  assert.deepEqual(normal.expectedPopOrder, ['out-of-memory', 'internal', 'validation']);
  assert.deepEqual(normal.popOrder, normal.expectedPopOrder);
  assert.deepEqual(normal.scopedErrors, {
    'out-of-memory': null,
    internal: null,
    validation: null,
  });
  assert.equal(normal.scopedErrorsComplete, true);
  assert.equal(normal.scopedErrorsEmpty, true);
  assert.equal(Object.isFrozen(normal), true);
  assert.equal(normal.records.every(Object.isFrozen), true);

  class MockGpuValidationError extends Error {}
  let capturedPop = 0;
  const captured = await drainImmediateAifErrorScopes({
    device: {
      async popErrorScope() {
        capturedPop += 1;
        return capturedPop === 2 ? new MockGpuValidationError('captured validation') : null;
      },
    },
    pushedFilters,
    trigger: 'failure-teardown',
  });
  assert.equal(capturedPop, 3);
  assert.equal(captured.pass, false);
  assert.equal(captured.scopedErrorsComplete, true);
  assert.equal(captured.scopedErrorsEmpty, false);
  assert.deepEqual(captured.scopedErrors.internal, {
    name: 'MockGpuValidationError',
    message: 'captured validation',
  });

  let rejectedPop = 0;
  const rejected = await drainImmediateAifErrorScopes({
    device: {
      async popErrorScope() {
        rejectedPop += 1;
        if (rejectedPop === 1) throw new Error('device lost while popping');
        return null;
      },
    },
    pushedFilters,
    trigger: 'failure-teardown',
  });
  assert.equal(rejectedPop, 3);
  assert.equal(rejected.pass, false);
  assert.equal(rejected.scopedErrorsComplete, false);
  assert.equal(rejected.scopedErrorsEmpty, false);
  assert.deepEqual(rejected.records[0].popFailure, {
    name: 'Error',
    message: 'device lost while popping',
  });
  const inapplicable = await drainImmediateAifErrorScopes({
    pushedFilters: [],
    trigger: 'failure-teardown',
  });
  assert.equal(inapplicable.pass, true);
  assert.equal(inapplicable.applicable, false);
  assert.equal(inapplicable.scopedErrors, null);
  await assert.rejects(
    drainImmediateAifErrorScopes({
      device: { popErrorScope() { return null; } },
      pushedFilters: ['internal'],
    }),
    /exact prefix/,
  );
});

test('buffer binding-layout snapshots resolve omitted WebGPU uniform defaults', () => {
  assert.deepEqual(immediateAifBufferBindingLayoutSnapshot({}), {
    hasOwnType: false,
    rawType: null,
    type: 'uniform',
    hasDynamicOffset: false,
    minBindingSize: 0,
  });
  assert.deepEqual(immediateAifBufferBindingLayoutSnapshot({
    type: 'read-only-storage',
    hasDynamicOffset: true,
    minBindingSize: 640,
  }), {
    hasOwnType: true,
    rawType: 'read-only-storage',
    type: 'read-only-storage',
    hasDynamicOffset: true,
    minBindingSize: 640,
  });
});

test('binding visibility pins Three uniform groups and exact shader-stage storage', () => {
  assert.equal(immediateAifExpectedBindingVisibility({
    resourceClass: 'buffer',
    bufferType: 'uniform',
    declaredStageVisibility: 3,
  }), 7);
  assert.notEqual(immediateAifExpectedBindingVisibility({
    resourceClass: 'buffer',
    bufferType: 'uniform',
    declaredStageVisibility: 3,
  }), 3);
  assert.equal(immediateAifExpectedBindingVisibility({
    resourceClass: 'buffer',
    bufferType: 'read-only-storage',
    declaredStageVisibility: 4,
  }), 4);
  assert.notEqual(immediateAifExpectedBindingVisibility({
    resourceClass: 'buffer',
    bufferType: 'read-only-storage',
    declaredStageVisibility: 4,
  }), 7);
  assert.throws(() => immediateAifExpectedBindingVisibility({
    resourceClass: 'buffer',
    bufferType: 'storage',
    declaredStageVisibility: 8,
  }), /stage mask/);
});

test('address and object-ID diagnostics pin Three first-use A/I/F vertex locations', () => {
  const resources = {
    position: { count: 6_440, resourceId: 439 },
    bucketBase: { count: 6_440, resourceId: 427 },
  };
  const position0 = {
    name: 'position', shaderLocation: 0, format: 'float32x3', stepMode: 'vertex',
    arrayType: 'Float32Array', itemSize: 3, normalized: false,
    count: 6_440, resourceId: 439,
  };
  const expected = {
    address: {
      A: [
        {
          name: 'bucketBase', shaderLocation: 0, format: 'uint32', stepMode: 'vertex',
          arrayType: 'Uint32Array', itemSize: 1, normalized: false,
          count: 6_440, resourceId: 427,
        },
        { ...position0, shaderLocation: 1 },
      ],
      I: [position0],
      F: [position0],
    },
    'object-id': {
      A: [
        position0,
        {
          name: 'bucketBase', shaderLocation: 1, format: 'uint32', stepMode: 'vertex',
          arrayType: 'Uint32Array', itemSize: 1, normalized: false,
          count: 6_440, resourceId: 427,
        },
      ],
      I: [position0],
      F: [position0],
    },
  };
  for (const diagnosticKind of ['address', 'object-id']) {
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      const laneResources = lane === 'A'
        ? resources
        : { position: resources.position };
      const actual = immediateAifExpectedDiagnosticVertexInputs({
        diagnosticKind,
        lane,
        resources: laneResources,
      });
      assert.deepEqual(actual, expected[diagnosticKind][lane]);
      assert.equal(Object.isFrozen(actual), true);
      assert.equal(actual.every(Object.isFrozen), true);
    }
  }
  assert.throws(
    () => immediateAifExpectedDiagnosticVertexInputs({
      diagnosticKind: 'address', lane: 'A', resources: { position: resources.position },
    }),
    /resources are not exact/,
  );
  assert.throws(
    () => immediateAifExpectedDiagnosticVertexInputs({
      diagnosticKind: 'object-id', lane: 'I', resources,
    }),
    /resources are not exact/,
  );
  assert.throws(
    () => immediateAifExpectedDiagnosticVertexInputs({
      diagnosticKind: 'production', lane: 'A', resources,
    }),
    /diagnostic kind/,
  );
  assert.throws(
    () => immediateAifExpectedDiagnosticVertexInputs({
      diagnosticKind: 'address', lane: 'P', resources,
    }),
    /diagnostic lane/,
  );
});

function sourceSlice(source, startMarker, endMarker, label) {
  const canonicalSource = source.replaceAll('\r\n', '\n');
  const start = canonicalSource.indexOf(startMarker);
  const end = canonicalSource.indexOf(endMarker, start + startMarker.length);
  assert.ok(start >= 0 && end > start, `${label} source markers are missing`);
  return canonicalSource.slice(start, end);
}

async function assertThreeDiagnosticFirstUseSemantics({ label, root }) {
  let sources;
  try {
    sources = await Promise.all([
      readFile(path.join(root, 'src/nodes/core/NodeBuilder.js'), 'utf8'),
      readFile(path.join(root, 'src/renderers/webgpu/nodes/WGSLNodeBuilder.js'), 'utf8'),
      readFile(path.join(root, 'src/renderers/common/RenderObject.js'), 'utf8'),
      readFile(path.join(root, 'src/renderers/webgpu/utils/WebGPUAttributeUtils.js'), 'utf8'),
      readFile(path.join(root, 'src/materials/nodes/NodeMaterial.js'), 'utf8'),
      readFile(path.join(root, 'src/nodes/core/AssignNode.js'), 'utf8'),
      readFile(path.join(root, 'src/nodes/accessors/Position.js'), 'utf8'),
    ]);
  } catch (error) {
    throw new Error(
      `${label} source root is not provisioned with the required Three.js source files: ${root}`,
      { cause: error },
    );
  }
  const [
    nodeBuilder,
    wgslBuilder,
    renderObject,
    attributeUtils,
    nodeMaterial,
    assignNode,
    positionAccessor,
  ] = sources;
  const getAttribute = sourceSlice(
    nodeBuilder,
    '\tgetAttribute( name, type ) {',
    '\n\t/**\n\t * Returns for the given node',
    `${label} NodeBuilder.getAttribute`,
  );
  const getWgslAttributes = sourceSlice(
    wgslBuilder,
    '\tgetAttributes( shaderStage ) {',
    '\n\t/**\n\t * Returns the members',
    `${label} WGSLNodeBuilder.getAttributes`,
  );
  const getRenderAttributes = sourceSlice(
    renderObject,
    '\tgetAttributes() {',
    '\n\t/**\n\t * Returns the vertex buffers',
    `${label} RenderObject.getAttributes`,
  );
  const createVertexBuffers = sourceSlice(
    attributeUtils,
    '\tcreateShaderVertexBuffers( renderObject ) {',
    '\n\t/**\n\t * Destroys the GPU buffer',
    `${label} WebGPUAttributeUtils.createShaderVertexBuffers`,
  );
  const setupPosition = sourceSlice(
    nodeMaterial,
    '\tsetupPosition( builder ) {',
    '\n\t/**\n\t * Setups the computation of the material\'s diffuse color',
    `${label} NodeMaterial.setupPosition`,
  );
  const generateAssign = sourceSlice(
    assignNode,
    '\tgenerate( builder, output ) {',
    '\n}\n\nexport default AssignNode;',
    `${label} AssignNode.generate`,
  );

  assert.match(getAttribute, /for \( const attribute of attributes \)/);
  assert.match(getAttribute, /attributes\.push\( attribute \)/);
  assert.match(getWgslAttributes, /const attributes = this\.getAttributesArray\(\)/);
  assert.match(
    getWgslAttributes,
    /@location\( \$\{index\} \) \$\{ name \} : \$\{ type \}/,
  );
  assert.match(getRenderAttributes, /for \( const nodeAttribute of nodeAttributes \)/);
  assert.match(getRenderAttributes, /attributes\.push\( attribute \)/);
  assert.match(createVertexBuffers, /for \( let slot = 0; slot < attributes\.length; slot \+\+ \)/);
  assert.match(createVertexBuffers, /shaderLocation: slot/);
  assert.match(
    setupPosition,
    /positionLocal\.assign\( subBuild\( this\.positionNode, 'POSITION', 'vec3' \) \)/,
  );
  assert.ok(
    generateAssign.indexOf('targetNode.build( builder )')
      < generateAssign.indexOf('sourceNode.build( builder, targetType )'),
    `${label} AssignNode must generate its position target before its custom source`,
  );
  assert.match(
    positionAccessor,
    /positionLocal = \/\*@__PURE__\*\/ positionGeometry\.toVarying\( 'positionLocal' \)/,
  );
}

test('installed r185 source preserves diagnostic first-use vertex location semantics', async () => {
  await assertThreeDiagnosticFirstUseSemantics({
    label: 'installed r185',
    root: path.resolve('node_modules/three'),
  });
});

const pinnedDevSourceRoot = process.env.THREE_IMMEDIATE_AIF_PINNED_DEV_SOURCE_ROOT?.trim();
test('explicitly provisioned pinned dev source preserves diagnostic first-use semantics', {
  skip: pinnedDevSourceRoot
    ? false
    : 'set THREE_IMMEDIATE_AIF_PINNED_DEV_SOURCE_ROOT to validate the pinned dev checkout',
}, async () => {
  await assertThreeDiagnosticFirstUseSemantics({
    label: 'explicitly provisioned pinned dev',
    root: path.resolve(pinnedDevSourceRoot),
  });
});

test('Phase 0 diagnostic materials preserve their purpose-specific first-use semantics',
  async () => {
  const pageSource = await readFile('src/phase0/immediate-aif-page.js', 'utf8');
  const objectIdMaterial = sourceSlice(
    pageSource,
    'function createObjectIdMaterial(',
    'function createObjectIdDiagnostics(',
    'object-ID diagnostic',
  );
  assert.match(objectIdMaterial, /material\.positionNode = Fn\(\(\) => \{/);
  assert.match(objectIdMaterial, /createVisibleIdAddressNode\(\{ addressMode \}\)/);
  assert.match(objectIdMaterial, /vec4\(positionGeometry, 1\)/);

  const addressMaterial = sourceSlice(
    pageSource,
    'function createAddressMaterial(',
    'function createAddressDiagnostics(',
    'address diagnostic',
  );
  assert.match(addressMaterial, /material\.vertexNode = Fn\(\(\) => \{/);
  const addressBaseUse = addressMaterial.indexOf('createVisibleIdAddressNode');
  const addressPositionUse = addressMaterial.indexOf('positionGeometry');
  assert.ok(addressBaseUse >= 0 && addressPositionUse > addressBaseUse,
    'address diagnostic must first use bucketBase before position in lane A');
  const capture = sourceSlice(
    pageSource,
    'async function captureDiagnosticProducer(',
    'function validateObjectIdPixels(',
    'captureDiagnosticProducer',
  );
  assert.match(capture, /immediateAifExpectedDiagnosticVertexInputs\(\{/);
  assert.doesNotMatch(capture, /name:\s*'position'[\s\S]*shaderLocation:\s*0/);
});

test('page failure teardown drains error scopes before resource and device destruction',
  async () => {
  const source = await readFile('src/phase0/immediate-aif-page.js', 'utf8');
  const pushStart = source.indexOf('for (const filter of IMMEDIATE_AIF_PHASE0_ERROR_SCOPE_FILTERS)');
  const pushRecord = source.indexOf('state.pendingErrorScopeFilters.push(filter)', pushStart);
  assert.ok(pushStart >= 0 && pushRecord > pushStart);

  const cleanupStart = source.indexOf('async function cleanup(state)');
  const cleanupEnd = source.indexOf('async function attachImmediateAifObservationChains', cleanupStart);
  const cleanupSource = source.slice(cleanupStart, cleanupEnd);
  const drainage = cleanupSource.indexOf("drainPageErrorScopes(state, 'failure-teardown')");
  const cleanupBoundary = cleanupSource.indexOf("instrumentation.mark('phase0-cleanup-start'");
  const deviceDestroy = cleanupSource.indexOf("await attempt('device'");
  assert.ok(drainage >= 0 && cleanupBoundary > drainage && deviceDestroy > cleanupBoundary);

  const runStart = source.indexOf('export async function runImmediateAifPhase0Page()');
  const runEnd = source.indexOf("if (typeof window !== 'undefined'", runStart);
  const runSource = source.slice(runStart, runEnd);
  assert.match(runSource, /scopeDrainage: null/);
  assert.match(runSource, /pendingErrorScopeFilters: \[\]/);
  assert.match(runSource, /errorScopeDrainage: null/);
});

function recordingEventTarget(calls) {
  const listeners = new Map();
  return {
    listeners,
    addEventListener(type, listener) {
      calls.push(`add:${type}`);
      assert.equal(listeners.has(type), false);
      listeners.set(type, listener);
    },
    removeEventListener(type, listener) {
      calls.push(`remove:${type}`);
      assert.equal(listeners.get(type), listener);
      listeners.delete(type);
    },
  };
}

test('unexpected cleanup throws preserve an execute failure and detach both page listeners',
  async () => {
  const calls = [];
  const eventTarget = recordingEventTarget(calls);
  const state = { cleanup: null };
  const executeError = new Error('primary execute failure');
  executeError.detail = { phase: 'address/A' };
  const teardownError = new Error('unexpected cleanup failure');
  const onError = () => {};
  const onUnhandledRejection = () => {};

  const outcome = await runImmediateAifPhase0ExecutionLifecycle({
    state,
    execute: async () => {
      calls.push('execute');
      throw executeError;
    },
    teardown: async () => {
      calls.push('teardown');
      throw teardownError;
    },
    eventTarget,
    onError,
    onUnhandledRejection,
  });

  assert.equal(outcome.status, 'technical-canary-failed');
  assert.equal(outcome.failure.message, 'primary execute failure');
  assert.deepEqual(outcome.failure.detail, { phase: 'address/A' });
  assert.equal(outcome.cleanupFailure.message, 'unexpected cleanup failure');
  assert.equal(state.cleanup.complete, false);
  assert.deepEqual(state.cleanup.unexpectedFailure, outcome.cleanupFailure);
  assert.deepEqual(state.cleanup.errors, [{
    label: 'unexpected-cleanup-throw',
    error: outcome.cleanupFailure,
  }]);
  assert.deepEqual(calls, [
    'add:error',
    'add:unhandledrejection',
    'execute',
    'teardown',
    'remove:error',
    'remove:unhandledrejection',
  ]);
  assert.equal(eventTarget.listeners.size, 0);
});

test('an unexpected cleanup throw becomes the primary failure after successful execution',
  async () => {
  const calls = [];
  const eventTarget = recordingEventTarget(calls);
  const state = { cleanup: null };
  const outcome = await runImmediateAifPhase0ExecutionLifecycle({
    state,
    execute: async () => { calls.push('execute'); },
    teardown: async () => {
      calls.push('teardown');
      throw new Error('cleanup-only failure');
    },
    eventTarget,
    onError: () => {},
    onUnhandledRejection: () => {},
  });

  assert.equal(outcome.status, 'technical-canary-failed');
  assert.equal(outcome.failure.message, 'Phase 0 cleanup threw unexpectedly.');
  assert.equal(outcome.failure.detail.message, 'cleanup-only failure');
  assert.equal(outcome.cleanupFailure.message, 'cleanup-only failure');
  assert.equal(state.cleanup.complete, false);
  assert.equal(eventTarget.listeners.size, 0);
  assert.deepEqual(calls.slice(-2), ['remove:error', 'remove:unhandledrejection']);
});

test('production lights realize world matrices before their transforms freeze', () => {
  const expected = IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION.lights;
  const scene = new Scene();
  const hemisphere = new HemisphereLight();
  hemisphere.position.fromArray(expected.hemisphere.position);
  realizeAndFreezeImmediateAifTransform(hemisphere);
  const directional = new DirectionalLight();
  directional.position.fromArray(expected.directional.position);
  directional.target.position.fromArray(expected.directional.target.position);
  realizeAndFreezeImmediateAifTransform(directional);
  realizeAndFreezeImmediateAifTransform(directional.target);
  scene.add(hemisphere, directional, directional.target);
  scene.updateMatrixWorld(true);

  assert.deepEqual(
    hemisphere.getWorldPosition(new Vector3()).toArray(),
    expected.hemisphere.worldPosition,
  );
  assert.deepEqual(hemisphere.matrixWorld.toArray(), expected.hemisphere.matrixWorld);
  assert.deepEqual(
    directional.getWorldPosition(new Vector3()).toArray(),
    expected.directional.worldPosition,
  );
  assert.deepEqual(directional.matrixWorld.toArray(), expected.directional.matrixWorld);
  assert.deepEqual(
    directional.target.getWorldPosition(new Vector3()).toArray(),
    expected.directional.target.worldPosition,
  );
  assert.deepEqual(
    directional.target.matrixWorld.toArray(),
    expected.directional.target.matrixWorld,
  );
  const direction = directional.target.getWorldPosition(new Vector3())
    .sub(directional.getWorldPosition(new Vector3()))
    .normalize()
    .toArray();
  direction.forEach((value, index) => {
    assert.ok(Math.abs(value - expected.directional.targetDirection[index]) < 1e-15);
  });
  assert.equal(hemisphere.matrixAutoUpdate, false);
  assert.equal(hemisphere.matrixWorldAutoUpdate, false);
  assert.equal(directional.matrixAutoUpdate, false);
  assert.equal(directional.matrixWorldAutoUpdate, false);
  assert.equal(directional.target.matrixAutoUpdate, false);
  assert.equal(directional.target.matrixWorldAutoUpdate, false);
});

test('Phase 0 camera realizes the exact WebGPU reversed-depth projection before rendering',
  async () => {
  const expected = IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION.camera;
  const camera = new PerspectiveCamera(
    expected.fov,
    expected.aspect,
    expected.near,
    expected.far,
  );
  assert.equal(camera.reversedDepth, false);
  realizeImmediateAifReversedDepthCamera(camera, {
    coordinateSystem: WebGPUCoordinateSystem,
    reversedDepthBuffer: true,
  });
  assert.equal(camera.coordinateSystem, expected.coordinateSystem);
  assert.equal(camera.reversedDepth, true);
  assert.deepEqual(camera.projectionMatrix.toArray(), expected.projectionMatrix);
  assert.deepEqual(
    camera.projectionMatrixInverse.toArray(),
    expected.projectionMatrixInverse,
  );
  assert.throws(() => realizeImmediateAifReversedDepthCamera(
    new PerspectiveCamera(),
    { coordinateSystem: WebGPUCoordinateSystem, reversedDepthBuffer: false },
  ), /reversed-depth renderer state/);
  const rendererSource = await readFile(
    'node_modules/three/src/renderers/common/Renderer.js',
    'utf8',
  );
  const reversedTransition = rendererSource.indexOf('camera._reversedDepth = true');
  const projectionRefresh = rendererSource.indexOf(
    'camera.updateProjectionMatrix();',
    reversedTransition,
  );
  assert.ok(reversedTransition >= 0 && projectionRefresh > reversedTransition);
});

test('Phase 0 failure artifacts remain bounded and JSON-safe for cyclic GPU-like details', () => {
  class FakeGPUBuffer {
    get mappedRange() {
      throw new Error('host object cannot be inspected');
    }
  }
  const cyclic = { label: 'cycle' };
  cyclic.self = cyclic;
  const error = new Error('synthetic GPU failure');
  error.detail = {
    cyclic,
    gpuBuffer: new FakeGPUBuffer(),
    callback() {},
    bigint: 12n,
    bytes: new Uint8Array(128).fill(7),
  };

  const serialized = serializeImmediateAifPhase0Error(error);
  const json = JSON.stringify(serialized);
  assert.equal(serialized.name, 'Error');
  assert.equal(serialized.message, 'synthetic GPU failure');
  assert.equal(serialized.detail.cyclic.self.__phase0Unserializable, true);
  assert.equal(serialized.detail.cyclic.self.reason, 'circular-reference');
  assert.equal(serialized.detail.gpuBuffer.reason, 'non-plain-host-object');
  assert.equal(serialized.detail.callback.reason, 'json-unsupported-value');
  assert.equal(serialized.detail.bigint.reason, 'json-unsupported-scalar');
  assert.equal(serialized.detail.bytes.preview.length, 32);
  assert.equal(serialized.detail.bytes.truncated, true);
  assert.ok(json.length < 20_000);

  const hostile = new Proxy({}, {
    get() {
      throw new Error('no property access');
    },
  });
  assert.doesNotThrow(() => JSON.stringify(serializeImmediateAifPhase0Error(hostile)));
});

function fakeAddressFixture() {
  const output = new Uint32Array(IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount);
  output.fill(IMMEDIATE_AIF_UNUSED_ADDRESS);
  output[0] = 7;
  output[1] = 8;
  output[2_048] = 2_049;
  const objectIds = Uint32Array.from([7, 8, 2_049]);
  const objectBuckets = new Uint32Array(IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount);
  for (let id = 0; id < objectBuckets.length; id += 1) {
    objectBuckets[id] = Math.floor(id / IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCapacity);
  }
  return {
    addressOracle: {
      expectedOutputObjectIds: output,
      objectIds,
      draws: Array.from(
        { length: IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCount },
        (_, sourceBucket) => ({ sourceBucket }),
      ),
    },
    scenario: {
      objectCount: IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount,
      objectBuckets,
    },
  };
}

test('Phase 0 packed address oracle covers exactly 256 by 256 RGBA8 pixels', () => {
  const fixture = fakeAddressFixture();
  const bytes = createImmediateAifPackedAddressBytes(fixture.addressOracle);
  assert.equal(bytes.byteLength, 256 * 256 * 4);
  assert.deepEqual(Array.from(bytes.subarray(0, 12)), [
    8, 0, 0, 255,
    9, 0, 0, 255,
    0, 0, 0, 0,
  ]);
  assert.deepEqual(
    Array.from(bytes.subarray(2_048 * 4, 2_048 * 4 + 4)),
    [2, 8, 0, 255],
  );
  const exact = compareImmediateAifPackedAddressBytes({
    actual: bytes,
    expected: bytes,
    ...fixture,
  });
  assert.equal(exact.pass, true);
  assert.equal(exact.activeCount, 3);
  assert.deepEqual(exact.mismatchCounts, {
    duplicate: 0,
    missing: 0,
    hidden: 0,
    wrongBucket: 0,
    outOfRange: 0,
    active: 0,
    padding: 0,
  });
  assert.deepEqual(IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET, {
    width: 256,
    height: 256,
    pixelCount: 65_536,
    byteLength: 262_144,
    format: 'rgba8unorm',
    encoding: 'rgb24-object-id-plus-one-transparent-zero-background',
  });
});

test('Phase 0 packed address comparison fails closed with classified corruption', () => {
  const fixture = fakeAddressFixture();
  const expected = createImmediateAifPackedAddressBytes(fixture.addressOracle);
  const actual = expected.slice();
  actual.fill(0, 0, 4);
  actual.set(actual.subarray(4, 8), 2 * 4);
  actual.set([255, 255, 255, 255], 3 * 4);
  actual.set([13, 0, 0, 255], 4 * 4);
  const observed = compareImmediateAifPackedAddressBytes({
    actual,
    expected,
    ...fixture,
  });
  assert.equal(observed.pass, false);
  assert.ok(observed.mismatchCounts.active >= 1);
  assert.ok(observed.mismatchCounts.padding >= 1);
  assert.ok(observed.mismatchCounts.missing >= 1);
  assert.ok(observed.mismatchCounts.duplicate >= 1);
  assert.ok(observed.mismatchCounts.hidden >= 1);
  assert.ok(observed.mismatchCounts.outOfRange >= 1);
});

function callback(lane) {
  return { lane, pass: true };
}

function coverageSnapshot(
  definition,
  scenarioId = 'v99',
  installationId = `${scenarioId}/fixture/${definition.scheduleId}`,
) {
  const lanes = IMMEDIATE_AIF_PHASE0_LANE_ORDERS.flatMap((order) => order);
  const productionPhases = lanes.map((lane, index) => (
    `phase0/${scenarioId}/${definition.scheduleId}/${lane}/capture-${index + 1}/production`
  ));
  const renderPassPhaseOrder = productionPhases.flatMap((phase) => [phase, phase]);
  const sharedSemantics = [
    'matrix', 'bounds', 'objectBucket', 'bucketBase',
    'bucketCapacity', 'cullOrder', 'visibleIds', 'overflow',
  ];
  let attributeId = 1;
  const immutableResources = sharedSemantics.map((semantic) => ({
    semantic: `shared.${semantic}`,
    attributeId: attributeId ++,
    gpuBufferId: `buffer-shared-${semantic}`,
  }));
  immutableResources.push({
    semantic: 'diagnostic.address.commonPosition',
    attributeId: attributeId ++,
    gpuBufferId: 'diagnostic-commonPosition-buffer',
  }, {
    semantic: 'diagnostic.address.featurePosition',
    attributeId: attributeId ++,
    gpuBufferId: 'diagnostic-featurePosition-buffer',
  });
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    for (const semantic of ['normal', 'position', 'uv']) {
      immutableResources.push({
        semantic: `geometry.${lane}.${semantic}`,
        attributeId: attributeId ++,
        gpuBufferId: `buffer-geometry-${semantic}`,
      });
    }
    if (lane === 'A') {
      immutableResources.push({
        semantic: 'geometry.A.bucketBase',
        attributeId: attributeId ++,
        gpuBufferId: 'buffer-a-base',
      });
    }
    immutableResources.push({
      semantic: `geometry.${lane}.index`,
      attributeId: attributeId ++,
      gpuBufferId: 'buffer-geometry-index',
    });
    immutableResources.push({
      semantic: `geometry.${lane}.indirectCommands`,
      attributeId: attributeId ++,
      gpuBufferId: `buffer-${lane.toLowerCase()}`,
    });
  }
  const immutableResourceIds = [...new Set(
    immutableResources.map((resource) => resource.gpuBufferId),
  )];
  const immutableTextureResources = [
    'production.color',
    'production.depth',
    'address.color',
    'objectId.color',
    'objectId.depth',
  ].map((semantic) => ({
    semantic,
    textureUuid: `texture-uuid-${semantic}`,
    gpuTextureId: `texture-gpu-${semantic}`,
  }));
  const productionRenderPassTopology = [];
  const productionCommandTopology = [];
  const commandEncoderTransfers = [];
  const commandEncoders = [];
  const queueSubmissions = [];
  const readbackCreations = [];
  for (const [index, phase] of productionPhases.entries()) {
    const base = 110 + index * 22;
    const encoderDefinitions = [
      { suffix: 'clear', sequence: base, begin: base + 1, end: base + 2,
        finish: base + 3, submit: base + 4,
        renderPassEncoderIds: [`render-pass-clear-${index}`], transferSequences: [] },
      { suffix: 'render', sequence: base + 5, begin: base + 6,
        execute: base + 7, end: base + 8, finish: base + 9, submit: base + 10,
        renderPassEncoderIds: [`render-pass-render-${index}`], transferSequences: [] },
      { suffix: 'color', creation: base + 11, sequence: base + 12,
        transfer: base + 13, finish: base + 14, submit: base + 15,
        renderPassEncoderIds: [], transferSequences: [base + 7] },
      { suffix: 'depth', creation: base + 16, sequence: base + 17,
        transfer: base + 18, finish: base + 19, submit: base + 20,
        renderPassEncoderIds: [], transferSequences: [base + 18] },
    ];
    encoderDefinitions[2].transferSequences = [encoderDefinitions[2].transfer];
    productionRenderPassTopology.push({
      phase,
      pass: true,
      clearCommandEncoderId: `command-encoder-${index}-clear`,
      clearEncoderId: `render-pass-clear-${index}`,
      clearBeginSequence: encoderDefinitions[0].begin,
      clearEndSequence: encoderDefinitions[0].end,
      renderCommandEncoderId: `command-encoder-${index}-render`,
      renderEncoderId: `render-pass-render-${index}`,
      renderBeginSequence: encoderDefinitions[1].begin,
      renderEndSequence: encoderDefinitions[1].end,
      expectedBundleId: `bundle-${index}`,
      clearDrawLikeEvents: [],
      renderDrawLikeEvents: [{
        sequence: encoderDefinitions[1].execute,
        method: 'executeBundles',
        bundleIds: [`bundle-${index}`],
      }],
    });
    for (const definition of encoderDefinitions) {
      const commandBufferId = `command-buffer-${index}-${definition.suffix}`;
      commandEncoders.push({
        sequence: definition.sequence,
        capturePhase: phase,
        commandEncoderId: `command-encoder-${index}-${definition.suffix}`,
        renderPassEncoderIds: definition.renderPassEncoderIds,
        computePassEncoderIds: [],
        transferSequences: definition.transferSequences,
        finishCallCount: 1,
        finishSequence: definition.finish,
        commandBufferId,
      });
      queueSubmissions.push({
        sequence: definition.submit,
        capturePhase: phase,
        commandBufferIds: [commandBufferId],
      });
    }
    for (const [channelIndex, channel] of ['color', 'depth'].entries()) {
      const definition = encoderDefinitions[channelIndex + 2];
      readbackCreations.push({
        sequence: definition.creation,
        capturePhase: phase,
        category: 'createBuffer',
        resourceClass: 'readback-staging',
        resourceId: `staging-${index * 2 + channelIndex}`,
        size: 3_686_400,
        usage: 9,
        mappedAtCreation: false,
      });
      commandEncoderTransfers.push({
        sequence: definition.transfer,
        capturePhase: phase,
        commandEncoderId: `command-encoder-${index}-${channel}`,
        method: 'copyTextureToBuffer',
        source: {
          textureId: `texture-gpu-production.${channel}`,
          mipLevel: 0,
          origin: { x: 0, y: 0, z: 0 },
          aspect: 'all',
        },
        destination: {
          bufferId: `staging-${index * 2 + channelIndex}`,
          offset: 0,
          bytesPerRow: 5_120,
          rowsPerImage: null,
        },
        copySize: { width: 1_280, height: 720, depthOrArrayLayers: 1 },
      });
    }
    const leg = (definition) => ({
      pass: true,
      commandEncoderId: `command-encoder-${index}-${definition.suffix}`,
      commandEncoderSequence: definition.sequence,
      commandBufferId: `command-buffer-${index}-${definition.suffix}`,
      finishSequence: definition.finish,
      submitSequence: definition.submit,
    });
    const readbackLeg = (definition, channel, channelIndex) => ({
      ...leg(definition),
      channel,
      sourceSemantic: `production.${channel}`,
      sourceTextureId: `texture-gpu-production.${channel}`,
      sourceMipLevel: 0,
      sourceOrigin: { x: 0, y: 0, z: 0 },
      sourceAspect: 'all',
      stagingBufferId: `staging-${index * 2 + channelIndex}`,
      stagingCreationSequence: definition.creation,
      stagingSize: 3_686_400,
      stagingUsage: 9,
      destinationOffset: 0,
      destinationBytesPerRow: 5_120,
      destinationRowsPerImage: null,
      copySize: { width: 1_280, height: 720, depthOrArrayLayers: 1 },
      transferSequence: definition.transfer,
    });
    productionCommandTopology.push({
      phase,
      pass: true,
      chronologyExact: true,
      clear: {
        ...leg(encoderDefinitions[0]),
        renderPassEncoderId: `render-pass-clear-${index}`,
        renderPassBeginSequence: encoderDefinitions[0].begin,
        renderPassEndSequence: encoderDefinitions[0].end,
        executeSequence: null,
      },
      render: {
        ...leg(encoderDefinitions[1]),
        renderPassEncoderId: `render-pass-render-${index}`,
        renderPassBeginSequence: encoderDefinitions[1].begin,
        renderPassEndSequence: encoderDefinitions[1].end,
        executeSequence: encoderDefinitions[1].execute,
      },
      colorReadback: readbackLeg(encoderDefinitions[2], 'color', 0),
      depthReadback: readbackLeg(encoderDefinitions[3], 'depth', 1),
    });
  }
  const persistentRecords = [{
    sequence: 1,
    capturePhase: 'phase0/prime/A',
    category: 'createBuffer',
    resourceId: 'persistent-buffer-1',
    resourceClass: 'persistent-buffer',
  }];
  return {
    ...definition,
    scheduleInstallationId: installationId,
    productionChallenge: {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-ordered-production-challenge',
      pass: true,
      installationId,
      scenarioId,
      scheduleId: definition.scheduleId,
      startSequence: 100,
      endSequence: 600,
      callbackCount: 18,
      expectedLanes: lanes,
      observedLanes: lanes,
      productionPhases,
      renderPassCount: 36,
      expectedRenderPassPhaseOrder: renderPassPhaseOrder,
      observedRenderPassPhaseOrder: renderPassPhaseOrder,
      productionRenderPassTopology,
      productionRenderPassTopologyExact: true,
      computePassCount: 0,
      computePasses: [],
      productionExecuteSequences: productionCommandTopology.map(
        (record) => record.render.executeSequence,
      ),
      executeSequencesStrictlyIncreasing: true,
      diagnosticRenderPassCount: 0,
      unexpectedRenderPassCount: 0,
      forbiddenCreationCount: 0,
      forbiddenCreations: [],
      creationEvents: readbackCreations,
      transientReadbackBufferCreationCount: 36,
      queueWriteBufferCount: 0,
      queueWriteBuffers: [],
      queueWriteTextureCount: 0,
      queueWriteTextures: [],
      queueCopyExternalImageToTextureCount: 0,
      queueCopyExternalImagesToTexture: [],
      textureDestructionCount: 0,
      textureDestructions: [],
      frozenResourceIds: ['buffer-a-base', 'buffer-a', 'buffer-i', 'buffer-f'],
      diagnosticPositionResourceIds: [
        'diagnostic-commonPosition-buffer',
        'diagnostic-featurePosition-buffer',
      ],
      installationImmutableResourceIds: [
        'buffer-a-base', 'buffer-a', 'buffer-i', 'buffer-f',
        'diagnostic-commonPosition-buffer', 'diagnostic-featurePosition-buffer',
      ],
      frozenBufferWriteCount: 0,
      frozenBufferWrites: [],
      frozenBufferMapCount: 0,
      frozenBufferMaps: [],
      installationFreezeBoundary: {
        pass: true,
        installationId,
        installationStartSequence: 5,
        installationQueueCompleteSequence: 51,
        installationCompleteSequence: 90,
        challengeStartSequence: 100,
        frozenResourceIds: ['buffer-a-base', 'buffer-a', 'buffer-i', 'buffer-f'],
        diagnosticPositionResourceIds: [
          'diagnostic-commonPosition-buffer',
          'diagnostic-featurePosition-buffer',
        ],
        installationImmutableResourceIds: [
          'buffer-a-base', 'buffer-a', 'buffer-i', 'buffer-f',
          'diagnostic-commonPosition-buffer', 'diagnostic-featurePosition-buffer',
        ],
        prescribedInstallationWrites: [],
        boundaryQueueWrites: [],
        boundaryMaps: [],
        boundaryCommandWrites: [],
      },
      mergedGeometryRealizationQueueCompleteSequence: 70,
      geometryRealizationBeforeChallenge: true,
      sharedGpuCommitmentCompleteSequence: 50,
      sharedGpuCommitmentBeforeChallenge: true,
      immutableResources,
      immutableResourceIds,
      immutableBufferWriteCount: 0,
      immutableBufferWrites: [],
      immutableBufferMapCount: 0,
      immutableBufferMaps: [],
      immutableTextureResources,
      immutableTextureIds: immutableTextureResources.map(
        (resource) => resource.gpuTextureId,
      ),
      commandEncoderTransferCount: commandEncoderTransfers.length,
      commandEncoderTransfers,
      immutableBufferCommandWriteCount: 0,
      immutableBufferCommandWrites: [],
      immutableTextureCommandWriteCount: 0,
      immutableTextureCommandWrites: [],
      expectedReadbackBufferSize: 3_686_400,
      expectedReadbackBufferUsage: 9,
      productionCommandTopology,
      productionCommandTopologyExact: true,
      commandReadbackTopologyExact: true,
      commandEncoderCount: commandEncoders.length,
      commandEncoders,
      queueSubmissionCount: queueSubmissions.length,
      queueSubmissions,
      commandSubmissionTopologyExact: true,
      persistentInventory: {
        pass: true,
        exact: true,
        preflight: { records: persistentRecords },
        postflight: { records: structuredClone(persistentRecords) },
      },
    },
    laneOrders: {
      results: IMMEDIATE_AIF_PHASE0_LANE_ORDERS.map((order) => ({
        order: [...order],
        records: order.map((lane) => ({ lane, callback: callback(lane) })),
      })),
    },
  };
}

test('ordered production challenge contract rejects phase and execution interleaving', () => {
  const challenge = coverageSnapshot(IMMEDIATE_AIF_PHASE0_SNAPSHOT_PLAN[0])
    .productionChallenge;
  assert.equal(validateImmediateAifOrderedProductionChallengeEvidence(challenge), challenge);
  const swappedPhase = structuredClone(challenge);
  [swappedPhase.observedRenderPassPhaseOrder[1], swappedPhase.observedRenderPassPhaseOrder[2]] = [
    swappedPhase.observedRenderPassPhaseOrder[2],
    swappedPhase.observedRenderPassPhaseOrder[1],
  ];
  assert.throws(
    () => validateImmediateAifOrderedProductionChallengeEvidence(swappedPhase),
    /sequencing is malformed/,
  );
  const reversedExecution = structuredClone(challenge);
  reversedExecution.productionExecuteSequences[7] =
    reversedExecution.productionExecuteSequences[6];
  assert.throws(
    () => validateImmediateAifOrderedProductionChallengeEvidence(reversedExecution),
    /sequencing is malformed/,
  );
  const frozenWrite = structuredClone(challenge);
  frozenWrite.frozenBufferWriteCount = 1;
  frozenWrite.frozenBufferWrites = [{ bufferId: 'buffer-a' }];
  assert.throws(
    () => validateImmediateAifOrderedProductionChallengeEvidence(frozenWrite),
    /sequencing is malformed/,
  );
  const immutableWrite = structuredClone(challenge);
  immutableWrite.immutableBufferWriteCount = 1;
  immutableWrite.immutableBufferWrites = [{ bufferId: 'buffer-geometry-position' }];
  assert.throws(
    () => validateImmediateAifOrderedProductionChallengeEvidence(immutableWrite),
    /sequencing is malformed/,
  );
  const textureWrite = structuredClone(challenge);
  textureWrite.queueWriteTextureCount = 1;
  textureWrite.queueWriteTextures = [{ textureId: 'texture-1' }];
  assert.throws(
    () => validateImmediateAifOrderedProductionChallengeEvidence(textureWrite),
    /sequencing is malformed/,
  );
  const lateGeometryRealization = structuredClone(challenge);
  lateGeometryRealization.mergedGeometryRealizationQueueCompleteSequence =
    lateGeometryRealization.startSequence;
  assert.throws(
    () => validateImmediateAifOrderedProductionChallengeEvidence(lateGeometryRealization),
    /sequencing is malformed/,
  );
});

function auditSnapshot(definition) {
  return {
    ...definition,
    laneCaptures: IMMEDIATE_AIF_PHASE0_LANES.map(callback),
  };
}

function queueWriteRecord({
  sequence,
  capturePhase,
  bufferId,
  sourceType,
  sourceByteLength,
  semantic,
  sourceSha256,
  sourceWitnessId,
  bufferOffset = 0,
  dataOffset = 0,
  size = null,
}) {
  const sourceElementByteSize = sourceType === 'Uint32Array'
    || sourceType === 'Float32Array'
    ? 4
    : 1;
  const sourceElementCount = sourceByteLength / sourceElementByteSize;
  const selectedSourceElementLength = size ?? sourceElementCount - dataOffset;
  const selectedSourceByteOffset = dataOffset * sourceElementByteSize;
  const selectedSourceByteLength = selectedSourceElementLength * sourceElementByteSize;
  return {
    sequence,
    capturePhase,
    bufferId,
    bufferOffset,
    sourceId: `queue-write-source-${sequence}`,
    sourceBackingBufferId: `queue-write-array-buffer-${sequence}`,
    sourceType,
    sourceByteLength,
    sourceByteOffset: 0,
    sourceElementByteSize,
    sourceElementCount,
    dataOffset,
    size,
    selectedSourceElementOffset: dataOffset,
    selectedSourceElementLength,
    selectedSourceByteOffset,
    selectedSourceByteLength,
    selectedSourceSha256: sourceSha256,
    semantic,
    sourceSha256,
    ...(sourceWitnessId === undefined ? {} : { sourceWitnessId }),
  };
}

function scheduleInstallation(scenarioId, scheduleId, ordinal, challenge = null) {
  const digest = 'd'.repeat(64);
  const installationId = `${scenarioId}/${ordinal}/${scheduleId}`;
  const scenarioOrdinal = scenarioId === 'v99' ? ordinal : ordinal - 4;
  const initialV99 = scenarioId === 'v99' && ordinal === 1;
  const basePhase = `phase0/schedule-install/${scenarioId}/${ordinal}-${scheduleId}`;
  const attributeCapturePhase = `${basePhase}/attribute-realization`;
  const diagnosticCapturePhase =
    `${basePhase}/diagnostic-position-realization`;
  const commonPositionDigest = '1'.repeat(64);
  const featurePositionDigest = {
    canonical: '2'.repeat(64),
    S1: '3'.repeat(64),
    S2: '4'.repeat(64),
  }[scheduleId];
  const expectedWriteSemantics = scenarioOrdinal === 1
    ? scenarioId === 'v99' ? [] : ['A.bucketBase']
    : ['A.bucketBase', 'F.indirectCommands'];
  const managerDefinitions = [
    ['A.bucketBase', 1, 'Uint32Array', 128, `buffer-base-${ordinal}`, 9, 20],
    ['A.indirectCommands', 4, 'Uint32Array', 640,
      `buffer-command-${ordinal}-A`, 21, 24],
    ['I.indirectCommands', 4, 'Uint32Array', 640,
      `buffer-command-${ordinal}-I`, 27, 30],
    ['F.indirectCommands', 4, 'Uint32Array', 640,
      `buffer-command-${ordinal}-F`, 39, 50],
  ];
  const managerCalls = managerDefinitions.map(([
    semantic, attributeType, arrayType, byteLength, gpuBufferId,
    startSequence, completeSequence,
  ]) => ({
      semantic,
      attributeType,
      manager: 'renderer._attributes',
      managerConstructor: 'Attributes',
      gpuBufferId,
      dirtyBeforeUpdate: expectedWriteSemantics.includes(semantic),
      arrayType,
      byteLength,
      cpuSha256: digest,
      startSequence,
      completeSequence,
    }));
  const queueWriteBuffers = expectedWriteSemantics.map((semantic) => {
    const manager = managerCalls.find((record) => record.semantic === semantic);
    return queueWriteRecord({
      sequence: semantic === 'A.bucketBase' ? 10 + ordinal : 40 + ordinal,
      capturePhase: attributeCapturePhase,
      bufferId: manager.gpuBufferId,
      sourceType: manager.arrayType,
      sourceByteLength: manager.byteLength,
      semantic,
      sourceSha256: manager.cpuSha256,
    });
  });
  const expectedDiagnosticWriteSemantics = initialV99 ? [] : ['featurePosition'];
  const commonRecord = {
    semantic: 'commonPosition',
    manager: 'renderer._attributes',
    managerConstructor: 'Attributes',
    dirtyBeforeUpdate: false,
    attributeType: 1,
    attributeId: 500,
    attributeVersion: 0,
    beforeDataVersion: 0,
    afterDataVersion: 0,
    arrayType: 'Float32Array',
    itemSize: 3,
    count: 1,
    byteLength: 12,
    cpuExact: true,
    cpuSha256: commonPositionDigest,
    witnessId: `sha256-${commonPositionDigest}-12`,
    gpuBufferId: 'diagnostic-commonPosition-buffer',
    gpuBufferSize: 12,
    gpuBufferCreation: {
      sequence: 1,
      capturePhase: 'phase0/diagnostic-prime/A/address',
      resourceId: 'diagnostic-commonPosition-buffer',
      resourceClass: 'persistent-or-upload-buffer',
      size: 12,
      usage: 44,
      mappedAtCreation: true,
    },
    updateStartSequence: 53,
    updateCompleteSequence: 55,
  };
  const featureRecord = {
    semantic: 'featurePosition',
    manager: 'renderer._attributes',
    managerConstructor: 'Attributes',
    dirtyBeforeUpdate: !initialV99,
    attributeType: 1,
    attributeId: 501,
    attributeVersion: initialV99 ? 0 : 1,
    beforeDataVersion: 0,
    afterDataVersion: initialV99 ? 0 : 1,
    arrayType: 'Float32Array',
    itemSize: 3,
    count: 1,
    byteLength: 12,
    cpuExact: true,
    cpuSha256: featurePositionDigest,
    witnessId: `sha256-${featurePositionDigest}-12`,
    gpuBufferId: 'diagnostic-featurePosition-buffer',
    gpuBufferSize: 12,
    gpuBufferCreation: {
      sequence: 2,
      capturePhase: 'phase0/diagnostic-prime/A/address',
      resourceId: 'diagnostic-featurePosition-buffer',
      resourceClass: 'persistent-or-upload-buffer',
      size: 12,
      usage: 44,
      mappedAtCreation: true,
    },
    updateStartSequence: 59,
    updateCompleteSequence: 70,
  };
  const diagnosticQueueWriteBuffers = initialV99 ? [] : [queueWriteRecord({
    sequence: 60 + ordinal,
    capturePhase: diagnosticCapturePhase,
    bufferId: featureRecord.gpuBufferId,
    sourceType: 'Float32Array',
    sourceByteLength: featureRecord.byteLength,
    semantic: featureRecord.semantic,
    sourceSha256: featureRecord.cpuSha256,
    sourceWitnessId: featureRecord.witnessId,
  })];
  const diagnosticPositionRealization = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-address-position-realization',
    pass: true,
    installationId,
    scenarioId,
    scheduleId,
    basePhase,
    capturePhase: diagnosticCapturePhase,
    startSequence: 52,
    queueCompleteSequence: 71,
    expectedWriteSemantics: expectedDiagnosticWriteSemantics,
    observedWriteSemantics: expectedDiagnosticWriteSemantics,
    queueWriteBufferCount: diagnosticQueueWriteBuffers.length,
    queueWriteBuffers: diagnosticQueueWriteBuffers,
    queueWritesExact: true,
    records: [commonRecord, featureRecord],
    laneBindings: {
      A: { semantic: 'commonPosition', attributeId: 500,
        gpuBufferId: 'diagnostic-commonPosition-buffer' },
      I: { semantic: 'commonPosition', attributeId: 500,
        gpuBufferId: 'diagnostic-commonPosition-buffer' },
      F: { semantic: 'featurePosition', attributeId: 501,
        gpuBufferId: 'diagnostic-featurePosition-buffer' },
    },
  };
  const boundChallenge = challenge === null ? null : {
    ...structuredClone(challenge),
    installationId,
    scenarioId,
    scheduleId,
    installationFreezeBoundary: {
      ...structuredClone(challenge.installationFreezeBoundary),
      installationId,
    },
  };
  const commonProjection = {
    renderer: {
      rendererConstructor: 'WebGPURenderer',
      backendConstructor: 'WebGPUBackend',
      backendId: 'backend-1',
      viewport: { width: 1_280, height: 720, devicePixelRatio: 1 },
      coordinateSystem: 2_001,
      autoClear: true,
      autoClearColor: true,
      autoClearDepth: true,
      autoClearStencil: true,
      clearColor: 0x030711,
      clearAlpha: 1,
      clearDepth: 0,
      clearStencil: 0,
      sortObjects: false,
      toneMapping: 0,
      toneMappingExposure: 1,
      outputColorSpace: 'srgb',
      reversedDepthBuffer: true,
      trackTimestamp: false,
      backendDeviceId: 'device-1',
      backendParameterDeviceId: 'device-1',
      backendDeviceMatchesParameter: true,
      activeRenderTargetUuid: null,
      activeCubeFace: 0,
      activeMipmapLevel: 0,
    },
    camera: { reversedDepth: true },
    materials: Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [lane, {}])),
    lights: { hemisphere: {}, directional: { target: {} } },
    scene: {
      activeLane: '__selected-lane__',
      laneRoots: Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane) => [lane, {}])),
      directChildren: [],
    },
    targets: { production: {}, address: {}, objectId: {} },
  };
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-schedule-installation-preflight',
    pass: true,
    frozen: true,
    installationId,
    ordinal,
    scenarioOrdinal,
    scenarioId,
    scheduleId,
    basePhase,
    startSequence: 5,
    queueCompleteSequence: 51,
    completeSequence: 90,
    diagnosticPositionRealization,
    commandSetSha256: digest,
    managerCalls,
    expectedWriteSemantics,
    observedDirtySemantics: expectedWriteSemantics,
    observedWriteSemantics: expectedWriteSemantics,
    queueWriteBufferCount: queueWriteBuffers.length,
    queueWriteBuffers,
    queueWritesExact: true,
    commands: Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane, index) => [
      lane,
      {
        lane,
        exact: true,
        byteLength: 640,
        frozenWords: Array.from({ length: 160 }, () => 0),
        expectedWords: Array.from({ length: 160 }, () => 0),
        gpuSha256: digest,
        expectedSha256: digest,
        gpuBufferId: `buffer-command-${ordinal}-${lane}`,
        readbackStartSequence: 75 + index * 2,
        readbackCompleteSequence: 76 + index * 2,
      },
    ])),
    aBucketBase: { exact: true, cpuSha256: digest },
    frozenResourceIds: [
      `buffer-base-${ordinal}`,
      `buffer-command-${ordinal}-A`,
      `buffer-command-${ordinal}-I`,
      `buffer-command-${ordinal}-F`,
    ],
    commonState: {
      pass: true,
      exactProjection: true,
      exactSha256: true,
      preflight: {
        sha256: digest,
        rawSha256: digest,
        projection: commonProjection,
        rawProjection: commonProjection,
        laneSelection: { exact: true },
      },
      postflight: {
        sha256: digest,
        rawSha256: digest,
        projection: commonProjection,
        rawProjection: commonProjection,
        laneSelection: { exact: true },
      },
    },
    orderedChallenge: boundChallenge,
  };
}

function mergedGeometryRealization() {
  const digest = 'd'.repeat(64);
  const semantics = ['bucketBase', 'normal', 'position', 'uv', 'index'];
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-merged-geometry-realization',
    pass: true,
    frozen: true,
    capturePhase: 'phase0/resources/merged-geometry-realization',
    geometryUuid: 'geometry-uuid',
    startSequence: 51,
    queueCompleteSequence: 70,
    expectedAttributeNames: semantics.slice(0, 4),
    recordCount: 5,
    records: semantics.map((semantic, index) => ({
      semantic,
      resourceKind: index === 4 ? 'index' : 'vertex-attribute',
      manager: 'renderer._attributes',
      managerConstructor: 'Attributes',
      attributeId: index + 1,
      attributeVersion: 0,
      attributeType: index === 4 ? 2 : 1,
      beforeDataVersion: semantic === 'uv' ? null : 0,
      afterDataVersion: 0,
      arrayType: index === 4 ? 'Uint32Array' : 'Float32Array',
      byteLength: 4,
      cpuSha256: digest,
      gpuBufferId: `geometry-buffer-${semantic}`,
      startSequence: 52 + index * 3,
      completeSequence: 53 + index * 3,
    })),
    managerVersionsExact: true,
    queueWriteBufferCount: 0,
    queueWriteBuffers: [],
    frozenResourceIds: semantics.map((semantic) => `geometry-buffer-${semantic}`),
  };
}

function gpuByteWitnessEvidence() {
  const digest = 'd'.repeat(64);
  const witnessId = `sha256-${digest}-4`;
  const shared = ['matrix', 'bounds', 'objectBucket', 'bucketBase',
    'bucketCapacity', 'cullOrder', 'visibleIds', 'overflow'];
  const semantics = [
    ...['v99', 'v20'].flatMap((scenarioId) => (
      shared.map((semantic) => `shared/${scenarioId}/${semantic}`)
    )),
    ...['bucketBase', 'normal', 'position', 'uv', 'index'].map(
      (semantic) => `merged-geometry/${semantic}`,
    ),
  ];
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-gpu-byte-witnesses',
    pass: true,
    witnessCount: 1,
    referenceCount: semantics.length,
    witnesses: {
      [witnessId]: {
        schemaVersion: 1,
        kind: 'immediate-aif-phase0-gpu-byte-witness',
        witnessId,
        encoding: 'base64-exact-bytes',
        byteLength: 4,
        sha256: digest,
        bytesBase64: 'AAAAAA==',
      },
    },
    references: semantics.map((semantic) => ({
      witnessId,
      semantic,
      gpuBufferId: semantic.startsWith('shared/')
        ? `shared-buffer-${semantic.split('/')[1]}-${semantic.split('/')[2]}`
        : `geometry-buffer-${semantic.split('/')[1]}`,
      capturePhase: `phase0/resources/${semantic}`,
      byteLength: 4,
      sha256: digest,
    })),
  };
}

function persistentResourceFreeze(scenarioId) {
  const retained = [{
    sequence: 2,
    capturePhase: 'phase0/prime/A',
    category: 'createRenderBundleEncoder',
    resourceId: 'retained-bundle',
    resourceClass: 'render-bundle',
    encoderId: 'retained-encoder',
    bundleId: 'retained-bundle',
    finishSequence: 3,
    nativeFinishReturned: true,
  }];
  const persistent = [{
    sequence: 1,
    capturePhase: null,
    category: 'createBuffer',
    resourceId: 'persistent-buffer',
    resourceClass: 'persistent-buffer',
  }];
  const scheduleIds = ['S1', 'S2', 'canonical'];
  const diagnosticBundles = scheduleIds.flatMap((scheduleId, scheduleIndex) => (
    ['address', 'object-id'].map((kind, kindIndex) => ({
      sequence: 20 + scheduleIndex * 6 + kindIndex * 2,
      capturePhase:
        `phase0/diagnostic-rerecord/${scenarioId}/${scheduleId}/I/${kind}`,
      category: 'createRenderBundleEncoder',
      resourceId: `diagnostic-${scheduleId}-${kind}`,
      resourceClass: 'render-bundle',
      encoderId: `diagnostic-encoder-${scheduleId}-${kind}`,
      bundleId: `diagnostic-${scheduleId}-${kind}`,
      finishSequence: 21 + scheduleIndex * 6 + kindIndex * 2,
      nativeFinishReturned: true,
    }))
  ));
  const productionBundles = scheduleIds.map((scheduleId, index) => ({
    sequence: 40 + index * 2,
    capturePhase: `phase0/${scenarioId}/${scheduleId}/I/capture-${index + 1}/production`,
    category: 'createRenderBundleEncoder',
    resourceId: `production-${scheduleId}`,
    resourceClass: 'render-bundle',
    encoderId: `production-encoder-${scheduleId}`,
    bundleId: `production-${scheduleId}`,
    finishSequence: 41 + index * 2,
    nativeFinishReturned: true,
  }));
  const addedBundles = [...diagnosticBundles, ...productionBundles];
  const preflightRecords = [...persistent, ...retained];
  const postflightRecords = [...persistent, ...retained, ...addedBundles];
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-persistent-resource-freeze-comparison',
    pass: true,
    scenarioId,
    preflight: {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-persistent-resource-freeze',
      pass: true,
      scenarioId,
      markerSequence: 10,
      resourceCount: preflightRecords.length,
      records: preflightRecords,
    },
    postflight: {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-persistent-resource-freeze',
      pass: true,
      scenarioId,
      markerSequence: 100,
      resourceCount: postflightRecords.length,
      records: postflightRecords,
    },
    prescribedPersistentDelta: {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-prescribed-persistent-resource-delta',
      pass: true,
      scenarioId,
      nonBundlePersistentExact: true,
      exactRetainedBundles: true,
      expectedAddedBundleCount: 9,
      addedBundleCount: 9,
      addedBundleTopologyExact: true,
      diagnosticBundles,
      productionBundles,
      productionSchedules: scheduleIds,
      addedBundles,
    },
  };
}

function globalCommandLedgerFixture() {
  const commandEncoderTraces = [];
  const queueSubmissions = [];
  const commandEncoderTransferCalls = [];
  const renderPassTraces = [];
  const computePassTraces = [];
  const renderBundleTraces = Array.from({ length: 27 }, (_, index) => ({
    encoderId: `global-render-bundle-${index}`,
    capturePhase: `phase0/bundle/${index}`,
    bundleId: `global-bundle-${index}`,
    events: index < 21
      ? Array.from({ length: 32 }, (__, callIndex) => ({
        sequence: 1_000 + index * 40 + callIndex,
        method: 'setImmediates',
        dataSize: 1,
      }))
      : [],
  }));
  const transferLedger = [];
  const bufferSourceSemantics = [
    ...['matrix', 'bounds', 'objectBucket', 'bucketBase', 'bucketCapacity',
      'cullOrder', 'visibleIds', 'overflow'].map((semantic) => `shared.${semantic}`),
    ...IMMEDIATE_AIF_PHASE0_LANES.map((lane) => `command.${lane}`),
    ...['bucketBase', 'normal', 'position', 'uv', 'index'].map(
      (semantic) => `merged-geometry.${semantic}`,
    ),
  ];
  const textureSourceSemantics = [
    'production.color', 'production.depth', 'address.color',
    'objectId.color', 'objectId.depth',
  ];
  const sourceInventory = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-readback-source-inventory',
    pass: true,
    bufferSourceCount: 16,
    textureSourceCount: 5,
    bufferSources: bufferSourceSemantics.map((semantic, index) => ({
      semantic,
      resourceKind: 'buffer',
      attributeId: 700 + index,
      resourceId: `global-source-buffer-${index}`,
    })),
    textureSources: textureSourceSemantics.map((semantic) => ({
      semantic,
      resourceKind: 'texture',
      textureUuid: `texture-uuid-${semantic}`,
      resourceId: `texture-gpu-${semantic}`,
    })),
  };
  const addSubmission = (record, submitSequence) => {
    commandEncoderTraces.push(record);
    queueSubmissions.push({
      sequence: submitSequence,
      capturePhase: record.capturePhase,
      commandBufferIds: [record.commandBufferId],
    });
  };
  for (let index = 0; index < 798; index += 1) {
    const base = 10_000 + index * 10;
    const capturePhase = `phase0/render/${index}`;
    const encoderId = `global-render-command-encoder-${index}`;
    const passEncoderId = `global-render-pass-${index}`;
    addSubmission({
      sequence: base,
      capturePhase,
      commandEncoderId: encoderId,
      renderPassEncoderIds: [passEncoderId],
      computePassEncoderIds: [],
      transferSequences: [],
      finishCallCount: 1,
      finishSequence: base + 2,
      commandBufferId: `global-render-command-buffer-${index}`,
    }, base + 3);
    const attachment = index < 5 ? {
      textureId: `texture-gpu-${textureSourceSemantics[index]}`,
      viewId: `texture-view-${index}`,
    } : null;
    const depthAttachment = attachment !== null
      && textureSourceSemantics[index].endsWith('.depth');
    renderPassTraces.push({
      encoderId: passEncoderId,
      events: [],
      colorAttachments: attachment !== null && !depthAttachment ? [attachment] : [],
      depthStencilAttachment: depthAttachment
        ? attachment
        : null,
    });
  }
  for (let index = 0; index < 6; index += 1) {
    const base = 30_000 + index * 10;
    const capturePhase = `phase0/compute/${index}`;
    const passEncoderId = `global-compute-pass-${index}`;
    addSubmission({
      sequence: base,
      capturePhase,
      commandEncoderId: `global-compute-command-encoder-${index}`,
      renderPassEncoderIds: [],
      computePassEncoderIds: [passEncoderId],
      transferSequences: [],
      finishCallCount: 1,
      finishSequence: base + 2,
      commandBufferId: `global-compute-command-buffer-${index}`,
    }, base + 3);
    computePassTraces.push({ encoderId: passEncoderId, events: [] });
  }
  for (let index = 0; index < 825; index += 1) {
    const base = 40_000 + index * 10;
    const capturePhase = `phase0/readback/${index % 666}`;
    const stagingBufferId = `global-staging-buffer-${index}`;
    const commandEncoderId = `global-transfer-command-encoder-${index}`;
    const commandBufferId = `global-transfer-command-buffer-${index}`;
    const sourceBinding = sourceInventory.bufferSources[index % 16];
    const transfer = {
      sequence: base + 2,
      capturePhase,
      commandEncoderId,
      method: 'copyBufferToBuffer',
      sourceBufferId: sourceBinding.resourceId,
      sourceOffset: 0,
      destinationBufferId: stagingBufferId,
      destinationOffset: 0,
      size: 4,
    };
    const commandEncoder = {
      sequence: base + 1,
      capturePhase,
      commandEncoderId,
      renderPassEncoderIds: [],
      computePassEncoderIds: [],
      transferSequences: [transfer.sequence],
      finishCallCount: 1,
      finishSequence: base + 3,
      commandBufferId,
    };
    const submission = {
      sequence: base + 4,
      capturePhase,
      commandBufferIds: [commandBufferId],
    };
    const mapEvents = [
      {
        sequence: base + 5,
        capturePhase,
        resourceId: stagingBufferId,
        method: 'mapAsync',
        mode: 1,
        offset: 0,
        size: 4,
      },
      {
        sequence: base + 6,
        capturePhase,
        resourceId: stagingBufferId,
        method: 'getMappedRange',
        mode: null,
        offset: 0,
        size: 4,
      },
    ];
    commandEncoderTransferCalls.push(transfer);
    commandEncoderTraces.push(commandEncoder);
    queueSubmissions.push(submission);
    transferLedger.push({
      pass: true,
      capturePhase,
      stagingBufferId,
      stagingSize: 4,
      stagingUsage: 9,
      stagingMappedAtCreation: false,
      createSequence: base,
      transfer,
      sourceBinding,
      commandEncoder,
      submission,
      mapEvents,
      destroySequence: base + 7,
      destroyCapturePhase: capturePhase,
      destroyCallCount: 1,
      transferShapeExact: true,
    });
  }
  return {
    stagingLifecycle: {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-readback-staging-lifecycle',
      pass: true,
      createdCount: 825,
      expectedCreatedCount: 825,
      readbackPhaseCount: 666,
      expectedReadbackPhaseCount: 666,
      liveCount: 0,
      transferCount: 825,
      expectedTransferCount: 825,
      transferMethodCounts: {
        copyBufferToBuffer: 825,
        copyTextureToBuffer: 0,
        clearBuffer: 0,
        copyBufferToTexture: 0,
        copyTextureToTexture: 0,
        resolveQuerySet: 0,
        writeTimestamp: 0,
      },
      expectedCommandEncoderCount: 1_629,
      commandEncoderCount: 1_629,
      commandEncoderClassCounts: {
        renderOnly: 798,
        computeOnly: 6,
        transferOnly: 825,
      },
      expectedQueueSubmissionCount: 1_629,
      queueSubmissionCount: 1_629,
      globalQueueWriteTextureCount: 0,
      globalCopyExternalImageToTextureCount: 0,
      commandLedgerPass: true,
      sourceInventory,
      transferLedger,
    },
    lifecycle: {
      renderBundleTraces,
      renderPassTraces,
      computePassTraces,
      commandEncoderTraces,
      queueSubmissions,
      commandEncoderTransferCalls,
      queueWriteTextureCalls: [],
      queueCopyExternalImageToTextureCalls: [],
      bufferMapEvents: transferLedger.flatMap((record) => record.mapEvents),
    },
    encoderState: {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-encoder-state-exclusion-evidence',
      pass: true,
      expectedRenderBundleEncoderCount: 27,
      renderBundleEncoderCount: 27,
      expectedImmediateRenderBundleCount: 21,
      immediateRenderBundleCount: 21,
      expectedRenderBundleSetImmediatesCount: 672,
      renderBundleSetImmediatesCount: 672,
      renderBundleImmediateTraces: renderBundleTraces.map((trace) => ({
        encoderId: trace.encoderId,
        capturePhase: trace.capturePhase,
        bundleId: trace.bundleId,
        callCount: trace.events.filter(
          (event) => event.method === 'setImmediates',
        ).length,
        calls: trace.events.filter((event) => event.method === 'setImmediates'),
      })),
      renderPassSetImmediatesCount: 0,
      renderPassSetImmediatesCalls: [],
      computePassSetImmediatesCount: 0,
      computePassSetImmediatesCalls: [],
      setBindGroupCount: 0,
      dynamicOffsetCallCount: 0,
      dynamicOffsetCalls: [],
      occlusionQueryCallCount: 0,
      occlusionQueryCalls: [],
    },
  };
}

function bufferMapLifecycleFixture(globalCommandLedger) {
  const stagingRecords = globalCommandLedger.stagingLifecycle.transferLedger.map((entry) => ({
    pass: true,
    resourceId: entry.stagingBufferId,
    resourceClass: 'readback-staging',
    capturePhase: entry.capturePhase,
    createSequence: entry.createSequence,
    size: entry.stagingSize,
    usage: entry.stagingUsage,
    mappedAtCreation: false,
    mapEventCount: entry.mapEvents.length,
    mapEvents: entry.mapEvents,
    firstUse: { sequence: entry.transfer.sequence, kind: entry.transfer.method },
    persistentMappedGrammar: false,
    persistentUnmappedGrammar: false,
    stagingGrammar: true,
  }));
  const persistentRecord = {
    pass: true,
    resourceId: 'persistent-unmapped-buffer',
    resourceClass: 'persistent-or-upload-buffer',
    capturePhase: 'phase0/renderer-init',
    createSequence: 1,
    size: 4,
    usage: 12,
    mappedAtCreation: false,
    mapEventCount: 0,
    mapEvents: [],
    firstUse: null,
    persistentMappedGrammar: false,
    persistentUnmappedGrammar: true,
    stagingGrammar: false,
  };
  const records = [...stagingRecords, persistentRecord];
  const rawEvents = globalCommandLedger.lifecycle.bufferMapEvents;
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-buffer-map-lifecycle',
    pass: true,
    bufferCount: records.length,
    persistentBufferCount: 1,
    persistentMappedAtCreationCount: 0,
    stagingBufferCount: 825,
    rawMapEventCount: rawEvents.length,
    representedMapEventCount: rawEvents.length,
    unmatchedEventCount: 0,
    unmatchedEvents: [],
    records,
  };
}

const FULL_UNIFORM_WRITE_RANGES = Object.freeze([
  Object.freeze({ bufferOffset: 0, dataOffset: 0, size: null }),
]);

const MULTI_PARTIAL_UNIFORM_WRITE_RANGES = Object.freeze([
  Object.freeze({ bufferOffset: 0, dataOffset: 0, size: 4 }),
  Object.freeze({ bufferOffset: 32, dataOffset: 8, size: 4 }),
]);

const OVERLAPPING_UNIFORM_WRITE_RANGES = Object.freeze([
  Object.freeze({ bufferOffset: 0, dataOffset: 0, size: 8 }),
  Object.freeze({ bufferOffset: 16, dataOffset: 4, size: 8 }),
]);

function queueWriteBufferLedgerFixture(
  scenarios,
  globalCommandLedger,
  uniformRanges = MULTI_PARTIAL_UNIFORM_WRITE_RANGES,
  uniformUseKind = 'render-pass-draw',
) {
  const categoryOrder = [
    'renderer-initialization',
    'scenario-matrix-realization',
    'live-compute',
    'shared-resource-commitment',
    'merged-geometry-realization',
    'schedule-attribute-realization',
    'schedule-diagnostic-position-realization',
    'production-render',
    'diagnostic-render',
  ];
  const computeUniform = uniformUseKind === 'compute-dispatch';
  const bundleUniform = uniformUseKind === 'executed-render-bundle';
  assert.ok(computeUniform || bundleUniform || uniformUseKind === 'render-pass-draw');
  const uniformCapturePhase = computeUniform
    ? 'phase0/live/v99/I'
    : 'phase0/v99/canonical/A/capture-900/production';
  const uniformBufferId = 'fixture-uniform-buffer';
  const uniformBufferSize = 64;
  const uniformBindGroupId = 'fixture-uniform-bind-group';
  const uniformBindGroupLayoutId = 'fixture-uniform-bind-group-layout';
  const uniformPassEncoderId = computeUniform
    ? 'global-compute-pass-0'
    : 'global-render-pass-0';
  const uniformCommandEncoderId = computeUniform
    ? 'global-compute-command-encoder-0'
    : 'global-render-command-encoder-0';
  const uniformCommandBufferId = computeUniform
    ? 'global-compute-command-buffer-0'
    : 'global-render-command-buffer-0';
  const sequenceBase = computeUniform ? 30_000 : 10_000;
  const uniformSetBindGroupSequence = sequenceBase + 1;
  const uniformWriteSequence = sequenceBase + 2;
  const uniformDrawSequence = sequenceBase + 6;
  const uniformFinishSequence = sequenceBase + 7;
  const uniformSubmitSequence = sequenceBase + 8;
  const uniformBundle = bundleUniform
    ? globalCommandLedger.lifecycle.renderBundleTraces.at(-1)
    : null;
  const uniformCommandEncoder = globalCommandLedger.lifecycle.commandEncoderTraces.find(
    (record) => record.commandEncoderId === uniformCommandEncoderId,
  );
  const uniformSubmission = globalCommandLedger.lifecycle.queueSubmissions.find(
    (record) => record.commandBufferIds[0] === uniformCommandBufferId,
  );
  const uniformPass = (computeUniform
    ? globalCommandLedger.lifecycle.computePassTraces
    : globalCommandLedger.lifecycle.renderPassTraces).find(
    (record) => record.encoderId === uniformPassEncoderId,
  );
  assert.ok(uniformCommandEncoder && uniformSubmission && uniformPass);
  uniformCommandEncoder.capturePhase = uniformCapturePhase;
  uniformCommandEncoder.finishSequence = uniformFinishSequence;
  uniformSubmission.capturePhase = uniformCapturePhase;
  uniformSubmission.sequence = uniformSubmitSequence;
  if (bundleUniform) {
    Object.assign(uniformBundle, {
      finishSequence: 9_703,
      events: [{
        sequence: 9_700,
        method: 'setBindGroup',
        index: 0,
        bindGroupId: uniformBindGroupId,
        dynamicOffsets: null,
        dynamicOffsetStart: null,
        dynamicOffsetLength: null,
        selectedDynamicOffsets: null,
      }, {
        sequence: 9_702,
        method: 'draw',
        vertexCount: 3,
        instanceCount: 1,
        firstVertex: 0,
        firstInstance: 0,
      }],
    });
  }
  Object.assign(uniformPass, {
    sequence: sequenceBase,
    capturePhase: uniformCapturePhase,
    commandEncoderId: uniformCommandEncoderId,
    events: bundleUniform ? [{
      sequence: uniformDrawSequence,
      method: 'executeBundles',
      bundleIds: [uniformBundle.bundleId],
    }] : [{
      sequence: uniformSetBindGroupSequence,
      method: 'setBindGroup',
      index: 0,
      bindGroupId: uniformBindGroupId,
      dynamicOffsets: null,
      dynamicOffsetStart: null,
      dynamicOffsetLength: null,
      selectedDynamicOffsets: null,
    }, {
      sequence: uniformDrawSequence,
      ...(computeUniform ? {
        method: 'dispatchWorkgroups',
        x: 1,
        y: 1,
        z: 1,
      } : {
        method: 'draw',
        vertexCount: 3,
        instanceCount: 1,
        firstVertex: 0,
        firstInstance: 0,
      }),
    }],
  });
  globalCommandLedger.encoderState.setBindGroupCount = 1;
  const sources = [
    ...scenarios.map((scenario) => ({
      ownerKind: 'scenario-matrix-realization',
      ownerId: scenario.scenarioId,
      capturePhase: scenario.matrixGpuRealization.capturePhase,
      records: scenario.matrixGpuRealization.queueWriteBuffers,
    })),
    ...scenarios.map((scenario) => ({
      ownerKind: 'shared-resource-commitment',
      ownerId: scenario.scenarioId,
      capturePhase: scenario.gpuResourceCommitments.capturePhase,
      records: scenario.gpuResourceCommitments.queueWriteBuffers,
    })),
    ...scenarios.flatMap((scenario) => scenario.scheduleInstallations.map(
      (installation) => ({
        ownerKind: 'schedule-attribute-realization',
        ownerId: installation.installationId,
        capturePhase: `${installation.basePhase}/attribute-realization`,
        records: installation.queueWriteBuffers,
      }),
    )),
    ...scenarios.flatMap((scenario) => scenario.scheduleInstallations.map(
      (installation) => ({
        ownerKind: 'schedule-diagnostic-position-realization',
        ownerId: installation.installationId,
        capturePhase: installation.diagnosticPositionRealization.capturePhase,
        records: installation.diagnosticPositionRealization.queueWriteBuffers,
      }),
    )),
  ];
  const liveAttributeDefinitions = [
    ['v99', 'F', ['shared.visibleIds', 'shared.overflow']],
    ['v99', 'A', ['shared.visibleIds', 'shared.overflow']],
    ['v20', 'A', [
      'shared.bounds', 'shared.objectBucket', 'shared.bucketBase',
      'shared.bucketCapacity', 'shared.cullOrder', 'shared.visibleIds',
      'shared.overflow', 'command.A',
    ]],
    ['v20', 'F', ['shared.visibleIds', 'shared.overflow', 'command.F']],
    ['v20', 'I', ['shared.visibleIds', 'shared.overflow', 'command.I']],
  ];
  let liveSequence = 500;
  const liveRecords = liveAttributeDefinitions.flatMap(([
    scenarioId, lane, semantics,
  ]) => semantics.map((semantic) => queueWriteRecord({
    sequence: liveSequence ++,
    capturePhase: `phase0/live/${scenarioId}/${lane}`,
    bufferId: `fixture-live-${semantic}`,
    sourceType: 'Uint32Array',
    sourceByteLength: 4,
    semantic,
    sourceSha256: 'd'.repeat(64),
  })));
  const uniformBufferCreation = {
    sequence: 9_800,
    capturePhase: 'phase0/prime/A',
    method: 'createBuffer',
    resourceId: uniformBufferId,
    resourceClass: 'persistent-or-upload-buffer',
    size: uniformBufferSize,
    usage: 72,
    mappedAtCreation: false,
  };
  const uniformBindGroupLayout = {
    sequence: 9_801,
    capturePhase: 'phase0/prime/A',
    layoutId: uniformBindGroupLayoutId,
    label: null,
    entries: [{
      binding: 0,
      visibility: 7,
      buffer: { type: null, hasDynamicOffset: false, minBindingSize: 0 },
      sampler: null,
      texture: null,
      storageTexture: null,
      externalTexture: null,
    }],
  };
  const uniformBindGroupCreation = {
    sequence: 9_802,
    capturePhase: 'phase0/prime/A',
    method: 'createBindGroup',
    resourceId: uniformBindGroupId,
    resourceClass: 'bind-group',
    layoutId: uniformBindGroupLayoutId,
    entries: [{
      binding: 0,
      bufferId: uniformBufferId,
      bufferOffset: 0,
      bufferSize: null,
      resourceId: null,
    }],
  };
  const uniformBindingProof = {
    exact: true,
    bindGroupId: uniformBindGroupId,
    bindGroupCreateSequence: uniformBindGroupCreation.sequence,
    bindGroupCapturePhase: uniformBindGroupCreation.capturePhase,
    bindGroupLayoutId: uniformBindGroupLayoutId,
    bindGroupLayoutCreateSequence: uniformBindGroupLayout.sequence,
    binding: 0,
    bufferId: uniformBufferId,
    bufferOffset: 0,
    bufferSize: null,
    bufferRawType: null,
    bufferType: 'uniform',
    hasDynamicOffset: false,
    minBindingSize: 0,
    layoutMatchCount: 1,
    coordinateMatchCount: 1,
  };
  const uniformExecutionProof = {
    useKind: uniformUseKind,
    useSequence: uniformDrawSequence,
    passEncoderId: uniformPassEncoderId,
    commandEncoderId: uniformCommandEncoderId,
    commandEncoderMatchCount: 1,
    commandBufferId: uniformCommandBufferId,
    commandEncoderFinishSequence: uniformFinishSequence,
    submissionMatchCount: 1,
    submitSequence: uniformSubmitSequence,
    ...(bundleUniform ? {
      bindGroupIndex: 0,
      activeSetBindGroupSequence: 9_700,
      activeTargetBindGroupMatchCount: 1,
      bundleId: uniformBundle.bundleId,
      bundleMatchCount: 1,
      bundleSetBindGroupSequence: 9_700,
      bundleDrawSequence: 9_702,
      bundleFinishSequence: 9_703,
    } : {
      setBindGroupSequence: uniformSetBindGroupSequence,
      bindGroupIndex: 0,
      activeSetBindGroupSequence: uniformSetBindGroupSequence,
      activeTargetBindGroupMatchCount: 1,
    }),
    bindGroupId: uniformBindGroupId,
    pass: true,
  };
  const uniformRawRecords = uniformRanges.map((range, index) => queueWriteRecord({
    sequence: uniformWriteSequence + index,
    capturePhase: uniformCapturePhase,
    bufferId: uniformBufferId,
    sourceType: 'Float32Array',
    sourceByteLength: uniformBufferSize,
    sourceSha256: (index + 1).toString(16).repeat(64),
    bufferOffset: range.bufferOffset,
    dataOffset: range.dataOffset,
    size: range.size,
  }));
  const categoryFor = (capturePhase) => {
    if (/^phase0\/live\//.test(capturePhase)) return 'live-compute';
    if (/matrix-realization$/.test(capturePhase)) return 'scenario-matrix-realization';
    if (/shared-commitment$/.test(capturePhase)) return 'shared-resource-commitment';
    if (/attribute-realization$/.test(capturePhase)) return 'schedule-attribute-realization';
    return 'schedule-diagnostic-position-realization';
  };
  const semanticBindingFor = (record) => ({
    semantic: record.semantic?.startsWith('shared.')
      || record.semantic?.startsWith('command.')
      ? record.semantic
      : record.semantic === 'matrix'
      ? 'shared.matrix'
      : record.semantic === 'visibleIds'
        ? 'shared.visibleIds'
        : record.semantic === 'A.bucketBase'
          ? 'geometry.A.bucketBase'
          : record.semantic === 'F.indirectCommands'
            ? 'command.F'
            : 'diagnostic.address.featurePosition',
    attributeId: `fixture-attribute-${record.bufferId}`,
  });
  const sourceRecords = [...sources.flatMap((source) => source.records), ...liveRecords];
  const creationByBuffer = new Map();
  for (const record of sourceRecords) {
    if (!creationByBuffer.has(record.bufferId)) {
      creationByBuffer.set(record.bufferId, {
        sequence: 0,
        capturePhase: 'phase0/renderer-init',
        method: 'createBuffer',
        resourceId: record.bufferId,
        resourceClass: 'persistent-or-upload-buffer',
        size: record.sourceByteLength,
        usage: 44,
        mappedAtCreation: false,
      });
    }
  }
  creationByBuffer.set(uniformBufferId, uniformBufferCreation);
  const attributeRecords = sourceRecords.map((record) => {
    const creation = creationByBuffer.get(record.bufferId);
    return {
      category: categoryFor(record.capturePhase),
      ...structuredClone(record),
      destinationOwner: {
        exact: true,
        resourceId: record.bufferId,
        resourceClass: creation.resourceClass,
        createSequence: creation.sequence,
        createCapturePhase: creation.capturePhase,
        size: creation.size,
        usage: creation.usage,
        mappedAtCreation: creation.mappedAtCreation,
        destinationBindingClass: 'known-attribute',
        semanticBindings: [semanticBindingFor(record)],
        uniformBindingProofs: [],
      },
    };
  });
  const uniformLedgerRecords = uniformRawRecords.map((record) => ({
    category: computeUniform ? 'live-compute' : 'production-render',
    ...structuredClone(record),
    destinationOwner: {
      exact: true,
      resourceId: uniformBufferId,
      resourceClass: uniformBufferCreation.resourceClass,
      createSequence: uniformBufferCreation.sequence,
      createCapturePhase: uniformBufferCreation.capturePhase,
      size: uniformBufferCreation.size,
      usage: uniformBufferCreation.usage,
      mappedAtCreation: uniformBufferCreation.mappedAtCreation,
      destinationBindingClass: 'uniform-bind-group-buffer',
      semanticBindings: [],
      uniformBindingProofs: [uniformBindingProof],
    },
  }));
  const records = [...attributeRecords, ...uniformLedgerRecords]
    .sort((left, right) => left.sequence - right.sequence);
  const expectedSpecializedCounts = {
    'scenario-matrix-realization': 1,
    'shared-resource-commitment': 2,
    'schedule-attribute-realization': 13,
    'schedule-diagnostic-position-realization': 7,
  };
  const uniformWriteGrammar = uniformLedgerRecords.map((record) => {
    const sourceRange = {
      start: record.selectedSourceByteOffset,
      byteLength: record.selectedSourceByteLength,
      end: record.selectedSourceByteOffset + record.selectedSourceByteLength,
    };
    const destinationRange = {
      start: record.bufferOffset,
      byteLength: record.selectedSourceByteLength,
      end: record.bufferOffset + record.selectedSourceByteLength,
    };
    const boundRange = { start: 0, byteLength: uniformBufferSize, end: uniformBufferSize };
    const rangePositive = sourceRange.byteLength > 0;
    const rangeAligned = [
      sourceRange.start,
      sourceRange.byteLength,
      destinationRange.start,
      destinationRange.byteLength,
    ].every((value) => Number.isInteger(value) && value % 4 === 0);
    const sourceRangeWithinData = sourceRange.start >= 0
      && sourceRange.end <= uniformBufferSize;
    const destinationRangeWithinCreation = destinationRange.start >= 0
      && destinationRange.end <= uniformBufferSize;
    const destinationRangeWithinBinding = destinationRange.start >= boundRange.start
      && destinationRange.end <= boundRange.end;
    const sourceDestinationOffsetExact = sourceRange.start
      === destinationRange.start - boundRange.start;
    return {
      pass: rangePositive
        && rangeAligned
        && sourceRangeWithinData
        && destinationRangeWithinCreation
        && destinationRangeWithinBinding
        && sourceDestinationOffsetExact,
      sequence: record.sequence,
      capturePhase: record.capturePhase,
      category: record.category,
      bufferId: record.bufferId,
      creationMatchCount: 1,
      creation: uniformBufferCreation,
      binding: uniformBindingProof,
      alignmentBytes: 4,
      sourceRange,
      destinationRange,
      boundRange,
      rangePositive,
      rangeAligned,
      sourceRangeWithinData,
      destinationRangeWithinCreation,
      destinationRangeWithinBinding,
      sourceDestinationOffsetExact,
      boundByteLength: uniformBufferSize,
      samePhaseDestinationWriteCount: uniformLedgerRecords.length,
      initializationOnly: false,
      executionProofs: [uniformExecutionProof],
    };
  });
  const orderedDestinationRanges = uniformWriteGrammar.map((record) => ({
    sequence: record.sequence,
    ...record.destinationRange,
  })).sort((left, right) => left.start - right.start || left.sequence - right.sequence);
  let furthestDestinationEnd = -1;
  let observedDestinationOverlapCount = 0;
  for (const range of orderedDestinationRanges) {
    if (range.start < furthestDestinationEnd) observedDestinationOverlapCount += 1;
    furthestDestinationEnd = Math.max(furthestDestinationEnd, range.end);
  }
  const uniformWriteGroups = uniformWriteGrammar.length === 0 ? [] : [{
    pass: uniformWriteGrammar.every((record) => record.pass),
    groupId: `${uniformCapturePhase}\u0000${uniformBufferId}`,
    capturePhase: uniformCapturePhase,
    category: computeUniform ? 'live-compute' : 'production-render',
    bufferId: uniformBufferId,
    creation: uniformBufferCreation,
    binding: uniformBindingProof,
    boundRange: { start: 0, byteLength: uniformBufferSize, end: uniformBufferSize },
    writeCount: uniformWriteGrammar.length,
    writeSequences: uniformWriteGrammar.map((record) => record.sequence),
    writeSequencesStrictlyIncreasing: true,
    sameCreationAndBinding: true,
    overlapPolicy: 'observed-not-prescribed',
    observedDestinationOverlapCount,
    orderedDestinationRanges,
    everyWriteExecuted: true,
    records: uniformWriteGrammar,
  }];
  return {
    ledger: {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-queue-write-buffer-ledger',
      pass: true,
      expectedSpecializedCounts,
      specializedCounts: { ...expectedSpecializedCounts },
      specializedBindingsExact: true,
      destinationPolicyExact: true,
      liveAttributeWriteGrammarExact: true,
      liveAttributeWriteGrammar: [
        ['v99', 'I', []],
        ...liveAttributeDefinitions,
      ].map(([scenarioId, lane, expectedSemantics]) => {
        const capturePhase = `phase0/live/${scenarioId}/${lane}`;
        const phaseRecords = records.filter(
          (record) => record.capturePhase === capturePhase,
        );
        const attributeRecords = phaseRecords.filter(
          (record) => record.destinationOwner.destinationBindingClass === 'known-attribute',
        );
        const observedSemantics = attributeRecords.map(
          (record) => record.destinationOwner.semanticBindings[0].semantic,
        );
        return {
          pass: true,
          scenarioId,
          lane,
          capturePhase,
          expectedSemantics,
          observedSemantics,
          attributeWriteCount: attributeRecords.length,
          uniformWriteCount: phaseRecords.length - attributeRecords.length,
          attributeRecords,
        };
      }),
      mergedGeometryWriteCount: 0,
      rendererInitializationWriteCount: 0,
      genericUniformWriteCount: computeUniform ? 0 : uniformLedgerRecords.length,
      uniformWriteGrammarExact: uniformWriteGrammar.every((record) => record.pass),
      uniformWriteGrammar,
      uniformWriteGroupCount: uniformWriteGroups.length,
      uniformWriteGroupsExact: uniformWriteGroups.every((group) => group.pass),
      uniformWriteGroups,
      callCount: records.length,
      categoryOrder,
      partitions: categoryOrder.map((category) => ({
        category,
        count: records.filter((record) => record.category === category).length,
        records: records.filter((record) => record.category === category),
      })),
      unclassifiedCount: 0,
      unclassified: [],
      specializedSources: sources.map((source) => ({
        ownerKind: source.ownerKind,
        ownerId: source.ownerId,
        capturePhase: source.capturePhase,
        recordCount: source.records.length,
        sequences: source.records.map((record) => record.sequence),
        records: source.records,
        rawExact: true,
      })),
      records,
    },
    rawCalls: records.map((record) => ({
      sequence: record.sequence,
      capturePhase: record.capturePhase,
      bufferId: record.bufferId,
      bufferOffset: record.bufferOffset,
      sourceId: record.sourceId,
      sourceBackingBufferId: record.sourceBackingBufferId,
      sourceType: record.sourceType,
      sourceByteLength: record.sourceByteLength,
      sourceByteOffset: record.sourceByteOffset,
      sourceElementByteSize: record.sourceElementByteSize,
      sourceElementCount: record.sourceElementCount,
      dataOffset: record.dataOffset,
      size: record.size,
      selectedSourceElementOffset: record.selectedSourceElementOffset,
      selectedSourceElementLength: record.selectedSourceElementLength,
      selectedSourceByteOffset: record.selectedSourceByteOffset,
      selectedSourceByteLength: record.selectedSourceByteLength,
      selectedSourceSha256: record.selectedSourceSha256,
    })),
    bufferCreations: [...creationByBuffer.values()],
    bindGroupCreations: [uniformBindGroupCreation],
    bindGroupLayouts: [uniformBindGroupLayout],
  };
}

function completeShape({
  uniformRanges = MULTI_PARTIAL_UNIFORM_WRITE_RANGES,
  uniformUseKind = 'render-pass-draw',
} = {}) {
  const digest = 'd'.repeat(64);
  const targetTextureIds = [
    'texture-gpu-production.color',
    'texture-gpu-production.depth',
    'texture-gpu-address.color',
    'texture-gpu-objectId.color',
    'texture-gpu-objectId.depth',
  ];
  const globalCommandLedger = globalCommandLedgerFixture();
  const scenarios = ['v99', 'v20'].map((scenarioId, scenarioIndex) => {
    const baseOrdinal = scenarioIndex * 4 + 1;
    const sharedSemantics = [
      'matrix', 'bounds', 'objectBucket', 'bucketBase',
      'bucketCapacity', 'cullOrder', 'visibleIds', 'overflow',
    ];
    const sharedManagerCalls = sharedSemantics.map((semantic, index) => ({
      semantic,
      manager: 'renderer._attributes',
      managerConstructor: 'Attributes',
      attributeId: index + 101,
      attributeVersion: 1,
      attributeType: 3,
      beforeDataVersion: semantic === 'visibleIds' ? 0 : 1,
      afterDataVersion: 1,
      gpuBufferId: `shared-buffer-${scenarioId}-${semantic}`,
      startSequence: 6 + index * 3,
      completeSequence: 7 + index * 3
        + (semantic === 'visibleIds' ? 1 + scenarioIndex : 0),
    }));
    const sharedRecords = sharedSemantics.map((semantic, index) => ({
      semantic,
      attributeId: index + 101,
      attributeVersion: 1,
      gpuBufferId: `shared-buffer-${scenarioId}-${semantic}`,
      cpuSha256: digest,
      cpuSourceSha256: digest,
      gpuSha256: digest,
      rawWitnessId: `sha256-${digest}-4`,
      readbackStartSequence: 31 + index * 2,
      readbackCompleteSequence: 32 + index * 2,
      exact: true,
    }));
    const visibleUpload = queueWriteRecord({
      sequence: 25 + scenarioIndex,
      capturePhase: `phase0/resources/${scenarioId}/shared-commitment`,
      bufferId: `shared-buffer-${scenarioId}-visibleIds`,
      sourceType: 'Uint32Array',
      sourceByteLength: 262_144,
      semantic: 'visibleIds',
      sourceSha256: digest,
      sourceWitnessId: `sha256-${digest}-4`,
    });
    const matrixWrite = queueWriteRecord({
      sequence: 2,
      capturePhase: `phase0/scenario-load/${scenarioId}/matrix-realization`,
      bufferId: `shared-buffer-${scenarioId}-matrix`,
      sourceType: 'Float32Array',
      sourceByteLength: 4_194_304,
      semantic: 'matrix',
      sourceSha256: digest,
    });
    const snapshots = [
      ...IMMEDIATE_AIF_PHASE0_SNAPSHOT_PLAN.slice(0, 3).map((definition, index) => (
        coverageSnapshot(
          definition,
          scenarioId,
          `${scenarioId}/${baseOrdinal + index}/${definition.scheduleId}`,
        )
      )),
      ...IMMEDIATE_AIF_PHASE0_SNAPSHOT_PLAN.slice(3).map(auditSnapshot),
    ];
    const freeze = persistentResourceFreeze(scenarioId);
    return {
      scenarioId,
      scenarioLoadBoundary: scenarioId === 'v20'
        ? { pass: true, fromScenarioId: 'v99', toScenarioId: 'v20' }
        : null,
      snapshots,
      persistentResourceFreeze: freeze,
      matrixGpuRealization: {
        schemaVersion: 1,
        kind: 'immediate-aif-phase0-scenario-matrix-realization',
        pass: true,
        scenarioId,
        capturePhase: `phase0/scenario-load/${scenarioId}/matrix-realization`,
        attributeId: 101,
        attributeVersion: 1,
        attributeType: 3,
        beforeDataVersion: scenarioId === 'v99' ? null : 0,
        afterDataVersion: 1,
        gpuBufferId: `shared-buffer-${scenarioId}-matrix`,
        cpuSha256: digest,
        startSequence: 1,
        updateCompleteSequence: 2 + (scenarioId === 'v20' ? 1 : 0),
        queueCompleteSequence: 3 + (scenarioId === 'v20' ? 1 : 0),
        expectedQueueWriteBufferCount: scenarioId === 'v99' ? 0 : 1,
        queueWriteBufferCount: scenarioId === 'v99' ? 0 : 1,
        queueWriteBuffers: scenarioId === 'v99' ? [] : [matrixWrite],
        queueWriteExact: true,
      },
      freeze: {
        visibleIdsAttributeId: 107,
        visibleIdsAttributeVersion: 1,
        visibleIdsGpuBufferId: `shared-buffer-${scenarioId}-visibleIds`,
        visibleIdsSha256: digest,
        visibleIdsCpuSha256: digest,
        gpuResourceCommitmentCapturePhase:
          `phase0/resources/${scenarioId}/shared-commitment`,
        gpuResourceCommitmentCompleteSequence: 50,
      },
      scheduleInstallations: ['canonical', 'S1', 'S2', 'canonical'].map(
        (scheduleId, index) => scheduleInstallation(
          scenarioId,
          scheduleId,
          baseOrdinal + index,
          index < 3 ? snapshots[index].productionChallenge : null,
        ),
      ),
      gpuResourceCommitments: {
        schemaVersion: 1,
        kind: 'immediate-aif-phase0-shared-gpu-resource-commitments',
        pass: true,
        scenarioId,
        capturePhase: `phase0/resources/${scenarioId}/shared-commitment`,
        startSequence: 5,
        queueCompleteSequence: 30,
        completeSequence: 50,
        semantics: sharedSemantics,
        managerCalls: sharedManagerCalls,
        managerVersionsExact: true,
        queueWriteBufferCount: 1,
        queueWriteBuffers: [visibleUpload],
        visibleIdsUpload: visibleUpload,
        visibleIdsUploadExact: true,
        records: sharedRecords,
      },
    };
  });
  const addressPositionRecords = scenarios.flatMap(
    (scenario) => scenario.scheduleInstallations.map(
      (installation) => installation.diagnosticPositionRealization,
    ),
  );
  const addressPositionWitnesses = Object.fromEntries(
    ['1', '2', '3', '4'].map((value) => {
      const sha256 = value.repeat(64);
      const witnessId = `sha256-${sha256}-12`;
      return [witnessId, {
        schemaVersion: 1,
        kind: 'immediate-aif-phase0-address-position-cpu-byte-witness',
        witnessId,
        encoding: 'base64-exact-bytes',
        byteLength: 12,
        sha256,
        bytesBase64: 'AAAAAAAAAAAAAAAA',
      }];
    }),
  );
  const queueWriteBufferLedger = queueWriteBufferLedgerFixture(
    scenarios,
    globalCommandLedger,
    uniformRanges,
    uniformUseKind,
  );
  globalCommandLedger.lifecycle.queueWriteBufferCalls = queueWriteBufferLedger.rawCalls;
  const bufferMapLifecycle = bufferMapLifecycleFixture(globalCommandLedger);
  const textureCreations = targetTextureIds.map((resourceId, index) => ({
    sequence: 200 + index * 2,
    capturePhase: 'phase0/renderer-init',
    method: 'createTexture',
    resourceId,
    resourceClass: 'persistent-texture',
  }));
  const textureViews = targetTextureIds.map((textureId, index) => ({
    sequence: 201 + index * 2,
    capturePhase: 'phase0/renderer-init',
    viewId: `texture-view-${index}`,
    textureId,
    descriptor: null,
  }));
  const textureInventory = targetTextureIds.map((textureId, index) => ({
    pass: true,
    textureId,
    classification: 'declared-render-target',
    createSequence: textureCreations[index].sequence,
    capturePhase: textureCreations[index].capturePhase,
    viewIds: [textureViews[index].viewId],
    boundViewIds: [],
    attachmentViewIds: [textureViews[index].viewId],
  }));
  const computeModules = Array.from({ length: 6 }, (_, index) => ({
    moduleId: `compute-module-${index}`,
    sequence: 300 + index,
    code: '@compute @workgroup_size(1) fn main() {}',
  }));
  const computePipelineLayouts = Array.from({ length: 6 }, (_, index) => ({
    sequence: 320 + index,
    capturePhase: `phase0/live/v99/${IMMEDIATE_AIF_PHASE0_LANES[index % 3]}`,
    layoutId: `compute-layout-${index}`,
    bindGroupLayoutIds: [],
    label: null,
    hasOwnImmediateSize: true,
    immediateSize: 0,
  }));
  const computePipelines = Array.from({ length: 6 }, (_, index) => ({
    pipelineId: `compute-pipeline-${index}`,
    sequence: 340 + index,
    layoutId: computePipelineLayouts[index].layoutId,
    immediateSize: null,
    descriptor: { immediateSize: null },
  }));
  const computePipelineLayoutBindings = computePipelines.map((pipeline, index) => ({
    pipelineId: pipeline.pipelineId,
    pipelineLayoutId: pipeline.layoutId,
    layoutMatchCount: 1,
    hasOwnImmediateSize: true,
    immediateSize: 0,
    pass: true,
  }));
  const computeImmediateState = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-compute-immediate-state-exclusion',
    pass: true,
    shaderModuleCount: 6,
    shaderModules: computeModules.map((module) => ({
      moduleId: module.moduleId,
      creationSequence: module.sequence,
      requiresImmediateAddressSpaceCount: 0,
      immediateVariableDeclarationCount: 0,
    })),
    pipelineCount: 6,
    pipelines: computePipelines.map((pipeline) => ({
      pipelineId: pipeline.pipelineId,
      creationSequence: pipeline.sequence,
      immediateSize: pipeline.immediateSize,
      descriptorImmediateSize: pipeline.descriptor.immediateSize,
    })),
    pipelineLayoutBindingCount: 6,
    pipelineLayoutBindings: computePipelineLayoutBindings,
    pipelineLayoutCount: 6,
    pipelineLayouts: computePipelineLayouts,
  };
  return {
    schemaVersion: 1,
    kind: IMMEDIATE_AIF_PHASE0_PAGE_KIND,
    status: 'technical-canary-complete',
    executionMode: 'technical-canary',
    analysisEligible: false,
    efficacyAnalysisAllowed: false,
    numericalDecision: null,
    timingCaptured: false,
    cleanupFailure: null,
    cleanup: {
      complete: true,
      deviceDestroyIntentional: true,
      terminalDeviceLoss: { reason: 'destroyed', message: '' },
      cleanupStartSequence: 1_000_000,
      globalFinalUse: {
        schemaVersion: 1,
        kind: 'immediate-aif-phase0-global-gpu-final-use',
        pass: true,
        finalUseSequence: 999_999,
      },
      persistentBufferCleanup: {
        pass: true,
        terminalDeviceCoverage: true,
        records: [{
          pass: true,
          disposition: 'explicit-destroy-exactly-once',
          destroyCallCount: 1,
          destroySequence: 1_000_001,
          destroyCapturePhase: 'phase0/cleanup/persistent-buffers',
          afterGlobalFinalUse: true,
        }],
      },
      persistentTextureCleanup: {
        pass: true,
        terminalDeviceCoverage: true,
        resourceCount: 5,
        explicitlyDestroyedCount: 5,
        declaredRenderTargetCount: 5,
        internalTextureCount: 0,
        allowedDestroyCapturePhases: ['phase0/cleanup/renderer'],
        records: targetTextureIds.map((resourceId, index) => ({
          resourceId,
          pass: true,
          disposition: 'explicit-destroy-exactly-once',
          destroyCallCount: 1,
          destroySequence: 1_000_010 + index,
          destroyCapturePhase: 'phase0/cleanup/renderer',
          afterGlobalFinalUse: true,
        })),
      },
    },
    evidence: {
      plan: immediateAifPhase0PlanEvidence(),
      shaders: {
        modules: computeModules,
        computeImmediateState,
      },
      pipelines: {
        bindGroupLayouts: queueWriteBufferLedger.bindGroupLayouts,
        layouts: computePipelineLayouts,
        render: [],
        compute: computePipelines,
      },
      observationChallenge: {
        schemaVersion: 1,
        kind: 'immediate-aif-phase0-observation-challenge',
        nonce: '0123456789abcdef0123456789abcdef',
        nonceEntropyBits: 128,
        encoding: IMMEDIATE_AIF_PHASE0_OBSERVATION_ENCODING,
        fieldNames: [...IMMEDIATE_AIF_PHASE0_OBSERVATION_FIELD_NAMES],
        fieldCount: IMMEDIATE_AIF_PHASE0_OBSERVATION_FIELD_NAMES.length,
        fieldCommitments: IMMEDIATE_AIF_PHASE0_OBSERVATION_FIELD_NAMES.map(
          (name) => ({ name, byteLength: 1, sha256: 'a'.repeat(64) }),
        ),
        sha256: 'b'.repeat(64),
        chains: Array.from({ length: 390 }, (_, index) => ({
          schemaVersion: 1,
          chainId: `chain-${index}`,
          sha256: 'c'.repeat(64),
        })),
        chainCount: 390,
      },
      outputWitnesses: {
        schemaVersion: 1,
        kind: 'immediate-aif-phase0-output-byte-witnesses',
        pass: true,
        expectedChallengedCallbackCount: 126,
        challengedCallbackCount: 126,
        expectedObservationCount: 384,
        observationCount: 384,
        callbackReferences: Array.from({ length: 378 }, () => ({})),
      },
      resources: {
        gpuCreations: [
          ...queueWriteBufferLedger.bufferCreations,
          ...queueWriteBufferLedger.bindGroupCreations,
          ...textureCreations,
        ],
        textureViews,
        geometryFixtures: {
          schemaVersion: 2,
          pass: true,
          sourceManifest: { bucketCount: 32 },
          realizationBindingExact: true,
          mergedProduction: {
            gpuRecords: ['bucketBase', 'normal', 'position', 'uv', 'index'].map(
              (semantic) => ({
                semantic,
                gpuResident: true,
                exact: true,
                cpuSha256: digest,
                gpuSha256: digest,
                gpuBufferId: `geometry-buffer-${semantic}`,
                rawWitnessId: `sha256-${digest}-4`,
                boundByProductionBundle: semantic !== 'uv',
              }),
            ),
          },
        },
        commonResources: {
          pass: true,
          expected: structuredClone(IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION),
          camera: {
            pass: true,
            reversedDepth: true,
            projectionMatrix: [
              ...IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION.camera.projectionMatrix,
            ],
            projectionMatrixInverse: [
              ...IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION.camera.projectionMatrixInverse,
            ],
          },
        },
        mergedGeometryRealization: mergedGeometryRealization(),
        gpuByteWitnesses: gpuByteWitnessEvidence(),
        addressDiagnosticPositions: {
          schemaVersion: 1,
          kind: 'immediate-aif-phase0-address-diagnostic-position-evidence',
          pass: true,
          expectedRecordCount: 8,
          recordCount: 8,
          witnessCount: 4,
          witnesses: addressPositionWitnesses,
          scheduleOrder: ['v99', 'v20'].flatMap((scenarioId) => (
            ['canonical', 'S1', 'S2', 'canonical'].map(
              (scheduleId) => `${scenarioId}/${scheduleId}`,
            )
          )),
          expectedScheduleOrder: ['v99', 'v20'].flatMap((scenarioId) => (
            ['canonical', 'S1', 'S2', 'canonical'].map(
              (scheduleId) => `${scenarioId}/${scheduleId}`,
            )
          )),
          stableResourceIdentity: true,
          exactScheduleContent: true,
          featureHashesBySchedule: {
            canonical: ['2'.repeat(64)],
            S1: ['3'.repeat(64)],
            S2: ['4'.repeat(64)],
          },
          records: addressPositionRecords,
        },
        scheduleInstallations: {
          pass: true,
          expectedInstallationCount: 8,
          installationCount: 8,
          expectedCommandReadbackCount: 24,
          commandReadbackCount: 24,
          expectedOrderedChallengeCount: 6,
          orderedChallengeCount: 6,
        },
        persistentResourceFreezes: {
          pass: true,
          records: scenarios.map((scenario) => scenario.persistentResourceFreeze),
        },
        stagingLifecycle: globalCommandLedger.stagingLifecycle,
        bufferMapLifecycle,
        encoderState: globalCommandLedger.encoderState,
        queueWriteBufferLedger: queueWriteBufferLedger.ledger,
        querySets: { pass: true, count: 0 },
        externalTextures: { pass: true, count: 0, methodInstrumented: true },
        textureDestructionBeforeCleanup: {
          pass: true,
          textureCount: 5,
          textureIds: targetTextureIds,
          declaredTargetTextureCount: 5,
          declaredTargetTextures: targetTextureIds.map((gpuTextureId, index) => ({
            semantic: [
              'production.color', 'production.depth', 'address.color',
              'objectId.color', 'objectId.depth',
            ][index],
            gpuTextureId,
          })),
          declaredTargetTextureIds: targetTextureIds,
          targetTextureInventoryExact: true,
          internalTextureCount: 0,
          internalTextureIds: [],
          textureInventory,
          destructionRecords: [],
        },
        samplers: { pass: true },
        renderPassTermination: { pass: true },
      },
      lifecycle: globalCommandLedger.lifecycle,
      scenarios,
    },
  };
}

const QUEUE_WRITE_RAW_FIELD_NAMES = Object.freeze([
  'sequence',
  'capturePhase',
  'bufferId',
  'bufferOffset',
  'sourceId',
  'sourceBackingBufferId',
  'sourceType',
  'sourceByteLength',
  'sourceByteOffset',
  'sourceElementByteSize',
  'sourceElementCount',
  'dataOffset',
  'size',
  'selectedSourceElementOffset',
  'selectedSourceElementLength',
  'selectedSourceByteOffset',
  'selectedSourceByteLength',
  'selectedSourceSha256',
]);

function forgeUniformWriteRecord(result, uniformIndex, mutate) {
  const ledger = result.evidence.resources.queueWriteBufferLedger;
  const sequence = ledger.uniformWriteGrammar[uniformIndex].sequence;
  const record = ledger.records.find((candidate) => candidate.sequence === sequence);
  assert.ok(record);
  mutate(record);
  const rawProjection = Object.fromEntries(QUEUE_WRITE_RAW_FIELD_NAMES.map(
    (name) => [name, structuredClone(record[name])],
  ));
  const duplicateRecords = [
    result.evidence.lifecycle.queueWriteBufferCalls.find(
      (candidate) => candidate.sequence === sequence,
    ),
    ...ledger.partitions.flatMap((partition) => partition.records).filter(
      (candidate) => candidate.sequence === sequence,
    ),
  ];
  for (const duplicate of duplicateRecords) {
    assert.ok(duplicate);
    Object.assign(duplicate, structuredClone(rawProjection));
  }
  const grammar = ledger.uniformWriteGrammar[uniformIndex];
  grammar.capturePhase = record.capturePhase;
  grammar.category = record.category;
  grammar.bufferId = record.bufferId;
  grammar.sourceRange = {
    start: record.selectedSourceByteOffset,
    byteLength: record.selectedSourceByteLength,
    end: record.selectedSourceByteOffset + record.selectedSourceByteLength,
  };
  grammar.destinationRange = {
    start: record.bufferOffset,
    byteLength: record.selectedSourceByteLength,
    end: record.bufferOffset + record.selectedSourceByteLength,
  };
  // Keep all self-authored booleans forged true; the contract must reconstruct them.
  grammar.pass = true;
  grammar.rangePositive = true;
  grammar.rangeAligned = true;
  grammar.sourceRangeWithinData = true;
  grammar.destinationRangeWithinCreation = true;
  grammar.destinationRangeWithinBinding = true;
  grammar.sourceDestinationOffsetExact = true;
  const group = ledger.uniformWriteGroups.find((candidate) => (
    candidate.writeSequences.includes(sequence)
  ));
  assert.ok(group);
  const groupRecordIndex = group.records.findIndex(
    (candidate) => candidate.sequence === sequence,
  );
  group.records[groupRecordIndex] = structuredClone(grammar);
  group.orderedDestinationRanges = group.records.map((candidate) => ({
    sequence: candidate.sequence,
    ...candidate.destinationRange,
  })).sort((left, right) => left.start - right.start || left.sequence - right.sequence);
  let furthestEnd = -1;
  group.observedDestinationOverlapCount = 0;
  for (const range of group.orderedDestinationRanges) {
    if (range.start < furthestEnd) group.observedDestinationOverlapCount += 1;
    furthestEnd = Math.max(furthestEnd, range.end);
  }
  return { ledger, record, grammar, group };
}

function forgeUniformExecutionProof(result, uniformIndex, mutate) {
  const ledger = result.evidence.resources.queueWriteBufferLedger;
  const grammar = ledger.uniformWriteGrammar[uniformIndex];
  const proof = grammar.executionProofs[0];
  mutate(proof);
  const group = ledger.uniformWriteGroups.find(
    (candidate) => candidate.writeSequences.includes(grammar.sequence),
  );
  const groupRecord = group.records.find(
    (candidate) => candidate.sequence === grammar.sequence,
  );
  groupRecord.executionProofs[0] = structuredClone(proof);
  return proof;
}

function insertUniformSameSlotRebind(result, useKind) {
  const ledger = result.evidence.resources.queueWriteBufferLedger;
  const proof = ledger.uniformWriteGrammar[0].executionProofs[0];
  const traces = useKind === 'compute-dispatch'
    ? result.evidence.lifecycle.computePassTraces
    : useKind === 'render-pass-draw'
      ? result.evidence.lifecycle.renderPassTraces
      : result.evidence.lifecycle.renderBundleTraces;
  const trace = traces.find((candidate) => (
    useKind === 'executed-render-bundle'
      ? candidate.bundleId === proof.bundleId
      : candidate.encoderId === proof.passEncoderId
  ));
  assert.ok(trace);
  const useSequence = useKind === 'executed-render-bundle'
    ? proof.bundleDrawSequence
    : proof.useSequence;
  const useIndex = trace.events.findIndex((event) => event.sequence === useSequence);
  assert.ok(useIndex > 0);
  trace.events.splice(useIndex, 0, {
    sequence: useSequence - 1,
    method: 'setBindGroup',
    index: proof.bindGroupIndex,
    bindGroupId: 'fixture-intervening-bind-group',
    dynamicOffsets: null,
    dynamicOffsetStart: null,
    dynamicOffsetLength: null,
    selectedDynamicOffsets: null,
  });
  result.evidence.resources.encoderState.setBindGroupCount += 1;
}

test('uniform queue writes accept full, multi-partial, and observed overlapping ranges', () => {
  const full = completeShape({ uniformRanges: FULL_UNIFORM_WRITE_RANGES });
  assert.equal(validateImmediateAifPhase0PageResultShape(full), full);
  assert.equal(
    full.evidence.resources.queueWriteBufferLedger.uniformWriteGroups[0].writeCount,
    1,
  );

  const partial = completeShape({ uniformRanges: MULTI_PARTIAL_UNIFORM_WRITE_RANGES });
  assert.equal(validateImmediateAifPhase0PageResultShape(partial), partial);
  assert.deepEqual(
    partial.evidence.resources.queueWriteBufferLedger.uniformWriteGroups[0]
      .orderedDestinationRanges.map(({ start, byteLength }) => ({ start, byteLength })),
    [{ start: 0, byteLength: 16 }, { start: 32, byteLength: 16 }],
  );
  const [directRecord] = partial.evidence.resources.queueWriteBufferLedger.records.filter(
    (record) => record.bufferId === 'fixture-uniform-buffer',
  );
  const [directProof] = partial.evidence.resources.queueWriteBufferLedger
    .uniformWriteGrammar[0].executionProofs;
  assert.ok(directProof.activeSetBindGroupSequence < directRecord.sequence);
  assert.ok(directRecord.sequence < directProof.useSequence);
  assert.equal(directProof.bindGroupIndex, 0);
  assert.equal(directProof.activeTargetBindGroupMatchCount, 1);

  const overlapping = completeShape({ uniformRanges: OVERLAPPING_UNIFORM_WRITE_RANGES });
  assert.equal(validateImmediateAifPhase0PageResultShape(overlapping), overlapping);
  const [overlapGroup] = overlapping.evidence.resources.queueWriteBufferLedger
    .uniformWriteGroups;
  assert.equal(overlapGroup.overlapPolicy, 'observed-not-prescribed');
  assert.equal(overlapGroup.observedDestinationOverlapCount, 1);
  assert.equal(overlapGroup.records.length, overlapGroup.writeCount);
  assert.equal(new Set(overlapGroup.writeSequences).size, overlapGroup.writeCount);
  assert.equal(
    overlapGroup.records.every((record) => record.executionProofs.length === 1),
    true,
  );

  const compute = completeShape({
    uniformRanges: MULTI_PARTIAL_UNIFORM_WRITE_RANGES,
    uniformUseKind: 'compute-dispatch',
  });
  assert.equal(validateImmediateAifPhase0PageResultShape(compute), compute);
  assert.deepEqual(
    compute.evidence.resources.queueWriteBufferLedger.uniformWriteGrammar.map(
      (record) => record.executionProofs[0].useKind,
    ),
    ['compute-dispatch', 'compute-dispatch'],
  );

  const bundle = completeShape({ uniformUseKind: 'executed-render-bundle' });
  assert.equal(validateImmediateAifPhase0PageResultShape(bundle), bundle);
  const [bundleProof] = bundle.evidence.resources.queueWriteBufferLedger
    .uniformWriteGrammar[0].executionProofs;
  const [bundleRecord] = bundle.evidence.resources.queueWriteBufferLedger.records.filter(
    (record) => record.bufferId === 'fixture-uniform-buffer',
  );
  assert.ok(bundleProof.bundleFinishSequence < bundleRecord.sequence);
  assert.ok(bundleRecord.sequence < bundleProof.useSequence);
  assert.equal(bundleProof.bindGroupIndex, 0);
  assert.equal(bundleProof.activeTargetBindGroupMatchCount, 1);
});

test('uniform queue write range and execution grammar rejects forged mutations', () => {
  const outOfRange = completeShape();
  forgeUniformWriteRecord(outOfRange, 1, (record) => {
    Object.assign(record, {
      sourceByteLength: 128,
      sourceElementByteSize: 4,
      sourceElementCount: 32,
      dataOffset: 15,
      size: 4,
      selectedSourceElementOffset: 15,
      selectedSourceElementLength: 4,
      selectedSourceByteOffset: 60,
      selectedSourceByteLength: 16,
      bufferOffset: 60,
    });
  });
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(outOfRange),
    /Queue write-buffer ledger evidence/,
  );

  const misaligned = completeShape();
  forgeUniformWriteRecord(misaligned, 0, (record) => {
    Object.assign(record, {
      sourceType: 'Uint16Array',
      sourceElementByteSize: 2,
      sourceElementCount: 32,
      dataOffset: 1,
      size: 2,
      selectedSourceElementOffset: 1,
      selectedSourceElementLength: 2,
      selectedSourceByteOffset: 2,
      selectedSourceByteLength: 4,
      bufferOffset: 2,
    });
  });
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(misaligned),
    /Queue write-buffer ledger evidence/,
  );

  const unexecuted = completeShape();
  const unexecutedLedger = unexecuted.evidence.resources.queueWriteBufferLedger;
  unexecutedLedger.uniformWriteGrammar[0].executionProofs[0]
    .setBindGroupSequence = 10_003;
  unexecutedLedger.uniformWriteGroups[0].records[0].executionProofs[0]
    .setBindGroupSequence = 10_003;
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(unexecuted),
    /Queue write-buffer ledger evidence/,
  );

  const unexecutedCompute = completeShape({ uniformUseKind: 'compute-dispatch' });
  const computeLedger = unexecutedCompute.evidence.resources.queueWriteBufferLedger;
  computeLedger.uniformWriteGrammar[0].executionProofs[0]
    .setBindGroupSequence = 30_003;
  computeLedger.uniformWriteGroups[0].records[0].executionProofs[0]
    .setBindGroupSequence = 30_003;
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(unexecutedCompute),
    /Queue write-buffer ledger evidence/,
  );

  for (const useKind of [
    'render-pass-draw',
    'compute-dispatch',
    'executed-render-bundle',
  ]) {
    const rebound = completeShape({ uniformUseKind: useKind });
    insertUniformSameSlotRebind(rebound, useKind);
    assert.throws(
      () => validateImmediateAifPhase0PageResultShape(rebound),
      /Queue write-buffer ledger evidence/,
      `${useKind} accepted an intervening same-slot bind group`,
    );
  }

  const forgedBindGroupIndex = completeShape();
  forgeUniformExecutionProof(forgedBindGroupIndex, 0, (proof) => {
    proof.bindGroupIndex = 1;
  });
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(forgedBindGroupIndex),
    /Queue write-buffer ledger evidence/,
  );

  const forgedActiveSequence = completeShape({
    uniformUseKind: 'executed-render-bundle',
  });
  forgeUniformExecutionProof(forgedActiveSequence, 0, (proof) => {
    proof.activeSetBindGroupSequence = 9_701;
  });
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(forgedActiveSequence),
    /Queue write-buffer ledger evidence/,
  );

  const multiPhase = completeShape();
  forgeUniformWriteRecord(multiPhase, 1, (record) => {
    record.capturePhase = 'phase0/v99/canonical/F/capture-901/production';
  });
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(multiPhase),
    /Queue write-buffer ledger evidence/,
  );

  const forgedOverlapObservation = completeShape({
    uniformRanges: OVERLAPPING_UNIFORM_WRITE_RANGES,
  });
  forgedOverlapObservation.evidence.resources.queueWriteBufferLedger
    .uniformWriteGroups[0].observedDestinationOverlapCount = 0;
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(forgedOverlapObservation),
    /Queue write-buffer ledger evidence/,
  );
});

test('schedule installation contract binds four manager updates and three frozen commands', () => {
  const installation = scheduleInstallation('v99', 'S1', 2,
    coverageSnapshot(IMMEDIATE_AIF_PHASE0_SNAPSHOT_PLAN[1]).productionChallenge);
  assert.equal(validateImmediateAifScheduleInstallationEvidence(installation), installation);
  const missingManagerCall = structuredClone(installation);
  missingManagerCall.managerCalls.pop();
  assert.throws(
    () => validateImmediateAifScheduleInstallationEvidence(missingManagerCall),
    /evidence is malformed/,
  );
  const aliasedCommand = structuredClone(installation);
  aliasedCommand.commands.I.frozenWords[4] = 1;
  assert.throws(
    () => validateImmediateAifScheduleInstallationEvidence(aliasedCommand),
    /evidence is malformed/,
  );
  const missingRawProjection = structuredClone(installation);
  delete missingRawProjection.commonState.postflight.rawProjection;
  assert.throws(
    () => validateImmediateAifScheduleInstallationEvidence(missingRawProjection),
    /evidence is malformed/,
  );
  const rendererMutation = structuredClone(installation);
  rendererMutation.commonState.postflight.rawProjection.renderer.trackTimestamp = true;
  assert.throws(
    () => validateImmediateAifScheduleInstallationEvidence(rendererMutation),
    /evidence is malformed/,
  );
});

test('page result shape requires 36 unique coverage cells plus restored/postflight audits', () => {
  const result = completeShape();
  assert.equal(validateImmediateAifPhase0PageResultShape(result), result);
  for (const cleanupFailure of [undefined, {
    name: 'Error', message: 'hidden cleanup failure', stack: null, detail: null,
  }]) {
    const invalidCleanup = structuredClone(result);
    if (cleanupFailure === undefined) delete invalidCleanup.cleanupFailure;
    else invalidCleanup.cleanupFailure = cleanupFailure;
    assert.throws(
      () => validateImmediateAifPhase0PageResultShape(invalidCleanup),
      /cannot retain a cleanup failure/,
    );
  }
  const ordinaryDepthPlan = structuredClone(result);
  ordinaryDepthPlan.evidence.plan.renderConfiguration.camera.reversedDepth = false;
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(ordinaryDepthPlan),
    /frozen Phase 0 render configuration/,
  );
  const ordinaryDepthResource = structuredClone(result);
  ordinaryDepthResource.evidence.resources.commonResources.camera.reversedDepth = false;
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(ordinaryDepthResource),
    /common resource evidence/,
  );
  const duplicate = structuredClone(result);
  duplicate.evidence.scenarios[0].snapshots[0].laneOrders.results.pop();
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(duplicate),
    /lane orders are incomplete/,
  );
  const missingAuditLane = structuredClone(result);
  missingAuditLane.evidence.scenarios[1].snapshots[4].laneCaptures.pop();
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(missingAuditLane),
    /audit captures are incomplete/,
  );
  const interleavedDiagnostic = structuredClone(result);
  interleavedDiagnostic.evidence.scenarios[0].snapshots[0]
    .productionChallenge.diagnosticRenderPassCount = 1;
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(interleavedDiagnostic),
    /production challenge is not isolated/,
  );
  const missingSchedulePreflight = structuredClone(result);
  missingSchedulePreflight.evidence.scenarios[1].scheduleInstallations.pop();
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(missingSchedulePreflight),
    /schedule preflights are incomplete|Queue write-buffer ledger evidence/,
  );
  const wrongStagingCount = structuredClone(result);
  wrongStagingCount.evidence.resources.stagingLifecycle.createdCount = 824;
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(wrongStagingCount),
    /common resource evidence/,
  );
  const terminalOnlyBufferCleanup = structuredClone(result);
  terminalOnlyBufferCleanup.cleanup.persistentBufferCleanup.records[0] = {
    pass: true,
    disposition: 'terminal-device-destroy',
    destroyCallCount: 0,
  };
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(terminalOnlyBufferCleanup),
    /persistent-buffer cleanup/,
  );
  const swappedVisibleUpload = structuredClone(result);
  swappedVisibleUpload.evidence.scenarios[0]
    .gpuResourceCommitments.visibleIdsUpload.bufferId = 'buffer-swapped';
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(swappedVisibleUpload),
    /GPU resource commitments/,
  );
  const missingMatrixWrite = structuredClone(result);
  missingMatrixWrite.evidence.scenarios[1].matrixGpuRealization.queueWriteBuffers = [];
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(missingMatrixWrite),
    /matrix realization|Queue write-buffer ledger evidence/,
  );
  const forgedPersistentDelta = structuredClone(result);
  forgedPersistentDelta.evidence.resources.persistentResourceFreezes.records[0]
    .prescribedPersistentDelta.addedBundleCount = 8;
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(forgedPersistentDelta),
    /Persistent resource freeze evidence/,
  );
  const missingRawGpuBytes = structuredClone(result);
  const [gpuWitness] = Object.values(
    missingRawGpuBytes.evidence.resources.gpuByteWitnesses.witnesses,
  );
  delete gpuWitness.bytesBase64;
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(missingRawGpuBytes),
    /GPU byte witness evidence/,
  );
  const forgedTypedWriteSelection = structuredClone(result);
  forgedTypedWriteSelection.evidence.resources.queueWriteBufferLedger.records[0]
    .selectedSourceByteLength += 4;
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(forgedTypedWriteSelection),
    /Queue write-buffer ledger evidence/,
  );
  const hiddenTargetViewUse = structuredClone(result);
  hiddenTargetViewUse.evidence.resources.textureDestructionBeforeCleanup
    .textureInventory[0].boundViewIds.push('texture-view-0');
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(hiddenTargetViewUse),
    /Texture inventory evidence/,
  );
  const computeLayoutLeak = structuredClone(result);
  computeLayoutLeak.evidence.pipelines.layouts[0].immediateSize = 4;
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(computeLayoutLeak),
    /Compute immediate-state exclusion/,
  );
  const swappedReadbackSource = structuredClone(result);
  swappedReadbackSource.evidence.resources.stagingLifecycle.transferLedger[0]
    .sourceBinding = swappedReadbackSource.evidence.resources.stagingLifecycle
      .sourceInventory.textureSources[0];
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(swappedReadbackSource),
    /Global command\/readback ledger evidence/,
  );
  const latePersistentMap = structuredClone(result);
  latePersistentMap.evidence.resources.bufferMapLifecycle.records.at(-1)
    .mapEvents.push({ sequence: 999, method: 'mapAsync' });
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(latePersistentMap),
    /Buffer map lifecycle evidence/,
  );
  const earlyTextureDestroy = structuredClone(result);
  earlyTextureDestroy.cleanup.persistentTextureCleanup.records[0].destroySequence =
    earlyTextureDestroy.cleanup.cleanupStartSequence;
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(earlyTextureDestroy),
    /persistent-buffer cleanup/,
  );
  const timingLeak = structuredClone(result);
  timingLeak.timingCaptured = true;
  assert.throws(
    () => validateImmediateAifPhase0PageResultShape(timingLeak),
    /correctness-only claim boundary/,
  );
});

test('target and frozen-plan contracts reject unbound or partial inputs', () => {
  const target = {
    schemaVersion: 1,
    key: 'r185',
    sourceFamily: 'installed-r185-overlay',
    expectedRevision: '185',
    scope: 'full Phase 0',
    implemented: ['A', 'I', 'F'],
    deferred: [],
    rawExpectation: { reportSha256: 'a'.repeat(64) },
    observationNonce: '0123456789abcdef0123456789abcdef',
  };
  assert.equal(validateImmediateAifPhase0TargetConfiguration(target), target);
  assert.throws(
    () => validateImmediateAifPhase0TargetConfiguration({ ...target, key: '' }),
    /malformed/,
  );
  assert.throws(
    () => validateImmediateAifPhase0TargetConfiguration({ ...target, rawExpectation: null }),
    /rawExpectation/,
  );
  assert.throws(
    () => validateImmediateAifPhase0TargetConfiguration({
      ...target,
      observationNonce: 'ABCDEF',
    }),
    /observationNonce/,
  );
  const plan = immediateAifPhase0PlanEvidence();
  assert.deepEqual(plan.lanes, ['A', 'I', 'F']);
  assert.equal(plan.workload.objectCount, 65_536);
  assert.equal(plan.workload.bucketCount, 32);
  assert.equal(plan.workload.geometryTier, 'medium');
  assert.equal(plan.workload.layout, 'baseline');
  assert.equal(plan.workload.seed, 0xb1ad_2026);
  assert.equal(plan.workload.commandByteLength, 640);
  assert.equal(plan.workload.bucketBases.at(-1), 63_488);
  assert.deepEqual(
    plan.snapshotPlan.map((entry) => entry.label),
    ['canonical-preflight', 'S1', 'S2', 'canonical-restored', 'canonical-postflight'],
  );
});

test('nonce observation commitments use exact ordered u32be UTF-8 framing', async () => {
  const names = ['alpha', 'beta'];
  const values = ['A', '\u{1f680}'];
  const commitment = await commitImmediateAifObservationFields(names, values);
  const framed = Buffer.concat(values.map((value) => {
    const bytes = Buffer.from(value, 'utf8');
    const prefix = Buffer.alloc(4);
    prefix.writeUInt32BE(bytes.length);
    return Buffer.concat([prefix, bytes]);
  }));
  assert.equal(commitment.encoding, IMMEDIATE_AIF_PHASE0_OBSERVATION_ENCODING);
  assert.deepEqual(commitment.fieldNames, names);
  assert.equal(commitment.sha256, createHash('sha256').update(framed).digest('hex'));
  assert.deepEqual(
    commitment.fieldCommitments.map(({ name, byteLength }) => ({ name, byteLength })),
    [{ name: 'alpha', byteLength: 1 }, { name: 'beta', byteLength: 4 }],
  );
});

test('nonce challenge commits every locked root field and per-use chain', async () => {
  const nonce = '0123456789abcdef0123456789abcdef';
  const chainEvidence = { pipelineId: 'pipeline-1', output: { sha256: 'a'.repeat(64) } };
  const chain = await createImmediateAifObservationChain({
    nonce,
    chainId: 'v99/S1/order-0-AIF/position-0-A/production',
    kind: 'immediate-aif-phase0-production-observation-chain',
    chainEvidence,
  });
  assert.deepEqual(chain.fieldNames, [...IMMEDIATE_AIF_PHASE0_CHAIN_FIELD_NAMES]);
  const evidence = {
    shaders: {},
    pipelines: {},
    commands: {},
    scenarios: [],
    addressWitnesses: {},
    outputWitnesses: {},
    resources: {},
    lifecycle: {
      renderBundleTraces: [],
      renderPassTraces: [],
      computePassTraces: [],
      instrumentationMarkers: [],
    },
  };
  const challenge = await createImmediateAifObservationChallenge({
    nonce,
    target: { key: 'r185' },
    evidence,
    chains: [chain],
  });
  assert.equal(challenge.kind, 'immediate-aif-phase0-observation-challenge');
  assert.deepEqual(challenge.fieldNames, [...IMMEDIATE_AIF_PHASE0_OBSERVATION_FIELD_NAMES]);
  assert.equal(challenge.fieldCount, 12);
  assert.equal(challenge.chainCount, 1);
  const changed = await createImmediateAifObservationChallenge({
    nonce,
    target: { key: 'pinned-dev' },
    evidence,
    chains: [chain],
  });
  assert.notEqual(challenge.sha256, changed.sha256);
});

test('browser entry is runner-bound, publishes one deterministic global, and has no timer seam', async () => {
  const [source, html] = await Promise.all([
    readFile('src/phase0/immediate-aif-page.js', 'utf8'),
    readFile('phase0-immediate-aif.html', 'utf8'),
  ]);
  assert.equal(IMMEDIATE_AIF_PHASE0_PAGE_GLOBAL, '__THREE_IMMEDIATE_AIF_PHASE0__');
  assert.match(source, /\/\* THREE_IMMEDIATE_PHASE0_TARGET_CONFIGURATION_V1 \*\/ null/);
  assert.match(source, /window\[IMMEDIATE_AIF_PHASE0_PAGE_GLOBAL\] = published/);
  assert.match(source, /runtimeCoverage/);
  assert.match(source, /mutateAndDetachImmediateBaseSource/);
  assert.match(source, /visibleIdsGpuBufferId/);
  assert.match(source, /gpuResourceCommitments/);
  assert.match(source, /observationChallenge/);
  assert.match(source, /creationOrdinal/);
  assert.match(source, /renderPassDescriptorSnapshot/);
  assert.match(source, /queue\.writeBuffer/);
  assert.match(source, /queue\.writeTexture/);
  assert.match(source, /bufferMapEvents/);
  assert.match(source, /EXPECTED_READBACK_STAGING_COUNT = 825/);
  assert.match(source, /EXPECTED_READBACK_PHASE_COUNT = 666/);
  assert.match(source, /THREE_ATTRIBUTE_TYPE_INDEX = 2/);
  assert.match(source, /THREE_ATTRIBUTE_TYPE_STORAGE = 3/);
  assert.match(source, /async function realizeScenarioMatrixAttribute/);
  assert.match(source, /async function realizeMergedGeometryForFreeze/);
  assert.match(source, /async function realizeScheduleInstallation/);
  assert.match(source, /manager\.update\(call\.attribute, call\.attributeType\)/);
  assert.match(source, /manager\.update\(attribute, THREE_ATTRIBUTE_TYPE_STORAGE\)/);
  assert.match(source, /async function captureCommonRenderStateCommitment/);
  assert.match(source, /productionLightProjection/);
  assert.match(source, /realizeAndFreezeImmediateAifTransform\(hemisphere\)/);
  assert.match(source, /realizeAndFreezeImmediateAifTransform\(directional\)/);
  assert.match(source, /realizeAndFreezeImmediateAifTransform\(directional\.target\)/);
  assert.match(source, /realizeImmediateAifReversedDepthCamera\(camera, renderer\)/);
  assert.match(source, /reversedDepth: camera\.reversedDepth/);
  assert.match(source, /gpuByteWitnesses/);
  assert.match(source, /textureViews/);
  assert.match(source, /pipelineDescriptor/);
  assert.match(source, /colorTargetExact/);
  assert.match(source, /topology\.activeRootCount === 1/);
  assert.match(source, /topology\.activeLane === 'F'/);
  assert.match(source, /rawEvents: events/);
  assert.match(source, /method === 'executeBundles'/);
  assert.match(source, /function pipelineBindingTopologyEvidence/);
  assert.match(source, /shaderDeclaredBindingCoordinates/);
  assert.match(source, /function bindingLayoutSemanticsEvidence/);
  assert.match(source, /layout\.bindGroupLayoutIds\.length === expectedGroupIndices\.length/);
  assert.match(source, /bindGroupLayouts\.length === expectedGroupIndices\.length/);
  assert.match(source, /bindingSemantics\.every\(\(record\) => record\.pass\)/);
  assert.match(source, /immediateAifExpectedBindingVisibility/);
  assert.match(source, /pinned-three-node-uniforms-all-stages/);
  assert.match(source, /type: entry\.buffer\.type \?\? 'uniform'/);
  assert.match(source, /shaderLayoutBindingTopologyExact/);
  assert.match(source, /shaderBindingSemantics\.every\(\(binding\) => binding\.pass\)/);
  assert.match(source, /uniformWriteGrammarExact/);
  assert.match(source, /uniformWriteGroupsExact/);
  assert.match(source, /activeBindGroupUseWitness/);
  assert.match(source, /activeTargetBindGroupMatchCount/);
  assert.match(source, /selectedTraceProof/);
  assert.match(source, /sourceDestinationOffsetExact/);
  assert.match(source, /overlapPolicy: 'observed-not-prescribed'/);
  assert.match(source, /liveAttributeWriteGrammarExact/);
  assert.match(source, /indirectCommandAccess/);
  assert.doesNotMatch(source, /performance\.now\s*\(/);
  assert.doesNotMatch(source, /Date\.now\s*\(/);
  assert.doesNotMatch(source, /trackTimestamp\s*:\s*true/);
  assert.doesNotMatch(source, /targetPadding/);
  const callbackStart = source.indexOf('async function captureLaneCallback');
  const callbackEnd = source.indexOf('function snapshotRelationalEvidence', callbackStart);
  assert.ok(callbackStart >= 0 && callbackEnd > callbackStart);
  assert.doesNotMatch(source.slice(callbackStart, callbackEnd), /Promise\.all/);
  const productionOnlyStart = source.indexOf('async function captureProductionLaneCallback');
  const enrichmentStart = source.indexOf('async function enrichLaneCallback');
  const wrapperStart = source.indexOf('async function captureLaneCallback');
  assert.ok(productionOnlyStart >= 0
    && enrichmentStart > productionOnlyStart
    && wrapperStart > enrichmentStart);
  const productionOnlySource = source.slice(productionOnlyStart, enrichmentStart);
  assert.doesNotMatch(productionOnlySource, /captureAddressOutput|captureObjectIdOutput/);
  const coverageStart = source.indexOf('async function runCoverageSnapshot');
  const coverageEnd = source.indexOf('function selectLaneForAudit', coverageStart);
  const coverageSource = source.slice(coverageStart, coverageEnd);
  const orderCall = coverageSource.indexOf('strategy.runAllLaneOrders');
  const challengeComplete = coverageSource.indexOf(
    "'phase0-ordered-production-challenge-complete'",
  );
  const enrichment = coverageSource.indexOf('await enrichLaneCallback');
  assert.ok(orderCall >= 0 && challengeComplete > orderCall && enrichment > challengeComplete);
  assert.match(coverageSource, /captureProductionLaneCallback/);
  assert.match(source, /immutableBufferWriteCount/);
  const scenarioStart = source.indexOf('async function runScenario');
  const scenarioEnd = source.indexOf('async function executePhase0', scenarioStart);
  const scenarioSource = source.slice(scenarioStart, scenarioEnd);
  const matrixRealization = scenarioSource.indexOf('realizeScenarioMatrixAttribute');
  const liveValidationStart = scenarioSource.indexOf('strategy.beginLiveValidation');
  const visibleFreeze = scenarioSource.indexOf('strategy.freezeValidatedVisibleIds');
  const sharedCommitment = scenarioSource.indexOf('captureSharedGpuResourceCommitments');
  const primeStart = scenarioSource.indexOf("if (scenarioId === 'v99')");
  const geometryRealization = scenarioSource.indexOf('realizeMergedGeometryForFreeze');
  const activeLaneSelection = scenarioSource.indexOf("strategy.selectActiveLane('A')");
  const initialSchedulePreflight = scenarioSource.indexOf('realizeScheduleInstallation');
  const authoritativeFreeze = scenarioSource.indexOf('capturePersistentResourceInventory');
  assert.ok(matrixRealization >= 0 && matrixRealization < liveValidationStart);
  assert.ok(visibleFreeze > liveValidationStart && sharedCommitment > visibleFreeze);
  assert.ok(primeStart > sharedCommitment
    && geometryRealization > primeStart
    && activeLaneSelection > geometryRealization
    && initialSchedulePreflight > activeLaneSelection
    && authoritativeFreeze > initialSchedulePreflight);
  assert.match(source, /destroyUndestroyedPersistentBuffers/);
  assert.match(html, /src="\/src\/phase0\/immediate-aif-page\.js"/);
  assert.match(html, /id="status"/);
  assert.match(html, /id="canvas-host"/);
});
