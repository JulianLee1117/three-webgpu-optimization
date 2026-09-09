import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  adapterMatchesStaticGpu,
  browserGpuMatchesStaticIdentity,
  closeThreeImmediatePhase0IntegrationState,
  commitThreeImmediatePhase0ObservationFields,
  createThreeImmediatePhase0ExpectedCommandWords,
  equalThreeImmediatePhase0PersistedNumberArrays,
  injectThreeImmediatePhase0Target,
  normalizeThreeImmediatePhase0ComputeShader,
  normalizeThreeImmediatePhase0DiagnosticShaders,
  normalizeThreeImmediatePhase0RenderShaders,
  parseThreeImmediatePhase0Arguments,
  reconstructThreeImmediatePhase0UniformQueueWriteEvidence,
  reconstructThreeImmediatePhase0DependencyClosure,
  THREE_IMMEDIATE_PHASE0_EXPECTED_CHROME_IDENTITY,
  validateThreeImmediatePhase0AddressWitnesses,
  validateThreeImmediatePhase0AddressDiagnosticPositions,
  validateThreeImmediatePhase0BufferMapLifecycle,
  validateThreeImmediatePhase0BundleEventGrammar,
  validateThreeImmediatePhase0ChallengeWriteClosure,
  validateThreeImmediatePhase0CommandSubmissionClosure,
  validateThreeImmediatePhase0DiagnosticVertexInputs,
  validateThreeImmediatePhase0DiagnosticStorageBindings,
  validateThreeImmediatePhase0DeviceInstrumentationClosure,
  validateThreeImmediatePhase0DepthStencilAttachmentDefaults,
  validateThreeImmediatePhase0ErrorScopeDrainage,
  validateThreeImmediatePhase0GlobalCommandClosure,
  validateThreeImmediatePhase0LoadedSourceClosure,
  validateThreeImmediatePhase0OutputWitnesses,
  validateThreeImmediatePhase0PipelineLayoutImmediateState,
  validateThreeImmediatePhase0PinnedRenderConfiguration,
  validateThreeImmediatePhase0PinnedGeometryManifests,
  validateThreeImmediatePhase0PageResult,
  validateThreeImmediatePhase0ProductionReadbackTransfers,
  validateThreeImmediatePhase0ProductionStorageBindings,
  validateThreeImmediatePhase0QueueWriteBufferLedger,
  validateThreeImmediatePhase0ReadbackSourceBijection,
  validateThreeImmediatePhase0RenderPassDescriptorDefaults,
  validateThreeImmediatePhase0RenderPipelineDescriptor,
  validateThreeImmediatePhase0Report,
  validateThreeImmediatePhase0ShaderDeclaredPipelineLayouts,
  validateThreeImmediatePhase0UniformQueueWriteEvidence,
  verifyThreeImmediatePhase0Artifact,
} from '../scripts/probe-three-immediate-phase0.mjs';
import {
  createImmediateAifAddressOracle,
  createImmediateAifCommandOracle,
  createImmediateAifPhase0Scenarios,
  IMMEDIATE_AIF_PHASE0_LANES,
  IMMEDIATE_AIF_PHASE0_LANE_ORDERS,
  IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION,
  IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS,
  IMMEDIATE_AIF_PHASE0_SCHEDULES,
  IMMEDIATE_AIF_PHASE0_WORKLOAD,
} from '../src/phase0/immediate-aif-plan.js';
import { createImmediateAifPackedAddressBytes } from '../src/phase0/immediate-aif-page-contract.js';
import { createIndexedGeometryFixtures } from '../src/scenes/geometry-fixtures.js';
import {
  fingerprintFixedSubsetScenario,
  fingerprintImmediateAifPhase0GeometryFixtures,
} from '../src/scenes/geometry-fingerprints.js';
import { normalizeLiveIndirectCommandComputeShader } from
  '../src/validation/live-compute-shader-normalization.js';
import { normalizeImmediateAifRenderShaderLanes } from
  '../src/validation/first-instance-shader-evidence.js';

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const TARGET_PIXELS = 1_280 * 720;
const TARGET_BYTES = TARGET_PIXELS * 4;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runnerPath = path.join(repoRoot, 'scripts', 'probe-three-immediate-phase0.mjs');

test('runner pins the exact reversed-depth Phase 0 render configuration', () => {
  const bytes = Buffer.from(JSON.stringify(IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION));
  assert.equal(bytes.byteLength, 7_389);
  assert.equal(sha256(bytes),
    'ef63b73377dbb5f65548ea06cf6a706861544e788478a2d6919c3c617ce20a64');
  assert.equal(validateThreeImmediatePhase0PinnedRenderConfiguration(
    IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION,
  ), true);
  assert.deepEqual(IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION.camera.projectionMatrix, [
    1.2062851427866268, 0, 0, 0,
    0, 2.1445069205095586, 0, 0,
    0, 0, 0.00010001000100010001, -1,
    0, 0, 0.10001000100010002, 0,
  ]);
  assert.deepEqual(IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION.camera.projectionMatrixInverse, [
    0.8289913922755531, 0, 0, 0,
    0, 0.46630765815499864, 0, 0,
    0, 0, 0, 9.998999999999999,
    0, 0, -1, 0.001,
  ]);
  for (const mutate of [
    (configuration) => { configuration.camera.reversedDepth = false; },
    (configuration) => { configuration.camera.projectionMatrix[10] = -1.0002; },
    (configuration) => { configuration.camera.projectionMatrixInverse[11] = 10; },
  ]) {
    const changed = structuredClone(IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION);
    mutate(changed);
    assert.equal(validateThreeImmediatePhase0PinnedRenderConfiguration(changed), false);
  }
  const circular = {};
  circular.self = circular;
  const throwingAccessor = {};
  Object.defineProperty(throwingAccessor, 'camera', {
    enumerable: true,
    get() { throw new Error('must not be evaluated'); },
  });
  for (const malformed of [
    null, undefined, Symbol('configuration'), 1n, () => {},
    { camera: undefined }, { camera: Number.NaN }, circular, throwingAccessor,
  ]) {
    assert.doesNotThrow(() => validateThreeImmediatePhase0PinnedRenderConfiguration(malformed));
    assert.equal(validateThreeImmediatePhase0PinnedRenderConfiguration(malformed), false);
  }
});

function scenarioSet() {
  const geometries = createIndexedGeometryFixtures(32, 'medium');
  try {
    return createImmediateAifPhase0Scenarios({
      geometrySpheres: geometries.map((geometry) => geometry.boundingSphere.clone()),
    });
  } finally {
    geometries.forEach((geometry) => geometry.dispose());
  }
}

function phase0VertexInputs(feature) {
  const position = {
    name: 'position', shaderLocation: 0, format: 'float32x3', stepMode: 'vertex',
    arrayType: 'Float32Array', itemSize: 3, normalized: false,
    count: 6_440, resourceId: 700,
  };
  const normal = {
    name: 'normal', shaderLocation: feature ? 1 : 2, format: 'float32x3',
    stepMode: 'vertex', arrayType: 'Float32Array', itemSize: 3,
    normalized: false, count: 6_440, resourceId: 701,
  };
  return feature ? [position, normal] : [position, {
    name: 'bucketBase', shaderLocation: 1, format: 'uint32', stepMode: 'vertex',
    arrayType: 'Uint32Array', itemSize: 1, normalized: false,
    count: 6_440, resourceId: 702,
  }, normal];
}

function diagnosticVertexInputs(lane, purpose = 'address', positionResourceId = 800) {
  const position = {
    name: 'position', shaderLocation: 0, format: 'float32x3', stepMode: 'vertex',
    arrayType: 'Float32Array', itemSize: 3, normalized: false,
    count: 6_440, resourceId: positionResourceId,
  };
  if (lane !== 'A') return [position];
  const bucketBase = {
    name: 'bucketBase', shaderLocation: 0, format: 'uint32', stepMode: 'vertex',
    arrayType: 'Uint32Array', itemSize: 1, normalized: false,
    count: 6_440, resourceId: 702,
  };
  return purpose === 'address'
    ? [bucketBase, { ...position, shaderLocation: 1 }]
    : [position, { ...bucketBase, shaderLocation: 1 }];
}

function normalErrorScopeDrainageFixture() {
  const scopedErrors = {
    'out-of-memory': null,
    internal: null,
    validation: null,
  };
  const drainage = {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-error-scope-drainage',
    pass: true,
    applicable: true,
    trigger: 'normal-completion',
    pushOrder: ['validation', 'internal', 'out-of-memory'],
    expectedPopOrder: ['out-of-memory', 'internal', 'validation'],
    popOrder: ['out-of-memory', 'internal', 'validation'],
    records: [
      { filter: 'out-of-memory', settled: true, returnedError: null, popFailure: null },
      { filter: 'internal', settled: true, returnedError: null, popFailure: null },
      { filter: 'validation', settled: true, returnedError: null, popFailure: null },
    ],
    scopedErrors,
    scopedErrorsComplete: true,
    scopedErrorsEmpty: true,
  };
  return {
    gpuErrors: {
      scopeDrainage: structuredClone(drainage),
      scoped: structuredClone(scopedErrors),
    },
    evidence: { lifecycle: {
      scopedErrors: structuredClone(scopedErrors),
      cleanup: { errorScopeDrainage: structuredClone(drainage) },
    } },
    cleanup: { errorScopeDrainage: structuredClone(drainage) },
  };
}

function exhaustiveReadbackSourceFixture() {
  const sharedLengths = {
    matrix: 4_194_304, bounds: 1_048_576, objectBucket: 262_144,
    bucketBase: 128, bucketCapacity: 128, cullOrder: 262_144,
    visibleIds: 262_144, overflow: 4,
  };
  const shared = Object.fromEntries(Object.entries(sharedLengths).map(
    ([semantic, byteLength], index) => [semantic, {
      semantic, attributeId: 100 + index, gpuBufferId: `shared-${semantic}`, byteLength,
    }],
  ));
  const command = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane, index) => [
    lane, { attributeId: 200 + index, gpuBufferId: `command-${lane}`, byteLength: 640 },
  ]));
  const mergedLengths = {
    bucketBase: 25_760, normal: 77_280, position: 77_280, uv: 51_520, index: 49_464,
  };
  const merged = Object.fromEntries(Object.entries(mergedLengths).map(
    ([semantic, byteLength], index) => [semantic, {
      semantic, attributeId: 300 + index, gpuBufferId: `merged-${semantic}`, byteLength,
    }],
  ));
  const targetSemantics = [
    'production.color', 'production.depth', 'address.color',
    'objectId.color', 'objectId.depth',
  ];
  const targets = Object.fromEntries(targetSemantics.map((semantic, index) => [semantic, {
    semantic, textureUuid: `texture-uuid-${index}`, gpuTextureId: `texture-${semantic}`,
  }]));
  const bufferSources = [
    ...Object.keys(sharedLengths).map((semantic) => ({
      semantic: `shared.${semantic}`, resourceKind: 'buffer',
      attributeId: shared[semantic].attributeId, resourceId: shared[semantic].gpuBufferId,
    })),
    ...IMMEDIATE_AIF_PHASE0_LANES.map((lane) => ({
      semantic: `command.${lane}`, resourceKind: 'buffer',
      attributeId: command[lane].attributeId, resourceId: command[lane].gpuBufferId,
    })),
    ...Object.keys(mergedLengths).map((semantic) => ({
      semantic: `merged-geometry.${semantic}`, resourceKind: 'buffer',
      attributeId: merged[semantic].attributeId, resourceId: merged[semantic].gpuBufferId,
    })),
  ];
  const textureSources = targetSemantics.map((semantic) => ({
    semantic, resourceKind: 'texture', textureUuid: targets[semantic].textureUuid,
    resourceId: targets[semantic].gpuTextureId,
  }));
  const callbackEntries = [];
  let serial = 1;
  const schedules = ['canonical', 'S1', 'S2', 'canonical', 'canonical'];
  const scenarios = ['v99', 'v20'].map((scenarioId, scenarioIndex) => {
    const snapshots = schedules.map((scheduleId, snapshotIndex) => {
      const installationOrdinal = scenarioIndex * 4 + Math.min(snapshotIndex + 1, 4);
      const installationId = `${scenarioId}/${installationOrdinal}/${scheduleId}`;
      const callback = (lane) => {
        const phase = `phase0/${scenarioId}/${scheduleId}/${lane}/capture-${serial++}`;
        const value = {
          lane, phase, scheduleInstallationId: installationId,
          phases: { production: `${phase}/production` },
        };
        callbackEntries.push({ lane, phase, callback: value });
        return value;
      };
      const rerecord = [1, 2, 3].includes(snapshotIndex)
        ? { callback: callback('I') } : null;
      if (snapshotIndex < 3) {
        return {
          rerecord,
          laneOrders: {
            results: IMMEDIATE_AIF_PHASE0_LANE_ORDERS.map((order) => ({
              order,
              records: order.map((lane) => ({ callback: callback(lane) })),
            })),
          },
        };
      }
      return {
        rerecord,
        laneCaptures: IMMEDIATE_AIF_PHASE0_LANES.map(
          (lane) => ({ callback: callback(lane) }),
        ),
      };
    });
    return {
      scenarioId,
      gpuResourceCommitments: { records: Object.values(shared) },
      snapshots,
    };
  });
  const installations = ['v99', 'v20'].flatMap((scenarioId, scenarioIndex) => (
    ['canonical', 'S1', 'S2', 'canonical'].map((scheduleId, localIndex) => {
      const ordinal = scenarioIndex * 4 + localIndex + 1;
      return {
        ordinal,
        basePhase: `phase0/schedule-install/${scenarioId}/${ordinal}-${scheduleId}`,
        commands: command,
      };
    })
  ));
  const specs = [];
  const addBuffer = (phase, semantic, source, byteLength) => specs.push({
    phase, semantic, kind: 'buffer', resourceId: source.gpuBufferId,
    byteLength, sourceBinding: bufferSources.find((record) => record.semantic === semantic),
  });
  const addTexture = (phase, semantic) => {
    const address = semantic === 'address.color';
    const width = address ? 256 : 1_280;
    const height = address ? 256 : 720;
    specs.push({
      phase, semantic, kind: 'texture', resourceId: targets[semantic].gpuTextureId,
      byteLength: width * height * 4, width, height,
      sourceBinding: textureSources.find((record) => record.semantic === semantic),
    });
  };
  for (const entry of callbackEntries) {
    addTexture(`${entry.phase}/production`, 'production.color');
    addTexture(`${entry.phase}/production`, 'production.depth');
    addBuffer(`${entry.phase}/command-readback`, `command.${entry.lane}`,
      command[entry.lane], 640);
    addTexture(`${entry.phase}/address`, 'address.color');
    addTexture(`${entry.phase}/object-id`, 'objectId.color');
    addBuffer(`${entry.phase}/visible-id-readback`, 'shared.visibleIds',
      shared.visibleIds, 262_144);
  }
  for (const scenarioId of ['v99', 'v20']) {
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      const phase = `phase0/live/${scenarioId}/${lane}`;
      addBuffer(phase, 'shared.overflow', shared.overflow, 4);
      addBuffer(phase, `command.${lane}`, command[lane], 640);
      addBuffer(phase, 'shared.visibleIds', shared.visibleIds, 262_144);
    }
    for (const semantic of Object.keys(sharedLengths)) {
      addBuffer(`phase0/resources/${scenarioId}/shared-commitment`,
        `shared.${semantic}`, shared[semantic], shared[semantic].byteLength);
    }
  }
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    addTexture(`phase0/prime/${lane}`, 'production.color');
    addTexture(`phase0/prime/${lane}`, 'production.depth');
  }
  for (const semantic of Object.keys(mergedLengths)) {
    addBuffer('phase0/resources/global/geometry-postflight',
      `merged-geometry.${semantic}`, merged[semantic], merged[semantic].byteLength);
  }
  for (const installation of installations) {
    for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
      addBuffer(`${installation.basePhase}/command-readback/${lane}`,
        `command.${lane}`, command[lane], 640);
    }
  }
  assert.equal(specs.length, 825);
  const stagingCreations = [];
  const transfers = [];
  const transferLedger = [];
  for (const [index, spec] of specs.entries()) {
    const stagingBufferId = `readback-staging-${index}`;
    const transfer = spec.kind === 'buffer' ? {
      sequence: index + 1, capturePhase: spec.phase, method: 'copyBufferToBuffer',
      sourceBufferId: spec.resourceId, sourceOffset: 0,
      destinationBufferId: stagingBufferId, destinationOffset: 0, size: spec.byteLength,
    } : {
      sequence: index + 1, capturePhase: spec.phase, method: 'copyTextureToBuffer',
      source: { textureId: spec.resourceId, mipLevel: 0,
        origin: { x: 0, y: 0, z: 0 }, aspect: 'all' },
      destination: { bufferId: stagingBufferId, offset: 0,
        bytesPerRow: spec.width * 4, rowsPerImage: null },
      copySize: { width: spec.width, height: spec.height, depthOrArrayLayers: 1 },
    };
    const staging = {
      resourceId: stagingBufferId, capturePhase: spec.phase,
      size: spec.byteLength, usage: 9, mappedAtCreation: false,
    };
    stagingCreations.push(staging);
    transfers.push(transfer);
    transferLedger.push({
      capturePhase: spec.phase, stagingBufferId, transfer,
      sourceBinding: spec.sourceBinding,
    });
  }
  const evidence = {
    scenarios,
    resources: {
      scheduleInstallations: { records: installations },
      geometryFixtures: { mergedProduction: { gpuRecords: Object.values(merged) } },
      textureDestructionBeforeCleanup: { declaredTargetTextures: Object.values(targets) },
      stagingLifecycle: {
        sourceInventory: {
          schemaVersion: 1, kind: 'immediate-aif-phase0-readback-source-inventory',
          pass: true, bufferSourceCount: 16, textureSourceCount: 5,
          bufferSources, textureSources,
        },
      },
    },
  };
  return { evidence, stagingCreations, transfers, transferLedger };
}

function phase0StorageBindings() {
  return [{
    semantic: 'matrix', group: 1, binding: 3, access: 'read', visibility: 1,
    elementType: 'mat4x4<f32>', count: 65_536, byteLength: 65_536 * 64,
    resourceId: 'matrix-attribute',
  }, {
    semantic: 'visibleIds', group: 1, binding: 4, access: 'read', visibility: 'vertex',
    elementType: 'u32', count: 65_536, byteLength: 65_536 * 4,
    resourceId: 'visible-attribute',
  }];
}

function phase0VertexShader({ feature, immediate = false, matrix, visible }) {
  const header = immediate
    ? 'requires immediate_address_space;\n\n// immediate data\nvar<immediate> threeImmediateDrawBase : u32;\n'
    : '';
  const parameters = feature
    ? `fn main( @builtin( instance_index ) instanceIndex : u32,
\t@location( 0 ) position : vec3<f32>,
\t@location( 1 ) normal : vec3<f32> ) -> VaryingsStruct {`
    : `fn main( @builtin( instance_index ) instanceIndex : u32,
\t@location( 0 ) position : vec3<f32>,
\t@location( 1 ) bucketBase : u32,
\t@location( 2 ) normal : vec3<f32> ) -> VaryingsStruct {`;
  const address = immediate ? '( threeImmediateDrawBase + instanceIndex )'
    : feature ? 'instanceIndex' : '( bucketBase + instanceIndex )';
  return `${header}
struct ${matrix}Struct { value : array< mat4x4<f32> > };
@binding( 3 ) @group( 1 )
var<storage, read> ${matrix} : ${matrix}Struct;
struct ${visible}Struct { value : array< u32 > };
@binding( 4 ) @group( 1 )
var<storage, read> ${visible} : ${visible}Struct;
struct VaryingsStruct { @builtin( position ) position : vec4<f32> };
@vertex
${parameters}
\tvar objectMatrix : mat4x4<f32>;
\tobjectMatrix = ${matrix}.value[ ${visible}.value[ ${address} ] ];
\tvar varyings : VaryingsStruct;
\tvaryings.position = objectMatrix * vec4<f32>( position + normal * 0.0, 1.0 );
\treturn varyings;
}
`;
}

function phase0ShaderLanes() {
  const fragmentShader = '@fragment\nfn main() -> @location( 0 ) vec4<f32> { return vec4<f32>(1.0); }\n';
  return Object.fromEntries([
    ['A', false, false, 100], ['I', true, true, 200], ['F', true, false, 300],
  ].map(([lane, feature, immediate, base]) => [lane, {
    vertexShader: phase0VertexShader({
      feature, immediate, matrix: `NodeBuffer_${base}`, visible: `NodeBuffer_${base + 1}`,
    }),
    fragmentShader,
    vertexInputs: phase0VertexInputs(feature),
    storageBindings: phase0StorageBindings(),
  }]));
}

function diagnosticShaderFixture({ lane, purpose, base }) {
  const immediate = lane === 'I';
  const header = immediate
    ? 'requires immediate_address_space;\n\n// immediate data\nvar<immediate> threeImmediateDrawBase : u32;\n'
    : '';
  const parameters = lane === 'A'
    ? '@builtin( instance_index ) instanceIndex : u32,\n\t@location( 0 ) position : vec3<f32>,\n\t@location( 1 ) bucketBase : u32'
    : '@builtin( instance_index ) instanceIndex : u32,\n\t@location( 0 ) position : vec3<f32>';
  const address = lane === 'A' ? '( bucketBase + instanceIndex )'
    : lane === 'I' ? '( threeImmediateDrawBase + instanceIndex )' : 'instanceIndex';
  const visible = `NodeBuffer_${base + 1}`;
  const matrix = `NodeBuffer_${base}`;
  const declarations = `${purpose === 'object-id' ? `struct ${matrix}Struct { value : array< mat4x4<f32> > };\n@binding( 3 ) @group( 1 )\nvar<storage, read> ${matrix} : ${matrix}Struct;\n` : ''}struct ${visible}Struct { value : array< u32 > };\n@binding( 4 ) @group( 1 )\nvar<storage, read> ${visible} : ${visible}Struct;`;
  const addressVariable = purpose === 'object-id' ? 'phase0ObjectAddress' : 'phase0SourceAddress';
  const objectVariable = purpose === 'object-id' ? 'phase0ObjectId' : 'phase0AddressObjectId';
  const body = purpose === 'object-id'
    ? `${addressVariable} = ${address};\n\t${objectVariable} = ${visible}.value[ ${addressVariable} ];\n\tlet objectMatrix = ${matrix}.value[ ${objectVariable} ];\n\tlet clip = objectMatrix * vec4<f32>( position, 1.0 );`
    : `${addressVariable} = ${address};\n\t${objectVariable} = ${visible}.value[ ${addressVariable} ];\n\tlet clip = vec4<f32>( position, 1.0 );`;
  return {
    vertexShader: `${header}${declarations}\nvar<private> ${addressVariable} : u32;\nvar<private> ${objectVariable} : u32;\nstruct VaryingsStruct { @builtin( position ) position : vec4<f32> };\n@vertex\nfn main( ${parameters} ) -> VaryingsStruct {\n\t${body}\n\tvar varyings : VaryingsStruct;\n\tvaryings.position = clip + vec4<f32>( f32(${objectVariable}) * 0.0 );\n\treturn varyings;\n}\n`,
    fragmentShader: '@fragment\nfn main() -> @location( 0 ) vec4<f32> { return vec4<f32>(1.0); }\n',
    storageBindings: [
      ...(purpose === 'object-id' ? [{
        semantic: 'matrix', group: 1, binding: 3, access: 'read', visibility: 1,
        elementType: 'mat4x4<f32>', count: 65_536, byteLength: 4_194_304,
        resourceId: 'matrix-attribute',
      }] : []),
      {
        semantic: 'visibleIds', group: 1, binding: 4, access: 'read', visibility: 1,
        elementType: 'u32', count: 65_536, byteLength: 262_144,
        resourceId: 'visible-attribute',
      },
    ],
  };
}

function diagnosticShaderLanes(purpose) {
  return Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane, index) => [
    lane, diagnosticShaderFixture({ lane, purpose, base: 1_000 + index * 10 }),
  ]));
}

function addressPositionOracleFixture() {
  const geometries = createIndexedGeometryFixtures(32, 'medium');
  try {
    const spans = [];
    const firstIndexes = [];
    let vertexCount = 0;
    let indexCount = 0;
    for (const geometry of geometries) {
      spans.push({ start: vertexCount, end: vertexCount + geometry.getAttribute('position').count });
      firstIndexes.push(indexCount);
      vertexCount += geometry.getAttribute('position').count;
      indexCount += geometry.index.count;
    }
    const mergedIndex = new Uint32Array(indexCount);
    let indexCursor = 0;
    let vertexCursor = 0;
    for (const geometry of geometries) {
      for (const value of geometry.index.array) mergedIndex[indexCursor++] = value + vertexCursor;
      vertexCursor += geometry.getAttribute('position').count;
    }
    const build = (scheduleId, feature) => {
      const values = new Float32Array(vertexCount * 3);
      const bases = IMMEDIATE_AIF_PHASE0_SCHEDULES[scheduleId].sourceBaseByDraw;
      for (let bucket = 0; bucket < 32; bucket += 1) {
        const span = spans[bucket];
        const targetBase = bucket * 2_048;
        const z = feature ? targetBase - bases[bucket] : targetBase;
        for (let vertex = span.start; vertex < span.end; vertex += 1) {
          values[vertex * 3 + 2] = z;
        }
        const triangle = mergedIndex.slice(firstIndexes[bucket], firstIndexes[bucket] + 3);
        const xy = [-0.375, -0.375, 0.375, -0.375, 0, 0.375];
        triangle.forEach((vertex, corner) => {
          values[vertex * 3] = xy[corner * 2];
          values[vertex * 3 + 1] = xy[corner * 2 + 1];
        });
      }
      return values;
    };
    return {
      common: build('canonical', false),
      feature: Object.fromEntries(IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS.map(
        (scheduleId) => [scheduleId, build(scheduleId, true)],
      )),
    };
  } finally {
    geometries.forEach((geometry) => geometry.dispose());
  }
}

function addressDiagnosticPositionFixture() {
  const oracle = addressPositionOracleFixture();
  const buffers = { commonPosition: 'buffer-address-common', featurePosition: 'buffer-address-feature' };
  const attributes = { commonPosition: 9_001, featurePosition: 9_002 };
  const creationFor = (semantic) => ({
    sequence: semantic === 'commonPosition' ? 1 : 2,
    capturePhase: `phase0/diagnostic-prime/${semantic === 'commonPosition' ? 'A' : 'F'}/address`,
    resourceId: buffers[semantic],
    resourceClass: 'persistent-or-upload-buffer',
    size: 77_280,
    usage: 44,
    mappedAtCreation: true,
  });
  const gpuCreations = Object.keys(buffers).map((semantic) => ({
    method: 'createBuffer', ...creationFor(semantic),
  }));
  const queueWriteBufferCalls = [];
  const records = [];
  const scheduleInstallations = [];
  const installations = ['v99', 'v20'].flatMap((scenarioId, scenarioIndex) => (
    ['canonical', 'S1', 'S2', 'canonical'].map((scheduleId, localIndex) => ({
      scenarioId,
      scheduleId,
      ordinal: scenarioIndex * 4 + localIndex + 1,
    }))
  ));
  for (let index = 0; index < installations.length; index += 1) {
    const { scenarioId, scheduleId, ordinal } = installations[index];
    const installationId = `${scenarioId}/${ordinal}/${scheduleId}`;
    const base = 100 + index * 100;
    const capturePhase = `phase0/schedule-install/${scenarioId}/${ordinal}-${scheduleId}/diagnostic-position-realization`;
    const positionValues = {
      commonPosition: oracle.common,
      featurePosition: oracle.feature[scheduleId],
    };
    const positionRecords = ['commonPosition', 'featurePosition'].map((semantic, semanticIndex) => {
      const values = positionValues[semantic];
      const hash = sha256(Buffer.from(values.buffer, values.byteOffset, values.byteLength));
      const attributeVersion = semantic === 'commonPosition' ? 0 : index + 1;
      return {
        semantic,
        manager: 'renderer._attributes',
        managerConstructor: 'Attributes',
        attributeType: 1,
        attributeId: attributes[semantic],
        attributeVersion,
        beforeDataVersion: semantic === 'commonPosition' || index === 0
          ? attributeVersion : attributeVersion - 1,
        afterDataVersion: attributeVersion,
        dirtyBeforeUpdate: semantic === 'featurePosition' && index > 0,
        arrayType: 'Float32Array',
        itemSize: 3,
        count: 6_440,
        byteLength: 77_280,
        cpuExact: true,
        cpuSha256: hash,
        witnessId: `sha256-${hash}-77280`,
        gpuBufferId: buffers[semantic],
        gpuBufferSize: 77_280,
        gpuBufferCreation: creationFor(semantic),
        updateStartSequence: base + 21 + semanticIndex * 3,
        updateCompleteSequence: base + 23 + semanticIndex * 3,
      };
    });
    const queueWrites = index === 0 ? [] : [{
      sequence: base + 25,
      capturePhase,
      bufferId: buffers.featurePosition,
      bufferOffset: 0,
      sourceType: 'Float32Array',
      sourceByteLength: 77_280,
      dataOffset: 0,
      size: null,
    }];
    queueWriteBufferCalls.push(...queueWrites);
    const realization = {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-address-position-realization',
      pass: true,
      installationId,
      scenarioId,
      scheduleId,
      capturePhase,
      startSequence: base + 20,
      queueCompleteSequence: base + 30,
      expectedWriteSemantics: index === 0 ? [] : ['featurePosition'],
      observedWriteSemantics: index === 0 ? [] : ['featurePosition'],
      queueWriteBufferCount: queueWrites.length,
      queueWriteBuffers: queueWrites,
      queueWritesExact: true,
      records: positionRecords,
      laneBindings: {
        A: { semantic: 'commonPosition', attributeId: attributes.commonPosition,
          gpuBufferId: buffers.commonPosition },
        I: { semantic: 'commonPosition', attributeId: attributes.commonPosition,
          gpuBufferId: buffers.commonPosition },
        F: { semantic: 'featurePosition', attributeId: attributes.featurePosition,
          gpuBufferId: buffers.featurePosition },
      },
    };
    records.push(realization);
    scheduleInstallations.push({
      installationId,
      scenarioId,
      scheduleId,
      startSequence: base,
      queueCompleteSequence: base + 10,
      completeSequence: base + 50,
      diagnosticPositionRealization: structuredClone(realization),
      commands: Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map((lane, laneIndex) => [
        lane, {
          readbackStartSequence: base + 32 + laneIndex * 4,
          readbackCompleteSequence: base + 34 + laneIndex * 4,
        },
      ])),
    });
  }
  const distinct = [oracle.common, ...Object.values(oracle.feature)];
  const witnesses = Object.fromEntries(distinct.map((values) => {
    const bytes = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
    const hash = sha256(bytes);
    const witnessId = `sha256-${hash}-${bytes.byteLength}`;
    return [witnessId, {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-address-position-cpu-byte-witness',
      witnessId,
      encoding: 'base64-exact-bytes',
      byteLength: bytes.byteLength,
      sha256: hash,
      bytesBase64: bytes.toString('base64'),
    }];
  }).sort(([left], [right]) => left.localeCompare(right)));
  return {
    evidence: {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-address-diagnostic-position-evidence',
      pass: true,
      expectedRecordCount: 8,
      recordCount: 8,
      witnessCount: Object.keys(witnesses).length,
      witnesses,
      records,
    },
    lifecycle: { queueWriteBufferCalls },
    gpuCreations,
    scheduleInstallations,
  };
}

function computeShaderFixture(identifier = 'NodeBuffer_900', binding = 4) {
  return `struct FixedSliceIndexedDraw {
  indexCount : u32,
  instanceCount : atomic<u32>,
  firstIndex : u32,
  baseVertex : i32,
  firstInstance : u32
};
struct ${identifier}Struct { value : array<FixedSliceIndexedDraw> };
@binding( ${binding} ) @group( 0 )
var<storage, read_write> ${identifier} : ${identifier}Struct;
@compute @workgroup_size( 64, 1, 1 )
fn main() { atomicAdd( &${identifier}.value[ 0u ].instanceCount, 1u ); }`;
}

function computeBindingFixture(binding = 4) {
  return [{
    semantic: 'indirectCommands', kind: 'storage-buffer', group: 0, binding,
    access: 'readWrite', attributeType: 'IndirectStorageBufferAttribute',
    arrayType: 'Uint32Array', itemSize: 5, count: 32, byteLength: 640,
  }];
}

function pipelineDescriptorFixture({ purpose = 'production', lane = 'A' } = {}) {
  const inputs = purpose === 'production'
    ? phase0VertexInputs(lane !== 'A') : diagnosticVertexInputs(lane, purpose);
  return {
    label: `fixture-${purpose}-${lane}`,
    hasOwnImmediateSize: false,
    immediateSize: null,
    layoutId: 'pipeline-layout-1',
    vertex: {
      moduleId: 'shader-module-1', entryPoint: 'main', constants: null,
      buffers: inputs.map((input) => ({
        arrayStride: input.itemSize * 4,
        stepMode: input.stepMode,
        attributes: [{ format: input.format, offset: 0, shaderLocation: input.shaderLocation }],
      })),
    },
    fragment: {
      moduleId: 'shader-module-2', entryPoint: 'main', constants: null,
      targets: [{ format: 'rgba8unorm', writeMask: 15, blend: null }],
    },
    primitive: {
      topology: 'triangle-list', stripIndexFormat: null, frontFace: 'ccw',
      cullMode: purpose === 'production' ? 'back' : 'none', unclippedDepth: false,
    },
    depthStencil: purpose === 'address' ? null : {
      format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater-equal',
      stencilFront: null, stencilBack: null, stencilReadMask: 0xffff_ffff,
      stencilWriteMask: 0xffff_ffff, depthBias: 0, depthBiasSlopeScale: 0,
      depthBiasClamp: 0,
    },
    multisample: { count: 1, mask: 0xffff_ffff, alphaToCoverageEnabled: false },
  };
}

test('Phase 0 CLI ignores ambient browser overrides and help/verify stay non-launching', () => {
  assert.deepEqual(THREE_IMMEDIATE_PHASE0_EXPECTED_CHROME_IDENTITY, {
    canonicalPath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    productVersion: '152.0.7977.82',
    byteLength: 4_461_720,
    sha256: '8cd23aec3a30479b9b8db2063e70526c88a7ec2c99cd744603eb26ab598733ed',
  });
  const prior = process.env.BROWSER_PATH;
  process.env.BROWSER_PATH = 'C:/hostile/ambient/chromium.exe';
  try {
    assert.deepEqual(parseThreeImmediatePhase0Arguments(['--target', 'r185']), {
      target: 'r185',
      browser: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
      verify: null,
      help: false,
    });
  } finally {
    if (prior === undefined) delete process.env.BROWSER_PATH;
    else process.env.BROWSER_PATH = prior;
  }
  assert.equal(parseThreeImmediatePhase0Arguments(['--help']).help, true);
  assert.deepEqual(parseThreeImmediatePhase0Arguments(['--verify', 'artifact.json']), {
    target: null,
    browser: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    verify: 'artifact.json',
    help: false,
  });
  assert.throws(
    () => parseThreeImmediatePhase0Arguments(['--verify', 'x', '--browser', 'chrome.exe']),
    /cannot be combined/,
  );
});

test('offline verifier and executable CLI reject every falsey JSON root', async () => {
  const parent = path.join(repoRoot, '.generated', 'phase0-verifier-tests');
  await mkdir(parent, { recursive: true });
  const directory = await mkdtemp(path.join(parent, 'falsey-'));
  try {
    for (const [index, value] of [null, false, 0, ''].entries()) {
      const filename = path.join(directory, `falsey-${index}.json`);
      await writeFile(filename, `${JSON.stringify(value)}\n`);
      const direct = await verifyThreeImmediatePhase0Artifact(filename);
      assert.equal(direct.valid, false);
      assert.match(direct.reasons.join('\n'), /non-null plain object/u);
      const child = spawnSync(process.execPath, [runnerPath, '--verify', filename], {
        cwd: repoRoot, encoding: 'utf8', timeout: 10_000,
      });
      assert.equal(child.status, 1, child.stderr);
      assert.match(child.stdout, /non-null plain object/u);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('failed or absent page evidence fails closed and surfaces the primary page failure', () => {
  const failedPage = {
    schemaVersion: 1,
    kind: 'three-immediate-aif-phase0-page-result',
    status: 'technical-canary-failed',
    failure: {
      name: 'Error',
      message: 'address/A producer chain failed.',
      stack: 'Error: address/A producer chain failed.',
    },
    evidence: null,
  };
  const pageValidation = validateThreeImmediatePhase0PageResult(failedPage);
  assert.equal(pageValidation.valid, false);
  assert.match(pageValidation.reasons.join('\n'),
    /page reported technical-canary-failed: Error: address\/A producer chain failed\./u);
  assert.doesNotMatch(pageValidation.reasons.join('\n'), /error-scope drainage/u);

  const completedWithoutEvidence = {
    ...failedPage,
    status: 'technical-canary-complete',
    cleanupFailure: null,
    failure: null,
  };
  const missingEvidenceValidation = validateThreeImmediatePhase0PageResult(
    completedWithoutEvidence,
  );
  assert.equal(missingEvidenceValidation.valid, false);
  assert.match(missingEvidenceValidation.reasons.join('\n'), /absent or non-object evidence/u);

  for (const cleanupFailure of [undefined, {
    name: 'Error', message: 'hidden cleanup failure', stack: null, detail: null,
  }]) {
    const invalidCleanup = structuredClone(completedWithoutEvidence);
    if (cleanupFailure === undefined) delete invalidCleanup.cleanupFailure;
    else invalidCleanup.cleanupFailure = cleanupFailure;
    const validation = validateThreeImmediatePhase0PageResult(invalidCleanup);
    assert.equal(validation.valid, false);
    assert.match(validation.reasons.join('\n'), /retained a cleanup failure/u);

    const reportValidationWithInvalidCleanup = validateThreeImmediatePhase0Report({
      status: 'technical-canary-pass',
      target: { key: 'r185' },
      raw: { report: null },
      observationNonce: null,
      integration: {
        pageResult: { ...invalidCleanup, evidence: {} },
      },
    });
    assert.equal(reportValidationWithInvalidCleanup.valid, false);
    assert.match(
      reportValidationWithInvalidCleanup.reasons.join('\n'),
      /retained a cleanup failure/u,
    );
  }

  const failedReport = {
    status: 'technical-canary-failed',
    target: { key: 'r185' },
    integration: { pageResult: failedPage },
    failure: {
      name: 'Phase0PageValidationError',
      message: 'Phase 0 page failed independent validation: address/A producer chain failed.',
      stack: null,
      pageStatus: failedPage.status,
      pageFailure: structuredClone(failedPage.failure),
    },
  };
  const reportValidation = validateThreeImmediatePhase0Report(failedReport);
  assert.equal(reportValidation.valid, false);
  assert.match(reportValidation.reasons.join('\n'), /Phase0PageValidationError/u);
  assert.match(reportValidation.reasons.join('\n'), /address\/A producer chain failed\./u);
  assert.doesNotMatch(reportValidation.reasons.join('\n'), /error-scope drainage/u);

  const circular = {};
  circular.self = circular;
  for (const malformed of [null, undefined, false, 0, '', Symbol('page'), 1n, circular]) {
    const page = validateThreeImmediatePhase0PageResult(malformed);
    const report = validateThreeImmediatePhase0Report(malformed);
    assert.equal(page.valid, false);
    assert.equal(report.valid, false);
    assert.match(page.reasons.join('\n'), /strict JSON object/u);
    assert.match(report.reasons.join('\n'), /strict JSON object/u);
  }

  const hostilePrototype = new Proxy({}, {
    getPrototypeOf() {
      throw new Error('hostile getPrototypeOf trap');
    },
  });
  for (const validate of [
    validateThreeImmediatePhase0PageResult,
    validateThreeImmediatePhase0Report,
  ]) {
    const validation = validate(hostilePrototype);
    assert.equal(validation.valid, false);
    assert.match(validation.reasons.join('\n'), /strict JSON object/u);
  }

  const hostileGet = new Proxy({ status: 'technical-canary-failed' }, {
    get(target, property, receiver) {
      if (property === 'status') throw new Error('hostile status getter');
      return Reflect.get(target, property, receiver);
    },
  });
  const hostilePageValidation = validateThreeImmediatePhase0PageResult(hostileGet);
  const hostileReportValidation = validateThreeImmediatePhase0Report(hostileGet);
  assert.equal(hostilePageValidation.valid, false);
  assert.equal(hostileReportValidation.valid, false);
  assert.match(hostilePageValidation.reasons.join('\n'), /validation failed closed/u);
  assert.match(hostileReportValidation.reasons.join('\n'), /validation failed closed/u);
});

test('passing-page error scopes require exact normal-completion LIFO drainage', () => {
  const fixture = normalErrorScopeDrainageFixture();
  assert.deepEqual(validateThreeImmediatePhase0ErrorScopeDrainage(fixture), {
    valid: true,
    reasons: [],
  });
  for (const mutate of [
    (value) => { value.gpuErrors.scopeDrainage = null; },
    (value) => { value.gpuErrors.scopeDrainage = {}; },
    (value) => { value.gpuErrors.scopeDrainage.applicable = false; },
    (value) => { value.gpuErrors.scopeDrainage.pushOrder.reverse(); },
    (value) => { value.gpuErrors.scopeDrainage.popOrder.reverse(); },
    (value) => { value.gpuErrors.scopeDrainage.records.reverse(); },
    (value) => { value.gpuErrors.scopeDrainage.records[0].settled = false; },
    (value) => {
      value.gpuErrors.scopeDrainage.records[0].returnedError = {
        name: 'GPUValidationError', message: 'captured',
      };
    },
    (value) => {
      value.gpuErrors.scopeDrainage.records[0].popFailure = {
        name: 'Error', message: 'rejected',
      };
    },
    (value) => { value.gpuErrors.scopeDrainage.scopedErrors = {}; },
    (value) => { value.gpuErrors.scoped = {}; },
    (value) => { value.evidence.lifecycle.scopedErrors.internal = { name: 'x', message: 'y' }; },
    (value) => { value.cleanup.errorScopeDrainage = null; },
    (value) => { value.evidence.lifecycle.cleanup.errorScopeDrainage.trigger = 'failure-teardown'; },
  ]) {
    const tampered = normalErrorScopeDrainageFixture();
    mutate(tampered);
    const validation = validateThreeImmediatePhase0ErrorScopeDrainage(tampered);
    assert.equal(validation.valid, false);
    assert.match(validation.reasons.join('\n'), /error-scope drainage/u);
  }
});

test('cleanup attempts context, browser, and server after the first close failure', async () => {
  const calls = [];
  const state = {
    diagnostics: { shutdownStarted: false },
    context: { close: async () => { calls.push('context'); throw new Error('context failed'); } },
    browser: {
      close: async () => { calls.push('browser'); },
      isConnected: () => false,
    },
    server: { close: async () => { calls.push('server'); } },
    browserRecord: { closedCleanly: false },
    serverRecord: { closedCleanly: false },
    shutdownRecord: {
      schemaVersion: 1,
      context: { present: true, attempted: false, succeeded: false, error: null },
      browser: { present: true, attempted: false, succeeded: false, error: null },
      server: { present: true, attempted: false, succeeded: false, error: null },
      complete: false,
    },
  };
  await assert.rejects(
    closeThreeImmediatePhase0IntegrationState(state), AggregateError,
  );
  assert.deepEqual(calls, ['context', 'browser', 'server']);
  assert.equal(state.shutdownRecord.context.attempted, true);
  assert.equal(state.shutdownRecord.context.succeeded, false);
  assert.match(state.shutdownRecord.context.error.message, /context failed/u);
  assert.equal(state.shutdownRecord.browser.succeeded, true);
  assert.equal(state.shutdownRecord.server.succeeded, true);
  assert.equal(state.shutdownRecord.complete, true);
});

test('served-response closure is bijective and content-addressed', () => {
  const origin = 'http://127.0.0.1:49152';
  const moduleResponse = {
    url: `${origin}/src/example.js`, status: 200, resourceType: 'script',
    fromServiceWorker: false, contentType: 'text/javascript',
    byteLength: 17, sha256: '1'.repeat(64),
  };
  const documentResponse = {
    url: `${origin}/phase0-immediate-aif.html`, status: 200, resourceType: 'document',
    fromServiceWorker: false, contentType: 'text/html',
    byteLength: 23, sha256: '2'.repeat(64),
  };
  const source = {
    serverOrigin: origin,
    overlayStart: { root: path.join(repoRoot, '.generated', 'three-immediate-overlay') },
    loadedModules: [{
      servedPath: 'repo:/src/example.js',
      servedResponses: [{
        url: moduleResponse.url,
        byteLength: moduleResponse.byteLength,
        sha256: moduleResponse.sha256,
      }],
    }],
    loadedResources: [moduleResponse, documentResponse].sort(
      (left, right) => left.url.localeCompare(right.url, 'en'),
    ),
  };
  source.loadedResourceUrls = source.loadedResources.map((record) => record.url);
  assert.deepEqual(validateThreeImmediatePhase0LoadedSourceClosure(source), {
    valid: true, reasons: [],
  });
  const digestTamper = structuredClone(source);
  digestTamper.loadedModules[0].servedResponses[0].sha256 = '3'.repeat(64);
  assert.equal(validateThreeImmediatePhase0LoadedSourceClosure(digestTamper).valid, false);
  const duplicate = structuredClone(source);
  duplicate.loadedResources.push(structuredClone(duplicate.loadedResources[0]));
  duplicate.loadedResourceUrls.push(duplicate.loadedResources[0].url);
  assert.equal(validateThreeImmediatePhase0LoadedSourceClosure(duplicate).valid, false);
});

test('dependency closure is reconstructed instead of trusting a page-authored pass flag', () => {
  const paths = [
    'overlay:/build/three.module.js',
    'overlay:/build/three.core.js',
    'overlay:/build/three.tsl.js',
    'overlay:/build/three.webgpu.js',
  ];
  const aliases = Object.fromEntries([
    'three', 'three/core-runtime', 'three/tsl', 'three/webgpu',
  ].map((name, index) => [name, { servedPath: paths[index], sha256: `${index + 1}`.repeat(64) }]));
  const modules = paths.map((servedPath, index) => ({
    servedPath, inputSha256: `${index + 1}`.repeat(64), transformCount: 1,
  }));
  assert.deepEqual(reconstructThreeImmediatePhase0DependencyClosure(modules, aliases), {
    exact: true,
    loadedModuleCount: 4,
    duplicateModuleCount: 0,
    unpatchedThreeModuleCount: 0,
    externalModuleCount: 0,
    aliasModulesPresent: true,
  });
  modules[0].transformCount = 2;
  assert.equal(
    reconstructThreeImmediatePhase0DependencyClosure(modules, aliases).exact,
    false,
  );
});

test('nonce commitments use exact ordered length framing and injection is single-site', () => {
  const fields = ['first', 'second'];
  const values = ['a', 'bc'];
  const framed = Buffer.concat([
    Buffer.from([0, 0, 0, 1]), Buffer.from('a'),
    Buffer.from([0, 0, 0, 2]), Buffer.from('bc'),
  ]);
  const commitment = commitThreeImmediatePhase0ObservationFields(fields, values);
  assert.equal(commitment.sha256, sha256(framed));
  assert.deepEqual(commitment.fieldCommitments, [
    { name: 'first', byteLength: 1, sha256: sha256(Buffer.from('a')) },
    { name: 'second', byteLength: 2, sha256: sha256(Buffer.from('bc')) },
  ]);
  const marker = '/* THREE_IMMEDIATE_PHASE0_TARGET_CONFIGURATION_V1 */ null';
  const injected = injectThreeImmediatePhase0Target(marker, { key: 'r185' });
  assert.match(injected, /Object\.freeze\(\{"key":"r185"\}\)/u);
  assert.throws(() => injectThreeImmediatePhase0Target(`${marker}\n${marker}`, {}),
    /Expected one/);
});

test('persisted render-state zero sign follows explicit JSON canonicalization', () => {
  const source = { worldDirection: [-0, -0, -1] };
  const reread = JSON.parse(JSON.stringify(source));
  assert.equal(Object.is(source.worldDirection[0], reread.worldDirection[0]), false);
  assert.equal(equalThreeImmediatePhase0PersistedNumberArrays(
    source.worldDirection, reread.worldDirection,
  ), true);
  assert.equal(equalThreeImmediatePhase0PersistedNumberArrays(
    [0, 0, -1], [0, Number.MIN_VALUE, -1],
  ), false);
});

test('runner-owned WGSL normalization independently accepts only the approved A/I/F contrast', () => {
  const lanes = phase0ShaderLanes();
  const runner = normalizeThreeImmediatePhase0RenderShaders(lanes);
  const page = normalizeImmediateAifRenderShaderLanes(lanes);
  assert.equal(runner.pass, true, runner.reasons.join('\n'));
  assert.deepEqual(runner, page);
  assert.equal(new Set(Object.values(runner.lanes).map(
    (lane) => lane.normalizedVertexShader,
  )).size, 1);

  const extraImmediate = structuredClone(lanes);
  extraImmediate.I.vertexShader = extraImmediate.I.vertexShader.replace(
    'var<immediate> threeImmediateDrawBase : u32;\n',
    'var<immediate> threeImmediateDrawBase : u32;\nvar<immediate> hidden : u32;\n',
  );
  assert.equal(normalizeThreeImmediatePhase0RenderShaders(extraImmediate).pass, false);

  const nonGenerated = structuredClone(lanes);
  nonGenerated.F.vertexShader = nonGenerated.F.vertexShader
    .replaceAll('NodeBuffer_300', 'HostileMatrix')
    .replaceAll('NodeBuffer_301', 'HostileVisible');
  assert.equal(normalizeImmediateAifRenderShaderLanes(nonGenerated).pass, true);
  assert.equal(normalizeThreeImmediatePhase0RenderShaders(nonGenerated).pass, false);
});

test('runner-owned diagnostic WGSL normalization accepts only the named address transports', () => {
  for (const purpose of ['address', 'object-id']) {
    const lanes = diagnosticShaderLanes(purpose);
    const audit = normalizeThreeImmediatePhase0DiagnosticShaders(lanes, { purpose });
    assert.equal(audit.pass, true, `${audit.reasons.join('\n')}\n${
      Object.entries(audit.lanes).map(([lane, record]) => (
        `${lane}: ${JSON.stringify(record.normalizedVertexShader)}`
      )).join('\n')
    }`);
    assert.equal(new Set(Object.values(audit.lanes).map(
      (lane) => lane.normalizedVertexSha256,
    )).size, 1);

    const wrongAddress = structuredClone(lanes);
    wrongAddress.A.vertexShader = wrongAddress.A.vertexShader.replace(
      '( bucketBase + instanceIndex )', 'instanceIndex',
    );
    assert.equal(normalizeThreeImmediatePhase0DiagnosticShaders(
      wrongAddress, { purpose },
    ).pass, false);

    const extraImmediate = structuredClone(lanes);
    extraImmediate.I.vertexShader = extraImmediate.I.vertexShader.replace(
      'var<immediate> threeImmediateDrawBase : u32;\n',
      'var<immediate> threeImmediateDrawBase : u32;\nvar<immediate> hidden : u32;\n',
    );
    assert.equal(normalizeThreeImmediatePhase0DiagnosticShaders(
      extraImmediate, { purpose },
    ).pass, false);

    const fragmentMutation = structuredClone(lanes);
    fragmentMutation.F.fragmentShader += '// hidden fragment divergence\n';
    assert.equal(normalizeThreeImmediatePhase0DiagnosticShaders(
      fragmentMutation, { purpose },
    ).pass, false);
  }
});

test('runner-owned compute normalizer matches the page record but rejects extra raw divergence', () => {
  const source = computeShaderFixture();
  const bindings = computeBindingFixture();
  assert.deepEqual(
    normalizeThreeImmediatePhase0ComputeShader(source, bindings),
    normalizeLiveIndirectCommandComputeShader(source, bindings),
  );
  assert.throws(() => normalizeThreeImmediatePhase0ComputeShader(
    `${source}\n// NodeBuffer_900 must not be normalized from a comment`, bindings,
  ), /token inventory/u);
  assert.throws(() => normalizeThreeImmediatePhase0ComputeShader(
    source.replace('firstInstance : u32', 'firstInstance : i32'), bindings,
  ), /command struct/u);
});

test('render pipeline descriptor gate distinguishes production, diagnostics, and exact depth', () => {
  for (const [purpose, lane] of [
    ['production', 'A'], ['production', 'I'], ['object-id', 'A'], ['object-id', 'F'],
    ['address', 'A'],
  ]) {
    const descriptor = pipelineDescriptorFixture({ purpose, lane });
    assert.deepEqual(validateThreeImmediatePhase0RenderPipelineDescriptor(descriptor, {
      purpose, lane,
      vertexInputs: purpose === 'production'
        ? phase0VertexInputs(lane !== 'A') : diagnosticVertexInputs(lane, purpose),
    }), { valid: true, reasons: [] });
  }
  const mutations = [
    ['production', 'A', (value) => { value.primitive.cullMode = 'none'; }],
    ['address', 'I', (value) => { value.primitive.cullMode = 'back'; }],
    ['object-id', 'F', (value) => { value.depthStencil.format = 'depth24plus'; }],
    ['address', 'A', (value) => { value.depthStencil = pipelineDescriptorFixture().depthStencil; }],
    ['production', 'A', (value) => { value.fragment.targets[0].format = 'rgba8uint'; }],
    ['production', 'A', (value) => { value.vertex.buffers[0].attributes[0].offset = 4; }],
    ['production', 'A', (value) => { value.multisample.mask = 0xffff_fffe; }],
  ];
  for (const [purpose, lane, mutate] of mutations) {
    const descriptor = pipelineDescriptorFixture({ purpose, lane });
    mutate(descriptor);
    assert.equal(validateThreeImmediatePhase0RenderPipelineDescriptor(descriptor, {
      purpose, lane,
      vertexInputs: purpose === 'production'
        ? phase0VertexInputs(lane !== 'A') : diagnosticVertexInputs(lane, purpose),
    }).valid, false, `${purpose}/${lane} mutation passed`);
  }
});

test('clear and Three render passes use role-specific descriptor defaults', () => {
  assert.equal(validateThreeImmediatePhase0RenderPassDescriptorDefaults(
    { label: null, maxDrawCount: null }, { role: 'clear' },
  ).valid, true);
  assert.equal(validateThreeImmediatePhase0RenderPassDescriptorDefaults(
    { label: '', maxDrawCount: 50_000_000 }, { role: 'render' },
  ).valid, true);
  for (const trace of [
    { label: null, maxDrawCount: null },
    { label: '', maxDrawCount: null },
    { label: '', maxDrawCount: 49_999_999 },
  ]) {
    assert.equal(validateThreeImmediatePhase0RenderPassDescriptorDefaults(
      trace, { role: 'render' },
    ).valid, false);
  }
  assert.equal(validateThreeImmediatePhase0RenderPassDescriptorDefaults(
    { label: '', maxDrawCount: 50_000_000 }, { role: 'clear' },
  ).valid, false);
});

test('non-stencil depth attachments retain Three zero stencil-clear defaults', () => {
  const attachment = {
    depthLoadOp: 'clear',
    depthStoreOp: 'store',
    depthClearValue: 0,
    depthReadOnly: false,
    stencilLoadOp: null,
    stencilStoreOp: null,
    stencilClearValue: 0,
    stencilReadOnly: false,
  };
  assert.equal(validateThreeImmediatePhase0DepthStencilAttachmentDefaults(
    attachment, { loadOp: 'clear' },
  ).valid, true);
  assert.equal(validateThreeImmediatePhase0DepthStencilAttachmentDefaults(
    { ...attachment, depthLoadOp: 'load' }, { loadOp: 'load' },
  ).valid, true);

  for (const mutate of [
    (value) => { value.stencilClearValue = null; },
    (value) => { value.stencilClearValue = 1; },
    (value) => { value.stencilLoadOp = 'clear'; },
    (value) => { value.stencilStoreOp = 'store'; },
    (value) => { value.stencilReadOnly = true; },
  ]) {
    const changed = structuredClone(attachment);
    mutate(changed);
    assert.equal(validateThreeImmediatePhase0DepthStencilAttachmentDefaults(
      changed, { loadOp: 'clear' },
    ).valid, false);
  }
});

test('pipeline layouts carry the exact production, diagnostic, and compute immediate size', () => {
  for (const [lane, immediateSize] of [['A', 0], ['I', 4], ['F', 0]]) {
    const layout = {
      layoutId: `layout-${lane}`,
      hasOwnImmediateSize: true,
      immediateSize,
    };
    assert.equal(validateThreeImmediatePhase0PipelineLayoutImmediateState(
      layout, { lane },
    ).valid, true);
    layout.immediateSize = immediateSize === 0 ? 4 : 0;
    assert.equal(validateThreeImmediatePhase0PipelineLayoutImmediateState(
      layout, { lane },
    ).valid, false);
  }
  const compute = {
    layoutId: 'layout-compute', hasOwnImmediateSize: true, immediateSize: 0,
  };
  assert.equal(validateThreeImmediatePhase0PipelineLayoutImmediateState(
    compute, { compute: true },
  ).valid, true);
  compute.immediateSize = 4;
  assert.equal(validateThreeImmediatePhase0PipelineLayoutImmediateState(
    compute, { compute: true },
  ).valid, false);
  compute.immediateSize = 0;
  compute.hasOwnImmediateSize = false;
  assert.equal(validateThreeImmediatePhase0PipelineLayoutImmediateState(
    compute, { compute: true },
  ).valid, false);
});

test('I bundle grammar starts with and preserves all 32 immediate/draw pairs', () => {
  const setup = [
    { method: 'setPipeline' },
    { method: 'setBindGroup', index: 0, bindGroupId: 'bind-group', dynamicOffsets: null },
    { method: 'setIndexBuffer', bufferId: 'index', indexFormat: 'uint32', offset: 0,
      size: null },
    { method: 'setVertexBuffer', slot: 0, bufferId: 'position', offset: 0, size: null },
  ];
  const pairs = Array.from({ length: 32 }, (_, index) => ([
    { method: 'setImmediates', draw: index },
    { method: 'drawIndexedIndirect', draw: index },
  ])).flat();
  const events = [...setup, ...pairs, { method: 'finish' }];
  assert.equal(validateThreeImmediatePhase0BundleEventGrammar(
    events, { lane: 'I' },
  ).valid, true);

  const missingFirstImmediate = events.filter((_, index) => index !== setup.length);
  assert.equal(validateThreeImmediatePhase0BundleEventGrammar(
    missingFirstImmediate, { lane: 'I' },
  ).valid, false);

  const movedFirstImmediate = [...events];
  const [firstImmediate] = movedFirstImmediate.splice(setup.length, 1);
  movedFirstImmediate.splice(setup.length + 1, 0, firstImmediate);
  assert.equal(validateThreeImmediatePhase0BundleEventGrammar(
    movedFirstImmediate, { lane: 'I' },
  ).valid, false);
});

function uniformQueueWriteRecord({
  sequence,
  bufferOffset = 0,
  selectedSourceByteOffset = bufferOffset,
  selectedSourceByteLength = 16,
  capturePhase = 'phase0/prime/A',
} = {}) {
  return {
    category: 'production-render',
    sequence,
    capturePhase,
    bufferId: 'uniform-buffer',
    bufferOffset,
    sourceByteLength: 16,
    selectedSourceByteOffset,
    selectedSourceByteLength,
    destinationOwner: {
      destinationBindingClass: 'uniform-bind-group-buffer',
      uniformBindingProofs: [{
        exact: true,
        bindGroupId: 'uniform-bind-group',
        bufferOffset: 0,
        bufferSize: null,
        hasDynamicOffset: false,
      }],
    },
  };
}

function directUniformWriteFixture(records, { setSequence = 10 } = {}) {
  const capturePhase = records[0].capturePhase;
  return {
    records,
    bufferCreations: [{
      resourceId: 'uniform-buffer', size: 16, usage: 72, mappedAtCreation: false,
    }],
    lifecycle: {
      computePassTraces: [],
      renderBundleTraces: [],
      renderPassTraces: [{
        encoderId: 'render-pass',
        commandEncoderId: 'command-encoder',
        capturePhase,
        events: [{
          method: 'setBindGroup', sequence: setSequence, index: 0,
          bindGroupId: 'uniform-bind-group', dynamicOffsets: null,
          dynamicOffsetStart: null, dynamicOffsetLength: null,
          selectedDynamicOffsets: null,
        }, { method: 'drawIndexed', sequence: 30 }],
      }],
      commandEncoderTraces: [{
        commandEncoderId: 'command-encoder', commandBufferId: 'command-buffer',
        finishSequence: 40,
      }],
      queueSubmissions: [{
        commandBufferIds: ['command-buffer'], capturePhase, sequence: 50,
      }],
    },
  };
}

function assertUniformWriteFixturePasses(fixture) {
  const reconstruction = reconstructThreeImmediatePhase0UniformQueueWriteEvidence(
    fixture.records,
    { bufferCreations: fixture.bufferCreations, lifecycle: fixture.lifecycle },
  );
  assert.equal(reconstruction.valid, true);
  assert.deepEqual(validateThreeImmediatePhase0UniformQueueWriteEvidence(
    reconstruction.expected,
    { records: fixture.records, bufferCreations: fixture.bufferCreations,
      lifecycle: fixture.lifecycle },
  ), { valid: true, reasons: [], expected: reconstruction.expected });
  return reconstruction.expected;
}

test('uniform queue writes accept exact full, partial, and overlapping byte ranges', () => {
  const full = directUniformWriteFixture([
    uniformQueueWriteRecord({ sequence: 20 }),
  ]);
  const fullExpected = assertUniformWriteFixturePasses(full);
  assert.equal(fullExpected.uniformWriteGroups[0].writeCount, 1);
  assert.equal(fullExpected.uniformWriteGroups[0].observedDestinationOverlapCount, 0);

  const partial = directUniformWriteFixture([
    uniformQueueWriteRecord({ sequence: 20, selectedSourceByteLength: 8 }),
    uniformQueueWriteRecord({ sequence: 21, bufferOffset: 8,
      selectedSourceByteOffset: 8, selectedSourceByteLength: 8 }),
  ]);
  const partialExpected = assertUniformWriteFixturePasses(partial);
  assert.equal(partialExpected.uniformWriteGroups[0].writeCount, 2);
  assert.equal(partialExpected.uniformWriteGroups[0].observedDestinationOverlapCount, 0);
  assert.equal(partialExpected.uniformWriteGroups[0].overlapPolicy,
    'observed-not-prescribed');

  const overlapping = directUniformWriteFixture([
    uniformQueueWriteRecord({ sequence: 20, selectedSourceByteLength: 12 }),
    uniformQueueWriteRecord({ sequence: 21, bufferOffset: 8,
      selectedSourceByteOffset: 8, selectedSourceByteLength: 8 }),
  ]);
  const overlapExpected = assertUniformWriteFixturePasses(overlapping);
  assert.equal(overlapExpected.uniformWriteGroups[0].observedDestinationOverlapCount, 1);

  // A direct bind may precede the write; it is the active binding at the draw
  // that matters. A post-write bind is accepted as well.
  assertUniformWriteFixturePasses(directUniformWriteFixture([
    uniformQueueWriteRecord({ sequence: 20 }),
  ], { setSequence: 25 }));
});

test('uniform queue writes reject range, order, active-slot, and retained-proof forgery', () => {
  const valid = directUniformWriteFixture([
    uniformQueueWriteRecord({ sequence: 20, selectedSourceByteLength: 8 }),
    uniformQueueWriteRecord({ sequence: 21, bufferOffset: 8,
      selectedSourceByteOffset: 8, selectedSourceByteLength: 8 }),
  ]);
  const validExpected = assertUniformWriteFixturePasses(valid);

  for (const mutate of [
    (fixture) => { fixture.records[1].bufferOffset = 6; },
    (fixture) => { fixture.records[1].selectedSourceByteOffset = 4; },
    (fixture) => { fixture.records[1].selectedSourceByteLength = 12; },
    (fixture) => { fixture.records.reverse(); },
    (fixture) => {
      fixture.lifecycle.renderPassTraces[0].events.splice(1, 0, {
        method: 'setBindGroup', sequence: 25, index: 0,
        bindGroupId: 'intervening-bind-group', dynamicOffsets: null,
        dynamicOffsetStart: null, dynamicOffsetLength: null,
        selectedDynamicOffsets: null,
      });
    },
    (fixture) => {
      fixture.lifecycle.renderPassTraces[0].events.splice(1, 0, {
        method: 'setBindGroup', sequence: 25, index: 1,
        bindGroupId: 'uniform-bind-group', dynamicOffsets: null,
        dynamicOffsetStart: null, dynamicOffsetLength: null,
        selectedDynamicOffsets: null,
      });
    },
  ]) {
    const changed = structuredClone(valid);
    mutate(changed);
    assert.equal(reconstructThreeImmediatePhase0UniformQueueWriteEvidence(
      changed.records,
      { bufferCreations: changed.bufferCreations, lifecycle: changed.lifecycle },
    ).valid, false);
  }

  const forged = structuredClone(validExpected);
  forged.uniformWriteGrammar[0].executionProofs[0].activeSetBindGroupSequence = 999;
  assert.equal(validateThreeImmediatePhase0UniformQueueWriteEvidence(
    forged,
    { records: valid.records, bufferCreations: valid.bufferCreations,
      lifecycle: valid.lifecycle },
  ).valid, false);
  const forgedIndex = structuredClone(validExpected);
  forgedIndex.uniformWriteGrammar[0].executionProofs[0].bindGroupIndex = 1;
  assert.equal(validateThreeImmediatePhase0UniformQueueWriteEvidence(
    forgedIndex,
    { records: valid.records, bufferCreations: valid.bufferCreations,
      lifecycle: valid.lifecycle },
  ).valid, false);
});

test('cached-bundle uniform proof binds historical setup to a later exact execution', () => {
  const record = uniformQueueWriteRecord({ sequence: 40 });
  const fixture = {
    records: [record],
    bufferCreations: [{
      resourceId: 'uniform-buffer', size: 16, usage: 72, mappedAtCreation: false,
    }],
    lifecycle: {
      computePassTraces: [],
      renderBundleTraces: [{
        bundleId: 'render-bundle', finishSequence: 30,
        events: [{
          method: 'setBindGroup', sequence: 10, index: 0,
          bindGroupId: 'uniform-bind-group', dynamicOffsets: null,
          dynamicOffsetStart: null, dynamicOffsetLength: null,
          selectedDynamicOffsets: null,
        }, { method: 'drawIndexedIndirect', sequence: 20 }],
      }],
      renderPassTraces: [{
        encoderId: 'render-pass', commandEncoderId: 'command-encoder',
        capturePhase: record.capturePhase,
        events: [{ method: 'executeBundles', sequence: 50,
          bundleIds: ['render-bundle'] }],
      }],
      commandEncoderTraces: [{
        commandEncoderId: 'command-encoder', commandBufferId: 'command-buffer',
        finishSequence: 60,
      }],
      queueSubmissions: [{
        commandBufferIds: ['command-buffer'], capturePhase: record.capturePhase,
        sequence: 70,
      }],
    },
  };
  const expected = assertUniformWriteFixturePasses(fixture);
  assert.equal(expected.uniformWriteGrammar[0].executionProofs[0].useKind,
    'executed-render-bundle');
  assert.ok(expected.uniformWriteGrammar[0].executionProofs[0].bundleFinishSequence
    < record.sequence);

  const earlyExecution = structuredClone(fixture);
  earlyExecution.lifecycle.renderPassTraces[0].events[0].sequence = 35;
  assert.equal(reconstructThreeImmediatePhase0UniformQueueWriteEvidence(
    earlyExecution.records,
    { bufferCreations: earlyExecution.bufferCreations,
      lifecycle: earlyExecution.lifecycle },
  ).valid, false);
});

test('compute and cached-bundle proofs reject same-slot rebinds before consuming use', () => {
  const record = uniformQueueWriteRecord({
    sequence: 20, capturePhase: 'phase0/live/v99/A',
  });
  record.category = 'live-compute';
  const computeFixture = {
    records: [record],
    bufferCreations: [{
      resourceId: 'uniform-buffer', size: 16, usage: 72, mappedAtCreation: false,
    }],
    lifecycle: {
      renderPassTraces: [],
      renderBundleTraces: [],
      computePassTraces: [{
        encoderId: 'compute-pass', commandEncoderId: 'compute-command-encoder',
        capturePhase: record.capturePhase,
        events: [{
          method: 'setBindGroup', sequence: 10, index: 0,
          bindGroupId: 'uniform-bind-group', dynamicOffsets: null,
          dynamicOffsetStart: null, dynamicOffsetLength: null,
          selectedDynamicOffsets: null,
        }, { method: 'dispatchWorkgroups', sequence: 30 }],
      }],
      commandEncoderTraces: [{
        commandEncoderId: 'compute-command-encoder', commandBufferId: 'compute-command-buffer',
        finishSequence: 40,
      }],
      queueSubmissions: [{
        commandBufferIds: ['compute-command-buffer'], capturePhase: record.capturePhase,
        sequence: 50,
      }],
    },
  };
  assertUniformWriteFixturePasses(computeFixture);
  const reboundCompute = structuredClone(computeFixture);
  reboundCompute.lifecycle.computePassTraces[0].events.splice(1, 0, {
    method: 'setBindGroup', sequence: 25, index: 0,
    bindGroupId: 'intervening-bind-group', dynamicOffsets: null,
    dynamicOffsetStart: null, dynamicOffsetLength: null,
    selectedDynamicOffsets: null,
  });
  assert.equal(reconstructThreeImmediatePhase0UniformQueueWriteEvidence(
    reboundCompute.records,
    { bufferCreations: reboundCompute.bufferCreations,
      lifecycle: reboundCompute.lifecycle },
  ).valid, false);

  const bundleRecord = uniformQueueWriteRecord({ sequence: 40 });
  const bundleFixture = {
    records: [bundleRecord],
    bufferCreations: computeFixture.bufferCreations,
    lifecycle: {
      computePassTraces: [],
      renderBundleTraces: [{
        bundleId: 'render-bundle', finishSequence: 30,
        events: [{
          method: 'setBindGroup', sequence: 10, index: 0,
          bindGroupId: 'uniform-bind-group', dynamicOffsets: null,
          dynamicOffsetStart: null, dynamicOffsetLength: null,
          selectedDynamicOffsets: null,
        }, {
          method: 'setBindGroup', sequence: 15, index: 0,
          bindGroupId: 'intervening-bind-group', dynamicOffsets: null,
          dynamicOffsetStart: null, dynamicOffsetLength: null,
          selectedDynamicOffsets: null,
        }, { method: 'drawIndexedIndirect', sequence: 20 }],
      }],
      renderPassTraces: [{
        encoderId: 'render-pass', commandEncoderId: 'command-encoder',
        capturePhase: bundleRecord.capturePhase,
        events: [{ method: 'executeBundles', sequence: 50,
          bundleIds: ['render-bundle'] }],
      }],
      commandEncoderTraces: [{
        commandEncoderId: 'command-encoder', commandBufferId: 'command-buffer',
        finishSequence: 60,
      }],
      queueSubmissions: [{
        commandBufferIds: ['command-buffer'], capturePhase: bundleRecord.capturePhase,
        sequence: 70,
      }],
    },
  };
  assert.equal(reconstructThreeImmediatePhase0UniformQueueWriteEvidence(
    bundleFixture.records,
    { bufferCreations: bundleFixture.bufferCreations,
      lifecycle: bundleFixture.lifecycle },
  ).valid, false);
});

function completeQueueWriteLedgerFixture() {
  const sharedSemantics = [
    'matrix', 'bounds', 'objectBucket', 'bucketBase', 'bucketCapacity', 'cullOrder',
    'visibleIds', 'overflow',
  ];
  let nextAttributeId = 1;
  const resource = (semantic, prefix) => ({
    semantic,
    attributeId: nextAttributeId++,
    gpuBufferId: `${prefix}-${semantic}`,
  });
  const shared = sharedSemantics.map((semantic) => resource(semantic, 'shared'));
  const merged = ['bucketBase', 'normal', 'position', 'uv', 'index'].map(
    (semantic) => resource(semantic, 'geometry'),
  );
  const commands = Object.fromEntries(['A', 'I', 'F'].map((lane) => [lane, {
    attributeId: nextAttributeId++, gpuBufferId: `command-${lane}`,
  }]));
  const diagnosticPositions = ['commonPosition', 'featurePosition'].map(
    (semantic) => resource(semantic, 'diagnostic'),
  );
  const knownResources = [...shared, ...merged, ...Object.values(commands),
    ...diagnosticPositions];
  const bufferCreations = knownResources.map((entry) => ({
    method: 'createBuffer',
    resourceId: entry.gpuBufferId,
    resourceClass: 'persistent-or-upload-buffer',
    sequence: 1,
    capturePhase: 'phase0/setup',
    size: 16,
    usage: 140,
    mappedAtCreation: false,
  }));
  bufferCreations.push({
    method: 'createBuffer',
    resourceId: 'uniform-buffer',
    resourceClass: 'persistent-or-upload-buffer',
    sequence: 1,
    capturePhase: 'phase0/setup',
    size: 16,
    usage: 72,
    mappedAtCreation: false,
  });
  let sequence = 100;
  const rawRecords = [];
  const addWrite = (capturePhase, bufferId, {
    bufferOffset = 0, dataOffset = 0, size = null,
  } = {}) => {
    const sourceElementCount = 4;
    const selectedSourceElementLength = size ?? sourceElementCount - dataOffset;
    const record = {
      sequence: sequence++,
      capturePhase,
      bufferId,
      bufferOffset,
      sourceId: `source-${sequence}`,
      sourceBackingBufferId: `source-backing-${sequence}`,
      sourceType: 'Uint32Array',
      sourceByteLength: 16,
      sourceByteOffset: 0,
      sourceElementByteSize: 4,
      sourceElementCount,
      dataOffset,
      size,
      selectedSourceElementOffset: dataOffset,
      selectedSourceElementLength,
      selectedSourceByteOffset: dataOffset * 4,
      selectedSourceByteLength: selectedSourceElementLength * 4,
      selectedSourceSha256: 'a'.repeat(64),
    };
    rawRecords.push(record);
    return record;
  };
  const idOf = (semantic) => shared.find(
    (entry) => entry.semantic === semantic,
  ).gpuBufferId;

  const scenarios = ['v99', 'v20'].map((scenarioId) => ({
    scenarioId,
    matrixGpuRealization: {
      capturePhase: `phase0/scenario-load/${scenarioId}/matrix-realization`,
      queueWriteBuffers: [],
    },
    gpuResourceCommitments: {
      capturePhase: `phase0/resources/${scenarioId}/shared-commitment`,
      records: shared,
      queueWriteBuffers: [],
    },
  }));
  scenarios[1].matrixGpuRealization.queueWriteBuffers.push(addWrite(
    scenarios[1].matrixGpuRealization.capturePhase, idOf('matrix'),
  ));
  for (const scenario of scenarios) {
    scenario.gpuResourceCommitments.queueWriteBuffers.push(addWrite(
      scenario.gpuResourceCommitments.capturePhase, idOf('visibleIds'),
    ));
  }

  const scheduleDefinitions = [
    ['v99', 0, 'canonical'], ['v99', 1, 'S1'],
    ['v99', 2, 'S2'], ['v99', 3, 'canonical'],
    ['v20', 4, 'canonical'], ['v20', 5, 'S1'],
    ['v20', 6, 'S2'], ['v20', 7, 'canonical'],
  ];
  const attributeWriteCounts = [4, 1, 1, 1, 2, 1, 1, 2];
  const diagnosticWriteCounts = [0, 1, 1, 1, 1, 1, 1, 1];
  const attributeDestinations = [
    merged.find((entry) => entry.semantic === 'bucketBase').gpuBufferId,
    commands.A.gpuBufferId,
    commands.I.gpuBufferId,
    commands.F.gpuBufferId,
  ];
  const installations = scheduleDefinitions.map(([
    scenarioId, ordinal, scheduleId,
  ], index) => {
    const basePhase = `phase0/schedule-install/${scenarioId}/${ordinal}-${scheduleId}`;
    const queueWriteBuffers = Array.from({ length: attributeWriteCounts[index] }, (_, write) => (
      addWrite(`${basePhase}/attribute-realization`,
        attributeDestinations[write % attributeDestinations.length])
    ));
    const diagnosticQueueWrites = Array.from(
      { length: diagnosticWriteCounts[index] },
      () => addWrite(`${basePhase}/diagnostic-position-realization`,
        diagnosticPositions[1].gpuBufferId),
    );
    return {
      installationId: `installation-${index}`,
      basePhase,
      commands,
      queueWriteBuffers,
      diagnosticPositionRealization: {
        capturePhase: `${basePhase}/diagnostic-position-realization`,
        queueWriteBuffers: diagnosticQueueWrites,
      },
    };
  });

  const expectedLiveWrites = [
    ['v99', 'I', []],
    ['v99', 'F', ['visibleIds', 'overflow']],
    ['v99', 'A', ['visibleIds', 'overflow']],
    ['v20', 'A', [
      'bounds', 'objectBucket', 'bucketBase', 'bucketCapacity', 'cullOrder',
      'visibleIds', 'overflow', 'command.A',
    ]],
    ['v20', 'F', ['visibleIds', 'overflow', 'command.F']],
    ['v20', 'I', ['visibleIds', 'overflow', 'command.I']],
  ];
  const storageBindGroupId = 'live-storage-bind-group';
  const computePassTraces = [];
  for (const [scenarioId, lane, semantics] of expectedLiveWrites) {
    const capturePhase = `phase0/live/${scenarioId}/${lane}`;
    const phaseWrites = semantics.map((semantic) => addWrite(
      capturePhase,
      semantic.startsWith('command.')
        ? commands[semantic.at(-1)].gpuBufferId : idOf(semantic),
    ));
    const lastWrite = phaseWrites.at(-1)?.sequence ?? sequence;
    computePassTraces.push({
      capturePhase,
      events: [{
        method: 'setBindGroup', sequence: lastWrite + 1,
        bindGroupId: storageBindGroupId,
        dynamicOffsets: null, dynamicOffsetStart: null,
        dynamicOffsetLength: null, selectedDynamicOffsets: null,
      }],
    });
  }

  const uniformPhase = 'phase0/prime/A';
  addWrite(uniformPhase, 'uniform-buffer', { size: 3 });
  addWrite(uniformPhase, 'uniform-buffer', { bufferOffset: 8, dataOffset: 2, size: 2 });
  const uniformBindGroup = {
    method: 'createBindGroup', resourceId: 'uniform-bind-group', sequence: 10,
    capturePhase: 'phase0/setup', layoutId: 'uniform-bind-group-layout',
    entries: [{ binding: 0, bufferId: 'uniform-buffer', bufferOffset: 0, bufferSize: null }],
  };
  const liveBufferIds = [...new Set(expectedLiveWrites.flatMap(([, , semantics]) => (
    semantics.map((semantic) => semantic.startsWith('command.')
      ? commands[semantic.at(-1)].gpuBufferId : idOf(semantic))
  )))];
  const storageBindGroup = {
    method: 'createBindGroup', resourceId: storageBindGroupId, sequence: 10,
    capturePhase: 'phase0/setup', layoutId: 'live-storage-bind-group-layout',
    entries: liveBufferIds.map((bufferId, binding) => ({
      binding, bufferId, bufferOffset: 0, bufferSize: 16,
    })),
  };
  const lifecycle = {
    queueWriteBufferCalls: rawRecords,
    computePassTraces,
    renderBundleTraces: [],
    renderPassTraces: [{
      encoderId: 'uniform-render-pass', commandEncoderId: 'uniform-command-encoder',
      capturePhase: uniformPhase,
      events: [{
        method: 'setBindGroup', sequence: 50, index: 0,
        bindGroupId: uniformBindGroup.resourceId, dynamicOffsets: null,
        dynamicOffsetStart: null, dynamicOffsetLength: null,
        selectedDynamicOffsets: null,
      }, { method: 'drawIndexed', sequence: sequence + 10 }],
    }],
    commandEncoderTraces: [{
      commandEncoderId: 'uniform-command-encoder', commandBufferId: 'uniform-command-buffer',
      finishSequence: sequence + 20,
    }],
    queueSubmissions: [{
      commandBufferIds: ['uniform-command-buffer'], capturePhase: uniformPhase,
      sequence: sequence + 30,
    }],
  };
  const resources = {
    gpuCreations: [...bufferCreations, uniformBindGroup, storageBindGroup],
    scheduleInstallations: { records: installations },
    geometryFixtures: { mergedProduction: { gpuRecords: merged } },
    addressDiagnosticPositions: { records: [{ records: diagnosticPositions }] },
  };
  const pipelines = {
    bindGroupLayouts: [{
      layoutId: 'uniform-bind-group-layout', sequence: 9,
      entries: [{
        binding: 0,
        buffer: { type: null, hasDynamicOffset: false, minBindingSize: 0 },
      }],
    }, {
      layoutId: 'live-storage-bind-group-layout', sequence: 9,
      entries: liveBufferIds.map((_, binding) => ({
        binding,
        buffer: { type: 'storage', hasDynamicOffset: false, minBindingSize: 0 },
      })),
    }],
  };
  return { lifecycle, scenarios, resources, pipelines };
}

test('complete queue ledger accepts independently reconstructed overlapping uniform writes', () => {
  const fixture = completeQueueWriteLedgerFixture();
  const initial = validateThreeImmediatePhase0QueueWriteBufferLedger({}, fixture);
  assert.equal(initial.valid, false);
  assert.ok(initial.expected);
  const accepted = validateThreeImmediatePhase0QueueWriteBufferLedger(
    initial.expected, fixture,
  );
  assert.deepEqual(accepted, {
    valid: true, reasons: [], expected: initial.expected,
  });
  assert.equal(initial.expected.uniformWriteGroups[0].writeCount, 2);
  assert.equal(initial.expected.uniformWriteGroups[0].observedDestinationOverlapCount, 1);

  const extra = structuredClone(initial.expected);
  extra.uniformWriteGroups[0].observedDestinationOverlapCount = 0;
  assert.equal(validateThreeImmediatePhase0QueueWriteBufferLedger(extra, fixture).valid, false);
});

test('raw WGSL declarations exhaust pipeline layouts and render-bundle bind groups', () => {
  const vertex = `
struct Frame { value: vec4<f32> };
@group(0) @binding(0) var<uniform> frame: Frame;
struct Values { value: array<u32> };
@binding(3) @group(1) var<storage, read> values: Values;
@vertex fn main() -> @builtin(position) vec4<f32> { return frame.value; }
`;
  const fragment = `
struct Frame { value: vec4<f32> };
@binding(0) @group(0) var<uniform> frame: Frame;
@fragment fn main() -> @location(0) vec4<f32> { return frame.value; }
`;
  const fixture = {
    shaders: { modules: [
      { moduleId: 'vertex-module', code: vertex },
      { moduleId: 'fragment-module', code: fragment },
    ] },
    pipelines: {
      render: [{ pipelineId: 'pipeline', vertexModuleId: 'vertex-module',
        fragmentModuleId: 'fragment-module', layoutId: 'pipeline-layout', sequence: 10 }],
      compute: [],
      layouts: [{ layoutId: 'pipeline-layout', sequence: 3,
        bindGroupLayoutIds: ['frame-layout', 'storage-layout'] }],
      bindGroupLayouts: [{ layoutId: 'frame-layout', sequence: 1, entries: [{
        binding: 0, visibility: 7,
        buffer: { type: null, hasDynamicOffset: false, minBindingSize: 0 },
        sampler: null, texture: null, storageTexture: null,
      }] }, { layoutId: 'storage-layout', sequence: 2, entries: [{
        binding: 3, visibility: 1,
        buffer: { type: 'read-only-storage', hasDynamicOffset: false, minBindingSize: 0 },
        sampler: null, texture: null, storageTexture: null,
      }] }],
    },
    resources: {
      textureViews: [],
      gpuCreations: [
        { method: 'createBuffer', resourceId: 'frame-buffer', sequence: 4, size: 16 },
        { method: 'createBuffer', resourceId: 'storage-buffer', sequence: 5, size: 256 },
        { method: 'createBindGroup', resourceId: 'frame-group', layoutId: 'frame-layout',
          sequence: 11, entries: [{ binding: 0, bufferId: 'frame-buffer',
            bufferOffset: 0, bufferSize: null, resourceId: null }] },
        { method: 'createBindGroup', resourceId: 'storage-group', layoutId: 'storage-layout',
          sequence: 12, entries: [{ binding: 3, bufferId: 'storage-buffer',
            bufferOffset: 0, bufferSize: null, resourceId: null }] },
      ],
    },
    lifecycle: { renderBundleTraces: [{
      bundleId: 'bundle', finishSequence: 30,
      events: [
        { method: 'setPipeline', pipelineId: 'pipeline', sequence: 20 },
        { method: 'setBindGroup', index: 0, bindGroupId: 'frame-group', sequence: 21,
          dynamicOffsets: null, dynamicOffsetStart: null, dynamicOffsetLength: null,
          selectedDynamicOffsets: null },
        { method: 'setBindGroup', index: 1, bindGroupId: 'storage-group', sequence: 22,
          dynamicOffsets: null, dynamicOffsetStart: null, dynamicOffsetLength: null,
          selectedDynamicOffsets: null },
        { method: 'drawIndexedIndirect', sequence: 23 },
        { method: 'finish', sequence: 30 },
      ],
    }] },
  };
  const audit = (value) => validateThreeImmediatePhase0ShaderDeclaredPipelineLayouts(value);
  assert.equal(audit(fixture).valid, true, audit(fixture).reasons.join('; '));
  for (const mutate of [
    (value) => {
      value.pipelines.layouts[0].bindGroupLayoutIds.push('trailing-empty-layout');
      value.pipelines.bindGroupLayouts.push({
        layoutId: 'trailing-empty-layout', sequence: 2, entries: [],
      });
    },
    (value) => { value.pipelines.bindGroupLayouts[1].entries[0].buffer.type = 'storage'; },
    (value) => { value.lifecycle.renderBundleTraces[0].events[1].selectedDynamicOffsets = []; },
    (value) => { value.resources.gpuCreations[3].entries[0].bufferId = 'missing-buffer'; },
  ]) {
    const tampered = structuredClone(fixture);
    mutate(tampered);
    assert.equal(audit(tampered).valid, false);
  }
});

test('diagnostic vertex inputs are purpose-specific and bind the exact merged resources', () => {
  const mergedPosition = { attributeId: 701, gpuBufferId: 'merged-position' };
  const mergedBucketBase = { attributeId: 702, gpuBufferId: 'merged-bucket-base' };
  const makeProducer = (lane, { objectId = false } = {}) => {
    const purpose = objectId ? 'object-id' : 'address';
    const inputs = diagnosticVertexInputs(lane, purpose, objectId ? 701 : 800);
    return {
      vertexInputs: inputs,
      expectedVertexInputs: structuredClone(inputs),
      vertexInputsExact: true,
      pipelineVertexInputs: inputs.map((input) => ({
        shaderLocation: input.shaderLocation, format: input.format, stepMode: input.stepMode,
      })),
      pipelineVertexInputsExact: true,
      rawEvents: inputs.map((input, slot) => ({
        method: 'setVertexBuffer', slot,
        bufferId: input.name === 'bucketBase'
          ? 'merged-bucket-base' : objectId ? 'merged-position' : 'address-common-position',
        offset: 0, size: null,
      })),
    };
  };
  const address = makeProducer('A');
  const objectId = makeProducer('A', { objectId: true });
  const immediate = makeProducer('I');
  const firstInstance = makeProducer('F');
  const objectImmediate = makeProducer('I', { objectId: true });
  const objectFirstInstance = makeProducer('F', { objectId: true });
  assert.deepEqual(address.vertexInputs.map(({ name, shaderLocation }) => ({
    name, shaderLocation,
  })), [
    { name: 'bucketBase', shaderLocation: 0 },
    { name: 'position', shaderLocation: 1 },
  ]);
  assert.deepEqual(objectId.vertexInputs.map(({ name, shaderLocation }) => ({
    name, shaderLocation,
  })), [
    { name: 'position', shaderLocation: 0 },
    { name: 'bucketBase', shaderLocation: 1 },
  ]);
  for (const producer of [
    immediate, firstInstance, objectImmediate, objectFirstInstance,
  ]) {
    assert.deepEqual(producer.vertexInputs.map(({ name, shaderLocation }) => ({
      name, shaderLocation,
    })), [{ name: 'position', shaderLocation: 0 }]);
  }
  assert.equal(validateThreeImmediatePhase0DiagnosticVertexInputs(address, {
    purpose: 'address', lane: 'A', mergedBucketBase,
  }).valid, true);
  assert.equal(validateThreeImmediatePhase0DiagnosticVertexInputs(objectId, {
    purpose: 'object-id', lane: 'A', mergedPosition, mergedBucketBase,
  }).valid, true);
  assert.equal(validateThreeImmediatePhase0DiagnosticVertexInputs(immediate, {
    purpose: 'address', lane: 'I',
  }).valid, true);
  assert.equal(validateThreeImmediatePhase0DiagnosticVertexInputs(firstInstance, {
    purpose: 'address', lane: 'F',
  }).valid, true);
  assert.equal(validateThreeImmediatePhase0DiagnosticVertexInputs(objectImmediate, {
    purpose: 'object-id', lane: 'I', mergedPosition,
  }).valid, true);
  assert.equal(validateThreeImmediatePhase0DiagnosticVertexInputs(objectFirstInstance, {
    purpose: 'object-id', lane: 'F', mergedPosition,
  }).valid, true);
  assert.equal(validateThreeImmediatePhase0DiagnosticVertexInputs(address, {
    lane: 'A', mergedBucketBase,
  }).valid, false);
  assert.equal(validateThreeImmediatePhase0DiagnosticVertexInputs(address, {
    purpose: 'object-id', lane: 'A', mergedBucketBase,
  }).valid, false);
  assert.equal(validateThreeImmediatePhase0DiagnosticVertexInputs(objectId, {
    purpose: 'address', lane: 'A', mergedPosition, mergedBucketBase,
  }).valid, false);
  for (const mutate of [
    (value) => value.vertexInputs.splice(1, 0, {
      ...value.vertexInputs[0], name: 'normal', shaderLocation: 1, resourceId: 999,
    }),
    (value) => { value.rawEvents[0].slot = 1; },
    (value) => { value.vertexInputs[0].resourceId = 703; },
    (value) => { value.rawEvents[0].bufferId = 'wrong-bucket-base'; },
    (value) => {
      value.vertexInputs.reverse();
      value.expectedVertexInputs.reverse();
      value.pipelineVertexInputs.reverse();
    },
  ]) {
    const tampered = structuredClone(address);
    mutate(tampered);
    assert.equal(validateThreeImmediatePhase0DiagnosticVertexInputs(tampered, {
      purpose: 'address', lane: 'A', mergedBucketBase,
    }).valid, false);
  }
});

test('production storage bindings join bundle bind groups to committed matrix and visible IDs', () => {
  const bundle = {
    pipelineId: 'production-A', finishSequence: 50,
    rawEvents: [{ method: 'setBindGroup', sequence: 30, index: 1, bindGroupId: 'bind-group-1',
      dynamicOffsets: null, dynamicOffsetStart: null, dynamicOffsetLength: null,
      selectedDynamicOffsets: null }],
  };
  const shaders = { lanes: { A: { storageBindings: phase0StorageBindings() } } };
  const resources = {
    gpuResourceCommitments: { records: [
      { semantic: 'matrix', attributeId: 'matrix-attribute', gpuBufferId: 'matrix-buffer',
        byteLength: 65_536 * 64 },
      { semantic: 'visibleIds', attributeId: 'visible-attribute',
        gpuBufferId: 'visible-buffer', byteLength: 65_536 * 4 },
    ] },
    gpuCreations: [{
      method: 'createBindGroup', resourceId: 'bind-group-1', layoutId: 'bgl-1', sequence: 20,
      entries: [
        { binding: 3, bufferId: 'matrix-buffer', bufferOffset: 0, bufferSize: null },
        { binding: 4, bufferId: 'visible-buffer', bufferOffset: 0, bufferSize: null },
      ],
    }, {
      method: 'createBuffer', resourceId: 'matrix-buffer', sequence: 10,
      size: 65_536 * 64, usage: 172,
    }, {
      method: 'createBuffer', resourceId: 'visible-buffer', sequence: 11,
      size: 65_536 * 4, usage: 172,
    }],
  };
  const pipelines = {
    render: [{ pipelineId: 'production-A', layoutId: 'layout-1' }],
    layouts: [{ layoutId: 'layout-1', bindGroupLayoutIds: ['unused-bgl', 'bgl-1'] }],
    bindGroupLayouts: [{ layoutId: 'bgl-1', sequence: 5,
      entries: [{ binding: 3, visibility: 1,
        buffer: { type: 'read-only-storage', hasDynamicOffset: false, minBindingSize: 0 } },
      { binding: 4, visibility: 1,
        buffer: { type: 'read-only-storage', hasDynamicOffset: false, minBindingSize: 0 } }] }],
  };
  const audit = (value) => validateThreeImmediatePhase0ProductionStorageBindings(
    value.bundle, { lane: 'A', scenarioId: 'v99', shaders: value.shaders,
      resources: value.resources, pipelines: value.pipelines },
  );
  const fixture = { bundle, shaders, resources, pipelines };
  assert.deepEqual(audit(fixture), { valid: true, reasons: [] });
  for (const mutate of [
    (value) => { value.resources.gpuCreations[0].entries[0].bufferId = 'visible-buffer'; },
    (value) => { value.shaders.lanes.A.storageBindings[0].binding = 4; },
    (value) => { value.resources.gpuCreations[0].layoutId = 'stale-layout'; },
    (value) => { value.pipelines.bindGroupLayouts[0].entries[0].buffer.type = 'storage'; },
    (value) => { value.bundle.rawEvents[0].selectedDynamicOffsets = []; },
  ]) {
    const tampered = structuredClone(fixture);
    mutate(tampered);
    assert.equal(audit(tampered).valid, false);
  }
});

test('diagnostic storage bindings independently join raw bind groups to scenario commitments', () => {
  const makeFixture = (kind) => {
    const semantics = kind === 'address' ? ['visibleIds'] : ['matrix', 'visibleIds'];
    const allBindings = phase0StorageBindings();
    const bindings = allBindings.filter((binding) => semantics.includes(binding.semantic));
    const commitments = {
      matrix: { semantic: 'matrix', attributeId: 'matrix-attribute',
        gpuBufferId: 'matrix-buffer', byteLength: 4_194_304 },
      visibleIds: { semantic: 'visibleIds', attributeId: 'visible-attribute',
        gpuBufferId: 'visible-buffer', byteLength: 262_144 },
    };
    const entries = bindings.map((binding) => ({
      binding: binding.binding,
      bufferId: commitments[binding.semantic].gpuBufferId,
      bufferOffset: 0,
      bufferSize: null,
    }));
    const producer = {
      pipelineId: `${kind}-pipeline`,
      pipelineLayoutId: 'diagnostic-pipeline-layout',
      finishSequence: 50,
      storageBindings: bindings,
      rawEvents: [{
        method: 'setBindGroup', sequence: 30, index: 1,
        bindGroupId: 'diagnostic-bind-group', dynamicOffsets: null,
        dynamicOffsetStart: null, dynamicOffsetLength: null, selectedDynamicOffsets: null,
      }, { method: 'drawIndexedIndirect', sequence: 40 }],
      boundStorage: bindings.map((binding) => ({
        ...binding,
        pass: true,
        bindGroupId: 'diagnostic-bind-group',
        bindGroupLayoutId: 'diagnostic-bgl',
        expectedGpuBufferId: commitments[binding.semantic].gpuBufferId,
        observedGpuBufferId: commitments[binding.semantic].gpuBufferId,
      })),
    };
    const resources = {
      gpuResourceCommitments: { records: semantics.map((semantic) => commitments[semantic]) },
      gpuCreations: [{
        method: 'createBindGroup', resourceId: 'diagnostic-bind-group',
        layoutId: 'diagnostic-bgl', sequence: 20, entries,
      }, ...semantics.map((semantic, index) => ({
        method: 'createBuffer', resourceId: commitments[semantic].gpuBufferId,
        resourceClass: 'persistent-or-upload-buffer', sequence: 10 + index,
        size: commitments[semantic].byteLength, usage: 172,
      }))],
    };
    const pipelines = {
      render: [{ pipelineId: `${kind}-pipeline`, layoutId: 'diagnostic-pipeline-layout' }],
      layouts: [{
        layoutId: 'diagnostic-pipeline-layout', bindGroupLayoutIds: ['unused', 'diagnostic-bgl'],
      }],
      bindGroupLayouts: [{
        layoutId: 'diagnostic-bgl', sequence: 5,
        entries: bindings.map((binding) => ({
          binding: binding.binding,
          visibility: 1,
          buffer: { type: 'read-only-storage', hasDynamicOffset: false, minBindingSize: 0 },
        })),
      }],
    };
    return { producer, resources, pipelines };
  };
  for (const kind of ['address', 'object-id']) {
    const fixture = makeFixture(kind);
    const audit = (value) => validateThreeImmediatePhase0DiagnosticStorageBindings(
      value.producer, { kind, scenarioId: 'v99', resources: value.resources,
        pipelines: value.pipelines },
    );
    assert.deepEqual(audit(fixture), { valid: true, reasons: [] });
    for (const mutate of [
      (value) => { value.producer.rawEvents[0].dynamicOffsets = []; },
      (value) => { value.resources.gpuCreations[0].entries[0].bufferId = 'duplicate-buffer'; },
      (value) => { value.pipelines.bindGroupLayouts[0].entries[0].binding += 1; },
      (value) => { value.resources.gpuCreations.at(-1).sequence = 25; },
      (value) => { value.producer.boundStorage[0].observedGpuBufferId = 'stale-buffer'; },
    ]) {
      const tampered = structuredClone(fixture);
      mutate(tampered);
      assert.equal(audit(tampered).valid, false, `${kind} coherent storage mutation passed`);
    }
  }
});

test('ordered production challenges reject texture injection and immutable buffer mutation', () => {
  const base = {
    queueWriteBufferCount: 1,
    queueWriteBuffers: [{ bufferId: 'renderer-uniform' }],
    queueWriteTextureCount: 0,
    queueWriteTextures: [],
    queueCopyExternalImageToTextureCount: 0,
    queueCopyExternalImagesToTexture: [],
    frozenBufferWriteCount: 0,
    frozenBufferWrites: [],
    frozenBufferMapCount: 0,
    frozenBufferMaps: [],
    immutableBufferWriteCount: 0,
    immutableBufferWrites: [],
    immutableBufferMapCount: 0,
    immutableBufferMaps: [],
  };
  const options = {
    queueWrites: base.queueWriteBuffers,
    queueTextureWrites: [],
    maps: [],
    frozenIds: ['command-buffer'],
    immutableIds: ['command-buffer', 'visible-ids', 'geometry-position'],
  };
  assert.deepEqual(validateThreeImmediatePhase0ChallengeWriteClosure(base, options), {
    valid: true, reasons: [],
  });
  const textureInjected = structuredClone(base);
  textureInjected.queueWriteTextureCount = 1;
  textureInjected.queueWriteTextures = [{ textureId: 'production-target' }];
  assert.equal(validateThreeImmediatePhase0ChallengeWriteClosure(textureInjected, {
    ...options, queueTextureWrites: textureInjected.queueWriteTextures,
  }).valid, false);
  const externalInjected = structuredClone(base);
  externalInjected.queueCopyExternalImageToTextureCount = 1;
  externalInjected.queueCopyExternalImagesToTexture = [{ textureId: 'production-target' }];
  assert.equal(validateThreeImmediatePhase0ChallengeWriteClosure(externalInjected, {
    ...options, queueExternalCopies: externalInjected.queueCopyExternalImagesToTexture,
  }).valid, false);
  const immutableWrite = structuredClone(base);
  immutableWrite.queueWriteBufferCount = 2;
  immutableWrite.queueWriteBuffers.push({ bufferId: 'visible-ids' });
  immutableWrite.immutableBufferWriteCount = 1;
  immutableWrite.immutableBufferWrites = [{ bufferId: 'visible-ids' }];
  assert.equal(validateThreeImmediatePhase0ChallengeWriteClosure(immutableWrite, {
    ...options, queueWrites: immutableWrite.queueWriteBuffers,
  }).valid, false);
});

test('device instrumentation requires importExternalTexture coverage and globally zero texture injection', () => {
  const expected = [
    'createShaderModule', 'createPipelineLayout', 'createBindGroupLayout',
    'createRenderPipeline', 'createRenderPipelineAsync',
    'createComputePipeline', 'createComputePipelineAsync',
    'createBuffer', 'createTexture', 'createBindGroup', 'createSampler', 'createQuerySet',
    'importExternalTexture',
    'queue.writeBuffer', 'queue.writeTexture',
    'queue.copyExternalImageToTexture', 'queue.submit',
    'createRenderBundleEncoder', 'createCommandEncoder',
  ];
  const fixture = {
    patchedMethods: expected, gpuCreations: [], queueWriteTextures: [],
    queueExternalCopies: [],
  };
  assert.equal(validateThreeImmediatePhase0DeviceInstrumentationClosure(fixture).valid, true);
  const omitted = structuredClone(fixture);
  omitted.patchedMethods.splice(11, 1);
  assert.equal(validateThreeImmediatePhase0DeviceInstrumentationClosure(omitted).valid, false);
  const imported = structuredClone(fixture);
  imported.gpuCreations.push({ method: 'importExternalTexture' });
  assert.equal(validateThreeImmediatePhase0DeviceInstrumentationClosure(imported).valid, false);
  const primeWrite = structuredClone(fixture);
  primeWrite.queueWriteTextures.push({ capturePhase: 'phase0/prime/A' });
  assert.equal(validateThreeImmediatePhase0DeviceInstrumentationClosure(primeWrite).valid, false);
  const postflightCopy = structuredClone(fixture);
  postflightCopy.queueExternalCopies.push({ capturePhase: 'phase0/v20/postflight' });
  assert.equal(validateThreeImmediatePhase0DeviceInstrumentationClosure(postflightCopy).valid, false);
});

function challengedReadbackFixture() {
  const phase = 'phase0/v99/canonical/A/capture-1/production';
  const stagingCreations = [0, 1].map((index) => ({
    sequence: 10 + index * 5,
    capturePhase: phase,
    category: 'createBuffer',
    resourceId: `staging-${index}`,
    resourceClass: 'readback-staging',
  }));
  const gpuCreations = stagingCreations.map((record) => ({
    sequence: record.sequence,
    capturePhase: phase,
    method: 'createBuffer',
    resourceId: record.resourceId,
    resourceClass: 'readback-staging',
    size: TARGET_BYTES,
    usage: 9,
  }));
  const transfers = ['production-color', 'production-depth'].map((textureId, index) => ({
    sequence: 12 + index * 5,
    capturePhase: phase,
    commandEncoderId: `encoder-${index + 2}`,
    method: 'copyTextureToBuffer',
    source: {
      textureId, mipLevel: 0, origin: { x: 0, y: 0, z: 0 }, aspect: 'all',
    },
    destination: {
      bufferId: `staging-${index}`, offset: 0, bytesPerRow: 5_120,
      rowsPerImage: null,
    },
    copySize: { width: 1_280, height: 720, depthOrArrayLayers: 1 },
  }));
  const renderPasses = [
    { sequence: 2, capturePhase: phase, encoderId: 'render-pass-clear' },
    { sequence: 7, capturePhase: phase, encoderId: 'render-pass-draw' },
  ];
  const commandEncoders = [
    {
      sequence: 1, capturePhase: phase, commandEncoderId: 'encoder-0',
      renderPassEncoderIds: ['render-pass-clear'], computePassEncoderIds: [],
      transferSequences: [], finishCallCount: 1, finishSequence: 3,
      commandBufferId: 'command-buffer-0', finishDescriptor: null,
    },
    {
      sequence: 6, capturePhase: phase, commandEncoderId: 'encoder-1',
      renderPassEncoderIds: ['render-pass-draw'], computePassEncoderIds: [],
      transferSequences: [], finishCallCount: 1, finishSequence: 8,
      commandBufferId: 'command-buffer-1', finishDescriptor: null,
    },
    {
      sequence: 11, capturePhase: phase, commandEncoderId: 'encoder-2',
      renderPassEncoderIds: [], computePassEncoderIds: [],
      transferSequences: [12], finishCallCount: 1, finishSequence: 13,
      commandBufferId: 'command-buffer-2', finishDescriptor: null,
    },
    {
      sequence: 16, capturePhase: phase, commandEncoderId: 'encoder-3',
      renderPassEncoderIds: [], computePassEncoderIds: [],
      transferSequences: [17], finishCallCount: 1, finishSequence: 18,
      commandBufferId: 'command-buffer-3', finishDescriptor: null,
    },
  ];
  const queueSubmissions = [4, 9, 14, 19].map((sequence, index) => ({
    sequence, capturePhase: phase, commandBufferIds: [`command-buffer-${index}`],
  }));
  return {
    phase, stagingCreations, gpuCreations, transfers,
    renderPasses, commandEncoders, queueSubmissions,
  };
}

test('challenged readbacks and submissions reject coherent transfer/execution tampering', () => {
  const fixture = challengedReadbackFixture();
  const readback = (value) => validateThreeImmediatePhase0ProductionReadbackTransfers({
    phases: [value.phase], transfers: value.transfers,
    stagingCreations: value.stagingCreations, gpuCreations: value.gpuCreations,
    colorTextureId: 'production-color', depthTextureId: 'production-depth',
  });
  assert.deepEqual(readback(fixture), { valid: true, reasons: [] });
  for (const mutate of [
    (value) => { value.transfers[1].source.textureId = 'production-color'; },
    (value) => { value.transfers.reverse(); },
    (value) => { value.transfers[0].source.origin.x = 1; },
    (value) => { value.transfers[0].destination.bytesPerRow = 5_376; },
    (value) => { value.transfers[0].copySize.width = 1_279; },
    (value) => { value.gpuCreations[0].usage = 8; },
  ]) {
    const tampered = structuredClone(fixture);
    mutate(tampered);
    assert.equal(readback(tampered).valid, false);
  }
  const submission = (value) => validateThreeImmediatePhase0CommandSubmissionClosure({
    phases: [value.phase], commandEncoders: value.commandEncoders,
    queueSubmissions: value.queueSubmissions, renderPasses: value.renderPasses,
    transfers: value.transfers,
  });
  assert.deepEqual(submission(fixture), { valid: true, reasons: [] });
  const unused = structuredClone(fixture);
  unused.commandEncoders[3].transferSequences = [];
  assert.equal(submission(unused).valid, false);
  const preencoded = structuredClone(fixture);
  preencoded.commandEncoders[3].sequence = -5;
  assert.equal(submission(preencoded).valid, false);
  const resubmitted = structuredClone(fixture);
  resubmitted.queueSubmissions[3].commandBufferIds = ['command-buffer-2'];
  assert.equal(submission(resubmitted).valid, false);
});

function globalCommandClosureFixture() {
  const base = challengedReadbackFixture();
  base.renderPasses[0].commandEncoderId = 'encoder-0';
  base.renderPasses[1].commandEncoderId = 'encoder-1';
  const stagingCreations = base.gpuCreations.map((record, index) => ({
    ...record,
    mappedAtCreation: false,
    destroyed: true,
    destroyCallCount: 1,
    destroySequence: index === 0 ? 22 : 25,
    destroyCapturePhase: base.phase,
  }));
  const bufferMapEvents = [
    { sequence: 20, capturePhase: base.phase, resourceId: 'staging-0',
      resourceClass: 'readback-staging', method: 'mapAsync', mode: 1, offset: 0, size: null },
    { sequence: 21, capturePhase: base.phase, resourceId: 'staging-0',
      resourceClass: 'readback-staging', method: 'getMappedRange', mode: null, offset: 0,
      size: null },
    { sequence: 23, capturePhase: base.phase, resourceId: 'staging-1',
      resourceClass: 'readback-staging', method: 'mapAsync', mode: 1, offset: 0, size: null },
    { sequence: 24, capturePhase: base.phase, resourceId: 'staging-1',
      resourceClass: 'readback-staging', method: 'getMappedRange', mode: null, offset: 0,
      size: null },
  ];
  const bufferDestructions = stagingCreations.map((record) => ({
    sequence: record.destroySequence, capturePhase: base.phase,
    resourceId: record.resourceId, resourceClass: 'readback-staging',
    method: 'destroy', callCount: 1,
  }));
  const textureCreations = ['production-color', 'production-depth'].map(
    (resourceId, index) => ({
      sequence: -2 + index, capturePhase: 'phase0/target-init', method: 'createTexture',
      resourceId, format: index === 0 ? 'rgba8unorm' : 'depth32float',
      size: { width: 1_280, height: 720, depthOrArrayLayers: 1 },
    }),
  );
  return {
    commandEncoders: base.commandEncoders,
    renderPasses: base.renderPasses,
    computePasses: [],
    transfers: base.transfers,
    queueSubmissions: base.queueSubmissions,
    stagingCreations,
    bufferMapEvents,
    bufferDestructions,
    bufferCreations: stagingCreations,
    textureCreations,
    queueWriteTextures: [],
    queueExternalCopies: [],
    expectedRenderPassCount: 2,
    expectedComputePassCount: 0,
    expectedTransferCount: 2,
  };
}

test('global command ledger is an exact encoder/submit/staging lifecycle partition', () => {
  const fixture = globalCommandClosureFixture();
  const audit = (value) => validateThreeImmediatePhase0GlobalCommandClosure(value);
  assert.deepEqual(audit(fixture), { valid: true, reasons: [] });
  const mutations = [
    (value) => value.transfers.push({ ...value.transfers[0], sequence: 18,
      destination: { ...value.transfers[0].destination, bufferId: 'staging-1' } }),
    (value) => { value.transfers[1].destination.bufferId = 'staging-0'; },
    (value) => { value.commandEncoders[0].transferSequences = [value.transfers[0].sequence]; },
    (value) => { value.queueSubmissions[3].commandBufferIds = ['unknown-command-buffer']; },
    (value) => { value.queueSubmissions[3].commandBufferIds = ['command-buffer-2']; },
    (value) => { value.bufferMapEvents[0].sequence = 13; },
    (value) => { value.stagingCreations[0].mappedAtCreation = true; },
    (value) => { value.transfers[0].method = 'clearBuffer'; },
    (value) => value.queueWriteTextures.push({ sequence: 5 }),
    (value) => value.queueExternalCopies.push({ sequence: 5 }),
  ];
  for (const mutate of mutations) {
    const tampered = structuredClone(fixture);
    mutate(tampered);
    assert.equal(audit(tampered).valid, false);
  }
});

test('buffer map closure includes transfer use when proving mapped-at-creation chronology', () => {
  const persistent = {
    method: 'createBuffer', resourceId: 'persistent-source',
    resourceClass: 'persistent-or-upload-buffer', capturePhase: 'phase0/init',
    sequence: 1, size: 16, usage: 8, mappedAtCreation: true,
  };
  const persistentMaps = [
    { sequence: 2, capturePhase: 'phase0/init', resourceId: persistent.resourceId,
      method: 'getMappedRange', mode: null, offset: 0, size: null },
    { sequence: 3, capturePhase: 'phase0/init', resourceId: persistent.resourceId,
      method: 'unmap', mode: null, offset: null, size: null },
  ];
  const staging = [];
  const transferLedger = [];
  const transfers = [];
  const stagingMaps = [];
  for (let index = 0; index < 825; index += 1) {
    const base = 10_000 + index * 10;
    const capturePhase = `phase0/readback/${index}`;
    const resourceId = `staging-${index}`;
    const transfer = {
      sequence: index === 0 ? 4 : base + 1,
      capturePhase,
      method: 'copyBufferToBuffer',
      sourceBufferId: persistent.resourceId,
      destinationBufferId: resourceId,
    };
    const mapEvents = [
      { sequence: base + 2, capturePhase, resourceId, method: 'mapAsync',
        mode: 1, offset: 0, size: 4 },
      { sequence: base + 3, capturePhase, resourceId, method: 'getMappedRange',
        mode: null, offset: 0, size: 4 },
    ];
    staging.push({
      method: 'createBuffer', resourceId, resourceClass: 'readback-staging',
      capturePhase, sequence: base, size: 4, usage: 9, mappedAtCreation: false,
    });
    transfers.push(transfer);
    stagingMaps.push(...mapEvents);
    transferLedger.push({ pass: true, stagingBufferId: resourceId, transfer, mapEvents });
  }
  const options = {
    lifecycle: {
      queueWriteBufferCalls: [], renderBundleTraces: [], renderPassTraces: [],
      computePassTraces: [], commandEncoderTransferCalls: transfers,
      bufferMapEvents: [...persistentMaps, ...stagingMaps],
    },
    resources: { gpuCreations: [persistent, ...staging] },
    stagingTransferLedger: transferLedger,
  };
  const reconstruction = validateThreeImmediatePhase0BufferMapLifecycle(null, options);
  assert.equal(reconstruction.valid, false);
  assert.equal(validateThreeImmediatePhase0BufferMapLifecycle(
    reconstruction.evidence, options,
  ).valid, true);
  const bad = structuredClone(options);
  bad.lifecycle.bufferMapEvents[1].sequence = 5;
  assert.equal(validateThreeImmediatePhase0BufferMapLifecycle(
    reconstruction.evidence, bad,
  ).valid, false);
});

test('all 825 readbacks are bound to the independently expected semantic source', () => {
  const fixture = exhaustiveReadbackSourceFixture();
  assert.equal(validateThreeImmediatePhase0ReadbackSourceBijection(
    fixture.evidence, fixture,
  ).valid, true);
  for (const mutate of [
    (value) => { value.transfers[0].source.textureId = 'texture-objectId.color'; },
    (value) => { value.transfers[0].source.origin.x = 1; },
    (value) => { value.transfers[0].destination.bytesPerRow += 256; },
    (value) => { value.transferLedger[0].sourceBinding.semantic = 'objectId.color'; },
  ]) {
    const bad = structuredClone(fixture);
    mutate(bad);
    assert.equal(validateThreeImmediatePhase0ReadbackSourceBijection(
      bad.evidence, bad,
    ).valid, false);
  }
});

function staticGpuFixture(name = 'NVIDIA GeForce RTX 5070 Ti') {
  return {
    gpuStaticTelemetryIdentity: {
      devices: [{ name, driverVersion: '581.35' }],
    },
  };
}

function cdpGpuFixture(deviceString = 'NVIDIA GeForce RTX 5070 Ti') {
  return {
    schemaVersion: 1,
    kind: 'chrome-cdp-static-gpu-system-info',
    modelName: '',
    modelVersion: '',
    gpu: {
      devices: [{
        vendorId: 0x10de,
        deviceId: 0x2f04,
        vendorString: 'NVIDIA',
        deviceString,
        driverVersion: '32.0.15.8135',
      }],
    },
  };
}

test('privacy-reduced adapter is accepted only through exact CDP/static 5070 Ti association', () => {
  const environment = staticGpuFixture();
  const systemInfo = cdpGpuFixture();
  const adapter = {
    vendor: 'nvidia', architecture: 'blackwell', device: '', description: '',
    isFallbackAdapter: false,
  };
  assert.equal(browserGpuMatchesStaticIdentity(systemInfo, environment), true);
  assert.equal(adapterMatchesStaticGpu(adapter, environment, systemInfo), true);
  assert.equal(adapterMatchesStaticGpu(adapter, environment, cdpGpuFixture(
    'NVIDIA GeForce RTX 5080',
  )), false);
  const softwareAugmented = cdpGpuFixture();
  softwareAugmented.gpu.devices.push({
    vendorId: 0x1ae0,
    deviceId: 0xc0de,
    vendorString: 'Google',
    deviceString: 'SwiftShader Device (Subzero)',
    driverVersion: '0',
  });
  assert.equal(browserGpuMatchesStaticIdentity(softwareAugmented, environment), false);
  assert.equal(adapterMatchesStaticGpu(adapter, environment, softwareAugmented), false);
  assert.equal(adapterMatchesStaticGpu(adapter, staticGpuFixture(
    'NVIDIA GeForce RTX 5080',
  ), cdpGpuFixture('NVIDIA GeForce RTX 5080')), false);
  assert.equal(adapterMatchesStaticGpu({ ...adapter, architecture: 'ada' },
    environment, systemInfo), false);
});

function addressWitnessFixture() {
  const scenarios = scenarioSet();
  const expectedWitnessIds = [];
  const witnesses = {};
  for (const scenarioId of ['v99', 'v20']) {
    for (const scheduleId of IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS) {
      for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
        const witnessId = `${scenarioId}/${scheduleId}/${lane}`;
        const bytes = createImmediateAifPackedAddressBytes(createImmediateAifAddressOracle({
          lane, scenario: scenarios[scenarioId], scheduleId,
        }));
        const hash = sha256(bytes);
        expectedWitnessIds.push(witnessId);
        witnesses[witnessId] = {
          schemaVersion: 1,
          kind: 'immediate-aif-phase0-address-byte-witness',
          witnessId, scenarioId, scheduleId, lane,
          encoding: 'base64-rgba8unorm',
          byteLength: bytes.byteLength,
          sha256: hash,
          expectedSha256: hash,
          bytesBase64: Buffer.from(bytes).toString('base64'),
        };
      }
    }
  }
  return {
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-address-byte-witnesses',
    pass: true,
    witnessCount: expectedWitnessIds.length,
    expectedWitnessIds,
    witnesses,
  };
}

test('address witness validator accepts the full independent oracle and rejects raw tampering', () => {
  const fixture = addressWitnessFixture();
  assert.deepEqual(validateThreeImmediatePhase0AddressWitnesses(fixture), {
    valid: true, reasons: [],
  });
  const tampered = structuredClone(fixture);
  const record = tampered.witnesses['v99/canonical/A'];
  const bytes = Buffer.from(record.bytesBase64, 'base64');
  bytes[0] ^= 1;
  record.bytesBase64 = bytes.toString('base64');
  assert.equal(validateThreeImmediatePhase0AddressWitnesses(tampered).valid, false);
});

test('address diagnostic position witnesses and all eight GPU realizations are independently frozen', () => {
  const fixture = addressDiagnosticPositionFixture();
  const audit = (value) => validateThreeImmediatePhase0AddressDiagnosticPositions(
    value.evidence,
    {
      lifecycle: value.lifecycle,
      gpuCreations: value.gpuCreations,
      scheduleInstallations: value.scheduleInstallations,
    },
  );
  assert.deepEqual(audit(fixture), { valid: true, reasons: [] });

  for (const mutate of [
    (value) => {
      const [witness] = Object.values(value.evidence.witnesses);
      const bytes = Buffer.from(witness.bytesBase64, 'base64');
      bytes[0] ^= 1;
      witness.bytesBase64 = bytes.toString('base64');
      witness.sha256 = sha256(bytes);
    },
    (value) => { value.evidence.records[1].records[1].beforeDataVersion += 1; },
    (value) => { value.lifecycle.queueWriteBufferCalls[0].sourceByteLength -= 4; },
    (value) => { value.evidence.records[4].laneBindings.F.gpuBufferId = 'swapped-buffer'; },
    (value) => { value.scheduleInstallations[2].diagnosticPositionRealization.pass = false; },
    (value) => { value.scheduleInstallations[3].completeSequence = 425; },
    (value) => { value.gpuCreations[0].capturePhase = 'phase0/late-hostile-create'; },
  ]) {
    const tampered = structuredClone(fixture);
    mutate(tampered);
    assert.equal(audit(tampered).valid, false);
  }
});

test('runner-local five-word commands cross-check the shared page builder without accepting it', () => {
  const scenarios = scenarioSet();
  const geometries = createIndexedGeometryFixtures(32, 'medium');
  try {
    for (const scenarioId of ['v99', 'v20']) {
      for (const scheduleId of IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS) {
        for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
          const shared = createImmediateAifCommandOracle({
            lane, scenario: scenarios[scenarioId], sourceGeometries: geometries, scheduleId,
          });
          assert.deepEqual(
            createThreeImmediatePhase0ExpectedCommandWords(scenarioId, scheduleId, lane),
            shared.postCullCommands,
          );
          assert.deepEqual(
            createThreeImmediatePhase0ExpectedCommandWords(
              scenarioId, scheduleId, lane, { initial: true },
            ),
            shared.initialCommands,
          );
        }
      }
    }
  } finally {
    geometries.forEach((geometry) => geometry.dispose());
  }
});

test('pinned geometry/scenario digests reject a coherently rehashed shared manifest mutation', async () => {
  const geometries = createIndexedGeometryFixtures(32, 'medium');
  try {
    const scenarios = createImmediateAifPhase0Scenarios({
      geometrySpheres: geometries.map((geometry) => geometry.boundingSphere.clone()),
    });
    const source = await fingerprintImmediateAifPhase0GeometryFixtures(geometries, 'medium');
    const scenarioManifests = {
      v99: await fingerprintFixedSubsetScenario(scenarios.v99, 0xb1ad_2026),
      v20: await fingerprintFixedSubsetScenario(scenarios.v20, 0xb1ad_2026),
    };
    assert.deepEqual(validateThreeImmediatePhase0PinnedGeometryManifests(
      source, scenarioManifests,
    ), { valid: true, reasons: [] });
    const tampered = structuredClone(source);
    tampered.geometries[0].name = 'coherently-rehashed-substitute';
    const { sha256: oldRecordSha, ...recordPayload } = tampered.geometries[0];
    tampered.geometries[0].sha256 = sha256(Buffer.from(JSON.stringify(recordPayload)));
    const { sha256: oldManifestSha, ...manifestPayload } = tampered;
    tampered.sha256 = sha256(Buffer.from(JSON.stringify(manifestPayload)));
    assert.notEqual(oldRecordSha, tampered.geometries[0].sha256);
    assert.notEqual(oldManifestSha, tampered.sha256);
    assert.equal(validateThreeImmediatePhase0PinnedGeometryManifests(
      tampered, scenarioManifests,
    ).valid, false);
  } finally {
    geometries.forEach((geometry) => geometry.dispose());
  }
});

function outputBytesAndRecords() {
  const scenarios = scenarioSet();
  const color = Buffer.alloc(TARGET_BYTES);
  color.set([21, 43, 65, 255], 4);
  const depth = Buffer.alloc(TARGET_BYTES);
  depth.writeFloatLE(0.5, 4);
  const objects = {};
  for (const scenarioId of ['v99', 'v20']) {
    const bytes = Buffer.alloc(TARGET_BYTES);
    const objectId = scenarios[scenarioId].expectedVisibleIds[0];
    const encoded = objectId + 1;
    bytes.set([encoded & 0xff, (encoded >>> 8) & 0xff, (encoded >>> 16) & 0xff, 255], 4);
    objects[scenarioId] = { bytes, objectId };
  }
  const makeOutput = (channel, bytes, extra) => ({
    format: channel === 'depth' ? 'depth32float'
      : channel === 'objectId' ? 'rgba8unorm-object-id-plus-one' : 'rgba8unorm',
    arrayType: channel === 'depth' ? 'Float32Array' : 'Uint8Array',
    byteLength: bytes.length,
    sha256: sha256(bytes),
    witnessId: `${channel}/${sha256(bytes)}`,
    ...extra,
  });
  const colorOutput = makeOutput('color', color, {
    baselineRgba: [0, 0, 0, 0], nonClearPixelCount: 1, contentPass: true,
  });
  const depthOutput = makeOutput('depth', depth, {
    zeroDepthCount: TARGET_PIXELS - 1, nonzeroFiniteDepthCount: 1, contentPass: true,
  });
  const objectOutputs = Object.fromEntries(['v99', 'v20'].map((scenarioId) => [
    scenarioId, makeOutput('objectId', objects[scenarioId].bytes, {}),
  ]));
  const decodedFor = (scenarioId) => ({
    pass: true,
    coveredPixelCount: 1,
    backgroundPixelCount: TARGET_PIXELS - 1,
    outOfRangePixelCount: 0,
    hiddenPixelCount: 0,
    uniqueObservedIdCount: 1,
    observedObjectIds: [objects[scenarioId].objectId],
    observedIdCounts: [{ objectId: objects[scenarioId].objectId, pixelCount: 1 }],
  });
  const witness = (channel, output, bytes, decoded = null) => ({
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-output-byte-witness',
    witnessId: output.witnessId,
    channel,
    format: output.format,
    arrayType: output.arrayType,
    byteLength: output.byteLength,
    sha256: output.sha256,
    encoding: 'base64-exact-bytes',
    bytesBase64: bytes.toString('base64'),
    decoded,
  });
  return {
    colorOutput, depthOutput, objectOutputs,
    witnessRecords: [
      witness('color', colorOutput, color),
      witness('depth', depthOutput, depth),
      ...['v99', 'v20'].map((scenarioId) => witness(
        'objectId', objectOutputs[scenarioId], objects[scenarioId].bytes,
        decodedFor(scenarioId),
      )),
    ],
  };
}

function outputWitnessFixture() {
  const { colorOutput, depthOutput, objectOutputs, witnessRecords } = outputBytesAndRecords();
  const callbacks = [];
  const makeCallback = (scenarioId, scheduleId, lane, phase) => {
    const callback = {
      pass: true, phase, scenarioId, scheduleId, lane,
      phases: {
        production: `${phase}/production`,
        objectId: `${phase}/object-id`,
      },
      output: {
        color: structuredClone(colorOutput),
        depth: structuredClone(depthOutput),
        objectId: structuredClone(objectOutputs[scenarioId]),
      },
    };
    callbacks.push(callback);
    return callback;
  };
  const scenarios = ['v99', 'v20'].map((scenarioId) => {
    const snapshot = (label, scheduleId, index) => {
      const base = `fixture/${scenarioId}/${label}`;
      if (index < 3) {
        return {
          rerecord: index === 0 ? null : {
            callback: makeCallback(scenarioId, scheduleId, 'I', `${base}/rerecord-I`),
          },
          laneOrders: {
            results: IMMEDIATE_AIF_PHASE0_LANE_ORDERS.map((order, orderIndex) => ({
              order,
              records: order.map((lane, position) => ({
                lane,
                callback: makeCallback(
                  scenarioId, scheduleId, lane,
                  `${base}/order-${orderIndex}-${order.join('')}/position-${position}-${lane}`,
                ),
              })),
            })),
          },
        };
      }
      return {
        rerecord: index === 3 ? {
          callback: makeCallback(scenarioId, scheduleId, 'I', `${base}/rerecord-I`),
        } : null,
        laneCaptures: IMMEDIATE_AIF_PHASE0_LANES.map((lane) => (
          makeCallback(scenarioId, scheduleId, lane, `${base}/audit-${lane}`)
        )),
      };
    };
    return {
      scenarioId,
      prime: scenarioId === 'v99' ? {
        records: ['A', 'F', 'I'].map((lane) => ({
          lane,
          callback: {
            pass: true,
            output: {
              color: structuredClone(colorOutput),
              depth: structuredClone(depthOutput),
            },
          },
        })),
      } : null,
      snapshots: [
        snapshot('canonical-preflight', 'canonical', 0),
        snapshot('S1', 'S1', 1),
        snapshot('S2', 'S2', 2),
        snapshot('canonical-restored', 'canonical', 3),
        snapshot('canonical-postflight', 'canonical', 4),
      ],
    };
  });
  assert.equal(callbacks.length, 126);
  const references = callbacks.flatMap((callback, callbackIndex) => (
    ['color', 'depth', 'objectId'].map((channel) => {
      const output = callback.output[channel];
      return {
        callbackOrdinal: callbackIndex + 1,
        phase: callback.phase,
        scenarioId: callback.scenarioId,
        scheduleId: callback.scheduleId,
        lane: callback.lane,
        channel,
        witnessId: output.witnessId,
        sha256: output.sha256,
        byteLength: output.byteLength,
      };
    })
  ));
  const observations = [];
  const appendObservation = (output, context, capturePhase, observationKind, channel) => {
    observations.push({
      ordinal: observations.length + 1,
      observationId: `${capturePhase}/${channel}`,
      observationKind, capturePhase,
      scenarioId: context.scenarioId, scheduleId: context.scheduleId, lane: context.lane,
      channel, witnessId: output.witnessId, sha256: output.sha256,
      byteLength: output.byteLength,
    });
  };
  for (const record of scenarios[0].prime.records) {
    for (const channel of ['color', 'depth']) {
      appendObservation(record.callback.output[channel], {
        scenarioId: 'v99', scheduleId: 'canonical', lane: record.lane,
      }, `phase0/prime/${record.lane}`, 'bundle-prime', channel);
    }
  }
  for (const callback of callbacks) {
    for (const channel of ['color', 'depth']) {
      appendObservation(callback.output[channel], callback,
        callback.phases.production, 'challenged-render-callback', channel);
    }
    appendObservation(callback.output.objectId, callback,
      callback.phases.objectId, 'challenged-render-callback', 'objectId');
  }
  const objectIdBaselines = {};
  for (const scenarioId of ['v99', 'v20']) {
    for (const scheduleId of IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS) {
      const witnessId = objectOutputs[scenarioId].witnessId;
      const laneWitnessIds = Object.fromEntries(IMMEDIATE_AIF_PHASE0_LANES.map(
        (lane) => [lane, [witnessId]],
      ));
      objectIdBaselines[`${scenarioId}/${scheduleId}`] = {
        scenarioId, scheduleId, baselineLane: 'A', baselineWitnessId: witnessId,
        laneWitnessIds, exact: true,
      };
    }
  }
  witnessRecords.sort((left, right) => left.witnessId.localeCompare(right.witnessId));
  const witnesses = Object.fromEntries(witnessRecords.map((record) => [record.witnessId, record]));
  return {
    scenarios,
    evidence: {
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-output-byte-witnesses',
      pass: true,
      encoding: 'base64-exact-bytes',
      channels: ['color', 'depth', 'objectId'],
      expectedChallengedCallbackCount: 126,
      challengedCallbackCount: 126,
      expectedObservationCount: 384,
      observationCount: 384,
      witnessCount: Object.keys(witnesses).length,
      witnesses,
      observations,
      callbackReferences: references,
      objectIdBaselines,
    },
  };
}

test('raw output fixture covers all 126 callbacks and rejects alpha/hash and all-clear tampering', () => {
  const fixture = outputWitnessFixture();
  assert.deepEqual(validateThreeImmediatePhase0OutputWitnesses(
    fixture.evidence, fixture.scenarios,
  ), { valid: true, reasons: [] });

  const alphaTampered = structuredClone(fixture);
  const objectId = Object.keys(alphaTampered.evidence.witnesses).find(
    (id) => id.startsWith('objectId/'),
  );
  const objectBytes = Buffer.from(
    alphaTampered.evidence.witnesses[objectId].bytesBase64, 'base64',
  );
  objectBytes[7] = 254;
  alphaTampered.evidence.witnesses[objectId].bytesBase64 = objectBytes.toString('base64');
  assert.equal(validateThreeImmediatePhase0OutputWitnesses(
    alphaTampered.evidence, alphaTampered.scenarios,
  ).valid, false);

  const clearTampered = structuredClone(fixture);
  const colorId = Object.keys(clearTampered.evidence.witnesses).find(
    (id) => id.startsWith('color/'),
  );
  clearTampered.evidence.witnesses[colorId].bytesBase64 = Buffer.alloc(TARGET_BYTES)
    .toString('base64');
  assert.equal(validateThreeImmediatePhase0OutputWitnesses(
    clearTampered.evidence, clearTampered.scenarios,
  ).valid, false);
});
