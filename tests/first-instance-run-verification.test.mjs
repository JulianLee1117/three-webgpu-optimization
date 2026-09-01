import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {
  summarizeInput,
  validateBenchmarkPlan,
  validateProtocolMatrix,
} from '../analysis/summarize.mjs';
import { summarizeFirstInstanceCrossoverRows } from '../analysis/first-instance-crossover-summary.mjs';
import {
  FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
  firstInstanceCrossoverScheduleSha256,
  firstInstanceRenderParityIdentity,
  firstInstanceValidationSemanticSha256,
  validateFirstInstanceTrialEvidence,
} from '../scripts/first-instance-evidence-validation.mjs';
import {
  FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
  firstInstanceCrossoverFrame,
} from '../src/benchmark/first-instance-crossover-schedule.js';
import {
  FIRST_INSTANCE_CROSSOVER_LANES,
  FIRST_INSTANCE_CROSSOVER_MODE,
  buildFirstInstanceCrossoverPlan,
} from '../src/benchmark/plan.js';
import { createFirstInstanceShaderEvidence } from '../src/validation/first-instance-shader-evidence.js';
import {
  createNvidiaTelemetryCoverageAudit,
  NVIDIA_QUERY_FIELDS,
  TELEMETRY_CSV_FIELDS,
  summarizeTelemetryRows,
} from '../scripts/nvidia-telemetry.mjs';

const [PORTABLE, FEATURE] = FIRST_INSTANCE_CROSSOVER_LANES;
const OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;
const RECORD_BYTES = 20;
const RUN_ID = 'first-instance-render-only-o65536-b32-test';
const SCENARIO_SEED = 123;
const REQUIRED_ARTIFACTS = [
  'frames.csv',
  'metadata.json',
  'trial-summaries.json',
  'validation-artifacts.json',
  'workload-manifests.json',
  'gpu-telemetry-summary.json',
];

function sha(value) {
  return createHash('sha256').update(value).digest('hex');
}

function shaJson(value) {
  return sha(JSON.stringify(value));
}

function jsonBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
}

function fnv1a64Text(value) {
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(value)) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, '0');
}

function protocol() {
  return {
    matrix: 'first-instance-render-only-o65536-b32',
    matrixKind: 'first-instance-render-only',
    representationScaleRole: null,
    layouts: ['baseline'],
    depthBinCount: null,
    reversedDepthBuffer: true,
    minimumStorageBuffersPerShaderStage: null,
    renderParity: 'preflight, timing-start, and postflight paired portable/feature exact validation plus two stable offscreen captures per lane of rgba8 color, depth32float, and encoded object ID',
    firstInstanceCrossover: {
      requiredFeature: 'indirect-first-instance',
      lanes: [...FIRST_INSTANCE_CROSSOVER_LANES],
      blockSize: 8,
      warmupBlocks: 40,
      measuredBlocks: 60,
      patterns: ['PFPFFPFP', 'FPFPPFPF'],
      expectedMeasuredRowsPerLane: 240,
      expectedRenderCallsPerFrame: 1,
      expectedRenderTimestampUidCount: 1,
      expectedComputeTimestampsPerFrame: 0,
      commandSegments: 2,
      commandRecordsPerSegment: BUCKET_COUNT,
      scheduleSha256ByOrientation: {
        0: firstInstanceCrossoverScheduleSha256(0),
        1: firstInstanceCrossoverScheduleSha256(1),
      },
    },
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    allowedObjectCounts: [4096, 16_384, OBJECT_COUNT],
    allowedBucketCounts: [1, 4, 32, 128],
    allowedHeterogeneousComparators: ['coalesced-v11', 'historical-v10'],
    heterogeneousComparator: null,
    modes: [FIRST_INSTANCE_CROSSOVER_MODE],
    visibilityLevels: [0.99, 0.2],
    repetitions: 12,
    warmupFrames: 320,
    measuredFrames: 480,
    maximumCpuTimerQuantumMs: 0.01,
    ordering: 'twelve-repetition-two-visibility-eight-frame-crossover-with-pairwise-balanced-command-segment-visibility-order-and-starting-orientation',
    threeBlocksScheduling: null,
  };
}

function plan() {
  return buildFirstInstanceCrossoverPlan({
    runId: RUN_ID,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  }).map((trial) => ({ ...trial, runId: RUN_ID }));
}

function segmentMap(order) {
  return Object.fromEntries(FIRST_INSTANCE_CROSSOVER_LANES.map((lane) => {
    const index = order.indexOf(lane);
    return [lane, {
      index,
      recordBase: index * BUCKET_COUNT,
      byteBase: index * BUCKET_COUNT * RECORD_BYTES,
    }];
  }));
}

function visibleCounts(visibility) {
  const expected = Math.round(OBJECT_COUNT * visibility);
  const base = Math.floor(expected / BUCKET_COUNT);
  const result = new Array(BUCKET_COUNT).fill(base);
  for (let index = 0; index < expected - base * BUCKET_COUNT; index += 1) {
    result[index] += 1;
  }
  return result;
}

function timedVertexShader(feature) {
  const matrixName = feature ? 'NodeBuffer_200' : 'NodeBuffer_100';
  const visibleName = feature ? 'NodeBuffer_201' : 'NodeBuffer_101';
  const inputs = feature
    ? `fn main( @builtin( instance_index ) instanceIndex : u32,
  @location( 0 ) position : vec3<f32>,
  @location( 1 ) normal : vec3<f32> ) -> VaryingsStruct {`
    : `fn main( @builtin( instance_index ) instanceIndex : u32,
  @location( 0 ) position : vec3<f32>,
  @location( 1 ) bucketBase : u32,
  @location( 2 ) normal : vec3<f32> ) -> VaryingsStruct {`;
  const address = feature ? 'instanceIndex' : '( bucketBase + instanceIndex )';
  return `
struct ${matrixName}Struct { value : array< mat4x4<f32> > };
@binding( 3 ) @group( 1 ) var<storage, read> ${matrixName} : ${matrixName}Struct;
struct ${visibleName}Struct { value : array< u32 > };
@binding( 4 ) @group( 1 ) var<storage, read> ${visibleName} : ${visibleName}Struct;
struct VaryingsStruct { @builtin( position ) position : vec4<f32> };
@vertex
${inputs}
  var objectMatrix : mat4x4<f32>;
  objectMatrix = ${matrixName}.value[ ${visibleName}.value[ ${address} ] ];
  var transformed : vec3<f32> = ( objectMatrix * vec4<f32>( position, 1.0 ) ).xyz;
  var varyings : VaryingsStruct;
  varyings.position = vec4<f32>( transformed + normal * 0.0, 1.0 );
  return varyings;
}`;
}

const TIMED_FRAGMENT_SHADER = `
@fragment fn main() -> @location( 0 ) vec4<f32> {
  return vec4<f32>( 0.4, 0.65, 0.95, 1.0 );
}`;

function timedVertexInputs(feature) {
  const position = {
    name: 'position', shaderLocation: 0, format: 'float32x3', stepMode: 'vertex',
    arrayType: 'Float32Array', itemSize: 3, normalized: false, count: 30,
    resourceId: 700,
  };
  const normal = {
    name: 'normal', shaderLocation: feature ? 1 : 2, format: 'float32x3',
    stepMode: 'vertex', arrayType: 'Float32Array', itemSize: 3,
    normalized: false, count: 30, resourceId: 701,
  };
  return feature ? [position, normal] : [position, {
    name: 'bucketBase', shaderLocation: 1, format: 'uint32', stepMode: 'vertex',
    arrayType: 'Uint32Array', itemSize: 1, normalized: false, count: 30,
    resourceId: 702,
  }, normal];
}

function timedStorageBindings() {
  return [
    {
      semantic: 'matrix', group: 1, binding: 3, access: 'read', visibility: 'vertex',
      elementType: 'mat4x4<f32>', count: OBJECT_COUNT, byteLength: OBJECT_COUNT * 64,
      resourceId: 'matrix-resource',
    },
    {
      semantic: 'visibleIds', group: 1, binding: 4, access: 'read', visibility: 'vertex',
      elementType: 'u32', count: OBJECT_COUNT, byteLength: OBJECT_COUNT * 4,
      resourceId: 'visible-ids-resource',
    },
  ];
}

let shaderEvidencePromise;
async function timedShaderEvidence() {
  shaderEvidencePromise ??= (async () => {
    const sources = {
      [PORTABLE]: {
        vertexShader: timedVertexShader(false), fragmentShader: TIMED_FRAGMENT_SHADER,
        vertexInputs: timedVertexInputs(false), storageBindings: timedStorageBindings(),
      },
      [FEATURE]: {
        vertexShader: timedVertexShader(true), fragmentShader: TIMED_FRAGMENT_SHADER,
        vertexInputs: timedVertexInputs(true), storageBindings: timedStorageBindings(),
      },
    };
    const evidence = await createFirstInstanceShaderEvidence({
      portable: sources[PORTABLE], feature: sources[FEATURE],
    });
    assert.equal(evidence.pass, true, evidence.reasons.join('\n'));
    return {
      ...evidence,
      captureApi: 'renderer.debug.getShaderAsync(scene, camera, mesh)',
      rawSources: Object.fromEntries(FIRST_INSTANCE_CROSSOVER_LANES.map((lane) => [lane, {
        vertexShader: sources[lane].vertexShader,
        fragmentShader: sources[lane].fragmentShader,
      }])),
    };
  })();
  return structuredClone(await shaderEvidencePromise);
}

function challengeVertexShader(lane) {
  const portable = lane === PORTABLE;
  const prefix = portable ? 'portable' : 'feature';
  const storageStruct = portable ? 'PortableVisibleIds' : 'FeatureVisibleIds';
  const storageVariable = portable ? 'portableVisibleIds' : 'featureVisibleIds';
  const address = portable ? '( bucketBase + instanceIndex )' : 'instanceIndex';
  const inputs = portable
    ? `@builtin( instance_index ) instanceIndex : u32,
  @location( 0 ) bucketBase : u32,
  @location( 1 ) position : vec3<f32>`
    : `@builtin( instance_index ) instanceIndex : u32,
  @location( 0 ) position : vec3<f32>`;
  return `
struct ${storageStruct} { value : array< u32 > };
@binding( 4 ) @group( 1 ) var<storage, read> ${storageVariable} : ${storageStruct};
struct VaryingsStruct {
  @builtin( position ) position : vec4<f32>,
  @location( 0 ) @interpolate( flat, either ) vAddressChallengeObjectId : u32
};
@vertex fn main( ${inputs} ) -> VaryingsStruct {
  var ${prefix}ChallengeAddress : u32;
  ${prefix}ChallengeAddress = ${address};
  let ${prefix}ChallengeObjectId : u32 = ${storageVariable}.value[ ${prefix}ChallengeAddress ];
  let pixelX : u32 = ${prefix}ChallengeAddress % 256u;
  let pixelY : u32 = ${prefix}ChallengeAddress / 256u;
  var output : VaryingsStruct;
  output.position = vec4<f32>( vec3<f32>( position.xy + vec2<f32>( f32( pixelX ), f32( pixelY ) ) * 0.0, position.z ), 1.0 );
  output.vAddressChallengeObjectId = ${prefix}ChallengeObjectId;
  return output;
}`;
}

const CHALLENGE_FRAGMENT_SHADER = `
struct OutputStruct { @location( 0 ) color : vec4<f32> };
var<private> output : OutputStruct;
var<private> encoded : u32;
@fragment fn main( @location( 0 ) @interpolate( flat, either ) vAddressChallengeObjectId : u32 ) -> OutputStruct {
  encoded = vAddressChallengeObjectId + 1u;
  output.color = vec4<f32>( f32( encoded & 255u ) / 255.0, f32( ( encoded >> 8u ) & 255u ) / 255.0, f32( ( encoded >> 16u ) & 255u ) / 255.0, 1.0 );
  return output;
}`;

function challengeShader(lane) {
  const vertexShader = challengeVertexShader(lane);
  const position = {
    name: 'position', shaderLocation: lane === PORTABLE ? 1 : 0,
    format: 'float32x3', stepMode: 'vertex', arrayType: 'Float32Array',
    itemSize: 3, normalized: false, count: 30,
    resourceId: 'challenge-position-resource',
  };
  return {
    vertexSha256: sha(vertexShader),
    fragmentSha256: sha(CHALLENGE_FRAGMENT_SHADER),
    vertexLines: [], fragmentLines: [],
    rawSources: { vertexShader, fragmentShader: CHALLENGE_FRAGMENT_SHADER },
    vertexInputs: lane === PORTABLE ? [{
      name: 'bucketBase', shaderLocation: 0, format: 'uint32', stepMode: 'vertex',
      arrayType: 'Uint32Array', itemSize: 1, normalized: false, count: 30,
      resourceId: 'challenge-bucket-base-resource',
    }, position] : [position],
    storageBindings: [{
      semantic: 'visibleIds', group: 1, binding: 4, access: 'read', visibility: 'vertex',
      elementType: 'u32', count: OBJECT_COUNT, byteLength: OBJECT_COUNT * 4,
      resourceId: 'visible-ids-resource',
    }],
  };
}

function attributeEvidence(name, id) {
  return {
    name,
    id,
    version: 0,
    arrayType: name === 'bucketBase' || name === 'index'
      ? 'Uint32Array'
      : 'Float32Array',
    itemSize: name === 'uv' ? 2 : name === 'bucketBase' || name === 'index' ? 1 : 3,
    count: 30,
    normalized: false,
    gpuType: null,
    byteLength: 120,
    isInstancedBufferAttribute: false,
    meshPerAttribute: null,
    sha256: sha(name),
  };
}

function geometryEvidence() {
  const commonAttributes = Object.fromEntries(
    ['normal', 'position', 'uv'].map((name, index) => {
      const attribute = attributeEvidence(name, 10 + index);
      return [name, {
        sameObject: true,
        portable: { ...attribute },
        feature: { ...attribute },
      }];
    }),
  );
  const index = attributeEvidence('index', 20);
  return {
    schemaVersion: 1,
    kind: 'shared-first-instance-geometry-evidence',
    pass: true,
    portableAttributeNames: ['bucketBase', 'normal', 'position', 'uv'],
    featureAttributeNames: ['normal', 'position', 'uv'],
    sharedIndex: { sameObject: true, portable: { ...index }, feature: { ...index } },
    commonAttributes,
    bucketBase: attributeEvidence('bucketBase', 21),
    bucketBaseMismatchCount: 0,
    noInstanceSteppedAttributes: true,
  };
}

function membershipEvidence(visibility) {
  const counts = visibleCounts(visibility);
  const count = counts.reduce((sum, value) => sum + value, 0);
  const digest = visibility === 0.99 ? '1'.repeat(64) : '2'.repeat(64);
  return {
    membership: {
      pass: true,
      expectedCount: count,
      listedCount: count,
      duplicateIds: 0,
      outOfRangeIds: 0,
      wrongBucketIds: 0,
      listedHiddenIds: 0,
      missingVisibleIds: 0,
      overflow: 0,
      errors: 0,
    },
    membershipDigests: {
      schemaVersion: 1,
      hashAlgorithm: 'sha256',
      encoding: 'sorted-uint32-little-endian',
      pass: true,
      invalidExpectedIds: 0,
      truncatedActualIds: 0,
      expected: { count, sha256: digest },
      actual: { count, sha256: digest },
      perBucket: counts.map((bucketCount, bucket) => {
        const bucketDigest = ((bucket % 15) + 1).toString(16).repeat(64);
        return {
          bucket,
          match: true,
          expected: { count: bucketCount, sha256: bucketDigest },
          actual: {
            count: bucketCount,
            declaredCount: bucketCount,
            sha256: bucketDigest,
          },
        };
      }),
    },
  };
}

function commandRecords(lane, visibility) {
  const counts = visibleCounts(visibility);
  let firstIndex = 0;
  return counts.map((instanceCount, bucket) => {
    const indexCount = 12 + bucket * 3;
    const command = {
      indexCount,
      instanceCount,
      firstIndex,
      baseVertex: 0,
      firstInstance: lane === PORTABLE ? 0 : bucket * (OBJECT_COUNT / BUCKET_COUNT),
    };
    firstIndex += indexCount;
    return { bucket, actual: { ...command }, expected: { ...command } };
  });
}

function packingEvidence(order, visibility) {
  const count = Math.round(OBJECT_COUNT * visibility);
  const laneCommands = Object.fromEntries(FIRST_INSTANCE_CROSSOVER_LANES.map((lane) => [
    lane,
    {
      pass: true,
      errors: [],
      commandCount: BUCKET_COUNT,
      totalInstanceCount: count,
      records: commandRecords(lane, visibility),
    },
  ]));
  return {
    schemaVersion: 1,
    kind: 'first-instance-crossover-exact-frozen-packing-validation',
    pass: true,
    metadata: {
      pass: true,
      errors: [],
      laneCommandSegmentOrder: [...order],
      recordsPerLane: BUCKET_COUNT,
      commandSegmentByteLength: BUCKET_COUNT * RECORD_BYTES,
    },
    visibleIds: {
      pass: true,
      exactExpectedPacking: true,
      elementCount: OBJECT_COUNT,
      sha256: visibility === 0.99 ? '4'.repeat(64) : '5'.repeat(64),
    },
    padding: {
      pass: true,
      sentinel: 0xffff_ffff,
      paddingCount: OBJECT_COUNT - count,
      paddingSentinelCount: OBJECT_COUNT - count,
      activeSentinelCount: 0,
      corruptPaddingAddresses: [],
    },
    commands: {
      pass: true,
      exactExpectedPhysicalPacking: true,
      pairPass: true,
      lanes: laneCommands,
    },
    commitments: {
      schemaVersion: 1,
      hashAlgorithm: 'sha256',
      encoding: 'uint32-little-endian',
      visibleIdsSha256: visibility === 0.99 ? '4'.repeat(64) : '5'.repeat(64),
      physicalCommandsSha256: '6'.repeat(64),
      logicalPairSha256: visibility === 0.99 ? '7'.repeat(64) : '8'.repeat(64),
      paddingSha256: '9'.repeat(64),
      commandCoresEqual: true,
      lanes: {
        [PORTABLE]: { commandsSha256: 'a'.repeat(64), coreSha256: 'c'.repeat(64) },
        [FEATURE]: { commandsSha256: 'b'.repeat(64), coreSha256: 'c'.repeat(64) },
      },
      pairs: Array.from({ length: BUCKET_COUNT }, (_, bucket) => ({
        bucket,
        coreEqual: true,
        portableFirstInstance: 0,
        featureFirstInstance: bucket * (OBJECT_COUNT / BUCKET_COUNT),
        expectedFeatureFirstInstance: bucket * (OBJECT_COUNT / BUCKET_COUNT),
        sha256: ((bucket % 13) + 1).toString(16).repeat(64),
      })),
    },
  };
}

function targetEvidence() {
  return {
    width: 256,
    height: 256,
    pixelCount: OBJECT_COUNT,
    pass: true,
    configuredFormat: 'RGBAFormat/UnsignedByteType',
    backendFormat: 'rgba8unorm',
    readbackArrayType: 'Uint8Array',
    bytesPerPixel: 4,
    bytesPerRow: 1024,
    rowAlignmentBytes: 256,
    origin: 'top-left',
    sampleLocation: 'integer-plus-half',
    samples: 0,
    depthBuffer: false,
    stencilBuffer: false,
    scissorTest: false,
    viewportExact: true,
    scissorExact: true,
    colorSpace: 'none',
    flipY: false,
    generateMipmaps: false,
  };
}

function addressChallenge(lane, segments, visibility) {
  const expectedVisibleCount = Math.round(OBJECT_COUNT * visibility);
  const resetDigest = sha(Buffer.alloc(OBJECT_COUNT * 4));
  return {
    schemaVersion: 1,
    kind: 'render-target-all-address-challenge',
    pass: true,
    lane,
    outputStage: 'fragment',
    addressTransport: 'vertex-address-to-rgba8-pixel',
    encoding: 'rgb24-object-id-plus-one-transparent-zero-background',
    target: targetEvidence(),
    shader: challengeShader(lane),
    commandSegment: { ...segments[lane] },
    reset: {
      pass: true,
      pixelCount: OBJECT_COUNT,
      addressCount: OBJECT_COUNT,
      byteLength: OBJECT_COUNT * 4,
      sha256: resetDigest,
      expectedSha256: resetDigest,
    },
    pixelCount: OBJECT_COUNT,
    addressCount: OBJECT_COUNT,
    byteLength: OBJECT_COUNT * 4,
    sha256: visibility === 0.99 ? 'd'.repeat(64) : 'e'.repeat(64),
    expectedSha256: visibility === 0.99 ? 'd'.repeat(64) : 'e'.repeat(64),
    exactExpectedBytes: true,
    activeAddressCount: expectedVisibleCount,
    paddingAddressCount: OBJECT_COUNT - expectedVisibleCount,
    targetPaddingPixelCount: 0,
    activeMismatchCount: 0,
    paddingMismatchCount: 0,
    targetPaddingMismatchCount: 0,
    coverage: {
      allBucketsActive: true,
      nonzeroBucketBaseCount: BUCKET_COUNT - 1,
      activeRowCount: Math.ceil(expectedVisibleCount / 256),
      nonzeroEncodedChannelPixelCounts: { red: 100, green: 100, blue: 1 },
      alphaEncodedPixelCount: expectedVisibleCount,
    },
  };
}

function lifecycleEvidence(order, segments, shaderEvidence) {
  return {
    kind: 'first-instance-crossover-static-resource-lifecycle',
    bundlesPrimed: true,
    bundleStaticFlags: { [PORTABLE]: true, [FEATURE]: true },
    allBundlesStatic: true,
    bundleCount: 2,
    meshCount: 2,
    activeRenderObjectCount: 1,
    activeLane: PORTABLE,
    laneSelectionSerial: 8,
    laneCommandSegmentOrder: [...order],
    commandSegments: structuredClone(segments),
    rootUuid: 'root',
    rootVersion: null,
    bundleUuids: { [PORTABLE]: 'portable-bundle', [FEATURE]: 'feature-bundle' },
    bundleVersions: { [PORTABLE]: 0, [FEATURE]: 0 },
    bundleRecordCounts: { [PORTABLE]: 1, [FEATURE]: 1 },
    meshUuids: { [PORTABLE]: 'portable-mesh', [FEATURE]: 'feature-mesh' },
    geometryUuids: { [PORTABLE]: 'portable-geometry', [FEATURE]: 'feature-geometry' },
    materialUuids: { [PORTABLE]: 'portable-material', [FEATURE]: 'feature-material' },
    materialVersions: { [PORTABLE]: 0, [FEATURE]: 0 },
    commonAttributeIds: { normal: 10, position: 11, uv: 12 },
    commonAttributeVersions: { normal: 0, position: 0, uv: 0 },
    indexAttributeId: 20,
    indexAttributeVersion: 0,
    bucketBaseAttributeId: 21,
    bucketBaseAttributeVersion: 0,
    matrixAttributeId: 30,
    matrixAttributeVersion: 0,
    visibleIdsAttributeId: 31,
    visibleIdsAttributeVersion: 0,
    indirectAttributeId: 32,
    indirectAttributeVersion: 0,
    shaderEvidence: {
      pass: true,
      portableVertexSha256: shaderEvidence.portable.raw.vertex.sha256,
      featureVertexSha256: shaderEvidence.feature.raw.vertex.sha256,
      normalizedVertexSha256: shaderEvidence.portable.normalizedVertex.sha256,
      fragmentSha256: shaderEvidence.portable.raw.fragment.sha256,
      storageBindings: {
        [PORTABLE]: structuredClone(shaderEvidence.portable.storageBindings),
        [FEATURE]: structuredClone(shaderEvidence.feature.storageBindings),
      },
    },
    configuredComputeDispatches: 0,
    configuredComputeSubmissions: 0,
  };
}

async function exactValidation(spec) {
  const segments = segmentMap(spec.laneCommandSegmentOrder);
  const shaderEvidence = await timedShaderEvidence();
  const lifecycle = lifecycleEvidence(spec.laneCommandSegmentOrder, segments, shaderEvidence);
  return {
    schemaVersion: 1,
    kind: 'first-instance-crossover-exact-paired-snapshots',
    pass: true,
    expectedIdsMatchScenario: true,
    frozenPacking: packingEvidence(spec.laneCommandSegmentOrder, spec.visibilityFraction),
    ...membershipEvidence(spec.visibilityFraction),
    geometry: geometryEvidence(),
    addressChallenges: {
      pass: true,
      byteIdentical: true,
      geometry: {
        schemaVersion: 1,
        kind: 'fragment-address-challenge-geometry-evidence',
        pass: true,
        topology: 'triangle-list',
        pixelLocalCoordinates: true,
        target: { width: 256, height: 256, pixelCount: OBJECT_COUNT },
        indexCount: BUCKET_COUNT * 3,
        addressedTriangleCount: BUCKET_COUNT,
        degenerateTriangleCount: 0,
        addressedTrianglesPerSubmittedInstance: 1,
        positionMismatchCount: 0,
        bucketBaseMismatchCount: 0,
        indexMismatchCount: 0,
        attributesExact: true,
        sharedPayloadExact: true,
        indirectIdentityExact: true,
        laneOffsetsExact: { [PORTABLE]: true, [FEATURE]: true },
        expectedInstanceCount: OBJECT_COUNT / BUCKET_COUNT,
      },
      lanes: {
        [PORTABLE]: addressChallenge(
          PORTABLE,
          segments,
          spec.visibilityFraction,
        ),
        [FEATURE]: addressChallenge(
          FEATURE,
          segments,
          spec.visibilityFraction,
        ),
      },
    },
    shaderEvidence,
    lifecycle,
  };
}

function parityLane() {
  const color = {
    format: 'rgba8unorm', arrayType: 'Uint8Array',
    byteLength: 1280 * 720 * 4, sha256: '3'.repeat(64),
  };
  const depth = {
    format: 'depth32float', arrayType: 'Float32Array',
    byteLength: 1280 * 720 * 4, sha256: '4'.repeat(64),
  };
  const objectId = {
    format: 'rgba8unorm-object-id-plus-one', arrayType: 'Uint8Array',
    byteLength: 1280 * 720 * 4, sha256: '5'.repeat(64),
  };
  return {
    schemaVersion: 1,
    kind: 'fixed-camera-offscreen-exact-render-parity',
    pass: true,
    width: 1280,
    height: 720,
    captures: 2,
    material: { type: 'MeshStandardNodeMaterial' },
    color,
    depth,
    objectId,
    objectIdValidation: {
      pass: true,
      encoding: 'rgb24-object-id-plus-one-zero-background',
      backgroundPixels: 920_600,
      coveredPixels: 1_000,
      outOfRangePixels: 0,
      nonVisiblePixels: 0,
    },
    reversedDepthBuffer: true,
    stability: {
      pass: true,
      firstCapture: {
        color: { ...color }, depth: { ...depth }, objectId: { ...objectId },
      },
      first: {
        colorSha256: color.sha256,
        depthSha256: depth.sha256,
        objectIdSha256: objectId.sha256,
      },
    },
  };
}

function renderParity(validation) {
  return {
    schemaVersion: 1,
    kind: 'first-instance-crossover-exact-render-parity',
    pass: true,
    laneIds: [...FIRST_INSTANCE_CROSSOVER_LANES],
    crossLaneExact: true,
    lanes: { [PORTABLE]: parityLane(), [FEATURE]: parityLane() },
    snapshotValidation: structuredClone(validation),
  };
}

function completionInvariant(spec, validation, serialBase) {
  const staticLifecycle = structuredClone(validation.lifecycle);
  delete staticLifecycle.activeLane;
  delete staticLifecycle.laneSelectionSerial;
  const lifecycleCommitment = fnv1a64Text(JSON.stringify(staticLifecycle));
  const segments = segmentMap(spec.laneCommandSegmentOrder);
  return {
    pass: true,
    kind: 'first-instance-crossover-static-resource-invariant',
    bundlesPrimed: true,
    bundleStaticFlags: { [PORTABLE]: true, [FEATURE]: true },
    allBundlesStatic: true,
    bundleCount: 2,
    meshCount: 2,
    activeRenderObjectCount: 1,
    bundleRecordCounts: { [PORTABLE]: 1, [FEATURE]: 1 },
    shaderEvidencePass: true,
    plannedLaneCommandSegmentOrder: spec.laneCommandSegmentOrder.join('|'),
    observedLaneCommandSegmentOrder: spec.laneCommandSegmentOrder.join('|'),
    configuredCommandOrderExact: true,
    plannedCommandSegments: JSON.stringify(segments),
    observedCommandSegments: structuredClone(segments),
    commandSegmentsExact: true,
    representation: {
      kind: 'frozen-first-instance-addressing-crossover',
      activeRenderObjectCount: 1,
      geometryIdentityCount: 2,
      materialIdentityCount: 2,
      commonIndexIdentityCount: 1,
      commandRecordCount: BUCKET_COUNT * 2,
      visibleIdsCount: OBJECT_COUNT,
    },
    lifecycleExact: true,
    staticLifecycleAtTimingStart: structuredClone(staticLifecycle),
    staticLifecycleAtTimingEnd: structuredClone(staticLifecycle),
    lifecycleCommitmentAtTimingStart: lifecycleCommitment,
    lifecycleCommitmentAtTimingEnd: lifecycleCommitment,
    selectorWriteSerialAtTimingStart: serialBase + 10,
    selectorWriteSerialAtTimingEnd:
      serialBase + 10 + FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
    selectorWritesDuringTiming: FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
    strategySelectionSerialAtTimingStart: serialBase + 20,
    strategySelectionSerialAtTimingEnd:
      serialBase + 20 + FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
    strategySelectionsDuringTiming: FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
    renderCallSerialAtTimingStart: serialBase + 30,
    renderCallSerialAtTimingEnd:
      serialBase + 30 + FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
    renderCallsDuringTiming: FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
    computeCallSerialAtTimingStart: serialBase + 5,
    computeCallSerialAtTimingEnd: serialBase + 5,
    computeCallsDuringTiming: 0,
    renderTargetTextureUuidAtTimingStart: 'timed-target',
    renderTargetTextureUuidAtTimingEnd: 'timed-target',
    renderTargetWidthAtTimingStart: 1280,
    renderTargetWidthAtTimingEnd: 1280,
    renderTargetHeightAtTimingStart: 720,
    renderTargetHeightAtTimingEnd: 720,
    renderTargetSamplesAtTimingStart: 0,
    renderTargetSamplesAtTimingEnd: 0,
    renderTargetDepthBufferAtTimingStart: true,
    renderTargetDepthBufferAtTimingEnd: true,
    cameraViewFnv64AtTimingStart: '0123456789abcdef',
    cameraViewFnv64AtTimingEnd: '0123456789abcdef',
    cameraProjectionFnv64AtTimingStart: 'fedcba9876543210',
    cameraProjectionFnv64AtTimingEnd: 'fedcba9876543210',
    totalPipelineCacheEntriesAtTimingStart: 12,
    totalPipelineCacheEntriesAtTimingEnd: 12,
    computePipelineCacheEntriesAtTimingStart: 2,
    computePipelineCacheEntriesAtTimingEnd: 2,
    expectedTimedFrameCount: FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
  };
}

function retainedRows(spec, validation, invariant) {
  const segments = segmentMap(spec.laneCommandSegmentOrder);
  const scheduleSha256 = firstInstanceCrossoverScheduleSha256(
    spec.superblockOrientationOffset,
  );
  const expectedVisibleCount = Math.round(OBJECT_COUNT * spec.visibilityFraction);
  return Array.from({ length: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES }, (_, index) => {
    const scheduled = firstInstanceCrossoverFrame(index, spec.superblockOrientationOffset);
    const segment = segments[scheduled.laneId];
    const ordinal = FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES + index + 1;
    const gpuRenderMs = scheduled.laneId === FEATURE
      ? spec.visibilityFraction === 0.99 ? 0.18 : 0.2
      : 0.22;
    return {
      schemaVersion: 2,
      runId: spec.runId,
      trialId: spec.trialId,
      planIndex: spec.planIndex,
      repetitionIndex: spec.repetitionIndex,
      modeOrderPosition: spec.modeOrderPosition,
      visibilityOrderPosition: spec.visibilityOrderPosition,
      layoutOrderPosition: spec.layoutOrderPosition,
      plannedModeOrder: spec.modeOrder.join('|'),
      plannedVisibilityOrder: spec.visibilityOrder.join('|'),
      plannedLayoutOrder: spec.layoutOrder.join('|'),
      frameIndex: index,
      phaseFrameIndex: index,
      modeId: spec.modeId,
      objectCount: spec.objectCount,
      bucketCount: spec.bucketCount,
      targetVisibilityFraction: spec.visibilityFraction,
      scenarioLayout: spec.layout,
      expectedVisibleCount,
      protocolWarmupFrames: FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
      protocolMeasuredFrames: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
      validationKind: validation.kind,
      validationPass: true,
      usesCompute: false,
      configuredDrawCommands: BUCKET_COUNT,
      configuredRenderObjects: 1,
      configuredComputeDispatches: 0,
      configuredComputeSubmissions: 0,
      configuredSubmittedInstances: expectedVisibleCount,
      timestampAvailable: true,
      expectedRenderTimestampUidCount: 1,
      plannedLaneCommandSegmentOrder: spec.laneCommandSegmentOrder.join('|'),
      plannedCommandSegments: JSON.stringify(segments),
      plannedScheduleSha256: scheduleSha256,
      superblockOrientationOffset: spec.superblockOrientationOffset,
      lifecycleCommitmentAtTimingStart: invariant.lifecycleCommitmentAtTimingStart,
      rootUuidAtTimingStart: validation.lifecycle.rootUuid,
      rootVersionAtTimingStart: validation.lifecycle.rootVersion,
      bundleUuidsAtTimingStart: FIRST_INSTANCE_CROSSOVER_LANES.map(
        (lane) => validation.lifecycle.bundleUuids[lane],
      ).join('|'),
      bundleVersionsAtTimingStart: FIRST_INSTANCE_CROSSOVER_LANES.map(
        (lane) => validation.lifecycle.bundleVersions[lane],
      ).join('|'),
      meshUuidsAtTimingStart: FIRST_INSTANCE_CROSSOVER_LANES.map(
        (lane) => validation.lifecycle.meshUuids[lane],
      ).join('|'),
      geometryUuidsAtTimingStart: FIRST_INSTANCE_CROSSOVER_LANES.map(
        (lane) => validation.lifecycle.geometryUuids[lane],
      ).join('|'),
      materialUuidsAtTimingStart: FIRST_INSTANCE_CROSSOVER_LANES.map(
        (lane) => validation.lifecycle.materialUuids[lane],
      ).join('|'),
      commonAttributeIdsAtTimingStart:
        JSON.stringify(validation.lifecycle.commonAttributeIds),
      commonAttributeVersionsAtTimingStart:
        JSON.stringify(validation.lifecycle.commonAttributeVersions),
      indexAttributeIdAtTimingStart: validation.lifecycle.indexAttributeId,
      indexAttributeVersionAtTimingStart: validation.lifecycle.indexAttributeVersion,
      bucketBaseAttributeIdAtTimingStart: validation.lifecycle.bucketBaseAttributeId,
      bucketBaseAttributeVersionAtTimingStart:
        validation.lifecycle.bucketBaseAttributeVersion,
      shaderPortableVertexSha256AtTimingStart:
        validation.lifecycle.shaderEvidence.portableVertexSha256,
      shaderFeatureVertexSha256AtTimingStart:
        validation.lifecycle.shaderEvidence.featureVertexSha256,
      shaderNormalizedVertexSha256AtTimingStart:
        validation.lifecycle.shaderEvidence.normalizedVertexSha256,
      shaderFragmentSha256AtTimingStart:
        validation.lifecycle.shaderEvidence.fragmentSha256,
      matrixAttributeIdAtTimingStart: validation.lifecycle.matrixAttributeId,
      matrixAttributeVersionAtTimingStart: validation.lifecycle.matrixAttributeVersion,
      visibleIdsAttributeIdAtTimingStart: validation.lifecycle.visibleIdsAttributeId,
      visibleIdsAttributeVersionAtTimingStart:
        validation.lifecycle.visibleIdsAttributeVersion,
      indirectAttributeIdAtTimingStart: validation.lifecycle.indirectAttributeId,
      indirectAttributeVersionAtTimingStart:
        validation.lifecycle.indirectAttributeVersion,
      computeCallSerialAtTimingStart: invariant.computeCallSerialAtTimingStart,
      selectorWriteSerialAtTimingStart: invariant.selectorWriteSerialAtTimingStart,
      strategySelectionSerialAtTimingStart:
        invariant.strategySelectionSerialAtTimingStart,
      renderCallSerialAtTimingStart: invariant.renderCallSerialAtTimingStart,
      renderTargetTextureUuidAtTimingStart:
        invariant.renderTargetTextureUuidAtTimingStart,
      renderTargetWidthAtTimingStart: invariant.renderTargetWidthAtTimingStart,
      renderTargetHeightAtTimingStart: invariant.renderTargetHeightAtTimingStart,
      renderTargetSamplesAtTimingStart: invariant.renderTargetSamplesAtTimingStart,
      renderTargetDepthBufferAtTimingStart:
        invariant.renderTargetDepthBufferAtTimingStart,
      cameraViewFnv64AtTimingStart: invariant.cameraViewFnv64AtTimingStart,
      cameraProjectionFnv64AtTimingStart:
        invariant.cameraProjectionFnv64AtTimingStart,
      totalPipelineCacheEntriesAtTimingStart:
        invariant.totalPipelineCacheEntriesAtTimingStart,
      computePipelineCacheEntriesAtTimingStart:
        invariant.computePipelineCacheEntriesAtTimingStart,
      crossoverBlockIndex: scheduled.crossoverBlockIndex,
      withinBlockPosition: scheduled.withinBlockPosition,
      crossoverPattern: scheduled.pattern,
      crossoverPatternIndex: scheduled.patternIndex,
      laneId: scheduled.laneId,
      commandSegmentIndex: segment.index,
      commandRecordBase: segment.recordBase,
      commandByteBase: segment.byteBase,
      selectorWriteSerial: invariant.selectorWriteSerialAtTimingStart + ordinal,
      strategySelectionSerial: invariant.strategySelectionSerialAtTimingStart + ordinal,
      renderCallSerial: invariant.renderCallSerialAtTimingStart + ordinal,
      gpuFrameId: spec.planIndex * 1_000 + index,
      gpuRenderTimestampUidCount: 1,
      cpuCommonUpdateMs: 0.01,
      cpuComputeSubmitMs: null,
      cpuRenderSubmitMs: 0.02,
      cpuSubmitTotalMs: 0.02,
      cpuFrameBodyMs: 0.04,
      gpuComputeMs: null,
      gpuRenderMs,
      gpuPassTotalMs: gpuRenderMs,
    };
  });
}

function geometryManifest() {
  const geometries = Array.from({ length: BUCKET_COUNT }, (_, bucket) => {
    const attribute = (name, itemSize) => ({
      arrayType: 'Float32Array',
      count: 30,
      itemSize,
      normalized: false,
      sha256: sha(`geometry-${bucket}-${name}`),
    });
    const canonical = {
      bucket,
      family: bucket % 4,
      name: `fixture-${bucket}`,
      attributes: {
        normal: attribute('normal', 3),
        position: attribute('position', 3),
        uv: attribute('uv', 2),
      },
      index: {
        arrayType: 'Uint32Array',
        count: 12 + bucket * 3,
        itemSize: 1,
        normalized: false,
        sha256: sha(`geometry-${bucket}-index`),
      },
      drawRange: { start: 0, count: 'Infinity' },
      boundingBox: { min: [-1, -1, -1], max: [1, 1, 1] },
      boundingSphere: { center: [0, 0, 0], radius: 1 },
    };
    return { ...canonical, sha256: shaJson(canonical) };
  });
  const canonical = {
    schemaVersion: 1,
    generator: 'createIndexedGeometryFixtures',
    tier: 'medium',
    bucketCount: BUCKET_COUNT,
    geometries,
  };
  return { ...canonical, sha256: shaJson(canonical) };
}

function scenarioManifest(visibility) {
  const expectedVisibleCount = Math.round(OBJECT_COUNT * visibility);
  const expectedVisibleIdsCanonicalSha256 = visibility === 0.99
    ? '1'.repeat(64)
    : '2'.repeat(64);
  const arrays = {
    bucketCounts: { arrayType: 'Uint32Array', length: BUCKET_COUNT, sha256: sha(`bucketCounts-${visibility}`) },
    bucketBases: { arrayType: 'Uint32Array', length: BUCKET_COUNT, sha256: sha(`bucketBases-${visibility}`) },
    visibleCounts: { arrayType: 'Uint32Array', length: BUCKET_COUNT, sha256: sha(`visibleCounts-${visibility}`) },
    objectBuckets: { arrayType: 'Uint32Array', length: OBJECT_COUNT, sha256: sha(`objectBuckets-${visibility}`) },
    matrices: { arrayType: 'Float32Array', length: OBJECT_COUNT * 16, sha256: sha(`matrices-${visibility}`) },
    bounds: { arrayType: 'Float32Array', length: OBJECT_COUNT * 4, sha256: sha(`bounds-${visibility}`) },
    expectedVisibleIds: {
      arrayType: 'Uint32Array',
      length: expectedVisibleCount,
      sha256: expectedVisibleIdsCanonicalSha256,
    },
    cullOrder: { arrayType: 'Uint32Array', length: OBJECT_COUNT, sha256: sha(`cullOrder-${visibility}`) },
  };
  const canonical = {
    schemaVersion: 1,
    generator: 'createFixedSubsetScenario',
    seed: SCENARIO_SEED,
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
    visibilityFraction: visibility,
    layout: 'baseline',
    depthBinRange: null,
    expectedVisibleCount,
    expectedVisibleIdsCanonicalSha256,
    arrays,
  };
  return { ...canonical, sha256: shaJson(canonical) };
}

function provenanceRecord() {
  return {
    status: 'available',
    commit: '1'.repeat(40),
    tree: '2'.repeat(40),
    ref: 'refs/heads/test',
    dirty: false,
    stagedChanges: 0,
    unstagedChanges: 0,
    untrackedFiles: 0,
    porcelainEntryCount: 0,
    porcelainByteCount: 0,
    porcelainSha256: sha(''),
    trackedFileCount: 100,
    trackedFilesSha256: '3'.repeat(64),
    packageLockTracked: true,
    packageLockSha256: '4'.repeat(64),
  };
}

function csvEscape(value) {
  if (value === null || value === undefined) return '';
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function rowsToCsv(rows) {
  const headers = Object.keys(rows[0]);
  return [
    headers.map(csvEscape).join(','),
    ...rows.map((row) => headers.map((field) => csvEscape(row[field])).join(',')),
  ].join('\n');
}

function percentile(rows, field, fraction) {
  const values = rows.map((row) => row[field]).filter(Number.isFinite)
    .sort((left, right) => left - right);
  if (values.length === 0) return null;
  return values[Math.min(
    values.length - 1,
    Math.max(0, Math.ceil(values.length * fraction) - 1),
  )];
}

function trialTiming(rows) {
  return {
    cpuCommonUpdateP50Ms: percentile(rows, 'cpuCommonUpdateMs', 0.5),
    cpuCommonUpdateP95Ms: percentile(rows, 'cpuCommonUpdateMs', 0.95),
    cpuFrameBodyP50Ms: percentile(rows, 'cpuFrameBodyMs', 0.5),
    cpuFrameBodyP95Ms: percentile(rows, 'cpuFrameBodyMs', 0.95),
    cpuSubmitP50Ms: percentile(rows, 'cpuSubmitTotalMs', 0.5),
    cpuSubmitP95Ms: percentile(rows, 'cpuSubmitTotalMs', 0.95),
    gpuComputeP50Ms: percentile(rows, 'gpuComputeMs', 0.5),
    gpuComputeP95Ms: percentile(rows, 'gpuComputeMs', 0.95),
    gpuRenderP50Ms: percentile(rows, 'gpuRenderMs', 0.5),
    gpuRenderP95Ms: percentile(rows, 'gpuRenderMs', 0.95),
    gpuPassTotalP50Ms: percentile(rows, 'gpuPassTotalMs', 0.5),
    gpuPassTotalP95Ms: percentile(rows, 'gpuPassTotalMs', 0.95),
  };
}

function unavailableTelemetryReport() {
  const processSnapshot = (label, runElapsedMs) => ({
    label,
    status: 'unavailable',
    capturedAtIso: '2026-08-31T20:00:01.000Z',
    runElapsedMs,
    reason: 'command-not-found',
    rawNonemptyLineCount: 0,
    parsedRecordCount: 0,
    malformedLineCount: 0,
    stdoutByteCount: 0,
    stdoutTruncated: false,
    stderrByteCount: 0,
    processes: [],
  });
  const rows = [];
  const collectorStartedRunElapsedMs = null;
  const collectorStopRequestedRunElapsedMs = null;
  return {
    provider: 'nvidia-smi',
    status: 'unavailable',
    reason: 'command-not-found',
    command: 'nvidia-smi',
    sampling: {
      processModel: 'one-long-lived-process',
      requestedIntervalMs: 250,
      queryFields: [...NVIDIA_QUERY_FIELDS],
      outputFile: 'gpu-telemetry.csv',
      collectorStartedRunElapsedMs,
      collectorStopRequestedRunElapsedMs,
      malformedLineCount: 0,
      stderrByteCount: 0,
      exit: null,
    },
    summary: {
      sampleCount: 0,
      gpuCount: 0,
      firstObservedAtIso: null,
      lastObservedAtIso: null,
      gpus: [],
    },
    coverageAudit: createNvidiaTelemetryCoverageAudit(rows, {
      collectorStartedRunElapsedMs,
      collectorStopRequestedRunElapsedMs,
    }),
    computeProcesses: {
      pre: processSnapshot('pre-run', 1),
      post: processSnapshot('post-run', 2),
    },
    acceptanceBoundary: {
      affectsTechnicalRunAcceptance: false,
      candidateEnvironmentReviewRequired: true,
      automaticPstateRejectionThreshold: null,
    },
  };
}

function telemetryRowsToCsv(rows) {
  return [
    TELEMETRY_CSV_FIELDS.join(','),
    ...rows.map((row) => TELEMETRY_CSV_FIELDS.map(
      (field) => csvEscape(row[field]),
    ).join(',')),
  ].join('\n');
}

function availableTelemetryReport() {
  const rows = [{
    observedAtIso: '2026-08-31T20:00:01.000Z',
    runElapsedMs: 1,
    runId: RUN_ID,
    trialId: null,
    planIndex: null,
    repetitionIndex: null,
    modeId: null,
    visibilityFraction: null,
    layout: null,
    phase: 'startup',
    gpuIndex: 0,
    gpuName: 'Fixture GPU',
    gpuUuid: 'GPU-fixture',
    pstate: 'P8',
    graphicsClockMHz: 300,
    memoryClockMHz: 405,
    gpuUtilizationPercent: 0,
    memoryUtilizationPercent: 0,
    memoryUsedMiB: 100,
    memoryTotalMiB: 24_576,
    temperatureC: 35,
    powerDrawW: 20,
  }];
  const report = unavailableTelemetryReport();
  report.status = 'available';
  report.reason = null;
  report.sampling.collectorStartedRunElapsedMs = 0;
  report.sampling.collectorStopRequestedRunElapsedMs = 2;
  report.sampling.exit = { code: 0, signal: null };
  report.summary = summarizeTelemetryRows(rows);
  report.coverageAudit = createNvidiaTelemetryCoverageAudit(rows, {
    collectorStartedRunElapsedMs: report.sampling.collectorStartedRunElapsedMs,
    collectorStopRequestedRunElapsedMs: report.sampling.collectorStopRequestedRunElapsedMs,
  });
  return { report, csv: `${telemetryRowsToCsv(rows)}\n` };
}

async function buildFixture() {
  const benchmarkPlan = plan();
  const geometry = geometryManifest();
  const scenarios = Object.fromEntries([0.99, 0.2].map((visibility) => {
    const value = scenarioManifest(visibility);
    return [String(visibility), value];
  }));
  const rows = [];
  const validationArtifacts = [];
  const trialSummaries = [];
  const parityByCell = {};
  for (const spec of benchmarkPlan) {
    const validation = await exactValidation(spec);
    const parity = renderParity(validation);
    const invariant = completionInvariant(spec, validation, spec.planIndex * 1_000);
    const trialRows = retainedRows(spec, validation, invariant);
    const pageSummary = {
      accepted: true,
      timestampAvailable: true,
      rowCount: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
      missingRenderFrames: 0,
      invalidRenderTimestampUidCountFrames: 0,
      expectedRenderTimestampUidCount: 1,
      missingComputeFrames: 0,
      quantumNs: 32,
      classification: 'fine',
      completionInvariant: invariant,
    };
    const trialEvidence = await validateFirstInstanceTrialEvidence({
      spec,
      environment: { indirectFirstInstanceAvailable: true },
      validation,
      renderParity: parity,
      rows: trialRows,
      summary: pageSummary,
      protocol: {
        schemaVersion: 2,
        warmupFrames: FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
        measuredFrames: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
        plannedScheduleSha256: firstInstanceCrossoverScheduleSha256(
          spec.superblockOrientationOffset,
        ),
      },
    });
    assert.equal(trialEvidence.pass, true, trialEvidence.rejectionReasons.join('\n'));
    const scenario = scenarios[String(spec.visibilityFraction)];
    const capture = {
      capturedAt: '2026-08-31T20:00:01.000Z',
      workload: {
        scenarioSeed: SCENARIO_SEED,
        geometryFixtureSha256: geometry.sha256,
        scenarioSha256: scenario.sha256,
      },
      validation: {
        payloadSha256: shaJson(validation),
        semanticSha256: firstInstanceValidationSemanticSha256(validation),
        payload: validation,
      },
      renderParity: parity,
      renderParitySemanticSha256: firstInstanceRenderParityIdentity(parity),
      renderParityOutputSha256: firstInstanceRenderParityIdentity(parity),
      accepted: true,
      rejectionReasons: [],
    };
    parityByCell[`baseline|${spec.visibilityFraction}`] =
      firstInstanceRenderParityIdentity(parity);
    const artifact = {
      schemaVersion: 2,
      trialId: spec.trialId,
      planIndex: spec.planIndex,
      repetitionIndex: spec.repetitionIndex,
      modeId: spec.modeId,
      visibilityFraction: spec.visibilityFraction,
      layout: spec.layout,
      objectCount: spec.objectCount,
      bucketCount: spec.bucketCount,
      laneCommandSegmentOrder: [...spec.laneCommandSegmentOrder],
      superblockOrientationOffset: spec.superblockOrientationOffset,
      plannedScheduleSha256: firstInstanceCrossoverScheduleSha256(
        spec.superblockOrientationOffset,
      ),
      selectedConfig: {
        strategyId: spec.modeId,
        objectCount: spec.objectCount,
        bucketCount: spec.bucketCount,
        visibilityFraction: spec.visibilityFraction,
        layout: spec.layout,
        laneCommandSegmentOrder: [...spec.laneCommandSegmentOrder],
        superblockOrientationOffset: spec.superblockOrientationOffset,
      },
      status: 'accepted',
      rejectionReasons: [],
      pre: structuredClone(capture),
      timingStart: structuredClone(capture),
      post: structuredClone(capture),
      firstInstanceTrialEvidence: trialEvidence,
    };
    artifact.sha256 = shaJson(artifact);
    validationArtifacts.push(artifact);
    trialSummaries.push({
      ...spec,
      plannedScheduleSha256: artifact.plannedScheduleSha256,
      selectedConfig: structuredClone(artifact.selectedConfig),
      startedAt: '2026-08-31T20:00:00.000Z',
      completedAt: '2026-08-31T20:00:02.000Z',
      elapsedMs: 2_000,
      validation: {
        pass: true,
        kind: validation.kind,
        artifactSha256: artifact.sha256,
        firstInstanceSemanticSha256: trialEvidence.semanticSha256,
      },
      timestamps: {
        accepted: true,
        available: true,
        rowCount: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
        missingRenderFrames: 0,
        missingComputeFrames: 0,
        expectedRenderTimestampUidCount: 1,
        invalidRenderTimestampUidCountFrames: 0,
        quantumNs: 32,
        classification: 'fine',
      },
      completionInvariant: invariant,
      timing: trialTiming(trialRows),
      accepted: true,
      rejectionReasons: [],
    });
    rows.push(...trialRows);
  }
  const analysis = summarizeFirstInstanceCrossoverRows(rows);
  const telemetrySummary = unavailableTelemetryReport();
  const start = provenanceRecord();
  const metadata = {
    schemaVersion: 2,
    runId: RUN_ID,
    status: 'complete',
    startedAt: '2026-08-31T20:00:00.000Z',
    completedAt: '2026-08-31T20:00:02.000Z',
    elapsedMs: 2_000,
    environment: {
      browser: {
        executable: 'chrome.exe',
        version: 'fixture-browser-1',
        headless: true,
        args: [
          '--enable-unsafe-webgpu',
          '--enable-webgpu-developer-features',
          '--disable-background-timer-throttling',
          '--disable-renderer-backgrounding',
        ],
        viewport: { width: 1280, height: 900, deviceScaleFactor: 1 },
      },
      backend: 'Fixture GPU · vulkan',
      benchmarkPage: {
        threeRevision: '185',
        userAgent: 'fixture-browser/1',
        adapterInfo: {
          vendor: 'fixture-vendor',
          architecture: 'fixture-architecture',
          device: 'fixture-device',
          description: 'Fixture GPU',
          backend: 'vulkan',
          type: 'discrete-gpu',
          driver: 'fixture-driver',
          isFallbackAdapter: false,
        },
        rendererBackend: 'WebGPUBackend',
        coordinateSystem: 2001,
        indirectFirstInstanceAvailable: true,
        reversedDepth: true,
        rendererReversedDepthBuffer: true,
        timestampAvailable: true,
        crossOriginIsolated: true,
        performanceNowQuantumMs: 0.005,
        viewport: { width: 1280, height: 720, devicePixelRatio: 1 },
      },
      gpuTelemetry: telemetrySummary,
    },
    evidenceStatus: 'candidate',
    sourceProvenance: {
      start,
      end: structuredClone(start),
      stable: true,
    },
    workload: {
      scenarioGenerator: 'createFixedSubsetScenario',
      scenarioSeed: SCENARIO_SEED,
      manifestArtifact: 'workload-manifests.json',
      geometryFixtureSha256: geometry.sha256,
      scenarioSha256ByVisibility: Object.fromEntries([0.99, 0.2].map(
        (visibility) => [String(visibility), scenarios[String(visibility)].sha256],
      )),
      scenarioSha256ByCell: Object.fromEntries([0.99, 0.2].map(
        (visibility) => [`baseline|${visibility}`, scenarios[String(visibility)].sha256],
      )),
      renderParitySha256ByCell: parityByCell,
      physicalBinSequenceSha256ByPair: null,
    },
    protocol: protocol(),
    plan: benchmarkPlan,
    expectedTrialCount: benchmarkPlan.length,
    completedTrialCount: benchmarkPlan.length,
    acceptedTrialCount: benchmarkPlan.length,
    frameRowCount: rows.length,
    validationArtifactCount: validationArtifacts.length,
    validationArtifactSha256: validationArtifacts.map((artifact) => artifact.sha256),
    firstInstanceAnalysisAudit: {
      schemaVersion: analysis.schemaVersion,
      kind: analysis.kind,
      deltaConvention: analysis.deltaConvention,
      nTrials: analysis.nTrials,
      nRows: analysis.nRows,
      sha256: shaJson(analysis),
    },
    pageErrors: [],
    error: null,
  };
  return {
    metadata,
    rows,
    validationArtifacts,
    trialSummaries,
    workloadManifests: {
      schemaVersion: 1,
      hashAlgorithm: 'sha256',
      geometryFixturesBySha256: { [geometry.sha256]: geometry },
      scenariosBySha256: Object.fromEntries(Object.values(scenarios).map(
        (scenario) => [scenario.sha256, scenario],
      )),
      invalidObservations: [],
    },
    telemetrySummary,
    telemetryCsv: `${telemetryRowsToCsv([])}\n`,
    telemetryCsvPresent: true,
  };
}

let baseFixturePromise;
async function createRunDirectory(t, mutate = () => undefined) {
  baseFixturePromise ??= buildFixture();
  const fixture = structuredClone(await baseFixturePromise);
  await mutate(fixture);
  for (const [index, artifact] of fixture.validationArtifacts.entries()) {
    delete artifact.sha256;
    artifact.sha256 = shaJson(artifact);
    fixture.trialSummaries[index].validation.artifactSha256 = artifact.sha256;
  }
  fixture.metadata.validationArtifactSha256 = fixture.validationArtifacts.map(
    (artifact) => artifact.sha256,
  );
  const directory = await mkdtemp(
    path.join(os.tmpdir(), 'three-webgpu-first-instance-analysis-'),
  );
  t.after(async () => rm(directory, { recursive: true, force: true }));
  const artifacts = new Map([
    ['frames.csv', Buffer.from(rowsToCsv(fixture.rows))],
    ['metadata.json', jsonBytes(fixture.metadata)],
    ['trial-summaries.json', jsonBytes(fixture.trialSummaries)],
    ['validation-artifacts.json', jsonBytes(fixture.validationArtifacts)],
    ['workload-manifests.json', jsonBytes(fixture.workloadManifests)],
    ['gpu-telemetry-summary.json', jsonBytes(fixture.telemetrySummary)],
  ]);
  if (fixture.telemetryCsvPresent) {
    artifacts.set('gpu-telemetry.csv', Buffer.from(fixture.telemetryCsv));
  }
  await Promise.all([...artifacts].map(([name, contents]) => (
    writeFile(path.join(directory, name), contents)
  )));
  const telemetryPresent = fixture.telemetryCsvPresent === true;
  const telemetryEvidenceAvailable = fixture.telemetrySummary.status === 'available'
    && telemetryPresent;
  const telemetryAbsenceReason = fixture.telemetrySummary.reason
    ?? `telemetry status: ${fixture.telemetrySummary.status ?? 'unavailable'}`;
  const manifest = {
    schemaVersion: 2,
    runId: RUN_ID,
    hashAlgorithm: 'sha256',
    requiredFiles: [...REQUIRED_ARTIFACTS],
    optionalFiles: [{
      name: 'gpu-telemetry.csv',
      present: telemetryPresent,
      evidenceAvailable: telemetryEvidenceAvailable,
      absenceReason: telemetryEvidenceAvailable ? null : telemetryAbsenceReason,
    }],
    files: [
      ...[...artifacts].map(([name, contents]) => ({
        name,
        role: `test ${name}`,
        required: REQUIRED_ARTIFACTS.includes(name),
        present: true,
        bytes: contents.length,
        sha256: sha(contents),
        absenceReason: null,
      })),
      ...(!telemetryPresent ? [{
        name: 'gpu-telemetry.csv',
        role: 'optional device telemetry samples',
        required: false,
        present: false,
        bytes: null,
        sha256: null,
        absenceReason: telemetryAbsenceReason,
      }] : []),
    ],
  };
  await writeFile(
    path.join(directory, 'artifact-manifest.json'),
    jsonBytes(manifest),
  );
  return directory;
}

test('first-instance protocol and plan validators bind the exact design', () => {
  const validProtocol = protocol();
  const validPlan = plan();
  assert.doesNotThrow(() => validateProtocolMatrix(validProtocol));
  assert.doesNotThrow(() => validateBenchmarkPlan(validPlan, {
    runId: RUN_ID,
    protocol: validProtocol,
  }));

  const changedSchedule = structuredClone(validProtocol);
  changedSchedule.firstInstanceCrossover.scheduleSha256ByOrientation[0] =
    'f'.repeat(64);
  assert.throws(() => validateProtocolMatrix(changedSchedule), /schedule commitments/);

  const changedFactor = structuredClone(validPlan);
  changedFactor[0].laneCommandSegmentOrder.reverse();
  assert.throws(
    () => validateBenchmarkPlan(changedFactor, { runId: RUN_ID, protocol: validProtocol }),
    /preregistered first-instance crossover factor/,
  );
});

test('verified first-instance run reconstructs the exact numerical summary', async (t) => {
  const directory = await createRunDirectory(t);
  const summary = await summarizeInput(directory);
  assert.equal(summary.kind, 'indirect-first-instance-crossover-summary');
  assert.equal(summary.nTrials, 24);
  assert.equal(summary.nRows, 11_520);
  assert.equal(summary.artifactVerification.status, 'consistent');
  assert.equal(summary.artifactVerification.evidenceStatus, 'candidate');
  assert.equal(summary.artifactVerification.sourceProvenanceStable, true);
});

test('verified first-instance run accepts coherent available telemetry evidence', async (t) => {
  const { report, csv } = availableTelemetryReport();
  const directory = await createRunDirectory(t, (fixture) => {
    fixture.telemetrySummary = report;
    fixture.metadata.environment.gpuTelemetry = structuredClone(report);
    fixture.telemetryCsv = csv;
    fixture.telemetryCsvPresent = true;
  });
  const summary = await summarizeInput(directory);
  assert.equal(summary.artifactVerification.status, 'consistent');
  assert.equal(summary.artifactVerification.verifiedArtifactCount, 7);
});

test('verified first-instance run rejects a rehashed command-segment row tamper', async (t) => {
  const directory = await createRunDirectory(t, ({ rows }) => {
    rows[0].commandByteBase += RECORD_BYTES;
  });
  await assert.rejects(summarizeInput(directory), /commandByteBase|command segment/);
});

test('verified first-instance run rejects rehashed postflight evidence tampering', async (t) => {
  const directory = await createRunDirectory(t, ({ validationArtifacts }) => {
    const capture = validationArtifacts[0].post;
    capture.validation.payload.expectedIdsMatchScenario = false;
    capture.validation.payloadSha256 = shaJson(capture.validation.payload);
    capture.validation.semanticSha256 = firstInstanceValidationSemanticSha256(
      capture.validation.payload,
    );
  });
  await assert.rejects(
    summarizeInput(directory),
    /changed semanticSha256 across captures|post first-instance payload failed|expected IDs match scenario/,
  );
});

test('verified first-instance run rejects a rehashed analyzer-audit tamper', async (t) => {
  const directory = await createRunDirectory(t, ({ metadata }) => {
    metadata.firstInstanceAnalysisAudit.sha256 = 'f'.repeat(64);
  });
  await assert.rejects(summarizeInput(directory), /firstInstanceAnalysisAudit/);
});

test('verified first-instance run binds trial timing summaries to retained rows', async (t) => {
  const cases = [
    [
      'omitted timing record',
      ({ trialSummaries }) => { delete trialSummaries[0].timing; },
      /timing must be an object/,
    ],
    [
      'rehashed p50 tamper',
      ({ trialSummaries }) => { trialSummaries[0].timing.gpuRenderP50Ms += 0.01; },
      /differs from its retained rows/,
    ],
  ];
  for (const [name, mutate, pattern] of cases) {
    await t.test(name, async (subtest) => {
      const directory = await createRunDirectory(subtest, mutate);
      await assert.rejects(summarizeInput(directory), pattern);
    });
  }
});

test('verified first-instance run rejects invalid startup timer isolation evidence', async (t) => {
  const cases = [
    [
      'cross-origin isolation is false',
      (page) => { page.crossOriginIsolated = false; },
      /cross-origin isolation/,
    ],
    [
      'timer quantum exceeds the preregistered maximum',
      (page) => { page.performanceNowQuantumMs = 0.02; },
      /CPU timer quantum/,
    ],
    [
      'timer quantum is null',
      (page) => { page.performanceNowQuantumMs = null; },
      /CPU timer quantum/,
    ],
    [
      'timer quantum is zero',
      (page) => { page.performanceNowQuantumMs = 0; },
      /CPU timer quantum/,
    ],
    [
      'timer quantum is negative',
      (page) => { page.performanceNowQuantumMs = -0.001; },
      /CPU timer quantum/,
    ],
  ];
  for (const [name, mutatePage, pattern] of cases) {
    await t.test(name, async (subtest) => {
      const directory = await createRunDirectory(subtest, ({ metadata }) => {
        mutatePage(metadata.environment.benchmarkPage);
      });
      await assert.rejects(summarizeInput(directory), pattern);
    });
  }
});

test('verified first-instance run binds browser, backend, adapter, and viewport identity', async (t) => {
  const cases = [
    [
      'browser metadata omitted',
      ({ metadata }) => { delete metadata.environment.browser; },
      /environment\.browser must be an object/,
    ],
    [
      'browser launch arguments changed',
      ({ metadata }) => { metadata.environment.browser.args.pop(); },
      /browser launch identity/,
    ],
    [
      'runner backend text changed',
      ({ metadata }) => { metadata.environment.backend = 'different backend'; },
      /runner backend identity/,
    ],
    [
      'Three.js revision changed',
      ({ metadata }) => { metadata.environment.benchmarkPage.threeRevision = '186'; },
      /pinned Three\.js revision/,
    ],
    [
      'render viewport changed',
      ({ metadata }) => { metadata.environment.benchmarkPage.viewport.width = 1279; },
      /page viewport/,
    ],
    [
      'adapter identity omitted',
      ({ metadata }) => { delete metadata.environment.benchmarkPage.adapterInfo; },
      /adapterInfo must be an object/,
    ],
    [
      'renderer backend changed',
      ({ metadata }) => { metadata.environment.benchmarkPage.rendererBackend = null; },
      /pinned WebGPU renderer identity/,
    ],
    [
      'coordinate system changed',
      ({ metadata }) => { metadata.environment.benchmarkPage.coordinateSystem = 2000; },
      /pinned WebGPU renderer identity/,
    ],
    [
      'user agent omitted',
      ({ metadata }) => { metadata.environment.benchmarkPage.userAgent = ''; },
      /userAgent must be a non-empty string/,
    ],
  ];
  for (const [name, mutate, pattern] of cases) {
    await t.test(name, async (subtest) => {
      const directory = await createRunDirectory(subtest, mutate);
      await assert.rejects(summarizeInput(directory), pattern);
    });
  }
});

test('verified first-instance run rejects rehashed telemetry coherence tampering', async (t) => {
  await t.test('available report without its CSV evidence', async (subtest) => {
    const { report, csv } = availableTelemetryReport();
    const directory = await createRunDirectory(subtest, (fixture) => {
      fixture.telemetrySummary = report;
      fixture.metadata.environment.gpuTelemetry = structuredClone(report);
      fixture.telemetryCsv = csv;
      fixture.telemetryCsvPresent = false;
    });
    await assert.rejects(
      summarizeInput(directory),
      /manifest presence\/evidence fields|requires gpu-telemetry\.csv/,
    );
  });

  await t.test('malformed telemetry report schema', async (subtest) => {
    const directory = await createRunDirectory(subtest, (fixture) => {
      delete fixture.telemetrySummary.sampling.queryFields;
      delete fixture.metadata.environment.gpuTelemetry.sampling.queryFields;
    });
    await assert.rejects(
      summarizeInput(directory),
      /gpu-telemetry-summary\.json sampling has an unexpected schema/,
    );
  });

  await t.test('coverage audit differs from CSV reconstruction', async (subtest) => {
    const { report, csv } = availableTelemetryReport();
    report.coverageAudit = { ...report.coverageAudit, finalMaximumGapMs: 0 };
    const directory = await createRunDirectory(subtest, (fixture) => {
      fixture.telemetrySummary = report;
      fixture.metadata.environment.gpuTelemetry = structuredClone(report);
      fixture.telemetryCsv = csv;
      fixture.telemetryCsvPresent = true;
    });
    await assert.rejects(
      summarizeInput(directory),
      /does not reconstruct telemetry coverage audit/,
    );
  });
});
