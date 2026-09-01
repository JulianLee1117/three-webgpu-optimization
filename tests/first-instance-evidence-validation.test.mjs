import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import {
  FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
  firstInstanceCrossoverScheduleSha256,
  firstInstanceRenderParityIdentity,
  firstInstanceValidationSemanticSha256,
  validateFirstInstanceCrossoverRenderParity,
  validateFirstInstanceCrossoverValidation,
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
} from '../src/benchmark/plan.js';
import { createFirstInstanceShaderEvidence } from '../src/validation/first-instance-shader-evidence.js';

const [PORTABLE, FEATURE] = FIRST_INSTANCE_CROSSOVER_LANES;
const OBJECT_COUNT = 512;
const BUCKET_COUNT = 2;
const VISIBLE_COUNT = Math.round(OBJECT_COUNT * 0.99);
const VISIBLE_COUNTS = [253, 254];
const INDEX_COUNTS = [12, 18];
const BUCKET_BASES = [0, 256];
const RECORDS_PER_LANE = 2;
const COMMAND_SEGMENT_BYTE_LENGTH = RECORDS_PER_LANE * 20;
const ZERO_RESET_SHA256 = sha256Bytes(Buffer.alloc(OBJECT_COUNT * 4));

function sha256Bytes(value) {
  return createHash('sha256').update(value).digest('hex');
}

function sha(value) {
  return sha256Bytes(value);
}

function fnv1a64Text(value) {
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(value)) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, '0');
}

function timedVertexShader({ feature, matrixName, visibleName }) {
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
struct ${matrixName}Struct {
  value : array< mat4x4<f32> >
};
@binding( 3 ) @group( 1 )
var<storage, read> ${matrixName} : ${matrixName}Struct;

struct ${visibleName}Struct {
  value : array< u32 >
};
@binding( 4 ) @group( 1 )
var<storage, read> ${visibleName} : ${visibleName}Struct;

struct VaryingsStruct {
  @builtin( position ) position : vec4<f32>
};

@vertex
${inputs}
  var objectMatrix : mat4x4<f32>;
  objectMatrix = ${matrixName}.value[ ${visibleName}.value[ ${address} ] ];
  var transformed : vec3<f32> = ( objectMatrix * vec4<f32>( position, 1.0 ) ).xyz;
  var varyings : VaryingsStruct;
  varyings.position = vec4<f32>( transformed + normal * 0.0, 1.0 );
  return varyings;
}
`;
}

const TIMED_FRAGMENT_SHADER = `
@fragment
fn main() -> @location( 0 ) vec4<f32> {
  return vec4<f32>( 0.4, 0.65, 0.95, 1.0 );
}
`;

function timedVertexInputs(feature) {
  const position = {
    name: 'position', shaderLocation: 0, format: 'float32x3', stepMode: 'vertex',
    arrayType: 'Float32Array', itemSize: 3, normalized: false, count: 30, resourceId: 700,
  };
  const normal = {
    name: 'normal', shaderLocation: feature ? 1 : 2, format: 'float32x3', stepMode: 'vertex',
    arrayType: 'Float32Array', itemSize: 3, normalized: false, count: 30, resourceId: 701,
  };
  if (feature) return [position, normal];
  return [position, {
    name: 'bucketBase', shaderLocation: 1, format: 'uint32', stepMode: 'vertex',
    arrayType: 'Uint32Array', itemSize: 1, normalized: false, count: 30, resourceId: 702,
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

async function timedShaderEvidence() {
  const sources = {
    [PORTABLE]: {
      vertexShader: timedVertexShader({
        feature: false,
        matrixName: 'NodeBuffer_100',
        visibleName: 'NodeBuffer_101',
      }),
      fragmentShader: TIMED_FRAGMENT_SHADER,
      vertexInputs: timedVertexInputs(false),
      storageBindings: timedStorageBindings(),
    },
    [FEATURE]: {
      vertexShader: timedVertexShader({
        feature: true,
        matrixName: 'NodeBuffer_200',
        visibleName: 'NodeBuffer_201',
      }),
      fragmentShader: TIMED_FRAGMENT_SHADER,
      vertexInputs: timedVertexInputs(true),
      storageBindings: timedStorageBindings(),
    },
  };
  const evidence = await createFirstInstanceShaderEvidence({
    portable: sources[PORTABLE],
    feature: sources[FEATURE],
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
struct ${storageStruct} {
  value : array< u32 >
};
@binding( 4 ) @group( 1 )
var<storage, read> ${storageVariable} : ${storageStruct};

struct VaryingsStruct {
  @builtin( position ) position : vec4<f32>,
  @location( 0 ) @interpolate( flat, either ) vAddressChallengeObjectId : u32
};

@vertex
fn main( ${inputs} ) -> VaryingsStruct {
  var ${prefix}ChallengeAddress : u32;
  ${prefix}ChallengeAddress = ${address};
  let ${prefix}ChallengeObjectId : u32 = ${storageVariable}.value[ ${prefix}ChallengeAddress ];
  let pixelX : u32 = ${prefix}ChallengeAddress % 256u;
  let pixelY : u32 = ${prefix}ChallengeAddress / 256u;
  var output : VaryingsStruct;
  output.position = vec4<f32>( vec3<f32>( position.xy + vec2<f32>( f32( pixelX ), f32( pixelY ) ) * 0.0, position.z ), 1.0 );
  output.vAddressChallengeObjectId = ${prefix}ChallengeObjectId;
  return output;
}
`;
}

const CHALLENGE_FRAGMENT_SHADER = `
struct OutputStruct {
  @location( 0 ) color : vec4<f32>
};
var<private> output : OutputStruct;
var<private> encoded : u32;
@fragment
fn main( @location( 0 ) @interpolate( flat, either ) vAddressChallengeObjectId : u32 ) -> OutputStruct {
  encoded = vAddressChallengeObjectId + 1u;
  output.color = vec4<f32>( f32( encoded & 255u ) / 255.0, f32( ( encoded >> 8u ) & 255u ) / 255.0, f32( ( encoded >> 16u ) & 255u ) / 255.0, 1.0 );
  return output;
}
`;

function challengeShader(lane) {
  const vertexShader = challengeVertexShader(lane);
  const position = {
    name: 'position', shaderLocation: lane === PORTABLE ? 1 : 0,
    format: 'float32x3', stepMode: 'vertex',
    arrayType: 'Float32Array', itemSize: 3, normalized: false, count: 30,
    resourceId: 'challenge-position-resource',
  };
  return {
    vertexSha256: sha(vertexShader),
    fragmentSha256: sha(CHALLENGE_FRAGMENT_SHADER),
    vertexLines: [],
    fragmentLines: [],
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

function targetEvidence() {
  return {
    width: 256,
    height: 2,
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

function expectedSegments(order) {
  return Object.fromEntries(FIRST_INSTANCE_CROSSOVER_LANES.map((lane) => {
    const index = order.indexOf(lane);
    return [lane, {
      index,
      recordBase: index * RECORDS_PER_LANE,
      byteBase: index * COMMAND_SEGMENT_BYTE_LENGTH,
    }];
  }));
}

function attributeEvidence(name, id, digest = sha(name)) {
  return {
    name,
    id,
    version: 0,
    arrayType: name === 'bucketBase' || name === 'index' ? 'Uint32Array' : 'Float32Array',
    itemSize: name === 'uv' ? 2 : name === 'bucketBase' || name === 'index' ? 1 : 3,
    count: 30,
    normalized: false,
    gpuType: null,
    byteLength: 120,
    isInstancedBufferAttribute: false,
    meshPerAttribute: null,
    sha256: digest,
  };
}

function geometryEvidence() {
  const attributes = Object.fromEntries(['normal', 'position', 'uv'].map((name, index) => {
    const attribute = attributeEvidence(name, 10 + index);
    return [name, {
      sameObject: true,
      portable: { ...attribute },
      feature: { ...attribute },
    }];
  }));
  const index = attributeEvidence('index', 20);
  return {
    schemaVersion: 1,
    kind: 'shared-first-instance-geometry-evidence',
    pass: true,
    portableAttributeNames: ['bucketBase', 'normal', 'position', 'uv'],
    featureAttributeNames: ['normal', 'position', 'uv'],
    sharedIndex: { sameObject: true, portable: { ...index }, feature: { ...index } },
    commonAttributes: attributes,
    bucketBase: attributeEvidence('bucketBase', 21),
    bucketBaseMismatchCount: 0,
    noInstanceSteppedAttributes: true,
  };
}

function membershipEvidence() {
  return {
    membership: {
      pass: true,
      expectedCount: VISIBLE_COUNT,
      listedCount: VISIBLE_COUNT,
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
      expected: { count: VISIBLE_COUNT, sha256: '1'.repeat(64) },
      actual: { count: VISIBLE_COUNT, sha256: '1'.repeat(64) },
      perBucket: [0, 1].map((bucket) => ({
        bucket,
        match: true,
        expected: { count: VISIBLE_COUNTS[bucket], sha256: String(bucket + 2).repeat(64) },
        actual: {
          count: VISIBLE_COUNTS[bucket],
          declaredCount: VISIBLE_COUNTS[bucket],
          sha256: String(bucket + 2).repeat(64),
        },
      })),
    },
  };
}

function commandRecords(lane) {
  let firstIndex = 0;
  return INDEX_COUNTS.map((indexCount, bucket) => {
    const command = {
      indexCount,
      instanceCount: VISIBLE_COUNTS[bucket],
      firstIndex,
      baseVertex: 0,
      firstInstance: lane === PORTABLE ? 0 : BUCKET_BASES[bucket],
    };
    firstIndex += indexCount;
    return { bucket, actual: { ...command }, expected: { ...command } };
  });
}

function packingEvidence(order) {
  const laneCommands = Object.fromEntries(FIRST_INSTANCE_CROSSOVER_LANES.map((lane) => [lane, {
    pass: true,
    errors: [],
    commandCount: BUCKET_COUNT,
    totalInstanceCount: VISIBLE_COUNT,
    records: commandRecords(lane),
  }]));
  return {
    schemaVersion: 1,
    kind: 'first-instance-crossover-exact-frozen-packing-validation',
    pass: true,
    metadata: {
      pass: true,
      errors: [],
      laneCommandSegmentOrder: [...order],
      recordsPerLane: RECORDS_PER_LANE,
      commandSegmentByteLength: COMMAND_SEGMENT_BYTE_LENGTH,
    },
    visibleIds: {
      pass: true,
      exactExpectedPacking: true,
      elementCount: OBJECT_COUNT,
      sha256: '4'.repeat(64),
    },
    padding: {
      pass: true,
      sentinel: 0xffff_ffff,
      paddingCount: OBJECT_COUNT - VISIBLE_COUNT,
      paddingSentinelCount: OBJECT_COUNT - VISIBLE_COUNT,
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
      visibleIdsSha256: '4'.repeat(64),
      physicalCommandsSha256: '5'.repeat(64),
      logicalPairSha256: '6'.repeat(64),
      paddingSha256: '7'.repeat(64),
      commandCoresEqual: true,
      lanes: {
        [PORTABLE]: { commandsSha256: '8'.repeat(64), coreSha256: 'a'.repeat(64) },
        [FEATURE]: { commandsSha256: '9'.repeat(64), coreSha256: 'a'.repeat(64) },
      },
      pairs: BUCKET_BASES.map((base, bucket) => ({
        bucket,
        coreEqual: true,
        portableFirstInstance: 0,
        featureFirstInstance: base,
        expectedFeatureFirstInstance: base,
        sha256: String(bucket + 11).repeat(64).slice(0, 64),
      })),
    },
  };
}

function addressChallenge(lane, segments) {
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
      sha256: ZERO_RESET_SHA256,
      expectedSha256: ZERO_RESET_SHA256,
    },
    pixelCount: OBJECT_COUNT,
    addressCount: OBJECT_COUNT,
    byteLength: OBJECT_COUNT * 4,
    sha256: 'b'.repeat(64),
    expectedSha256: 'b'.repeat(64),
    exactExpectedBytes: true,
    activeAddressCount: VISIBLE_COUNT,
    paddingAddressCount: OBJECT_COUNT - VISIBLE_COUNT,
    targetPaddingPixelCount: 0,
    activeMismatchCount: 0,
    paddingMismatchCount: 0,
    targetPaddingMismatchCount: 0,
    coverage: {
      allBucketsActive: true,
      nonzeroBucketBaseCount: BUCKET_COUNT - 1,
      activeRowCount: 2,
      nonzeroEncodedChannelPixelCounts: { red: 398, green: 200, blue: 0 },
      alphaEncodedPixelCount: VISIBLE_COUNT,
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
    rootVersion: 0,
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

async function exactValidation(order) {
  const segments = expectedSegments(order);
  const shaderEvidence = await timedShaderEvidence();
  const membership = membershipEvidence();
  const lifecycle = lifecycleEvidence(order, segments, shaderEvidence);
  return {
    schemaVersion: 1,
    kind: 'first-instance-crossover-exact-paired-snapshots',
    pass: true,
    expectedIdsMatchScenario: true,
    frozenPacking: packingEvidence(order),
    ...membership,
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
        target: { width: 256, height: 2, pixelCount: OBJECT_COUNT },
        indexCount: 30,
        addressedTriangleCount: BUCKET_COUNT,
        degenerateTriangleCount: 8,
        addressedTrianglesPerSubmittedInstance: 1,
        positionMismatchCount: 0,
        bucketBaseMismatchCount: 0,
        indexMismatchCount: 0,
        attributesExact: true,
        sharedPayloadExact: true,
        indirectIdentityExact: true,
        laneOffsetsExact: { [PORTABLE]: true, [FEATURE]: true },
        expectedInstanceCount: 256,
      },
      lanes: {
        [PORTABLE]: addressChallenge(PORTABLE, segments),
        [FEATURE]: addressChallenge(FEATURE, segments),
      },
    },
    shaderEvidence,
    lifecycle,
  };
}

function parityLane() {
  const color = { format: 'rgba8unorm', arrayType: 'Uint8Array', byteLength: 1280 * 720 * 4, sha256: 'c'.repeat(64) };
  const depth = { format: 'depth32float', arrayType: 'Float32Array', byteLength: 1280 * 720 * 4, sha256: 'd'.repeat(64) };
  const objectId = { format: 'rgba8unorm-object-id-plus-one', arrayType: 'Uint8Array', byteLength: 1280 * 720 * 4, sha256: 'e'.repeat(64) };
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
        color: { ...color },
        depth: { ...depth },
        objectId: { ...objectId },
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

function completionInvariant(spec, validation) {
  const staticLifecycle = structuredClone(validation.lifecycle);
  delete staticLifecycle.activeLane;
  delete staticLifecycle.laneSelectionSerial;
  const commitment = fnv1a64Text(JSON.stringify(staticLifecycle));
  const segments = expectedSegments(spec.laneCommandSegmentOrder);
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
      commandRecordCount: RECORDS_PER_LANE * 2,
      visibleIdsCount: OBJECT_COUNT,
    },
    lifecycleExact: true,
    staticLifecycleAtTimingStart: structuredClone(staticLifecycle),
    staticLifecycleAtTimingEnd: structuredClone(staticLifecycle),
    lifecycleCommitmentAtTimingStart: commitment,
    lifecycleCommitmentAtTimingEnd: commitment,
    selectorWriteSerialAtTimingStart: 10,
    selectorWriteSerialAtTimingEnd: 10 + FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
    selectorWritesDuringTiming: FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
    strategySelectionSerialAtTimingStart: 20,
    strategySelectionSerialAtTimingEnd: 20 + FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
    strategySelectionsDuringTiming: FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
    renderCallSerialAtTimingStart: 30,
    renderCallSerialAtTimingEnd: 30 + FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
    renderCallsDuringTiming: FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT,
    computeCallSerialAtTimingStart: 5,
    computeCallSerialAtTimingEnd: 5,
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
  const segments = expectedSegments(spec.laneCommandSegmentOrder);
  const scheduleSha256 = firstInstanceCrossoverScheduleSha256(
    spec.superblockOrientationOffset,
  );
  return Array.from({ length: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES }, (_, index) => {
    const scheduled = firstInstanceCrossoverFrame(index, spec.superblockOrientationOffset);
    const segment = segments[scheduled.laneId];
    const ordinal = FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES + index + 1;
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
      expectedVisibleCount: VISIBLE_COUNT,
      protocolWarmupFrames: FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
      protocolMeasuredFrames: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
      validationKind: validation.kind,
      validationPass: true,
      usesCompute: false,
      configuredDrawCommands: BUCKET_COUNT,
      configuredRenderObjects: 1,
      configuredComputeDispatches: 0,
      configuredComputeSubmissions: 0,
      configuredSubmittedInstances: VISIBLE_COUNT,
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
      strategySelectionSerialAtTimingStart: invariant.strategySelectionSerialAtTimingStart,
      renderCallSerialAtTimingStart: invariant.renderCallSerialAtTimingStart,
      renderTargetTextureUuidAtTimingStart: invariant.renderTargetTextureUuidAtTimingStart,
      renderTargetWidthAtTimingStart: invariant.renderTargetWidthAtTimingStart,
      renderTargetHeightAtTimingStart: invariant.renderTargetHeightAtTimingStart,
      renderTargetSamplesAtTimingStart: invariant.renderTargetSamplesAtTimingStart,
      renderTargetDepthBufferAtTimingStart: invariant.renderTargetDepthBufferAtTimingStart,
      cameraViewFnv64AtTimingStart: invariant.cameraViewFnv64AtTimingStart,
      cameraProjectionFnv64AtTimingStart: invariant.cameraProjectionFnv64AtTimingStart,
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
      gpuFrameId: 1_000 + index,
      cpuCommonUpdateMs: 0.01,
      cpuComputeSubmitMs: null,
      cpuRenderSubmitMs: 0.1,
      cpuSubmitTotalMs: 0.1,
      cpuFrameBodyMs: 0.12,
      gpuComputeMs: null,
      gpuRenderMs: 0.2,
      gpuRenderTimestampUidCount: 1,
      gpuPassTotalMs: 0.2,
    };
  });
}

async function artifact({ repetitionIndex = 0, visibilityOrderPosition = 0 } = {}) {
  const visibilityOrder = repetitionIndex % 2 === 0 ? [0.99, 0.2] : [0.2, 0.99];
  const commandOrders = [
    [PORTABLE, FEATURE], [FEATURE, PORTABLE], [FEATURE, PORTABLE],
    [PORTABLE, FEATURE], [FEATURE, PORTABLE], [PORTABLE, FEATURE],
    [PORTABLE, FEATURE], [FEATURE, PORTABLE], [FEATURE, PORTABLE],
    [PORTABLE, FEATURE], [PORTABLE, FEATURE], [FEATURE, PORTABLE],
  ];
  const orientationOffsets = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1];
  const spec = {
    runId: 'run',
    trialId: 'trial',
    planIndex: repetitionIndex * 2 + visibilityOrderPosition,
    repetitionIndex,
    modeId: FIRST_INSTANCE_CROSSOVER_MODE,
    modeOrder: [FIRST_INSTANCE_CROSSOVER_MODE],
    modeOrderPosition: 0,
    visibilityFraction: visibilityOrder[visibilityOrderPosition],
    visibilityOrder,
    visibilityOrderPosition,
    layout: 'baseline',
    layoutOrder: ['baseline'],
    layoutOrderPosition: 0,
    laneCommandSegmentOrder: [...commandOrders[repetitionIndex]],
    superblockOrientationOffset: orientationOffsets[repetitionIndex],
    objectCount: OBJECT_COUNT,
    bucketCount: BUCKET_COUNT,
  };
  const validation = await exactValidation(spec.laneCommandSegmentOrder);
  const invariant = completionInvariant(spec, validation);
  const rows = retainedRows(spec, validation, invariant);
  const summary = {
    accepted: true,
    timestampAvailable: true,
    rowCount: rows.length,
    missingRenderFrames: 0,
    invalidRenderTimestampUidCountFrames: 0,
    expectedRenderTimestampUidCount: 1,
    missingComputeFrames: 0,
    quantumNs: 32,
    classification: 'fine',
    completionInvariant: invariant,
  };
  return {
    spec,
    environment: {
      indirectFirstInstanceAvailable: true,
      timestampAvailable: true,
    },
    validation,
    renderParity: renderParity(validation),
    rows,
    summary,
    protocol: {
      schemaVersion: 2,
      warmupFrames: FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
      measuredFrames: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
      plannedScheduleSha256: firstInstanceCrossoverScheduleSha256(
        spec.superblockOrientationOffset,
      ),
    },
  };
}

function clone(value) {
  return structuredClone(value);
}

test('strict first-instance evidence accepts the complete causal artifact', async () => {
  const value = await artifact();
  const result = await validateFirstInstanceTrialEvidence(value);
  assert.equal(result.pass, true, result.rejectionReasons.join('\n'));
  assert.deepEqual(result.rejectionReasons, []);
  assert.match(result.semanticSha256, /^[0-9a-f]{64}$/);
  assert.equal(result.semanticSha256, firstInstanceValidationSemanticSha256(value.validation));
  assert.match(firstInstanceRenderParityIdentity(value.renderParity), /^[0-9a-f]{64}$/);
  assert.deepEqual(await validateFirstInstanceCrossoverValidation(
    value.validation,
    { spec: value.spec, environment: value.environment },
  ), []);
  assert.deepEqual(validateFirstInstanceCrossoverRenderParity(
    value.renderParity,
    { spec: value.spec, validation: value.validation },
  ), []);
});

test('reversed physical command segments remain valid while parity identity stays logical', async () => {
  const forward = await artifact();
  const reversed = await artifact({ repetitionIndex: 1, visibilityOrderPosition: 1 });
  const result = await validateFirstInstanceTrialEvidence(reversed);
  assert.equal(result.pass, true, result.rejectionReasons.join('\n'));
  assert.equal(
    firstInstanceRenderParityIdentity(reversed.renderParity),
    firstInstanceRenderParityIdentity(forward.renderParity),
  );
  assert.notEqual(
    firstInstanceValidationSemanticSha256(reversed.validation),
    firstInstanceValidationSemanticSha256(forward.validation),
  );
});

test('validator rejects environment, command, shader, oracle, lifecycle, and row tampering', async () => {
  const pristine = await artifact();
  const cases = [
    ['indirectFirstInstanceAvailable', (value) => {
      value.environment.indirectFirstInstanceAvailable = false;
    }],
    ['fifth command word', (value) => {
      value.validation.frozenPacking.commands.lanes[FEATURE].records[1].actual.firstInstance = 0;
    }],
    ['cross-lane indexCount', (value) => {
      value.validation.frozenPacking.commands.lanes[FEATURE].records[0].actual.indexCount += 1;
    }],
    ['no visible instances', (value) => {
      value.validation.frozenPacking.commands.lanes[PORTABLE].records[1].actual.instanceCount = 0;
    }],
    ['challenge raw vertex WGSL is missing', (value) => {
      delete value.validation.addressChallenges.lanes[PORTABLE].shader.rawSources;
    }],
    ['raw vertex WGSL digest', (value) => {
      value.validation.addressChallenges.lanes[PORTABLE].shader.vertexSha256 = '0'.repeat(64);
    }],
    ['normalized vertex WGSL identity', (value) => {
      const shader = value.validation.addressChallenges.lanes[FEATURE].shader;
      shader.rawSources.vertexShader = shader.rawSources.vertexShader.replace(
        'let pixelX',
        'let unrelated : u32 = 7u;\n  let pixelX',
      );
      shader.vertexSha256 = sha(shader.rawSources.vertexShader);
    }],
    ['RGB24 green-byte extraction', (value) => {
      for (const lane of FIRST_INSTANCE_CROSSOVER_LANES) {
        const shader = value.validation.addressChallenges.lanes[lane].shader;
        shader.rawSources.fragmentShader = shader.rawSources.fragmentShader.replace(
          'encoded >> 8u',
          'encoded >> 7u',
        );
        shader.fragmentSha256 = sha(shader.rawSources.fragmentShader);
      }
    }],
    ['encoded fragment alpha', (value) => {
      for (const lane of FIRST_INSTANCE_CROSSOVER_LANES) {
        const shader = value.validation.addressChallenges.lanes[lane].shader;
        shader.rawSources.fragmentShader = shader.rawSources.fragmentShader.replace(
          ', 1.0 );',
          ', 0.5 );',
        );
        shader.fragmentSha256 = sha(shader.rawSources.fragmentShader);
      }
    }],
    ['visibleIds resource identity', (value) => {
      value.validation.addressChallenges.lanes[FEATURE]
        .shader.storageBindings[0].resourceId = 'replacement-visible-ids';
    }],
    ['challenge storage group', (value) => {
      value.validation.addressChallenges.lanes[FEATURE]
        .shader.storageBindings[0].group = 2;
    }],
    ['configuredFormat', (value) => {
      value.validation.addressChallenges.lanes[PORTABLE]
        .target.configuredFormat = 'unknown';
    }],
    ['transparent all-zero reset digest', (value) => {
      const reset = value.validation.addressChallenges.lanes[FEATURE].reset;
      reset.sha256 = 'f'.repeat(64);
      reset.expectedSha256 = reset.sha256;
    }],
    ['exactExpectedBytes', (value) => {
      value.validation.addressChallenges.lanes[PORTABLE].exactExpectedBytes = false;
    }],
    ['green encoding coverage', (value) => {
      value.validation.addressChallenges.lanes[FEATURE]
        .coverage.nonzeroEncodedChannelPixelCounts.green = 0;
    }],
    ['render parity cross-lane depth', (value) => {
      value.renderParity.lanes[FEATURE].depth.sha256 = '0'.repeat(64);
    }],
    ['completion static lifecycle equality', (value) => {
      value.summary.completionInvariant.staticLifecycleAtTimingEnd.meshUuids[FEATURE]
        = 'replacement-mesh';
    }],
    ['stable renderTargetTextureUuid', (value) => {
      value.summary.completionInvariant.renderTargetTextureUuidAtTimingEnd
        = 'replacement-target';
    }],
    ['stable cameraViewFnv64', (value) => {
      value.summary.completionInvariant.cameraViewFnv64AtTimingEnd
        = '1111111111111111';
    }],
    ['stable totalPipelineCacheEntries', (value) => {
      value.summary.completionInvariant.totalPipelineCacheEntriesAtTimingEnd += 1;
    }],
    ['strategySelectionsDuringTiming', (value) => {
      value.summary.completionInvariant.strategySelectionsDuringTiming -= 1;
    }],
    ['computeCallsDuringTiming', (value) => {
      value.summary.completionInvariant.computeCallsDuringTiming = 1;
      value.summary.completionInvariant.computeCallSerialAtTimingEnd += 1;
    }],
    ['row 17 laneId', (value) => {
      value.rows[17].laneId = value.rows[17].laneId === PORTABLE ? FEATURE : PORTABLE;
    }],
    ['row 18 schemaVersion', (value) => {
      value.rows[18].schemaVersion = 1;
    }],
    ['row 200 strategySelectionSerial', (value) => {
      value.rows[200].strategySelectionSerial += 1;
    }],
    ['row 479 render timestamp UID count', (value) => {
      value.rows[479].gpuRenderTimestampUidCount = 2;
    }],
    ['retained row count', (value) => {
      value.rows.pop();
      value.summary.rowCount -= 1;
    }],
    ['planned schedule sha256', (value) => {
      value.protocol.plannedScheduleSha256 = '0'.repeat(64);
    }],
  ];

  for (const [message, mutate] of cases) {
    const value = clone(pristine);
    mutate(value);
    const result = await validateFirstInstanceTrialEvidence(value);
    assert.equal(result.pass, false, `${message} unexpectedly passed`);
    assert.match(result.rejectionReasons.join('\n'), new RegExp(message), message);
  }
});

test('validator fails clearly when runner-only feature evidence is absent', async () => {
  const value = await artifact();
  delete value.environment;
  const result = await validateFirstInstanceTrialEvidence(value);
  assert.equal(result.pass, false);
  assert.match(result.rejectionReasons.join('\n'), /environment indirectFirstInstanceAvailable/);
});
