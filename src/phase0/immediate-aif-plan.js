import {
  INDEXED_INDIRECT_STRIDE_BYTES,
  INDEXED_INDIRECT_STRIDE_UINTS,
  createIndexedIndirectCommands,
} from '../culling/indexed-command-layout.js';
import { createFixedSubsetScenario } from '../scenes/fixed-subsets.js';

export const IMMEDIATE_AIF_PHASE0_LANES = Object.freeze(['A', 'I', 'F']);
export const IMMEDIATE_AIF_PHASE0_VISIBILITY_FRACTIONS = Object.freeze([0.99, 0.2]);
export const IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS = Object.freeze([
  'canonical',
  'S1',
  'S2',
]);
export const IMMEDIATE_AIF_UNUSED_ADDRESS = 0xffff_ffff;

export const IMMEDIATE_AIF_PHASE0_LANE_ORDERS = Object.freeze([
  Object.freeze(['A', 'I', 'F']),
  Object.freeze(['A', 'F', 'I']),
  Object.freeze(['F', 'I', 'A']),
  Object.freeze(['I', 'A', 'F']),
  Object.freeze(['I', 'F', 'A']),
  Object.freeze(['F', 'A', 'I']),
]);

const OBJECT_COUNT = 65_536;
const BUCKET_COUNT = 32;
const BUCKET_CAPACITY = OBJECT_COUNT / BUCKET_COUNT;
const BUCKET_COUNTS = Array.from({ length: BUCKET_COUNT }, () => BUCKET_CAPACITY);
const BUCKET_BASES = Array.from(
  { length: BUCKET_COUNT },
  (_, bucket) => bucket * BUCKET_CAPACITY,
);
const HIGH_VISIBLE_COUNTS = [
  ...Array.from({ length: 17 }, () => 2_028),
  ...Array.from({ length: 15 }, () => 2_027),
];
const LOW_VISIBLE_COUNTS = [
  ...Array.from({ length: 19 }, () => 410),
  ...Array.from({ length: 13 }, () => 409),
];
const COUNT_CLASS_GROUPS = [
  Array.from({ length: 17 }, (_, index) => index),
  [17, 18],
  Array.from({ length: 13 }, (_, index) => index + 19),
];

function deepFreeze(value) {
  if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const nested of Object.values(value)) deepFreeze(nested);
    Object.freeze(value);
  }
  return value;
}

function rotateGroupsRight(groups) {
  return groups.flatMap((group) => [group.at(-1), ...group.slice(0, -1)]);
}

function rotateGroupsLeft(groups) {
  return groups.flatMap((group) => [...group.slice(1), group[0]]);
}

function createSchedule(id, sourceBucketByDraw) {
  return deepFreeze({
    id,
    sourceBucketByDraw,
    sourceBaseByDraw: sourceBucketByDraw.map((bucket) => BUCKET_BASES[bucket]),
  });
}

export const IMMEDIATE_AIF_PHASE0_COUNT_CLASS_GROUPS = deepFreeze(
  COUNT_CLASS_GROUPS.map((group) => [...group]),
);

export const IMMEDIATE_AIF_PHASE0_SCHEDULES = deepFreeze({
  canonical: createSchedule(
    'canonical',
    Array.from({ length: BUCKET_COUNT }, (_, bucket) => bucket),
  ),
  S1: createSchedule('S1', rotateGroupsRight(COUNT_CLASS_GROUPS)),
  S2: createSchedule('S2', rotateGroupsLeft(COUNT_CLASS_GROUPS)),
});

export const IMMEDIATE_AIF_PHASE0_WORKLOAD = deepFreeze({
  objectCount: OBJECT_COUNT,
  bucketCount: BUCKET_COUNT,
  geometryTier: 'medium',
  layout: 'baseline',
  seed: 0xb1ad_2026,
  bucketCapacity: BUCKET_CAPACITY,
  bucketCounts: BUCKET_COUNTS,
  bucketBases: BUCKET_BASES,
  drawCount: BUCKET_COUNT,
  indirectStrideUints: INDEXED_INDIRECT_STRIDE_UINTS,
  indirectStrideBytes: INDEXED_INDIRECT_STRIDE_BYTES,
  commandUint32Length: BUCKET_COUNT * INDEXED_INDIRECT_STRIDE_UINTS,
  commandByteLength: BUCKET_COUNT * INDEXED_INDIRECT_STRIDE_BYTES,
  indirectOffsets: Array.from(
    { length: BUCKET_COUNT },
    (_, bucket) => bucket * INDEXED_INDIRECT_STRIDE_BYTES,
  ),
  visibilityLevels: [
    {
      id: 'v99',
      fraction: 0.99,
      expectedVisibleCount: 64_881,
      visibleCounts: HIGH_VISIBLE_COUNTS,
    },
    {
      id: 'v20',
      fraction: 0.2,
      expectedVisibleCount: 13_107,
      visibleCounts: LOW_VISIBLE_COUNTS,
    },
  ],
});

export const IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION = deepFreeze({
  viewport: { width: 1_280, height: 720, devicePixelRatio: 1 },
  renderer: {
    antialias: false,
    samples: 0,
    powerPreference: 'high-performance',
    forceWebGL: false,
    alpha: true,
    depth: true,
    stencil: false,
    logarithmicDepthBuffer: false,
    reversedDepthBuffer: true,
    trackTimestamp: false,
    autoClear: true,
    autoClearColor: true,
    autoClearDepth: true,
    autoClearStencil: true,
    toneMapping: 0,
    toneMappingExposure: 1,
    sortObjects: false,
    outputColorSpace: 'srgb',
  },
  camera: {
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
  },
  clear: {
    color: 0x030711,
    alpha: 1,
    colorEnabled: true,
    depthEnabled: true,
    stencilEnabled: false,
    reversedDepthClearValue: 0,
  },
  material: {
    type: 'MeshStandardNodeMaterial',
    color: 0x68a7f2,
    roughness: 0.68,
    metalness: 0.08,
    emissive: 0x000000,
    emissiveIntensity: 1,
    opacity: 1,
    transparent: false,
    side: 0,
    depthTest: true,
    depthWrite: true,
    blending: 1,
    blendSrc: 204,
    blendDst: 205,
    blendEquation: 100,
    blendSrcAlpha: null,
    blendDstAlpha: null,
    blendEquationAlpha: null,
    blendColor: [0, 0, 0],
    blendAlpha: 0,
    premultipliedAlpha: false,
    colorWrite: true,
    depthFunc: 3,
    vertexColors: false,
    toneMapped: true,
    wireframe: false,
    alphaTest: 0,
    alphaHash: false,
    alphaToCoverage: false,
    forceSinglePass: false,
    polygonOffset: false,
    polygonOffsetFactor: 0,
    polygonOffsetUnits: 0,
    stencilWrite: false,
    stencilWriteMask: 255,
    stencilFunc: 519,
    stencilRef: 0,
    stencilFuncMask: 255,
    stencilFail: 7_680,
    stencilZFail: 7_680,
    stencilZPass: 7_680,
    fog: true,
    lights: true,
    flatShading: false,
    dithering: false,
    precision: null,
    bumpScale: 1,
    normalScale: [1, 1],
    normalMapType: 0,
    displacementScale: 1,
    displacementBias: 0,
    aoMapIntensity: 1,
    lightMapIntensity: 1,
    envMapIntensity: 1,
    envMapRotation: [0, 0, 0, 'XYZ'],
    clipIntersection: false,
    clipShadows: false,
    shadowSide: null,
    clippingPlanes: null,
    wireframeLinewidth: 1,
    wireframeLinecap: 'round',
    wireframeLinejoin: 'round',
    maps: {
      map: null,
      alphaMap: null,
      aoMap: null,
      bumpMap: null,
      normalMap: null,
      displacementMap: null,
      roughnessMap: null,
      metalnessMap: null,
      emissiveMap: null,
      envMap: null,
      lightMap: null,
    },
    visible: true,
  },
  lights: {
    hemisphere: {
      type: 'HemisphereLight',
      name: 'phase0-hemisphere-light',
      skyColor: 0x9bc5ff,
      groundColor: 0x182038,
      intensity: 2.2,
      position: [0, 1, 0],
      worldPosition: [0, 1, 0],
      quaternion: [0, 0, 0, 1],
      scale: [1, 1, 1],
      matrix: [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 1, 0, 1,
      ],
      matrixWorld: [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 1, 0, 1,
      ],
      matrixAutoUpdate: false,
      matrixWorldAutoUpdate: false,
      visible: true,
      layersMask: 1,
      castShadow: false,
      receiveShadow: false,
      frustumCulled: true,
      renderOrder: 0,
      effectiveNormalizedDirection: [0, 1, 0],
    },
    directional: {
      type: 'DirectionalLight',
      name: 'phase0-directional-light',
      color: 0xffffff,
      intensity: 3.2,
      position: [20, 35, 60],
      worldPosition: [20, 35, 60],
      quaternion: [0, 0, 0, 1],
      scale: [1, 1, 1],
      matrix: [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        20, 35, 60, 1,
      ],
      matrixWorld: [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        20, 35, 60, 1,
      ],
      matrixAutoUpdate: false,
      matrixWorldAutoUpdate: false,
      visible: true,
      layersMask: 1,
      castShadow: false,
      receiveShadow: false,
      frustumCulled: true,
      renderOrder: 0,
      targetDirection: [
        -0.27668578554642986,
        -0.48420012470625223,
        -0.8300573566392895,
      ],
      effectiveNormalizedDirection: [
        0.27668578554642986,
        0.48420012470625223,
        0.8300573566392895,
      ],
      target: {
        type: 'Object3D',
        name: 'phase0-directional-light-target',
        addedToProductionScene: true,
        position: [0, 0, 0],
        worldPosition: [0, 0, 0],
        quaternion: [0, 0, 0, 1],
        scale: [1, 1, 1],
        matrix: [
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1,
        ],
        matrixWorld: [
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1,
        ],
        matrixAutoUpdate: false,
        matrixWorldAutoUpdate: false,
        visible: true,
        layersMask: 1,
        castShadow: false,
        receiveShadow: false,
        frustumCulled: true,
        renderOrder: 0,
      },
    },
  },
  targets: {
    production: {
      width: 1_280,
      height: 720,
      depth: 1,
      count: 1,
      samples: 0,
      colorFormat: 'rgba8unorm',
      colorType: 'uint8',
      colorSpace: '',
      depthFormat: 'depth32float',
      depthBuffer: true,
      stencilBuffer: false,
      resolveDepthBuffer: true,
      resolveStencilBuffer: true,
      multiview: false,
      useArrayDepthTexture: false,
      colorTexture: {
        format: 1_023,
        type: 1_009,
        colorSpace: '',
        minFilter: 1_006,
        magFilter: 1_006,
        generateMipmaps: false,
        flipY: false,
        internalFormat: null,
        anisotropy: 1,
        wrapS: 1_001,
        wrapT: 1_001,
        wrapR: null,
      },
      depthTexture: {
        format: 1_026,
        type: 1_015,
        colorSpace: '',
        minFilter: 1_003,
        magFilter: 1_003,
        generateMipmaps: false,
        flipY: false,
        internalFormat: null,
        anisotropy: 1,
        wrapS: 1_001,
        wrapT: 1_001,
        wrapR: null,
      },
      gpuColorDescriptor: {
        label: '',
        size: { width: 1_280, height: 720, depthOrArrayLayers: 1 },
        mipLevelCount: 1,
        sampleCount: 1,
        dimension: '2d',
        format: 'rgba8unorm',
        usage: 23,
        viewFormats: [],
        textureBindingViewDimension: null,
      },
      gpuDepthDescriptor: {
        label: '',
        size: { width: 1_280, height: 720, depthOrArrayLayers: 1 },
        mipLevelCount: 1,
        sampleCount: 1,
        dimension: '2d',
        format: 'depth32float',
        usage: 23,
        viewFormats: [],
        textureBindingViewDimension: null,
      },
    },
    objectId: {
      width: 1_280,
      height: 720,
      depth: 1,
      count: 1,
      samples: 0,
      colorFormat: 'rgba8unorm',
      colorType: 'uint8',
      colorSpace: '',
      depthFormat: 'depth32float',
      depthBuffer: true,
      stencilBuffer: false,
      resolveDepthBuffer: true,
      resolveStencilBuffer: true,
      multiview: false,
      useArrayDepthTexture: false,
      colorTexture: {
        format: 1_023,
        type: 1_009,
        colorSpace: '',
        minFilter: 1_006,
        magFilter: 1_006,
        generateMipmaps: false,
        flipY: false,
        internalFormat: null,
        anisotropy: 1,
        wrapS: 1_001,
        wrapT: 1_001,
        wrapR: null,
      },
      depthTexture: {
        format: 1_026,
        type: 1_015,
        colorSpace: '',
        minFilter: 1_003,
        magFilter: 1_003,
        generateMipmaps: false,
        flipY: false,
        internalFormat: null,
        anisotropy: 1,
        wrapS: 1_001,
        wrapT: 1_001,
        wrapR: null,
      },
      gpuColorDescriptor: {
        label: '',
        size: { width: 1_280, height: 720, depthOrArrayLayers: 1 },
        mipLevelCount: 1,
        sampleCount: 1,
        dimension: '2d',
        format: 'rgba8unorm',
        usage: 23,
        viewFormats: [],
        textureBindingViewDimension: null,
      },
      gpuDepthDescriptor: {
        label: '',
        size: { width: 1_280, height: 720, depthOrArrayLayers: 1 },
        mipLevelCount: 1,
        sampleCount: 1,
        dimension: '2d',
        format: 'depth32float',
        usage: 23,
        viewFormats: [],
        textureBindingViewDimension: null,
      },
    },
    address: {
      width: 256,
      height: 256,
      depth: 1,
      count: 1,
      samples: 0,
      colorFormat: 'rgba8unorm',
      colorType: 'uint8',
      colorSpace: '',
      depthBuffer: false,
      stencilBuffer: false,
      resolveDepthBuffer: true,
      resolveStencilBuffer: true,
      multiview: false,
      useArrayDepthTexture: false,
      colorTexture: {
        format: 1_023,
        type: 1_009,
        colorSpace: '',
        minFilter: 1_006,
        magFilter: 1_006,
        generateMipmaps: false,
        flipY: false,
        internalFormat: null,
        anisotropy: 1,
        wrapS: 1_001,
        wrapT: 1_001,
        wrapR: null,
      },
      depthTexture: null,
      gpuColorDescriptor: {
        label: '',
        size: { width: 256, height: 256, depthOrArrayLayers: 1 },
        mipLevelCount: 1,
        sampleCount: 1,
        dimension: '2d',
        format: 'rgba8unorm',
        usage: 23,
        viewFormats: [],
        textureBindingViewDimension: null,
      },
      gpuDepthDescriptor: null,
    },
  },
});

export const IMMEDIATE_AIF_PHASE0_PLAN = deepFreeze({
  schemaVersion: 1,
  kind: 'webgpu-immediate-aif-phase0-technical-canary',
  executionMode: 'technical-canary',
  analysisEligible: false,
  efficacyAnalysisAllowed: false,
  numericalDecision: null,
  lanes: IMMEDIATE_AIF_PHASE0_LANES,
  laneOrders: IMMEDIATE_AIF_PHASE0_LANE_ORDERS,
  scheduleIds: IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS,
  schedules: IMMEDIATE_AIF_PHASE0_SCHEDULES,
  countClassGroups: IMMEDIATE_AIF_PHASE0_COUNT_CLASS_GROUPS,
  workload: IMMEDIATE_AIF_PHASE0_WORKLOAD,
  renderConfiguration: IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION,
});

function exactArray(actual, expected) {
  return actual?.length === expected.length
    && expected.every((value, index) => actual[index] === value);
}

function requireExactUint32(value, expected, label) {
  if (!(value instanceof Uint32Array) || !exactArray(value, expected)) {
    throw new RangeError(`${label} does not match the frozen Phase 0 workload.`);
  }
}

function exactTypedBytes(actual, expected) {
  if (!ArrayBuffer.isView(actual)
    || !ArrayBuffer.isView(expected)
    || actual.constructor !== expected.constructor
    || actual.byteLength !== expected.byteLength) return false;
  const actualBytes = new Uint8Array(actual.buffer, actual.byteOffset, actual.byteLength);
  const expectedBytes = new Uint8Array(expected.buffer, expected.byteOffset, expected.byteLength);
  return actualBytes.every((value, index) => value === expectedBytes[index]);
}

function visibilityLevelForScenario(scenario) {
  return IMMEDIATE_AIF_PHASE0_WORKLOAD.visibilityLevels.find(
    (level) => level.fraction === scenario?.visibilityFraction,
  );
}

export function validateImmediateAifPhase0Lane(lane, label = 'lane') {
  if (!IMMEDIATE_AIF_PHASE0_LANES.includes(lane)) {
    throw new RangeError(`${label} must be A, I, or F.`);
  }
  return lane;
}

export function validateImmediateAifPhase0Scenario(scenario, expectedScenario = null) {
  if (!scenario || typeof scenario !== 'object') {
    throw new TypeError('Phase 0 scenario must be an object.');
  }
  const level = visibilityLevelForScenario(scenario);
  if (!level) {
    throw new RangeError('Phase 0 scenario visibility must be exactly 0.99 or 0.2.');
  }
  if (scenario.objectCount !== OBJECT_COUNT
    || scenario.bucketCount !== BUCKET_COUNT
    || scenario.layout !== IMMEDIATE_AIF_PHASE0_WORKLOAD.layout
    || scenario.scenarioId !== level.id
    || scenario.seed !== IMMEDIATE_AIF_PHASE0_WORKLOAD.seed
    || scenario.geometryTier !== IMMEDIATE_AIF_PHASE0_WORKLOAD.geometryTier) {
    throw new RangeError('Scenario dimensions or layout do not match the frozen Phase 0 workload.');
  }
  requireExactUint32(scenario.bucketCounts, BUCKET_COUNTS, 'scenario.bucketCounts');
  requireExactUint32(scenario.bucketBases, BUCKET_BASES, 'scenario.bucketBases');
  requireExactUint32(scenario.visibleCounts, level.visibleCounts, 'scenario.visibleCounts');
  if (!(scenario.expectedVisibleIds instanceof Uint32Array)
    || scenario.expectedVisibleCount !== level.expectedVisibleCount
    || scenario.expectedVisibleIds.length !== level.expectedVisibleCount) {
    throw new RangeError('Scenario visible IDs do not match the frozen Phase 0 workload.');
  }
  if (!(scenario.objectBuckets instanceof Uint32Array)
    || scenario.objectBuckets.length !== OBJECT_COUNT
    || scenario.objectBuckets.some((bucket, objectId) => (
      bucket !== Math.floor(objectId / BUCKET_CAPACITY)
    ))) {
    throw new RangeError(`scenario.objectBuckets must contain ${OBJECT_COUNT} uint32 values.`);
  }
  if (!(scenario.cullOrder instanceof Uint32Array)
    || scenario.cullOrder.length !== OBJECT_COUNT
    || scenario.cullOrder.some((value, index) => value !== index)) {
    throw new RangeError('scenario.cullOrder must be the frozen identity order.');
  }
  if (!(scenario.matrices instanceof Float32Array)
    || scenario.matrices.length !== OBJECT_COUNT * 16
    || !(scenario.bounds instanceof Float32Array)
    || scenario.bounds.length !== OBJECT_COUNT * 4) {
    throw new RangeError('Scenario matrix/bounds buffers do not match the frozen workload.');
  }
  let visibleCursor = 0;
  for (let bucket = 0; bucket < BUCKET_COUNT; bucket += 1) {
    const base = BUCKET_BASES[bucket];
    for (let local = 0; local < level.visibleCounts[bucket]; local += 1) {
      if (scenario.expectedVisibleIds[visibleCursor] !== base + local) {
        throw new RangeError('Scenario visible IDs are not the frozen per-bucket prefix.');
      }
      visibleCursor += 1;
    }
  }
  if (expectedScenario !== null) {
    for (const name of [
      'scenarioId', 'seed', 'geometryTier', 'objectCount', 'bucketCount',
      'visibilityFraction', 'expectedVisibleCount', 'layout',
    ]) {
      if (!Object.is(scenario[name], expectedScenario[name])) {
        throw new Error(`scenario.${name} differs from the regenerated canonical workload.`);
      }
    }
    for (const name of [
      'bucketCounts', 'bucketBases', 'visibleCounts', 'objectBuckets', 'cullOrder',
      'matrices', 'bounds', 'expectedVisibleIds',
    ]) {
      if (!exactTypedBytes(scenario[name], expectedScenario[name])) {
        throw new Error(`scenario.${name} bytes differ from the regenerated canonical workload.`);
      }
    }
  }
  return level;
}

export function createImmediateAifPhase0Scenarios({ geometrySpheres } = {}) {
  if (!Array.isArray(geometrySpheres) || geometrySpheres.length !== BUCKET_COUNT) {
    throw new RangeError(`geometrySpheres must contain exactly ${BUCKET_COUNT} spheres.`);
  }
  const scenarios = {};
  for (const level of IMMEDIATE_AIF_PHASE0_WORKLOAD.visibilityLevels) {
    const generated = createFixedSubsetScenario({
      objectCount: OBJECT_COUNT,
      bucketCount: BUCKET_COUNT,
      visibilityFraction: level.fraction,
      geometrySpheres,
      seed: IMMEDIATE_AIF_PHASE0_WORKLOAD.seed,
      layout: IMMEDIATE_AIF_PHASE0_WORKLOAD.layout,
    });
    const scenario = Object.freeze({
      ...generated,
      scenarioId: level.id,
      seed: IMMEDIATE_AIF_PHASE0_WORKLOAD.seed,
      geometryTier: IMMEDIATE_AIF_PHASE0_WORKLOAD.geometryTier,
    });
    validateImmediateAifPhase0Scenario(scenario);
    scenarios[level.id] = scenario;
  }
  return Object.freeze(scenarios);
}

function resolveSchedule(scheduleId) {
  if (!IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS.includes(scheduleId)) {
    throw new RangeError('scheduleId must be canonical, S1, or S2.');
  }
  return IMMEDIATE_AIF_PHASE0_SCHEDULES[scheduleId];
}

function sourceBasesForScenario(scenario, schedule) {
  return Uint32Array.from(
    schedule.sourceBucketByDraw,
    (sourceBucket) => scenario.bucketBases[sourceBucket],
  );
}

function resolveFirstIndexes(sourceGeometries, firstIndexes) {
  if (firstIndexes !== null) {
    if (!(firstIndexes instanceof Uint32Array) || firstIndexes.length !== BUCKET_COUNT) {
      throw new RangeError(`firstIndexes must be a Uint32Array of length ${BUCKET_COUNT}.`);
    }
    return firstIndexes.slice();
  }
  const resolved = new Uint32Array(BUCKET_COUNT);
  let cursor = 0;
  for (let bucket = 0; bucket < BUCKET_COUNT; bucket += 1) {
    const indexCount = sourceGeometries[bucket]?.index?.count;
    if (!Number.isInteger(indexCount) || indexCount <= 0) {
      throw new Error(`sourceGeometries[${bucket}] must be nonempty and indexed.`);
    }
    if (cursor > 0xffff_ffff) {
      throw new RangeError('Cumulative firstIndex exceeds uint32 capacity.');
    }
    resolved[bucket] = cursor;
    cursor += indexCount;
    if (!Number.isSafeInteger(cursor)) {
      throw new RangeError('Cumulative index count exceeds JavaScript safe integer capacity.');
    }
  }
  return resolved;
}

export function createImmediateAifCommandOracle({
  lane,
  scenario,
  sourceGeometries,
  firstIndexes = null,
  scheduleId = 'canonical',
} = {}) {
  validateImmediateAifPhase0Lane(lane);
  validateImmediateAifPhase0Scenario(scenario);
  if (!Array.isArray(sourceGeometries) || sourceGeometries.length !== BUCKET_COUNT) {
    throw new RangeError(`sourceGeometries must contain exactly ${BUCKET_COUNT} geometries.`);
  }
  const schedule = resolveSchedule(scheduleId);
  const resolvedFirstIndexes = resolveFirstIndexes(sourceGeometries, firstIndexes);
  const sourceBaseByDraw = sourceBasesForScenario(scenario, schedule);
  const firstInstances = lane === 'F' ? sourceBaseByDraw : null;
  const initialLayout = createIndexedIndirectCommands(
    sourceGeometries,
    scenario.bucketCounts,
    null,
    resolvedFirstIndexes,
    firstInstances,
  );
  const postCullLayout = createIndexedIndirectCommands(
    sourceGeometries,
    scenario.bucketCounts,
    scenario.visibleCounts,
    resolvedFirstIndexes,
    firstInstances,
  );
  if (initialLayout.commands.byteLength !== IMMEDIATE_AIF_PHASE0_WORKLOAD.commandByteLength
    || postCullLayout.commands.byteLength !== IMMEDIATE_AIF_PHASE0_WORKLOAD.commandByteLength
    || !exactArray(postCullLayout.offsets, IMMEDIATE_AIF_PHASE0_WORKLOAD.indirectOffsets)) {
    throw new Error('Command layout does not match the frozen 32-draw Phase 0 shape.');
  }
  return {
    kind: 'webgpu-immediate-aif-command-oracle',
    lane,
    scheduleId,
    sourceBaseByDraw,
    bucketBaseAttributeByDraw: lane === 'A' ? sourceBaseByDraw : null,
    indirectImmediateBases: lane === 'I' ? sourceBaseByDraw : null,
    indirectFirstInstances: lane === 'F' ? sourceBaseByDraw : null,
    firstIndexes: resolvedFirstIndexes,
    offsets: postCullLayout.offsets,
    recordCount: postCullLayout.recordCount,
    initialCommands: initialLayout.commands,
    postCullCommands: postCullLayout.commands,
  };
}

function createVisibleIdPacking(scenario) {
  const packed = new Uint32Array(OBJECT_COUNT);
  packed.fill(IMMEDIATE_AIF_UNUSED_ADDRESS);
  const cursors = new Uint32Array(BUCKET_COUNT);
  let previousObjectId = -1;
  for (const objectId of scenario.expectedVisibleIds) {
    if (objectId <= previousObjectId || objectId >= OBJECT_COUNT) {
      throw new RangeError('scenario.expectedVisibleIds must be strictly ascending and in range.');
    }
    previousObjectId = objectId;
    const bucket = scenario.objectBuckets[objectId];
    if (bucket >= BUCKET_COUNT) {
      throw new RangeError(`scenario.objectBuckets[${objectId}] is out of range.`);
    }
    const cursor = cursors[bucket];
    if (cursor >= scenario.visibleCounts[bucket]) {
      throw new Error(`Visible-ID packing overflowed bucket ${bucket}.`);
    }
    packed[scenario.bucketBases[bucket] + cursor] = objectId;
    cursors[bucket] += 1;
  }
  requireExactUint32(cursors, scenario.visibleCounts, 'packed visible counts');
  return packed;
}

export function createImmediateAifAddressOracle({
  lane,
  scenario,
  scheduleId = 'canonical',
} = {}) {
  validateImmediateAifPhase0Lane(lane);
  const level = validateImmediateAifPhase0Scenario(scenario);
  const schedule = resolveSchedule(scheduleId);
  const sourceBaseByDraw = sourceBasesForScenario(scenario, schedule);
  const canonicalVisibleIds = createVisibleIdPacking(scenario);
  const activeTargetAddresses = new Uint32Array(level.expectedVisibleCount);
  const activeSourceAddresses = new Uint32Array(level.expectedVisibleCount);
  const builtinInstanceIndices = new Uint32Array(level.expectedVisibleCount);
  const baseOperands = new Uint32Array(level.expectedVisibleCount);
  const objectIds = new Uint32Array(level.expectedVisibleCount);
  const targetAddressToSourceAddress = new Uint32Array(OBJECT_COUNT);
  const expectedOutputObjectIds = new Uint32Array(OBJECT_COUNT);
  targetAddressToSourceAddress.fill(IMMEDIATE_AIF_UNUSED_ADDRESS);
  expectedOutputObjectIds.fill(IMMEDIATE_AIF_UNUSED_ADDRESS);
  const sourceAddressSeen = new Uint8Array(OBJECT_COUNT);
  const draws = [];
  let cursor = 0;

  for (let draw = 0; draw < BUCKET_COUNT; draw += 1) {
    const sourceBucket = schedule.sourceBucketByDraw[draw];
    const submittedCount = scenario.visibleCounts[draw];
    if (scenario.visibleCounts[sourceBucket] !== submittedCount) {
      throw new Error(
        `${scheduleId} is unsafe for visibility ${level.id}: draw ${draw} changes count class.`,
      );
    }
    const targetBase = scenario.bucketBases[draw];
    const sourceBase = sourceBaseByDraw[draw];
    if (sourceBase + submittedCount
      > scenario.bucketBases[sourceBucket] + scenario.bucketCounts[sourceBucket]) {
      throw new Error(`${scheduleId} source range for draw ${draw} exceeds its bucket slice.`);
    }
    draws.push(Object.freeze({
      draw,
      sourceBucket,
      targetBase,
      sourceBase,
      submittedCount,
    }));

    for (let localInstance = 0; localInstance < submittedCount; localInstance += 1) {
      const targetAddress = targetBase + localInstance;
      const sourceAddress = sourceBase + localInstance;
      const builtinInstanceIndex = lane === 'F' ? sourceAddress : localInstance;
      const baseOperand = lane === 'F' ? 0 : sourceBase;
      if (builtinInstanceIndex + baseOperand !== sourceAddress) {
        throw new Error(`Address oracle failed to reconstruct ${lane} draw ${draw}.`);
      }
      if (sourceAddressSeen[sourceAddress] !== 0) {
        throw new Error(`${scheduleId} reads active source address ${sourceAddress} twice.`);
      }
      const objectId = canonicalVisibleIds[sourceAddress];
      if (objectId === IMMEDIATE_AIF_UNUSED_ADDRESS) {
        throw new Error(`${scheduleId} reads padding at active source address ${sourceAddress}.`);
      }
      sourceAddressSeen[sourceAddress] = 1;
      activeTargetAddresses[cursor] = targetAddress;
      activeSourceAddresses[cursor] = sourceAddress;
      builtinInstanceIndices[cursor] = builtinInstanceIndex;
      baseOperands[cursor] = baseOperand;
      objectIds[cursor] = objectId;
      targetAddressToSourceAddress[targetAddress] = sourceAddress;
      expectedOutputObjectIds[targetAddress] = objectId;
      cursor += 1;
    }
  }
  if (cursor !== level.expectedVisibleCount) {
    throw new Error('Address oracle did not cover the frozen active invocation count.');
  }

  return {
    kind: 'webgpu-immediate-aif-address-oracle',
    lane,
    scheduleId,
    visibilityId: level.id,
    sourceBaseByDraw,
    draws: Object.freeze(draws),
    canonicalVisibleIds,
    activeTargetAddresses,
    activeSourceAddresses,
    builtinInstanceIndices,
    baseOperands,
    objectIds,
    targetAddressToSourceAddress,
    expectedOutputObjectIds,
  };
}
