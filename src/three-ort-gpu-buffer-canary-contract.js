export const ORT_GPU_BUFFER_CANARY_KIND =
  'three-webgpu-onnxruntime-borrowed-gpu-buffer-correctness-canary';

export const ORT_MODEL_BASE64 =
  'CAMSBmNoZW50YTpwChUKAVgKAVcSAVkaBW11bF8xIgNNdWwSCG11bCB0ZXN0KiMIAwgCEAEiGAAAgD8AAABAAABAQAAAgEAAAKBAAADAQEIBV1oTCgFYEg4KDAgBEggKAggDCgIIAmITCgFZEg4KDAgBEggKAggDCgIIAkIECgAQBw==';
export const ORT_MODEL_BYTES = 130;
export const ORT_MODEL_SHA256 =
  '71f431c4e9321ec6fbeb158d02ed240459a7dcc98673fa79a4f439ce42efaf10';

export const ORT_TARGET_PHYSICAL_BYTES = 32;
export const ORT_TARGET_LOGICAL_BYTES = 24;
export const ORT_TARGET_ITEM_SIZE = 1;
export const ORT_TARGET_ITEM_COUNT = 6;

export const ORT_ORACLE_SIZE = Object.freeze({ width: 256, height: 256 });
export const ORT_ORACLE_SENTINEL = 0x4f525433;
export const ORT_MARKER_NDC_SIZE = 0.18;
export const ORT_ORACLE_MIN_PIXELS_PER_MARKER = 300;
export const ORT_ORACLE_MAX_PIXELS_PER_MARKER = 800;
export const ORT_ORACLE_SPATIAL_TOLERANCE_PIXELS = 2;

export const ORT_CANARY_RUNS = Object.freeze([
  Object.freeze({
    runIndex: 0,
    label: 'run-1',
    input: Object.freeze([-0.75, -0.375, 0.25, -0.1875, 0, 0.125]),
    expectedOutput: Object.freeze([-0.75, -0.75, 0.75, -0.75, 0, 0.75]),
    expectedBits: Object.freeze([
      0xbf400000, 0xbf400000,
      0x3f400000, 0xbf400000,
      0x00000000, 0x3f400000,
    ]),
  }),
  Object.freeze({
    runIndex: 1,
    label: 'run-2',
    input: Object.freeze([-0.625, -0.25, 0.125, -0.125, 0.125, 0.0625]),
    expectedOutput: Object.freeze([-0.625, -0.5, 0.375, -0.5, 0.625, 0.375]),
    expectedBits: Object.freeze([
      0xbf200000, 0xbf000000,
      0x3ec00000, 0xbf000000,
      0x3f200000, 0x3ec00000,
    ]),
  }),
]);

function requireRun(runIndex) {
  if (!Number.isInteger(runIndex) || runIndex < 0 || runIndex >= ORT_CANARY_RUNS.length) {
    throw new RangeError(`ORT canary runIndex must be 0 or 1; received ${runIndex}.`);
  }
  return ORT_CANARY_RUNS[runIndex];
}

function tupleKey(tuple) {
  return tuple.join(',');
}

function compareTupleKeys(left, right) {
  const leftWords = left.split(',').map(Number);
  const rightWords = right.split(',').map(Number);
  for (let index = 0; index < 4; index += 1) {
    if (leftWords[index] !== rightWords[index]) return leftWords[index] - rightWords[index];
  }
  return 0;
}

export function decodeOrtModelBytes(base64 = ORT_MODEL_BASE64) {
  if (typeof base64 !== 'string' || base64.length === 0) {
    throw new TypeError('ORT model base64 must be a non-empty string.');
  }
  if (typeof globalThis.atob !== 'function') {
    throw new Error('This environment does not provide atob().');
  }

  const binary = globalThis.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

export function expectedOrtGpuBufferTuples(runIndex) {
  const run = requireRun(runIndex);
  return [0, 1, 2].map((instance) => [
    run.expectedBits[instance * 2],
    run.expectedBits[instance * 2 + 1],
    instance,
    ORT_ORACLE_SENTINEL,
  ]);
}

export function ndcToOraclePixel(
  x,
  y,
  width = ORT_ORACLE_SIZE.width,
  height = ORT_ORACLE_SIZE.height,
) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    throw new TypeError('NDC coordinates must be finite numbers.');
  }
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new RangeError('Oracle dimensions must be positive integers.');
  }

  return {
    x: ((x + 1) * 0.5 * width) - 0.5,
    y: ((1 - y) * 0.5 * height) - 0.5,
  };
}

export function analyzeOrtGpuBufferOracle(
  words,
  runIndex,
  {
    width = ORT_ORACLE_SIZE.width,
    height = ORT_ORACLE_SIZE.height,
    minimumPixelsPerMarker = ORT_ORACLE_MIN_PIXELS_PER_MARKER,
    maximumPixelsPerMarker = ORT_ORACLE_MAX_PIXELS_PER_MARKER,
    spatialTolerancePixels = ORT_ORACLE_SPATIAL_TOLERANCE_PIXELS,
  } = {},
) {
  const run = requireRun(runIndex);
  if (!(words instanceof Uint32Array)) {
    throw new TypeError('ORT GPU-buffer oracle readback must be a Uint32Array.');
  }
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new RangeError('Oracle dimensions must be positive integers.');
  }
  if (words.length !== width * height * 4) {
    throw new RangeError(
      `ORT GPU-buffer oracle has ${words.length} words; expected ${width * height * 4}.`,
    );
  }
  if (!Number.isInteger(minimumPixelsPerMarker) || minimumPixelsPerMarker <= 0) {
    throw new RangeError('minimumPixelsPerMarker must be a positive integer.');
  }
  if (!Number.isInteger(maximumPixelsPerMarker)
    || maximumPixelsPerMarker < minimumPixelsPerMarker) {
    throw new RangeError('maximumPixelsPerMarker must be an integer at least as large as the minimum.');
  }
  if (!Number.isFinite(spatialTolerancePixels) || spatialTolerancePixels < 0) {
    throw new RangeError('spatialTolerancePixels must be a non-negative finite number.');
  }

  const expectedTuples = expectedOrtGpuBufferTuples(runIndex);
  const expectedByKey = new Map(expectedTuples.map((tuple, instance) => [
    tupleKey(tuple),
    {
      instance,
      tuple,
      count: 0,
      sumX: 0,
      sumY: 0,
      minX: width,
      minY: height,
      maxX: -1,
      maxY: -1,
    },
  ]));
  const unexpectedMap = new Map();
  let backgroundPixels = 0;

  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const offset = pixel * 4;
    const tuple = [words[offset], words[offset + 1], words[offset + 2], words[offset + 3]];
    const isBackground = tuple.every((value) => value === 0);
    if (isBackground) {
      backgroundPixels += 1;
      continue;
    }

    const x = pixel % width;
    const y = Math.floor(pixel / width);
    const key = tupleKey(tuple);
    const expected = expectedByKey.get(key);
    if (expected) {
      expected.count += 1;
      expected.sumX += x;
      expected.sumY += y;
      expected.minX = Math.min(expected.minX, x);
      expected.minY = Math.min(expected.minY, y);
      expected.maxX = Math.max(expected.maxX, x);
      expected.maxY = Math.max(expected.maxY, y);
      continue;
    }

    const unexpected = unexpectedMap.get(key) ?? {
      tuple,
      count: 0,
      firstPixel: { x, y },
    };
    unexpected.count += 1;
    unexpectedMap.set(key, unexpected);
  }

  const tuples = [...expectedByKey.values()].map((record) => {
    const expectedPosition = [
      run.expectedOutput[record.instance * 2],
      run.expectedOutput[record.instance * 2 + 1],
    ];
    const expectedCentroid = ndcToOraclePixel(
      expectedPosition[0],
      expectedPosition[1],
      width,
      height,
    );
    const centroid = record.count > 0
      ? { x: record.sumX / record.count, y: record.sumY / record.count }
      : null;
    const centroidError = centroid === null
      ? null
      : {
        x: Math.abs(centroid.x - expectedCentroid.x),
        y: Math.abs(centroid.y - expectedCentroid.y),
      };
    const sufficient = record.count >= minimumPixelsPerMarker;
    const notOversized = record.count <= maximumPixelsPerMarker;
    const spatialMismatch = centroidError === null
      || centroidError.x > spatialTolerancePixels
      || centroidError.y > spatialTolerancePixels;

    return {
      instance: record.instance,
      tuple: record.tuple,
      expectedPosition,
      count: record.count,
      sufficient,
      notOversized,
      centroid,
      expectedCentroid,
      centroidError,
      spatialMismatch,
      bounds: record.count > 0
        ? {
          minX: record.minX,
          minY: record.minY,
          maxX: record.maxX,
          maxY: record.maxY,
          width: record.maxX - record.minX + 1,
          height: record.maxY - record.minY + 1,
        }
        : null,
    };
  });
  const unexpected = [...unexpectedMap.entries()]
    .sort(([left], [right]) => compareTupleKeys(left, right))
    .map(([, record]) => record);
  const spatialMismatches = tuples
    .filter((record) => record.spatialMismatch)
    .map((record) => record.instance);
  const coveredPixels = tuples.reduce((sum, record) => sum + record.count, 0);

  return {
    runIndex,
    runLabel: run.label,
    width,
    height,
    totalPixels: width * height,
    backgroundPixels,
    coveredPixels,
    minimumPixelsPerMarker,
    maximumPixelsPerMarker,
    spatialTolerancePixels,
    markerNdcSize: ORT_MARKER_NDC_SIZE,
    expectedPixelTuples: expectedTuples,
    tuples,
    unexpected,
    spatialMismatches,
    exact: backgroundPixels > 0
      && unexpected.length === 0
      && tuples.every((record) => (
        record.sufficient && record.notOversized && record.spatialMismatch === false
      )),
  };
}
