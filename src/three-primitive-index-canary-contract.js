export const PRIMITIVE_INDEX_CANARY_KIND =
  'three-webgpu-primitive-index-correctness-canary';

export const PRIMITIVE_INDEX_SENTINEL = 0x5a17;

export const PRIMITIVE_INDEX_POSITIONS = Object.freeze([
  0, 0, 0,
  0.9, 0.9, 0,
  -0.9, 0.9, 0,
  -0.9, -0.9, 0,
  0.9, -0.9, 0,
]);

export const PRIMITIVE_INDEX_INDICES = Object.freeze([
  0, 1, 2,
  0, 2, 3,
  0, 3, 4,
  0, 4, 1,
]);

export const PRIMITIVE_INDEX_INSTANCE_TRANSLATIONS = Object.freeze([
  Object.freeze([-1.35, 0, 0]),
  Object.freeze([1.35, 0, 0]),
]);

export const PRIMITIVE_INDEX_ORACLE_SIZE = Object.freeze({
  width: 320,
  height: 180,
});

export const PRIMITIVE_INDEX_MIN_PIXELS_PER_TUPLE = 1_000;

export function expectedPrimitiveIndexTuples() {
  const tuples = [];
  for (let instance = 0; instance < 2; instance += 1) {
    for (let primitive = 0; primitive < 4; primitive += 1) {
      tuples.push([primitive, instance, PRIMITIVE_INDEX_SENTINEL, 1]);
    }
  }
  return tuples;
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

export function analyzePrimitiveIndexOracle(
  words,
  {
    width = PRIMITIVE_INDEX_ORACLE_SIZE.width,
    height = PRIMITIVE_INDEX_ORACLE_SIZE.height,
    minimumPixelsPerTuple = PRIMITIVE_INDEX_MIN_PIXELS_PER_TUPLE,
  } = {},
) {
  if (!(words instanceof Uint32Array)) {
    throw new TypeError('primitiveIndex oracle readback must be a Uint32Array.');
  }
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new RangeError('primitiveIndex oracle dimensions must be positive integers.');
  }
  if (words.length !== width * height * 4) {
    throw new RangeError(
      `primitiveIndex oracle has ${words.length} words; expected ${width * height * 4}.`,
    );
  }

  const histogram = new Map();
  let instanceSpatialMismatches = 0;
  for (let offset = 0; offset < words.length; offset += 4) {
    const key = `${words[offset]},${words[offset + 1]},${words[offset + 2]},${words[offset + 3]}`;
    histogram.set(key, (histogram.get(key) ?? 0) + 1);
    if (words[offset + 2] === PRIMITIVE_INDEX_SENTINEL && words[offset + 3] === 1) {
      const pixel = offset / 4;
      const x = pixel % width;
      const expectedInstance = x < width / 2 ? 0 : 1;
      if (words[offset + 1] !== expectedInstance) instanceSpatialMismatches += 1;
    }
  }

  const backgroundKey = '0,0,0,0';
  const expectedTuples = expectedPrimitiveIndexTuples();
  const expectedKeys = new Set(expectedTuples.map(tupleKey));
  const unexpected = [...histogram.entries()]
    .filter(([key]) => key !== backgroundKey && expectedKeys.has(key) === false)
    .sort(([left], [right]) => compareTupleKeys(left, right))
    .map(([key, count]) => ({ tuple: key.split(',').map(Number), count }));
  const tuples = expectedTuples.map((tuple) => {
    const count = histogram.get(tupleKey(tuple)) ?? 0;
    return { tuple, count, sufficient: count >= minimumPixelsPerTuple };
  });
  const primitiveIdsByInstance = [0, 1].map((instance) => (
    tuples
      .filter((record) => record.tuple[1] === instance && record.count > 0)
      .map((record) => record.tuple[0])
      .sort((left, right) => left - right)
  ));
  const resetProven = primitiveIdsByInstance.every((ids) => (
    ids.length === 4 && ids.every((value, index) => value === index)
  ));
  const histogramRecords = [...histogram.entries()]
    .sort(([left], [right]) => compareTupleKeys(left, right))
    .map(([key, count]) => ({ tuple: key.split(',').map(Number), count }));

  return {
    backgroundPixels: histogram.get(backgroundKey) ?? 0,
    exact: unexpected.length === 0
      && tuples.every((record) => record.sufficient)
      && resetProven
      && instanceSpatialMismatches === 0
      && (histogram.get(backgroundKey) ?? 0) > 0,
    expectedPixelTuples: expectedTuples,
    height,
    histogram: histogramRecords,
    instanceSpatialMismatches,
    minimumPixelsPerTuple,
    primitiveIdsByInstance,
    resetProven,
    totalPixels: width * height,
    tuples,
    unexpected,
    width,
  };
}
