export const INDEXED_INDIRECT_STRIDE_UINTS = 5;
export const INDEXED_INDIRECT_STRIDE_BYTES = INDEXED_INDIRECT_STRIDE_UINTS * Uint32Array.BYTES_PER_ELEMENT;

export function createIndexedIndirectCommands(geometries, capacities, initialCounts = null) {
  if (geometries.length !== capacities.length) {
    throw new RangeError('geometries and capacities must have equal lengths.');
  }
  if (initialCounts && initialCounts.length !== capacities.length) {
    throw new RangeError('initialCounts and capacities must have equal lengths.');
  }

  // TSL r185 specializes a single struct storage binding as a scalar struct while
  // `.element()` still emits indexed access. Padding to two records preserves the
  // array binding shape required by the one-bucket parity experiment.
  const recordCount = Math.max(2, geometries.length);
  const commands = new Uint32Array(recordCount * INDEXED_INDIRECT_STRIDE_UINTS);
  const signed = new Int32Array(commands.buffer);
  const offsets = new Uint32Array(geometries.length);

  for (let bucket = 0; bucket < geometries.length; bucket += 1) {
    const geometry = geometries[bucket];
    if (!geometry.index) throw new Error(`Geometry bucket ${bucket} must be indexed.`);
    const base = bucket * INDEXED_INDIRECT_STRIDE_UINTS;
    commands[base] = geometry.index.count;
    commands[base + 1] = initialCounts ? initialCounts[bucket] : 0;
    commands[base + 2] = geometry.drawRange.start > 0 ? geometry.drawRange.start : 0;
    signed[base + 3] = 0;
    commands[base + 4] = 0;
    offsets[bucket] = bucket * INDEXED_INDIRECT_STRIDE_BYTES;
  }

  return {
    commands,
    recordCount,
    capacities: Uint32Array.from(capacities),
    offsets,
  };
}
