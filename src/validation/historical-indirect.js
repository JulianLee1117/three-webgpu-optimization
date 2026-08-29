import { INDEXED_INDIRECT_STRIDE_UINTS } from '../culling/indexed-command-layout.js';

function asSignedWords(commands) {
  return new Int32Array(commands.buffer, commands.byteOffset, commands.length);
}

/**
 * Validate Three Blocks 0.10 heterogeneous indirect commands and remap its
 * dynamically allocated survivor ranges into the harness's bucket-major form.
 */
export function decodeHistoricalIndirectResults({
  commands,
  survivorIds,
  geometries,
  expectedCounts,
  bucketBases,
  capacities,
  objectCount,
}) {
  const bucketCount = geometries.length;
  const errors = [];

  if (!(commands instanceof Uint32Array)) {
    throw new TypeError('Historical indirect commands must be a Uint32Array.');
  }
  if (!(survivorIds instanceof Uint32Array)) {
    throw new TypeError('Historical survivor IDs must be a Uint32Array.');
  }
  if (commands.length !== bucketCount * INDEXED_INDIRECT_STRIDE_UINTS) {
    errors.push(
      `expected ${bucketCount * INDEXED_INDIRECT_STRIDE_UINTS} command words, received ${commands.length}`,
    );
  }
  if (expectedCounts.length !== bucketCount
    || bucketBases.length !== bucketCount
    || capacities.length !== bucketCount) {
    throw new RangeError('Historical validation arrays must match the geometry count.');
  }

  const signed = asSignedWords(commands);
  const actualCounts = new Uint32Array(bucketCount);
  const actualIds = new Uint32Array(objectCount);
  actualIds.fill(objectCount);
  const ranges = [];
  let expectedFirstIndex = 0;
  let totalCommandSurvivors = 0;

  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const geometry = geometries[bucket];
    if (!geometry.index) {
      errors.push(`bucket ${bucket}: source geometry is not indexed`);
      continue;
    }

    const base = bucket * INDEXED_INDIRECT_STRIDE_UINTS;
    if (base + INDEXED_INDIRECT_STRIDE_UINTS > commands.length) {
      errors.push(`bucket ${bucket}: command is missing`);
      expectedFirstIndex += geometry.index.count;
      continue;
    }

    const indexCount = commands[base];
    const instanceCount = commands[base + 1];
    const firstIndex = commands[base + 2];
    const baseVertex = signed[base + 3];
    const firstInstance = commands[base + 4];
    actualCounts[bucket] = instanceCount;
    totalCommandSurvivors += instanceCount;

    if (indexCount !== geometry.index.count) errors.push(`bucket ${bucket}: indexCount`);
    if (instanceCount !== expectedCounts[bucket]) errors.push(`bucket ${bucket}: instanceCount`);
    if (firstIndex !== expectedFirstIndex) errors.push(`bucket ${bucket}: firstIndex`);
    if (baseVertex !== 0) errors.push(`bucket ${bucket}: baseVertex`);
    if (instanceCount > capacities[bucket]) errors.push(`bucket ${bucket}: capacity overflow`);

    ranges.push({ bucket, firstInstance, instanceCount });
    expectedFirstIndex += geometry.index.count;
  }

  if (totalCommandSurvivors !== survivorIds.length) {
    errors.push(
      `command survivor total ${totalCommandSurvivors} does not match readback length ${survivorIds.length}`,
    );
  }

  const coverage = new Uint8Array(survivorIds.length);
  for (const { bucket, firstInstance, instanceCount } of ranges) {
    const end = firstInstance + instanceCount;
    if (end > survivorIds.length) {
      errors.push(`bucket ${bucket}: survivor range exceeds compacted readback`);
    }

    const capacity = capacities[bucket];
    const readableCount = firstInstance < survivorIds.length
      ? Math.min(instanceCount, capacity, survivorIds.length - firstInstance)
      : 0;
    for (let slot = 0; slot < readableCount; slot += 1) {
      const sourceSlot = firstInstance + slot;
      if (coverage[sourceSlot] !== 0) {
        errors.push(`bucket ${bucket}: survivor range overlaps slot ${sourceSlot}`);
      }
      coverage[sourceSlot] += 1;
      actualIds[bucketBases[bucket] + slot] = survivorIds[sourceSlot];
    }
  }

  for (let slot = 0; slot < coverage.length; slot += 1) {
    if (coverage[slot] === 0) errors.push(`compacted survivor slot ${slot} is unclaimed`);
  }

  return {
    actualCounts,
    actualIds,
    commandValidation: {
      pass: errors.length === 0,
      errors,
      totalCommandSurvivors,
      survivorReadbackLength: survivorIds.length,
    },
  };
}
