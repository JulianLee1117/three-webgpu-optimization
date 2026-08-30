function requireUint32Array(value, label) {
  if (!(value instanceof Uint32Array)) {
    throw new TypeError(`${label} must be a Uint32Array.`);
  }
}

function uint32LittleEndianBytes(values) {
  const bytes = new Uint8Array(values.length * Uint32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setUint32(index * Uint32Array.BYTES_PER_ELEMENT, values[index], true);
  }
  return bytes;
}

export async function sha256CanonicalUint32(values) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error('Web Crypto SHA-256 is unavailable.');
  const normalized = values instanceof Uint32Array ? values : Uint32Array.from(values);
  const digest = await subtle.digest('SHA-256', uint32LittleEndianBytes(normalized));
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

function sortedUint32(values) {
  return Uint32Array.from(values).sort();
}

async function digestSet(values) {
  const sorted = sortedUint32(values);
  return {
    count: sorted.length,
    sha256: await sha256CanonicalUint32(sorted),
  };
}

export async function createMembershipDigestEvidence({
  expectedIds,
  actualIds,
  actualCounts,
  objectBuckets,
  bucketBases,
  capacities,
}) {
  requireUint32Array(expectedIds, 'expectedIds');
  requireUint32Array(actualIds, 'actualIds');
  requireUint32Array(actualCounts, 'actualCounts');
  requireUint32Array(objectBuckets, 'objectBuckets');
  requireUint32Array(bucketBases, 'bucketBases');
  requireUint32Array(capacities, 'capacities');
  const bucketCount = capacities.length;
  if (actualCounts.length !== bucketCount || bucketBases.length !== bucketCount) {
    throw new RangeError('Membership digest bucket arrays must have equal lengths.');
  }

  const expectedByBucket = Array.from({ length: bucketCount }, () => []);
  let invalidExpectedIds = 0;
  for (const objectId of expectedIds) {
    const bucket = objectBuckets[objectId];
    if (objectId >= objectBuckets.length || bucket >= bucketCount) {
      invalidExpectedIds += 1;
    } else {
      expectedByBucket[bucket].push(objectId);
    }
  }

  const actualByBucket = [];
  let truncatedActualIds = 0;
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const declaredCount = actualCounts[bucket];
    const capturedCount = Math.min(declaredCount, capacities[bucket]);
    truncatedActualIds += declaredCount - capturedCount;
    const base = bucketBases[bucket];
    actualByBucket.push(Array.from(actualIds.subarray(base, base + capturedCount)));
  }

  const perBucket = await Promise.all(expectedByBucket.map(async (expected, bucket) => {
    const [expectedDigest, actualDigest] = await Promise.all([
      digestSet(expected),
      digestSet(actualByBucket[bucket]),
    ]);
    return {
      bucket,
      expected: expectedDigest,
      actual: {
        declaredCount: actualCounts[bucket],
        ...actualDigest,
      },
      match: expectedDigest.count === actualDigest.count
        && expectedDigest.sha256 === actualDigest.sha256,
    };
  }));

  const [expected, actual] = await Promise.all([
    digestSet(expectedIds),
    digestSet(actualByBucket.flat()),
  ]);
  return {
    schemaVersion: 1,
    hashAlgorithm: 'sha256',
    encoding: 'sorted-uint32-little-endian',
    pass: invalidExpectedIds === 0
      && truncatedActualIds === 0
      && expected.count === actual.count
      && expected.sha256 === actual.sha256
      && perBucket.every((bucket) => bucket.match),
    invalidExpectedIds,
    truncatedActualIds,
    expected,
    actual,
    perBucket,
  };
}
