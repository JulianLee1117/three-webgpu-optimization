import assert from 'node:assert/strict';
import test from 'node:test';
import {
  createMembershipDigestEvidence,
  sha256CanonicalUint32,
} from '../src/validation/membership-digests.js';

test('canonical uint32 digest is order-sensitive before membership sorting', async () => {
  assert.notEqual(
    await sha256CanonicalUint32(Uint32Array.of(1, 2, 3)),
    await sha256CanonicalUint32(Uint32Array.of(3, 2, 1)),
  );
});

test('membership digests prove aggregate and per-bucket sets independent of append order', async () => {
  const evidence = await createMembershipDigestEvidence({
    expectedIds: Uint32Array.of(0, 2, 4, 5),
    actualIds: Uint32Array.of(2, 0, 99, 99, 5, 4),
    actualCounts: Uint32Array.of(2, 2),
    objectBuckets: Uint32Array.of(0, 0, 0, 1, 1, 1),
    bucketBases: Uint32Array.of(0, 4),
    capacities: Uint32Array.of(4, 2),
  });
  assert.equal(evidence.pass, true);
  assert.equal(evidence.expected.sha256, evidence.actual.sha256);
  assert.deepEqual(evidence.perBucket.map((bucket) => bucket.match), [true, true]);
});

test('membership digests reject a cross-bucket substitution with equal aggregate count', async () => {
  const evidence = await createMembershipDigestEvidence({
    expectedIds: Uint32Array.of(0, 2, 4, 5),
    actualIds: Uint32Array.of(0, 4, 99, 99, 2, 5),
    actualCounts: Uint32Array.of(2, 2),
    objectBuckets: Uint32Array.of(0, 0, 0, 1, 1, 1),
    bucketBases: Uint32Array.of(0, 4),
    capacities: Uint32Array.of(4, 2),
  });
  assert.equal(evidence.pass, false);
  assert.deepEqual(evidence.perBucket.map((bucket) => bucket.match), [false, false]);
});
