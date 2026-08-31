import { sha256CanonicalUint32 } from '../validation/membership-digests.js';

function toHex(buffer) {
  return [...new Uint8Array(buffer)]
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

async function sha256Bytes(bytes) {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error('Web Crypto SHA-256 is unavailable.');
  const exactBytes = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return toHex(await subtle.digest('SHA-256', exactBytes));
}

async function sha256Json(value) {
  return sha256Bytes(new TextEncoder().encode(JSON.stringify(value)));
}

async function fingerprintAttribute(attribute) {
  if (!attribute?.array || !ArrayBuffer.isView(attribute.array)) {
    throw new TypeError('Geometry fingerprints require typed-array buffer attributes.');
  }
  return {
    arrayType: attribute.array.constructor.name,
    count: attribute.count,
    itemSize: attribute.itemSize,
    normalized: attribute.normalized === true,
    sha256: await sha256Bytes(attribute.array),
  };
}

function vectorRecord(vector) {
  return [vector.x, vector.y, vector.z];
}

export async function fingerprintGeometryFixture(geometry, bucket) {
  if (!geometry?.index || !geometry.boundingBox || !geometry.boundingSphere) {
    throw new Error(`Geometry bucket ${bucket} is missing indexed bounds data.`);
  }
  const attributeEntries = await Promise.all(
    Object.keys(geometry.attributes).sort().map(async (name) => (
      [name, await fingerprintAttribute(geometry.getAttribute(name))]
    )),
  );
  const record = {
    bucket,
    family: bucket % 4,
    name: geometry.name,
    attributes: Object.fromEntries(attributeEntries),
    index: await fingerprintAttribute(geometry.index),
    drawRange: {
      start: geometry.drawRange.start,
      count: Number.isFinite(geometry.drawRange.count) ? geometry.drawRange.count : 'Infinity',
    },
    boundingBox: {
      min: vectorRecord(geometry.boundingBox.min),
      max: vectorRecord(geometry.boundingBox.max),
    },
    boundingSphere: {
      center: vectorRecord(geometry.boundingSphere.center),
      radius: geometry.boundingSphere.radius,
    },
  };
  return { ...record, sha256: await sha256Json(record) };
}

export async function fingerprintGeometryFixtures(geometries, tier) {
  const records = await Promise.all(
    geometries.map((geometry, bucket) => fingerprintGeometryFixture(geometry, bucket)),
  );
  const manifest = {
    schemaVersion: 1,
    generator: 'createIndexedGeometryFixtures',
    tier,
    bucketCount: records.length,
    geometries: records,
  };
  return { ...manifest, sha256: await sha256Json(manifest) };
}

export async function fingerprintFixedSubsetScenario(scenario, seed) {
  const arrayNames = [
    'bucketCounts',
    'bucketBases',
    'visibleCounts',
    'objectBuckets',
    'matrices',
    'bounds',
    'expectedVisibleIds',
    'cullOrder',
  ];
  const arrays = Object.fromEntries(await Promise.all(arrayNames.map(async (name) => {
    const value = scenario[name];
    if (!ArrayBuffer.isView(value)) {
      throw new TypeError(`Scenario fingerprint requires typed array ${name}.`);
    }
    return [name, {
      arrayType: value.constructor.name,
      length: value.length,
      sha256: await sha256Bytes(value),
    }];
  })));
  const record = {
    schemaVersion: 1,
    generator: 'createFixedSubsetScenario',
    seed,
    objectCount: scenario.objectCount,
    bucketCount: scenario.bucketCount,
    visibilityFraction: scenario.visibilityFraction,
    layout: scenario.layout ?? 'baseline',
    depthBinRange: scenario.depthBinRange
      ? {
        near: scenario.depthBinRange.near,
        far: scenario.depthBinRange.far,
      }
      : null,
    expectedVisibleCount: scenario.expectedVisibleCount,
    expectedVisibleIdsCanonicalSha256: await sha256CanonicalUint32(
      Uint32Array.from(scenario.expectedVisibleIds).sort(),
    ),
    arrays,
  };
  return { ...record, sha256: await sha256Json(record) };
}
