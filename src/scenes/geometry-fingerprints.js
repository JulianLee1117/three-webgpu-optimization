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

async function fingerprintPhase0Attribute(attribute) {
  if (!attribute?.array || !ArrayBuffer.isView(attribute.array)) {
    throw new TypeError('Phase 0 geometry fingerprints require typed-array buffer attributes.');
  }
  return {
    arrayType: attribute.array.constructor.name,
    byteLength: attribute.array.byteLength,
    count: attribute.count,
    itemSize: attribute.itemSize,
    normalized: attribute.normalized === true,
    usage: attribute.usage,
    gpuType: attribute.gpuType ?? null,
    sha256: await sha256Bytes(attribute.array),
  };
}

async function fingerprintPhase0MorphAttributes(morphAttributes) {
  const result = {};
  for (const name of Object.keys(morphAttributes ?? {}).sort()) {
    result[name] = [];
    for (const attribute of morphAttributes[name]) {
      result[name].push(await fingerprintPhase0Attribute(attribute));
    }
  }
  return result;
}

/**
 * Phase 0's stronger geometry identity. This is deliberately a distinct v2
 * schema so the long-lived v1 artifact hashes above remain reproducible.
 */
export async function fingerprintImmediateAifPhase0GeometryFixture(geometry, bucket) {
  if (!geometry?.index || !geometry.boundingBox || !geometry.boundingSphere) {
    throw new Error(`Phase 0 geometry bucket ${bucket} is missing indexed bounds data.`);
  }
  const attributeEntries = [];
  for (const name of Object.keys(geometry.attributes).sort()) {
    attributeEntries.push([
      name,
      await fingerprintPhase0Attribute(geometry.getAttribute(name)),
    ]);
  }
  const record = {
    bucket,
    family: bucket % 4,
    name: geometry.name,
    attributes: Object.fromEntries(attributeEntries),
    index: await fingerprintPhase0Attribute(geometry.index),
    morphAttributes: await fingerprintPhase0MorphAttributes(geometry.morphAttributes),
    morphTargetsRelative: geometry.morphTargetsRelative === true,
    groups: geometry.groups.map((group) => ({
      start: group.start,
      count: group.count,
      materialIndex: group.materialIndex,
    })),
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

export async function fingerprintImmediateAifPhase0GeometryFixtures(geometries, tier) {
  const records = [];
  for (let bucket = 0; bucket < geometries.length; bucket += 1) {
    records.push(await fingerprintImmediateAifPhase0GeometryFixture(
      geometries[bucket],
      bucket,
    ));
  }
  const manifest = {
    schemaVersion: 2,
    kind: 'immediate-aif-phase0-geometry-fixture-manifest',
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
