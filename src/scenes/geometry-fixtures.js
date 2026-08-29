import {
  BoxGeometry,
  CylinderGeometry,
  SphereGeometry,
  TorusGeometry,
} from 'three';

const TWO_PI = Math.PI * 2;

function bucketPhase(bucket, salt) {
  const mixed = (Math.imul(bucket + 1, 0x9e37_79b1) ^ salt) >>> 0;
  return (mixed / 0x1_0000_0000) * TWO_PI;
}

function applyBucketDeformation(geometry, bucket) {
  const position = geometry.getAttribute('position');
  const normal = geometry.getAttribute('normal');
  if (!position || !normal || position.count !== normal.count) {
    throw new Error(`Geometry bucket ${bucket} requires matched positions and normals.`);
  }

  geometry.computeBoundingSphere();
  const amplitude = geometry.boundingSphere.radius * 0.0125;
  const phaseA = bucketPhase(bucket, 0xa511_e9b3);
  const phaseB = bucketPhase(bucket, 0x63d8_35f1);

  for (let vertex = 0; vertex < position.count; vertex += 1) {
    const x = position.getX(vertex);
    const y = position.getY(vertex);
    const z = position.getZ(vertex);
    const wave = (
      Math.sin(x * 3.17 + y * 4.11 - z * 2.73 + phaseA) * 0.65
      + Math.cos(x * 5.23 - y * 2.37 + z * 3.79 + phaseB) * 0.35
    );
    const displacement = amplitude * wave;
    position.setXYZ(
      vertex,
      x + normal.getX(vertex) * displacement,
      y + normal.getY(vertex) * displacement,
      z + normal.getZ(vertex) * displacement,
    );
  }

  position.needsUpdate = true;
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  geometry.name = `controlled-family-${bucket % 4}-bucket-${bucket}`;
  return geometry;
}

function createVariant(bucket, tier) {
  const quality = tier === 'high' ? 2 : tier === 'medium' ? 1 : 0;
  switch (bucket % 4) {
    case 0:
      return new BoxGeometry(1.4, 1.0, 1.2, 2 + quality * 2, 2 + quality * 2, 2 + quality * 2);
    case 1:
      return new SphereGeometry(0.8, 12 + quality * 8, 8 + quality * 4);
    case 2:
      return new CylinderGeometry(0.45, 0.75, 1.5, 10 + quality * 6, 2 + quality * 2);
    default:
      return new TorusGeometry(0.62, 0.2, 6 + quality * 4, 12 + quality * 8);
  }
}

export function createIndexedGeometryFixtures(bucketCount, tier = 'medium') {
  const geometries = [];
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const geometry = applyBucketDeformation(createVariant(bucket, tier), bucket);
    if (!geometry.index) throw new Error(`Geometry bucket ${bucket} is unexpectedly non-indexed.`);
    if (!geometry.getAttribute('normal')) throw new Error(`Geometry bucket ${bucket} has no normals.`);
    if (!geometry.getAttribute('uv')) throw new Error(`Geometry bucket ${bucket} has no UVs.`);
    geometries.push(geometry);
  }
  return geometries;
}
