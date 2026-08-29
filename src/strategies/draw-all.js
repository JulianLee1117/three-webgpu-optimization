import {
  BundleGroup,
  Mesh,
  StorageBufferAttribute,
} from 'three/webgpu';
import { createStorageTransformMaterial } from '../materials/storage-transform.js';
import { createIndexedBucketGeometry } from '../render/indexed-bucket-geometry.js';

export function buildDrawAllStrategy({ scenario, sourceGeometries }) {
  const matrixAttribute = new StorageBufferAttribute(scenario.matrices, 16);
  const material = createStorageTransformMaterial({
    matrixAttribute,
    objectCount: scenario.objectCount,
  });
  const root = new BundleGroup();
  root.name = 'bundled-indexed-draw-all';
  const geometries = [];

  for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
    const geometry = createIndexedBucketGeometry(
      sourceGeometries[bucket],
      scenario.bucketBases[bucket],
      scenario.bucketCounts[bucket],
    );
    const mesh = new Mesh(geometry, material);
    mesh.frustumCulled = false;
    root.add(mesh);
    geometries.push(geometry);
  }

  return {
    id: 'draw-all',
    root,
    geometries,
    materials: [material],
    storageAttributes: [matrixAttribute],
    computeNodes: [],
    usesCompute: false,
    configuredDrawCommands: scenario.bucketCount,
    configuredComputeDispatches: 0,
    configuredComputeSubmissions: 0,
    configuredSubmittedInstances: scenario.objectCount,
    update() {},
    submitCompute() {},
    async validate() {
      return {
        pass: true,
        kind: 'draw-all-reference',
        expectedVisibleCount: scenario.expectedVisibleCount,
      };
    },
  };
}
