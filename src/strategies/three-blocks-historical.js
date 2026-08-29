import {
  BundleGroup,
  Color,
  Matrix4,
  MeshStandardNodeMaterial,
} from 'three/webgpu';
import { IndirectBatchedMesh } from 'three-blocks-v10/indirect-batching';
import { decodeHistoricalIndirectResults } from '../validation/historical-indirect.js';
import { compareMembership } from '../validation/membership.js';

const HISTORICAL_STORAGE_FIELDS = Object.freeze([
  '_refPosFallbackSSBO',
  '_refNrmFallbackSSBO',
  'refPosSSBO',
  'refNrmSSBO',
  'outVisSSBO',
  '_matricesFallbackSB',
  'matricesSB',
  'geometryIdSB',
  'geometrySpheresSB',
  'blockCountsSSBO',
  'blockOffsetsSSBO',
  '_linearIdSSBO',
  '_outIdSSBO',
  '_indirect',
  '_multiIndirect',
  'geometryRangesSSBO',
  'geometryCountsSSBO',
  'geometryOffsetsSSBO',
  'geometryHeadsSSBO',
  'geomAllocHeadSSBO',
]);

const HISTORICAL_COMPUTE_FIELDS = Object.freeze([
  'initAllK',
  'clearArgs',
  'clearVis',
  'clearBlockCounts',
  'countBlocks',
  'prefixBlocks',
  'scatter',
  'singleGeometrySelect',
  'clearGeomCounts',
  'clearGeomHeads',
  'clearGeomAllocHead',
  'scatterByGeometry',
  'writeMultiIndirect',
]);

const HISTORICAL_MESH_STORAGE_FIELDS = Object.freeze([
  'matricesSB',
  'colorsSB',
  'geometryIdSB',
  'survivorIdSB',
  'drawFirstInstanceSB',
  'drawInstanceCountSB',
  '_multiIndirect',
  '_internalSurvivorSB',
  '_workgroupIndirect',
]);

function collectUniqueFields(target, names) {
  return [...new Set(names.map((name) => target?.[name]).filter(Boolean))];
}

function requireHistoricalDiagnostics(mesh) {
  const culler = mesh.culler;
  if (!culler
    || typeof culler.readIndirectArgsAll !== 'function'
    || typeof culler.readSurvivorIndicesAsync !== 'function') {
    throw new TypeError(
      'three-blocks@0.10.0 diagnostics changed: heterogeneous readback is unavailable.',
    );
  }
  return culler;
}

function requireSafeCleanup(renderer, attributes, computeNodes) {
  const rendererDelete = renderer?._attributes?.delete;
  if (attributes.some((attribute) => typeof attribute.dispose !== 'function')
    && typeof rendererDelete !== 'function') {
    throw new TypeError(
      'Three.js r185 attribute cleanup is unavailable for three-blocks@0.10.0 resources.',
    );
  }
  if (computeNodes.some((node) => typeof node.dispose !== 'function')) {
    throw new TypeError(
      'Three.js r185 compute-node cleanup is unavailable for three-blocks@0.10.0 resources.',
    );
  }
}

function requireCompatibleScenario(scenario, sourceGeometries) {
  if (!Number.isInteger(scenario.bucketCount) || scenario.bucketCount <= 1) {
    throw new RangeError(
      'The historical heterogeneous lane requires at least two geometry buckets.',
    );
  }
  if (sourceGeometries.length !== scenario.bucketCount) {
    throw new RangeError('sourceGeometries length must equal scenario.bucketCount.');
  }
  if (scenario.bucketCounts.length !== scenario.bucketCount
    || scenario.bucketBases.length !== scenario.bucketCount) {
    throw new RangeError('The scenario bucket layout does not match scenario.bucketCount.');
  }
  if (scenario.matrices.length !== scenario.objectCount * 16) {
    throw new RangeError('The scenario must provide one full matrix per object.');
  }

  let nextBase = 0;
  for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
    const count = scenario.bucketCounts[bucket];
    if (!Number.isInteger(count) || count <= 0) {
      throw new RangeError(`scenario.bucketCounts[${bucket}] must be a positive integer.`);
    }
    if (scenario.bucketBases[bucket] !== nextBase) {
      throw new RangeError('The historical lane requires contiguous bucket-major objects.');
    }
    if (!sourceGeometries[bucket].index) {
      throw new Error(`Historical geometry bucket ${bucket} must be indexed.`);
    }
    nextBase += count;
  }
  if (nextBase !== scenario.objectCount) {
    throw new RangeError('The scenario bucket counts must sum to scenario.objectCount.');
  }
}

function failedReadback(message) {
  return {
    pass: false,
    kind: 'three-blocks-historical-exact-membership',
    readbackErrors: [message],
  };
}

export function buildThreeBlocksHistoricalStrategy({ scenario, sourceGeometries, renderer }) {
  if (!renderer || typeof renderer.compute !== 'function') {
    throw new TypeError('The historical Three Blocks strategy requires an initialized renderer.');
  }
  if (renderer.hasFeature('indirect-first-instance') !== true) {
    throw new Error(
      'three-blocks@0.10.0 heterogeneous batching requires the indirect-first-instance WebGPU feature.',
    );
  }
  requireCompatibleScenario(scenario, sourceGeometries);

  const maxVertexCount = sourceGeometries.reduce(
    (sum, geometry) => sum + geometry.getAttribute('position').count,
    0,
  );
  const maxIndexCount = sourceGeometries.reduce(
    (sum, geometry) => sum + geometry.index.count,
    0,
  );
  const material = new MeshStandardNodeMaterial();
  material.color = new Color(0x68a7f2);
  material.roughness = 0.68;
  material.metalness = 0.08;

  let mesh;
  try {
    mesh = new IndirectBatchedMesh(
      scenario.objectCount,
      maxVertexCount,
      maxIndexCount,
      material,
    );
    mesh.name = 'three-blocks-v10-historical-indirect-batch';
    mesh.beginBulkUpdate();

    const geometryIds = sourceGeometries.map((geometry) => mesh.addGeometry(geometry));
    const matrix = new Matrix4();
    for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
      const base = scenario.bucketBases[bucket];
      const count = scenario.bucketCounts[bucket];
      for (let localIndex = 0; localIndex < count; localIndex += 1) {
        const objectId = base + localIndex;
        const instanceId = mesh.addInstance(geometryIds[bucket]);
        if (instanceId !== objectId) {
          throw new Error(
            `three-blocks@0.10.0 returned instance ${instanceId}; expected stable object ID ${objectId}.`,
          );
        }
        matrix.fromArray(scenario.matrices, objectId * 16);
        mesh.setMatrixAt(instanceId, matrix);
      }
    }
    mesh.endBulkUpdate(false);
    mesh.updateMatrixWorld(true);
    mesh.enableInternalCulling(renderer);
  } catch (error) {
    mesh?.dispose?.();
    material.dispose();
    throw error;
  }

  const culler = requireHistoricalDiagnostics(mesh);
  const priorOnBeforeRender = mesh.onBeforeRender;
  // Bundle replay does not invoke object callbacks. Explicit scheduling is
  // required every frame, and suppressing the callback also prevents a second
  // cull when the bundle is initially recorded or explicitly rebuilt.
  mesh.onBeforeRender = () => {};

  const root = new BundleGroup();
  root.name = 'three-blocks-v10-historical-indirect-bundle';
  root.add(mesh);

  const storageAttributes = [
    ...collectUniqueFields(mesh, HISTORICAL_MESH_STORAGE_FIELDS),
    ...collectUniqueFields(culler, HISTORICAL_STORAGE_FIELDS),
  ];
  const uniqueStorageAttributes = [...new Set(storageAttributes)];
  const computeNodes = collectUniqueFields(culler, HISTORICAL_COMPUTE_FIELDS);
  try {
    requireSafeCleanup(renderer, uniqueStorageAttributes, computeNodes);
  } catch (error) {
    mesh.onBeforeRender = priorOnBeforeRender;
    mesh.geometry.setIndirect(null);
    culler.dispose?.();
    mesh.dispose();
    material.dispose();
    throw error;
  }

  let activeCamera = null;
  let disposed = false;

  return {
    id: 'three-blocks-historical',
    root,
    // The batch owns and disposes its merged geometry.
    geometries: [],
    materials: [material],
    storageAttributes: uniqueStorageAttributes,
    computeNodes,
    usesCompute: true,
    configuredDrawCommands: scenario.bucketCount,
    configuredComputeDispatches: 9,
    configuredComputeSubmissions: 1,
    configuredSubmittedInstances: null,
    update(camera) {
      activeCamera = camera;
    },
    submitCompute() {
      if (!activeCamera) throw new Error('Historical Three Blocks culling requires a camera.');
      mesh.updateInternalCulling(activeCamera);
    },
    async validate(_renderer, expectedIds) {
      let commands;
      let survivorIds;
      try {
        commands = await culler.readIndirectArgsAll();
        survivorIds = await culler.readSurvivorIndicesAsync();
      } catch (error) {
        return failedReadback(`Three Blocks 0.10 diagnostic readback failed: ${String(error)}`);
      }

      let decoded;
      try {
        decoded = decodeHistoricalIndirectResults({
          commands,
          survivorIds,
          geometries: sourceGeometries,
          expectedCounts: scenario.visibleCounts,
          bucketBases: scenario.bucketBases,
          capacities: scenario.bucketCounts,
          objectCount: scenario.objectCount,
        });
      } catch (error) {
        return failedReadback(`Three Blocks 0.10 diagnostic output was malformed: ${String(error)}`);
      }

      const membership = compareMembership({
        expectedIds,
        actualIds: decoded.actualIds,
        actualCounts: decoded.actualCounts,
        objectBuckets: scenario.objectBuckets,
        bucketBases: scenario.bucketBases,
        capacities: scenario.bucketCounts,
        objectCount: scenario.objectCount,
      });
      return {
        pass: decoded.commandValidation.pass && membership.pass,
        kind: 'three-blocks-historical-exact-membership',
        commandValidation: decoded.commandValidation,
        membership,
        readbackErrors: [],
      };
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      mesh.onBeforeRender = priorOnBeforeRender;
      mesh.geometry.setIndirect(null);
      culler.dispose();
      mesh.dispose();
      mesh.culler = null;
    },
  };
}
