import {
  BundleGroup,
  Color,
  InstancedMesh,
  MeshStandardNodeMaterial,
  StorageInstancedBufferAttribute,
} from 'three/webgpu';
import { ComputeInstanceCulling } from 'three-blocks-v11/instance-culling';
import { compareMembership, validateIndexedCommands } from '../validation/membership.js';
import { createMembershipDigestEvidence } from '../validation/membership-digests.js';
import {
  disposeRetainedComputeNodes,
  retainPinnedV11ComputeNodes,
  retainPinnedV11StorageAttributes,
} from './pinned-v11-compute-lifecycle.js';

function failedReadback(id, message) {
  return {
    pass: false,
    kind: `${id}-exact-membership`,
    readbackErrors: Array.isArray(message) ? message : [message],
  };
}

function createPublicPerBucketSchedule(entries) {
  return {
    configuredComputeDispatches: 4 * entries.length,
    configuredComputeSubmissions: entries.length,
    submitCompute() {
      for (const entry of entries) entry.culler.update();
    },
  };
}

function createPinnedCoalescedSchedule(entries, renderer) {
  // Pinned, undocumented scheduling probe for exactly three-blocks@0.11.0.
  // It accesses runtime compute-node properties only to isolate submission
  // coalescing; three-blocks-current remains the public-API comparison lane.
  const phases = ['clearArgs', 'clearVis', 'selectPack', 'capInstanceCount'];
  const nodes = [];
  for (const entry of entries) {
    for (const phase of phases) {
      nodes.push(entry.computeNodeByField.get(phase));
    }
  }

  let prewarmed = false;
  return {
    configuredComputeDispatches: nodes.length,
    configuredComputeSubmissions: 1,
    submitCompute() {
      if (!prewarmed) {
        // The harness calls submit once before timing. Public update performs
        // each culler's one-time initialization and establishes a valid result.
        for (const entry of entries) entry.culler.update();
        prewarmed = true;
        return;
      }
      renderer.compute(nodes);
    },
  };
}

function disposeCullerEntries(entries) {
  for (const entry of entries) entry.geometry.setIndirect(null);
  for (const entry of entries) {
    entry.culler.dispose();
    entry.mesh.onBeforeRender = entry.priorOnBeforeRender;
  }
  disposeRetainedComputeNodes(entries.map((entry) => entry.computeNodes));
}

function buildThreeBlocksStrategy(
  { scenario, sourceGeometries, renderer },
  { id, createSchedule },
) {
  if (!renderer || typeof renderer.compute !== 'function') {
    throw new TypeError('The Three Blocks strategy requires an initialized renderer.');
  }
  if (!Number.isInteger(scenario.bucketCount) || scenario.bucketCount <= 0) {
    throw new RangeError('The Three Blocks strategy requires at least one geometry bucket.');
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
    const base = scenario.bucketBases[bucket];
    if (!Number.isInteger(count) || count <= 0) {
      throw new RangeError(`scenario.bucketCounts[${bucket}] must be a positive integer.`);
    }
    if (base !== nextBase) {
      throw new RangeError('The Three Blocks strategy requires contiguous bucket-major objects.');
    }
    if (!sourceGeometries[bucket].index) {
      throw new Error(`The controlled Three Blocks comparator requires indexed geometry at bucket ${bucket}.`);
    }
    nextBase += count;
  }
  if (nextBase !== scenario.objectCount) {
    throw new RangeError('The scenario bucket counts must sum to scenario.objectCount.');
  }

  const root = new BundleGroup();
  root.name = id === 'three-blocks-coalesced'
    ? 'three-blocks-v11-coalesced-probe-bundle'
    : 'three-blocks-v11-explicit-bundle';
  const entries = [];

  for (let bucket = 0; bucket < scenario.bucketCount; bucket += 1) {
    const base = scenario.bucketBases[bucket];
    const count = scenario.bucketCounts[bucket];
    const geometry = sourceGeometries[bucket].clone();
    geometry.computeBoundingSphere();

    const material = new MeshStandardNodeMaterial();
    material.color = new Color(0x68a7f2);
    material.roughness = 0.68;
    material.metalness = 0.08;

    const matrixStart = base * 16;
    const matrixEnd = (base + count) * 16;
    const matrixAttribute = new StorageInstancedBufferAttribute(
      scenario.matrices.subarray(matrixStart, matrixEnd),
      16,
    );
    const mesh = new InstancedMesh(geometry, material, count);
    mesh.name = `three-blocks-v11-bucket-${bucket}`;
    mesh.instanceMatrix = matrixAttribute;
    mesh.instanceMatrix.needsUpdate = true;

    const priorOnBeforeRender = mesh.onBeforeRender;
    const culler = new ComputeInstanceCulling(mesh, renderer, {
      count,
      enabled: true,
      sortObjects: false,
      forceSort: false,
      instanceMatrixStorage: matrixAttribute,
      frustumPadXY: 0,
      frustumPadZNear: 0,
      frustumPadZFar: 0,
    });
    const retainedCompute = retainPinnedV11ComputeNodes(culler, bucket);
    const retainedStorageAttributes = retainPinnedV11StorageAttributes(culler, bucket);

    // The mesh-bound constructor installs an automatic update. This benchmark
    // invokes the public update API explicitly before replaying the bundle.
    mesh.onBeforeRender = priorOnBeforeRender;
    root.add(mesh);
    entries.push({
      bucket,
      base,
      count,
      geometry,
      material,
      matrixAttribute,
      mesh,
      priorOnBeforeRender,
      culler,
      computeNodeByField: retainedCompute.byField,
      computeNodes: retainedCompute.nodes,
      storageAttributes: [matrixAttribute, ...retainedStorageAttributes],
    });
  }

  let schedule;
  try {
    schedule = createSchedule(entries, renderer);
  } catch (error) {
    disposeCullerEntries(entries);
    for (const entry of entries) {
      entry.geometry.dispose();
      entry.material.dispose();
      entry.matrixAttribute.dispose();
    }
    throw error;
  }

  let disposed = false;

  return {
    id,
    root,
    geometries: entries.map((entry) => entry.geometry),
    materials: entries.map((entry) => entry.material),
    storageAttributes: [...new Set(entries.flatMap((entry) => entry.storageAttributes))],
    // Pinned package nodes are released explicitly by dispose() because v0.11.0
    // disposes their buffers but does not emit ComputeNode disposal events.
    computeNodes: [],
    usesCompute: true,
    configuredDrawCommands: scenario.bucketCount,
    configuredRenderObjects: scenario.bucketCount,
    configuredComputeDispatches: schedule.configuredComputeDispatches,
    configuredComputeSubmissions: schedule.configuredComputeSubmissions,
    configuredSubmittedInstances: null,
    update(camera) {
      for (const entry of entries) entry.culler.setCameraUniforms(camera);
    },
    submitCompute: schedule.submitCompute,
    async validate(_renderer, expectedIds) {
      let nativeCommands;
      try {
        nativeCommands = await Promise.all(
          entries.map(async (entry) => {
            try {
              return await entry.culler.readIndirectArgs();
            } catch (error) {
              throw new Error(`bucket ${entry.bucket}: ${String(error)}`);
            }
          }),
        );
      } catch (error) {
        return failedReadback(id, `Three Blocks command readback failed: ${String(error)}`);
      }

      const malformedCommands = [];
      for (let bucket = 0; bucket < nativeCommands.length; bucket += 1) {
        const command = nativeCommands[bucket];
        if (!(command instanceof Uint32Array)) {
          malformedCommands.push(`bucket ${bucket}: Three Blocks returned no indirect command.`);
        } else if (command.length !== 5) {
          malformedCommands.push(
            `bucket ${bucket}: expected 5 indirect command values, received ${command.length}.`,
          );
        }
      }
      if (malformedCommands.length > 0) return failedReadback(id, malformedCommands);

      let nativeSurvivors;
      try {
        nativeSurvivors = await Promise.all(
          entries.map(async (entry) => {
            try {
              return await entry.culler.readSurvivorIndicesAsync();
            } catch (error) {
              throw new Error(`bucket ${entry.bucket}: ${String(error)}`);
            }
          }),
        );
      } catch (error) {
        return failedReadback(id, `Three Blocks survivor readback failed: ${String(error)}`);
      }

      const malformedSurvivors = [];
      for (let bucket = 0; bucket < nativeSurvivors.length; bucket += 1) {
        if (!(nativeSurvivors[bucket] instanceof Uint32Array)) {
          malformedSurvivors.push(
            `bucket ${bucket}: Three Blocks returned no survivor-ID array.`,
          );
        }
      }
      if (malformedSurvivors.length > 0) return failedReadback(id, malformedSurvivors);

      const commands = new Uint32Array(scenario.bucketCount * 5);
      const actualCounts = new Uint32Array(scenario.bucketCount);
      const survivorIds = new Uint32Array(scenario.objectCount);
      survivorIds.fill(scenario.objectCount);
      const readbackErrors = [];

      for (const entry of entries) {
        const command = nativeCommands[entry.bucket];
        const localSurvivors = nativeSurvivors[entry.bucket];
        commands.set(command, entry.bucket * 5);
        actualCounts[entry.bucket] = command[1];

        if (localSurvivors.length !== actualCounts[entry.bucket]) {
          readbackErrors.push(
            `bucket ${entry.bucket}: indirect count ${actualCounts[entry.bucket]} does not match survivor readback length ${localSurvivors.length}.`,
          );
        }
        if (localSurvivors.length > entry.count) {
          readbackErrors.push(
            `bucket ${entry.bucket}: survivor readback length ${localSurvivors.length} exceeds capacity ${entry.count}.`,
          );
        }

        let invalidLocalIds = 0;
        const safeLength = Math.min(localSurvivors.length, entry.count);
        for (let slot = 0; slot < safeLength; slot += 1) {
          const localId = localSurvivors[slot];
          if (localId >= entry.count) {
            invalidLocalIds += 1;
          } else {
            survivorIds[entry.base + slot] = entry.base + localId;
          }
        }
        if (invalidLocalIds > 0) {
          readbackErrors.push(
            `bucket ${entry.bucket}: ${invalidLocalIds} survivor IDs exceed local capacity ${entry.count}.`,
          );
        }
      }

      const membership = compareMembership({
        expectedIds,
        actualIds: survivorIds,
        actualCounts,
        objectBuckets: scenario.objectBuckets,
        bucketBases: scenario.bucketBases,
        capacities: scenario.bucketCounts,
        objectCount: scenario.objectCount,
      });
      const commandValidation = validateIndexedCommands({
        commands,
        geometries: entries.map((entry) => entry.geometry),
        expectedCounts: scenario.visibleCounts,
      });
      const membershipDigests = await createMembershipDigestEvidence({
        expectedIds,
        actualIds: survivorIds,
        actualCounts,
        objectBuckets: scenario.objectBuckets,
        bucketBases: scenario.bucketBases,
        capacities: scenario.bucketCounts,
      });

      return {
        pass: readbackErrors.length === 0
          && membership.pass
          && membershipDigests.pass
          && commandValidation.pass,
        kind: `${id}-exact-membership`,
        membership,
        membershipDigests,
        commandValidation,
        readbackErrors,
      };
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      disposeCullerEntries(entries);
    },
  };
}

export function buildThreeBlocksCurrentStrategy(options) {
  return buildThreeBlocksStrategy(options, {
    id: 'three-blocks-current',
    createSchedule: createPublicPerBucketSchedule,
  });
}

export function buildThreeBlocksCoalescedStrategy(options) {
  return buildThreeBlocksStrategy(options, {
    id: 'three-blocks-coalesced',
    createSchedule: createPinnedCoalescedSchedule,
  });
}
