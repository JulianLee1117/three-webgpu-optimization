import assert from 'node:assert/strict';
import test from 'node:test';
import {
  FIRST_INSTANCE_SHADER_EVIDENCE_SCHEMA_VERSION,
  createFirstInstanceLaneShaderEvidence,
  createFirstInstanceShaderEvidence,
} from '../src/validation/first-instance-shader-evidence.js';
import {
  runtimeStorageBindingEvidence,
  runtimeVertexInputEvidence,
} from '../src/strategies/first-instance-crossover.js';

const OBJECT_COUNT = 4_096;
const VERTEX_COUNT = 6_440;

function vertexShader({
  feature,
  matrixName,
  visibleIdsName,
  extraBeforeAddress = '',
}) {
  const inputs = feature
    ? `fn main( @builtin( instance_index ) instanceIndex : u32,
\t@location( 0 ) position : vec3<f32>,
\t@location( 1 ) normal : vec3<f32> ) -> VaryingsStruct {`
    : `fn main( @builtin( instance_index ) instanceIndex : u32,
\t@location( 0 ) position : vec3<f32>,
\t@location( 1 ) bucketBase : u32,
\t@location( 2 ) normal : vec3<f32> ) -> VaryingsStruct {`;
  const visibleIndex = feature ? 'instanceIndex' : '( bucketBase + instanceIndex )';
  return `
struct ${matrixName}Struct {
\tvalue : array< mat4x4<f32> >
};
@binding( 3 ) @group( 1 )
var<storage, read> ${matrixName} : ${matrixName}Struct;

struct ${visibleIdsName}Struct {
\tvalue : array< u32 >
};
@binding( 4 ) @group( 1 )
var<storage, read> ${visibleIdsName} : ${visibleIdsName}Struct;

struct VaryingsStruct {
\t@builtin( position ) position : vec4<f32>
};

@vertex
${inputs}
\tvar objectMatrix : mat4x4<f32>;
${extraBeforeAddress}\tobjectMatrix = ${matrixName}.value[ ${visibleIdsName}.value[ ${visibleIndex} ] ];
\tvar transformed : vec3<f32> = ( objectMatrix * vec4<f32>( position, 1.0 ) ).xyz;
\tvar varyings : VaryingsStruct;
\tvaryings.position = vec4<f32>( transformed + normal * 0.0, 1.0 );
\treturn varyings;
}
`;
}

function fragmentShader(extra = '') {
  return `
@fragment
fn main() -> @location( 0 ) vec4<f32> {
${extra}\treturn vec4<f32>( 0.4, 0.65, 0.95, 1.0 );
}
`;
}

function computeBindings(binding = 4) {
  return [{
    semantic: 'indirectCommands',
    kind: 'storage-buffer',
    group: 0,
    binding,
    access: 'readWrite',
    attributeType: 'IndirectStorageBufferAttribute',
    arrayType: 'Uint32Array',
    itemSize: 5,
    count: 32,
    byteLength: 640,
  }];
}

function computeShader(identifier, binding = 4) {
  return `struct FixedSliceIndexedDraw {
  indexCount : u32,
  instanceCount : atomic<u32>,
  firstIndex : u32,
  baseVertex : i32,
  firstInstance : u32
};
struct ${identifier}Struct {
  value : array<FixedSliceIndexedDraw>
};
@binding( ${binding} ) @group( 0 )
var<storage, read_write> ${identifier} : ${identifier}Struct;
@compute @workgroup_size( 64, 1, 1 )
fn main() {
  atomicAdd( &${identifier}.value[ instanceIndex ].instanceCount, 1u );
}`;
}

function computePhases(identifierBase) {
  return {
    reset: {
      shader: computeShader(`NodeBuffer_${identifierBase}`),
      bindings: computeBindings(),
    },
    cull: {
      computeShader: computeShader(`NodeBuffer_${identifierBase + 1}`),
      bindings: computeBindings(),
    },
  };
}

function vertexInputs(feature) {
  const common = {
    position: {
      name: 'position',
      shaderLocation: 0,
      format: 'float32x3',
      stepMode: 'vertex',
      arrayType: 'Float32Array',
      itemSize: 3,
      normalized: false,
      count: VERTEX_COUNT,
      resourceId: 700,
    },
    normal: {
      name: 'normal',
      shaderLocation: feature ? 1 : 2,
      format: 'float32x3',
      stepMode: 'vertex',
      arrayType: 'Float32Array',
      itemSize: 3,
      normalized: false,
      count: VERTEX_COUNT,
      resourceId: 701,
    },
  };
  if (feature) return [common.position, common.normal];
  return [
    common.position,
    {
      name: 'bucketBase',
      shaderLocation: 1,
      format: 'uint32',
      stepMode: 'vertex',
      arrayType: 'Uint32Array',
      itemSize: 1,
      normalized: false,
      count: VERTEX_COUNT,
      resourceId: 702,
    },
    common.normal,
  ];
}

function storageBindings() {
  return [
    {
      semantic: 'matrix',
      group: 1,
      binding: 3,
      access: 'read',
      visibility: 1,
      elementType: 'mat4x4<f32>',
      count: OBJECT_COUNT,
      byteLength: OBJECT_COUNT * 64,
      resourceId: 'shared-matrix-attribute',
    },
    {
      semantic: 'visibleIds',
      group: 1,
      binding: 4,
      access: 'read',
      visibility: 'vertex',
      elementType: 'u32',
      count: OBJECT_COUNT,
      byteLength: OBJECT_COUNT * 4,
      resourceId: 'shared-visible-ids-attribute',
    },
  ];
}

function contrast(overrides = {}) {
  const portableOptions = {
    feature: false,
    matrixName: 'NodeBuffer_1864',
    visibleIdsName: 'NodeBuffer_1865',
    ...(overrides.portableVertexOptions ?? {}),
  };
  const featureOptions = {
    feature: true,
    matrixName: 'NodeBuffer_2716',
    visibleIdsName: 'NodeBuffer_2717',
    ...(overrides.featureVertexOptions ?? {}),
  };
  return {
    portable: {
      vertexShader: vertexShader(portableOptions),
      fragmentShader: fragmentShader(overrides.portableFragmentExtra),
      vertexInputs: vertexInputs(false),
      storageBindings: storageBindings(),
    },
    feature: {
      vertexShader: vertexShader(featureOptions),
      fragmentShader: fragmentShader(overrides.featureFragmentExtra),
      vertexInputs: vertexInputs(true),
      storageBindings: storageBindings(),
    },
  };
}

function clone(value) {
  return structuredClone(value);
}

function assertFailed(evidence, pattern) {
  assert.equal(evidence.pass, false);
  assert.match(evidence.reasons.join('\n'), pattern);
}

test('shader evidence accepts only the pinned first-instance addressing contrast', async () => {
  const evidence = await createFirstInstanceShaderEvidence(contrast());
  assert.equal(evidence.schemaVersion, FIRST_INSTANCE_SHADER_EVIDENCE_SCHEMA_VERSION);
  assert.equal(evidence.kind, 'indirect-first-instance-shader-evidence');
  assert.equal(evidence.pass, true, evidence.reasons.join('\n'));
  assert.deepEqual(evidence.reasons, []);
  assert.deepEqual(evidence.portable.occurrenceCounts, {
    instanceIndex: 2,
    bucketBase: 2,
  });
  assert.deepEqual(evidence.feature.occurrenceCounts, {
    instanceIndex: 2,
    bucketBase: 0,
  });
  assert.equal(evidence.comparison.rawVertexDifferent, true);
  assert.equal(evidence.comparison.rawFragmentEqual, true);
  assert.equal(evidence.comparison.normalizedVertexEqual, true);
  assert.equal(evidence.comparison.normalizedVertexSha256Equal, true);
  assert.notEqual(evidence.portable.raw.vertex.sha256, evidence.feature.raw.vertex.sha256);
  assert.equal(evidence.portable.raw.fragment.sha256, evidence.feature.raw.fragment.sha256);
  assert.equal(
    evidence.portable.normalizedVertex.sha256,
    evidence.feature.normalizedVertex.sha256,
  );
  assert.match(evidence.portable.raw.vertex.sha256, /^[a-f0-9]{64}$/);
  assert.equal(
    evidence.portable.semanticMappings[0].variableName,
    'NodeBuffer_1864',
  );
  assert.equal(
    evidence.feature.semanticMappings[1].variableName,
    'NodeBuffer_2717',
  );
  assert.deepEqual(Object.keys(evidence.feature).sort(), [
    'normalizedVertex',
    'occurrenceCounts',
    'raw',
    'semanticMappings',
    'storageBindings',
    'vertexInputs',
  ]);
});

test('lane-local shader evidence audits feature semantics without a portable lane', async () => {
  const portable = await createFirstInstanceLaneShaderEvidence({
    laneId: 'portable',
    lane: contrast().portable,
    compute: computePhases(3100),
  });
  const first = await createFirstInstanceLaneShaderEvidence({
    laneId: 'feature',
    lane: contrast().feature,
    compute: computePhases(4100),
  });
  const second = await createFirstInstanceLaneShaderEvidence({
    laneId: 'feature',
    lane: contrast({
      featureVertexOptions: {
        matrixName: 'NodeBuffer_9100',
        visibleIdsName: 'NodeBuffer_9101',
      },
    }).feature,
    compute: computePhases(9200),
  });
  assert.equal(portable.pass, true, portable.reasons.join('\n'));
  assert.equal(first.pass, true, first.reasons.join('\n'));
  assert.equal(second.pass, true, second.reasons.join('\n'));
  assert.equal(first.kind, 'indirect-first-instance-lane-shader-evidence');
  assert.equal(first.laneId, 'feature');
  assert.equal(first.addressMode, 'indirect-first-instance');
  assert.equal(first.render.occurrenceCounts.bucketBase, 0);
  assert.equal(first.render.vertexInputs.some((input) => input.name === 'bucketBase'), false);
  assert.equal(first.render.raw.vertex.source.includes('bucketBase'), false);
  assert.equal(first.render.normalizedFragment.source, first.render.raw.fragment.source);
  assert.equal(first.compute.reset.executableFirstInstanceReferences, 0);
  assert.equal(portable.addressMode, 'bucket-base');
  assert.equal(portable.render.occurrenceCounts.bucketBase, 2);
  assert.equal(
    portable.normalizedSemantics.vertexSha256,
    first.normalizedSemantics.vertexSha256,
  );
  assert.equal(
    portable.normalizedSemantics.fragmentSha256,
    first.normalizedSemantics.fragmentSha256,
  );
  assert.notEqual(first.render.raw.vertex.sha256, second.render.raw.vertex.sha256);
  assert.equal(
    first.normalizedSemantics.vertexSha256,
    second.normalizedSemantics.vertexSha256,
  );
  assert.equal(
    first.normalizedSemantics.fragmentSha256,
    second.normalizedSemantics.fragmentSha256,
  );
  for (const phase of ['reset', 'cull']) {
    assert.equal(
      portable.normalizedSemantics.computeSha256ByPhase[phase],
      first.normalizedSemantics.computeSha256ByPhase[phase],
    );
    assert.notEqual(first.compute[phase].raw.sha256, second.compute[phase].raw.sha256);
    assert.equal(
      first.normalizedSemantics.computeSha256ByPhase[phase],
      second.normalizedSemantics.computeSha256ByPhase[phase],
    );
  }
});

test('lane-local shader evidence fails closed on lane mislabeling', async () => {
  const input = contrast();
  const featureAsPortable = await createFirstInstanceLaneShaderEvidence({
    laneId: 'portable',
    lane: input.feature,
  });
  assertFailed(featureAsPortable, /portable WGSL contains an unexpected bucketBase identifier count/);
  const portableAsFeature = await createFirstInstanceLaneShaderEvidence({
    laneId: 'feature',
    lane: input.portable,
  });
  assertFailed(portableAsFeature, /feature WGSL contains an unexpected bucketBase identifier count/);
  const unknown = await createFirstInstanceLaneShaderEvidence({
    laneId: 'experimental',
    lane: input.feature,
  });
  assertFailed(unknown, /laneId must be exactly portable or feature/);
  assert.equal(unknown.laneId, null);
});

test('feature lane-local evidence rejects forbidden inputs, references, and compute access', async () => {
  const forbiddenReference = contrast({
    featureVertexOptions: {
      extraBeforeAddress: '\tlet forbiddenBucketBase : u32 = bucketBase;\n',
    },
  });
  assertFailed(await createFirstInstanceLaneShaderEvidence({
    laneId: 'feature',
    lane: forbiddenReference.feature,
  }), /unexpected bucketBase identifier count/);

  const forbiddenInput = contrast();
  forbiddenInput.feature.vertexInputs.push({ ...forbiddenInput.portable.vertexInputs[1] });
  assertFailed(await createFirstInstanceLaneShaderEvidence({
    laneId: 'feature',
    lane: forbiddenInput.feature,
  }), /unexpected input count/);

  const forbiddenCompute = computePhases(5100);
  forbiddenCompute.reset.shader = forbiddenCompute.reset.shader.replace(
    '.instanceCount',
    '.firstInstance',
  );
  assertFailed(await createFirstInstanceLaneShaderEvidence({
    laneId: 'feature',
    lane: contrast().feature,
    compute: forbiddenCompute,
  }), /must not read or write the indirect firstInstance field/);
});

test('shader evidence rejects an extra vertex operation after targeted normalization', async () => {
  const evidence = await createFirstInstanceShaderEvidence(contrast({
    featureVertexOptions: {
      extraBeforeAddress: '\tlet unrelatedPipelineChange : u32 = 7u;\n',
    },
  }));
  assertFailed(evidence, /normalized vertex WGSL differs/);
});

test('shader evidence rejects any additional instanceIndex use', async () => {
  const evidence = await createFirstInstanceShaderEvidence(contrast({
    featureVertexOptions: {
      extraBeforeAddress: '\tlet leakedInstance : u32 = instanceIndex;\n',
    },
  }));
  assertFailed(evidence, /exactly two instanceIndex identifiers/);
  assert.equal(evidence.feature, null);
});

test('shader evidence requires byte-identical fragment WGSL', async () => {
  const evidence = await createFirstInstanceShaderEvidence(contrast({
    featureFragmentExtra: '\tlet fragmentConfound : f32 = 0.0;\n',
  }));
  assertFailed(evidence, /raw fragment WGSL is not byte-identical/);
  assert.equal(evidence.comparison.rawFragmentEqual, false);
});

test('semantic buffer mapping rejects a metadata/type substitution', async () => {
  const input = contrast();
  input.feature.storageBindings[0].elementType = 'u32';
  const evidence = await createFirstInstanceShaderEvidence(input);
  assertFailed(evidence, /matrix\.elementType must be mat4x4<f32>/);
});

test('numeric NodeBuffer names outside semantic bindings are never blindly normalized', async () => {
  const evidence = await createFirstInstanceShaderEvidence(contrast({
    featureVertexOptions: {
      extraBeforeAddress: '\tlet NodeBuffer_9999 : u32 = 0u;\n',
    },
  }));
  assertFailed(evidence, /normalized vertex WGSL differs/);
});

test('runtime vertex metadata rejects an instance-stepped common attribute', async () => {
  const input = contrast();
  input.feature.vertexInputs[1].stepMode = 'instance';
  const evidence = await createFirstInstanceShaderEvidence(input);
  assertFailed(evidence, /stepMode must equal vertex/);
});

test('cross-lane runtime metadata requires shared resource identities', async () => {
  const input = contrast();
  input.feature.storageBindings[1].resourceId = 'different-visible-ids-attribute';
  const evidence = await createFirstInstanceShaderEvidence(input);
  assertFailed(evidence, /visibleIds must use the same runtime resource/);
  assert.equal(evidence.comparison.normalizedVertexEqual, true);
});

test('a swapped storage coordinate cannot pass through numeric identifier normalization', async () => {
  const input = clone(contrast());
  input.feature.storageBindings[0].binding = 4;
  input.feature.storageBindings[1].binding = 3;
  const evidence = await createFirstInstanceShaderEvidence(input);
  assertFailed(evidence, /matrix storage must precede visibleIds/);
});

function fakeRuntimeVertexEvidence() {
  const attributes = [
    {
      id: 700,
      array: new Float32Array(VERTEX_COUNT * 3),
      itemSize: 3,
      normalized: false,
      count: VERTEX_COUNT,
    },
    {
      id: 702,
      array: new Uint32Array(VERTEX_COUNT),
      itemSize: 1,
      normalized: false,
      count: VERTEX_COUNT,
    },
    {
      id: 701,
      array: new Float32Array(VERTEX_COUNT * 3),
      itemSize: 3,
      normalized: false,
      count: VERTEX_COUNT,
    },
  ];
  const renderObject = {
    getNodeBuilderState: () => ({
      nodeAttributes: [{ name: 'position' }, { name: 'bucketBase' }, { name: 'normal' }],
    }),
    getAttributes: () => attributes,
  };
  const renderer = {
    backend: {
      attributeUtils: {
        createShaderVertexBuffers: () => [
          {
            arrayStride: 12,
            stepMode: 'vertex',
            attributes: [{ shaderLocation: 0, format: 'float32x3' }],
          },
          {
            arrayStride: 4,
            stepMode: 'vertex',
            attributes: [{ shaderLocation: 1, format: 'uint32' }],
          },
          {
            arrayStride: 12,
            stepMode: 'vertex',
            attributes: [{ shaderLocation: 2, format: 'float32x3' }],
          },
        ],
      },
    },
  };
  return { renderer, renderObject };
}

function fakeStorageBinding(attribute, { access = 'readOnly', visibility = 1 } = {}) {
  return {
    isStorageBuffer: true,
    access,
    visibility,
    attribute,
    buffer: attribute.array,
    nodeUniform: { value: attribute },
    getVisibility() {
      return this.visibility;
    },
  };
}

test('r185 vertex evidence comes from render-object attributes and backend layouts', () => {
  const { renderer, renderObject } = fakeRuntimeVertexEvidence();
  const inputs = runtimeVertexInputEvidence(renderer, renderObject);
  assert.deepEqual(inputs.map((input) => ({
    name: input.name,
    shaderLocation: input.shaderLocation,
    format: input.format,
    stepMode: input.stepMode,
    resourceId: input.resourceId,
  })), [
    {
      name: 'position',
      shaderLocation: 0,
      format: 'float32x3',
      stepMode: 'vertex',
      resourceId: 700,
    },
    {
      name: 'bucketBase',
      shaderLocation: 1,
      format: 'uint32',
      stepMode: 'vertex',
      resourceId: 702,
    },
    {
      name: 'normal',
      shaderLocation: 2,
      format: 'float32x3',
      stepMode: 'vertex',
      resourceId: 701,
    },
  ]);
});

test('r185 storage evidence joins actual bindings to WGSL coordinates and resources', () => {
  const matrixAttribute = {
    id: 800,
    count: OBJECT_COUNT,
    array: new Float32Array(OBJECT_COUNT * 16),
  };
  const visibleIdsAttribute = {
    id: 801,
    count: OBJECT_COUNT,
    array: new Uint32Array(OBJECT_COUNT),
  };
  const renderObject = {
    getBindings: () => [
      { bindings: [{ isStorageBuffer: false }] },
      {
        bindings: [
          { isStorageBuffer: false },
          { isStorageBuffer: false },
          { isStorageBuffer: false },
          fakeStorageBinding(matrixAttribute),
          fakeStorageBinding(visibleIdsAttribute),
        ],
      },
    ],
  };
  const records = runtimeStorageBindingEvidence(
    vertexShader({
      feature: false,
      matrixName: 'NodeBuffer_1864',
      visibleIdsName: 'NodeBuffer_1865',
    }),
    renderObject,
    { matrixAttribute, visibleIdsAttribute },
  );
  assert.deepEqual(records, [
    {
      semantic: 'matrix',
      group: 1,
      binding: 3,
      access: 'read',
      visibility: 1,
      elementType: 'mat4x4<f32>',
      count: OBJECT_COUNT,
      byteLength: OBJECT_COUNT * 64,
      resourceId: 800,
    },
    {
      semantic: 'visibleIds',
      group: 1,
      binding: 4,
      access: 'read',
      visibility: 1,
      elementType: 'u32',
      count: OBJECT_COUNT,
      byteLength: OBJECT_COUNT * 4,
      resourceId: 801,
    },
  ]);
});

test('r185 storage evidence rejects an unrecognized bound storage resource', () => {
  const matrixAttribute = {
    id: 800,
    count: OBJECT_COUNT,
    array: new Float32Array(OBJECT_COUNT * 16),
  };
  const visibleIdsAttribute = {
    id: 801,
    count: OBJECT_COUNT,
    array: new Uint32Array(OBJECT_COUNT),
  };
  const unknownAttribute = {
    id: 802,
    count: OBJECT_COUNT,
    array: new Uint32Array(OBJECT_COUNT),
  };
  const renderObject = {
    getBindings: () => [
      { bindings: [{ isStorageBuffer: false }] },
      {
        bindings: [
          { isStorageBuffer: false },
          { isStorageBuffer: false },
          { isStorageBuffer: false },
          fakeStorageBinding(matrixAttribute),
          fakeStorageBinding(visibleIdsAttribute),
          fakeStorageBinding(unknownAttribute),
        ],
      },
    ],
  };
  assert.throws(
    () => runtimeStorageBindingEvidence(
      vertexShader({
        feature: false,
        matrixName: 'NodeBuffer_1864',
        visibleIdsName: 'NodeBuffer_1865',
      }),
      renderObject,
      { matrixAttribute, visibleIdsAttribute },
    ),
    /unexpected storage resource/,
  );
});
