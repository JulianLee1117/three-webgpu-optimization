import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import {
  validateLiveComputeShaderEvidence,
} from '../scripts/live-first-instance-evidence-validation.mjs';
import {
  normalizeLiveIndirectCommandComputeShader,
} from '../src/validation/live-compute-shader-normalization.js';

function bindings(binding = 4) {
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

function computeSource(identifier, operation = 'atomicAdd', binding = 4) {
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
  ${operation}( &${identifier}.value[ instanceIndex ].instanceCount, 1u );
}`;
}

test('live compute normalization changes only the resolved lane-local command symbol', () => {
  const portable = normalizeLiveIndirectCommandComputeShader(
    computeSource('NodeBuffer_1853'),
    bindings(),
  );
  const feature = normalizeLiveIndirectCommandComputeShader(
    computeSource('NodeBuffer_1870'),
    bindings(),
  );
  assert.equal(portable.normalizedShader, feature.normalizedShader);
  assert.deepEqual(portable.audit, {
    schemaVersion: 1,
    kind: 'live-indirect-command-wgsl-identifier-normalization',
    pass: true,
    semantic: 'indirectCommands',
    group: 0,
    binding: 4,
    generatedVariableIdentifier: 'NodeBuffer_1853',
    generatedStructIdentifier: 'NodeBuffer_1853Struct',
    canonicalVariableIdentifier: 'LiveLaneIndirectCommands',
    canonicalStructIdentifier: 'LiveLaneIndirectCommandsStruct',
    commandStructIdentifier: 'FixedSliceIndexedDraw',
    commandStructFields: [
      'indexCount:u32',
      'instanceCount:atomic<u32>',
      'firstIndex:u32',
      'baseVertex:i32',
      'firstInstance:u32',
    ],
    variableTokenCount: 2,
    structTokenCount: 2,
    commentIdentifierTokenCount: 0,
  });
});

test('live compute normalization does not hide any other shader difference', () => {
  const portable = normalizeLiveIndirectCommandComputeShader(
    computeSource('NodeBuffer_1853', 'atomicAdd'),
    bindings(),
  );
  const feature = normalizeLiveIndirectCommandComputeShader(
    computeSource('NodeBuffer_1870', 'atomicStore'),
    bindings(),
  );
  assert.notEqual(portable.normalizedShader, feature.normalizedShader);
  for (const changed of [
    computeSource('NodeBuffer_1870').replace('1u );', '2u );'),
    computeSource('NodeBuffer_1870').replace('fn main()', 'fn  main()'),
    `${computeSource('NodeBuffer_1870')}\nstruct NodeBuffer_999Struct { value: u32 };`,
  ]) {
    assert.notEqual(
      portable.normalizedShader,
      normalizeLiveIndirectCommandComputeShader(changed, bindings()).normalizedShader,
    );
  }
});

test('live compute normalization fails closed on an unproven coordinate or identifier', () => {
  assert.throws(
    () => normalizeLiveIndirectCommandComputeShader(
      computeSource('NodeBuffer_1853'),
      bindings(7),
    ),
    /resolve exactly one WGSL storage declaration/,
  );
  assert.throws(
    () => normalizeLiveIndirectCommandComputeShader(
      computeSource('commandOutput'),
      bindings(),
    ),
    /pinned r185 indirectCommands WGSL identifiers are malformed/,
  );
  assert.throws(
    () => normalizeLiveIndirectCommandComputeShader(
      computeSource('NodeBuffer_1853').replace('read_write', 'read'),
      bindings(),
    ),
    /resolve exactly one WGSL storage declaration/,
  );
  assert.throws(
    () => normalizeLiveIndirectCommandComputeShader(
      `${computeSource('NodeBuffer_1853')}\n// NodeBuffer_1853 is lane-local`,
      bindings(),
    ),
    /occurs inside a WGSL comment/,
  );
  assert.throws(
    () => normalizeLiveIndirectCommandComputeShader(
      computeSource('NodeBuffer_1853').replace('firstInstance : u32', 'firstInstance : i32'),
      bindings(),
    ),
    /indexed-indirect command struct is malformed/,
  );
  assert.throws(
    () => normalizeLiveIndirectCommandComputeShader(
      computeSource('NodeBuffer_1853'),
      [{ ...bindings()[0], itemSize: 4 }],
    ),
    /runtime binding shape is malformed/,
  );
  assert.throws(
    () => normalizeLiveIndirectCommandComputeShader(
      computeSource('NodeBuffer_1853').replaceAll(
        'FixedSliceIndexedDraw',
        'OtherIndexedDraw',
      ),
      bindings(),
    ),
    /wrapper element type is malformed/,
  );
  assert.throws(
    () => normalizeLiveIndirectCommandComputeShader(
      `${computeSource('NodeBuffer_1853')}\n@binding( 4 ) @group( 0 )\n`
        + 'var<storage, read_write> NodeBuffer_1853 : NodeBuffer_1853Struct;',
      bindings(),
    ),
    /resolve exactly one WGSL storage declaration/,
  );
});

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function phaseBindings(phase, lane) {
  const semantics = phase === 'reset'
    ? ['indirectCommands', 'overflow']
    : [
      'cullOrder', 'bounds', 'uniforms', 'objectBucket', 'indirectCommands',
      'bucketCapacity', 'visibleIds', 'bucketBase', 'overflow',
    ];
  return semantics.map((semantic, binding) => semantic === 'uniforms'
    ? {
      group: 0,
      binding,
      kind: 'uniform-buffer',
      semantic,
      access: null,
      visibility: 4,
      byteLength: 96,
      resourceId: null,
      attributeType: null,
      arrayType: null,
      itemSize: null,
      count: null,
      uniformValues: [1, 2, 3, 4],
    }
    : {
      group: 0,
      binding,
      kind: 'storage-buffer',
      semantic,
      access: semantic === 'indirectCommands' ? 'readWrite' : 'readOnly',
      visibility: 4,
      byteLength: semantic === 'indirectCommands' ? 640 : 262_144,
      resourceId: semantic === 'indirectCommands' ? `${lane}-commands` : semantic,
      attributeType: semantic === 'indirectCommands'
        ? 'IndirectStorageBufferAttribute'
        : 'StorageBufferAttribute',
      arrayType: 'Uint32Array',
      itemSize: semantic === 'indirectCommands' ? 5 : 1,
      count: semantic === 'indirectCommands' ? 32 : 65_536,
      uniformValues: null,
    });
}

function computePhaseEvidence(phase) {
  const binding = phase === 'reset' ? 0 : 4;
  const count = phase === 'reset' ? 32 : 65_536;
  const dispatch = phase === 'reset' ? [1, 1, 1] : [1024, 1, 1];
  const sources = {
    portable: computeSource('NodeBuffer_1853', 'atomicAdd', binding),
    feature: computeSource('NodeBuffer_1870', 'atomicAdd', binding),
  };
  const laneBindings = {
    portable: phaseBindings(phase, 'portable'),
    feature: phaseBindings(phase, 'feature'),
  };
  const normalized = Object.fromEntries(['portable', 'feature'].map((lane) => [
    lane,
    normalizeLiveIndirectCommandComputeShader(sources[lane], laneBindings[lane]),
  ]));
  const storageSemantics = laneBindings.portable
    .filter((record) => record.kind === 'storage-buffer')
    .map((record) => record.semantic);
  const resourceComparisons = storageSemantics.map((semantic) => {
    const portable = laneBindings.portable.find((record) => record.semantic === semantic);
    const feature = laneBindings.feature.find((record) => record.semantic === semantic);
    return {
      semantic,
      group: portable.group,
      binding: portable.binding,
      portableResourceId: portable.resourceId,
      featureResourceId: feature.resourceId,
      shouldShare: semantic !== 'indirectCommands',
      pass: true,
    };
  });
  return {
    pass: true,
    rawWgslEqual: false,
    normalizedWgslEqual: true,
    rawDifferenceRestrictedToLaneLocalIndirectBinding: true,
    workgroupDeclaration: '@workgroup_size( 64, 1, 1 )',
    executionEqual: true,
    fixedWorkloadExact: true,
    fixedExpectation: { count, workgroupSize: [64, 1, 1], derivedDispatchSize: dispatch },
    bindings: {
      pass: true,
      shapeEqual: true,
      uniformValuesEqual: true,
      resourceComparisons,
    },
    wordFour: Object.fromEntries(['portable', 'feature'].map((lane) => [lane, {
      pass: true,
      declarationCount: 1,
      executableAccessCount: 0,
    }])),
    normalization: Object.fromEntries(['portable', 'feature'].map((lane) => [
      lane,
      normalized[lane].audit,
    ])),
    sourceDigests: Object.fromEntries(['portable', 'feature'].map((lane) => [lane, {
      rawSha256: sha256(sources[lane]),
      normalizedSha256: sha256(normalized[lane].normalizedShader),
    }])),
    lanes: Object.fromEntries(['portable', 'feature'].map((lane) => [lane, {
      computeShader: sources[lane],
      bindings: laneBindings[lane],
      execution: {
        count,
        workgroupSize: [64, 1, 1],
        invocationsPerWorkgroup: 64,
        derivedDispatchSize: dispatch,
        runtimeDispatchSize: dispatch,
        runtimeMatchesDerived: true,
      },
    }])),
  };
}

function computeEvidence() {
  return {
    schemaVersion: 1,
    kind: 'live-first-instance-compute-shader-evidence',
    pass: true,
    dispatchDimensionsEqual: true,
    dispatchDimensions: { reset: [1, 1, 1], cull: [1024, 1, 1] },
    fixedWorkloadExact: true,
    maxStorageBindingCount: 8,
    phases: { reset: computePhaseEvidence('reset'), cull: computePhaseEvidence('cull') },
  };
}

test('strict compute verifier reconstructs normalization audits and hashes', () => {
  const evidence = computeEvidence();
  assert.deepEqual(validateLiveComputeShaderEvidence(evidence), []);

  const auditTamper = structuredClone(evidence);
  auditTamper.phases.reset.normalization.feature.variableTokenCount += 1;
  assert.match(
    validateLiveComputeShaderEvidence(auditTamper).join('\n'),
    /normalization audit/,
  );

  const hashTamper = structuredClone(evidence);
  hashTamper.phases.cull.sourceDigests.portable.rawSha256 = '0'.repeat(64);
  assert.match(validateLiveComputeShaderEvidence(hashTamper).join('\n'), /raw WGSL SHA-256/);
});
