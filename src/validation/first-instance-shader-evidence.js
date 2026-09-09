import { normalizeLiveIndirectCommandComputeShader } from './live-compute-shader-normalization.js';

const SCHEMA_VERSION = 1;

const LANE_IDS = Object.freeze({
  PORTABLE: 'portable',
  FEATURE: 'feature',
});

const STORAGE_SEMANTICS = Object.freeze({
  MATRIX: 'matrix',
  VISIBLE_IDS: 'visibleIds',
});

const SEMANTIC_BUFFER_NAMES = Object.freeze({
  [STORAGE_SEMANTICS.MATRIX]: '__FIRST_INSTANCE_MATRIX_BUFFER__',
  [STORAGE_SEMANTICS.VISIBLE_IDS]: '__FIRST_INSTANCE_VISIBLE_IDS_BUFFER__',
});

const SEMANTIC_STRUCT_NAMES = Object.freeze({
  [STORAGE_SEMANTICS.MATRIX]: '__FIRST_INSTANCE_MATRIX_BUFFER_STRUCT__',
  [STORAGE_SEMANTICS.VISIBLE_IDS]: '__FIRST_INSTANCE_VISIBLE_IDS_BUFFER_STRUCT__',
});

const EXPECTED_ELEMENT_TYPES = Object.freeze({
  [STORAGE_SEMANTICS.MATRIX]: 'mat4x4<f32>',
  [STORAGE_SEMANTICS.VISIBLE_IDS]: 'u32',
});

const ELEMENT_BYTE_SIZES = Object.freeze({
  [STORAGE_SEMANTICS.MATRIX]: 64,
  [STORAGE_SEMANTICS.VISIBLE_IDS]: 4,
});

const PHASE0_LANE_IDS = Object.freeze(['A', 'I', 'F']);
const IMMEDIATE_REQUIREMENT = 'immediate_address_space';
const IMMEDIATE_VARIABLE = 'threeImmediateDrawBase';

const EXPECTED_VERTEX_INPUTS = Object.freeze({
  [LANE_IDS.PORTABLE]: Object.freeze([
    Object.freeze({
      kind: 'builtin',
      builtin: 'instance_index',
      name: 'instanceIndex',
      wgslType: 'u32',
    }),
    Object.freeze({
      kind: 'location',
      location: 0,
      name: 'position',
      wgslType: 'vec3<f32>',
    }),
    Object.freeze({
      kind: 'location',
      location: 1,
      name: 'bucketBase',
      wgslType: 'u32',
    }),
    Object.freeze({
      kind: 'location',
      location: 2,
      name: 'normal',
      wgslType: 'vec3<f32>',
    }),
  ]),
  [LANE_IDS.FEATURE]: Object.freeze([
    Object.freeze({
      kind: 'builtin',
      builtin: 'instance_index',
      name: 'instanceIndex',
      wgslType: 'u32',
    }),
    Object.freeze({
      kind: 'location',
      location: 0,
      name: 'position',
      wgslType: 'vec3<f32>',
    }),
    Object.freeze({
      kind: 'location',
      location: 1,
      name: 'normal',
      wgslType: 'vec3<f32>',
    }),
  ]),
});

const EXPECTED_RUNTIME_INPUTS = Object.freeze({
  [LANE_IDS.PORTABLE]: Object.freeze([
    Object.freeze({
      name: 'position',
      shaderLocation: 0,
      format: 'float32x3',
      stepMode: 'vertex',
      arrayType: 'Float32Array',
      itemSize: 3,
      normalized: false,
    }),
    Object.freeze({
      name: 'bucketBase',
      shaderLocation: 1,
      format: 'uint32',
      stepMode: 'vertex',
      arrayType: 'Uint32Array',
      itemSize: 1,
      normalized: false,
    }),
    Object.freeze({
      name: 'normal',
      shaderLocation: 2,
      format: 'float32x3',
      stepMode: 'vertex',
      arrayType: 'Float32Array',
      itemSize: 3,
      normalized: false,
    }),
  ]),
  [LANE_IDS.FEATURE]: Object.freeze([
    Object.freeze({
      name: 'position',
      shaderLocation: 0,
      format: 'float32x3',
      stepMode: 'vertex',
      arrayType: 'Float32Array',
      itemSize: 3,
      normalized: false,
    }),
    Object.freeze({
      name: 'normal',
      shaderLocation: 1,
      format: 'float32x3',
      stepMode: 'vertex',
      arrayType: 'Float32Array',
      itemSize: 3,
      normalized: false,
    }),
  ]),
});

class ShaderEvidenceError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ShaderEvidenceError';
  }
}

function assertAudit(condition, message) {
  if (!condition) throw new ShaderEvidenceError(message);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function normalizedType(value) {
  return typeof value === 'string' ? value.replace(/\s+/g, '') : null;
}

function countIdentifier(source, identifier) {
  return source.match(new RegExp(`\\b${escapeRegExp(identifier)}\\b`, 'g'))?.length ?? 0;
}

function replaceIdentifier(source, identifier, replacement) {
  return source.replace(
    new RegExp(`\\b${escapeRegExp(identifier)}\\b`, 'g'),
    replacement,
  );
}

function countMatches(source, pattern) {
  return [...source.matchAll(new RegExp(pattern.source, pattern.flags.includes('g')
    ? pattern.flags
    : `${pattern.flags}g`))].length;
}

function findMatchingDelimiter(source, start, open, close) {
  let depth = 0;
  for (let index = start; index < source.length; index += 1) {
    if (source[index] === open) depth += 1;
    else if (source[index] === close) {
      depth -= 1;
      if (depth === 0) return index;
    }
  }
  return -1;
}

function parseVertexInputs(vertexShader) {
  const entryMatch = /@vertex\s+fn\s+main\s*\(/m.exec(vertexShader);
  assertAudit(entryMatch !== null, 'vertex WGSL must expose one @vertex fn main entry point.');
  assertAudit(
    countMatches(vertexShader, /@vertex\s+fn\s+main\s*\(/m) === 1,
    'vertex WGSL must expose exactly one @vertex fn main entry point.',
  );
  const openIndex = vertexShader.indexOf('(', entryMatch.index);
  const closeIndex = findMatchingDelimiter(vertexShader, openIndex, '(', ')');
  assertAudit(closeIndex > openIndex, 'vertex WGSL main parameters are malformed.');
  const parameters = vertexShader.slice(openIndex + 1, closeIndex);
  const parameterPattern = /@(builtin|location)\s*\(\s*([^)]+?)\s*\)\s*([A-Za-z_]\w*)\s*:\s*([^,\n)]+)/g;
  const parsed = [];
  for (const match of parameters.matchAll(parameterPattern)) {
    const annotationKind = match[1];
    const annotationValue = match[2].trim();
    const name = match[3];
    const wgslType = normalizedType(match[4]);
    if (annotationKind === 'builtin') {
      parsed.push({ kind: 'builtin', builtin: annotationValue, name, wgslType });
    } else {
      assertAudit(/^\d+$/.test(annotationValue), 'vertex location must be an integer.');
      parsed.push({
        kind: 'location',
        location: Number(annotationValue),
        name,
        wgslType,
      });
    }
  }
  const residual = parameters.replace(parameterPattern, '').replaceAll(',', '').trim();
  assertAudit(residual === '', `vertex WGSL contains an unsupported entry input: ${residual}`);
  return parsed;
}

function expectedVertexInputRecords(laneId) {
  return EXPECTED_VERTEX_INPUTS[laneId].map((input) => ({ ...input }));
}

function validateParsedVertexInputs(laneId, parsed) {
  const expected = expectedVertexInputRecords(laneId);
  assertAudit(
    JSON.stringify(parsed) === JSON.stringify(expected),
    `${laneId} vertex WGSL inputs differ from the pinned addressing contrast.`,
  );
}

function validateResourceId(value, label) {
  const valid = (typeof value === 'string' && value.length > 0)
    || (typeof value === 'number' && Number.isFinite(value));
  assertAudit(valid, `${label}.resourceId must be a nonempty string or finite number.`);
}

function validateRuntimeVertexInputs(laneId, vertexInputs) {
  assertAudit(Array.isArray(vertexInputs), `${laneId}.vertexInputs must be an array.`);
  const expected = EXPECTED_RUNTIME_INPUTS[laneId];
  assertAudit(
    vertexInputs.length === expected.length,
    `${laneId}.vertexInputs has an unexpected input count.`,
  );
  const sanitized = vertexInputs.map((input, index) => {
    assertAudit(input && typeof input === 'object', `${laneId}.vertexInputs[${index}] is invalid.`);
    const expectedInput = expected[index];
    for (const [name, expectedValue] of Object.entries(expectedInput)) {
      assertAudit(
        input[name] === expectedValue,
        `${laneId}.vertexInputs[${index}].${name} must equal ${String(expectedValue)}.`,
      );
    }
    assertAudit(
      Number.isInteger(input.count) && input.count > 0,
      `${laneId}.vertexInputs[${index}].count must be a positive integer.`,
    );
    validateResourceId(input.resourceId, `${laneId}.vertexInputs[${index}]`);
    return {
      ...expectedInput,
      count: input.count,
      resourceId: input.resourceId,
    };
  });
  const vertexCount = sanitized[0].count;
  assertAudit(
    sanitized.every((input) => input.count === vertexCount),
    `${laneId} vertex input counts must be identical.`,
  );
  return sanitized;
}

function normalizeVisibility(value) {
  if (value === 'vertex' || value === 1) return 'vertex';
  return null;
}

function validateStorageBindingMetadata(laneId, storageBindings) {
  assertAudit(Array.isArray(storageBindings), `${laneId}.storageBindings must be an array.`);
  assertAudit(storageBindings.length === 2, `${laneId} must expose exactly two storage bindings.`);
  const bySemantic = new Map();
  for (let index = 0; index < storageBindings.length; index += 1) {
    const binding = storageBindings[index];
    assertAudit(binding && typeof binding === 'object', `${laneId}.storageBindings[${index}] is invalid.`);
    assertAudit(
      Object.values(STORAGE_SEMANTICS).includes(binding.semantic),
      `${laneId}.storageBindings[${index}] has an unknown semantic.`,
    );
    assertAudit(
      !bySemantic.has(binding.semantic),
      `${laneId} duplicates the ${binding.semantic} storage semantic.`,
    );
    assertAudit(
      Number.isInteger(binding.group) && binding.group >= 0,
      `${laneId}.${binding.semantic}.group must be a nonnegative integer.`,
    );
    assertAudit(
      Number.isInteger(binding.binding) && binding.binding >= 0,
      `${laneId}.${binding.semantic}.binding must be a nonnegative integer.`,
    );
    assertAudit(binding.access === 'read', `${laneId}.${binding.semantic}.access must be read.`);
    assertAudit(
      normalizeVisibility(binding.visibility) === 'vertex',
      `${laneId}.${binding.semantic}.visibility must be vertex-only.`,
    );
    const expectedElementType = EXPECTED_ELEMENT_TYPES[binding.semantic];
    assertAudit(
      normalizedType(binding.elementType) === expectedElementType,
      `${laneId}.${binding.semantic}.elementType must be ${expectedElementType}.`,
    );
    assertAudit(
      Number.isInteger(binding.count) && binding.count > 0,
      `${laneId}.${binding.semantic}.count must be a positive integer.`,
    );
    const expectedByteLength = binding.count * ELEMENT_BYTE_SIZES[binding.semantic];
    assertAudit(
      binding.byteLength === expectedByteLength,
      `${laneId}.${binding.semantic}.byteLength must equal ${expectedByteLength}.`,
    );
    validateResourceId(binding.resourceId, `${laneId}.${binding.semantic}`);
    bySemantic.set(binding.semantic, {
      semantic: binding.semantic,
      group: binding.group,
      binding: binding.binding,
      access: 'read',
      visibility: 'vertex',
      elementType: expectedElementType,
      count: binding.count,
      byteLength: binding.byteLength,
      resourceId: binding.resourceId,
    });
  }
  for (const semantic of Object.values(STORAGE_SEMANTICS)) {
    assertAudit(bySemantic.has(semantic), `${laneId} is missing the ${semantic} storage binding.`);
  }
  const matrix = bySemantic.get(STORAGE_SEMANTICS.MATRIX);
  const visibleIds = bySemantic.get(STORAGE_SEMANTICS.VISIBLE_IDS);
  assertAudit(
    matrix.group === visibleIds.group && matrix.binding < visibleIds.binding,
    `${laneId} matrix storage must precede visibleIds in the same bind group.`,
  );
  assertAudit(
    matrix.count === visibleIds.count,
    `${laneId} matrix and visibleIds storage counts must match.`,
  );
  return bySemantic;
}

function parseStorageDeclarations(vertexShader) {
  const declarationPattern = /((?:@(binding|group)\s*\(\s*\d+\s*\)\s*){2})var\s*<\s*storage\s*,\s*(read|read_write)\s*>\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)\s*;/g;
  const declarations = [];
  for (const match of vertexShader.matchAll(declarationPattern)) {
    const annotations = match[1];
    const coordinates = {};
    for (const annotation of annotations.matchAll(/@(binding|group)\s*\(\s*(\d+)\s*\)/g)) {
      assertAudit(
        coordinates[annotation[1]] === undefined,
        'storage WGSL repeats a binding coordinate annotation.',
      );
      coordinates[annotation[1]] = Number(annotation[2]);
    }
    assertAudit(
      Number.isInteger(coordinates.group) && Number.isInteger(coordinates.binding),
      'storage WGSL must provide group and binding coordinates.',
    );
    declarations.push({
      group: coordinates.group,
      binding: coordinates.binding,
      access: match[3],
      variableName: match[4],
      structName: match[5],
    });
  }
  return declarations;
}

function parseStorageStructElementType(vertexShader, structName) {
  const pattern = new RegExp(
    `struct\\s+${escapeRegExp(structName)}\\s*\\{([\\s\\S]*?)\\}\\s*;`,
    'g',
  );
  const matches = [...vertexShader.matchAll(pattern)];
  assertAudit(matches.length === 1, `storage struct ${structName} must be declared exactly once.`);
  const field = /^\s*value\s*:\s*array\s*<([\s\S]+)>\s*,?\s*$/.exec(matches[0][1]);
  assertAudit(field !== null, `storage struct ${structName} must contain only an array value field.`);
  return normalizedType(field[1]);
}

function storageCoordinateKey(group, binding) {
  return `${group}:${binding}`;
}

function mapSemanticStorageIdentifiers(laneId, vertexShader, metadataBySemantic) {
  const declarations = parseStorageDeclarations(vertexShader);
  assertAudit(
    declarations.length === metadataBySemantic.size,
    `${laneId} vertex WGSL storage declaration count differs from runtime metadata.`,
  );
  const declarationsByCoordinate = new Map();
  for (const declaration of declarations) {
    const key = storageCoordinateKey(declaration.group, declaration.binding);
    assertAudit(
      !declarationsByCoordinate.has(key),
      `${laneId} vertex WGSL duplicates storage coordinate ${key}.`,
    );
    declarationsByCoordinate.set(key, declaration);
  }

  let normalized = vertexShader;
  const mappings = [];
  for (const semantic of Object.values(STORAGE_SEMANTICS)) {
    const metadata = metadataBySemantic.get(semantic);
    const key = storageCoordinateKey(metadata.group, metadata.binding);
    const declaration = declarationsByCoordinate.get(key);
    assertAudit(declaration !== undefined, `${laneId} WGSL is missing ${semantic} at ${key}.`);
    assertAudit(
      declaration.access === metadata.access,
      `${laneId} WGSL ${semantic} access differs from runtime metadata.`,
    );
    const elementType = parseStorageStructElementType(vertexShader, declaration.structName);
    assertAudit(
      elementType === metadata.elementType,
      `${laneId} WGSL ${semantic} element type must be ${metadata.elementType}.`,
    );
    assertAudit(
      countIdentifier(vertexShader, declaration.variableName) === 2,
      `${laneId} WGSL ${semantic} variable must have one declaration and one use.`,
    );
    assertAudit(
      countIdentifier(vertexShader, declaration.structName) === 2,
      `${laneId} WGSL ${semantic} struct must have one declaration and one binding use.`,
    );
    mappings.push({
      semantic,
      group: metadata.group,
      binding: metadata.binding,
      access: metadata.access,
      elementType,
      variableName: declaration.variableName,
      structName: declaration.structName,
    });
    normalized = replaceIdentifier(
      normalized,
      declaration.structName,
      SEMANTIC_STRUCT_NAMES[semantic],
    );
    normalized = replaceIdentifier(
      normalized,
      declaration.variableName,
      SEMANTIC_BUFFER_NAMES[semantic],
    );
  }
  return { normalized, mappings };
}

function replaceUnique(source, pattern, replacement, failureMessage) {
  assertAudit(countMatches(source, pattern) === 1, failureMessage);
  return source.replace(pattern, replacement);
}

function normalizeAddressingContrast(laneId, vertexShader) {
  const matrixName = escapeRegExp(SEMANTIC_BUFFER_NAMES[STORAGE_SEMANTICS.MATRIX]);
  const visibleName = escapeRegExp(SEMANTIC_BUFFER_NAMES[STORAGE_SEMANTICS.VISIBLE_IDS]);
  const expectedIndex = laneId === LANE_IDS.PORTABLE
    ? '\\(\\s*bucketBase\\s*\\+\\s*instanceIndex\\s*\\)'
    : 'instanceIndex';
  const fullAddressPattern = new RegExp(
    `${matrixName}\\s*\\.\\s*value\\s*\\[\\s*${visibleName}`
      + `\\s*\\.\\s*value\\s*\\[\\s*${expectedIndex}\\s*\\]\\s*\\]`,
    'g',
  );
  assertAudit(
    countMatches(vertexShader, fullAddressPattern) === 1,
    `${laneId} WGSL must contain exactly one pinned matrix/visibleIds address expression.`,
  );
  const visibleIndexPattern = new RegExp(
    `(${visibleName}\\s*\\.\\s*value\\s*\\[\\s*)${expectedIndex}(\\s*\\])`,
    'g',
  );
  return replaceUnique(
    vertexShader,
    visibleIndexPattern,
    '$1__FIRST_INSTANCE_ADDRESS_INDEX__$2',
    `${laneId} WGSL visibleIds address must be unique.`,
  );
}

function normalizeVertexEntryContrast(laneId, vertexShader) {
  let normalized = vertexShader;
  if (laneId === LANE_IDS.PORTABLE) {
    const bucketParameter = /@location\s*\(\s*1\s*\)\s*bucketBase\s*:\s*u32\s*,\s*/g;
    normalized = replaceUnique(
      normalized,
      bucketParameter,
      '',
      'portable WGSL bucketBase entry parameter must be unique.',
    );
    const normalParameter = /@location\s*\(\s*2\s*\)\s*normal\s*:\s*vec3\s*<\s*f32\s*>/g;
    normalized = replaceUnique(
      normalized,
      normalParameter,
      (match) => match.replace(/\(\s*2\s*\)/, '( 1 )'),
      'portable WGSL normal location must be uniquely addressable.',
    );
  }
  return normalized;
}

function auditLane(laneId, lane) {
  assertAudit(lane && typeof lane === 'object', `${laneId} lane must be an object.`);
  assertAudit(
    typeof lane.vertexShader === 'string' && lane.vertexShader.length > 0,
    `${laneId}.vertexShader must be a nonempty string.`,
  );
  assertAudit(
    typeof lane.fragmentShader === 'string' && lane.fragmentShader.length > 0,
    `${laneId}.fragmentShader must be a nonempty string.`,
  );
  const instanceIndexOccurrences = countIdentifier(lane.vertexShader, 'instanceIndex');
  const bucketBaseOccurrences = countIdentifier(lane.vertexShader, 'bucketBase');
  assertAudit(
    instanceIndexOccurrences === 2,
    `${laneId} WGSL must contain exactly two instanceIndex identifiers.`,
  );
  assertAudit(
    bucketBaseOccurrences === (laneId === LANE_IDS.PORTABLE ? 2 : 0),
    `${laneId} WGSL contains an unexpected bucketBase identifier count.`,
  );
  const parsedVertexInputs = parseVertexInputs(lane.vertexShader);
  validateParsedVertexInputs(laneId, parsedVertexInputs);
  const runtimeVertexInputs = validateRuntimeVertexInputs(laneId, lane.vertexInputs);
  const storageMetadata = validateStorageBindingMetadata(laneId, lane.storageBindings);
  const semanticMapping = mapSemanticStorageIdentifiers(
    laneId,
    lane.vertexShader,
    storageMetadata,
  );
  let normalizedVertexShader = normalizeAddressingContrast(
    laneId,
    semanticMapping.normalized,
  );
  normalizedVertexShader = normalizeVertexEntryContrast(laneId, normalizedVertexShader);
  assertAudit(
    countIdentifier(normalizedVertexShader, 'bucketBase') === 0,
    `${laneId} normalization left a bucketBase identifier.`,
  );
  assertAudit(
    countIdentifier(normalizedVertexShader, 'instanceIndex') === 1,
    `${laneId} normalization changed an unapproved instanceIndex use.`,
  );
  return {
    laneId,
    vertexShader: lane.vertexShader,
    fragmentShader: lane.fragmentShader,
    normalizedVertexShader,
    parsedVertexInputs,
    runtimeVertexInputs,
    storageBindings: Object.fromEntries(
      [...storageMetadata.entries()].map(([semantic, binding]) => [semantic, binding]),
    ),
    semanticMappings: semanticMapping.mappings,
    occurrenceCounts: {
      instanceIndex: instanceIndexOccurrences,
      bucketBase: bucketBaseOccurrences,
    },
  };
}

function validateCrossLaneRuntimeIdentity(portableAudit, featureAudit) {
  for (const name of ['position', 'normal']) {
    const portable = portableAudit.runtimeVertexInputs.find((input) => input.name === name);
    const feature = featureAudit.runtimeVertexInputs.find((input) => input.name === name);
    assertAudit(
      portable.resourceId === feature.resourceId,
      `${name} must use the same runtime resource in both lanes.`,
    );
    assertAudit(portable.count === feature.count, `${name} count differs between lanes.`);
  }
  const portableBucket = portableAudit.runtimeVertexInputs.find(
    (input) => input.name === 'bucketBase',
  );
  assertAudit(
    portableBucket.count === portableAudit.runtimeVertexInputs[0].count,
    'portable bucketBase count must equal the common vertex count.',
  );
  for (const semantic of Object.values(STORAGE_SEMANTICS)) {
    const portable = portableAudit.storageBindings[semantic];
    const feature = featureAudit.storageBindings[semantic];
    for (const property of [
      'group', 'binding', 'access', 'visibility', 'elementType', 'count', 'byteLength',
    ]) {
      assertAudit(
        portable[property] === feature[property],
        `${semantic}.${property} differs between lanes.`,
      );
    }
    assertAudit(
      portable.resourceId === feature.resourceId,
      `${semantic} must use the same runtime resource in both lanes.`,
    );
  }
}

function encodedBytes(value) {
  return new TextEncoder().encode(value);
}

function toHex(buffer) {
  return [...new Uint8Array(buffer)]
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

async function shaderDigest(value) {
  const bytes = encodedBytes(value);
  assertAudit(
    globalThis.crypto?.subtle && typeof globalThis.crypto.subtle.digest === 'function',
    'Web Crypto SHA-256 support is required for shader evidence.',
  );
  return {
    byteLength: bytes.byteLength,
    sha256: toHex(await globalThis.crypto.subtle.digest('SHA-256', bytes)),
  };
}

async function laneDigestRecord(audit) {
  if (!audit) return null;
  const [vertex, fragment, normalizedVertex] = await Promise.all([
    shaderDigest(audit.vertexShader),
    shaderDigest(audit.fragmentShader),
    shaderDigest(audit.normalizedVertexShader),
  ]);
  return {
    raw: { vertex, fragment },
    normalizedVertex,
    vertexInputs: audit.runtimeVertexInputs,
    storageBindings: audit.storageBindings,
    semanticMappings: audit.semanticMappings,
    occurrenceCounts: audit.occurrenceCounts,
  };
}

function laneAddressMode(laneId) {
  if (laneId === LANE_IDS.PORTABLE) return 'bucket-base';
  if (laneId === LANE_IDS.FEATURE) return 'indirect-first-instance';
  throw new ShaderEvidenceError(
    `laneId must be exactly ${LANE_IDS.PORTABLE} or ${LANE_IDS.FEATURE}.`,
  );
}

function sourceDigestRecord(source, digest) {
  return { source, ...digest };
}

async function laneLocalRenderRecord(audit) {
  const legacy = await laneDigestRecord(audit);
  const normalizedFragment = await shaderDigest(audit.fragmentShader);
  return {
    laneId: audit.laneId,
    addressMode: laneAddressMode(audit.laneId),
    raw: {
      vertex: sourceDigestRecord(audit.vertexShader, legacy.raw.vertex),
      fragment: sourceDigestRecord(audit.fragmentShader, legacy.raw.fragment),
    },
    normalizedVertex: sourceDigestRecord(
      audit.normalizedVertexShader,
      legacy.normalizedVertex,
    ),
    normalizedFragment: sourceDigestRecord(audit.fragmentShader, normalizedFragment),
    fragmentNormalization: 'identity-no-approved-lane-contrast',
    vertexInputs: legacy.vertexInputs,
    storageBindings: legacy.storageBindings,
    semanticMappings: legacy.semanticMappings,
    occurrenceCounts: legacy.occurrenceCounts,
  };
}

function computePhaseSource(phase, label) {
  assertAudit(phase && typeof phase === 'object', `${label} must be an object.`);
  const shader = phase.shader;
  const computeShader = phase.computeShader;
  assertAudit(
    shader === undefined || computeShader === undefined || shader === computeShader,
    `${label}.shader and ${label}.computeShader disagree.`,
  );
  const source = shader ?? computeShader;
  assertAudit(
    typeof source === 'string' && source.length > 0,
    `${label} requires a nonempty shader or computeShader string.`,
  );
  return source;
}

async function auditComputePhase(phaseName, phase) {
  const label = `compute.${phaseName}`;
  const source = computePhaseSource(phase, label);
  assertAudit(Array.isArray(phase.bindings), `${label}.bindings must be an array.`);
  assertAudit(
    countMatches(source, /@compute\s+@workgroup_size\s*\([^)]*\)\s+fn\s+main\s*\(/m) === 1,
    `${label} must expose exactly one @compute @workgroup_size fn main entry point.`,
  );
  const executableFirstInstanceReferences = countMatches(
    source,
    /\.\s*firstInstance\b/g,
  );
  assertAudit(
    executableFirstInstanceReferences === 0,
    `${label} must not read or write the indirect firstInstance field.`,
  );
  const normalized = normalizeLiveIndirectCommandComputeShader(source, phase.bindings);
  const [rawDigest, normalizedDigest] = await Promise.all([
    shaderDigest(source),
    shaderDigest(normalized.normalizedShader),
  ]);
  return {
    raw: sourceDigestRecord(source, rawDigest),
    normalized: sourceDigestRecord(normalized.normalizedShader, normalizedDigest),
    normalization: normalized.audit,
    executableFirstInstanceReferences,
    bindings: structuredClone(phase.bindings),
  };
}

async function laneLocalComputeRecords(compute) {
  if (compute === null || compute === undefined) return {};
  assertAudit(compute && typeof compute === 'object' && !Array.isArray(compute),
    'compute must be a reset/cull phase record.');
  assertAudit(
    JSON.stringify(Object.keys(compute).sort()) === JSON.stringify(['cull', 'reset']),
    'compute must contain exactly reset and cull phases.',
  );
  const records = {};
  for (const phaseName of ['reset', 'cull']) {
    records[phaseName] = await auditComputePhase(phaseName, compute[phaseName]);
  }
  return records;
}

function legacyLaneRecord(render) {
  if (render === null) return null;
  return {
    raw: {
      vertex: {
        byteLength: render.raw.vertex.byteLength,
        sha256: render.raw.vertex.sha256,
      },
      fragment: {
        byteLength: render.raw.fragment.byteLength,
        sha256: render.raw.fragment.sha256,
      },
    },
    normalizedVertex: {
      byteLength: render.normalizedVertex.byteLength,
      sha256: render.normalizedVertex.sha256,
    },
    vertexInputs: render.vertexInputs,
    storageBindings: render.storageBindings,
    semanticMappings: render.semanticMappings,
    occurrenceCounts: render.occurrenceCounts,
  };
}

/**
 * Audits one independently constructed portable or feature lane. The returned
 * raw and narrowly normalized sources plus their digests are self-contained so
 * an offline verifier can compare vertex, fragment, and compute semantics from
 * different fresh browser/device sessions without constructing the other lane.
 * `compute`, when supplied, must contain exactly `reset` and `cull`; each phase
 * accepts either `shader` or `computeShader` plus its runtime `bindings`.
 */
export async function createFirstInstanceLaneShaderEvidence({
  laneId,
  lane,
  compute = null,
} = {}) {
  const reasons = [];
  let render = null;
  let computeRecords = {};
  let addressMode = null;
  try {
    addressMode = laneAddressMode(laneId);
    render = await laneLocalRenderRecord(auditLane(laneId, lane));
  } catch (error) {
    reasons.push(error.message);
  }
  if (render !== null) {
    try {
      computeRecords = await laneLocalComputeRecords(compute);
    } catch (error) {
      reasons.push(error.message);
    }
  }
  return {
    schemaVersion: SCHEMA_VERSION,
    kind: 'indirect-first-instance-lane-shader-evidence',
    pass: reasons.length === 0,
    reasons,
    laneId: addressMode === null ? null : laneId,
    addressMode,
    render,
    compute: computeRecords,
    normalizedSemantics: render === null ? null : {
      vertexSha256: render.normalizedVertex.sha256,
      fragmentSha256: render.normalizedFragment.sha256,
      computeSha256ByPhase: Object.fromEntries(
        Object.entries(computeRecords).map(([phaseName, record]) => [
          phaseName,
          record.normalized.sha256,
        ]),
      ),
    },
  };
}

/**
 * Creates fail-closed evidence for the fixed-slice bucketBase versus indirect-first-instance
 * shader contrast. Runtime metadata must describe only shader-used vertex inputs and the two
 * vertex-visible storage bindings. `resourceId` values must be stable primitive identity tokens;
 * common resources are required to use the same token in both lanes.
 */
export async function createFirstInstanceShaderEvidence({ portable, feature } = {}) {
  const reasons = [];
  const [portableEvidence, featureEvidence] = await Promise.all([
    createFirstInstanceLaneShaderEvidence({ laneId: LANE_IDS.PORTABLE, lane: portable }),
    createFirstInstanceLaneShaderEvidence({ laneId: LANE_IDS.FEATURE, lane: feature }),
  ]);
  reasons.push(...portableEvidence.reasons.map((reason) => `portable: ${reason}`));
  reasons.push(...featureEvidence.reasons.map((reason) => `feature: ${reason}`));
  const portableAudit = portableEvidence.render === null ? null : {
    vertexShader: portableEvidence.render.raw.vertex.source,
    fragmentShader: portableEvidence.render.raw.fragment.source,
    normalizedVertexShader: portableEvidence.render.normalizedVertex.source,
    runtimeVertexInputs: portableEvidence.render.vertexInputs,
    storageBindings: portableEvidence.render.storageBindings,
  };
  const featureAudit = featureEvidence.render === null ? null : {
    vertexShader: featureEvidence.render.raw.vertex.source,
    fragmentShader: featureEvidence.render.raw.fragment.source,
    normalizedVertexShader: featureEvidence.render.normalizedVertex.source,
    runtimeVertexInputs: featureEvidence.render.vertexInputs,
    storageBindings: featureEvidence.render.storageBindings,
  };

  let normalizedVertexEqual = false;
  let rawFragmentEqual = false;
  let rawVertexDifferent = false;
  if (portableAudit && featureAudit) {
    try {
      validateCrossLaneRuntimeIdentity(portableAudit, featureAudit);
    } catch (error) {
      reasons.push(`cross-lane runtime metadata: ${error.message}`);
    }
    normalizedVertexEqual = portableAudit.normalizedVertexShader
      === featureAudit.normalizedVertexShader;
    rawFragmentEqual = portableAudit.fragmentShader === featureAudit.fragmentShader;
    rawVertexDifferent = portableAudit.vertexShader !== featureAudit.vertexShader;
    if (!normalizedVertexEqual) {
      reasons.push('normalized vertex WGSL differs outside the approved addressing contrast.');
    }
    if (!rawFragmentEqual) reasons.push('raw fragment WGSL is not byte-identical.');
    if (!rawVertexDifferent) reasons.push('raw vertex WGSL unexpectedly has no lane contrast.');
  }

  const portableRecord = legacyLaneRecord(portableEvidence.render);
  const featureRecord = legacyLaneRecord(featureEvidence.render);
  return {
    schemaVersion: SCHEMA_VERSION,
    kind: 'indirect-first-instance-shader-evidence',
    pass: reasons.length === 0,
    reasons,
    portable: portableRecord,
    feature: featureRecord,
    comparison: {
      rawVertexDifferent,
      rawFragmentEqual,
      normalizedVertexEqual,
      normalizedVertexSha256Equal: portableRecord !== null
        && featureRecord !== null
        && portableRecord.normalizedVertex.sha256 === featureRecord.normalizedVertex.sha256,
    },
  };
}

function phase0ImmediateContrast(vertexShader) {
  assertAudit(
    typeof vertexShader === 'string' && vertexShader.length > 0,
    'I.vertexShader must be a nonempty string.',
  );
  const requirementPattern = /requires\s+immediate_address_space\s*;/g;
  const declarationPattern = /var\s*<\s*immediate\s*>\s+threeImmediateDrawBase\s*:\s*u32\s*;/g;
  const injectionPattern = /requires\s+immediate_address_space\s*;\r?\n\r?\n\/\/ immediate data\r?\nvar\s*<\s*immediate\s*>\s+threeImmediateDrawBase\s*:\s*u32\s*;\r?\n/g;
  const parenthesizedAddressPattern = /\(\s*threeImmediateDrawBase\s*\+\s*instanceIndex\s*\)/g;
  const bareAddressPattern = /threeImmediateDrawBase\s*\+\s*instanceIndex/g;
  const requirementCount = countMatches(vertexShader, requirementPattern);
  const declarationCount = countMatches(vertexShader, declarationPattern);
  const totalImmediateDeclarationCount = countMatches(
    vertexShader,
    /var\s*<\s*immediate\s*>\s+[A-Za-z_]\w*\s*:/g,
  );
  const injectionCount = countMatches(vertexShader, injectionPattern);
  const parenthesizedAddressCount = countMatches(vertexShader, parenthesizedAddressPattern);
  const bareAddressCount = countMatches(vertexShader, bareAddressPattern);
  const variableTokenCount = countIdentifier(vertexShader, IMMEDIATE_VARIABLE);
  assertAudit(requirementCount === 1,
    'I WGSL must require immediate_address_space exactly once.');
  assertAudit(declarationCount === 1,
    'I WGSL must declare threeImmediateDrawBase exactly once.');
  assertAudit(totalImmediateDeclarationCount === 1,
    'I WGSL must contain exactly one total immediate variable declaration.');
  assertAudit(injectionCount === 1,
    'I WGSL must contain the exact overlay immediate declaration block once.');
  assertAudit(bareAddressCount === 1,
    'I WGSL must use threeImmediateDrawBase + instanceIndex exactly once.');
  assertAudit(variableTokenCount === 2,
    'I WGSL must contain only the declaration and address use of threeImmediateDrawBase.');

  let normalized = vertexShader.replace(injectionPattern, '');
  if (parenthesizedAddressCount === 1) {
    normalized = normalized.replace(parenthesizedAddressPattern, 'instanceIndex');
  } else {
    assertAudit(parenthesizedAddressCount === 0,
      'I WGSL contains multiple parenthesized immediate address expressions.');
    normalized = normalized.replace(bareAddressPattern, 'instanceIndex');
  }
  assertAudit(countIdentifier(normalized, IMMEDIATE_VARIABLE) === 0,
    'I normalization left an immediate variable token.');
  assertAudit(countIdentifier(normalized, IMMEDIATE_REQUIREMENT) === 0,
    'I normalization left an immediate feature token.');
  return {
    normalized,
    audit: {
      schemaVersion: SCHEMA_VERSION,
      kind: 'immediate-aif-phase0-immediate-wgsl-normalization',
      pass: true,
      requirementCount,
      declarationCount,
      totalImmediateDeclarationCount,
      injectionCount,
      addressExpressionCount: bareAddressCount,
      parenthesizedAddressCount,
      variableTokenCount,
      replacement: 'instanceIndex',
    },
  };
}

function assertNoImmediateContrast(laneId, vertexShader) {
  assertAudit(
    countIdentifier(vertexShader, IMMEDIATE_REQUIREMENT) === 0
      && countIdentifier(vertexShader, IMMEDIATE_VARIABLE) === 0
      && countMatches(vertexShader, /var\s*<\s*immediate\s*>/g) === 0,
    `${laneId} WGSL unexpectedly contains an immediate-address-space token.`,
  );
}

function phase0RenderLaneNormalizationRecord(
  laneId,
  originalLane,
  audit,
  immediateNormalization,
) {
  return {
    laneId,
    addressMode: laneId === 'A'
      ? 'bucket-base'
      : laneId === 'I'
        ? 'immediate-base'
        : 'indirect-first-instance',
    rawVertexShader: originalLane.vertexShader,
    rawFragmentShader: originalLane.fragmentShader,
    normalizedVertexShader: audit.normalizedVertexShader,
    vertexInputs: audit.runtimeVertexInputs,
    storageBindings: audit.storageBindings,
    semanticMappings: audit.semanticMappings,
    occurrenceCounts: audit.occurrenceCounts,
    immediateNormalization,
  };
}

/**
 * Synchronously performs the narrowly-scoped Phase 0 A/I/F render
 * normalization. A is first
 * audited as the existing portable bucket-base lane and F as the existing
 * first-instance lane. I may differ from F only by the overlay's exact
 * immediate declaration block and its single address expression. All
 * generated storage identifiers are then canonicalized by the established
 * first-instance semantic normalizer before byte equality is required.
 */
export function normalizeImmediateAifRenderShaderLanes(lanes = {}) {
  const reasons = [];
  const audits = {};
  let immediateNormalization = null;
  try {
    assertAudit(lanes && typeof lanes === 'object' && !Array.isArray(lanes),
      'Phase 0 render lanes must be an object.');
    for (const laneId of PHASE0_LANE_IDS) {
      assertAudit(lanes[laneId] && typeof lanes[laneId] === 'object',
        `Phase 0 render lane ${laneId} is missing.`);
    }
    assertNoImmediateContrast('A', lanes.A.vertexShader);
    assertNoImmediateContrast('F', lanes.F.vertexShader);
    audits.A = auditLane(LANE_IDS.PORTABLE, lanes.A);
    audits.F = auditLane(LANE_IDS.FEATURE, lanes.F);
    immediateNormalization = phase0ImmediateContrast(lanes.I.vertexShader);
    audits.I = auditLane(LANE_IDS.FEATURE, {
      ...lanes.I,
      vertexShader: immediateNormalization.normalized,
    });
    validateCrossLaneRuntimeIdentity(audits.A, audits.F);
    validateCrossLaneRuntimeIdentity(audits.A, audits.I);
  } catch (error) {
    reasons.push(error.message);
  }

  const records = {};
  if (PHASE0_LANE_IDS.every((laneId) => audits[laneId] !== undefined)) {
    for (const laneId of PHASE0_LANE_IDS) {
      records[laneId] = phase0RenderLaneNormalizationRecord(
        laneId,
        lanes[laneId],
        audits[laneId],
        laneId === 'I' ? immediateNormalization.audit : null,
      );
    }
    const normalizedVertexEqual = PHASE0_LANE_IDS.every(
      (laneId) => records[laneId].normalizedVertexShader
        === records.A.normalizedVertexShader,
    );
    const rawFragmentEqual = PHASE0_LANE_IDS.every(
      (laneId) => records[laneId].rawFragmentShader === records.A.rawFragmentShader,
    );
    const rawVertexPairwiseDifferent = new Set(
      PHASE0_LANE_IDS.map((laneId) => records[laneId].rawVertexShader),
    ).size === PHASE0_LANE_IDS.length;
    if (!normalizedVertexEqual) {
      reasons.push('A/I/F normalized vertex WGSL differs outside the approved address transports.');
    }
    if (!rawFragmentEqual) {
      reasons.push('A/I/F raw fragment WGSL is not byte-identical.');
    }
    if (!rawVertexPairwiseDifferent) {
      reasons.push('A/I/F raw vertex WGSL does not expose three distinct address transports.');
    }
    return {
      schemaVersion: SCHEMA_VERSION,
      kind: 'immediate-aif-phase0-render-shader-normalization',
      pass: reasons.length === 0,
      reasons,
      lanes: records,
      comparison: {
        normalizedVertexEqual,
        rawFragmentEqual,
        rawVertexPairwiseDifferent,
      },
    };
  }

  return {
    schemaVersion: SCHEMA_VERSION,
    kind: 'immediate-aif-phase0-render-shader-normalization',
    pass: false,
    reasons,
    lanes: records,
    comparison: null,
  };
}

/** Adds deterministic SHA-256 commitments to the synchronous Phase 0 audit. */
export async function createImmediateAifRenderShaderEvidence(lanes = {}) {
  const normalized = normalizeImmediateAifRenderShaderLanes(lanes);
  if (!normalized.pass) {
    return {
      ...normalized,
      commonVertexSha256: null,
      commonFragmentSha256: null,
    };
  }
  const records = {};
  for (const laneId of PHASE0_LANE_IDS) {
    const record = normalized.lanes[laneId];
    const [rawVertex, rawFragment, normalizedVertex] = await Promise.all([
      shaderDigest(record.rawVertexShader),
      shaderDigest(record.rawFragmentShader),
      shaderDigest(record.normalizedVertexShader),
    ]);
    records[laneId] = {
      ...record,
      rawVertexSha256: rawVertex.sha256,
      rawFragmentSha256: rawFragment.sha256,
      normalizedSha256: normalizedVertex.sha256,
      normalizedVertexSha256: normalizedVertex.sha256,
      byteLengths: {
        rawVertex: rawVertex.byteLength,
        rawFragment: rawFragment.byteLength,
        normalizedVertex: normalizedVertex.byteLength,
      },
    };
  }
  const normalizedVertexSha256Equal = PHASE0_LANE_IDS.every(
    (laneId) => records[laneId].normalizedSha256 === records.A.normalizedSha256,
  );
  const rawFragmentSha256Equal = PHASE0_LANE_IDS.every(
    (laneId) => records[laneId].rawFragmentSha256 === records.A.rawFragmentSha256,
  );
  return {
    ...normalized,
    pass: normalized.pass && normalizedVertexSha256Equal && rawFragmentSha256Equal,
    reasons: [
      ...normalized.reasons,
      ...(normalizedVertexSha256Equal
        ? []
        : ['A/I/F normalized vertex WGSL hashes differ.']),
      ...(rawFragmentSha256Equal
        ? []
        : ['A/I/F raw fragment WGSL hashes differ.']),
    ],
    lanes: records,
    commonVertexSha256: normalizedVertexSha256Equal
      ? records.A.normalizedSha256
      : null,
    commonFragmentSha256: rawFragmentSha256Equal
      ? records.A.rawFragmentSha256
      : null,
    comparison: {
      ...normalized.comparison,
      normalizedVertexSha256Equal,
      rawFragmentSha256Equal,
    },
  };
}

export const FIRST_INSTANCE_SHADER_EVIDENCE_SCHEMA_VERSION = SCHEMA_VERSION;
