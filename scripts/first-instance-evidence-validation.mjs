import { createHash } from 'node:crypto';
import {
  FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE,
  FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
  FIRST_INSTANCE_CROSSOVER_PATTERNS,
  FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
  firstInstanceCrossoverFrame,
} from '../src/benchmark/first-instance-crossover-schedule.js';
import {
  FIRST_INSTANCE_CROSSOVER_COMMAND_SEGMENT_ORDERS,
  FIRST_INSTANCE_CROSSOVER_LANES,
  FIRST_INSTANCE_CROSSOVER_MODE,
  FIRST_INSTANCE_CROSSOVER_ORIENTATION_OFFSETS,
  FIRST_INSTANCE_CROSSOVER_REPETITIONS,
} from '../src/benchmark/plan.js';
import { INDEXED_INDIRECT_STRIDE_BYTES } from '../src/culling/indexed-command-layout.js';
import { createFirstInstanceShaderEvidence } from '../src/validation/first-instance-shader-evidence.js';

const [PORTABLE, FEATURE] = FIRST_INSTANCE_CROSSOVER_LANES;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const FNV64_PATTERN = /^[0-9a-f]{16}$/;
const TIMED_FRAME_COUNT = FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES
  + FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES;
const ADDRESS_TARGET_WIDTH = 256;
const PARITY_WIDTH = 1280;
const PARITY_HEIGHT = 720;
const PARITY_PIXEL_COUNT = PARITY_WIDTH * PARITY_HEIGHT;

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isNonnegativeInteger(value) {
  return Number.isInteger(value) && value >= 0;
}

function isPositiveInteger(value) {
  return Number.isInteger(value) && value > 0;
}

function isIdentity(value) {
  return (typeof value === 'string' && value.length > 0) || isNonnegativeInteger(value);
}

function requireCondition(condition, message, reasons) {
  if (!condition) reasons.push(message);
}

function requireEqual(actual, expected, label, reasons) {
  if (!Object.is(actual, expected)) {
    reasons.push(`${label} is ${JSON.stringify(actual)}; expected ${JSON.stringify(expected)}`);
  }
}

function requireSha256(value, label, reasons) {
  if (!SHA256_PATTERN.test(value ?? '')) reasons.push(`${label} is not a lowercase SHA-256 digest`);
}

function requireFNV64(value, label, reasons) {
  if (!FNV64_PATTERN.test(value ?? '')) reasons.push(`${label} is not a lowercase FNV-1a-64 digest`);
}

function requireExactArray(actual, expected, label, reasons) {
  if (!Array.isArray(actual) || actual.length !== expected.length) {
    reasons.push(`${label} is not the exact expected array`);
    return false;
  }
  for (let index = 0; index < expected.length; index += 1) {
    requireEqual(actual[index], expected[index], `${label}[${index}]`, reasons);
  }
  return true;
}

function uniqueReasons(reasons) {
  return [...new Set(reasons)];
}

function sha256Text(value) {
  return createHash('sha256').update(value).digest('hex');
}

function sha256ZeroBytes(byteLength) {
  return createHash('sha256').update(Buffer.alloc(byteLength)).digest('hex');
}

function sha256Json(value) {
  return sha256Text(JSON.stringify(value));
}

function fnv1a64Text(value) {
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(value)) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, '0');
}

function stableLifecycle(lifecycle) {
  if (!isRecord(lifecycle)) return null;
  const clone = structuredClone(lifecycle);
  delete clone.activeLane;
  delete clone.laneSelectionSerial;
  return clone;
}

function semanticValidation(validation) {
  if (!isRecord(validation)) return validation;
  const clone = structuredClone(validation);
  if (isRecord(clone.lifecycle)) {
    delete clone.lifecycle.activeLane;
    delete clone.lifecycle.laneSelectionSerial;
  }
  return clone;
}

function semanticValidationSha256(validation) {
  return sha256Json(semanticValidation(validation));
}

export function firstInstanceValidationSemanticSha256(validation) {
  return semanticValidationSha256(validation);
}

export function firstInstanceRenderParityIdentity(parity) {
  return sha256Json({
    schemaVersion: parity?.schemaVersion,
    kind: parity?.kind,
    laneIds: parity?.laneIds,
    crossLaneExact: parity?.crossLaneExact,
    lanes: Object.fromEntries(FIRST_INSTANCE_CROSSOVER_LANES.map((lane) => [lane, {
      width: parity?.lanes?.[lane]?.width,
      height: parity?.lanes?.[lane]?.height,
      reversedDepthBuffer: parity?.lanes?.[lane]?.reversedDepthBuffer,
      material: parity?.lanes?.[lane]?.material,
      color: parity?.lanes?.[lane]?.color,
      depth: parity?.lanes?.[lane]?.depth,
      objectId: parity?.lanes?.[lane]?.objectId,
      objectIdValidation: parity?.lanes?.[lane]?.objectIdValidation,
    }])),
    logicalSnapshot: {
      visibleIdsSha256:
        parity?.snapshotValidation?.frozenPacking?.commitments?.visibleIdsSha256 ?? null,
      logicalPairSha256:
        parity?.snapshotValidation?.frozenPacking?.commitments?.logicalPairSha256 ?? null,
      membershipSha256:
        parity?.snapshotValidation?.membershipDigests?.actual?.sha256 ?? null,
    },
  });
}

function schedulePhase(frameCount, orientationOffset) {
  return Array.from({ length: frameCount }, (_, phaseFrameIndex) => {
    const frame = firstInstanceCrossoverFrame(phaseFrameIndex, orientationOffset);
    return {
      phaseFrameIndex,
      crossoverBlockIndex: frame.crossoverBlockIndex,
      withinBlockPosition: frame.withinBlockPosition,
      patternIndex: frame.patternIndex,
      pattern: frame.pattern,
      laneId: frame.laneId,
    };
  });
}

export function firstInstanceCrossoverScheduleSha256(orientationOffset) {
  return sha256Json({
    schemaVersion: 1,
    kind: 'indirect-first-instance-crossover-frame-schedule',
    blockSize: FIRST_INSTANCE_CROSSOVER_BLOCK_SIZE,
    warmupFrames: FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
    measuredFrames: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
    orientationOffset,
    patterns: FIRST_INSTANCE_CROSSOVER_PATTERNS,
    warmup: schedulePhase(FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES, orientationOffset),
    measured: schedulePhase(FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES, orientationOffset),
  });
}

function commandSegments(spec, reasons) {
  const order = spec?.laneCommandSegmentOrder;
  const exactOrder = Array.isArray(order)
    && order.length === FIRST_INSTANCE_CROSSOVER_LANES.length
    && new Set(order).size === FIRST_INSTANCE_CROSSOVER_LANES.length
    && FIRST_INSTANCE_CROSSOVER_LANES.every((lane) => order.includes(lane));
  requireCondition(exactOrder, 'spec laneCommandSegmentOrder is not an exact lane permutation', reasons);
  if (!exactOrder || !isPositiveInteger(spec?.bucketCount)) return null;
  const recordsPerLane = Math.max(2, spec.bucketCount);
  const byteLength = recordsPerLane * INDEXED_INDIRECT_STRIDE_BYTES;
  return Object.fromEntries(FIRST_INSTANCE_CROSSOVER_LANES.map((lane) => {
    const index = order.indexOf(lane);
    return [lane, {
      index,
      recordBase: index * recordsPerLane,
      byteBase: index * byteLength,
    }];
  }));
}

function validateSegment(actual, expected, label, reasons) {
  if (!isRecord(actual) || !expected) {
    reasons.push(`${label} command segment is missing`);
    return;
  }
  requireEqual(actual.index, expected.index, `${label}.index`, reasons);
  requireEqual(actual.recordBase, expected.recordBase, `${label}.recordBase`, reasons);
  requireEqual(actual.byteBase, expected.byteBase, `${label}.byteBase`, reasons);
}

function normalizeVisibility(value) {
  return value === 'vertex' || value === 1 ? 'vertex' : null;
}

function countMatches(value, expression) {
  return [...value.matchAll(expression)].length;
}

function identifierPattern(name) {
  return new RegExp(`\\b${name.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\$&')}\\b`, 'g');
}

function replaceIdentifier(source, before, after) {
  return source.replace(identifierPattern(before), after);
}

function challengeStorageDeclaration(vertexShader, reasons, lane) {
  const declaration = /((?:@(binding|group)\s*\(\s*\d+\s*\)\s*){2})var\s*<\s*storage\s*,\s*(read|read_write)\s*>\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)\s*;/g;
  const matches = [...vertexShader.matchAll(declaration)];
  requireEqual(matches.length, 1, `${lane} challenge storage declaration count`, reasons);
  if (matches.length !== 1) return null;
  const coordinates = {};
  for (const annotation of matches[0][1].matchAll(/@(binding|group)\s*\(\s*(\d+)\s*\)/g)) {
    coordinates[annotation[1]] = Number(annotation[2]);
  }
  requireCondition(Number.isInteger(coordinates.group), `${lane} challenge storage group is missing`, reasons);
  requireCondition(Number.isInteger(coordinates.binding), `${lane} challenge storage binding is missing`, reasons);
  requireEqual(matches[0][3], 'read', `${lane} challenge storage access`, reasons);
  const structName = matches[0][5];
  const structPattern = new RegExp(`struct\\s+${structName}\\s*\\{([\\s\\S]*?)\\}\\s*;`);
  const struct = structPattern.exec(vertexShader);
  requireCondition(struct !== null, `${lane} challenge storage struct is missing`, reasons);
  if (struct !== null) {
    requireCondition(
      /^\s*value\s*:\s*array\s*<\s*u32\s*>\s*,?\s*$/.test(struct[1]),
      `${lane} challenge storage is not exactly an array<u32> value`,
      reasons,
    );
  }
  return {
    group: coordinates.group,
    binding: coordinates.binding,
    variableName: matches[0][4],
    structName,
  };
}

function challengeAddressVariable(vertexShader, lane, reasons) {
  const expression = lane === PORTABLE
    ? /([A-Za-z_]\w*ChallengeAddress)\s*=\s*\(\s*bucketBase\s*\+\s*instanceIndex\s*\)\s*;/g
    : /([A-Za-z_]\w*ChallengeAddress)\s*=\s*instanceIndex\s*;/g;
  const matches = [...vertexShader.matchAll(expression)];
  requireEqual(matches.length, 1, `${lane} challenge address assignment count`, reasons);
  return matches.length === 1 ? matches[0][1] : null;
}

function normalizedChallengeVertex(vertexShader, lane, storage, addressVariable, reasons) {
  let normalized = vertexShader;
  normalized = replaceIdentifier(normalized, storage.structName, '__VISIBLE_IDS_STRUCT__');
  normalized = replaceIdentifier(normalized, storage.variableName, '__VISIBLE_IDS_BUFFER__');
  normalized = replaceIdentifier(normalized, addressVariable, '__ADDRESS_SSA__');
  normalized = normalized.replaceAll(`${lane}Challenge`, '__LANE_CHALLENGE__');
  if (lane === PORTABLE) {
    const before = normalized;
    normalized = normalized.replace(
      /@location\s*\(\s*0\s*\)\s*bucketBase\s*:\s*u32\s*,\s*/,
      '',
    );
    requireCondition(normalized !== before, 'portable challenge bucketBase entry input was not removable', reasons);
    const beforePositionRenumber = normalized;
    normalized = normalized.replace(
      /@location\s*\(\s*1\s*\)\s*position\s*:\s*vec3\s*<\s*f32\s*>/,
      '@location( 0 ) position : vec3<f32>',
    );
    requireCondition(
      normalized !== beforePositionRenumber,
      'portable challenge position input was not renumberable from location 1 to 0',
      reasons,
    );
    normalized = normalized.replace(
      /__ADDRESS_SSA__\s*=\s*\(\s*bucketBase\s*\+\s*instanceIndex\s*\)\s*;/,
      '__ADDRESS_SSA__ = __FIRST_INSTANCE_ADDRESS__;',
    );
  } else {
    normalized = normalized.replace(
      /__ADDRESS_SSA__\s*=\s*instanceIndex\s*;/,
      '__ADDRESS_SSA__ = __FIRST_INSTANCE_ADDRESS__;',
    );
  }
  return normalized
    .replace(/\s+/g, ' ')
    .replace(/\s+\)/g, ')')
    .replace(/\(\s+/g, '(')
    .trim();
}

function validateChallengeFragmentEncoding(fragmentShader, varyingName, lane, reasons) {
  if (!varyingName) return;
  const escapedVarying = varyingName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const assignment = new RegExp(
    `(?:let|var)?\\s*([A-Za-z_]\\w*)\\s*(?::\\s*u32)?\\s*=\\s*\\(?\\s*`
      + `(?:[A-Za-z_]\\w*\\.)?${escapedVarying}\\s*\\+\\s*1u\\s*\\)?\\s*;`,
    'g',
  );
  const assignments = [...fragmentShader.matchAll(assignment)];
  requireEqual(assignments.length, 1, `${lane} challenge RGB24 plus-one assignment count`, reasons);
  if (assignments.length !== 1) return;
  const encoded = assignments[0][1].replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const compact = fragmentShader.replace(/\s+/g, '');
  requireCondition(
    new RegExp(`${encoded}&255u`).test(compact),
    `${lane} challenge RGB24 red-byte extraction is missing`,
    reasons,
  );
  requireCondition(
    new RegExp(`\\(${encoded}>>8u\\)&255u|${encoded}>>8u&255u`).test(compact),
    `${lane} challenge RGB24 green-byte extraction is missing`,
    reasons,
  );
  requireCondition(
    new RegExp(`\\(${encoded}>>16u\\)&255u|${encoded}>>16u&255u`).test(compact),
    `${lane} challenge RGB24 blue-byte extraction is missing`,
    reasons,
  );
  const colorAssignments = [
    ...fragmentShader.matchAll(/output\s*\.\s*color\s*=\s*vec4\s*<\s*f32\s*>\s*\(([\s\S]*?)\)\s*;/g),
  ];
  requireEqual(
    colorAssignments.length,
    1,
    `${lane} challenge encoded fragment color assignment count`,
    reasons,
  );
  if (colorAssignments.length === 1) {
    const components = [];
    let start = 0;
    let depth = 0;
    const body = colorAssignments[0][1];
    for (let index = 0; index < body.length; index += 1) {
      if (body[index] === '(') depth += 1;
      if (body[index] === ')') depth -= 1;
      if (body[index] === ',' && depth === 0) {
        components.push(body.slice(start, index).trim());
        start = index + 1;
      }
    }
    components.push(body.slice(start).trim());
    requireEqual(components.length, 4, `${lane} challenge encoded fragment component count`, reasons);
    requireEqual(components[3]?.replace(/\s+/g, ''), '1.0', `${lane} challenge encoded fragment alpha`, reasons);
  }
  requireEqual(
    countMatches(fragmentShader, /return\s+output\s*;/g),
    1,
    `${lane} challenge encoded fragment output return count`,
    reasons,
  );
}

function validateChallengeRuntime(
  shader,
  lane,
  timedVisibleIdsBinding,
  storageDeclaration,
  reasons,
) {
  const inputs = shader?.vertexInputs;
  const expectedNames = lane === PORTABLE ? ['bucketBase', 'position'] : ['position'];
  requireCondition(Array.isArray(inputs), `${lane} challenge runtime vertexInputs are missing`, reasons);
  if (Array.isArray(inputs)) {
    requireExactArray(inputs.map((entry) => entry?.name), expectedNames, `${lane} challenge vertex input names`, reasons);
    for (let index = 0; index < inputs.length; index += 1) {
      const input = inputs[index];
      const name = expectedNames[index];
      requireEqual(input?.shaderLocation, index, `${lane} challenge ${name} shaderLocation`, reasons);
      requireEqual(input?.stepMode, 'vertex', `${lane} challenge ${name} stepMode`, reasons);
      requireEqual(input?.normalized, false, `${lane} challenge ${name} normalized`, reasons);
      requireEqual(input?.format, name === 'bucketBase' ? 'uint32' : 'float32x3', `${lane} challenge ${name} format`, reasons);
      requireEqual(input?.arrayType, name === 'bucketBase' ? 'Uint32Array' : 'Float32Array', `${lane} challenge ${name} arrayType`, reasons);
      requireEqual(input?.itemSize, name === 'bucketBase' ? 1 : 3, `${lane} challenge ${name} itemSize`, reasons);
      requireCondition(isPositiveInteger(input?.count), `${lane} challenge ${name} count is not positive`, reasons);
      requireCondition(isIdentity(input?.resourceId), `${lane} challenge ${name} resource identity is missing`, reasons);
    }
  }

  const bindings = shader?.storageBindings;
  requireCondition(Array.isArray(bindings), `${lane} challenge runtime storageBindings are missing`, reasons);
  if (!Array.isArray(bindings)) return;
  requireEqual(bindings.length, 1, `${lane} challenge runtime storage binding count`, reasons);
  const binding = bindings[0];
  requireEqual(binding?.semantic, 'visibleIds', `${lane} challenge storage semantic`, reasons);
  requireEqual(binding?.group, storageDeclaration?.group, `${lane} challenge storage group`, reasons);
  requireEqual(binding?.binding, storageDeclaration?.binding, `${lane} challenge storage binding`, reasons);
  requireEqual(binding?.access, 'read', `${lane} challenge storage access`, reasons);
  requireEqual(normalizeVisibility(binding?.visibility), 'vertex', `${lane} challenge storage visibility`, reasons);
  requireEqual(binding?.elementType?.replace(/\s+/g, ''), 'u32', `${lane} challenge storage element type`, reasons);
  requireEqual(binding?.count, timedVisibleIdsBinding?.count, `${lane} challenge storage count`, reasons);
  requireEqual(binding?.byteLength, timedVisibleIdsBinding?.byteLength, `${lane} challenge storage byteLength`, reasons);
  requireEqual(binding?.resourceId, timedVisibleIdsBinding?.resourceId, `${lane} challenge visibleIds resource identity`, reasons);
}

function validateChallengeShader(challenge, lane, timedVisibleIdsBinding, reasons) {
  const shader = challenge?.shader;
  if (!isRecord(shader)) {
    reasons.push(`${lane} challenge shader evidence is missing`);
    return null;
  }
  const vertexShader = shader.rawSources?.vertexShader;
  const fragmentShader = shader.rawSources?.fragmentShader;
  if (typeof vertexShader !== 'string' || vertexShader.length === 0) {
    reasons.push(`${lane} challenge raw vertex WGSL is missing`);
    return null;
  }
  if (typeof fragmentShader !== 'string' || fragmentShader.length === 0) {
    reasons.push(`${lane} challenge raw fragment WGSL is missing`);
    return null;
  }
  requireSha256(shader.vertexSha256, `${lane} challenge vertex sha256`, reasons);
  requireSha256(shader.fragmentSha256, `${lane} challenge fragment sha256`, reasons);
  requireEqual(shader.vertexSha256, sha256Text(vertexShader), `${lane} challenge raw vertex WGSL digest`, reasons);
  requireEqual(shader.fragmentSha256, sha256Text(fragmentShader), `${lane} challenge raw fragment WGSL digest`, reasons);
  requireEqual(countMatches(vertexShader, /@vertex\s+fn\s+main\s*\(/g), 1, `${lane} challenge @vertex main count`, reasons);
  requireEqual(countMatches(fragmentShader, /@fragment\s+fn\s+main\s*\(/g), 1, `${lane} challenge @fragment main count`, reasons);
  requireEqual(countMatches(vertexShader, /@builtin\s*\(\s*instance_index\s*\)\s*instanceIndex\s*:\s*u32/g), 1, `${lane} challenge instance_index input count`, reasons);
  requireEqual(
    countMatches(vertexShader, lane === PORTABLE
      ? /@location\s*\(\s*0\s*\)\s*bucketBase\s*:\s*u32/g
      : /@location\s*\(\s*0\s*\)\s*position\s*:\s*vec3\s*<\s*f32\s*>/g),
    1,
    `${lane} challenge pinned primary vertex input`,
    reasons,
  );
  if (lane === PORTABLE) {
    requireEqual(
      countMatches(vertexShader, /@location\s*\(\s*1\s*\)\s*position\s*:\s*vec3\s*<\s*f32\s*>/g),
      1,
      'portable challenge pinned position input',
      reasons,
    );
  }
  requireCondition(!/\batomic\b|read_write/.test(vertexShader), `${lane} challenge vertex WGSL contains atomics or read_write storage`, reasons);
  requireCondition(!/\batomic\b|var\s*<\s*storage/.test(fragmentShader), `${lane} challenge fragment WGSL accesses storage or atomics`, reasons);
  if (lane === PORTABLE) {
    requireEqual(countMatches(vertexShader, /\bbucketBase\b/g), 2, 'portable challenge bucketBase identifier count', reasons);
  } else {
    requireEqual(countMatches(vertexShader, /\bbucketBase\b/g), 0, 'feature challenge bucketBase identifier count', reasons);
  }
  const storage = challengeStorageDeclaration(vertexShader, reasons, lane);
  const addressVariable = challengeAddressVariable(vertexShader, lane, reasons);
  if (!storage || !addressVariable) return null;
  const escapedAddress = addressVariable.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const escapedStorage = storage.variableName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  requireEqual(
    countMatches(vertexShader, new RegExp(`${escapedStorage}\\s*\\.\\s*value\\s*\\[\\s*${escapedAddress}\\s*\\]`, 'g')),
    1,
    `${lane} challenge visibleIds address use count`,
    reasons,
  );
  requireCondition(
    new RegExp(`\\b${escapedAddress}\\b\\s*%\\s*${ADDRESS_TARGET_WIDTH}u?`).test(vertexShader),
    `${lane} challenge address SSA does not feed pixel x`,
    reasons,
  );
  requireCondition(
    new RegExp(`\\b${escapedAddress}\\b\\s*/\\s*${ADDRESS_TARGET_WIDTH}u?`).test(vertexShader),
    `${lane} challenge address SSA does not feed pixel y`,
    reasons,
  );
  const flatU32 = /@location\s*\(\s*\d+\s*\)\s*@interpolate\s*\(\s*flat\s*,\s*either\s*\)\s*([A-Za-z_]\w*)\s*:\s*u32/g;
  const vertexVaryings = [...vertexShader.matchAll(flatU32)].map((match) => match[1]);
  const fragmentVaryings = [...fragmentShader.matchAll(flatU32)].map((match) => match[1]);
  requireEqual(vertexVaryings.length, 1, `${lane} challenge flat u32 vertex varying count`, reasons);
  requireExactArray(fragmentVaryings, vertexVaryings, `${lane} challenge flat u32 fragment varying`, reasons);
  validateChallengeFragmentEncoding(fragmentShader, vertexVaryings[0], lane, reasons);
  validateChallengeRuntime(
    shader,
    lane,
    timedVisibleIdsBinding,
    storage,
    reasons,
  );
  return {
    vertexShader,
    fragmentShader,
    normalizedVertex: normalizedChallengeVertex(
      vertexShader,
      lane,
      storage,
      addressVariable,
      reasons,
    ),
  };
}

function validateTarget(target, objectCount, label, reasons) {
  const height = Math.ceil(objectCount / ADDRESS_TARGET_WIDTH);
  const pixelCount = ADDRESS_TARGET_WIDTH * height;
  if (!isRecord(target)) {
    reasons.push(`${label} target metadata is missing`);
    return { height, pixelCount };
  }
  const exact = {
    pass: true,
    width: ADDRESS_TARGET_WIDTH,
    height,
    pixelCount,
    configuredFormat: 'RGBAFormat/UnsignedByteType',
    backendFormat: 'rgba8unorm',
    readbackArrayType: 'Uint8Array',
    bytesPerPixel: 4,
    bytesPerRow: ADDRESS_TARGET_WIDTH * 4,
    rowAlignmentBytes: 256,
    origin: 'top-left',
    sampleLocation: 'integer-plus-half',
    samples: 0,
    depthBuffer: false,
    stencilBuffer: false,
    scissorTest: false,
    viewportExact: true,
    scissorExact: true,
    colorSpace: 'none',
    flipY: false,
    generateMipmaps: false,
  };
  for (const [field, expected] of Object.entries(exact)) {
    requireEqual(target[field], expected, `${label} target.${field}`, reasons);
  }
  requireEqual(target.bytesPerRow % target.rowAlignmentBytes, 0, `${label} row alignment remainder`, reasons);
  return { height, pixelCount };
}

function validateTargetShape(target, objectCount, label, reasons) {
  const height = Math.ceil(objectCount / ADDRESS_TARGET_WIDTH);
  const pixelCount = ADDRESS_TARGET_WIDTH * height;
  requireEqual(target?.width, ADDRESS_TARGET_WIDTH, `${label} target.width`, reasons);
  requireEqual(target?.height, height, `${label} target.height`, reasons);
  requireEqual(target?.pixelCount, pixelCount, `${label} target.pixelCount`, reasons);
  return { height, pixelCount };
}

function validateAddressChallenges(validation, spec, segments, timedVisibleIdsBinding, reasons) {
  const addressChallenges = validation?.addressChallenges;
  if (!isRecord(addressChallenges)) {
    reasons.push('addressChallenges evidence is missing');
    return;
  }
  requireEqual(addressChallenges.pass, true, 'addressChallenges.pass', reasons);
  requireEqual(addressChallenges.byteIdentical, true, 'addressChallenges.byteIdentical', reasons);
  const geometry = addressChallenges.geometry;
  const { height, pixelCount } = validateTargetShape(
    geometry?.target,
    spec.objectCount,
    'address challenge geometry',
    reasons,
  );
  requireEqual(geometry?.schemaVersion, 1, 'address challenge geometry schemaVersion', reasons);
  requireEqual(geometry?.kind, 'fragment-address-challenge-geometry-evidence', 'address challenge geometry kind', reasons);
  requireEqual(geometry?.pass, true, 'address challenge geometry pass', reasons);
  requireEqual(geometry?.topology, 'triangle-list', 'address challenge topology', reasons);
  requireEqual(geometry?.pixelLocalCoordinates, true, 'address challenge pixel-local coordinates', reasons);
  requireCondition(isPositiveInteger(geometry?.indexCount), 'address challenge indexCount is not positive', reasons);
  requireEqual(geometry?.indexCount % 3, 0, 'address challenge triangle alignment', reasons);
  requireEqual(geometry?.addressedTriangleCount, spec.bucketCount, 'address challenge addressed triangle count', reasons);
  requireEqual(geometry?.degenerateTriangleCount, geometry?.indexCount / 3 - spec.bucketCount, 'address challenge degenerate triangle count', reasons);
  requireEqual(geometry?.addressedTrianglesPerSubmittedInstance, 1, 'address challenge addressed triangles per instance', reasons);
  for (const field of ['positionMismatchCount', 'bucketBaseMismatchCount', 'indexMismatchCount']) {
    requireEqual(geometry?.[field], 0, `address challenge geometry ${field}`, reasons);
  }
  for (const field of ['attributesExact', 'sharedPayloadExact', 'indirectIdentityExact']) {
    requireEqual(geometry?.[field], true, `address challenge geometry ${field}`, reasons);
  }
  for (const lane of FIRST_INSTANCE_CROSSOVER_LANES) {
    requireEqual(geometry?.laneOffsetsExact?.[lane], true, `${lane} challenge geometry lane offset`, reasons);
  }

  const laneShaders = {};
  const expectedVisibleCount = validation?.membership?.expectedCount;
  for (const lane of FIRST_INSTANCE_CROSSOVER_LANES) {
    const challenge = addressChallenges.lanes?.[lane];
    const label = `${lane} address challenge`;
    if (!isRecord(challenge)) {
      reasons.push(`${label} is missing`);
      continue;
    }
    requireEqual(challenge.schemaVersion, 1, `${label} schemaVersion`, reasons);
    requireEqual(challenge.kind, 'render-target-all-address-challenge', `${label} kind`, reasons);
    requireEqual(challenge.pass, true, `${label} pass`, reasons);
    requireEqual(challenge.lane, lane, `${label} lane`, reasons);
    requireEqual(challenge.outputStage, 'fragment', `${label} outputStage`, reasons);
    requireEqual(challenge.addressTransport, 'vertex-address-to-rgba8-pixel', `${label} addressTransport`, reasons);
    requireEqual(challenge.encoding, 'rgb24-object-id-plus-one-transparent-zero-background', `${label} encoding`, reasons);
    validateTarget(challenge.target, spec.objectCount, label, reasons);
    validateSegment(challenge.commandSegment, segments?.[lane], label, reasons);
    requireEqual(challenge.pixelCount, pixelCount, `${label} pixelCount`, reasons);
    requireEqual(challenge.addressCount, spec.objectCount, `${label} addressCount`, reasons);
    requireEqual(challenge.byteLength, pixelCount * 4, `${label} byteLength`, reasons);
    requireSha256(challenge.sha256, `${label} sha256`, reasons);
    requireEqual(challenge.sha256, challenge.expectedSha256, `${label} exact output digest`, reasons);
    requireEqual(challenge.exactExpectedBytes, true, `${label} exactExpectedBytes`, reasons);
    requireEqual(challenge.activeAddressCount, expectedVisibleCount, `${label} activeAddressCount`, reasons);
    requireEqual(challenge.paddingAddressCount, spec.objectCount - expectedVisibleCount, `${label} paddingAddressCount`, reasons);
    requireEqual(challenge.targetPaddingPixelCount, pixelCount - spec.objectCount, `${label} targetPaddingPixelCount`, reasons);
    requireEqual(challenge.activeMismatchCount, 0, `${label} activeMismatchCount`, reasons);
    requireEqual(challenge.paddingMismatchCount, 0, `${label} paddingMismatchCount`, reasons);
    requireEqual(challenge.targetPaddingMismatchCount, 0, `${label} targetPaddingMismatchCount`, reasons);
    const reset = challenge.reset;
    requireEqual(reset?.pass, true, `${label} reset pass`, reasons);
    requireEqual(reset?.pixelCount, pixelCount, `${label} reset pixelCount`, reasons);
    requireEqual(reset?.addressCount, spec.objectCount, `${label} reset addressCount`, reasons);
    requireEqual(reset?.byteLength, pixelCount * 4, `${label} reset byteLength`, reasons);
    requireSha256(reset?.sha256, `${label} reset sha256`, reasons);
    requireEqual(reset?.sha256, reset?.expectedSha256, `${label} reset exact digest`, reasons);
    requireEqual(reset?.sha256, sha256ZeroBytes(pixelCount * 4), `${label} transparent all-zero reset digest`, reasons);
    const coverage = challenge.coverage;
    requireEqual(coverage?.allBucketsActive, true, `${label} allBucketsActive`, reasons);
    requireEqual(coverage?.nonzeroBucketBaseCount, spec.bucketCount - 1, `${label} nonzeroBucketBaseCount`, reasons);
    requireCondition(coverage?.activeRowCount > 1, `${label} activeRowCount does not exercise multiple rows`, reasons);
    requireCondition(coverage?.activeRowCount <= height, `${label} activeRowCount exceeds target height`, reasons);
    requireCondition(coverage?.nonzeroEncodedChannelPixelCounts?.red > 0, `${label} red encoding coverage is empty`, reasons);
    requireCondition(coverage?.nonzeroEncodedChannelPixelCounts?.green > 0, `${label} green encoding coverage is empty`, reasons);
    requireCondition(isNonnegativeInteger(coverage?.nonzeroEncodedChannelPixelCounts?.blue), `${label} blue encoding coverage is invalid`, reasons);
    requireEqual(coverage?.alphaEncodedPixelCount, expectedVisibleCount, `${label} alpha encoding coverage`, reasons);
    laneShaders[lane] = validateChallengeShader(challenge, lane, timedVisibleIdsBinding, reasons);
  }

  const portableChallenge = addressChallenges.lanes?.[PORTABLE];
  const featureChallenge = addressChallenges.lanes?.[FEATURE];
  requireEqual(portableChallenge?.sha256, featureChallenge?.sha256, 'address challenge cross-lane output digest', reasons);
  requireEqual(portableChallenge?.reset?.sha256, featureChallenge?.reset?.sha256, 'address challenge cross-lane reset digest', reasons);
  requireEqual(
    sha256Json(portableChallenge?.coverage),
    sha256Json(featureChallenge?.coverage),
    'address challenge cross-lane coverage metadata',
    reasons,
  );
  requireEqual(portableChallenge?.shader?.fragmentSha256, featureChallenge?.shader?.fragmentSha256, 'address challenge cross-lane fragment digest', reasons);
  requireEqual(laneShaders[PORTABLE]?.fragmentShader, laneShaders[FEATURE]?.fragmentShader, 'address challenge raw fragment WGSL byte identity', reasons);
  requireEqual(laneShaders[PORTABLE]?.normalizedVertex, laneShaders[FEATURE]?.normalizedVertex, 'address challenge normalized vertex WGSL identity', reasons);
  if (laneShaders[PORTABLE] && laneShaders[FEATURE]) {
    requireCondition(
      laneShaders[PORTABLE].vertexShader !== laneShaders[FEATURE].vertexShader,
      'address challenge raw vertex WGSL has no lane contrast',
      reasons,
    );
  }
  const portableInput = portableChallenge?.shader?.vertexInputs?.find((entry) => entry.name === 'position');
  const featureInput = featureChallenge?.shader?.vertexInputs?.find((entry) => entry.name === 'position');
  requireEqual(portableInput?.resourceId, featureInput?.resourceId, 'address challenge shared position resource', reasons);
}

async function validateTimedShaderEvidence(shaderEvidence, spec, reasons) {
  if (!isRecord(shaderEvidence)) {
    reasons.push('timed shaderEvidence is missing');
    return null;
  }
  requireEqual(shaderEvidence.schemaVersion, 1, 'timed shaderEvidence schemaVersion', reasons);
  requireEqual(shaderEvidence.kind, 'indirect-first-instance-shader-evidence', 'timed shaderEvidence kind', reasons);
  requireEqual(shaderEvidence.pass, true, 'timed shaderEvidence pass', reasons);
  requireExactArray(shaderEvidence.reasons, [], 'timed shaderEvidence reasons', reasons);
  requireEqual(shaderEvidence.captureApi, 'renderer.debug.getShaderAsync(scene, camera, mesh)', 'timed shader capture API', reasons);
  const inputs = {};
  for (const lane of FIRST_INSTANCE_CROSSOVER_LANES) {
    const raw = shaderEvidence.rawSources?.[lane];
    requireCondition(typeof raw?.vertexShader === 'string' && raw.vertexShader.length > 0, `${lane} timed raw vertex WGSL is missing`, reasons);
    requireCondition(typeof raw?.fragmentShader === 'string' && raw.fragmentShader.length > 0, `${lane} timed raw fragment WGSL is missing`, reasons);
    const storage = shaderEvidence[lane]?.storageBindings;
    inputs[lane] = {
      vertexShader: raw?.vertexShader,
      fragmentShader: raw?.fragmentShader,
      vertexInputs: shaderEvidence[lane]?.vertexInputs,
      storageBindings: isRecord(storage) ? Object.values(storage) : storage,
    };
    requireEqual(shaderEvidence[lane]?.raw?.vertex?.sha256, sha256Text(raw?.vertexShader ?? ''), `${lane} timed raw vertex digest`, reasons);
    requireEqual(shaderEvidence[lane]?.raw?.fragment?.sha256, sha256Text(raw?.fragmentShader ?? ''), `${lane} timed raw fragment digest`, reasons);
  }
  const rebuilt = await createFirstInstanceShaderEvidence(inputs);
  requireEqual(rebuilt.pass, true, 'recomputed timed shader evidence pass', reasons);
  for (const reason of rebuilt.reasons) reasons.push(`recomputed timed shader evidence: ${reason}`);
  for (const lane of FIRST_INSTANCE_CROSSOVER_LANES) {
    requireEqual(rebuilt[lane]?.raw?.vertex?.sha256, shaderEvidence[lane]?.raw?.vertex?.sha256, `${lane} rebuilt raw vertex digest`, reasons);
    requireEqual(rebuilt[lane]?.raw?.fragment?.sha256, shaderEvidence[lane]?.raw?.fragment?.sha256, `${lane} rebuilt raw fragment digest`, reasons);
    requireEqual(rebuilt[lane]?.normalizedVertex?.sha256, shaderEvidence[lane]?.normalizedVertex?.sha256, `${lane} rebuilt normalized vertex digest`, reasons);
  }
  for (const field of ['rawVertexDifferent', 'rawFragmentEqual', 'normalizedVertexEqual', 'normalizedVertexSha256Equal']) {
    requireEqual(shaderEvidence.comparison?.[field], true, `timed shader comparison.${field}`, reasons);
  }
  const visibleBindings = FIRST_INSTANCE_CROSSOVER_LANES.map(
    (lane) => shaderEvidence[lane]?.storageBindings?.visibleIds,
  );
  requireEqual(visibleBindings[0]?.resourceId, visibleBindings[1]?.resourceId, 'timed visibleIds shared resource', reasons);
  requireEqual(visibleBindings[0]?.count, spec.objectCount, 'timed visibleIds element count', reasons);
  requireEqual(visibleBindings[0]?.byteLength, spec.objectCount * 4, 'timed visibleIds byteLength', reasons);
  return visibleBindings[0] ?? null;
}

function validateMembership(validation, spec, reasons) {
  const membership = validation?.membership;
  requireEqual(membership?.pass, true, 'membership pass', reasons);
  requireEqual(
    membership?.expectedCount,
    Math.round(spec.objectCount * spec.visibilityFraction),
    'membership count versus target visibility fraction',
    reasons,
  );
  requireEqual(membership?.expectedCount, membership?.listedCount, 'membership expected/listed count', reasons);
  requireEqual(membership?.expectedCount, validation?.frozenPacking?.visibleIds?.elementCount
    - validation?.frozenPacking?.padding?.paddingCount, 'membership count versus frozen active addresses', reasons);
  for (const field of [
    'duplicateIds', 'outOfRangeIds', 'wrongBucketIds', 'listedHiddenIds',
    'missingVisibleIds', 'overflow', 'errors',
  ]) requireEqual(membership?.[field], 0, `membership ${field}`, reasons);
  const digests = validation?.membershipDigests;
  requireEqual(digests?.pass, true, 'membership digest pass', reasons);
  requireEqual(digests?.invalidExpectedIds, 0, 'membership digest invalidExpectedIds', reasons);
  requireEqual(digests?.truncatedActualIds, 0, 'membership digest truncatedActualIds', reasons);
  requireEqual(digests?.expected?.count, membership?.expectedCount, 'membership digest expected count', reasons);
  requireEqual(digests?.actual?.count, membership?.expectedCount, 'membership digest actual count', reasons);
  requireSha256(digests?.expected?.sha256, 'membership expected sha256', reasons);
  requireEqual(digests?.actual?.sha256, digests?.expected?.sha256, 'membership actual/expected digest', reasons);
  requireEqual(digests?.perBucket?.length, spec.bucketCount, 'membership per-bucket digest count', reasons);
  if (Array.isArray(digests?.perBucket)) {
    for (let bucket = 0; bucket < digests.perBucket.length; bucket += 1) {
      const record = digests.perBucket[bucket];
      requireEqual(record?.bucket, bucket, `membership bucket ${bucket} index`, reasons);
      requireEqual(record?.match, true, `membership bucket ${bucket} match`, reasons);
      requireCondition(record?.expected?.count > 0, `membership bucket ${bucket} is empty`, reasons);
      requireEqual(record?.actual?.count, record?.expected?.count, `membership bucket ${bucket} count`, reasons);
      requireEqual(record?.actual?.declaredCount, record?.expected?.count, `membership bucket ${bucket} declared count`, reasons);
      requireEqual(record?.actual?.sha256, record?.expected?.sha256, `membership bucket ${bucket} digest`, reasons);
    }
  }
}

function validatePacking(validation, spec, segments, reasons) {
  const packing = validation?.frozenPacking;
  if (!isRecord(packing)) {
    reasons.push('frozenPacking validation is missing');
    return;
  }
  const recordsPerLane = Math.max(2, spec.bucketCount);
  const commandSegmentByteLength = recordsPerLane * INDEXED_INDIRECT_STRIDE_BYTES;
  requireEqual(packing.schemaVersion, 1, 'frozenPacking schemaVersion', reasons);
  requireEqual(packing.kind, 'first-instance-crossover-exact-frozen-packing-validation', 'frozenPacking kind', reasons);
  requireEqual(packing.pass, true, 'frozenPacking pass', reasons);
  requireEqual(packing.metadata?.pass, true, 'frozenPacking metadata pass', reasons);
  requireExactArray(packing.metadata?.errors, [], 'frozenPacking metadata errors', reasons);
  requireExactArray(packing.metadata?.laneCommandSegmentOrder, spec.laneCommandSegmentOrder, 'frozenPacking command segment order', reasons);
  requireEqual(packing.metadata?.recordsPerLane, recordsPerLane, 'frozenPacking recordsPerLane', reasons);
  requireEqual(packing.metadata?.commandSegmentByteLength, commandSegmentByteLength, 'frozenPacking commandSegmentByteLength', reasons);
  requireEqual(packing.visibleIds?.pass, true, 'frozen visibleIds pass', reasons);
  requireEqual(packing.visibleIds?.exactExpectedPacking, true, 'frozen exact visibleIds packing', reasons);
  requireEqual(packing.visibleIds?.elementCount, spec.objectCount, 'frozen visibleIds element count', reasons);
  requireSha256(packing.visibleIds?.sha256, 'frozen visibleIds sha256', reasons);
  requireEqual(packing.padding?.pass, true, 'frozen padding pass', reasons);
  requireEqual(packing.padding?.sentinel, 0xffff_ffff, 'frozen padding sentinel', reasons);
  requireEqual(packing.padding?.paddingCount, spec.objectCount - validation?.membership?.expectedCount, 'frozen padding count', reasons);
  requireEqual(packing.padding?.paddingSentinelCount, packing.padding?.paddingCount, 'frozen padding sentinel count', reasons);
  requireEqual(packing.padding?.activeSentinelCount, 0, 'frozen active sentinel count', reasons);
  requireExactArray(packing.padding?.corruptPaddingAddresses, [], 'frozen corrupt padding addresses', reasons);
  requireEqual(packing.commands?.pass, true, 'frozen commands pass', reasons);
  requireEqual(packing.commands?.exactExpectedPhysicalPacking, true, 'frozen exact physical commands', reasons);
  requireEqual(packing.commands?.pairPass, true, 'frozen command pair pass', reasons);

  const commandRecords = {};
  for (const lane of FIRST_INSTANCE_CROSSOVER_LANES) {
    const command = packing.commands?.lanes?.[lane];
    commandRecords[lane] = command?.records;
    requireEqual(command?.pass, true, `${lane} commands pass`, reasons);
    requireExactArray(command?.errors, [], `${lane} command errors`, reasons);
    requireEqual(command?.commandCount, spec.bucketCount, `${lane} command count`, reasons);
    requireEqual(command?.totalInstanceCount, validation?.membership?.expectedCount, `${lane} total instance count`, reasons);
    requireEqual(command?.records?.length, spec.bucketCount, `${lane} command record count`, reasons);
  }

  const pairs = packing.commitments?.pairs;
  requireEqual(pairs?.length, spec.bucketCount, 'command commitment pair count', reasons);
  if (Array.isArray(pairs)
    && Array.isArray(commandRecords[PORTABLE])
    && Array.isArray(commandRecords[FEATURE])) {
    let firstIndexCursor = 0;
    let instanceTotal = 0;
    let positiveFeatureBaseCount = 0;
    for (let bucket = 0; bucket < spec.bucketCount; bucket += 1) {
      const portable = commandRecords[PORTABLE][bucket];
      const feature = commandRecords[FEATURE][bucket];
      const pair = pairs[bucket];
      requireEqual(portable?.bucket, bucket, `portable bucket ${bucket} record index`, reasons);
      requireEqual(feature?.bucket, bucket, `feature bucket ${bucket} record index`, reasons);
      requireEqual(pair?.bucket, bucket, `command pair bucket ${bucket} index`, reasons);
      for (const field of ['indexCount', 'instanceCount', 'firstIndex', 'baseVertex', 'firstInstance']) {
        requireEqual(portable?.actual?.[field], portable?.expected?.[field], `portable bucket ${bucket} ${field}`, reasons);
        requireEqual(feature?.actual?.[field], feature?.expected?.[field], `feature bucket ${bucket} ${field}`, reasons);
      }
      for (const field of ['indexCount', 'instanceCount', 'firstIndex', 'baseVertex']) {
        requireEqual(feature?.actual?.[field], portable?.actual?.[field], `bucket ${bucket} cross-lane ${field}`, reasons);
      }
      requireCondition(portable?.actual?.instanceCount > 0, `bucket ${bucket} has no visible instances`, reasons);
      requireEqual(portable?.actual?.firstIndex, firstIndexCursor, `bucket ${bucket} contiguous firstIndex`, reasons);
      firstIndexCursor += portable?.actual?.indexCount ?? 0;
      instanceTotal += portable?.actual?.instanceCount ?? 0;
      requireEqual(portable?.actual?.baseVertex, 0, `portable bucket ${bucket} baseVertex`, reasons);
      requireEqual(feature?.actual?.baseVertex, 0, `feature bucket ${bucket} baseVertex`, reasons);
      requireEqual(portable?.actual?.firstInstance, 0, `portable bucket ${bucket} fifth command word`, reasons);
      requireEqual(feature?.actual?.firstInstance, pair?.expectedFeatureFirstInstance, `feature bucket ${bucket} fifth command word`, reasons);
      requireEqual(pair?.portableFirstInstance, 0, `command pair ${bucket} portable firstInstance`, reasons);
      requireEqual(pair?.featureFirstInstance, pair?.expectedFeatureFirstInstance, `command pair ${bucket} feature firstInstance`, reasons);
      requireEqual(pair?.coreEqual, true, `command pair ${bucket} core equality`, reasons);
      if (pair?.featureFirstInstance > 0) positiveFeatureBaseCount += 1;
      requireSha256(pair?.sha256, `command pair ${bucket} sha256`, reasons);
      if (bucket === 0) requireEqual(pair?.featureFirstInstance, 0, 'first feature bucket base', reasons);
      if (bucket > 0) {
        requireCondition(
          pair?.featureFirstInstance > pairs[bucket - 1]?.featureFirstInstance,
          `feature bucket ${bucket} base is not strictly increasing`,
          reasons,
        );
      }
    }
    requireEqual(instanceTotal, validation?.membership?.expectedCount, 'non-vacuous command instance total', reasons);
    requireEqual(positiveFeatureBaseCount, spec.bucketCount - 1, 'nonzero feature firstInstance bucket count', reasons);
  }

  const commitments = packing.commitments;
  requireEqual(commitments?.schemaVersion, 1, 'command commitments schemaVersion', reasons);
  requireEqual(commitments?.hashAlgorithm, 'sha256', 'command commitments hashAlgorithm', reasons);
  requireEqual(commitments?.encoding, 'uint32-little-endian', 'command commitments encoding', reasons);
  requireEqual(commitments?.commandCoresEqual, true, 'command core commitment equality', reasons);
  for (const field of ['visibleIdsSha256', 'physicalCommandsSha256', 'logicalPairSha256', 'paddingSha256']) {
    requireSha256(commitments?.[field], `command commitments ${field}`, reasons);
  }
  requireEqual(commitments?.visibleIdsSha256, packing.visibleIds?.sha256, 'visibleIds commitment identity', reasons);
  for (const lane of FIRST_INSTANCE_CROSSOVER_LANES) {
    requireSha256(commitments?.lanes?.[lane]?.commandsSha256, `${lane} logical commands sha256`, reasons);
    requireSha256(commitments?.lanes?.[lane]?.coreSha256, `${lane} command core sha256`, reasons);
  }
  requireEqual(commitments?.lanes?.[PORTABLE]?.coreSha256, commitments?.lanes?.[FEATURE]?.coreSha256, 'logical command core digest equality', reasons);
  requireCondition(
    commitments?.lanes?.[PORTABLE]?.commandsSha256 !== commitments?.lanes?.[FEATURE]?.commandsSha256,
    'portable and feature logical command digests do not expose a fifth-word contrast',
    reasons,
  );

  for (const lane of FIRST_INSTANCE_CROSSOVER_LANES) {
    validateSegment(validation?.lifecycle?.commandSegments?.[lane], segments?.[lane], `${lane} lifecycle`, reasons);
  }
}

function validateGeometry(geometry, reasons) {
  requireEqual(geometry?.schemaVersion, 1, 'shared geometry schemaVersion', reasons);
  requireEqual(geometry?.kind, 'shared-first-instance-geometry-evidence', 'shared geometry kind', reasons);
  requireEqual(geometry?.pass, true, 'shared geometry pass', reasons);
  requireExactArray(geometry?.portableAttributeNames, ['bucketBase', 'normal', 'position', 'uv'], 'portable geometry attributes', reasons);
  requireExactArray(geometry?.featureAttributeNames, ['normal', 'position', 'uv'], 'feature geometry attributes', reasons);
  requireEqual(geometry?.noInstanceSteppedAttributes, true, 'geometry no instance-stepped attributes', reasons);
  requireEqual(geometry?.bucketBaseMismatchCount, 0, 'geometry bucketBase mismatch count', reasons);
  requireEqual(geometry?.bucketBase?.arrayType, 'Uint32Array', 'geometry bucketBase array type', reasons);
  requireEqual(geometry?.bucketBase?.itemSize, 1, 'geometry bucketBase item size', reasons);
  requireEqual(geometry?.bucketBase?.isInstancedBufferAttribute, false, 'geometry bucketBase instancing', reasons);
  requireEqual(geometry?.sharedIndex?.sameObject, true, 'geometry shared index identity', reasons);
  requireEqual(geometry?.sharedIndex?.portable?.id, geometry?.sharedIndex?.feature?.id, 'geometry shared index resource ID', reasons);
  requireEqual(geometry?.sharedIndex?.portable?.sha256, geometry?.sharedIndex?.feature?.sha256, 'geometry shared index digest', reasons);
  requireSha256(geometry?.sharedIndex?.portable?.sha256, 'geometry shared index sha256', reasons);
  requireExactArray(Object.keys(geometry?.commonAttributes ?? {}).sort(), ['normal', 'position', 'uv'], 'geometry common attribute names', reasons);
  for (const name of ['normal', 'position', 'uv']) {
    const record = geometry?.commonAttributes?.[name];
    requireEqual(record?.sameObject, true, `geometry ${name} shared object`, reasons);
    requireEqual(record?.portable?.id, record?.feature?.id, `geometry ${name} resource ID`, reasons);
    requireEqual(record?.portable?.version, record?.feature?.version, `geometry ${name} version`, reasons);
    requireEqual(record?.portable?.sha256, record?.feature?.sha256, `geometry ${name} digest`, reasons);
    requireSha256(record?.portable?.sha256, `geometry ${name} sha256`, reasons);
    requireEqual(record?.portable?.isInstancedBufferAttribute, false, `geometry ${name} portable instancing`, reasons);
    requireEqual(record?.feature?.isInstancedBufferAttribute, false, `geometry ${name} feature instancing`, reasons);
  }
}

function validateLifecycle(lifecycle, spec, segments, shaderEvidence, reasons) {
  if (!isRecord(lifecycle)) {
    reasons.push('first-instance lifecycle evidence is missing');
    return;
  }
  requireEqual(lifecycle.kind, 'first-instance-crossover-static-resource-lifecycle', 'lifecycle kind', reasons);
  requireEqual(lifecycle.bundlesPrimed, true, 'lifecycle bundlesPrimed', reasons);
  requireEqual(lifecycle.allBundlesStatic, true, 'lifecycle allBundlesStatic', reasons);
  requireEqual(lifecycle.bundleCount, 2, 'lifecycle bundleCount', reasons);
  requireEqual(lifecycle.meshCount, 2, 'lifecycle meshCount', reasons);
  requireEqual(lifecycle.activeRenderObjectCount, 1, 'lifecycle activeRenderObjectCount', reasons);
  requireCondition(FIRST_INSTANCE_CROSSOVER_LANES.includes(lifecycle.activeLane), 'lifecycle activeLane is unsupported', reasons);
  requireCondition(isNonnegativeInteger(lifecycle.laneSelectionSerial), 'lifecycle laneSelectionSerial is invalid', reasons);
  requireExactArray(lifecycle.laneCommandSegmentOrder, spec.laneCommandSegmentOrder, 'lifecycle command segment order', reasons);
  requireEqual(lifecycle.configuredComputeDispatches, 0, 'lifecycle configured compute dispatches', reasons);
  requireEqual(lifecycle.configuredComputeSubmissions, 0, 'lifecycle configured compute submissions', reasons);
  requireEqual(lifecycle.shaderEvidence?.pass, true, 'lifecycle shader evidence pass', reasons);
  requireEqual(lifecycle.shaderEvidence?.portableVertexSha256, shaderEvidence?.portable?.raw?.vertex?.sha256, 'lifecycle portable shader digest', reasons);
  requireEqual(lifecycle.shaderEvidence?.featureVertexSha256, shaderEvidence?.feature?.raw?.vertex?.sha256, 'lifecycle feature shader digest', reasons);
  requireEqual(lifecycle.shaderEvidence?.normalizedVertexSha256, shaderEvidence?.portable?.normalizedVertex?.sha256, 'lifecycle normalized shader digest', reasons);
  requireEqual(lifecycle.shaderEvidence?.fragmentSha256, shaderEvidence?.portable?.raw?.fragment?.sha256, 'lifecycle fragment shader digest', reasons);
  for (const lane of FIRST_INSTANCE_CROSSOVER_LANES) {
    validateSegment(lifecycle.commandSegments?.[lane], segments?.[lane], `${lane} lifecycle`, reasons);
    requireEqual(lifecycle.bundleStaticFlags?.[lane], true, `${lane} static bundle flag`, reasons);
    requireEqual(lifecycle.bundleRecordCounts?.[lane], 1, `${lane} bundle record count`, reasons);
    for (const field of ['bundleUuids', 'meshUuids', 'geometryUuids', 'materialUuids']) {
      requireCondition(isIdentity(lifecycle[field]?.[lane]), `${lane} lifecycle ${field} identity is missing`, reasons);
    }
    requireCondition(
      lifecycle.bundleVersions?.[lane] === null
        || isNonnegativeInteger(lifecycle.bundleVersions?.[lane]),
      `${lane} bundle version is invalid`,
      reasons,
    );
    requireCondition(isNonnegativeInteger(lifecycle.materialVersions?.[lane]), `${lane} material version is invalid`, reasons);
  }
  for (const field of ['bundleUuids', 'meshUuids', 'geometryUuids', 'materialUuids']) {
    requireCondition(lifecycle[field]?.[PORTABLE] !== lifecycle[field]?.[FEATURE], `lifecycle ${field} lanes are not distinct`, reasons);
  }
  requireCondition(isIdentity(lifecycle.rootUuid), 'lifecycle rootUuid is missing', reasons);
  requireCondition(
    lifecycle.rootVersion === null || isNonnegativeInteger(lifecycle.rootVersion),
    'lifecycle rootVersion is invalid',
    reasons,
  );
  for (const field of ['indexAttributeId', 'bucketBaseAttributeId', 'matrixAttributeId', 'visibleIdsAttributeId', 'indirectAttributeId']) {
    requireCondition(isIdentity(lifecycle[field]), `lifecycle ${field} is missing`, reasons);
  }
  for (const field of ['indexAttributeVersion', 'bucketBaseAttributeVersion', 'matrixAttributeVersion', 'visibleIdsAttributeVersion', 'indirectAttributeVersion']) {
    requireCondition(isNonnegativeInteger(lifecycle[field]), `lifecycle ${field} is invalid`, reasons);
  }
  requireExactArray(Object.keys(lifecycle.commonAttributeIds ?? {}).sort(), ['normal', 'position', 'uv'], 'lifecycle common attribute IDs', reasons);
  requireExactArray(Object.keys(lifecycle.commonAttributeVersions ?? {}).sort(), ['normal', 'position', 'uv'], 'lifecycle common attribute versions', reasons);
  for (const name of ['normal', 'position', 'uv']) {
    requireCondition(isIdentity(lifecycle.commonAttributeIds?.[name]), `lifecycle ${name} attribute ID is missing`, reasons);
    requireCondition(isNonnegativeInteger(lifecycle.commonAttributeVersions?.[name]), `lifecycle ${name} attribute version is invalid`, reasons);
  }
}

export async function validateFirstInstanceCrossoverValidation(validation, {
  spec,
  environment,
} = {}) {
  const reasons = [];
  if (!isRecord(spec)) reasons.push('first-instance trial spec is missing');
  requireEqual(spec?.modeId, FIRST_INSTANCE_CROSSOVER_MODE, 'spec modeId', reasons);
  requireCondition(isPositiveInteger(spec?.objectCount), 'spec objectCount is not positive', reasons);
  requireCondition(isPositiveInteger(spec?.bucketCount), 'spec bucketCount is not positive', reasons);
  requireCondition(isNonnegativeInteger(spec?.planIndex), 'spec planIndex is invalid', reasons);
  requireCondition(isNonnegativeInteger(spec?.repetitionIndex), 'spec repetitionIndex is invalid', reasons);
  requireCondition(
    spec?.repetitionIndex < FIRST_INSTANCE_CROSSOVER_REPETITIONS,
    'spec repetitionIndex exceeds the preregistered matrix',
    reasons,
  );
  requireCondition(typeof spec?.trialId === 'string' && spec.trialId.length > 0, 'spec trialId is missing', reasons);
  requireExactArray(spec?.modeOrder, [FIRST_INSTANCE_CROSSOVER_MODE], 'spec mode order', reasons);
  requireEqual(spec?.modeOrderPosition, 0, 'spec modeOrderPosition', reasons);
  requireCondition(spec?.visibilityFraction === 0.99 || spec?.visibilityFraction === 0.2, 'spec visibilityFraction is not preregistered', reasons);
  requireCondition(
    Array.isArray(spec?.visibilityOrder)
      && spec.visibilityOrder.length === 2
      && new Set(spec.visibilityOrder).size === 2
      && spec.visibilityOrder.includes(0.99)
      && spec.visibilityOrder.includes(0.2),
    'spec visibilityOrder is not the exact preregistered pair',
    reasons,
  );
  requireCondition(spec?.visibilityOrderPosition === 0 || spec?.visibilityOrderPosition === 1, 'spec visibilityOrderPosition is invalid', reasons);
  requireEqual(spec?.visibilityOrder?.[spec?.visibilityOrderPosition], spec?.visibilityFraction, 'spec visibility order position', reasons);
  const expectedVisibilityOrder = spec?.repetitionIndex % 2 === 0
    ? [0.99, 0.2]
    : [0.2, 0.99];
  requireExactArray(spec?.visibilityOrder, expectedVisibilityOrder, 'spec preregistered visibility order', reasons);
  requireEqual(
    spec?.planIndex,
    spec?.repetitionIndex * 2 + spec?.visibilityOrderPosition,
    'spec preregistered planIndex',
    reasons,
  );
  requireEqual(spec?.layout, 'baseline', 'spec layout', reasons);
  requireExactArray(spec?.layoutOrder, ['baseline'], 'spec layout order', reasons);
  requireEqual(spec?.layoutOrderPosition, 0, 'spec layoutOrderPosition', reasons);
  requireCondition(spec?.objectCount > ADDRESS_TARGET_WIDTH, 'spec objectCount does not exercise multiple address rows', reasons);
  requireCondition(spec?.bucketCount > 1, 'spec bucketCount does not exercise nonzero bucket bases', reasons);
  requireCondition(spec?.superblockOrientationOffset === 0 || spec?.superblockOrientationOffset === 1, 'spec superblockOrientationOffset is invalid', reasons);
  requireExactArray(
    spec?.laneCommandSegmentOrder,
    FIRST_INSTANCE_CROSSOVER_COMMAND_SEGMENT_ORDERS[spec?.repetitionIndex] ?? [],
    'spec preregistered command-segment order',
    reasons,
  );
  requireEqual(
    spec?.superblockOrientationOffset,
    FIRST_INSTANCE_CROSSOVER_ORIENTATION_OFFSETS[spec?.repetitionIndex],
    'spec preregistered orientation offset',
    reasons,
  );
  requireEqual(environment?.indirectFirstInstanceAvailable, true, 'environment indirectFirstInstanceAvailable', reasons);
  const segments = commandSegments(spec, reasons);
  if (!isRecord(validation)) {
    return uniqueReasons([...reasons, 'first-instance exact validation is missing']);
  }
  requireEqual(validation.schemaVersion, 1, 'first-instance validation schemaVersion', reasons);
  requireEqual(validation.kind, 'first-instance-crossover-exact-paired-snapshots', 'first-instance validation kind', reasons);
  requireEqual(validation.pass, true, 'first-instance validation pass', reasons);
  requireEqual(validation.expectedIdsMatchScenario, true, 'expected IDs match scenario', reasons);
  validateMembership(validation, spec, reasons);
  validatePacking(validation, spec, segments, reasons);
  validateGeometry(validation.geometry, reasons);
  const timedVisibleIdsBinding = await validateTimedShaderEvidence(
    validation.shaderEvidence,
    spec,
    reasons,
  );
  validateAddressChallenges(validation, spec, segments, timedVisibleIdsBinding, reasons);
  validateLifecycle(validation.lifecycle, spec, segments, validation.shaderEvidence, reasons);
  return uniqueReasons(reasons);
}

function validateParityLane(lane, label, reasons) {
  requireEqual(lane?.schemaVersion, 1, `${label} schemaVersion`, reasons);
  requireEqual(lane?.kind, 'fixed-camera-offscreen-exact-render-parity', `${label} kind`, reasons);
  requireEqual(lane?.pass, true, `${label} pass`, reasons);
  requireEqual(lane?.width, PARITY_WIDTH, `${label} width`, reasons);
  requireEqual(lane?.height, PARITY_HEIGHT, `${label} height`, reasons);
  requireEqual(lane?.captures, 2, `${label} captures`, reasons);
  requireEqual(lane?.reversedDepthBuffer, true, `${label} reversedDepthBuffer`, reasons);
  requireEqual(lane?.stability?.pass, true, `${label} stability pass`, reasons);
  const channelSchemas = {
    color: { format: 'rgba8unorm', arrayType: 'Uint8Array' },
    depth: { format: 'depth32float', arrayType: 'Float32Array' },
    objectId: { format: 'rgba8unorm-object-id-plus-one', arrayType: 'Uint8Array' },
  };
  for (const [channel, schema] of Object.entries(channelSchemas)) {
    requireEqual(lane?.[channel]?.format, schema.format, `${label} ${channel} format`, reasons);
    requireEqual(lane?.[channel]?.arrayType, schema.arrayType, `${label} ${channel} arrayType`, reasons);
    requireEqual(lane?.[channel]?.byteLength, PARITY_PIXEL_COUNT * 4, `${label} ${channel} byteLength`, reasons);
    requireSha256(lane?.[channel]?.sha256, `${label} ${channel} sha256`, reasons);
    requireEqual(lane?.[channel]?.sha256, lane?.stability?.firstCapture?.[channel]?.sha256, `${label} stable ${channel} digest`, reasons);
    requireEqual(
      lane?.stability?.first?.[`${channel}Sha256`],
      lane?.[channel]?.sha256,
      `${label} first ${channel} digest`,
      reasons,
    );
  }
  requireEqual(lane?.objectIdValidation?.pass, true, `${label} object-ID validation pass`, reasons);
  requireEqual(lane?.objectIdValidation?.encoding, 'rgb24-object-id-plus-one-zero-background', `${label} object-ID encoding`, reasons);
  requireCondition(lane?.objectIdValidation?.coveredPixels > 0, `${label} object-ID coverage is empty`, reasons);
  requireCondition(isNonnegativeInteger(lane?.objectIdValidation?.backgroundPixels), `${label} background pixel count is invalid`, reasons);
  requireEqual(lane?.objectIdValidation?.coveredPixels + lane?.objectIdValidation?.backgroundPixels, PARITY_PIXEL_COUNT, `${label} classified pixel count`, reasons);
  requireEqual(lane?.objectIdValidation?.outOfRangePixels, 0, `${label} out-of-range pixels`, reasons);
  requireEqual(lane?.objectIdValidation?.nonVisiblePixels, 0, `${label} non-visible pixels`, reasons);
}

export function validateFirstInstanceCrossoverRenderParity(parity, {
  spec,
  validation,
} = {}) {
  const reasons = [];
  if (!isRecord(parity)) return ['first-instance render-parity evidence is missing'];
  requireEqual(parity.schemaVersion, 1, 'first-instance render parity schemaVersion', reasons);
  requireEqual(parity.kind, 'first-instance-crossover-exact-render-parity', 'first-instance render parity kind', reasons);
  requireEqual(parity.pass, true, 'first-instance render parity pass', reasons);
  requireExactArray(parity.laneIds, FIRST_INSTANCE_CROSSOVER_LANES, 'first-instance render parity lanes', reasons);
  requireEqual(parity.crossLaneExact, true, 'first-instance render parity crossLaneExact', reasons);
  for (const lane of FIRST_INSTANCE_CROSSOVER_LANES) {
    validateParityLane(parity.lanes?.[lane], `${lane} render parity`, reasons);
  }
  const portable = parity.lanes?.[PORTABLE];
  const feature = parity.lanes?.[FEATURE];
  for (const field of ['material', 'color', 'depth', 'objectId', 'objectIdValidation']) {
    requireEqual(sha256Json(portable?.[field]), sha256Json(feature?.[field]), `render parity cross-lane ${field}`, reasons);
  }
  requireEqual(portable?.reversedDepthBuffer, feature?.reversedDepthBuffer, 'render parity cross-lane reversedDepthBuffer', reasons);
  requireEqual(semanticValidationSha256(parity.snapshotValidation), semanticValidationSha256(validation), 'render parity snapshot validation identity', reasons);
  requireEqual(parity.snapshotValidation?.pass, true, 'render parity snapshot validation pass', reasons);
  requireEqual(parity.snapshotValidation?.kind, 'first-instance-crossover-exact-paired-snapshots', 'render parity snapshot validation kind', reasons);
  requireEqual(spec?.modeId, FIRST_INSTANCE_CROSSOVER_MODE, 'render parity spec modeId', reasons);
  return uniqueReasons(reasons);
}

export function validateFirstInstanceCrossoverCompletionInvariant(invariant, {
  spec,
  validation,
} = {}) {
  const reasons = [];
  const segments = commandSegments(spec, reasons);
  if (!isRecord(invariant)) return uniqueReasons([...reasons, 'first-instance completion invariant is missing']);
  requireEqual(invariant.pass, true, 'completion invariant pass', reasons);
  requireEqual(invariant.kind, 'first-instance-crossover-static-resource-invariant', 'completion invariant kind', reasons);
  for (const [field, expected] of Object.entries({
    bundlesPrimed: true,
    allBundlesStatic: true,
    bundleCount: 2,
    meshCount: 2,
    activeRenderObjectCount: 1,
    shaderEvidencePass: true,
    configuredCommandOrderExact: true,
    commandSegmentsExact: true,
    lifecycleExact: true,
  })) requireEqual(invariant[field], expected, `completion invariant ${field}`, reasons);
  for (const lane of FIRST_INSTANCE_CROSSOVER_LANES) {
    requireEqual(invariant.bundleStaticFlags?.[lane], true, `completion ${lane} bundle static`, reasons);
    requireEqual(invariant.bundleRecordCounts?.[lane], 1, `completion ${lane} bundle record count`, reasons);
    validateSegment(invariant.observedCommandSegments?.[lane], segments?.[lane], `completion ${lane}`, reasons);
  }
  const orderText = spec?.laneCommandSegmentOrder?.join('|');
  requireEqual(invariant.plannedLaneCommandSegmentOrder, orderText, 'completion planned command order', reasons);
  requireEqual(invariant.observedLaneCommandSegmentOrder, orderText, 'completion observed command order', reasons);
  let parsedSegments = null;
  try {
    parsedSegments = JSON.parse(invariant.plannedCommandSegments);
  } catch {
    reasons.push('completion plannedCommandSegments is not valid JSON');
  }
  requireEqual(sha256Json(parsedSegments), sha256Json(segments), 'completion planned command segments', reasons);
  const representation = invariant.representation;
  requireEqual(representation?.kind, 'frozen-first-instance-addressing-crossover', 'completion representation kind', reasons);
  requireEqual(representation?.activeRenderObjectCount, 1, 'completion representation active objects', reasons);
  requireEqual(representation?.geometryIdentityCount, 2, 'completion representation geometry identities', reasons);
  requireEqual(representation?.materialIdentityCount, 2, 'completion representation material identities', reasons);
  requireEqual(representation?.commonIndexIdentityCount, 1, 'completion representation common index', reasons);
  requireEqual(representation?.commandRecordCount, Math.max(2, spec?.bucketCount) * 2, 'completion representation command records', reasons);
  requireEqual(representation?.visibleIdsCount, spec?.objectCount, 'completion representation visible IDs', reasons);

  requireEqual(sha256Json(invariant.staticLifecycleAtTimingStart), sha256Json(invariant.staticLifecycleAtTimingEnd), 'completion static lifecycle equality', reasons);
  requireEqual(sha256Json(invariant.staticLifecycleAtTimingStart), sha256Json(stableLifecycle(validation?.lifecycle)), 'completion lifecycle versus validation', reasons);
  requireFNV64(invariant.lifecycleCommitmentAtTimingStart, 'completion lifecycle start commitment', reasons);
  requireEqual(invariant.lifecycleCommitmentAtTimingStart, invariant.lifecycleCommitmentAtTimingEnd, 'completion lifecycle commitment stability', reasons);
  requireEqual(invariant.lifecycleCommitmentAtTimingStart, fnv1a64Text(JSON.stringify(invariant.staticLifecycleAtTimingStart)), 'completion lifecycle commitment recomputation', reasons);
  validateLifecycle(
    { ...invariant.staticLifecycleAtTimingEnd, activeLane: PORTABLE, laneSelectionSerial: 0 },
    spec,
    segments,
    validation?.shaderEvidence,
    reasons,
  );

  const serials = [
    ['selectorWriteSerial', 'selectorWritesDuringTiming', TIMED_FRAME_COUNT],
    ['strategySelectionSerial', 'strategySelectionsDuringTiming', TIMED_FRAME_COUNT],
    ['renderCallSerial', 'renderCallsDuringTiming', TIMED_FRAME_COUNT],
    ['computeCallSerial', 'computeCallsDuringTiming', 0],
  ];
  for (const [prefix, countField, expectedCount] of serials) {
    requireCondition(isNonnegativeInteger(invariant[`${prefix}AtTimingStart`]), `${prefix} start is invalid`, reasons);
    requireCondition(isNonnegativeInteger(invariant[`${prefix}AtTimingEnd`]), `${prefix} end is invalid`, reasons);
    requireEqual(invariant[countField], expectedCount, `completion ${countField}`, reasons);
    requireEqual(invariant[`${prefix}AtTimingEnd`] - invariant[`${prefix}AtTimingStart`], expectedCount, `completion ${prefix} serial delta`, reasons);
  }
  requireEqual(invariant.expectedTimedFrameCount, TIMED_FRAME_COUNT, 'completion expectedTimedFrameCount', reasons);
  for (const field of [
    'renderTargetTextureUuid', 'renderTargetWidth', 'renderTargetHeight',
    'renderTargetSamples', 'renderTargetDepthBuffer', 'cameraViewFnv64',
    'cameraProjectionFnv64', 'totalPipelineCacheEntries',
    'computePipelineCacheEntries',
  ]) requireEqual(invariant[`${field}AtTimingStart`], invariant[`${field}AtTimingEnd`], `completion stable ${field}`, reasons);
  requireCondition(isIdentity(invariant.renderTargetTextureUuidAtTimingStart), 'completion render-target texture identity is missing', reasons);
  requireEqual(invariant.renderTargetWidthAtTimingStart, PARITY_WIDTH, 'completion render-target width', reasons);
  requireEqual(invariant.renderTargetHeightAtTimingStart, PARITY_HEIGHT, 'completion render-target height', reasons);
  requireEqual(invariant.renderTargetSamplesAtTimingStart, 0, 'completion render-target samples', reasons);
  requireEqual(invariant.renderTargetDepthBufferAtTimingStart, true, 'completion render-target depthBuffer', reasons);
  requireFNV64(invariant.cameraViewFnv64AtTimingStart, 'completion camera view digest', reasons);
  requireFNV64(invariant.cameraProjectionFnv64AtTimingStart, 'completion camera projection digest', reasons);
  requireCondition(isNonnegativeInteger(invariant.totalPipelineCacheEntriesAtTimingStart), 'completion total pipeline-cache count is invalid', reasons);
  requireCondition(isNonnegativeInteger(invariant.computePipelineCacheEntriesAtTimingStart), 'completion compute pipeline-cache count is invalid', reasons);
  return uniqueReasons(reasons);
}

function allRowsEqual(rows, field, expected, reasons) {
  for (let index = 0; index < rows.length; index += 1) {
    requireEqual(rows[index]?.[field], expected, `row ${index} ${field}`, reasons);
  }
}

export function validateFirstInstanceCrossoverRows({
  spec,
  rows,
  summary,
  validation,
  protocol,
} = {}) {
  const reasons = [];
  const segments = commandSegments(spec, reasons);
  const expectedScheduleSha256 = firstInstanceCrossoverScheduleSha256(
    spec?.superblockOrientationOffset,
  );
  requireEqual(protocol?.schemaVersion, 2, 'first-instance protocol schemaVersion', reasons);
  requireEqual(protocol?.warmupFrames, FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES, 'first-instance protocol warmupFrames', reasons);
  requireEqual(protocol?.measuredFrames, FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES, 'first-instance protocol measuredFrames', reasons);
  requireSha256(protocol?.plannedScheduleSha256, 'first-instance protocol planned schedule sha256', reasons);
  requireEqual(protocol?.plannedScheduleSha256, expectedScheduleSha256, 'first-instance protocol planned schedule sha256', reasons);
  requireCondition(Array.isArray(rows), 'first-instance retained rows are missing', reasons);
  if (!Array.isArray(rows)) return uniqueReasons(reasons);
  requireEqual(rows.length, FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES, 'first-instance retained row count', reasons);
  requireEqual(summary?.accepted, true, 'first-instance timing summary accepted', reasons);
  requireEqual(summary?.timestampAvailable, true, 'first-instance timestamp availability', reasons);
  requireEqual(summary?.rowCount, FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES, 'first-instance timing summary rowCount', reasons);
  requireEqual(summary?.missingRenderFrames, 0, 'first-instance missing render frames', reasons);
  requireEqual(summary?.invalidRenderTimestampUidCountFrames, 0, 'first-instance invalid render timestamp UID frames', reasons);
  requireEqual(summary?.expectedRenderTimestampUidCount, 1, 'first-instance expected render timestamp UID count', reasons);
  requireEqual(summary?.missingComputeFrames, 0, 'first-instance missing compute frames', reasons);
  requireEqual(summary?.classification, 'fine', 'first-instance timestamp classification', reasons);
  requireCondition(Number.isFinite(summary?.quantumNs) && summary.quantumNs > 0 && summary.quantumNs <= 10_000, 'first-instance timestamp quantum exceeds 10,000 ns or is invalid', reasons);
  const invariant = summary?.completionInvariant;
  const baseSelector = invariant?.selectorWriteSerialAtTimingStart;
  const baseSelection = invariant?.strategySelectionSerialAtTimingStart;
  const baseRender = invariant?.renderCallSerialAtTimingStart;
  const seenGpuFrames = new Set();
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const scheduled = firstInstanceCrossoverFrame(index, spec?.superblockOrientationOffset);
    const segment = segments?.[scheduled.laneId];
    const ordinal = FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES + index + 1;
    requireEqual(row?.frameIndex, index, `row ${index} frameIndex`, reasons);
    requireEqual(row?.phaseFrameIndex, index, `row ${index} phaseFrameIndex`, reasons);
    requireEqual(row?.crossoverBlockIndex, scheduled.crossoverBlockIndex, `row ${index} crossoverBlockIndex`, reasons);
    requireEqual(row?.withinBlockPosition, scheduled.withinBlockPosition, `row ${index} withinBlockPosition`, reasons);
    requireEqual(row?.crossoverPattern, scheduled.pattern, `row ${index} crossoverPattern`, reasons);
    requireEqual(row?.crossoverPatternIndex, scheduled.patternIndex, `row ${index} crossoverPatternIndex`, reasons);
    requireEqual(row?.laneId, scheduled.laneId, `row ${index} laneId`, reasons);
    requireEqual(row?.commandSegmentIndex, segment?.index, `row ${index} commandSegmentIndex`, reasons);
    requireEqual(row?.commandRecordBase, segment?.recordBase, `row ${index} commandRecordBase`, reasons);
    requireEqual(row?.commandByteBase, segment?.byteBase, `row ${index} commandByteBase`, reasons);
    requireEqual(row?.selectorWriteSerialAtTimingStart, baseSelector, `row ${index} selector start serial`, reasons);
    requireEqual(row?.strategySelectionSerialAtTimingStart, baseSelection, `row ${index} strategy selection start serial`, reasons);
    requireEqual(row?.renderCallSerialAtTimingStart, baseRender, `row ${index} render-call start serial`, reasons);
    requireEqual(row?.selectorWriteSerial, baseSelector + ordinal, `row ${index} selectorWriteSerial`, reasons);
    requireEqual(row?.strategySelectionSerial, baseSelection + ordinal, `row ${index} strategySelectionSerial`, reasons);
    requireEqual(row?.renderCallSerial, baseRender + ordinal, `row ${index} renderCallSerial`, reasons);
    requireCondition(isNonnegativeInteger(row?.gpuFrameId), `row ${index} gpuFrameId is invalid`, reasons);
    requireCondition(!seenGpuFrames.has(row?.gpuFrameId), `row ${index} gpuFrameId is duplicated`, reasons);
    seenGpuFrames.add(row?.gpuFrameId);
    if (index > 0) {
      requireEqual(row?.gpuFrameId, rows[index - 1]?.gpuFrameId + 1, `row ${index} gpuFrameId sequence`, reasons);
      requireEqual(row?.selectorWriteSerial, rows[index - 1]?.selectorWriteSerial + 1, `row ${index} selector serial sequence`, reasons);
      requireEqual(row?.strategySelectionSerial, rows[index - 1]?.strategySelectionSerial + 1, `row ${index} strategy selection sequence`, reasons);
      requireEqual(row?.renderCallSerial, rows[index - 1]?.renderCallSerial + 1, `row ${index} render-call sequence`, reasons);
    }
    requireEqual(row?.gpuRenderTimestampUidCount, 1, `row ${index} render timestamp UID count`, reasons);
    requireCondition(Number.isFinite(row?.gpuRenderMs) && row.gpuRenderMs >= 0, `row ${index} gpuRenderMs is invalid`, reasons);
    requireEqual(row?.gpuComputeMs, null, `row ${index} gpuComputeMs`, reasons);
    requireEqual(row?.gpuPassTotalMs, row?.gpuRenderMs, `row ${index} gpu pass total`, reasons);
    requireEqual(row?.cpuComputeSubmitMs, null, `row ${index} cpuComputeSubmitMs`, reasons);
    requireEqual(row?.usesCompute, false, `row ${index} usesCompute`, reasons);
    requireEqual(row?.configuredDrawCommands, spec?.bucketCount, `row ${index} configuredDrawCommands`, reasons);
    requireEqual(row?.configuredRenderObjects, 1, `row ${index} configuredRenderObjects`, reasons);
    requireEqual(row?.configuredComputeDispatches, 0, `row ${index} configuredComputeDispatches`, reasons);
    requireEqual(row?.configuredComputeSubmissions, 0, `row ${index} configuredComputeSubmissions`, reasons);
    requireEqual(row?.configuredSubmittedInstances, validation?.membership?.expectedCount, `row ${index} configuredSubmittedInstances`, reasons);
    requireEqual(row?.expectedVisibleCount, validation?.membership?.expectedCount, `row ${index} expectedVisibleCount`, reasons);
  }
  const expectedRunId = spec?.runId ?? rows[0]?.runId;
  requireCondition(
    typeof expectedRunId === 'string' && expectedRunId.length > 0,
    'first-instance row runId is missing',
    reasons,
  );
  const common = {
    schemaVersion: 2,
    runId: expectedRunId,
    trialId: spec?.trialId,
    planIndex: spec?.planIndex,
    repetitionIndex: spec?.repetitionIndex,
    modeOrderPosition: spec?.modeOrderPosition,
    visibilityOrderPosition: spec?.visibilityOrderPosition,
    layoutOrderPosition: spec?.layoutOrderPosition,
    plannedModeOrder: spec?.modeOrder?.join('|'),
    plannedVisibilityOrder: spec?.visibilityOrder?.join('|'),
    plannedLayoutOrder: spec?.layoutOrder?.join('|'),
    modeId: FIRST_INSTANCE_CROSSOVER_MODE,
    objectCount: spec?.objectCount,
    bucketCount: spec?.bucketCount,
    targetVisibilityFraction: spec?.visibilityFraction,
    scenarioLayout: spec?.layout,
    plannedLaneCommandSegmentOrder: spec?.laneCommandSegmentOrder?.join('|'),
    plannedCommandSegments: JSON.stringify(segments),
    superblockOrientationOffset: spec?.superblockOrientationOffset,
    plannedScheduleSha256: expectedScheduleSha256,
    protocolWarmupFrames: FIRST_INSTANCE_CROSSOVER_WARMUP_FRAMES,
    protocolMeasuredFrames: FIRST_INSTANCE_CROSSOVER_MEASURED_FRAMES,
    validationKind: 'first-instance-crossover-exact-paired-snapshots',
    validationPass: true,
    timestampAvailable: true,
    expectedRenderTimestampUidCount: 1,
    lifecycleCommitmentAtTimingStart: invariant?.lifecycleCommitmentAtTimingStart,
    rootUuidAtTimingStart: validation?.lifecycle?.rootUuid,
    rootVersionAtTimingStart: validation?.lifecycle?.rootVersion,
    bundleUuidsAtTimingStart: FIRST_INSTANCE_CROSSOVER_LANES.map(
      (lane) => validation?.lifecycle?.bundleUuids?.[lane],
    ).join('|'),
    bundleVersionsAtTimingStart: FIRST_INSTANCE_CROSSOVER_LANES.map(
      (lane) => validation?.lifecycle?.bundleVersions?.[lane],
    ).join('|'),
    meshUuidsAtTimingStart: FIRST_INSTANCE_CROSSOVER_LANES.map(
      (lane) => validation?.lifecycle?.meshUuids?.[lane],
    ).join('|'),
    geometryUuidsAtTimingStart: FIRST_INSTANCE_CROSSOVER_LANES.map(
      (lane) => validation?.lifecycle?.geometryUuids?.[lane],
    ).join('|'),
    materialUuidsAtTimingStart: FIRST_INSTANCE_CROSSOVER_LANES.map(
      (lane) => validation?.lifecycle?.materialUuids?.[lane],
    ).join('|'),
    commonAttributeIdsAtTimingStart:
      JSON.stringify(validation?.lifecycle?.commonAttributeIds),
    commonAttributeVersionsAtTimingStart:
      JSON.stringify(validation?.lifecycle?.commonAttributeVersions),
    indexAttributeIdAtTimingStart: validation?.lifecycle?.indexAttributeId,
    indexAttributeVersionAtTimingStart: validation?.lifecycle?.indexAttributeVersion,
    bucketBaseAttributeIdAtTimingStart: validation?.lifecycle?.bucketBaseAttributeId,
    bucketBaseAttributeVersionAtTimingStart:
      validation?.lifecycle?.bucketBaseAttributeVersion,
    shaderPortableVertexSha256AtTimingStart:
      validation?.lifecycle?.shaderEvidence?.portableVertexSha256,
    shaderFeatureVertexSha256AtTimingStart:
      validation?.lifecycle?.shaderEvidence?.featureVertexSha256,
    shaderNormalizedVertexSha256AtTimingStart:
      validation?.lifecycle?.shaderEvidence?.normalizedVertexSha256,
    shaderFragmentSha256AtTimingStart:
      validation?.lifecycle?.shaderEvidence?.fragmentSha256,
    matrixAttributeIdAtTimingStart: validation?.lifecycle?.matrixAttributeId,
    matrixAttributeVersionAtTimingStart: validation?.lifecycle?.matrixAttributeVersion,
    visibleIdsAttributeIdAtTimingStart: validation?.lifecycle?.visibleIdsAttributeId,
    visibleIdsAttributeVersionAtTimingStart:
      validation?.lifecycle?.visibleIdsAttributeVersion,
    indirectAttributeIdAtTimingStart: validation?.lifecycle?.indirectAttributeId,
    indirectAttributeVersionAtTimingStart: validation?.lifecycle?.indirectAttributeVersion,
    computeCallSerialAtTimingStart: invariant?.computeCallSerialAtTimingStart,
    renderTargetTextureUuidAtTimingStart: invariant?.renderTargetTextureUuidAtTimingStart,
    renderTargetWidthAtTimingStart: invariant?.renderTargetWidthAtTimingStart,
    renderTargetHeightAtTimingStart: invariant?.renderTargetHeightAtTimingStart,
    renderTargetSamplesAtTimingStart: invariant?.renderTargetSamplesAtTimingStart,
    renderTargetDepthBufferAtTimingStart: invariant?.renderTargetDepthBufferAtTimingStart,
    cameraViewFnv64AtTimingStart: invariant?.cameraViewFnv64AtTimingStart,
    cameraProjectionFnv64AtTimingStart: invariant?.cameraProjectionFnv64AtTimingStart,
    totalPipelineCacheEntriesAtTimingStart: invariant?.totalPipelineCacheEntriesAtTimingStart,
    computePipelineCacheEntriesAtTimingStart: invariant?.computePipelineCacheEntriesAtTimingStart,
  };
  for (const [field, expected] of Object.entries(common)) allRowsEqual(rows, field, expected, reasons);
  return uniqueReasons(reasons);
}

export async function validateFirstInstanceCrossoverTrialEvidence({
  spec,
  environment,
  validation,
  renderParity,
  rows,
  summary,
  protocol,
} = {}) {
  const reasons = [];
  reasons.push(...await validateFirstInstanceCrossoverValidation(validation, {
    spec,
    environment,
  }));
  reasons.push(...validateFirstInstanceCrossoverRenderParity(renderParity, {
    spec,
    validation,
  }));
  reasons.push(...validateFirstInstanceCrossoverCompletionInvariant(
    summary?.completionInvariant,
    { spec, validation },
  ));
  reasons.push(...validateFirstInstanceCrossoverRows({
    spec,
    rows,
    summary,
    validation,
    protocol,
  }));
  const rejectionReasons = uniqueReasons(reasons);
  return {
    pass: rejectionReasons.length === 0,
    rejectionReasons,
    semanticSha256: validation ? semanticValidationSha256(validation) : null,
  };
}

export const validateFirstInstanceTrialEvidence = validateFirstInstanceCrossoverTrialEvidence;

export const FIRST_INSTANCE_CROSSOVER_TIMED_FRAME_COUNT = TIMED_FRAME_COUNT;
