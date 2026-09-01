const WGSL_IDENTIFIER = '[A-Za-z_][A-Za-z0-9_]*';
const GENERATED_BUFFER_IDENTIFIER = /^NodeBuffer_[0-9]+$/;
const CANONICAL_VARIABLE = 'LiveLaneIndirectCommands';
const CANONICAL_STRUCT = `${CANONICAL_VARIABLE}Struct`;

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function tokenCount(source, identifier) {
  return source.match(new RegExp(`\\b${escapeRegExp(identifier)}\\b`, 'g'))?.length ?? 0;
}

function exactStructBody(source, identifier) {
  const declarations = [...source.matchAll(new RegExp(
    `\\bstruct\\s+${escapeRegExp(identifier)}\\s*\\{([^{}]*)\\}\\s*;`,
    'g',
  ))];
  if (declarations.length !== 1) {
    throw new Error(`${identifier} must have exactly one non-nested WGSL struct declaration.`);
  }
  return declarations[0][1].replace(/\s+/g, '').replace(/,$/, '');
}

function commentTokenCount(source, identifier) {
  const comments = source.match(/\/\/[^\r\n]*|\/\*[\s\S]*?\*\//g) ?? [];
  return comments.reduce((count, comment) => count + tokenCount(comment, identifier), 0);
}

function requireCoordinate(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new TypeError(`${label} must be a nonnegative integer.`);
  }
  return value;
}

/**
 * Canonicalize only the generated WGSL identifiers belonging to the lane-local
 * indirect-command storage binding. The raw source remains part of the
 * evidence; exact equality after this transform proves that no other source
 * token differs between the compute lanes.
 */
export function normalizeLiveIndirectCommandComputeShader(source, bindings) {
  if (typeof source !== 'string' || source.length === 0) {
    throw new TypeError('A non-empty raw compute WGSL source is required.');
  }
  if (!Array.isArray(bindings)) {
    throw new TypeError('Runtime compute bindings must be an array.');
  }
  const matches = bindings.filter((binding) => binding?.semantic === 'indirectCommands');
  if (matches.length !== 1 || matches[0]?.kind !== 'storage-buffer') {
    throw new Error('Exactly one indirectCommands storage binding is required.');
  }
  if (matches[0].access !== 'readWrite'
    || matches[0].attributeType !== 'IndirectStorageBufferAttribute'
    || matches[0].arrayType !== 'Uint32Array'
    || matches[0].itemSize !== 5
    || matches[0].count !== 32
    || matches[0].byteLength !== 32 * 5 * Uint32Array.BYTES_PER_ELEMENT) {
    throw new Error('The pinned r185 indirectCommands runtime binding shape is malformed.');
  }
  const group = requireCoordinate(matches[0].group, 'indirectCommands group');
  const binding = requireCoordinate(matches[0].binding, 'indirectCommands binding');
  const declaration = new RegExp(
    `@binding\\s*\\(\\s*${binding}\\s*\\)\\s*`
      + `@group\\s*\\(\\s*${group}\\s*\\)\\s*`
      + `var\\s*<\\s*storage\\s*,\\s*read_write\\s*>\\s*`
      + `(${WGSL_IDENTIFIER})\\s*:\\s*(${WGSL_IDENTIFIER})\\s*;`,
    'g',
  );
  const declarations = [...source.matchAll(declaration)];
  if (declarations.length !== 1) {
    throw new Error(
      'The indirectCommands coordinate must resolve exactly one WGSL storage declaration.',
    );
  }
  const variableIdentifier = declarations[0][1];
  const structIdentifier = declarations[0][2];
  if (!GENERATED_BUFFER_IDENTIFIER.test(variableIdentifier)
    || structIdentifier !== `${variableIdentifier}Struct`) {
    throw new Error('The pinned r185 indirectCommands WGSL identifiers are malformed.');
  }
  if (source.includes(CANONICAL_VARIABLE) || source.includes(CANONICAL_STRUCT)) {
    throw new Error('Raw compute WGSL already contains the reserved canonical identifier.');
  }
  const wrapperBody = exactStructBody(source, structIdentifier);
  const wrapperMatch = wrapperBody.match(new RegExp(
    `^value:array<(${WGSL_IDENTIFIER})>$`,
  ));
  if (!wrapperMatch) {
    throw new Error(
      `The pinned r185 indirectCommands wrapper struct is malformed: ${JSON.stringify(wrapperBody)}.`,
    );
  }
  const commandStructIdentifier = wrapperMatch[1];
  if (commandStructIdentifier !== 'FixedSliceIndexedDraw') {
    throw new Error('The pinned r185 indirectCommands wrapper element type is malformed.');
  }
  const commandStructBody = exactStructBody(source, commandStructIdentifier);
  if (commandStructBody
    !== 'indexCount:u32,instanceCount:atomic<u32>,firstIndex:u32,baseVertex:i32,firstInstance:u32') {
    throw new Error('The pinned r185 indexed-indirect command struct is malformed.');
  }
  const variableTokenCount = tokenCount(source, variableIdentifier);
  const structTokenCount = tokenCount(source, structIdentifier);
  const commentIdentifierTokenCount = commentTokenCount(source, variableIdentifier)
    + commentTokenCount(source, structIdentifier);
  if (variableTokenCount < 2 || structTokenCount !== 2) {
    throw new Error('The pinned r185 indirectCommands WGSL token counts are malformed.');
  }
  if (commentIdentifierTokenCount !== 0) {
    throw new Error('The generated indirectCommands identifier occurs inside a WGSL comment.');
  }
  const normalizedShader = source
    .replace(
      new RegExp(`\\b${escapeRegExp(structIdentifier)}\\b`, 'g'),
      CANONICAL_STRUCT,
    )
    .replace(
      new RegExp(`\\b${escapeRegExp(variableIdentifier)}\\b`, 'g'),
      CANONICAL_VARIABLE,
    );
  return {
    normalizedShader,
    audit: {
      schemaVersion: 1,
      kind: 'live-indirect-command-wgsl-identifier-normalization',
      pass: true,
      semantic: 'indirectCommands',
      group,
      binding,
      generatedVariableIdentifier: variableIdentifier,
      generatedStructIdentifier: structIdentifier,
      canonicalVariableIdentifier: CANONICAL_VARIABLE,
      canonicalStructIdentifier: CANONICAL_STRUCT,
      commandStructIdentifier,
      commandStructFields: [
        'indexCount:u32',
        'instanceCount:atomic<u32>',
        'firstIndex:u32',
        'baseVertex:i32',
        'firstInstance:u32',
      ],
      variableTokenCount,
      structTokenCount,
      commentIdentifierTokenCount,
    },
  };
}
