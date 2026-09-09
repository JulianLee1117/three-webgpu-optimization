import {
  IMMEDIATE_AIF_PHASE0_LANES,
  IMMEDIATE_AIF_PHASE0_LANE_ORDERS,
  IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS,
  IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION,
  IMMEDIATE_AIF_PHASE0_WORKLOAD,
  IMMEDIATE_AIF_UNUSED_ADDRESS,
} from './immediate-aif-plan.js';

export const IMMEDIATE_AIF_PHASE0_PAGE_KIND =
  'three-immediate-aif-phase0-page-result';
export const IMMEDIATE_AIF_PHASE0_PAGE_GLOBAL =
  '__THREE_IMMEDIATE_AIF_PHASE0__';
export const IMMEDIATE_AIF_PHASE0_OBSERVATION_NONCE_PATTERN = /^[0-9a-f]{32}$/;
export const IMMEDIATE_AIF_PHASE0_OBSERVATION_ENCODING =
  'ordered-utf8-fields-u32be-byte-length-prefix-v1';
export const IMMEDIATE_AIF_PHASE0_OBSERVATION_DOMAIN =
  'three-immediate-aif-phase0-observation-v1';
export const IMMEDIATE_AIF_PHASE0_CHAIN_DOMAIN =
  'three-immediate-aif-phase0-chain-v1';
export const IMMEDIATE_AIF_PHASE0_OBSERVATION_FIELD_NAMES = Object.freeze([
  'domain',
  'nonce',
  'target',
  'shaders',
  'pipelines',
  'commands',
  'scenarios',
  'addressWitnesses',
  'outputWitnesses',
  'resources',
  'chains',
  'executionTraces',
]);
export const IMMEDIATE_AIF_PHASE0_CHAIN_FIELD_NAMES = Object.freeze([
  'domain',
  'nonce',
  'chainId',
  'chainEvidence',
]);
export const IMMEDIATE_AIF_PHASE0_ERROR_SCOPE_FILTERS = Object.freeze([
  'validation',
  'internal',
  'out-of-memory',
]);
export const IMMEDIATE_AIF_PHASE0_SNAPSHOT_PLAN = Object.freeze([
  Object.freeze({ label: 'canonical-preflight', scheduleId: 'canonical' }),
  Object.freeze({ label: 'S1', scheduleId: 'S1' }),
  Object.freeze({ label: 'S2', scheduleId: 'S2' }),
  Object.freeze({ label: 'canonical-restored', scheduleId: 'canonical' }),
  Object.freeze({ label: 'canonical-postflight', scheduleId: 'canonical' }),
]);
export const IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET = Object.freeze({
  width: 256,
  height: 256,
  pixelCount: 65_536,
  byteLength: 65_536 * 4,
  format: 'rgba8unorm',
  encoding: 'rgb24-object-id-plus-one-transparent-zero-background',
});

const IMMEDIATE_AIF_PHASE0_DIAGNOSTIC_KINDS = Object.freeze([
  'address',
  'object-id',
]);

const POSITION_VERTEX_INPUT = Object.freeze({
  name: 'position',
  shaderLocation: 0,
  format: 'float32x3',
  stepMode: 'vertex',
  arrayType: 'Float32Array',
  itemSize: 3,
  normalized: false,
});

const DIAGNOSTIC_VERTEX_INPUT_LAYOUTS = Object.freeze({
  address: Object.freeze({
    A: Object.freeze([
      Object.freeze({
        name: 'bucketBase',
        shaderLocation: 0,
        format: 'uint32',
        stepMode: 'vertex',
        arrayType: 'Uint32Array',
        itemSize: 1,
        normalized: false,
      }),
      Object.freeze({ ...POSITION_VERTEX_INPUT, shaderLocation: 1 }),
    ]),
    I: Object.freeze([POSITION_VERTEX_INPUT]),
    F: Object.freeze([POSITION_VERTEX_INPUT]),
  }),
  'object-id': Object.freeze({
    A: Object.freeze([
      POSITION_VERTEX_INPUT,
      Object.freeze({
        name: 'bucketBase',
        shaderLocation: 1,
        format: 'uint32',
        stepMode: 'vertex',
        arrayType: 'Uint32Array',
        itemSize: 1,
        normalized: false,
      }),
    ]),
    I: Object.freeze([POSITION_VERTEX_INPUT]),
    F: Object.freeze([POSITION_VERTEX_INPUT]),
  }),
});

function requireObject(value, label) {
  if (value === null || typeof value !== 'object') {
    throw new TypeError(`${label} must be an object.`);
  }
  return value;
}

function exactArray(left, right) {
  return left?.length === right.length
    && right.every((value, index) => left[index] === value);
}

function exactValue(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function serializeImmediateAifErrorScopeValue(error) {
  return Object.freeze({
    name: String(error?.constructor?.name ?? error?.name ?? 'GPUError').slice(0, 256),
    message: String(error?.message ?? error).slice(0, 4_096),
  });
}

export async function drainImmediateAifErrorScopes({
  device = null,
  pushedFilters = [],
  trigger = 'failure-teardown',
} = {}) {
  if (!['normal-completion', 'failure-teardown'].includes(trigger)) {
    throw new RangeError('Phase 0 error-scope drainage trigger is invalid.');
  }
  if (!Array.isArray(pushedFilters)
    || pushedFilters.length > IMMEDIATE_AIF_PHASE0_ERROR_SCOPE_FILTERS.length
    || !exactArray(
      pushedFilters,
      IMMEDIATE_AIF_PHASE0_ERROR_SCOPE_FILTERS.slice(0, pushedFilters.length),
    )) {
    throw new Error('Phase 0 pushed error-scope filters are not an exact prefix.');
  }
  const expectedPopOrder = [...pushedFilters].reverse();
  if (expectedPopOrder.length === 0) {
    return Object.freeze({
      schemaVersion: 1,
      kind: 'immediate-aif-phase0-error-scope-drainage',
      pass: true,
      applicable: false,
      trigger,
      pushOrder: Object.freeze([]),
      expectedPopOrder: Object.freeze([]),
      popOrder: Object.freeze([]),
      records: Object.freeze([]),
      scopedErrors: null,
      scopedErrorsComplete: true,
      scopedErrorsEmpty: true,
    });
  }
  if (typeof device?.popErrorScope !== 'function') {
    throw new TypeError('Phase 0 error-scope drainage requires device.popErrorScope().');
  }

  const scopedErrors = {};
  const records = [];
  for (const filter of expectedPopOrder) {
    try {
      const error = await device.popErrorScope();
      const serializedError = error === null
        ? null
        : serializeImmediateAifErrorScopeValue(error);
      scopedErrors[filter] = serializedError;
      records.push(Object.freeze({
        filter,
        settled: true,
        returnedError: serializedError,
        popFailure: null,
      }));
    } catch (error) {
      const popFailure = serializeImmediateAifErrorScopeValue(error);
      scopedErrors[filter] = null;
      records.push(Object.freeze({
        filter,
        settled: false,
        returnedError: null,
        popFailure,
      }));
    }
  }
  const scopedErrorsComplete = records.every((record) => record.settled);
  const scopedErrorsEmpty = scopedErrorsComplete
    && records.every((record) => record.returnedError === null);
  return Object.freeze({
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-error-scope-drainage',
    pass: scopedErrorsEmpty,
    applicable: true,
    trigger,
    pushOrder: Object.freeze([...pushedFilters]),
    expectedPopOrder: Object.freeze(expectedPopOrder),
    popOrder: Object.freeze(records.map((record) => record.filter)),
    records: Object.freeze(records),
    scopedErrors: Object.freeze(scopedErrors),
    scopedErrorsComplete,
    scopedErrorsEmpty,
  });
}

export function immediateAifExpectedDiagnosticVertexInputs({
  diagnosticKind,
  lane,
  resources,
} = {}) {
  if (!IMMEDIATE_AIF_PHASE0_DIAGNOSTIC_KINDS.includes(diagnosticKind)) {
    throw new RangeError('Phase 0 diagnostic kind must be address or object-id.');
  }
  if (!IMMEDIATE_AIF_PHASE0_LANES.includes(lane)) {
    throw new RangeError('Phase 0 diagnostic lane must be A, I, or F.');
  }
  requireObject(resources, 'Phase 0 diagnostic vertex resources');
  const layout = DIAGNOSTIC_VERTEX_INPUT_LAYOUTS[diagnosticKind][lane];
  const expectedNames = layout.map((input) => input.name).sort();
  if (!exactArray(Object.keys(resources).sort(), expectedNames)) {
    throw new Error(`Phase 0 ${diagnosticKind}/${lane} diagnostic resources are not exact.`);
  }
  return Object.freeze(layout.map((input) => {
    const resource = requireObject(
      resources[input.name],
      `Phase 0 ${diagnosticKind}/${lane} ${input.name} resource`,
    );
    if (!Number.isInteger(resource.count)
      || resource.count <= 0
      || !Number.isInteger(resource.resourceId)
      || resource.resourceId < 0) {
      throw new Error(
        `Phase 0 ${diagnosticKind}/${lane} ${input.name} resource is malformed.`,
      );
    }
    return Object.freeze({
      ...input,
      count: resource.count,
      resourceId: resource.resourceId,
    });
  }));
}

export function normalizeImmediateAifCommonRenderStateProjection(rawProjection) {
  requireObject(rawProjection, 'Common render-state projection');
  const projection = structuredClone(rawProjection);
  const scene = requireObject(projection.scene, 'Common render-state scene');
  const laneRoots = requireObject(scene.laneRoots, 'Common render-state lane roots');
  const laneRootUuids = new Set();
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    const root = requireObject(laneRoots[lane], `Common render-state lane ${lane}`);
    if (typeof root.uuid !== 'string' || typeof root.visible !== 'boolean') {
      throw new Error(`Common render-state lane ${lane} is malformed.`);
    }
    laneRootUuids.add(root.uuid);
    root.visible = '__selected-lane-dependent__';
  }
  scene.activeLane = '__selected-lane__';
  if (!Array.isArray(scene.directChildren)) {
    throw new Error('Common render-state direct children are malformed.');
  }
  for (const child of scene.directChildren) {
    if (laneRootUuids.has(child?.uuid)) child.visible = '__selected-lane-dependent__';
  }
  return projection;
}

const FAILURE_SERIALIZATION_LIMITS = Object.freeze({
  depth: 6,
  entries: 256,
  keysPerObject: 64,
  arrayLength: 64,
  stringLength: 4_096,
  binaryPreviewLength: 32,
});

function failureSerializationMarker(reason, type = null, extra = null) {
  const marker = {
    __phase0Unserializable: true,
    reason,
  };
  if (type !== null) marker.type = type;
  if (extra !== null) Object.assign(marker, extra);
  return marker;
}

function bestEffortTypeName(value) {
  try {
    const name = value?.constructor?.name;
    if (typeof name === 'string' && name.length > 0) return name.slice(0, 256);
  } catch {
    // Host objects are allowed to reject constructor inspection.
  }
  return typeof value;
}

function bestEffortProperty(value, key) {
  try {
    return { pass: true, value: value?.[key] };
  } catch (error) {
    let message = 'property access threw';
    try {
      message = String(error?.message ?? error).slice(0, 512);
    } catch {
      // Keep the static fallback.
    }
    return {
      pass: false,
      marker: failureSerializationMarker('property-access-threw', null, { message }),
    };
  }
}

function sanitizeFailureValue(value, state, depth) {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') {
    return typeof value === 'string'
      ? value.slice(0, FAILURE_SERIALIZATION_LIMITS.stringLength)
      : value;
  }
  if (typeof value === 'number') {
    return Number.isFinite(value)
      ? value
      : failureSerializationMarker('non-finite-number', 'number', { value: String(value) });
  }
  if (typeof value === 'bigint') {
    return failureSerializationMarker('json-unsupported-scalar', 'bigint', {
      value: String(value).slice(0, FAILURE_SERIALIZATION_LIMITS.stringLength),
    });
  }
  if (typeof value === 'undefined' || typeof value === 'symbol' || typeof value === 'function') {
    return failureSerializationMarker('json-unsupported-value', typeof value);
  }
  if (depth >= FAILURE_SERIALIZATION_LIMITS.depth) {
    return failureSerializationMarker('maximum-depth-exceeded', bestEffortTypeName(value));
  }
  if (state.entries >= FAILURE_SERIALIZATION_LIMITS.entries) {
    return failureSerializationMarker('maximum-entry-budget-exceeded', bestEffortTypeName(value));
  }
  if (state.ancestors.has(value)) {
    return failureSerializationMarker('circular-reference', bestEffortTypeName(value));
  }

  state.entries += 1;
  state.ancestors.add(value);
  try {
    if (ArrayBuffer.isView(value)) {
      let byteLength = null;
      let preview = [];
      try {
        byteLength = value.byteLength;
        const bytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
        preview = Array.from(bytes.subarray(0, FAILURE_SERIALIZATION_LIMITS.binaryPreviewLength));
      } catch {
        return failureSerializationMarker('unreadable-binary-view', bestEffortTypeName(value));
      }
      return {
        __phase0BinarySummary: true,
        type: bestEffortTypeName(value),
        byteLength,
        preview,
        truncated: byteLength > preview.length,
      };
    }
    if (value instanceof ArrayBuffer) {
      try {
        const bytes = new Uint8Array(value);
        const preview = Array.from(
          bytes.subarray(0, FAILURE_SERIALIZATION_LIMITS.binaryPreviewLength),
        );
        return {
          __phase0BinarySummary: true,
          type: 'ArrayBuffer',
          byteLength: bytes.byteLength,
          preview,
          truncated: bytes.byteLength > preview.length,
        };
      } catch {
        return failureSerializationMarker('unreadable-array-buffer', 'ArrayBuffer');
      }
    }
    if (Array.isArray(value)) {
      const length = Math.min(value.length, FAILURE_SERIALIZATION_LIMITS.arrayLength);
      const result = [];
      for (let index = 0; index < length; index += 1) {
        const property = bestEffortProperty(value, index);
        result.push(property.pass
          ? sanitizeFailureValue(property.value, state, depth + 1)
          : property.marker);
      }
      if (value.length > length) {
        result.push(failureSerializationMarker('array-truncated', 'Array', {
          omittedEntryCount: value.length - length,
        }));
      }
      return result;
    }

    let prototype;
    try {
      prototype = Object.getPrototypeOf(value);
    } catch {
      return failureSerializationMarker('prototype-inspection-threw', bestEffortTypeName(value));
    }
    if (prototype !== Object.prototype && prototype !== null) {
      return failureSerializationMarker('non-plain-host-object', bestEffortTypeName(value));
    }

    let keys;
    try {
      keys = Object.keys(value).sort();
    } catch {
      return failureSerializationMarker('key-enumeration-threw', bestEffortTypeName(value));
    }
    const retainedKeys = keys.slice(0, FAILURE_SERIALIZATION_LIMITS.keysPerObject);
    const result = {};
    for (const key of retainedKeys) {
      const property = bestEffortProperty(value, key);
      result[key] = property.pass
        ? sanitizeFailureValue(property.value, state, depth + 1)
        : property.marker;
    }
    if (keys.length > retainedKeys.length) {
      result.__phase0Truncated = {
        reason: 'object-key-limit',
        omittedKeyCount: keys.length - retainedKeys.length,
      };
    }
    return result;
  } catch (error) {
    let message = 'failure detail sanitization threw';
    try {
      message = String(error?.message ?? error).slice(0, 512);
    } catch {
      // Keep the static fallback.
    }
    return failureSerializationMarker('sanitization-threw', bestEffortTypeName(value), { message });
  } finally {
    state.ancestors.delete(value);
  }
}

/**
 * Convert an arbitrary thrown value into a bounded JSON-safe failure artifact.
 * This deliberately summarizes host/GPU objects instead of cloning or probing them.
 */
export function serializeImmediateAifPhase0Error(error) {
  try {
    const nameProperty = bestEffortProperty(error, 'name');
    const messageProperty = bestEffortProperty(error, 'message');
    const stackProperty = bestEffortProperty(error, 'stack');
    const detailProperty = bestEffortProperty(error, 'detail');
    let fallbackMessage = 'Unknown Phase 0 failure';
    try {
      fallbackMessage = String(error).slice(0, FAILURE_SERIALIZATION_LIMITS.stringLength);
    } catch {
      // Keep the static fallback.
    }
    return {
      name: nameProperty.pass
        ? String(nameProperty.value ?? 'Error').slice(0, 256)
        : 'UninspectableError',
      message: messageProperty.pass
        ? String(messageProperty.value ?? fallbackMessage).slice(
          0,
          FAILURE_SERIALIZATION_LIMITS.stringLength,
        )
        : fallbackMessage,
      stack: stackProperty.pass && stackProperty.value != null
        ? String(stackProperty.value).slice(0, 8_192)
        : null,
      detail: detailProperty.pass && detailProperty.value != null
        ? sanitizeFailureValue(detailProperty.value, {
          ancestors: new WeakSet(),
          entries: 0,
        }, 0)
        : detailProperty.pass ? null : detailProperty.marker,
    };
  } catch {
    return {
      name: 'UnserializableError',
      message: 'Phase 0 failure could not be inspected safely.',
      stack: null,
      detail: failureSerializationMarker('top-level-error-inspection-threw'),
    };
  }
}

export function bytesOf(value) {
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (!ArrayBuffer.isView(value)) {
    throw new TypeError('Expected an ArrayBuffer or typed-array view.');
  }
  return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
}

export function immediateAifQueueWriteBufferSourceSelection(
  data,
  dataOffset = 0,
  size = undefined,
) {
  const sourceBytes = bytesOf(data);
  const elementByteSize = ArrayBuffer.isView(data) && !(data instanceof DataView)
    ? Number(data.BYTES_PER_ELEMENT)
    : 1;
  const elementCount = sourceBytes.byteLength / elementByteSize;
  const elementOffset = Number(dataOffset);
  const elementLength = size === undefined
    ? elementCount - elementOffset
    : Number(size);
  if (!Number.isInteger(elementByteSize)
    || elementByteSize <= 0
    || !Number.isInteger(elementCount)
    || !Number.isInteger(elementOffset)
    || !Number.isInteger(elementLength)
    || elementOffset < 0
    || elementLength < 0
    || elementOffset + elementLength > elementCount) {
    throw new RangeError('queue.writeBuffer source selection is out of range.');
  }
  const byteOffset = elementOffset * elementByteSize;
  const byteLength = elementLength * elementByteSize;
  return Object.freeze({
    elementByteSize,
    elementCount,
    elementOffset,
    elementLength,
    byteOffset,
    byteLength,
    bytes: sourceBytes.slice(byteOffset, byteOffset + byteLength),
  });
}

export function immediateAifBufferBindingLayoutSnapshot(buffer) {
  if (buffer === null || typeof buffer !== 'object') return null;
  return Object.freeze({
    hasOwnType: Object.hasOwn(buffer, 'type'),
    rawType: buffer.type ?? null,
    type: buffer.type ?? 'uniform',
    hasDynamicOffset: buffer.hasDynamicOffset ?? false,
    minBindingSize: Number(buffer.minBindingSize ?? 0),
  });
}

export function immediateAifExpectedBindingVisibility({
  resourceClass,
  bufferType = null,
  declaredStageVisibility,
}) {
  if (!Number.isInteger(declaredStageVisibility)
    || declaredStageVisibility <= 0
    || (declaredStageVisibility & ~0x7) !== 0) {
    throw new RangeError('Shader-declared visibility must be a nonempty WebGPU stage mask.');
  }
  // Pinned Three constructs NodeUniformsGroup layouts with visibility 7 even when a
  // particular generated module references that group from fewer shader stages.
  if (resourceClass === 'buffer' && bufferType === 'uniform') return 0x7;
  return declaredStageVisibility;
}

export function realizeAndFreezeImmediateAifTransform(object) {
  if (object === null || typeof object !== 'object'
    || typeof object.updateMatrix !== 'function'
    || typeof object.updateMatrixWorld !== 'function') {
    throw new TypeError('A matrix-capable Three object is required.');
  }
  object.updateMatrix();
  object.updateMatrixWorld(true);
  object.matrixAutoUpdate = false;
  object.matrixWorldAutoUpdate = false;
  return object;
}

export function realizeImmediateAifReversedDepthCamera(camera, renderer) {
  if (camera === null || typeof camera !== 'object'
    || typeof camera.updateProjectionMatrix !== 'function') {
    throw new TypeError('A projection-capable Three camera is required.');
  }
  if (renderer?.reversedDepthBuffer !== true
    || !Number.isInteger(renderer.coordinateSystem)) {
    throw new TypeError('The Phase 0 reversed-depth renderer state is required.');
  }
  // Mirror the pinned Renderer._updateCamera transition before the first
  // render so every preflight sees the projection that will actually be used.
  camera.coordinateSystem = renderer.coordinateSystem;
  camera._reversedDepth = true;
  camera.updateProjectionMatrix();
  if (camera.reversedDepth !== true) {
    throw new Error('The Phase 0 camera did not enter reversed-depth state.');
  }
  return camera;
}

export function hexOf(buffer) {
  return [...new Uint8Array(buffer)]
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

export async function sha256Bytes(value) {
  const subtle = globalThis.crypto?.subtle;
  if (subtle === undefined) throw new Error('Web Crypto SHA-256 is unavailable.');
  return hexOf(await subtle.digest('SHA-256', bytesOf(value)));
}

export async function sha256Text(value) {
  return sha256Bytes(new TextEncoder().encode(value));
}

function validateObservationNonce(nonce) {
  if (!IMMEDIATE_AIF_PHASE0_OBSERVATION_NONCE_PATTERN.test(nonce)) {
    throw new Error('observationNonce must be exactly 128 bits of lowercase hexadecimal.');
  }
  return nonce;
}

function jsonObservationField(value, name) {
  const encoded = JSON.stringify(value);
  if (typeof encoded !== 'string') {
    throw new TypeError(`Observation field ${name} is not JSON-serializable.`);
  }
  return encoded;
}

/**
 * Hash an ordered field vector as u32be(byteLength) || UTF-8 bytes per field.
 * Individual commitments intentionally hash the unprefixed field bytes.
 */
export async function commitImmediateAifObservationFields(fieldNames, fieldValues) {
  if (!Array.isArray(fieldNames)
    || !Array.isArray(fieldValues)
    || fieldNames.length === 0
    || fieldNames.length !== fieldValues.length
    || !fieldNames.every((name) => typeof name === 'string' && name.length > 0)
    || !fieldValues.every((value) => typeof value === 'string')) {
    throw new TypeError('Observation fields must be matching nonempty string arrays.');
  }
  const encoder = new TextEncoder();
  const encodedFields = fieldValues.map((value) => encoder.encode(value));
  const byteLength = encodedFields.reduce((total, bytes) => total + 4 + bytes.length, 0);
  const framed = new Uint8Array(byteLength);
  const view = new DataView(framed.buffer);
  let offset = 0;
  for (const bytes of encodedFields) {
    view.setUint32(offset, bytes.length, false);
    offset += 4;
    framed.set(bytes, offset);
    offset += bytes.length;
  }
  const fieldCommitments = [];
  for (let index = 0; index < encodedFields.length; index += 1) {
    fieldCommitments.push({
      name: fieldNames[index],
      byteLength: encodedFields[index].byteLength,
      sha256: await sha256Bytes(encodedFields[index]),
    });
  }
  return {
    encoding: IMMEDIATE_AIF_PHASE0_OBSERVATION_ENCODING,
    fieldNames: [...fieldNames],
    fieldCount: fieldNames.length,
    fieldCommitments,
    sha256: await sha256Bytes(framed),
  };
}

export async function createImmediateAifObservationChain({
  nonce,
  chainId,
  kind,
  chainEvidence,
} = {}) {
  validateObservationNonce(nonce);
  if (typeof chainId !== 'string' || chainId.length === 0
    || typeof kind !== 'string' || kind.length === 0) {
    throw new TypeError('Observation chainId and kind must be nonempty strings.');
  }
  const commitment = await commitImmediateAifObservationFields(
    IMMEDIATE_AIF_PHASE0_CHAIN_FIELD_NAMES,
    [
      IMMEDIATE_AIF_PHASE0_CHAIN_DOMAIN,
      nonce,
      chainId,
      jsonObservationField(chainEvidence, 'chainEvidence'),
    ],
  );
  return Object.freeze({
    schemaVersion: 1,
    kind,
    chainId,
    ...commitment,
  });
}

export async function createImmediateAifObservationChallenge({
  nonce,
  target,
  evidence,
  chains,
} = {}) {
  validateObservationNonce(nonce);
  requireObject(target, 'observation target');
  requireObject(evidence, 'observation evidence');
  if (!Array.isArray(chains) || !chains.every((chain) => (
    chain?.schemaVersion === 1
      && typeof chain.chainId === 'string'
      && typeof chain.sha256 === 'string'
  ))) {
    throw new TypeError('Observation chains must be a complete chain-record array.');
  }
  const executionTraces = {
    renderBundles: evidence.lifecycle?.renderBundleTraces,
    renderPasses: evidence.lifecycle?.renderPassTraces,
    computePasses: evidence.lifecycle?.computePassTraces,
    markers: evidence.lifecycle?.instrumentationMarkers,
  };
  const commitment = await commitImmediateAifObservationFields(
    IMMEDIATE_AIF_PHASE0_OBSERVATION_FIELD_NAMES,
    [
      IMMEDIATE_AIF_PHASE0_OBSERVATION_DOMAIN,
      nonce,
      jsonObservationField(target, 'target'),
      jsonObservationField(evidence.shaders, 'shaders'),
      jsonObservationField(evidence.pipelines, 'pipelines'),
      jsonObservationField(evidence.commands, 'commands'),
      jsonObservationField(evidence.scenarios, 'scenarios'),
      jsonObservationField(evidence.addressWitnesses, 'addressWitnesses'),
      jsonObservationField(evidence.outputWitnesses, 'outputWitnesses'),
      jsonObservationField(evidence.resources, 'resources'),
      jsonObservationField(chains, 'chains'),
      jsonObservationField(executionTraces, 'executionTraces'),
    ],
  );
  return Object.freeze({
    schemaVersion: 1,
    kind: 'immediate-aif-phase0-observation-challenge',
    nonce,
    nonceEntropyBits: 128,
    ...commitment,
    chains: [...chains],
    chainCount: chains.length,
  });
}

export function validateImmediateAifPhase0TargetConfiguration(configuration) {
  requireObject(configuration, 'Phase 0 target configuration');
  if (configuration.schemaVersion !== 1
    || typeof configuration.key !== 'string'
    || configuration.key.length === 0
    || typeof configuration.sourceFamily !== 'string'
    || configuration.sourceFamily.length === 0
    || typeof configuration.expectedRevision !== 'string'
    || configuration.expectedRevision.length === 0
    || typeof configuration.scope !== 'string'
    || !Array.isArray(configuration.implemented)
    || !Array.isArray(configuration.deferred)) {
    throw new Error('The runner-injected Phase 0 target configuration is malformed.');
  }
  if (configuration.rawExpectation !== undefined
    && (configuration.rawExpectation === null
      || typeof configuration.rawExpectation !== 'object')) {
    throw new Error('target.rawExpectation must be an object when supplied.');
  }
  validateObservationNonce(configuration.observationNonce);
  return configuration;
}

export function createImmediateAifPackedAddressBytes(addressOracle) {
  requireObject(addressOracle, 'addressOracle');
  const values = addressOracle.expectedOutputObjectIds;
  if (!(values instanceof Uint32Array)
    || values.length !== IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount
    || IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.pixelCount
      !== IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount) {
    throw new Error('Address oracle does not cover the frozen 256 by 256 target.');
  }
  const output = new Uint8Array(IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.byteLength);
  for (let address = 0; address < values.length; address += 1) {
    const objectId = values[address];
    if (objectId === IMMEDIATE_AIF_UNUSED_ADDRESS) continue;
    if (objectId >= IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount
      || objectId > 0x00ff_fffe) {
      throw new RangeError(`Address ${address} contains an unencodable object ID.`);
    }
    const encoded = objectId + 1;
    const byteOffset = address * 4;
    output[byteOffset] = encoded & 0xff;
    output[byteOffset + 1] = (encoded >>> 8) & 0xff;
    output[byteOffset + 2] = (encoded >>> 16) & 0xff;
    output[byteOffset + 3] = 0xff;
  }
  return output;
}

function decodeObjectId(bytes, address) {
  const offset = address * 4;
  const encoded = bytes[offset]
    | (bytes[offset + 1] << 8)
    | (bytes[offset + 2] << 16);
  return encoded === 0 && bytes[offset + 3] === 0
    ? IMMEDIATE_AIF_UNUSED_ADDRESS
    : (encoded - 1) >>> 0;
}

function pixelExact(actual, expected, address) {
  const offset = address * 4;
  return actual[offset] === expected[offset]
    && actual[offset + 1] === expected[offset + 1]
    && actual[offset + 2] === expected[offset + 2]
    && actual[offset + 3] === expected[offset + 3];
}

/**
 * Independently classifies every observed address pixel. The exact byte check
 * remains the primary oracle; the counters make failures locally diagnosable
 * without accepting a page-provided boolean as proof.
 */
export function compareImmediateAifPackedAddressBytes({
  actual,
  expected,
  addressOracle,
  scenario,
} = {}) {
  const actualBytes = bytesOf(actual);
  const expectedBytes = bytesOf(expected);
  if (actualBytes.byteLength !== IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.byteLength
    || expectedBytes.byteLength !== IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.byteLength) {
    throw new RangeError('Address output must contain exactly 256 by 256 RGBA8 pixels.');
  }
  requireObject(addressOracle, 'addressOracle');
  requireObject(scenario, 'scenario');
  const activeExpected = new Set(addressOracle.objectIds);
  const activeObserved = new Map();
  const mismatchCounts = {
    duplicate: 0,
    missing: 0,
    hidden: 0,
    wrongBucket: 0,
    outOfRange: 0,
    active: 0,
    padding: 0,
  };
  let activeCount = 0;
  let paddingCount = 0;

  for (let address = 0; address < IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET.pixelCount;
    address += 1) {
    const expectedObjectId = addressOracle.expectedOutputObjectIds[address];
    const active = expectedObjectId !== IMMEDIATE_AIF_UNUSED_ADDRESS;
    if (active) activeCount += 1;
    else paddingCount += 1;
    if (!pixelExact(actualBytes, expectedBytes, address)) {
      mismatchCounts[active ? 'active' : 'padding'] += 1;
    }
    const observedObjectId = decodeObjectId(actualBytes, address);
    if (observedObjectId === IMMEDIATE_AIF_UNUSED_ADDRESS) continue;
    if (observedObjectId >= scenario.objectCount) {
      mismatchCounts.outOfRange += 1;
      continue;
    }
    activeObserved.set(
      observedObjectId,
      (activeObserved.get(observedObjectId) ?? 0) + 1,
    );
    if (!activeExpected.has(observedObjectId)) mismatchCounts.hidden += 1;
    if (active) {
      const targetBucket = Math.floor(address / IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCapacity);
      const expectedSourceBucket = addressOracle.draws[targetBucket]?.sourceBucket;
      if (scenario.objectBuckets[observedObjectId] !== expectedSourceBucket) {
        mismatchCounts.wrongBucket += 1;
      }
    }
  }
  for (const objectId of activeExpected) {
    const count = activeObserved.get(objectId) ?? 0;
    if (count === 0) mismatchCounts.missing += 1;
    if (count > 1) mismatchCounts.duplicate += count - 1;
  }
  const exact = actualBytes.every((value, index) => value === expectedBytes[index]);
  return Object.freeze({
    pass: exact && Object.values(mismatchCounts).every((count) => count === 0),
    exactExpectedBytes: exact,
    activeCount,
    paddingCount,
    mismatchCounts: Object.freeze(mismatchCounts),
  });
}

export function validateImmediateAifOrderedProductionChallengeEvidence(challenge) {
  requireObject(challenge, 'Ordered production challenge');
  const expectedLanes = IMMEDIATE_AIF_PHASE0_LANE_ORDERS.flatMap((order) => order);
  const expectedImmutableSemantics = [
    'matrix',
    'bounds',
    'objectBucket',
    'bucketBase',
    'bucketCapacity',
    'cullOrder',
    'visibleIds',
    'overflow',
  ].map((semantic) => `shared.${semantic}`);
  expectedImmutableSemantics.push(
    'diagnostic.address.commonPosition',
    'diagnostic.address.featurePosition',
  );
  for (const lane of IMMEDIATE_AIF_PHASE0_LANES) {
    for (const semantic of ['normal', 'position', 'uv']) {
      expectedImmutableSemantics.push(`geometry.${lane}.${semantic}`);
    }
    if (lane === 'A') expectedImmutableSemantics.push('geometry.A.bucketBase');
    expectedImmutableSemantics.push(`geometry.${lane}.index`);
    expectedImmutableSemantics.push(`geometry.${lane}.indirectCommands`);
  }
  const expectedPhaseOrder = challenge.productionPhases?.flatMap(
    (phase) => [phase, phase],
  );
  const sequencesStrictlyIncrease = challenge.productionExecuteSequences?.every(
    (sequence, index) => Number.isInteger(sequence)
      && sequence > challenge.startSequence
      && sequence < challenge.endSequence
      && (index === 0 || sequence > challenge.productionExecuteSequences[index - 1]),
  ) === true;
  const stagingResourceIds = new Set(
    challenge.creationEvents?.map((record) => record.resourceId),
  );
  const encodedCommandBufferIds = challenge.commandEncoders?.map(
    (record) => record.commandBufferId,
  );
  const submittedCommandBufferIds = challenge.queueSubmissions?.flatMap(
    (record) => record.commandBufferIds,
  );
  const commandSubmissionRawExact = challenge.commandEncoders?.length === 72
    && challenge.queueSubmissions?.length === 72
    && challenge.commandEncoders.every((record) => (
      record.finishCallCount === 1
        && Number.isInteger(record.sequence)
        && Number.isInteger(record.finishSequence)
        && record.sequence < record.finishSequence
        && typeof record.commandBufferId === 'string'
        && challenge.productionPhases.includes(record.capturePhase)
        && ((record.renderPassEncoderIds?.length === 1
          && record.computePassEncoderIds?.length === 0
          && record.transferSequences?.length === 0)
          || (record.renderPassEncoderIds?.length === 0
            && record.computePassEncoderIds?.length === 0
            && record.transferSequences?.length === 1))
    ))
    && challenge.queueSubmissions.every((record) => (
      Number.isInteger(record.sequence)
        && record.commandBufferIds?.length === 1
        && challenge.productionPhases.includes(record.capturePhase)
    ))
    && new Set(encodedCommandBufferIds).size === 72
    && new Set(submittedCommandBufferIds).size === 72
    && exactArray(
      [...encodedCommandBufferIds].sort(),
      [...submittedCommandBufferIds].sort(),
    )
    && challenge.commandEncoders.every((record) => (
      challenge.queueSubmissions.some((submission) => (
        submission.commandBufferIds[0] === record.commandBufferId
          && submission.sequence > record.finishSequence
      ))
    ));
  const expectedReadbackBufferSize = 1_280 * 720 * 4;
  const expectedReadbackBufferUsage = 0x0001 | 0x0008;
  const productionTextureBySemantic = new Map(
    challenge.immutableTextureResources?.map((record) => [record.semantic, record]) ?? [],
  );
  const productionCommandTopologyRawExact =
    challenge.productionCommandTopology?.length === 18
    && Array.isArray(challenge.productionPhases)
    && Array.isArray(challenge.productionRenderPassTopology)
    && Array.isArray(challenge.productionExecuteSequences)
    && Array.isArray(challenge.commandEncoders)
    && Array.isArray(challenge.queueSubmissions)
    && Array.isArray(challenge.commandEncoderTransfers)
    && Array.isArray(challenge.creationEvents)
    && challenge.productionCommandTopology.every((topology, index) => {
      const phase = challenge.productionPhases?.[index];
      const renderTopology = challenge.productionRenderPassTopology?.[index];
      const encoderFor = (leg) => challenge.commandEncoders.filter(
        (record) => record.commandEncoderId === leg?.commandEncoderId,
      );
      const submissionFor = (leg) => challenge.queueSubmissions.filter(
        (record) => exactArray(record.commandBufferIds, [leg?.commandBufferId]),
      );
      const renderLegExact = (leg, name) => {
        const encoders = encoderFor(leg);
        const submissions = submissionFor(leg);
        const encoder = encoders[0];
        const submission = submissions[0];
        const expectedPassEncoderId = name === 'clear'
          ? renderTopology?.clearEncoderId
          : renderTopology?.renderEncoderId;
        const expectedCommandEncoderId = name === 'clear'
          ? renderTopology?.clearCommandEncoderId
          : renderTopology?.renderCommandEncoderId;
        const expectedBeginSequence = name === 'clear'
          ? renderTopology?.clearBeginSequence
          : renderTopology?.renderBeginSequence;
        const expectedEndSequence = name === 'clear'
          ? renderTopology?.clearEndSequence
          : renderTopology?.renderEndSequence;
        return leg?.pass === true
          && encoders.length === 1
          && submissions.length === 1
          && leg.commandEncoderId === expectedCommandEncoderId
          && leg.commandEncoderSequence === encoder.sequence
          && leg.renderPassEncoderId === expectedPassEncoderId
          && leg.renderPassBeginSequence === expectedBeginSequence
          && leg.renderPassEndSequence === expectedEndSequence
          && leg.commandBufferId === encoder.commandBufferId
          && leg.finishSequence === encoder.finishSequence
          && leg.submitSequence === submission.sequence
          && encoder.capturePhase === phase
          && exactArray(encoder.renderPassEncoderIds, [expectedPassEncoderId])
          && encoder.computePassEncoderIds?.length === 0
          && encoder.transferSequences?.length === 0
          && encoder.finishCallCount === 1
          && encoder.sequence < expectedBeginSequence
          && expectedBeginSequence < expectedEndSequence
          && expectedEndSequence < encoder.finishSequence
          && encoder.finishSequence < submission.sequence
          && (name === 'clear'
            ? leg.executeSequence === null
            : leg.executeSequence === challenge.productionExecuteSequences[index]
              && expectedBeginSequence < leg.executeSequence
              && leg.executeSequence < expectedEndSequence);
      };
      const readbackLegExact = (leg, channel) => {
        const expectedTexture = productionTextureBySemantic.get(`production.${channel}`);
        const encoders = encoderFor(leg);
        const submissions = submissionFor(leg);
        const transfers = challenge.commandEncoderTransfers.filter(
          (record) => record.sequence === leg?.transferSequence,
        );
        const stagingRecords = challenge.creationEvents.filter(
          (record) => record.resourceId === leg?.stagingBufferId,
        );
        const encoder = encoders[0];
        const submission = submissions[0];
        const transfer = transfers[0];
        const staging = stagingRecords[0];
        return leg?.pass === true
          && leg.channel === channel
          && leg.sourceSemantic === `production.${channel}`
          && expectedTexture !== undefined
          && leg.sourceTextureId === expectedTexture.gpuTextureId
          && leg.sourceMipLevel === 0
          && exactArray(
            [leg.sourceOrigin?.x, leg.sourceOrigin?.y, leg.sourceOrigin?.z],
            [0, 0, 0],
          )
          && leg.sourceAspect === 'all'
          && leg.stagingSize === expectedReadbackBufferSize
          && leg.stagingUsage === expectedReadbackBufferUsage
          && leg.destinationOffset === 0
          && leg.destinationBytesPerRow === 5_120
          && leg.destinationRowsPerImage === null
          && exactArray(
            [leg.copySize?.width, leg.copySize?.height, leg.copySize?.depthOrArrayLayers],
            [1_280, 720, 1],
          )
          && encoders.length === 1
          && submissions.length === 1
          && transfers.length === 1
          && stagingRecords.length === 1
          && transfer.capturePhase === phase
          && transfer.commandEncoderId === leg.commandEncoderId
          && transfer.method === 'copyTextureToBuffer'
          && transfer.source?.textureId === expectedTexture.gpuTextureId
          && transfer.source.mipLevel === 0
          && exactArray(
            [transfer.source.origin?.x, transfer.source.origin?.y, transfer.source.origin?.z],
            [0, 0, 0],
          )
          && transfer.source.aspect === 'all'
          && transfer.destination?.bufferId === leg.stagingBufferId
          && transfer.destination.offset === 0
          && transfer.destination.bytesPerRow === 5_120
          && transfer.destination.rowsPerImage === null
          && exactArray(
            [
              transfer.copySize?.width,
              transfer.copySize?.height,
              transfer.copySize?.depthOrArrayLayers,
            ],
            [1_280, 720, 1],
          )
          && staging.capturePhase === phase
          && staging.category === 'createBuffer'
          && staging.resourceClass === 'readback-staging'
          && staging.size === expectedReadbackBufferSize
          && staging.usage === expectedReadbackBufferUsage
          && staging.mappedAtCreation === false
          && leg.stagingCreationSequence === staging.sequence
          && leg.commandEncoderSequence === encoder.sequence
          && leg.commandBufferId === encoder.commandBufferId
          && leg.finishSequence === encoder.finishSequence
          && leg.submitSequence === submission.sequence
          && encoder.capturePhase === phase
          && encoder.renderPassEncoderIds?.length === 0
          && encoder.computePassEncoderIds?.length === 0
          && exactArray(encoder.transferSequences, [transfer.sequence])
          && encoder.finishCallCount === 1
          && staging.sequence < encoder.sequence
          && encoder.sequence < transfer.sequence
          && transfer.sequence < encoder.finishSequence
          && encoder.finishSequence < submission.sequence;
      };
      const phaseEncoders = challenge.commandEncoders.filter(
        (record) => record.capturePhase === phase,
      );
      const phaseSubmissions = challenge.queueSubmissions.filter(
        (record) => record.capturePhase === phase,
      );
      const phaseTransfers = challenge.commandEncoderTransfers.filter(
        (record) => record.capturePhase === phase,
      );
      return topology.pass === true
        && topology.phase === phase
        && topology.chronologyExact === true
        && phaseEncoders.length === 4
        && phaseSubmissions.length === 4
        && phaseTransfers.length === 2
        && renderLegExact(topology.clear, 'clear')
        && renderLegExact(topology.render, 'render')
        && readbackLegExact(topology.colorReadback, 'color')
        && readbackLegExact(topology.depthReadback, 'depth')
        && topology.colorReadback.sourceTextureId
          !== topology.depthReadback.sourceTextureId
        && topology.colorReadback.stagingBufferId
          !== topology.depthReadback.stagingBufferId
        && topology.clear.submitSequence < topology.render.commandEncoderSequence
        && topology.render.submitSequence
          < topology.colorReadback.stagingCreationSequence
        && topology.colorReadback.submitSequence
          < topology.depthReadback.stagingCreationSequence;
    });
  if (challenge.schemaVersion !== 1
    || challenge.kind !== 'immediate-aif-phase0-ordered-production-challenge'
    || challenge.pass !== true
    || typeof challenge.installationId !== 'string'
    || !['v99', 'v20'].includes(challenge.scenarioId)
    || !IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS.includes(challenge.scheduleId)
    || !Number.isInteger(challenge.startSequence)
    || !Number.isInteger(challenge.endSequence)
    || challenge.startSequence >= challenge.endSequence
    || challenge.callbackCount !== 18
    || !exactArray(challenge.expectedLanes, expectedLanes)
    || !exactArray(challenge.observedLanes, expectedLanes)
    || !Array.isArray(challenge.productionPhases)
    || challenge.productionPhases.length !== 18
    || new Set(challenge.productionPhases).size !== 18
    || challenge.renderPassCount !== 36
    || !exactArray(challenge.expectedRenderPassPhaseOrder, expectedPhaseOrder)
    || !exactArray(challenge.observedRenderPassPhaseOrder, expectedPhaseOrder)
    || challenge.productionRenderPassTopology?.length !== 18
    || !challenge.productionRenderPassTopology.every((record, index) => (
      record.pass === true
        && record.phase === challenge.productionPhases[index]
        && typeof record.clearCommandEncoderId === 'string'
        && typeof record.clearEncoderId === 'string'
        && Number.isInteger(record.clearBeginSequence)
        && Number.isInteger(record.clearEndSequence)
        && record.clearBeginSequence < record.clearEndSequence
        && typeof record.renderCommandEncoderId === 'string'
        && typeof record.renderEncoderId === 'string'
        && Number.isInteger(record.renderBeginSequence)
        && Number.isInteger(record.renderEndSequence)
        && record.renderBeginSequence < record.renderEndSequence
        && record.clearDrawLikeEvents?.length === 0
        && record.renderDrawLikeEvents?.length === 1
        && record.renderDrawLikeEvents[0].method === 'executeBundles'
        && exactArray(record.renderDrawLikeEvents[0].bundleIds, [record.expectedBundleId])
    ))
    || challenge.productionRenderPassTopologyExact !== true
    || challenge.computePassCount !== 0
    || challenge.computePasses?.length !== 0
    || !Array.isArray(challenge.productionExecuteSequences)
    || challenge.productionExecuteSequences.length !== 18
    || !sequencesStrictlyIncrease
    || challenge.executeSequencesStrictlyIncreasing !== true
    || challenge.diagnosticRenderPassCount !== 0
    || challenge.unexpectedRenderPassCount !== 0
    || challenge.forbiddenCreationCount !== 0
    || challenge.forbiddenCreations?.length !== 0
    || challenge.creationEvents?.length !== challenge.transientReadbackBufferCreationCount
    || challenge.transientReadbackBufferCreationCount !== 36
    || new Set(challenge.creationEvents.map((record) => record.resourceId)).size !== 36
    || !challenge.creationEvents.every((record) => (
      record.category === 'createBuffer'
        && record.resourceClass === 'readback-staging'
        && challenge.productionPhases.includes(record.capturePhase)
        && record.size === expectedReadbackBufferSize
        && record.usage === expectedReadbackBufferUsage
        && record.mappedAtCreation === false
    ))
    || challenge.queueWriteBufferCount !== challenge.queueWriteBuffers?.length
    || challenge.queueWriteTextureCount !== challenge.queueWriteTextures?.length
    || challenge.queueWriteTextureCount !== 0
    || challenge.queueWriteTextures.length !== 0
    || challenge.queueCopyExternalImageToTextureCount !== 0
    || challenge.queueCopyExternalImagesToTexture?.length !== 0
    || challenge.textureDestructionCount !== 0
    || challenge.textureDestructions?.length !== 0
    || challenge.frozenResourceIds?.length !== 4
    || new Set(challenge.frozenResourceIds).size !== 4
    || !challenge.frozenResourceIds.every((resourceId) => typeof resourceId === 'string')
    || challenge.diagnosticPositionResourceIds?.length !== 2
    || new Set(challenge.diagnosticPositionResourceIds).size !== 2
    || challenge.installationImmutableResourceIds?.length !== 6
    || new Set(challenge.installationImmutableResourceIds).size !== 6
    || !exactArray(
      challenge.installationImmutableResourceIds,
      [...challenge.frozenResourceIds, ...challenge.diagnosticPositionResourceIds],
    )
    || challenge.frozenBufferWriteCount !== 0
    || challenge.frozenBufferWrites?.length !== 0
    || challenge.frozenBufferMapCount !== 0
    || challenge.frozenBufferMaps?.length !== 0
    || challenge.installationFreezeBoundary?.pass !== true
    || challenge.installationFreezeBoundary.installationId !== challenge.installationId
    || !Number.isInteger(
      challenge.installationFreezeBoundary.installationStartSequence,
    )
    || !Number.isInteger(
      challenge.installationFreezeBoundary.installationQueueCompleteSequence,
    )
    || !Number.isInteger(
      challenge.installationFreezeBoundary.installationCompleteSequence,
    )
    || challenge.installationFreezeBoundary.installationStartSequence
      >= challenge.installationFreezeBoundary.installationQueueCompleteSequence
    || challenge.installationFreezeBoundary.installationQueueCompleteSequence
      >= challenge.installationFreezeBoundary.installationCompleteSequence
    || challenge.installationFreezeBoundary.installationCompleteSequence
      >= challenge.startSequence
    || challenge.installationFreezeBoundary.challengeStartSequence
      !== challenge.startSequence
    || !exactArray(
      challenge.installationFreezeBoundary.frozenResourceIds,
      challenge.frozenResourceIds,
    )
    || !exactArray(
      challenge.installationFreezeBoundary.diagnosticPositionResourceIds,
      challenge.diagnosticPositionResourceIds,
    )
    || !exactArray(
      challenge.installationFreezeBoundary.installationImmutableResourceIds,
      challenge.installationImmutableResourceIds,
    )
    || !Array.isArray(
      challenge.installationFreezeBoundary.prescribedInstallationWrites,
    )
    || !challenge.installationFreezeBoundary.prescribedInstallationWrites.every(
      (record) => Number.isInteger(record.sequence)
        && record.sequence
          > challenge.installationFreezeBoundary.installationStartSequence
        && record.sequence
          < challenge.installationFreezeBoundary.installationCompleteSequence
        && challenge.installationImmutableResourceIds.includes(record.bufferId),
    )
    || challenge.installationFreezeBoundary.boundaryQueueWrites?.length !== 0
    || challenge.installationFreezeBoundary.boundaryMaps?.length !== 0
    || challenge.installationFreezeBoundary.boundaryCommandWrites?.length !== 0
    || !Number.isInteger(challenge.mergedGeometryRealizationQueueCompleteSequence)
    || challenge.mergedGeometryRealizationQueueCompleteSequence >= challenge.startSequence
    || challenge.geometryRealizationBeforeChallenge !== true
    || !Number.isInteger(challenge.sharedGpuCommitmentCompleteSequence)
    || challenge.sharedGpuCommitmentCompleteSequence >= challenge.startSequence
    || challenge.sharedGpuCommitmentBeforeChallenge !== true
    || challenge.immutableResources?.length !== expectedImmutableSemantics.length
    || !exactArray(
      challenge.immutableResources.map((resource) => resource.semantic).sort(),
      [...expectedImmutableSemantics].sort(),
    )
    || !challenge.immutableResources.every((resource) => (
      Number.isInteger(resource.attributeId)
        && typeof resource.gpuBufferId === 'string'
    ))
    || challenge.immutableResourceIds?.length !== 18
    || new Set(challenge.immutableResourceIds).size !== 18
    || !challenge.immutableResourceIds.every((resourceId) => typeof resourceId === 'string')
    || !challenge.immutableResources.every(
      (resource) => challenge.immutableResourceIds.includes(resource.gpuBufferId),
    )
    || !challenge.frozenResourceIds.every(
      (resourceId) => challenge.immutableResourceIds.includes(resourceId),
    )
    || !challenge.diagnosticPositionResourceIds.every(
      (resourceId) => challenge.immutableResourceIds.includes(resourceId),
    )
    || challenge.immutableBufferWriteCount !== 0
    || challenge.immutableBufferWrites?.length !== 0
    || challenge.immutableBufferMapCount !== 0
    || challenge.immutableBufferMaps?.length !== 0
    || challenge.immutableTextureResources?.length !== 5
    || !exactArray(
      challenge.immutableTextureResources.map((record) => record.semantic),
      [
        'production.color',
        'production.depth',
        'address.color',
        'objectId.color',
        'objectId.depth',
      ],
    )
    || !challenge.immutableTextureResources.every((record) => (
      typeof record.textureUuid === 'string' && typeof record.gpuTextureId === 'string'
    ))
    || challenge.immutableTextureIds?.length !== 5
    || new Set(challenge.immutableTextureIds).size !== 5
    || !exactArray(
      challenge.immutableTextureIds,
      challenge.immutableTextureResources.map((record) => record.gpuTextureId),
    )
    || challenge.commandEncoderTransferCount !== 36
    || challenge.commandEncoderTransfers?.length !== 36
    || !challenge.productionPhases.every((phase) => (
      challenge.commandEncoderTransfers.filter(
        (record) => record.capturePhase === phase,
      ).length === 2
    ))
    || !challenge.commandEncoderTransfers.every((record) => (
      record.method === 'copyTextureToBuffer'
        && challenge.productionPhases.includes(record.capturePhase)
        && challenge.immutableTextureResources.slice(0, 2).some(
          (texture) => texture.gpuTextureId === record.source?.textureId,
        )
        && stagingResourceIds.has(record.destination?.bufferId)
        && record.source.mipLevel === 0
        && exactArray(
          [record.source.origin?.x, record.source.origin?.y, record.source.origin?.z],
          [0, 0, 0],
        )
        && record.source.aspect === 'all'
        && record.destination.offset === 0
        && record.destination.bytesPerRow === 5_120
        && record.destination.rowsPerImage === null
        && record.copySize?.width === 1_280
        && record.copySize?.height === 720
        && record.copySize?.depthOrArrayLayers === 1
    ))
    || challenge.immutableBufferCommandWriteCount !== 0
    || challenge.immutableBufferCommandWrites?.length !== 0
    || challenge.immutableTextureCommandWriteCount !== 0
    || challenge.immutableTextureCommandWrites?.length !== 0
    || challenge.expectedReadbackBufferSize !== expectedReadbackBufferSize
    || challenge.expectedReadbackBufferUsage !== expectedReadbackBufferUsage
    || challenge.productionCommandTopology?.length !== 18
    || challenge.productionCommandTopologyExact !== true
    || !productionCommandTopologyRawExact
    || challenge.commandReadbackTopologyExact !== true
    || challenge.commandEncoderCount !== 72
    || challenge.queueSubmissionCount !== 72
    || challenge.commandSubmissionTopologyExact !== true
    || !commandSubmissionRawExact
    || challenge.persistentInventory?.pass !== true
    || challenge.persistentInventory.exact !== true
    || !Array.isArray(challenge.persistentInventory.preflight?.records)
    || challenge.persistentInventory.preflight.records.length === 0
    || !Array.isArray(challenge.persistentInventory.postflight?.records)
    || JSON.stringify(challenge.persistentInventory.preflight.records)
      !== JSON.stringify(challenge.persistentInventory.postflight.records)) {
    throw new Error('Ordered production challenge sequencing is malformed.');
  }
  return challenge;
}

export function validateImmediateAifScheduleInstallationEvidence(installation) {
  requireObject(installation, 'Schedule installation');
  const expectedManagerSemantics = [
    'A.bucketBase',
    'A.indirectCommands',
    'I.indirectCommands',
    'F.indirectCommands',
  ];
  const expectedAttributeTypes = [1, 4, 4, 4];
  const managerCalls = installation.managerCalls;
  const commands = installation.commands;
  const commonState = installation.commonState;
  const diagnosticPositions = installation.diagnosticPositionRealization;
  const expectedScenarioOrdinal = installation.scenarioId === 'v99'
    ? installation.ordinal
    : installation.ordinal - 4;
  const expectedInstallationWriteSemantics = expectedScenarioOrdinal === 1
    ? installation.scenarioId === 'v99'
      ? []
      : ['A.bucketBase']
    : ['A.bucketBase', 'F.indirectCommands'];
  const expectedDiagnosticWriteSemantics = installation.scenarioId === 'v99'
      && installation.ordinal === 1
    ? []
    : ['featurePosition'];
  const rawCommonProjectionExact = [
    commonState?.preflight?.rawProjection,
    commonState?.postflight?.rawProjection,
  ].every((projection) => {
    const renderer = projection?.renderer;
    return renderer?.rendererConstructor === 'WebGPURenderer'
      && renderer.backendConstructor === 'WebGPUBackend'
      && typeof renderer.backendId === 'string'
      && renderer.viewport?.width === 1_280
      && renderer.viewport?.height === 720
      && renderer.viewport?.devicePixelRatio === 1
      && renderer.coordinateSystem === 2_001
      && renderer.autoClear === true
      && renderer.autoClearColor === true
      && renderer.autoClearDepth === true
      && renderer.autoClearStencil === true
      && renderer.clearColor === 0x030711
      && renderer.clearAlpha === 1
      && renderer.clearDepth === 0
      && renderer.clearStencil === 0
      && renderer.sortObjects === false
      && renderer.toneMapping === 0
      && renderer.toneMappingExposure === 1
      && renderer.outputColorSpace === 'srgb'
      && renderer.reversedDepthBuffer === true
      && renderer.trackTimestamp === false
      && typeof renderer.backendDeviceId === 'string'
      && renderer.backendDeviceId === renderer.backendParameterDeviceId
      && renderer.backendDeviceMatchesParameter === true
      && renderer.activeRenderTargetUuid === null
      && renderer.activeCubeFace === 0
      && renderer.activeMipmapLevel === 0
      && projection.camera !== null
      && typeof projection.camera === 'object'
      && projection.camera.reversedDepth === true
      && exactArray(Object.keys(projection.materials ?? {}), IMMEDIATE_AIF_PHASE0_LANES)
      && projection.lights?.hemisphere !== null
      && projection.lights?.directional?.target !== null
      && exactArray(Object.keys(projection.scene?.laneRoots ?? {}),
        IMMEDIATE_AIF_PHASE0_LANES)
      && projection.targets?.production !== null
      && projection.targets?.address !== null
      && projection.targets?.objectId !== null;
  });
  if (installation.schemaVersion !== 1
    || installation.kind !== 'immediate-aif-phase0-schedule-installation-preflight'
    || installation.pass !== true
    || installation.frozen !== true
    || typeof installation.installationId !== 'string'
    || installation.installationId.length === 0
    || !Number.isInteger(installation.ordinal)
    || installation.ordinal <= 0
    || installation.scenarioOrdinal !== expectedScenarioOrdinal
    || installation.scenarioOrdinal < 1
    || installation.scenarioOrdinal > 4
    || !['v99', 'v20'].includes(installation.scenarioId)
    || !IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS.includes(installation.scheduleId)
    || installation.basePhase
      !== `phase0/schedule-install/${installation.scenarioId}/${installation.ordinal}-${installation.scheduleId}`
    || !Number.isInteger(installation.startSequence)
    || !Number.isInteger(installation.queueCompleteSequence)
    || !Number.isInteger(installation.completeSequence)
    || installation.startSequence >= installation.queueCompleteSequence
    || installation.queueCompleteSequence >= installation.completeSequence
    || !Array.isArray(managerCalls)
    || managerCalls.length !== 4
    || !exactArray(managerCalls.map((call) => call.semantic), expectedManagerSemantics)
    || !exactArray(managerCalls.map((call) => call.attributeType), expectedAttributeTypes)
    || !managerCalls.every((call) => call.manager === 'renderer._attributes'
      && call.managerConstructor === 'Attributes'
      && typeof call.gpuBufferId === 'string'
      && typeof call.dirtyBeforeUpdate === 'boolean'
      && typeof call.arrayType === 'string'
      && Number.isInteger(call.byteLength)
      && call.byteLength > 0
      && /^[0-9a-f]{64}$/.test(call.cpuSha256 ?? '')
      && Number.isInteger(call.startSequence)
      && Number.isInteger(call.completeSequence)
      && call.startSequence < call.completeSequence
      && call.completeSequence < installation.queueCompleteSequence)
    || !managerCalls.every((call, index) => (
      index === 0 || call.startSequence > managerCalls[index - 1].completeSequence
    ))
    || installation.queueWritesExact !== true
    || !exactArray(
      installation.expectedWriteSemantics,
      expectedInstallationWriteSemantics,
    )
    || !exactArray(
      installation.observedDirtySemantics,
      expectedInstallationWriteSemantics,
    )
    || !exactArray(
      installation.observedWriteSemantics,
      expectedInstallationWriteSemantics,
    )
    || installation.queueWriteBufferCount !== expectedInstallationWriteSemantics.length
    || installation.queueWriteBuffers?.length !== expectedInstallationWriteSemantics.length
    || !installation.queueWriteBuffers.every((write, index) => {
      const managerCall = managerCalls.find(
        (record) => record.semantic === expectedInstallationWriteSemantics[index],
      );
      return managerCall !== undefined
        && write.capturePhase === `${installation.basePhase}/attribute-realization`
        && write.bufferId === managerCall.gpuBufferId
        && write.bufferOffset === 0
        && write.sourceType === managerCall.arrayType
        && write.sourceByteLength === managerCall.byteLength
        && write.dataOffset === 0
        && write.size === null
        && write.semantic === managerCall.semantic
        && write.sourceSha256 === managerCall.cpuSha256
        && write.sequence > managerCall.startSequence
        && write.sequence < managerCall.completeSequence;
    })
    || diagnosticPositions?.schemaVersion !== 1
    || diagnosticPositions.kind
      !== 'immediate-aif-phase0-address-position-realization'
    || diagnosticPositions.pass !== true
    || diagnosticPositions.installationId !== installation.installationId
    || diagnosticPositions.scenarioId !== installation.scenarioId
    || diagnosticPositions.scheduleId !== installation.scheduleId
    || typeof diagnosticPositions.capturePhase !== 'string'
    || !Number.isInteger(diagnosticPositions.startSequence)
    || !Number.isInteger(diagnosticPositions.queueCompleteSequence)
    || diagnosticPositions.startSequence <= installation.queueCompleteSequence
    || diagnosticPositions.startSequence >= diagnosticPositions.queueCompleteSequence
    || diagnosticPositions.queueCompleteSequence >= installation.completeSequence
    || !exactArray(
      diagnosticPositions.expectedWriteSemantics,
      expectedDiagnosticWriteSemantics,
    )
    || !exactArray(
      diagnosticPositions.observedWriteSemantics,
      expectedDiagnosticWriteSemantics,
    )
    || diagnosticPositions.queueWriteBufferCount
      !== expectedDiagnosticWriteSemantics.length
    || diagnosticPositions.queueWriteBuffers?.length
      !== expectedDiagnosticWriteSemantics.length
    || diagnosticPositions.queueWritesExact !== true
    || diagnosticPositions.records?.length !== 2
    || !exactArray(
      diagnosticPositions.records.map((record) => record.semantic),
      ['commonPosition', 'featurePosition'],
    )
    || !diagnosticPositions.records.every((record) => (
      record.manager === 'renderer._attributes'
        && record.managerConstructor === 'Attributes'
        && record.attributeType === 1
        && Number.isInteger(record.attributeId)
        && Number.isInteger(record.attributeVersion)
        && record.afterDataVersion === record.attributeVersion
        && record.arrayType === 'Float32Array'
        && record.itemSize === 3
        && Number.isInteger(record.count)
        && record.count > 0
        && record.byteLength === record.count * 3 * 4
        && record.cpuExact === true
        && /^[0-9a-f]{64}$/.test(record.cpuSha256 ?? '')
        && record.witnessId === `sha256-${record.cpuSha256}-${record.byteLength}`
        && typeof record.gpuBufferId === 'string'
        && record.gpuBufferSize === record.byteLength
        && record.gpuBufferCreation?.resourceId === record.gpuBufferId
        && record.gpuBufferCreation.resourceClass === 'persistent-or-upload-buffer'
        && record.gpuBufferCreation.size === record.byteLength
        && record.gpuBufferCreation.usage === 44
        && record.gpuBufferCreation.mappedAtCreation === true
        && Number.isInteger(record.gpuBufferCreation.sequence)
        && record.gpuBufferCreation.sequence < record.updateStartSequence
        && Number.isInteger(record.updateStartSequence)
        && Number.isInteger(record.updateCompleteSequence)
        && record.updateStartSequence < record.updateCompleteSequence
        && record.updateCompleteSequence < diagnosticPositions.queueCompleteSequence
    ))
    || diagnosticPositions.records[0].dirtyBeforeUpdate !== false
    || diagnosticPositions.records[1].dirtyBeforeUpdate
      !== (expectedDiagnosticWriteSemantics.length === 1)
    || !diagnosticPositions.queueWriteBuffers.every((write) => {
      const feature = diagnosticPositions.records[1];
      return write.capturePhase === diagnosticPositions.capturePhase
        && write.bufferId === feature.gpuBufferId
        && write.bufferOffset === 0
        && write.sourceType === 'Float32Array'
        && write.sourceByteLength === feature.byteLength
        && write.dataOffset === 0
        && write.size === null
        && write.semantic === feature.semantic
        && write.sourceSha256 === feature.cpuSha256
        && write.sourceWitnessId === feature.witnessId
        && write.sequence > feature.updateStartSequence
        && write.sequence < feature.updateCompleteSequence;
    })
    || diagnosticPositions.laneBindings?.A?.semantic !== 'commonPosition'
    || diagnosticPositions.laneBindings?.I?.semantic !== 'commonPosition'
    || diagnosticPositions.laneBindings?.F?.semantic !== 'featurePosition'
    || diagnosticPositions.laneBindings.A.attributeId
      !== diagnosticPositions.records[0].attributeId
    || diagnosticPositions.laneBindings.I.attributeId
      !== diagnosticPositions.records[0].attributeId
    || diagnosticPositions.laneBindings.F.attributeId
      !== diagnosticPositions.records[1].attributeId
    || diagnosticPositions.laneBindings.A.gpuBufferId
      !== diagnosticPositions.records[0].gpuBufferId
    || diagnosticPositions.laneBindings.I.gpuBufferId
      !== diagnosticPositions.records[0].gpuBufferId
    || diagnosticPositions.laneBindings.F.gpuBufferId
      !== diagnosticPositions.records[1].gpuBufferId
    || !exactArray(Object.keys(commands ?? {}), IMMEDIATE_AIF_PHASE0_LANES)
    || !IMMEDIATE_AIF_PHASE0_LANES.every((lane) => {
      const command = commands[lane];
      return command?.lane === lane
        && command.exact === true
        && command.byteLength === IMMEDIATE_AIF_PHASE0_WORKLOAD.commandByteLength
        && command.frozenWords?.length === IMMEDIATE_AIF_PHASE0_WORKLOAD.commandUint32Length
        && command.expectedWords?.length === IMMEDIATE_AIF_PHASE0_WORKLOAD.commandUint32Length
        && command.frozenWords.every((word, index) => word === command.expectedWords[index])
        && /^[0-9a-f]{64}$/.test(command.gpuSha256 ?? '')
        && command.gpuSha256 === command.expectedSha256
        && typeof command.gpuBufferId === 'string'
        && Number.isInteger(command.readbackStartSequence)
        && Number.isInteger(command.readbackCompleteSequence)
        && command.readbackStartSequence > installation.queueCompleteSequence
        && command.readbackStartSequence > diagnosticPositions.queueCompleteSequence
        && command.readbackCompleteSequence > command.readbackStartSequence;
    })
    || !/^[0-9a-f]{64}$/.test(installation.commandSetSha256 ?? '')
    || installation.aBucketBase?.exact !== true
    || !/^[0-9a-f]{64}$/.test(installation.aBucketBase.cpuSha256 ?? '')
    || installation.frozenResourceIds?.length !== 4
    || new Set(installation.frozenResourceIds).size !== 4
    || !exactArray(
      installation.frozenResourceIds,
      [
        managerCalls[0]?.gpuBufferId,
        commands?.A?.gpuBufferId,
        commands?.I?.gpuBufferId,
        commands?.F?.gpuBufferId,
      ],
    )
    || commonState?.pass !== true
    || commonState.exactProjection !== true
    || commonState.exactSha256 !== true
    || !/^[0-9a-f]{64}$/.test(commonState.preflight?.sha256 ?? '')
    || commonState.preflight.sha256 !== commonState.postflight?.sha256
    || !/^[0-9a-f]{64}$/.test(commonState.preflight.rawSha256 ?? '')
    || !/^[0-9a-f]{64}$/.test(commonState.postflight.rawSha256 ?? '')
    || commonState.preflight.laneSelection?.exact !== true
    || commonState.postflight.laneSelection?.exact !== true
    || typeof commonState.preflight.projection !== 'object'
    || typeof commonState.preflight.rawProjection !== 'object'
    || typeof commonState.postflight.projection !== 'object'
    || typeof commonState.postflight.rawProjection !== 'object'
    || !rawCommonProjectionExact) {
    throw new Error('Schedule installation evidence is malformed.');
  }
  if (installation.orderedChallenge !== null) {
    validateImmediateAifOrderedProductionChallengeEvidence(
      installation.orderedChallenge,
    );
    if (installation.orderedChallenge.installationId !== installation.installationId
      || installation.orderedChallenge.scenarioId !== installation.scenarioId
      || installation.orderedChallenge.scheduleId !== installation.scheduleId) {
      throw new Error('Schedule installation challenge binding is malformed.');
    }
  }
  return installation;
}

function validateMergedGeometryRealizationEvidence(realization) {
  requireObject(realization, 'Merged geometry realization');
  const semantics = ['bucketBase', 'normal', 'position', 'uv', 'index'];
  if (realization.schemaVersion !== 1
    || realization.kind !== 'immediate-aif-phase0-merged-geometry-realization'
    || realization.pass !== true
    || realization.frozen !== true
    || realization.capturePhase !== 'phase0/resources/merged-geometry-realization'
    || typeof realization.geometryUuid !== 'string'
    || !Number.isInteger(realization.startSequence)
    || !Number.isInteger(realization.queueCompleteSequence)
    || realization.startSequence >= realization.queueCompleteSequence
    || !exactArray(realization.expectedAttributeNames, semantics.slice(0, 4))
    || realization.recordCount !== 5
    || realization.records?.length !== 5
    || !exactArray(realization.records.map((record) => record.semantic), semantics)
    || !exactArray(realization.records.map((record) => record.attributeType), [1, 1, 1, 1, 2])
    || !realization.records.every((record, index) => (
      record.manager === 'renderer._attributes'
        && record.managerConstructor === 'Attributes'
        && record.resourceKind === (index === 4 ? 'index' : 'vertex-attribute')
        && Number.isInteger(record.attributeId)
        && Number.isInteger(record.attributeVersion)
        && record.afterDataVersion === record.attributeVersion
        && (record.semantic === 'uv'
          ? record.beforeDataVersion === null
          : record.beforeDataVersion === record.attributeVersion)
        && typeof record.gpuBufferId === 'string'
        && Number.isInteger(record.startSequence)
        && Number.isInteger(record.completeSequence)
        && record.startSequence < record.completeSequence
        && record.completeSequence < realization.queueCompleteSequence
        && Number.isInteger(record.byteLength)
        && record.byteLength > 0
        && /^[0-9a-f]{64}$/.test(record.cpuSha256 ?? '')
    ))
    || realization.frozenResourceIds?.length !== 5
    || realization.managerVersionsExact !== true
    || realization.queueWriteBufferCount !== 0
    || realization.queueWriteBuffers?.length !== 0
    || new Set(realization.frozenResourceIds).size !== 5
    || !exactArray(
      realization.frozenResourceIds,
      realization.records.map((record) => record.gpuBufferId),
    )) {
    throw new Error('Merged geometry realization evidence is malformed.');
  }
  return realization;
}

function validateGpuByteWitnessEvidence(evidence) {
  requireObject(evidence, 'GPU byte witnesses');
  const sharedSemantics = ['matrix', 'bounds', 'objectBucket', 'bucketBase',
    'bucketCapacity', 'cullOrder', 'visibleIds', 'overflow'];
  const expectedSemantics = [
    ...['v99', 'v20'].flatMap((scenarioId) => (
      sharedSemantics.map((semantic) => `shared/${scenarioId}/${semantic}`)
    )),
    ...['bucketBase', 'normal', 'position', 'uv', 'index'].map(
      (semantic) => `merged-geometry/${semantic}`,
    ),
  ];
  const witnesses = evidence.witnesses;
  const references = evidence.references;
  if (evidence.schemaVersion !== 1
    || evidence.kind !== 'immediate-aif-phase0-gpu-byte-witnesses'
    || evidence.pass !== true
    || witnesses === null
    || typeof witnesses !== 'object'
    || Array.isArray(witnesses)
    || evidence.witnessCount !== Object.keys(witnesses).length
    || evidence.witnessCount <= 0
    || evidence.referenceCount !== expectedSemantics.length
    || references?.length !== expectedSemantics.length
    || !exactArray(
      references.map((reference) => reference.semantic).sort(),
      [...expectedSemantics].sort(),
    )
    || !references.every((reference) => {
      const witness = witnesses[reference.witnessId];
      return typeof reference.gpuBufferId === 'string'
        && typeof reference.capturePhase === 'string'
        && reference.capturePhase.length > 0
        && Number.isInteger(reference.byteLength)
        && reference.byteLength > 0
        && /^[0-9a-f]{64}$/.test(reference.sha256 ?? '')
        && witness?.witnessId === reference.witnessId
        && witness.sha256 === reference.sha256
        && witness.byteLength === reference.byteLength;
    })
    || !Object.entries(witnesses).every(([witnessId, witness]) => (
      witness?.schemaVersion === 1
        && witness.kind === 'immediate-aif-phase0-gpu-byte-witness'
        && witness.witnessId === witnessId
        && witness.encoding === 'base64-exact-bytes'
        && Number.isInteger(witness.byteLength)
        && witness.byteLength > 0
        && /^[0-9a-f]{64}$/.test(witness.sha256 ?? '')
        && witnessId === `sha256-${witness.sha256}-${witness.byteLength}`
        && typeof witness.bytesBase64 === 'string'
        && witness.bytesBase64.length > 0
    ))) {
    throw new Error('GPU byte witness evidence is malformed.');
  }
  return evidence;
}

function validateAddressDiagnosticPositionEvidence(evidence) {
  requireObject(evidence, 'Address diagnostic position evidence');
  const records = evidence.records;
  const witnesses = evidence.witnesses;
  const expectedScheduleOrder = ['v99', 'v20'].flatMap((scenarioId) => (
    ['canonical', 'S1', 'S2', 'canonical'].map(
      (scheduleId) => `${scenarioId}/${scheduleId}`,
    )
  ));
  const commonRecords = records?.map((record) => record.records?.[0]);
  const featureRecords = records?.map((record) => record.records?.[1]);
  const featureHashes = Object.fromEntries(['canonical', 'S1', 'S2'].map(
    (scheduleId) => [scheduleId, [...new Set(records?.filter(
      (record) => record.scheduleId === scheduleId,
    ).map((record) => record.records[1].cpuSha256) ?? [])]],
  ));
  if (evidence.schemaVersion !== 1
    || evidence.kind !== 'immediate-aif-phase0-address-diagnostic-position-evidence'
    || evidence.pass !== true
    || evidence.expectedRecordCount !== 8
    || evidence.recordCount !== 8
    || records?.length !== 8
    || !exactArray(evidence.scheduleOrder, expectedScheduleOrder)
    || !exactArray(evidence.expectedScheduleOrder, expectedScheduleOrder)
    || !exactArray(
      records.map((record) => `${record.scenarioId}/${record.scheduleId}`),
      expectedScheduleOrder,
    )
    || evidence.stableResourceIdentity !== true
    || evidence.exactScheduleContent !== true
    || new Set(commonRecords.map(
      (record) => `${record.attributeId}/${record.gpuBufferId}`,
    )).size !== 1
    || new Set(featureRecords.map(
      (record) => `${record.attributeId}/${record.gpuBufferId}`,
    )).size !== 1
    || commonRecords[0].attributeId === featureRecords[0].attributeId
    || commonRecords[0].gpuBufferId === featureRecords[0].gpuBufferId
    || new Set(commonRecords.map((record) => record.cpuSha256)).size !== 1
    || !Object.values(featureHashes).every((hashes) => hashes.length === 1)
    || new Set(Object.values(featureHashes).map((hashes) => hashes[0])).size !== 3
    || JSON.stringify(evidence.featureHashesBySchedule) !== JSON.stringify(featureHashes)
    || records.reduce((count, record) => count + record.queueWriteBufferCount, 0) !== 7
    || witnesses === null
    || typeof witnesses !== 'object'
    || Array.isArray(witnesses)
    || evidence.witnessCount !== 4
    || Object.keys(witnesses).length !== 4
    || !Object.entries(witnesses).every(([witnessId, witness]) => (
      witness?.schemaVersion === 1
        && witness.kind === 'immediate-aif-phase0-address-position-cpu-byte-witness'
        && witness.witnessId === witnessId
        && witness.encoding === 'base64-exact-bytes'
        && Number.isInteger(witness.byteLength)
        && witness.byteLength > 0
        && /^[0-9a-f]{64}$/.test(witness.sha256 ?? '')
        && witnessId === `sha256-${witness.sha256}-${witness.byteLength}`
        && typeof witness.bytesBase64 === 'string'
        && witness.bytesBase64.length > 0
    ))
    || !records.every((record) => record.records.every((position) => {
      const witness = witnesses[position.witnessId];
      return witness?.sha256 === position.cpuSha256
        && witness.byteLength === position.byteLength;
    }))) {
    throw new Error('Address diagnostic position evidence is malformed.');
  }
  return evidence;
}

const QUEUE_WRITE_BUFFER_CATEGORY_ORDER = Object.freeze([
  'renderer-initialization',
  'scenario-matrix-realization',
  'live-compute',
  'shared-resource-commitment',
  'merged-geometry-realization',
  'schedule-attribute-realization',
  'schedule-diagnostic-position-realization',
  'production-render',
  'diagnostic-render',
]);

function queueWriteBufferCategory(capturePhase) {
  if (capturePhase === 'phase0/renderer-init') return 'renderer-initialization';
  if (/^phase0\/scenario-load\/(?:v99|v20)\/matrix-realization$/.test(capturePhase)) {
    return 'scenario-matrix-realization';
  }
  if (/^phase0\/live\/(?:v99|v20)\/[AIF]$/.test(capturePhase)) return 'live-compute';
  if (/^phase0\/resources\/(?:v99|v20)\/shared-commitment$/.test(capturePhase)) {
    return 'shared-resource-commitment';
  }
  if (capturePhase === 'phase0/resources/merged-geometry-realization') {
    return 'merged-geometry-realization';
  }
  if (/^phase0\/schedule-install\/(?:v99|v20)\/\d+-(?:canonical|S1|S2)\/attribute-realization$/.test(capturePhase)) {
    return 'schedule-attribute-realization';
  }
  if (/^phase0\/schedule-install\/(?:v99|v20)\/\d+-(?:canonical|S1|S2)\/diagnostic-position-realization$/.test(capturePhase)) {
    return 'schedule-diagnostic-position-realization';
  }
  if (/^phase0\/prime\/[AIF]$/.test(capturePhase)
    || /^phase0\/(?:v99|v20)\/(?:canonical|S1|S2)\/[AIF]\/capture-\d+\/production$/.test(capturePhase)) {
    return 'production-render';
  }
  if (/^phase0\/diagnostic-(?:prime|rerecord)\//.test(capturePhase)
    || /^phase0\/(?:v99|v20)\/(?:canonical|S1|S2)\/[AIF]\/capture-\d+\/(?:address|object-id)$/.test(capturePhase)) {
    return 'diagnostic-render';
  }
  return null;
}

function queueWriteBufferRawProjection(record) {
  return {
    sequence: record.sequence,
    capturePhase: record.capturePhase,
    bufferId: record.bufferId,
    bufferOffset: record.bufferOffset,
    sourceId: record.sourceId,
    sourceBackingBufferId: record.sourceBackingBufferId,
    sourceType: record.sourceType,
    sourceByteLength: record.sourceByteLength,
    sourceByteOffset: record.sourceByteOffset,
    sourceElementByteSize: record.sourceElementByteSize,
    sourceElementCount: record.sourceElementCount,
    dataOffset: record.dataOffset,
    size: record.size,
    selectedSourceElementOffset: record.selectedSourceElementOffset,
    selectedSourceElementLength: record.selectedSourceElementLength,
    selectedSourceByteOffset: record.selectedSourceByteOffset,
    selectedSourceByteLength: record.selectedSourceByteLength,
    selectedSourceSha256: record.selectedSourceSha256,
  };
}

function validateQueueWriteBufferLedgerEvidence(
  ledger,
  lifecycle,
  scenarios,
  pipelines,
  resources,
) {
  requireObject(ledger, 'Queue write-buffer ledger');
  const rawCalls = lifecycle.queueWriteBufferCalls;
  const records = ledger.records;
  const specializedCategories = new Set([
    'scenario-matrix-realization',
    'shared-resource-commitment',
    'schedule-attribute-realization',
    'schedule-diagnostic-position-realization',
  ]);
  const expectedSpecializedCounts = {
    'scenario-matrix-realization': 1,
    'shared-resource-commitment': 2,
    'schedule-attribute-realization': 13,
    'schedule-diagnostic-position-realization': 7,
  };
  const expectedSources = [
    ...scenarios.map((scenario) => ({
      ownerKind: 'scenario-matrix-realization', ownerId: scenario.scenarioId,
      records: scenario.matrixGpuRealization.queueWriteBuffers,
    })),
    ...scenarios.map((scenario) => ({
      ownerKind: 'shared-resource-commitment', ownerId: scenario.scenarioId,
      records: scenario.gpuResourceCommitments.queueWriteBuffers,
    })),
    ...scenarios.flatMap((scenario) => scenario.scheduleInstallations.map((installation) => ({
      ownerKind: 'schedule-attribute-realization', ownerId: installation.installationId,
      records: installation.queueWriteBuffers,
    }))),
    ...scenarios.flatMap((scenario) => scenario.scheduleInstallations.map((installation) => ({
      ownerKind: 'schedule-diagnostic-position-realization',
      ownerId: installation.installationId,
      records: installation.diagnosticPositionRealization.queueWriteBuffers,
    }))),
  ];
  const sourceSequences = ledger.specializedSources?.flatMap(
    (source) => source.sequences,
  ).sort((left, right) => left - right);
  const ledgerSpecializedSequences = records?.filter(
    (record) => specializedCategories.has(record.category),
  ).map((record) => record.sequence).sort((left, right) => left - right);
  const rawExact = Array.isArray(rawCalls)
    && Array.isArray(records)
    && rawCalls.length === records.length
    && rawCalls.every((raw, index) => exactValue(
      queueWriteBufferRawProjection(raw),
      queueWriteBufferRawProjection(records[index]),
    ));
  const partitionsExact = ledger.partitions?.length === QUEUE_WRITE_BUFFER_CATEGORY_ORDER.length
    && ledger.partitions.every((partition, index) => {
      const category = QUEUE_WRITE_BUFFER_CATEGORY_ORDER[index];
      const expected = records.filter((record) => record.category === category);
      return partition.category === category
        && partition.count === expected.length
        && exactValue(partition.records, expected);
    });
  const sourcesExact = ledger.specializedSources?.length === expectedSources.length
    && ledger.specializedSources.every((source, index) => {
      const expected = expectedSources[index];
      return source.ownerKind === expected.ownerKind
        && source.ownerId === expected.ownerId
        && source.rawExact === true
        && source.recordCount === expected.records.length
        && exactValue(source.records, expected.records)
        && exactArray(source.sequences, expected.records.map((record) => record.sequence));
    });
  const bufferCreations = resources.gpuCreations?.filter(
    (resource) => resource.method === 'createBuffer',
  ) ?? [];
  const bindGroups = resources.gpuCreations?.filter(
    (resource) => resource.method === 'createBindGroup',
  ) ?? [];
  const bindGroupLayouts = pipelines.bindGroupLayouts ?? [];
  const ownerProofExact = (record) => {
    const owner = record.destinationOwner;
    const creations = bufferCreations.filter(
      (creation) => creation.resourceId === record.bufferId,
    );
    const creation = creations.length === 1 ? creations[0] : null;
    if (creation === null
      || owner?.resourceId !== record.bufferId
      || owner.resourceClass !== creation.resourceClass
      || owner.createSequence !== creation.sequence
      || owner.createCapturePhase !== creation.capturePhase
      || owner.size !== creation.size
      || owner.usage !== creation.usage
      || owner.mappedAtCreation !== creation.mappedAtCreation) return false;
    const proofs = owner.uniformBindingProofs ?? [];
    const allBufferBindingOccurrences = bindGroups.flatMap((bindGroup) => (
      bindGroup.entries?.filter((entry) => entry.bufferId === record.bufferId).map(
        (entry) => ({ bindGroup, entry }),
      ) ?? []
    ));
    const uniformProofsExact = proofs.every((proof) => {
      const matchingBindGroups = bindGroups.filter(
        (bindGroup) => bindGroup.resourceId === proof.bindGroupId,
      );
      const bindGroup = matchingBindGroups.length === 1 ? matchingBindGroups[0] : null;
      const matchingEntries = bindGroup?.entries?.filter((entry) => (
        entry.binding === proof.binding && entry.bufferId === record.bufferId
      )) ?? [];
      const matchingLayouts = bindGroupLayouts.filter(
        (layout) => layout.layoutId === proof.bindGroupLayoutId,
      );
      const matchingCoordinates = matchingLayouts.flatMap((layout) => (
        layout.entries.filter((entry) => entry.binding === proof.binding)
      ));
      const entry = matchingEntries.length === 1 ? matchingEntries[0] : null;
      const coordinate = matchingCoordinates.length === 1 ? matchingCoordinates[0] : null;
      return proof.exact === true
        && matchingBindGroups.length === 1
        && matchingEntries.length === 1
        && matchingLayouts.length === 1
        && matchingCoordinates.length === 1
        && bindGroup.layoutId === proof.bindGroupLayoutId
        && proof.bindGroupCreateSequence === bindGroup.sequence
        && proof.bindGroupCapturePhase === bindGroup.capturePhase
        && proof.bindGroupLayoutCreateSequence === matchingLayouts[0].sequence
        && proof.bufferId === record.bufferId
        && proof.bufferOffset === entry.bufferOffset
        && proof.bufferSize === entry.bufferSize
        && proof.bufferRawType === coordinate.buffer?.type
        && proof.bufferType === 'uniform'
        && (coordinate.buffer?.type ?? 'uniform') === proof.bufferType
        && coordinate.buffer.hasDynamicOffset === proof.hasDynamicOffset
        && coordinate.buffer.minBindingSize === proof.minBindingSize
        && proof.layoutMatchCount === 1
        && proof.coordinateMatchCount === 1;
    });
    return owner.exact === true
      && (owner.destinationBindingClass === 'known-attribute'
        ? owner.semanticBindings?.length > 0 && proofs.length === 0
        : owner.destinationBindingClass === 'uniform-bind-group-buffer'
          && owner.semanticBindings?.length === 0
          && proofs.length === 1
          && allBufferBindingOccurrences.length === 1
          && allBufferBindingOccurrences[0].bindGroup.resourceId === proofs[0].bindGroupId
          && allBufferBindingOccurrences[0].entry.binding === proofs[0].binding
          && uniformProofsExact);
  };
  const uniformOnlyCategories = new Set([
    'renderer-initialization', 'production-render', 'diagnostic-render',
  ]);
  const policyExact = records?.every((record) => {
    const owner = record.destinationOwner;
    const semantics = owner.semanticBindings?.map((binding) => binding.semantic) ?? [];
    const uniform = owner.destinationBindingClass === 'uniform-bind-group-buffer';
    if (uniformOnlyCategories.has(record.category)) return uniform;
    if (record.category === 'live-compute') {
      const match = record.capturePhase.match(/^phase0\/live\/(?:v99|v20)\/([AIF])$/);
      const lane = match?.[1] ?? null;
      const allowed = [
        'shared.bounds', 'shared.objectBucket', 'shared.bucketBase',
        'shared.bucketCapacity', 'shared.cullOrder', 'shared.visibleIds',
        'shared.overflow', `command.${lane}`,
      ];
      return uniform || (semantics.length > 0
        && semantics.every((semantic) => allowed.includes(semantic)));
    }
    if (record.category === 'scenario-matrix-realization') {
      return semantics.includes('shared.matrix');
    }
    if (record.category === 'shared-resource-commitment') {
      return semantics.includes('shared.visibleIds');
    }
    if (record.category === 'schedule-attribute-realization') {
      return semantics.some((semantic) => semantic === 'geometry.A.bucketBase'
        || /^command\.[AIF]$/.test(semantic));
    }
    if (record.category === 'schedule-diagnostic-position-realization') {
      return semantics.includes('diagnostic.address.featurePosition');
    }
    return false;
  }) === true;
  const expectedLiveAttributeWrites = [
    ['v99', 'I', []],
    ['v99', 'F', ['shared.visibleIds', 'shared.overflow']],
    ['v99', 'A', ['shared.visibleIds', 'shared.overflow']],
    ['v20', 'A', [
      'shared.bounds', 'shared.objectBucket', 'shared.bucketBase',
      'shared.bucketCapacity', 'shared.cullOrder', 'shared.visibleIds',
      'shared.overflow', 'command.A',
    ]],
    ['v20', 'F', ['shared.visibleIds', 'shared.overflow', 'command.F']],
    ['v20', 'I', ['shared.visibleIds', 'shared.overflow', 'command.I']],
  ];
  const expectedLiveGrammar = expectedLiveAttributeWrites.map(([
    scenarioId, lane, expectedSemantics,
  ]) => {
    const capturePhase = `phase0/live/${scenarioId}/${lane}`;
    const phaseRecords = records?.filter((record) => record.capturePhase === capturePhase) ?? [];
    const attributeRecords = phaseRecords.filter(
      (record) => record.destinationOwner.destinationBindingClass === 'known-attribute',
    );
    const observedSemantics = attributeRecords.flatMap(
      (record) => record.destinationOwner.semanticBindings.map((binding) => binding.semantic),
    );
    return {
      pass: exactArray([...observedSemantics].sort(), [...expectedSemantics].sort()),
      scenarioId,
      lane,
      capturePhase,
      expectedSemantics,
      observedSemantics,
      attributeWriteCount: attributeRecords.length,
      uniformWriteCount: phaseRecords.length - attributeRecords.length,
      attributeRecords,
    };
  });
  const uniformLedgerRecords = records?.filter(
    (record) => record.destinationOwner.destinationBindingClass
      === 'uniform-bind-group-buffer',
  ) ?? [];
  const activeBindGroupAtUse = (
    events,
    bindGroupId,
    useSequence,
    usePredicate,
  ) => {
    const activeSlots = new Map();
    const useMatches = [];
    for (const event of events ?? []) {
      if (event.method === 'setBindGroup') {
        activeSlots.set(event.index, event);
        continue;
      }
      if (event.sequence !== useSequence || !usePredicate(event)) continue;
      const activeMatches = [...activeSlots].filter(([, active]) => (
        active.bindGroupId === bindGroupId && active.sequence < event.sequence
      ));
      useMatches.push({ event, activeMatches });
    }
    const selected = useMatches.length === 1 ? useMatches[0] : null;
    const activeMatchCount = selected?.activeMatches.length ?? 0;
    const activeMatch = activeMatchCount === 1 ? selected.activeMatches[0] : null;
    return {
      useMatchCount: useMatches.length,
      activeMatchCount,
      bindGroupIndex: activeMatch?.[0] ?? null,
      activeSetBindGroupSequence: activeMatch?.[1].sequence ?? null,
    };
  };
  const uniformExecutionProofExact = (record, proof) => {
    const encoders = lifecycle.commandEncoderTraces.filter(
      (encoder) => encoder.commandEncoderId === proof.commandEncoderId,
    );
    const encoder = encoders.length === 1 ? encoders[0] : null;
    const submissions = encoder === null ? [] : lifecycle.queueSubmissions.filter(
      (submission) => exactArray(submission.commandBufferIds, [encoder.commandBufferId]),
    );
    const renderPasses = lifecycle.renderPassTraces.filter(
      (trace) => trace.encoderId === proof.passEncoderId,
    );
    const computePasses = lifecycle.computePassTraces.filter(
      (trace) => trace.encoderId === proof.passEncoderId,
    );
    const passTrace = renderPasses[0] ?? computePasses[0] ?? null;
    const directReplay = passTrace === null ? null : activeBindGroupAtUse(
      passTrace.events,
      proof.bindGroupId,
      proof.useSequence,
      proof.useKind === 'compute-dispatch'
        ? (event) => event.method === 'dispatchWorkgroups'
          || event.method === 'dispatchWorkgroupsIndirect'
        : (event) => /^draw/.test(event.method),
    );
    const directReplayExact = directReplay !== null
      && directReplay.useMatchCount === 1
      && directReplay.activeMatchCount === 1
      && proof.bindGroupIndex === directReplay.bindGroupIndex
      && proof.activeSetBindGroupSequence === directReplay.activeSetBindGroupSequence
      && proof.activeTargetBindGroupMatchCount === directReplay.activeMatchCount
      && proof.setBindGroupSequence === directReplay.activeSetBindGroupSequence;
    const bundleMatches = lifecycle.renderBundleTraces.filter(
      (bundle) => bundle.bundleId === proof.bundleId,
    );
    const bundle = bundleMatches.length === 1 ? bundleMatches[0] : null;
    const bundleReplay = bundle === null ? null : activeBindGroupAtUse(
      bundle.events,
      proof.bindGroupId,
      proof.bundleDrawSequence,
      (event) => /^draw/.test(event.method),
    );
    const executeMatches = passTrace?.events.filter((event) => (
      event.method === 'executeBundles'
        && event.sequence === proof.useSequence
        && event.bundleIds?.includes(proof.bundleId)
    )) ?? [];
    const bundleReplayExact = bundleReplay !== null
      && bundleReplay.useMatchCount === 1
      && bundleReplay.activeMatchCount === 1
      && proof.bundleMatchCount === bundleMatches.length
      && bundleMatches.length === 1
      && proof.bindGroupIndex === bundleReplay.bindGroupIndex
      && proof.activeSetBindGroupSequence === bundleReplay.activeSetBindGroupSequence
      && proof.activeTargetBindGroupMatchCount === bundleReplay.activeMatchCount
      && proof.bundleSetBindGroupSequence === bundleReplay.activeSetBindGroupSequence
      && proof.bundleFinishSequence === bundle.finishSequence
      && proof.bundleDrawSequence < bundle.finishSequence
      && bundle.finishSequence < proof.useSequence
      && executeMatches.length === 1;
    const directUse = proof.useKind === 'compute-dispatch'
      ? computePasses.length === 1
        && renderPasses.length === 0
        && directReplayExact
      : proof.useKind === 'render-pass-draw'
        ? renderPasses.length === 1
          && computePasses.length === 0
          && directReplayExact
        : proof.useKind === 'executed-render-bundle'
          ? renderPasses.length === 1
            && computePasses.length === 0
            && bundleReplayExact
          : false;
    return proof.pass === true
      && encoders.length === 1
      && proof.commandEncoderMatchCount === encoders.length
      && submissions.length === 1
      && proof.submissionMatchCount === submissions.length
      && passTrace !== null
      && passTrace.capturePhase === record.capturePhase
      && passTrace.commandEncoderId === proof.commandEncoderId
      && proof.commandBufferId === encoder.commandBufferId
      && proof.commandEncoderFinishSequence === encoder.finishSequence
      && proof.submitSequence === submissions[0].sequence
      && submissions[0].capturePhase === record.capturePhase
      && record.sequence < proof.useSequence
      && proof.useSequence < encoder.finishSequence
      && encoder.finishSequence < submissions[0].sequence
      && directUse;
  };
  const expectedUniformGrammarExact = ledger.uniformWriteGrammar?.length
    === uniformLedgerRecords.length
    && ledger.uniformWriteGrammar.every((grammar, index) => {
      const record = uniformLedgerRecords[index];
      const creations = bufferCreations.filter(
        (candidate) => candidate.resourceId === record.bufferId,
      );
      const creation = creations.length === 1 ? creations[0] : null;
      const proofs = record.destinationOwner.uniformBindingProofs;
      const binding = proofs.length === 1 ? proofs[0] : null;
      const boundStart = binding?.bufferOffset ?? null;
      const boundByteLength = binding === null || creation === null
        ? null
        : binding.bufferSize ?? creation.size - binding.bufferOffset;
      const boundEnd = boundStart === null || boundByteLength === null
        ? null
        : boundStart + boundByteLength;
      const sourceRange = {
        start: record.selectedSourceByteOffset,
        byteLength: record.selectedSourceByteLength,
        end: record.selectedSourceByteOffset + record.selectedSourceByteLength,
      };
      const destinationRange = {
        start: record.bufferOffset,
        byteLength: record.selectedSourceByteLength,
        end: record.bufferOffset + record.selectedSourceByteLength,
      };
      const rangePositive = Number.isInteger(sourceRange.byteLength)
        && sourceRange.byteLength > 0
        && destinationRange.byteLength === sourceRange.byteLength;
      const rangeAligned = [
        sourceRange.start,
        sourceRange.byteLength,
        destinationRange.start,
        destinationRange.byteLength,
      ].every((value) => Number.isInteger(value) && value % 4 === 0);
      const sourceRangeWithinData = Number.isInteger(record.sourceByteLength)
        && sourceRange.start >= 0
        && sourceRange.end <= record.sourceByteLength;
      const destinationRangeWithinCreation = Number.isInteger(creation?.size)
        && destinationRange.start >= 0
        && destinationRange.end <= creation.size;
      const destinationRangeWithinBinding = Number.isInteger(boundStart)
        && Number.isInteger(boundByteLength)
        && boundByteLength > 0
        && destinationRange.start >= boundStart
        && destinationRange.end <= boundEnd;
      const sourceDestinationOffsetExact = Number.isInteger(boundStart)
        && sourceRange.start === destinationRange.start - boundStart;
      const samePhaseDestinationWriteCount = records.filter(
        (candidate) => candidate.capturePhase === record.capturePhase
          && candidate.bufferId === record.bufferId,
      ).length;
      const executionExact = record.category !== 'renderer-initialization'
        && grammar.executionProofs?.length === 1
        && uniformExecutionProofExact(record, grammar.executionProofs[0]);
      return grammar.pass === true
        && grammar.sequence === record.sequence
        && grammar.capturePhase === record.capturePhase
        && grammar.category === record.category
        && grammar.bufferId === record.bufferId
        && grammar.creationMatchCount === creations.length
        && creations.length === 1
        && exactValue(grammar.creation, creation)
        && exactValue(grammar.binding, binding)
        && grammar.alignmentBytes === 4
        && exactValue(grammar.sourceRange, sourceRange)
        && exactValue(grammar.destinationRange, destinationRange)
        && exactValue(grammar.boundRange, {
          start: boundStart,
          byteLength: boundByteLength,
          end: boundEnd,
        })
        && grammar.rangePositive === rangePositive
        && rangePositive
        && grammar.rangeAligned === rangeAligned
        && rangeAligned
        && grammar.sourceRangeWithinData === sourceRangeWithinData
        && sourceRangeWithinData
        && grammar.destinationRangeWithinCreation === destinationRangeWithinCreation
        && destinationRangeWithinCreation
        && grammar.destinationRangeWithinBinding === destinationRangeWithinBinding
        && destinationRangeWithinBinding
        && grammar.sourceDestinationOffsetExact === sourceDestinationOffsetExact
        && sourceDestinationOffsetExact
        && grammar.boundByteLength === boundByteLength
        && grammar.samePhaseDestinationWriteCount === samePhaseDestinationWriteCount
        && samePhaseDestinationWriteCount >= 1
        && grammar.initializationOnly === false
        && creation.usage === 72
        && binding !== null
        && binding.hasDynamicOffset === false
        && executionExact;
    });
  const uniformGrammarBySequence = new Map(
    (ledger.uniformWriteGrammar ?? []).map((grammar) => [grammar.sequence, grammar]),
  );
  const expectedUniformGroupMap = new Map();
  for (const record of uniformLedgerRecords) {
    const groupId = `${record.capturePhase}\u0000${record.bufferId}`;
    const group = expectedUniformGroupMap.get(groupId) ?? [];
    group.push(record);
    expectedUniformGroupMap.set(groupId, group);
  }
  const expectedUniformGroupsExact = ledger.uniformWriteGroups?.length
    === expectedUniformGroupMap.size
    && ledger.uniformWriteGroups.every((group, index) => {
      const expectedEntry = [...expectedUniformGroupMap][index];
      if (expectedEntry === undefined) return false;
      const [groupId, groupLedgerRecords] = expectedEntry;
      const groupGrammar = groupLedgerRecords.map(
        (record) => uniformGrammarBySequence.get(record.sequence),
      );
      if (groupGrammar.some((record) => record === undefined)) return false;
      const [first] = groupGrammar;
      const orderedDestinationRanges = groupGrammar.map((record) => ({
        sequence: record.sequence,
        ...record.destinationRange,
      })).sort((left, right) => left.start - right.start || left.sequence - right.sequence);
      let furthestDestinationEnd = -1;
      let observedDestinationOverlapCount = 0;
      for (const range of orderedDestinationRanges) {
        if (range.start < furthestDestinationEnd) observedDestinationOverlapCount += 1;
        furthestDestinationEnd = Math.max(furthestDestinationEnd, range.end);
      }
      const writeSequences = groupGrammar.map((record) => record.sequence);
      const writeSequencesStrictlyIncreasing = writeSequences.every(
        (sequence, sequenceIndex) => sequenceIndex === 0
          || sequence > writeSequences[sequenceIndex - 1],
      );
      const sameCreationAndBinding = groupGrammar.every((record) => (
        exactValue(record.creation, first.creation)
          && exactValue(record.binding, first.binding)
      ));
      return group.pass === true
        && group.groupId === groupId
        && group.capturePhase === first.capturePhase
        && group.category === first.category
        && group.bufferId === first.bufferId
        && exactValue(group.creation, first.creation)
        && exactValue(group.binding, first.binding)
        && exactValue(group.boundRange, first.boundRange)
        && group.writeCount === groupGrammar.length
        && group.writeCount >= 1
        && exactArray(group.writeSequences, writeSequences)
        && group.writeSequencesStrictlyIncreasing === writeSequencesStrictlyIncreasing
        && writeSequencesStrictlyIncreasing
        && group.sameCreationAndBinding === sameCreationAndBinding
        && sameCreationAndBinding
        && group.overlapPolicy === 'observed-not-prescribed'
        && group.observedDestinationOverlapCount === observedDestinationOverlapCount
        && exactValue(group.orderedDestinationRanges, orderedDestinationRanges)
        && group.everyWriteExecuted === true
        && groupGrammar.every((record) => record.pass === true
          && record.executionProofs.length === 1)
        && exactValue(group.records, groupGrammar);
    })
    && new Set((ledger.uniformWriteGroups ?? []).map((group) => group.groupId)).size
      === expectedUniformGroupMap.size
    && (ledger.uniformWriteGroups ?? []).reduce(
      (count, group) => count + group.writeCount,
      0,
    ) === uniformLedgerRecords.length;
  if (ledger.schemaVersion !== 1
    || ledger.kind !== 'immediate-aif-phase0-queue-write-buffer-ledger'
    || ledger.pass !== true
    || ledger.callCount !== rawCalls?.length
    || ledger.unclassifiedCount !== 0
    || ledger.unclassified?.length !== 0
    || !exactArray(ledger.categoryOrder, QUEUE_WRITE_BUFFER_CATEGORY_ORDER)
    || !exactValue(ledger.expectedSpecializedCounts, expectedSpecializedCounts)
    || !exactValue(ledger.specializedCounts, expectedSpecializedCounts)
    || ledger.specializedBindingsExact !== true
    || ledger.destinationPolicyExact !== true
    || ledger.liveAttributeWriteGrammarExact !== true
    || !exactValue(ledger.liveAttributeWriteGrammar, expectedLiveGrammar)
    || !expectedLiveGrammar.every((record) => record.pass)
    || ledger.uniformWriteGrammarExact !== true
    || !expectedUniformGrammarExact
    || ledger.uniformWriteGroupCount !== expectedUniformGroupMap.size
    || ledger.uniformWriteGroupsExact !== true
    || !expectedUniformGroupsExact
    || uniformGrammarBySequence.size !== uniformLedgerRecords.length
    || ledger.mergedGeometryWriteCount !== 0
    || ledger.rendererInitializationWriteCount !== 0
    || ledger.genericUniformWriteCount !== records.filter(
      (record) => uniformOnlyCategories.has(record.category),
    ).length
    || !rawExact
    || !partitionsExact
    || !sourcesExact
    || !policyExact
    || !exactArray(sourceSequences, ledgerSpecializedSequences)
    || new Set(records.map((record) => record.sequence)).size !== records.length
    || !records.every((record) => record.category === queueWriteBufferCategory(
      record.capturePhase,
    )
      && ownerProofExact(record)
      && record.destinationOwner.resourceId === record.bufferId
      && Number.isInteger(record.destinationOwner.createSequence)
      && record.destinationOwner.createSequence < record.sequence
      && record.destinationOwner.resourceClass === 'persistent-or-upload-buffer'
      && Number.isInteger(record.sourceElementByteSize)
      && record.sourceElementByteSize > 0
      && Number.isInteger(record.sourceElementCount)
      && record.selectedSourceElementOffset === record.dataOffset
      && record.selectedSourceElementLength
        === (record.size ?? record.sourceElementCount - record.dataOffset)
      && record.selectedSourceByteOffset
        === record.selectedSourceElementOffset * record.sourceElementByteSize
      && record.selectedSourceByteLength
        === record.selectedSourceElementLength * record.sourceElementByteSize
      && /^[0-9a-f]{64}$/.test(record.selectedSourceSha256 ?? '')
    )) {
    throw new Error('Queue write-buffer ledger evidence is malformed.');
  }
  return ledger;
}

function validateBufferMapLifecycleEvidence(evidence, lifecycle, staging) {
  requireObject(evidence, 'Buffer map lifecycle');
  const rawEvents = lifecycle.bufferMapEvents;
  const records = evidence.records;
  const representedEvents = records?.flatMap((record) => record.mapEvents) ?? [];
  const stagingById = new Map(staging.transferLedger.map(
    (record) => [record.stagingBufferId, record],
  ));
  const recordsExact = records?.every((record) => {
    const persistent = record.resourceClass === 'persistent-or-upload-buffer';
    if (persistent) {
      const noMapUsage = (record.usage & 0x0003) === 0;
      const mappedGrammar = record.mappedAtCreation === true
        && record.mapEvents.length === 2
        && record.mapEvents[0].method === 'getMappedRange'
        && record.mapEvents[0].offset === 0
        && (record.mapEvents[0].size === null || record.mapEvents[0].size === record.size)
        && record.mapEvents[1].method === 'unmap'
        && record.mapEvents.every((event) => event.capturePhase === record.capturePhase)
        && record.createSequence < record.mapEvents[0].sequence
        && record.mapEvents[0].sequence < record.mapEvents[1].sequence
        && (record.firstUse === null
          || record.mapEvents[1].sequence < record.firstUse.sequence);
      const unmappedGrammar = record.mappedAtCreation === false
        && record.mapEvents.length === 0;
      return record.pass === true && noMapUsage
        && (mappedGrammar || unmappedGrammar)
        && record.persistentMappedGrammar === mappedGrammar
        && record.persistentUnmappedGrammar === unmappedGrammar;
    }
    const stagingRecord = stagingById.get(record.resourceId);
    const stagingGrammar = record.resourceClass === 'readback-staging'
      && stagingRecord?.pass === true
      && record.mappedAtCreation === false
      && record.mapEvents.length === 2
      && exactValue(record.mapEvents, stagingRecord.mapEvents)
      && record.mapEvents[0].method === 'mapAsync'
      && record.mapEvents[1].method === 'getMappedRange'
      && stagingRecord.transfer.sequence < record.mapEvents[0].sequence
      && record.firstUse?.sequence === stagingRecord.transfer.sequence;
    return record.pass === true
      && record.stagingGrammar === stagingGrammar
      && stagingGrammar;
  }) === true;
  if (evidence.schemaVersion !== 1
    || evidence.kind !== 'immediate-aif-phase0-buffer-map-lifecycle'
    || evidence.pass !== true
    || !Array.isArray(rawEvents)
    || !Array.isArray(records)
    || evidence.bufferCount !== records.length
    || evidence.bufferCount <= 825
    || evidence.stagingBufferCount !== 825
    || evidence.persistentBufferCount !== records.length - 825
    || evidence.persistentBufferCount <= 0
    || evidence.rawMapEventCount !== rawEvents.length
    || evidence.representedMapEventCount !== representedEvents.length
    || evidence.unmatchedEventCount !== 0
    || evidence.unmatchedEvents?.length !== 0
    || !recordsExact
    || new Set(records.map((record) => record.resourceId)).size !== records.length
    || !exactArray(
      representedEvents.map((event) => event.sequence).sort((a, b) => a - b),
      rawEvents.map((event) => event.sequence).sort((a, b) => a - b),
    )) {
    throw new Error('Buffer map lifecycle evidence is malformed.');
  }
  return evidence;
}

function validateEncoderStateEvidence(evidence, lifecycle) {
  requireObject(evidence, 'Encoder state exclusion');
  const bundleTraces = lifecycle.renderBundleTraces;
  const renderPassTraces = lifecycle.renderPassTraces;
  const computePassTraces = lifecycle.computePassTraces;
  const bundleImmediateTraces = bundleTraces?.map((trace) => ({
    encoderId: trace.encoderId,
    capturePhase: trace.capturePhase,
    bundleId: trace.bundleId,
    callCount: trace.events.filter((event) => event.method === 'setImmediates').length,
    calls: trace.events.filter((event) => event.method === 'setImmediates'),
  }));
  const renderPassImmediateCalls = renderPassTraces?.flatMap(
    (trace) => trace.events.filter((event) => event.method === 'setImmediates'),
  );
  const computeImmediateCalls = computePassTraces?.flatMap(
    (trace) => trace.events.filter((event) => event.method === 'setImmediates'),
  );
  const allBindGroups = [...(bundleTraces ?? []), ...(renderPassTraces ?? []),
    ...(computePassTraces ?? [])].flatMap(
    (trace) => trace.events.filter((event) => event.method === 'setBindGroup'),
  );
  const dynamicCalls = allBindGroups.filter((event) => event.dynamicOffsets !== null
    || event.dynamicOffsetStart !== null
    || event.dynamicOffsetLength !== null
    || event.selectedDynamicOffsets !== null);
  const occlusionCalls = renderPassTraces?.flatMap((trace) => trace.events.filter(
    (event) => event.method === 'beginOcclusionQuery'
      || event.method === 'endOcclusionQuery',
  ));
  if (evidence.schemaVersion !== 1
    || evidence.kind !== 'immediate-aif-phase0-encoder-state-exclusion-evidence'
    || evidence.pass !== true
    || evidence.expectedRenderBundleEncoderCount !== 27
    || evidence.renderBundleEncoderCount !== 27
    || bundleTraces?.length !== 27
    || evidence.expectedImmediateRenderBundleCount !== 21
    || evidence.immediateRenderBundleCount !== 21
    || bundleImmediateTraces.filter((record) => record.callCount === 32).length !== 21
    || bundleImmediateTraces.filter((record) => record.callCount === 0).length !== 6
    || evidence.expectedRenderBundleSetImmediatesCount !== 672
    || evidence.renderBundleSetImmediatesCount !== 672
    || !exactValue(evidence.renderBundleImmediateTraces, bundleImmediateTraces)
    || evidence.renderPassSetImmediatesCount !== 0
    || evidence.renderPassSetImmediatesCalls?.length !== 0
    || renderPassImmediateCalls.length !== 0
    || evidence.computePassSetImmediatesCount !== 0
    || evidence.computePassSetImmediatesCalls?.length !== 0
    || computeImmediateCalls.length !== 0
    || evidence.setBindGroupCount !== allBindGroups.length
    || evidence.dynamicOffsetCallCount !== 0
    || evidence.dynamicOffsetCalls?.length !== 0
    || dynamicCalls.length !== 0
    || evidence.occlusionQueryCallCount !== 0
    || evidence.occlusionQueryCalls?.length !== 0
    || occlusionCalls.length !== 0) {
    throw new Error('Encoder state exclusion evidence is malformed.');
  }
  return evidence;
}

function validateComputeImmediateStateEvidence(evidence, shaders, pipelines) {
  requireObject(evidence, 'Compute immediate-state exclusion');
  const computeModules = shaders.modules?.filter((module) => /@compute\b/.test(
    module.code ?? '',
  ));
  const computePipelines = pipelines.compute;
  const pipelineLayouts = pipelines.layouts;
  const expectedLayoutBindings = computePipelines?.map((pipeline) => {
    const matches = pipelineLayouts?.filter(
      (layout) => layout.layoutId === pipeline.layoutId,
    ) ?? [];
    const layout = matches.length === 1 ? matches[0] : null;
    return {
      pipelineId: pipeline.pipelineId,
      pipelineLayoutId: pipeline.layoutId,
      layoutMatchCount: matches.length,
      hasOwnImmediateSize: layout?.hasOwnImmediateSize ?? false,
      immediateSize: layout?.immediateSize ?? null,
      pass: matches.length === 1
        && layout.hasOwnImmediateSize === true
        && layout.immediateSize === 0,
    };
  });
  const expectedLayouts = [...new Map((expectedLayoutBindings ?? [])
    .filter((binding) => binding.layoutMatchCount === 1)
    .map((binding) => [
      binding.pipelineLayoutId,
      pipelineLayouts.find((layout) => layout.layoutId === binding.pipelineLayoutId),
    ])).values()];
  if (evidence.schemaVersion !== 1
    || evidence.kind !== 'immediate-aif-phase0-compute-immediate-state-exclusion'
    || evidence.pass !== true
    || !Array.isArray(computeModules)
    || computeModules.length === 0
    || evidence.shaderModuleCount !== computeModules.length
    || evidence.shaderModules?.length !== computeModules.length
    || !computeModules.every((module, index) => (
      evidence.shaderModules[index].moduleId === module.moduleId
        && evidence.shaderModules[index].requiresImmediateAddressSpaceCount === 0
        && evidence.shaderModules[index].immediateVariableDeclarationCount === 0
        && !/requires\s+immediate_address_space\s*;/.test(module.code ?? '')
        && !/var\s*<\s*immediate\s*>/.test(module.code ?? '')
    ))
    || !Array.isArray(computePipelines)
    || computePipelines.length === 0
    || !Array.isArray(pipelineLayouts)
    || evidence.pipelineCount !== computePipelines.length
    || evidence.pipelines?.length !== computePipelines.length
    || !computePipelines.every((pipeline, index) => (
      evidence.pipelines[index].pipelineId === pipeline.pipelineId
        && (pipeline.immediateSize === null || pipeline.immediateSize === 0)
        && (pipeline.descriptor?.immediateSize === null
          || pipeline.descriptor?.immediateSize === 0)
        && evidence.pipelines[index].immediateSize === pipeline.immediateSize
        && evidence.pipelines[index].descriptorImmediateSize
          === pipeline.descriptor?.immediateSize
    ))
    || evidence.pipelineLayoutBindingCount !== computePipelines.length
    || evidence.pipelineLayoutBindings?.length !== computePipelines.length
    || !exactValue(evidence.pipelineLayoutBindings, expectedLayoutBindings)
    || !expectedLayoutBindings.every((binding) => binding.pass)
    || evidence.pipelineLayoutCount !== expectedLayouts.length
    || !exactValue(evidence.pipelineLayouts, expectedLayouts)
    || !expectedLayouts.every((layout) => (
      layout.hasOwnImmediateSize === true && layout.immediateSize === 0
    ))) {
    throw new Error('Compute immediate-state exclusion evidence is malformed.');
  }
  return evidence;
}

function validateTextureInventoryEvidence(evidence, resources, lifecycle) {
  requireObject(evidence, 'Texture inventory');
  const creations = resources.gpuCreations?.filter(
    (resource) => resource.method === 'createTexture',
  );
  const views = resources.textureViews;
  const bindGroupViewIds = new Set(resources.gpuCreations?.filter(
    (resource) => resource.method === 'createBindGroup',
  ).flatMap((resource) => resource.entries ?? []).map(
    (entry) => entry.resourceId,
  ).filter((resourceId) => typeof resourceId === 'string'));
  const attachmentViewIdsByTexture = new Map();
  for (const trace of lifecycle.renderPassTraces ?? []) {
    for (const attachment of trace.colorAttachments ?? []) {
      if (typeof attachment?.textureId !== 'string'
        || typeof attachment?.viewId !== 'string') continue;
      const ids = attachmentViewIdsByTexture.get(attachment.textureId) ?? new Set();
      ids.add(attachment.viewId);
      attachmentViewIdsByTexture.set(attachment.textureId, ids);
    }
    const depth = trace.depthStencilAttachment;
    if (typeof depth?.textureId === 'string' && typeof depth?.viewId === 'string') {
      const ids = attachmentViewIdsByTexture.get(depth.textureId) ?? new Set();
      ids.add(depth.viewId);
      attachmentViewIdsByTexture.set(depth.textureId, ids);
    }
  }
  if (!Array.isArray(creations)
    || !Array.isArray(views)
    || evidence.textureCount !== creations.length
    || evidence.textureInventory?.length !== creations.length
    || evidence.declaredTargetTextureCount !== 5
    || evidence.declaredTargetTextureIds?.length !== 5
    || new Set(evidence.declaredTargetTextureIds).size !== 5
    || evidence.targetTextureInventoryExact !== true
    || evidence.internalTextureCount !== creations.length - 5
    || !evidence.textureInventory.every((record) => {
      const creation = creations.find((candidate) => candidate.resourceId === record.textureId);
      const childViews = views.filter((view) => view.textureId === record.textureId);
      const expectedClassification = evidence.declaredTargetTextureIds.includes(record.textureId)
        ? 'declared-render-target'
        : 'renderer-internal';
      const expectedBoundViewIds = childViews.map((view) => view.viewId).filter(
        (viewId) => bindGroupViewIds.has(viewId),
      );
      const expectedAttachmentViewIds = childViews.map((view) => view.viewId).filter(
        (viewId) => attachmentViewIdsByTexture.get(record.textureId)?.has(viewId),
      );
      return record.pass === true
        && creation !== undefined
        && record.createSequence === creation.sequence
        && record.capturePhase === creation.capturePhase
        && record.classification === expectedClassification
        && exactArray(record.viewIds, childViews.map((view) => view.viewId))
        && exactArray(record.boundViewIds, expectedBoundViewIds)
        && exactArray(record.attachmentViewIds, expectedAttachmentViewIds)
        && new Set(record.viewIds).size === record.viewIds.length
        && childViews.length > 0
        && (expectedClassification === 'declared-render-target'
          ? expectedAttachmentViewIds.length === childViews.length
            && expectedBoundViewIds.length === 0
          : expectedBoundViewIds.length === childViews.length
            && expectedAttachmentViewIds.length === 0);
    })) {
    throw new Error('Texture inventory evidence is malformed.');
  }
  return evidence;
}

function validateGlobalCommandLedgerEvidence(staging, lifecycle) {
  requireObject(staging, 'Readback staging lifecycle');
  requireObject(lifecycle, 'Command encoder lifecycle');
  const encoders = lifecycle.commandEncoderTraces;
  const submissions = lifecycle.queueSubmissions;
  const operations = lifecycle.commandEncoderTransferCalls;
  const renderPasses = lifecycle.renderPassTraces;
  const computePasses = lifecycle.computePassTraces;
  const encodedCommandBufferIds = encoders?.map((record) => record.commandBufferId);
  const submittedCommandBufferIds = submissions?.flatMap(
    (record) => record.commandBufferIds,
  );
  const renderPassEncoderIds = encoders?.flatMap(
    (record) => record.renderPassEncoderIds,
  );
  const computePassEncoderIds = encoders?.flatMap(
    (record) => record.computePassEncoderIds,
  );
  const operationSequences = encoders?.flatMap(
    (record) => record.transferSequences,
  );
  const operationDestinationId = (operation) => (
    operation.method === 'copyBufferToBuffer'
      ? operation.destinationBufferId
      : operation.method === 'copyTextureToBuffer'
        ? operation.destination?.bufferId
        : null
  );
  const expectedBufferSourceSemantics = [
    ...['matrix', 'bounds', 'objectBucket', 'bucketBase', 'bucketCapacity',
      'cullOrder', 'visibleIds', 'overflow'].map((semantic) => `shared.${semantic}`),
    ...IMMEDIATE_AIF_PHASE0_LANES.map((lane) => `command.${lane}`),
    ...['bucketBase', 'normal', 'position', 'uv', 'index'].map(
      (semantic) => `merged-geometry.${semantic}`,
    ),
  ];
  const expectedTextureSourceSemantics = [
    'production.color',
    'production.depth',
    'address.color',
    'objectId.color',
    'objectId.depth',
  ];
  const sourceInventoryExact = staging.sourceInventory?.schemaVersion === 1
    && staging.sourceInventory.kind === 'immediate-aif-phase0-readback-source-inventory'
    && staging.sourceInventory.pass === true
    && staging.sourceInventory.bufferSourceCount === 16
    && staging.sourceInventory.textureSourceCount === 5
    && exactArray(
      staging.sourceInventory.bufferSources?.map((record) => record.semantic),
      expectedBufferSourceSemantics,
    )
    && exactArray(
      staging.sourceInventory.textureSources?.map((record) => record.semantic),
      expectedTextureSourceSemantics,
    )
    && new Set(staging.sourceInventory.bufferSources.map(
      (record) => record.resourceId,
    )).size === 16
    && new Set(staging.sourceInventory.textureSources.map(
      (record) => record.resourceId,
    )).size === 5;
  const encoderClasses = encoders?.map((record) => (
    record.renderPassEncoderIds.length === 1
      && record.computePassEncoderIds.length === 0
      && record.transferSequences.length === 0
      ? 'renderOnly'
      : record.renderPassEncoderIds.length === 0
        && record.computePassEncoderIds.length === 1
        && record.transferSequences.length === 0
        ? 'computeOnly'
        : record.renderPassEncoderIds.length === 0
          && record.computePassEncoderIds.length === 0
          && record.transferSequences.length === 1
          ? 'transferOnly'
          : 'mixed-or-empty'
  ));
  const rawTopologyExact = encoders?.length === 1_629
    && submissions?.length === 1_629
    && operations?.length === 825
    && renderPasses?.length === 798
    && computePasses?.length === 6
    && operations.every((record) => (
      (record.method === 'copyBufferToBuffer' || record.method === 'copyTextureToBuffer')
        && Number.isInteger(record.sequence)
        && typeof record.capturePhase === 'string'
        && record.capturePhase.length > 0
        && typeof record.commandEncoderId === 'string'
    ))
    && encoders.every((record) => (
      typeof record.capturePhase === 'string'
        && record.capturePhase.length > 0
        && Number.isInteger(record.sequence)
        && record.finishCallCount === 1
        && Number.isInteger(record.finishSequence)
        && record.sequence < record.finishSequence
        && typeof record.commandBufferId === 'string'
    ))
    && encoderClasses.filter((value) => value === 'renderOnly').length === 798
    && encoderClasses.filter((value) => value === 'computeOnly').length === 6
    && encoderClasses.filter((value) => value === 'transferOnly').length === 825
    && !encoderClasses.includes('mixed-or-empty')
    && submissions.every((record) => (
      typeof record.capturePhase === 'string'
        && record.capturePhase.length > 0
        && Number.isInteger(record.sequence)
        && record.commandBufferIds?.length === 1
    ))
    && new Set(encodedCommandBufferIds).size === 1_629
    && new Set(submittedCommandBufferIds).size === 1_629
    && exactArray(
      [...encodedCommandBufferIds].sort(),
      [...submittedCommandBufferIds].sort(),
    )
    && encoders.every((encoder) => submissions.filter((submission) => (
      exactArray(submission.commandBufferIds, [encoder.commandBufferId])
        && submission.sequence > encoder.finishSequence
    )).length === 1)
    && new Set(renderPassEncoderIds).size === 798
    && exactArray(
      [...renderPassEncoderIds].sort(),
      [...renderPasses.map((record) => record.encoderId)].sort(),
    )
    && new Set(computePassEncoderIds).size === 6
    && exactArray(
      [...computePassEncoderIds].sort(),
      [...computePasses.map((record) => record.encoderId)].sort(),
    )
    && new Set(operationSequences).size === 825
    && exactArray(
      [...operationSequences].sort((left, right) => left - right),
      [...operations.map((record) => record.sequence)].sort((left, right) => left - right),
    )
    && lifecycle.queueWriteTextureCalls?.length === 0
    && lifecycle.queueCopyExternalImageToTextureCalls?.length === 0;
  const ledgerRawExact = staging.transferLedger?.length === 825
    && staging.transferLedger.every((record) => {
      const matchingOperations = operations.filter(
        (operation) => operation.sequence === record.transfer?.sequence,
      );
      const matchingEncoders = encoders.filter(
        (encoder) => encoder.commandEncoderId === record.commandEncoder?.commandEncoderId,
      );
      const matchingSubmissions = submissions.filter((submission) => (
        exactArray(submission.commandBufferIds, [record.commandEncoder?.commandBufferId])
      ));
      const operation = matchingOperations[0];
      const encoder = matchingEncoders[0];
      const submission = matchingSubmissions[0];
      const mapAsync = record.mapEvents?.[0];
      const getMappedRange = record.mapEvents?.[1];
      const transferShapeExact = operation?.method === 'copyBufferToBuffer'
        ? typeof operation.sourceBufferId === 'string'
          && operation.sourceOffset >= 0
          && operation.destinationBufferId === record.stagingBufferId
          && operation.destinationOffset === 0
          && operation.size === record.stagingSize
        : operation?.method === 'copyTextureToBuffer'
          && typeof operation.source?.textureId === 'string'
          && operation.source.mipLevel >= 0
          && Number.isInteger(operation.source.origin?.x)
          && Number.isInteger(operation.source.origin?.y)
          && Number.isInteger(operation.source.origin?.z)
          && operation.source.aspect === 'all'
          && operation.destination?.bufferId === record.stagingBufferId
          && operation.destination.offset === 0
          && Number.isInteger(operation.destination.bytesPerRow)
          && operation.destination.bytesPerRow > 0
          && operation.destination.bytesPerRow % 256 === 0
          && operation.destination.rowsPerImage === null
          && Number.isInteger(operation.copySize?.width)
          && operation.copySize.width > 0
          && Number.isInteger(operation.copySize?.height)
          && operation.copySize.height > 0
          && operation.copySize.depthOrArrayLayers === 1
          && record.stagingSize === (operation.copySize.height - 1)
            * operation.destination.bytesPerRow + operation.copySize.width * 4;
      const expectedMappedSize = operation?.method === 'copyBufferToBuffer'
        ? record.stagingSize
        : null;
      const operationSourceId = operation?.method === 'copyBufferToBuffer'
        ? operation.sourceBufferId
        : operation?.source?.textureId ?? null;
      const expectedSourceKind = operation?.method === 'copyBufferToBuffer'
        ? 'buffer'
        : 'texture';
      return record.pass === true
        && typeof record.capturePhase === 'string'
        && record.capturePhase.length > 0
        && typeof record.stagingBufferId === 'string'
        && Number.isInteger(record.stagingSize)
        && record.stagingSize > 0
        && record.stagingUsage === 9
        && record.stagingMappedAtCreation === false
        && matchingOperations.length === 1
        && matchingEncoders.length === 1
        && matchingSubmissions.length === 1
        && record.sourceBinding?.resourceKind === expectedSourceKind
        && record.sourceBinding.resourceId === operationSourceId
        && (expectedSourceKind === 'buffer'
          ? staging.sourceInventory.bufferSources
          : staging.sourceInventory.textureSources).some((source) => (
          exactValue(source, record.sourceBinding)
        ))
        && operation.capturePhase === record.capturePhase
        && operation.commandEncoderId === encoder.commandEncoderId
        && encoder.capturePhase === record.capturePhase
        && encoder.renderPassEncoderIds.length === 0
        && encoder.computePassEncoderIds.length === 0
        && exactArray(encoder.transferSequences, [operation.sequence])
        && submission.capturePhase === record.capturePhase
        && record.transferShapeExact === true
        && transferShapeExact
        && record.mapEvents?.length === 2
        && mapAsync.method === 'mapAsync'
        && mapAsync.resourceId === record.stagingBufferId
        && mapAsync.mode === 1
        && mapAsync.offset === 0
        && mapAsync.size === expectedMappedSize
        && getMappedRange.method === 'getMappedRange'
        && getMappedRange.resourceId === record.stagingBufferId
        && getMappedRange.offset === 0
        && getMappedRange.size === expectedMappedSize
        && record.createSequence < encoder.sequence
        && encoder.sequence < operation.sequence
        && operation.sequence < encoder.finishSequence
        && encoder.finishSequence < submission.sequence
        && submission.sequence < mapAsync.sequence
        && mapAsync.sequence < getMappedRange.sequence
        && getMappedRange.sequence < record.destroySequence
        && record.destroyCapturePhase === record.capturePhase
        && record.destroyCallCount === 1;
    })
    && new Set(staging.transferLedger.map((record) => record.stagingBufferId)).size === 825
    && new Set(operations.map(operationDestinationId)).size === 825
    && exactArray(
      [...staging.transferLedger.map((record) => record.stagingBufferId)].sort(),
      [...operations.map(operationDestinationId)].sort(),
    );
  if (staging.schemaVersion !== 1
    || staging.kind !== 'immediate-aif-phase0-readback-staging-lifecycle'
    || staging.pass !== true
    || staging.createdCount !== 825
    || staging.expectedCreatedCount !== 825
    || staging.readbackPhaseCount !== 666
    || staging.expectedReadbackPhaseCount !== 666
    || staging.liveCount !== 0
    || staging.transferCount !== 825
    || staging.expectedTransferCount !== 825
    || staging.commandEncoderCount !== 1_629
    || staging.expectedCommandEncoderCount !== 1_629
    || staging.queueSubmissionCount !== 1_629
    || staging.expectedQueueSubmissionCount !== 1_629
    || staging.commandEncoderClassCounts?.renderOnly !== 798
    || staging.commandEncoderClassCounts.computeOnly !== 6
    || staging.commandEncoderClassCounts.transferOnly !== 825
    || staging.transferMethodCounts?.copyBufferToBuffer
      + staging.transferMethodCounts?.copyTextureToBuffer !== 825
    || staging.transferMethodCounts?.clearBuffer !== 0
    || staging.transferMethodCounts?.copyBufferToTexture !== 0
    || staging.transferMethodCounts?.copyTextureToTexture !== 0
    || staging.transferMethodCounts?.resolveQuerySet !== 0
    || staging.transferMethodCounts?.writeTimestamp !== 0
    || staging.globalQueueWriteTextureCount !== 0
    || staging.globalCopyExternalImageToTextureCount !== 0
    || staging.commandLedgerPass !== true
    || !sourceInventoryExact
    || !rawTopologyExact
    || !ledgerRawExact) {
    throw new Error('Global command/readback ledger evidence is malformed.');
  }
  return staging;
}

function validatePersistentResourceFreezeEvidence(freeze) {
  requireObject(freeze, 'Persistent resource freeze');
  const before = freeze.preflight;
  const after = freeze.postflight;
  const delta = freeze.prescribedPersistentDelta;
  const beforeRecords = before?.records;
  const afterRecords = after?.records;
  const beforeNonBundles = beforeRecords?.filter(
    (record) => record.resourceClass !== 'render-bundle',
  );
  const afterNonBundles = afterRecords?.filter(
    (record) => record.resourceClass !== 'render-bundle',
  );
  const beforeBundleIds = new Set(beforeRecords?.filter(
    (record) => record.resourceClass === 'render-bundle',
  ).map((record) => record.bundleId));
  const addedBundles = afterRecords?.filter(
    (record) => record.resourceClass === 'render-bundle'
      && !beforeBundleIds.has(record.bundleId),
  );
  if (freeze.schemaVersion !== 1
    || freeze.kind !== 'immediate-aif-phase0-persistent-resource-freeze-comparison'
    || freeze.pass !== true
    || !['v99', 'v20'].includes(freeze.scenarioId)
    || before?.schemaVersion !== 1
    || before.kind !== 'immediate-aif-phase0-persistent-resource-freeze'
    || before.pass !== true
    || before.scenarioId !== freeze.scenarioId
    || !Number.isInteger(before.markerSequence)
    || before.resourceCount !== beforeRecords?.length
    || beforeRecords.length === 0
    || after?.schemaVersion !== 1
    || after.kind !== 'immediate-aif-phase0-persistent-resource-freeze'
    || after.pass !== true
    || after.scenarioId !== freeze.scenarioId
    || !Number.isInteger(after.markerSequence)
    || after.markerSequence <= before.markerSequence
    || after.resourceCount !== afterRecords?.length
    || !beforeRecords.every((record) => record.sequence < before.markerSequence)
    || !afterRecords.every((record) => record.sequence < after.markerSequence)
    || JSON.stringify(beforeNonBundles) !== JSON.stringify(afterNonBundles)
    || addedBundles?.length !== 9
    || delta?.schemaVersion !== 1
    || delta.kind !== 'immediate-aif-phase0-prescribed-persistent-resource-delta'
    || delta.pass !== true
    || delta.scenarioId !== freeze.scenarioId
    || delta.nonBundlePersistentExact !== true
    || delta.exactRetainedBundles !== true
    || delta.expectedAddedBundleCount !== 9
    || delta.addedBundleCount !== 9
    || delta.addedBundleTopologyExact !== true
    || delta.diagnosticBundles?.length !== 6
    || delta.productionBundles?.length !== 3
    || !exactArray(delta.productionSchedules, ['S1', 'S2', 'canonical'])
    || JSON.stringify(delta.addedBundles) !== JSON.stringify(addedBundles)) {
    throw new Error('Persistent resource freeze evidence is malformed.');
  }
  return freeze;
}

export function validateImmediateAifPhase0PageResultShape(result) {
  requireObject(result, 'Phase 0 page result');
  if (result.schemaVersion !== 1
    || result.kind !== IMMEDIATE_AIF_PHASE0_PAGE_KIND
    || result.executionMode !== 'technical-canary'
    || result.analysisEligible !== false
    || result.efficacyAnalysisAllowed !== false
    || result.numericalDecision !== null
    || result.timingCaptured !== false) {
    throw new Error('Phase 0 page result crosses the correctness-only claim boundary.');
  }
  if (![
    'technical-canary-complete',
    'technical-canary-unsupported',
    'technical-canary-failed',
  ].includes(result.status)) {
    throw new Error('Phase 0 page result has an unknown status.');
  }
  if (result.status === 'technical-canary-complete') {
    if (result.cleanupFailure !== null) {
      throw new Error('A complete Phase 0 result cannot retain a cleanup failure.');
    }
    const frozenPlan = result.evidence?.plan;
    if (!exactValue(
      frozenPlan?.renderConfiguration,
      IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION,
    )
      || frozenPlan.renderConfiguration.camera.reversedDepth !== true) {
      throw new Error('The frozen Phase 0 render configuration is malformed.');
    }
    if (result.cleanup?.complete !== true
      || result.cleanup.deviceDestroyIntentional !== true
      || result.cleanup.terminalDeviceLoss?.reason !== 'destroyed'
      || !Number.isInteger(result.cleanup.cleanupStartSequence)
      || result.cleanup.globalFinalUse?.pass !== true
      || !Number.isInteger(result.cleanup.globalFinalUse.finalUseSequence)
      || result.cleanup.globalFinalUse.finalUseSequence >= result.cleanup.cleanupStartSequence
      || result.cleanup.persistentBufferCleanup?.pass !== true
      || result.cleanup.persistentBufferCleanup.terminalDeviceCoverage !== true
      || !Array.isArray(result.cleanup.persistentBufferCleanup.records)
      || result.cleanup.persistentBufferCleanup.records.length === 0
      || !result.cleanup.persistentBufferCleanup.records.every((record) => (
        record.pass === true
          && record.disposition === 'explicit-destroy-exactly-once'
          && record.destroyCallCount === 1
          && record.afterGlobalFinalUse === true
          && record.destroySequence > result.cleanup.cleanupStartSequence
          && /^phase0\/cleanup\//.test(record.destroyCapturePhase ?? '')
      ))
      || result.cleanup.persistentTextureCleanup?.pass !== true
      || result.cleanup.persistentTextureCleanup.terminalDeviceCoverage !== true
      || result.cleanup.persistentTextureCleanup.resourceCount < 5
      || result.cleanup.persistentTextureCleanup.explicitlyDestroyedCount
        !== result.cleanup.persistentTextureCleanup.resourceCount
      || result.cleanup.persistentTextureCleanup.declaredRenderTargetCount !== 5
      || !Array.isArray(result.cleanup.persistentTextureCleanup.allowedDestroyCapturePhases)
      || !result.cleanup.persistentTextureCleanup.allowedDestroyCapturePhases.every(
        (phase) => /^phase0\/cleanup\//.test(phase ?? ''),
      )
      || !Array.isArray(result.cleanup.persistentTextureCleanup.records)
      || !result.cleanup.persistentTextureCleanup.records.every((record) => (
        record.pass === true
          && record.disposition === 'explicit-destroy-exactly-once'
          && record.destroyCallCount === 1
          && record.afterGlobalFinalUse === true
          && record.destroySequence > result.cleanup.cleanupStartSequence
          && /^phase0\/cleanup\//.test(record.destroyCapturePhase ?? '')
      ))) {
      throw new Error('A complete Phase 0 result needs exact persistent-buffer cleanup.');
    }
    const challenge = result.evidence?.observationChallenge;
    if (challenge?.schemaVersion !== 1
      || challenge.kind !== 'immediate-aif-phase0-observation-challenge'
      || !IMMEDIATE_AIF_PHASE0_OBSERVATION_NONCE_PATTERN.test(challenge.nonce)
      || challenge.nonceEntropyBits !== 128
      || challenge.encoding !== IMMEDIATE_AIF_PHASE0_OBSERVATION_ENCODING
      || !exactArray(challenge.fieldNames, IMMEDIATE_AIF_PHASE0_OBSERVATION_FIELD_NAMES)
      || challenge.fieldCount !== IMMEDIATE_AIF_PHASE0_OBSERVATION_FIELD_NAMES.length
      || !Array.isArray(challenge.fieldCommitments)
      || challenge.fieldCommitments.length !== challenge.fieldCount
      || !Array.isArray(challenge.chains)
      || challenge.chainCount !== 390
      || challenge.chains.length !== challenge.chainCount
      || typeof challenge.sha256 !== 'string') {
      throw new Error('A complete Phase 0 result needs the exact nonce observation challenge.');
    }
    const outputWitnesses = result.evidence?.outputWitnesses;
    if (outputWitnesses?.schemaVersion !== 1
      || outputWitnesses.kind !== 'immediate-aif-phase0-output-byte-witnesses'
      || outputWitnesses.pass !== true
      || outputWitnesses.expectedChallengedCallbackCount !== 126
      || outputWitnesses.challengedCallbackCount !== 126
      || outputWitnesses.expectedObservationCount !== 384
      || outputWitnesses.observationCount !== 384
      || !Array.isArray(outputWitnesses.callbackReferences)
      || outputWitnesses.callbackReferences.length !== 378) {
      throw new Error('A complete Phase 0 result needs exact raw output witnesses.');
    }
    const resources = result.evidence?.resources;
    if (resources?.geometryFixtures?.schemaVersion !== 2
      || resources.geometryFixtures.pass !== true
      || resources.geometryFixtures.sourceManifest?.bucketCount !== 32
      || resources.geometryFixtures.realizationBindingExact !== true
      || resources.geometryFixtures.mergedProduction?.gpuRecords?.length !== 5
      || resources.addressDiagnosticPositions?.pass !== true
      || resources.queueWriteBufferLedger?.pass !== true
      || resources.bufferMapLifecycle?.pass !== true
      || resources.encoderState?.pass !== true
      || !exactArray(
        resources.geometryFixtures.mergedProduction.gpuRecords.map(
          (record) => record.semantic,
        ),
        ['bucketBase', 'normal', 'position', 'uv', 'index'],
      )
      || !resources.geometryFixtures.mergedProduction.gpuRecords.every((record) => (
        record.gpuResident === true
          && record.exact === true
          && record.gpuSha256 === record.cpuSha256
          && typeof record.gpuBufferId === 'string'
          && typeof record.rawWitnessId === 'string'
      ))
      || resources.geometryFixtures.mergedProduction.gpuRecords.find(
        (record) => record.semantic === 'uv',
      )?.boundByProductionBundle !== false
      || !['bucketBase', 'normal', 'position', 'index'].every((semantic) => (
        resources.geometryFixtures.mergedProduction.gpuRecords.find(
          (record) => record.semantic === semantic,
        )?.boundByProductionBundle === true
      ))
      || resources.commonResources?.pass !== true
      || !exactValue(
        resources.commonResources.expected,
        IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION,
      )
      || resources.commonResources.camera?.pass !== true
      || resources.commonResources.camera.reversedDepth !== true
      || !exactArray(
        resources.commonResources.camera.projectionMatrix,
        IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION.camera.projectionMatrix,
      )
      || !exactArray(
        resources.commonResources.camera.projectionMatrixInverse,
        IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION.camera.projectionMatrixInverse,
      )
      || resources.scheduleInstallations?.pass !== true
      || resources.scheduleInstallations.expectedInstallationCount !== 8
      || resources.scheduleInstallations.installationCount !== 8
      || resources.scheduleInstallations.expectedCommandReadbackCount !== 24
      || resources.scheduleInstallations.commandReadbackCount !== 24
      || resources.scheduleInstallations.expectedOrderedChallengeCount !== 6
      || resources.scheduleInstallations.orderedChallengeCount !== 6
      || resources.persistentResourceFreezes?.pass !== true
      || resources.persistentResourceFreezes.records?.length !== 2
      || resources.stagingLifecycle?.pass !== true
      || resources.stagingLifecycle.createdCount !== 825
      || resources.stagingLifecycle.readbackPhaseCount !== 666
      || resources.querySets?.pass !== true
      || resources.querySets.count !== 0
      || resources.externalTextures?.pass !== true
      || resources.externalTextures.count !== 0
      || resources.externalTextures.methodInstrumented !== true
      || resources.textureDestructionBeforeCleanup?.pass !== true
      || resources.textureDestructionBeforeCleanup.textureCount < 5
      || resources.textureDestructionBeforeCleanup.declaredTargetTextureCount !== 5
      || resources.textureDestructionBeforeCleanup.targetTextureInventoryExact !== true
      || resources.textureDestructionBeforeCleanup.declaredTargetTextureIds?.length !== 5
      || resources.textureDestructionBeforeCleanup.destructionRecords?.length !== 0
      || resources.samplers?.pass !== true
      || resources.renderPassTermination?.pass !== true) {
      throw new Error('A complete Phase 0 result needs exact common resource evidence.');
    }
    validateGlobalCommandLedgerEvidence(
      resources.stagingLifecycle,
      result.evidence.lifecycle,
    );
    validateBufferMapLifecycleEvidence(
      resources.bufferMapLifecycle,
      result.evidence.lifecycle,
      resources.stagingLifecycle,
    );
    validateEncoderStateEvidence(resources.encoderState, result.evidence.lifecycle);
    validateTextureInventoryEvidence(
      resources.textureDestructionBeforeCleanup,
      resources,
      result.evidence.lifecycle,
    );
    validateComputeImmediateStateEvidence(
      result.evidence.shaders?.computeImmediateState,
      result.evidence.shaders,
      result.evidence.pipelines,
    );
    validateMergedGeometryRealizationEvidence(resources.mergedGeometryRealization);
    validateGpuByteWitnessEvidence(resources.gpuByteWitnesses);
    validateAddressDiagnosticPositionEvidence(resources.addressDiagnosticPositions);
    if (!resources.geometryFixtures.mergedProduction.gpuRecords.every((record) => (
      resources.gpuByteWitnesses.references.some((reference) => (
        reference.semantic === `merged-geometry/${record.semantic}`
          && reference.witnessId === record.rawWitnessId
          && reference.gpuBufferId === record.gpuBufferId
          && reference.sha256 === record.gpuSha256
      ))
    ))) {
      throw new Error('Merged geometry byte witnesses are not resource-bound.');
    }
    for (const freeze of resources.persistentResourceFreezes.records) {
      validatePersistentResourceFreezeEvidence(freeze);
    }
    const scenarios = result.evidence?.scenarios;
    if (!Array.isArray(scenarios)
      || !exactArray(scenarios.map((entry) => entry.scenarioId), ['v99', 'v20'])) {
      throw new Error('A complete Phase 0 result must contain v99 then v20.');
    }
    validateQueueWriteBufferLedgerEvidence(
      resources.queueWriteBufferLedger,
      result.evidence.lifecycle,
      scenarios,
      result.evidence.pipelines,
      resources,
    );
    for (const scenario of scenarios) {
      if (!Array.isArray(scenario.snapshots)
        || !exactArray(
          scenario.snapshots.map((entry) => entry.label),
          IMMEDIATE_AIF_PHASE0_SNAPSHOT_PLAN.map((entry) => entry.label),
        )
        || !exactArray(
          scenario.snapshots.map((entry) => entry.scheduleId),
          IMMEDIATE_AIF_PHASE0_SNAPSHOT_PLAN.map((entry) => entry.scheduleId),
        )) {
        throw new Error(`${scenario.scenarioId} snapshot sequence is incomplete.`);
      }
      for (const [snapshotIndex, snapshot] of scenario.snapshots.entries()) {
        if (snapshotIndex < 3) {
          const orders = snapshot.laneOrders?.results;
          if (!Array.isArray(orders)
            || orders.length !== IMMEDIATE_AIF_PHASE0_LANE_ORDERS.length
            || !orders.every((entry, index) => (
              exactArray(entry.order, IMMEDIATE_AIF_PHASE0_LANE_ORDERS[index])
              && exactArray(entry.records?.map((record) => record.lane), entry.order)
            ))) {
            throw new Error(`${scenario.scenarioId}/${snapshot.label} lane orders are incomplete.`);
          }
          try {
            validateImmediateAifOrderedProductionChallengeEvidence(
              snapshot.productionChallenge,
            );
            if (snapshot.scheduleInstallationId
                !== snapshot.productionChallenge.installationId
              || snapshot.productionChallenge.scenarioId !== scenario.scenarioId
              || snapshot.productionChallenge.scheduleId !== snapshot.scheduleId
              || snapshot.productionChallenge
                .mergedGeometryRealizationQueueCompleteSequence
                  !== resources.mergedGeometryRealization.queueCompleteSequence) {
              throw new Error('snapshot challenge binding mismatch');
            }
          } catch {
            throw new Error(
              `${scenario.scenarioId}/${snapshot.label} production challenge is not isolated.`,
            );
          }
        } else if (!exactArray(
          snapshot.laneCaptures?.map((record) => record.lane),
          IMMEDIATE_AIF_PHASE0_LANES,
        )) {
          throw new Error(`${scenario.scenarioId}/${snapshot.label} audit captures are incomplete.`);
        }
      }
      if (!Array.isArray(scenario.scheduleInstallations)
        || scenario.scheduleInstallations.length !== 4
        || !exactArray(
          scenario.scheduleInstallations.map((entry) => entry.scheduleId),
          ['canonical', 'S1', 'S2', 'canonical'],
        )
        || !scenario.scheduleInstallations.every((entry, index) => {
          try {
            validateImmediateAifScheduleInstallationEvidence(entry);
            const positionMatches = resources.addressDiagnosticPositions.records.filter(
              (record) => record.installationId === entry.installationId,
            );
            return entry.ordinal === (scenario.scenarioId === 'v99' ? index + 1 : index + 5)
              && positionMatches.length === 1
              && JSON.stringify(positionMatches[0])
                === JSON.stringify(entry.diagnosticPositionRealization)
              && (index < 3
                ? entry.orderedChallenge?.installationId
                  === scenario.snapshots[index].productionChallenge?.installationId
                : entry.orderedChallenge === null);
          } catch {
            return false;
          }
        })) {
        throw new Error(`${scenario.scenarioId} schedule preflights are incomplete.`);
      }
      const gpuCommitments = scenario.gpuResourceCommitments;
      const expectedSharedSemantics = [
        'matrix',
        'bounds',
        'objectBucket',
        'bucketBase',
        'bucketCapacity',
        'cullOrder',
        'visibleIds',
        'overflow',
      ];
      if (gpuCommitments?.schemaVersion !== 1
        || gpuCommitments.kind !== 'immediate-aif-phase0-shared-gpu-resource-commitments'
        || gpuCommitments.pass !== true
        || gpuCommitments.scenarioId !== scenario.scenarioId
        || gpuCommitments.capturePhase
          !== `phase0/resources/${scenario.scenarioId}/shared-commitment`
        || !Number.isInteger(gpuCommitments.startSequence)
        || !Number.isInteger(gpuCommitments.queueCompleteSequence)
        || !Number.isInteger(gpuCommitments.completeSequence)
        || !(gpuCommitments.startSequence < gpuCommitments.queueCompleteSequence
          && gpuCommitments.queueCompleteSequence < gpuCommitments.completeSequence
          && gpuCommitments.completeSequence
            < scenario.snapshots[0].productionChallenge.startSequence)
        || !scenario.snapshots.slice(0, 3).every((snapshot) => (
          snapshot.productionChallenge.sharedGpuCommitmentCompleteSequence
            === gpuCommitments.completeSequence
        ))
        || !exactArray(gpuCommitments.semantics, expectedSharedSemantics)
        || gpuCommitments.managerCalls?.length !== 8
        || !exactArray(
          gpuCommitments.managerCalls.map((record) => record.semantic),
          expectedSharedSemantics,
        )
        || !gpuCommitments.managerCalls.every((record) => (
          record.manager === 'renderer._attributes'
            && record.managerConstructor === 'Attributes'
            && record.attributeType === 3
            && typeof record.gpuBufferId === 'string'
            && Number.isInteger(record.attributeVersion)
            && record.afterDataVersion === record.attributeVersion
            && (record.semantic === 'visibleIds'
              ? record.beforeDataVersion === record.attributeVersion - 1
              : record.beforeDataVersion === record.attributeVersion)
            && record.startSequence < record.completeSequence
            && record.completeSequence < gpuCommitments.queueCompleteSequence
        ))
        || gpuCommitments.managerVersionsExact !== true
        || gpuCommitments.queueWriteBufferCount !== 1
        || gpuCommitments.queueWriteBuffers?.length !== 1
        || gpuCommitments.visibleIdsUploadExact !== true
        || JSON.stringify(gpuCommitments.visibleIdsUpload)
          !== JSON.stringify(gpuCommitments.queueWriteBuffers[0])
        || gpuCommitments.visibleIdsUpload.capturePhase !== gpuCommitments.capturePhase
        || gpuCommitments.visibleIdsUpload.bufferId
          !== gpuCommitments.managerCalls.find(
            (record) => record.semantic === 'visibleIds',
          )?.gpuBufferId
        || gpuCommitments.visibleIdsUpload.bufferOffset !== 0
        || gpuCommitments.visibleIdsUpload.sourceType !== 'Uint32Array'
        || gpuCommitments.visibleIdsUpload.sourceByteLength !== 262_144
        || gpuCommitments.visibleIdsUpload.dataOffset !== 0
        || gpuCommitments.visibleIdsUpload.size !== null
        || gpuCommitments.visibleIdsUpload.semantic !== 'visibleIds'
        || gpuCommitments.visibleIdsUpload.sourceSha256
          !== gpuCommitments.records.find((record) => record.semantic === 'visibleIds')
            ?.cpuSourceSha256
        || gpuCommitments.visibleIdsUpload.sourceWitnessId
          !== gpuCommitments.records.find((record) => record.semantic === 'visibleIds')
            ?.rawWitnessId
        || gpuCommitments.visibleIdsUpload.sequence
          <= gpuCommitments.managerCalls.find(
            (record) => record.semantic === 'visibleIds',
          )?.startSequence
        || gpuCommitments.visibleIdsUpload.sequence
          >= gpuCommitments.managerCalls.find(
            (record) => record.semantic === 'visibleIds',
          )?.completeSequence
        || !Array.isArray(gpuCommitments.records)
        || gpuCommitments.records.length !== 8
        || !exactArray(
          gpuCommitments.records.map((record) => record.semantic),
          expectedSharedSemantics,
        )
        || !gpuCommitments.records.every((record) => (
          record.exact === true
            && record.cpuSha256 === record.gpuSha256
            && typeof record.gpuBufferId === 'string'
            && typeof record.rawWitnessId === 'string'
            && record.readbackStartSequence > gpuCommitments.queueCompleteSequence
            && record.readbackCompleteSequence > record.readbackStartSequence
            && gpuCommitments.managerCalls.find(
              (managerRecord) => managerRecord.semantic === record.semantic,
            )?.gpuBufferId === record.gpuBufferId
        ))) {
        throw new Error(`${scenario.scenarioId} GPU resource commitments are incomplete.`);
      }
      const visibleIdsCommitment = gpuCommitments.records.find(
        (record) => record.semantic === 'visibleIds',
      );
      const matrixRealization = scenario.matrixGpuRealization;
      const matrixWrite = matrixRealization?.queueWriteBuffers?.[0] ?? null;
      if (matrixRealization?.schemaVersion !== 1
        || matrixRealization.kind !== 'immediate-aif-phase0-scenario-matrix-realization'
        || matrixRealization.pass !== true
        || matrixRealization.scenarioId !== scenario.scenarioId
        || matrixRealization.capturePhase
          !== `phase0/scenario-load/${scenario.scenarioId}/matrix-realization`
        || matrixRealization.attributeType !== 3
        || matrixRealization.afterDataVersion !== matrixRealization.attributeVersion
        || typeof matrixRealization.gpuBufferId !== 'string'
        || !/^[0-9a-f]{64}$/.test(matrixRealization.cpuSha256 ?? '')
        || !(matrixRealization.startSequence < matrixRealization.updateCompleteSequence
          && matrixRealization.updateCompleteSequence
            < matrixRealization.queueCompleteSequence
          && matrixRealization.queueCompleteSequence < gpuCommitments.startSequence)
        || matrixRealization.queueWriteExact !== true
        || (scenario.scenarioId === 'v99'
          ? matrixRealization.beforeDataVersion !== null
            || matrixRealization.expectedQueueWriteBufferCount !== 0
            || matrixRealization.queueWriteBufferCount !== 0
            || matrixRealization.queueWriteBuffers?.length !== 0
          : matrixRealization.beforeDataVersion
              !== matrixRealization.attributeVersion - 1
            || matrixRealization.expectedQueueWriteBufferCount !== 1
            || matrixRealization.queueWriteBufferCount !== 1
            || matrixRealization.queueWriteBuffers?.length !== 1
            || matrixWrite.capturePhase !== matrixRealization.capturePhase
            || matrixWrite.bufferId !== matrixRealization.gpuBufferId
            || matrixWrite.bufferOffset !== 0
            || matrixWrite.sourceType !== 'Float32Array'
            || matrixWrite.sourceByteLength !== 4_194_304
            || matrixWrite.dataOffset !== 0
            || matrixWrite.size !== null
            || matrixWrite.semantic !== 'matrix'
            || matrixWrite.sourceSha256 !== matrixRealization.cpuSha256)
        || gpuCommitments.managerCalls[0].gpuBufferId !== matrixRealization.gpuBufferId
        || gpuCommitments.managerCalls[0].attributeId !== matrixRealization.attributeId
        || gpuCommitments.managerCalls[0].attributeVersion
          !== matrixRealization.attributeVersion
        || gpuCommitments.records[0].attributeId !== matrixRealization.attributeId
        || gpuCommitments.records[0].attributeVersion
          !== matrixRealization.attributeVersion
        || gpuCommitments.records[0].gpuBufferId !== matrixRealization.gpuBufferId
        || gpuCommitments.records[0].cpuSha256 !== matrixRealization.cpuSha256) {
        throw new Error(`${scenario.scenarioId} matrix realization is incomplete.`);
      }
      if (!gpuCommitments.records.every((record) => (
        resources.gpuByteWitnesses.references.some((reference) => (
          reference.semantic === `shared/${scenario.scenarioId}/${record.semantic}`
            && reference.witnessId === record.rawWitnessId
            && reference.gpuBufferId === record.gpuBufferId
            && reference.sha256 === record.gpuSha256
        ))
      ))) {
        throw new Error(`${scenario.scenarioId} shared byte witnesses are not resource-bound.`);
      }
      if (scenario.freeze?.visibleIdsAttributeId !== visibleIdsCommitment.attributeId
        || scenario.freeze.visibleIdsAttributeVersion !== visibleIdsCommitment.attributeVersion
        || scenario.freeze.visibleIdsGpuBufferId !== visibleIdsCommitment.gpuBufferId
        || scenario.freeze.visibleIdsSha256 !== visibleIdsCommitment.gpuSha256
        || scenario.freeze.visibleIdsCpuSha256 !== visibleIdsCommitment.cpuSha256
        || scenario.freeze.gpuResourceCommitmentCapturePhase !== gpuCommitments.capturePhase
        || scenario.freeze.gpuResourceCommitmentCompleteSequence
          !== gpuCommitments.completeSequence) {
        throw new Error(`${scenario.scenarioId} survivor freeze is not GPU-bound.`);
      }
      if (scenario.scenarioId === 'v20'
        && (scenario.scenarioLoadBoundary?.pass !== true
          || scenario.scenarioLoadBoundary.fromScenarioId !== 'v99'
          || scenario.scenarioLoadBoundary.toScenarioId !== 'v20')) {
        throw new Error('The v99-to-v20 load boundary is incomplete.');
      }
    }
  }
  return result;
}

export function immediateAifPhase0PlanEvidence() {
  return Object.freeze({
    lanes: [...IMMEDIATE_AIF_PHASE0_LANES],
    laneOrders: IMMEDIATE_AIF_PHASE0_LANE_ORDERS.map((order) => [...order]),
    scheduleIds: [...IMMEDIATE_AIF_PHASE0_SCHEDULE_IDS],
    snapshotPlan: IMMEDIATE_AIF_PHASE0_SNAPSHOT_PLAN.map((entry) => ({ ...entry })),
    workload: {
      objectCount: IMMEDIATE_AIF_PHASE0_WORKLOAD.objectCount,
      bucketCount: IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCount,
      bucketCapacity: IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCapacity,
      geometryTier: IMMEDIATE_AIF_PHASE0_WORKLOAD.geometryTier,
      layout: IMMEDIATE_AIF_PHASE0_WORKLOAD.layout,
      seed: IMMEDIATE_AIF_PHASE0_WORKLOAD.seed,
      bucketCounts: [...IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketCounts],
      bucketBases: [...IMMEDIATE_AIF_PHASE0_WORKLOAD.bucketBases],
      drawCount: IMMEDIATE_AIF_PHASE0_WORKLOAD.drawCount,
      commandUint32Length: IMMEDIATE_AIF_PHASE0_WORKLOAD.commandUint32Length,
      commandByteLength: IMMEDIATE_AIF_PHASE0_WORKLOAD.commandByteLength,
      indirectOffsets: [...IMMEDIATE_AIF_PHASE0_WORKLOAD.indirectOffsets],
      visibilityLevels: IMMEDIATE_AIF_PHASE0_WORKLOAD.visibilityLevels.map((level) => ({
        id: level.id,
        fraction: level.fraction,
        expectedVisibleCount: level.expectedVisibleCount,
        visibleCounts: [...level.visibleCounts],
      })),
    },
    addressTarget: { ...IMMEDIATE_AIF_PHASE0_ADDRESS_TARGET },
    renderConfiguration: structuredClone(IMMEDIATE_AIF_PHASE0_RENDER_CONFIGURATION),
  });
}
