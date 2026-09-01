function timestampTypes(includeCompute) {
  return includeCompute ? ['render', 'compute'] : ['render'];
}

const TIMESTAMP_TYPES = Object.freeze(['render', 'compute']);
const TIMESTAMP_PREFIX_BY_TYPE = Object.freeze({ render: 'r', compute: 'c' });
const TIMESTAMP_TYPE_BY_PREFIX = Object.freeze({ r: 'render', c: 'compute' });
const THREE_R185_TIMESTAMP_UID = /^(c|r):([1-9]\d*):(0|[1-9]\d*):f(0|[1-9]\d*)$/;

const registeredComputeTimestampBackends = new WeakMap();

function timestampBackend(renderer) {
  const backend = renderer?.backend;
  if (!backend
    || typeof backend.updateTimeStampUID !== 'function'
    || typeof backend.get !== 'function') {
    throw new TypeError(
      'Compute timestamp-group registration requires an initialized Three.js backend.',
    );
  }
  return backend;
}

/**
 * Gives one exact r185 compute-group Array identity an unambiguous timestamp UID.
 * The Array is deliberately left untouched: all attribution lives in this
 * module's WeakMap and one narrowly scoped backend wrapper.
 */
export function registerComputeTimestampGroup(renderer, computeGroup, contextId) {
  if (!Array.isArray(computeGroup)) {
    throw new TypeError('A registered compute timestamp group must be an Array identity.');
  }
  if (!Number.isSafeInteger(contextId) || contextId <= 0) {
    throw new RangeError('Compute timestamp contextId must be a positive safe integer.');
  }

  const backend = timestampBackend(renderer);
  let state = registeredComputeTimestampBackends.get(backend);
  if (state === undefined) {
    const originalUpdateTimeStampUID = backend.updateTimeStampUID;
    state = {
      contextIds: new Set(),
      groups: new WeakMap(),
      originalUpdateTimeStampUID,
      registrationSerial: 0,
      wrapper: null,
    };
    state.wrapper = function updateRegisteredComputeTimestampUID(abstractContext) {
      const registeredContextId = state.groups.get(abstractContext);
      if (registeredContextId === undefined) {
        return Reflect.apply(state.originalUpdateTimeStampUID, this, [abstractContext]);
      }
      if (this !== backend) {
        throw new Error('Registered compute timestamp group used with a different backend.');
      }
      const info = this.renderer?.info ?? renderer?.info;
      const frameCall = info?.compute?.frameCalls;
      const frame = info?.frame;
      if (!Number.isSafeInteger(frameCall) || frameCall <= 0
        || !Number.isSafeInteger(frame) || frame < 0) {
        throw new Error('Registered compute timestamp group has invalid renderer frame state.');
      }
      const contextData = this.get(abstractContext);
      if ((typeof contextData !== 'object' && typeof contextData !== 'function')
        || contextData === null) {
        throw new Error('Registered compute timestamp group lacks backend context data.');
      }
      contextData.timestampUID = `c:${frameCall}:${registeredContextId}:f${frame}`;
      return undefined;
    };
    Object.defineProperty(backend, 'updateTimeStampUID', {
      configurable: true,
      enumerable: false,
      value: state.wrapper,
      writable: true,
    });
    registeredComputeTimestampBackends.set(backend, state);
  } else if (backend.updateTimeStampUID !== state.wrapper) {
    throw new Error('The registered compute timestamp UID backend wrapper was replaced.');
  }

  if (state.groups.has(computeGroup)) {
    throw new Error('Compute timestamp group Array identity is already registered.');
  }
  if (state.contextIds.has(contextId)) {
    throw new Error(`Compute timestamp contextId ${contextId} is already registered.`);
  }
  state.groups.set(computeGroup, contextId);
  state.contextIds.add(contextId);
  state.registrationSerial += 1;
  return Object.freeze({
    schemaVersion: 1,
    kind: 'three-r185-compute-timestamp-group-registration',
    contextId,
    registrationSerial: state.registrationSerial,
    backendIdentity: timestampObjectId(backend),
    backendWrapperIdentity: timestampObjectId(state.wrapper),
    computeGroupIdentity: timestampObjectId(computeGroup),
  });
}

const timestampObjectIds = new WeakMap();
let nextTimestampObjectId = 1;

function timestampObjectId(value) {
  if ((typeof value !== 'object' && typeof value !== 'function') || value === null) {
    return null;
  }
  if (!timestampObjectIds.has(value)) {
    timestampObjectIds.set(value, nextTimestampObjectId);
    nextTimestampObjectId += 1;
  }
  return timestampObjectIds.get(value);
}

export function timestampSupport(renderer) {
  return renderer.backend?.trackTimestamp === true;
}

export function setTimestampTracking(renderer, enabled) {
  if (renderer.backend) renderer.backend.trackTimestamp = enabled;
}

function timestampPoolRecord(pool) {
  if (!pool) return null;
  return {
    poolIdentity: timestampObjectId(pool),
    querySetIdentity: timestampObjectId(pool.querySet),
    resolveBufferIdentity: timestampObjectId(pool.resolveBuffer),
    resultBufferIdentity: timestampObjectId(pool.resultBuffer),
    maxQueries: pool.maxQueries ?? null,
    currentQueryIndex: pool.currentQueryIndex ?? null,
    queryOffsetCount: pool.queryOffsets instanceof Map ? pool.queryOffsets.size : null,
    queryOffsetUids: pool.queryOffsets instanceof Map ? [...pool.queryOffsets.keys()] : null,
    frameCount: Array.isArray(pool.frames) ? pool.frames.length : null,
    frames: Array.isArray(pool.frames) ? [...pool.frames] : null,
    timestampUidCount: pool.timestamps instanceof Map ? pool.timestamps.size : null,
    timestampUids: pool.timestamps instanceof Map ? [...pool.timestamps.keys()].sort() : null,
    pendingResolve: pool.pendingResolve === false ? false : Boolean(pool.pendingResolve),
    isDisposed: pool.isDisposed === true,
    resultBufferMapState: pool.resultBuffer?.mapState ?? null,
  };
}

export function timestampPoolDiagnostics(renderer) {
  const pools = renderer?.backend?.timestampQueryPool;
  return {
    schemaVersion: 1,
    kind: 'three-r185-timestamp-pool-diagnostics',
    backendTrackingEnabled: renderer?.backend?.trackTimestamp === true,
    render: timestampPoolRecord(pools?.render),
    compute: timestampPoolRecord(pools?.compute),
  };
}

function assertPrimedTimestampPool(record, type) {
  if (!record
    || !Number.isInteger(record.poolIdentity)
    || !Number.isInteger(record.querySetIdentity)
    || !Number.isInteger(record.resolveBufferIdentity)
    || !Number.isInteger(record.resultBufferIdentity)) {
    throw new Error(`${type} timestamp pool was not fully allocated by the untimed prime.`);
  }
  if (!Number.isInteger(record.maxQueries) || record.maxQueries < 2) {
    throw new Error(`${type} timestamp pool has an invalid capacity.`);
  }
  if (record.currentQueryIndex !== 0
    || record.queryOffsetCount !== 0
    || record.frameCount !== 1
    || !Number.isInteger(record.timestampUidCount)
    || record.timestampUidCount < 1
    || record.pendingResolve !== false
    || record.isDisposed !== false
    || record.resultBufferMapState !== 'unmapped') {
    throw new Error(`${type} timestamp pool did not return one resolved prime UID in a clean state.`);
  }
}

export async function preprimeTimestampPools(renderer, {
  submitCompute,
  submitRender,
} = {}) {
  if (typeof submitCompute !== 'function' || typeof submitRender !== 'function') {
    throw new TypeError('Timestamp pre-prime requires compute and render callbacks.');
  }
  if (!renderer?.backend || renderer.backend.trackTimestamp === true) {
    throw new Error('Timestamp pre-prime requires initialized, disabled timestamp tracking.');
  }
  const before = timestampPoolDiagnostics(renderer);
  setTimestampTracking(renderer, true);
  try {
    await submitCompute();
    await submitRender();
    const maps = await resolveTimestampMaps(renderer, {
      includeCompute: true,
      collect: true,
      strictUidGrammar: true,
    });
    for (const type of TIMESTAMP_TYPES) {
      const resolution = maps.resolutions[type];
      if (maps.uidRecords[type].length !== 1
        || resolution.positiveDurationCount !== 1
        || resolution.nonpositiveDurationCount !== 0) {
        throw new Error(`${type} timestamp pre-prime did not resolve one positive UID.`);
      }
    }
  } finally {
    setTimestampTracking(renderer, false);
  }
  const after = timestampPoolDiagnostics(renderer);
  assertPrimedTimestampPool(after.render, 'render');
  assertPrimedTimestampPool(after.compute, 'compute');
  for (const type of ['render', 'compute']) {
    const previous = before[type];
    const current = after[type];
    const previousUidCount = previous?.timestampUidCount ?? 0;
    if (current.timestampUidCount !== previousUidCount + 1) {
      throw new Error(`${type} timestamp pre-prime did not add exactly one resolved UID.`);
    }
    if (previous && current.poolIdentity !== previous.poolIdentity) {
      throw new Error(`${type} timestamp pre-prime replaced an existing query pool.`);
    }
  }
  return {
    schemaVersion: 1,
    kind: 'three-r185-timestamp-pool-preprime',
    before,
    after,
    addedTimestampUidCount: { render: 1, compute: 1 },
  };
}

function emptyTimestampResolution() {
  return {
    quantumNs: null,
    classification: 'unresolved',
    recordCount: 0,
    positiveDurationCount: 0,
    nonpositiveDurationCount: 0,
  };
}

function createTimestampMaps(includeCompute, strictUidGrammar) {
  return {
    schemaVersion: 2,
    kind: 'three-r185-resolved-timestamp-maps',
    includedTypes: timestampTypes(includeCompute),
    strictUidGrammar,
    render: new Map(),
    compute: new Map(),
    uidCounts: {
      render: new Map(),
      compute: new Map(),
    },
    frames: {
      render: [],
      compute: [],
    },
    uidRecords: {
      render: [],
      compute: [],
    },
    uidDurations: {
      render: new Map(),
      compute: new Map(),
    },
    uidsByFrame: {
      render: new Map(),
      compute: new Map(),
    },
    resolutions: {
      render: emptyTimestampResolution(),
      compute: emptyTimestampResolution(),
    },
  };
}

function parseTimestampUid(uid, expectedType, strictUidGrammar) {
  if (typeof uid !== 'string') {
    throw new TypeError(`${expectedType} timestamp UID must be a string.`);
  }
  if (!strictUidGrammar) {
    const legacyMatch = /:f(\d+)$/.exec(uid);
    if (legacyMatch === null) return null;
    const frameId = Number(legacyMatch[1]);
    if (!Number.isSafeInteger(frameId)) {
      throw new Error(`${expectedType} timestamp UID exceeds safe frame integers: ${uid}`);
    }
    return {
      uid,
      type: expectedType,
      callIndex: null,
      contextId: null,
      frameId,
    };
  }
  const match = THREE_R185_TIMESTAMP_UID.exec(uid);
  if (match === null) {
    throw new Error(`Malformed Three.js r185 ${expectedType} timestamp UID: ${uid}`);
  }
  const type = TIMESTAMP_TYPE_BY_PREFIX[match[1]];
  if (type !== expectedType || match[1] !== TIMESTAMP_PREFIX_BY_TYPE[expectedType]) {
    throw new Error(
      `Three.js r185 timestamp UID ${uid} belongs to ${type}, not ${expectedType}.`,
    );
  }
  const callIndex = Number(match[2]);
  const contextId = Number(match[3]);
  const frameId = Number(match[4]);
  if (!Number.isSafeInteger(callIndex)
    || !Number.isSafeInteger(contextId)
    || !Number.isSafeInteger(frameId)) {
    throw new Error(`Three.js r185 ${expectedType} timestamp UID exceeds safe integers: ${uid}`);
  }
  return { uid, type, callIndex, contextId, frameId };
}

function resolvedFrames(renderer, pool, type) {
  const value = renderer.backend?.getTimestampFrames?.(type) ?? pool?.frames ?? [];
  if (!Array.isArray(value)) {
    throw new TypeError(`${type} timestamp frames must be an Array.`);
  }
  const frames = [...value];
  const seen = new Set();
  for (const frame of frames) {
    if (!Number.isSafeInteger(frame) || frame < 0) {
      throw new Error(`${type} timestamp pool contains an invalid frame ID.`);
    }
    if (seen.has(frame)) {
      throw new Error(`${type} timestamp pool contains duplicate frame ${frame}.`);
    }
    seen.add(frame);
  }
  return frames;
}

function resolutionForRecords(records) {
  const positiveDurationsNs = [];
  let nonpositiveDurationCount = 0;
  for (const record of records) {
    if (record.durationMs > 0) {
      positiveDurationsNs.push(Math.round(record.durationMs * 1e6));
    } else {
      nonpositiveDurationCount += 1;
    }
  }
  const base = {
    recordCount: records.length,
    positiveDurationCount: positiveDurationsNs.length,
    nonpositiveDurationCount,
  };
  if (nonpositiveDurationCount > 0) {
    return { quantumNs: null, classification: 'invalid', ...base };
  }
  if (positiveDurationsNs.length === 0) {
    return { quantumNs: null, classification: 'unresolved', ...base };
  }
  const quantumNs = positiveDurationsNs.reduce(gcd);
  return {
    quantumNs,
    classification: quantumNs >= 10_000 ? 'quantized' : 'fine',
    ...base,
  };
}

export async function resolveTimestampMaps(renderer, {
  includeCompute,
  collect,
  strictUidGrammar = false,
}) {
  if (typeof strictUidGrammar !== 'boolean') {
    throw new TypeError('strictUidGrammar must be a boolean.');
  }
  const maps = createTimestampMaps(includeCompute, strictUidGrammar);
  if (!timestampSupport(renderer)) return maps;

  const types = maps.includedTypes;
  await Promise.all(types.map((type) => renderer.resolveTimestampsAsync(type)));

  for (const type of types) {
    const pool = renderer.backend?.timestampQueryPool?.[type];
    if (!pool) continue;
    if (!(pool.timestamps instanceof Map)) {
      throw new TypeError(`${type} timestamp pool lacks a resolved timestamp Map.`);
    }
    const frameList = resolvedFrames(renderer, pool, type);
    const frames = new Set(frameList);
    const records = [];
    for (const [uid, durationMs] of pool.timestamps) {
      const parsed = parseTimestampUid(uid, type, strictUidGrammar);
      if (!Number.isFinite(durationMs)) {
        throw new Error(`${type} timestamp UID ${uid} has a non-finite duration.`);
      }
      if (collect && parsed !== null && frames.has(parsed.frameId)) {
        records.push({ ...parsed, durationMs });
      }
    }
    if (!collect) continue;

    records.sort((left, right) => left.frameId - right.frameId
      || left.callIndex - right.callIndex
      || left.contextId - right.contextId
      || left.uid.localeCompare(right.uid));
    maps.frames[type] = frameList;
    maps.uidRecords[type] = records;
    for (const frame of frameList) {
      maps[type].set(frame, 0);
      maps.uidCounts[type].set(frame, 0);
      maps.uidsByFrame[type].set(frame, []);
    }
    for (const record of records) {
      maps.uidDurations[type].set(record.uid, record.durationMs);
      maps.uidsByFrame[type].get(record.frameId).push(record.uid);
      maps[type].set(record.frameId, maps[type].get(record.frameId) + record.durationMs);
      maps.uidCounts[type].set(
        record.frameId,
        maps.uidCounts[type].get(record.frameId) + 1,
      );
    }
    maps.resolutions[type] = resolutionForRecords(records);
  }
  return maps;
}

function gcd(left, right) {
  let a = Math.abs(Math.trunc(left));
  let b = Math.abs(Math.trunc(right));
  while (b !== 0) [a, b] = [b, a % b];
  return a;
}

export function timestampResolution(maps) {
  const includedTypes = Array.isArray(maps?.includedTypes)
    ? maps.includedTypes
    : TIMESTAMP_TYPES.filter((type) => maps?.[type] instanceof Map && maps[type].size > 0);
  const resolutions = includedTypes.map((type) => (
    maps?.resolutions?.[type]
      ?? resolutionForRecords(
        [...(maps?.[type]?.values?.() ?? [])].map((durationMs, index) => ({
          uid: `${type}:${index}`,
          durationMs,
        })),
      )
  ));
  if (resolutions.some((resolution) => resolution.classification === 'invalid')) {
    return { quantumNs: null, classification: 'invalid' };
  }
  if (resolutions.length === 0 || resolutions.some(
    (resolution) => resolution.classification === 'unresolved',
  )) {
    return { quantumNs: null, classification: 'unresolved' };
  }
  const quantumNs = Math.max(...resolutions.map((resolution) => resolution.quantumNs));
  return {
    quantumNs,
    classification: resolutions.some((resolution) => resolution.classification === 'quantized')
      ? 'quantized'
      : 'fine',
  };
}
