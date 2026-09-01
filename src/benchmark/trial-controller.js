import {
  partitionResolvedTimestampMaps,
  resolveTimestampMaps,
  setTimestampTracking,
  timestampPoolDiagnostics,
  timestampResolution,
  timestampSupport,
} from './gpu-timestamps.js';

function percentile(values, fraction) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * fraction) - 1))];
}

function requirePositiveFrameCount(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive integer.`);
  }
  return value;
}

const TIMESTAMP_TYPES = Object.freeze(['render', 'compute']);
const MAX_TIMESTAMP_QUANTUM_NS = 1_000;
const TIMESTAMP_QUERIES_PER_UID = 2;

function sameJson(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function arrayValuesAreUnique(values) {
  return Array.isArray(values) && new Set(values).size === values.length;
}

function timestampPoolStaticIdentity(pool) {
  return {
    poolIdentity: pool?.poolIdentity ?? null,
    querySetIdentity: pool?.querySetIdentity ?? null,
    resolveBufferIdentity: pool?.resolveBufferIdentity ?? null,
    resultBufferIdentity: pool?.resultBufferIdentity ?? null,
    maxQueries: pool?.maxQueries ?? null,
    isDisposed: pool?.isDisposed ?? null,
  };
}

function validateContinuousTimestampStart(diagnostics, requiredQueries, requiredTypes) {
  if (diagnostics?.backendTrackingEnabled !== false) {
    throw new Error('Continuous timestamp timing must start with tracking disabled.');
  }
  for (const type of requiredTypes) {
    const pool = diagnostics?.[type];
    if (!pool
      || pool.currentQueryIndex !== 0
      || pool.queryOffsetCount !== 0
      || !Array.isArray(pool.queryOffsetUids)
      || pool.queryOffsetUids.length !== pool.queryOffsetCount
      || !arrayValuesAreUnique(pool.queryOffsetUids)
      || pool.pendingResolve !== false
      || pool.isDisposed !== false
      || pool.resultBufferMapState !== 'unmapped'
      || !Number.isInteger(pool.maxQueries)
      || pool.maxQueries < requiredQueries
      || !Array.isArray(pool.frames)
      || pool.frames.length !== pool.frameCount
      || !arrayValuesAreUnique(pool.frames)
      || !Number.isInteger(pool.timestampUidCount)
      || !Array.isArray(pool.timestampUids)
      || pool.timestampUids.length !== pool.timestampUidCount
      || !arrayValuesAreUnique(pool.timestampUids)) {
      throw new Error(
        `${type} timestamp pool cannot retain the continuous timed interval.`,
      );
    }
  }
}

function continuousTimestampUidsMatchRows(rows, type, uids) {
  if (!Array.isArray(uids) || uids.length !== rows.length) return false;
  const prefix = type === 'compute' ? 'c' : 'r';
  return rows.every((row, index) => {
    const callIndex = type === 'compute'
      ? row.computeFrameCallIndex
      : row.renderFrameCallIndex;
    const match = /^(c|r):([1-9]\d*):(0|[1-9]\d*):f(0|[1-9]\d*)$/.exec(uids[index]);
    const uidCallIndex = Number(match?.[2]);
    const uidContextId = Number(match?.[3]);
    const uidFrameId = Number(match?.[4]);
    if (!Number.isSafeInteger(row.gpuFrameId) || row.gpuFrameId < 0
      || !Number.isSafeInteger(callIndex) || callIndex <= 0
      || match?.[1] !== prefix
      || !Number.isSafeInteger(uidCallIndex) || uidCallIndex !== callIndex
      || !Number.isSafeInteger(uidContextId) || uidContextId < 0
      || !Number.isSafeInteger(uidFrameId) || uidFrameId !== row.gpuFrameId
      || (type === 'compute' && uidContextId !== row.computeTimestampContextId)) {
      return false;
    }
    return true;
  });
}

function validateContinuousTimestampPreResolve({
  start,
  beforeResolve,
  rows,
  requiredQueries,
  requiredTypes,
}) {
  if (beforeResolve?.backendTrackingEnabled !== true) {
    throw new Error('Continuous timestamp tracking stopped before final resolution.');
  }
  for (const type of requiredTypes) {
    const initial = start[type];
    const before = beforeResolve?.[type];
    if (!before
      || !sameJson(timestampPoolStaticIdentity(before), timestampPoolStaticIdentity(initial))
      || before.currentQueryIndex !== requiredQueries
      || before.queryOffsetCount !== rows.length
      || !continuousTimestampUidsMatchRows(rows, type, before.queryOffsetUids)
      || before.pendingResolve !== false
      || before.resultBufferMapState !== 'unmapped'
      || before.timestampUidCount !== initial.timestampUidCount
      || !sameJson(before.timestampUids, initial.timestampUids)
      || before.frameCount !== initial.frameCount
      || !sameJson(before.frames, initial.frames)) {
      throw new Error(
        `${type} timestamp pool does not prove one unresolved continuous timed interval.`,
      );
    }
  }
}

function validateContinuousTimestampPostResolve({
  start,
  afterResolve,
  frames,
  requiredTypes,
}) {
  for (const type of requiredTypes) {
    const initial = start[type];
    const after = afterResolve?.[type];
    if (!after
      || !sameJson(timestampPoolStaticIdentity(after), timestampPoolStaticIdentity(initial))
      || after.currentQueryIndex !== 0
      || after.queryOffsetCount !== 0
      || !Array.isArray(after.queryOffsetUids)
      || after.queryOffsetUids.length !== after.queryOffsetCount
      || !arrayValuesAreUnique(after.queryOffsetUids)
      || after.pendingResolve !== false
      || after.resultBufferMapState !== 'unmapped'
      || after.frameCount !== frames.length
      || !sameJson(after.frames, frames)
      || !arrayValuesAreUnique(after.frames)
      || after.timestampUidCount !== initial.timestampUidCount + frames.length
      || !Array.isArray(after.timestampUids)
      || after.timestampUids.length !== after.timestampUidCount
      || !arrayValuesAreUnique(after.timestampUids)) {
      throw new Error(
        `${type} timestamp pool did not resolve the complete continuous timed interval.`,
      );
    }
  }
}

function createTimestampPhaseResults() {
  return {
    schemaVersion: 1,
    kind: 'three-r185-timestamp-phase-results',
    warmup: null,
    measurement: null,
  };
}

function serializeTimestampPhase(phase, maps) {
  return {
    schemaVersion: 1,
    kind: 'three-r185-timestamp-phase-result',
    phase,
    includedTypes: [...maps.includedTypes],
    strictUidGrammar: maps.strictUidGrammar,
    pools: Object.fromEntries(TIMESTAMP_TYPES.map((type) => [type, {
      type,
      included: maps.includedTypes.includes(type),
      frames: [...maps.frames[type]],
      uidRecords: maps.uidRecords[type].map((record) => ({ ...record })),
      resolution: { ...maps.resolutions[type] },
    }])),
  };
}

function timestampPoolQualityValid(resolution) {
  return resolution !== null
    && typeof resolution === 'object'
    && resolution.recordCount > 0
    && resolution.positiveDurationCount === resolution.recordCount
    && resolution.nonpositiveDurationCount === 0
    && Number.isFinite(resolution.quantumNs)
    && resolution.quantumNs > 0
    && resolution.quantumNs <= MAX_TIMESTAMP_QUANTUM_NS
    && (resolution.classification === 'fine' || resolution.classification === 'quantized');
}

function joinTimestampRows(rows, maps, context) {
  const recordsByUid = Object.fromEntries(TIMESTAMP_TYPES.map((type) => [
    type,
    new Map(maps.uidRecords[type].map((record) => [record.uid, record])),
  ]));
  return rows.map((row) => {
    const gpuComputeTimestampUids = context.usesCompute
      ? [...(maps.uidsByFrame.compute.get(row.gpuFrameId) ?? [])]
      : [];
    const gpuComputeTimestampRecords = gpuComputeTimestampUids.map(
      (uid) => recordsByUid.compute.get(uid),
    );
    const gpuComputeTimestampUidCount = context.usesCompute
      ? gpuComputeTimestampUids.length
      : 0;
    const expectedComputeTimestampUidCount = context.expectedComputeTimestampUidCount ?? null;
    const computeTimestampUidCountValid = !context.usesCompute
      || (expectedComputeTimestampUidCount === null
        ? gpuComputeTimestampUidCount > 0
        : gpuComputeTimestampUidCount === expectedComputeTimestampUidCount);
    const computeTimestampDurationValid = !context.usesCompute
      || (gpuComputeTimestampRecords.length > 0
        && gpuComputeTimestampRecords.every((record) => record?.durationMs > 0));
    const gpuComputeMs = context.usesCompute
      && computeTimestampUidCountValid
      && computeTimestampDurationValid
      ? gpuComputeTimestampRecords.reduce((total, record) => total + record.durationMs, 0)
      : null;
    const gpuRenderTimestampUids = [
      ...(maps.uidsByFrame.render.get(row.gpuFrameId) ?? []),
    ];
    const gpuRenderTimestampRecords = gpuRenderTimestampUids.map(
      (uid) => recordsByUid.render.get(uid),
    );
    const gpuRenderTimestampUidCount = gpuRenderTimestampUids.length;
    const expectedRenderTimestampUidCount = context.expectedRenderTimestampUidCount ?? null;
    const renderTimestampUidCountValid = expectedRenderTimestampUidCount === null
      ? gpuRenderTimestampUidCount > 0
      : gpuRenderTimestampUidCount === expectedRenderTimestampUidCount;
    const renderTimestampDurationValid = gpuRenderTimestampRecords.length > 0
      && gpuRenderTimestampRecords.every((record) => record?.durationMs > 0);
    const gpuRenderMs = renderTimestampUidCountValid && renderTimestampDurationValid
      ? gpuRenderTimestampRecords.reduce((total, record) => total + record.durationMs, 0)
      : null;
    const gpuPassTotalMs = gpuRenderMs !== null
      && (!context.usesCompute || gpuComputeMs !== null)
      ? gpuRenderMs + (gpuComputeMs ?? 0)
      : null;
    return {
      ...row,
      gpuComputeMs,
      gpuComputeTimestampUidCount,
      gpuComputeTimestampUids: JSON.stringify(gpuComputeTimestampUids),
      gpuComputeTimestampRecords: JSON.stringify(gpuComputeTimestampRecords),
      gpuComputeTimestampDurationValid: computeTimestampDurationValid,
      gpuRenderMs,
      gpuRenderTimestampUidCount,
      gpuRenderTimestampUids: JSON.stringify(gpuRenderTimestampUids),
      gpuRenderTimestampRecords: JSON.stringify(gpuRenderTimestampRecords),
      gpuRenderTimestampDurationValid: renderTimestampDurationValid,
      gpuPassTotalMs,
    };
  });
}

export class TrialController {
  constructor(renderer, {
    warmupFrames,
    measuredFrames,
    onStatus,
    onComplete,
    onError,
    validateCompletion = null,
  }) {
    this.renderer = renderer;
    this.defaultWarmupFrames = requirePositiveFrameCount(warmupFrames, 'warmupFrames');
    this.defaultMeasuredFrames = requirePositiveFrameCount(measuredFrames, 'measuredFrames');
    this.warmupFrames = this.defaultWarmupFrames;
    this.measuredFrames = this.defaultMeasuredFrames;
    this.onStatus = onStatus;
    this.onComplete = onComplete;
    this.onError = onError;
    this.validateCompletion = validateCompletion;
    this.phase = 'idle';
    this.remaining = 0;
    this.rows = [];
    this.context = null;
    this.error = null;
    this.timestampPhases = createTimestampPhaseResults();
    this.timestampResolutionTopology = null;
    this.continuousTimestampPoolsAtStart = null;
    this.deferWarmupTimestampResolution = false;
    this._warmupRows = [];
  }

  fail(error) {
    setTimestampTracking(this.renderer, false);
    this.remaining = 0;
    this.phase = 'error';
    this.error = error instanceof Error ? error : new Error(String(error));
    const message = this.error.message;
    this.onStatus?.(`Trial failed: ${message}`);
    this.onError?.(this.error);
  }

  get active() {
    return this.phase === 'warmup' || this.phase === 'measure';
  }

  get resolving() {
    return this.phase.startsWith('resolving');
  }

  get warmupRows() {
    return this._warmupRows;
  }

  get frameDescriptor() {
    if (!this.active) return null;
    const phaseFrameCount = this.phase === 'warmup'
      ? this.warmupFrames
      : this.measuredFrames;
    return Object.freeze({
      phase: this.phase,
      phaseFrameIndex: phaseFrameCount - this.remaining,
      phaseFrameCount,
    });
  }

  async start(context, {
    warmupFrames = this.defaultWarmupFrames,
    measuredFrames = this.defaultMeasuredFrames,
    deferWarmupTimestampResolution = false,
  } = {}) {
    if (!['idle', 'complete', 'error'].includes(this.phase)) throw new Error('A trial is already active.');
    this.warmupFrames = requirePositiveFrameCount(warmupFrames, 'warmupFrames');
    this.measuredFrames = requirePositiveFrameCount(measuredFrames, 'measuredFrames');
    if (typeof deferWarmupTimestampResolution !== 'boolean') {
      throw new TypeError('deferWarmupTimestampResolution must be a boolean.');
    }
    this.deferWarmupTimestampResolution = deferWarmupTimestampResolution;
    this.phase = 'resolving-start';
    this.error = null;
    this.context = context;
    this.rows = [];
    this.timestampPhases = createTimestampPhaseResults();
    this.timestampResolutionTopology = null;
    this.continuousTimestampPoolsAtStart = null;
    this._warmupRows = [];
    try {
      if (this.deferWarmupTimestampResolution) {
        if (this.context.usesCompute !== true
          || this.context.strictTimestampUidAttribution !== true
          || this.context.expectedRenderTimestampUidCount !== 1
          || this.context.expectedComputeTimestampUidCount !== 1) {
          throw new Error(
            'Continuous timestamp timing requires one strictly attributed render and compute UID per frame.',
          );
        }
        const requiredQueries = (this.warmupFrames + this.measuredFrames)
          * TIMESTAMP_QUERIES_PER_UID;
        const requiredTypes = this.context.usesCompute ? TIMESTAMP_TYPES : ['render'];
        const diagnostics = timestampPoolDiagnostics(this.renderer);
        validateContinuousTimestampStart(diagnostics, requiredQueries, requiredTypes);
        this.continuousTimestampPoolsAtStart = diagnostics;
        setTimestampTracking(this.renderer, true);
      } else {
        setTimestampTracking(this.renderer, true);
        await resolveTimestampMaps(this.renderer, {
          includeCompute: true,
          collect: false,
          strictUidGrammar: this.context.strictTimestampUidAttribution === true,
        });
      }
    } catch (error) {
      this.fail(error);
      return;
    }
    this.phase = 'warmup';
    this.remaining = this.warmupFrames;
    this.onStatus?.(`Warming ${this.warmupFrames} frames.`);
  }

  recordFrame(row) {
    if (!this.active) return;
    if (this.phase === 'warmup') {
      const warmupFrameIndex = this._warmupRows.length;
      this._warmupRows.push({
        ...row,
        phase: 'warmup',
        frameIndex: warmupFrameIndex,
        warmupFrameIndex,
      });
    } else if (this.phase === 'measure') {
      this.rows.push({
        ...this.context,
        frameIndex: this.rows.length,
        ...row,
      });
    }
    this.remaining -= 1;
    if (this.remaining > 0) return;
    if (this.phase === 'warmup') {
      if (this.deferWarmupTimestampResolution) {
        this.phase = 'measure';
        this.remaining = this.measuredFrames;
        this.onStatus?.(`Measuring ${this.measuredFrames} frames.`);
        return;
      }
      this.phase = 'resolving-warmup';
      void this.finishWarmup().catch((error) => this.fail(error));
    } else {
      this.phase = 'resolving-measurement';
      void this.finishMeasurement().catch((error) => this.fail(error));
    }
  }

  async finishWarmup() {
    const maps = await resolveTimestampMaps(this.renderer, {
      includeCompute: this.context.usesCompute,
      collect: true,
      strictUidGrammar: this.context.strictTimestampUidAttribution === true,
    });
    this.timestampPhases.warmup = serializeTimestampPhase('warmup', maps);
    this._warmupRows = Object.freeze(
      joinTimestampRows(this._warmupRows, maps, this.context)
        .map((row) => Object.freeze(row)),
    );
    this.rows = [];
    this.phase = 'measure';
    this.remaining = this.measuredFrames;
    this.onStatus?.(`Measuring ${this.measuredFrames} frames.`);
  }

  async finishMeasurement() {
    let poolsBeforePostMeasurementResolve = null;
    const continuousRows = this.deferWarmupTimestampResolution
      ? [...this._warmupRows, ...this.rows]
      : null;
    const continuousFrames = continuousRows?.map((row) => row.gpuFrameId) ?? null;
    const requiredTypes = this.context.usesCompute ? TIMESTAMP_TYPES : ['render'];
    const requiredQueries = this.deferWarmupTimestampResolution
      ? continuousRows.length * TIMESTAMP_QUERIES_PER_UID
      : null;
    if (this.deferWarmupTimestampResolution) {
      poolsBeforePostMeasurementResolve = timestampPoolDiagnostics(this.renderer);
      validateContinuousTimestampPreResolve({
        start: this.continuousTimestampPoolsAtStart,
        beforeResolve: poolsBeforePostMeasurementResolve,
        rows: continuousRows,
        requiredQueries,
        requiredTypes,
      });
    }
    const resolvedMaps = await resolveTimestampMaps(this.renderer, {
      includeCompute: this.context.usesCompute,
      collect: true,
      strictUidGrammar: this.context.strictTimestampUidAttribution === true,
    });
    const poolsAfterPostMeasurementResolve = this.deferWarmupTimestampResolution
      ? timestampPoolDiagnostics(this.renderer)
      : null;
    if (this.deferWarmupTimestampResolution) {
      validateContinuousTimestampPostResolve({
        start: this.continuousTimestampPoolsAtStart,
        afterResolve: poolsAfterPostMeasurementResolve,
        frames: continuousFrames,
        requiredTypes,
      });
    }
    let maps = resolvedMaps;
    if (this.deferWarmupTimestampResolution) {
      const warmupFrames = this._warmupRows.map((row) => row.gpuFrameId);
      const measurementFrames = this.rows.map((row) => row.gpuFrameId);
      const partitioned = partitionResolvedTimestampMaps(resolvedMaps, {
        warmupFrames,
        measurementFrames,
      });
      this.timestampPhases.warmup = serializeTimestampPhase('warmup', partitioned.warmup);
      this._warmupRows = Object.freeze(
        joinTimestampRows(this._warmupRows, partitioned.warmup, this.context)
          .map((row) => Object.freeze(row)),
      );
      maps = partitioned.measurement;
      const combinedFrames = [...warmupFrames, ...measurementFrames];
      this.timestampResolutionTopology = Object.freeze({
        schemaVersion: 1,
        kind: 'three-r185-timestamp-resolution-topology',
        mode: 'single-post-measurement',
        warmupBoundaryResolveBatchCount: 0,
        postMeasurementResolveBatchCount: 1,
        queriesPerTimestampUid: TIMESTAMP_QUERIES_PER_UID,
        requiredQueriesPerType: requiredQueries,
        resolvedFrameCountByType: Object.freeze(Object.fromEntries(
          resolvedMaps.includedTypes.map((type) => [type, resolvedMaps.frames[type].length]),
        )),
        firstGpuFrameId: combinedFrames[0],
        lastGpuFrameId: combinedFrames.at(-1),
        intervalContiguous: combinedFrames.every(
          (frameId, index) => index === 0 || frameId === combinedFrames[index - 1] + 1,
        ),
        poolsAtStart: this.continuousTimestampPoolsAtStart,
        poolsBeforePostMeasurementResolve,
        poolsAfterPostMeasurementResolve,
      });
    }
    this.timestampPhases.measurement = serializeTimestampPhase('measurement', maps);
    const joined = joinTimestampRows(this.rows, maps, this.context);
    const missingRenderFrames = joined.filter((row) => row.gpuRenderMs === null).length;
    const invalidRenderTimestampUidCountFrames = joined.filter(
      (row) => (
        this.context.expectedRenderTimestampUidCount === null
        || this.context.expectedRenderTimestampUidCount === undefined
          ? row.gpuRenderTimestampUidCount <= 0
          : row.gpuRenderTimestampUidCount !== this.context.expectedRenderTimestampUidCount
      ),
    ).length;
    const missingComputeFrames = this.context.usesCompute
      ? joined.filter((row) => row.gpuComputeMs === null).length
      : 0;
    const invalidComputeTimestampUidCountFrames = this.context.usesCompute
      ? joined.filter(
        (row) => (
          this.context.expectedComputeTimestampUidCount === null
          || this.context.expectedComputeTimestampUidCount === undefined
            ? row.gpuComputeTimestampUidCount <= 0
            : row.gpuComputeTimestampUidCount
              !== this.context.expectedComputeTimestampUidCount
        ),
      ).length
      : 0;
    const invalidRenderTimestampDurationFrames = joined.filter(
      (row) => row.gpuRenderTimestampDurationValid !== true,
    ).length;
    const invalidComputeTimestampDurationFrames = this.context.usesCompute
      ? joined.filter((row) => row.gpuComputeTimestampDurationValid !== true).length
      : 0;
    const missingWarmupRenderFrames = this._warmupRows.filter(
      (row) => row.gpuRenderMs === null,
    ).length;
    const invalidWarmupRenderTimestampUidCountFrames = this._warmupRows.filter(
      (row) => (
        this.context.expectedRenderTimestampUidCount === null
        || this.context.expectedRenderTimestampUidCount === undefined
          ? row.gpuRenderTimestampUidCount <= 0
          : row.gpuRenderTimestampUidCount !== this.context.expectedRenderTimestampUidCount
      ),
    ).length;
    const invalidWarmupRenderTimestampDurationFrames = this._warmupRows.filter(
      (row) => row.gpuRenderTimestampDurationValid !== true,
    ).length;
    const missingWarmupComputeFrames = this.context.usesCompute
      ? this._warmupRows.filter((row) => row.gpuComputeMs === null).length
      : 0;
    const invalidWarmupComputeTimestampUidCountFrames = this.context.usesCompute
      ? this._warmupRows.filter(
        (row) => (
          this.context.expectedComputeTimestampUidCount === null
          || this.context.expectedComputeTimestampUidCount === undefined
            ? row.gpuComputeTimestampUidCount <= 0
            : row.gpuComputeTimestampUidCount
              !== this.context.expectedComputeTimestampUidCount
        ),
      ).length
      : 0;
    const invalidWarmupComputeTimestampDurationFrames = this.context.usesCompute
      ? this._warmupRows.filter((row) => row.gpuComputeTimestampDurationValid !== true).length
      : 0;
    const warmupRenderTimestampPoolQualityValid = timestampPoolQualityValid(
      this.timestampPhases.warmup?.pools?.render?.resolution,
    );
    const warmupComputeTimestampPoolQualityValid = !this.context.usesCompute
      || timestampPoolQualityValid(
        this.timestampPhases.warmup?.pools?.compute?.resolution,
      );
    const renderTimestampPoolQualityValid = timestampPoolQualityValid(
      maps.resolutions.render,
    );
    const computeTimestampPoolQualityValid = !this.context.usesCompute
      || timestampPoolQualityValid(maps.resolutions.compute);
    const warmupTimestampFrameCountValid = TIMESTAMP_TYPES.every((type) => (
      (type !== 'compute' || this.context.usesCompute)
        ? this.timestampPhases.warmup?.pools?.[type]?.frames?.length === this.warmupFrames
        : true
    ));
    const measurementTimestampFrameCountValid = TIMESTAMP_TYPES.every((type) => (
      (type !== 'compute' || this.context.usesCompute)
        ? maps.frames[type].length === this.measuredFrames
        : true
    ));
    const resolution = timestampResolution(maps);
    const gpuTotals = joined.map((row) => row.gpuPassTotalMs).filter(Number.isFinite);
    const completionResult = this.validateCompletion === null
      ? { pass: true }
      : this.validateCompletion(this.context);
    const completionInvariant = completionResult !== null
      && typeof completionResult === 'object'
      && !Array.isArray(completionResult)
      && typeof completionResult.pass === 'boolean'
      ? completionResult
      : { pass: false, kind: 'invalid-completion-invariant-result' };
    const summary = {
      rowCount: joined.length,
      warmupRowCount: this._warmupRows.length,
      missingWarmupRenderFrames,
      invalidWarmupRenderTimestampUidCountFrames,
      invalidWarmupRenderTimestampDurationFrames,
      missingWarmupComputeFrames,
      invalidWarmupComputeTimestampUidCountFrames,
      invalidWarmupComputeTimestampDurationFrames,
      missingRenderFrames,
      invalidRenderTimestampUidCountFrames,
      invalidRenderTimestampDurationFrames,
      expectedRenderTimestampUidCount: this.context.expectedRenderTimestampUidCount ?? null,
      missingComputeFrames,
      invalidComputeTimestampUidCountFrames,
      invalidComputeTimestampDurationFrames,
      expectedComputeTimestampUidCount: this.context.expectedComputeTimestampUidCount ?? null,
      timestampAvailable: timestampSupport(this.renderer),
      timestampResolutions: Object.fromEntries(TIMESTAMP_TYPES.map((type) => [
        type,
        { ...maps.resolutions[type] },
      ])),
      renderTimestampPoolQualityValid,
      computeTimestampPoolQualityValid,
      warmupRenderTimestampPoolQualityValid,
      warmupComputeTimestampPoolQualityValid,
      warmupTimestampFrameCountValid,
      measurementTimestampFrameCountValid,
      timestampPhases: this.timestampPhases,
      timestampResolutionTopology: this.timestampResolutionTopology,
      ...resolution,
      cpuSubmitP50Ms: percentile(joined.map((row) => row.cpuSubmitTotalMs), 0.5),
      cpuSubmitP95Ms: percentile(joined.map((row) => row.cpuSubmitTotalMs), 0.95),
      gpuPassP50Ms: percentile(gpuTotals, 0.5),
      gpuPassP95Ms: percentile(gpuTotals, 0.95),
      completionInvariant,
      accepted: joined.length === this.measuredFrames
        && this._warmupRows.length === this.warmupFrames
        && missingWarmupRenderFrames === 0
        && invalidWarmupRenderTimestampUidCountFrames === 0
        && invalidWarmupRenderTimestampDurationFrames === 0
        && missingWarmupComputeFrames === 0
        && invalidWarmupComputeTimestampUidCountFrames === 0
        && invalidWarmupComputeTimestampDurationFrames === 0
        && missingRenderFrames === 0
        && invalidRenderTimestampUidCountFrames === 0
        && invalidRenderTimestampDurationFrames === 0
        && missingComputeFrames === 0
        && invalidComputeTimestampUidCountFrames === 0
        && invalidComputeTimestampDurationFrames === 0
        && renderTimestampPoolQualityValid
        && computeTimestampPoolQualityValid
        && warmupRenderTimestampPoolQualityValid
        && warmupComputeTimestampPoolQualityValid
        && warmupTimestampFrameCountValid
        && measurementTimestampFrameCountValid
        && completionInvariant.pass === true,
    };
    setTimestampTracking(this.renderer, false);
    this.phase = 'complete';
    this.onStatus?.(
      summary.accepted
        ? 'Trial timing complete.'
        : 'Trial rejected: incomplete timestamps or failed completion invariant.',
    );
    this.onComplete?.({
      rows: joined,
      summary,
      context: this.context,
      timestampPhases: this.timestampPhases,
      warmupRows: this._warmupRows,
    });
  }
}
