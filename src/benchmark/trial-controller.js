import {
  resolveTimestampMaps,
  setTimestampTracking,
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
  } = {}) {
    if (!['idle', 'complete', 'error'].includes(this.phase)) throw new Error('A trial is already active.');
    this.warmupFrames = requirePositiveFrameCount(warmupFrames, 'warmupFrames');
    this.measuredFrames = requirePositiveFrameCount(measuredFrames, 'measuredFrames');
    this.phase = 'resolving-start';
    this.error = null;
    this.context = context;
    this.rows = [];
    this.timestampPhases = createTimestampPhaseResults();
    this._warmupRows = [];
    try {
      setTimestampTracking(this.renderer, true);
      await resolveTimestampMaps(this.renderer, {
        includeCompute: true,
        collect: false,
        strictUidGrammar: this.context.strictTimestampUidAttribution === true,
      });
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
    const maps = await resolveTimestampMaps(this.renderer, {
      includeCompute: this.context.usesCompute,
      collect: true,
      strictUidGrammar: this.context.strictTimestampUidAttribution === true,
    });
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
