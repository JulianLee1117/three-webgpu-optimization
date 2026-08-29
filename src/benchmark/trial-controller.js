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

export class TrialController {
  constructor(renderer, { warmupFrames, measuredFrames, onStatus, onComplete, onError }) {
    this.renderer = renderer;
    this.warmupFrames = warmupFrames;
    this.measuredFrames = measuredFrames;
    this.onStatus = onStatus;
    this.onComplete = onComplete;
    this.onError = onError;
    this.phase = 'idle';
    this.remaining = 0;
    this.rows = [];
    this.context = null;
    this.error = null;
  }

  fail(error) {
    setTimestampTracking(this.renderer, false);
    this.remaining = 0;
    this.phase = 'error';
    this.error = error instanceof Error ? error : new Error(String(error));
    const message = this.error.message;
    this.onStatus?.(`Trial failed while resolving timestamps: ${message}`);
    this.onError?.(this.error);
  }

  get active() {
    return this.phase === 'warmup' || this.phase === 'measure';
  }

  get resolving() {
    return this.phase.startsWith('resolving');
  }

  async start(context) {
    if (!['idle', 'complete', 'error'].includes(this.phase)) throw new Error('A trial is already active.');
    this.phase = 'resolving-start';
    this.error = null;
    this.context = context;
    this.rows = [];
    try {
      setTimestampTracking(this.renderer, true);
      await resolveTimestampMaps(this.renderer, { includeCompute: true, collect: false });
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
    if (this.phase === 'measure') {
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
    await resolveTimestampMaps(this.renderer, {
      includeCompute: this.context.usesCompute,
      collect: false,
    });
    this.rows = [];
    this.phase = 'measure';
    this.remaining = this.measuredFrames;
    this.onStatus?.(`Measuring ${this.measuredFrames} frames.`);
  }

  async finishMeasurement() {
    const maps = await resolveTimestampMaps(this.renderer, {
      includeCompute: this.context.usesCompute,
      collect: true,
    });
    const joined = this.rows.map((row) => {
      const gpuComputeMs = this.context.usesCompute ? (maps.compute.get(row.gpuFrameId) ?? null) : null;
      const gpuRenderMs = maps.render.get(row.gpuFrameId) ?? null;
      const gpuPassTotalMs = gpuRenderMs !== null
        && (!this.context.usesCompute || gpuComputeMs !== null)
        ? gpuRenderMs + (gpuComputeMs ?? 0)
        : null;
      return {
        ...row,
        gpuComputeMs,
        gpuRenderMs,
        gpuPassTotalMs,
      };
    });
    const missingRenderFrames = joined.filter((row) => row.gpuRenderMs === null).length;
    const missingComputeFrames = this.context.usesCompute
      ? joined.filter((row) => row.gpuComputeMs === null).length
      : 0;
    const resolution = timestampResolution(maps);
    const gpuTotals = joined.map((row) => row.gpuPassTotalMs).filter(Number.isFinite);
    const summary = {
      rowCount: joined.length,
      missingRenderFrames,
      missingComputeFrames,
      timestampAvailable: timestampSupport(this.renderer),
      ...resolution,
      cpuSubmitP50Ms: percentile(joined.map((row) => row.cpuSubmitTotalMs), 0.5),
      cpuSubmitP95Ms: percentile(joined.map((row) => row.cpuSubmitTotalMs), 0.95),
      gpuPassP50Ms: percentile(gpuTotals, 0.5),
      gpuPassP95Ms: percentile(gpuTotals, 0.95),
      accepted: joined.length === this.measuredFrames
        && missingRenderFrames === 0
        && missingComputeFrames === 0,
    };
    setTimestampTracking(this.renderer, false);
    this.phase = 'complete';
    this.onStatus?.(summary.accepted ? 'Trial timing complete.' : 'Trial rejected: missing timestamp frames.');
    this.onComplete?.({ rows: joined, summary, context: this.context });
  }
}
