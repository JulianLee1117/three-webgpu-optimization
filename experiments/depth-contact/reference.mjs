/**
 * CPU reference for a DELIBERATELY ASYNCHRONOUS depth-contact adapter.
 *
 * This never passes stale data to an API promising a current-frame depth map.
 * Captured camera coordinates are used by every method, including the baseline.
 * Preserving those coordinates is required bookkeeping, not the novel candidate.
 *
 * Scene: finite, axis-aligned solid columns with analytic top heightfields. The
 * oracle answers membership for vertical point probes, not sphere/rigid-body
 * collision or contacts against vertical side walls. Scripted probes run at 60Hz
 * without solver feedback so every method receives exactly the same queries.
 * Residual penetration is measured AFTER one proposed upward correction.
 *
 * Snapshot contract:
 * { captureTime, availableTime, width, height,
 *   camera: { x,y,z,yaw,width,height }, depth: Float32Array(width*height) }
 * depth is linear eye distance along a downward orthographic ray; Infinity means
 * background. Pixel centers are (column+.5)/width, (row+.5)/height. Camera yaw
 * rotates the horizontal ray grid around world +Y. No object IDs, true velocities,
 * object correspondences, scene handles or future samples enter the adapters.
 *
 * Query contract: adapter.ingest(snapshot); adapter.query({x,y,z,id?}, nowSeconds)
 * returns {valid, contact, height, correctedY, ageSeconds, reason}. A valid height
 * is in world units. Contact is y <= height + contactSlop. CorrectedY never moves
 * a probe downward. Missing/expired depth produces no contact and no correction.
 *
 * The candidate estimates ONE global translation from the depth mask's world
 * centroid and mean height in two snapshots. It is intentionally falsified by
 * independent object motion, reversal and deformation controls. This synthetic
 * feasibility screen cannot establish novelty, real-scene tracking quality,
 * end-to-end WebGPU cost, or production collision correctness.
 *
 * Node: node experiments/depth-contact/reference.mjs [--self-test] [--quick]
 *       [--scene=motion-reversal] [--rate=5] [--resolution=64] [--seconds=4]
 * No dependencies, file writes, network requests, workers, or GPU work.
 */

export const PROTOCOL = Object.freeze({
  version: 'delayed-depth-contact-reference-v1',
  simulationHz: 60,
  snapshotRatesHz: [5, 10],
  deliveryDelayMs: 80,
  measuredSeconds: 4,
  warmupSeconds: 1,
  resolution: 64,
  probeColumns: 12,
  probeRows: 8,
  probeHeightBands: [0.18, 0.34, 0.49],
  contactSlop: 0.002,
  edgeBandWorld: 0.12,
  postReversalWindowSeconds: 0.35,
  requiredFalsePositiveReduction: 0.50,
  maximumAdditionalMissRate: 0.05,
  minimumBaselineFalsePositivesForReductionGate: 20,
  maximumControlFalsePositiveRateIncrease: 0.002,
  maximumResidualPenetrationIncreaseWorld: 0.005,
  cost: {
    normalizedQueryBatchSize: 256,
    maximumP95QueryBatchMs: 1.0,
    maximumP95SnapshotIngestMs: 2.0,
    scope: 'Single-thread Node CPU adapter only; excludes depth capture/readback and GPU work.'
  },
  baselineSelection: 'Per scenario/rate: fewest false positives among prespecified baselines with miss rate <= captured hold + 0.05. Ties favor fewer misses. This optimistic selection uses evaluation labels and makes the kill/go comparison harder; it is not a deployable selector.',
  decision: 'Require separate motion-scenario gates, edge/reversal controls and cost gates. No pooling can hide a scenario failure. Passing is only a reason for a narrow next experiment, not a novel-research claim.'
});

export const SCENE_NAMES = Object.freeze([
  'static', 'camera-motion', 'linear-translation', 'vertical-translation',
  'edge-sweep', 'motion-reversal', 'opposing-surfaces', 'deforming-surface'
]);
export const METHOD_NAMES = Object.freeze([
  'captured-hold', 'captured-expiry', 'captured-smoothing',
  'captured-smoothing-expiry', 'global-constant-velocity'
]);

const assert = (condition, message) => { if (!condition) throw new Error(message); };
const finite = value => typeof value === 'number' && Number.isFinite(value);
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const fraction = value => value - Math.floor(value);
const triangle = value => 1 - 4 * Math.abs(fraction(value) - 0.5);
const now = () => performance.now();
const median = values => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
const percentile = (values, p) => values.length ? [...values].sort((a, b) => a - b)[Math.ceil(p * values.length) - 1] : 0;

/** Plain-data scene definitions; the formulas in surfacePose/topHeight map to WGSL. */
export function makeScene(name = 'linear-translation') {
  assert(SCENE_NAMES.includes(name), `Unknown scene: ${name}`);
  const platform = overrides => ({
    x: 0, z: 0, y: 0.35, halfWidth: 0.72, halfDepth: 0.68,
    tiltX: 0.025, tiltZ: -0.015,
    motion: { kind: 'linear', vx: 0, vy: 0, vz: 0 }, deformation: null, ...overrides
  });
  let surfaces = [platform({})];
  if (name === 'linear-translation') surfaces = [platform({ x: -1.1, motion: { kind: 'linear', vx: 0.4, vy: 0, vz: 0 } })];
  if (name === 'vertical-translation') surfaces = [platform({ y: 0.52, tiltX: 0, tiltZ: 0, motion: { kind: 'linear', vx: 0, vy: -0.07, vz: 0 } })];
  if (name === 'edge-sweep') surfaces = [platform({ x: -1.25, halfWidth: 0.28, motion: { kind: 'linear', vx: 0.5, vy: 0, vz: 0 } })];
  if (name === 'motion-reversal') surfaces = [platform({ motion: { kind: 'triangle', amplitudeX: 0.78, period: 2.4, phase: 0.25 } })];
  if (name === 'opposing-surfaces') surfaces = [
    platform({ x: -0.85, z: -0.52, halfWidth: 0.42, halfDepth: 0.42, motion: { kind: 'linear', vx: 0.34, vy: 0, vz: 0 } }),
    platform({ x: 0.85, z: 0.52, halfWidth: 0.42, halfDepth: 0.42, motion: { kind: 'linear', vx: -0.34, vy: 0, vz: 0 } })
  ];
  if (name === 'deforming-surface') surfaces = [platform({ deformation: { amplitude: 0.13, angularSpeed: 2.2, spatialFrequencyX: 1.6 } })];
  return { name, surfaces, movingCamera: name === 'camera-motion', control: ['static', 'camera-motion'].includes(name) };
}

export function surfacePose(surface, time) {
  const motion = surface.motion;
  if (motion.kind === 'triangle') {
    return { x: surface.x + motion.amplitudeX * triangle(time / motion.period + motion.phase), y: surface.y, z: surface.z };
  }
  return { x: surface.x + motion.vx * time, y: surface.y + motion.vy * time, z: surface.z + motion.vz * time };
}

function topHeight(surface, pose, x, z, time) {
  const u = x - pose.x, v = z - pose.z;
  let height = pose.y + surface.tiltX * u + surface.tiltZ * v;
  if (surface.deformation) {
    const deformation = surface.deformation;
    height += deformation.amplitude * Math.sin(deformation.angularSpeed * time + deformation.spatialFrequencyX * u);
  }
  return height;
}

/** Highest visible top height; null denotes no solid column at this world X/Z. */
export function surfaceHeight(scene, x, z, time) {
  let height = null;
  for (const surface of scene.surfaces) {
    const pose = surfacePose(surface, time);
    if (Math.abs(x - pose.x) > surface.halfWidth || Math.abs(z - pose.z) > surface.halfDepth) continue;
    const candidate = topHeight(surface, pose, x, z, time);
    if (height === null || candidate > height) height = candidate;
  }
  return height;
}

export function cameraAt(scene, time) {
  return {
    x: scene.movingCamera ? 0.38 * Math.sin(0.9 * time) : 0,
    y: 3,
    z: scene.movingCamera ? 0.30 * Math.cos(0.75 * time) : 0,
    yaw: scene.movingCamera ? 0.18 * Math.sin(1.1 * time) : 0,
    width: 5, height: 5
  };
}

export function pixelToWorld(camera, u, v) {
  const a = (u - 0.5) * camera.width, b = (v - 0.5) * camera.height;
  const c = Math.cos(camera.yaw), s = Math.sin(camera.yaw);
  return { x: camera.x + c * a - s * b, z: camera.z + s * a + c * b };
}

export function worldToPixel(camera, x, z) {
  const dx = x - camera.x, dz = z - camera.z, c = Math.cos(camera.yaw), s = Math.sin(camera.yaw);
  return { u: (c * dx + s * dz) / camera.width + 0.5, v: (-s * dx + c * dz) / camera.height + 0.5 };
}

export function captureDepth(scene, captureTime, options = {}) {
  const resolution = options.resolution ?? PROTOCOL.resolution;
  assert(Number.isInteger(resolution) && resolution >= 16 && resolution <= 256, 'Resolution must be an integer between 16 and 256.');
  const camera = cameraAt(scene, captureTime), depth = new Float32Array(resolution * resolution);
  for (let row = 0; row < resolution; row++) for (let column = 0; column < resolution; column++) {
    const point = pixelToWorld(camera, (column + 0.5) / resolution, (row + 0.5) / resolution);
    const height = surfaceHeight(scene, point.x, point.z, captureTime);
    depth[row * resolution + column] = height === null ? Infinity : camera.y - height;
  }
  return {
    captureTime, availableTime: options.availableTime ?? captureTime,
    width: resolution, height: resolution, camera, depth
  };
}

/** All four neighbors must exist and lie within a declared height jump guard. */
export function sampleSnapshot(snapshot, x, z, options = {}) {
  const uv = worldToPixel(snapshot.camera, x, z);
  const px = uv.u * snapshot.width - 0.5, py = uv.v * snapshot.height - 0.5;
  const x0 = Math.floor(px), y0 = Math.floor(py), tx = px - x0, ty = py - y0;
  if (x0 < 0 || y0 < 0 || x0 + 1 >= snapshot.width || y0 + 1 >= snapshot.height) return { valid: false, height: null, reason: 'outside-capture' };
  const offset = y0 * snapshot.width + x0;
  const values = [snapshot.depth[offset], snapshot.depth[offset + 1], snapshot.depth[offset + snapshot.width], snapshot.depth[offset + snapshot.width + 1]];
  if (!values.every(Number.isFinite)) return { valid: false, height: null, reason: 'background-or-edge' };
  if (Math.max(...values) - Math.min(...values) > (options.maximumHeightJump ?? 0.15)) return { valid: false, height: null, reason: 'height-discontinuity' };
  const depth = (1 - ty) * ((1 - tx) * values[0] + tx * values[1]) + ty * ((1 - tx) * values[2] + tx * values[3]);
  return { valid: true, height: snapshot.camera.y - depth, reason: 'sampled' };
}

/** Global statistics from visible depth pixels only; object labels are unavailable. */
export function summarizeDepth(snapshot) {
  let count = 0, sumX = 0, sumY = 0, sumZ = 0;
  for (let row = 0; row < snapshot.height; row++) for (let column = 0; column < snapshot.width; column++) {
    const depth = snapshot.depth[row * snapshot.width + column];
    if (!Number.isFinite(depth)) continue;
    const point = pixelToWorld(snapshot.camera, (column + 0.5) / snapshot.width, (row + 0.5) / snapshot.height);
    count++; sumX += point.x; sumY += snapshot.camera.y - depth; sumZ += point.z;
  }
  return count ? { count, x: sumX / count, y: sumY / count, z: sumZ / count } : null;
}

export function makeAdapter(method = 'captured-hold', options = {}) {
  assert(METHOD_NAMES.includes(method), `Unknown method: ${method}`);
  const settings = {
    contactSlop: PROTOCOL.contactSlop, expirySeconds: 0.15,
    smoothingLatestWeight: 0.75, maximumPredictionAgeSeconds: 0.35,
    maximumHorizontalSpeed: 2.0, maximumVerticalSpeed: 0.6,
    centroidDeadbandPixels: 0.25, maximumHeightJump: 0.15, ...options
  };
  let latest = null, previous = null, latestSummary = null;
  let velocity = { x: 0, y: 0, z: 0 };
  const predicts = method === 'global-constant-velocity';
  const smooths = method.includes('smoothing'), expires = method.includes('expiry');
  return {
    method,
    ingest(snapshot) {
      assert(snapshot.depth instanceof Float32Array && snapshot.depth.length === snapshot.width * snapshot.height, 'Invalid depth snapshot.');
      assert(finite(snapshot.captureTime) && finite(snapshot.availableTime) && snapshot.availableTime >= snapshot.captureTime, 'Invalid capture/delivery times.');
      assert(!latest || snapshot.captureTime > latest.captureTime, 'Snapshots must arrive in strictly increasing capture order.');
      previous = latest; latest = snapshot;
      if (predicts) {
        const oldSummary = latestSummary;
        latestSummary = summarizeDepth(snapshot);
        velocity = { x: 0, y: 0, z: 0 };
        if (previous && oldSummary && latestSummary) {
          const dt = latest.captureTime - previous.captureTime;
          const dx = latestSummary.x - oldSummary.x, dz = latestSummary.z - oldSummary.z;
          const deadband = settings.centroidDeadbandPixels * Math.max(snapshot.camera.width / snapshot.width, snapshot.camera.height / snapshot.height);
          if (Math.hypot(dx, dz) > deadband) {
            velocity.x = dx / dt; velocity.z = dz / dt;
            const speed = Math.hypot(velocity.x, velocity.z);
            if (speed > settings.maximumHorizontalSpeed) {
              velocity.x *= settings.maximumHorizontalSpeed / speed;
              velocity.z *= settings.maximumHorizontalSpeed / speed;
            }
          }
          velocity.y = clamp((latestSummary.y - oldSummary.y) / dt, -settings.maximumVerticalSpeed, settings.maximumVerticalSpeed);
        }
      }
    },
    query(probe, time) {
      assert(finite(probe.x) && finite(probe.y) && finite(probe.z) && finite(time), 'Probe/time must be finite.');
      const ageSeconds = latest ? time - latest.captureTime : null;
      const reject = reason => ({ valid: false, contact: false, height: null, correctedY: probe.y, ageSeconds, reason });
      if (!latest) return reject('no-snapshot');
      assert(time + 1e-9 >= latest.availableTime, 'A future/unavailable snapshot reached a query.');
      if (expires && ageSeconds > settings.expirySeconds) return reject('expired');
      if (predicts && ageSeconds > settings.maximumPredictionAgeSeconds) return reject('prediction-horizon');
      let sample = sampleSnapshot(latest,
        probe.x - (predicts ? velocity.x * ageSeconds : 0),
        probe.z - (predicts ? velocity.z * ageSeconds : 0), settings);
      if (!sample.valid) return reject(sample.reason);
      if (predicts) sample.height += velocity.y * ageSeconds;
      if (smooths && previous) {
        const old = sampleSnapshot(previous, probe.x, probe.z, settings);
        if (old.valid) sample.height = settings.smoothingLatestWeight * sample.height + (1 - settings.smoothingLatestWeight) * old.height;
      }
      const contact = probe.y <= sample.height + settings.contactSlop;
      return {
        valid: true, contact, height: sample.height,
        correctedY: contact ? Math.max(probe.y, sample.height) : probe.y,
        ageSeconds, reason: predicts ? 'globally-predicted' : smooths ? 'height-smoothed' : 'captured-camera'
      };
    },
    get debugState() { return { latestCaptureTime: latest?.captureTime ?? null, velocity: { ...velocity } }; }
  };
}

export function makeProbes(time, options = {}) {
  const columns = options.probeColumns ?? PROTOCOL.probeColumns, rows = options.probeRows ?? PROTOCOL.probeRows;
  const bands = options.probeHeightBands ?? PROTOCOL.probeHeightBands, probes = [];
  for (let row = 0; row < rows; row++) for (let column = 0; column < columns; column++) for (let band = 0; band < bands.length; band++) {
    const id = (row * columns + column) * bands.length + band;
    probes.push({ id,
      x: -1.65 + (column + 0.5) * 3.3 / columns + 0.012 * Math.sin(0.5 * time + id * 0.1),
      z: -1.05 + (row + 0.5) * 2.1 / rows,
      y: bands[band] + 0.025 * Math.sin(0.9 * time + id * 0.37)
    });
  }
  return probes;
}

function nearFootprintEdge(scene, probe, time, band) {
  return scene.surfaces.some(surface => {
    const pose = surfacePose(surface, time);
    const dx = Math.abs(probe.x - pose.x) - surface.halfWidth, dz = Math.abs(probe.z - pose.z) - surface.halfDepth;
    const distance = dx <= 0 && dz <= 0 ? -Math.max(dx, dz) : Math.hypot(Math.max(0, dx), Math.max(0, dz));
    return distance <= band;
  });
}

export function isPostReversal(scene, time, window = PROTOCOL.postReversalWindowSeconds) {
  return scene.surfaces.some(surface => {
    const motion = surface.motion;
    if (motion.kind !== 'triangle') return false;
    const phase = fraction(time / motion.period + motion.phase);
    const elapsed = phase < 0.5 ? phase * motion.period : (phase - 0.5) * motion.period;
    return elapsed <= window;
  });
}

export function oracleContact(scene, probe, time, options = {}) {
  const height = surfaceHeight(scene, probe.x, probe.z, time);
  return {
    height, contact: height !== null && probe.y <= height + (options.contactSlop ?? PROTOCOL.contactSlop),
    penetration: height === null ? 0 : Math.max(0, height - probe.y),
    edge: nearFootprintEdge(scene, probe, time, options.edgeBandWorld ?? PROTOCOL.edgeBandWorld)
  };
}

function configuration(options = {}) {
  const config = {
    scene: 'linear-translation', snapshotHz: 5, deliveryDelayMs: PROTOCOL.deliveryDelayMs,
    durationSeconds: PROTOCOL.measuredSeconds, warmupSeconds: PROTOCOL.warmupSeconds,
    resolution: PROTOCOL.resolution, ...options
  };
  assert([5, 10].includes(config.snapshotHz), 'Prespecified snapshot rate must be 5 or 10Hz.');
  assert(finite(config.deliveryDelayMs) && config.deliveryDelayMs >= 0 && config.deliveryDelayMs <= 200, 'Delay must be between 0 and 200ms.');
  assert(finite(config.durationSeconds) && config.durationSeconds > 0 && config.durationSeconds <= 10, 'Duration must be >0 and <=10 seconds.');
  assert(finite(config.warmupSeconds) && config.warmupSeconds >= 0.5 && config.warmupSeconds <= 2, 'Warmup must be between 0.5 and 2 seconds.');
  return config;
}

/** Lazy frames keep memory bounded; capture and delivery occur on the 60Hz clock. */
export function* generateDataset(options = {}) {
  const config = configuration(options), scene = typeof config.scene === 'string' ? makeScene(config.scene) : config.scene;
  const periodFrames = PROTOCOL.simulationHz / config.snapshotHz;
  const frames = Math.ceil((config.durationSeconds + config.warmupSeconds) * PROTOCOL.simulationHz);
  const pending = [];
  for (let frame = 0; frame < frames; frame++) {
    const time = frame / PROTOCOL.simulationHz;
    if (frame % periodFrames === 0) pending.push(captureDepth(scene, time, {
      resolution: config.resolution, availableTime: time + config.deliveryDelayMs / 1000
    }));
    const deliveries = [];
    while (pending.length && pending[0].availableTime <= time + 1e-9) deliveries.push(pending.shift());
    yield {
      frame, time, measured: time + 1e-9 >= config.warmupSeconds, deliveries,
      probes: makeProbes(time, config), postReversal: isPostReversal(scene, time), scene
    };
  }
}

function emptyCounts() {
  return { queries: 0, truePositive: 0, falsePositive: 0, trueNegative: 0, falseNegative: 0,
    abstained: 0, residualPenetrationSum: 0, residualPenetrationMax: 0,
    falseLiftSum: 0, falseLiftMax: 0 };
}

function addObservation(counts, probe, oracle, prediction) {
  counts.queries++;
  counts[oracle.contact ? prediction.contact ? 'truePositive' : 'falseNegative' : prediction.contact ? 'falsePositive' : 'trueNegative']++;
  if (!prediction.valid) counts.abstained++;
  const residual = oracle.height === null ? 0 : Math.max(0, oracle.height - prediction.correctedY);
  const justifiedY = oracle.height === null ? probe.y : Math.max(probe.y, oracle.height);
  const falseLift = Math.max(0, prediction.correctedY - justifiedY);
  counts.residualPenetrationSum += residual;
  counts.residualPenetrationMax = Math.max(counts.residualPenetrationMax, residual);
  counts.falseLiftSum += falseLift;
  counts.falseLiftMax = Math.max(counts.falseLiftMax, falseLift);
}

function finishCounts(counts) {
  const positives = counts.truePositive + counts.falseNegative, negatives = counts.trueNegative + counts.falsePositive;
  return {
    ...counts, oracleContacts: positives, oracleNoncontacts: negatives,
    falsePositiveRate: negatives ? counts.falsePositive / negatives : 0,
    missRate: positives ? counts.falseNegative / positives : 0,
    meanResidualPenetrationOnOracleContacts: positives ? counts.residualPenetrationSum / positives : 0,
    meanFalseLiftPerQuery: counts.queries ? counts.falseLiftSum / counts.queries : 0
  };
}

function compareStratum(candidate, baseline, control = false) {
  const opportunity = !control && baseline.falsePositive >= PROTOCOL.minimumBaselineFalsePositivesForReductionGate;
  const falsePositivePass = opportunity
    ? candidate.falsePositiveRate <= baseline.falsePositiveRate * (1 - PROTOCOL.requiredFalsePositiveReduction)
    : candidate.falsePositiveRate <= baseline.falsePositiveRate + PROTOCOL.maximumControlFalsePositiveRateIncrease;
  const missPass = candidate.missRate <= baseline.missRate + PROTOCOL.maximumAdditionalMissRate;
  const penetrationPass = candidate.meanResidualPenetrationOnOracleContacts <= baseline.meanResidualPenetrationOnOracleContacts + PROTOCOL.maximumResidualPenetrationIncreaseWorld;
  return {
    baselineFalsePositives: baseline.falsePositive, candidateFalsePositives: candidate.falsePositive,
    falsePositiveReduction: baseline.falsePositive ? 1 - candidate.falsePositive / baseline.falsePositive : null,
    additionalMissRate: candidate.missRate - baseline.missRate,
    additionalMeanResidualPenetration: candidate.meanResidualPenetrationOnOracleContacts - baseline.meanResidualPenetrationOnOracleContacts,
    reductionOpportunity: opportunity, falsePositivePass, missPass, penetrationPass,
    pass: falsePositivePass && missPass && penetrationPass
  };
}

/** Conservative kill/go screen; the strongest baseline is explicitly oracle-selected. */
export function decisionForScene(methods, control = false) {
  const hold = methods['captured-hold'].all;
  const eligible = METHOD_NAMES.filter(name => name !== 'global-constant-velocity')
    .filter(name => methods[name].all.missRate <= hold.missRate + PROTOCOL.maximumAdditionalMissRate)
    .sort((a, b) => methods[a].all.falsePositive - methods[b].all.falsePositive || methods[a].all.falseNegative - methods[b].all.falseNegative);
  const baselineName = eligible[0], candidate = methods['global-constant-velocity'], baseline = methods[baselineName];
  const all = compareStratum(candidate.all, baseline.all, control);
  const edge = compareStratum(candidate.edge, baseline.edge, control);
  const reversal = candidate.postReversal.queries ? compareStratum(candidate.postReversal, baseline.postReversal, control) : null;
  const queryCostPass = candidate.cost.p95MsPer256QueryBatch <= PROTOCOL.cost.maximumP95QueryBatchMs;
  const ingestCostPass = candidate.cost.p95SnapshotIngestMs <= PROTOCOL.cost.maximumP95SnapshotIngestMs;
  const pass = all.pass && edge.pass && (!reversal || reversal.pass) && queryCostPass && ingestCostPass;
  return {
    baselineName, eligibleBaselineNames: eligible,
    all, edge, postReversal: reversal,
    cost: { queryCostPass, ingestCostPass },
    pass,
    meaningfulReduction: pass && all.reductionOpportunity
  };
}

export function evaluate(options = {}) {
  const config = configuration(options), scene = typeof config.scene === 'string' ? makeScene(config.scene) : config.scene;
  const adapters = Object.fromEntries(METHOD_NAMES.map(name => [name, makeAdapter(name, config.adapterOptions)]));
  const accumulators = Object.fromEntries(METHOD_NAMES.map(name => [name, {
    all: emptyCounts(), edge: emptyCounts(), postReversal: emptyCounts(),
    queryTimes: [], ingestTimes: [], normalizedQueryTimes: []
  }]));
  const wallStart = now();
  let measuredFrames = 0, deliveredSnapshots = 0;
  for (const frame of generateDataset({ ...config, scene })) {
    for (const snapshot of frame.deliveries) {
      deliveredSnapshots++;
      for (const name of METHOD_NAMES) {
        const start = now(); adapters[name].ingest(snapshot); const elapsed = now() - start;
        if (frame.measured) accumulators[name].ingestTimes.push(elapsed);
      }
    }
    if (!frame.measured) {
      // Exercise the same paths before recording cost, without collecting labels.
      for (const name of METHOD_NAMES) for (const probe of frame.probes) adapters[name].query(probe, frame.time);
      continue;
    }
    measuredFrames++;
    const oracle = frame.probes.map(probe => oracleContact(scene, probe, frame.time));
    for (const name of METHOD_NAMES) {
      const accumulator = accumulators[name], start = now();
      const predictions = frame.probes.map(probe => adapters[name].query(probe, frame.time));
      const elapsed = now() - start;
      accumulator.queryTimes.push(elapsed);
      accumulator.normalizedQueryTimes.push(elapsed / frame.probes.length * PROTOCOL.cost.normalizedQueryBatchSize);
      for (let index = 0; index < frame.probes.length; index++) {
        addObservation(accumulator.all, frame.probes[index], oracle[index], predictions[index]);
        if (oracle[index].edge) addObservation(accumulator.edge, frame.probes[index], oracle[index], predictions[index]);
        if (frame.postReversal) addObservation(accumulator.postReversal, frame.probes[index], oracle[index], predictions[index]);
      }
    }
  }
  const methods = Object.fromEntries(METHOD_NAMES.map(name => {
    const record = accumulators[name];
    return [name, {
      all: finishCounts(record.all), edge: finishCounts(record.edge), postReversal: finishCounts(record.postReversal),
      cost: {
        measuredQueryBatches: record.queryTimes.length,
        measuredSnapshotIngests: record.ingestTimes.length,
        medianMsPer256QueryBatch: median(record.normalizedQueryTimes),
        p95MsPer256QueryBatch: percentile(record.normalizedQueryTimes, 0.95),
        medianSnapshotIngestMs: median(record.ingestTimes),
        p95SnapshotIngestMs: percentile(record.ingestTimes, 0.95)
      }
    }];
  }));
  return {
    scene: scene.name, control: scene.control, config: { ...config, scene: scene.name },
    measuredFrames, deliveredSnapshots, methods, decision: decisionForScene(methods, scene.control),
    elapsedCpuWallMs: now() - wallStart
  };
}

export function runScreen(options = {}) {
  const scenes = options.scenes ?? SCENE_NAMES, rates = options.rates ?? PROTOCOL.snapshotRatesHz;
  const results = [];
  for (const scene of scenes) for (const snapshotHz of rates) results.push(evaluate({ ...options, scene, snapshotHz }));
  const failed = results.filter(result => !result.decision.pass);
  const meaningful = results.filter(result => result.decision.meaningfulReduction);
  const complete = results.length === SCENE_NAMES.length * PROTOCOL.snapshotRatesHz.length &&
    SCENE_NAMES.every(scene => PROTOCOL.snapshotRatesHz.every(rate => results.some(result => result.scene === scene && result.config.snapshotHz === rate))) &&
    results.every(result => result.config.durationSeconds === PROTOCOL.measuredSeconds &&
      result.config.warmupSeconds === PROTOCOL.warmupSeconds && result.config.resolution === PROTOCOL.resolution &&
      result.config.deliveryDelayMs === PROTOCOL.deliveryDelayMs && !result.config.adapterOptions &&
      result.config.probeColumns === undefined && result.config.probeRows === undefined && result.config.probeHeightBands === undefined);
  return {
    kind: 'delayed-depth-contact-cpu-screen-v1', protocol: PROTOCOL,
    limitations: [
      'Deterministic analytic finite heightfields and vertical point probes, not realistic image-depth noise or full collision dynamics.',
      'One global centroid velocity cannot represent independent object motion, deformation, occlusion, or unknown correspondence.',
      'All lanes use captured camera coordinates and the same conservative edge-aware sampler.',
      'Snapshots at 5/10Hz arrive after 80ms, rounded up by the 60Hz delivery clock; age is measured from capture, not arrival.',
      'Expiry/smoothing baselines are prespecified; the strongest eligible baseline is chosen with evaluation labels for an optimistic comparison.',
      'CPU timings are approximate local adapter costs, include JavaScript allocation, and exclude depth generation, transfer, readback, GPU work and browser integration.',
      'No statistical confidence or generalization claim follows from repeated deterministic probe queries.'
    ],
    decision: {
      completePrespecifiedScreen: complete,
      failedSceneRates: failed.map(result => `${result.scene}@${result.config.snapshotHz}Hz`),
      meaningfulReductionSceneRates: meaningful.map(result => `${result.scene}@${result.config.snapshotHz}Hz`),
      action: !complete ? 'incomplete-screen' : failed.length ? 'no-go-for-general-adapter' : meaningful.length ? 'narrow-followup-only' : 'no-demonstrated-opportunity'
    }, results
  };
}

export function runSelfTests() {
  const close = (a, b, epsilon = 1e-6) => assert(Math.abs(a - b) <= epsilon, `${a} differs from ${b}.`);
  const camera = cameraAt(makeScene('camera-motion'), 1.2), point = pixelToWorld(camera, 0.23, 0.79);
  const uv = worldToPixel(camera, point.x, point.z); close(uv.u, 0.23); close(uv.v, 0.79);
  const scene = makeScene('static'), first = captureDepth(scene, 0), second = captureDepth(scene, 0.2);
  close(sampleSnapshot(first, 0, 0).height, surfaceHeight(scene, 0, 0, 0));
  assert(!sampleSnapshot(first, 2, 2).valid, 'Background must not become a surface.');
  const hold = makeAdapter('captured-hold'); hold.ingest(first); hold.ingest(second);
  assert(hold.query({ x: 0, y: 0.2, z: 0 }, 0.3).contact, 'A point below the top must contact.');
  assert(!hold.query({ x: 0, y: 0.6, z: 0 }, 0.3).contact, 'A point above the top must not contact.');
  const expiry = makeAdapter('captured-expiry'); expiry.ingest(first);
  assert(expiry.query({ x: 0, y: 0.2, z: 0 }, 0.2).reason === 'expired', 'Expiry must use capture age.');
  const moving = makeScene('vertical-translation'), predictor = makeAdapter('global-constant-velocity');
  predictor.ingest(captureDepth(moving, 0)); predictor.ingest(captureDepth(moving, 0.2));
  close(predictor.query({ x: 0, y: 0.2, z: 0 }, 0.3).height, surfaceHeight(moving, 0, 0, 0.3), 3e-6);
  const cameraScene = makeScene('camera-motion');
  close(sampleSnapshot(captureDepth(cameraScene, 0), 0, 0).height, sampleSnapshot(captureDepth(cameraScene, 1), 0, 0).height);
  assert(isPostReversal(makeScene('motion-reversal'), 1.81), 'Reversal window should be present after a turn.');
  assert(!isPostReversal(makeScene('motion-reversal'), 1.5), 'Reversal window should not include ordinary motion.');
  const frames = [...generateDataset({ durationSeconds: 0.1, warmupSeconds: 0.5, resolution: 16 })];
  assert(frames.every(frame => frame.deliveries.every(snapshot => snapshot.availableTime <= frame.time + 1e-9)), 'No future samples may be delivered.');
  assert(frames.find(frame => frame.deliveries.length).frame === 5, '80ms delivery should first be observed at frame 5.');
  assert(Object.keys(first).sort().join(',') === 'availableTime,camera,captureTime,depth,height,width', 'Snapshot must not leak scene/pose metadata.');
  return { status: 'passed', checks: ['camera inverse', 'analytic height', 'background rejection', 'contact signs', 'capture-age expiry', 'vertical prediction', 'camera-only invariance', 'reversal labels', 'causal delivery', 'snapshot metadata isolation'] };
}

async function main(args) {
  if (args.includes('--self-test')) { console.log(JSON.stringify(runSelfTests())); return; }
  const value = prefix => args.find(argument => argument.startsWith(prefix))?.slice(prefix.length);
  const quick = args.includes('--quick'), scene = value('--scene='), rate = value('--rate=');
  const options = {
    scenes: scene ? [scene] : quick ? ['linear-translation', 'motion-reversal', 'opposing-surfaces'] : SCENE_NAMES,
    rates: rate ? [Number(rate)] : PROTOCOL.snapshotRatesHz,
    resolution: Number(value('--resolution=') ?? PROTOCOL.resolution),
    durationSeconds: Number(value('--seconds=') ?? (quick ? 2 : PROTOCOL.measuredSeconds))
  };
  const result = runScreen(options);
  console.log(JSON.stringify(result, null, 2));
}

if (typeof process !== 'undefined' && process.argv?.[1]) {
  const { pathToFileURL } = await import('node:url');
  if (import.meta.url === pathToFileURL(process.argv[1]).href) {
    main(process.argv.slice(2)).catch(error => { console.error(error.stack ?? error); process.exitCode = 1; });
  }
}
