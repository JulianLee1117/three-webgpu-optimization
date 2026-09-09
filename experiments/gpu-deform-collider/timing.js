// Prospective throughput screen. This helper does not establish correctness or
// complete PROTOCOL.md. The caller owns the device, validation and watchdog.
const METHODS = Object.freeze(['refit', 'interval', 'swept']);
const ORDERS = Object.freeze([
  ['refit', 'interval', 'swept'],
  ['interval', 'swept', 'refit'],
  ['swept', 'refit', 'interval'],
  ['refit', 'swept', 'interval'],
  ['swept', 'interval', 'refit'],
  ['interval', 'refit', 'swept'],
].map(Object.freeze));

function median(values) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function synchronous(callback, args, name) {
  const result = callback(...args);
  if (result && typeof result.then === 'function') throw Error(`${name} must submit synchronously, not return a Promise.`);
}

/**
 * prepare(method,.7,1.5) is called ONCE at the beginning of each batch. For
 * swept, it should construct the window bounds once; step must then reuse them.
 * step(method,time) submits deformation, method-specific updates and identical
 * queries. Times are evenly spaced and include both endpoints. Both callbacks
 * must submit to this device's queue synchronously, with already-built kernels
 * and buffers. They must not await, read back, render, or change query workload.
 *
 * Six rounds use all lane-order permutations once. Smaller rounds are permitted
 * for an explicitly partial canary. One full batch per method warms up first.
 * CPU submission time measures callbacks, excluding this helper's markers.
 * Wall time includes queue completion (and timestamp instrumentation if enabled).
 * GPU timestamps bracket all submissions, so they measure ELAPSED GPU TIMELINE
 * including CPU submission gaps, never the sum of active kernel execution time.
 * Timestamp mapping happens after the wall timer stops and before the next batch.
 * No GPU buffers are allocated during any timed interval. All owned resources
 * are disposed on success or failure; this helper never destroys the device.
 */
export async function runPairedBatches({device, prepare, step, steps = 16, rounds = 6}) {
  if (!device?.queue || typeof device.queue.onSubmittedWorkDone !== 'function') throw Error('Provide an initialized GPUDevice.');
  for (const [name, callback] of [['prepare', prepare], ['step', step]]) {
    if (typeof callback !== 'function' || callback.constructor?.name === 'AsyncFunction') throw Error(`${name} must be a synchronous function.`);
  }
  if (!Number.isInteger(steps) || steps < 2 || steps > 32) throw Error('steps must be an integer from 2 to 32.');
  if (!Number.isInteger(rounds) || rounds < 1 || rounds > 6) throw Error('rounds must be an integer from 1 to 6.');
  const start = .7, end = 1.5;
  const times = Array.from({length: steps}, (_, i) => i === steps - 1 ? end : start + (end - start) * i / (steps - 1));
  const timestampsEnabled = device.features?.has('timestamp-query') === true;
  const batches = [];
  let querySet = null, resolveBuffer = null, readback = null;

  try {
    if (timestampsEnabled) {
      querySet = device.createQuerySet({label: 'paired-batch-timeline', type: 'timestamp', count: 2});
      resolveBuffer = device.createBuffer({label: 'paired-batch-timestamp-resolve', size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC});
      readback = device.createBuffer({label: 'paired-batch-timestamp-readback', size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    }

    function markers() {
      if (!timestampsEnabled) return null;
      // Build measurement command buffers before starting any CPU wall timer.
      const before = device.createCommandEncoder({label: 'paired-batch-begin'});
      before.beginComputePass({label: 'timeline-begin-marker', timestampWrites: {querySet, beginningOfPassWriteIndex: 0}}).end();
      const after = device.createCommandEncoder({label: 'paired-batch-end'});
      after.beginComputePass({label: 'timeline-end-marker', timestampWrites: {querySet, endOfPassWriteIndex: 1}}).end();
      after.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
      after.copyBufferToBuffer(resolveBuffer, 0, readback, 0, 16);
      return [before.finish(), after.finish()];
    }

    async function batch(method, block, position) {
      const commands = markers();
      const wallStart = performance.now();
      if (commands) device.queue.submit([commands[0]]);
      const submitStart = performance.now();
      synchronous(prepare, [method, start, end], 'prepare');
      for (const time of times) synchronous(step, [method, time], 'step');
      const cpuSubmitMs = performance.now() - submitStart;
      if (commands) device.queue.submit([commands[1]]);
      await device.queue.onSubmittedWorkDone();
      const wallBatchMs = performance.now() - wallStart;
      let gpuTimelineMs = null;
      if (timestampsEnabled) {
        await readback.mapAsync(GPUMapMode.READ);
        try {
          const stamps = new BigUint64Array(readback.getMappedRange(0, 16));
          const elapsedNs = stamps[1] - stamps[0];
          if (elapsedNs < 0n || elapsedNs > BigInt(Number.MAX_SAFE_INTEGER)) throw Error('Invalid or out-of-range GPU timestamp duration.');
          gpuTimelineMs = Number(elapsedNs) / 1e6;
        } finally { readback.unmap(); }
      }
      if (![cpuSubmitMs, wallBatchMs].every(value => Number.isFinite(value) && value >= 0)) throw Error('Invalid CPU timing duration.');
      return {block, position, method, cpuSubmitMs, cpuSubmitPerStepMs: cpuSubmitMs / steps,
        wallBatchMs, wallPerStepMs: wallBatchMs / steps, gpuTimelineMs,
        gpuTimelinePerStepMs: gpuTimelineMs === null ? null : gpuTimelineMs / steps};
    }

    await device.queue.onSubmittedWorkDone();
    for (const [position, method] of METHODS.entries()) await batch(method, -1, position);
    const orders = ORDERS.slice(0, rounds).map(order => [...order]);
    for (const [block, order] of orders.entries()) {
      for (const [position, method] of order.entries()) batches.push(await batch(method, block, position));
    }
    const metrics = ['cpuSubmitMs', 'cpuSubmitPerStepMs', 'wallBatchMs', 'wallPerStepMs', 'gpuTimelineMs', 'gpuTimelinePerStepMs'];
    const medians = Object.fromEntries(METHODS.map(method => [method, Object.fromEntries(metrics.map(metric =>
      [metric, median(batches.filter(batch => batch.method === method).map(batch => batch[metric]).filter(value => value !== null))]))]));
    const pairedDifferences = orders.map((_, block) => {
      const lanes = Object.fromEntries(batches.filter(batch => batch.block === block).map(batch => [batch.method, batch]));
      return {block, ...Object.fromEntries(['interval', 'swept'].map(method => [method, Object.fromEntries(metrics.map(metric =>
        [metric, lanes[method][metric] === null ? null : lanes[method][metric] - lanes.refit[metric]]))]))};
    });
    return {kind: 'gpu-deform-collider-paired-throughput-v1', status: 'completed', steps, rounds,
      methods: [...METHODS], start, end, times, orders, completeSixPermutations: rounds === 6,
      warmupBatches: METHODS.length, timestampsEnabled, batches, medians, pairedDifferences,
      metrics: {
        cpuSubmitMs: 'CPU callback execution and submission for prepare + all steps; excludes measurement-marker encoding/submission.',
        wallBatchMs: 'CPU-observed completed prepare + all steps; includes timestamp-marker submissions, resolution/copy and queue-completion latency when enabled; excludes timestamp mapping.',
        gpuTimelineMs: 'Elapsed GPU timeline from beginning marker to ending marker, including submission gaps; NOT active kernel execution time. Null if timestamp-query was not enabled on this device.',
        perStep: 'Batch metric divided by steps, including amortized prepare cost.',
        pairedDifferences: 'Comparator minus exact refit within the same permutation block; negative favors the comparator.',
      },
      scope: 'Prospective single-session throughput diagnostic; correctness must be established separately and this does not confirm the full PROTOCOL.md matrix.'};
  } finally {
    if (readback?.mapState === 'mapped') readback.unmap();
    readback?.destroy(); resolveBuffer?.destroy(); querySet?.destroy();
  }
}
