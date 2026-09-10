import { targetSamplesFromMask, solveSamples } from './solver.mjs';

/** CPU worker entry. Terminating the worker is the owner's cancellation path. */
export function processRequest(message, emit) {
  const {
    id, mask, width, height, resolution = 48, grid = 25, iterations = 400,
    background = 0.04, learningRate = 0.0009, smoothness = 0.00001,
  } = message;
  if ((typeof id !== 'string' && typeof id !== 'number') || (typeof id === 'number' && !Number.isFinite(id))) throw new TypeError('Request requires a finite number or string id');
  if (!Number.isInteger(resolution) || resolution < 8 || resolution > 64) throw new RangeError('Worker resolution is bounded to [8,64]');
  if (typeof emit !== 'function') throw new TypeError('A message emitter is required');
  const data = mask instanceof ArrayBuffer ? new Uint8Array(mask) : mask;
  const started = performance.now();
  const target = targetSamplesFromMask(data, width, height, resolution * resolution, { background });
  const result = solveSamples(target, {
    resolution, grid, iterations, background, learningRate, smoothness,
    onProgress: ({ iteration, height, loss }) => {
      emit({ type: 'progress', id, iteration, loss, height, grid }, [height.buffer]);
    },
  });
  result.pipelineMilliseconds = performance.now() - started;
  emit({ type: 'result', id, ...result }, [result.lens.height.buffer, result.target.buffer]);
  return result;
}

if (typeof WorkerGlobalScope !== 'undefined' && globalThis instanceof WorkerGlobalScope) {
  globalThis.onmessage = ({ data }) => {
    try { processRequest(data, (value, transfer) => globalThis.postMessage(value, transfer)); }
    catch (error) { globalThis.postMessage({ type: 'error', id: data?.id, message: String(error.message ?? error) }); }
  };
}
