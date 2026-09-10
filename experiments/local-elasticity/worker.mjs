import { prepareBasis, evaluateBasis } from './basis.mjs';
import { createMechanics } from './mechanics.mjs';
import { sampleQuadrature } from './assets.mjs';

let mechanics, state, points, pins, stepCount = 0;
onmessage = async ({ data }) => {
  try {
    if (data.type === 'prepare') {
      const start = performance.now();
      const quadrature = sampleQuadrature(data.positions, data.quadratureCount ?? 256);
      postMessage({ type: 'progress', text: 'Constructing a mechanical basis from this file…' });
      const basis = prepareBasis({ positions: quadrature.positions, volumes: quadrature.volumes, centerCount: 48, modeCount: 6 });
      const quadratureFields = evaluateBasis(basis, quadrature.positions);
      const basisMs = performance.now() - start;
      mechanics = createMechanics({ restPositions: quadrature.positions, volumes: quadrature.volumes, ...quadratureFields, density: 1, young: data.young ?? 40, poisson: 0.3 });
      state = { q: new Float64Array(12 * basis.modeCount), velocity: new Float64Array(12 * basis.modeCount) };
      points = quadrature.positions;
      const lowest = Array.from({ length: points.length / 3 }, (_, i) => i).sort((a, b) => points[a * 3 + 1] - points[b * 3 + 1]);
      pins = lowest.slice(0, 3).map(index => ({ index, target: Array.from(points.subarray(index * 3, index * 3 + 3)) }));
      const count = data.positions.length / 3, H = basis.modeCount;
      const weights = new Float64Array(count * H), gradients = new Float64Array(count * H * 3);
      for (let first = 0; first < count; first += 1024) {
        const chunk = evaluateBasis(basis, data.positions.subarray(first * 3, Math.min(count, first + 1024) * 3));
        weights.set(chunk.weights, first * H); gradients.set(chunk.gradients, first * H * 3);
        if (first % 16384 === 0) postMessage({ type: 'progress', text: `Binding the captured appearance: ${Math.round(100 * first / count)}%` });
      }
      stepCount = 0;
      postMessage({ type: 'ready', weights, gradients, modeCount: H, q: state.q, quadrature: points,
        diagnostics: { basisMs, prepareMs: performance.now() - start, pointCount: count, quadratureCount: points.length / 3, centerCount: basis.centerCount, modeCount: H, residuals: basis.residuals, eigenvalues: basis.eigenvalues, mechanics: mechanics.diagnostics, assumption: quadrature.assumption } });
    } else if (data.type === 'step') {
      if (!mechanics) throw Error('Prepare an object first.');
      const start = performance.now();
      const result = mechanics.step(state, { dt: 1 / 60, gravity: [0, -1.5, 0], pins: data.pinned ? pins : [], drag: data.drag ?? null, floor: -0.62, maxIterations: 10 });
      state = result.state;
      if (!state.q.every(Number.isFinite) || Math.max(...state.q.map(Math.abs)) > 100) throw Error('Simulation left the bounded numerical envelope.');
      stepCount++;
      postMessage({ type: 'state', q: state.q, stepCount, simulatedSeconds: stepCount / 60, solveMs: performance.now() - start, diagnostics: result.diagnostics });
    } else if (data.type === 'reset') {
      state.q.fill(0); state.velocity.fill(0); stepCount = 0;
      postMessage({ type: 'state', q: state.q, stepCount, simulatedSeconds: 0, solveMs: 0, diagnostics: { reset: true } });
    }
  } catch (error) { postMessage({ type: 'error', message: String(error.stack ?? error) }); }
};
