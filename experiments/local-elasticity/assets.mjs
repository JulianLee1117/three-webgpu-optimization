// Fixed-width antimatter15 .splat data. No mesh conversion or asset-specific rig.
export function decodeSplat(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength % 32 || buffer.byteLength < 32 || buffer.byteLength > 5_000_000) throw Error('Use a .splat file smaller than 5 MB.');
  const count = buffer.byteLength / 32, view = new DataView(buffer);
  const positions = new Float64Array(count * 3);
  for (let i = 0; i < count; i++) {
    for (let d = 0; d < 3; d++) {
      const x = view.getFloat32(i * 32 + d * 4, true);
      const s = view.getFloat32(i * 32 + 12 + d * 4, true);
      if (!Number.isFinite(x) || !Number.isFinite(s) || s <= 0) throw Error('Invalid splat position or scale.');
      positions[3 * i + d] = x;
    }
  }
  // Dataset convention is -Y up. Both bundled objects follow it. Uploaded files
  // can select the opposite convention explicitly in the UI.
  return { positions, count };
}

export function normalizePositions(input, { flipY = true } = {}) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < input.length; i++) { const d = i % 3; min[d] = Math.min(min[d], input[i]); max[d] = Math.max(max[d], input[i]); }
  const scale = 1 / Math.max(...max.map((v, d) => v - min[d]));
  if (!Number.isFinite(scale) || scale <= 0) throw Error('Degenerate object bounds.');
  const center = min.map((v, d) => (v + max[d]) / 2);
  const positions = Float64Array.from(input, (v, i) => (v - center[i % 3]) * scale * (flipY && i % 3 === 1 ? -1 : 1));
  return { positions, scale, center, flipY };
}

export function sampleQuadrature(positions, count = 384, seed = 12345) {
  const n = positions.length / 3;
  if (!Number.isInteger(count) || count < 64 || count > 1024 || count > n) throw Error('Invalid quadrature count.');
  // Uniform sample without replacement, preserving the source point measure.
  // Equal weights are a declared modeling assumption, not recovered solid volume.
  const indices = Uint32Array.from({ length: n }, (_, i) => i);
  let state = seed >>> 0;
  const random = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 4294967296; };
  const samples = new Float64Array(count * 3);
  for (let i = 0; i < count; i++) {
    const j = i + Math.floor(random() * (n - i));
    [indices[i], indices[j]] = [indices[j], indices[i]];
    samples.set(positions.subarray(indices[i] * 3, indices[i] * 3 + 3), i * 3);
  }
  return { positions: samples, volumes: new Float64Array(count).fill(1 / count), indices: indices.slice(0, count), assumption: 'Uniform source-point quadrature; total model volume 1, uncalibrated.' };
}
