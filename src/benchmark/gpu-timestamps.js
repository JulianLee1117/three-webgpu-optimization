function timestampTypes(includeCompute) {
  return includeCompute ? ['render', 'compute'] : ['render'];
}

export function timestampSupport(renderer) {
  return renderer.backend?.trackTimestamp === true;
}

export function setTimestampTracking(renderer, enabled) {
  if (renderer.backend) renderer.backend.trackTimestamp = enabled;
}

export async function resolveTimestampMaps(renderer, { includeCompute, collect }) {
  const maps = { render: new Map(), compute: new Map() };
  if (!timestampSupport(renderer)) return maps;

  const types = timestampTypes(includeCompute);
  await Promise.all(types.map((type) => renderer.resolveTimestampsAsync(type)));
  if (!collect) return maps;

  for (const type of types) {
    const pool = renderer.backend?.timestampQueryPool?.[type];
    const frames = new Set(renderer.backend?.getTimestampFrames?.(type) ?? pool?.frames ?? []);
    for (const frame of frames) maps[type].set(frame, 0);
    for (const [uid, durationMs] of pool?.timestamps ?? []) {
      const match = uid.match(/:f(\d+)$/);
      if (!match) continue;
      const frame = Number(match[1]);
      if (!frames.has(frame)) continue;
      maps[type].set(frame, (maps[type].get(frame) ?? 0) + durationMs);
    }
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
  const durationsNs = [];
  for (const map of Object.values(maps)) {
    for (const durationMs of map.values()) {
      const nanoseconds = Math.round(durationMs * 1e6);
      if (nanoseconds > 0) durationsNs.push(nanoseconds);
    }
  }
  if (durationsNs.length === 0) {
    return { quantumNs: null, classification: 'unresolved' };
  }
  const quantumNs = durationsNs.reduce(gcd);
  return {
    quantumNs,
    classification: quantumNs >= 10_000 ? 'quantized' : 'fine',
  };
}
