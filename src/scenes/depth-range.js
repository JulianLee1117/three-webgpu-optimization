function requireScenarioArrays(scenario) {
  if (!(scenario?.bounds instanceof Float32Array)
    || scenario.bounds.length !== scenario.objectCount * 4) {
    throw new RangeError('scenario.bounds must contain one Float32 vec4 per object.');
  }
  if (!(scenario.expectedVisibleIds instanceof Uint32Array)
    || scenario.expectedVisibleIds.length === 0) {
    throw new RangeError('scenario.expectedVisibleIds must contain at least one object.');
  }
}

/**
 * Returns the fixed range used to classify visible sphere-nearest view depths.
 * The small symmetric pad keeps the extrema away from bin boundaries after f32
 * uniform conversion without changing the relative ordering between layouts.
 */
export function computeScenarioDepthBinRange(scenario, camera) {
  requireScenarioArrays(scenario);
  if (!camera?.matrixWorldInverse?.elements) {
    throw new TypeError('A camera with a matrixWorldInverse is required.');
  }

  camera.updateMatrixWorld();
  const view = camera.matrixWorldInverse.elements;
  let near = Number.POSITIVE_INFINITY;
  let far = Number.NEGATIVE_INFINITY;

  for (const objectId of scenario.expectedVisibleIds) {
    const offset = objectId * 4;
    const x = scenario.bounds[offset];
    const y = scenario.bounds[offset + 1];
    const z = scenario.bounds[offset + 2];
    const radius = scenario.bounds[offset + 3];
    const viewZ = view[2] * x + view[6] * y + view[10] * z + view[14];
    const nearestDepth = -viewZ - radius;
    near = Math.min(near, nearestDepth);
    far = Math.max(far, nearestDepth);
  }

  const span = far - near;
  const padding = Math.max(1e-3, span * 1e-4);
  if (!Number.isFinite(span) || span < 0) {
    throw new Error('Could not derive a finite visible depth range.');
  }
  if (span === 0) {
    return Object.freeze({ near: near - padding, far: far + padding });
  }
  return Object.freeze({ near: near - padding, far: far + padding });
}
