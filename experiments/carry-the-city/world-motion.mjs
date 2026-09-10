/**
 * Velocity inheritance from actual, composed world positions.
 *
 * Record a supported object's final world-space anchor once per simulation
 * frame, after the town transform and deformation/blend have been updated.
 * Retain the two positions themselves: reconstructing an old point through a
 * current matrix, current blend or modified source clip loses real movement.
 *
 * This is a finite-difference velocity of kinematic animation. It neither
 * models propulsion nor infers the force of a collision.
 */

function vector(value, label) {
  const p = Array.isArray(value) || ArrayBuffer.isView(value)
    ? Array.from(value) : [value?.x, value?.y, value?.z];
  if (p.length !== 3 || !p.every(Number.isFinite)) throw new TypeError(`${label} must contain three finite coordinates`);
  return p;
}

function key(id) {
  if (typeof id !== 'string' || id.length === 0 || id.length > 128) throw new TypeError('Anchor id must be a short nonempty string');
}

/**
 * A bounded history for a few potential release anchors. record() copies its
 * input; later scene graph mutations cannot rewrite the preceding frame.
 * maxDeltaTime rejects stale history rather than treating a pause as movement.
 */
export function createWorldMotionHistory({ maxPoints = 8, maxDeltaTime = 0.25 } = {}) {
  if (!Number.isInteger(maxPoints) || maxPoints < 1 || maxPoints > 64) throw new RangeError('maxPoints must be an integer in [1, 64]');
  if (!Number.isFinite(maxDeltaTime) || maxDeltaTime <= 0 || maxDeltaTime > 1) throw new RangeError('maxDeltaTime must be in (0, 1] seconds');
  const points = new Map();

  return Object.freeze({
    get size() { return points.size; },

    record(id, worldPosition, time, { discontinuous = false } = {}) {
      key(id);
      const position = vector(worldPosition, 'World position');
      if (!Number.isFinite(time)) throw new TypeError('Time must be finite simulation seconds');
      if (typeof discontinuous !== 'boolean') throw new TypeError('discontinuous must be a boolean');
      const old = points.get(id);
      if (!old && points.size >= maxPoints) throw new RangeError('World motion anchor limit reached');
      const elapsed = old ? time - old.current.time : 0;
      const reason = discontinuous ? 'discontinuous'
        : !old ? 'first-sample'
          : elapsed <= 0 ? 'non-increasing-time'
            : elapsed > maxDeltaTime ? 'stale-history' : 'tracked';
      const previous = reason === 'tracked' ? old.current : null;
      points.set(id, { current: { position, time }, previous, reason });
      return { hasVelocity: previous !== null, deltaTime: previous ? elapsed : 0, reason };
    },

    /**
     * Velocity is the preceding frame interval's average in world units/second.
     * An optional explicit WORLD-space kick is added once, independently of the
     * inherited motion. Calling this does not consume or modify the history.
     */
    release(id, { kick = [0, 0, 0] } = {}) {
      key(id);
      const added = vector(kick, 'World-space kick');
      const entry = points.get(id);
      if (!entry) throw new RangeError(`World motion anchor has not been recorded: ${id}`);
      const { current, previous, reason } = entry;
      const deltaTime = previous ? current.time - previous.time : 0;
      const inheritedVelocity = previous
        ? current.position.map((v, c) => (v - previous.position[c]) / deltaTime)
        : [0, 0, 0];
      const velocity = inheritedVelocity.map((v, c) => v + added[c]);
      if (![...inheritedVelocity, ...velocity].every(Number.isFinite)) throw new RangeError('World motion exceeds finite velocity range');
      return {
        position: current.position.slice(), velocity, inheritedVelocity,
        hasVelocity: previous !== null, deltaTime, reason,
      };
    },

    forget(id) { key(id); return points.delete(id); },
    clear() { points.clear(); },
  });
}
