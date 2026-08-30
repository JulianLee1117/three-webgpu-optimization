export function disposeStrategyResources(renderer, strategy) {
  if (!strategy) return;
  const storageAttributes = new Set(strategy.storageAttributes ?? []);
  for (const geometry of strategy.geometries ?? []) {
    if (geometry.indirect) storageAttributes.add(geometry.indirect);
  }
  const deleteAttribute = renderer?._attributes?.delete;
  if (storageAttributes.size > 0 && typeof deleteAttribute !== 'function') {
    throw new TypeError(
      'Three.js r185 renderer attribute cleanup is unavailable for benchmark resources.',
    );
  }

  strategy.root?.removeFromParent();
  strategy.dispose?.(renderer);
  for (const geometry of strategy.geometries ?? []) {
    geometry.setIndirect?.(null);
    geometry.dispose();
  }
  for (const material of strategy.materials ?? []) material.dispose();
  for (const computeNode of strategy.computeNodes ?? []) computeNode?.dispose?.();
  // BufferAttribute.dispose() only emits an event in Three r185. Standalone
  // storage and indirect attributes have no listener in the common renderer,
  // so the pinned attribute manager must destroy their GPU buffers explicitly.
  for (const attribute of storageAttributes) deleteAttribute.call(renderer._attributes, attribute);
}
