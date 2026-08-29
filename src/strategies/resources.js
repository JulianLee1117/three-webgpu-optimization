export function disposeStrategyResources(renderer, strategy) {
  if (!strategy) return;
  strategy.root?.removeFromParent();
  strategy.dispose?.(renderer);
  for (const geometry of strategy.geometries ?? []) geometry.dispose();
  for (const material of strategy.materials ?? []) material.dispose();
  for (const computeNode of strategy.computeNodes ?? []) computeNode?.dispose?.();
  for (const attribute of strategy.storageAttributes ?? []) {
    if (typeof attribute.dispose === 'function') {
      attribute.dispose();
    } else {
      renderer?._attributes?.delete?.(attribute);
    }
  }
}
