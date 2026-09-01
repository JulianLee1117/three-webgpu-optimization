export const FIRST_INSTANCE_STANDALONE_BOOT_ID =
  'first-instance-live-standalone-deployment-v1';

const MODE_ID = 'first-instance-live-standalone-deployment';
const BOOT_KEYS = Object.freeze([
  'benchmarkBoot',
  'standaloneLane',
  'standaloneVisibilityOrder',
]);
const LANES = Object.freeze(['portable', 'feature']);
const VISIBILITY_ORDERS = Object.freeze([
  Object.freeze([0.99, 0.2]),
  Object.freeze([0.2, 0.99]),
]);

function exactOrder(left, right) {
  return left.length === right.length
    && left.every((value, index) => Object.is(value, right[index]));
}

export function parseFirstInstanceStandaloneBoot(search = '') {
  const parameters = new URLSearchParams(search);
  const bootValues = parameters.getAll('benchmarkBoot');
  if (bootValues.length === 0) return null;
  if (bootValues.length !== 1 || bootValues[0] !== FIRST_INSTANCE_STANDALONE_BOOT_ID) {
    throw new Error('Standalone benchmark boot identifier is invalid.');
  }
  const keys = [...parameters.keys()];
  if (keys.length !== BOOT_KEYS.length
    || !BOOT_KEYS.every((key) => parameters.getAll(key).length === 1)
    || keys.some((key) => !BOOT_KEYS.includes(key))) {
    throw new Error('Standalone benchmark boot query differs from its exact schema.');
  }
  const laneId = parameters.get('standaloneLane');
  if (!LANES.includes(laneId)) {
    throw new Error('Standalone benchmark boot lane is invalid.');
  }
  const visibilityOrder = parameters.get('standaloneVisibilityOrder')
    .split(',')
    .map(Number);
  if (!VISIBILITY_ORDERS.some((candidate) => exactOrder(candidate, visibilityOrder))) {
    throw new Error('Standalone benchmark boot visibility order is invalid.');
  }
  return Object.freeze({
    schemaVersion: 1,
    kind: 'first-instance-standalone-initial-page-boot',
    bootId: FIRST_INSTANCE_STANDALONE_BOOT_ID,
    modeId: MODE_ID,
    laneId,
    visibilityOrder: Object.freeze([...visibilityOrder]),
    objectCount: 65_536,
    bucketCount: 32,
    layout: 'baseline',
    initialRebuildCount: 1,
    priorStrategyConstructionCount: 0,
  });
}

export function firstInstanceStandaloneBootSearch({ laneId, visibilityOrder }) {
  if (!LANES.includes(laneId)
    || !Array.isArray(visibilityOrder)
    || !VISIBILITY_ORDERS.some((candidate) => exactOrder(candidate, visibilityOrder))) {
    throw new Error('Cannot encode an invalid standalone benchmark boot configuration.');
  }
  const parameters = new URLSearchParams();
  parameters.set('benchmarkBoot', FIRST_INSTANCE_STANDALONE_BOOT_ID);
  parameters.set('standaloneLane', laneId);
  parameters.set('standaloneVisibilityOrder', visibilityOrder.join(','));
  return `?${parameters.toString()}`;
}
