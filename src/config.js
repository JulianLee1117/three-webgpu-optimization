export const THREE_REVISION = '185';

export const VIEWPORT = Object.freeze({
  width: 1280,
  height: 720,
  devicePixelRatio: 1,
});

export const CAMERA = Object.freeze({
  fov: 50,
  aspect: VIEWPORT.width / VIEWPORT.height,
  near: 0.1,
  far: 1000,
  position: Object.freeze([0, 0, 140]),
  target: Object.freeze([0, 0, 0]),
});

export const DEVELOPMENT_PROTOCOL = Object.freeze({
  objectCount: 16_384,
  bucketCounts: Object.freeze([1, 4, 32]),
  visibilityFractions: Object.freeze([0.2, 0.8, 0.99]),
  warmupFrames: 300,
  measuredFrames: 240,
  seed: 0xb1ad_2026,
});

export const PUBLICATION_PROTOCOL = Object.freeze({
  warmupFrames: 300,
  measuredFrames: 600,
  repetitions: 6,
  bucketCounts: Object.freeze([1, 4, 32, 128, 512]),
});
