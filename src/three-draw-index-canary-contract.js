export const THREE_DRAW_INDEX_CANARY_KIND =
  'three-webgpu-draw-index-correctness-canary';
export const WGSL_IMMEDIATE_FEATURE = 'immediate_address_space';
export const DRAW_INDEX_SYMBOL = 'nodeDrawIndex';
export const DRAW_COUNT = 4;
export const TARGET_WIDTH = 4;
export const TARGET_HEIGHT = 1;
export const INDIRECT_WORD_COUNT = 5;
export const BATCH_ORDER = Object.freeze([2, 0, 3, 1]);
export const INDIRECT_OFFSETS = Object.freeze(BATCH_ORDER.map((index) => index * 20));
export const RERECORD_INDIRECT_OFFSETS = Object.freeze([20, 60, 0, 40]);

function rgbaPixels(redValues, greenValues = redValues.map(() => 0)) {
  const output = [];
  for (let index = 0; index < redValues.length; index += 1) {
    output.push(redValues[index], greenValues[index], 0, 255);
  }
  return output;
}

export function expectedDirectOutput() {
  return [1, 0, 0, 255];
}

export function expectedIndirectOutput(offsets = INDIRECT_OFFSETS) {
  const drawOrdinalByGeometry = Array(DRAW_COUNT).fill(-1);
  offsets.forEach((offset, drawOrdinal) => {
    drawOrdinalByGeometry[offset / 20] = drawOrdinal;
  });
  return rgbaPixels(drawOrdinalByGeometry.map((drawOrdinal) => drawOrdinal + 1));
}

export function expectedBatchedOutput(order = BATCH_ORDER) {
  const drawOrdinalByInstance = Array(DRAW_COUNT).fill(-1);
  order.forEach((instanceId, drawOrdinal) => {
    drawOrdinalByInstance[instanceId] = drawOrdinal;
  });
  return rgbaPixels(
    drawOrdinalByInstance.map((drawOrdinal) => drawOrdinal + 1),
    Array.from({ length: DRAW_COUNT }, (_, instanceId) => instanceId + 1),
  );
}

export function expectedBatchedControlOutput() {
  return rgbaPixels(
    Array(DRAW_COUNT).fill(0),
    Array.from({ length: DRAW_COUNT }, (_, instanceId) => instanceId + 1),
  );
}

export function createIndirectCommands() {
  const commands = new Uint32Array(DRAW_COUNT * INDIRECT_WORD_COUNT);
  for (let draw = 0; draw < DRAW_COUNT; draw += 1) {
    const word = draw * INDIRECT_WORD_COUNT;
    commands[word] = 3;
    commands[word + 1] = 1;
    commands[word + 2] = draw * 3;
    commands[word + 3] = 0;
    commands[word + 4] = 0;
  }
  return commands;
}
