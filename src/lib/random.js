export function mulberry32(seed) {
  let state = seed >>> 0;
  return function random() {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

export function shuffledRange(length, seed) {
  const values = Uint32Array.from({ length }, (_, index) => index);
  const random = mulberry32(seed);
  for (let index = values.length - 1; index > 0; index -= 1) {
    const target = Math.floor(random() * (index + 1));
    const temporary = values[index];
    values[index] = values[target];
    values[target] = temporary;
  }
  return values;
}
