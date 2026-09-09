import test from 'node:test';
import assert from 'node:assert/strict';
import {Matrix3, Vector2, Vector3, Vector4} from 'three';
import {StorageBufferAttribute} from 'three/webgpu';
import {uniform, attribute, storage, instanceIndex, float, vec2, vec3, vec4, sin, cos, exp, tanh, dot, abs, Fn} from 'three/tsl';
import {gradients, evaluateGradients, AUTOGRAD_LIMITS} from './autograd.js';

const near = (actual, expected, tolerance = 1e-10) => assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);
const arrayNear = (actual, expected, tolerance) => {assert.equal(actual.length, expected.length); actual.forEach((v, i) => near(v, expected[i], tolerance));};

test('shared DAG accumulates reverse adjoints rather than overwriting them', () => {
  const x = uniform(0), y = uniform(0), shared = x.mul(y);
  const loss = shared.mul(shared).add(shared).add(x);
  const answer = evaluateGradients(loss, [x, y], new Map([[x, 2], [y, 3]]));
  near(answer.value, 44);
  arrayNear(answer.gradients, [40, 26]);
  assert.ok(answer.stats.reverseAccumulations > 0);
  const symbolic = gradients(loss, [x, y]);
  assert.equal(symbolic.value, loss);
  assert.ok(symbolic.gradients.every(node => node.isNode));
  assert.equal(symbolic.stats.scalarTapeOps, answer.stats.scalarTapeOps);
});

test('broadcast, vector parameter widths, repeated swizzles, and dot reduce correctly', () => {
  const x = uniform(0), w = uniform(new Vector4());
  const z = w.xxyw.mul(x).add(vec4(x));
  const loss = dot(z, vec4(1, 2, 3, 4));
  const answer = evaluateGradients(loss, [w, x], new Map([[w, [2, 3, 5, 7]], [x, 11]]));
  // w.x occurs twice; w.z never occurs; scalar x broadcasts to all outputs.
  arrayNear(answer.gradients[0], [33, 33, 0, 44]);
  near(answer.gradients[1], 1 * 3 + 2 * 3 + 3 * 4 + 4 * 8);
  near(answer.value, answer.gradients[1] * 11);
  const symbolic = gradients(loss, [w, x]);
  assert.ok(symbolic.gradients[0].isNode && symbolic.gradients[1].isNode);
  assert.equal(symbolic.stats.scalarParameters, 5);
});

test('scalar/vector conversion uses Three format padding and truncation', () => {
  const x = uniform(0), q = uniform(new Vector2()), r = uniform(new Vector3());
  const loss = dot(vec4(q), vec4(1)).add(dot(vec4(r), vec4(1))).add(dot(vec3(x), vec3(1))).add(float(vec4(x)));
  const answer = evaluateGradients(loss, [x, q, r], new Map([[x, 2], [q, [3, 4]], [r, [5, 6, 7]]]));
  near(answer.value, 7 + 1 + 18 + 1 + 6 + 2);
  near(answer.gradients[0], 4);
  arrayNear(answer.gradients[1], [1, 1]);
  arrayNear(answer.gradients[2], [1, 1, 1]);
  const truncate = evaluateGradients(dot(vec2(r), vec2(2, 3)), [r], new Map([[r, [5, 6, 7]]]));
  arrayNear(truncate.gradients[0], [2, 3, 0]);
});

test('analytic neural-network residual gradient with vector weights and unused component', () => {
  const weights = uniform(new Vector4()), gain = uniform(0), input = attribute('input', 'vec2'), target = uniform(0);
  const hidden = tanh(dot(weights.xy, input).add(weights.z));
  const residual = hidden.mul(gain).sub(target);
  const loss = residual.mul(residual);
  const w = [.3, -.4, .2, 99], g = 1.2, p = [.7, -.8], desired = -.15;
  const answer = evaluateGradients(loss, [weights, gain], new Map([[weights, w], [gain, g], [input, p], [target, desired]]), {constants: [input, target]});
  const h = Math.tanh(w[0] * p[0] + w[1] * p[1] + w[2]), error = g * h - desired;
  const delta = 2 * error * g * (1 - h * h);
  near(answer.value, error * error);
  arrayNear(answer.gradients[0], [delta * p[0], delta * p[1], delta, 0]);
  near(answer.gradients[1], 2 * error * h);
  const symbolic = gradients(loss, [weights, gain], {constants: [input, target]});
  assert.ok(symbolic.gradients.every(node => node.isNode));
});

test('random shared scalar DAGs match independently evaluated central differences', () => {
  let state = 0x7926a;
  const random = () => {state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296;};
  let checked = 0;
  for (let trial = 0; trial < 80; trial++) {
    const variables = [uniform(0), uniform(0), uniform(0)];
    const pool = variables.map((node, i) => ({node, value: x => x[i]}));
    for (let operation = 0; operation < 24; operation++) {
      const a = pool[Math.floor(random() * pool.length)], b = pool[Math.floor(random() * pool.length)];
      let next;
      switch (Math.floor(random() * 8)) {
        case 0: next = {node: a.node.add(b.node), value: x => a.value(x) + b.value(x)}; break;
        case 1: next = {node: a.node.sub(b.node), value: x => a.value(x) - b.value(x)}; break;
        case 2: next = {node: a.node.mul(b.node).mul(.2), value: x => a.value(x) * b.value(x) * .2}; break;
        case 3: next = {node: a.node.div(b.node.mul(b.node).add(1)), value: x => a.value(x) / (b.value(x) ** 2 + 1)}; break;
        case 4: next = {node: sin(a.node), value: x => Math.sin(a.value(x))}; break;
        case 5: next = {node: cos(a.node), value: x => Math.cos(a.value(x))}; break;
        case 6: next = {node: exp(tanh(a.node)), value: x => Math.exp(Math.tanh(a.value(x)))}; break;
        case 7: next = {node: tanh(a.node).negate(), value: x => -Math.tanh(a.value(x))}; break;
      }
      pool.push(next);
    }
    const a = pool.at(-1), b = pool.at(-2), expression = a.node.mul(a.node).add(b.node);
    const objective = x => a.value(x) ** 2 + b.value(x);
    const point = variables.map(() => random() - .5), values = new Map(variables.map((v, i) => [v, point[i]]));
    const answer = evaluateGradients(expression, variables, values);
    near(answer.value, objective(point));
    for (let coordinate = 0; coordinate < 3; coordinate++) {
      const epsilon = 1e-5, left = [...point], right = [...point];
      left[coordinate] -= epsilon; right[coordinate] += epsilon;
      const expected = (objective(right) - objective(left)) / (2 * epsilon);
      near(answer.gradients[coordinate], expected, 3e-6);
      checked++;
    }
  }
  assert.equal(checked, 240);
});

test('exact storage swizzle boundaries work; storage is otherwise rejected', () => {
  const buffer = storage(new StorageBufferAttribute(new Float32Array(8), 4), 'vec4', 2);
  const weight = buffer.element(instanceIndex).x;
  const p = buffer.element(instanceIndex).yzw;
  const loss = weight.mul(dot(p, p));
  const answer = evaluateGradients(loss, [weight], new Map([[weight, 2], [p, [1, 2, 3]]]), {constants: [p]});
  near(answer.value, 28); near(answer.gradients[0], 14);
  assert.ok(gradients(loss, [weight], {constants: [p]}).gradients[0].isNode);
  assert.throws(() => gradients(loss, [weight]), /unbound/);
  const integers = storage(new StorageBufferAttribute(new Uint32Array(4), 4), 'uvec4', 1).element(instanceIndex).x;
  assert.throws(() => gradients(integers, [integers]), /floating-point/);
});

test('unused parameters have matching-width zero gradients and constant root is valid', () => {
  const x = uniform(0), q = uniform(new Vector3());
  const answer = evaluateGradients(float(7), [x, q], new Map([[x, 1], [q, [2, 3, 4]]]));
  near(answer.value, 7); near(answer.gradients[0], 0); arrayNear(answer.gradients[1], [0, 0, 0]);
  assert.equal(answer.stats.unusedInputs, 2);
});

test('rejects unbound inputs, unsupported nodes, mutations, side effects and invalid domains', () => {
  const x = uniform(0), y = uniform(0);
  assert.throws(() => gradients(x.add(y), [x]), /unbound/);
  assert.throws(() => gradients(x, [x], {constants: [x]}), /Duplicate/);
  assert.throws(() => gradients(x, [x, x]), /Duplicate/);
  assert.throws(() => gradients(vec2(x), [x]), /scalar/);
  assert.throws(() => gradients(abs(x), [x]), /Unsupported MathNode/);
  assert.throws(() => gradients(x.toVar(), [x]), /VarNode/);
  assert.throws(() => gradients(x.toConst(), [x]), /VarNode/);
  assert.throws(() => gradients(Fn(() => x.add(1))(), [x]), /Unsupported|unsupported/);
  assert.throws(() => gradients(x.add(1).before(y), [x]), /side effects/);
  assert.throws(() => evaluateGradients(x, [x], new Map()), /Missing/);
  assert.throws(() => evaluateGradients(x, [x], new Map([[x, NaN]])), /finite/);
  assert.throws(() => evaluateGradients(x.div(y), [x, y], new Map([[x, 1], [y, 0]])), /Nonfinite/);
  assert.throws(() => evaluateGradients(exp(x), [x], new Map([[x, 1000]])), /Nonfinite/);
});

test('rejects cycles and enforces depth, scalar parameter and operation bounds', () => {
  const x = uniform(0), cyclic = x.add(1);
  const op = cyclic.node;
  assert.equal(op.constructor.type, 'OperatorNode');
  op.aNode = op;
  assert.throws(() => gradients(cyclic, [x]), /Cyclic/);
  const variables = Array.from({length: 129}, () => uniform(0));
  assert.throws(() => gradients(variables[0], variables), /128 designated/);
  let deep = x;
  for (let i = 0; i < 129; i++) deep = deep.add(1);
  assert.throws(() => gradients(deep, [x]), /depth exceeds/);
  let layer = Array.from({length: 2200}, (_, i) => sin(x.add(i / 2200)));
  while (layer.length > 1) {
    const next = [];
    for (let i = 0; i < layer.length; i += 2) next.push(layer[i + 1] ? layer[i].add(layer[i + 1]) : layer[i]);
    layer = next;
  }
  assert.throws(() => gradients(layer[0], [x]), /4096/);
  assert.equal(AUTOGRAD_LIMITS.maxScalarTapeOps, 4096);
});

test('actual returned native gradient expressions match independent vector finite differences', () => {
  const q = uniform(new Vector4()), d = uniform(new Vector4()), gain = uniform(0);
  const r = sin(q).add(cos(d)).mul(exp(q.mul(.1))).div(d.mul(d).add(1));
  const loss = dot(tanh(r), vec4(.3, -.7, 1.1, -.5)).mul(gain);
  const variables = [q, d, gain], symbolic = gradients(loss, variables);
  const independent = point => {
    const coefficients = [.3, -.7, 1.1, -.5];
    let sum = 0;
    for (let i = 0; i < 4; i++) {
      const numerator = (Math.sin(point[i]) + Math.cos(point[i + 4])) * Math.exp(.1 * point[i]);
      sum += coefficients[i] * Math.tanh(numerator / (point[i + 4] ** 2 + 1));
    }
    return sum * point[8];
  };
  let state = 0x72910;
  const random = () => {state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296;};
  for (let trial = 0; trial < 16; trial++) {
    const point = Array.from({length: 9}, () => 4 * random() - 2);
    const values = new Map([[q, point.slice(0, 4)], [d, point.slice(4, 8)], [gain, point[8]]]);
    const numeric = evaluateGradients(loss, variables, values);
    near(numeric.value, independent(point));
    const nativeScalars = [...'xyzw'].map(axis => symbolic.gradients[0][axis])
      .concat([...'xyzw'].map(axis => symbolic.gradients[1][axis]), symbolic.gradients[2]);
    const numericScalars = [...numeric.gradients[0], ...numeric.gradients[1], numeric.gradients[2]];
    for (let coordinate = 0; coordinate < point.length; coordinate++) {
      const low = [...point], high = [...point], epsilon = 1e-5;
      low[coordinate] -= epsilon; high[coordinate] += epsilon;
      const finiteDifference = (independent(high) - independent(low)) / (2 * epsilon);
      // This evaluates the returned TSL expression graph itself. It catches
      // symbolic packing/broadcast/backend errors that checking .isNode misses.
      const returnedExpression = evaluateGradients(nativeScalars[coordinate], variables, values).value;
      near(returnedExpression, finiteDifference, 3e-6);
      near(numericScalars[coordinate], finiteDifference, 3e-6);
    }
  }
});

test('generated gradient preserves exact constants and handles analytic cancellation', () => {
  const x = uniform(0), p = attribute('observation', 'vec2'), unrelated = uniform(new Vector3());
  const shared = x.mul(p.x).add(p.y), loss = shared.mul(shared).sub(shared.mul(shared)).add(x.mul(x).mul(x));
  const variables = [x, unrelated], constants = [p];
  const values = new Map([[x, -1.25], [p, [.3, -.7]], [unrelated, [9, 8, 7]]]);
  const symbolic = gradients(loss, variables, {constants});
  near(evaluateGradients(symbolic.gradients[0], variables, values, {constants}).value, 3 * 1.25 ** 2);
  for (const axis of 'xyz') near(evaluateGradients(symbolic.gradients[1][axis], variables, values, {constants}).value, 0);
  assert.throws(() => evaluateGradients(symbolic.gradients[0], variables, values), /unbound/);
});

test('integer, boolean, matrix and nested integer storage boundaries fail closed', () => {
  const invalid = [uniform(1, 'int'), uniform(1, 'uint'), uniform(false, 'bool'),
    uniform(new Vector3(), 'ivec3'), uniform(new Matrix3())];
  for (const input of invalid) assert.throws(() => gradients(input, [input]), /float|type/);
  const integerBuffer = storage(new StorageBufferAttribute(new Int32Array(8), 4), 'ivec4', 2);
  const nested = integerBuffer.element(instanceIndex).yz.x;
  assert.throws(() => gradients(nested, [nested]), /floating-point/);
  const q = uniform(new Vector2());
  assert.throws(() => gradients(q.toUint(), [q]), /Unsupported conversion/);
});

test('all designated CPU inputs obey finite-value and width contracts even when unused', () => {
  const x = uniform(0), q = uniform(new Vector3()), constant = attribute('unusedConstant', 'vec2');
  assert.throws(() => evaluateGradients(float(7), [x], new Map([[x, NaN]])), /finite/);
  assert.throws(() => evaluateGradients(float(7), [q], new Map([[q, [1, 2]]])), /width/);
  assert.throws(() => evaluateGradients(x.mul(x), [x], new Map([[x, 2], [constant, [1, Infinity]]]), {constants: [constant]}), /finite/);
});
