import {float, vec2, vec3, sin, cos} from 'three/tsl';

// Standard forward-mode automatic differentiation of a bounded pure r185 TSL
// DAG. This is a software integration experiment, not a new AD algorithm.
export const DERIVATIVE_LIMITS = Object.freeze({maxNodes: 128, maxDepth: 32});
const axes = 'xyz';
const inputTypes = new Set(['UniformNode', 'AttributeNode', 'PropertyNode', 'SplitNode']);
const typeName = node => node?.constructor?.type ?? node?.constructor?.name ?? 'unknown';

function canonical(input, stats) {
  let node = input;
  const seen = new Set();
  for (;;) {
    if (!node?.isNode || seen.has(node)) throw Error('Expected an acyclic native TSL node.');
    seen.add(node);
    if (node._beforeNodes?.length) throw Error('Nodes with before() side effects are unsupported.');
    if (node.self?.isNode && node.self !== node) { node = node.self; continue; }
    if (typeName(node) !== 'VarNode') return node;
    if (node.intent !== true || node.name !== null || node.readOnly === true) throw Error('Explicit or mutable VarNode is unsupported.');
    stats.intentWrappers.add(node);
    node = node.node;
  }
}

function inputWidth(node) {
  if (typeName(node) === 'SplitNode' && /^[xyz]{1,3}$/.test(node.components) && [1, 3].includes(node.components.length)) return node.components.length;
  const type = node.nodeType;
  if (type === 'float' || type === 'f32') return 1;
  if (type === 'vec3' || type === 'vec3f') return 3;
  if (type == null && typeof node.value === 'number') return 1;
  if (type == null && node.value?.isVector3) return 3;
  throw Error(`A designated ${typeName(node)} needs float or vec3 type.`);
}

function constantValues(node) {
  const width = inputWidth(node);
  const values = width === 1 ? [node.value] : [node.value?.x, node.value?.y, node.value?.z];
  if (values.some(value => typeof value !== 'number' || !Number.isFinite(value))) throw Error('Constants must be finite float/vec3 values.');
  return values;
}

const tslBackend = {
  name: 'tsl', c: float,
  add: (a, b) => a.add(b), sub: (a, b) => a.sub(b), mul: (a, b) => a.mul(b), sin, cos,
  unpack(node, width) { return width === 1 ? [node] : [...axes].map(axis => node[axis]); },
  pack: values => values.length === 1 ? values[0] : values.length === 2 ? vec2(...values) : vec3(...values),
};

const numericBackend = {
  name: 'cpu', c: value => value,
  add: (a, b) => a + b, sub: (a, b) => a - b, mul: (a, b) => a * b, sin: Math.sin, cos: Math.cos,
  unpack(value, width) {
    const values = width === 1 ? [value] : Array.from(value ?? []);
    if (values.length !== width || values.some(v => typeof v !== 'number' || !Number.isFinite(v))) throw Error('CPU input values must be finite scalar/vec3 values.');
    return values;
  },
  pack: values => values.length === 1 ? values[0] : values,
};

function interpret(rootNode, variable, {constants = []}, B, values) {
  if (!Array.isArray(constants)) throw Error('constants must be an array of exact native input nodes.');
  if (B.name === 'cpu' && !(values instanceof Map)) throw Error('CPU values must be a Map keyed by exact native input-node identity.');
  const details = {intentWrappers: new Set(), nodes: 0, operations: 0, depth: 0, usedInputs: new Set()};
  const inputs = new Map(), cache = new Map(), active = new Set(), numericValues = new Map();
  const variableNode = canonical(variable, details);
  if (!inputTypes.has(typeName(variableNode)) || inputWidth(variableNode) !== 1) throw Error('Differentiation variable must be a designated scalar input.');
  if (values) {
    for (const [key, value] of values) {
      const node = canonical(key, details);
      if (numericValues.has(node)) throw Error('Duplicate canonical CPU input value.');
      numericValues.set(node, value);
    }
  }
  for (const input of [variable, ...constants]) {
    const node = canonical(input, details);
    if (!inputTypes.has(typeName(node))) throw Error('Inputs may only designate uniform/attribute/property nodes or explicit scalar/vec3 swizzles.');
    if (inputs.has(node)) throw Error('An input cannot be both variable and constant, or occur twice.');
    const width = inputWidth(node);
    if (B.name === 'cpu' && !numericValues.has(node)) throw Error('Missing CPU value for designated input.');
    const components = B.unpack(B.name === 'tsl' ? node : numericValues.get(node), width);
    inputs.set(node, components.map(value => ({value, derivative: B.c(node === variableNode ? 1 : 0), zero: node !== variableNode})));
  }

  const zero = () => B.c(0);
  function combine(a, b, operation) {
    if (a.length !== b.length && a.length !== 1 && b.length !== 1) throw Error('Unsupported vector-width mismatch.');
    return Array.from({length: Math.max(a.length, b.length)}, (_, i) => operation(a[a.length === 1 ? 0 : i], b[b.length === 1 ? 0 : i]));
  }
  function add(a, b) {
    const derivative = a.zero ? b.derivative : b.zero ? a.derivative : B.add(a.derivative, b.derivative);
    return {value: B.add(a.value, b.value), derivative, zero: a.zero && b.zero};
  }
  function subtract(a, b) {
    const derivative = b.zero ? a.derivative : a.zero ? B.sub(zero(), b.derivative) : B.sub(a.derivative, b.derivative);
    return {value: B.sub(a.value, b.value), derivative, zero: a.zero && b.zero};
  }
  function multiply(a, b) {
    let derivative;
    if (a.zero && b.zero) derivative = zero();
    else if (a.zero) derivative = B.mul(a.value, b.derivative);
    else if (b.zero) derivative = B.mul(a.derivative, b.value);
    else derivative = B.add(B.mul(a.derivative, b.value), B.mul(a.value, b.derivative));
    return {value: B.mul(a.value, b.value), derivative, zero: a.zero && b.zero};
  }
  function unary(a, method) {
    if (method === 'negate') return {value: B.sub(zero(), a.value), derivative: a.zero ? zero() : B.sub(zero(), a.derivative), zero: a.zero};
    const value = B[method](a.value);
    const factor = a.zero ? undefined : method === 'sin' ? B.cos(a.value) : B.sub(zero(), B.sin(a.value));
    return {value, derivative: a.zero ? zero() : B.mul(factor, a.derivative), zero: a.zero};
  }

  function visit(input, depth = 1) {
    const node = canonical(input, details);
    details.depth = Math.max(details.depth, depth);
    if (depth > DERIVATIVE_LIMITS.maxDepth) throw Error('Expression depth exceeds 32.');
    if (active.has(node)) throw Error('Cyclic expression DAG.');
    if (cache.has(node)) return cache.get(node);
    if (++details.nodes > DERIVATIVE_LIMITS.maxNodes) throw Error('Expression DAG exceeds 128 nodes.');
    active.add(node);
    const child = value => visit(value, depth + 1);
    let result;
    if (inputs.has(node)) {
      result = inputs.get(node);
      details.usedInputs.add(node);
    } else {
      switch (typeName(node)) {
        case 'ConstNode':
          result = constantValues(node).map(value => ({value: B.c(value), derivative: zero(), zero: true}));
          break;
        case 'OperatorNode': {
          const operation = {'+': add, '-': subtract, '*': multiply}[node.op];
          if (!operation || !node.bNode) throw Error(`Unsupported operator: ${node.op}.`);
          result = combine(child(node.aNode), child(node.bNode), operation);
          details.operations++;
          break;
        }
        case 'MathNode':
          if (!['sin', 'cos', 'negate'].includes(node.method) || node.bNode || node.cNode) throw Error(`Unsupported MathNode method: ${node.method}.`);
          result = child(node.aNode).map(value => unary(value, node.method));
          details.operations++;
          break;
        case 'JoinNode':
          if (!['vec3', 'vec3f'].includes(node.nodeType)) throw Error('Only vec3 joins are supported.');
          result = node.nodes.flatMap(child);
          if (result.length !== 3) throw Error('vec3 joins must contain exactly three scalar components.');
          break;
        case 'SplitNode': {
          const source = child(node.node);
          if (typeof node.components !== 'string' || !/^[xyz]{1,3}$/.test(node.components)) throw Error('Only xyz swizzles up to width three are supported.');
          result = [...node.components].map(axis => {
            const index = source.length === 1 ? 0 : axes.indexOf(axis);
            if (index >= source.length) throw Error('Swizzle accesses an absent component.');
            return source[index];
          });
          break;
        }
        case 'ConvertNode': {
          const source = child(node.node);
          if (node.convertTo === 'float') result = [source[0]];
          else if (node.convertTo === 'vec3' && source.length === 1) result = [source[0], source[0], source[0]];
          else if (node.convertTo === 'vec3' && source.length === 3) result = source;
          else throw Error(`Unsupported conversion to ${node.convertTo}.`);
          break;
        }
        default:
          throw Error(`Unsupported or unbound TSL node: ${typeName(node)}.`);
      }
    }
    active.delete(node);
    cache.set(node, result);
    return result;
  }
  const result = visit(rootNode);
  if (result.length !== 1 && result.length !== 3) throw Error('The root must produce float or vec3.');
  return {
    value: B.pack(result.map(component => component.value)),
    derivative: B.pack(result.map(component => component.derivative)),
    stats: Object.freeze({backend: B.name, nodes: details.nodes, operations: details.operations, maxDepth: details.depth,
      components: result.length, inputCount: details.usedInputs.size, unusedInputs: inputs.size - details.usedInputs.size,
      variableUsed: details.usedInputs.has(variableNode), unwrappedIntents: details.intentWrappers.size, method: 'standard forward-mode AD'}),
  };
}

/**
 * Differentiate a pure float/vec3 native TSL expression w.r.t. one scalar input.
 * The exact variable node and constants are opaque input boundaries. The caller
 * promises each designated constant (e.g. storage.element(i).xyz) is independent
 * of the variable and all expressions are immutable, finite, and side-effect
 * free. No unknown input is silently treated as constant. Supports +,-,*,
 * sin,cos,negate, float/vec3 construction and xyz swizzles. Explicit variables,
 * functions, unbound storage, textures, branches and nonsmooth math fail closed.
 *
 * For velocity of a rendered triangle, interpolate derivatives at its three
 * rest vertices; differentiating at the interpolated rest position instead
 * describes the underlying nonlinear map, not generally the linear triangle.
 */
export function differentiate(rootNode, variable, options = {}) {
  const {derivative, stats} = interpret(rootNode, variable, options, tslBackend);
  return {derivative, stats};
}

/** Same traversal and derivative rules using CPU numbers; no GPU or FD oracle. */
export function evaluateDerivative(rootNode, variable, values, options = {}) {
  return interpret(rootNode, variable, options, numericBackend, values);
}

async function selfTest() {
  const {default: assert} = await import('node:assert/strict');
  const {uniform, attribute, storage, instanceIndex, Fn, abs} = await import('three/tsl');
  const {StorageBufferAttribute} = await import('three/webgpu');
  const p = attribute('position', 'vec3'), t = uniform(0, 'float'), unknown = uniform(0, 'float');
  const angle = p.y.mul(1.2).add(t.mul(.6)), s = sin(angle), c = cos(angle);
  const graph = {
    wave: vec3(p.x, p.y.add(sin(p.x.mul(2.1).add(t)).mul(.65)), p.z),
    ripple: vec3(p.x, p.y.add(sin(p.x.mul(5).add(t)).mul(cos(p.z.mul(4).sub(t))).mul(.45)), p.z),
    twist: vec3(p.x.mul(c).sub(p.z.mul(s)), p.y, p.x.mul(s).add(p.z.mul(c))),
  };
  const expected = (kind, q, phase) => {
    if (kind === 'wave') return {value: [q[0], q[1] + .65 * Math.sin(2.1 * q[0] + phase), q[2]], derivative: [0, .65 * Math.cos(2.1 * q[0] + phase), 0]};
    if (kind === 'ripple') {
      const a = 5 * q[0] + phase, b = 4 * q[2] - phase;
      return {value: [q[0], q[1] + .45 * Math.sin(a) * Math.cos(b), q[2]], derivative: [0, .45 * Math.cos(a - b), 0]};
    }
    const a = 1.2 * q[1] + .6 * phase;
    return {value: [q[0] * Math.cos(a) - q[2] * Math.sin(a), q[1], q[0] * Math.sin(a) + q[2] * Math.cos(a)],
      derivative: [-.6 * (q[0] * Math.sin(a) + q[2] * Math.cos(a)), 0, .6 * (q[0] * Math.cos(a) - q[2] * Math.sin(a))]};
  };
  let state = 1209384;
  const random = () => {state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296;};
  let maxError = 0;
  const close = (actual, reference) => {
    const error = Math.abs(actual - reference);
    maxError = Math.max(maxError, error);
    assert.ok(error < 1e-11, `${actual} differs from independent derivative ${reference}`);
  };
  for (let i = 0; i < 500; i++) {
    const q = [random() * 4 - 2, random() * 4 - 2, random() * 4 - 2], phase = random() * 6 - 3;
    for (const [kind, expression] of Object.entries(graph)) {
      const actual = evaluateDerivative(expression, t, new Map([[p, q], [t, phase]]), {constants: [p]});
      const reference = expected(kind, q, phase);
      actual.value.forEach((v, axis) => close(v, reference.value[axis]));
      actual.derivative.forEach((v, axis) => close(v, reference.derivative[axis]));
      const symbolic = differentiate(expression, t, {constants: [p]});
      assert.ok(symbolic.derivative.isNode);
      assert.equal(actual.stats.nodes, symbolic.stats.nodes);
    }
  }
  const scalar = evaluateDerivative(t.mul(t).sub(t.mul(3)).negate(), t, new Map([[t, 2]]));
  close(scalar.value, 2); close(scalar.derivative, -1);
  const swizzle = evaluateDerivative(vec3(t, t.mul(2), t.mul(3)).zyx, t, new Map([[t, 2]]));
  assert.deepEqual(swizzle.derivative, [3, 2, 1]);
  const constant = evaluateDerivative(p, t, new Map([[t, 1], [p, [1, 2, 3]]]), {constants: [p]});
  assert.deepEqual(constant.derivative, [0, 0, 0]);
  const base = storage(new StorageBufferAttribute(new Float32Array(4), 4), 'vec4', 1).element(instanceIndex).xyz;
  assert.ok(differentiate(base.add(vec3(t)), t, {constants: [base]}).derivative.isNode);
  assert.throws(() => differentiate(base.add(vec3(t)), t), /unbound/);
  assert.throws(() => differentiate(t.add(unknown), t), /unbound/);
  assert.throws(() => differentiate(t, t, {constants: [t]}), /both variable/);
  assert.throws(() => differentiate(t, p), /scalar/);
  assert.throws(() => differentiate(t.div(2), t), /Unsupported operator/);
  assert.throws(() => differentiate(abs(t), t), /Unsupported MathNode/);
  assert.throws(() => differentiate(t.toVar(), t), /VarNode/);
  assert.throws(() => differentiate(Fn(() => t.add(1))(), t), /unsupported|Unsupported/);
  assert.throws(() => evaluateDerivative(t, t, new Map()), /Missing CPU/);
  console.log(JSON.stringify({passed: true, samples: 1500, maxError, method: 'standard forward-mode AD', gpuExecuted: false}));
}

if (typeof process !== 'undefined' && process.versions?.node && process.argv.includes('--self-test')) {
  const {pathToFileURL} = await import('node:url');
  if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await selfTest();
}
