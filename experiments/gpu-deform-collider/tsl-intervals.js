import {float, vec2, vec3, min, max, abs, sin, cos, ceil, floor, select} from 'three/tsl';

// Experimental interval interpretation of a PURE native Three r185 expression
// DAG. This is ordinary interval arithmetic with heuristic float padding, NOT a
// formal floating-point proof or a compiler for arbitrary TSL programs.
export const INTERVAL_LIMITS = Object.freeze({inputMagnitude: 32, maxNodes: 128, maxDepth: 32, padding: 1e-5, trigAbsoluteMargin: .001, maxEnvelope: 1e12});
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
    // Older TSL proxies expose their underlying node through .self.
    if (node.self?.isNode && node.self !== node) { node = node.self; continue; }
    if (typeName(node) !== 'VarNode') return node;
    // r185 wraps normal arithmetic/construction in unnamed variable intents.
    // Explicit .toVar()/.toConst() declarations are intentionally rejected.
    if (node.intent !== true || node.name !== null) throw Error('Explicit or mutable VarNode is unsupported.');
    if (stats) stats.intentWrappers.add(node);
    node = node.node;
  }
}

function inputWidth(node) {
  // An explicitly bound storageElement.xyz is an opaque vec3 input boundary;
  // its storage/index expressions are not part of the interpreted deformation.
  if (typeName(node) === 'SplitNode' && /^[xyz]{1,3}$/.test(node.components) && [1, 3].includes(node.components.length)) return node.components.length;
  const type = node.nodeType;
  if (type === 'float' || type === 'f32') return 1;
  if (type === 'vec3' || type === 'vec3f') return 3;
  if (type == null && typeof node.value === 'number') return 1;
  if (type == null && node.value?.isVector3) return 3;
  throw Error(`A bound ${typeName(node)} needs an explicit float or vec3 type.`);
}

function constantValues(node) {
  const width = inputWidth(node);
  const values = width === 1 ? [node.value] : [node.value?.x, node.value?.y, node.value?.z];
  if (values.some(value => typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > 32)) {
    throw Error('Constants must be finite float/vec3 values in [-32,32].');
  }
  return values;
}

const tslBackend = {
  name: 'tsl', c: float,
  add: (a, b) => a.add(b), sub: (a, b) => a.sub(b), mul: (a, b) => a.mul(b),
  min, max, abs, sin, cos, ceil, floor,
  le: (a, b) => a.lessThanEqual(b), ge: (a, b) => a.greaterThanEqual(b),
  or: (a, b) => a.or(b), and: (a, b) => a.and(b), select,
  unpack(value, width) {
    if (!value?.isNode) throw Error('TSL interval endpoints must be TSL nodes.');
    return width === 1 ? [value] : axes.split('').map(axis => value[axis]);
  },
  pack: values => values.length === 1 ? values[0] : (values.length === 2 ? vec2(...values) : vec3(...values)),
};

const numericBackend = {
  name: 'cpu', c: value => value,
  add: (a, b) => a + b, sub: (a, b) => a - b, mul: (a, b) => a * b,
  min: Math.min, max: Math.max, abs: Math.abs, sin: Math.sin, cos: Math.cos, ceil: Math.ceil, floor: Math.floor,
  le: (a, b) => a <= b, ge: (a, b) => a >= b,
  or: (a, b) => a || b, and: (a, b) => a && b, select: (condition, yes, no) => condition ? yes : no,
  unpack(value, width) {
    const values = width === 1 ? [value] : Array.from(value ?? []);
    if (values.length !== width || values.some(v => typeof v !== 'number' || !Number.isFinite(v) || Math.abs(v) > 32)) {
      throw Error('CPU input endpoints must be finite scalar/vec3 values in [-32,32].');
    }
    return values;
  },
  pack: values => values.length === 1 ? values[0] : values,
};

function interpret(rootNode, bindings, B) {
  if (!(bindings instanceof Map)) throw Error('bindings must be a Map keyed by native input-node identity.');
  const details = {intentWrappers: new Set(), nodes: 0, operations: 0, inflatedComponents: 0, depth: 0, inputs: new Set()};
  const inputs = new Map(), cache = new Map(), active = new Set();
  for (const [key, range] of bindings) {
    const node = canonical(key, details);
    if (!inputTypes.has(typeName(node))) throw Error('Bindings may only replace uniform/attribute/property inputs or explicit scalar/vec3 swizzles.');
    if (inputs.has(node)) throw Error('Duplicate canonical input binding.');
    const width = inputWidth(node);
    const lo = B.unpack(range?.lo, width), hi = B.unpack(range?.hi, width);
    if (B.name === 'cpu' && lo.some((value, i) => value > hi[i])) throw Error('Interval lower endpoints must not exceed upper endpoints.');
    inputs.set(node, lo.map((value, i) => ({lo: value, hi: hi[i], magnitude: 32})));
  }

  function padded(lo, hi, magnitude, extraMargin = 0) {
    if (!Number.isFinite(magnitude) || magnitude > INTERVAL_LIMITS.maxEnvelope) throw Error('Expression growth exceeds the bounded float envelope.');
    details.inflatedComponents++;
    const padding = B.add(B.c(extraMargin), B.mul(B.c(INTERVAL_LIMITS.padding), B.add(B.c(1), B.max(B.abs(lo), B.abs(hi)))));
    return {lo: B.sub(lo, padding), hi: B.add(hi, padding), magnitude: magnitude + extraMargin + INTERVAL_LIMITS.padding * (1 + magnitude)};
  }
  function binary(a, b, operation) {
    if (a.length !== b.length && a.length !== 1 && b.length !== 1) throw Error('Unsupported vector-width mismatch.');
    return Array.from({length: Math.max(a.length, b.length)}, (_, i) => operation(a[a.length === 1 ? 0 : i], b[b.length === 1 ? 0 : i]));
  }
  function add(a, b) { return padded(B.add(a.lo, b.lo), B.add(a.hi, b.hi), a.magnitude + b.magnitude); }
  function sub(a, b) { return padded(B.sub(a.lo, b.hi), B.sub(a.hi, b.lo), a.magnitude + b.magnitude); }
  function multiply(a, b) {
    const products = [B.mul(a.lo, b.lo), B.mul(a.lo, b.hi), B.mul(a.hi, b.lo), B.mul(a.hi, b.hi)];
    return padded(products.reduce((x, y) => B.min(x, y)), products.reduce((x, y) => B.max(x, y)), a.magnitude * b.magnitude);
  }
  function trig(a, method) {
    const fn = B[method], period = B.c(1 / (2 * Math.PI));
    // Does [lo,hi] contain phase + k*2π? Include both endpoints.
    const contains = phase => B.le(B.ceil(B.mul(B.sub(a.lo, B.c(phase)), period)), B.floor(B.mul(B.sub(a.hi, B.c(phase)), period)));
    const maximumPhase = method === 'sin' ? Math.PI / 2 : 0;
    const minimumPhase = method === 'sin' ? -Math.PI / 2 : Math.PI;
    // Avoid relying on accurate phase reduction outside the tested trig domain.
    const outside = B.or(B.le(a.lo, B.c(-32)), B.ge(a.hi, B.c(32)));
    const lo = B.select(B.or(outside, contains(minimumPhase)), B.c(-1), B.min(fn(a.lo), fn(a.hi)));
    const hi = B.select(B.or(outside, contains(maximumPhase)), B.c(1), B.max(fn(a.lo), fn(a.hi)));
    // 1e-5 alone is below WGSL's permitted sine error even on [-π,π].
    // This larger explicit margin still does NOT prove larger-angle portability.
    return padded(lo, hi, 1, INTERVAL_LIMITS.trigAbsoluteMargin);
  }
  function absolute(a) {
    const aLo = B.abs(a.lo), aHi = B.abs(a.hi);
    const crossesZero = B.and(B.le(a.lo, B.c(0)), B.ge(a.hi, B.c(0)));
    return padded(B.select(crossesZero, B.c(0), B.min(aLo, aHi)), B.max(aLo, aHi), a.magnitude);
  }

  function visit(input, depth = 1) {
    const node = canonical(input, details);
    details.depth = Math.max(details.depth, depth);
    if (depth > INTERVAL_LIMITS.maxDepth) throw Error('Expression depth exceeds 32.');
    if (active.has(node)) throw Error('Cyclic expression DAG.');
    if (cache.has(node)) return cache.get(node);
    if (++details.nodes > INTERVAL_LIMITS.maxNodes) throw Error('Expression DAG exceeds 128 nodes.');
    active.add(node);
    const child = value => visit(value, depth + 1);
    let result;
    if (inputs.has(node)) {
      details.inputs.add(node);
      result = inputs.get(node);
    } else {
      const type = typeName(node);
      switch (type) {
        case 'ConstNode':
          result = constantValues(node).map(value => ({lo: B.c(value), hi: B.c(value), magnitude: Math.abs(value)}));
          break;
        case 'OperatorNode': {
          const operation = {'+': add, '-': sub, '*': multiply}[node.op];
          if (!operation || !node.bNode) throw Error(`Unsupported operator: ${node.op}.`);
          result = binary(child(node.aNode), child(node.bNode), operation);
          details.operations++;
          break;
        }
        case 'MathNode': {
          const a = child(node.aNode);
          if (node.cNode) throw Error('Three-input math is unsupported.');
          if (['sin', 'cos', 'abs', 'negate'].includes(node.method)) {
            if (node.bNode) throw Error('Unexpected second input to unary math.');
            result = a.map(value => node.method === 'abs' ? absolute(value) : node.method === 'negate'
              ? padded(B.sub(B.c(0), value.hi), B.sub(B.c(0), value.lo), value.magnitude) : trig(value, node.method));
          } else if (node.method === 'min' || node.method === 'max') {
            const operation = B[node.method];
            result = binary(a, child(node.bNode), (x, y) => padded(operation(x.lo, y.lo), operation(x.hi, y.hi), Math.max(x.magnitude, y.magnitude)));
          } else throw Error(`Unsupported MathNode method: ${node.method}.`);
          details.operations++;
          break;
        }
        case 'JoinNode': {
          if (!['vec3', 'vec3f'].includes(node.nodeType)) throw Error('Only vec3 joins are supported.');
          result = node.nodes.flatMap(child);
          if (result.length !== 3) throw Error('vec3 joins must contain exactly three scalar components.');
          break;
        }
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
          throw Error(`Unsupported or unbound TSL node: ${type}.`);
      }
    }
    active.delete(node);
    cache.set(node, result);
    return result;
  }
  const result = visit(rootNode);
  if (result.length !== 1 && result.length !== 3) throw Error('The root must produce float or vec3.');
  return {
    lo: B.pack(result.map(value => value.lo)), hi: B.pack(result.map(value => value.hi)),
    stats: Object.freeze({backend: B.name, nodes: details.nodes, operations: details.operations, inflatedComponents: details.inflatedComponents,
      maxDepth: details.depth, inputCount: details.inputs.size, unusedBindings: inputs.size - details.inputs.size,
      unwrappedIntents: details.intentWrappers.size, components: result.length, padding: INTERVAL_LIMITS.padding,
      trigAbsoluteMargin: INTERVAL_LIMITS.trigAbsoluteMargin, inputMagnitudeLimit: 32, formalFloatProof: false}),
  };
}

/**
 * Compile a pure native TSL float/vec3 expression into componentwise intervals.
 * bindings maps the EXACT input node (uniform/attribute/property, or an explicit
 * storageElement.xyz swizzle treated as an opaque input) to {lo,hi} TSL
 * nodes. Every runtime endpoint must be finite, ordered, and inside [-32,32].
 * The caller owns that validation; arbitrary GPU endpoints cannot be inspected
 * here. Pure expressions must not be assigned elsewhere or mutate after compile.
 * Native r185's automatic unnamed VarIntent wrappers are transparently unwrapped;
 * explicit variables, functions, storage reads, textures and unknown nodes fail.
 * Arithmetic is +,-,*, sin,cos,abs,min,max,negate; no division/matrices/control flow.
 * Trig adds a 0.001 absolute margin. Its [-32,32] phase/accuracy behavior is an
 * empirical fixture contract, not a portable WGSL guarantee outside [-π,π].
 * The result has practical per-operation padding, not a formal GPU-error proof.
 */
export function compileInterval(rootNode, bindings) { return interpret(rootNode, bindings, tslBackend); }

/** Same source-DAG traversal/arithmetic on CPU numbers, for containment checks. */
export function evaluateInterval(rootNode, bindings) { return interpret(rootNode, bindings, numericBackend); }

if (typeof process !== 'undefined' && process.argv?.includes('--self-test')) {
  const assert = (await import('node:assert/strict')).default;
  const {uniform, attribute, Fn} = await import('three/tsl');
  const p = attribute('position', 'vec3'), t = uniform(0, 'float'), x = uniform(0, 'float');
  const s = sin(t), c = cos(t);
  const expression = vec3(p.x.mul(c).sub(p.z.mul(s)), p.y.add(sin(p.x.mul(2).add(t)).mul(.25)), p.x.mul(s).add(p.z.mul(c)));
  const ranges = new Map([[p, {lo: [-1, -.5, -.2], hi: [.8, .9, 1.1]}], [t, {lo: .2, hi: 1.3}]]);
  const bounds = evaluateInterval(expression, ranges);
  let seed = 1234567;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  for (let sample = 0; sample < 2000; sample++) {
    const q = ranges.get(p).lo.map((v, i) => v + random() * (ranges.get(p).hi[i] - v));
    const phase = .2 + random() * 1.1;
    const expected = [q[0] * Math.cos(phase) - q[2] * Math.sin(phase), q[1] + Math.sin(2 * q[0] + phase) * .25, q[0] * Math.sin(phase) + q[2] * Math.cos(phase)];
    expected.forEach((v, i) => assert.ok(v >= bounds.lo[i] && v <= bounds.hi[i], 'independent twist sample escapes interval'));
  }
  const scalar = (node, lo, hi) => evaluateInterval(node, new Map([[x, {lo, hi}]]));
  const peak = scalar(sin(x), 1.5, 1.7); assert.ok(peak.hi >= 1 && peak.hi < 1.002);
  const narrow = scalar(sin(x), -.1, .1); assert.ok(narrow.lo > -.102 && narrow.hi < .102);
  const minimum = scalar(cos(x), 3, 3.2); assert.ok(minimum.lo <= -1 && minimum.lo > -1.002);
  const wide = scalar(sin(x), -8, 8); assert.ok(wide.lo <= -1 && wide.hi >= 1);
  const crossing = scalar(abs(x), -2, 1); assert.ok(crossing.lo <= 0 && crossing.hi >= 2);
  const product = evaluateInterval(x.mul(t), new Map([[x, {lo: -2, hi: 3}], [t, {lo: -4, hi: -1}]]));
  assert.ok(product.lo <= -12 && product.hi >= 8);
  const swizzle = evaluateInterval(vec3(p.zy, p.x).negate(), new Map([[p, {lo: [-1, 2, 3], hi: [0, 4, 5]}]]));
  assert.ok(swizzle.lo[0] <= -5 && swizzle.hi[2] >= 1);
  const symbolic = compileInterval(expression, new Map([[p, {lo: vec3(-1, -.5, -.2), hi: vec3(.8, .9, 1.1)}], [t, {lo: float(.2), hi: float(1.3)}]]));
  assert.ok(symbolic.lo.isNode && symbolic.hi.isNode);
  assert.equal(symbolic.stats.nodes, bounds.stats.nodes);
  assert.throws(() => evaluateInterval(x, new Map()), /unbound/);
  assert.throws(() => scalar(x.div(2), 0, 1), /Unsupported operator/);
  assert.throws(() => scalar(x.toVar(), 0, 1), /VarNode/);
  assert.throws(() => scalar(Fn(() => x.add(1))(), 0, 1), /unsupported|Unsupported/);
  assert.throws(() => scalar(x, 2, 1), /lower/);
  assert.throws(() => scalar(x, -33, 1), /finite/);
  console.log(JSON.stringify({message: 'Native TSL interval CPU tests passed; no GPU execution.', stats: bounds.stats, samples: 2000}));
}
