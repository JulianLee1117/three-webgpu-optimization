import {float, vec2, vec3, vec4, sin, cos, exp, tanh, exp2, sqrt, abs, floor, fract, pow} from 'three/tsl';
import D_GGX from 'three/src/nodes/functions/BSDF/D_GGX.js';
import F_Schlick from 'three/src/nodes/functions/BSDF/F_Schlick.js';
import V_GGX_SmithCorrelated from 'three/src/nodes/functions/BSDF/V_GGX_SmithCorrelated.js';
import BRDF_Lambert from 'three/src/nodes/functions/BSDF/BRDF_Lambert.js';
import BRDF_GGX from 'three/src/nodes/functions/BSDF/BRDF_GGX.js';
import BRDF_GGX_Multiscatter from 'three/src/nodes/functions/BSDF/BRDF_GGX_Multiscatter.js';

/**
 * Reverse-mode VJP for a bounded, pure native Three r185 TSL expression DAG.
 *
 * gradients(loss, [weights, bias], {constants: [position, target]}) returns
 * {value: loss, gradients: [dWeights, dBias], stats}. The loss must be scalar;
 * each designated variable may be float/vec2/vec3/vec4. Gradients preserve its
 * width and order. Repeated uses, repeated swizzles and scalar broadcasts sum
 * their adjoints. evaluateGradients has the same rules, with a third argument
 * Map(inputNode -> finite number/array), and returns a numeric scalar value.
 *
 * Designated nodes are EXACT opaque input boundaries. A storage.element(i).x
 * or .xyzw swizzle may be a variable or constant, without traversing its storage
 * and index subtree. The caller promises designated constants do not depend on
 * any variable, designated scalar components do not alias one another, values
 * do not mutate during evaluation, and no source expression
 * is assigned elsewhere. Unknown/unbound storage, uniforms, functions, explicit
 * variables, side effects, matrices, non-float types and unsupported math fail.
 * Normal unnamed r185 VarIntent expression wrappers are transparently unwrapped.
 *
 * Material-mips extension of the separately frozen trainable-tsl compiler.
 * Supported: +,-,*,/, sin/cos/exp/exp2/tanh/negate/oneMinus/dot, sqrt/abs,
 * min/max/clamp/mix/length/normalize, floor/fract, and pow with a literal scalar
 * exponent; vector joins, xyzw swizzles,
 * float/vector conversions. Vector extension follows Three NodeBuilder.format:
 * vec2 -> vec3 appends 0; vec3 -> vec4 appends 1; scalar -> vector broadcasts.
 * Ordinary swizzles must address existing vector components (scalar swizzles
 * broadcast); implicit padded accesses such as vec2.z are rejected. Exact
 * designated opaque input boundaries retain their caller-supplied values.
 * Runtime divisions and exponentials must stay finite and in their domains.
 * CPU checks catch nonfinite intermediate values/gradients; symbolic compilation
 * cannot inspect GPU runtime inputs. The generated graph uses ordinary floating
 * point and standard AD, not exact arithmetic or a numerical stability guarantee.
 * No GPU execution, optimizer, resource allocation or training occurs here.
 *
 * Piecewise conventions: min/max ties route to the first operand; clamp uses
 * zero partials for every input at either exact bound; abs(0) has derivative 0;
 * floor/fract use 0/1, valid only away from integer discontinuities. clamp bounds
 * must be ordered, pow bases positive, differentiated sqrt arguments positive,
 * and normalize inputs nonzero. These are caller-owned GPU domain contracts.
 * Arbitrary conditionals and general Fn bodies still reject. Only the imported
 * stock helper identities below may be expanded, with unchanged JS functions,
 * pure node arguments and pure outputs; iridescence/anisotropy flags reject.
 * DFGLUT is not expanded because its body creates a texture.
 *
 * options.replacements is an optional Map(exactOriginalNode -> pureGraph).
 * It explicitly substitutes the differentiated model, including its CPU value;
 * symbolic .value remains the ORIGINAL root. It is not a proof of primal parity.
 * Callers must independently compare against the actual renderer. When a
 * replacement uses selected texels as designated constants, independence holds
 * only locally inside a fixed interpolation cell, away from its boundaries.
 * No general texture/storage interpretation or arbitrary JS callbacks occur.
 */
export const AUTOGRAD_LIMITS = Object.freeze({maxScalarTapeOps: 4096, maxScalarParameters: 128, maxDepth: 128});
const axes = 'xyzw';
const designatedTypes = new Set(['UniformNode', 'AttributeNode', 'PropertyNode', 'SplitNode']);
const typeName = node => node?.constructor?.type ?? node?.constructor?.name ?? 'unknown';
const widthOfType = type => ({float: 1, f32: 1, vec2: 2, vec2f: 2, vec3: 3, vec3f: 3, vec4: 4, vec4f: 4})[type];
const stockFunctions = new Map([
  ['D_GGX', D_GGX], ['F_Schlick', F_Schlick], ['V_GGX_SmithCorrelated', V_GGX_SmithCorrelated],
  ['BRDF_Lambert', BRDF_Lambert], ['BRDF_GGX', BRDF_GGX], ['BRDF_GGX_Multiscatter', BRDF_GGX_Multiscatter],
].map(([name, fn]) => [fn.shaderNode, {name, jsFunc: fn.shaderNode.jsFunc}]));
export const STOCK_FUNCTION_NAMES = Object.freeze([...stockFunctions.values()].map(entry => entry.name));

function canonical(input, intents) {
  let node = input;
  const seen = new Set();
  for (;;) {
    if (!node?.isNode || seen.has(node)) throw Error('Expected an acyclic native TSL node.');
    seen.add(node);
    if (node._beforeNodes?.length) throw Error('Nodes with before() side effects are unsupported.');
    if (node.self?.isNode && node.self !== node) {node = node.self; continue;}
    if (typeName(node) !== 'VarNode') return node;
    if (node.intent !== true || node.name !== null || node.readOnly === true) throw Error('Explicit or mutable VarNode is unsupported.');
    intents.add(node);
    node = node.node;
  }
}

function inputWidth(node) {
  if (typeName(node) === 'SplitNode' && /^[xyzw]{1,4}$/.test(node.components)) {
    // Inspect only type metadata on opaque storage/leaf boundaries, never
    // interpret a storage access or its index. Integer swizzles must not be
    // silently accepted as differentiable floating-point parameters.
    let source = node.node;
    const seen = new Set([node]);
    while (source) {
      if (seen.has(source)) throw Error('Cyclic designated input type.');
      seen.add(source);
      if (widthOfType(source.nodeType)) return node.components.length;
      if (source.nodeType != null) throw Error('Designated swizzle source must have a floating-point type.');
      if (['SplitNode', 'StorageArrayElementNode', 'ArrayElementNode'].includes(typeName(source))) {source = source.node; continue;}
      throw Error('Designated swizzle source needs explicit floating-point type metadata.');
    }
    throw Error('Designated swizzle source is missing.');
  }
  const width = widthOfType(node.nodeType);
  if (width) return width;
  if (node.nodeType == null) {
    if (typeof node.value === 'number') return 1;
    for (const size of [2, 3, 4]) if (node.value?.[`isVector${size}`]) return size;
  }
  throw Error(`A designated ${typeName(node)} needs float/vec2/vec3/vec4 type.`);
}

function constantValues(node) {
  const width = inputWidth(node);
  const values = width === 1 ? [node.value] : [...axes.slice(0, width)].map(axis => node.value?.[axis]);
  if (values.some(value => typeof value !== 'number' || !Number.isFinite(value))) throw Error('Constants must be finite float/vector values.');
  return values;
}

const tslBackend = {
  name: 'tsl', c: float, sin, cos, exp, tanh, exp2, sqrt, abs, floor, fract, pow,
  add: (a, b) => a.add(b), sub: (a, b) => a.sub(b), mul: (a, b) => a.mul(b), div: (a, b) => a.div(b),
  min: (a, b) => a.min(b), max: (a, b) => a.max(b),
  choose: (relation, a, b, yes, no) => a[relation](b).select(yes, no),
  unpack: (node, width) => width === 1 ? [node] : [...axes.slice(0, width)].map(axis => node[axis]),
  pack: values => values.length === 1 ? values[0] : [null, null, vec2, vec3, vec4][values.length](...values),
};
const numericBackend = {
  name: 'cpu', c: value => value, sin: Math.sin, cos: Math.cos, exp: Math.exp, tanh: Math.tanh,
  exp2: x => 2 ** x, sqrt: Math.sqrt, abs: Math.abs, floor: Math.floor, fract: x => x - Math.floor(x), pow: Math.pow,
  min: Math.min, max: Math.max,
  choose: (relation, a, b, yes, no) => ({lessThan: a < b, greaterThan: a > b, lessThanEqual: a <= b, greaterThanEqual: a >= b, equal: a === b})[relation] ? yes : no,
  add: (a, b) => a + b, sub: (a, b) => a - b, mul: (a, b) => a * b, div: (a, b) => a / b,
  unpack(value, width) {
    const values = width === 1 ? [value] : Array.from(value ?? []);
    if (values.length !== width || values.some(v => typeof v !== 'number' || !Number.isFinite(v))) throw Error('CPU inputs must be finite numbers or matching-width arrays.');
    return values;
  },
  pack: values => values.length === 1 ? values[0] : values,
};

function interpret(rootNode, variables, {constants = [], replacements = new Map()}, B, values) {
  if (!Array.isArray(variables) || !Array.isArray(constants)) throw Error('variables and constants must be arrays of exact native input nodes.');
  if (!(replacements instanceof Map)) throw Error('replacements must be a Map of exact nodes to pure graphs.');
  if (B.name === 'cpu' && !(values instanceof Map)) throw Error('CPU values must be a Map keyed by exact input-node identity.');
  const intents = new Set(), inputs = new Map(), numericValues = new Map(), cache = new Map(), active = new Set();
  const tape = [], variableRecords = [], usedInputs = new Set();
  let scalarParameters = 0, nativeNodes = 0, maximumDepth = 0, reverseAccumulations = 0, replacementNodes = 0;
  const substitution = new Map(), expandedFunctions = new Set();
  for (const [key, graph] of replacements) {
    const original = canonical(key, intents), replacement = canonical(graph, intents);
    if (substitution.has(original)) throw Error('Duplicate canonical replacement node.');
    if (original === replacement) throw Error('Replacement cannot refer to itself.');
    substitution.set(original, replacement);
  }
  if (values) for (const [key, value] of values) {
    const node = canonical(key, intents);
    if (numericValues.has(node)) throw Error('Duplicate canonical CPU input value.');
    numericValues.set(node, value);
  }
  for (const [isVariable, collection] of [[true, variables], [false, constants]]) for (const input of collection) {
    const node = canonical(input, intents);
    if (!designatedTypes.has(typeName(node))) throw Error('Designated inputs must be uniforms, attributes, properties, or exact scalar/vector swizzles.');
    if (inputs.has(node)) throw Error('Duplicate or overlapping variable/constant input.');
    if (substitution.has(node)) throw Error('A designated input cannot also be replaced.');
    const width = inputWidth(node);
    if (isVariable && (scalarParameters += width) > AUTOGRAD_LIMITS.maxScalarParameters) throw Error('More than 128 designated scalar parameters.');
    if (B.name === 'cpu' && !numericValues.has(node)) throw Error('Missing CPU value for designated input.');
    // Validate the complete declared input contract even for disconnected
    // parameters/constants whose nodes are never reached from the loss.
    const cpuComponents = B.name === 'cpu' ? B.unpack(numericValues.get(node), width) : null;
    const record = {node, width, isVariable, ids: null, cpuComponents};
    inputs.set(node, record);
    if (isVariable) variableRecords.push(record);
  }
  function finite(value) {
    if (B.name === 'cpu' && !Number.isFinite(value)) throw Error('Nonfinite intermediate value, local derivative, or adjoint.');
    return value;
  }
  function append(value, parents = [], differentiable = parents.some(([id]) => tape[id].differentiable)) {
    if (tape.length >= AUTOGRAD_LIMITS.maxScalarTapeOps) throw Error('More than 4096 scalar tape operations.');
    parents = parents.filter(([id]) => tape[id].differentiable);
    parents.forEach(([, partial]) => finite(partial));
    const id = tape.length;
    tape.push({value: finite(value), parents, differentiable});
    return id;
  }
  const leaf = value => append(B.c(value));
  const v = id => tape[id].value;
  function add(a, b) {return append(B.add(v(a), v(b)), [[a, B.c(1)], [b, B.c(1)]]);}
  function subtract(a, b) {return append(B.sub(v(a), v(b)), [[a, B.c(1)], [b, B.c(-1)]]);}
  function multiply(a, b) {return append(B.mul(v(a), v(b)), [[a, v(b)], [b, v(a)]]);}
  function divide(a, b) {return append(B.div(v(a), v(b)), [[a, B.div(B.c(1), v(b))], [b, B.div(B.sub(B.c(0), v(a)), B.mul(v(b), v(b)))]]);}
  function unary(a, method) {
    const value = method === 'negate' ? B.sub(B.c(0), v(a)) : method === 'oneMinus' ? B.sub(B.c(1), v(a)) : B[method](v(a));
    const derivative = method === 'sin' ? B.cos(v(a)) : method === 'cos' ? B.sub(B.c(0), B.sin(v(a)))
      : method === 'exp' ? value : method === 'exp2' ? B.mul(B.c(Math.LN2), value)
      : method === 'tanh' ? B.sub(B.c(1), B.mul(value, value))
      : method === 'sqrt' ? B.div(B.c(.5), value)
      : method === 'floor' ? B.c(0) : method === 'fract' ? B.c(1)
      : method === 'abs' ? B.choose('greaterThan', v(a), B.c(0), B.c(1), B.choose('lessThan', v(a), B.c(0), B.c(-1), B.c(0))) : B.c(-1);
    return append(value, [[a, derivative]]);
  }
  function extremum(a, b, method) {
    const first = B.choose(method === 'max' ? 'greaterThanEqual' : 'lessThanEqual', v(a), v(b), B.c(1), B.c(0));
    return append(B[method](v(a), v(b)), [[a, first], [b, B.sub(B.c(1), first)]]);
  }
  function clampValue(a, low, high) {
    if (B.name === 'cpu' && v(low) > v(high)) throw Error('clamp bounds must be ordered.');
    const atBoundary = value => B.choose('equal', v(a), v(low), B.c(0), B.choose('equal', v(a), v(high), B.c(0), value));
    const inside = B.choose('greaterThan', v(a), v(low), B.choose('lessThan', v(a), v(high), B.c(1), B.c(0)), B.c(0));
    return append(B.min(B.max(v(a), v(low)), v(high)), [[a, inside],
      [low, atBoundary(B.choose('lessThan', v(a), v(low), B.c(1), B.c(0)))],
      [high, atBoundary(B.choose('greaterThan', v(a), v(high), B.c(1), B.c(0)))]]);
  }
  function binary(a, b, operation) {
    if (a.length !== b.length && a.length !== 1 && b.length !== 1) throw Error('Unsupported vector-width mismatch.');
    return Array.from({length: Math.max(a.length, b.length)}, (_, i) => operation(a[a.length === 1 ? 0 : i], b[b.length === 1 ? 0 : i]));
  }
  function ternary(a, b, c, operation) {
    const width = Math.max(a.length, b.length, c.length);
    if ([a, b, c].some(x => x.length !== 1 && x.length !== width)) throw Error('Unsupported vector-width mismatch.');
    return Array.from({length: width}, (_, i) => operation(...[a, b, c].map(x => x[x.length === 1 ? 0 : i])));
  }
  function convert(source, width) {
    if (source.length >= width) return source.slice(0, width);
    if (source.length === 1) return Array(width).fill(source[0]);
    const result = [...source];
    while (result.length < width) result.push(leaf(result.length === 3 ? 1 : 0));
    return result;
  }
  function visit(input, depth = 1) {
    const node = canonical(input, intents);
    maximumDepth = Math.max(maximumDepth, depth);
    if (depth > AUTOGRAD_LIMITS.maxDepth) throw Error('Expression depth exceeds 128.');
    if (active.has(node)) throw Error('Cyclic expression DAG.');
    if (cache.has(node)) return cache.get(node);
    // Repeated structural views need a cap as well as arithmetic scalar entries.
    if (++nativeNodes > AUTOGRAD_LIMITS.maxScalarTapeOps) throw Error('More than 4096 native expression nodes.');
    active.add(node);
    const child = input => visit(input, depth + 1);
    let result;
    if (substitution.has(node)) {
      replacementNodes++;
      result = child(substitution.get(node));
    } else if (node.isShaderCallNodeInternal === true) {
      const entry = stockFunctions.get(node.shaderNode);
      if (!entry || node.shaderNode.jsFunc !== entry.jsFunc) throw Error('Unsupported or modified stock Fn.');
      if (!Array.isArray(node.rawInputs) || node.rawInputs.length !== 1 || Object.getPrototypeOf(node.rawInputs[0]) !== Object.prototype) throw Error('Stock Fn requires one plain named-input object.');
      const args = {};
      for (const [key, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(node.rawInputs[0]))) {
        if (!('value' in descriptor)) throw Error('Stock Fn input accessors are unsupported.');
        const value = descriptor.value;
        if (value === undefined) continue;
        if (key.startsWith('USE_')) throw Error('Optional stock Fn branches are unsupported.');
        if (!value?.isNode && (typeof value !== 'number' || !Number.isFinite(value))) throw Error('Stock Fn arguments must be native nodes or finite float literals.');
        const inputNode = value?.isNode ? value : float(value);
        child(inputNode); // Even unused arguments must satisfy the pure input contract.
        args[key] = inputNode;
      }
      expandedFunctions.add(entry.name);
      result = child(entry.jsFunc(args));
    } else if (inputs.has(node)) {
      const record = inputs.get(node);
      const values = B.name === 'tsl' ? B.unpack(node, record.width) : record.cpuComponents;
      result = values.map(value => append(value, [], record.isVariable));
      record.ids = result;
      usedInputs.add(node);
    } else switch (typeName(node)) {
      case 'ConstNode': result = constantValues(node).map(leaf); break;
      case 'OperatorNode': {
        const operation = {'+': add, '-': subtract, '*': multiply, '/': divide}[node.op];
        if (!operation || !node.bNode) throw Error(`Unsupported operator: ${node.op}.`);
        result = binary(child(node.aNode), child(node.bNode), operation);
        break;
      }
      case 'MathNode': {
        const a = child(node.aNode);
        if (['sin', 'cos', 'exp', 'exp2', 'tanh', 'negate', 'oneMinus', 'sqrt', 'abs', 'floor', 'fract'].includes(node.method) && !node.bNode && !node.cNode) result = a.map(id => unary(id, node.method));
        else if (['length', 'normalize'].includes(node.method) && !node.bNode && !node.cNode) {
          const length = unary(a.map(id => multiply(id, id)).reduce(add), 'sqrt');
          result = node.method === 'length' ? [length] : a.map(id => divide(id, length));
        } else if (['min', 'max'].includes(node.method) && node.bNode && !node.cNode) result = binary(a, child(node.bNode), (x, y) => extremum(x, y, node.method));
        else if (node.method === 'clamp' && node.bNode && node.cNode) result = ternary(a, child(node.bNode), child(node.cNode), clampValue);
        else if (node.method === 'mix' && node.bNode && node.cNode) result = ternary(a, child(node.bNode), child(node.cNode), (x, y, t) => add(multiply(x, subtract(leaf(1), t)), multiply(y, t)));
        else if (node.method === 'pow' && node.bNode && !node.cNode) {
          const exponentNode = canonical(node.bNode, intents);
          if (typeName(exponentNode) !== 'ConstNode' || inputWidth(exponentNode) !== 1) throw Error('pow requires a literal scalar exponent.');
          const exponent = constantValues(exponentNode)[0];
          result = a.map(id => {
            if (B.name === 'cpu' && !(v(id) > 0)) throw Error('pow base must be positive.');
            return append(B.pow(v(id), B.c(exponent)), [[id, exponent === 0 ? B.c(0) : B.mul(B.c(exponent), B.pow(v(id), B.c(exponent - 1)))]]);
          });
        } else if (node.method === 'dot' && node.bNode && !node.cNode) {
          const b = child(node.bNode);
          if (a.length !== b.length || a.length < 2) throw Error('dot requires matching vectors of width two to four.');
          result = [a.map((id, i) => multiply(id, b[i])).reduce(add)];
        } else throw Error(`Unsupported MathNode method: ${node.method}.`);
        break;
      }
      case 'JoinNode': {
        const width = widthOfType(node.nodeType);
        if (width === 1 || (node.nodeType != null && !width)) throw Error('Only float vector joins are supported.');
        result = node.nodes.flatMap(child);
        if (result.length < 2 || result.length > 4 || (width && result.length !== width)) throw Error('Vector join length must match its type.');
        break;
      }
      case 'SplitNode': {
        if (typeof node.components !== 'string' || !/^[xyzw]{1,4}$/.test(node.components)) throw Error('Only xyzw swizzles up to width four are supported.');
        const source = child(node.node);
        result = [...node.components].map(axis => {
          const index = source.length === 1 ? 0 : axes.indexOf(axis);
          if (index >= source.length) throw Error('Swizzle accesses an absent component.');
          return source[index];
        });
        break;
      }
      case 'ConvertNode': {
        const width = widthOfType(node.convertTo);
        if (!width) throw Error(`Unsupported conversion to ${node.convertTo}.`);
        result = convert(child(node.node), width);
        break;
      }
      default: throw Error(`Unsupported or unbound TSL node: ${typeName(node)}.`);
    }
    active.delete(node); cache.set(node, result); return result;
  }
  const output = visit(rootNode);
  if (output.length !== 1) throw Error('The loss root must be scalar.');
  const adjoints = new Array(tape.length);
  adjoints[output[0]] = B.c(1);
  for (let id = tape.length - 1; id >= 0; id--) {
    if (adjoints[id] === undefined) continue;
    for (const [parent, partial] of tape[id].parents) {
      const contribution = finite(B.mul(adjoints[id], partial));
      adjoints[parent] = finite(adjoints[parent] === undefined ? contribution : B.add(adjoints[parent], contribution));
      reverseAccumulations++;
    }
  }
  return {
    value: B.name === 'tsl' ? rootNode : v(output[0]),
    gradients: variableRecords.map(record => B.pack(Array.from({length: record.width}, (_, i) => record.ids && adjoints[record.ids[i]] !== undefined ? adjoints[record.ids[i]] : B.c(0)))),
    stats: Object.freeze({backend: B.name, method: 'standard reverse-mode VJP', nativeNodes, scalarTapeOps: tape.length,
      scalarParameters, reverseAccumulations, maxDepth: maximumDepth, designatedInputs: inputs.size, usedInputs: usedInputs.size,
      unusedInputs: inputs.size - usedInputs.size, unwrappedIntents: intents.size, replacementNodes,
      unusedReplacements: substitution.size - replacementNodes, differentiatedModelSubstituted: replacementNodes > 0,
      expandedStockFunctions: Object.freeze([...expandedFunctions])}),
  };
}

export function gradients(rootNode, variables, options = {}) {return interpret(rootNode, variables, options, tslBackend);}
export function evaluateGradients(rootNode, variables, values, options = {}) {return interpret(rootNode, variables, options, numericBackend, values);}
