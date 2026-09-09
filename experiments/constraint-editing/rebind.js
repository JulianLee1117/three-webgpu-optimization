import {nodeObject} from 'three/tsl';

/**
 * Clone a bounded, pure native Three 0.185.1 expression onto exact new inputs.
 *
 * rebind(expression, new Map([[positionInput, sampledPosition], [time, sampleTime]]))
 * returns a native TSL node. rebindWithStats returns {root, stats}.
 *
 * Every nonconstant source leaf must be an explicitly bound uniform, attribute,
 * property or exact scalar/vector swizzle. Node identity is canonicalized through
 * .self and ordinary unnamed VarIntent wrappers; expression equivalence does not
 * count as identity. One source node produces one clone, preserving DAG sharing.
 * Source nodes, input maps, and vector constant values are never modified.
 *
 * Replacement expressions are caller-owned opaque input boundaries. This allows
 * exact GPU storage swizzles such as samples.element(instanceIndex).xyz without
 * interpreting their indexing or storage subtree. The caller must provide pure,
 * immutable replacements with the same scalar/vector width as the old input.
 * This utility rejects explicit/mutable variables and before() on the boundary,
 * but cannot prove purity, alias freedom or runtime domains inside opaque input
 * expressions. Unbound storage and texture/function expressions always reject.
 *
 * Only the scalar/vector arithmetic used by the material-mips AD subset is
 * cloned. This is graph rebinding, not differentiation, optimization or a general
 * TSL compiler. No GPU work, formula transcription, texture allocation or user JS
 * function execution occurs here. Three internal node shapes are version-sensitive.
 */
export const REBIND_LIMITS = Object.freeze({maxNodes: 4096, maxDepth: 128, maxBindings: 256});
const kind = n => n?.constructor?.type ?? n?.constructor?.name ?? 'unknown';
const widths = {float: 1, f32: 1, vec2: 2, vec2f: 2, vec3: 3, vec3f: 3, vec4: 4, vec4f: 4};
const boundaryKinds = new Set(['UniformNode', 'AttributeNode', 'PropertyNode', 'SplitNode']);
const unary = new Set(['sin', 'cos', 'exp', 'exp2', 'tanh', 'negate', 'oneMinus', 'sqrt', 'abs', 'floor', 'fract', 'length', 'normalize']);
const binary = new Set(['min', 'max', 'dot', 'pow']);
const ternary = new Set(['clamp', 'mix']);

function canonical(input, intents) {
  let n = input;
  const seen = new Set();
  for (;;) {
    if (!n?.isNode || seen.has(n)) throw Error('Expected an acyclic native TSL node.');
    seen.add(n);
    if (n._beforeNodes?.length) throw Error('before() side effects are unsupported.');
    if (n.self?.isNode && n.self !== n) {n = n.self; continue;}
    if (kind(n) !== 'VarNode') return n;
    if (n.intent !== true || n.name !== null || n.readOnly === true) throw Error('Explicit or mutable VarNode is unsupported.');
    intents.add(n); n = n.node;
  }
}

function knownWidth(n) {
  if (kind(n) === 'SplitNode') {
    if (typeof n.components !== 'string' || !/^[xyzw]{1,4}$/.test(n.components)) throw Error('Only existing xyzw components are supported.');
    // A width-one swizzle of integer storage is still an integer. Inspect the
    // available type metadata without evaluating an opaque storage/index tree.
    const seen = new Set([n]);
    for (let source = n.node; source && !seen.has(source); source = source.node) {
      seen.add(source);
      if (source.nodeType != null) {
        if (!widths[source.nodeType]) throw Error('Only floating scalar/vector swizzle sources are supported.');
        break;
      }
    }
    return n.components.length;
  }
  const type = kind(n) === 'ConvertNode' ? n.convertTo : n.nodeType;
  if (type != null) {
    if (!widths[type]) throw Error('Only floating scalar/vector types are supported.');
    return widths[type];
  }
  if (typeof n.value === 'number') return 1;
  for (const width of [2, 3, 4]) if (n.value?.[`isVector${width}`]) return width;
  return null; // Some replacement expression widths need a NodeBuilder to infer.
}

function constantCopy(n) {
  const width = knownWidth(n);
  if (!width) throw Error('Constant needs an explicit floating scalar/vector type.');
  const values = width === 1 ? [n.value] : [...'xyzw'.slice(0, width)].map(axis => n.value?.[axis]);
  if (values.some(x => typeof x !== 'number' || !Number.isFinite(x))) throw Error('Constants must be finite floating values.');
  return width === 1 ? n.value : n.value.clone();
}

export function rebindWithStats(rootNode, bindings) {
  if (!(bindings instanceof Map)) throw Error('bindings must be a Map keyed by exact native input nodes.');
  if (bindings.size > REBIND_LIMITS.maxBindings) throw Error('More than 256 exact input bindings.');
  const intents = new Set(), inputs = new Map(), used = new Set(), cache = new Map(), active = new Set();
  let visitedNodes = 0, clonedNodes = 0, maximumDepth = 0;
  for (const [key, value] of bindings) {
    const source = canonical(key, intents), replacement = canonical(value, intents);
    if (!boundaryKinds.has(kind(source))) throw Error('Binding keys must be exact uniform, attribute, property or swizzle inputs.');
    if (inputs.has(source)) throw Error('Duplicate canonical input binding.');
    if (replacement.isShaderCallNodeInternal || ['TextureNode', 'ConditionalNode', 'AssignNode'].includes(kind(replacement))) throw Error('Function, texture and effectful replacement boundaries are unsupported.');
    const sourceWidth = knownWidth(source), targetWidth = knownWidth(replacement);
    if (!sourceWidth) throw Error('Source input needs an explicit floating scalar/vector type.');
    if (targetWidth && sourceWidth !== targetWidth) throw Error('Replacement input width does not match the source input.');
    inputs.set(source, nodeObject(replacement));
  }
  function visit(input, depth = 1) {
    const n = canonical(input, intents);
    maximumDepth = Math.max(maximumDepth, depth);
    if (depth > REBIND_LIMITS.maxDepth) throw Error('Expression depth exceeds 128.');
    if (active.has(n)) throw Error('Cyclic expression DAG.');
    if (cache.has(n)) return cache.get(n);
    if (++visitedNodes > REBIND_LIMITS.maxNodes) throw Error('More than 4096 native expression nodes.');
    if (inputs.has(n)) {
      const result = inputs.get(n); used.add(n); cache.set(n, result); return result;
    }
    active.add(n);
    const child = input => visit(input, depth + 1);
    let result;
    switch (kind(n)) {
      case 'ConstNode': result = new n.constructor(constantCopy(n), n.nodeType); break;
      case 'OperatorNode':
        if (!['+', '-', '*', '/'].includes(n.op) || !n.aNode || !n.bNode) throw Error(`Unsupported operator ${n.op}.`);
        result = new n.constructor(n.op, child(n.aNode), child(n.bNode)); break;
      case 'MathNode': {
        let args;
        if (unary.has(n.method) && n.aNode && !n.bNode && !n.cNode) args = [child(n.aNode)];
        else if (binary.has(n.method) && n.aNode && n.bNode && !n.cNode) {
          if (n.method === 'pow') {
            const exponent = canonical(n.bNode, intents);
            if (kind(exponent) !== 'ConstNode' || knownWidth(exponent) !== 1) throw Error('pow requires a literal scalar exponent.');
          }
          args = [child(n.aNode), child(n.bNode)];
        } else if (ternary.has(n.method) && n.aNode && n.bNode && n.cNode) args = [child(n.aNode), child(n.bNode), child(n.cNode)];
        else throw Error(`Unsupported MathNode method or arity: ${n.method}.`);
        // MathNode min/max interpret a fourth argument as another operand, even
        // when it is null. Preserve actual arity instead of padding arguments.
        result = new n.constructor(n.method, ...args); break;
      }
      case 'JoinNode':
        if ((n.nodeType != null && !widths[n.nodeType]) || !Array.isArray(n.nodes) || n.nodes.length < 1 || n.nodes.length > 4) throw Error('Only floating vector joins are supported.');
        result = new n.constructor(n.nodes.map(child), n.nodeType); break;
      case 'SplitNode':
        knownWidth(n); result = new n.constructor(child(n.node), n.components); break;
      case 'ConvertNode':
        if (!widths[n.convertTo]) throw Error(`Unsupported conversion to ${n.convertTo}.`);
        result = new n.constructor(child(n.node), n.convertTo); break;
      default: throw Error(`Unsupported or unbound source node: ${kind(n)}.`);
    }
    clonedNodes++;
    result = nodeObject(result); active.delete(n); cache.set(n, result); return result;
  }
  const root = visit(rootNode);
  return {root, stats: Object.freeze({visitedNodes, clonedNodes, maxDepth: maximumDepth,
    bindings: inputs.size, usedBindings: used.size, unusedBindings: inputs.size - used.size,
    unwrappedIntents: intents.size, method: 'exact-input pure native TSL DAG cloning'})};
}

export function rebind(rootNode, bindings) {return rebindWithStats(rootNode, bindings).root;}
