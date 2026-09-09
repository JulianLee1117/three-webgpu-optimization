import {float, vec2, vec3, vec4, ivec2, nodeObject, textureLoad} from 'three/tsl';
import {HalfFloatType, RGFormat, LinearFilter, ClampToEdgeWrapping} from 'three';
import D_GGX from '../../node_modules/three/src/nodes/functions/BSDF/D_GGX.js';
import F_Schlick from '../../node_modules/three/src/nodes/functions/BSDF/F_Schlick.js';
import V_GGX_SmithCorrelated from '../../node_modules/three/src/nodes/functions/BSDF/V_GGX_SmithCorrelated.js';
import BRDF_GGX from '../../node_modules/three/src/nodes/functions/BSDF/BRDF_GGX.js';
import BRDF_GGX_Multiscatter from '../../node_modules/three/src/nodes/functions/BSDF/BRDF_GGX_Multiscatter.js';
import DFGLUT from '../../node_modules/three/src/nodes/functions/BSDF/DFGLUT.js';
import {normalView} from '../../node_modules/three/src/nodes/accessors/Normal.js';
import {positionViewDirection} from '../../node_modules/three/src/nodes/accessors/Position.js';

export const METAL_F0 = Object.freeze([.65, .3, .12]);
export const SHADING_SOURCE_FILES = Object.freeze([
  'node_modules/three/package.json',
  ...['D_GGX', 'F_Schlick', 'V_GGX_SmithCorrelated', 'BRDF_GGX', 'BRDF_GGX_Multiscatter', 'DFGLUT']
    .map(name => `node_modules/three/src/nodes/functions/BSDF/${name}.js`),
  'node_modules/three/src/nodes/functions/PhysicalLightingModel.js',
  'node_modules/three/src/nodes/functions/material/getRoughness.js',
  'node_modules/three/src/nodes/tsl/TSLCore.js',
]);
export const SHADING_SCOPE = Object.freeze({
  three: '0.185.1', metalness: 1, f90: 1, diffuse: 0,
  lighting: 'Opaque isotropic metallic direct punctual-light response; no shadows, IBL, tone mapping or color-space conversion.',
  adaptation: 'Allowlisted upstream Fn JavaScript bodies expanded without changing their arithmetic; exact normal/view globals rebound to explicit inputs.',
  lutDerivative: 'Piecewise bilinear coordinate derivative of the unchanged RG16F DFG table; cell-selection texel loads are locally constant. Cell boundaries and clamping kinks are nondifferentiable.',
  privateAPI: 'Fn.shaderNode.jsFunc and native node representation are private/version-sensitive implementation details.',
});

const kind = n => n?.constructor?.type ?? n?.constructor?.name;
function self(n) {return n?.self?.isNode && n.self !== n ? n.self : n;}
function unwrap(n) {
  n = self(n);
  while (kind(n) === 'VarNode' && n.intent === true && n.name === null && !n.readOnly) n = self(n.node);
  return n;
}
const approved = new Map([D_GGX, F_Schlick, V_GGX_SmithCorrelated, BRDF_GGX, BRDF_GGX_Multiscatter, DFGLUT]
  .map(fn => [fn.shaderNode, fn]));

// Invoke only this exact upstream initializer. The table is not copied into this
// file, altered, approximated or retrained. Allocation here is a CPU DataTexture.
const dfgInitial = DFGLUT.shaderNode.jsFunc({roughness: float(.5), dotNV: float(.5)});
const dfgTextureNode = unwrap(unwrap(dfgInitial).node);
export const DFG_TEXTURE = dfgTextureNode.value;
if (!DFG_TEXTURE?.isDataTexture || DFG_TEXTURE.image.width !== 16 || DFG_TEXTURE.image.height !== 16
  || !(DFG_TEXTURE.image.data instanceof Uint16Array) || DFG_TEXTURE.image.data.length !== 512
  || DFG_TEXTURE.type !== HalfFloatType || DFG_TEXTURE.format !== RGFormat
  || DFG_TEXTURE.minFilter !== LinearFilter || DFG_TEXTURE.magFilter !== LinearFilter
  || DFG_TEXTURE.wrapS !== ClampToEdgeWrapping || DFG_TEXTURE.wrapT !== ClampToEdgeWrapping
  || DFG_TEXTURE.generateMipmaps || DFG_TEXTURE.flipY) throw Error('Unexpected upstream DFG texture contract');

export function decodeFloat16(bits) {
  const sign = bits & 0x8000 ? -1 : 1, exponent = (bits >>> 10) & 31, fraction = bits & 1023;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return exponent === 0 ? sign * 2 ** -14 * (fraction / 1024) : sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}
export const DFG_DATA = Object.freeze(Array.from(DFG_TEXTURE.image.data, decodeFloat16));
const clampNumber = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
function texel(x, y) {
  const i = 2 * (clampNumber(y, 0, 15) * 16 + clampNumber(x, 0, 15));
  return [DFG_DATA[i], DFG_DATA[i + 1]];
}
export function dfgCell(roughness, dotNV) {
  if (![roughness, dotNV].every(Number.isFinite)) throw Error('Nonfinite DFG coordinates');
  const x = 16 * roughness - .5, y = 16 * dotNV - .5;
  const ix = Math.floor(x), iy = Math.floor(y);
  return {fraction: [x - ix, y - iy], texels: [texel(ix, iy), texel(ix+1, iy), texel(ix, iy+1), texel(ix+1, iy+1)]};
}
export function numericDFG(roughness, dotNV) {
  const {fraction: [tx, ty], texels: [a, b, c, d]} = dfgCell(roughness, dotNV);
  return a.map((_, i) => (a[i] * (1-tx) + b[i] * tx) * (1-ty) + (c[i] * (1-tx) + d[i] * tx) * ty);
}
export function normalFromSlopes(x, y) {return vec3(x, y, 1).normalize();}

/**
 * Expand the actual upstream pure functions into a native expression. Global
 * normal/view accessors are replaced BEFORE entering their context/mutable code.
 * Unknown functions, mutating nodes and unbound accessors fail. No global node or
 * upstream helper is patched. Textures keep actual hardware sampling in forward.
 */
export function expandStockMetallic(expression, bindings, opaqueInputs = []) {
  const substitutions = new Map([...bindings].map(([a, b]) => [self(a), b]));
  const opaque = new Set(opaqueInputs.map(self));
  const seen = new Map(), active = new Set(), textures = [], expandedFunctions = new Set();
  let visited = 0;
  function visit(input) {
    const n = self(input);
    if (!n?.isNode) throw Error('Expected a native TSL node');
    if (substitutions.has(n)) return substitutions.get(n);
    if (opaque.has(n)) return nodeObject(n);
    if (seen.has(n)) return seen.get(n);
    if (active.has(n) || ++visited > 4096) throw Error('Cyclic or oversized stock shading graph');
    if (n._beforeNodes?.length) throw Error('Stock shading graph contains side effects');
    active.add(n); let result;
    const type = kind(n);
    if (type === 'VarNode') {
      if (!n.intent || n.name !== null || n.readOnly) throw Error('Unexpected explicit stock variable');
      result = visit(n.node);
    } else if (n.isShaderCallNodeInternal) {
      if (!approved.has(n.shaderNode)) throw Error('Unapproved upstream shader function');
      const args = n.rawInputs;
      if (!Array.isArray(args) || args.length !== 1 || !args[0] || args[0].isNode) throw Error('Unexpected upstream Fn inputs');
      const params = Object.fromEntries(Object.entries(args[0]).map(([key, value]) =>
        [key, typeof value === 'number' ? float(value) : value]));
      expandedFunctions.add(n.shaderNode.jsFunc);
      result = visit(n.shaderNode.jsFunc(params));
    } else if (type === 'ConstNode') result = nodeObject(n);
    else if (type === 'OperatorNode') result = nodeObject(new n.constructor(n.op, visit(n.aNode), visit(n.bNode)));
    else if (type === 'MathNode') {
      // MathNode max/min use arguments.length for variadic folding. Supplying a
      // fourth null argument would create an unintended extra max/min node.
      const args = [n.method, visit(n.aNode)];
      if (n.bNode) args.push(visit(n.bNode));
      if (n.cNode) args.push(visit(n.cNode));
      result = nodeObject(new n.constructor(...args));
    }
    else if (type === 'SplitNode') result = nodeObject(new n.constructor(visit(n.node), n.components));
    else if (type === 'JoinNode') result = nodeObject(new n.constructor(n.nodes.map(visit), n.nodeType));
    else if (type === 'ConvertNode') result = nodeObject(new n.constructor(visit(n.node), n.convertTo));
    else if (type === 'TextureNode') {
      if (n.value !== DFG_TEXTURE || !n.sampler || !n.uvNode || n.biasNode || n.compareNode || n.gradNode || n.offsetNode || n.depthNode || n.gatherNode) throw Error('Unexpected stock texture operation');
      const clone = n.clone(); clone.uvNode = visit(n.uvNode);
      clone.levelNode = float(0); // The exact table has one mip; compute uses explicit LOD 0.
      result = nodeObject(clone); textures.push(result);
    } else throw Error(`Unsupported/unbound stock shading node ${type}`);
    active.delete(n); seen.set(n, result); return result;
  }
  return {forward: visit(expression), textureNodes: textures, expandedFunctionCount: expandedFunctions.size, visitedNodes: visited};
}

function dfgReplacement(sample) {
  const uv = sample.uvNode, scaled = uv.mul(16).sub(.5), lower = scaled.floor();
  const fraction = scaled.sub(lower), upper = lower.add(1);
  const cell = [lower, vec2(upper.x, lower.y), vec2(lower.x, upper.y), upper];
  const loads = cell.map(p => {
    const raw = textureLoad(DFG_TEXTURE, ivec2(p.clamp(0, 15)), 0);
    // Truthful opaque read type: RG16F textureLoad returns vec4(r,g,0,1).
    raw.nodeType = 'vec4';
    return raw.rg;
  });
  const a = loads[0].mul(fraction.x.oneMinus()).add(loads[1].mul(fraction.x));
  const b = loads[2].mul(fraction.x.oneMinus()).add(loads[3].mul(fraction.x));
  const value = a.mul(fraction.y.oneMinus()).add(b.mul(fraction.y));
  return {replacement: vec4(value, 0, 1), constants: loads, uv, cell};
}

/**
 * Explicit input bridge. Directions are normalized here. Bind parameter/input
 * nodes as opaque leaves while expanding source, then differentiate `forward`
 * using {constants: [...inputs, ...result.constants], replacements}.
 * The four selected DFG texels per lookup are STOP-GRADIENT cell values: they
 * may depend discretely on UV, but are constant on each open bilinear cell.
 */
export function createMetallicShading({normal, view, light, roughness, geometryRoughness = float(0), f0 = vec3(...METAL_F0), inputs = []}) {
  if (![normal, view, light, roughness].every(n => n?.isNode)) throw Error('Expected normal/view/light/roughness TSL inputs');
  const N = normal.normalize(), V = view.normalize(), L = light.normalize();
  const r = roughness.max(.0525).add(geometryRoughness).min(1);
  const expression = BRDF_GGX_Multiscatter({lightDirection: L, f0, f90: float(1), roughness: r});
  const expanded = expandStockMetallic(expression, new Map([[normalView, N], [positionViewDirection, V]]),
    [normal, view, light, roughness, geometryRoughness, f0, ...inputs]);
  const replacements = new Map(), constants = [], lutLookups = [];
  for (const sample of expanded.textureNodes) {
    const replacement = dfgReplacement(sample);
    replacements.set(sample, replacement.replacement); constants.push(...replacement.constants);
    lutLookups.push({sample, ...replacement});
  }
  if (lutLookups.length !== 2) throw Error('Expected exactly two stock DFG lookups');
  const color = expanded.forward.mul(N.dot(L).clamp());
  return {...expanded, forward: color, color, replacements, constants, lutLookups, texture: DFG_TEXTURE, normal: N, effectiveRoughness: r};
}

// Independent double-precision oracle; transcription deliberately separate from
// the graph expander. Equations attributed to the exact files in SOURCE_FILES.
const dot3 = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
export function normalizeNumeric(v) {
  if (!Array.isArray(v) && !ArrayBuffer.isView(v)) throw Error('Expected numeric vec3');
  if (v.length !== 3 || !Array.from(v).every(Number.isFinite)) throw Error('Expected finite vec3');
  const length = Math.hypot(...v); if (!(length > 0)) throw Error('Cannot normalize zero vector');
  return Array.from(v, x => x / length);
}
export function numericBRDF(normal, roughness, light, view, f0 = METAL_F0, geometryRoughness = 0) {
  if (![roughness, geometryRoughness, ...f0].every(Number.isFinite)) throw Error('Nonfinite material');
  const n = normalizeNumeric(normal), l = normalizeNumeric(light), v = normalizeNumeric(view);
  const h = normalizeNumeric(l.map((x, i) => x + v[i]));
  const nl = clampNumber(dot3(n, l), 0, 1), nv = clampNumber(dot3(n, v), 0, 1);
  const nh = clampNumber(dot3(n, h), 0, 1), vh = clampNumber(dot3(v, h), 0, 1);
  const r = Math.min(Math.max(roughness, .0525) + geometryRoughness, 1), a2 = r ** 4;
  const denom = 1 - nh * nh * (1 - a2), D = a2 / (denom * denom) / Math.PI;
  const gv = nl * Math.sqrt(a2 + (1-a2)*nv*nv), gl = nv * Math.sqrt(a2 + (1-a2)*nl*nl);
  const visibility = .5 / Math.max(gv + gl, 1e-6);
  const fresnel = 2 ** ((-5.55473*vh-6.98316)*vh);
  const dv = numericDFG(r, nv), dl = numericDFG(r, nl);
  const ev = 1-dv[0]-dv[1], el = 1-dl[0]-dl[1];
  return Array.from(f0, color => {
    const F = color*(1-fresnel) + fresnel, avg = color + (1-color)*.047619;
    const ms = (color*dv[0]+dv[1])*(color*dl[0]+dl[1])*avg / (1-ev*el*avg*avg+1e-6);
    return nl*(F*visibility*D + ms*ev*el);
  });
}
