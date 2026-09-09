import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {DataUtils, Vector3, DataTexture} from 'three';
import {float, vec3, uniform, Fn, texture} from 'three/tsl';
import {normalView} from '../../node_modules/three/src/nodes/accessors/Normal.js';
import {positionViewDirection} from '../../node_modules/three/src/nodes/accessors/Position.js';
import {createMetallicShading, expandStockMetallic, normalFromSlopes, numericBRDF,
  numericDFG, dfgCell, DFG_DATA, DFG_TEXTURE, decodeFloat16, SHADING_SOURCE_FILES} from './shading.js';

const typename = n => n.constructor.type ?? n.constructor.name;
const array = v => Array.isArray(v) ? v : [v];
function map2(a, b, fn) {
  a = array(a); b = array(b); const width = Math.max(a.length, b.length);
  const result = Array.from({length: width}, (_, i) => fn(a[a.length === 1 ? 0 : i], b[b.length === 1 ? 0 : i]));
  return width === 1 ? result[0] : result;
}
const unary = (a, fn) => Array.isArray(a) ? a.map(fn) : fn(a);
// Test-only interpreter of the EXPANDED actual upstream forward graph. It does
// not call numericBRDF or use its equations. Texture sampling is independently
// evaluated from the original half-float texture bits.
function evaluate(node, bindings = new Map(), replacements = new Map(), cache = new Map()) {
  node = node.self ?? node;
  if (bindings.has(node)) return bindings.get(node);
  if (replacements.has(node)) return evaluate(replacements.get(node), bindings, replacements, cache);
  if (cache.has(node)) return cache.get(node);
  const ev = child => evaluate(child, bindings, replacements, cache);
  let value; const kind = typename(node);
  if (kind === 'VarNode') value = ev(node.node);
  else if (kind === 'UniformNode' || kind === 'ConstNode') {
    value = typeof node.value === 'number' ? node.value : node.value.toArray();
  } else if (kind === 'OperatorNode') {
    const f = {'+': (a,b) => a+b, '-': (a,b) => a-b, '*': (a,b) => a*b, '/': (a,b) => a/b}[node.op];
    if (!f) throw Error('Unexpected test operator ' + node.op);
    value = map2(ev(node.aNode), ev(node.bNode), f);
  } else if (kind === 'MathNode') {
    const a = ev(node.aNode), b = node.bNode ? ev(node.bNode) : undefined;
    if (node.method === 'dot') value = array(a).reduce((sum,x,i) => sum + x*array(b)[i], 0);
    else if (node.method === 'normalize') {const v = array(a), len = Math.hypot(...v); value = v.map(x => x/len);}
    else if (node.method === 'clamp') value = map2(map2(a,b,Math.max), ev(node.cNode), Math.min);
    else if (node.method === 'min' || node.method === 'max') {
      value = map2(a,b,Math[node.method]);
      if (!Number.isFinite(array(value)[0])) throw Error(`Invalid ${node.method} operands ${JSON.stringify(a)}, ${JSON.stringify(b)}`);
    }
    else {
      const f = {oneMinus: x => 1-x, negate: x => -x, exp2: x => 2**x, sqrt: Math.sqrt, floor: Math.floor, fract: x => x-Math.floor(x)}[node.method];
      if (!f) throw Error('Unexpected test method ' + node.method); value = unary(a,f);
    }
  } else if (kind === 'JoinNode') value = node.nodes.flatMap(n => array(ev(n)));
  else if (kind === 'SplitNode') {
    const v = array(ev(node.node)); value = [...node.components].map(x => v['xyzw'.indexOf(x)]);
    if (value.length === 1) value = value[0];
  } else if (kind === 'ConvertNode') value = /i(?:nt|vec)/.test(node.convertTo) ? unary(ev(node.node), Math.trunc) : ev(node.node);
  else if (kind === 'TextureNode') {
    const uv = ev(node.uvNode), image = node.value.image;
    const at = (x,y) => {
      const offset = 2*(Math.max(0,Math.min(15,y))*16 + Math.max(0,Math.min(15,x)));
      return [DataUtils.fromHalfFloat(image.data[offset]), DataUtils.fromHalfFloat(image.data[offset+1]), 0, 1];
    };
    if (!node.sampler) value = at(uv[0],uv[1]);
    else {
      const x = uv[0]*16-.5, y = uv[1]*16-.5, ix = Math.floor(x), iy = Math.floor(y), tx = x-ix, ty = y-iy;
      const aa=at(ix,iy),bb=at(ix+1,iy),cc=at(ix,iy+1),dd=at(ix+1,iy+1);
      value = aa.map((_,i) => aa[i]*(1-tx)*(1-ty)+bb[i]*tx*(1-ty)+cc[i]*(1-tx)*ty+dd[i]*tx*ty);
    }
  } else throw Error('Unexpected test node ' + kind);
  if (!array(value).every(Number.isFinite)) throw Error(`Nonfinite test evaluation ${kind}/${node.method ?? node.components ?? node.op ?? ''}: ${JSON.stringify(value)}`);
  cache.set(node, value); return value;
}
function fixture() {
  const n = uniform(new Vector3(0,0,1)), v = uniform(new Vector3(0,0,1)), l = uniform(new Vector3(.2,.3,1)), r = uniform(.4);
  return {n,v,l,r,bridge:createMetallicShading({normal:n,view:v,light:l,roughness:r})};
}
const maxDiff = (a,b) => Math.max(...array(a).map((x,i) => Math.abs(x-array(b)[i])));

test('pinned Three dependency and exact upstream half-float table', () => {
  const pkg = JSON.parse(readFileSync(new URL('../../node_modules/three/package.json', import.meta.url)));
  assert.equal(pkg.version, '0.185.1');
  assert.equal(DFG_DATA.length,512);
  for(let i=0;i<512;i++) assert.equal(DFG_DATA[i],DataUtils.fromHalfFloat(DFG_TEXTURE.image.data[i]));
  assert.equal(decodeFloat16(0x3c00),1); assert.equal(decodeFloat16(0xc000),-2);
  assert.equal(decodeFloat16(1),2**-24); assert.equal(decodeFloat16(0x7c00),Infinity);
  assert.ok(Number.isNaN(decodeFloat16(0x7c01)));
  for(const name of SHADING_SOURCE_FILES) assert.equal(createHash('sha256').update(readFileSync(new URL('../../'+name,import.meta.url))).digest('hex').length,64);
});

test('DFG bilinear centers and clamp-to-edge exactly address table entries', () => {
  for(let y=0;y<16;y++)for(let x=0;x<16;x++)assert.deepEqual(numericDFG((x+.5)/16,(y+.5)/16),DFG_DATA.slice(2*(16*y+x),2*(16*y+x)+2));
  assert.deepEqual(numericDFG(-1,-1),DFG_DATA.slice(0,2));
  assert.deepEqual(numericDFG(2,2),DFG_DATA.slice(-2));
  assert.deepEqual(dfgCell(.5,.5).fraction,[.5,.5]);
});

test('expanded native upstream graph agrees with independent metallic oracle', () => {
  const {n,v,l,r,bridge} = fixture();
  let state = 731; const random = () => ((state=(Math.imul(state,1664525)+1013904223)>>>0)/2**32);
  let max = 0;
  for(let i=0;i<256;i++) {
    const N=[(random()-.5)*.7,(random()-.5)*.7,1],V=[(random()-.5)*1.4,(random()-.5)*1.4,1],L=[(random()-.5)*1.8,(random()-.5)*1.8,1];
    const R=i<4?[0,.0525,.5,1][i]:.08+.9*random();
    const expected=numericBRDF(N,R,L,V), actual=evaluate(bridge.forward,new Map([[n.self??n,N],[v.self??v,V],[l.self??l,L],[r.self??r,R]]));
    max=Math.max(max,maxDiff(expected,actual));
    assert.ok(maxDiff(expected,actual)<=1e-9+1e-10*Math.max(...expected),`sample ${i}: ${maxDiff(expected,actual)}`);
  }
  assert.equal(bridge.expandedFunctionCount,6); assert.equal(bridge.lutLookups.length,2);
  assert.ok(max<1e-8);
});

test('DFG replacement preserves full primal graph, including edge clamps', () => {
  const {n,v,l,r,bridge}=fixture();
  const replacements=new Map([...bridge.replacements].map(([k,v])=>[k.self??k,v]));
  for(const R of [0,.0525,.0625,.193,.5,.96875,1,1.2])for(const N of [[0,0,1],[.35,-.2,1],[1.2,.4,1]]) {
    const values=new Map([[n.self??n,N],[v.self??v,[.2,-.1,1]],[l.self??l,[-.4,.35,1]],[r.self??r,R]]);
    assert.ok(maxDiff(evaluate(bridge.forward,values),evaluate(bridge.forward,values,replacements))<1e-11);
  }
  assert.equal(bridge.constants.length,8);
  for(const sample of bridge.textureNodes) {assert.equal(sample.value,DFG_TEXTURE);assert.equal(sample.sampler,true);}
});

test('slope graph and explicit geometry roughness preserve upstream clamp behavior', () => {
  const sx=uniform(.2),sy=uniform(-.1),r=uniform(.2),n=normalFromSlopes(sx,sy);
  const bridge=createMetallicShading({normal:n,view:vec3(0,0,1),light:vec3(.1,.2,1),roughness:r,geometryRoughness:float(.13),inputs:[sx,sy]});
  assert.ok(maxDiff(evaluate(bridge.forward),numericBRDF([.2,-.1,1],.2,[.1,.2,1],[0,0,1],undefined,.13))<1e-10);
  assert.deepEqual(numericBRDF([0,0,1],0,[.2,.1,1],[0,0,1]),numericBRDF([0,0,1],.0525,[.2,.1,1],[0,0,1]));
});

test('unknown functions, explicit variables and foreign textures fail closed', () => {
  assert.throws(()=>expandStockMetallic(Fn(()=>float(1))(),new Map()),/Unapproved/);
  assert.throws(()=>expandStockMetallic(float(1).toVar('mutable'),new Map()),/explicit/);
  const foreign=new DataTexture(new Uint8Array(4),1,1);
  assert.throws(()=>expandStockMetallic(texture(foreign),new Map()),/Unexpected stock texture/);
  foreign.dispose();
});

test('source globals and original DFG sampler are unchanged by repeated construction', () => {
  const normalNode=normalView.node,viewNode=positionViewDirection.node;
  fixture();fixture();
  assert.equal(normalView.node,normalNode);assert.equal(positionViewDirection.node,viewNode);
  assert.equal(DFG_TEXTURE.image.data.length,512);assert.equal(DFG_TEXTURE.generateMipmaps,false);
});
