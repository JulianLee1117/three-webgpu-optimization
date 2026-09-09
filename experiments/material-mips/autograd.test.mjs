import test from 'node:test';
import assert from 'node:assert/strict';
import {Matrix3, Vector2, Vector3, Vector4} from 'three';
import {StorageBufferAttribute} from 'three/webgpu';
import {uniform, attribute, storage, instanceIndex, float, vec2, vec3, vec4, sin, cos, exp, tanh, dot, abs, mix, Fn} from 'three/tsl';
import {gradients, evaluateGradients, AUTOGRAD_LIMITS} from './autograd.js';
import D_GGX from 'three/src/nodes/functions/BSDF/D_GGX.js';
import F_Schlick from 'three/src/nodes/functions/BSDF/F_Schlick.js';
import V_GGX_SmithCorrelated from 'three/src/nodes/functions/BSDF/V_GGX_SmithCorrelated.js';
import {createMetallicShading, normalFromSlopes, dfgCell, numericBRDF} from './shading.js';

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
  assert.throws(() => gradients(x.log(), [x]), /Unsupported MathNode/);
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

// Independent evaluator of generated native expressions, including their
// explicit conditional nodes. This does not run the AD traversal or its rules.
function nativeValue(root, values) {
  const canonical = input => {
    let n = input;
    while (n.self?.isNode && n.self !== n || n.constructor.type === 'VarNode') n = n.self?.isNode && n.self !== n ? n.self : n.node;
    return n;
  };
  const inputs = new Map([...values].map(([n, x]) => [canonical(n), Array.isArray(x) ? x : [x]])), cache = new Map();
  const broadcast = (arrays, fn) => Array.from({length: Math.max(...arrays.map(x => x.length))}, (_, i) => fn(...arrays.map(x => x[x.length === 1 ? 0 : i])));
  function visit(input) {
    const n = canonical(input);
    if (inputs.has(n)) return inputs.get(n);
    if (cache.has(n)) return cache.get(n);
    const type = n.constructor.type;
    let out;
    if (type === 'ConstNode') out = typeof n.value === 'number' || typeof n.value === 'boolean' ? [n.value] : n.value.toArray();
    else if (type === 'OperatorNode') {
      const operations = {'+': (a,b)=>a+b, '-':(a,b)=>a-b, '*':(a,b)=>a*b, '/':(a,b)=>a/b,
        '>':(a,b)=>a>b, '<':(a,b)=>a<b, '>=':(a,b)=>a>=b, '<=':(a,b)=>a<=b, '==':(a,b)=>a===b};
      assert.ok(operations[n.op], `independent unsupported operator ${n.op}`);
      out = broadcast([visit(n.aNode), visit(n.bNode)], operations[n.op]);
    } else if (type === 'ConditionalNode') out = broadcast([visit(n.condNode), visit(n.ifNode), visit(n.elseNode)], (c,a,b)=>c?a:b);
    else if (type === 'MathNode') {
      const operations = {sin:Math.sin, cos:Math.cos, exp:Math.exp, exp2:x=>2**x, tanh:Math.tanh, sqrt:Math.sqrt,
        abs:Math.abs, floor:Math.floor, fract:x=>x-Math.floor(x), negate:x=>-x, oneMinus:x=>1-x,
        pow:Math.pow, min:Math.min, max:Math.max, clamp:(x,l,h)=>Math.min(Math.max(x,l),h), mix:(a,b,t)=>a*(1-t)+b*t};
      const a = visit(n.aNode);
      if(n.method==='dot')out=[a.reduce((s,x,i)=>s+x*visit(n.bNode)[i],0)];
      else if(n.method==='length')out=[Math.hypot(...a)];
      else if(n.method==='normalize'){const length=Math.hypot(...a);out=a.map(x=>x/length);}
      else {assert.ok(operations[n.method], `independent unsupported math ${n.method}`);out=broadcast([a,...[n.bNode,n.cNode].filter(Boolean).map(visit)],operations[n.method]);}
    } else if(type==='JoinNode')out=n.nodes.flatMap(visit);
    else if(type==='SplitNode'){const a=visit(n.node);out=[...n.components].map(axis=>a[a.length===1?0:'xyzw'.indexOf(axis)]);}
    else if(type==='ConvertNode'){
      const width={float:1,vec2:2,vec3:3,vec4:4}[n.convertTo],a=visit(n.node);
      assert.ok(width);out=a.length===1?Array(width).fill(a[0]):a.slice(0,width);
      while(out.length<width)out.push(out.length===3?1:0);
    } else throw Error(`independent unsupported node ${type}`);
    cache.set(n,out);return out;
  }
  const result=visit(root);return result.length===1?result[0]:result;
}

test('material math numeric and actual native gradients match independent finite differences', () => {
  const q=uniform(new Vector3()),t=uniform(0),coeff=vec3(.3,-.5,.9);
  const loss=mix(q.normalize(),q.zxy,t).dot(coeff).add(q.length().mul(.2))
    .add(t.add(1).sqrt()).add(t.exp2()).add(t.add(2).pow(1.7))
    .add(q.y.abs()).add(q.x.oneMinus()).add(q.x.min(q.z)).add(q.y.max(q.z))
    .add(t.clamp(.1,.8)).add(q.x.floor()).add(q.y.fract());
  const symbolic=gradients(loss,[q,t]);
  const objective=([x,y,z,t])=>{
    const norm=Math.hypot(x,y,z),a=[x/norm,y/norm,z/norm],b=[z,x,y];
    return [.3,-.5,.9].reduce((s,c,i)=>s+c*(a[i]*(1-t)+b[i]*t),0)
      +norm*.2+Math.sqrt(t+1)+2**t+(t+2)**1.7+Math.abs(y)+1-x+Math.min(x,z)+Math.max(y,z)
      +Math.min(Math.max(t,.1),.8)+Math.floor(x)+y-Math.floor(y);
  };
  let checked=0;
  for(let trial=0;trial<24;trial++){
    const point=[.31+trial*.013,-.77+trial*.009,1.2-trial*.008,[.03,.3,.6,1.05][trial%4]];
    const values=new Map([[q,point.slice(0,3)],[t,point[3]]]);
    const result=evaluateGradients(loss,[q,t],values);
    near(result.value,objective(point));
    const emitted=[...nativeValue(symbolic.gradients[0],values),nativeValue(symbolic.gradients[1],values)];
    const numeric=[...result.gradients[0],result.gradients[1]];
    for(let i=0;i<4;i++){const lo=[...point],hi=[...point];lo[i]-=1e-6;hi[i]+=1e-6;
      const expected=(objective(hi)-objective(lo))/2e-6;near(numeric[i],expected,3e-6);near(emitted[i],expected,3e-6);checked++;}
  }
  assert.equal(checked,96);
});

test('piecewise boundary conventions are explicit, deterministic, and emitted correctly', () => {
  const x=uniform(0),low=uniform(0),high=uniform(0);
  for(const point of [[0,0,1],[1,0,1],[-1,0,1],[2,0,1],[.5,0,1]]){
    const values=new Map([[x,point[0]],[low,point[1]],[high,point[2]]]),loss=x.clamp(low,high);
    const expected=point[0]===0||point[0]===1?[0,0,0]:point[0]<0?[0,1,0]:point[0]>1?[0,0,1]:[1,0,0];
    arrayNear(evaluateGradients(loss,[x,low,high],values).gradients,expected);
    arrayNear(gradients(loss,[x,low,high]).gradients.map(n=>nativeValue(n,values)),expected);
  }
  for(const method of ['min','max']){
    const values=new Map([[x,2],[low,2]]),loss=x[method](low);
    arrayNear(evaluateGradients(loss,[x,low],values).gradients,[1,0]);
    arrayNear(gradients(loss,[x,low]).gradients.map(n=>nativeValue(n,values)),[1,0]);
  }
  near(nativeValue(gradients(x.abs(),[x]).gradients[0],new Map([[x,0]])),0);
  assert.throws(()=>evaluateGradients(x.clamp(low,high),[x,low,high],new Map([[x,.5],[low,2],[high,1]])),/ordered/);
  assert.throws(()=>evaluateGradients(x.sqrt(),[x],new Map([[x,0]])),/Nonfinite/);
  assert.throws(()=>evaluateGradients(x.pow(.5),[x],new Map([[x,-1]])),/positive/);
  assert.throws(()=>gradients(x.pow(low),[x],{constants:[low]}),/literal scalar/);
  assert.throws(()=>gradients(x.greaterThan(0).select(x,x.negate()),[x]),/Unsupported/);
});

test('actual allowlisted stock helpers expand and match independent GGX equations', () => {
  const alpha=uniform(0),n=uniform(0),v=uniform(0),f0=uniform(0),f90=uniform(0);
  const d=D_GGX({alpha,dotNH:n}),f=F_Schlick({f0,f90,dotVH:v}),vis=V_GGX_SmithCorrelated({alpha,dotNL:n,dotNV:v});
  const loss=d.mul(f).mul(vis),variables=[alpha,n,v,f0,f90],symbolic=gradients(loss,variables);
  assert.deepEqual([...symbolic.stats.expandedStockFunctions].sort(),['D_GGX','F_Schlick','V_GGX_SmithCorrelated'].sort());
  const objective=([a,n,v,f0,f90])=>{
    const a2=a*a,denom=1-n*n*(1-a2),D=a2/(Math.PI*denom*denom),Fresnel=2**((-5.55473*v-6.98316)*v);
    const F=f0*(1-Fresnel)+f90*Fresnel,V=.5/Math.max(1e-6,n*Math.sqrt(a2+(1-a2)*v*v)+v*Math.sqrt(a2+(1-a2)*n*n));
    return D*F*V;
  };
  for(const point of [[.2,.6,.8,.04,1],[.6,.95,.4,.7,1],[.8,.3,.2,.2,.9]]){
    const values=new Map(variables.map((n,i)=>[n,point[i]])),answer=evaluateGradients(loss,variables,values);
    near(answer.value,objective(point));
    for(let i=0;i<point.length;i++){const lo=[...point],hi=[...point];lo[i]-=1e-6;hi[i]+=1e-6;
      const expected=(objective(hi)-objective(lo))/2e-6;near(answer.gradients[i],expected,3e-6);near(nativeValue(symbolic.gradients[i],values),expected,3e-6);}
  }
  assert.throws(()=>gradients(Fn(()=>alpha.mul(alpha))(),[alpha]),/Unsupported/);
  assert.throws(()=>gradients(D_GGX({alpha,dotNH:n,unused:alpha.toVar()}),variables),/VarNode/);
});

test('explicit substitutions differentiate the surrogate while retaining the original symbolic primal', () => {
  const x=uniform(0),opaque=Fn(()=>x.add(7))(),loss=opaque.mul(opaque);
  const replacements=new Map([[opaque,x.mul(x)]]),options={replacements},values=new Map([[x,3]]);
  const symbolic=gradients(loss,[x],options),answer=evaluateGradients(loss,[x],values,options);
  assert.equal(symbolic.value,loss);near(answer.value,81);near(answer.gradients[0],108);
  near(nativeValue(symbolic.gradients[0],values),108);
  assert.equal(answer.stats.replacementNodes,1);assert.equal(answer.stats.differentiatedModelSubstituted,true);
  assert.throws(()=>gradients(loss,[x]),/Unsupported/);
  assert.throws(()=>gradients(loss,[x],{replacements:new Map([[opaque,opaque]])}),/itself/);
  assert.throws(()=>gradients(loss,[x],{replacements:new Map([[opaque,opaque.add(1)]])}),/Cyclic/);
  assert.throws(()=>gradients(loss,[x],{replacements:new Map([[opaque,x.toVar()]])}),/VarNode/);
  assert.throws(()=>gradients(x,[x],{replacements:new Map([[x,x.add(1)]])}),/also be replaced/);
});

test('complete stock metallic bridge derivatives match independent resampled DFG finite differences', () => {
  const x=uniform(0),y=uniform(0),roughness=uniform(0),variables=[x,y,roughness];
  const view=[.1,.2,1],light=[.3,-.2,1];
  const shading=createMetallicShading({normal:normalFromSlopes(x,y),view:vec3(...view),light:vec3(...light),roughness,inputs:variables});
  const loss=dot(shading.forward,vec3(1)),options={constants:shading.constants,replacements:shading.replacements};
  const symbolic=gradients(loss,variables,options);
  const objective=p=>numericBRDF([p[0],p[1],1],p[2],light,view).reduce((a,b)=>a+b,0);
  for(const point of [[.13,-.17,.23],[-.21,.16,.4],[.07,.23,.72]]){
    const values=new Map(variables.map((v,i)=>[v,point[i]]));
    for(const lookup of shading.lutLookups){
      const uv=nativeValue(lookup.uv,values),cell=dfgCell(...uv);
      lookup.constants.forEach((node,i)=>values.set(node,cell.texels[i]));
    }
    const numeric=evaluateGradients(loss,variables,values,options);
    near(numeric.value,objective(point),1e-9);
    for(let i=0;i<3;i++){
      const lo=[...point],hi=[...point];lo[i]-=1e-6;hi[i]+=1e-6;
      // Recompute the actual interpolation coordinates and selected texels for
      // both perturbed samples; do not finite-difference a frozen surrogate.
      const expected=(objective(hi)-objective(lo))/2e-6;
      near(numeric.gradients[i],expected,1e-5);near(nativeValue(symbolic.gradients[i],values),expected,1e-5);
    }
  }
  assert.equal(symbolic.stats.replacementNodes,2);
  assert.equal(symbolic.stats.unusedReplacements,0);
});
