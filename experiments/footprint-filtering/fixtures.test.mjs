import test from 'node:test';
import assert from 'node:assert/strict';
import {Vector2} from 'three';
import {uniform} from 'three/tsl';
import {evaluateGradients} from '../trainable-tsl/autograd.js';
import {KINDS,CELLS,CPU_PROTOCOL,makeFixture,numericFixture,analyticTerms,integrateBox,quadratureBox,sinc,manualFilteredFixture} from './fixtures.mjs';

const close=(a,b,tolerance=CPU_PROTOCOL.absoluteTolerance)=>a.forEach((value,i)=>assert.ok(Math.abs(value-b[i])<=tolerance,`${value} != ${b[i]}`));

test('manual Fourier identities match independent original products and native forward graphs',()=>{
  const uv=uniform(new Vector2()),time=uniform(0),frequency=uniform(1);
  let seed=9351;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  for(const kind of KINDS){
    const graph=makeFixture(kind,{uv,time,frequency});
    for(let i=0;i<24;i++){
      const u=2*random()-1,v=2*random()-1,t=random(),f=32*random(),expected=numericFixture(kind,u,v,t,{frequency:f});
      const values=new Map([[uv,[u,v]],[time,t],[frequency,f]]);
      const actual=[graph.x,graph.y,graph.z].map(node=>evaluateGradients(node,[],values,{constants:[uv,time,frequency]}).value);
      close(actual,expected,2e-12);close(integrateBox(kind,{u,v,t,frequency:f,du:[0,0],dv:[0,0],dt:0}),expected,2e-12);
      assert.ok(expected.every(x=>x>=.1-1e-12&&x<=.9+1e-12));
    }
  }
});

test('manual exact box integral agrees with independently converged direct-product quadrature',()=>{
  for(const kind of KINDS)for(const cell of CELLS){
    const coarse=quadratureBox(kind,cell,{orders:CPU_PROTOCOL.quadratureOrders}),fine=quadratureBox(kind,cell,{orders:CPU_PROTOCOL.convergenceOrders});
    close(coarse.rgb,fine.rgb);close(integrateBox(kind,cell),fine.rgb);
    assert.ok(coarse.samples<=65536&&fine.samples<=65536);
  }
});

test('correlated weave preserves a beat when filtering one carrier independently gives zero',()=>{
  const box=CELLS.find(c=>c.id==='correlated-carrier-zero');
  const exact=integrateBox('weave',box);
  // A spans exactly one cycle along this pixel edge; E[sin(A)] is zero.
  // Incorrect independence would set .5+.4*E[sin(A)]*E[sin(B)] to .5.
  assert.ok(Math.abs(exact[0]-.5)>.1,`The adversarial beat is too weak: ${exact[0]}`);
  close(exact,quadratureBox('weave',box,{orders:[48,48,24]}).rgb);
});

test('squared-wave DC is retained and zero-frequency three-wave cancellation is explicit',()=>{
  const box={u:.123,v:.231,t:.137,frequency:10,du:[1/30,0],dv:[0,0],dt:0};
  close([integrateBox('energy',box)[0]],[.5],1e-12);
  // Averaging sin first and then squaring would incorrectly return .1.
  assert.ok(Math.abs(integrateBox('energy',box)[0]-.1)>.39);
  const terms=analyticTerms('interference',17);
  assert.ok(terms.every(channel=>channel.terms.some(term=>term.phase[0]===0&&term.phase[1]===0&&Math.abs(term.phase[2]-Math.PI)<1e-12)));
});

test('box orientation symmetry, sinc removable singularity, and bounded input validation',()=>{
  assert.equal(sinc(0),1);assert.equal(sinc(1e-12),1);
  for(const kind of KINDS){const box=CELLS.find(c=>c.id==='combined');close(integrateBox(kind,box),integrateBox(kind,{...box,du:box.dv,dv:box.du}));
    close(integrateBox(kind,box),integrateBox(kind,{...box,du:box.du.map(x=>-x),dv:box.dv.map(x=>-x)}));}
  assert.throws(()=>integrateBox('weave',{u:0,v:0,t:0,dt:-1}));assert.throws(()=>numericFixture('weave',0,0,0,{frequency:Infinity}));
  assert.throws(()=>quadratureBox('weave',CELLS[6],{orders:[64,64,64]}));
  const uv=uniform(new Vector2()),time=uniform(0),frequency=uniform(1);
  assert.ok(manualFilteredFixture('weave',{uv,time,frequency,du:[0,0],dv:[0,0],dt:0}).isNode);
});
