import test from 'node:test';
import assert from 'node:assert/strict';
import {compareRenderMasks} from './render-parity.js';

function rectangle() {
  const mask=new Uint8Array(256*192);
  for(let y=30;y<80;y++)for(let x=40;x<90;x++)mask[y*256+x]=1;
  return mask;
}
test('identical nonempty masks pass exactly',()=>{
  const mask=rectangle(),result=compareRenderMasks(mask,mask);
  assert.equal(result.passed,true);assert.equal(result.differingPixels,0);assert.equal(result.nativeForeground,2500);
});
test('a single boundary discrepancy passes the declared precision allowance',()=>{
  const mask=rectangle(),other=mask.slice();other[30*256+40]=0;
  const result=compareRenderMasks(mask,other);
  assert.equal(result.passed,true);assert.equal(result.differingPixels,1);assert.equal(result.interiorMismatches,0);
});
test('a single interior hole fails even below the pixel-count allowance',()=>{
  const mask=rectangle(),other=mask.slice();other[50*256+50]=0;
  const result=compareRenderMasks(mask,other);
  assert.equal(result.passed,false);assert.equal(result.differingPixels,1);assert.equal(result.interiorMismatches,1);
});
test('matching empty masks cannot pass',()=>{
  const mask=new Uint8Array(256*192);assert.equal(compareRenderMasks(mask,mask).passed,false);
});
