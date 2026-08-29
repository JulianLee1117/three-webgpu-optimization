import assert from 'node:assert/strict';
import test from 'node:test';
import { parseCsv, summarizeCsv } from '../analysis/summarize.mjs';

test('CSV parsing preserves quoted commas and escaped quotes', () => {
  const parsed = parseCsv('name,note\r\n"fixed,slice","two ""dispatches"""\r\n');
  assert.deepEqual(parsed.records, [{ name: 'fixed,slice', note: 'two "dispatches"' }]);
});

test('analysis pairs modes by repetitionIndex within a visibility cell', () => {
  const csv = [
    'modeId,targetVisibilityFraction,repetitionIndex,usesCompute,gpuPassTotalMs,gpuComputeMs,gpuRenderMs,cpuCommonUpdateMs,cpuFrameBodyMs,cpuSubmitTotalMs',
    'draw-all,0.2,0,false,10,,10,0.1,1.4,1',
    'three-blocks-current,0.2,0,true,8,2,6,0.8,3.2,2',
    'three-blocks-historical,0.2,0,true,7,1,6,0.6,2.8,1.8',
    'fixed-slice,0.2,0,true,4,1,3,0.3,2.1,1.5',
    'draw-all,0.2,1,false,12,,12,0.2,1.6,1.1',
    'three-blocks-current,0.2,1,true,10,2,8,1,3.6,2.1',
    'three-blocks-historical,0.2,1,true,9,1,8,0.8,3.2,1.9',
    'fixed-slice,0.2,1,true,6,1,5,0.5,2.3,1.6',
  ].join('\n');

  const summary = summarizeCsv(csv);
  assert.equal(summary.repetitionColumn, 'repetitionIndex');
  const fixed = summary.groups.find((group) => group.modeId === 'fixed-slice');
  const drawAll = summary.groups.find((group) => group.modeId === 'draw-all');
  assert.equal(fixed.nTrials, 2);
  assert.equal(drawAll.medianAcrossTrials.gpuPassTotalMs, 11);
  assert.equal(fixed.medianAcrossTrials.cpuCommonUpdateMs, 0.4);
  assert.equal(fixed.medianAcrossTrials.cpuFrameBodyMs, 2.2);
  assert.equal(fixed.medianAcrossTrials.cpuSubmitTotalMs, 1.55);
  const versusDrawAll = summary.comparisons.fixedSliceVsDrawAll[0];
  const versusThreeBlocks = summary.comparisons.fixedSliceVsThreeBlocksCurrent[0];
  const versusHistorical = summary.comparisons.fixedSliceVsThreeBlocksHistorical[0];
  assert.equal(versusDrawAll.nPairs, 2);
  assert.equal(versusThreeBlocks.nPairs, 2);
  assert.equal(versusHistorical.nPairs, 2);
  assert.equal(versusDrawAll.medianPairedDelta.gpuPassTotalMs.absoluteMs, -6);
  assert.equal(versusThreeBlocks.medianPairedDelta.gpuPassTotalMs.absoluteMs, -4);
  assert.equal(versusHistorical.medianPairedDelta.gpuPassTotalMs.absoluteMs, -3);
  assert.ok(Math.abs(versusDrawAll.medianPairedDelta.cpuCommonUpdateMs.absoluteMs - 0.25) < 1e-12);
  assert.ok(Math.abs(versusThreeBlocks.medianPairedDelta.cpuFrameBodyMs.absoluteMs + 1.2) < 1e-12);
  assert.ok(Math.abs(versusHistorical.medianPairedDelta.cpuSubmitTotalMs.absoluteMs + 0.3) < 1e-12);
});

test('analysis rejects missing GPU compute data for compute modes', () => {
  const csv = [
    'modeId,targetVisibilityFraction,repetitionIndex,usesCompute,gpuPassTotalMs,gpuComputeMs,gpuRenderMs,cpuCommonUpdateMs,cpuFrameBodyMs,cpuSubmitTotalMs',
    'fixed-slice,0.2,0,true,4,,3,0.3,2.1,1.5',
  ].join('\n');
  assert.throws(() => summarizeCsv(csv), /compute mode but has no gpuComputeMs/);
});

test('analysis requires full-frame and common-update CPU timing without redefining submit timing', () => {
  const csv = [
    'modeId,targetVisibilityFraction,usesCompute,gpuPassTotalMs,gpuComputeMs,gpuRenderMs,cpuSubmitTotalMs',
    'draw-all,0.2,false,10,,10,1',
  ].join('\n');
  assert.throws(
    () => summarizeCsv(csv),
    /CSV is missing required columns: cpuCommonUpdateMs, cpuFrameBodyMs/,
  );
});
