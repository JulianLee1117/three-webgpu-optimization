import assert from 'node:assert/strict';
import { execFile as execFileCallback } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const execFile = promisify(execFileCallback);
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const candidateRoot = path.join(
  repoRoot,
  '.generated',
  'three-primitive-index-current-dev-candidate',
);
const patchPath = path.join(repoRoot, 'patches', 'three-webgpu-primitive-index.patch');

const baseCommit = 'c4ffe022f2a4f982b42b7da5af79a87066a138ae';
const candidateCommit = 'c7880322c2be17269210c305636b929f43d671f1';
const candidateTree = 'd3dc72787fb4247224dd3ef6bd1623c18cf96bd3';
const patchSha256 = '3b4ff0ec2897c4d6cbd0bc2dfd7a94c4298134f03ccac33240f9763e41e25d15';

const expectedPaths = [
  'src/Three.TSL.js',
  'src/nodes/core/IndexNode.js',
  'src/nodes/core/NodeBuilder.js',
  'src/renderers/webgl-fallback/nodes/GLSLNodeBuilder.js',
  'src/renderers/webgpu/nodes/WGSLNodeBuilder.js',
  'src/renderers/webgpu/utils/WebGPUConstants.js',
  'test/unit/src/renderers/webgpu/WebGPUPrimitiveIndex.tests.js',
  'test/unit/three.source.unit.js',
  'tsl/content/Guide.md',
];

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

async function git(args, encoding = null) {
  const { stdout } = await execFile('git', ['-C', candidateRoot, ...args], {
    encoding,
    maxBuffer: 64 * 1024 * 1024,
    windowsHide: true,
  });
  return stdout;
}

test('tracked primitiveIndex patch is the exact clean pinned candidate diff', async () => {
  const [head, tree, status, regenerated, tracked] = await Promise.all([
    git(['rev-parse', 'HEAD'], 'utf8'),
    git(['rev-parse', 'HEAD^{tree}'], 'utf8'),
    git(['status', '--porcelain=v1', '--untracked-files=all'], 'utf8'),
    git([
      'diff',
      '--binary',
      '--full-index',
      '--no-ext-diff',
      `${baseCommit}..${candidateCommit}`,
    ]),
    readFile(patchPath),
  ]);

  assert.equal(head.trim(), candidateCommit, 'candidate commit changed');
  assert.equal(tree.trim(), candidateTree, 'candidate tree changed');
  assert.equal(status, '', 'candidate worktree is not clean');
  assert.equal(tracked.compare(regenerated), 0, 'tracked patch differs byte-for-byte');
  assert.equal(tracked.length, 14_708, 'tracked patch byte length changed');
  assert.equal(sha256(tracked), patchSha256, 'tracked patch SHA-256 changed');

  const paths = [...tracked.toString('utf8').matchAll(
    /^diff --git a\/(.+) b\/(.+)$/gmu,
  )].map((match) => {
    assert.equal(match[1], match[2], 'patch renames an unexpected file');
    return match[1];
  });
  assert.deepEqual(paths, expectedPaths, 'patch file scope changed');
});

test('tracked primitiveIndex patch contains its public API and safety evidence', async () => {
  const source = await readFile(patchPath, 'utf8');
  const requiredFragments = [
    '+export const primitiveIndex = TSL.primitiveIndex;',
    "+\tPrimitiveIndex: 'primitive-index'",
    "+\t\t\treturn this.renderer.hasFeature( 'primitive-index' );",
    "+\t\tthis.enableDirective( 'primitive_index' );",
    "+\t\treturn this.getBuiltin( 'primitive_index', 'primitiveIndex', 'u32' );",
    "'enable primitive_index;'",
    "'@builtin( primitive_index ) primitiveIndex : u32'",
    'primitiveIndex is only available in fragment shaders.',
    'primitiveIndex requires the WebGPU "primitive-index" feature.',
    'primitiveIndex is not supported by the WebGL backend.',
    'diff --git a/test/unit/src/renderers/webgpu/WebGPUPrimitiveIndex.tests.js',
    "QUnit.test( 'device negotiation'",
    "QUnit.test( 'conditional WGSL generation'",
    "QUnit.test( 'feature and stage validation'",
    "+import './src/renderers/webgpu/WebGPUPrimitiveIndex.tests.js';",
    'diff --git a/tsl/content/Guide.md',
    '+::: api primitiveIndex : `uint` - The per-instance primitive index in a fragment shader. Requires the WebGPU `primitive-index` feature. :::',
  ];

  for (const fragment of requiredFragments) {
    assert.equal(source.includes(fragment), true, `patch is missing: ${fragment}`);
  }
});
