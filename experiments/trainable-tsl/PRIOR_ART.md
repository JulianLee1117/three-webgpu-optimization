# Trainable native TSL: prior art and claim boundary

Research checked on 2026-09-09. This is a bounded integration experiment, not
evidence of a new differentiation algorithm or an accepted performance result.
The prospective experiment is defined in [PROTOCOL.md](PROTOCOL.md).

The useful question is whether an author can write a supported, ordinary Three.js
TSL forward expression once, automatically obtain its parameter derivatives, fit
it to observations on WebGPU, and render/export the learned expression without
maintaining a separate backward program or translating the model into an ML
framework. AI coding tools could author such a forward expression, but the
experiment does not test AI code generation or autonomous world creation.

## Closest existing implementations

- **Slang / Slang.D:** Slang already generates forward and reverse derivatives,
  including custom derivative support. The 2023 Slang.D paper establishes shader
  autodiff and fused neural graphics workloads. Its browser playground compiles
  locally with WebAssembly and executes WGSL through WebGPU; this was documented
  on 2024-11-20. Browser shader autodiff is therefore established prior art, not a
  missing web capability. The practical distinction here is accepting native TSL
  node objects already used by a Three material, rather than authoring Slang.
  [Official autodiff guide](https://shader-slang.org/slang/user-guide/autodiff),
  [Slang.D paper](https://research.nvidia.com/labs/rtr/publication/bangaru2023slangd/bangaru2023slangd.pdf),
  [browser announcement](https://shader-slang.org/blog/2024/11/20/theres-a-lot-going-on-with-slang/),
  [playground source at 631c0d8, 2026-06-11](https://github.com/shader-slang/slang-playground/tree/631c0d838fef1e7992e2a71a91c71ef89cb84f10).
  An independent [SlangWebGPU autodiff example, 2024-11-29 revision](https://github.com/eliemichel/SlangWebGPU/tree/6071f1da03b6482ac21552c112ffed94ae8f8633/examples/05_autodiff)
  also exists. These implementations were inspected, not benchmarked locally.

- **jax-js:** a JavaScript ML framework compiling array operations to WebGPU and
  WebAssembly. Its feature list includes `grad`, `value_and_grad`, `jvp`, `vjp`
  and `jit`; browser training examples already exist. It is a credible alternative
  for training a small field and using its weights in Three. This experiment
  only investigates avoiding a separate array-program representation when the
  forward model is already a native TSL graph. No speed, memory, maturity or
  generality advantage over jax-js is established.
  [Repository, MIT, 970fa22, 2026-09-06](https://github.com/ekzhang/jax-js/tree/970fa22d934ce2e617cd3a993c6ecc8b736496f2),
  [pinned feature list](https://github.com/ekzhang/jax-js/blob/970fa22d934ce2e617cd3a993c6ecc8b736496f2/FEATURES.md).

- **glsl-autodiff:** Dave Pagurek's 2021 library already builds a shader operation
  graph through a JavaScript API and emits both the expression and requested
  derivatives. Its deformation examples automatically correct vertex normals.
  The proposed work is not the first JavaScript shader graph differentiator,
  derivative-based deformation system, or write-once shader workflow. Its
  narrower distinction is inspecting native TSL nodes and training their
  parameters through Three's WebGPU path.
  [Author's introduction, 2021-08-13](https://www.davepagurek.com/programming/glsl-autodiff/),
  [source, MIT, f2f6f7b, 2023-07-02](https://github.com/davepagurek/glsl-autodiff/tree/f2f6f7ba45859c222c37eb8d518a0d18b410e0de).

- **three-ntc:** this is especially close: a Three.js runtime and GPU trainer for
  learned material textures, with TSL decoding, fitting, Adam updates and model
  export. At the checked revision its training kernel explicitly describes a
  hand-differentiated backward pass, and its shared dense-layer kernels contain
  manually authored activation derivatives and backward propagation. Thus
  training neural fields inside Three is already implemented; automatic backward
  generation for a supported author-supplied native graph is the proposed
  addition. We have not integrated this prototype into three-ntc and cannot claim
  it supports that trainer's complete graph or improves its materials.
  [Repository, MIT, 2d35250, 2026-09-09](https://github.com/bhouston/three-ntc/tree/2d35250a708c110c2e3afe4108da79eaaf91bc44),
  [training kernel, lines 85 onward](https://github.com/bhouston/three-ntc/blob/2d35250a708c110c2e3afe4108da79eaaf91bc44/packages/three-ntc-trainer/src/NTCGPUComputeTSL.ts#L85),
  [manual activation derivative, line 42](https://github.com/bhouston/three-ntc/blob/2d35250a708c110c2e3afe4108da79eaaf91bc44/packages/three-ntc-trainer/src/NTCGPUKernelsTSL.ts#L42),
  [dense backward kernel, line 218](https://github.com/bhouston/three-ntc/blob/2d35250a708c110c2e3afe4108da79eaaf91bc44/packages/three-ntc-trainer/src/NTCGPUKernelsTSL.ts#L218).

- **A-delta, SIGGRAPH 2022:** automatically differentiates shader programs,
  optimizes parameters against target images, and exports fitted GLSL for
  subsequent editing and animation. It even treats certain discontinuities,
  which this continuous-expression prototype does not. This directly rules out
  broad claims about inventing editable fitted shaders or shader-parameter
  optimization. [Primary paper and implementation links](https://gfx.cs.princeton.edu/gfx/pubs/Yang_2022_AAF/index.php).

## Native Three boundary

This repository installs Three `0.185.1`. TSL represents arithmetic through
`OperatorNode` and `MathNode`, with explicit operator/method and child-node fields.
That structure makes a bounded traversal plausible. The official language
documentation describes the authoring system; it does not establish an autodiff
API. A local keyword check of the installed `src/nodes` found no advertised
autodiff implementation. That search is not a proof that no community solution
exists. [TSL documentation](https://github.com/mrdoob/three.js/wiki/Three.js-Shading-Language),
[r185 operator source](https://github.com/mrdoob/three.js/blob/2431a09f46f34c560bc8e44b33be0e567723d5b9/src/nodes/math/OperatorNode.js),
[r185 math source](https://github.com/mrdoob/three.js/blob/2431a09f46f34c560bc8e44b33be0e567723d5b9/src/nodes/math/MathNode.js).
The tag links describe r185; each local report must hash the actual installed
`0.185.1` files rather than equating a tag with the npm patch release.

The prototype depends on node representation details, not a stable upstream
autodiff contract. Unsupported nodes must fail explicitly. Native TSL input does
not imply all TSL, all materials, texture sampling, mutation, control flow,
visibility, rasterization or complete scene derivatives are supported.

## Decision

Proceed as a small capability test. A positive result would demonstrate a useful
native-graph training workflow across two structurally different smooth fields,
with verified derivatives and reusable weights. It would not establish a
breakthrough, a generally superior training framework, or an AI-generated-world
system. A tiny attractive surface alone is insufficient: the unsupported-graph
behavior, independent gradient checks, GPU finite-difference control and held-out
error gates are essential. Keep negative outcomes and limitations alongside any
working demonstration.
