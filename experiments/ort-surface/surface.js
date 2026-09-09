import * as THREE from '/.generated/ort-surface/three.webgpu.js';
import { Fn, storage, instanceIndex, vertexIndex, uniform, uint, float, vec3, vec4, mix } from '/.generated/ort-surface/three.tsl.js';
import * as ort from '/node_modules/onnxruntime-web-dev/dist/ort.webgpu.bundle.min.mjs';

const $ = id => document.getElementById(id);
const weights = await (await fetch('./wave.json')).json();
const errors = [];
let app, busy = false, playing = false, phase = 'idle';
const events = [], targets = new WeakMap(), groups = new WeakMap(), passes = new WeakMap();
const owners = new WeakMap();
const watched = buffer => targets.get(buffer);
function record(method, buffers, details = {}) {
  const roles = buffers.map(watched).filter(Boolean);
  if (roles.length) events.push({ phase, method, roles, ...details });
}
function wrap(proto, name, inspect) {
  const original = proto[name];
  proto[name] = function(...args) { const value = original.apply(this, args); inspect.call(this, args, value); return value; };
}
// Observe actual native buffer operations, including operations hidden inside ORT.
wrap(GPUQueue.prototype, 'writeBuffer', a => record('writeBuffer', [a[0]]));
wrap(GPUCommandEncoder.prototype, 'copyBufferToBuffer', a => record('copyBufferToBuffer', [a[0], a[2]], { bytes: a[4] }));
wrap(GPUCommandEncoder.prototype, 'clearBuffer', a => record('clearBuffer', [a[0]]));
wrap(GPUCommandEncoder.prototype, 'copyBufferToTexture', a => record('copyBufferToTexture', [a[0].buffer]));
wrap(GPUCommandEncoder.prototype, 'copyTextureToBuffer', a => record('copyTextureToBuffer', [a[1].buffer]));
wrap(GPUBuffer.prototype, 'mapAsync', function() { record('mapAsync', [this]); });
wrap(GPUBuffer.prototype, 'getMappedRange', function() { record('getMappedRange', [this]); });
wrap(GPUBuffer.prototype, 'destroy', function() { record('destroy', [this]); });
wrap(GPUDevice.prototype, 'createCommandEncoder', function(_a, encoder) { owners.set(encoder, this); });
for (const name of ['beginComputePass', 'beginRenderPass']) {
  wrap(GPUCommandEncoder.prototype, name, function(_a, pass) { owners.set(pass, owners.get(this)); });
}
wrap(GPUDevice.prototype, 'createBindGroup', (a, group) => {
  const bindings = a[0].entries.filter(e => watched(e.resource?.buffer)).map(e => ({
    role: watched(e.resource.buffer), binding: e.binding, offset: e.resource.offset ?? 0, size: e.resource.size ?? e.resource.buffer.size
  })); groups.set(group, bindings);
});
for (const [proto, command] of [[GPUComputePassEncoder.prototype, 'dispatchWorkgroups'], [GPURenderPassEncoder.prototype, 'drawIndexed']]) {
  wrap(proto, 'setBindGroup', function(a) { let active = passes.get(this); if (!active) passes.set(this, active = new Map()); active.set(a[0], groups.get(a[1]) ?? []); });
  wrap(proto, command, function(a) {
    const bindings = [...(passes.get(this)?.values() ?? [])].flat();
    if (bindings.length) events.push({ phase, method: command, bindings, args: a, sharedDevice: owners.get(this) === app?.device });
  });
}

function failUnless(condition, message) { if (!condition) throw new Error(message); }
function status(message) { $('status').textContent = message; }
async function scoped(label, body) {
  phase = label;
  for (const type of ['out-of-memory', 'internal', 'validation']) app.device.pushErrorScope(type);
  let value, failure;
  try { value = await body(); await app.device.queue.onSubmittedWorkDone(); }
  catch (error) { failure = error; }
  finally {
    for (let i = 0; i < 3; i++) { const e = await app.device.popErrorScope(); if (e) { errors.push({ phase, message: e.message }); failure ??= new Error(e.message); } }
    phase = 'idle';
  }
  if (failure) throw failure;
  return value;
}

function makeMaterial(values, side) {
  const material = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide, wireframe: true });
  const height = values.element(vertexIndex);
  material.positionNode = Fn(() => {
    const x = float(vertexIndex.mod(uint(side))).div(side - 1).mul(2).sub(1);
    const z = float(vertexIndex.div(uint(side))).div(side - 1).mul(2).sub(1);
    return vec3(x.mul(3.3), height.mul(2.2), z.mul(3.3));
  })();
  material.colorNode = mix(vec3(0.025, 0.24, 0.34), vec3(0.43, 1, 0.65), height.add(0.7).div(1.4));
  return material;
}

async function initialize(side = 128) {
  failUnless([64, 128, 256].includes(side), 'Resolution is capped at 65,536 vertices.');
  if (app) await teardown();
  $('size').value = String(side); $('lane').value = 'shared'; $('bytes').textContent = '0 bytes';
  $('path-label').textContent = 'Shared-buffer path, per update'; $('result').textContent = '';
  failUnless(navigator.gpu, 'WebGPU is unavailable in this browser.');
  status('Loading a small model and one GPU device…');
  const adapter = await navigator.gpu.requestAdapter();
  failUnless(adapter, 'No WebGPU adapter.');
  const device = await adapter.requestDevice();
  const n = side * side;
  app = { adapter, device, side, n, disposed: false };
  device.addEventListener('uncapturederror', e => errors.push({ phase, message: e.error.message }));
  device.lost.then(info => { if (!app?.disposed && info.reason !== 'destroyed') errors.push({ phase, message: `Device lost: ${info.message}` }); });
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = { wasm: '/node_modules/onnxruntime-web-dev/dist/ort-wasm-simd-threaded.asyncify.wasm' };
  ort.env.webgpu.adapter = adapter;
  app.session = await ort.InferenceSession.create(new Uint8Array(await (await fetch('./wave.onnx')).arrayBuffer()), {
    executionProviders: [{ name: 'webgpu', device, validationMode: 'full' }]
  });
  // ORT's global env property describes its default device. With a supplied
  // per-session device, verify actual command-encoder ownership below instead.
  app.ortGlobalDeviceMatches = await ort.env.webgpu.device === device;
  for (const [name, elements] of [['input', n * 3], ['output', n]]) {
    app[name] = device.createBuffer({ label: `ort-surface-${name}`, size: elements * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    targets.set(app[name], name);
    app[`${name}Attribute`] = new THREE.ExternalStorageBufferAttribute(app[name], { device, type: 'float', count: elements });
    app[`${name}Tensor`] = ort.Tensor.fromGpuBuffer(app[name], { dataType: 'float32', dims: [n, name === 'input' ? 3 : 1] });
  }
  const inputValues = storage(app.inputAttribute);
  const clock = uniform(0);
  app.clock = clock;
  app.seed = Fn(() => {
    const id = instanceIndex;
    const base = id.mul(uint(3));
    inputValues.element(base).assign(float(id.mod(uint(side))).div(side - 1).mul(2).sub(1));
    inputValues.element(base.add(uint(1))).assign(float(id.div(uint(side))).div(side - 1).mul(2).sub(1));
    inputValues.element(base.add(uint(2))).assign(clock);
  })().compute(n);
  app.roundtripAttribute = new THREE.StorageBufferAttribute(n, 1);
  app.sharedMaterial = makeMaterial(storage(app.outputAttribute).toReadOnly(), side);
  app.roundtripMaterial = makeMaterial(storage(app.roundtripAttribute).toReadOnly(), side);
  // PlaneGeometry supplies the shared topology; positionNode reads the ONNX tensor.
  app.geometry = new THREE.PlaneGeometry(2, 2, side - 1, side - 1);
  app.mesh = new THREE.Mesh(app.geometry, app.sharedMaterial);
  app.mesh.frustumCulled = false;
  app.scene = new THREE.Scene(); app.scene.add(app.mesh);
  app.camera = new THREE.PerspectiveCamera(42, 1, .1, 100);
  app.camera.position.set(7.2, 5.8, 8.5); app.camera.lookAt(0, -.2, 0);
  app.renderer = new THREE.WebGPURenderer({ device, antialias: false });
  app.renderer.setPixelRatio(1);
  app.renderer.setClearColor(0x090e14, 1);
  await app.renderer.init();
  const stage = $('stage');
  app.renderer.setSize(Math.min(stage.clientWidth, 1100), Math.min(stage.clientHeight, 650));
  app.camera.aspect = app.renderer.domElement.width / app.renderer.domElement.height;
  app.camera.updateProjectionMatrix();
  stage.prepend(app.renderer.domElement);
  app.target = new THREE.RenderTarget(512, 384, { samples: 0 });
  await update(0.9, 'shared');
  $('play').disabled = $('check').disabled = false;
  $('start').textContent = 'Reload';
  status(`${n.toLocaleString()} vertices ready. Animation is paused.`);
  return { ortVersion: ort.env.versions.web, threeRevision: THREE.REVISION, n,
    adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description } };
}

async function infer(t) {
  app.clock.value = t;
  app.renderer.compute(app.seed);
  const result = await app.session.run({ X: app.inputTensor }, { Y: app.outputTensor });
  failUnless(result.Y === app.outputTensor && result.Y.gpuBuffer === app.output, 'ORT replaced its preallocated output.');
}
async function draw(lane, toTarget = false) {
  if (lane === 'roundtrip') {
    // A deliberately explicit, fair baseline: same inference, download, then upload.
    const data = await app.renderer.getArrayBufferAsync(app.outputAttribute);
    app.roundtripAttribute.array.set(new Float32Array(data));
    app.roundtripAttribute.needsUpdate = true;
  }
  app.mesh.material = lane === 'shared' ? app.sharedMaterial : app.roundtripMaterial;
  app.renderer.setRenderTarget(toTarget ? app.target : null);
  app.renderer.render(app.scene, app.camera);
}
async function update(t, lane) {
  return scoped(`update-${lane}`, async () => { await infer(t); await draw(lane); });
}
function cpuReference(id, t) {
  const x = (id % app.side) / (app.side - 1) * 2 - 1;
  const z = Math.floor(id / app.side) / (app.side - 1) * 2 - 1;
  let y = weights.C;
  for (let k = 0; k < weights.hidden; k++) y += Math.sin(x * weights.W[0][k] + z * weights.W[1][k] + t * weights.W[2][k] + weights.B[k]) * weights.V[k];
  return y;
}
async function validate() {
  status('Checking numerical output, image parity and native buffer operations…');
  const report = { kind: 'ort-surface-correctness-v1', n: app.n, samples: [], errors,
    outputBytes: app.n * 4, ortVersion: ort.env.versions.web, threeRevision: THREE.REVISION };
  for (const t of [.375, 2.25]) {
    const start = events.length;
    await scoped('proof-shared', async () => { await infer(t); await draw('shared', true); });
    const trace = events.slice(start);
    failUnless(trace.filter(e => e.bindings).every(e => e.sharedDevice), 'A target command used a different GPU device.');
    const forbidden = trace.filter(e => ['writeBuffer', 'copyBufferToBuffer', 'copyBufferToTexture', 'copyTextureToBuffer', 'clearBuffer', 'mapAsync', 'getMappedRange', 'destroy'].includes(e.method));
    failUnless(forbidden.length === 0, 'Shared path touched CPU/copy operations on its target: ' + JSON.stringify(forbidden));
    failUnless(trace.some(e => e.method === 'dispatchWorkgroups' && e.bindings.some(b => b.role === 'input')), 'No input compute dispatch observed.');
    failUnless(trace.some(e => e.method === 'dispatchWorkgroups' && e.bindings.some(b => b.role === 'output')), 'No output inference dispatch observed.');
    failUnless(trace.some(e => e.method === 'drawIndexed' && e.bindings.some(b => b.role === 'output' && b.size === app.n * 4 && b.offset === 0)), 'No exact output-buffer draw binding observed.');
    const backend = app.renderer.backend.get(app.outputAttribute);
    failUnless(backend.buffer === app.output && backend.ownsBuffer === false, 'Three failed borrowed-buffer identity.');
    const sample = await scoped('oracle-readback', async () => {
      const sharedImage = await app.renderer.readRenderTargetPixelsAsync(app.target, 0, 0, 512, 384);
      const y = new Float32Array(await app.renderer.getArrayBufferAsync(app.outputAttribute));
      let maxError = 0; for (let id = 0; id < app.n; id++) maxError = Math.max(maxError, Math.abs(y[id] - cpuReference(id, t)));
      failUnless(Number.isFinite(maxError) && maxError < 2e-5, `CPU reference error ${maxError}`);
      await draw('roundtrip', true);
      const copiedImage = await app.renderer.readRenderTargetPixelsAsync(app.target, 0, 0, 512, 384);
      let unequalBytes = 0; for (let i = 0; i < sharedImage.length; i++) if (sharedImage[i] !== copiedImage[i]) unequalBytes++;
      failUnless(unequalBytes === 0, `Image mismatch: ${unequalBytes} bytes`);
      // Require actual non-background content, so two empty pictures cannot pass.
      const first = sharedImage.slice(0, 4); let nonBackground = 0;
      for (let i = 0; i < sharedImage.length; i += 4) if (first.some((v,k) => v !== sharedImage[i+k])) nonBackground++;
      failUnless(nonBackground > 1000, 'Rendered image is empty.');
      return { time: t, maxAbsoluteError: maxError, unequalImageBytes: unequalBytes, nonBackgroundPixels: nonBackground, trace };
    });
    report.samples.push(sample);
  }
  failUnless(errors.length === 0, 'GPU errors observed.');
  await scoped('preview', () => draw('shared'));
  report.status = 'passed';
  status('PASS · identical images, GPU-resident handoff.');
  $('result').textContent = `All ${app.n.toLocaleString()} heights checked at two times.\nImage differences: 0 bytes.\nShared output transfers: 0 bytes.\nRound trip: ${(app.n * 8 / 1024).toLocaleString()} KiB/update (download + upload).`;
  return report;
}

async function animate() {
  if (playing) { playing = false; return; }
  playing = true; $('play').textContent = 'Stop';
  const start = performance.now(); let frames = 0;
  try {
    while (playing && !document.hidden && performance.now() - start < 8000 && frames < 120) {
      const begin = performance.now();
      await update((.9 + (begin - start) * .0007) % (2 * Math.PI), $('lane').value); frames++;
      await new Promise(resolve => setTimeout(resolve, Math.max(0, 1000 / 15 - (performance.now() - begin))));
    }
    status(`Paused after ${frames} updates. Click Animate to run again.`);
    return { frames, cappedUpdatesPerSecond: 15, maximumSeconds: 8, stopped: true };
  } finally { playing = false; $('play').textContent = 'Animate 8s'; }
}

async function recordAnimation() {
  const stream = app.renderer.domElement.captureStream(15);
  const type = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find(MediaRecorder.isTypeSupported);
  failUnless(type, 'This browser cannot record WebM.');
  const recorder = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 2500000 });
  const chunks = [];
  recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  const finished = new Promise(resolve => recorder.onstop = resolve);
  try {
    recorder.start();
    const animation = await animate();
    recorder.stop(); await finished;
    const data = new Uint8Array(await new Blob(chunks, { type }).arrayBuffer());
    let binary = ''; for (const byte of data) binary += String.fromCharCode(byte);
    return { animation, mimeType: type, base64: btoa(binary) };
  } finally { for (const track of stream.getTracks()) track.stop(); }
}

async function teardown() {
  if (!app || app.disposed) return;
  phase = 'teardown'; playing = false;
  const old = app;
  await old.device.queue.onSubmittedWorkDone();
  old.sharedMaterial?.dispose(); old.roundtripMaterial?.dispose(); old.seed?.dispose();
  old.geometry?.dispose(); old.target?.dispose();
  old.inputAttribute?.dispose(); old.outputAttribute?.dispose();
  await old.renderer?.dispose(); old.renderer?.domElement.remove();
  await old.session?.release();
  const prematureDestroy = events.filter(e => e.phase === 'teardown' && e.method === 'destroy');
  failUnless(prematureDestroy.length === 0, 'A borrower destroyed an external buffer.');
  phase = 'owner-cleanup';
  old.input?.destroy(); old.output?.destroy(); old.disposed = true;
  old.device.destroy(); phase = 'idle';
  return { borrowerDestroyedBuffers: prematureDestroy.length, disposed: true };
}
async function action(body) {
  if (busy) return;
  busy = true; $('start').disabled = $('check').disabled = true;
  try { return await body(); } catch (error) { status(`Stopped: ${error.message}`); console.error(error); throw error; }
  finally { busy = false; $('start').disabled = false; $('check').disabled = !app || app.disposed; }
}
$('start').onclick = () => action(() => initialize(Number($('size').value))).catch(() => {});
$('check').onclick = () => action(validate).catch(() => {});
$('play').onclick = () => playing ? (playing = false) : action(animate).catch(() => {});
$('lane').onchange = () => { $('bytes').textContent = $('lane').value === 'shared' ? '0 bytes' : `${(app?.n ?? Number($('size').value) ** 2) * 8 / 1024} KiB`; $('path-label').textContent = $('lane').value === 'shared' ? 'Shared-buffer path, per update' : 'Download + upload, per update'; };
document.addEventListener('visibilitychange', () => { if (document.hidden) playing = false; });
window.ortSurface = { initialize, validate, teardown, update, animate, recordAnimation, events, errors, get state() { return { busy, playing, n: app?.n }; } };
