import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GaussianSplat } from 'three/addons/objects/GaussianSplat.js';
import { SPLATLoader } from 'three/addons/loaders/SPLATLoader.js';
import { decodeSplat, normalizePositions } from './assets.mjs';
import { createSplatDeformer, deformPoint } from './deform.js';

const $ = id => document.getElementById(id), view = $('view');
let worker, renderer, scene, camera, controls, splat, deformer, positions, fields, q, normalization, grip;
let running = false, pending = false, resetPending = false, timer, watchdog, until = 0, generation = 0, drag = null, plane, lastStepAt = 0;
const evidence = { prepared: false, frames: 0, steps: [], errors: [], preparation: null };
window.elasticityDemo = { evidence, getState: () => ({ running, pending, ready: !!deformer, q: q ? Array.from(q) : null }), start: () => start(), stop: () => stop(), reset };

function stop() { running = false; clearTimeout(timer); $('play').textContent = 'Run for 12 seconds'; }
function reset() {
  stop(); drag = null; if (controls) controls.enabled = true;
  if (!worker || !deformer || resetPending) return;
  // Worker messages are FIFO: reset follows an already running solve.
  pending = true; resetPending = true; worker.postMessage({ type: 'reset' });
}
function fail(error) {
  generation++; evidence.errors.push(String(error));
  dispose().catch(disposalError => evidence.errors.push(String(disposalError)));
  $('error').textContent = String(error); $('busy').style.display = 'none'; $('prepare').disabled = false;
}
function render() {
  if (!renderer) return;
  if (grip) {
    grip.visible = !!drag;
    if (drag) {
      grip.children[0].position.set(...drag.target);
      const a = grip.children[1].geometry.getAttribute('position');
      a.array.set(deformPoint(drag.restPosition, drag.weights, q), 0); a.array.set(drag.target, 3); a.needsUpdate = true;
    }
  }
  renderer.render(scene, camera); evidence.frames++;
}
async function dispose() {
  stop(); clearTimeout(watchdog); worker?.terminate(); worker = null; controls?.dispose();
  const owned = renderer; renderer = null; owned?.domElement.remove(); deformer = null; pending = false; resetPending = false; drag = null;
  fields = null; q = null; positions = null; grip = null;
  evidence.prepared = false;
  $('play').disabled = true; $('reset').disabled = true;
  await owned?.dispose();
}
function start() { if (!worker || !deformer || pending) return; running = true; until = performance.now() + 12000; $('play').textContent = 'Pause'; tick(); }
function tick() {
  if (!running || pending || !worker) return;
  if (performance.now() >= until) { stop(); $('status').textContent += ' · Paused'; return; }
  pending = true;
  lastStepAt = performance.now();
  worker.postMessage({ type: 'step', pinned: $('pinned').checked, drag });
}

async function openRenderer(data, geometry, own) {
  const created = new THREE.WebGPURenderer({ antialias: false });
  created.setPixelRatio(1); created.setSize(view.clientWidth, view.clientHeight); created.setClearColor('#14282d');
  try { await created.init(); } catch (error) { await created.dispose(); throw error; }
  if (own !== generation || document.hidden) { await created.dispose(); return false; }
  if (!created.backend.isWebGPUBackend) { await created.dispose(); throw Error('This experiment requires a WebGPU browser.'); }
  renderer = created;
  scene = new THREE.Scene(); camera = new THREE.PerspectiveCamera(42, view.clientWidth / view.clientHeight, 0.01, 30);
  camera.position.set(1.2, 0.55, 1.7); camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
  splat = new GaussianSplat(geometry); scene.add(splat);
  const grid = new THREE.GridHelper(2.4, 12, '#536c65', '#2b4342'); grid.position.y = -0.63; scene.add(grid);
  grip = new THREE.Group(); grip.visible = false;
  grip.add(new THREE.Mesh(new THREE.SphereGeometry(0.014, 8, 6), new THREE.MeshBasicMaterial({ color: '#ffcb87', depthTest: false })));
  const stringGeometry = new THREE.BufferGeometry(); stringGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
  const string = new THREE.Line(stringGeometry, new THREE.LineBasicMaterial({ color: '#ffcb87', depthTest: false })); string.frustumCulled = false; grip.add(string); scene.add(grip);
  controls = new OrbitControls(camera, renderer.domElement); controls.minDistance = 0.55; controls.maxDistance = 4; controls.enableDamping = false; controls.addEventListener('change', render);
  fields = { weights: data.weights, gradients: data.gradients, modeCount: data.modeCount }; q = data.q;
  deformer = createSplatDeformer(splat, positions, geometry.getAttribute('covariance').array, fields);
  view.prepend(renderer.domElement); $('empty').style.display = 'none';
  deformer.update(renderer, q); render();
  setupPicking();
  return true;
}

function setupPicking() {
  const canvas = renderer.domElement, raycaster = new THREE.Raycaster(), projected = new THREE.Vector3();
  const xy = event => { const r = canvas.getBoundingClientRect(); return new THREE.Vector2((event.clientX - r.left) / r.width * 2 - 1, -(event.clientY - r.top) / r.height * 2 + 1); };
  canvas.addEventListener('pointerdown', event => {
    if (event.button !== 0 || !fields) return;
    const mouse = xy(event), pixelRadius = 16 / canvas.clientHeight * 2;
    let best = -1, distance = pixelRadius * pixelRadius, point;
    camera.updateMatrixWorld();
    for (let i = 0; i < positions.length / 3; i++) {
      const x = deformPoint(positions.subarray(i * 3, i * 3 + 3), fields.weights.subarray(i * fields.modeCount, (i + 1) * fields.modeCount), q);
      projected.set(...x).project(camera);
      if (projected.z < -1 || projected.z > 1) continue;
      const d = (projected.x - mouse.x) ** 2 * camera.aspect ** 2 + (projected.y - mouse.y) ** 2;
      if (d < distance) { distance = d; best = i; point = x; }
    }
    if (best < 0) return;
    controls.enabled = false; event.stopImmediatePropagation(); event.preventDefault(); canvas.setPointerCapture(event.pointerId);
    plane = new THREE.Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()), new THREE.Vector3(...point));
    drag = { restPosition: Array.from(positions.subarray(best * 3, best * 3 + 3)), weights: Array.from(fields.weights.subarray(best * fields.modeCount, (best + 1) * fields.modeCount)), target: point, stiffness: 25 };
    evidence.lastGrab = { index: best, restPosition: drag.restPosition, startPosition: point };
    if (!running) start();
  }, { capture: true });
  canvas.addEventListener('pointermove', event => {
    if (!drag) return;
    raycaster.setFromCamera(xy(event), camera);
    const hit = raycaster.ray.intersectPlane(plane, new THREE.Vector3());
    if (hit) { drag.target = hit.toArray().map((x, d) => Math.max(drag.restPosition[d] - 0.55, Math.min(drag.restPosition[d] + 0.55, x))); render(); }
  });
  const release = () => { drag = null; if (controls) controls.enabled = true; render(); };
  canvas.addEventListener('pointerup', release); canvas.addEventListener('pointercancel', release); canvas.addEventListener('lostpointercapture', release);
}

$('prepare').onclick = async () => {
  generation++; const own = generation; await dispose(); if (own !== generation || document.hidden) return;
  evidence.prepared = false; evidence.errors = []; evidence.steps = [];
  $('error').textContent = ''; $('prepare').disabled = true; $('empty').style.display = 'grid'; $('busy').style.display = 'block';
  $('empty').querySelector('strong').textContent = 'Preparing on this computer';
  try {
    const upload = $('upload').files[0], asset = $('asset').value;
    if (upload && upload.size > 4_800_000) throw Error('Use a .splat file up to 4.8 MB (150,000 splats).');
    const bytes = upload ? await upload.arrayBuffer() : await (await fetch(new URL(`assets/${asset}.splat`, import.meta.url))).arrayBuffer();
    if (own !== generation || document.hidden) return;
    if (bytes.byteLength < 256 * 32 || bytes.byteLength > 150000 * 32) throw Error('Use a .splat file with 256 to 150,000 splats.');
    const decoded = decodeSplat(bytes); normalization = normalizePositions(decoded.positions, { flipY: $('flip').checked }); positions = normalization.positions;
    const geometry = new SPLATLoader().parse(bytes);
    geometry.getAttribute('position').array.set(positions);
    const cov = geometry.getAttribute('covariance').array, s2 = normalization.scale ** 2;
    for (let i = 0; i < cov.length; i++) cov[i] *= s2 * (normalization.flipY && (i % 6 === 1 || i % 6 === 4) ? -1 : 1);
    geometry.computeBoundingBox(); geometry.computeBoundingSphere();
    $('credit').textContent = upload ? `Local file: ${upload.name}` : asset === 'plant' ? 'Houseplant: phone capture by Marcel Padilla · CC BY 4.0 · splats collection.' : 'Spot by Keenan Crane (CC0); mesh-to-splat conversion by Marcel Padilla.';
    worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
    worker.onerror = error => fail(error.message);
    watchdog = setTimeout(() => { worker?.terminate(); fail('Preparation exceeded the 90-second limit.'); }, 90000);
    worker.onmessage = async ({ data }) => {
      if (own !== generation) return;
      try {
        if (data.type === 'progress') $('status').textContent = data.text;
        if (data.type === 'error') { fail(data.message); return; }
        if (data.type === 'ready') {
          clearTimeout(watchdog); evidence.preparation = data.diagnostics;
          if (!await openRenderer(data, geometry, own)) return;
          evidence.prepared = true; $('prepare').disabled = false; $('play').disabled = false; $('reset').disabled = false; $('busy').style.display = 'none';
          $('metrics').textContent = `${decoded.count.toLocaleString()} splats · ${data.modeCount} mechanical fields\nPrepared in ${(data.diagnostics.prepareMs / 1000).toFixed(2)} s`;
          $('status').textContent = 'Ready · Grab the object or run the simulation';
        }
        if (data.type === 'state') {
          if (data.diagnostics.reset) resetPending = false;
          pending = resetPending; q = data.q; evidence.steps.push({ solveMs: data.solveMs, ...data.diagnostics }); if (evidence.steps.length > 600) evidence.steps.shift();
          deformer.update(renderer, q); render();
          $('status').textContent = `Simulated ${data.simulatedSeconds.toFixed(2)} s · solve ${data.solveMs.toFixed(1)} ms · ${data.diagnostics.status ?? 'Reset'} · ${running ? 'Live' : 'Paused'}`;
          if (running) timer = setTimeout(tick, Math.max(0, 33 - (performance.now() - lastStepAt)));
        }
      } catch (error) { if (own === generation) fail(error); }
    };
    worker.postMessage({ type: 'prepare', positions });
  } catch (error) { if (own === generation) fail(error); }
};
$('play').onclick = () => running ? stop() : start();
$('reset').onclick = reset;
addEventListener('resize', () => { if (!renderer) return; renderer.setSize(view.clientWidth, view.clientHeight); camera.aspect = view.clientWidth / view.clientHeight; camera.updateProjectionMatrix(); render(); });
document.addEventListener('visibilitychange', () => { if (document.hidden) { generation++; dispose().catch(error => evidence.errors.push(String(error))); $('prepare').disabled = false; $('status').textContent = 'Resources released while hidden · Prepare to reopen'; } });
addEventListener('pagehide', () => { generation++; dispose().catch(error => evidence.errors.push(String(error))); });
