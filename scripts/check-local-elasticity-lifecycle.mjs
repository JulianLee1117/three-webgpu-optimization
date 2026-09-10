import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, realpath } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

if (process.argv.length > 2) throw Error('No options: one bounded lifecycle cohort.');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'), hash = x => createHash('sha256').update(x).digest('hex');
const prefix = '/experiments/local-elasticity/', sources = ['app.html', 'app.js', 'worker.mjs', 'assets.mjs', 'basis.mjs', 'deform.js', 'mechanics.mjs'];
const report = { kind: 'local-elasticity-lifecycle-v2', status: 'failed', createdAt: new Date().toISOString(), sourceHashes: {}, sourceSnapshots: {}, servedHashes: {},
  errors: [], requestFailures: [], cases: [], instrumentation: 'Synthetic document.hidden/visibilityState and visibilitychange/pagehide events. Real asset fetch, Worker, renderer.init/dispose/render/compute remain active. Explicit gates delay fetch completion, init completion, and delivery of one worker state callback. The fourth case injects NaN into the first ready-message basis gradient, causing the real deformer input guard to throw after renderer initialization. This is not a native operating-system visibility test.',
  limits: { watchdogMs: 45000, sequentialPages: 4, maxRealDevices: 3, maxConcurrentRealDevices: 1, quietObservationMs: 250 } };
for (const name of [...sources.map(n => prefix.slice(1) + n), 'scripts/check-local-elasticity-lifecycle.mjs']) {
  const raw = await readFile(path.join(root, name)); report.sourceHashes[name] = hash(raw); report.sourceSnapshots[name] = raw.toString('utf8');
}
const bootstrap = `<script type="module">import {WebGPURenderer} from '/node_modules/three-r186/src/Three.WebGPU.js'; window.lifecycleHarness.installRenderer(WebGPURenderer); await import('/experiments/local-elasticity/app.js'); window.lifecycleHarness.appLoaded=true;</script>`;
const htmlOriginal = report.sourceSnapshots[prefix.slice(1) + 'app.html'];
const html = htmlOriginal.replace('<script type="module" src="/experiments/local-elasticity/app.js"></script>', bootstrap);
if (html === htmlOriginal) throw Error('Expected app bootstrap was not found; instrumentation must fail closed.');
report.overlay = { path: prefix + 'app.html', originalSHA256: hash(htmlOriginal), servedSHA256: hash(html), servedText: html, reason: 'Install transparent lifecycle counters before importing the unchanged app module.' };

// Serialized by Playwright as a browser init script. Its complete source is
// retained in this runner snapshot. Only the explicitly labeled fault case
// corrupts physics input; ordinary cases keep the real numerical data unchanged.
function installHarness() {
  const counters = { workersCreated: 0, workersTerminated: 0, liveWorkers: 0, workerPosts: 0, stepPosts: 0, stateDeliveries: 0,
    heldStates: 0, renderersInitialized: 0, deviceCount: 0, liveDevices: 0, maxLiveDevices: 0, disposeStarted: 0, disposeFinished: 0, renderCalls: 0, computeCalls: 0,
    fetchesHeld: 0, initsHeld: 0, injectedSetupFaults: 0, deviceLoss: [], gpuErrors: [], events: [] };
  const config = { holdFetch: false, holdInit: false, holdState: false, injectBadGradientOnce: false };
  let hidden = false;
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => hidden });
  Object.defineProperty(document, 'visibilityState', { configurable: true, get: () => hidden ? 'hidden' : 'visible' });
  const nativeSetTimeout = window.setTimeout.bind(window), nativeClearTimeout = window.clearTimeout.bind(window), timers = new Map();
  window.setTimeout = (callback, delay, ...args) => {
    if (typeof callback !== 'function') return nativeSetTimeout(callback, delay, ...args);
    let id; id = nativeSetTimeout(() => { timers.delete(id); callback(...args); }, delay);
    timers.set(id, { delay: Number(delay) || 0 }); return id;
  };
  window.clearTimeout = id => { timers.delete(id); return nativeClearTimeout(id); };
  const held = { fetch: [], init: [], state: [] };
  const record = (type, more = {}) => counters.events.push({ type, at: performance.now(), ...more });
  const nativeFetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const response = await nativeFetch(...args);
    if (config.holdFetch && String(args[0]).includes('/assets/spot.splat')) {
      counters.fetchesHeld++; record('fetch-held'); await new Promise(resolve => held.fetch.push(resolve));
    }
    return response;
  };
  const NativeWorker = window.Worker;
  const onmessageDescriptor = Object.getOwnPropertyDescriptor(NativeWorker.prototype, 'onmessage');
  if (!onmessageDescriptor?.set) throw Error('Native Worker.onmessage instrumentation is unavailable');
  window.Worker = class InstrumentedWorker extends NativeWorker {
    constructor(...args) {
      super(...args); counters.workersCreated++; counters.liveWorkers++; record('worker-created');
      let terminated = false, handler = null;
      const originalTerminate = this.terminate.bind(this);
      this.terminate = () => { if (!terminated) { terminated = true; counters.workersTerminated++; counters.liveWorkers--; record('worker-terminated'); } return originalTerminate(); };
      Object.defineProperty(this, 'onmessage', { configurable: true, get: () => handler, set: fn => {
        handler = fn;
        onmessageDescriptor.set.call(this, fn == null ? null : event => {
          if (config.injectBadGradientOnce && event.data?.type === 'ready') {
            config.injectBadGradientOnce = false;
            if (!(event.data.gradients instanceof Float64Array) || event.data.gradients.length === 0) throw Error('Fault-injection input contract changed');
            event.data.gradients[0] = NaN; counters.injectedSetupFaults++;
            record('injected-invalid-first-gradient', { expectedFailure: 'Invalid gradients value' });
          }
          const deliver = () => { if (event.data?.type === 'state') counters.stateDeliveries++; return fn.call(this, event); };
          if (config.holdState && event.data?.type === 'state') { counters.heldStates++; held.state.push(deliver); record('state-held'); } else deliver();
        });
      } });
    }
    postMessage(...args) { counters.workerPosts++; if (args[0]?.type === 'step') counters.stepPosts++; return super.postMessage(...args); }
  };
  const initialized = new WeakSet(), disposed = new WeakSet();
  window.lifecycleHarness = {
    config, counters, appLoaded: false,
    installRenderer(Renderer) {
      const originalInit = Renderer.prototype.init, originalDispose = Renderer.prototype.dispose, originalRender = Renderer.prototype.render, originalCompute = Renderer.prototype.compute;
      Renderer.prototype.init = async function (...args) {
        const result = await originalInit.apply(this, args);
        if (!initialized.has(this)) {
          initialized.add(this); counters.renderersInitialized++; record('renderer-initialized');
          const device = this.backend?.device;
          if (this.backend?.isWebGPUBackend && device) {
            counters.deviceCount++; counters.liveDevices++; counters.maxLiveDevices = Math.max(counters.maxLiveDevices, counters.liveDevices); record('device-created');
            device.addEventListener('uncapturederror', event => counters.gpuErrors.push(String(event.error.message)));
            device.lost.then(loss => { counters.liveDevices--; counters.deviceLoss.push(loss.reason); record('device-lost', { reason: loss.reason }); });
          }
        }
        if (config.holdInit) { counters.initsHeld++; record('init-held'); await new Promise(resolve => held.init.push(resolve)); }
        return result;
      };
      Renderer.prototype.dispose = async function (...args) {
        if (disposed.has(this)) return originalDispose.apply(this, args);
        disposed.add(this); counters.disposeStarted++; record('dispose-started');
        const result = await originalDispose.apply(this, args);
        counters.disposeFinished++; record('dispose-finished'); return result;
      };
      Renderer.prototype.render = function (...args) { counters.renderCalls++; return originalRender.apply(this, args); };
      Renderer.prototype.compute = function (...args) { counters.computeCalls++; return originalCompute.apply(this, args); };
    },
    setHidden(value) { hidden = value; record('synthetic-visibility', { hidden: value }); document.dispatchEvent(new Event('visibilitychange')); },
    pagehide() { record('synthetic-pagehide'); window.dispatchEvent(new PageTransitionEvent('pagehide', { persisted: false })); },
    release(kind) { config[{ fetch: 'holdFetch', init: 'holdInit', state: 'holdState' }[kind]] = false; record('release', { kind, count: held[kind].length }); for (const release of held[kind].splice(0)) release(); },
    snapshot() { return { counters: structuredClone(counters), hidden, pendingTimers: Array.from(timers.values()), appLoaded: this.appLoaded,
      state: window.elasticityDemo?.getState() ?? null, evidence: window.elasticityDemo?.evidence ?? null, canvasCount: document.querySelectorAll('#view canvas').length,
      prepareDisabled: document.querySelector('#prepare')?.disabled, statusText: document.querySelector('#status')?.textContent,
      errorText: document.querySelector('#error')?.textContent }; }
  };
}

const prefixes = [prefix, '/node_modules/three-r186/src/', '/node_modules/three-r186/examples/jsm/'];
const server = createServer(async (request, response) => {
  try {
    if (!['GET', 'HEAD'].includes(request.method)) { response.writeHead(405).end(); return; }
    let name = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
    if (name === '/') name = prefix + 'app.html';
    if (name === '/favicon.ico') { response.writeHead(204).end(); return; }
    if (name.includes('\\') || name.includes('\0') || name.split('/').includes('..')) throw Error('Invalid route');
    const allowed = prefixes.find(p => name.startsWith(p)); if (!allowed) throw Error('Route outside allowlist');
    const base = await realpath(path.join(root, allowed)), file = await realpath(path.join(root, name)), relative = path.relative(base, file);
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) throw Error('Path containment failed');
    const raw = name === prefix + 'app.html' ? Buffer.from(html) : await readFile(file), digest = hash(raw);
    if (report.servedHashes[name] && report.servedHashes[name] !== digest) throw Error('Source changed during run');
    report.servedHashes[name] = digest;
    response.writeHead(200, { 'Content-Type': name.endsWith('.html') ? 'text/html; charset=utf-8' : name.endsWith('.splat') ? 'application/octet-stream' : 'text/javascript; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
    response.end(request.method === 'HEAD' ? undefined : raw);
  } catch (error) { report.requestFailures.push({ url: request.url, message: String(error.message) }); response.writeHead(404).end('Not found'); }
});
await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
const output = path.join(root, 'results/development/local-elasticity-lifecycle', new Date().toISOString().replaceAll(':', '-'));
await mkdir(output, { recursive: true });
const telemetry = () => { try { return execFileSync('nvidia-smi', ['--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw', '--format=csv,noheader'], { encoding: 'utf8', windowsHide: true, timeout: 3000 }).trim(); } catch { return null; } };
const assert = (condition, message) => { if (!condition) throw Error(message); };
let browser, page, expired = false;
report.preGpu = telemetry();
const watchdog = setTimeout(() => { expired = true; browser?.close().catch(() => {}); }, report.limits.watchdogMs);
async function snapshot() { return JSON.parse(await page.evaluate(() => JSON.stringify(window.lifecycleHarness.snapshot()))); }
async function openPage(config) {
  page = await browser.newPage({ viewport: { width: 800, height: 640 } });
  page.on('pageerror', error => report.errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') report.errors.push(message.text()); });
  await page.addInitScript(installHarness);
  await page.goto(`http://127.0.0.1:${server.address().port}/`, { timeout: 15000 });
  await page.waitForFunction(() => window.lifecycleHarness?.appLoaded, null, { timeout: 15000 });
  await page.selectOption('#asset', 'spot');
  assert(await page.inputValue('#asset') === 'spot', 'The bounded test asset was not selected');
  await page.evaluate(config => Object.assign(window.lifecycleHarness.config, config), config);
}
async function quietAfter(before, { expectedError = null } = {}) {
  await page.waitForTimeout(report.limits.quietObservationMs);
  await page.evaluate(() => window.lifecycleHarness.setHidden(false));
  await page.waitForTimeout(report.limits.quietObservationMs);
  const after = await snapshot();
  assert(after.counters.workerPosts === before.counters.workerPosts && after.counters.renderCalls === before.counters.renderCalls && after.counters.computeCalls === before.counters.computeCalls, 'Old work resurrected after cancellation/show');
  assert(after.counters.liveWorkers === 0 && after.counters.liveDevices === 0 && after.canvasCount === 0, 'Resources survived cancellation');
  assert(!after.state.ready && !after.state.running && !after.state.pending && !after.evidence.prepared && after.state.q === null, 'App retained a ready/running state after cancellation');
  assert(after.pendingTimers.length === 0, 'A timeout survived cancellation');
  const errorsExpected = expectedError == null ? after.evidence.errors.length === 0 : after.evidence.errors.length === 1 && after.evidence.errors[0].includes(expectedError);
  assert(after.counters.gpuErrors.length === 0 && errorsExpected, 'Cancellation raised unexpected GPU/app errors');
  return after;
}
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true }); report.browser = browser.version();
  if (expired) throw Error('Startup exceeded watchdog');
  await openPage({ holdFetch: true }); await page.click('#prepare');
  await page.waitForFunction(() => window.lifecycleHarness.counters.fetchesHeld === 1, null, { timeout: 10000 });
  const fetchCase = { id: 'delayed-fetch-hidden', before: await snapshot() }; report.cases.push(fetchCase);
  await page.evaluate(() => { window.lifecycleHarness.setHidden(true); window.lifecycleHarness.release('fetch'); });
  await page.waitForTimeout(30); fetchCase.after = await quietAfter(await snapshot());
  assert(fetchCase.after.counters.workersCreated === 0 && fetchCase.after.counters.deviceCount === 0, 'Stale fetch created resources'); fetchCase.passed = true;
  await page.close(); page = null;

  await openPage({ holdInit: true }); await page.click('#prepare');
  await page.waitForFunction(() => window.lifecycleHarness.counters.initsHeld === 1 || window.elasticityDemo.evidence.errors.length, null, { timeout: 15000 });
  const initCase = { id: 'real-device-delayed-init-hidden', before: await snapshot() }; report.cases.push(initCase);
  assert(initCase.before.counters.deviceCount === 1 && initCase.before.counters.liveDevices === 1 && initCase.before.counters.renderCalls === 0, 'Real init gate was not reached');
  await page.evaluate(() => { window.lifecycleHarness.setHidden(true); window.lifecycleHarness.release('init'); });
  await page.waitForFunction(() => window.lifecycleHarness.counters.disposeFinished === 1 && window.lifecycleHarness.counters.liveDevices === 0, null, { timeout: 5000 });
  initCase.after = await quietAfter(await snapshot());
  assert(initCase.after.counters.renderCalls === 0 && initCase.after.counters.computeCalls === 0 && initCase.after.counters.deviceLoss[0] === 'destroyed', 'Stale renderer performed work or failed to destroy its device'); initCase.passed = true;
  await page.close(); page = null;

  await openPage({ holdState: true }); await page.click('#prepare');
  await page.waitForFunction(() => window.elasticityDemo.evidence.prepared || window.elasticityDemo.evidence.errors.length, null, { timeout: 15000 });
  assert((await snapshot()).evidence.prepared, 'App failed normal preparation');
  await page.evaluate(() => window.elasticityDemo.start());
  await page.waitForFunction(() => window.lifecycleHarness.counters.heldStates === 1 || window.elasticityDemo.evidence.errors.length, null, { timeout: 5000 });
  const activeCase = { id: 'existing-renderer-hidden-with-stale-state-and-pagehide', before: await snapshot() }; report.cases.push(activeCase);
  assert(activeCase.before.state.pending && activeCase.before.counters.stepPosts === 1 && activeCase.before.counters.renderCalls === 1, 'Live pending step gate was not reached');
  await page.evaluate(() => { window.lifecycleHarness.setHidden(true); window.lifecycleHarness.pagehide(); window.lifecycleHarness.release('state'); });
  await page.waitForFunction(() => window.lifecycleHarness.counters.disposeFinished === 1 && window.lifecycleHarness.counters.liveDevices === 0, null, { timeout: 5000 });
  activeCase.after = await quietAfter(await snapshot());
  assert(activeCase.after.counters.disposeStarted === 1 && activeCase.after.counters.disposeFinished === 1 && activeCase.after.counters.stateDeliveries === 1 && activeCase.after.evidence.steps.length === 0, 'Late state was applied, or disposal was not idempotent/awaited'); activeCase.passed = true;
  await page.close(); page = null;

  await openPage({ injectBadGradientOnce: true });
  const faultCase = { id: 'injected-deformer-setup-failure-disposes-real-device', initial: await snapshot(),
    fault: 'Ready-message gradients[0] replaced with NaN; original deformer finite-input guard must reject after real renderer.init().' }; report.cases.push(faultCase);
  await page.click('#prepare');
  await page.waitForFunction(() => window.elasticityDemo.evidence.errors.length > 0 && window.lifecycleHarness.counters.disposeFinished === 1 && window.lifecycleHarness.counters.liveDevices === 0, null, { timeout: 15000 });
  faultCase.after = await quietAfter(await snapshot(), { expectedError: 'Invalid gradients value' });
  assert(faultCase.after.counters.injectedSetupFaults === 1 && faultCase.after.counters.deviceCount === 1 && faultCase.after.counters.disposeStarted === 1 && faultCase.after.counters.deviceLoss[0] === 'destroyed', 'Injected setup failure did not dispose its initialized device exactly once');
  assert(faultCase.after.counters.renderCalls === 0 && faultCase.after.counters.computeCalls === 0 && faultCase.after.errorText.includes('Invalid gradients value') && !faultCase.after.prepareDisabled, 'Setup failure dispatched GPU work or did not expose a recoverable error');
  faultCase.passed = true;
  await page.close(); page = null;
  report.deviceCount = report.cases.reduce((sum, c) => sum + c.after.counters.deviceCount, 0);
  assert(report.deviceCount === 3 && report.cases.every(c => c.after.counters.maxLiveDevices <= 1), 'Device budget mismatch');
  assert(report.errors.length === 0 && report.requestFailures.length === 0, 'Browser or route errors');
  for (const [name, expected] of Object.entries(report.sourceHashes)) assert(hash(await readFile(path.join(root, name))) === expected, `Captured source changed: ${name}`);
  for (const [name, expected] of Object.entries(report.servedHashes)) if (name !== report.overlay.path) assert(hash(await readFile(path.join(root, name))) === expected, `Served source changed: ${name}`);
  report.status = 'passed';
} catch (error) {
  report.failure = String(error.stack ?? error); process.exitCode = 1;
  if (page && !page.isClosed()) try { report.partial = await snapshot(); } catch {}
} finally {
  clearTimeout(watchdog);
  if (page && !page.isClosed()) {
    try {
      await page.evaluate(() => {
        window.lifecycleHarness.setHidden(true); window.lifecycleHarness.pagehide();
        for (const kind of ['fetch', 'init', 'state']) window.lifecycleHarness.release(kind);
      });
      await page.waitForFunction(() => window.lifecycleHarness.counters.liveWorkers === 0 && window.lifecycleHarness.counters.liveDevices === 0, null, { timeout: 5000 });
      report.failureTeardown = await snapshot();
    } catch (error) { report.failureTeardownError = String(error.message); }
  }
  await browser?.close(); await new Promise(resolve => server.close(resolve));
  report.watchdogExpired = expired; report.postGpu = telemetry();
  const raw = JSON.stringify(report, null, 2) + '\n'; await writeFile(path.join(output, 'report.json'), raw, { flag: 'wx' }); await writeFile(path.join(output, 'report.sha256'), hash(raw) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ output, sha256: hash(raw), status: report.status, failure: report.failure, cases: report.cases.map(c => ({ id: c.id, passed: c.passed, after: c.after?.counters })),
    errors: report.errors, requestFailures: report.requestFailures, preGpu: report.preGpu, postGpu: report.postGpu, watchdogExpired: expired }, null, 2));
}
