// Runs the exact packaged static server and UI. Invoke only when the shared GPU
// test slot is clear. One browser, one page/device at a time, 45-second watchdog.
import { chromium } from 'playwright-core';
import { spawn, execFileSync } from 'node:child_process';
import { createServer, request } from 'node:http';
import { readFile, writeFile, mkdir, realpath } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const option = process.argv.find(x => x.startsWith('--package='));
if (!option) throw Error('Required: --package=deliverables/local-elasticity-<timestamp>; optional --static-only');
const staticOnly = process.argv.includes('--static-only');
const packagePath = await realpath(path.resolve(root, option.slice(10)));
const deliverables = await realpath(path.join(root, 'deliverables'));
if (!packagePath.startsWith(deliverables + path.sep)) throw Error('Package must be inside workspace deliverables');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const assert = (ok, text) => { if (!ok) throw Error(text); };
const manifestBytes = await readFile(path.join(packagePath, 'manifest.json'));
const manifest = JSON.parse(manifestBytes);
const report = { kind: 'local-elasticity-packaged-ui-v1', status: 'failed', created: new Date().toISOString(),
  package: path.relative(root, packagePath).replaceAll('\\', '/'), manifestSHA256: sha(manifestBytes),
  three: manifest.three, staticOnly, errors: [], network: [], cases: [], watchdogExpired: false };
const output = path.join(root, 'results/development/local-elasticity-bundle', report.created.replaceAll(':', '-'));
await mkdir(output, { recursive: true });
const reservation = createServer();
await new Promise(resolve => reservation.listen(0, '127.0.0.1', resolve));
const port = reservation.address().port;
await new Promise(resolve => reservation.close(resolve));
const origin = `http://127.0.0.1:${port}`;
let browser, server, watchdog;
function statusAt(route) {
  return new Promise((resolve, reject) => {
    const req = request({ hostname: '127.0.0.1', port, path: route }, response => {
      response.resume(); response.on('end', () => resolve(response.statusCode));
    }); req.on('error', reject); req.end();
  });
}
const sanitize = value => String(value).replaceAll(root, '<workspace>/').replaceAll(root.replaceAll('\\', '/'), '<workspace>/');
try {
  report.verification = JSON.parse(execFileSync(process.execPath, [path.join(packagePath, 'verify.mjs')], { windowsHide: true, encoding: 'utf8' }));
  server = spawn(process.execPath, [path.join(packagePath, 'serve.mjs'), '--port=' + port], { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(Error('Packaged server startup timed out')), 5000);
    server.once('error', reject); server.once('exit', code => reject(Error('Server exited: ' + code)));
    server.stdout.once('data', () => { clearTimeout(timer); resolve(); });
  });
  report.routes = [];
  for (const route of ['/', '/experiments/local-elasticity/app.html', ...Object.keys(manifest.files)
    .filter(name => name.endsWith('.splat') || /^assets\/.*\.js$/.test(name)).map(name => '/' + name)]) {
    const status = await statusAt(route); report.routes.push({ route, status }); assert(status === 200, 'Missing packaged route: ' + route);
  }
  for (const route of ['/../package.json', '/%2e%2e/package.json', '/assets/%2e%2e/package.json', '/%5cWindows/system.ini', '/%00', '/%252e%252e/package.json']) {
    const status = await statusAt(route); report.routes.push({ route, status }); assert(status === 400 || status === 404, 'Path rejection failed: ' + route);
  }
  if (!staticOnly) {
    watchdog = setTimeout(() => { report.watchdogExpired = true; browser?.close().catch(() => {}); }, 45000);
    browser = await chromium.launch({ ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : { channel: 'chrome' }), headless: true });
    report.browser = browser.version();
    for (const asset of ['plant', 'spot']) {
      const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
      await page.addInitScript(() => {
        window.__bundleDeviceRequests = 0;
        if (typeof GPUAdapter !== 'undefined') {
          const original = GPUAdapter.prototype.requestDevice;
          GPUAdapter.prototype.requestDevice = function (...args) {
            window.__bundleDeviceRequests++; return original.apply(this, args);
          };
        }
      });
      page.on('pageerror', error => report.errors.push(sanitize(error.message)));
      page.on('console', message => { if (message.type() === 'error') report.errors.push(sanitize(message.text())); });
      page.on('request', req => {
        const url = new URL(req.url()); if (!['http:', 'https:'].includes(url.protocol)) return;
        const local = url.origin === origin; report.network.push({ path: local ? url.pathname : url.href, local });
        if (!local) report.errors.push('Unexpected external runtime request: ' + url.href);
      });
      page.on('response', response => { if (response.status() >= 400) report.errors.push('HTTP ' + response.status() + ' ' + new URL(response.url()).pathname); });
      const cell = { asset, status: 'failed' }; report.cases.push(cell);
      await page.goto(origin + '/', { timeout: 10000 });
      await page.waitForFunction(() => !!window.elasticityDemo);
      cell.idle = await page.evaluate(() => window.elasticityDemo.getState());
      assert(!cell.idle.ready, 'GPU renderer unexpectedly ready while idle');
      await page.selectOption('#asset', asset);
      if (asset === 'spot') {
        cell.countBoundaries = [];
        for (const count of [255, 150001]) {
          const buffer = Buffer.alloc(count * 32);
          for (let i = 0; i < count; i++) {
            for (let axis = 0; axis < 3; axis++) { buffer.writeFloatLE(((i + axis) % 31) / 31, i * 32 + 4 * axis); buffer.writeFloatLE(.01, i * 32 + 12 + 4 * axis); }
          }
          await page.setInputFiles('#upload', { name: `count-boundary-${count}.splat`, mimeType: 'application/octet-stream', buffer });
          await page.click('#prepare');
          await page.waitForFunction(() => window.elasticityDemo.evidence.errors.length > 0, null, { timeout: 5000 });
          const result = await page.evaluate(() => ({ state: window.elasticityDemo.getState(), errors: window.elasticityDemo.evidence.errors,
            deviceRequests: window.__bundleDeviceRequests }));
          assert(!result.state.ready && result.deviceRequests === 0, 'Count boundary allocated a GPU device');
          cell.countBoundaries.push({ count, bytes: buffer.length, ...result });
        }
        await page.setInputFiles('#upload', path.join(root, 'experiments/local-elasticity/assets/spot.splat'));
        cell.localUpload = 'Bundled Spot bytes selected through the file input; not an unseen-asset generalization test';
      }
      await page.click('#prepare');
      await page.waitForFunction(() => window.elasticityDemo.evidence.prepared || window.elasticityDemo.evidence.errors.length, null, { timeout: 18000 });
      cell.prepared = await page.evaluate(() => window.elasticityDemo.evidence);
      assert(cell.prepared.prepared && !cell.prepared.errors.length, 'Packaged preparation failed');
      const settle = async () => { await page.evaluate(() => window.elasticityDemo.stop()); await page.waitForFunction(() => !window.elasticityDemo.getState().pending); };
      const count = () => page.evaluate(() => window.elasticityDemo.evidence.steps.length);
      const snapshot = () => page.evaluate(() => ({ state: window.elasticityDemo.getState(), evidence: window.elasticityDemo.evidence }));
      await page.click('#play');
      await page.waitForFunction(() => window.elasticityDemo.evidence.steps.length >= 4 || window.elasticityDemo.evidence.errors.length, null, { timeout: 5000 });
      await settle();
      if (asset === 'spot') {
        cell.released = await snapshot();
        cell.nonconverged = cell.released.evidence.steps.filter(step => step.status && step.status !== 'converged');
        await page.click('#reset'); await page.waitForFunction(() => window.elasticityDemo.getState().q.every(x => x === 0));
        cell.finalEvidence = await page.evaluate(() => window.elasticityDemo.evidence);
        cell.reset = true; cell.functionalPassed = !cell.released.evidence.errors.length;
        cell.allSolvesConverged = !cell.nonconverged.length;
        cell.status = cell.functionalPassed && cell.allSolvesConverged ? 'passed' : 'failed-gate';
        await page.goto('about:blank'); await page.close(); continue;
      }
      const first = await count();
      await page.mouse.move(asset === 'plant' ? 490 : 550, asset === 'plant' ? 400 : 390); await page.mouse.down();
      await page.waitForFunction(() => !!window.elasticityDemo.evidence.lastGrab, null, { timeout: 2000 });
      await page.mouse.move(asset === 'plant' ? 655 : 715, asset === 'plant' ? 360 : 350, { steps: 8 });
      await page.waitForFunction(n => window.elasticityDemo.evidence.steps.length >= n + 24 || window.elasticityDemo.evidence.errors.length, first, { timeout: 7000 });
      await settle(); cell.pulled = await snapshot();
      await page.screenshot({ path: path.join(output, asset + '-pulled.png') });
      await page.mouse.up(); const releaseCount = await count();
      await page.evaluate(() => window.elasticityDemo.start());
      await page.waitForFunction(n => window.elasticityDemo.evidence.steps.length >= n + 12 || window.elasticityDemo.evidence.errors.length, releaseCount, { timeout: 5000 });
      await settle(); cell.released = await snapshot();
      assert(!cell.released.evidence.errors.length, 'Worker/render errors');
      assert(Math.max(...cell.released.state.q.map((x, i) => Math.abs(x - cell.pulled.state.q[i]))) > 1e-5, 'No release evolution');
      cell.nonconverged = cell.released.evidence.steps.filter(step => step.status && step.status !== 'converged');
      cell.resetDuringPending = await page.evaluate(() => {
        window.elasticityDemo.start(); const before = window.elasticityDemo.getState();
        document.getElementById('reset').click(); return { before, requested: window.elasticityDemo.getState() };
      });
      assert(cell.resetDuringPending.before.pending, 'Reset test did not exercise an in-flight solve');
      await page.waitForFunction(() => { const s = window.elasticityDemo.getState(); return !s.pending && !s.running && s.q.every(x => x === 0); });
      await page.waitForTimeout(100);
      cell.resetDuringPending.after = await page.evaluate(() => window.elasticityDemo.getState());
      assert(!cell.resetDuringPending.after.pending && !cell.resetDuringPending.after.running && cell.resetDuringPending.after.q.every(x => x === 0), 'Queued reset did not remain at rest');
      cell.finalEvidence = await page.evaluate(() => window.elasticityDemo.evidence);
      cell.nonconverged = cell.finalEvidence.steps.filter(step => step.status && step.status !== 'converged');
      cell.reset = true; cell.functionalPassed = true; cell.allSolvesConverged = !cell.nonconverged.length;
      cell.status = cell.allSolvesConverged ? 'passed' : 'functional-only';
      await page.goto('about:blank'); await page.close();
    }
    assert(!report.errors.length && !report.watchdogExpired, 'Packaged UI errors/watchdog');
    report.renderFrames = report.cases.reduce((sum, cell) => sum + cell.finalEvidence.frames, 0);
    assert(report.renderFrames <= 80, 'Exceeded the declared80-frame UI budget');
    report.gates = { functional: report.cases.every(cell => cell.functionalPassed),
      allSolvesConverged: report.cases.every(cell => cell.allSolvesConverged) };
  }
  report.status = staticOnly || report.gates.functional && report.gates.allSolvesConverged ? 'passed' : 'functional-only';
  if (report.status !== 'passed') process.exitCode = 1;
} catch (error) { report.failure = sanitize(error.stack ?? error); process.exitCode = 1; }
finally {
  clearTimeout(watchdog); await browser?.close().catch(() => {}); server?.kill();
  const bytes = Buffer.from(JSON.stringify(report, null, 2) + '\n');
  await writeFile(path.join(output, 'report.json'), bytes, { flag: 'wx' });
  await writeFile(path.join(output, 'report.sha256'), sha(bytes) + '  report.json\n', { flag: 'wx' });
  console.log(JSON.stringify({ report: path.relative(root, path.join(output, 'report.json')), status: report.status,
    sha256: sha(bytes), staticOnly, failure: report.failure, cases: report.cases.map(c => ({ asset: c.asset, status: c.status, steps: c.released?.evidence.steps.length, nonconverged: c.nonconverged?.length })) }));
}
