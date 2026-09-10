import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, realpath } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

// Exactly one small, disposable browser/device run; no resident UI or benchmark loop.
if (process.argv.length > 2) throw Error('No options supported: this executes one bounded GPU correctness probe.');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const sourceFiles = ['scripts/check-local-elasticity-gpu.mjs', ...['deform.js', 'basis.mjs', 'assets.mjs', 'gpu-probe.js', 'gpu-probe.html'].map(p => `experiments/local-elasticity/${p}`),
  'experiments/local-elasticity/assets/manifest.json', 'node_modules/three-r186/package.json',
  'node_modules/three-r186/examples/jsm/objects/GaussianSplat.js', 'node_modules/three-r186/examples/jsm/gpgpu/CountingSort.js',
  'node_modules/three-r186/examples/jsm/utils/GaussianSplatUtils.js', 'node_modules/three-r186/examples/jsm/loaders/SPLATLoader.js'];
const report = { kind: 'local-elasticity-gpu-report-v1', status: 'failed', createdAt: new Date().toISOString(), sourceHashes: {}, sourceSnapshots: {}, servedHashes: {},
  requestFailures: [], errors: [], cpuOnlyWhileIdle: true, limits: { wallClockMs: 45000, renderSize: [256, 192], maxDeformationDispatches: 8, rendererCount: 1 } };
for (const file of sourceFiles) { const raw = await readFile(path.join(root, file)); report.sourceHashes[file] = hash(raw); report.sourceSnapshots[file] = raw.toString('utf8'); }
const assetPath = 'experiments/local-elasticity/assets/spot.splat';
report.asset = { path: assetPath, sha256: hash(await readFile(path.join(root, assetPath))) };
const prefixes = ['/experiments/local-elasticity/', '/node_modules/three-r186/'];
const server = createServer(async (request, response) => {
  try {
    if (!['GET', 'HEAD'].includes(request.method)) { response.writeHead(405, { Allow: 'GET, HEAD' }).end(); return; }
    let name = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
    if (name === '/') name = '/experiments/local-elasticity/gpu-probe.html';
    if (name === '/favicon.ico') { response.writeHead(204).end(); return; }
    if (name.includes('\\') || name.includes('\0') || name.split('/').includes('..')) throw Error('Invalid route');
    const prefix = prefixes.find(p => name.startsWith(p)); if (!prefix) throw Error('Outside experiment allowlist');
    const base = await realpath(path.join(root, prefix)), file = await realpath(path.join(root, name)), relative = path.relative(base, file);
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) throw Error('Route containment failure');
    const raw = await readFile(file), digest = hash(raw);
    if (report.servedHashes[name] && report.servedHashes[name] !== digest) throw Error('Source changed during execution');
    report.servedHashes[name] = digest;
    response.writeHead(200, { 'Content-Type': name.endsWith('.html') ? 'text/html; charset=utf-8' : name.endsWith('.splat') ? 'application/octet-stream' : name.endsWith('.json') ? 'application/json' : 'text/javascript; charset=utf-8',
      'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
    response.end(request.method === 'HEAD' ? undefined : raw);
  } catch (error) { report.requestFailures.push({ url: request.url, message: String(error.message) }); response.writeHead(404).end('Not found'); }
});
await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
const output = path.join(root, 'results/development/local-elasticity-gpu', new Date().toISOString().replaceAll(':', '-') + '-covariance');
await mkdir(output, { recursive: true });
function telemetry() { try { return execFileSync('nvidia-smi', ['--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw', '--format=csv,noheader'], { encoding: 'utf8', timeout: 3000, windowsHide: true }).trim(); } catch { return null; } }
let browser, page, expired = false;
report.preGpu = telemetry();
const watchdog = setTimeout(() => { expired = true; browser?.close().catch(() => {}); }, report.limits.wallClockMs);
try {
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  if (expired) throw Error('Browser startup exceeded watchdog');
  report.browser = browser.version();
  page = await browser.newPage({ viewport: { width: 320, height: 240 } });
  page.on('pageerror', error => report.errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') report.errors.push(message.text()); });
  await page.goto(`http://127.0.0.1:${server.address().port}/`, { timeout: 15000 });
  await page.waitForFunction(() => window.localElasticityProbe, null, { timeout: 15000 });
  const json = await page.evaluate(async () => JSON.stringify(await window.localElasticityProbe.runProbe(), (_, v) => ArrayBuffer.isView(v) ? Array.from(v) : v));
  report.probe = JSON.parse(json);
  if (report.probe.status !== 'passed') throw Error(report.probe.failure ?? 'Probe gate failed');
  if (report.errors.length || report.requestFailures.length) throw Error('Browser or route error');
  if (report.probe.lifecycle.rendererCount !== 1 || report.probe.lifecycle.deformationDispatches !== 8 || report.probe.lifecycle.renderCount !== 8 || !report.probe.lifecycle.disposed || report.probe.lifecycle.deviceLossReason !== 'destroyed') throw Error('Unexpected bounded device lifecycle');
  for (const [name, expected] of Object.entries(report.sourceHashes)) if (hash(await readFile(path.join(root, name))) !== expected) throw Error(`Captured source changed: ${name}`);
  for (const [name, expected] of Object.entries(report.servedHashes)) if (hash(await readFile(path.join(root, name))) !== expected) throw Error(`Executed source changed: ${name}`);
  if (report.probe.groups.find(g => g.id === 'spot-128').input.preparation.assetSHA256 !== report.asset.sha256) throw Error('Executed asset hash differs');
  report.status = 'passed';
} catch (error) {
  report.failure = String(error.stack ?? error); process.exitCode = 1;
  if (!report.probe && page && !page.isClosed()) try { report.partialProbe = JSON.parse(await page.evaluate(() => JSON.stringify(window.localElasticityProbeResult ?? null, (_, v) => ArrayBuffer.isView(v) ? Array.from(v) : v))); } catch {}
} finally {
  clearTimeout(watchdog); await browser?.close(); await new Promise(resolve => server.close(resolve));
  report.watchdogExpired = expired; report.postGpu = telemetry();
  const raw = JSON.stringify(report, null, 2) + '\n';
  await writeFile(path.join(output, 'report.json'), raw, { flag: 'wx' }); await writeFile(path.join(output, 'report.sha256'), hash(raw) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ output, sha256: hash(raw), status: report.status, failure: report.failure, cells: report.probe?.cells.map(c => ({ group: c.group, state: c.state, center: c.centerCheck, covariance: c.covarianceCheck })),
    lifecycle: report.probe?.lifecycle, errors: report.errors, requestFailures: report.requestFailures, preGpu: report.preGpu, postGpu: report.postGpu, watchdogExpired: expired }, null, 2));
}
