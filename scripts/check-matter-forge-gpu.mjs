import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, realpath } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

if (process.argv.length > 2) throw Error('No options: one bounded four-fixture GPU correctness run on an ephemeral loopback port.');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sha = raw => createHash('sha256').update(raw).digest('hex');
const prefix = '/experiments/matter-forge/';
const sourceFiles = ['scripts/check-matter-forge-gpu.mjs', ...['probe.html', 'probe.mjs', 'cpu.mjs', 'gpu.mjs'].map(file => 'experiments/matter-forge/' + file)];
const allowed = new Set(sourceFiles.slice(1).map(file => '/' + file));
const report = { kind: 'matter-forge-gpu-parity-report-v1', createdAt: new Date().toISOString(), status: 'failed',
  sourceHashes: {}, sourceSnapshots: {}, servedHashes: {}, errors: [], requestFailures: [],
  limits: { wallClockMs: 45000, deviceCount: 1, fixtures: 4, particlesPerFixture: 64, totalSteps: 25, gridN: 12, dispatches: 125 } };
const output = path.join(root, 'results/development/matter-forge-gpu', report.createdAt.replaceAll(':', '-'));
await mkdir(output, { recursive: true });
let browser, page, server, watchdog, expired = false;
function telemetry() {
  try { return execFileSync('nvidia-smi', ['--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw', '--format=csv,noheader'], { encoding: 'utf8', timeout: 3000, windowsHide: true }).trim(); }
  catch { return null; }
}
try {
  for (const name of sourceFiles) { const raw = await readFile(path.join(root, name)); report.sourceHashes[name] = sha(raw); report.sourceSnapshots[name] = raw.toString('utf8'); }
  const base = await realpath(path.join(root, prefix));
  server = createServer(async (request, response) => {
    try {
      if (!['GET', 'HEAD'].includes(request.method)) { response.writeHead(405).end(); return; }
      let name = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
      if (name === '/') name = prefix + 'probe.html';
      if (name === '/favicon.ico') { response.writeHead(204).end(); return; }
      if (name.includes('\\') || name.includes('\0') || name.split('/').includes('..') || !allowed.has(name)) throw Error('Outside exact source allowlist');
      const file = await realpath(path.join(root, name)), relative = path.relative(base, file);
      if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) throw Error('Route containment');
      const raw = await readFile(file), digest = sha(raw);
      if (digest !== report.sourceHashes[name.slice(1)]) throw Error('Source changed after pre-run snapshot');
      report.servedHashes[name] = digest;
      response.writeHead(200, { 'Content-Type': name.endsWith('.html') ? 'text/html; charset=utf-8' : 'text/javascript; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
      response.end(request.method === 'HEAD' ? undefined : raw);
    } catch (error) { report.requestFailures.push({ url: request.url, message: String(error.message) }); response.writeHead(404).end('Not found'); }
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  report.preGpu = telemetry();
  watchdog = setTimeout(() => { expired = true; browser?.close().catch(() => {}); }, report.limits.wallClockMs);
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  if (expired) throw Error('Watchdog during browser launch');
  report.browser = browser.version();
  page = await browser.newPage({ viewport: { width: 96, height: 96 } });
  page.on('pageerror', error => report.errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') report.errors.push(message.text()); });
  await page.goto(`http://127.0.0.1:${server.address().port}/`, { timeout: 15000 });
  await page.waitForFunction(() => window.matterForgeProbe, null, { timeout: 15000 });
  report.probe = await page.evaluate(() => window.matterForgeProbe.runProbe());
  if (report.probe.status !== 'passed') throw Error(report.probe.failure ?? 'Probe gates failed');
  if (report.errors.length || report.requestFailures.length) throw Error('Browser or route errors');
  if (!report.probe.lifecycle.pass) throw Error('Device/simulator lifecycle gate failed');
  for (const [name, digest] of Object.entries(report.sourceHashes)) if (sha(await readFile(path.join(root, name))) !== digest) throw Error(`Captured source changed during run: ${name}`);
  for (const name of allowed) if (report.servedHashes[name] !== report.sourceHashes[name.slice(1)]) throw Error(`Expected source was not executed: ${name}`);
  report.status = 'passed';
} catch (error) {
  report.failure = String(error.stack ?? error); process.exitCode = 1;
  if (!report.probe && page && !page.isClosed()) try { report.partialProbe = await page.evaluate(() => window.matterForgeProbeResult ?? null); } catch {}
} finally {
  clearTimeout(watchdog);
  try { await browser?.close(); } catch (error) { report.cleanupFailure = String(error.message); report.status = 'failed'; process.exitCode = 1; }
  if (server?.listening) await new Promise(resolve => server.close(resolve));
  report.watchdogExpired = expired; report.postGpu = telemetry();
  if (expired) { report.status = 'failed'; process.exitCode = 1; }
  const raw = JSON.stringify(report, null, 2) + '\n';
  await writeFile(path.join(output, 'report.json'), raw, { flag: 'wx' });
  await writeFile(path.join(output, 'report.sha256'), sha(raw) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ output, status: report.status, sha256: sha(raw), failure: report.failure,
    cells: report.probe?.cells.map(({ id, status, errors, momentumError, metadata, minimumSolidDeterminant }) => ({ id, status, errors, momentumError, metadata, minimumSolidDeterminant })),
    lifecycle: report.probe?.lifecycle, errors: report.errors, requestFailures: report.requestFailures, preGpu: report.preGpu, postGpu: report.postGpu }, null, 2));
}
