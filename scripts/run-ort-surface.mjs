import { createServer } from 'node:http';
import { chromium } from 'playwright-core';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const serve = process.argv.includes('--serve');
const animate = process.argv.includes('--animate');
const record = process.argv.includes('--record');
const sizeArg = process.argv.find(a => a.startsWith('--size='));
const side = sizeArg ? Number(sizeArg.split('=')[1]) : 128;
if (![64,128,256].includes(side)) throw new Error('Supported grid widths: 64, 128, 256.');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream' };
const server = createServer(async (req, res) => {
  const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  if (!['/experiments/ort-surface/', '/.generated/ort-surface/', '/node_modules/onnxruntime-web-dev/dist/'].some(prefix => pathname.startsWith(prefix))) { res.writeHead(404).end(); return; }
  const file = path.resolve(root, '.' + pathname + (pathname.endsWith('/') ? 'index.html' : ''));
  if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
  try {
    const bytes = await readFile(file);
    res.writeHead(200, { 'Content-Type': types[path.extname(file)] ?? 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp', 'Cache-Control': 'no-store' }); res.end(bytes);
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(serve ? 5186 : 0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}/experiments/ort-surface/`;
if (serve) { console.log(url); }
else {
  const out = path.join(root, 'results/development/ort-surface', new Date().toISOString().replaceAll(':','-') + `-${side}`);
  await mkdir(out, { recursive: true });
  const report = { kind: 'ort-surface-development-run-v1', started: new Date().toISOString(), side,
    claim: 'Correctness and observed buffer handoff operations, not a performance benchmark.',
    browserErrors: [], status: 'failed', manifest: JSON.parse(await readFile(path.join(root, '.generated/ort-surface/manifest.json'))) };
  let browser, page;
  const deadline = setTimeout(() => { browser?.close().catch(() => {}); }, 90000);
  try {
    browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: [] });
    report.browser = browser.version();
    page = await browser.newPage({ viewport: { width: 1280, height: 880 }, deviceScaleFactor: 1 });
    page.on('pageerror', e => report.browserErrors.push(e.message));
    page.on('console', msg => { if (msg.type() === 'error') { report.browserErrors.push(msg.text()); console.log('Browser: ' + msg.text()); } });
    page.on('requestfailed', req => report.browserErrors.push(`${req.url()}: ${req.failure()?.errorText}`));
    console.log('Opening ' + url);
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 20000 });
    await page.waitForFunction(() => !!window.ortSurface, null, { timeout: 15000 });
    console.log('Initializing bounded GPU proof');
    report.environment = await page.evaluate(side => window.ortSurface.initialize(side), side);
    report.proof = await page.evaluate(() => window.ortSurface.validate());
    if (record) {
      console.log('Recording eight seconds of real demo animation');
      const video = await page.evaluate(() => window.ortSurface.recordAnimation());
      await writeFile(path.join(out, 'surface.webm'), Buffer.from(video.base64, 'base64'), { flag: 'wx' });
      report.animation = video.animation;
    } else if (animate) {
      console.log('Checking the eight-second animation limit');
      report.animation = await page.evaluate(() => window.ortSurface.animate());
      if (!(report.animation.frames > 1 && report.animation.frames <= 120 && report.animation.stopped)) throw new Error('Animation did not satisfy its frame cap.');
    }
    await page.screenshot({ path: path.join(out, 'surface.png') });
    report.cleanup = await page.evaluate(() => window.ortSurface.teardown());
    report.finalGpuErrors = await page.evaluate(() => window.ortSurface.errors);
    if (report.finalGpuErrors.length) throw new Error('GPU errors after animation or cleanup.');
    if (report.browserErrors.length) throw new Error('Browser errors: ' + report.browserErrors.join('; '));
    report.status = 'passed';
  } catch (error) {
    report.failure = String(error.stack ?? error);
    if (page) {
      report.pageState = await page.evaluate(() => ({ status: document.getElementById('status')?.textContent, events: window.ortSurface?.events, errors: window.ortSurface?.errors })).catch(() => null);
      report.cleanupAfterFailure = await page.evaluate(() => window.ortSurface?.teardown()).catch(e => ({ error: e.message }));
    }
  } finally {
    clearTimeout(deadline);
    await browser?.close(); await new Promise(resolve => server.close(resolve));
    report.finished = new Date().toISOString();
    const bytes = JSON.stringify(report, null, 2) + '\n';
    await writeFile(path.join(out, 'report.json'), bytes, { flag: 'wx' });
    await writeFile(path.join(out, 'report.json.sha256'), createHash('sha256').update(bytes).digest('hex') + '\n', { flag: 'wx' });
    console.log(JSON.stringify({ status: report.status, out, failure: report.failure, proof: report.proof?.samples.map(s => ({ time: s.time, error: s.maxAbsoluteError, unequalImageBytes: s.unequalImageBytes })) }, null, 2));
    if (report.status !== 'passed') process.exitCode = 1;
  }
}
