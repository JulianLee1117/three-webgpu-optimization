import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, realpath, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2), serve = args.includes('--serve'), interaction = args.includes('--interaction'), asset = args.find(x => x.startsWith('--asset='))?.slice(8) ?? 'spot';
if (args.some(x => !['--serve', '--interaction', '--asset=spot', '--asset=plant'].includes(x))) throw Error('Options: --serve, --interaction, --asset=spot, --asset=plant');
const prefix = '/experiments/local-elasticity/', prefixes = [prefix, '/node_modules/three-r186/src/', '/node_modules/three-r186/examples/jsm/'];
const failures = [], hashes = {}, sha = bytes => createHash('sha256').update(bytes).digest('hex');
const server = createServer(async (req, res) => {
  try {
    if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405).end(); return; }
    let name = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname);
    if (name === '/') name = prefix + 'app.html';
    if (name === '/README.md') name = prefix + 'README.md';
    if (name === '/favicon.ico') { res.writeHead(204).end(); return; }
    const allowed = prefixes.find(x => name.startsWith(x));
    if (!allowed || name.includes('\\') || name.includes('\0') || name.split('/').includes('..')) throw Error('Unsupported route');
    const file = await realpath(path.join(root, name)), base = await realpath(path.join(root, allowed)), relative = path.relative(base, file);
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) throw Error('Path containment failed');
    const bytes = await readFile(file); hashes[name] = sha(bytes);
    const mime = name.endsWith('.html') ? 'text/html' : name.endsWith('.json') ? 'application/json' : name.endsWith('.md') ? 'text/plain' : name.endsWith('.splat') ? 'application/octet-stream' : name.endsWith('.png') ? 'image/png' : 'text/javascript';
    res.writeHead(200, { 'Content-Type': mime, 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' }); res.end(req.method === 'HEAD' ? undefined : bytes);
  } catch (error) { failures.push({ url: req.url, message: String(error) }); res.writeHead(404).end('Not found'); }
});
await new Promise((resolve, reject) => { server.once('error', reject); server.listen(serve ? 5199 : 0, '127.0.0.1', resolve); });
const url = `http://127.0.0.1:${server.address().port}/`;
if (serve) console.log(JSON.stringify({ url, mode: 'idle until Prepare is clicked; runs capped at 12 seconds' }));
else {
  const out = path.join(root, 'results/development/local-elasticity', new Date().toISOString().replaceAll(':', '-') + '-' + asset + '-ui');
  await mkdir(out, { recursive: true });
  const report = { status: 'failed', asset, errors: [], routes: failures, hashes, sources: {} };
  for (const name of (await readdir(path.join(root, prefix))).filter(x => /\.(js|mjs|html|py)$/.test(x))) report.sources[name] = sha(await readFile(path.join(root, prefix, name)));
  let browser;
  const watchdog = setTimeout(() => browser?.close().catch(() => {}), 60000);
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true }); report.browser = browser.version();
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    page.on('pageerror', error => report.errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') report.errors.push(message.text()); });
    await page.goto(url, { timeout: 15000 }); await page.selectOption('#asset', asset); await page.click('#prepare');
    await page.waitForFunction(() => window.elasticityDemo?.evidence.prepared || window.elasticityDemo?.evidence.errors.length, null, { timeout: 40000 });
    report.preparation = await page.evaluate(() => window.elasticityDemo.evidence);
    if (!report.preparation.prepared) throw Error(JSON.stringify(report.preparation.errors));
    await page.screenshot({ path: path.join(out, 'rest.png') });
    await page.click('#play');
    await page.waitForFunction(() => window.elasticityDemo.evidence.steps.length >= 4 || window.elasticityDemo.evidence.errors.length, null, { timeout: 10000 });
    await page.evaluate(() => window.elasticityDemo.stop());
    await page.waitForFunction(() => !window.elasticityDemo.getState().pending);
    report.state = await page.evaluate(() => ({ ...window.elasticityDemo.getState(), evidence: window.elasticityDemo.evidence }));
    if (report.state.evidence.errors.length || report.errors.length || failures.length) throw Error('Browser/worker errors');
    if (report.state.evidence.steps.length < 4 || !report.state.q.some(x => Math.abs(x) > 1e-8)) throw Error('No mechanical state evolution');
    await page.screenshot({ path: path.join(out, 'deformed.png') });
    if (interaction) {
      const before = report.state.evidence.steps.length;
      await page.mouse.move(asset === 'plant' ? 490 : 550, asset === 'plant' ? 400 : 390); await page.mouse.down();
      await page.waitForFunction(() => !!window.elasticityDemo.evidence.lastGrab, null, { timeout: 2000 });
      await page.mouse.move(asset === 'plant' ? 655 : 715, asset === 'plant' ? 360 : 350, { steps: 8 });
      await page.waitForFunction(n => window.elasticityDemo.evidence.steps.length >= n + 24 || window.elasticityDemo.evidence.errors.length, before, { timeout: 10000 });
      await page.evaluate(() => window.elasticityDemo.stop()); await page.waitForFunction(() => !window.elasticityDemo.getState().pending);
      report.pulled = await page.evaluate(() => ({ state: window.elasticityDemo.getState(), grab: window.elasticityDemo.evidence.lastGrab, evidence: window.elasticityDemo.evidence }));
      await page.screenshot({ path: path.join(out, 'pulled.png') }); await page.mouse.up();
      const releaseStart = report.pulled.evidence.steps.length; await page.evaluate(() => window.elasticityDemo.start());
      await page.waitForFunction(n => window.elasticityDemo.evidence.steps.length >= n + 12 || window.elasticityDemo.evidence.errors.length, releaseStart, { timeout: 10000 });
      await page.evaluate(() => window.elasticityDemo.stop()); await page.waitForFunction(() => !window.elasticityDemo.getState().pending);
      report.released = await page.evaluate(() => ({ state: window.elasticityDemo.getState(), evidence: window.elasticityDemo.evidence }));
      await page.screenshot({ path: path.join(out, 'released.png') });
      if (report.released.evidence.errors.length || report.errors.length) throw Error('Interaction produced errors');
      if (Math.max(...report.released.state.q.map((x, i) => Math.abs(x - report.pulled.state.q[i]))) < 1e-5) throw Error('Release did not change state');
    }
    await page.click('#reset'); await page.waitForFunction(() => window.elasticityDemo.getState().q.every(x => x === 0));
    report.reset = true; report.status = 'passed';
  } catch (error) { report.failure = String(error.stack ?? error); process.exitCode = 1; }
  finally {
    clearTimeout(watchdog); await browser?.close(); await new Promise(resolve => server.close(resolve));
    const bytes = JSON.stringify(report, (_, value) => ArrayBuffer.isView(value) ? Array.from(value) : value, 2) + '\n';
    await writeFile(path.join(out, 'report.json'), bytes); await writeFile(path.join(out, 'report.sha256'), sha(bytes) + '\n');
    console.log(JSON.stringify({ out, status: report.status, failure: report.failure, errors: report.errors, preparation: report.preparation?.preparation, steps: report.state?.evidence.steps.map(x => ({ status: x.status, solveMs: x.solveMs, pinError: x.pinError, minDeterminant: x.minDeterminant })) }, null, 2));
  }
}
