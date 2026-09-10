import { createServer } from 'node:http';
import { readFile, realpath, mkdir, writeFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { chromium } from 'playwright-core';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const serve = process.argv.includes('--serve');
if (process.argv.slice(2).some(value => value !== '--serve')) throw Error('Only option: --serve');
const prefix = '/experiments/carry-the-city/';
const allowed = [prefix, '/node_modules/three-r186/'];
const routes = [], servedHashes = {};
const sha = value => createHash('sha256').update(value).digest('hex');
const mimeTypes = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png', '.json': 'application/json' };
const server = createServer(async (req, res) => {
  try {
    if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405).end(); return; }
    let name = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname);
    if (name === '/') name = prefix + 'index.html';
    if (name === '/favicon.ico') { res.writeHead(204).end(); return; }
    const match = allowed.find(candidate => name.startsWith(candidate));
    if (!match || name.includes('\\') || name.includes('\0') || name.split('/').includes('..')) throw Error('Invalid route');
    const base = await realpath(path.join(root, match));
    const file = await realpath(path.join(root, name));
    const relative = path.relative(base, file);
    if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) throw Error('Outside source allowlist');
    const raw = await readFile(file), digest = sha(raw);
    if (!serve && servedHashes[name] && servedHashes[name] !== digest) throw Error('Served source changed');
    servedHashes[name] = digest;
    res.writeHead(200, {
      'Content-Type': mimeTypes[path.extname(name)] ?? 'application/octet-stream',
      'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
    });
    res.end(req.method === 'HEAD' ? undefined : raw);
  } catch (error) {
    routes.push({ url: req.url, error: error.message });
    res.writeHead(404).end();
  }
});
await new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(serve ? 5202 : 0, '127.0.0.1', resolve);
});
const url = `http://127.0.0.1:${server.address().port}/`;

if (serve) {
  console.log(JSON.stringify({ url, state: 'Intro remains idle until the player begins the voyage.' }));
} else {
  const began = Date.now();
  const out = path.join(root, 'results/development/carry-the-city', new Date().toISOString().replaceAll(':', '-'));
  await mkdir(out, { recursive: true });
  const report = {
    status: 'failed', startedAt: new Date(began).toISOString(), url,
    watchdogMs: 75000, viewport: { width: 1440, height: 1000 },
    errors: [], requestFailures: [], routes, checks: [], route: [], actions: [], screenshots: {},
    sourceHashes: {}, sourceSnapshots: {}, servedHashes,
    testMethod: 'Real Begin, pointer steering, mode/restart controls, a bounded real-time RAF sample, collision and rescue routes; public advance/setTarget methods. No writes to game state.',
  };
  let browser, page, timedOut = false;
  const watchdog = setTimeout(() => {
    timedOut = true;
    browser?.close().catch(() => {});
  }, 75000);
  function check(name, passed, detail = null) {
    report.checks.push({ name, passed: Boolean(passed), detail });
    if (!passed) throw Error(`Check failed: ${name}${detail ? ' ' + JSON.stringify(detail) : ''}`);
  }
  async function snapshot(label) {
    const value = await page.evaluate(() => {
      const s = window.carryCity.state;
      return {
        phase: s.phase, playing: s.playing, paused: s.paused, mode: s.mode,
        x: s.x, y: s.y, z: s.z, yaw: s.yaw, speed: s.speed, time: s.time,
        target: s.target ? { ...s.target } : null, elapsed: s.elapsed,
        rescued: s.rescued, lanterns: Array.from(s.lanterns), bumps: s.bumps,
      };
    });
    report.route.push({ label, ...value });
    return value;
  }
  async function screenshot(name) {
    const file = path.join(out, name + '.png');
    await page.screenshot({ path: file, animations: 'disabled' });
    report.screenshots[name] = { path: name + '.png', sha256: sha(await readFile(file)) };
  }
  async function advance(seconds, label) {
    report.actions.push({ type: 'advance', seconds, label });
    await page.evaluate(value => window.carryCity.advance(value, { renderEvery: 120 }), seconds);
    const value = await snapshot(label);
    check(`Finite movement: ${label}`, [value.x, value.y, value.z, value.time, value.elapsed].every(Number.isFinite));
    return value;
  }
  async function travel(x, z, label, { seconds = 18, threshold = .9, setTarget = true } = {}) {
    if (setTarget) {
      report.actions.push({ type: 'setTarget', x, z, label });
      await page.evaluate(([a, b]) => window.carryCity.setTarget(a, b), [x, z]);
    }
    let latest = await snapshot(label + ': target');
    for (let elapsed = 0; elapsed < seconds && Math.hypot(latest.x - x, latest.z - z) > threshold; elapsed += 2) {
      latest = await advance(2, `${label}: ${elapsed + 2}s`);
    }
    check(`Arrived: ${label}`, Math.hypot(latest.x - x, latest.z - z) <= threshold, latest);
    return latest;
  }
  try {
    for (const name of await readdir(path.join(root, prefix))) {
      if (!/\.(?:html|mjs|css)$/.test(name) || name.endsWith('.test.mjs')) continue;
      const key = prefix + name, source = await readFile(path.join(root, key), 'utf8');
      report.sourceSnapshots[key] = source; report.sourceHashes[key] = sha(source);
    }
    const runner = '/scripts/run-carry-the-city.mjs';
    report.sourceSnapshots[runner] = await readFile(fileURLToPath(import.meta.url), 'utf8');
    report.sourceHashes[runner] = sha(report.sourceSnapshots[runner]);
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    report.browser = browser.version();
    page = await browser.newPage({ viewport: report.viewport });
    page.setDefaultTimeout(10000);
    page.on('pageerror', error => report.errors.push({ type: 'pageerror', message: error.message }));
    page.on('console', message => {
      if (message.type() === 'error') report.errors.push({ type: 'console', message: message.text() });
    });
    page.on('requestfailed', request => report.requestFailures.push({ url: request.url(), error: request.failure()?.errorText }));
    await page.goto(url, { waitUntil: 'load', timeout: 30000 });
    await page.waitForFunction(() => window.carryCity?.evidence.ready || window.carryCity?.evidence.errors.length, null, { timeout: 30000 });
    const initialEvidence = await page.evaluate(() => window.carryCity.evidence);
    check('World prepared', initialEvidence.ready && initialEvidence.errors.length === 0, initialEvidence.errors);
    const intro = await snapshot('intro');
    check('Intro is idle', intro.phase === 'intro' && !intro.playing && intro.rescued === 0);
    check('Begin control is available', await page.locator('#begin').isEnabled());
    await screenshot('intro');

    await page.click('#begin'); report.actions.push({ type: 'click', selector: '#begin' });
    // Observe the real animation loop after a short warmup. These are browser
    // RAF intervals and application render counts, not GPU timings or a claim
    // about a target framerate. A timer also bounds the sample if RAF stalls.
    report.realtime = await page.evaluate(async () => {
      await new Promise(resolve => setTimeout(resolve, 250));
      return new Promise(resolve => {
        const begin = performance.now(), framesBefore = window.carryCity.evidence.frames;
        const simulationBefore = window.carryCity.state.time, intervals = [];
        let previous = null, request = 0, ended = false;
        const finish = () => {
          if (ended) return;
          ended = true; cancelAnimationFrame(request); clearTimeout(deadline);
          const sorted = intervals.slice().sort((a, b) => a - b);
          const percentile = p => sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] : null;
          resolve({
            method: 'Independent requestAnimationFrame observer alongside the real game loop; no manual simulation stepping during sample.',
            warmupMs: 250, requestedMs: 2000, observedMs: performance.now() - begin,
            observerCallbacks: intervals.length + (previous === null ? 0 : 1),
            applicationFrames: window.carryCity.evidence.frames - framesBefore,
            simulationSeconds: window.carryCity.state.time - simulationBefore,
            intervalsMs: intervals,
            meanIntervalMs: intervals.length ? intervals.reduce((a, b) => a + b, 0) / intervals.length : null,
            medianIntervalMs: percentile(.5), p95IntervalMs: percentile(.95), maxIntervalMs: percentile(1),
          });
        };
        const sample = now => {
          if (previous !== null) intervals.push(now - previous);
          previous = now;
          if (performance.now() - begin >= 2000) finish();
          else request = requestAnimationFrame(sample);
        };
        const deadline = setTimeout(finish, 2250);
        request = requestAnimationFrame(sample);
      });
    });
    report.actions.push({ type: 'observe-realtime-RAF', requestedMs: 2000, warmupMs: 250 });
    check('Real animation loop renders and advances', report.realtime.applicationFrames >= 2 && report.realtime.simulationSeconds > 0
      && report.realtime.intervalsMs.length > 0 && report.realtime.intervalsMs.every(value => Number.isFinite(value) && value > 0), report.realtime);
    await page.evaluate(() => window.carryCity.advance(0));
    const started = await snapshot('after actual Begin');
    check('Begin starts the voyage', started.playing && started.phase === 'voyage' && started.mode === 'swim');
    const pointer = await page.evaluate(() => window.carryCity.project(-9, 0, 6));
    check('Water pointer target is onscreen', pointer.x > 0 && pointer.x < 1440 && pointer.y > 0 && pointer.y < 1000, pointer);
    await page.mouse.move(pointer.x - 8, pointer.y + 5);
    await page.mouse.down();
    await page.mouse.move(pointer.x, pointer.y, { steps: 3 });
    await page.mouse.up();
    report.actions.push({ type: 'pointer-drag', projectedWorldPoint: [-9, 0, 6], client: pointer });
    const steered = await snapshot('after actual water pointer');
    check('Water pointer steers to its world point', steered.target && Math.hypot(steered.target.x + 9, steered.target.z - 6) < .1, steered.target);
    const first = await travel(-9, 6, 'first lantern', { setTarget: false });
    check('Swimming caused travel', Math.hypot(first.x - started.x, first.z - started.z) > 5);
    await screenshot('swim');

    await page.click('#fly'); report.actions.push({ type: 'click', selector: '#fly' });
    check('Moth button changes mode and UI', await page.evaluate(() => window.carryCity.state.mode === 'fly' && document.getElementById('fly').classList.contains('active')));
    const second = await travel(8, -5, 'second lantern');
    check('Flight clears the water', second.y > 4 && second.mode === 'fly', { y: second.y, mode: second.mode });
    // Select a visible wingbeat phase by allowing the simulation to advance.
    const wingTime = await page.evaluate(() => ((.5 - window.carryCity.state.time % 2) + 2) % 2);
    if (wingTime > .001) await advance(wingTime, 'wingbeat screenshot phase');
    await screenshot('fly');
    await travel(0, -17.2, 'lighthouse approach');
    await page.click('#swim'); report.actions.push({ type: 'click', selector: '#swim' });
    check('Koi button changes mode and UI', await page.evaluate(() => window.carryCity.state.mode === 'swim' && document.getElementById('swim').classList.contains('active')));
    let finished;
    for (let i = 0; i < 9; i++) {
      finished = await advance(2, `landing and rescue ${i + 1}`);
      if (finished.phase === 'done') break;
    }
    check('Reached the destination by movement', finished.z < -15 && Math.abs(finished.x) < 2.2, finished);
    check('All eight residents rescued', finished.phase === 'done' && finished.rescued === 8, finished);
    check('Finish screen is visible', await page.locator('#finish').isVisible());
    check('Route collected all three lanterns', finished.lanterns.length === 3, finished.lanterns);
    await screenshot('finish');
    report.evidence = await page.evaluate(() => window.carryCity.evidence);
    check('Start, both modes, docking and win were observed', ['start', 'docking', 'win'].every(type => report.evidence.events.some(e => e.type === type))
      && ['swim', 'fly'].every(kind => report.evidence.events.some(e => e.type === 'motion-borrowed' && e.kind === kind)));
    check('No game or browser errors', report.evidence.errors.length === 0 && report.errors.length === 0 && routes.length === 0,
      { game: report.evidence.errors, browser: report.errors, routes });

    // Restart after the completed rescue, then take a deliberately different
    // route into a reef. The event must arise from movement and contact.
    const priorRestarts = report.evidence.events.filter(e => e.type === 'restart').length;
    await page.click('#restart'); report.actions.push({ type: 'click', selector: '#restart' });
    const restarted = await snapshot('after actual restart');
    check('Restart resets the voyage', restarted.phase === 'intro' && !restarted.playing && restarted.mode === 'swim'
      && restarted.rescued === 0 && restarted.bumps === 0 && restarted.lanterns.length === 0 && restarted.target === null
      && restarted.time === 0 && restarted.elapsed === 0 && Math.hypot(restarted.x, restarted.z - 7) < 1e-9, restarted);
    check('Restart restores intro and hides the finish', await page.locator('#intro').isVisible() && !await page.locator('#finish').isVisible());
    check('Restart restores all resident HUD markers', await page.locator('#people').textContent() === '● ● ● ● ● ● ● ●');
    check('One actual restart event was recorded', await page.evaluate(count => window.carryCity.evidence.events.filter(e => e.type === 'restart').length === count + 1, priorRestarts));
    await page.click('#begin'); report.actions.push({ type: 'click', selector: '#begin', label: 'collision voyage' });
    await page.evaluate(() => window.carryCity.advance(0));
    const collisionEventStart = await page.evaluate(() => window.carryCity.evidence.events.length);
    await page.evaluate(() => window.carryCity.setTarget(-5, 0));
    report.actions.push({ type: 'setTarget', x: -5, z: 0, label: 'real reef contact' });
    let collisionState;
    for (let i = 0; i < 5; i++) {
      collisionState = await advance(1, `reef approach ${i + 1}`);
      if (collisionState.bumps > 0) break;
    }
    report.collision = {
      state: collisionState,
      events: await page.evaluate(start => window.carryCity.evidence.events.slice(start), collisionEventStart),
    };
    const impact = report.collision.events.find(e => e.type === 'reef-impact');
    const release = report.collision.events.find(e => e.type === 'cargo-released');
    check('Movement into the reef triggers contact', collisionState.mode === 'swim' && collisionState.bumps > 0 && Boolean(impact), report.collision);
    check('Reef contact releases cargo', Boolean(release) && release.time >= impact.time, release ?? null);
    const finiteVector = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
    check('Released cargo has finite position and velocity', finiteVector(release.position) && finiteVector(release.velocity)
      && finiteVector(release.inheritedVelocity), release);
    check('Cargo inherits a tracked simulation frame', release.hasVelocity === true && release.reason === 'tracked'
      && release.deltaTime > 0 && release.deltaTime <= .25, release);
    check('Cargo adds the explicit kick exactly once', release.velocity.every((value, i) => Math.abs(value - release.inheritedVelocity[i] - (i === 1 ? 1.8 : 0)) < 1e-9), release);
    await screenshot('collision');
    report.secondaryEvidence = await page.evaluate(() => window.carryCity.evidence);
    check('Restart and collision remain free of errors', report.secondaryEvidence.errors.length === 0 && report.errors.length === 0 && routes.length === 0,
      { game: report.secondaryEvidence.errors, browser: report.errors, routes });

    await page.evaluate(() => window.carryCity.dispose());
    report.disposal = await page.evaluate(async () => {
      const framesBefore = window.carryCity.evidence.frames;
      // The second call must be harmless. Wait one event-loop interval for an
      // intentional context-loss notification or stray animation callbacks.
      window.carryCity.dispose();
      await new Promise(resolve => setTimeout(resolve, 150));
      return { ...window.carryCity.evidence, framesAfterDisposal: window.carryCity.evidence.frames - framesBefore };
    });
    check('Explicit disposal completed', report.disposal.disposed);
    check('Disposal is idempotent and stops rendering', report.disposal.framesAfterDisposal === 0);
    check('Disposal did not produce game errors', report.disposal.errors.length === 0 && report.errors.length === 0,
      { game: report.disposal.errors, browser: report.errors });
    for (const [name, digest] of Object.entries(servedHashes)) {
      if (report.sourceHashes[name] && report.sourceHashes[name] !== digest) throw Error('Served source differs from snapshot: ' + name);
    }
    for (const group of [report.sourceHashes, servedHashes]) for (const [name, digest] of Object.entries(group)) {
      if (sha(await readFile(path.join(root, name))) !== digest) throw Error('Executed source changed: ' + name);
    }
    check('Executed source hashes remained stable', true);
    report.status = 'passed';
  } catch (error) {
    report.failure = String(error.stack ?? error);
    process.exitCode = 1;
  } finally {
    if (page && !page.isClosed()) {
      try { await page.evaluate(() => window.carryCity?.dispose()); } catch (error) { report.cleanupError = error.message; }
    }
    await browser?.close().catch(error => { report.closeError = error.message; });
    clearTimeout(watchdog);
    report.timedOut = timedOut;
    report.wallMs = Date.now() - began;
    if (timedOut) { report.status = 'failed'; report.failure = '75-second browser watchdog expired. ' + (report.failure ?? ''); process.exitCode = 1; }
    await new Promise(resolve => server.close(resolve));
    const raw = JSON.stringify(report, null, 2) + '\n';
    await writeFile(path.join(out, 'report.json'), raw);
    await writeFile(path.join(out, 'report.sha256'), sha(raw) + '\n');
    console.log(JSON.stringify({ out, status: report.status, failure: report.failure, wallMs: report.wallMs, checks: report.checks, errors: report.errors }, null, 2));
  }
}
