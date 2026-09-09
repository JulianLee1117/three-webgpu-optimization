import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const script = fileURLToPath(import.meta.url);
const root = path.resolve(path.dirname(script), '..');
const reportRoot = path.join(root, 'results/development/ort-handoff');
const lanes = ['borrowed', 'internal', 'gpu-copy', 'cpu-copy'];
const comparators = lanes.slice(1);
const bootstrapReplicates = 10000;
const bootstrapSeed = 731;
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const finiteNonnegative = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;

export function median(values) {
  assert(values.length > 0, 'Cannot calculate an empty median.');
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function nearestRank(values, probability) {
  assert(values.length > 0 && probability > 0 && probability <= 1, 'Invalid nearest-rank quantile.');
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.ceil(probability * sorted.length) - 1];
}

export function verifyDigest(bytes, expected) {
  assert(/^[a-f0-9]{64}$/i.test(expected.trim()), 'Malformed report SHA256 sidecar.');
  const digest = hash(bytes);
  assert(digest === expected.trim().toLowerCase(), 'Report SHA256 mismatch.');
  return digest;
}

function permutations(values) {
  return values.length ? values.flatMap((value, index) =>
    permutations(values.filter((_, other) => other !== index)).map(rest => [value, ...rest])) : [[]];
}

function expectedOrders() {
  const orders = permutations(lanes);
  let seed = 731;
  for (let index = orders.length - 1; index > 0; index--) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const other = seed % (index + 1);
    [orders[index], orders[other]] = [orders[other], orders[index]];
  }
  return orders;
}

export function validateReport(report) {
  assert(report.kind === 'ort-handoff-screen-v1', 'Unexpected report kind.');
  assert(report.status === 'passed' && report.stock === false && !report.failure,
    'Analysis requires a passed four-lane report.');
  assert([128, 256].includes(report.side), 'Unexpected grid size.');
  assert(Array.isArray(report.errors) && report.errors.length === 0, 'Browser errors were reported.');
  assert(report.finalState?.disposed === true && report.finalState.phase === 'disposed', 'Cleanup did not complete.');
  assert(Array.isArray(report.finalState.errors) && report.finalState.errors.length === 0, 'GPU errors were reported.');
  const environment = report.environment;
  assert(environment?.stock === false && environment.side === report.side && environment.n === report.side ** 2,
    'Environment dimensions or build mode do not match the report.');
  for (const key of ['internalBufferOwnedByThree', 'inputBufferSame', 'internalOutputBufferSame']) {
    assert(environment[key] === true, `Buffer identity/ownership gate failed: ${key}.`);
  }
  for (const value of [report.browser, environment.threeRevision, environment.ortVersion]) {
    assert(typeof value === 'string' && value.length > 0, 'Missing runtime version.');
  }
  assert(report.sources && Object.keys(report.sources).length > 0 &&
    Object.values(report.sources).every(value => /^[a-f0-9]{64}$/i.test(value)), 'Missing or invalid source hashes.');

  assert(Array.isArray(report.proof) && report.proof.length === 8, 'Expected eight correctness records.');
  const proofKeys = new Set();
  const contentByTime = new Map();
  for (const proof of report.proof) {
    assert(lanes.includes(proof.lane) && [.375, 2.25].includes(proof.t), 'Unexpected correctness lane or time.');
    const key = `${proof.lane}:${proof.t}`;
    assert(!proofKeys.has(key), `Duplicate correctness record: ${key}.`);
    proofKeys.add(key);
    assert(finiteNonnegative(proof.maxError) && proof.maxError < 2e-5, `CPU numerical oracle failed: ${key}.`);
    assert(proof.unequalBytes === 0, `Image parity failed: ${key}.`);
    assert(Number.isInteger(proof.nonBackgroundPixels) && proof.nonBackgroundPixels > 1000 &&
      proof.nonBackgroundPixels <= 384 * 256, `Nonempty-image gate failed: ${key}.`);
    if (contentByTime.has(proof.t)) {
      assert(contentByTime.get(proof.t) === proof.nonBackgroundPixels, 'Image parity and content counts disagree.');
    } else contentByTime.set(proof.t, proof.nonBackgroundPixels);
  }

  const warmup = report.timing?.warmup;
  assert(Array.isArray(warmup) && warmup.length === 32, 'Expected eight warmups per lane.');
  warmup.forEach((sample, index) => {
    assert(sample.lane === lanes[index % 4] && finiteNonnegative(sample.ms), 'Invalid warmup lane/order/timing.');
  });
  const samples = report.timing.samples;
  assert(Array.isArray(samples) && samples.length === 96, 'Expected 24 complete four-lane blocks.');
  const orders = expectedOrders();
  const blocks = Array.from({ length: 24 }, (_, block) => ({ block, order: [], latencyMs: {} }));
  samples.forEach((sample, index) => {
    const block = Math.floor(index / 4), position = index % 4;
    assert(sample.block === block && sample.position === position, 'Samples are missing, duplicated, or out of chronological block order.');
    assert(sample.lane === orders[block][position], 'Lane permutation differs from the prespecified seed-731 schedule.');
    assert(finiteNonnegative(sample.ms), 'Invalid measured latency.');
    assert(!Object.hasOwn(blocks[block].latencyMs, sample.lane), 'Duplicate lane in a block.');
    blocks[block].order.push(sample.lane);
    blocks[block].latencyMs[sample.lane] = sample.ms;
  });
  assert(new Set(blocks.map(block => block.order.join('|'))).size === 24, 'Expected all 24 distinct lane permutations.');
  for (const lane of lanes) for (let position = 0; position < 4; position++) {
    assert(samples.filter(sample => sample.lane === lane && sample.position === position).length === 6,
      'Lane/position balance failed.');
  }
  return blocks;
}

export function summarizeReport(report) {
  const blocks = validateReport(report);
  const pairedBlocks = blocks.map(block => ({
    ...block,
    comparatorMinusBorrowedMs: Object.fromEntries(comparators.map(lane =>
      [lane, block.latencyMs[lane] - block.latencyMs.borrowed]))
  }));
  const bootstrap = Object.fromEntries(comparators.map(lane => [lane, []]));
  let seed = bootstrapSeed;
  // Every replicate uses one shared set of block indices for all three contrasts.
  // This retains all four lanes in a sampled block, including their pairing.
  for (let replicate = 0; replicate < bootstrapReplicates; replicate++) {
    const indices = Array.from({ length: blocks.length }, () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return Math.floor(seed / 0x100000000 * blocks.length);
    });
    for (const lane of comparators) {
      bootstrap[lane].push(median(indices.map(index => pairedBlocks[index].comparatorMinusBorrowedMs[lane])));
    }
  }
  return {
    side: report.side,
    environment: report.environment,
    browser: report.browser,
    sourceHashes: report.sources,
    verification: {
      passedStatus: true, correctnessRecords: report.proof.length, completedCleanup: true,
      warmupsPerLane: 8, completeBlocks: blocks.length, samplesPerLane: 24,
      distinctPermutations: 24, samplesPerLanePosition: 6, scheduleSeed: 731
    },
    lanes: Object.fromEntries(lanes.map(lane => {
      const values = blocks.map(block => block.latencyMs[lane]);
      return [lane, { samples: values.length, medianMs: median(values), p95Ms: nearestRank(values, .95) }];
    })),
    pairedComparisons: Object.fromEntries(comparators.map(lane => {
      const differences = pairedBlocks.map(block => block.comparatorMinusBorrowedMs[lane]);
      return [lane, {
        comparator: lane, reference: 'borrowed', blocks: differences.length,
        medianDifferenceMs: median(differences),
        bootstrap95PercentileIntervalMs: [nearestRank(bootstrap[lane], .025), nearestRank(bootstrap[lane], .975)]
      }];
    })),
    pairedBlocks
  };
}

async function readVerifiedReport(filename) {
  const resolved = await realpath(filename);
  const resolvedRoot = await realpath(reportRoot);
  const relative = path.relative(resolvedRoot, resolved);
  assert(relative !== '' && !relative.startsWith('..' + path.sep) && !path.isAbsolute(relative) &&
    path.basename(resolved) === 'report.json', 'Input must be an immutable report.json under results/development/ort-handoff.');
  const bytes = await readFile(resolved);
  const digest = verifyDigest(bytes, await readFile(resolved + '.sha256', 'utf8'));
  return { path: resolved, sha256: digest, report: JSON.parse(bytes.toString('utf8')) };
}

async function main() {
  const explicit = process.argv.slice(2);
  if (explicit.includes('--help')) {
    console.log('Usage: node scripts/analyze-ort-handoff-screen.mjs [path/to/report.json ...]\nDefault: all passed four-lane reports under results/development/ort-handoff.');
    return;
  }
  const candidates = explicit.length ? explicit.map(filename => path.resolve(filename)) :
    (await readdir(reportRoot, { withFileTypes: true }))
      .filter(entry => entry.isDirectory() && /^\d{4}-\d{2}-\d{2}T/.test(entry.name))
      .map(entry => path.join(reportRoot, entry.name, 'report.json')).sort();
  assert(candidates.length > 0, 'No handoff reports found.');
  const verified = await Promise.all(candidates.map(readVerifiedReport));
  const selected = explicit.length ? verified : verified.filter(input => input.report.status === 'passed' && input.report.stock === false);
  assert(selected.length > 0, 'No passed four-lane reports found.');
  assert(new Set(selected.map(input => input.path)).size === selected.length, 'Duplicate input report.');
  const summary = {
    kind: 'ort-handoff-paired-analysis-v1', created: new Date().toISOString(),
    analyzerSha256: hash(await readFile(script)),
    endpoint: 'CPU-observed serialized update completion latency, including inference, rendering, transfers, scheduling, and synchronization.',
    units: 'milliseconds',
    differenceSign: 'Positive comparator-minus-borrowed values favor borrowed; negative values favor the comparator.',
    method: {
      laneMedian: 'Conventional sample median: mean of the two middle observations for even sample counts.',
      p95: 'Nearest rank: sorted observation ceil(0.95 * n). At n=24 this is the second-largest observation.',
      pairedStatistic: 'Median of within-block comparator-minus-borrowed differences; not a difference between marginal medians.',
      bootstrap: { unit: 'complete four-lane block', replicates: bootstrapReplicates, seed: bootstrapSeed,
        randomGenerator: 'LCG: uint32(1664525 * state + 1013904223); index=floor(state/2^32 * 24)',
        interval: 'Separate unadjusted 95% percentile intervals using nearest-rank 2.5% and 97.5% endpoints.' }
    },
    limitations: [
      'Each report is a single-session diagnostic. Sizes and reports are analyzed separately; they are not pooled.',
      'Bootstrap intervals describe within-session block resampling under an exchangeability assumption. They do not capture cross-session/device uncertainty or residual serial dependence.',
      'An interval crossing zero is inconclusive and is not proof of equivalence. No equivalence margin was prespecified.',
      'These are CPU-observed serialized latencies, not GPU execution times, steady-state throughput, or display latency.',
      'Report SHA256 checks validate file integrity; recorded source hashes are preserved, not asserted to match the current mutable workspace.',
      'Correctness records verify the recorded scalar/image and buffer-identity gates; they are not a native operation trace.'
    ],
    reports: selected.map(input => ({ sourceReport: input.path, sourceReportSha256: input.sha256, ...summarizeReport(input.report) }))
  };
  const out = path.join(root, 'results/development/ort-handoff-analysis', summary.created.replaceAll(':', '-'));
  await mkdir(out, { recursive: true });
  const bytes = JSON.stringify(summary, null, 2) + '\n';
  const outputFile = path.join(out, 'summary.json');
  await writeFile(outputFile, bytes, { flag: 'wx' });
  await writeFile(outputFile + '.sha256', hash(bytes) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ outputFile, reports: summary.reports.map(report => ({
    side: report.side, sourceReport: report.sourceReport, lanes: report.lanes, pairedComparisons: report.pairedComparisons
  })) }, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === script) {
  main().catch(error => { console.error(error.stack ?? error); process.exitCode = 1; });
}
