import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

import {
  THREE_IMMEDIATE_OVERLAY_TARGET_DEV,
  runThreeImmediateOverlayProbe,
} from './probe-three-immediate-overlay.mjs';

function parseArguments(argv) {
  const options = {
    outputPath: null,
    browserPath: process.env.BROWSER_PATH ?? null,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--output') {
      const value = argv[++index];
      if (value === undefined) throw new Error('--output requires a path.');
      options.outputPath = path.resolve(value);
    } else if (argument === '--browser') {
      const value = argv[++index];
      if (value === undefined) throw new Error('--browser requires a path.');
      options.browserPath = path.resolve(value);
    } else {
      throw new Error(`Unknown argument: ${argument}`);
    }
  }
  return options;
}

export function runThreeImmediateDevOverlayProbe(options = {}) {
  return runThreeImmediateOverlayProbe({
    ...options,
    target: THREE_IMMEDIATE_OVERLAY_TARGET_DEV,
  });
}

function isDirectExecution() {
  return process.argv[1] !== undefined
    && fileURLToPath(import.meta.url) === path.resolve(process.argv[1]);
}

if (isDirectExecution()) {
  const options = parseArguments(process.argv.slice(2));
  runThreeImmediateDevOverlayProbe(options).then(({ outputPath, envelope }) => {
    process.stdout.write(`${JSON.stringify({
      status: envelope.status,
      target: envelope.target,
      outputPath,
      pageStatus: envelope.pageResult?.status ?? null,
      servedSourceExact: envelope.servedSource?.exact ?? false,
      fullPhaseZeroStatus: envelope.fullPhaseZeroStatus,
      failure: envelope.failure,
    }, null, 2)}\n`);
    if (envelope.status !== 'development-checks-complete') process.exitCode = 1;
  }).catch((error) => {
    process.stderr.write(`${error.stack ?? error.message}\n`);
    process.exitCode = 1;
  });
}
