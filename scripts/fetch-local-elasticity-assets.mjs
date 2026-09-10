import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const pin = 'ac7f3850ceadbc0483d10c9f0b597c2ba5e89009';
const base = `https://raw.githubusercontent.com/marcelpadilla/splats/${pin}/data/`;
const assets = [
  { id: 'spot', file: 'spot/spot_10k.splat', bytes: 320000, sha256: '6eabe6708f0ca8734437340784d15b3da0a5c60640e711e74232babd0ae1c666', license: 'CC0-1.0', attribution: 'Spot by Keenan Crane; mesh-to-splat conversion by Marcel Padilla.', origin: 'mesh2splat' },
  { id: 'plant', file: 'plant/plant.splat', bytes: 3636736, sha256: 'ce9d490b18f16d6730d832227bf541431b8b1a5d167c5a3fe89af4f324c0d8fd', license: 'CC-BY-4.0', attribution: 'Gaussian splat by Marcel Padilla, from https://github.com/marcelpadilla/splats, licensed CC-BY-4.0.', origin: 'capture' },
];
const directory = path.join(root, 'experiments/local-elasticity/assets');
await mkdir(directory, { recursive: true });
for (const asset of assets) {
  const destination = path.join(directory, `${asset.id}.splat`);
  let bytes;
  try { bytes = await readFile(destination); } catch {
    const response = await fetch(base + asset.file, { signal: AbortSignal.timeout(20000) });
    if (!response.ok) throw Error(`Asset request failed: ${response.status}`);
    const declared = Number(response.headers.get('content-length'));
    if (declared && declared !== asset.bytes) throw Error('Unexpected asset size');
    bytes = Buffer.from(await response.arrayBuffer());
  }
  if (bytes.length !== asset.bytes || createHash('sha256').update(bytes).digest('hex') !== asset.sha256) throw Error(`Asset integrity failure: ${asset.id}`);
  await writeFile(destination, bytes);
  console.log(`${asset.id}: ${bytes.length} bytes, verified`);
}
await writeFile(path.join(directory, 'manifest.json'), JSON.stringify({ source: 'https://github.com/marcelpadilla/splats', pin, assets }, null, 2) + '\n');
