"""Small, deterministic CPU-only random-feature fit; no framework or GPU training.

The target is a synthetic traveling wave, not a learned physical simulation.
Produces an ONNX sine network and JSON weights for an independent CPU oracle.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import json
import struct
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
rng = np.random.default_rng(73021)
hidden = 32
# Random sine features plus several known frequencies. Only the last layer is fit.
w = rng.uniform(-3, 3, (3, hidden))
w[:, :4] = np.array([[2.4, 0.6, 1], [-0.8, 2, -1], [1.4, -1.6, .5], [.5, .5, 2]]).T
b = rng.uniform(-np.pi, np.pi, hidden)
b[:4] = [0, .4, 1, -.6]
x = rng.uniform([-1, -1, 0], [1, 1, np.pi * 2], (2048, 3))
def target(x):
    return (np.sin(x @ w[:, :4] + b[:4]) @ np.array([.32, .18, .12, .06]))[:, None]
h = np.sin(x @ w + b)
# Fit 32 features plus a bias. One CPU thread and <1 MB of training activations.
a = np.column_stack([h, np.ones(len(h))])
coef = np.linalg.solve(a.T @ a + 1e-7 * np.eye(hidden + 1), a.T @ target(x))
w, b, v, c = [np.asarray(z, dtype=np.float32) for z in (w, b, coef[:-1], coef[-1])]
validation = rng.uniform([-1, -1, 0], [1, 1, np.pi * 2], (512, 3)).astype(np.float32)
error = float(np.max(np.abs(np.sin(validation @ w + b) @ v + c - target(validation))))

# Minimal protobuf encoder for standard ONNX IR 8 / opset 13. No runtime dependency.
def varint(n):
    out = bytearray()
    while n > 127:
        out.append((n & 127) | 128); n >>= 7
    return bytes(out + bytes([n]))
def integer(n, value): return varint(n << 3) + varint(value)
def blob(n, value):
    if isinstance(value, str): value = value.encode()
    return varint((n << 3) | 2) + varint(len(value)) + value
def tensor(name, value):
    return b''.join(integer(1, d) for d in value.shape) + integer(2, 1) + blob(8, name) + blob(9, value.astype('<f4').tobytes())
def info(name, dims):
    shape = b''.join(blob(1, blob(2, d) if isinstance(d, str) else integer(1, d)) for d in dims)
    return blob(1, name) + blob(2, blob(1, integer(1, 1) + blob(2, shape)))
def node(op, inputs, output):
    return b''.join(blob(1, i) for i in inputs) + blob(2, output) + blob(4, op)
nodes = [('MatMul', ['X', 'W'], 'H0'), ('Add', ['H0', 'B'], 'H1'),
         ('Sin', ['H1'], 'H2'), ('MatMul', ['H2', 'V'], 'H3'), ('Add', ['H3', 'C'], 'Y')]
graph = b''.join(blob(1, node(*n)) for n in nodes) + blob(2, 'synthetic-wave-ridge-fit')
graph += b''.join(blob(5, tensor(n, z)) for n, z in [('W', w), ('B', b), ('V', v), ('C', c)])
graph += blob(11, info('X', ['N', 3])) + blob(12, info('Y', ['N', 1]))
model = integer(1, 8) + blob(2, 'three-webgpu-research') + blob(7, graph) + blob(8, integer(2, 13))
(root / 'wave.onnx').write_bytes(model)
metadata = dict(kind='ridge-trained sine features on a synthetic wave', seed=73021,
    hidden=hidden, trainingSamples=2048, validationSamples=512, maxValidationError=error,
    W=w.tolist(), B=b.tolist(), V=v.reshape(-1).tolist(), C=float(c[0]))
(root / 'wave.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(dict(bytes=len(model), maxValidationError=error, parameters=w.size+b.size+v.size+c.size)))
