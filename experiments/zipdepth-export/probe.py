"""CPU-only reproduction of ZipDepth's export fidelity discrepancy.

Uses the released Base NPU checkpoint and three deterministic procedural images.
This measures numerical fidelity to the checkpoint, not ground-truth depth quality.
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import copy
import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".local-research/zipdepth-export-source"
OUT = ROOT / "results/development/zipdepth-export" / datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(":", "-")
OUT.mkdir(parents=True, exist_ok=False)
parser = argparse.ArgumentParser()
parser.add_argument("--height", type=int, choices=(384, 512), default=384)
parser.add_argument("--width", type=int, choices=(384, 512), default=384)
args = parser.parse_args()
HEIGHT, WIDTH = args.height, args.width
sys.path.insert(0, str(SOURCE))
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.manual_seed(314159)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def errors(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite or incompatible output")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "maxAbs": float(np.abs(delta).max()),
        "meanAbs": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
        "relativeL1": float(np.abs(delta).sum() / max(np.abs(expected).sum(), 1e-30)),
        "allclose_rtol1e-4_atol1e-5": bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-5)),
    }


def main():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    for name, item in manifest["files"].items():
        if digest(SOURCE / name) != item["sha256"]:
            raise ValueError(f"Source checksum changed: {name}")
    from zipdepth.model.architecture import create_model
    from zipdepth.utils.model_utils import strip_state_dict_prefixes, fuse_remaining_conv_bn

    try:
        onnxsim_version = importlib.metadata.version("onnxsim")
    except importlib.metadata.PackageNotFoundError:
        onnxsim_version = None
    report["optionalDependencies"] = {"onnxsim": {"available": importlib.util.find_spec("onnxsim") is not None, "version": onnxsim_version}}
    report.update(source=manifest, versions={"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__, "python": sys.version}, threads={"torch": 2, "interop": 1, "ort": 2}, cudaBuild=torch.version.cuda, probeSha256=digest(Path(__file__)))
    ckpt = torch.load(SOURCE / "checkpoints/zipdepth_base_npu.pth", map_location="cpu", weights_only=True)
    sd = strip_state_dict_prefixes(ckpt.get("model_state_dict", ckpt))
    model = create_model(variant="base", global_mode="balanced", upsample_unfold=False).eval()
    model.load_state_dict(sd, strict=True)
    report["parameterCount"] = sum(v.numel() for v in model.parameters())
    report["checkpointLoad"] = "strict=True; all keys matched"
    unfused = copy.deepcopy(model).eval()
    model.fuse_for_inference()
    fuse_remaining_conv_bn(model)
    reference = copy.deepcopy(model).eval()

    # Remove exactly the attention-to-mean substitution, retaining other export code.
    original_path = SOURCE / "scripts/export.py"
    original_text = original_path.read_text(encoding="utf-8")
    bad = """        if type(m).__name__ == 'GlobalContextBlock':
            def _fwd(self_m, x, _h=s16[0], _w=s16[1]):
                ctx = F.avg_pool2d(x, kernel_size=(_h, _w))
                return x + self_m.transform(ctx)
            m.forward = types.MethodType(_fwd, m)

        elif type(m).__name__ == 'StripPoolingAttention':"""
    if original_text.count(bad) != 1:
        raise ValueError("Pinned exporter does not match minimal patch")
    corrected_text = original_text.replace(bad, "        if type(m).__name__ == 'StripPoolingAttention':")
    corrected_path = OUT / "export_corrected.py"
    corrected_path.write_bytes(corrected_text.encode("utf-8"))
    patch = "".join(difflib.unified_diff(original_text.splitlines(True), corrected_text.splitlines(True), fromfile="a/scripts/export.py", tofile="b/scripts/export.py"))
    (OUT / "preserve-global-context.patch").write_bytes(patch.encode("utf-8"))
    original = module_at("zipdepth_original_export", original_path)
    corrected = module_at("zipdepth_corrected_export", corrected_path)
    report["patchSha256"] = digest(OUT / "preserve-global-context.patch")

    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    yy /= np.float32(HEIGHT - 1)
    xx /= np.float32(WIDTH - 1)
    inputs = {
        "gradient": np.stack((xx, yy, (xx + yy) * np.float32(0.5)))[None],
        "checker": np.stack((np.floor(xx * 12) % 2, np.floor(yy * 12) % 2, (np.floor(xx * 12) + np.floor(yy * 12)) % 2)).astype(np.float32)[None],
        "constant": np.full((1, 3, HEIGHT, WIDTH), 0.5, dtype=np.float32),
    }
    references = {}
    report["shape"] = [1, 3, HEIGHT, WIDTH]
    report["fusionVsUnfusedCheckpoint"] = {}
    with torch.inference_mode():
        for name, x in inputs.items():
            references[name] = reference(torch.from_numpy(x)).numpy().copy()
            before_fusion = unfused(torch.from_numpy(x)).numpy().copy()
            report["fusionVsUnfusedCheckpoint"][name] = errors(references[name], before_fusion)
            if not report["fusionVsUnfusedCheckpoint"][name]["allclose_rtol1e-4_atol1e-5"]:
                raise AssertionError("Fusion fidelity gate failed")
            x.tofile(OUT / f"{name}-input.f32")
            references[name].tofile(OUT / f"{name}-reference.f32")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    report["lanes"] = {}
    for name, exporter in (("original", original), ("corrected", corrected)):
        candidate = copy.deepcopy(model)
        destination = OUT / f"{name}.onnx"
        print(f"Exporting {name}", flush=True)
        exporter.export_onnx(candidate, (1, 3, HEIGHT, WIDTH), str(destination), opset=17)
        graph = onnx.load(destination)
        onnx.checker.check_model(graph)
        record = {"bytes": destination.stat().st_size, "sha256": digest(destination), "operators": sorted(set(n.op_type for n in graph.graph.node)), "cases": {}}
        session = ort.InferenceSession(str(destination), options, providers=["CPUExecutionProvider"])
        for case, x in inputs.items():
            with torch.inference_mode():
                mutated = candidate(torch.from_numpy(x)).numpy().copy()
            result = session.run(["depth"], {"image": x})[0]
            result.tofile(OUT / f"{case}-{name}.f32")
            record["cases"][case] = {
                "onnxVsUnmodifiedReference": errors(result, references[case]),
                "mutatedTorchVsUnmodifiedReference": errors(mutated, references[case]),
                "onnxVsMutatedTorch": errors(result, mutated),
                "referenceRange": [float(references[case].min()), float(references[case].max())],
            }
        report["lanes"][name] = record
        del session, candidate
    # Regression gate is the preserved reference, not the model after export mutation.
    report["correctedPassesAllclose"] = all(v["onnxVsUnmodifiedReference"]["allclose_rtol1e-4_atol1e-5"] for v in report["lanes"]["corrected"]["cases"].values())
    report["originalFailsAnyAllclose"] = any(not v["onnxVsUnmodifiedReference"]["allclose_rtol1e-4_atol1e-5"] for v in report["lanes"]["original"]["cases"].values())
    if not report["correctedPassesAllclose"] or not report["originalFailsAnyAllclose"]:
        raise AssertionError("Hypothesis or fixed export fidelity gate did not pass; inspect preserved metrics")
    report["status"] = "passed"


report = {"kind": "zipdepth-export-fidelity-v1", "status": "failed", "scope": "Released Base NPU checkpoint; three procedural images; numerical fidelity, not depth accuracy."}
start = time.monotonic()
try:
    main()
except Exception:
    report["failure"] = traceback.format_exc()
finally:
    report["elapsedSeconds"] = time.monotonic() - start
    report["artifacts"] = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in OUT.iterdir() if p.is_file()}
    data = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (OUT / "report.json").write_bytes(data.encode("utf-8"))
    (OUT / "report.json.sha256").write_text(hashlib.sha256(data.encode()).hexdigest() + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "status": report["status"], "seconds": report["elapsedSeconds"], "lanes": report.get("lanes"), "failure": report.get("failure")}, indent=2))
if report["status"] != "passed":
    raise SystemExit(1)
