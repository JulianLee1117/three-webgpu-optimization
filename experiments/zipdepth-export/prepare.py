"""Fetch a pinned, small ZipDepth source/checkpoint fixture; do not execute it."""
from pathlib import Path
import hashlib
import json
import urllib.request

REVISION = "91f3fd21e131641f51e8d35736d1958350180e3a"
ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / ".local-research" / "zipdepth-export-source"
FILES = {
    "zipdepth/model/architecture.py": (200_000, "9fd74fa2053da5bd56b2f2eecfdc3a6bf5e5161da0df05c72da3b89f29b77521"),
    "zipdepth/utils/model_utils.py": (100_000, "cd396c8d4f1eeb93da564b879d2b62aee8025d11222188153a549457649ead5c"),
    "scripts/export.py": (100_000, "d4c5612877176c144445516fbcf16aa5922666eb674007f075816e99a3c2c69a"),
    "README.md": (100_000, "2c60d28b05ac697368b8c926a8fc776218842942effd6bc6cb2329016aee3f0f"),
    "LICENSE": (20_000, "0007e2ff761f1b89ad89327870b807cd4de00cb657b442b62de2f92fdc87d508"),
    "checkpoints/zipdepth_base_npu.pth": (32_000_000, "627c04fda584133ead4310074884a4a037061b4c01ba86e73e492ea30fab570d"),
}


def main():
    records = {}
    for name, (limit, expected_sha256) in FILES.items():
        url = f"https://raw.githubusercontent.com/fabiotosi92/ZipDepth/{REVISION}/{name}"
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        cached = target.exists()
        if cached:
            with target.open("rb") as cache:
                data = cache.read(limit + 1)
        else:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"File exceeds size bound: {name}")
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"Pinned checksum mismatch: {name}; expected {expected_sha256}, got {actual_sha256}")
        if not cached:
            target.write_bytes(data)
        records[name] = {"url": url, "bytes": len(data), "sha256": actual_sha256}
    # Only the two inspected modules are needed; avoid executing package initializers.
    for package in ("zipdepth", "zipdepth/model", "zipdepth/utils"):
        target = DEST / package / "__init__.py"
        if not target.exists():
            target.write_text("# Local isolated source fixture.\n", encoding="utf-8")
    manifest = {"revision": REVISION, "files": records}
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
