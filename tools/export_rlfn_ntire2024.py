#!/usr/bin/env python3
"""Export the MIT-licensed NTIRE 2024 RLFN baseline to a fixed-shape ONNX model.

The checkpoint is the official 0.317M-parameter RLFN baseline. It expects RGB
values in [0,255], takes 128x128 patches, and produces 4x output patches.
"""
from pathlib import Path
import urllib.request\nimport hashlib

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "app" / "src" / "main" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
SOURCE_URL = "https://raw.githubusercontent.com/Amazingren/NTIRE2024_ESR/6404fa48dde4a4adbf241a51debdc661aef20449/models/team00_RLFN.py"
WEIGHT_URL = "https://raw.githubusercontent.com/Amazingren/NTIRE2024_ESR/6404fa48dde4a4adbf241a51debdc661aef20449/model_zoo/team00_rlfn.pth"
SOURCE = ROOT / "tools" / "_ntire_team00_rlfn.py"
WEIGHTS = ROOT / "tools" / "_ntire_team00_rlfn.pth"
OUTPUT = ASSETS / "rlfn_ntire2024_x4.onnx"


def download(url: str, path: Path, expected_git_blob: str):
    request = urllib.request.Request(url, headers={"User-Agent": "AICamera3rd-model-builder/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
    if len(data) < 1024:
        raise RuntimeError(f"Unexpectedly small download: {url} ({len(data)} bytes)")
    path.write_bytes(data)
    print(f"Downloaded {path.name}: {len(data):,} bytes")


download(SOURCE_URL, SOURCE)
download(WEIGHT_URL, WEIGHTS)

# Import the exact upstream architecture without copying or modifying its code.
namespace = {}
exec(compile(SOURCE.read_text(), str(SOURCE), "exec"), namespace)
model = namespace["RLFN_Prune"]().cpu().eval()
state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)

dummy = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
with torch.inference_mode():
    expected = model(dummy)
if tuple(expected.shape) != (1, 3, 512, 512):
    raise RuntimeError(f"Unexpected RLFN output: {tuple(expected.shape)}")

torch.onnx.export(
    model,
    dummy,
    str(OUTPUT),
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["image"],
    output_names=["restored"],
    dynamic_axes=None,
)
onnx_model = onnx.load(str(OUTPUT))
onnx.checker.check_model(onnx_model)

session = ort.InferenceSession(str(OUTPUT), providers=["CPUExecutionProvider"])
sample = np.zeros((1, 3, 128, 128), dtype=np.float32)
result = session.run(["restored"], {"image": sample})[0]
if result.shape != (1, 3, 512, 512) or not np.isfinite(result).all():
    raise RuntimeError(f"RLFN ONNX smoke inference failed: shape={result.shape}")
if OUTPUT.stat().st_size < 500_000:
    raise RuntimeError(f"RLFN ONNX unexpectedly small: {OUTPUT.stat().st_size}")

print(f"RLFN ONNX ready: {OUTPUT.stat().st_size:,} bytes; 128x128 -> 512x512")
WEIGHTS.unlink(missing_ok=True)
SOURCE.unlink(missing_ok=True)
