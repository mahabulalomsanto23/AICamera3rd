#!/usr/bin/env python3
"""Fetch and smoke-test the offline NAFNet and ADE20K Lite models."""
from pathlib import Path
import io
import tarfile
import zipfile
import urllib.request

import numpy as np
import tensorflow as tf

ASSETS = Path("app/src/main/assets")
ASSETS.mkdir(parents=True, exist_ok=True)


def fetch(url: str, destination: Path):
    print("Downloading", url)
    request = urllib.request.Request(url, headers={"User-Agent": "AICamera3rd-build/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    if len(payload) < 1024 * 1024:
        raise RuntimeError(f"Model download unexpectedly small: {len(payload)} bytes")
    destination.write_bytes(payload)
    print("Downloaded", destination, len(payload), "bytes")


def extract_tflite(archive: Path, destination: Path):
    payload = archive.read_bytes()
    stream = io.BytesIO(payload)
    if zipfile.is_zipfile(stream):
        with zipfile.ZipFile(stream) as z:
            candidates = [n for n in z.namelist() if n.lower().endswith(".tflite")]
            if not candidates:
                raise RuntimeError("TFHub archive contains no .tflite file")
            destination.write_bytes(z.read(candidates[0]))
            return
    stream.seek(0)
    if tarfile.is_tarfile(archive):
        with tarfile.open(fileobj=stream, mode="r:*") as t:
            members = [m for m in t.getmembers() if m.name.lower().endswith(".tflite")]
            if not members:
                raise RuntimeError("TFHub archive contains no .tflite file")
            destination.write_bytes(t.extractfile(members[0]).read())
            return
    # Some TFHub Lite endpoints return the FlatBuffer directly.
    if len(payload) > 8 and payload[4:8] == b"TFL3":
        destination.write_bytes(payload)
        return
    raise RuntimeError("Unknown TFHub model archive format")


def smoke(path: Path, expected_input=None, min_classes=None):
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2)
    interpreter.allocate_tensors()
    ins = interpreter.get_input_details()
    outs = interpreter.get_output_details()
    print(path.name, "input", [(x["shape"].tolist(), str(x["dtype"])) for x in ins],
          "outputs", [(x["shape"].tolist(), str(x["dtype"])) for x in outs])
    if len(ins) != 1 or len(outs) != 1:
        raise RuntimeError(f"Unexpected tensor count for {path.name}")
    shape = ins[0]["shape"]
    if expected_input and list(shape) != expected_input:
        raise RuntimeError(f"{path.name}: expected input {expected_input}, got {shape}")
    if min_classes and (len(outs[0]["shape"]) != 4 or int(outs[0]["shape"][-1]) < min_classes):
        raise RuntimeError(f"{path.name}: expected at least {min_classes} classes, got {outs[0]['shape']}")
    sample = np.zeros(shape, dtype=ins[0]["dtype"])
    interpreter.set_tensor(ins[0]["index"], sample)
    interpreter.invoke()
    result = interpreter.get_tensor(outs[0]["index"])
    if not np.isfinite(result).all():
        raise RuntimeError(f"{path.name} produced NaN/Inf")
    return shape, outs[0]["shape"]


naf = ASSETS / "nafnet_sidd_width32_fp16.tflite"
fetch("https://huggingface.co/litert-community/NAFNet-SIDD-width32-LiteRT/resolve/main/nafnet_sidd_width32_fp16.tflite", naf)
smoke(naf, [1, 3, 256, 256])

seg_archive = ASSETS / "ade20k_tfhub_download"
seg_model = ASSETS / "deeplabv3_ade20k.tflite"
fetch("https://tfhub.dev/sayakpaul/deeplabv3-mobilenetv2-ade20k/1/default/lite/2?tf-hub-format=compressed", seg_archive)
extract_tflite(seg_archive, seg_model)
seg_archive.unlink(missing_ok=True)
smoke(seg_model, min_classes=100)
print("NAFNet and ADE20K model download + CPU smoke inference passed")
