#!/usr/bin/env python3
"""Download Google's FiveK MAXIM retouching model and export an FP16 TFLite asset."""
from pathlib import Path
import tensorflow as tf
from huggingface_hub import snapshot_download

MODEL_DIR = snapshot_download(
    repo_id="google/maxim-s2-enhancement-fivek",
    allow_patterns=["saved_model.pb", "variables/*"],
)
out_dir = Path("app/src/main/assets")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "maxim_fivek_fp16.tflite"

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
model = converter.convert()
out_path.write_bytes(model)

interpreter = tf.lite.Interpreter(model_content=model)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print("MAXIM FiveK TFLite:", out_path, len(model), "bytes")
print("Input:", inp["shape"], inp["dtype"], "Output:", out["shape"], out["dtype"])
if list(inp["shape"]) != [1, 256, 256, 3] or list(out["shape"]) != [1, 256, 256, 3]:
    raise RuntimeError("Unexpected MAXIM FiveK tensor shape; refusing to package an incompatible model")
