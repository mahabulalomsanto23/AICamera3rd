#!/usr/bin/env python3
"""Export Google's FiveK MAXIM SavedModel as a validated Android TFLite asset.

MAXIM's SavedModel uses tf.ScaleAndTranslate for antialiased bilinear resizing.
That op is not in the TFLite/Flex allowlist used by the Android runtime. For
TFLite export only, rewrite those nodes to the equivalent half-pixel
ResizeBilinear operation. This disables the extra antialias filter on
downsampling; it does not replace the learned MAXIM enhancement network.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from huggingface_hub import snapshot_download

MODEL_DIR = Path(snapshot_download(
    repo_id="google/maxim-s2-enhancement-fivek",
    allow_patterns=["saved_model.pb", "variables/*"],
))
saved_model_path = MODEL_DIR / "saved_model.pb"

saved_model = saved_model_pb2.SavedModel()
saved_model.ParseFromString(saved_model_path.read_bytes())
patched = 0


def patch_nodes(nodes):
    global patched
    for node in nodes:
        if node.op != "ScaleAndTranslate":
            continue
        image_input, size_input = node.input[0], node.input[1]
        node.op = "ResizeBilinear"
        del node.input[:]
        node.input.extend([image_input, size_input])
        node.attr.clear()
        node.attr["T"].type = tf.float32.as_datatype_enum
        node.attr["align_corners"].b = False
        node.attr["half_pixel_centers"].b = True
        patched += 1


for meta_graph in saved_model.meta_graphs:
    patch_nodes(meta_graph.graph_def.node)
    for function in meta_graph.graph_def.library.function:
        patch_nodes(function.node_def)

if patched == 0:
    raise RuntimeError("Expected MAXIM ScaleAndTranslate nodes were not found; refusing to export an unpatched model.")

saved_model_path.write_bytes(saved_model.SerializeToString())
print(f"Replaced {patched} ScaleAndTranslate nodes with ResizeBilinear")

out_dir = Path("app/src/main/assets")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "maxim_fivek_fp16.tflite"

converter = tf.lite.TFLiteConverter.from_saved_model(str(MODEL_DIR))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=model, num_threads=2)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
outputs = interpreter.get_output_details()
print("All MAXIM outputs:", [(o["index"], o["name"], o["shape"].tolist(), str(o["dtype"])) for o in outputs])
if list(inp["shape"]) != [1, 256, 256, 3]:
    raise RuntimeError(f"Unexpected MAXIM input tensor: {inp['shape']}")
matches = [(i, o) for i, o in enumerate(outputs) if list(o["shape"]) == [1, 256, 256, 3]]
if not matches:
    raise RuntimeError("No full-resolution MAXIM output [1,256,256,3]; outputs were: " +
                       repr([(o["name"], o["shape"].tolist()) for o in outputs]))

sample = np.zeros(inp["shape"], dtype=inp["dtype"])
interpreter.set_tensor(inp["index"], sample)
interpreter.invoke()
for _, out in matches:
    result = interpreter.get_tensor(out["index"])
    if not np.isfinite(result).all():
        raise RuntimeError(f"MAXIM smoke inference produced NaN/Inf for {out['name']}")

out_path.write_bytes(model)
print("MAXIM FiveK TFLite:", out_path, len(model), "bytes")
print("Input:", inp["shape"], inp["dtype"], "Full-resolution output(s):",
      [(o["index"], o["shape"].tolist(), str(o["dtype"])) for _, o in matches])
print("MAXIM TFLite conversion and smoke inference passed")
print("MAXIM TFLite conversion and smoke inference passed")
