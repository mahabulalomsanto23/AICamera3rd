# Generative restoration backend — Snapdragon 865 (SM8250)

## Decision

Use PiSA-SR / PiSA-Lite as the first generative-restoration candidate, behind an optional backend. Keep MAXIM, ADE20K, face protection and RLFN as the reliable offline fallback.

PiSA-SR is a one-step diffusion super-resolution method with separate pixel-fidelity and semantic-detail guidance controls. Semantic guidance must remain conservative because stronger guidance can create plausible but incorrect textures.

## Snapdragon 865 compatibility

The OnePlus 8 platform is Snapdragon 865 / SM8250 / Hexagon V66. Qualcomm's current QAIRT device matrix lists SM8250 as an Android target (SoC model 21, HTP V66). Therefore, the chipset is not categorically excluded from QNN.

However, the public PiSA-Lite QNN context binaries are compiled specifically for Snapdragon 8 Gen 3 / SM8650. They must not be copied into this app or loaded on SM8250. The PiSA-Lite README says its ONNX graphs need to be compiled separately for each target Snapdragon using Qualcomm AI Hub / QAIRT / QNN tooling.

## Required target package

A valid SM8250 package must include all three target-compiled graphs:
- encoder
- denoiser
- decoder

All three execute in order. The Android integration must call the QNN HTP backend through a native C++/JNI runner. The app must verify target SoC, graph metadata, runtime libraries and package checksums before enabling the feature.

Expected private app storage location:
`files/models/pisa_sm8250/`

Do not download or bundle the SM8650 binaries as a fallback. If no validated SM8250 package/runtime is present, retain the existing offline MAXIM + scene pipeline + RLFN path and report that generative restoration is unavailable.

## Quality and safety controls

- Keep pixel-fidelity and semantic-detail controls separate.
- Default semantic guidance low; expose stronger reconstruction only as an explicit setting after device validation.
- Blend generated output with the input to limit face, logo, text and geometry drift.
- Never describe generated texture as recovered factual detail.
- Compare input/output on the same user photos and log model load, inference time, peak memory and thermal state.

## Blockers before implementation can be enabled

1. Compile PiSA-Lite's ONNX graphs for SM8250 / HTP V66 with a compatible Qualcomm QAIRT/QNN toolchain.
2. Verify graph operators, quantization and context binaries on an actual OnePlus 8.
3. Obtain/ship the matching QNN Android runtime and native JNI runner under the applicable Qualcomm terms.
4. Confirm the upstream PiSA-Lite weight redistribution rights; its model card uses `license: other` and explicitly warns that upstream component terms apply.

## References

- Official PiSA-SR: https://github.com/csslc/PiSA-SR
- PiSA-Lite model card and Android integration notes: https://huggingface.co/Alptekinege/PiSA-Lite
- Qualcomm QAIRT supported-device matrix: https://docs.qualcomm.com/nav/home/QNN_general_overview.html?product=924033590759186372
