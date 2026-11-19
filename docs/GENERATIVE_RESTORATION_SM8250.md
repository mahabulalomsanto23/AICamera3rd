# Generative restoration backend — Snapdragon 865 (SM8250)

## Decision

Use PiSA-SR / PiSA-Lite as the first generative-restoration candidate, behind an optional backend. Keep MAXIM, ADE20K, face protection and RLFN as the reliable offline fallback.

PiSA-SR is a one-step diffusion super-resolution method with separate pixel-fidelity and semantic-detail guidance controls. Semantic guidance must remain conservative because stronger guidance can create plausible but incorrect textures.

## Snapdragon 865 compatibility

The OnePlus 8 platform is Snapdragon 865 / SM8250 / Hexagon V66. Qualcomm's QAIRT device matrix lists SM8250 as an Android target (SoC model 21, HTP V66). This means the chipset is supported by the proprietary Qualcomm QAIRT/QNN SDK at the platform level; it does not mean every open-source QNN integration or every precompiled model supports it.

Important distinctions:

- **QAIRT SDK:** lists SM8250 / V66 as an Android target.
- **Qualcomm AI Hub device catalog:** the published chipset attribute list includes Snapdragon 865+ but does not list the base Snapdragon 865. Do not assume AI Hub Workbench can compile/profile directly against a OnePlus 8 target.
- **ExecuTorch QNN backend:** its current documented supported mobile SoCs do not include SM8250. Do not assume its standard QNN backend can target this phone without upstream support or custom work.
- **PiSA-Lite binaries:** the public QNN context binaries are compiled specifically for Snapdragon 8 Gen 3 / SM8650. They must not be copied into this app or loaded on SM8250.

Therefore, the practical route is a direct QAIRT/QNN SDK integration and a separately compiled SM8250 package, subject to access to a compatible SDK release and a real OnePlus 8 for validation. AI Hub cloud compilation is not currently a confirmed route for the base 865.

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

1. Obtain a compatible Qualcomm QAIRT/QNN SDK release and confirm that its HTP runtime supports the OnePlus 8's SM8250 / V66 firmware.
2. Compile PiSA-Lite's ONNX graphs for SM8250 / HTP V66. Do not assume AI Hub Workbench exposes a base-Snapdragon-865 target; its published device attributes list 865+ but not base 865.
3. Verify graph operators, quantization and context binaries on an actual OnePlus 8. The current ExecuTorch QNN backend's documented chipset list does not include SM8250, so a direct QNN SDK C++/JNI runner or a separately validated backend is required.
4. Obtain/ship the matching QNN Android runtime and native JNI runner under the applicable Qualcomm terms.
5. Confirm the upstream PiSA-Lite weight redistribution rights; its model card uses `license: other` and explicitly warns that upstream component terms apply.

## References

- Official PiSA-SR: https://github.com/csslc/PiSA-SR
- PiSA-Lite model card and Android integration notes: https://huggingface.co/Alptekinege/PiSA-Lite
- Qualcomm QAIRT supported-device matrix: https://docs.qualcomm.com/nav/home/QNN_general_overview.html?product=924033590759186372
- Qualcomm AI Hub device attributes: https://dev.aihub.qualcomm.com/docs/hub/generated/qai_hub.Device.html
- Qualcomm AI Hub device selection: https://workbench.aihub.qualcomm.com/docs/hub/devices.html
- ExecuTorch QNN backend support matrix: https://android.googleid.googlesource.com/platform/external/executorch/+/HEAD/.wiki/backends/qnn/overview.md
