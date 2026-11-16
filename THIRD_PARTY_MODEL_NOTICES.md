# Third-party model notices

## NTIRE 2024 RLFN baseline

- Model: Residual Local Feature Network (RLFN), NTIRE 2024 Efficient Super-Resolution baseline.
- Upstream: https://github.com/Amazingren/NTIRE2024_ESR
- Upstream revision: 6404fa48dde4a4adbf241a51debdc661aef20449
- License: MIT (see upstream repository LICENSE).
- Model checkpoint: model_zoo/team00_rlfn.pth.
- Architecture source: models/team00_RLFN.py.
- Exported for this application to a fixed 1x3x128x128 input and 1x3x512x512 output ONNX graph.
- Runtime: ONNX Runtime Android CPU execution provider; no network access is used during inference.

The model is a conventional single-image super-resolution model trained for benchmark reconstruction. It can improve resolution/detail but does not guarantee recovery of details absent from the source image.
