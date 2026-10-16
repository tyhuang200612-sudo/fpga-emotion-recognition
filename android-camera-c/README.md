# TinyCNN-C 2026 live selfie inference app

This is **on-device facial expression inference**: live front-facing Camera2
preview -> ML Kit on-device face detector -> 48x48 luminance face ROI ->
genuine trained TinyCNN-C seed-2026 ONNX model -> three probability bars.

**Not** the earlier training progress monitor. The front camera stays local;
no networking permission, accounts, uploads or face-image storage.

The build is *fail closed*: `app/src/main/assets/model.onnx` must contain a
real checkpoint exported from `seed_2026/best.pt`; the builder verifies its
architecture, 3-class order and ONNX/PyTorch numerical parity. Do not ship
the app with random/demo/B-model weights. Only publish after the genuine C
checkpoint is archived and validated.

Preview ROI follows face detector bounds + a conservative 1.3x square context
crop, uses Y plane grayscale [0,1] like the actual model. False positives,
low-light errors and neutral/sad ambiguity are expected; predicted expression
is **not** a measure of underlying mental state. This is a personal test build,
not clinical use or verified FPGA INT8 inference.
