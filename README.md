# Pocket AI v0.1.2

This is the iPhone debugging build.

Changes from v0.1.1:
- Tests WebGPU with `requestAdapter()` instead of assuming `navigator.gpu` means a usable GPU.
- Falls back cleanly to WASM when WebGPU is unavailable.
- Disables Transformers.js Cache API/WASM caching on ordinary LAN HTTP origins where the Cache API is unavailable.
- Limits ONNX WASM to one thread for Safari/iPhone compatibility.
- Shows secure-context and WASM-thread diagnostics when initialization fails.
- Bumps the service-worker shell cache so the iPhone receives the new JavaScript.

The initial download still requires internet. After the model/runtime have been cached, the app is designed to run without a network connection.

Test:
1. `npm install`
2. `npm run dev -- --host 0.0.0.0`
3. Open the shown LAN address on the iPhone.
4. Tap Download AI.
5. If it fails after reaching 100%, copy the error shown on screen back to the chat.

The model is `onnx-community/Qwen2.5-0.5B-Instruct` with `q4`.
