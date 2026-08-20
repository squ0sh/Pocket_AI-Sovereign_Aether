# Pocket AI v0.1.1

This is the iPhone debugging build.

Changes from v0.1:
- Correct WebGPU detection via `navigator.gpu`.
- Uses Transformers.js browser model cache.
- Explicitly enables WASM runtime caching.
- Disables filesystem caching in the browser.
- Shows the actual model initialization error instead of silently returning to the download screen.
- Displays whether WebGPU and Cache API are available.

The initial download still requires internet. After the model/runtime have been cached, the app is designed to run without a network connection.

Test:
1. `npm install`
2. `npm run dev -- --host 0.0.0.0`
3. Open the shown LAN address on the iPhone.
4. Tap Download AI.
5. If it fails after reaching 100%, copy the error shown on screen back to the chat.

The model is `onnx-community/Qwen2.5-0.5B-Instruct` with `q4`.
