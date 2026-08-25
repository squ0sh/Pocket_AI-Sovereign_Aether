# Pocket AI — v0.3.0

## What this build is

Pocket AI is a browser-local PWA using MLC/WebLLM and WebGPU. This version keeps **Qwen 0.5B** as the working fallback and adds **Ternary Bonsai 1.7B** to the model selector as an experimental path.

## Version history

### v0.2.0 — MLC/WebLLM + WebGPU foundation
- Moved the PWA to MLC/WebLLM for browser-local inference.
- WebGPU is required.
- Qwen 0.5B was the initial small local model.
- No ONNX/WASM fallback.

### v0.2.1 — HTTPS + diagnostics
- Added automatic HTTPS development through `@vitejs/plugin-basic-ssl`.
- Kept real WebGPU adapter detection.
- Kept IndexedDB model caching in WebLLM.
- Added clearer secure-context/GPU diagnostics for iPhone Safari.
- Qwen 0.5B remained the working model.

### v0.3.0 — Qwen 0.5B + Ternary Bonsai 1.7B selector
- Model selector now contains exactly two local choices:
  - Qwen 0.5B — working fallback.
  - Ternary Bonsai 1.7B — experimental.
- Preserves the PWA, HTTPS, WebGPU, chat history, and local-storage approach.
- Adds the published Bonsai MLC/WebGPU artifact configuration.
- **Important:** the published Bonsai artifact requires a patched MLC/WebLLM runtime with the custom `bonsai_tq_f32` quantization profile. The stock WebLLM 0.2.84 dependency in this build does not contain that profile, so Bonsai is intentionally blocked with a clear diagnostic rather than pretending it works.

## Bonsai artifact

The experimental Bonsai MLC artifact is approximately 460 MB and uses a custom 2-bit symmetric group-quantized representation. Its model card reports successful WebGPU compilation and recommends a 4K context / 512 prefill smoke-test configuration. Browser success still depends on GPU memory, WebGPU support, cache quota, and the compatible patched runtime.

Source artifact: https://huggingface.co/welcoma/Ternary-Bonsai-1.7B-bonsai_tq_f32-MLC

## Run

```bash
npm install
npm run dev
```

Open the HTTPS address Vite prints. For iPhone testing, the device must establish a secure context and have WebGPU available.

## Model selection

Tap the model name in the top-right corner. The selected model is remembered locally. Refreshing after switching is intentional so the WebLLM engine starts cleanly.

## Next step

The next engineering task is to replace the stock WebLLM runtime with the Bonsai-compatible patched runtime, then test Bonsai generation on the iPhone. Do not remove Qwen 0.5B until Bonsai has successfully generated a response on the target device.
