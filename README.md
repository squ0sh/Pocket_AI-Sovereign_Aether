# Pocket AI — v0.3.2

## Current build

Pocket AI is a browser-local PWA using MLC/WebLLM and WebGPU. Qwen 0.5B remains the known-good baseline. Bonsai 1.7B Q1 is an experimental custom model.

## Version history

### v0.2.0 — WebLLM/WebGPU foundation
- Local browser inference through MLC/WebLLM.
- WebGPU capability detection.
- PWA-oriented chat interface.

### v0.2.1 — iPhone/Safari reliability
- HTTPS development configuration.
- WebGPU diagnostics.
- IndexedDB model caching.
- Qwen 0.5B as the known-good local model.

### v0.3.0 — Model selector
- Qwen 0.5B retained as the fallback.
- Added Bonsai experimental model selection.

### v0.3.1 — Bonsai Q1 experiment
- Switched the experimental Bonsai entry to Bonsai 1.7B Q1.
- Added generation diagnostics.
- Reduced the first browser test to a small context/prefill configuration.

### v0.3.2 — Correct custom ModelRecord
- Registers the Bonsai Q1 `ModelRecord` alongside the prebuilt WebLLM models.
- Fixes the `findModelRecord` initialization failure.
- Uses the correct Bonsai Q1 WebGPU WASM library.
- Keeps Qwen 0.5B on the stock WebLLM path.
- Uses a 64-token first-generation test.

## Models

| Model | Status | Runtime requirement |
|---|---|---|
| Qwen 0.5B | Known-good fallback | Stock WebLLM 0.2.84 |
| Bonsai 1.7B Q1 | Experimental | Bonsai `bonsai_q1_f32` runtime support |

## Bonsai Q1 configuration

The model is registered using the published configuration from the Bonsai Q1 MLC artifact:

- Model ID: `Bonsai-1.7B-q1-MLC`
- Model artifact: `welcoma/Bonsai-1.7B-bonsai_q1_f32-MLC`
- WebGPU library: `libs/bonsai-q1-1.7b-bonsai_q1_f32-webgpu.wasm`
- Browser smoke-test context: 4096
- Browser smoke-test prefill: 512

This build initially overrides those to **2048 context / 128 prefill** to reduce pressure on an iPhone.

The artifact author states that Bonsai Q1 requires a patched MLC/WebLLM runtime with the custom `bonsai_q1_f32` path. It is not expected to work with an unmodified upstream WebLLM runtime.

## Run

```bash
npm install
npm run dev
```

Open the HTTPS Vite address on the iPhone.

## Testing

1. Load Qwen 0.5B first if you want to verify the baseline.
2. Select **Bonsai 1.7B Q1**.
3. Load the model.
4. Try `hello` or `Say only: ready`.
5. If initialization fails, the UI displays the WebLLM/MLC stack trace. If generation fails after successful initialization, the UI keeps the generation error visible for diagnosis.

## Important limitation

A successful custom `ModelRecord` registration does not by itself provide the Bonsai Q1 runtime. The WebGPU WASM contains the compiled model library, but the runtime must understand the `bonsai_q1_f32` quantization path.

## References

Bonsai Q1 artifact:
https://huggingface.co/welcoma/Bonsai-1.7B-bonsai_q1_f32-MLC

WebLLM custom-model documentation:
https://llm.mlc.ai/docs/deploy/webllm.html
