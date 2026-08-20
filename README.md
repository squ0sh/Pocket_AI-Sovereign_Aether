# Pocket AI

A tiny PWA prototype for running a small language model locally in a phone browser.

## Current prototype

- Vanilla JavaScript
- PWA installable on iPhone and Android
- Transformers.js
- Qwen 2.5 0.5B Instruct
- Four-bit quantized model
- WebGPU when available, WASM/CPU fallback
- Browser-side model caching
- Chat history stored locally in localStorage
- No backend and no AI API

Hugging Face's Transformers.js documentation explicitly supports Qwen2.5-0.5B-Instruct in Q4 with WebGPU.

## Run it

```bash
npm install
npm run dev
```

For an actual phone test, use a secure HTTPS deployment or an HTTPS local-development tunnel. Service workers and WebGPU are browser security features and should be tested in a real mobile browser.

## Important

The first model download requires internet access. After the model has been downloaded and cached, the goal is for the app to continue working offline. Browser storage policies can vary by device/browser, so the prototype should be tested on the actual iPhone and Android phones we intend to support.


Created by: Joshua Rhoads

-Love, Truth, & Justice are the Law
