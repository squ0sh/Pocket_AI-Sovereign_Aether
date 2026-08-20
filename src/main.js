import { pipeline, TextStreamer, env } from "@huggingface/transformers";
import "./style.css";

const MODEL = "onnx-community/Qwen2.5-0.5B-Instruct";
const STORAGE_KEY = "pocket-ai-messages";

let generator = null;
let messages = loadMessages();

const chat = document.querySelector("#chat");
const welcome = document.querySelector("#welcome");
const loadButton = document.querySelector("#loadButton");
const modelButton = document.querySelector("#modelButton");
const composer = document.querySelector("#composer");
const input = document.querySelector("#input");
const send = document.querySelector("#send");
const status = document.querySelector("#status");
const progressWrap = document.querySelector("#progressWrap");
const progress = document.querySelector("#progress");
const progressText = document.querySelector("#progressText");
const errorBox = document.querySelector("#errorBox");

// Safari/iPhone can be served from an ordinary LAN HTTP origin (e.g.
// http://10.0.0.103:5173). In that context the Cache API is unavailable.
// Transformers.js will otherwise try to use its browser/WASM cache and can
// fail *after* the model files have finished downloading.
const cacheAvailable = typeof caches !== "undefined" &&
  (window.isSecureContext || location.hostname === "localhost" || location.hostname === "127.0.0.1");

env.useBrowserCache = cacheAvailable;
env.useWasmCache = cacheAvailable;
env.useFSCache = false;
env.cacheKey = "pocket-ai-transformers-v0.1.2";

// Keep the WASM backend conservative on Safari/iPhone. One thread avoids
// SharedArrayBuffer / worker-related runtime problems on restricted origins.
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.numThreads = 1;
}

renderMessages();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js"));
}

modelButton.addEventListener("click", () => {
  alert("Version one has one local model: Qwen 2.5 0.5B Instruct, four-bit quantized. Online models can be added later.");
});

loadButton.addEventListener("click", loadModel);

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || !generator) return;

  input.value = "";
  input.disabled = true;
  send.disabled = true;

  addMessage("user", text);
  const assistant = addMessage("assistant", "");
  saveMessages();

  const conversation = [
    { role: "system", content: "You are a helpful, concise assistant running locally on a phone. Answer naturally and honestly." },
    ...messages.slice(-12)
  ];

  try {
    let generated = "";
    const streamer = new TextStreamer(generator.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (token) => {
        generated += token;
        assistant.textContent = generated;
        scrollToBottom();
      }
    });

    await generator(conversation, {
      max_new_tokens: 256,
      do_sample: true,
      temperature: 0.7,
      top_p: 0.9,
      streamer
    });

    const last = messages[messages.length - 1];
    if (last?.role === "assistant") last.content = generated.trim();
    saveMessages();
  } catch (error) {
    assistant.textContent = `Local generation error:\n${formatError(error)}`;
    console.error(error);
  } finally {
    input.disabled = false;
    send.disabled = false;
    input.focus();
  }
});

async function loadModel() {
  if (generator) return;

  loadButton.disabled = true;
  progressWrap.hidden = false;
  errorBox.hidden = true;
  progress.style.width = "0%";
  status.textContent = "Loading local AI…";
  progressText.textContent = "Checking device and model cache…";

  const hasWebGPU = await detectWebGPU();
  const device = hasWebGPU ? "webgpu" : "wasm";

  try {
    if (hasWebGPU) {
      status.textContent = "Local AI · WebGPU";
      progressText.textContent = "WebGPU detected · preparing model…";
    } else {
      status.textContent = "Local AI · CPU fallback";
      progressText.textContent = "WebGPU unavailable · preparing CPU model…";
    }

    generator = await pipeline("text-generation", MODEL, {
      device,
      dtype: "q4",
      progress_callback: (info) => {
        if (info.status === "progress" && Number.isFinite(info.progress)) {
          const pct = Math.max(0, Math.min(100, info.progress));
          progress.style.width = `${pct}%`;
          progressText.textContent = `${info.file ?? "Model"} · ${Math.round(pct)}%`;
        } else if (info.status === "initiate") {
          progressText.textContent = `Downloading ${info.file ?? "model"}…`;
        } else if (info.status === "done") {
          progressText.textContent = "File ready · initializing runtime…";
        }
      }
    });

    status.textContent = hasWebGPU ? "Local AI · WebGPU" : "Local AI · CPU fallback";
    progressWrap.hidden = true;
    welcome.hidden = true;
    input.disabled = false;
    send.disabled = false;
    input.focus();
  } catch (error) {
    generator = null;
    status.textContent = `Local AI · ${device} failed`;
    progressText.textContent = "Model downloaded, but initialization failed.";
    errorBox.textContent = [
      "INITIALIZATION ERROR",
      "",
      formatError(error),
      "",
      `Browser: ${navigator.userAgent}`,
      `WebGPU API: ${hasWebGPU ? "available" : "not available"}`,
      `Cache API: ${cacheAvailable ? "available" : "not available on this origin"}`,
      `Secure context: ${window.isSecureContext ? "yes" : "no"}`,
      `WASM threads: ${env.backends?.onnx?.wasm?.numThreads ?? "default"}`,
      "",
      "Send this screen/error text back to us so we can diagnose the iPhone path."
    ].join("\n");
    errorBox.hidden = false;
    loadButton.disabled = false;
    console.error("Pocket AI model initialization failed:", error);
  }
}

async function detectWebGPU() {
  if (typeof navigator === "undefined" || !navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

function formatError(error) {
  if (!error) return "Unknown error";
  return error.stack || error.message || String(error);
}

function addMessage(role, content) {
  const bubble = document.createElement("div");
  bubble.className = `message ${role}`;
  bubble.textContent = content;
  chat.appendChild(bubble);
  messages.push({ role, content });
  scrollToBottom();
  return bubble;
}

function renderMessages() {
  for (const message of messages) {
    const bubble = document.createElement("div");
    bubble.className = `message ${message.role}`;
    bubble.textContent = message.content;
    chat.appendChild(bubble);
  }
  scrollToBottom();
}

function loadMessages() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveMessages() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-100)));
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chat.scrollTop = chat.scrollHeight;
  });
}