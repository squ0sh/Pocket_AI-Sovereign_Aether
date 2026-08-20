import { pipeline, TextStreamer } from "@huggingface/transformers";
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
    assistant.textContent = `Sorry — the local model could not generate a response.\n\n${error.message}`;
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
  status.textContent = "Loading local AI…";
  progressText.textContent = "Downloading model files…";

  try {
    const device = "gpu" in navigator ? "webgpu" : "wasm";

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
          progressText.textContent = "Preparing model…";
        }
      }
    });

    status.textContent = device === "webgpu"
      ? "Local AI · WebGPU"
      : "Local AI · CPU fallback";

    welcome.hidden = true;
    input.disabled = false;
    send.disabled = false;
    input.focus();
  } catch (error) {
    status.textContent = "Could not load local AI";
    progressText.textContent = error.message;
    loadButton.disabled = false;
    console.error(error);
  }
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
