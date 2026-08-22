import {
  CreateMLCEngine,
  prebuiltAppConfig,
} from "@mlc-ai/web-llm";
import "./style.css";

const STORAGE_KEY = "pocket-ai-messages-v2";
const MODEL_ID = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC";

let engine = null;
let hardware = null;
let messages = loadMessages();
let busy = false;

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
const hardwareBox = document.querySelector("#hardwareBox");

renderMessages();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch((error) => {
      console.warn("Service worker registration failed:", error);
    });
  });
}

modelButton.addEventListener("click", () => {
  const model = prebuiltAppConfig.model_list.find(
    (item) => item.model_id === MODEL_ID
  );

  alert(
    "Pocket AI\n\n" +
      "Model: " +
      MODEL_ID +
      "\n" +
      "Estimated GPU memory: " +
      Math.round(model?.vram_required_MB ?? 945) +
      " MB\n\n" +
      "WebLLM/MLC runs the model locally through WebGPU. " +
      "Nothing is sent to OpenAI or another AI API."
  );
});

loadButton.addEventListener("click", loadModel);

composer.addEventListener("submit", async (event) => {
  event.preventDefault();

  const text = input.value.trim();

  if (!text || !engine || busy) {
    return;
  }

  busy = true;

  input.value = "";
  input.disabled = true;
  send.disabled = true;

  // Add the user's message to our stored conversation.
  addMessage("user", text);
  saveMessages();

  // Build the conversation WITHOUT the empty assistant UI bubble.
  const conversation = [
    {
      role: "system",
      content:
        "You are a helpful, concise assistant running locally on the user's device. Answer naturally and honestly.",
    },
    ...messages.slice(-14),
  ];

  // This bubble is only for displaying the response.
  const assistant = addMessage("assistant", "");

  try {
    const chunks = await engine.chat.completions.create({
      messages: conversation,
      temperature: 0.7,
      top_p: 0.9,
      max_tokens: 256,
      stream: true,
    });

    let generated = "";

    for await (const chunk of chunks) {
      const delta = chunk.choices?.[0]?.delta?.content ?? "";

      if (delta) {
        generated += delta;
        assistant.textContent = generated;
        scrollToBottom();
      }
    }

    // Store the completed assistant response.
    messages.push({
      role: "assistant",
      content: generated.trim(),
    });

    saveMessages();
  } catch (error) {
    const formatted = formatError(error);

    assistant.textContent =
      "Local generation error:\n" + formatted;

    console.error("Pocket AI generation failed:", error);

    // Don't leave an empty assistant message in localStorage.
    messages = messages.filter(
      (message) =>
        !(message.role === "assistant" && message.content === "")
    );

    saveMessages();
  } finally {
    busy = false;

    input.disabled = false;
    send.disabled = false;

    input.focus();
  }
});

async function loadModel() {
  if (engine) {
    return;
  }

  loadButton.disabled = true;
  progressWrap.hidden = false;
  errorBox.hidden = true;

  progress.style.width = "0%";

  status.textContent = "Local AI · checking device...";
  progressText.textContent =
    "Checking WebGPU and device capabilities...";

  try {
    hardware = await inspectHardware();
    renderHardware(hardware);

    if (!hardware.secureContext) {
      throw new Error(
        "WebGPU requires a secure context (HTTPS). " +
          "This address is HTTP, so WebLLM cannot access the GPU here. " +
          "On the iPhone, open Pocket AI over HTTPS before loading the model."
      );
    }

    if (!hardware.webgpuApi) {
      throw new Error(
        "WebGPU is not exposed by this browser. " +
          "WebLLM requires WebGPU; there is intentionally no ONNX/WASM " +
          "fallback in this build because that runtime was crashing on iPhone."
      );
    }

    if (!hardware.adapter) {
      throw new Error(
        "WebGPU is exposed, but Safari could not provide a GPU adapter. " 
+
          "The browser/device combination cannot run WebLLM right now."
      );
    }

    status.textContent =
      "Local AI · WebGPU · " + hardware.label;

    progressText.textContent =
      "GPU available · downloading MLC model...";

    engine = await CreateMLCEngine(MODEL_ID, {
      appConfig: {
        ...prebuiltAppConfig,
        cacheBackend: "indexeddb",
      },

      initProgressCallback: (info) => {
        if (info?.progress != null) {
          const pct = Math.max(
            0,
            Math.min(100, Number(info.progress) * 100)
          );

          progress.style.width = pct + "%";
        }

        progressText.textContent =
          info?.text || "Preparing local GPU runtime...";
      },
    });

    const vendor = await safeGPUVendor(engine);

    if (vendor) {
      hardware.vendor = vendor;
      renderHardware(hardware);
    }

    status.textContent =
      "Local AI · WebGPU · " + hardware.label;

    progress.style.width = "100%";

    progressText.textContent =
      "Ready · model is running on this device.";

    setTimeout(() => {
      progressWrap.hidden = true;
    }, 500);

    welcome.hidden = true;

    input.disabled = false;
    send.disabled = false;

    input.focus();
  } catch (error) {
    engine = null;

    status.textContent =
      "Local AI · WebGPU unavailable";

    progressText.textContent =
      "The GPU runtime could not initialize.";

    errorBox.textContent = [
      "WEBLLM / MLC INITIALIZATION ERROR",
      "",
      formatError(error),
      "",
      "Secure context: " +
        (hardware?.secureContext ? "yes" : "no"),
      "WebGPU API: " +
        (hardware?.webgpuApi ? "available" : "not available"),
      "GPU adapter: " +
        (hardware?.adapter ? "available" : "not available"),
      "Browser: " + navigator.userAgent,
      "",
      "This build uses WebLLM/MLC only. " +
        "It does not fall back to the ONNX/WASM runtime " +
        "that was crashing on iPhone.",
    ].join("\n");

    errorBox.hidden = false;

    loadButton.disabled = false;

    console.error(
      "Pocket AI WebLLM initialization failed:",
      error
    );
  }
}

async function inspectHardware() {
  const result = {
    secureContext: window.isSecureContext,
    webgpuApi: "gpu" in navigator,
    adapter: null,
    vendor: "unknown GPU",
    label: "WebGPU",
    maxBuffer: 0,
    features: [],
  };

  if (!result.webgpuApi) {
    return result;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });

    if (!adapter) {
      return result;
    }

    result.adapter = true;
    result.features = [...adapter.features];

    result.maxBuffer =
      adapter.limits?.maxStorageBufferBindingSize ?? 0;

    const info = adapter.info;

    const parts = [
      info?.vendor,
      info?.architecture,
      info?.description,
    ].filter(Boolean);

    if (parts.length) {
      result.label = parts.join(" · ");
    }
  } catch (error) {
    console.warn(
      "WebGPU adapter inspection failed:",
      error
    );
  }

  return result;
}

function renderHardware(info) {
  if (!info) {
    return;
  }

  const maxBufferMB = info.maxBuffer
    ? Math.round(info.maxBuffer / 1024 / 1024)
    : 0;

  hardwareBox.textContent = [
    "Connection: " +
      (info.secureContext
        ? "HTTPS / secure"
        : "HTTP / not secure"),

    "WebGPU API: " +
      (info.webgpuApi ? "yes" : "no"),

    "GPU adapter: " +
      (info.adapter ? "yes" : "no"),

    "GPU: " + info.label,

    info.vendor && info.vendor !== "unknown GPU"
      ? "MLC vendor: " + info.vendor
      : null,

    maxBufferMB
      ? "Max storage buffer: " + maxBufferMB + " MB"
      : null,
  ]
    .filter(Boolean)
    .join("\n");
}

async function safeGPUVendor(mlcEngine) {
  try {
    return await mlcEngine.getGPUVendor();
  } catch {
    return "";
  }
}

function formatError(error) {
  if (!error) {
    return "Unknown error";
  }

  if (error.stack) {
    return error.stack;
  }

  if (error.message) {
    return error.message;
  }

  return String(error);
}

function addMessage(role, content) {
  const bubble = document.createElement("div");

  bubble.className = "message " + role;
  bubble.textContent = content;

  chat.appendChild(bubble);

  messages.push({
    role,
    content,
  });

  scrollToBottom();

  return bubble;
}

function renderMessages() {
  for (const message of messages) {
    const bubble = document.createElement("div");

    bubble.className = "message " + message.role;
    bubble.textContent = message.content;

    chat.appendChild(bubble);
  }

  scrollToBottom();
}

function loadMessages() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);

    if (!stored) {
      return [];
    }

    const parsed = JSON.parse(stored);

    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed.filter(
      (message) =>
        message &&
        (message.role === "user" ||
          message.role === "assistant") &&
        typeof message.content === "string"
    );
  } catch {
    return [];
  }
}

function saveMessages() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify(messages.slice(-100))
  );
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chat.scrollTop = chat.scrollHeight;
  });
}
