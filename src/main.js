import {
  CreateMLCEngine,
  prebuiltAppConfig,
} from "@mlc-ai/web-llm";
import "./style.css";

const STORAGE_KEY = "pocket-ai-messages-v3";

const MODELS = {
  qwen: {
    id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
    label: "Qwen 0.5B",
    description: "Working fallback · standard WebLLM",
    type: "standard",
  },

  bonsai: {
    id: "Bonsai-1.7B-q1-MLC",
    label: "Bonsai 1.7B Q1",
    description: "Experimental · bonsai_q1_f32",
    type: "bonsai-q1",

    model:
      
"https://huggingface.co/welcoma/Bonsai-1.7B-bonsai_q1_f32-MLC/resolve/main/",

    model_lib:
      
"https://huggingface.co/welcoma/Bonsai-1.7B-bonsai_q1_f32-MLC/resolve/main/libs/bonsai-q1-1.7b-bonsai_q1_f32-webgpu.wasm",

    overrides: {
      context_window_size: 4096,
      prefill_chunk_size: 512,
    },
  },
};

let selectedModel = "qwen";
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

modelButton.addEventListener("click", showModelSelector);
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

  addMessage("user", text);
  saveMessages();

  const conversation = [
    {
      role: "system",
      content:
        "You are a helpful, concise assistant running locally on the user's device. Answer naturally and honestly.",
    },
    ...messages.slice(-14),
  ];

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

    /*
     * The assistant placeholder was already pushed into messages
     * by addMessage(). Replace its empty content with the final
     * generated response instead of creating a duplicate entry.
     */
    const lastMessage = messages[messages.length - 1];

    if (
      lastMessage &&
      lastMessage.role === "assistant" &&
      lastMessage.content === ""
    ) {
      lastMessage.content = generated.trim();
    }

    saveMessages();
  } catch (error) {
    assistant.textContent =
      "Local generation error:\n\n" + formatError(error);

    console.error(
      "Pocket AI generation failed:",
      error
    );

    /*
     * Remove the empty assistant placeholder if generation failed.
     */
    const lastMessage = messages[messages.length - 1];

    if (
      lastMessage &&
      lastMessage.role === "assistant" &&
      lastMessage.content === ""
    ) {
      messages.pop();
    }

    saveMessages();
  } finally {
    busy = false;

    input.disabled = false;
    send.disabled = false;

    input.focus();
  }
});

async function showModelSelector() {
  if (engine) {
    alert(
      "Refresh the page before switching models.\n\n" +
        "Current model: " +
        MODELS[selectedModel].label
    );

    return;
  }

  const choice = prompt(
    "Choose model:\n\n" +
      "1 — Qwen 0.5B (working fallback)\n" +
      "2 — Bonsai 1.7B Q1 (experimental)\n\n" +
      "Enter 1 or 2:"
  );

  if (choice === "1") {
    selectedModel = "qwen";
  } else if (choice === "2") {
    selectedModel = "bonsai";
  } else {
    return;
  }

  modelButton.textContent =
    MODELS[selectedModel].label;

  status.textContent =
    "Local AI · " +
    MODELS[selectedModel].label;

  progressText.textContent =
    "Selected " +
    MODELS[selectedModel].label +
    ".";
}

function buildAppConfig() {
  const model = MODELS[selectedModel];

  /*
   * Keep the existing Qwen configuration untouched.
   */
  if (selectedModel === "qwen") {
    return {
      ...prebuiltAppConfig,
      cacheBackend: "indexeddb",
    };
  }

  /*
   * Bonsai Q1 custom model record.
   *
   * The published Bonsai artifact specifies this exact model,
   * model_id, model_lib and browser-smoke-test overrides.
   *
   * IMPORTANT:
   * This requires a WebLLM runtime containing the
   * `bonsai_q1_f32` runtime path.
   */
  return {
    ...prebuiltAppConfig,

    cacheBackend: "indexeddb",

    model_list: [
      {
        model: model.model,
        model_id: model.id,
        model_lib: model.model_lib,

        overrides: {
          context_window_size: 4096,
          prefill_chunk_size: 512,
        },
      },
    ],
  };
}

async function loadModel() {
  if (engine) {
    return;
  }

  loadButton.disabled = true;

  progressWrap.hidden = false;
  errorBox.hidden = true;

  progress.style.width = "0%";

  const model = MODELS[selectedModel];

  status.textContent =
    "Local AI · checking device...";

  progressText.textContent =
    "Checking WebGPU and device capabilities...";

  try {
    hardware = await inspectHardware();

    renderHardware(hardware);

    if (!hardware.secureContext) {
      throw new Error(
        "WebGPU requires a secure context (HTTPS). " +
          "Open Pocket AI over HTTPS."
      );
    }

    if (!hardware.webgpuApi) {
      throw new Error(
        "WebGPU is not exposed by this browser. " +
          "WebLLM requires WebGPU."
      );
    }

    if (!hardware.adapter) {
      throw new Error(
        "WebGPU is exposed, but Safari could not provide " +
          "a GPU adapter."
      );
    }

    status.textContent =
      "Local AI · WebGPU · " +
      hardware.label;

    progressText.textContent =
      "GPU available · loading " +
      model.label +
      "...";

    /*
     * IMPORTANT:
     *
     * We intentionally DO NOT block Bonsai here.
     *
     * The previous build contained a manual error that stopped
     * Bonsai before WebLLM even attempted to load it.
     *
     * Now we let the actual runtime tell us what is unsupported.
     */
    engine = await CreateMLCEngine(
      model.id,
      {
        appConfig: buildAppConfig(),

        initProgressCallback: (info) => {
          if (info?.progress != null) {
            const pct = Math.max(
              0,
              Math.min(
                100,
                Number(info.progress) * 100
              )
            );

            progress.style.width =
              pct + "%";
          }

          progressText.textContent =
            info?.text ||
            "Preparing local GPU runtime...";
        },
      }
    );

    const vendor =
      await safeGPUVendor(engine);

    if (vendor) {
      hardware.vendor = vendor;
      renderHardware(hardware);
    }

    status.textContent =
      "Local AI · " +
      model.label +
      " · WebGPU";

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
      "Local AI · model unavailable";

    progressText.textContent =
      "The selected model could not initialize.";

    /*
     * Capture as much diagnostic information as possible.
     * This is especially important for Bonsai because we need
     * to see the actual runtime error rather than only:
     *
     *   loadModel@main.js...
     */

    const diagnostics = [
      "WEBLLM / MLC INITIALIZATION ERROR",
      "",
      "Selected model: " +
        model.label,

      "Model ID: " +
        model.id,

      "Model type: " +
        model.type,

      "",

      "ERROR:",
      formatError(error),

      "",

      "DEVICE:",
      "Secure context: " +
        (hardware?.secureContext
          ? "yes"
          : "no"),

      "WebGPU API: " +
        (hardware?.webgpuApi
          ? "yes"
          : "no"),

      "GPU adapter: " +
        (hardware?.adapter
          ? "yes"
          : "no"),

      "GPU: " +
        (hardware?.label ||
          "unknown"),

      "Max storage buffer: " +
        (hardware?.maxBuffer
          ? Math.round(
              hardware.maxBuffer /
                1024 /
                1024
            ) + " MB"
          : "unknown"),

      "",

      "RUNTIME:",
      "@mlc-ai/web-llm",

      "",

      "BROWSER:",
      navigator.userAgent,
    ];

    if (selectedModel === "bonsai") {
      diagnostics.push(
        "",
        "BONSAI:",
        "Quantization: bonsai_q1_f32",
        "Architecture: Qwen3-shaped",
        "Context override: 4096",
        "Prefill override: 512",
        "",
        "The published Bonsai Q1 artifact requires",
        "a WebLLM/MLC runtime containing Bonsai Q1",
        "support. Stock upstream WebLLM may reject",
        "this model before generation."
      );
    }

    errorBox.textContent =
      diagnostics.join("\n");

    errorBox.hidden = false;

    loadButton.disabled = false;

    console.error(
      "Pocket AI WebLLM initialization failed:",
      error
    );

    /*
     * Also log the original object so Safari's console,
     * when available, can expose nested error properties.
     */
    console.error(
      "Full initialization error object:",
      error
    );
  }
}

async function inspectHardware() {
  const result = {
    secureContext:
      window.isSecureContext,

    webgpuApi:
      "gpu" in navigator,

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
    const adapter =
      await navigator.gpu.requestAdapter({
        powerPreference:
          "high-performance",
      });

    if (!adapter) {
      return result;
    }

    result.adapter = true;

    result.features =
      [...adapter.features];

    result.maxBuffer =
      adapter.limits
        ?.maxStorageBufferBindingSize ??
      0;

    const info =
      adapter.info;

    const parts = [
      info?.vendor,
      info?.architecture,
      info?.description,
    ].filter(Boolean);

    if (parts.length) {
      result.label =
        parts.join(" · ");
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

  const maxBufferMB =
    info.maxBuffer
      ? Math.round(
          info.maxBuffer /
            1024 /
            1024
        )
      : 0;

  hardwareBox.textContent = [
    "Connection: " +
      (info.secureContext
        ? "HTTPS / secure"
        : "HTTP / not secure"),

    "WebGPU API: " +
      (info.webgpuApi
        ? "yes"
        : "no"),

    "GPU adapter: " +
      (info.adapter
        ? "yes"
        : "no"),

    "GPU: " +
      info.label,

    info.vendor &&
    info.vendor !==
      "unknown GPU"
      ? "MLC vendor: " +
        info.vendor
      : null,

    maxBufferMB
      ? "Max storage buffer: " +
        maxBufferMB +
        " MB"
      : null,
  ]
    .filter(Boolean)
    .join("\n");
}

async function safeGPUVendor(
  mlcEngine
) {
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

  try {
    return JSON.stringify(
      error,
      Object.getOwnPropertyNames(error),
      2
    );
  } catch {
    return String(error);
  }
}

function addMessage(
  role,
  content
) {
  const bubble =
    document.createElement(
      "div"
    );

  bubble.className =
    "message " + role;

  bubble.textContent =
    content;

  chat.appendChild(
    bubble
  );

  messages.push({
    role,
    content,
  });

  scrollToBottom();

  return bubble;
}

function renderMessages() {
  for (const message of messages) {
    const bubble =
      document.createElement(
        "div"
      );

    bubble.className =
      "message " +
      message.role;

    bubble.textContent =
      message.content;

    chat.appendChild(
      bubble
    );
  }

  scrollToBottom();
}

function loadMessages() {
  try {
    const stored =
      localStorage.getItem(
        STORAGE_KEY
      );

    if (!stored) {
      return [];
    }

    const parsed =
      JSON.parse(stored);

    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed.filter(
      (message) =>
        message &&
        (
          message.role ===
            "user" ||
          message.role ===
            "assistant"
        ) &&
        typeof message.content ===
          "string"
    );
  } catch {
    return [];
  }
}

function saveMessages() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify(
      messages.slice(-100)
    )
  );
}

function scrollToBottom() {
  requestAnimationFrame(
    () => {
      chat.scrollTop =
        chat.scrollHeight;
    }
  );
}
