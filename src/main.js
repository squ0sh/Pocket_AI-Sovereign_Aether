import {
  CreateMLCEngine,
  prebuiltAppConfig,
} from "@mlc-ai/web-llm";
import "./style.css";

const STORAGE_KEY = "pocket-ai-chats-v3";
const LEGACY_STORAGE_KEY = "pocket-ai-messages-v2";
const MODEL_STORAGE_KEY = "pocket-ai-selected-model-v1";

const MODEL_OPTIONS = [
  { id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", name: "Qwen 0.5B", tier: "Fast", description: "Smallest option · best for speed and battery." },
  { id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC", name: "Qwen 1.5B", tier: "Balanced", description: "More capable while still designed for low-resource devices." },
  { id: "Qwen2.5-3B-Instruct-q4f16_1-MLC", name: "Qwen 3B", tier: "Powerful", description: "A larger Qwen model for stronger general responses." },
  { id: "Hermes-3-Llama-3.2-3B-q4f16_1-MLC", name: "Hermes 3 · 3B", tier: "Conversational", description: "3B conversational model with a different personality and behavior." },
];

let selectedModelId = loadSelectedModel();

let engine = null;
let hardware = null;
let chats = loadChats();
let activeChatId = chats[0]?.id ?? null;
let busy = false;
let switchingModel = false;

const chat = document.querySelector("#chat");
const welcome = document.querySelector("#welcome");
const loadButton = document.querySelector("#loadButton");
const modelButton = document.querySelector("#modelButton");
const modelSheet = document.querySelector("#modelSheet");
const closeModelButton = document.querySelector("#closeModelButton");
const modelList = document.querySelector("#modelList");
const composer = document.querySelector("#composer");
const input = document.querySelector("#input");
const send = document.querySelector("#send");
const status = document.querySelector("#status");
const progressWrap = document.querySelector("#progressWrap");
const progress = document.querySelector("#progress");
const progressText = document.querySelector("#progressText");
const errorBox = document.querySelector("#errorBox");
const hardwareBox = document.querySelector("#hardwareBox");

const menuButton = document.querySelector("#menuButton");
const closeDrawerButton = document.querySelector("#closeDrawerButton");
const drawer = document.querySelector("#historyDrawer");
const drawerBackdrop = document.querySelector("#drawerBackdrop");
const newChatButton = document.querySelector("#newChatButton");
const historyList = document.querySelector("#historyList");

updateModelButton();
renderModelList();
renderHistory();
renderActiveChat();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch((error) => {
      console.warn("Service worker registration failed:", error);
    });
  });
}

modelButton.addEventListener("click", openModelSheet);

loadButton.addEventListener("click", () => loadModel(selectedModelId));
closeModelButton.addEventListener("click", closeModelSheet);
modelSheet.addEventListener("click", (event) => {
  if (event.target === modelSheet) closeModelSheet();
});

menuButton.addEventListener("click", openDrawer);
closeDrawerButton.addEventListener("click", closeDrawer);
drawerBackdrop.addEventListener("click", closeDrawer);

newChatButton.addEventListener("click", () => {
  createNewChat();
  closeDrawer();
});

composer.addEventListener("submit", sendMessage);

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 130) + "px";
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

function getActiveChat() {
  return chats.find((item) => item.id === activeChatId) ?? null;
}

function createNewChat() {
  const now = Date.now();

  const newChat = {
    id: crypto.randomUUID(),
    title: "New Chat",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };

  chats.unshift(newChat);
  activeChatId = newChat.id;

  saveChats();
  renderHistory();
  renderActiveChat();

  if (engine) {
    input.disabled = false;
    send.disabled = false;
    input.focus();
  }
}

function selectChat(id) {
  if (busy) {
    return;
  }

  if (!chats.some((item) => item.id === id)) {
    return;
  }

  activeChatId = id;
  saveChats();
  renderHistory();
  renderActiveChat();
  closeDrawer();

  if (engine) {
    input.disabled = false;
    send.disabled = false;
    input.focus();
  }
}

function deleteChat(id) {
  if (busy) {
    return;
  }

  const index = chats.findIndex((item) => item.id === id);

  if (index === -1) {
    return;
  }

  const deleted = chats[index];

  if (
    deleted.messages.length > 0 &&
    !confirm(`Delete "${deleted.title}"?`)
  ) {
    return;
  }

  chats.splice(index, 1);

  if (chats.length === 0) {
    createNewChat();
    return;
  }

  if (activeChatId === id) {
    activeChatId = chats[0].id;
  }

  saveChats();
  renderHistory();
  renderActiveChat();
}

function renderHistory() {
  historyList.replaceChildren();

  if (!chats.length) {
    const empty = document.createElement("div");
    empty.className = "empty-history";
    empty.textContent = "No conversations yet.";
    historyList.appendChild(empty);
    return;
  }

  const sorted = [...chats].sort(
    (a, b) => b.updatedAt - a.updatedAt
  );

  for (const item of sorted) {
    const row = document.createElement("div");
    row.className =
      "history-item" +
      (item.id === activeChatId ? " active" : "");

    const main = document.createElement("button");
    main.type = "button";
    main.className = "history-item-main";
    main.addEventListener("click", () => selectChat(item.id));

    const title = document.createElement("span");
    title.className = "history-title";
    title.textContent = item.title || "New Chat";

    const time = document.createElement("span");
    time.className = "history-time";
    time.textContent = formatHistoryTime(item.updatedAt);

    main.append(title, time);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "history-delete";
    remove.setAttribute("aria-label", `Delete ${item.title || "chat"}`);
    remove.textContent = "×";
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteChat(item.id);
    });

    row.append(main, remove);
    historyList.appendChild(row);
  }
}

function renderActiveChat() {
  const active = getActiveChat();

  chat.replaceChildren();

  if (!active || active.messages.length === 0) {
    chat.appendChild(welcome);
    welcome.hidden = false;
    return;
  }

  welcome.hidden = true;

  for (const message of active.messages) {
    const bubble = createMessageBubble(
      message.role,
      message.content
    );
    chat.appendChild(bubble);
  }

  scrollToBottom();
}

async function sendMessage(event) {
  event.preventDefault();

  const text = input.value.trim();

  if (!text || !engine || busy) {
    return;
  }

  let active = getActiveChat();

  if (!active) {
    createNewChat();
    active = getActiveChat();
  }

  if (!active) {
    return;
  }

  busy = true;

  input.value = "";
  input.style.height = "auto";
  input.disabled = true;
  send.disabled = true;

  const userMessage = {
    role: "user",
    content: text,
  };

  active.messages.push(userMessage);

  if (active.title === "New Chat") {
    active.title = makeChatTitle(text);
  }

  active.updatedAt = Date.now();

  // Render the user's message immediately. The previous version stored it
  // correctly but forgot to create the visible user bubble.
  const userBubble = createMessageBubble("user", text);
  chat.appendChild(userBubble);
  scrollToBottom();

  saveChats();
  renderHistory();

  // Build the model context from real stored messages only.
  const conversation = [
    {
      role: "system",
      content:
        "You are a helpful, concise assistant running locally on the user's device. Answer naturally and honestly.",
    },
    ...active.messages.slice(-14),
  ];

  // Visual streaming bubble only. It is never stored as an empty message.
  const assistant = createMessageBubble("assistant", "");
  assistant.classList.add("streaming");
  chat.appendChild(assistant);
  scrollToBottom();

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

      if (!delta) {
        continue;
      }

      generated += delta;

      // WebLLM supplies streamed chunks. The browser paints them
      // immediately, giving the token-by-token typing effect.
      assistant.textContent = generated;
      scrollToBottom();
    }

    assistant.classList.remove("streaming");

    active.messages.push({
      role: "assistant",
      content: generated.trim(),
    });

    active.updatedAt = Date.now();

    saveChats();
    renderHistory();
  } catch (error) {
    assistant.classList.remove("streaming");
    assistant.textContent =
      "Local generation error:\n" + formatError(error);

    console.error("Pocket AI generation failed:", error);

    saveChats();
  } finally {
    busy = false;

    input.disabled = false;
    send.disabled = false;

    input.focus();
  }
}

function getSelectedModelOption() {
  return MODEL_OPTIONS.find((item) => item.id === selectedModelId) ?? MODEL_OPTIONS[0];
}

function updateModelButton() {
  modelButton.textContent = getSelectedModelOption().name;
}

function renderModelList() {
  modelList.replaceChildren();
  for (const option of MODEL_OPTIONS) {
    const record = prebuiltAppConfig.model_list.find((item) => item.model_id === option.id);
    if (!record) continue;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "model-option" + (option.id === selectedModelId ? " active" : "");
    button.disabled = switchingModel;
    const main = document.createElement("span");
    main.className = "model-option-main";
    const name = document.createElement("span");
    name.className = "model-option-name";
    name.textContent = option.name;
    const description = document.createElement("span");
    description.className = "model-option-description";
    description.textContent = option.description;
    const meta = document.createElement("span");
    meta.className = "model-option-meta";
    const memory = Math.round(record.vram_required_MB ?? 0);
    meta.textContent = memory ? `Estimated GPU memory · ${memory} MB` : "GPU memory estimate unavailable";
    main.append(name, description, meta);
    const badge = document.createElement("span");
    badge.className = "model-option-badge";
    badge.textContent = option.tier;
    button.append(main, badge);
    button.addEventListener("click", () => selectModel(option.id));
    modelList.appendChild(button);
  }
}

function openModelSheet() {
  renderModelList();
  modelSheet.classList.add("open");
  modelSheet.setAttribute("aria-hidden", "false");
}

function closeModelSheet() {
  modelSheet.classList.remove("open");
  modelSheet.setAttribute("aria-hidden", "true");
}

async function selectModel(modelId) {
  if (busy || switchingModel) return;
  if (modelId === selectedModelId && engine) {
    closeModelSheet();
    return;
  }
  selectedModelId = modelId;
  saveSelectedModel();
  updateModelButton();
  closeModelSheet();
  await loadModel(modelId);
}

async function loadModel(modelId = selectedModelId) {
  if (busy || switchingModel) return;

  selectedModelId = modelId;
  saveSelectedModel();
  updateModelButton();
  switchingModel = true;
  renderModelList();

  loadButton.disabled = true;
  progressWrap.hidden = false;
  errorBox.hidden = true;
  progress.style.width = "0%";
  status.textContent = `Local AI · ${getSelectedModelOption().name} · checking device...`;
  progressText.textContent = "Checking WebGPU and device capabilities...";

  try {
    hardware = await inspectHardware();
    renderHardware(hardware);

    if (!hardware.secureContext) {
      throw new Error("WebGPU requires a secure context (HTTPS). This address is HTTP, so WebLLM cannot access the GPU here. On the iPhone, open Pocket AI over HTTPS before loading the model.");
    }

    if (!hardware.webgpuApi) {
      throw new Error("WebGPU is not exposed by this browser. WebLLM requires WebGPU; there is intentionally no ONNX/WASM fallback in this build because that runtime was crashing on iPhone.");
    }

    if (!hardware.adapter) {
      throw new Error("WebGPU is exposed, but Safari could not provide a GPU adapter. The browser/device combination cannot run WebLLM right now.");
    }

    status.textContent = `Local AI · ${getSelectedModelOption().name} · WebGPU · ${hardware.label}`;
    progressText.textContent = `GPU available · loading ${getSelectedModelOption().name}...`;

    const engineConfig = {
      appConfig: {
        ...prebuiltAppConfig,
        cacheBackend: "indexeddb",
      },
      initProgressCallback: (info) => {
        if (info?.progress != null) {
          const pct = Math.max(0, Math.min(100, Number(info.progress) * 100));
          progress.style.width = `${pct}%`;
        }
        progressText.textContent = info?.text || "Preparing local GPU runtime...";
      },
    };

    if (engine) {
      // WebLLM supports switching the loaded model through reload().
      await engine.reload(selectedModelId);
    } else {
      engine = await CreateMLCEngine(selectedModelId, engineConfig);
    }

    const vendor = await safeGPUVendor(engine);
    if (vendor) {
      hardware.vendor = vendor;
      renderHardware(hardware);
    }

    status.textContent = `Local AI · ${getSelectedModelOption().name} · WebGPU · ${hardware.label}`;
    progress.style.width = "100%";
    progressText.textContent = "Ready · model is running on this device.";
    setTimeout(() => (progressWrap.hidden = true), 500);
    welcome.hidden = true;
    input.disabled = false;
    send.disabled = false;
    input.focus();
  } catch (error) {
    engine = null;
    status.textContent = `Local AI · ${getSelectedModelOption().name} · WebGPU unavailable`;
    progressText.textContent = "The GPU runtime could not initialize.";
    errorBox.textContent = [
      "WEBLLM / MLC INITIALIZATION ERROR",
      "",
      formatError(error),
      "",
      `Secure context: ${hardware?.secureContext ? "yes" : "no"}`,
      `WebGPU API: ${hardware?.webgpuApi ? "available" : "not available"}`,
      `GPU adapter: ${hardware?.adapter ? "available" : "not available"}`,
      `Model: ${selectedModelId}`,
      `Browser: ${navigator.userAgent}`,
      "",
      "This build uses WebLLM/MLC only. It does not fall back to the ONNX/WASM runtime that was crashing on iPhone.",
    ].join("\n");
    errorBox.hidden = false;
    loadButton.disabled = false;
    console.error("Pocket AI WebLLM initialization failed:", error);
  } finally {
    switchingModel = false;
    loadButton.disabled = !!engine;
    renderModelList();
    updateModelButton();
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

function createMessageBubble(role, content) {
  const bubble = document.createElement("div");

  bubble.className = "message " + role;
  bubble.textContent = content;

  return bubble;
}

function makeChatTitle(text) {
  const cleaned = text.replace(/\s+/g, " ").trim();

  if (!cleaned) {
    return "New Chat";
  }

  return cleaned.length > 42
    ? cleaned.slice(0, 42).trimEnd() + "…"
    : cleaned;
}

function formatHistoryTime(timestamp) {
  const date = new Date(timestamp);
  const now = new Date();

  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();

  if (sameDay) {
    return date.toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit",
    });
  }

  const sameYear = date.getFullYear() === now.getFullYear();

  if (sameYear) {
    return date.toLocaleDateString([], {
      month: "short",
      day: "numeric",
    });
  }

  return date.toLocaleDateString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function openDrawer() {
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  menuButton.setAttribute("aria-expanded", "true");
  drawerBackdrop.hidden = false;

  requestAnimationFrame(() => {
    drawerBackdrop.style.opacity = "1";
  });
}

function closeDrawer() {
  drawer.classList.remove("open");
  drawer.setAttribute("aria-hidden", "true");
  menuButton.setAttribute("aria-expanded", "false");
  drawerBackdrop.hidden = true;
}

function loadSelectedModel() {
  try {
    const saved = localStorage.getItem(MODEL_STORAGE_KEY);
    if (saved && MODEL_OPTIONS.some((option) => option.id === saved)) return saved;
  } catch (error) {
    console.warn("Pocket AI selected model could not be loaded:", error);
  }
  return MODEL_OPTIONS[0].id;
}

function saveSelectedModel() {
  try {
    localStorage.setItem(MODEL_STORAGE_KEY, selectedModelId);
  } catch (error) {
    console.warn("Pocket AI selected model could not be saved:", error);
  }
}

function loadChats() {
  try {
    const current = localStorage.getItem(STORAGE_KEY);

    if (current) {
      const parsed = JSON.parse(current);

      if (Array.isArray(parsed)) {
        return normalizeChats(parsed);
      }
    }

    // One-time migration from the previous single-thread storage format.
    const legacy = localStorage.getItem(LEGACY_STORAGE_KEY);

    if (legacy) {
      const parsedLegacy = JSON.parse(legacy);

      if (Array.isArray(parsedLegacy) && parsedLegacy.length) {
        const now = Date.now();

        const migrated = {
          id: crypto.randomUUID(),
          title: makeChatTitle(
            parsedLegacy.find((item) => item.role === "user")?.content ||
              "Previous Chat"
          ),
          createdAt: now,
          updatedAt: now,
          messages: parsedLegacy.filter(
            (message) =>
              message &&
              (message.role === "user" ||
                message.role === "assistant") &&
              typeof message.content === "string" &&
              message.content.trim() !== ""
          ),
        };

        const result = [migrated];
        localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify(result)
        );

        return result;
      }
    }
  } catch (error) {
    console.warn("Pocket AI chat history could not be loaded:", error);
  }

  return [
    {
      id: crypto.randomUUID(),
      title: "New Chat",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    },
  ];
}

function normalizeChats(items) {
  return items
    .filter((item) => item && typeof item === "object")
    .map((item) => ({
      id: item.id || crypto.randomUUID(),
      title: item.title || "New Chat",
      createdAt: Number(item.createdAt) || Date.now(),
      updatedAt: Number(item.updatedAt) || Date.now(),
      messages: Array.isArray(item.messages)
        ? item.messages.filter(
            (message) =>
              message &&
              (message.role === "user" ||
                message.role === "assistant") &&
              typeof message.content === "string" &&
              message.content.trim() !== ""
          )
        : [],
    }));
}

function saveChats() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify(chats)
  );
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chat.scrollTop = chat.scrollHeight;
  });
}
