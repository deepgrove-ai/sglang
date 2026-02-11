// ── Configuration ────────────────────────────────────────────────
// If served via the reverse proxy (serve.py), API is same-origin.
// If served via plain http.server, fall back to SGLang port on same host.
const SGLANG_PORT = 30080;
const isSameOriginProxy = window.location.port !== "4173"; // plain dev server uses 4173
const AUTO_API_BASE = isSameOriginProxy
  ? window.location.origin
  : `${window.location.protocol}//${window.location.hostname}:${SGLANG_PORT}`;

const DEFAULT_MODEL_ID = "mangrove-alltern-overlap";
const SYSTEM_PROMPT =
  "You are a helpful assistant. Reply with concise Markdown. Use LaTeX for math when useful.";

const URL_PARAMS = new URLSearchParams(window.location.search);
const API_BASE = (URL_PARAMS.get("api") || AUTO_API_BASE).replace(/\/+$/, "");
const PREFERRED_MODEL_ID = (URL_PARAMS.get("model") || DEFAULT_MODEL_ID).trim();

const state = {
  busy: false,
  modelId: null,
  sessionStarted: false,
  messages: [{ role: "system", content: SYSTEM_PROMPT }],
  abortController: null,
};

// ── DOM refs ─────────────────────────────────────────────────────
const appShell = document.querySelector(".app-shell");
const chatLog = document.getElementById("chat-log");
const composer = document.getElementById("composer");
const promptInput = document.getElementById("prompt-input");
const sendBtn = document.getElementById("send-btn");
const resetBtn = document.getElementById("reset-chat");
const typingTemplate = document.getElementById("typing-template");

if (typeof marked !== "undefined") {
  marked.setOptions({ gfm: true, breaks: true });
}

initialize();

async function initialize() {
  bindEvents();
  autoResizeInput();
  setSessionStarted(false);
  promptInput.focus();
}

function bindEvents() {
  composer.addEventListener("submit", handleSubmit);
  promptInput.addEventListener("input", autoResizeInput);
  promptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      composer.requestSubmit();
    }
  });

  resetBtn.addEventListener("click", () => {
    if (state.busy) {
      if (state.abortController) {
        state.abortController.abort();
      }
      return;
    }
    state.messages = [{ role: "system", content: SYSTEM_PROMPT }];
    state.sessionStarted = false;
    chatLog.innerHTML = "";
    promptInput.value = "";
    autoResizeInput();
    setSessionStarted(false);
    promptInput.focus();
  });

  chatLog.addEventListener("click", async (event) => {
    const copyBtn = event.target.closest(".copy-code-btn");
    if (!copyBtn) return;

    const codeBlock = copyBtn.closest(".code-wrap")?.querySelector("pre code");
    if (!codeBlock) return;

    const originalLabel = copyBtn.textContent;
    try {
      await navigator.clipboard.writeText(codeBlock.textContent || "");
      copyBtn.textContent = "Copied!";
      setTimeout(() => { copyBtn.textContent = originalLabel; }, 1500);
    } catch (error) {
      copyBtn.textContent = "Failed";
      setTimeout(() => { copyBtn.textContent = originalLabel; }, 1500);
      console.error(error);
    }
  });
}

// ── Submit handler ───────────────────────────────────────────────
async function handleSubmit(event) {
  event.preventDefault();
  if (state.busy) return;

  const userText = promptInput.value.trim();
  if (!userText) return;

  if (!state.sessionStarted) setSessionStarted(true);

  promptInput.value = "";
  autoResizeInput();
  addUserMessage(userText);
  state.messages.push({ role: "user", content: userText });

  const bubble = addTypingBubble();
  setBusy(true);

  try {
    let assistantReply;
    let wasStreamed = false;
    try {
      assistantReply = await streamAssistantReply(bubble);
      wasStreamed = true;
      state.messages.push({ role: "assistant", content: assistantReply });
    } catch (error) {
      if (error.name === "AbortError") {
        assistantReply = "*Request cancelled.*";
      } else {
        console.error("[RedMod] Chat error:", error);
        assistantReply =
          "Could not reach the model.\n\n```text\n" +
          error.message +
          "\n```\n\nAPI: `" + API_BASE + "`";
      }
    }

    if (wasStreamed) {
      finalizeStreamedBubble(bubble);
    } else {
      paintAssistantBubble(bubble, assistantReply, false);
    }
  } catch (error) {
    paintAssistantBubble(bubble,
      "Something went wrong.\n\n```text\n" + error.message + "\n```", false);
  } finally {
    setBusy(false);
    state.abortController = null;
    promptInput.focus();
  }
}

// ── State helpers ────────────────────────────────────────────────
function setSessionStarted(started) {
  state.sessionStarted = started;
  appShell.classList.toggle("is-home", !started);
}

function setBusy(isBusy) {
  state.busy = isBusy;
  sendBtn.disabled = isBusy;
  promptInput.disabled = isBusy;
  resetBtn.textContent = isBusy ? "Stop" : "New chat";
}

// ── Model resolution ─────────────────────────────────────────────
async function resolveModel() {
  if (state.modelId) return state.modelId;

  const response = await fetch(`${API_BASE}/v1/models`);
  if (!response.ok) {
    throw new Error(`Model discovery failed (${response.status}). API: ${API_BASE}`);
  }

  const payload = await response.json();
  const models = Array.isArray(payload?.data) ? payload.data : [];
  const match = models.find((m) => m?.id === PREFERRED_MODEL_ID);
  const modelId = match?.id || models[0]?.id;
  if (!modelId) throw new Error("No models returned by /v1/models.");

  state.modelId = modelId;
  return modelId;
}

// ── Streaming chat ───────────────────────────────────────────────
async function streamAssistantReply(bubble) {
  const modelId = await resolveModel();
  const controller = new AbortController();
  state.abortController = controller;

  const response = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: modelId,
      messages: state.messages,
      temperature: 0.7,
      top_p: 0.9,
      frequency_penalty: 0.3,
      presence_penalty: 0.3,
      stream: true,
    }),
    signal: controller.signal,
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Chat failed (${response.status}): ${trimText(body, 280)}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullText = "";
  let buffer = "";

  // Clear thinking indicator
  bubble.innerHTML = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith("data:")) continue;
      const data = trimmed.slice(5).trim();
      if (data === "[DONE]") continue;

      try {
        const parsed = JSON.parse(data);
        const delta = parsed?.choices?.[0]?.delta?.content;
        if (typeof delta === "string" && delta.length > 0) {
          fullText += delta;
          renderStreamChunk(bubble, fullText);
          scrollToBottom();
        }
      } catch {
        // skip malformed chunks
      }
    }
  }

  if (!fullText.trim()) {
    throw new Error("Assistant response was empty.");
  }

  return fullText;
}

// ── Rendering helpers ────────────────────────────────────────────
let _mathRenderTimer = null;
function renderStreamChunk(bubble, text) {
  bubble.innerHTML = markdownToHtml(text);
  decorateCodeBlocks(bubble);
  // Throttle math rendering during streaming (every 500ms)
  if (!_mathRenderTimer) {
    _mathRenderTimer = setTimeout(() => {
      renderMath(bubble);
      _mathRenderTimer = null;
    }, 500);
  }
}

function finalizeStreamedBubble(bubble) {
  clearTimeout(_mathRenderTimer);
  _mathRenderTimer = null;
  decorateCodeBlocks(bubble);
  renderMath(bubble);
  scrollToBottom();
}

function paintAssistantBubble(bubble, markdown, animate) {
  bubble.innerHTML = markdownToHtml(markdown);
  decorateCodeBlocks(bubble);
  renderMath(bubble);
  if (animate) applyWordFadeIn(bubble);
  scrollToBottom();
}

function markdownToHtml(markdown) {
  // Protect math blocks from being mangled by the Markdown parser.
  // We extract them, replace with placeholders, parse MD, then restore.
  const mathBlocks = [];
  const PLACEHOLDER = (i) => `%%MATH_${i}%%`;

  // Order matters: longer/greedier patterns first.
  // 1. Display math: $$ ... $$ and \[ ... \]
  // 2. Inline math:  $ ... $  and \( ... \)
  // 3. Also protect ```...``` code fences (already safe, but belt-and-suspenders)
  let protected_ = markdown;

  // Display: $$ ... $$ (can span lines)
  protected_ = protected_.replace(/\$\$([\s\S]+?)\$\$/g, (match) => {
    mathBlocks.push(match);
    return PLACEHOLDER(mathBlocks.length - 1);
  });

  // Display: \[ ... \] (can span lines)
  protected_ = protected_.replace(/\\\[([\s\S]+?)\\\]/g, (match) => {
    mathBlocks.push(match);
    return PLACEHOLDER(mathBlocks.length - 1);
  });

  // Inline: $ ... $ (single line, non-greedy, must have content)
  protected_ = protected_.replace(/\$([^\$\n]+?)\$/g, (match) => {
    mathBlocks.push(match);
    return PLACEHOLDER(mathBlocks.length - 1);
  });

  // Inline: \( ... \) (single line)
  protected_ = protected_.replace(/\\\((.+?)\\\)/g, (match) => {
    mathBlocks.push(match);
    return PLACEHOLDER(mathBlocks.length - 1);
  });

  // Parse Markdown
  let html = typeof marked !== "undefined"
    ? marked.parse(protected_)
    : escapeHtml(protected_);

  // Sanitize
  if (typeof DOMPurify !== "undefined") {
    html = DOMPurify.sanitize(html);
  }

  // Restore math blocks
  for (let i = 0; i < mathBlocks.length; i++) {
    html = html.replace(PLACEHOLDER(i), mathBlocks[i]);
  }

  return html;
}

function decorateCodeBlocks(root) {
  root.querySelectorAll("pre > code").forEach((codeEl) => {
    if (typeof hljs !== "undefined") {
      try { hljs.highlightElement(codeEl); } catch { /* skip */ }
    }

    const pre = codeEl.parentElement;
    if (!pre || pre.parentElement?.classList.contains("code-wrap")) return;

    const wrapper = document.createElement("div");
    wrapper.className = "code-wrap";

    const header = document.createElement("div");
    header.className = "code-header";

    const langLabel = document.createElement("span");
    langLabel.textContent = detectLanguage(codeEl);

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "copy-code-btn";
    copyBtn.textContent = "Copy";

    header.append(langLabel, copyBtn);
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.append(header, pre);
  });
}

function detectLanguage(codeEl) {
  const cls = Array.from(codeEl.classList).find((n) => n.startsWith("language-"));
  return cls ? cls.slice(9) || "text" : "text";
}

function renderMath(root) {
  if (typeof renderMathInElement !== "function") return;
  renderMathInElement(root, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: false,
    strict: "ignore",
  });
}

// ── Message DOM ──────────────────────────────────────────────────
function addUserMessage(content) {
  const bubble = createMessageShell("user");
  bubble.textContent = content;
  scrollToBottom();
}

function addTypingBubble() {
  const bubble = createMessageShell("assistant");
  bubble.innerHTML = "";
  bubble.append(typingTemplate.content.cloneNode(true));
  scrollToBottom();
  return bubble;
}

function createMessageShell(role) {
  const article = document.createElement("article");
  article.className = `msg msg-${role}`;

  const roleLabel = document.createElement("div");
  roleLabel.className = "msg-role";
  roleLabel.textContent = role === "user" ? "You" : "Assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  article.append(roleLabel, bubble);
  chatLog.appendChild(article);
  scrollToBottom();
  return bubble;
}

// ── Word fade-in (non-streamed only) ─────────────────────────────
function applyWordFadeIn(root) {
  const textNodes = [];
  const blocked = "pre, code, .katex, .katex-display, .copy-code-btn";
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);

  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (!node.nodeValue?.trim()) continue;
    if (node.parentElement?.closest(blocked)) continue;
    textNodes.push(node);
  }

  const totalChars = textNodes.reduce((n, nd) =>
    n + nd.nodeValue.replace(/\s+/g, "").length, 0);
  const letterMode = totalChars <= 900;

  let wi = 0;
  for (const node of textNodes) {
    const frag = document.createDocumentFragment();
    for (const seg of node.nodeValue.split(/(\s+)/)) {
      if (!seg) continue;
      if (/^\s+$/.test(seg)) { frag.append(document.createTextNode(seg)); continue; }

      const delay = Math.min(wi * 20, 2000);
      const span = document.createElement("span");
      span.style.setProperty("--word-delay", `${delay}ms`);

      if (letterMode) {
        span.className = "diff-word";
        Array.from(seg).forEach((ch, ci) => {
          const cs = document.createElement("span");
          cs.className = "diff-letter";
          cs.style.setProperty("--word-delay", `${delay}ms`);
          cs.style.setProperty("--letter-delay", `${ci * 10 + (wi % 3) * 4}ms`);
          cs.textContent = ch;
          span.append(cs);
        });
      } else {
        span.className = "diff-word--only";
        span.textContent = seg;
      }
      frag.append(span);
      wi++;
    }
    node.parentNode.replaceChild(frag, node);
  }
}

// ── Utilities ────────────────────────────────────────────────────
function autoResizeInput() {
  promptInput.style.height = "0px";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 200)}px`;
}

function scrollToBottom() {
  requestAnimationFrame(() => { chatLog.scrollTop = chatLog.scrollHeight; });
}

function trimText(text, max) {
  return text.length <= max ? text : text.slice(0, max - 1) + "\u2026";
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
