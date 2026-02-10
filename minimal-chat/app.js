const DEFAULT_API_BASE = "http://127.0.0.1:31080";
const DEFAULT_MODEL_ID = "mangrove-i2s-tuned";
const SYSTEM_PROMPT =
  "You are a helpful assistant. Reply with concise Markdown. Use LaTeX for math when useful.";

const state = {
  busy: false,
  modelId: null,
  sessionStarted: false,
  messages: [{ role: "system", content: SYSTEM_PROMPT }],
};

const URL_PARAMS = new URLSearchParams(window.location.search);
const apiBaseFromQuery = URL_PARAMS.get("api");
const modelFromQuery = URL_PARAMS.get("model");
const API_BASE = (apiBaseFromQuery || DEFAULT_API_BASE).replace(/\/+$/, "");
const PREFERRED_MODEL_ID = (modelFromQuery || DEFAULT_MODEL_ID).trim();

const appShell = document.querySelector(".app-shell");
const chatLog = document.getElementById("chat-log");
const composer = document.getElementById("composer");
const promptInput = document.getElementById("prompt-input");
const sendBtn = document.getElementById("send-btn");
const resetBtn = document.getElementById("reset-chat");
const typingTemplate = document.getElementById("typing-template");

if (typeof marked !== "undefined") {
  marked.setOptions({
    gfm: true,
    breaks: true,
  });
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
    if (state.busy) return;
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
      copyBtn.textContent = "Copied";
      window.setTimeout(() => {
        copyBtn.textContent = originalLabel;
      }, 1200);
    } catch (error) {
      copyBtn.textContent = "Copy failed";
      window.setTimeout(() => {
        copyBtn.textContent = originalLabel;
      }, 1200);
      console.error(error);
    }
  });
}

async function handleSubmit(event) {
  event.preventDefault();
  if (state.busy) return;

  const userText = promptInput.value.trim();
  if (!userText) return;

  if (!state.sessionStarted) {
    setSessionStarted(true);
  }

  promptInput.value = "";
  autoResizeInput();
  addUserMessage(userText);
  state.messages.push({ role: "user", content: userText });

  const typingBubble = addTypingBubble();
  setBusy(true);

  try {
    let assistantReply;
    try {
      assistantReply = await requestAssistantReply();
      state.messages.push({ role: "assistant", content: assistantReply });
    } catch (error) {
      assistantReply = "I could not reach the chat endpoint.\n\n```text\n" + error.message + "\n```";
    }

    paintAssistantBubble(typingBubble, assistantReply, true);
  } catch (error) {
    const fallback = "Something went wrong while rendering the response.\n\n```text\n" + error.message + "\n```";
    paintAssistantBubble(typingBubble, fallback, false);
  } finally {
    setBusy(false);
    promptInput.focus();
  }
}

function setSessionStarted(started) {
  state.sessionStarted = started;
  appShell.classList.toggle("is-home", !started);
}

function setBusy(isBusy) {
  state.busy = isBusy;
  sendBtn.disabled = isBusy;
  resetBtn.disabled = isBusy;
  promptInput.disabled = isBusy;
}

async function resolveModel() {
  if (state.modelId) return state.modelId;

  const response = await fetch(`${API_BASE}/v1/models`);
  if (!response.ok) {
    throw new Error(`Model discovery failed (${response.status}).`);
  }

  const payload = await response.json();
  const models = Array.isArray(payload?.data) ? payload.data : [];
  const preferredModel = models.find((item) => item?.id === PREFERRED_MODEL_ID);
  const modelId = preferredModel?.id || models[0]?.id;
  if (!modelId) {
    throw new Error("No models returned by /v1/models.");
  }

  state.modelId = modelId;
  return modelId;
}

async function requestAssistantReply() {
  const modelId = await resolveModel();
  const payload = {
    model: modelId,
    messages: state.messages,
    temperature: 0.35,
    stream: false,
  };

  const response = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Chat failed (${response.status}): ${trimText(body, 280)}`);
  }

  const data = await response.json();
  const text = data?.choices?.[0]?.message?.content;
  if (typeof text !== "string" || !text.trim()) {
    throw new Error("Assistant response was empty.");
  }

  return text;
}

function addUserMessage(content) {
  const bubble = createMessageShell("user");
  bubble.textContent = content;
  scrollToBottom();
}

function addAssistantMessage(content, animate = true) {
  const bubble = createMessageShell("assistant");
  paintAssistantBubble(bubble, content, animate);
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

function paintAssistantBubble(bubble, markdown, animate) {
  const html = markdownToHtml(markdown);
  bubble.innerHTML = html;
  decorateCodeBlocks(bubble);
  renderMath(bubble);
  if (animate) {
    applyWordFadeIn(bubble);
  }
  scrollToBottom();
}

function markdownToHtml(markdown) {
  const rawHtml = typeof marked !== "undefined" ? marked.parse(markdown) : escapeHtml(markdown);
  if (typeof DOMPurify !== "undefined") {
    return DOMPurify.sanitize(rawHtml);
  }
  return rawHtml;
}

function decorateCodeBlocks(root) {
  root.querySelectorAll("pre > code").forEach((codeElement) => {
    if (typeof hljs !== "undefined") {
      try {
        hljs.highlightElement(codeElement);
      } catch (error) {
        console.warn("Code highlight skipped.", error);
      }
    }

    const pre = codeElement.parentElement;
    if (!pre || pre.parentElement?.classList.contains("code-wrap")) return;

    const wrapper = document.createElement("div");
    wrapper.className = "code-wrap";

    const header = document.createElement("div");
    header.className = "code-header";

    const langLabel = document.createElement("span");
    langLabel.textContent = detectLanguage(codeElement);

    const copyButton = document.createElement("button");
    copyButton.type = "button";
    copyButton.className = "copy-code-btn";
    copyButton.textContent = "Copy";

    header.append(langLabel, copyButton);
    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.append(header, pre);
  });
}

function detectLanguage(codeElement) {
  const languageClass = Array.from(codeElement.classList).find((name) => name.startsWith("language-"));
  if (!languageClass) return "text";
  return languageClass.slice("language-".length) || "text";
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

function applyWordFadeIn(root) {
  const textNodes = [];
  const blockedSelector = "pre, code, .katex, .katex-display, .copy-code-btn";
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);

  while (walker.nextNode()) {
    const textNode = walker.currentNode;
    if (!textNode.nodeValue || !textNode.nodeValue.trim()) continue;
    const parent = textNode.parentElement;
    if (!parent || parent.closest(blockedSelector)) continue;
    textNodes.push(textNode);
  }

  const totalRenderableChars = textNodes.reduce((count, node) => {
    return count + node.nodeValue.replace(/\s+/g, "").length;
  }, 0);
  const shouldAnimateLetters = totalRenderableChars <= 900;

  let wordIndex = 0;
  for (const node of textNodes) {
    const fragment = document.createDocumentFragment();
    const segments = node.nodeValue.split(/(\s+)/);

    for (const segment of segments) {
      if (!segment) continue;
      if (/^\s+$/.test(segment)) {
        fragment.append(document.createTextNode(segment));
        continue;
      }

      const delayMs = Math.min(wordIndex * 24, 2400);
      const wordSpan = document.createElement("span");
      wordSpan.style.setProperty("--word-delay", `${delayMs}ms`);

      if (shouldAnimateLetters) {
        wordSpan.className = "diff-word";
        const chars = Array.from(segment);
        for (let charIndex = 0; charIndex < chars.length; charIndex += 1) {
          const charSpan = document.createElement("span");
          const letterDelayMs = charIndex * 12 + (wordIndex % 3) * 5;
          charSpan.className = "diff-letter";
          charSpan.style.setProperty("--word-delay", `${delayMs}ms`);
          charSpan.style.setProperty("--letter-delay", `${letterDelayMs}ms`);
          charSpan.textContent = chars[charIndex];
          wordSpan.append(charSpan);
        }
      } else {
        wordSpan.className = "diff-word--only";
        wordSpan.textContent = segment;
      }

      fragment.append(wordSpan);
      wordIndex += 1;
    }

    node.parentNode.replaceChild(fragment, node);
  }
}

function autoResizeInput() {
  promptInput.style.height = "0px";
  promptInput.style.height = `${promptInput.scrollHeight}px`;
}

function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

function trimText(text, maxLen) {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 1) + "...";
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
