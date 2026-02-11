# Minimal Chat UI

Single-chat, no-sidebar interface with a clean modern design.

## Features

- **Streaming responses** — tokens appear in real-time via SSE
- Word-by-word fade-in animation for assistant replies
- Markdown rendering with fenced code blocks
- Syntax highlighting and one-click code copy button
- LaTeX math rendering (`$...$`, `$$...$$`, `\(...\)`, `\[...\]`)
- OpenAI-compatible API calls for SGLang (`/v1/models`, `/v1/chat/completions`)
- Stop button to cancel in-flight requests

## Run

1. Start your SGLang server (default expected at `http://127.0.0.1:31080`).
2. Serve this folder:

```bash
cd ~/sglang/minimal-chat
python3 -m http.server 4173
```

3. Open:

```text
http://127.0.0.1:4173
```

Optional API override via query string:

```text
http://127.0.0.1:4173/?api=http://127.0.0.1:31080
```

Optional API + model override:

```text
http://127.0.0.1:4173/?api=http://127.0.0.1:31080&model=mangrove-i2s-tuned
```
