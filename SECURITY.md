# Security

`deepseek-web-tools` runs **in-process** in the Hermes agent host, with the agent's own
permissions. This file states exactly what it can reach, so you can decide before enabling it.

## Blast radius, in one sentence

It registers one tool that makes one HTTPS request to `api.deepseek.com` and returns text; the
only files it ever opens are the ones you pass in `files`, it starts no process, imports nothing
dynamic, and cannot act without a tool call from the model.

## Outbound traffic

| Endpoint | When | Credential |
|---|---|---|
| `POST https://api.deepseek.com/chat/completions` | one per `web_research` call | `Authorization: Bearer $DEEPSEEK_API_KEY` |

Nothing else is contacted — no telemetry, no update check, no vendor SDK. The prompt text, any
URLs/images you pass and **the bytes of any file named in `files`** are sent to DeepSeek, which is
the point of the tool: treat a query as public content and do not put secrets in it.

## Local files (`files`)

- Read **only** when a tool call names them: no directory walking, no globbing, no default path.
- Refused before being opened: relative paths, non-regular files, empty files, anything over the
  size caps, unsupported extensions, and every path that looks like a credential store
  (`.env*`, `~/.ssh`, `~/.gnupg`, `~/.local/secrets`, `~/.password-store`, `*/secrets/`). The
  refusal is reported in the answer; nothing is dropped silently.
- Encoded as a base64 `data:` URL in the request body, held in memory only — the plugin writes no
  temporary file and logs neither the path nor the contents.

## The API key

- Resolved through `agent.secret_scope.get_secret`, which honours the turn's profile scope. In
  multiplexed installs an unscoped read **raises** `UnscopedSecretError`; the plugin does not catch
  it into a bare `os.environ` read — the tool hides itself instead (`check_fn` returns false). The
  `os.environ` path exists only for standalone runs (test suite, `scripts/live_accept.py`), where
  the Hermes host package is not importable.
- The plugin never reads `.env` or any secrets file itself.
- Never logged, never included in an error message, never written to disk. Error bodies are
  truncated to 300 characters, and the request header is never echoed.
- A missing key makes the tool **hide** (`check_fn`), rather than fail on first call.

## `DEEPSEEK_BASE_URL` (test seam)

Accepted only as the exact origin `https://api.deepseek.com` (https, no userinfo, port 443/absent,
no path/query/fragment) or a loopback host (`127.0.0.1`, `localhost`, `::1`). Anything else is
refused with a warning that names the scheme and host only — never the raw value, which could carry
a credential. A freely redirectable base URL would hand the bearer key to whoever set the variable,
so there is deliberately no escape hatch: anyone who can set one variable can set two.

## Privilege surface

- **No hooks.** No `pre_tool_call`, no `pre_llm_call`, no `on_session_*` — this plugin cannot see,
  rewrite or veto anything else the agent does.
- **No overrides.** `web_research` is a new name; it replaces no built-in tool.
- **No side effects outside the response.** No writes, no `subprocess`, no `eval`, no dynamic
  imports.
- Failures are returned as a short string (`web_research failed: …`) and never raised: a plugin that
  throws inside a tool handler breaks the agent loop.

## What the vendor sees and keeps

DeepSeek executes the search and the page reads, so it sees the query, the URLs and any image you
pass. Read page text is returned to the model **encrypted inside a per-call `signature`**; the
plugin cannot decrypt it and does not try. Content that fails the vendor's safety review comes back
as a `url_not_allowed` / `content_filter` result, surfaced in the tool output as a failed call or a
non-`stop` `finish_reason`.

## Indirect prompt injection

`web_research` returns text the vendor's model wrote and page titles the vendor chose, and it puts
both into the tool result **the agent trusts**. Web content is attacker-controlled by definition, so
a page can carry instructions aimed at the model ("ignore your rules and…"). Two things follow:

- Treat a `web_research` result as untrusted content, not as an instruction. It is data with
  sources attached, and the sources are what make it checkable.
- The tool grants no capability of its own: it cannot write files, run commands or spawn anything.
  An injected instruction that reaches the agent still has to pass the same approval layers as any
  other model decision — which is why this plugin declares no hooks and no tool overrides.

## Reporting

Open an issue at https://github.com/fedebyes/hermes-deepseek-web-tools/issues. Please do not paste
API keys, full request bodies or private URLs into a report.
