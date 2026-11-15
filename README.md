# deepseek-web-tools

A Hermes plugin that gives the agent **one** tool — `web_research` — backed by DeepSeek's
server-side web tools. Declaring `deepseek_web_search` (and optionally `deepseek_read_url`) in a
single Chat Completions call lets the model search the web, open the pages it found and search
inside them **on DeepSeek's side**, then answer with structured citations. One call instead of a
search-then-fetch loop.

```
web_research(query="Is the Tokyo Tower taller than the Eiffel Tower? Check both official sites.")
→ Tokyo Tower is taller — 333 m (official Tokyo Tower site) vs. the Eiffel Tower's 330 m (official Eiffel Tower site).

  Sources:
  1. Tokyo Tower — https://en.tokyotower.co.jp/company/
  2. Key figures — https://www.toureiffel.paris/en/the-monument/key-figures

  Server steps: deep_search ×3
  Billed requests: search=2
  (deep thinking search; page text is not returned by the vendor - use web_extract when you need to read a page yourself)
```

## What it is and is not

**It is** the web layer of DeepSeek Chat inside one model call: the model decides whether to go
online, and a *deep thinking search* round makes up to ~20 calls of `search` → `open_page` →
`find_in_page`, checking facts against their sources before answering. That is not reachable with
Hermes' built-in tools: `web_search` returns metadata, `web_extract` returns page text, and neither
lets the model iterate inside the call.

**It is not** a replacement for `web_extract`. Page text never comes back — the vendor carries
each read page inside an encrypted `signature` — so this tool yields *answers with sources*. When a
pipeline needs the text itself (a note, a diff, a parser), keep using `web_extract`.

## Install

```bash
hermes plugins install fedebyes/hermes-deepseek-web-tools
hermes plugins enable deepseek-web-tools
```

Then restart the process that should see it: `hermes-gateway` (Telegram, cron) **and**
`hermes-webui` if you use the browser UI — a plugin registers its tools when the agent process
starts, so a session that was already running keeps the tool set it loaded at start.

Requires `DEEPSEEK_API_KEY` in the Hermes environment (the same key the model uses). Without it
the tool hides itself (`check_fn`) instead of failing at call time.

## Tool

| Param | Type | Notes |
|---|---|---|
| `query` | string, required | The question. Say how it should be sourced when it matters. |
| `urls` | string[] | Up to 10 http(s) pages to read as the basis of the answer. The vendor considers only the first 10 distinct URLs, and the model picks which to read. **Ignored with `thinking: true`**: the vendor then runs deep thinking search and does not read links separately. |
| `image_url` | string | An https image to search/reason about (image search is part of `deepseek_web_search`). Publicly reachable, ≤8192 characters. |
| `files` | string[] | Up to 3 **local** documents the question is about: absolute paths to PDF, DOCX/XLSX/PPTX, `.txt`/`.md`/`.csv` or an image (PNG/JPEG/WebP/GIF), ≤8 MB each and ≤12 MB total. Each file is **uploaded to the Files API first** (`rich_layout_recognition` for PDF/Word/PowerPoint so figures and tables reach the model, `user_data` for spreadsheets, plain text and images) and referenced by `file_id` — documents are refused inline by the vendor. Uploads are permanent, so the id is cached by content hash: asking about the same PDF twice costs one upload. Bytes are sent to DeepSeek, so credential paths (`.env*`, `~/.ssh`, `~/.local/secrets`, `~/.gnupg`) are refused outright, and every refusal is reported in the answer instead of being dropped silently. |
| `thinking` | boolean | Per-call override. `true` = deep thinking search (multi-step, more expensive), `false` = a single search step. |

Result: the answer, the citation list, the server steps actually used, and the billed request
counters — so a run is auditable without reading the raw response.

## Config

Four settings, under the `settings` subtree the loader actually reads — a value set anywhere else
in the entry is ignored:

```bash
hermes config set plugins.entries.deepseek-web-tools.settings.thinking true
```

| Key (`plugins.entries.deepseek-web-tools.settings.*`) | Default | Meaning |
|---|---|---|
| `model_name` | `deepseek-flash` | The only model the vendor documents for the server tools. Named `model_name` because `model` is a reserved settings root in Hermes. |
| `thinking` | `false` | Default for every call. **Off on purpose:** the server default is *enabled*, and deep thinking search bills reasoning tokens plus up to ~20 searches. |
| `with_read_url` | `true` | Also declare `deepseek_read_url` so URLs passed to the tool are read. Harmless when no URL is passed, inert when `thinking` is on. |
| `timeout_s` | `90` | Per-socket HTTP timeout, clamped to 5..120. Not a hard wall-clock bound — a server trickling the body can exceed it; the real ceiling is the agent's tool deadline (420 s), which the plugin does not own. |

## Cost and limits

- **$0.75 per 1,000 successful** search and fetch **queries**, billed per query — not per token,
  not per result count: one search call carrying three queries is three units. Failed calls, empty
  results and timeouts are not billed. The `Billed requests:` line in the tool result counts exactly
  these units, so a call can be reconciles against the vendor's usage page.
- Rate limits are per account and published on DeepSeek's Search & Fetch page (the documented
  numbers have moved before — the client's job is to survive them, not to encode them). Overruns
  are HTTP 429; the client retries 408/429/5xx up to three times with backoff.
- `thinking` and `urls` do not combine: with thinking on, the round *is* deep thinking search and
  the vendor skips URL reading, so a URL list buys nothing there — read the pages with
  `web_extract`, or keep `thinking` off for that call.
- A `thinking=false` call is typically one search; `thinking=true` can reach ~20 billed calls.
  Choose accordingly — one deliberate deep call usually beats five shallow ones.
- **"Not billed" is about the request, not the turn.** A search or fetch that never produced a
  result is free, but a *generation* failure (HTTP 200 with an `error` body) still bills the
  searches that already ran, and a retry after a 5xx re-runs the whole round: the vendor advises
  retrying those, so the trade is a possible double charge on a rare path. `web_research` reports
  the billed counters it was told about, so a suspicious run is checkable against the balance.

## Verify

```bash
# 1. offline unit tests (no network, no key)
uv run tests/test_client.py

# 2. schema/manifest gate used for catalog admission
hermes plugins validate .

# 3. one real call (~$0.002), asserting kinds/citations/billing
uv run scripts/live_accept.py
```

`live_accept.py` needs `DEEPSEEK_API_KEY`; without it it exits 2 and sends nothing.

## Security

See [SECURITY.md](SECURITY.md). Short version: one outbound host (`api.deepseek.com`), the key is
resolved through `agent.secret_scope.get_secret` (the profile-scoped accessor) and never
logged, no file I/O beyond the local documents you name in `files`, no subprocesses, no telemetry. `DEEPSEEK_BASE_URL` exists for tests only and is validated — an override that would
point the bearer key at another host is refused.

## Uninstall

```bash
hermes plugins disable deepseek-web-tools
hermes plugins remove deepseek-web-tools
```

## License

MIT. Community plugin, not affiliated with DeepSeek.
