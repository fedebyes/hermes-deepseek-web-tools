"""DeepSeek server-side web tools in one Chat Completions call.

Declaring ``deepseek_web_search`` / ``deepseek_read_url`` puts the web layer of DeepSeek
Chat inside the model call: the model decides whether to search, opens pages it found and
searches inside them, and answers with citations. Nothing is executed client-side and no
page text is returned - the vendor carries each read page in an encrypted ``signature`` -
so this module produces *answers with sources*, not page content.

Self-contained on purpose: no relative imports, so the test suite and the live acceptance
script can import it from its own path without a plugin host.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
MAX_URLS = 10          # vendor: first 10 distinct URLs of the last user message are considered
TIMEOUT_S = 90        # per attempt; 3 attempts + backoff bounds one call at ~4.5 min
                      # (PluginContext.register_tool exposes no execution_timeout, so the
                      # budget has to live here rather than in the tool registration)
MAX_ATTEMPTS = 3
MIN_TIMEOUT_S = 5
MAX_TIMEOUT_S = 120   # above this a single call can outlive the 420 s tool deadline, and the
                      # await aborts while the worker thread keeps billing the vendor round
MAX_FILES = 3         # a second, bigger document is a second call, not a bigger payload
MAX_FILE_BYTES = 8 * 1024 * 1024      # per file; the vendor answers 413 above its body limit
MAX_FILES_TOTAL_BYTES = 12 * 1024 * 1024
# What the vendor parses into context: PDFs and Office documents, plus images and plain text.
_FILE_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
# Attaching a file means its bytes leave this machine for the vendor, so the paths that hold
# credentials are refused outright - a cheap guard that does not depend on remembering.
_REFUSED_PATH_PARTS = ("/.env", "/.ssh/", "/.gnupg/", "/.local/secrets/", "/.password-store/",
                       "/secrets/")
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_DETAIL_LIMIT = 300    # never let a vendor error body flood the context

# Per-page outcomes (url_not_allowed, unavailable) arrive as call-level error codes on a 200,
# never as an HTTP status on POST /chat/completions - so these hints stay about the request.
_ERROR_HINTS: dict[int, str] = {
    400: "request rejected - an unknown parameter, a model_name the vendor does not serve, or a "
         "tool_choice naming a server tool",
    401: "DeepSeek API key missing or invalid",
    402: "DeepSeek account balance too low",
    403: "the request was refused by policy",
    404: "endpoint not found - check the base URL",
    413: "request body too large (image)",
    429: "rate limited",
}


class WebToolsError(RuntimeError):
    """A call that could not produce an answer, with the vendor's reason attached."""


# ─── key + endpoint ──────────────────────────────────────────────────────────

def resolve_api_key() -> str | None:
    """The key Hermes holds for this call, resolved with the caller's profile scope.

    Inside the plugin host this goes through ``agent.secret_scope.get_secret``, which honours
    the turn's profile scope and **raises** ``UnscopedSecretError`` when multiplexing is on and
    no scope is installed. That raise is deliberately not caught: falling back to
    ``os.environ`` here would hand the model call another profile's credential, which is the
    exact failure the scope exists to prevent.

    The ``os.environ`` read survives only for the standalone case - the test suite and
    ``scripts/live_accept.py`` run outside a Hermes host, where ``agent`` is not importable.
    """
    try:
        from agent.secret_scope import get_secret
    except ModuleNotFoundError as exc:
        # Only "there is no Hermes host here" may fall back (standalone tests / live_accept).
        # A transitive ImportError inside a real install is a broken core, not a standalone run,
        # and must not silently degrade into the unscoped read rule 11 bans.
        if (getattr(exc, "name", "") or "") != "agent":   # exact: `agentx` is not the host
            raise
        return os.environ.get("DEEPSEEK_API_KEY") or None
    return get_secret("DEEPSEEK_API_KEY") or None


def key_available() -> bool:
    """``check_fn`` for the tool registry: hide the tool when the key is missing or unscoped."""
    try:
        return resolve_api_key() is not None
    except Exception as exc:
        # Hide the tool on an unscoped read, but say it out loud: in a multiplexed install this
        # is a spawn-site scoping bug, and silence would look like "the key is not configured".
        # Anything else is a real failure and is left to the registry, which logs it fail-closed.
        if type(exc).__name__ == "UnscopedSecretError":
            logger.warning("DEEPSEEK_API_KEY is unscoped here (%s) - hiding web_research", exc)
            return False
        raise


def base_url() -> str:
    """``DEEPSEEK_BASE_URL`` is a test seam and is validated, never trusted blindly.

    Accepting an arbitrary host would hand the bearer key to whoever set the variable, so
    only two shapes pass: the exact origin ``https://api.deepseek.com``, or a loopback host
    (plain http allowed there). Anything else is ignored with a warning that names the
    scheme and host only - the raw value could carry a credential.
    """
    raw = (os.environ.get("DEEPSEEK_BASE_URL") or "").strip()
    if not raw:
        return DEFAULT_BASE_URL
    parts = None
    refuse_reason = "not the exact origin"
    try:
        parts = urllib.parse.urlsplit(raw)
        host = (parts.hostname or "").lower()
        bad_userinfo = bool(parts.username or parts.password)
        clean_tail = not parts.query and not parts.fragment and parts.path in ("", "/")
        exact = (parts.scheme == "https" and host == "api.deepseek.com"
                 and parts.port in (None, 443) and clean_tail and not bad_userinfo)
        loopback = (parts.scheme in ("http", "https") and host in _LOOPBACK_HOSTS
                    and not bad_userinfo)
    except ValueError as exc:               # unparseable URL, or a port urlsplit refuses
        exact = loopback = False
        refuse_reason = f"unparseable: {exc}"
    if not (exact or loopback):
        # Never re-parse: the value may be the unparseable one that just raised, and a second
        # urlsplit would turn "refuse and fall back" into a crash on every call.
        scheme = (parts.scheme or "?") if parts is not None else "unparseable"
        host = (parts.hostname or "?") if parts is not None else "unparseable"
        logger.warning("DEEPSEEK_BASE_URL ignored: %s://%s (%s); using the default endpoint",
                       scheme, host, refuse_reason)
        return DEFAULT_BASE_URL
    return raw.rstrip("/")


# ─── request ─────────────────────────────────────────────────────────────────

def billed_counters(usage: dict[str, Any] | None) -> str:
    """Only ``tool_*`` counters price a request; other ``server_tool_use`` keys are display stats.

    The vendor's usage block is a mixed bag and its vocabulary changes; filtering by prefix keeps
    a future ``tool_*`` counter billable and a future non-billed counter out of "Billed requests".
    """
    return ", ".join(f"{name[len('tool_'):]}={value}"
                     for name, value in sorted((usage or {}).items())
                     if name.startswith("tool_") and isinstance(value, int) and value)


def dedupe_urls(urls: list[str] | None, limit: int | None = MAX_URLS) -> list[str]:
    """Trim, drop empties and de-duplicate, keeping order — the vendor counts the first 10
    *distinct* URLs, so a duplicate would spend a slot the model never gets to use.

    ``limit=None`` returns every distinct page; ``run`` uses that to tell a duplicate from a page
    dropped by the cap, which are different things to report."""
    seen: set[str] = set()
    picked: list[str] = []
    for raw in urls or []:
        url = str(raw).strip()
        key = url.rstrip("/")                       # same page to the vendor, one slot here
        if not url or key in seen:
            continue
        seen.add(key)
        picked.append(url)
        if limit is not None and len(picked) == limit:
            break
    return picked


def clamp_timeout(value: Any, default: int = TIMEOUT_S) -> int:
    """Bound the per-attempt timeout: urlopen rejects 0 and negatives, and a huge value lets a
    call outlive the tool deadline (the vendor round is billed even when the await aborts)."""
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return default
    return max(MIN_TIMEOUT_S, min(seconds, MAX_TIMEOUT_S))


def as_bool(value: Any, *, default: bool = False) -> bool:
    """Coerce a tool argument or setting to a bool, treating "false" as false.

    ``bool("false")`` is True, which would silently turn a one-search call into a deep
    thinking search billed for up to ~20, so string forms are parsed explicitly.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_credentials(candidate: str) -> bool:
    """True for the paths nobody should ever hand to a third party, by name or by location."""
    lowered = candidate.lower()
    if os.path.basename(lowered).startswith(".env"):
        return True
    return any(part in lowered for part in _REFUSED_PATH_PARTS)


def file_parts(
    paths: list[str] | None,
    *,
    max_files: int = MAX_FILES,
    max_bytes: int = MAX_FILE_BYTES,
    max_total: int = MAX_FILES_TOTAL_BYTES,
) -> tuple[list[dict[str, str]], list[str]]:
    """Turn local file paths into ``file`` content parts — the vendor's file input.

    The vendor takes a base64 data URL (``file_data``) plus a ``filename``; there is no upload
    step, so the bytes ride in the request body and the size caps are what stands between a
    working call and a 413. Refusals are *reported*, never silent: a skipped file the model
    cannot see is the difference between an answer about the document and a confident answer
    about nothing. Credential paths are refused because attaching a file ships it to a third
    party, and that decision should never be one forgotten path away.
    """
    parts: list[dict[str, str]] = []
    notes: list[str] = []
    total = 0

    for raw in paths or []:
        path = str(raw).strip()
        if not path:
            continue
        if len(parts) >= max_files:
            notes.append(f"{path} skipped - at most {max_files} files per call")
            continue
        if not os.path.isabs(path):
            notes.append(f"{path} skipped - an absolute path is required")
            continue
        # What actually gets read is the link's *target*: a symlink named innocent.pdf pointing at
        # ~/.local/secrets/key.txt would sail past a check on the string the caller passed, so every
        # policy and filesystem decision below is made on the resolved path.
        target = os.path.realpath(path)
        if _looks_like_credentials(path) or _looks_like_credentials(target):
            notes.append(f"{os.path.basename(path)} skipped - credentials are never attached")
            continue
        lowered = target.lower()
        suffix = os.path.splitext(lowered)[1]
        mime = _FILE_MIME.get(suffix)
        if not mime:
            allowed = ", ".join(sorted(_FILE_MIME))
            notes.append(f"{path} skipped - unsupported type (allowed: {allowed})")
            continue
        try:
            size = os.path.getsize(target)
        except OSError:
            notes.append(f"{path} skipped - not readable")
            continue
        if not os.path.isfile(target):
            notes.append(f"{path} skipped - not a regular file")
            continue
        if size == 0:
            notes.append(f"{path} skipped - empty file")
            continue
        if size > max_bytes:
            notes.append(f"{os.path.basename(path)} skipped - {size} bytes, over the "
                         f"{max_bytes} byte limit for one file")
            continue
        if total + size > max_total:
            notes.append(f"{os.path.basename(path)} skipped - would put the payload over "
                         f"{max_total} bytes")
            continue
        try:
            with open(path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("ascii")
        except OSError:
            notes.append(f"{path} skipped - not readable")
            continue
        total += size
        parts.append({
            "type": "file",
            "file_data": f"data:{mime};base64,{encoded}",
            "filename": os.path.basename(target),
        })

    return parts, notes


def build_payload(
    *,
    query: str,
    urls: list[str] | None = None,
    image_url: str | None = None,
    files: list[dict[str, str]] | None = None,
    thinking: bool = False,
    model: str = DEFAULT_MODEL,
    with_read_url: bool = True,
) -> dict[str, Any]:
    """The wire body for one ``web_research`` call.

    URLs are appended to the prompt: ``deepseek_read_url`` only looks at URLs *in* the last
    user message, so a URL that is not in the text is a URL the model never sees. Attached
    files are content parts, not prompt text - the vendor parses them into the context.
    """
    text = query.strip()
    picked = dedupe_urls(urls)
    if picked:
        text = f"{text}\n\nPages to read:\n" + "\n".join(picked)

    content: Any = text
    if image_url or files:
        content = [{"type": "text", "text": text}]
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url.strip()}})
        content.extend(files or [])

    tools: list[dict[str, str]] = [{"type": "deepseek_web_search"}]
    if with_read_url:
        tools.append({"type": "deepseek_read_url"})

    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "tools": tools,
        "thinking": {"type": "enabled" if thinking else "disabled"},
        "stream": False,
    }


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """The vendor's message, truncated. A body never contains the request's Authorization."""
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:
        return ""
    try:
        parsed = json.loads(raw)
        message = str(((parsed or {}).get("error") or {}).get("message") or "")
        if message:
            return message[:_DETAIL_LIMIT]
    except (ValueError, AttributeError):
        pass
    return raw.strip()[:_DETAIL_LIMIT]


def post(
    payload: dict[str, Any],
    *,
    key: str,
    base: str | None = None,
    timeout: int = TIMEOUT_S,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """POST the payload, retrying only what the vendor says is worth retrying."""
    url = f"{(base or base_url()).rstrip('/')}/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error = "no attempt made"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        try:
            with opener(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _error_detail(exc)
            if exc.code in RETRY_STATUS and attempt < MAX_ATTEMPTS:
                sleep(min(2 ** (attempt - 1), 4))
                last_error = f"HTTP {exc.code}"
                continue
            hint = _ERROR_HINTS.get(exc.code)
            reason = f"HTTP {exc.code}" + (f" ({hint})" if hint else "")
            raise WebToolsError(f"{reason}: {detail}" if detail else reason) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt < MAX_ATTEMPTS:
                sleep(min(2 ** (attempt - 1), 4))
                last_error = f"network error: {exc}"
                continue
            raise WebToolsError(f"network error after {MAX_ATTEMPTS} attempts: {exc}") from exc

    raise WebToolsError(f"request failed: {last_error}")


# ─── response ────────────────────────────────────────────────────────────────

def parse_response(body: dict[str, Any]) -> dict[str, Any]:
    """Flatten a chat-completion with server tools into what a caller can use."""
    choices = body.get("choices") or []
    if not choices:
        # A generation failure arrives as HTTP 200 with {"error": {...}, "usage": {...}}: the
        # vendor's message is the actionable part ("please retry", a content refusal), and the
        # searches that already ran were billed, so it must not be collapsed into "no choices".
        error = body.get("error") or {}
        detail = str(error.get("message") or "").strip() if isinstance(error, dict) else str(error)
        code = error.get("code") if isinstance(error, dict) else None
        reason = f"generation failed{f' ({code})' if code else ''}"
        billed = billed_counters((body.get("usage") or {}).get("server_tool_use"))
        if billed:                                  # a failed generation still billed the round
            detail = f"{detail} (billed: {billed})" if detail else f"billed: {billed}"
        raise WebToolsError(f"{reason}: {detail}" if detail else reason)
    choice = choices[0] or {}
    message = choice.get("message") or {}

    calls: list[dict[str, Any]] = []
    for step in message.get("server_tool_calls") or []:
        for call in (step or {}).get("calls") or []:
            calls.append({
                "kind": (call or {}).get("kind"),
                "tool": (call or {}).get("type"),
                "action": (call or {}).get("action"),
                "status": (call or {}).get("status"),
                "error": (call or {}).get("error"),
                "page": (call or {}).get("page"),
                "sources": (call or {}).get("sources"),
            })

    citations: list[dict[str, Any]] = []
    for annotation in message.get("annotations") or []:
        if (annotation or {}).get("type") != "url_citation":
            continue
        cite = annotation.get("url_citation") or {}
        citations.append({
            "url": cite.get("url"),
            "title": cite.get("title"),
            "start_index": cite.get("start_index"),
            "end_index": cite.get("end_index"),
        })

    return {
        "answer": message.get("content") or "",
        "citations": citations,
        "calls": calls,
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage") or {},
    }


def render(result: dict[str, Any], *, thinking: bool) -> str:
    """The string the agent sees: answer first, then sources, then what the call cost."""
    answer = (result.get("answer") or "").strip()
    parts: list[str] = [answer or "(no answer text returned)"]

    citations = result.get("citations") or []
    if citations:
        lines = []
        for index, cite in enumerate(citations, start=1):
            title = (cite.get("title") or "").strip() or "(untitled)"
            lines.append(f"{index}. {title} - {cite.get('url')}")
        parts.append("Sources:\n" + "\n".join(lines))

    calls = result.get("calls") or []
    if calls:
        counts: dict[str, int] = {}
        for call in calls:
            kind = str(call.get("kind") or "?")
            counts[kind] = counts.get(kind, 0) + 1
        steps = " · ".join(f"{kind} ×{count}" for kind, count in sorted(counts.items()))
        parts.append(f"Server steps: {steps}")

    failures = []
    for call in calls:
        if call.get("status") == "failed":
            code = ((call.get("error") or {}) if isinstance(call.get("error"), dict) else {}).get("code")
            failures.append(f"{call.get('kind') or '?'}: {code or 'failed'}")
    if failures:
        parts.append("Failed calls: " + ", ".join(failures))

    billed = billed_counters((result.get("usage") or {}).get("server_tool_use"))
    if billed:
        parts.append(f"Billed requests: {billed}")

    finish = result.get("finish_reason")
    if finish and finish != "stop":
        parts.append(f"finish_reason: {finish}")

    mode = "deep thinking search" if thinking else "single search step"
    parts.append(f"({mode}; page text is not returned by the vendor - "
                 f"use web_extract when you need to read a page yourself)")
    return "\n\n".join(parts)


# ─── one call end to end ─────────────────────────────────────────────────────

def research(
    query: str,
    *,
    urls: list[str] | None = None,
    image_url: str | None = None,
    files: list[dict[str, str]] | None = None,
    thinking: bool = False,
    model: str = DEFAULT_MODEL,
    with_read_url: bool = True,
    timeout: int = TIMEOUT_S,
    key: str | None = None,
    base: str | None = None,
) -> dict[str, Any]:
    """Run one web_research call and return its parsed result (raises WebToolsError)."""
    resolved = key or resolve_api_key()
    if not resolved:
        raise WebToolsError(
            "DEEPSEEK_API_KEY is not available - set it in the Hermes environment ("
            "`hermes config set` or .env) to use web_research")
    payload = build_payload(query=query, urls=urls, image_url=image_url, files=files,
                            thinking=thinking, model=model, with_read_url=with_read_url)
    body = post(payload, key=resolved, base=base, timeout=timeout)
    return parse_response(body)


def run(args: dict[str, Any], *, thinking_default: bool = False,
        model: str = DEFAULT_MODEL, with_read_url: bool = True,
        timeout: int = TIMEOUT_S) -> str:
    """Tool handler body: validate the arguments, call, render. Never raises."""
    query = (args.get("query") or "").strip()
    if not query:
        return "web_research needs a non-empty `query`."

    if args.get("thinking") is None:
        thinking = thinking_default
    else:
        thinking = as_bool(args.get("thinking"), default=thinking_default)

    urls = args.get("urls") or []
    if isinstance(urls, str):                       # tolerate a bare string
        urls = [urls]
    elif isinstance(urls, dict):                    # a mapping would otherwise iterate its keys
        urls = list(urls.values())
    raw_urls = [str(u).strip() for u in urls if str(u).strip()]
    unique = dedupe_urls(raw_urls, limit=None)
    urls = unique[:MAX_URLS]
    duplicates = len(raw_urls) - len(unique)
    over_cap = len(unique) - len(urls)

    image_url = (args.get("image_url") or "").strip() or None

    raw_files = args.get("files") or []
    if isinstance(raw_files, str):                  # tolerate a bare path
        raw_files = [raw_files]
    elif isinstance(raw_files, dict):               # a mapping would otherwise iterate its keys
        raw_files = list(raw_files.values())
    parts, file_notes = file_parts([str(p) for p in raw_files])

    try:
        result = research(query, urls=urls or None, image_url=image_url, files=parts or None,
                          thinking=thinking, model=model, with_read_url=with_read_url,
                          timeout=clamp_timeout(timeout))
    except WebToolsError as exc:
        return f"web_research failed: {exc}"
    except Exception as exc:                        # never break the agent loop
        logger.debug("web_research unexpected failure: %s", exc)
        return f"web_research failed: {type(exc).__name__}: {exc}"

    text = render(result, thinking=thinking)
    notes = []
    if duplicates:
        notes.append(f"{duplicates} duplicate or blank URL(s) ignored")
    if over_cap:                                    # a different cause: 12 distinct pages, cap 10
        notes.append(f"{over_cap} URL(s) over the vendor's 10-page cap not sent")
    if notes:
        text += f"\n\n({'; '.join(notes)}.)"
    if file_notes:
        text += f"\n\n({'; '.join(file_notes)}.)"
    return text
