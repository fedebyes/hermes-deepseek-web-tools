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
TIMEOUT_S = 180
MAX_ATTEMPTS = 3
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_DETAIL_LIMIT = 300    # never let a vendor error body flood the context

_ERROR_HINTS: dict[int, str] = {
    400: "request rejected - tool_choice cannot name a server tool, and web_search_options is "
         "refused when these tools are declared",
    401: "DeepSeek API key missing or invalid",
    402: "DeepSeek account balance too low",
    403: "the page was rejected by the vendor's security policy",
    404: "no usable content for that URL",
    413: "request body too large (image)",
    429: "rate limited",
}


class WebToolsError(RuntimeError):
    """A call that could not produce an answer, with the vendor's reason attached."""


# ─── key + endpoint ──────────────────────────────────────────────────────────

def resolve_api_key() -> str | None:
    """The key Hermes already holds for the model, never read from a file here."""
    try:                                            # the plugin host's own resolver
        from agent.web_search_provider import get_provider_env
    except Exception:                               # standalone (tests, live accept)
        return os.environ.get("DEEPSEEK_API_KEY") or None
    try:
        return get_provider_env("DEEPSEEK_API_KEY") or None
    except Exception:
        return os.environ.get("DEEPSEEK_API_KEY") or None


def key_available() -> bool:
    """``check_fn`` for the tool registry: hide the tool when the key is missing."""
    return resolve_api_key() is not None


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
    try:
        parts = urllib.parse.urlsplit(raw)
        host = (parts.hostname or "").lower()
        bad_userinfo = bool(parts.username or parts.password)
        clean_tail = not parts.query and not parts.fragment and parts.path in ("", "/")
        exact = (parts.scheme == "https" and host == "api.deepseek.com"
                 and parts.port in (None, 443) and clean_tail and not bad_userinfo)
        loopback = (parts.scheme in ("http", "https") and host in _LOOPBACK_HOSTS
                    and not bad_userinfo)
    except ValueError:
        exact = loopback = False
    if not (exact or loopback):
        parts = urllib.parse.urlsplit(raw)
        logger.warning("DEEPSEEK_BASE_URL ignored: %s://%s is not the exact origin %s "
                       "and is not loopback; using the default endpoint",
                       parts.scheme or "?", parts.hostname or "?", DEFAULT_BASE_URL)
        return DEFAULT_BASE_URL
    return raw.rstrip("/")


# ─── request ─────────────────────────────────────────────────────────────────

def build_payload(
    *,
    query: str,
    urls: list[str] | None = None,
    image_url: str | None = None,
    thinking: bool = False,
    model: str = DEFAULT_MODEL,
    with_read_url: bool = True,
) -> dict[str, Any]:
    """The wire body for one ``web_research`` call.

    URLs are appended to the prompt: ``deepseek_read_url`` only looks at URLs *in* the last
    user message, so a URL that is not in the text is a URL the model never sees.
    """
    text = query.strip()
    picked = [u.strip() for u in (urls or []) if u and u.strip()][:MAX_URLS]
    if picked:
        text = f"{text}\n\nPages to read:\n" + "\n".join(picked)

    if image_url:
        content: Any = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url.strip()}},
        ]
    else:
        content = text

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
        raise WebToolsError("response carried no choices")
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

    usage = (result.get("usage") or {}).get("server_tool_use") or {}
    if usage:
        billed = ", ".join(f"{name.replace('tool_', '')}={value}"
                           for name, value in sorted(usage.items()) if value)
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
    payload = build_payload(query=query, urls=urls, image_url=image_url,
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
        thinking = bool(args.get("thinking"))

    urls = args.get("urls") or []
    if isinstance(urls, str):                       # tolerate a bare string
        urls = [urls]
    urls = [str(u).strip() for u in urls if str(u).strip()]
    dropped = len(urls) - MAX_URLS
    if dropped > 0:
        urls = urls[:MAX_URLS]

    image_url = (args.get("image_url") or "").strip() or None

    try:
        result = research(query, urls=urls or None, image_url=image_url,
                          thinking=thinking, model=model, with_read_url=with_read_url,
                          timeout=timeout)
    except WebToolsError as exc:
        return f"web_research failed: {exc}"
    except Exception as exc:                        # never break the agent loop
        logger.debug("web_research unexpected failure: %s", exc)
        return f"web_research failed: {type(exc).__name__}: {exc}"

    text = render(result, thinking=thinking)
    if dropped > 0:
        text += f"\n\n({dropped} extra URL(s) ignored: the vendor considers 10.)"
    return text
