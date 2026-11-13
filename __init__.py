"""deepseek-web-tools - DeepSeek's server-side web tools as one Hermes tool.

``web_research`` puts ``deepseek_web_search`` (and optionally ``deepseek_read_url``) inside a
single Chat Completions call: the model searches, opens pages and finds text inside them on
the vendor's side, then answers with structured citations. The plugin adds one tool and
nothing else - no hooks, no overrides, no config of its own beyond three knobs.
"""

from __future__ import annotations

import logging

from . import client
from .schema import WEB_RESEARCH_SCHEMA

logger = logging.getLogger(__name__)

TOOL_NAME = "web_research"
TOOLSET = "web"

_TRUE = {"1", "true", "yes", "on"}


def _cfg(ctx, key: str, default):
    """Read one setting, coerced to ``default``'s type; a bad value falls back to the default.

    A settings read that fails must never take the tool down (the plugin host may be running
    with no config on disk - validation and tests do exactly that).
    """
    try:
        value = ctx.get_config(key, default)
    except Exception:
        return default
    if value is None:
        return default
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in _TRUE
    if isinstance(default, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    text = str(value).strip()
    return text or default


def register(ctx) -> None:
    """Called once by the plugin loader."""
    model = _cfg(ctx, "model", client.DEFAULT_MODEL)
    thinking_default = _cfg(ctx, "thinking", False)
    with_read_url = _cfg(ctx, "with_read_url", True)
    timeout = _cfg(ctx, "timeout_s", client.TIMEOUT_S)

    def handle(args: dict, **_kwargs) -> str:
        """Run one web_research call. Returns a string; never raises into the agent loop."""
        try:
            return client.run(
                args if isinstance(args, dict) else {},
                thinking_default=thinking_default,
                model=model,
                with_read_url=with_read_url,
                timeout=timeout,
            )
        except Exception as exc:  # defensive: run() already swallows its own failures
            logger.debug("web_research handler failed: %s", exc)
            return f"web_research failed: {type(exc).__name__}: {exc}"

    ctx.register_tool(
        name=TOOL_NAME,
        toolset=TOOLSET,
        schema=WEB_RESEARCH_SCHEMA,
        handler=handle,
        check_fn=client.key_available,
        requires_env=["DEEPSEEK_API_KEY"],
        emoji="🌐",
        description=("Answer a question from the live web with citations, using DeepSeek's "
                     "server-side web tools (one call instead of a search-then-fetch loop)."),
    )
    logger.debug("deepseek-web-tools: registered %s (model=%s thinking=%s read_url=%s)",
                 TOOL_NAME, model, thinking_default, with_read_url)
