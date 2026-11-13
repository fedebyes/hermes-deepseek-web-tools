"""deepseek-web-tools - DeepSeek's server-side web tools as one Hermes tool.

``web_research`` puts ``deepseek_web_search`` (and optionally ``deepseek_read_url``) inside a
single Chat Completions call: the model searches, opens pages and finds text inside them on
the vendor's side, then answers with structured citations. The plugin adds one tool and
nothing else - no hooks, no overrides, and four settings (model_name, thinking,
with_read_url, timeout_s).
"""

from __future__ import annotations

import logging

from . import client
from .schema import WEB_RESEARCH_SCHEMA

logger = logging.getLogger(__name__)

TOOL_NAME = "web_research"
TOOLSET = "web"

def _cfg(ctx, key: str, default):
    """Read one setting, coerced to ``default``'s type; a bad value falls back to the default.

    Settings live under ``plugins.entries.<plugin-id>.settings.<key>`` - ``ctx.get_config``
    resolves that subtree (legacy ``.config`` still honoured), so a value set anywhere else in
    the entry is invisible here. A failed read must never take the tool down: the plugin host
    may run with no config on disk, which is exactly what validation does.
    """
    try:
        value = ctx.get_config(key, default)
    except Exception:
        return default
    if value is None:
        return default
    if isinstance(default, bool):
        return client.as_bool(value, default=default)
    if isinstance(default, int):
        if isinstance(value, bool):                 # bool is an int: refuse rather than 1s
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    text = str(value).strip()
    return text or default


def register(ctx) -> None:
    """Called once by the plugin loader."""
    # ``model`` is a reserved settings root in Hermes (_PLUGIN_SETTING_RESERVED_ROOTS),
    # so the key is ``model_name``: get_config would reject the shorter form.
    model = _cfg(ctx, "model_name", client.DEFAULT_MODEL)
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
