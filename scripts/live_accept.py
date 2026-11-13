#!/usr/bin/env python3
"""Live acceptance for deepseek-web-tools: one real call, assertions on the wire facts.

    uv run <plugin>/scripts/live_accept.py

Exit codes: 0 = accepted, 1 = the call ran but a premise broke, 2 = no key in this
environment (nothing was sent). Prints facts only; the key is never echoed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
KNOWN_KINDS = {"search", "deep_search", "read_link", "image_search", "deep_image_search"}
QUERY = ("Which is taller, the Eiffel Tower or the Tokyo Tower? "
         "Check both official sites and answer in one line.")


def load_client():
    spec = importlib.util.spec_from_file_location("dsweb_client", PLUGIN_DIR / "client.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    client = load_client()
    if not client.key_available():
        print("SKIP: DEEPSEEK_API_KEY is not available in this environment")
        return 2

    result = client.research(QUERY, thinking=True)          # deep thinking search on purpose

    problems: list[str] = []
    answer = (result.get("answer") or "").strip()
    if not answer:
        problems.append("empty answer")
    kinds = {call.get("kind") for call in result.get("calls") or []}
    if not kinds:
        problems.append("no server tool calls in the response")
    unknown = kinds - KNOWN_KINDS
    if unknown:
        problems.append(f"unexpected kinds: {sorted(unknown)}")
    if not result.get("citations"):
        problems.append("no citations")
    usage = (result.get("usage") or {}).get("server_tool_use") or {}
    billed = sum(int(usage.get(name) or 0)
                 for name in ("tool_search", "tool_fetch", "tool_image_search"))
    if billed < 1:
        problems.append("vendor billed zero requests")
    if result.get("finish_reason") != "stop":
        problems.append(f"finish_reason={result.get('finish_reason')!r}")

    print("answer:", answer[:200].replace("\n", " "))
    print("kinds:", sorted(k for k in kinds if k), "| citations:", len(result.get("citations") or []),
          "| billed requests:", billed)
    if problems:
        print("FAIL:", "; ".join(problems))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
