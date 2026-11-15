#!/usr/bin/env python3
"""Live acceptance for deepseek-web-tools: one real call, assertions on the wire facts.

    uv run <plugin>/scripts/live_accept.py                 # one real search call
    uv run <plugin>/scripts/live_accept.py --files-probe   # upload a throwaway .txt with a token
    uv run <plugin>/scripts/live_accept.py --files DOC --expect TOKEN   # assert it was read

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


def file_acceptance(client, path: Path | None = None, expect: str | None = None) -> int:
    """Upload a real document and prove the vendor read it.

    The upload path was merged once on unit tests alone and the vendor answered 400 — the only
    evidence that counts for `files` is a question whose answer exists solely inside the file.
    With no path a throwaway ``.txt`` carrying a random token is uploaded, so the endpoint, the
    ``purpose`` field and the ``file_id`` reference are all exercised on every run.
    """
    import random
    import tempfile

    token = (expect or f"ZXQ{random.randint(1000, 9999)}").upper()
    temporary = None
    if path is None:
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "acceptance-probe.txt"
        path.write_text(f"Field notes.\n\nThe verification token is {token}.\n", encoding="utf-8")
    if not path.is_file():
        print(f"FAIL: {path} is not a readable file")
        return 1
    try:
        result = client.research("Which verification token appears in this document? "
                                 "Answer with the token only.", files=[str(path)])
    finally:
        if temporary is not None:
            temporary.cleanup()
    answer = (result.get("answer") or "")
    print("files:", path.name, "| answer:", answer[:160].replace("\n", " "))
    if token.lower() not in answer.lower():
        print(f"FAIL: the answer does not contain {token} - the document did not reach the model")
        return 1
    print("OK (the document was parsed by the vendor)")
    return 0


def main() -> int:
    client = load_client()
    if not client.key_available():
        print("SKIP: DEEPSEEK_API_KEY is not available in this environment")
        return 2

    argv = sys.argv[1:]
    if "--files" in argv:
        index = argv.index("--files")
        extra = argv[index + 1:index + 2]
        provided = Path(extra[0]).expanduser() if extra else None
        expect = None
        if "--expect" in argv:
            position = argv.index("--expect")
            expect = (argv[position + 1:position + 2] or [None])[0]
        if provided is not None and expect is None:
            print("note: no --expect TOKEN given, so this only proves the upload succeeded;\n"
                  "      pass --expect <a word that is in the document> to assert it was read")
            return 0 if client.file_parts([str(provided)], key=client.resolve_api_key())[0] else 1
        return file_acceptance(client, provided, expect)
    if "--files-probe" in argv:
        return file_acceptance(client)

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
