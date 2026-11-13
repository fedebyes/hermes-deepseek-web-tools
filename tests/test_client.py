"""Offline unit tests for the deepseek-web-tools client (stdlib unittest, no pytest).

Run:  ``uv run config/asus/hermes-plugins/deepseek-web-tools/tests/test_client.py``

No network, no key, no plugin host: the transport is injected, so every wire behaviour
(retries, error mapping, parsing, rendering) is asserted without a vendor call. The live
behaviour is covered by ``scripts/live_accept.py``, which makes one real request.

pytest is deliberately not used: this directory is a package whose name contains a hyphen,
which pytest's collection cannot turn into a module name, and a config workaround there is
more moving parts than the suite is worth.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
KEY = "sk-UNIT-TEST-KEY"


def _load_client():
    spec = importlib.util.spec_from_file_location("dsweb_client", PLUGIN_DIR / "client.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["dsweb_client"] = module
    spec.loader.exec_module(module)
    return module


client = _load_client()


# ─── fakes ───────────────────────────────────────────────────────────────────

class _Response:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _opener(*script):
    """Play ``script``: a dict is a body, an int is an HTTP error with that status."""
    seen: list = []
    queue = list(script)

    def opener(request, timeout=None):
        seen.append(request)
        item = queue.pop(0)
        if isinstance(item, int):
            payload = json.dumps({"error": {"message": "vendor says no"}}).encode("utf-8")
            raise urllib.error.HTTPError(request.full_url, item, "error", {}, io.BytesIO(payload))
        return _Response(item)

    opener.seen = seen
    return opener


def _body(answer="answer", kinds=("deep_search",), citations=1, searches=2, fetches=0):
    calls = [{"kind": kind, "type": "deepseek_web_search", "status": "completed",
              "action": {"type": "search", "queries": ["q"]}} for kind in kinds]
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": answer,
                "annotations": [
                    {"type": "url_citation",
                     "url_citation": {"url": f"https://example.com/{i}", "title": f"Src {i}",
                                      "start_index": 0, "end_index": 5}}
                    for i in range(citations)
                ],
                "server_tool_calls": [{"index": 0, "reasoning_content": "", "calls": calls}],
            },
        }],
        "usage": {"server_tool_use": {"tool_search": searches, "tool_fetch": fetches,
                                      "tool_image_search": 0}},
    }


class _EnvMixin:
    """A DEEPSEEK_BASE_URL override that is restored even when an assertion fails."""

    def setUp(self):
        super().setUp()
        self._saved = os.environ.get("DEEPSEEK_BASE_URL")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("DEEPSEEK_BASE_URL", None)
        else:
            os.environ["DEEPSEEK_BASE_URL"] = self._saved
        super().tearDown()


# ─── payload ────────────────────────────────────────────────────────────────

class PayloadTests(unittest.TestCase):
    def test_defaults_declare_both_tools_and_disable_thinking(self):
        payload = client.build_payload(query="  what is new in deepseek v4.1  ")
        self.assertEqual(payload["model"], client.DEFAULT_MODEL)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertIs(payload["stream"], False)
        self.assertEqual([tool["type"] for tool in payload["tools"]],
                         ["deepseek_web_search", "deepseek_read_url"])
        self.assertEqual(payload["messages"][0]["content"], "what is new in deepseek v4.1")

    def test_thinking_and_read_url_are_independent_switches(self):
        payload = client.build_payload(query="q", thinking=True, with_read_url=False)
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual([tool["type"] for tool in payload["tools"]], ["deepseek_web_search"])

    def test_urls_are_embedded_in_the_prompt_and_capped_at_ten(self):
        urls = [f"https://example.com/{i}" for i in range(13)]
        content = client.build_payload(query="compare", urls=urls)["messages"][0]["content"]
        for url in urls[:10]:
            self.assertIn(url, content)
        self.assertNotIn("https://example.com/10", content)

    def test_image_becomes_a_content_block(self):
        content = client.build_payload(query="what is this",
                                       image_url="https://example.com/a.jpg")["messages"][0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "what is this"})
        self.assertEqual(content[1], {"type": "image_url",
                                      "image_url": {"url": "https://example.com/a.jpg"}})


# ─── endpoint guard ─────────────────────────────────────────────────────────

class BaseUrlGuardTests(_EnvMixin, unittest.TestCase):
    CASES = (
        ("", client.DEFAULT_BASE_URL),
        ("https://api.deepseek.com", "https://api.deepseek.com"),
        ("https://api.deepseek.com/", "https://api.deepseek.com"),
        ("http://127.0.0.1:8123", "http://127.0.0.1:8123"),
        ("http://localhost:9000", "http://localhost:9000"),
        ("https://api.deepseek.com.evil.example", client.DEFAULT_BASE_URL),
        ("http://api.deepseek.com", client.DEFAULT_BASE_URL),
        ("https://user:pw@api.deepseek.com", client.DEFAULT_BASE_URL),
        ("https://evil.example", client.DEFAULT_BASE_URL),
        ("https://api.deepseek.com/v1", client.DEFAULT_BASE_URL),
    )

    def test_guard(self):
        for value, expected in self.CASES:
            with self.subTest(value=value):
                os.environ["DEEPSEEK_BASE_URL"] = value
                self.assertEqual(client.base_url(), expected)


# ─── transport ──────────────────────────────────────────────────────────────

class TransportTests(unittest.TestCase):
    def _post(self, opener):
        return client.post({"model": "m"}, key=KEY, base="https://api.deepseek.com",
                           opener=opener, sleep=lambda _s: None)

    def test_retries_429_then_succeeds(self):
        opener = _opener(429, {"choices": []})
        self.assertEqual(self._post(opener), {"choices": []})
        self.assertEqual(len(opener.seen), 2)

    def test_gives_up_after_three_attempts(self):
        opener = _opener(500, 500, 500)
        with self.assertRaises(client.WebToolsError) as ctx:
            self._post(opener)
        self.assertEqual(len(opener.seen), 3)
        self.assertIn("HTTP 500", str(ctx.exception))

    def test_does_not_retry_a_400_and_carries_the_hint(self):
        opener = _opener(400)
        with self.assertRaises(client.WebToolsError) as ctx:
            self._post(opener)
        message = str(ctx.exception)
        self.assertEqual(len(opener.seen), 1)
        self.assertIn("tool_choice", message)      # the 400 hint
        self.assertIn("vendor says no", message)   # the vendor's own text
        self.assertNotIn(KEY, message)             # the key is never echoed

    def test_sends_bearer_auth_and_json(self):
        opener = _opener(_body())
        self._post(opener)
        request = opener.seen[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.get_header("Authorization"), f"Bearer {KEY}")
        self.assertEqual(request.get_header("Content-type"), "application/json")


# ─── response ───────────────────────────────────────────────────────────────

class ResponseTests(unittest.TestCase):
    def test_flattens_calls_and_citations(self):
        result = client.parse_response(_body(kinds=("deep_search", "read_link"),
                                             citations=2, searches=3, fetches=1))
        self.assertEqual(result["answer"], "answer")
        self.assertEqual([call["kind"] for call in result["calls"]], ["deep_search", "read_link"])
        self.assertEqual([cite["url"] for cite in result["citations"]],
                         ["https://example.com/0", "https://example.com/1"])
        self.assertEqual(result["usage"]["server_tool_use"]["tool_fetch"], 1)
        self.assertEqual(result["finish_reason"], "stop")

    def test_without_choices_raises(self):
        with self.assertRaises(client.WebToolsError):
            client.parse_response({"choices": []})

    def test_render_shows_answer_sources_steps_billing_and_the_caveat(self):
        text = client.render(client.parse_response(_body(citations=1, searches=2)), thinking=False)
        self.assertTrue(text.startswith("answer"))
        self.assertIn("Sources:", text)
        self.assertIn("https://example.com/0", text)
        self.assertIn("deep_search ×1", text)
        self.assertIn("search=2", text)
        self.assertIn("web_extract", text)         # the "page text is not returned" caveat
        self.assertIn("single search step", text)

    def test_flags_a_non_stop_finish_reason(self):
        body = _body()
        body["choices"][0]["finish_reason"] = "length"
        text = client.render(client.parse_response(body), thinking=True)
        self.assertIn("finish_reason: length", text)
        self.assertIn("deep thinking search", text)


# ─── run() ──────────────────────────────────────────────────────────────────

class RunTests(unittest.TestCase):
    def test_requires_a_query(self):
        out = client.run({"query": "   "})
        self.assertIn("non-empty `query`", out)
        self.assertFalse(out.startswith("web_research failed:"))

    def test_a_vendor_error_becomes_a_string(self):
        original = client.research
        client.research = lambda *_a, **_kw: (_ for _ in ()).throw(
            client.WebToolsError("HTTP 402 (DeepSeek account balance too low)"))
        try:
            out = client.run({"query": "q"})
        finally:
            client.research = original
        self.assertTrue(out.startswith("web_research failed: HTTP 402"))
        self.assertNotIn("Traceback", out)

    def _capture(self, args, **kwargs):
        captured: dict = {}

        def fake_research(query, **inner):
            captured.update(inner)
            return client.parse_response(_body(citations=0))

        original = client.research
        client.research = fake_research
        try:
            out = client.run(args, **kwargs)
        finally:
            client.research = original
        return captured, out

    def test_caps_urls_and_reports_the_dropped_ones(self):
        captured, out = self._capture({"query": "q",
                                       "urls": [f"https://e.com/{i}" for i in range(12)]})
        self.assertEqual(len(captured["urls"]), 10)
        self.assertIn("2 extra URL(s) ignored", out)

    def test_per_call_thinking_override_wins(self):
        captured, _ = self._capture({"query": "q", "thinking": True}, thinking_default=False)
        self.assertIs(captured["thinking"], True)
        captured, _ = self._capture({"query": "q"}, thinking_default=True)
        self.assertIs(captured["thinking"], True)

    def test_accepts_a_bare_string_url(self):
        captured, _ = self._capture({"query": "q", "urls": "https://example.com/x"})
        self.assertEqual(captured["urls"], ["https://example.com/x"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
