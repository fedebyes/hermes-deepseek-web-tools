"""Offline unit tests for the deepseek-web-tools client (stdlib unittest, no pytest).

Run:  ``uv run tests/test_client.py``

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
import tempfile
import types
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


# ─── file input ─────────────────────────────────────────────────────────────

class FileInputTests(unittest.TestCase):
    """``files`` is the vendor's document input: upload first, then reference the ``file_id``."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name) / "files.json"

    def _parts(self, paths, ids=("file-api-test-1",), **kwargs):
        """Run file_parts against a scripted Files API; requests are the wire evidence."""
        opener = _opener(*[{"id": file_id} for file_id in ids])
        parts, notes = client.file_parts(paths, key="k", opener=opener,
                                         cache_path=self.cache, **kwargs)
        return parts, notes, opener.seen

    @staticmethod
    def _form(request) -> bytes:
        return request.data if isinstance(request.data, bytes) else bytes(request.data)

    def _write(self, name: str, size: int = 16, contents: bytes | None = None) -> str:
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as handle:
            handle.write(contents if contents is not None else b"x" * size)
        return path

    def test_a_pdf_is_uploaded_and_referenced_by_file_id(self):
        path = self._write("report.pdf", contents=b"%PDF-1.4 fake")
        parts, notes, sent = self._parts([path])
        self.assertEqual(notes, [])
        self.assertEqual(parts, [{"type": "file", "file_id": "file-api-test-1"}])
        self.assertEqual(len(sent), 1)
        body = self._form(sent[0])
        self.assertEqual(sent[0].full_url, f"{client.base_url()}/files")
        self.assertIn(b'name="purpose"', body)
        # the vendor requires the purpose field before the file field for documents
        self.assertLess(body.index(b'name="purpose"'), body.index(b'name="file"'))
        self.assertIn(b"rich_layout_recognition", body)
        self.assertIn(b'filename="report.pdf"', body)
        self.assertIn(b"%PDF-1.4 fake", body)

    def test_the_purpose_follows_the_file_type(self):
        docx = self._write("a.docx")
        png = self._write("b.png")
        parts, notes, sent = self._parts([docx, png], ids=("file-api-1", "file-api-2"))
        self.assertEqual(notes, [])
        self.assertEqual([p["file_id"] for p in parts], ["file-api-1", "file-api-2"])
        first, second = (self._form(request) for request in sent)
        self.assertIn(b"rich_layout_recognition", first)      # Word: text AND figures
        self.assertIn(b'application/vnd.openxmlformats', first)
        self.assertIn(b"user_data", second)                   # images take user_data only
        self.assertIn(b"image/png", second)

    def test_the_same_file_is_uploaded_once_and_a_changed_one_again(self):
        path = self._write("report.pdf", contents=b"%PDF-v1")
        self._parts([path])
        parts, notes, sent = self._parts([path])               # unchanged -> cache hit
        self.assertEqual(parts, [{"type": "file", "file_id": "file-api-test-1"}])
        self.assertEqual(sent, [])
        self._write("report.pdf", contents=b"%PDF-v2-changed")
        _, _, sent = self._parts([path])                       # content changed -> new upload
        self.assertEqual(len(sent), 1)

    def test_a_missing_key_leaves_the_file_out_and_says_so(self):
        path = self._write("report.pdf")
        opener = _opener({"id": "file-api-never"})
        parts, notes = client.file_parts([path], key=None, opener=opener,
                                         cache_path=self.cache)
        self.assertEqual(parts, [])
        self.assertIn("no API key available to upload", notes[0])
        self.assertEqual(opener.seen, [])

    def test_an_upload_refusal_is_reported_not_raised(self):
        path = self._write("report.pdf")
        opener = _opener(403)                                  # the vendor refuses this upload
        parts, notes = client.file_parts([path], key="k", opener=opener,
                                         cache_path=self.cache)
        self.assertEqual(parts, [])
        self.assertIn("skipped - upload refused", notes[0])
        self.assertIn("HTTP 403", notes[0])

    def test_a_relative_path_is_refused_with_a_reason(self):
        parts, notes = client.file_parts(["report.pdf"])
        self.assertEqual(parts, [])
        self.assertIn("absolute path is required", notes[0])

    def test_an_unsupported_extension_is_refused_and_the_allowed_ones_named(self):
        path = self._write("archive.zip")
        parts, notes = client.file_parts([path])
        self.assertEqual(parts, [])
        self.assertIn("unsupported type", notes[0])
        self.assertIn(".pdf", notes[0])

    def test_a_file_over_the_size_cap_is_refused_not_truncated(self):
        path = self._write("big.pdf", size=client.MAX_FILE_BYTES + 1)
        parts, notes = client.file_parts([path])
        self.assertEqual(parts, [])
        self.assertIn("over the", notes[0])

    def test_an_empty_file_is_refused(self):
        path = self._write("empty.pdf", contents=b"")
        parts, notes = client.file_parts([path])
        self.assertEqual(parts, [])
        self.assertIn("empty file", notes[0])

    def test_a_missing_file_is_refused(self):
        parts, notes = client.file_parts([os.path.join(self.tmp.name, "nope.pdf")])
        self.assertEqual(parts, [])
        self.assertTrue(notes)

    def test_credentials_are_never_attached(self):
        dotenv = self._write(".env", contents=b"SECRET=1\n")
        nested = os.path.join(self.tmp.name, ".local", "secrets", "key.txt")
        os.makedirs(os.path.dirname(nested), exist_ok=True)
        with open(nested, "w", encoding="utf-8") as handle:
            handle.write("s\n")
        parts, notes = client.file_parts([dotenv, nested])
        self.assertEqual(parts, [])
        self.assertEqual(len(notes), 2)
        for note in notes:
            self.assertIn("credentials are never attached", note)
        # nothing was read: an empty part list is the assertion that the bytes never left
        self.assertEqual(parts, [])

    def test_a_symlink_into_a_credential_store_is_refused_by_its_target(self):
        """A link named innocent.pdf must be judged by where it points, not by its own name."""
        secret_dir = os.path.join(self.tmp.name, ".local", "secrets")
        os.makedirs(secret_dir, exist_ok=True)
        secret = os.path.join(secret_dir, "key.txt")
        with open(secret, "w", encoding="utf-8") as handle:
            handle.write("s\n")
        link = os.path.join(self.tmp.name, "innocent.pdf")
        os.symlink(secret, link)
        parts, notes = client.file_parts([link])
        self.assertEqual(parts, [])
        self.assertIn("credentials are never attached", notes[0])

    def test_a_legit_symlink_resolves_and_uploads_the_target_name(self):
        real = self._write("real.pdf")
        link = os.path.join(self.tmp.name, "alias.pdf")
        os.symlink(real, link)
        parts, notes, sent = self._parts([link])
        self.assertEqual(notes, [])
        self.assertEqual(len(parts), 1)
        self.assertIn(b'filename="real.pdf"', self._form(sent[0]))

    def test_a_dot_env_symlink_is_refused(self):
        env_file = self._write("dotenv-source", contents=b"SECRET=1\n")
        link = os.path.join(self.tmp.name, ".env")
        os.symlink(env_file, link)
        parts, notes = client.file_parts([link])
        self.assertEqual(parts, [])
        self.assertIn("credentials are never attached", notes[0])

    def test_at_most_three_files_and_the_extras_are_reported(self):
        paths = [self._write(f"f{i}.txt") for i in range(5)]
        parts, notes, _ = self._parts(paths, ids=tuple(f"file-api-{i}" for i in range(5)))
        self.assertEqual(len(parts), client.MAX_FILES)
        self.assertEqual(len(notes), 2)
        self.assertIn("at most 3 files", notes[0])

    def test_the_total_payload_cap_is_enforced(self):
        big = self._write("a.pdf", size=client.MAX_FILE_BYTES)
        second = self._write("b.pdf", size=client.MAX_FILE_BYTES)
        parts, notes, _ = self._parts([big, second])
        self.assertEqual(len(parts), 1)
        self.assertIn("would put the payload over", notes[0])

    def test_file_parts_follow_the_text_in_the_payload(self):
        path = self._write("report.pdf")
        parts, _, _ = self._parts([path])
        content = client.build_payload(query="summarise this", files=parts)["messages"][0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(content[0], {"type": "text", "text": "summarise this"})
        self.assertEqual(content[1], {"type": "file", "file_id": "file-api-test-1"})

    def test_no_files_keeps_the_plain_string_content(self):
        content = client.build_payload(query="q")["messages"][0]["content"]
        self.assertEqual(content, "q")


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
        # unparseable values must fall back, not raise: a malformed env var used to crash every
        # call because the warning branch re-parsed the string that had just failed
        ("https://[::1", client.DEFAULT_BASE_URL),
        ("https://[bad", client.DEFAULT_BASE_URL),
        ("https://api.deepseek.com:99999999", client.DEFAULT_BASE_URL),
        ("ftp://x", client.DEFAULT_BASE_URL),
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
        self.assertIn("2 URL(s) over the vendor's 10-page cap not sent", out)

    def test_per_call_thinking_override_wins(self):
        captured, _ = self._capture({"query": "q", "thinking": True}, thinking_default=False)
        self.assertIs(captured["thinking"], True)
        captured, _ = self._capture({"query": "q"}, thinking_default=True)
        self.assertIs(captured["thinking"], True)

    def test_accepts_a_bare_string_url(self):
        captured, _ = self._capture({"query": "q", "urls": "https://example.com/x"})
        self.assertEqual(captured["urls"], ["https://example.com/x"])

    def test_files_are_uploaded_and_passed_as_file_ids(self):
        import hashlib
        import time as _time
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "notes.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("hello")
            cache = Path(tmp) / "files.json"
            digest = hashlib.sha256(b"hello").hexdigest()
            cache.write_text(json.dumps({f"{digest}:user_data":
                                         {"id": "file-api-cached", "at": int(_time.time())}}))
            original_cache = client.FILES_CACHE
            client.FILES_CACHE = cache
            try:
                captured, out = self._capture({"query": "q", "files": [path]})
            finally:
                client.FILES_CACHE = original_cache
        self.assertEqual(captured["files"], [{"type": "file", "file_id": "file-api-cached"}])
        self.assertNotIn("skipped", out)

    def test_a_refused_file_is_reported_in_the_answer(self):
        captured, out = self._capture({"query": "q", "files": ["relative.txt"]})
        self.assertIsNone(captured.get("files"))
        self.assertIn("skipped - an absolute path is required", out)

    def test_a_bare_string_file_is_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "one.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("x")
            # no key here: the point is the string is accepted as a path, not that it uploads
            captured, out = self._capture({"query": "q", "files": path})
        self.assertIsNone(captured.get("files"))
        self.assertIn("one.txt", out)
        self.assertNotIn("absolute path is required", out)


# ─── credential scope ───────────────────────────────────────────────────────

class SecretScopeTests(unittest.TestCase):
    """The key must come from the scoped accessor, and an unscoped read must hide the tool.

    Falling back to ``os.environ`` on ``UnscopedSecretError`` would hand a multiplexed install
    another profile's credential (catalog rule 11), so the fallback is asserted against.
    """

    NAMES = ("agent", "agent.secret_scope")

    def setUp(self):
        self._saved = {name: sys.modules.get(name) for name in self.NAMES}
        self._saved_env = os.environ.get("DEEPSEEK_API_KEY")

    def tearDown(self):
        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if self._saved_env is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = self._saved_env

    def _install(self, get_secret):
        agent = types.ModuleType("agent")
        scope = types.ModuleType("agent.secret_scope")
        scope.get_secret = get_secret
        agent.secret_scope = scope
        sys.modules["agent"] = agent
        sys.modules["agent.secret_scope"] = scope

    def test_key_comes_from_the_scoped_accessor(self):
        seen = {}

        def get_secret(name, default=None):
            seen["name"] = name
            return "sk-from-scope"

        self._install(get_secret)
        self.assertEqual(client.resolve_api_key(), "sk-from-scope")
        self.assertEqual(seen["name"], "DEEPSEEK_API_KEY")

    def test_an_unscoped_read_hides_the_tool_and_never_touches_env(self):
        os.environ["DEEPSEEK_API_KEY"] = "sk-env-must-not-be-used"

        class UnscopedSecretError(RuntimeError):
            """Mirrors the core class by name: that is what the plugin keys on."""

        def get_secret(name, default=None):
            raise UnscopedSecretError("no secret scope installed")

        self._install(get_secret)
        self.assertFalse(client.key_available())
        with self.assertRaises(UnscopedSecretError):
            client.resolve_api_key()          # the raise is not swallowed either

    def test_any_other_secret_failure_is_left_to_the_registry(self):
        def get_secret(name, default=None):
            raise RuntimeError("a real bug, not a scope problem")

        self._install(get_secret)
        with self.assertRaises(RuntimeError):   # fail loud, do not hide a broken install
            client.key_available()


# ─── argument and body coercion ─────────────────────────────────────────────

class CoercionTests(unittest.TestCase):
    def test_as_bool_parses_string_forms(self):
        for value in ("false", "0", "off", "no", "", "  false  "):
            with self.subTest(value=value):
                self.assertIs(client.as_bool(value, default=True), False)
        for value in ("true", "1", "on", "yes", 1, 2.0, True):
            with self.subTest(value=value):
                self.assertIs(client.as_bool(value, default=False), True)
        self.assertIs(client.as_bool(None, default=True), True)

    def test_a_string_false_from_the_model_does_not_buy_a_deep_search(self):
        captured = {}

        def fake_research(query, **kwargs):
            captured.update(kwargs)
            return client.parse_response(_body(citations=0))

        original = client.research
        client.research = fake_research
        try:
            client.run({"query": "q", "thinking": "false"}, thinking_default=True)
        finally:
            client.research = original
        self.assertIs(captured["thinking"], False)

    def test_duplicate_urls_do_not_spend_slots(self):
        urls = ["https://a.example", "https://a.example", " https://b.example "]
        urls += [f"https://c.example/{i}" for i in range(12)]
        picked = client.dedupe_urls(urls)
        self.assertEqual(len(picked), 10)
        self.assertEqual(picked[:2], ["https://a.example", "https://b.example"])

    def test_a_duplicate_url_is_reported_only_when_one_was_dropped(self):
        captured = {}

        def fake_research(query, **kwargs):
            captured.update(kwargs)
            return client.parse_response(_body(citations=0))

        original = client.research
        client.research = fake_research
        try:
            out = client.run({"query": "q", "urls": ["https://a.example", "https://a.example"]})
        finally:
            client.research = original
        self.assertEqual(captured["urls"], ["https://a.example"])
        self.assertIn("1 duplicate or blank URL(s) ignored", out)

    def test_a_generation_failure_surfaces_the_vendor_message(self):
        body = {"error": {"message": "please retry", "code": "server_error"},
                "usage": {"server_tool_use": {"tool_search": 1}}}
        with self.assertRaises(client.WebToolsError) as ctx:
            client.parse_response(body)
        self.assertIn("please retry", str(ctx.exception))
        self.assertIn("server_error", str(ctx.exception))

    def test_a_generation_failure_still_reports_what_was_billed(self):
        body = {"error": {"message": "content refused"}, "usage": {
            "server_tool_use": {"tool_search": 12, "tool_fetch": 3, "tool_image_search": 0}}}
        with self.assertRaises(client.WebToolsError) as ctx:
            client.parse_response(body)
        self.assertIn("search=12", str(ctx.exception))
        self.assertIn("fetch=3", str(ctx.exception))       # the only record of a billed failure

    def test_a_trailing_slash_does_not_spend_a_second_slot(self):
        picked = client.dedupe_urls(["https://x.example/a", "https://x.example/a/"])
        self.assertEqual(picked, ["https://x.example/a"])

    def test_duplicates_and_the_page_cap_are_reported_as_different_causes(self):
        captured: dict = {}

        def fake_research(query, **kwargs):
            captured.update(kwargs)
            return client.parse_response(_body(citations=0))

        original = client.research
        client.research = fake_research
        try:
            urls = [f"https://e.com/{i}" for i in range(12)] + ["https://e.com/0"]
            out = client.run({"query": "q", "urls": urls})
        finally:
            client.research = original
        self.assertEqual(len(captured["urls"]), 10)
        self.assertIn("1 duplicate or blank URL(s) ignored", out)
        self.assertIn("2 URL(s) over the vendor's 10-page cap not sent", out)

    def test_only_tool_counters_are_called_billed(self):
        usage = {"tool_search": 3, "tool_fetch": 2, "tool_image_search": 0,
                 "tool_other": 5, "some_display_stat": 9, "unknown_not_an_int": "x"}
        self.assertEqual(client.billed_counters(usage),
                         "fetch=2, other=5, search=3")   # display stats are not money
        self.assertEqual(client.billed_counters(None), "")
        self.assertEqual(client.billed_counters({"tool_search": 0}), "")

    def test_a_dict_of_urls_does_not_leak_its_keys(self):
        captured = {}

        def fake_research(query, **kwargs):
            captured.update(kwargs)
            return client.parse_response(_body(citations=0))

        original = client.research
        client.research = fake_research
        try:
            client.run({"query": "q", "urls": {"a": "https://a.example"}})
        finally:
            client.research = original
        self.assertEqual(captured["urls"], ["https://a.example"])

    def test_the_timeout_is_clamped(self):
        self.assertEqual(client.clamp_timeout(0), client.MIN_TIMEOUT_S)
        self.assertEqual(client.clamp_timeout(-5), client.MIN_TIMEOUT_S)
        self.assertEqual(client.clamp_timeout(9999), client.MAX_TIMEOUT_S)
        self.assertEqual(client.clamp_timeout(30), 30)
        self.assertEqual(client.clamp_timeout("nonsense"), client.TIMEOUT_S)

    def test_the_clamped_timeout_reaches_the_call(self):
        captured = {}

        def fake_research(query, **kwargs):
            captured.update(kwargs)
            return client.parse_response(_body(citations=0))

        original = client.research
        client.research = fake_research
        try:
            client.run({"query": "q"}, timeout=9999)
        finally:
            client.research = original
        self.assertEqual(captured["timeout"], client.MAX_TIMEOUT_S)

    def test_a_failed_call_shows_its_error_code(self):
        body = _body(citations=0)
        body["choices"][0]["message"]["server_tool_calls"][0]["calls"][0].update(
            {"status": "failed", "error": {"code": "url_not_allowed"}})
        text = client.render(client.parse_response(body), thinking=False)
        self.assertIn("Failed calls:", text)
        self.assertIn("url_not_allowed", text)


# ─── registration and the settings contract ─────────────────────────────────

def _load_plugin_package():
    """Import the plugin package the way the loader does — as a package, so `from . import client`
    resolves. The directory name contains a hyphen, which is why this is not a plain import."""
    name = "dsweb_plugin_under_test"
    spec = importlib.util.spec_from_file_location(
        name, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)])
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RegisterTests(unittest.TestCase):
    """`hermes plugins validate` proves declared capabilities match registrations; it cannot see
    WHICH keys `get_config` is asked for. `model`, `plugins`, `security` and `settings` are
    reserved roots that raise, and `_cfg` maps a raise to the default — so renaming a setting back
    to `model` would leave it silently inert. These assertions are the only thing that catches it.
    """

    RESERVED = {"model", "plugins", "security", "settings"}

    def setUp(self):
        self.plugin = _load_plugin_package()

    @staticmethod
    def _ctx(tools, asked, raise_on_config=False):
        class Ctx:
            def get_config(self, key, default=None):
                asked.append(key)
                if raise_on_config:
                    raise ValueError("Rejected config path")
                return default

            def register_tool(self, **kwargs):
                tools.append(kwargs)

        return Ctx()

    def test_asks_only_legal_settings_keys_and_registers_one_tool(self):
        asked: list = []
        tools: list = []
        self.plugin.register(self._ctx(tools, asked))

        self.assertTrue(asked)
        roots = {key.split(".")[0] for key in asked}
        self.assertFalse(roots & self.RESERVED, f"reserved settings root in {sorted(roots)}")
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "web_research")
        self.assertEqual(tools[0]["toolset"], "web")
        self.assertTrue(callable(tools[0]["handler"]))
        self.assertTrue(callable(tools[0]["check_fn"]))

    def test_a_rejected_settings_read_does_not_take_the_tool_down(self):
        tools: list = []
        self.plugin.register(self._ctx(tools, [], raise_on_config=True))   # must not raise
        self.assertEqual(len(tools), 1)

    def test_the_handler_returns_a_string_when_no_key_is_available(self):
        tools: list = []
        self.plugin.register(self._ctx(tools, []))
        previous = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            out = tools[0]["handler"]({"query": "q"})
        finally:
            if previous is not None:
                os.environ["DEEPSEEK_API_KEY"] = previous
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("web_research failed:"), out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
