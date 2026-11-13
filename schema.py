"""JSON schema for the single tool this plugin registers.

Shaped like the bundled google_meet tools: ``{"name", "description", "parameters"}``,
``additionalProperties: false`` so a wrong argument is refused by the model's own
validator instead of being silently dropped here.
"""

from __future__ import annotations

from typing import Any


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


WEB_RESEARCH_SCHEMA = _schema(
    "web_research",
    "Answer a question from the live web using DeepSeek's server-side web tools: the model "
    "searches, opens pages and finds text inside them on the vendor's side, then answers with "
    "structured citations. One call instead of a search-then-fetch loop, so it is the right tool "
    "when the answer - not the page - is what you need. It does NOT return page text (the vendor "
    "keeps it encrypted), so use web_extract when you must read a page yourself. thinking=false "
    "(default): one search step, fast and cheapest. thinking=true: deep thinking search, up to "
    "~20 calls of search/open_page/find_in_page to check several facts against their sources - "
    "slower and billed per search, use it for multi-source verification.",
    {
        "query": {
            "type": "string",
            "description": "The question to answer, in natural language. Ask for what you want to "
                           "know and say how it should be sourced when it matters (e.g. 'from the "
                           "manufacturer's own spec page').",
        },
        "urls": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
            "description": "Optional http(s) pages to read as the basis of the answer. The vendor "
                           "considers the first 10 distinct URLs only, and the model picks which "
                           "ones to read. Ignored when thinking=true: with thinking on the vendor "
                           "runs deep thinking search and does not read links separately.",
        },
        "image_url": {
            "type": "string",
            "description": "Optional https URL of an image to search/reason about (image search is "
                           "part of the web-search tool). Publicly reachable, at most 8192 "
                           "characters; local files must be passed as a data: URL by the caller.",
        },
        "thinking": {
            "type": "boolean",
            "description": "Override the plugin default for this call. true = deep thinking search "
                           "(multi-step, more expensive), false = single search step.",
        },
    },
    required=["query"],
)
