"""Tests for core/manual.py's strip_inline_markdown and render_markdown_as_html.

strip_inline_markdown coverage is regression coverage for a bug where a
backtick code span's own content wasn't protected from the later
italic-underscore regex: once the backticks were stripped, an identifier
with two or more underscores (e.g. `_assign_nearest_leaflet`) had its own
underscores misread as italic markers, corrupting it (e.g. into
"assignnearest_leaflet"). render_markdown_as_html covers the GUI's Manual
button's renderer: the same multi-underscore code-span case, HTML-escaping
of special characters, and basic block structure (headings, lists, code
fences).
"""

from __future__ import annotations

from CALM.core.manual import render_markdown_as_html, strip_inline_markdown


def test_strip_inline_markdown_preserves_multi_underscore_code_spans() -> None:
    assert strip_inline_markdown("the `_assign_nearest_leaflet` function") == "the _assign_nearest_leaflet function"
    assert strip_inline_markdown("the `area_per_lipid.csv` file") == "the area_per_lipid.csv file"


def test_strip_inline_markdown_still_strips_real_emphasis_and_links() -> None:
    assert strip_inline_markdown("this is _italic_ text") == "this is italic text"
    assert strip_inline_markdown("this is **bold** text") == "this is bold text"
    assert strip_inline_markdown("this is __also bold__ text") == "this is also bold text"
    assert strip_inline_markdown("a [link](http://example.com/x) here") == "a link here"


def test_strip_inline_markdown_handles_code_span_and_real_emphasis_together() -> None:
    result = strip_inline_markdown("mixed `_code_here_` and _italic_ together")
    assert result == "mixed _code_here_ and italic together"


def test_strip_inline_markdown_handles_multiple_code_spans_on_one_line() -> None:
    result = strip_inline_markdown("both `_first_span_` and `_second_span_` survive")
    assert result == "both _first_span_ and _second_span_ survive"


def test_render_markdown_as_html_preserves_multi_underscore_code_spans() -> None:
    result = render_markdown_as_html("the `_assign_nearest_leaflet` function")
    assert "<code>_assign_nearest_leaflet</code>" in result
    assert "assignnearest_leaflet" not in result


def test_render_markdown_as_html_escapes_special_characters() -> None:
    result = render_markdown_as_html("a `<script>` tag and a & sign")
    assert "<script>" not in result  # never unescaped outside its own <code> wrapper
    assert "<code>&lt;script&gt;</code>" in result
    assert "&amp; sign" in result


def test_render_markdown_as_html_renders_headings_lists_and_code_fences() -> None:
    md = "# Title\n\n- one\n- two\n\n```\nraw code\n```\n"
    result = render_markdown_as_html(md, title="My Manual")
    assert "<title>My Manual</title>" in result
    assert "<h1>Title</h1>" in result
    assert "<ul>" in result and "<li>one</li>" in result and "<li>two</li>" in result
    assert "<pre><code>raw code</code></pre>" in result
