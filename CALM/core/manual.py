from __future__ import annotations

import argparse
import html
import re
import shutil
import textwrap
from importlib import resources
from pathlib import Path
from typing import Any


class ManualAction(argparse.Action):
    """Argparse action that prints a command manual and exits immediately."""

    def __init__(
        self,
        option_strings: list[str],
        dest: str = argparse.SUPPRESS,
        default: str = argparse.SUPPRESS,
        **kwargs: Any,
    ) -> None:
        self.manual_name: str = kwargs.pop("manual_name")
        kwargs.setdefault("nargs", 0)
        super().__init__(option_strings, dest=dest, default=default, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        print(render_manual(self.manual_name))
        parser.exit(0)


def render_manual(name: str) -> str:
    """Load a Markdown manual from CALM/manuals and render it for terminal output."""
    markdown = load_manual_markdown(name)
    return render_markdown_for_terminal(markdown)


def load_manual_markdown(name: str) -> str:
    filename = f"{name}.md"

    try:
        package_files = resources.files("CALM.manuals")
        return package_files.joinpath(filename).read_text(encoding="utf-8")
    except Exception:
        local_path = Path(__file__).resolve().parent.parent / "manuals" / filename
        if local_path.exists():
            return local_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"CALM manual not found: {filename}") from None


def render_markdown_for_terminal(markdown: str) -> str:
    """Render a command manual's Markdown for terminal output: keeps code
    blocks readable, converts headings to terminal-style section headers,
    and strips inline Markdown markers. Handles the specific subset of
    Markdown CALM's manuals use (headings, bullets, numbered lists, code
    fences, inline emphasis/links)."""
    width = shutil.get_terminal_size((100, 24)).columns
    width = max(72, min(width, 120))

    lines: list[str] = []
    in_code = False

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()

        if line.strip().startswith("```"):
            in_code = not in_code
            if not in_code:
                lines.append("")
            continue

        if in_code:
            lines.append(f"  {line}")
            continue

        if not line.strip():
            if lines and lines[-1] != "":
                lines.append("")
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            text = strip_inline_markdown(heading.group(2)).strip()
            if level == 1:
                lines.append(text.upper())
                lines.append("=" * min(len(text), width))
            elif level == 2:
                lines.append(text)
                lines.append("-" * min(len(text), width))
            else:
                lines.append(text + ":")
            continue

        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet:
            text = strip_inline_markdown(bullet.group(1)).strip()
            wrapped = textwrap.wrap(text, width=width - 4) or [""]
            lines.append(f"  • {wrapped[0]}")
            for cont in wrapped[1:]:
                lines.append(f"    {cont}")
            continue

        numbered = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
        if numbered:
            prefix = f"  {numbered.group(1)}. "
            text = strip_inline_markdown(numbered.group(2)).strip()
            wrapped = textwrap.wrap(text, width=width - len(prefix)) or [""]
            lines.append(prefix + wrapped[0])
            for cont in wrapped[1:]:
                lines.append(" " * len(prefix) + cont)
            continue

        text = strip_inline_markdown(line)
        wrapped = textwrap.wrap(text, width=width) or [""]
        lines.extend(wrapped)

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def strip_inline_markdown(text: str) -> str:
    """Strip Markdown code spans, bold/italic emphasis, and links, rendering plain text.

    Code-span content is protected from every later substitution (stashed
    behind a placeholder, restored verbatim at the end) - otherwise an
    identifier with two or more underscores (e.g. `_assign_nearest_leaflet`)
    would have its own underscores misread as italic markers once the
    surrounding backticks are gone, corrupting it (e.g. into
    "assignnearest_leaflet").
    """
    code_spans: list[str] = []

    def _stash_code(match: "re.Match[str]") -> str:
        code_spans.append(match.group(1))
        return f"\x00{len(code_spans) - 1}\x00"

    text = re.sub(r"`([^`]*)`", _stash_code, text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"__([^_]*)__", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    text = re.sub(r"_([^_]*)_", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    for i, code in enumerate(code_spans):
        text = text.replace(f"\x00{i}\x00", code)
    return text


def _inline_markdown_to_html(text: str) -> str:
    """Convert inline Markdown (code spans, bold/italic, links) to the equivalent HTML tags.

    Mirrors `strip_inline_markdown`'s structure - code-span content is
    stashed behind a placeholder before the other substitutions run (same
    reason: an identifier's own underscores must not be misread as italic
    markers once its backticks are gone), restored at the end already
    wrapped in `<code>`. The whole line is HTML-escaped first, so every
    substitution below already operates on safe text.
    """
    text = html.escape(text)
    code_spans: list[str] = []

    def _stash_code(match: "re.Match[str]") -> str:
        code_spans.append(f"<code>{match.group(1)}</code>")
        return f"\x00{len(code_spans) - 1}\x00"

    text = re.sub(r"`([^`]*)`", _stash_code, text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__([^_]*)__", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]*)\*", r"<em>\1</em>", text)
    text = re.sub(r"_([^_]*)_", r"<em>\1</em>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)

    for i, code in enumerate(code_spans):
        text = text.replace(f"\x00{i}\x00", code)
    return text


def render_markdown_as_html(markdown: str, title: str = "CALM manual") -> str:
    """Render a command manual's Markdown as a styled, self-contained HTML page.

    Handles the same Markdown subset `render_markdown_for_terminal` does
    (headings, bullets, numbered lists, code fences, inline emphasis/
    links), via the same line-by-line parse, emitting real HTML tags
    instead of plain text - a browser has no native Markdown rendering of
    its own, so opening a `.md` file directly just shows its raw source.
    """
    body: list[str] = []
    in_code = False
    code_lines: list[str] = []
    list_kind: str | None = None  # "ul"/"ol"/None - which list is currently open

    def close_list() -> None:
        nonlocal list_kind
        if list_kind is not None:
            body.append(f"</{list_kind}>")
            list_kind = None

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()

        if line.strip().startswith("```"):
            if not in_code:
                close_list()
                code_lines = []
            else:
                body.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
            in_code = not in_code
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            close_list()
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            close_list()
            level = len(heading.group(1))
            text = _inline_markdown_to_html(heading.group(2).strip())
            body.append(f"<h{level}>{text}</h{level}>")
            continue

        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet:
            if list_kind != "ul":
                close_list()
                body.append("<ul>")
                list_kind = "ul"
            body.append(f"<li>{_inline_markdown_to_html(bullet.group(1).strip())}</li>")
            continue

        numbered = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
        if numbered:
            if list_kind != "ol":
                close_list()
                body.append("<ol>")
                list_kind = "ol"
            body.append(f"<li>{_inline_markdown_to_html(numbered.group(2).strip())}</li>")
            continue

        close_list()
        body.append(f"<p>{_inline_markdown_to_html(line)}</p>")

    close_list()

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 860px;
        margin: 2rem auto; padding: 0 1.5rem; line-height: 1.5; color: #1a1a1a; }}
h1, h2, h3, h4, h5, h6 {{ margin-top: 1.6em; }}
code, pre {{ font-family: ui-monospace, "SF Mono", Consolas, monospace; background: #f2f2f2; border-radius: 4px; }}
code {{ padding: 0.1em 0.35em; }}
pre {{ padding: 0.75em 1em; overflow-x: auto; }}
pre code {{ background: none; padding: 0; }}
a {{ color: #0b5fff; }}
</style>
</head>
<body>
{chr(10).join(body)}
</body>
</html>
"""


def add_manual(parser: argparse.ArgumentParser, manual_name: str) -> None:
    parser.add_argument(
        "--man",
        action=ManualAction,
        manual_name=manual_name,
        help="show the full manual for this command and exit",
    )
