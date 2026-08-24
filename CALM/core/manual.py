from __future__ import annotations

import argparse
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


def add_manual(parser: argparse.ArgumentParser, manual_name: str) -> None:
    parser.add_argument(
        "--man",
        action=ManualAction,
        manual_name=manual_name,
        help="show the full manual for this command and exit",
    )
