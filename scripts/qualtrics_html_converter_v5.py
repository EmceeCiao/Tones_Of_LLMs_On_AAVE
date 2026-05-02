#!/usr/bin/env python3
"""Export presentation-friendly Qualtrics HTML with nested list handling."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Iterable


CODE_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
BLOCK_MATH_RE = re.compile(r"\\\[\s*(.*?)\s*\\\]", re.S)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)
ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.S)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
ORDERED_RE = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
UNORDERED_RE = re.compile(r"^(\s*)[-*]\s+(.*)$")
TABLE_DELIM_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
BLOCKQUOTE_RE = re.compile(r"^\s*>\s?(.*)$")


def escape_text(text: str) -> str:
    return html.escape(text, quote=False)


def apply_inline_markup(text: str) -> str:
    placeholders: dict[str, str] = {}

    def stash_code(match: re.Match[str]) -> str:
        key = f"__INLINE_CODE_{len(placeholders)}__"
        code_text = escape_text(match.group(1)).replace("&amp;", "&")
        placeholders[key] = (
            '<code style="font-family: Menlo, Consolas, monospace;">'
            f"{code_text}</code>"
        )
        return key

    text = INLINE_CODE_RE.sub(stash_code, text)
    escaped = escape_text(text)
    escaped = BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    escaped = ITALIC_RE.sub(r"<em>\1</em>", escaped)
    for key, value in placeholders.items():
        escaped = escaped.replace(key, value)
    return escaped


def field_label(field: str, index: int) -> str:
    if field == "prompt":
        return f"Prompt {index}"
    if field == "response_A":
        return "Response A"
    if field == "response_B":
        return "Response B"
    return field.replace("_", " ")


def split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def render_table(lines: list[str]) -> str:
    header = split_table_row(lines[0])
    body = [split_table_row(line) for line in lines[2:]]
    head_cells = "".join(
        f'<th style="border:1px solid #d0d7de; padding:6px 8px; text-align:left;">{apply_inline_markup(cell)}</th>'
        for cell in header
    )
    body_rows = []
    for row in body:
        cells = "".join(
            f'<td style="border:1px solid #d0d7de; padding:6px 8px; vertical-align:top;">{apply_inline_markup(cell)}</td>'
            for cell in row
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        '<table style="border-collapse:collapse; width:100%; margin:8px 0 12px 0;">'
        f"<thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def is_special_block_start(line: str) -> bool:
    stripped = line.strip()
    return bool(
        HEADING_RE.match(line)
        or TABLE_DELIM_RE.match(line)
        or BLOCKQUOTE_RE.match(line)
        or stripped == "<hr>"
        or stripped.startswith("__HTML_BLOCK_")
    )


def list_match(line: str):
    ordered = ORDERED_RE.match(line)
    if ordered:
        return "ol", ordered
    unordered = UNORDERED_RE.match(line)
    if unordered:
        return "ul", unordered
    return None, None


def collect_list(lines: list[str], start: int) -> tuple[str, int]:
    list_type, first_match = list_match(lines[start])
    if not first_match:
        raise ValueError("collect_list called on non-list line")

    base_indent = len(first_match.group(1))
    start_num = int(first_match.group(2)) if list_type == "ol" else None
    items: list[str] = []
    i = start

    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break

        current_type, match = list_match(lines[i])
        if not match or current_type != list_type or len(match.group(1)) != base_indent:
            break

        if list_type == "ol":
            content = match.group(3)
        else:
            content = match.group(2)

        item_parts = [apply_inline_markup(content)]
        i += 1

        while i < len(lines):
            if not lines[i].strip():
                lookahead = i + 1
                while lookahead < len(lines) and not lines[lookahead].strip():
                    lookahead += 1
                if lookahead >= len(lines):
                    i = lookahead
                    break
                next_type, next_match = list_match(lines[lookahead])
                if next_match and len(next_match.group(1)) == base_indent and next_type == list_type:
                    i = lookahead
                    break
                if next_match and len(next_match.group(1)) > base_indent:
                    i = lookahead
                    continue
                break

            next_type, next_match = list_match(lines[i])
            next_indent = indent_width(lines[i])

            if next_match and next_indent == base_indent and next_type == list_type:
                break

            if next_match and next_indent > base_indent:
                nested_html, i = collect_list(lines, i)
                item_parts.append(nested_html)
                continue

            if next_indent > base_indent and not is_special_block_start(lines[i]):
                item_parts.append(
                    f'<div style="margin:4px 0 0 0;">{apply_inline_markup(lines[i].strip())}</div>'
                )
                i += 1
                continue

            break

        items.append(f'<li style="margin:0 0 4px 0;">{"".join(item_parts)}</li>')

    style = 'style="margin:6px 0 12px 24px; padding:0;"'
    start_attr = f' start="{start_num}"' if list_type == "ol" and start_num and start_num != 1 else ""
    return f"<{list_type}{start_attr} {style}>{''.join(items)}</{list_type}>", i


def collect_paragraph(lines: list[str], start: int) -> tuple[str, int]:
    parts = []
    i = start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            break
        if (
            HEADING_RE.match(line)
            or ORDERED_RE.match(line)
            or UNORDERED_RE.match(line)
            or BLOCKQUOTE_RE.match(line)
            or TABLE_DELIM_RE.match(line)
            or line.strip() == "<hr>"
            or line.startswith("__HTML_BLOCK_")
        ):
            break
        parts.append(apply_inline_markup(line.rstrip()))
        i += 1
    return f'<p style="margin:0 0 12px 0;">{"<br>".join(parts)}</p>', i


def collect_blockquote(lines: list[str], start: int) -> tuple[str, int]:
    parts = []
    i = start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            break
        match = BLOCKQUOTE_RE.match(line)
        if not match:
            break
        parts.append(apply_inline_markup(match.group(1).rstrip()))
        i += 1
    return (
        '<blockquote style="margin:8px 0 12px 0; padding:0 0 0 12px; '
        'border-left:3px solid #d0d7de;">'
        f'<p style="margin:0;">{"<br>".join(parts)}</p>'
        "</blockquote>",
        i,
    )


def convert_blocks(text: str) -> str:
    placeholders: dict[str, str] = {}

    def stash(content: str) -> str:
        key = f"__HTML_BLOCK_{len(placeholders)}__"
        placeholders[key] = content
        return key

    def code_sub(match: re.Match[str]) -> str:
        language = match.group(1).strip()
        code = escape_text(match.group(2).strip("\n"))
        class_attr = f' class="language-{escape_text(language)}"' if language else ""
        block = (
            '<pre style="margin:8px 0 12px 0; padding:12px; background:#f6f8fa; '
            'border:1px solid #d0d7de; border-radius:6px; white-space:pre-wrap; '
            'overflow-wrap:anywhere; word-break:break-word;">'
            f"<code{class_attr}>{code}</code></pre>"
        )
        return stash(block)

    def math_sub(match: re.Match[str]) -> str:
        expr = escape_text(match.group(1).strip())
        block = (
            '<pre style="margin:8px 0 12px 0; padding:12px; background:#f6f8fa; '
            'border:1px solid #d0d7de; border-radius:6px; white-space:pre-wrap; '
            'overflow-wrap:anywhere; word-break:break-word;">'
            f"<code>{expr}</code></pre>"
        )
        return stash(block)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CODE_FENCE_RE.sub(code_sub, text)
    text = BLOCK_MATH_RE.sub(math_sub, text)
    text = re.sub(r"^\s*---+\s*$", "<hr>", text, flags=re.M)

    lines = text.split("\n")
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped in placeholders:
            blocks.append(placeholders[stripped])
            i += 1
            continue

        heading = HEADING_RE.match(line)
        if heading:
            level = min(len(heading.group(1)), 4)
            blocks.append(
                f'<h{level} style="margin:12px 0 8px 0;">{apply_inline_markup(heading.group(2).strip())}</h{level}>'
            )
            i += 1
            continue

        if stripped == "<hr>":
            blocks.append('<hr style="margin:12px 0; border:none; border-top:1px solid #d0d7de;">')
            i += 1
            continue

        if BLOCKQUOTE_RE.match(line):
            block, i = collect_blockquote(lines, i)
            blocks.append(block)
            continue

        if i + 1 < len(lines) and "|" in line and TABLE_DELIM_RE.match(lines[i + 1]):
            table_lines = [line, lines[i + 1]]
            j = i + 2
            while j < len(lines) and "|" in lines[j].strip():
                table_lines.append(lines[j])
                j += 1
            blocks.append(render_table(table_lines))
            i = j
            continue

        if ORDERED_RE.match(line) or UNORDERED_RE.match(line):
            block, i = collect_list(lines, i)
            blocks.append(block)
            continue

        para, i = collect_paragraph(lines, i)
        blocks.append(para)

    rendered = "".join(blocks)
    for key, value in placeholders.items():
        rendered = rendered.replace(key, value)
    return (
        '<div style="width:100%; max-width:100%; overflow-wrap:anywhere; word-break:break-word;">'
        f"{rendered}"
        "</div>"
    )


def transform_records(records: Iterable[dict], fields: list[str]) -> list[dict]:
    converted = []
    for row in records:
        new_row = dict(row)
        for field in fields:
            if field in row and isinstance(row[field], str):
                new_row[field] = convert_blocks(row[field])
        converted.append(new_row)
    return converted


def build_preview(records: list[dict], fields: list[str]) -> str:
    cards = []
    for idx, row in enumerate(records, start=1):
        sections = [
            f"<p><strong>Category:</strong> {escape_text(str(row.get('category', '')))}</p>",
            f"<p><strong>Prompt {idx}:</strong></p>",
            row.get("prompt", ""),
        ]
        for field in fields:
            if field.startswith("response_"):
                sections.append(f"<p><strong>{escape_text(field_label(field, idx))}:</strong></p>")
                sections.append(row.get(field, ""))
        cards.append(f"<section>{''.join(sections)}</section>")
    css = """
    <style>
    body { font-family: Georgia, serif; margin: 32px auto; max-width: 980px; line-height: 1.5; color: #1f2933; }
    section { border: 1px solid #d9e2ec; border-radius: 10px; padding: 20px; margin-bottom: 24px; background: #fff; }
    </style>
    """
    return f"<!doctype html><html><head><meta charset='utf-8'>{css}</head><body>{''.join(cards)}</body></html>"


def export_snippets(records: list[dict], fields: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_parts = ["<!doctype html><html><head><meta charset='utf-8'></head><body>"]
    for idx, row in enumerate(records, start=1):
        combined_parts = [f"<p><strong>Prompt {idx}:</strong></p>", str(row.get("prompt", ""))]
        for field in fields:
            if field not in row:
                continue
            label = field_label(field, idx)
            snippet = f"<p><strong>{escape_text(label)}:</strong></p>{row[field]}"
            file_path = output_dir / f"item_{idx:02d}_{field}.html"
            file_path.write_text(snippet, encoding="utf-8")
            if field != "prompt":
                combined_parts.append(f"<p><strong>{escape_text(label)}:</strong></p>")
                combined_parts.append(str(row[field]))
        combined_path = output_dir / f"item_{idx:02d}_combined.html"
        combined_path.write_text("".join(combined_parts), encoding="utf-8")
        index_parts.append(f"<p><strong>Prompt {idx}</strong>: <code>{escape_text(combined_path.name)}</code></p>")
    index_parts.append("</body></html>")
    (output_dir / "index.html").write_text("".join(index_parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["prompt", "response_A", "response_B"],
        help="Fields to convert to HTML in-place.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--preview-html", type=Path)
    parser.add_argument("--snippet-dir", type=Path)
    args = parser.parse_args()

    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Expected a top-level JSON list.")

    converted = transform_records(data, args.fields)
    output_json = args.output_json or args.input_json.with_name(f"{args.input_json.stem}_qualtrics_ready_v4.json")
    output_json.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.preview_html:
        args.preview_html.write_text(build_preview(converted, args.fields), encoding="utf-8")
    if args.snippet_dir:
        export_snippets(converted, args.fields, args.snippet_dir)

    print(f"Wrote {output_json}")
    if args.preview_html:
        print(f"Wrote {args.preview_html}")
    if args.snippet_dir:
        print(f"Wrote snippets to {args.snippet_dir}")


if __name__ == "__main__":
    main()
