#!/usr/bin/env python3
"""Aggregate randomized Round 1 Qualtrics response columns.

Qualtrics randomized whether the same two responses appeared as A/B in V1 or
V2. This script normalizes those columns so ratings for the same underlying
response land under the same dialect-labeled output column:

    V1_A == V2_B -> SAE
    V1_B == V2_A -> AAVE

The script intentionally avoids pandas/openpyxl so it can run with the current
project dependencies.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "Round_1_Results" / "LLM & AAVE ROUND 1_April 29, 2026_02.09.xlsx"
DEFAULT_OUTPUT = DEFAULT_INPUT.with_name(DEFAULT_INPUT.stem + "_aggregated.xlsx")

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
NS = {"m": MAIN_NS, "r": REL_NS}

QUESTION_RE = re.compile(
    r"^(P\d+)_V([12])(?:_([AB]))?_(Helpful|Clarity|Warmth|Preference|PreferenceWhy|Approp|AppropWhy)$"
)
SCORE_METRICS = ("Helpful", "Clarity", "Warmth")
DIALECTS = ("SAE", "AAVE")


def column_letters_to_index(letters: str) -> int:
    index = 0
    for char in letters:
        index = index * 26 + ord(char) - ord("A") + 1
    return index


def column_index_to_letters(index: int) -> str:
    letters = ""
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def cell_column_index(cell_ref: str) -> int:
    match = re.match(r"[A-Z]+", cell_ref)
    if not match:
        raise ValueError(f"Unexpected cell reference: {cell_ref}")
    return column_letters_to_index(match.group(0))


def read_shared_strings(archive: ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(item.itertext()) for item in root.findall("m:si", NS)]


def read_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value_node = cell.find("m:v", NS)
    if value_node is None or value_node.text is None:
        inline = cell.find("m:is", NS)
        return "".join(inline.itertext()) if inline is not None else ""

    value = value_node.text
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value)]
    return value


def first_sheet_path(archive: ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets_by_id = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
    first_sheet = workbook.find(".//m:sheet", NS)
    if first_sheet is None:
        raise ValueError("Workbook does not contain any sheets.")

    rel_id = first_sheet.attrib[f"{{{REL_NS}}}id"]
    target = targets_by_id[rel_id]
    return "xl/" + target.lstrip("/")


def read_xlsx_first_sheet(path: Path) -> list[dict[int, str]]:
    with ZipFile(path) as archive:
        shared_strings = read_shared_strings(archive)
        sheet = ET.fromstring(archive.read(first_sheet_path(archive)))

        rows: list[dict[int, str]] = []
        max_row = 0
        row_values_by_number: dict[int, dict[int, str]] = {}
        for row in sheet.findall(".//m:sheetData/m:row", NS):
            row_number = int(row.attrib["r"])
            max_row = max(max_row, row_number)
            row_values_by_number[row_number] = {
                cell_column_index(cell.attrib["r"]): read_cell_value(cell, shared_strings)
                for cell in row.findall("m:c", NS)
            }

        for row_number in range(1, max_row + 1):
            rows.append(row_values_by_number.get(row_number, {}))
        return rows


def dense_row(row: dict[int, str], width: int) -> list[str]:
    return [row.get(index, "") for index in range(1, width + 1)]


def nonempty(*values: str) -> str:
    for value in values:
        if value != "":
            return value
    return ""


def normalize_choice(value: str, version: str) -> str:
    """Convert displayed Response A/B choices to normalized dialect labels."""
    if value == "Response A":
        return "SAE" if version == "1" else "AAVE"
    if value == "Response B":
        return "AAVE" if version == "1" else "SAE"
    return value


def answered_version(row_by_header: dict[str, str], prompt: str) -> str:
    v1_values = [
        value
        for header, value in row_by_header.items()
        if header.startswith(f"{prompt}_V1_") and value != ""
    ]
    v2_values = [
        value
        for header, value in row_by_header.items()
        if header.startswith(f"{prompt}_V2_") and value != ""
    ]
    if v1_values and not v2_values:
        return "1"
    if v2_values and not v1_values:
        return "2"
    if v1_values and v2_values:
        return "both"
    return ""


def normalize_prompt_row(row_by_header: dict[str, str], prompt: str) -> dict[str, str]:
    output: dict[str, str] = {}

    for metric in SCORE_METRICS:
        output[f"{prompt}_SAE_{metric}"] = nonempty(
            row_by_header.get(f"{prompt}_V1_A_{metric}", ""),
            row_by_header.get(f"{prompt}_V2_B_{metric}", ""),
        )
        output[f"{prompt}_AAVE_{metric}"] = nonempty(
            row_by_header.get(f"{prompt}_V1_B_{metric}", ""),
            row_by_header.get(f"{prompt}_V2_A_{metric}", ""),
        )

    version = answered_version(row_by_header, prompt)
    output[f"{prompt}_AnsweredVersion"] = version

    preference_v1 = row_by_header.get(f"{prompt}_V1_Preference", "")
    preference_v2 = row_by_header.get(f"{prompt}_V2_Preference", "")
    if preference_v1:
        output[f"{prompt}_Preference"] = normalize_choice(preference_v1, "1")
    elif preference_v2:
        output[f"{prompt}_Preference"] = normalize_choice(preference_v2, "2")
    else:
        output[f"{prompt}_Preference"] = ""

    approp_v1 = row_by_header.get(f"{prompt}_V1_Approp", "")
    approp_v2 = row_by_header.get(f"{prompt}_V2_Approp", "")
    if approp_v1:
        output[f"{prompt}_Approp"] = normalize_choice(approp_v1, "1")
    elif approp_v2:
        output[f"{prompt}_Approp"] = normalize_choice(approp_v2, "2")
    else:
        output[f"{prompt}_Approp"] = ""

    output[f"{prompt}_PreferenceWhy"] = nonempty(
        row_by_header.get(f"{prompt}_V1_PreferenceWhy", ""),
        row_by_header.get(f"{prompt}_V2_PreferenceWhy", ""),
    )
    output[f"{prompt}_AppropWhy"] = nonempty(
        row_by_header.get(f"{prompt}_V1_AppropWhy", ""),
        row_by_header.get(f"{prompt}_V2_AppropWhy", ""),
    )
    return output


def build_aggregated_rows(raw_rows: list[dict[int, str]]) -> tuple[list[list[str]], list[list[str]]]:
    if len(raw_rows) < 2:
        raise ValueError("Expected at least two Qualtrics header rows.")

    width = max(max(row.keys(), default=0) for row in raw_rows)
    headers = dense_row(raw_rows[0], width)
    question_headers = {header for header in headers if QUESTION_RE.match(header)}
    metadata_headers = [header for header in headers if header and header not in question_headers]
    prompts = sorted({QUESTION_RE.match(header).group(1) for header in question_headers})

    output_headers = metadata_headers[:]
    for prompt in prompts:
        for dialect in DIALECTS:
            for metric in SCORE_METRICS:
                output_headers.append(f"{prompt}_{dialect}_{metric}")
        output_headers.extend(
            [
                f"{prompt}_AnsweredVersion",
                f"{prompt}_Preference",
                f"{prompt}_PreferenceWhy",
                f"{prompt}_Approp",
                f"{prompt}_AppropWhy",
            ]
        )

    output_rows = [output_headers]
    for row in raw_rows[2:]:
        dense = dense_row(row, width)
        row_by_header = {
            header: dense[index]
            for index, header in enumerate(headers)
            if header
        }
        output_row = [row_by_header.get(header, "") for header in metadata_headers]
        normalized_by_header: dict[str, str] = {}
        for prompt in prompts:
            normalized_by_header.update(normalize_prompt_row(row_by_header, prompt))
        output_row.extend(normalized_by_header.get(header, "") for header in output_headers[len(metadata_headers):])
        output_rows.append(output_row)

    mapping_rows = [
        ["Normalized label", "Source labels", "Meaning"],
        ["SAE", "V1_A; V2_B", "SAE response after Qualtrics randomization"],
        ["AAVE", "V1_B; V2_A", "AAVE response after Qualtrics randomization"],
        ["Preference", "V1_Preference; V2_Preference", "Response A/B converted to SAE/AAVE"],
        ["Approp", "V1_Approp; V2_Approp", "Response A/B converted to SAE/AAVE"],
        ["PreferenceWhy", "V1_PreferenceWhy; V2_PreferenceWhy", "Grouped by prompt regardless of version"],
        ["AppropWhy", "V1_AppropWhy; V2_AppropWhy", "Grouped by prompt regardless of version"],
    ]
    return output_rows, mapping_rows


def xml_text(value: str) -> str:
    return html.escape(value, quote=False)


def is_number(value: str) -> bool:
    return bool(re.fullmatch(r"-?(?:\d+\.?\d*|\.\d+)", value))


def worksheet_xml(rows: list[list[str]]) -> str:
    xml_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            if value == "":
                continue
            ref = f"{column_index_to_letters(col_index)}{row_index}"
            if row_index > 1 and is_number(value):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(
                    f'<c r="{ref}" t="inlineStr"><is><t>{xml_text(value)}</t></is></c>'
                )
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    max_width = max((len(row) for row in rows), default=1)
    max_height = max(len(rows), 1)
    dimension = f"A1:{column_index_to_letters(max_width)}{max_height}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<worksheet xmlns="{MAIN_NS}" xmlns:r="{REL_NS}">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        '</worksheet>'
    )


def write_xlsx(path: Path, sheets: list[tuple[str, list[list[str]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<Types xmlns="{CONTENT_TYPES_NS}">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(1, len(sheets) + 1)
            )
            + "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<Relationships xmlns="{PKG_REL_NS}">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<workbook xmlns="{MAIN_NS}" xmlns:r="{REL_NS}"><sheets>'
            + "".join(
                f'<sheet name="{html.escape(name, quote=True)}" sheetId="{i}" r:id="rId{i}"/>'
                for i, (name, _) in enumerate(sheets, start=1)
            )
            + "</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<Relationships xmlns="{PKG_REL_NS}">'
            + "".join(
                f'<Relationship Id="rId{i}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{i}.xml"/>'
                for i in range(1, len(sheets) + 1)
            )
            + "</Relationships>",
        )
        for i, (_, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{i}.xml", worksheet_xml(rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Qualtrics .xlsx export")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Aggregated .xlsx path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows = read_xlsx_first_sheet(args.input)
    aggregated_rows, mapping_rows = build_aggregated_rows(raw_rows)
    write_xlsx(args.output, [("Aggregated", aggregated_rows), ("Mapping", mapping_rows)])
    print(f"Wrote {args.output}")
    print(f"Rows: {len(aggregated_rows) - 1} responses")
    print(f"Columns: {len(aggregated_rows[0])}")


if __name__ == "__main__":
    main()
