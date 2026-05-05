from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import sqlparse

BASE_DIR = Path(__file__).resolve().parent
DB1_FILE = BASE_DIR / "db1.sql"
DB2_FILE = BASE_DIR / "db2.sql"
REPORT_DIR = BASE_DIR / "reports"
SUMMARY_DIR = BASE_DIR / "summary"
REPORT_FILE = REPORT_DIR / "db_comparison_report.csv"
SUMMARY_FILE = SUMMARY_DIR / "db_comparison_summary.csv"

IDENTIFIER_PATTERN = r"(?:`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[\w$]+)"
QUALIFIED_IDENTIFIER_PATTERN = rf"{IDENTIFIER_PATTERN}(?:\s*\.\s*{IDENTIFIER_PATTERN})?"
TABLE_NAME_PATTERN = re.compile(
    rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>{QUALIFIED_IDENTIFIER_PATTERN})",
    re.I | re.S,
)
CREATE_INDEX_PATTERN = re.compile(
    rf"^CREATE\s+(?P<unique>UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>{IDENTIFIER_PATTERN})\s+ON\s+(?P<table>{QUALIFIED_IDENTIFIER_PATTERN})\s*\((?P<cols>.+)\)\s*$",
    re.I | re.S,
)
ALTER_ADD_INDEX_PATTERN = re.compile(
    rf"^ALTER\s+TABLE\s+(?P<table>{QUALIFIED_IDENTIFIER_PATTERN})\s+ADD\s+(?P<unique>UNIQUE\s+)?(?:INDEX|KEY)\s+(?P<name>{IDENTIFIER_PATTERN})?\s*\((?P<cols>.+)\)\s*$",
    re.I | re.S,
)
ALTER_ADD_PRIMARY_PATTERN = re.compile(
    rf"^ALTER\s+TABLE\s+(?P<table>{QUALIFIED_IDENTIFIER_PATTERN})\s+ADD\s+PRIMARY\s+KEY\s*\((?P<cols>.+)\)\s*$",
    re.I | re.S,
)
COLUMN_BREAK_PATTERN = re.compile(
    r"\b(?:NOT\s+NULL|DEFAULT|AUTO_INCREMENT|PRIMARY\s+KEY|UNIQUE|CHECK|REFERENCES|COLLATE|COMMENT|ON\s+UPDATE|GENERATED|AS|STORED|VIRTUAL)\b",
    re.I,
)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_identifier(identifier: str) -> str:
    cleaned = (identifier or "").strip()
    if cleaned.startswith("`") and cleaned.endswith("`"):
        return cleaned[1:-1]
    if cleaned.startswith('"') and cleaned.endswith('"'):
        return cleaned[1:-1]
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return cleaned[1:-1]
    return cleaned


def normalize_name(name: str) -> str:
    return strip_identifier(name).strip().lower()


def normalize_compare_text(text: str) -> str:
    return collapse_whitespace((text or "").replace("`", "")).lower()


def normalize_datatype(datatype: str) -> str:
    cleaned = collapse_whitespace((datatype or "").replace("`", "")).upper()
    replacements = {
        "CHARACTER VARYING": "VARCHAR",
        "CHAR VARYING": "VARCHAR",
        "DOUBLE PRECISION": "DOUBLE",
        "INTEGER": "INT",
        "NUMERIC": "DECIMAL",
        "BOOL": "BOOLEAN",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    return re.sub(r"\s+", " ", cleaned)


def find_matching_paren(text: str, open_index: int) -> Optional[int]:
    depth = 0
    quote: Optional[str] = None
    i = open_index
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == quote:
                if quote in {"'", '"'} and i + 1 < len(text) and text[i + 1] == quote:
                    i += 2
                    continue
                quote = None
            i += 1
            continue

        if ch in {"'", '"', "`"}:
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def extract_parenthesized_block(text: str, start_index: int = 0) -> Optional[str]:
    open_index = text.find("(", start_index)
    if open_index == -1:
        return None
    close_index = find_matching_paren(text, open_index)
    if close_index is None:
        return None
    return text[open_index + 1 : close_index]


def split_top_level_commas(text: str) -> List[str]:
    parts: List[str] = []
    buffer: List[str] = []
    depth = 0
    quote: Optional[str] = None
    i = 0

    while i < len(text):
        ch = text[i]
        if quote is not None:
            buffer.append(ch)
            if ch == quote:
                if quote in {"'", '"'} and i + 1 < len(text) and text[i + 1] == quote:
                    buffer.append(text[i + 1])
                    i += 2
                    continue
                quote = None
            i += 1
            continue

        if ch in {"'", '"', "`"}:
            quote = ch
            buffer.append(ch)
        elif ch == "(":
            depth += 1
            buffer.append(ch)
        elif ch == ")":
            if depth > 0:
                depth -= 1
            buffer.append(ch)
        elif ch == "," and depth == 0:
            item = collapse_whitespace("".join(buffer))
            if item:
                parts.append(item)
            buffer = []
        else:
            buffer.append(ch)
        i += 1

    tail = collapse_whitespace("".join(buffer))
    if tail:
        parts.append(tail)
    return parts


def strip_named_constraint(text: str) -> str:
    return re.sub(
        rf"^CONSTRAINT\s+{IDENTIFIER_PATTERN}\s+",
        "",
        collapse_whitespace(text),
        flags=re.I,
    )


def extract_clause_value(text: str, clause_name: str) -> Optional[str]:
    pattern = re.compile(
        rf"\b{re.escape(clause_name)}\b\s+(.+?)(?=\s+(?:NOT\s+NULL|PRIMARY\s+KEY|UNIQUE|CHECK|REFERENCES|COLLATE|COMMENT|AUTO_INCREMENT|ON\s+UPDATE|GENERATED|AS|STORED|VIRTUAL)\b|$)",
        re.I | re.S,
    )
    match = pattern.search(text or "")
    if not match:
        return None
    return collapse_whitespace(match.group(1).rstrip(","))


def normalize_column_list(columns_text: str) -> str:
    columns = [normalize_compare_text(column) for column in split_top_level_commas(columns_text)]
    return ", ".join(columns)


def extract_table_name(statement: str) -> Optional[str]:
    match = TABLE_NAME_PATTERN.search(statement or "")
    if not match:
        return None
    raw_name = match.group("name")
    table_name = strip_identifier(raw_name.split(".")[-1])
    return table_name or None


def parse_column_definition(definition: str) -> Optional[Dict[str, Any]]:
    cleaned = collapse_whitespace(definition).rstrip(",")
    match = re.match(rf"^(?P<name>{IDENTIFIER_PATTERN})\s+(?P<rest>.+)$", cleaned, re.I | re.S)
    if not match:
        return None

    name = strip_identifier(match.group("name"))
    rest = collapse_whitespace(match.group("rest"))
    type_match = COLUMN_BREAK_PATTERN.search(rest)
    datatype = rest[: type_match.start()].strip() if type_match else rest.strip()
    clause_part = rest[type_match.start() :].strip() if type_match else ""

    extras: List[str] = []
    for keyword in [
        "AUTO_INCREMENT",
        "PRIMARY KEY",
        "UNIQUE",
        "CHECK",
        "REFERENCES",
        "GENERATED",
        "COLLATE",
        "COMMENT",
        "ON UPDATE",
        "VIRTUAL",
        "STORED",
    ]:
        if re.search(rf"\b{re.escape(keyword)}\b", clause_part, re.I):
            extras.append(keyword)

    return {
        "key": normalize_name(name),
        "name": name,
        "datatype": normalize_datatype(datatype),
        "not_null": bool(re.search(r"\bNOT\s+NULL\b", clause_part, re.I)),
        "default": extract_clause_value(clause_part, "DEFAULT"),
        "extras": tuple(sorted(set(extras))),
        "display": cleaned,
    }


def normalize_index_signature(clause: str) -> str:
    cleaned = collapse_whitespace(clause).rstrip(",")
    upper = cleaned.upper()
    if upper.startswith("CONSTRAINT "):
        cleaned = strip_named_constraint(cleaned)
        upper = cleaned.upper()

    unique = False
    if upper.startswith("UNIQUE "):
        unique = True
        cleaned = cleaned[len("UNIQUE ") :].strip()
        upper = cleaned.upper()

    if upper.startswith("KEY "):
        cleaned = cleaned[len("KEY ") :].strip()
    elif upper.startswith("INDEX "):
        cleaned = cleaned[len("INDEX ") :].strip()
    elif upper.startswith("CREATE INDEX "):
        cleaned = cleaned[len("CREATE INDEX ") :].strip()

    columns_text = extract_parenthesized_block(cleaned) or ""
    return f"unique={unique}|columns:{normalize_column_list(columns_text)}"


def normalize_constraint_signature(clause: str) -> str:
    cleaned = collapse_whitespace(clause).rstrip(",")
    cleaned = strip_named_constraint(cleaned)
    upper = cleaned.upper()

    if upper.startswith("PRIMARY KEY"):
        columns_text = extract_parenthesized_block(cleaned) or ""
        return f"primary|columns:{normalize_column_list(columns_text)}"

    if upper.startswith("UNIQUE"):
        columns_text = extract_parenthesized_block(cleaned) or ""
        return f"unique|columns:{normalize_column_list(columns_text)}"

    if upper.startswith("FOREIGN KEY"):
        columns_text = extract_parenthesized_block(cleaned) or ""
        references_match = re.search(
            rf"\bREFERENCES\b\s+(?P<table>{QUALIFIED_IDENTIFIER_PATTERN})\s*\((?P<cols>.+?)\)",
            cleaned,
            re.I | re.S,
        )
        ref_table = strip_identifier(references_match.group("table").split(".")[-1]) if references_match else ""
        ref_columns = normalize_column_list(references_match.group("cols")) if references_match else ""
        return (
            f"foreign|columns:{normalize_column_list(columns_text)}"
            f"|references:{normalize_name(ref_table)}({ref_columns})"
        )

    if upper.startswith("CHECK"):
        expression = extract_parenthesized_block(cleaned) or cleaned
        return f"check|expr:{normalize_compare_text(expression)}"

    return normalize_compare_text(cleaned)


def parse_standalone_statement(statement: str) -> Optional[Tuple[str, str, str, str]]:
    cleaned = collapse_whitespace(statement).rstrip(";")
    if not cleaned:
        return None

    create_index_match = CREATE_INDEX_PATTERN.match(cleaned)
    if create_index_match:
        table_name = strip_identifier(create_index_match.group("table").split(".")[-1])
        unique = bool(create_index_match.group("unique"))
        signature = f"unique={unique}|columns:{normalize_column_list(create_index_match.group('cols'))}"
        return normalize_name(table_name), "indexes", signature, cleaned

    alter_add_index_match = ALTER_ADD_INDEX_PATTERN.match(cleaned)
    if alter_add_index_match:
        table_name = strip_identifier(alter_add_index_match.group("table").split(".")[-1])
        unique = bool(alter_add_index_match.group("unique"))
        signature = f"unique={unique}|columns:{normalize_column_list(alter_add_index_match.group('cols'))}"
        return normalize_name(table_name), "indexes", signature, cleaned

    alter_add_primary_match = ALTER_ADD_PRIMARY_PATTERN.match(cleaned)
    if alter_add_primary_match:
        table_name = strip_identifier(alter_add_primary_match.group("table").split(".")[-1])
        signature = f"primary|columns:{normalize_column_list(alter_add_primary_match.group('cols'))}"
        return normalize_name(table_name), "constraints", signature, cleaned

    return None


def empty_table_entry(display_name: str) -> Dict[str, Any]:
    return {
        "display_name": display_name,
        "columns": {},
        "constraints": {},
        "indexes": {},
    }


def parse_sql_file(file_path: Path) -> Dict[str, Dict[str, Any]]:
    sql_text = file_path.read_text(encoding="utf-8", errors="ignore")
    statements = [collapse_whitespace(stmt) for stmt in sqlparse.split(sql_text)]
    database: Dict[str, Dict[str, Any]] = {}

    for statement in statements:
        if not statement:
            continue

        if statement.upper().startswith("CREATE TABLE"):
            table_name = extract_table_name(statement)
            if not table_name:
                continue

            table_key = normalize_name(table_name)
            table_entry = database.setdefault(table_key, empty_table_entry(table_name))
            table_entry["display_name"] = table_name

            body = extract_parenthesized_block(statement)
            if not body:
                continue

            for item in split_top_level_commas(body):
                item_clean = collapse_whitespace(item).rstrip(",")
                if not item_clean:
                    continue

                item_upper = item_clean.upper()
                if item_upper.startswith("CONSTRAINT "):
                    remainder = strip_named_constraint(item_clean)
                    remainder_upper = remainder.upper()
                    if remainder_upper.startswith(("UNIQUE KEY", "UNIQUE INDEX", "KEY ", "INDEX ")):
                        signature = normalize_index_signature(remainder)
                        table_entry["indexes"][signature] = remainder
                    elif remainder_upper.startswith(("PRIMARY KEY", "FOREIGN KEY", "CHECK")):
                        signature = normalize_constraint_signature(remainder)
                        table_entry["constraints"][signature] = remainder
                    else:
                        signature = normalize_constraint_signature(remainder)
                        table_entry["constraints"][signature] = remainder
                elif item_upper.startswith(("UNIQUE KEY", "UNIQUE INDEX", "KEY ", "INDEX ")):
                    signature = normalize_index_signature(item_clean)
                    table_entry["indexes"][signature] = item_clean
                elif item_upper.startswith(("PRIMARY KEY", "FOREIGN KEY", "CHECK")):
                    signature = normalize_constraint_signature(item_clean)
                    table_entry["constraints"][signature] = item_clean
                else:
                    column = parse_column_definition(item_clean)
                    if column is not None:
                        table_entry["columns"][column["key"]] = column
                    else:
                        signature = normalize_compare_text(item_clean)
                        table_entry["constraints"][signature] = item_clean
            continue

        parsed = parse_standalone_statement(statement)
        if parsed is None:
            continue

        table_key, bucket_name, signature, display_text = parsed
        table_entry = database.setdefault(table_key, empty_table_entry(table_key))
        table_entry[bucket_name][signature] = display_text

    return database


def compare_columns(
    table_name: str,
    left_columns: Dict[str, Dict[str, Any]],
    right_columns: Dict[str, Dict[str, Any]],
    report_rows: List[Dict[str, Any]],
) -> Tuple[int, int]:
    missing = 0
    changed = 0
    left_keys = set(left_columns)
    right_keys = set(right_columns)

    for column_key in sorted(left_keys - right_keys):
        left_column = left_columns[column_key]
        report_rows.append(
            {
                "Table": table_name,
                "Object Type": "Column",
                "Issue": "Missing in db2",
                "Object": left_column["name"],
                "Details": "Column exists only in DB1.",
                "DB1 Info": left_column["display"],
                "DB2 Info": "",
            }
        )
        missing += 1

    for column_key in sorted(right_keys - left_keys):
        right_column = right_columns[column_key]
        report_rows.append(
            {
                "Table": table_name,
                "Object Type": "Column",
                "Issue": "Missing in db1",
                "Object": right_column["name"],
                "Details": "Column exists only in DB2.",
                "DB1 Info": "",
                "DB2 Info": right_column["display"],
            }
        )
        missing += 1

    for column_key in sorted(left_keys & right_keys):
        left_column = left_columns[column_key]
        right_column = right_columns[column_key]
        changes: List[str] = []

        if left_column["datatype"] != right_column["datatype"]:
            changes.append(f"Type: {left_column['datatype']} -> {right_column['datatype']}")
        if left_column["not_null"] != right_column["not_null"]:
            changes.append(f"NotNull: {left_column['not_null']} -> {right_column['not_null']}")
        if normalize_compare_text(str(left_column["default"])) != normalize_compare_text(str(right_column["default"])):
            changes.append(f"Default: {left_column['default']} -> {right_column['default']}")
        if tuple(left_column["extras"]) != tuple(right_column["extras"]):
            left_extras = ", ".join(left_column["extras"]) or "None"
            right_extras = ", ".join(right_column["extras"]) or "None"
            changes.append(f"Extras: {left_extras} -> {right_extras}")

        if changes:
            report_rows.append(
                {
                    "Table": table_name,
                    "Object Type": "Column",
                    "Issue": "Changed",
                    "Object": left_column["name"],
                    "Details": "; ".join(changes),
                    "DB1 Info": left_column["display"],
                    "DB2 Info": right_column["display"],
                }
            )
            changed += 1

    return missing, changed


def compare_named_objects(
    table_name: str,
    object_type: str,
    left_objects: Dict[str, str],
    right_objects: Dict[str, str],
    report_rows: List[Dict[str, Any]],
) -> int:
    missing = 0
    left_keys = set(left_objects)
    right_keys = set(right_objects)

    for signature in sorted(left_keys - right_keys):
        report_rows.append(
            {
                "Table": table_name,
                "Object Type": object_type,
                "Issue": "Missing in db2",
                "Object": left_objects[signature],
                "Details": f"{object_type} exists only in DB1.",
                "DB1 Info": left_objects[signature],
                "DB2 Info": "",
            }
        )
        missing += 1

    for signature in sorted(right_keys - left_keys):
        report_rows.append(
            {
                "Table": table_name,
                "Object Type": object_type,
                "Issue": "Missing in db1",
                "Object": right_objects[signature],
                "Details": f"{object_type} exists only in DB2.",
                "DB1 Info": "",
                "DB2 Info": right_objects[signature],
            }
        )
        missing += 1

    return missing


def compare_databases(
    left_database: Dict[str, Dict[str, Any]],
    right_database: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    report_rows: List[Dict[str, Any]] = []

    left_tables = set(left_database)
    right_tables = set(right_database)
    all_tables = sorted(left_tables | right_tables)

    missing_tables = 0
    missing_columns = 0
    changed_columns = 0
    missing_indexes = 0
    missing_constraints = 0
    matched_tables = 0

    for table_key in all_tables:
        left_table = left_database.get(table_key)
        right_table = right_database.get(table_key)
        table_name = (left_table or right_table or empty_table_entry(table_key))["display_name"]

        if left_table is None:
            report_rows.append(
                {
                    "Table": table_name,
                    "Object Type": "Table",
                    "Issue": "Missing in db1",
                    "Object": table_name,
                    "Details": "Table exists only in DB2.",
                    "DB1 Info": "",
                    "DB2 Info": table_name,
                }
            )
            missing_tables += 1
            continue

        if right_table is None:
            report_rows.append(
                {
                    "Table": table_name,
                    "Object Type": "Table",
                    "Issue": "Missing in db2",
                    "Object": table_name,
                    "Details": "Table exists only in DB1.",
                    "DB1 Info": table_name,
                    "DB2 Info": "",
                }
            )
            missing_tables += 1
            continue

        matched_tables += 1
        missing, changed = compare_columns(table_name, left_table["columns"], right_table["columns"], report_rows)
        missing_columns += missing
        changed_columns += changed
        missing_indexes += compare_named_objects(
            table_name,
            "Index",
            left_table["indexes"],
            right_table["indexes"],
            report_rows,
        )
        missing_constraints += compare_named_objects(
            table_name,
            "Constraint",
            left_table["constraints"],
            right_table["constraints"],
            report_rows,
        )

    if not report_rows:
        report_rows.append(
            {
                "Table": "-",
                "Object Type": "-",
                "Issue": "No differences found",
                "Object": "-",
                "Details": "Parsed tables, columns, indexes, and constraints match between DB1 and DB2.",
                "DB1 Info": "",
                "DB2 Info": "",
            }
        )

    report_df = pd.DataFrame(
        report_rows,
        columns=["Table", "Object Type", "Issue", "Object", "Details", "DB1 Info", "DB2 Info"],
    )

    summary_rows = [
        ("Tables Parsed in DB1", len(left_tables)),
        ("Tables Parsed in DB2", len(right_tables)),
        ("Tables Compared", matched_tables),
        ("Missing Tables", missing_tables),
        ("Missing Columns", missing_columns),
        ("Changed Columns", changed_columns),
        ("Missing Indexes", missing_indexes),
        ("Missing Constraints", missing_constraints),
        ("Total Differences", missing_tables + missing_columns + changed_columns + missing_indexes + missing_constraints),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Count"])

    return report_df, summary_df


def write_outputs(report_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    report_df.to_csv(REPORT_FILE, index=False, quoting=csv.QUOTE_ALL)
    summary_df.to_csv(SUMMARY_FILE, index=False, quoting=csv.QUOTE_ALL)


def main() -> None:
    if not DB1_FILE.exists():
        raise FileNotFoundError(f"Missing schema file: {DB1_FILE}")
    if not DB2_FILE.exists():
        raise FileNotFoundError(f"Missing schema file: {DB2_FILE}")

    left_database = parse_sql_file(DB1_FILE)
    right_database = parse_sql_file(DB2_FILE)
    report_df, summary_df = compare_databases(left_database, right_database)
    write_outputs(report_df, summary_df)

    print(f"Summary CSV generated: {SUMMARY_FILE}")
    print(f"Detailed report CSV generated: {REPORT_FILE}")


if __name__ == "__main__":
    main()
