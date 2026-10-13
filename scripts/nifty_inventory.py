"""Audit every raw Nifty CSV/TXT without changing the source files.

CSV and TXT representations are paired only by archive year/month/file stem.
A SHA256 match across different canonical IDs does NOT identify one contract.
The archive month is provenance, not a verified expiry date.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

MONTHS = {
    name.lower(): number
    for number, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"),
        start=1,
    )
}
FIELDS = [
    "relative_path", "archive_year", "archive_month", "archive_month_dir",
    "suffix", "filename", "stem", "canonical_id", "size_bytes", "sha256",
    "nonempty_lines", "field_counts_json", "first_date", "first_time",
    "last_date", "last_time", "symbol_styles_json", "symbols_json",
    "group_file_count", "group_csv_count", "group_txt_count", "group_hash_count",
    "pair_status", "selection_needs_review", "selected",
]


def json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def symbol_style(symbol: str) -> str:
    if re.fullmatch(r"(?:CE|PE)\s+\d+(?:\.\d+)?", symbol):
        return "option_type_space_strike"
    if re.fullmatch(r"NIFTY\d+(?:\.\d+)?(?:CE|PE)", symbol):
        return "nifty_strike_option_type"
    if re.fullmatch(r"NIFTY\d{2}[A-Z]{3}\d{2}\d+(?:\.\d+)?(?:CE|PE)", symbol, flags=re.IGNORECASE):
        return "nifty_date_token_strike_option_type"
    return "other"


def inspect_file(path: Path, raw_root: Path) -> dict:
    relative = path.relative_to(raw_root)
    year_match = re.search(r"\b(20\d{2})\b", relative.parts[0])
    if year_match is None or len(relative.parts) < 3:
        raise ValueError(f"Cannot determine archive year/month for {relative}")
    year = int(year_match.group(1))
    month_dir = relative.parts[1]
    month = MONTHS.get(month_dir.split()[0].lower())
    if month is None:
        raise ValueError(f"Unknown archive month for {relative}")
    payload = path.read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    widths: Counter = Counter()
    symbols: Counter = Counter()
    first = last = None
    nonempty = 0
    for line in payload.splitlines():
        if not line.strip():
            continue
        nonempty += 1
        widths[str(line.count(b",") + 1)] += 1
        symbol = line.partition(b",")[0].decode("utf-8", errors="replace").strip()
        symbols[symbol] += 1
        if first is None:
            first = line
        last = line
    styles: Counter = Counter()
    for symbol, count in symbols.items():
        styles[symbol_style(symbol)] += count

    def row_datetime(line: bytes | None) -> tuple[str, str]:
        if line is None:
            return "", ""
        fields = line.decode("utf-8", errors="replace").split(",", 3)
        return (fields[1].strip() if len(fields) > 1 else "", fields[2].strip() if len(fields) > 2 else "")

    first_date, first_time = row_datetime(first)
    last_date, last_time = row_datetime(last)
    return {
        "relative_path": relative.as_posix(),
        "archive_year": year,
        "archive_month": month,
        "archive_month_dir": month_dir,
        "suffix": path.suffix.lower(),
        "filename": path.name,
        "stem": path.stem,
        "canonical_id": f"{year}/{month:02d}/{path.stem}",
        "size_bytes": len(payload),
        "sha256": sha256,
        "nonempty_lines": nonempty,
        "field_counts_json": json_text(widths),
        "first_date": first_date,
        "first_time": first_time,
        "last_date": last_date,
        "last_time": last_time,
        "symbol_styles_json": json_text(styles),
        "symbols_json": json_text(symbols),
    }


def assign_selection(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["canonical_id"]].append(record)
    for group in groups.values():
        csv_count = sum(record["suffix"] == ".csv" for record in group)
        txt_count = sum(record["suffix"] == ".txt" for record in group)
        hash_count = len({record["sha256"] for record in group})
        if csv_count and txt_count:
            status = "csv_txt_identical" if hash_count == 1 else "csv_txt_different"
        else:
            status = "csv_only" if csv_count else "txt_only"
        # Exclude only byte-identical representations within one canonical ID.
        # Conflicting representations remain selected for downstream review.
        by_hash: dict[str, list[dict]] = defaultdict(list)
        for record in group:
            by_hash[record["sha256"]].append(record)
        chosen_paths = {
            min(same_hash, key=lambda record: (record["suffix"] != ".csv", record["relative_path"]))["relative_path"]
            for same_hash in by_hash.values()
        }
        needs_review = csv_count > 1 or txt_count > 1 or hash_count > 1
        for record in group:
            record.update({
                "group_file_count": len(group),
                "group_csv_count": csv_count,
                "group_txt_count": txt_count,
                "group_hash_count": hash_count,
                "pair_status": status,
                "selection_needs_review": needs_review,
                "selected": record["relative_path"] in chosen_paths,
            })
    return groups


def make_summary(records: list[dict], groups: dict[str, list[dict]], raw_root: Path) -> dict:
    selected = [record for record in records if record["selected"]]
    all_hashes: dict[str, set[str]] = defaultdict(set)
    for record in selected:
        all_hashes[record["sha256"]].add(record["canonical_id"])
    cross_id_hashes = {key: sorted(ids) for key, ids in all_hashes.items() if len(ids) > 1}
    monthly: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in records:
        monthly[(record["archive_year"], record["archive_month"])].append(record)
    month_summaries = []
    for (year, month), month_records in sorted(monthly.items()):
        month_selected = [record for record in month_records if record["selected"]]
        width_counts: Counter = Counter()
        for record in month_selected:
            width_counts.update(json.loads(record["field_counts_json"]))
        month_summaries.append({
            "archive_year": year,
            "archive_month": month,
            "physical_files": len(month_records),
            "selected_files": len(month_selected),
            "physical_bytes": sum(record["size_bytes"] for record in month_records),
            "selected_bytes": sum(record["size_bytes"] for record in month_selected),
            "physical_nonempty_rows": sum(record["nonempty_lines"] for record in month_records),
            "selected_nonempty_rows": sum(record["nonempty_lines"] for record in month_selected),
            "selected_row_field_counts": dict(width_counts),
            "pair_status_groups": dict(Counter(record["pair_status"] for record in {record["canonical_id"]: record for record in month_records}.values())),
        })
    years = sorted({record["archive_year"] for record in records})
    width_counts: Counter = Counter()
    style_counts: Counter = Counter()
    for record in selected:
        width_counts.update(json.loads(record["field_counts_json"]))
        style_counts.update(json.loads(record["symbol_styles_json"]))
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "raw_root": str(raw_root),
        "method": {
            "source_mutations": "none",
            "pair_key": "archive_year/archive_month/file_stem (case sensitive)",
            "selection": "One file per unique hash within a pair key, CSV preferred, TXT fallback; conflicting bytes are retained and flagged for review.",
            "hash": "SHA256 of exact raw bytes, used only to compare representations of a pair key.",
            "row_count": "Nonempty physical lines; field count is commas + 1 for observed unquoted headerless records.",
            "archive_month_caveat": "Archive month is provenance and has not been asserted to be an expiry date.",
            "ninth_column_caveat": "The ninth field has no supplied vendor schema; its meaning is unverified.",
            "cross_id_hash_caveat": "Equal content across different canonical IDs is retained; do not deduplicate contracts by hash.",
        },
        "physical_files": len(records),
        "physical_bytes": sum(record["size_bytes"] for record in records),
        "physical_nonempty_rows": sum(record["nonempty_lines"] for record in records),
        "physical_suffix_counts": dict(Counter(record["suffix"] for record in records)),
        "selected_files": len(selected),
        "canonical_groups": len(groups),
        "selected_bytes": sum(record["size_bytes"] for record in selected),
        "selected_nonempty_rows": sum(record["nonempty_lines"] for record in selected),
        "excluded_identical_representation_files": sum(not record["selected"] for record in records),
        "excluded_identical_representation_rows": sum(record["nonempty_lines"] for record in records if not record["selected"]),
        "excluded_identical_representation_bytes": sum(record["size_bytes"] for record in records if not record["selected"]),
        "exclusion_note": "Excluded from the analysis selection only; original files remain untouched.",
        "selected_suffix_counts": dict(Counter(record["suffix"] for record in selected)),
        "selected_row_field_counts": dict(width_counts),
        "selected_symbol_style_row_counts": dict(style_counts),
        "pair_status_group_counts": dict(Counter(group[0]["pair_status"] for group in groups.values())),
        "selection_needs_review_groups": [key for key, group in groups.items() if group[0]["selection_needs_review"]],
        "same_format_multiple_file_groups": [key for key, group in groups.items() if group[0]["group_csv_count"] > 1 or group[0]["group_txt_count"] > 1],
        "zero_nonempty_line_files": [record["relative_path"] for record in records if not record["nonempty_lines"]],
        "multi_symbol_files": [record["relative_path"] for record in records if len(json.loads(record["symbols_json"])) > 1],
        "mixed_field_width_files": [record["relative_path"] for record in records if len(json.loads(record["field_counts_json"])) > 1],
        "cross_canonical_identical_hash_groups": len(cross_id_hashes),
        "cross_canonical_identical_hash_examples": dict(list(cross_id_hashes.items())[:20]),
        "missing_archive_months_inside_observed_years": [f"{year}-{month:02d}" for year in years for month in range(1, 13) if (year, month) not in monthly],
        "monthly": month_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("Nifty Options Data"))
    parser.add_argument("--inventory", type=Path, default=Path("data/processed/nifty_eda/file_inventory.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/nifty_eda/inventory_summary.json"))
    args = parser.parse_args()
    files = sorted(path for path in args.raw_root.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".txt"})
    if not files:
        raise SystemExit(f"No CSV/TXT input found under {args.raw_root}")
    records = []
    for number, path in enumerate(files, start=1):
        records.append(inspect_file(path, args.raw_root))
        if number % 500 == 0 or number == len(files):
            print(f"Audited {number:,}/{len(files):,} files", flush=True)
    groups = assign_selection(records)
    summary = make_summary(records, groups, args.raw_root)
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    with args.inventory.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key in {"physical_files", "selected_files", "selected_nonempty_rows", "pair_status_group_counts", "selected_row_field_counts", "selection_needs_review_groups", "missing_archive_months_inside_observed_years"}}, indent=2))


if __name__ == "__main__":
    main()
