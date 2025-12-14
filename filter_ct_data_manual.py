#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

ORIG_COLS = ["drug", "nct_id", "title", "status", "phase", "disease", "first_submit"]


def norm_basic(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def split_diseases(s: str, split_char: str) -> list[str]:
    if s is None:
        return []
    parts = [p.strip() for p in str(s).split(split_char)]
    return [p for p in parts if p]


def join_diseases(parts: list[str], split_char: str) -> str:
    return (split_char + " ").join(parts)


def load_blocklist(use_default: bool, blocklist_path: Path | None) -> set[str]:
    block: set[str] = set()

    if use_default:
        # Only entries that are not diseases (study population / endpoints / PK/BE concepts / interventions).
        block |= {
            # populations
            "healthy",
            "healthy volunteer",
            "healthy volunteers",
            "healthy subject",
            "healthy subjects",
            "healthy participant",
            "healthy participants",
            "healthy adult",
            "healthy adults",
            "healthy control",
            "healthy controls",

            # endpoints / study concepts
            "quality of life",
            "pharmacokinetics",
            "bioequivalence",

            # interventions / procedures (not diseases)
            "contraception",
            "smoking cessation",
            "anesthesia",
        }

    if blocklist_path is not None and blocklist_path.exists():
        for line in blocklist_path.read_text(encoding="utf-8").splitlines():
            k = norm_basic(line)
            if k:
                block.add(k)

    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-in", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("ct_manual_filter"))
    ap.add_argument("--split", type=str, default=";")
    ap.add_argument("--use-default-blocklist", action="store_true")
    ap.add_argument("--blocklist", type=Path, default=None)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    out_kept = args.outdir / "ctgov_manual_filtered.csv"
    out_dropped = args.outdir / "ctgov_manual_dropped_rows.csv"
    out_removed_long = args.outdir / "ctgov_manual_removed_terms_long.csv"
    out_summary = args.outdir / "ctgov_manual_filter_summary.json"

    df = pd.read_csv(args.ct_in)

    miss = set(ORIG_COLS) - set(df.columns)
    if miss:
        raise SystemExit(f"CT file missing columns: {sorted(miss)}")

    block = load_blocklist(args.use_default_blocklist, args.blocklist)

    removed_events = []
    kept_rows = []
    dropped_rows = []

    n_rows_in = len(df)
    n_rows_dropped = 0
    n_terms_removed = 0

    for i, r in df.iterrows():
        original = r["disease"]
        parts = split_diseases(original, args.split)

        kept = []
        removed = []
        for p in parts:
            if norm_basic(p) in block:
                removed.append(p)
            else:
                kept.append(p)

        if removed:
            for rm in removed:
                removed_events.append(
                    {
                        "row_index": int(i),
                        "nct_id": r.get("nct_id", ""),
                        "drug": r.get("drug", ""),
                        "removed_term": rm,
                        "original_disease_field": "" if pd.isna(original) else str(original),
                    }
                )
            n_terms_removed += len(removed)

        if len(kept) == 0:
            dropped_rows.append(r[ORIG_COLS].to_dict())
            n_rows_dropped += 1
            continue

        rr = r[ORIG_COLS].copy()
        rr["disease"] = join_diseases(kept, args.split)
        kept_rows.append(rr.to_dict())

    kept_df = pd.DataFrame(kept_rows, columns=ORIG_COLS)
    dropped_df = pd.DataFrame(dropped_rows, columns=ORIG_COLS)
    removed_df = pd.DataFrame(removed_events)

    kept_df.to_csv(out_kept, index=False, quoting=csv.QUOTE_MINIMAL)
    dropped_df.to_csv(out_dropped, index=False, quoting=csv.QUOTE_MINIMAL)
    removed_df.to_csv(out_removed_long, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "inputs": {
            "ct_in": str(args.ct_in),
            "split": args.split,
            "use_default_blocklist": bool(args.use_default_blocklist),
            "blocklist_path": str(args.blocklist) if args.blocklist else None,
        },
        "counts": {
            "rows_in": int(n_rows_in),
            "rows_out_kept": int(len(kept_df)),
            "rows_out_dropped_all_terms_removed": int(n_rows_dropped),
            "total_terms_removed": int(n_terms_removed),
            "unique_removed_terms": sorted({norm_basic(x) for x in removed_df["removed_term"].tolist()})
            if len(removed_df)
            else [],
        },
        "outputs": {
            "manual_filtered": str(out_kept),
            "dropped_rows": str(out_dropped),
            "removed_terms_long": str(out_removed_long),
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("CT manual filter summary")
    print(f"  input rows:   {n_rows_in}")
    print(f"  kept rows:    {len(kept_df)}  -> {out_kept}")
    print(f"  dropped rows: {n_rows_dropped} (all terms removed) -> {out_dropped}")
    print(f"  terms removed: {n_terms_removed} -> {out_removed_long}")
    print(f"  summary:      {out_summary}")


if __name__ == "__main__":
    main()
