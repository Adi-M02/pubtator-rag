#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

from pubtator_api import pubtator_resolve_best_entity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-path", type=Path, required=True)
    ap.add_argument("--split", type=str, default=";")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--norm-mode", choices=["basic", "basic_singular"], required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/ct_resolution_preview"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/ct_resolution_preview/cache.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.ct_path)
    for c in ["drug", "nct_id", "disease", "first_submit"]:
        if c not in df.columns:
            raise SystemExit(f"Missing required column in CT file: {c}")

    rows = []
    for _, r in df.iterrows():
        parts = [p.strip() for p in str(r["disease"]).split(args.split)]
        parts = [p for p in parts if p]
        if not parts:
            parts = [""]
        for d in parts:
            rows.append({"drug": r["drug"], "nct_id": r["nct_id"], "disease": d, "first_submit": r["first_submit"]})
    ct = pd.DataFrame(rows)

    freq = (
        ct.groupby("disease")
        .agg(n_trials=("nct_id", "nunique"), n_rows=("disease", "size"))
        .reset_index()
        .sort_values(["n_trials", "n_rows"], ascending=False)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    cache = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text(encoding="utf-8"))

    preview = freq.head(args.top).copy()
    out_rows = []
    for d in preview["disease"].astype(str).tolist():
        if d in cache:
            res = cache[d]
        else:
            res = pubtator_resolve_best_entity(d, concept="DISEASE", limit=10, norm_mode=args.norm_mode)
            res = {
                "best_id": res["best_id"],
                "best_label": res["best_label"],
                "status": res["status"],
                "candidates": res["candidates"][:5],
                "total_count": res["total_count"],
            }
            cache[d] = res

        out_rows.append(
            {
                "ct_disease": d,
                "n_trials": int(preview.loc[preview["disease"] == d, "n_trials"].iloc[0]),
                "n_rows": int(preview.loc[preview["disease"] == d, "n_rows"].iloc[0]),
                "resolved_label": res["best_label"],
                "resolved_id": res["best_id"],
                "resolve_status": res["status"],
                "top_candidates": json.dumps(res["candidates"]),
                "total_candidates_available": int(res["total_count"]),
            }
        )

    args.cache.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    out = pd.DataFrame(out_rows).sort_values(["n_trials", "n_rows"], ascending=False)
    out_path = args.out_dir / "ct_disease_resolution_preview_top.csv"
    out.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Cache: {args.cache}")

if __name__ == "__main__":
    main()
