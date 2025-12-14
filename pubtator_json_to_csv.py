#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-json", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--write-edge-agg", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    j = json.loads(args.pipeline_json.read_text(encoding="utf-8"))

    rows = []
    seen = set()

    for ind in (j.get("indications") or []):
        drug_name = str(ind.get("drug_name", "") or "")
        for ev in (ind.get("evidence") or []):
            disease_name = str(ev.get("disease_name", "") or "")
            disease_id = str(ev.get("disease_id", "") or "")

            articles = ev.get("articles") or []
            pmids = ev.get("pmids") or []

            if articles:
                for a in articles:
                    pmid = str(a.get("pmid", "") or "")
                    date = a.get("date", None)
                    key = (drug_name, disease_id, pmid, str(date) if date is not None else "")
                    if pmid and key not in seen:
                        seen.add(key)
                        rows.append({
                            "drug_name": drug_name,
                            "disease_name": disease_name,
                            "disease_id": disease_id,
                            "pmid": pmid,
                            "pmid_date": date,
                        })
            else:
                for pmid in pmids:
                    pmid = str(pmid or "")
                    key = (drug_name, disease_id, pmid, "")
                    if pmid and key not in seen:
                        seen.add(key)
                        rows.append({
                            "drug_name": drug_name,
                            "disease_name": disease_name,
                            "disease_id": disease_id,
                            "pmid": pmid,
                            "pmid_date": None,
                        })

    df = pd.DataFrame(rows, columns=["drug_name","disease_name","disease_id","pmid","pmid_date"])
    out_long = args.outdir / "drug_disease_pmid_long.csv"
    df.to_csv(out_long, index=False)

    if args.write_edge_agg and len(df):
        dfa = df.copy()
        dfa["year"] = pd.to_datetime(dfa["pmid_date"], errors="coerce", utc=True).dt.year
        agg = (
            dfa.groupby(["drug_name","disease_id","disease_name"], dropna=False)
               .agg(
                   n_pmids=("pmid", "nunique"),
                   year_min=("year", lambda s: int(s.min()) if s.notna().any() else None),
                   year_max=("year", lambda s: int(s.max()) if s.notna().any() else None),
               )
               .reset_index()
        )
        out_edges = args.outdir / "drug_disease_edges_from_pmids.csv"
        agg.to_csv(out_edges, index=False)

    print(f"Wrote: {out_long}")
    if args.write_edge_agg:
        print(f"Wrote: {args.outdir / 'drug_disease_edges_from_pmids.csv'}")

if __name__ == "__main__":
    main()
