#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_pubtator(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    need = {"drug_name", "disease_name", "pmid"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"PubTator CSV missing columns: {sorted(missing)}")

    df = df.copy()
    for c in ["drug_name", "disease_name", "pmid"]:
        df[c] = df[c].astype(str).str.strip()

    df = df[(df["pmid"] != "") & (df["drug_name"] != "") & (df["disease_name"] != "")]
    return df[["pmid", "drug_name", "disease_name"]]


def load_rentrez(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    need = {"drug", "disease", "pmid"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Rentrez CSV missing columns: {sorted(missing)}")

    df = df.copy()
    for c in ["drug", "disease", "pmid"]:
        df[c] = df[c].astype(str).str.strip()

    # This automatically removes rows like: "Lisdexamfetamine",NA,NA,FALSE,NA,NA
    df = df[(df["pmid"] != "") & (df["drug"] != "") & (df["disease"] != "")]
    return df[["pmid", "drug", "disease"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pubtator", default="pubtator_outputs/drug_disease_pmid_long.csv")
    ap.add_argument("--rentrez", default="rentrez_data/pubmed_results.csv")
    ap.add_argument("--out-dir", default=None, help="Default: eval_results/ under this script directory")
    ap.add_argument("--top-k", type=int, default=50, help="Top K shared PMIDs by unique claim count")
    args = ap.parse_args()

    app_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else (app_dir / "eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    pub_path = Path(args.pubtator)
    ren_path = Path(args.rentrez)
    if not pub_path.exists():
        raise FileNotFoundError(f"PubTator file not found: {pub_path}")
    if not ren_path.exists():
        raise FileNotFoundError(f"Rentrez file not found: {ren_path}")

    pub = load_pubtator(pub_path)
    ren = load_rentrez(ren_path)

    pub_pmids = set(pub["pmid"].unique())
    ren_pmids = set(ren["pmid"].unique())
    shared_pmids = sorted(pub_pmids & ren_pmids, key=lambda x: int(x) if str(x).isdigit() else str(x))

    # Raw row counts per PMID
    pub_rows = pub.groupby("pmid").size().rename("pub_rows").to_frame()
    ren_rows = ren.groupby("pmid").size().rename("rentrez_rows").to_frame()

    # Unique (drug, disease) counts per PMID
    pub_u = pub.drop_duplicates(subset=["pmid", "drug_name", "disease_name"])
    ren_u = ren.drop_duplicates(subset=["pmid", "drug", "disease"])
    pub_unique = pub_u.groupby("pmid").size().rename("pub_unique_claims").to_frame()
    ren_unique = ren_u.groupby("pmid").size().rename("rentrez_unique_claims").to_frame()

    shared = pd.DataFrame({"pmid": shared_pmids}).set_index("pmid")
    shared = (
        shared.join(pub_rows, how="left")
        .join(ren_rows, how="left")
        .join(pub_unique, how="left")
        .join(ren_unique, how="left")
        .fillna(0)
        .astype(int)
    )

    # Multi-claim flags: two versions
    shared["pub_multi_unique"] = shared["pub_unique_claims"] > 1
    shared["rentrez_multi_unique"] = shared["rentrez_unique_claims"] > 1
    shared["either_multi_unique"] = shared["pub_multi_unique"] | shared["rentrez_multi_unique"]

    shared["pub_multi_rows"] = shared["pub_rows"] > 1
    shared["rentrez_multi_rows"] = shared["rentrez_rows"] > 1
    shared["either_multi_rows"] = shared["pub_multi_rows"] | shared["rentrez_multi_rows"]

    shared["total_unique_claims_both_sources"] = shared["pub_unique_claims"] + shared["rentrez_unique_claims"]
    shared["total_rows_both_sources"] = shared["pub_rows"] + shared["rentrez_rows"]

    # Distribution table for quick understanding
    dist = (
        shared.reset_index()
        .groupby(["pub_unique_claims", "rentrez_unique_claims"])
        .size()
        .rename("n_shared_pmids")
        .reset_index()
        .sort_values(["n_shared_pmids", "pub_unique_claims", "rentrez_unique_claims"], ascending=[False, True, True])
    )

    summary = {
        "pub_rows_loaded_after_drop_empty": int(len(pub)),
        "rentrez_rows_loaded_after_drop_empty": int(len(ren)),
        "pub_unique_pmids": int(len(pub_pmids)),
        "rentrez_unique_pmids": int(len(ren_pmids)),
        "shared_unique_pmids": int(len(shared_pmids)),
        "shared_pmids_pub_multi_unique": int(shared["pub_multi_unique"].sum()),
        "shared_pmids_rentrez_multi_unique": int(shared["rentrez_multi_unique"].sum()),
        "shared_pmids_either_multi_unique": int(shared["either_multi_unique"].sum()),
        "shared_pmids_pub_multi_rows": int(shared["pub_multi_rows"].sum()),
        "shared_pmids_rentrez_multi_rows": int(shared["rentrez_multi_rows"].sum()),
        "shared_pmids_either_multi_rows": int(shared["either_multi_rows"].sum()),
        "shared_pmids_total_unique_claims_ge_3": int((shared["total_unique_claims_both_sources"] >= 3).sum()),
        "shared_pmids_total_rows_ge_3": int((shared["total_rows_both_sources"] >= 3).sum()),
    }

    counts_csv = out_dir / "shared_pmid_claim_counts.csv"
    dist_csv = out_dir / "shared_pmid_claim_count_distribution.csv"
    top_csv = out_dir / "shared_pmid_claim_counts_top.csv"
    summary_json = out_dir / "shared_pmid_claim_counts_summary.json"

    shared.reset_index().to_csv(counts_csv, index=False)
    dist.to_csv(dist_csv, index=False)
    shared.sort_values(
        ["total_unique_claims_both_sources", "total_rows_both_sources"],
        ascending=False,
    ).head(args.top_k).reset_index().to_csv(top_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Shared PMID claim count summary:")
    print(json.dumps(summary, indent=2))
    print("\nWrote:")
    print(f"  {counts_csv}")
    print(f"  {dist_csv}")
    print(f"  {top_csv}")
    print(f"  {summary_json}")


if __name__ == "__main__":
    main()
