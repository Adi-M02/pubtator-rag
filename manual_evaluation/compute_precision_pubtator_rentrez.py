#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def norm_na(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.upper() in {"", "NA", "N/A", "NAN", "NONE", "NULL"}:
        return ""
    return s


def norm_key(s: str) -> str:
    s = norm_na(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_json_cell(cell: str):
    s = norm_na(cell)
    if not s:
        return None
    if '""' in s and '\\"' not in s:
        s = s.replace('""', '"')
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"(\[.*\]|\{.*\})", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(1))
        except Exception:
            return None


def claim_key_pub(drug: str, disease: str, disease_id: str) -> str:
    return f"{norm_key(drug)}|||{norm_key(disease)}|||{norm_key(disease_id)}"


def claim_key_ren(drug: str, disease: str) -> str:
    return f"{norm_key(drug)}|||{norm_key(disease)}"


def load_manual_eval(manual_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(manual_csv, dtype=str).fillna("")
    need = {
        "pmid",
        "group",
        "pubtator_agree",
        "rentrez_agree",
        "pubtator_true_claims_json",
        "rentrez_true_claims_json",
    }
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Manual eval CSV missing columns: {sorted(missing)}")
    df["pmid"] = df["pmid"].map(norm_na)
    df["group"] = df["group"].map(norm_na)
    df["pubtator_agree"] = df["pubtator_agree"].map(norm_na)
    df["rentrez_agree"] = df["rentrez_agree"].map(norm_na)
    df = df[df["pmid"] != ""].copy()
    return df


def build_manual_true_sets(df: pd.DataFrame) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    pub_true: dict[str, set[str]] = {}
    ren_true: dict[str, set[str]] = {}

    for _, r in df.iterrows():
        pmid = r["pmid"]

        pub_set: set[str] = set()
        pub_arr = parse_json_cell(r.get("pubtator_true_claims_json", ""))
        if isinstance(pub_arr, list):
            for x in pub_arr:
                if not isinstance(x, dict):
                    continue
                dr = norm_na(x.get("drug", ""))
                di = norm_na(x.get("disease", ""))
                did = norm_na(x.get("disease_id", ""))
                if dr and di:
                    pub_set.add(claim_key_pub(dr, di, did))
        pub_true[pmid] = pub_set

        ren_set: set[str] = set()
        ren_arr = parse_json_cell(r.get("rentrez_true_claims_json", ""))
        if isinstance(ren_arr, list):
            for x in ren_arr:
                if not isinstance(x, dict):
                    continue
                dr = norm_na(x.get("drug", ""))
                di = norm_na(x.get("disease", ""))
                if dr and di:
                    ren_set.add(claim_key_ren(dr, di))
        ren_true[pmid] = ren_set

    return pub_true, ren_true


def load_rentrez_claims(rentrez_csv: Path, pmids: set[str], require_positive: bool) -> dict[str, set[str]]:
    df = pd.read_csv(rentrez_csv, dtype=str).fillna("")
    need = {"pmid", "drug", "disease"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Rentrez CSV missing columns: {sorted(missing)}")

    df["pmid"] = df["pmid"].map(norm_na)
    df["drug"] = df["drug"].map(norm_na)
    df["disease"] = df["disease"].map(norm_na)

    if require_positive and "positive" in df.columns:
        pos = df["positive"].fillna("").astype(str).str.strip().str.upper()
        df = df[pos.isin({"TRUE", "T", "1", "YES", "Y"})]

    df = df[df["pmid"].isin(pmids)]
    out: dict[str, set[str]] = defaultdict(set)
    for _, r in df.iterrows():
        if not r["pmid"] or not r["drug"] or not r["disease"]:
            continue
        out[r["pmid"]].add(claim_key_ren(r["drug"], r["disease"]))
    return dict(out)


def load_pubtator_claims(pub_csv: Path, pmids: set[str], chunksize: int) -> dict[str, set[str]]:
    usecols = ["drug_name", "disease_name", "disease_id", "pmid"]
    out: dict[str, set[str]] = defaultdict(set)

    it = pd.read_csv(pub_csv, dtype=str, usecols=usecols, chunksize=chunksize)
    for chunk in it:
        chunk = chunk.fillna("")
        chunk["pmid"] = chunk["pmid"].map(norm_na)
        chunk = chunk[chunk["pmid"].isin(pmids)]
        if chunk.empty:
            continue

        chunk["drug_name"] = chunk["drug_name"].map(norm_na)
        chunk["disease_name"] = chunk["disease_name"].map(norm_na)
        chunk["disease_id"] = chunk["disease_id"].map(norm_na)

        chunk = chunk[(chunk["drug_name"] != "") & (chunk["disease_name"] != "")]
        if chunk.empty:
            continue

        for _, r in chunk.iterrows():
            out[r["pmid"]].add(claim_key_pub(r["drug_name"], r["disease_name"], r["disease_id"]))

    return dict(out)


def precision_from_claims(
    df: pd.DataFrame,
    dataset: str,
    source_claims: dict[str, set[str]],
    manual_true: dict[str, set[str]],
    agree_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for _, r in df.iterrows():
        pmid = r["pmid"]
        grp = r["group"] or ""
        agree = r.get(agree_col, "")
        if agree not in {"Yes", "No"}:
            continue

        pred = source_claims.get(pmid, set())
        true = manual_true.get(pmid, set())

        tp = len(pred & true)
        fp = len(pred - true)
        missing_true = len(true - pred)

        rows.append(
            {
                "pmid": pmid,
                "group": grp,
                "dataset": dataset,
                "agree": agree,
                "pred_claims": len(pred),
                "true_claims": len(true),
                "tp": tp,
                "fp": fp,
                "missing_true": missing_true,
                "precision": (tp / (tp + fp)) if (tp + fp) else "",
            }
        )

    per_pmid = pd.DataFrame(rows)

    def summarize(sub: pd.DataFrame) -> dict:
        tp = int(sub["tp"].sum()) if not sub.empty else 0
        fp = int(sub["fp"].sum()) if not sub.empty else 0
        denom = tp + fp
        return {
            "dataset": dataset,
            "n_pmids": int(sub["pmid"].nunique()) if not sub.empty else 0,
            "tp": tp,
            "fp": fp,
            "pred_claims_total": int(sub["pred_claims"].sum()) if not sub.empty else 0,
            "precision": (tp / denom) if denom else None,
        }

    summary_rows = []
    for grp in ["single", "multi"]:
        summary_rows.append({"group": grp, **summarize(per_pmid[per_pmid["group"] == grp])})
    summary_rows.append({"group": "all", **summarize(per_pmid)})

    summary = pd.DataFrame(summary_rows)
    return summary, per_pmid


def combined_precision(pub_per_pmid: pd.DataFrame, ren_per_pmid: pd.DataFrame) -> pd.DataFrame:
    if pub_per_pmid.empty and ren_per_pmid.empty:
        return pd.DataFrame([{"group": "single", "dataset": "Combined", "n_pmids": 0, "tp": 0, "fp": 0, "pred_claims_total": 0, "precision": None},
                             {"group": "multi", "dataset": "Combined", "n_pmids": 0, "tp": 0, "fp": 0, "pred_claims_total": 0, "precision": None},
                             {"group": "all", "dataset": "Combined", "n_pmids": 0, "tp": 0, "fp": 0, "pred_claims_total": 0, "precision": None}])

    both = pd.concat([pub_per_pmid, ren_per_pmid], ignore_index=True)
    # combine per pmid (sum tp/fp across datasets) then aggregate per group
    pm_agg = both.groupby(["pmid", "group"], as_index=False).agg(
        tp=("tp", "sum"),
        fp=("fp", "sum"),
        pred_claims=("pred_claims", "sum"),
    )

    def summarize(sub: pd.DataFrame) -> dict:
        tp = int(sub["tp"].sum()) if not sub.empty else 0
        fp = int(sub["fp"].sum()) if not sub.empty else 0
        denom = tp + fp
        return {
            "dataset": "Combined",
            "n_pmids": int(sub["pmid"].nunique()) if not sub.empty else 0,
            "tp": tp,
            "fp": fp,
            "pred_claims_total": int(sub["pred_claims"].sum()) if not sub.empty else 0,
            "precision": (tp / denom) if denom else None,
        }

    rows = []
    for grp in ["single", "multi"]:
        rows.append({"group": grp, **summarize(pm_agg[pm_agg["group"] == grp])})
    rows.append({"group": "all", **summarize(pm_agg)})

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual_csv", type=str, default="manual_evaluation/eval_results/shared_pmids_eval_stratified_pmid_eval.csv")
    ap.add_argument("--rentrez_csv", type=str, default="rentrez_data/pubmed_results.csv")
    ap.add_argument("--pubtator_csv", type=str, default="pubtator_outputs/drug_disease_pmid_long.csv")
    ap.add_argument("--require_positive", action="store_true")
    ap.add_argument("--chunksize", type=int, default=250_000)
    ap.add_argument("--out_dir", type=str, default="manual_evaluation/eval_results/precision_eval")
    ap.add_argument("--out_tag", type=str, default="precision_v2_claim_level")
    args = ap.parse_args()

    manual_csv = Path(args.manual_csv)
    rentrez_csv = Path(args.rentrez_csv)
    pubtator_csv = Path(args.pubtator_csv)
    out_dir = Path(args.out_dir) / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_manual_eval(manual_csv)
    pmids = set(df["pmid"].tolist())

    pub_true, ren_true = build_manual_true_sets(df)

    ren_claims = load_rentrez_claims(rentrez_csv, pmids, require_positive=bool(args.require_positive))
    pub_claims = load_pubtator_claims(pubtator_csv, pmids, chunksize=int(args.chunksize))

    pub_summary, pub_per_pmid = precision_from_claims(
        df=df,
        dataset="PubTator",
        source_claims=pub_claims,
        manual_true=pub_true,
        agree_col="pubtator_agree",
    )
    ren_summary, ren_per_pmid = precision_from_claims(
        df=df,
        dataset="Rentrez",
        source_claims=ren_claims,
        manual_true=ren_true,
        agree_col="rentrez_agree",
    )

    comb_summary = combined_precision(pub_per_pmid, ren_per_pmid)

    pub_summary.to_csv(out_dir / "claim_level_precision_pubtator_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    ren_summary.to_csv(out_dir / "claim_level_precision_rentrez_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    comb_summary.to_csv(out_dir / "claim_level_precision_combined_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    pub_per_pmid.to_csv(out_dir / "claim_level_precision_pubtator_by_pmid.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    ren_per_pmid.to_csv(out_dir / "claim_level_precision_rentrez_by_pmid.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    meta = {
        "manual_csv": str(manual_csv),
        "pubtator_csv": str(pubtator_csv),
        "rentrez_csv": str(rentrez_csv),
        "require_positive": bool(args.require_positive),
        "n_manual_rows": int(len(df)),
        "n_manual_pmids": int(df["pmid"].nunique()),
        "n_pubtator_pmids_with_claims": int(len(pub_claims)),
        "n_rentrez_pmids_with_claims": int(len(ren_claims)),
        "out_dir": str(out_dir),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote outputs to: {out_dir}\n")
    print("Claim level precision summary (PubTator):")
    print(pub_summary.to_string(index=False))
    print("\nClaim level precision summary (Rentrez):")
    print(ren_summary.to_string(index=False))
    print("\nClaim level precision summary (Combined PubTator + Rentrez):")
    print(comb_summary.to_string(index=False))


if __name__ == "__main__":
    main()
