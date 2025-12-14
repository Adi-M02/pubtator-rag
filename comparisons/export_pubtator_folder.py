#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd


def norm_basic(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    out = []
    prev_space = False
    for ch in s:
        ok = ("a" <= ch <= "z") or ("0" <= ch <= "9")
        if ok:
            out.append(ch)
            prev_space = False
        else:
            if not prev_space:
                out.append(" ")
                prev_space = True
    return " ".join("".join(out).split())


def singularize_token(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("ses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def token_key(s: str) -> tuple:
    toks = norm_basic(s).split()
    toks = [singularize_token(t) for t in toks if t]
    c = Counter(toks)
    return tuple(sorted(c.items()))


def safe_year(dt) -> int | None:
    if pd.isna(dt):
        return None
    try:
        return int(dt.year)
    except Exception:
        return None


def atomic_write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pubtator-summary-csv", type=Path, required=True)
    ap.add_argument("--pubtator-json", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--agg-mode", choices=["max", "sum"], default="max")
    ap.add_argument("--write-articles-long", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pubtator_summary_csv)
    need = {"drug_name", "drug_id", "disease_name", "disease_id", "total_articles"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"PubTator summary missing columns: {sorted(miss)}")

    df["drug_name"] = df["drug_name"].astype(str)
    df["drug_key"] = df["drug_name"].map(norm_basic)

    df["disease_name"] = df["disease_name"].astype(str)
    df["disease_key_norm"] = df["disease_name"].map(norm_basic)
    df["disease_key_tokens"] = df["disease_name"].map(token_key).map(lambda x: json.dumps(list(x)))

    df["total_articles"] = pd.to_numeric(df["total_articles"], errors="coerce").fillna(0).astype(int)

    # Collapse drug_ids under drug_name by (drug_name, disease_id)
    gcols = ["drug_name", "drug_key", "disease_id"]
    def pick_label(x: pd.Series) -> str:
        vc = x.value_counts(dropna=False)
        return str(vc.index[0]) if len(vc) else ""

    agg = df.groupby(gcols, dropna=False).agg(
        disease_name=("disease_name", pick_label),
        disease_key_norm=("disease_key_norm", pick_label),
        disease_key_tokens=("disease_key_tokens", pick_label),
        weight_total_articles_sum=("total_articles", "sum"),
        weight_total_articles_max=("total_articles", "max"),
    ).reset_index()

    if args.agg_mode == "sum":
        agg["weight_total_articles"] = agg["weight_total_articles_sum"]
    else:
        agg["weight_total_articles"] = agg["weight_total_articles_max"]

    # Load JSON for PMIDs and year counts
    art_rows = []
    j = json.loads(args.pubtator_json.read_text(encoding="utf-8"))
    indications = j.get("indications", []) or []
    for ind in indications:
        drug_name = str(ind.get("drug_name", "") or "")
        drug_key = norm_basic(drug_name)
        evidence = ind.get("evidence", []) or []
        for ev in evidence:
            disease_id = str(ev.get("disease_id", "") or "")
            disease_name = str(ev.get("disease_name", "") or "")
            articles = ev.get("articles", []) or []
            for a in articles:
                pmid = str(a.get("pmid", "") or "")
                date_s = a.get("date", None)
                dt = pd.to_datetime(date_s, errors="coerce", utc=True)
                y = safe_year(dt)
                if pmid:
                    art_rows.append(
                        {
                            "drug_name": drug_name,
                            "drug_key": drug_key,
                            "disease_id": disease_id,
                            "disease_name_json": disease_name,
                            "pmid": pmid,
                            "date": None if pd.isna(dt) else dt.isoformat(),
                            "year": y,
                        }
                    )

    df_art = pd.DataFrame(art_rows) if art_rows else pd.DataFrame(
        columns=["drug_name","drug_key","disease_id","disease_name_json","pmid","date","year"]
    )

    # Collapsed PMID unions and year counts per (drug_name, disease_id)
    pmid_info = {}
    if len(df_art):
        for (drug_name, drug_key, disease_id), sub in df_art.groupby(["drug_name", "drug_key", "disease_id"], dropna=False):
            pmids = set(sub["pmid"].astype(str).tolist())
            years = [y for y in sub["year"].tolist() if isinstance(y, int)]
            yc = Counter(years)
            pmid_info[(drug_name, drug_key, disease_id)] = {
                "unique_pmids_merged": int(len(pmids)),
                "year_min": int(min(yc)) if yc else None,
                "year_max": int(max(yc)) if yc else None,
                "year_counts_json": json.dumps({str(k): int(v) for k, v in sorted(yc.items())}),
            }

    extra_cols = []
    for _, r in agg.iterrows():
        k = (r["drug_name"], r["drug_key"], r["disease_id"])
        info = pmid_info.get(k, None)
        if info is None:
            extra_cols.append({"unique_pmids_merged": 0, "year_min": None, "year_max": None, "year_counts_json": "{}"})
        else:
            extra_cols.append(info)

    df_extra = pd.DataFrame(extra_cols)
    out_edges = pd.concat([agg, df_extra], axis=1)

    edges_path = args.outdir / "pubtator_canonical_edges.csv"
    out_edges.to_csv(edges_path, index=False)

    art_path = None
    if args.write_articles_long:
        art_path = args.outdir / "pubtator_canonical_articles_long.csv"
        df_art.to_csv(art_path, index=False)

    decisions = {
        "pubtator_source_files": {
            "summary_csv": str(args.pubtator_summary_csv),
            "json": str(args.pubtator_json),
        },
        "collapse_rules": {
            "collapse_drug_ids_under_drug_name": True,
            "edge_key": ["drug_name", "disease_id"],
            "weight_total_articles_aggregation": args.agg_mode,
            "unique_pmids_merged": "union of PMIDs across all drug_ids under the same drug_name for a given disease_id (within collected evidence)",
            "year_counts": "derived from PubTator JSON evidence articles[].date, aggregated per (drug_name, disease_id)",
        },
        "string_keys": {
            "drug_key": "norm_basic(drug_name)",
            "disease_key_norm": "norm_basic(disease_name)",
            "disease_key_tokens": "token multiset with lightweight singularization, stored as JSON",
        },
        "notes": [
            "PMIDs per drug-disease are capped by the upstream collection; counts reflect collected evidence, not necessarily all PubMed.",
        ],
    }
    decisions_path = args.outdir / "pubtator_decisions.json"
    atomic_write_json(decisions_path, decisions)

    print(f"Wrote: {edges_path}")
    if art_path:
        print(f"Wrote: {art_path}")
    print(f"Wrote: {decisions_path}")


if __name__ == "__main__":
    main()
