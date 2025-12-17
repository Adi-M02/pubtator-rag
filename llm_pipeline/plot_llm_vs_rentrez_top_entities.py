#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SAMPLE_JSON = "manual_evaluation/eval_results/shared_pmids_eval_stratified_sample_pmids.json"


NA_SET = {"", "NA", "N/A", "NAN", "NONE", "NULL"}


def norm_na(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.upper() in NA_SET else s


def norm_key(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def resolve_path(p: str, script_dir: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    # treat relative paths as relative to project root (parent of script dir) by default
    return (script_dir.parent / pp).resolve()


def load_sample_pmids(sample_json: Path) -> list[str]:
    obj = json.loads(sample_json.read_text(encoding="utf-8"))
    pmids = obj.get("pmids", [])
    if not isinstance(pmids, list) or not pmids:
        raise ValueError(f"No PMIDs found in sample file: {sample_json}")
    out = []
    for x in pmids:
        s = norm_na(x)
        if s:
            out.append(s)
    if not out:
        raise ValueError(f"No usable PMIDs found in sample file: {sample_json}")
    return out


def find_default_sample_json(script_dir: Path) -> Path:
    project_dir = script_dir.parent
    candidates = [
        project_dir / DEFAULT_SAMPLE_JSON,
        script_dir / DEFAULT_SAMPLE_JSON,
        project_dir / "eval_results" / Path(DEFAULT_SAMPLE_JSON).name,
        project_dir / "manual_evaluation" / "eval_results" / Path(DEFAULT_SAMPLE_JSON).name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find the sample PMID JSON.\n"
        f"Tried: {', '.join(str(x) for x in candidates)}\n"
        "Pass --sample_json explicitly to fix."
    )


@dataclass
class SeriesCounts:
    counts: Counter
    display: dict[str, str]


def add_count(counter: Counter, display: dict[str, str], raw: str) -> None:
    raw = norm_na(raw)
    if not raw:
        return
    k = norm_key(raw)
    if not k:
        return
    counter[k] += 1
    # store a stable display label, prefer first seen
    if k not in display:
        display[k] = raw


def most_common_display(display: dict[str, str], key: str) -> str:
    return display.get(key, key)


def load_rentrez_counts(
    rentrez_csv: Path,
    sample_pmids: set[str],
    require_positive: bool,
) -> tuple[SeriesCounts, SeriesCounts]:
    df = pd.read_csv(rentrez_csv, dtype=str).fillna("")
    need = {"pmid", "drug", "disease"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Rentrez CSV missing columns: {sorted(missing)}")

    if require_positive and "positive" in df.columns:
        pos = df["positive"].fillna("").astype(str).str.strip().str.upper()
        df = df[pos.isin({"TRUE", "T", "1", "YES", "Y"})].copy()

    df["pmid"] = df["pmid"].map(norm_na)
    df["drug"] = df["drug"].map(norm_na)
    df["disease"] = df["disease"].map(norm_na)

    df = df[df["pmid"].isin(sample_pmids)].copy()

    drug_counts = Counter()
    drug_display: dict[str, str] = {}
    dis_counts = Counter()
    dis_display: dict[str, str] = {}

    for _, r in df.iterrows():
        add_count(drug_counts, drug_display, r["drug"])
        add_count(dis_counts, dis_display, r["disease"])

    return SeriesCounts(drug_counts, drug_display), SeriesCounts(dis_counts, dis_display)


def load_llm_counts(
    llm_jsonl: Path,
    sample_pmids: set[str],
    disease_mode: str,
) -> tuple[SeriesCounts, SeriesCounts]:
    drug_counts = Counter()
    drug_display: dict[str, str] = {}
    dis_counts = Counter()
    dis_display: dict[str, str] = {}

    with llm_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            pmid = norm_na(row.get("pmid", ""))
            if not pmid or pmid not in sample_pmids:
                continue

            llm_json = row.get("llm_json")
            if not isinstance(llm_json, dict):
                continue

            drugs = llm_json.get("drugs", [])
            if isinstance(drugs, list):
                for d in drugs:
                    add_count(drug_counts, drug_display, d)

            dcs = llm_json.get("disease_concepts", [])
            if not isinstance(dcs, list):
                continue

            for dc in dcs:
                if not isinstance(dc, dict):
                    continue
                canon = norm_na(dc.get("canonical", ""))
                aliases = dc.get("aliases", [])
                aliases = aliases if isinstance(aliases, list) else []

                if disease_mode == "canonical":
                    add_count(dis_counts, dis_display, canon)
                elif disease_mode == "canonical_plus_aliases":
                    if canon:
                        add_count(dis_counts, dis_display, canon)
                    for a in aliases:
                        add_count(dis_counts, dis_display, a)
                elif disease_mode == "aliases_only":
                    for a in aliases:
                        add_count(dis_counts, dis_display, a)
                else:
                    raise ValueError("Invalid --llm_disease_mode. Use: canonical, canonical_plus_aliases, aliases_only")

    return SeriesCounts(drug_counts, drug_display), SeriesCounts(dis_counts, dis_display)


def pick_top_union(
    a: Counter,
    b: Counter,
    top_n: int,
) -> list[str]:
    top_a = [k for k, _ in a.most_common(top_n)]
    top_b = [k for k, _ in b.most_common(top_n)]
    seen = set()
    out = []
    for k in top_a + top_b:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out[:top_n]


def build_compare_table(
    keys: list[str],
    llm: SeriesCounts,
    ren: SeriesCounts,
) -> pd.DataFrame:
    rows = []
    for k in keys:
        rows.append(
            {
                "entity_key": k,
                "entity_label": most_common_display(llm.display, k) if k in llm.counts else most_common_display(ren.display, k),
                "llm_count": int(llm.counts.get(k, 0)),
                "rentrez_count": int(ren.counts.get(k, 0)),
            }
        )
    df = pd.DataFrame(rows)
    # stable ordering: by max count desc, then label
    df["max_count"] = df[["llm_count", "rentrez_count"]].max(axis=1)
    df = df.sort_values(["max_count", "entity_label"], ascending=[False, True]).drop(columns=["max_count"])
    return df


def plot_grouped_bar(df: pd.DataFrame, title: str, out_png: Path) -> None:
    labels = df["entity_label"].tolist()
    llm_counts = df["llm_count"].tolist()
    ren_counts = df["rentrez_count"].tolist()

    x = list(range(len(labels)))
    w = 0.4

    plt.figure(figsize=(14, 6))
    plt.bar([i - w / 2 for i in x], llm_counts, width=w, label="LLM")
    plt.bar([i + w / 2 for i in x], ren_counts, width=w, label="Rentrez")
    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel("Count (mentions across PMIDs)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_jsonl", required=True, help="Path to llm_outputs.jsonl")
    ap.add_argument("--rentrez_csv", required=True, help="Path to rentrez CSV with columns pmid, drug, disease")
    ap.add_argument("--sample_json", default="", help="Path to *_sample_pmids.json. If empty, tries a default location.")
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--run_tag", default="llm_vs_rentrez_top_entities")
    ap.add_argument("--require_positive", action="store_true", help="If set and rentrez has 'positive', keep only TRUE-like rows")
    ap.add_argument(
        "--llm_disease_mode",
        default="canonical",
        choices=["canonical", "canonical_plus_aliases", "aliases_only"],
        help="How to count LLM diseases",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_root = script_dir.parent / "llm_compare"
    out_dir = out_root / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_jsonl = resolve_path(args.llm_jsonl, script_dir)
    rentrez_csv = resolve_path(args.rentrez_csv, script_dir)

    if args.sample_json:
        sample_json = resolve_path(args.sample_json, script_dir)
    else:
        sample_json = find_default_sample_json(script_dir)

    if not llm_jsonl.exists():
        raise FileNotFoundError(llm_jsonl)
    if not rentrez_csv.exists():
        raise FileNotFoundError(rentrez_csv)
    if not sample_json.exists():
        raise FileNotFoundError(sample_json)

    pmids = load_sample_pmids(sample_json)
    pmid_set = set(pmids)

    llm_drugs, llm_dis = load_llm_counts(llm_jsonl, pmid_set, disease_mode=args.llm_disease_mode)
    ren_drugs, ren_dis = load_rentrez_counts(rentrez_csv, pmid_set, require_positive=bool(args.require_positive))

    top_drug_keys = pick_top_union(llm_drugs.counts, ren_drugs.counts, top_n=int(args.top_n))
    top_dis_keys = pick_top_union(llm_dis.counts, ren_dis.counts, top_n=int(args.top_n))

    drug_df = build_compare_table(top_drug_keys, llm_drugs, ren_drugs)
    dis_df = build_compare_table(top_dis_keys, llm_dis, ren_dis)

    drug_csv = out_dir / "top_drugs_counts.csv"
    dis_csv = out_dir / "top_diseases_counts.csv"
    drug_df.to_csv(drug_csv, index=False)
    dis_df.to_csv(dis_csv, index=False)

    plot_grouped_bar(
        drug_df,
        title=f"Top {args.top_n} drugs (LLM vs Rentrez) on sampled PMIDs",
        out_png=out_dir / "top_drugs_bar.png",
    )
    plot_grouped_bar(
        dis_df,
        title=f"Top {args.top_n} diseases (LLM vs Rentrez) on sampled PMIDs",
        out_png=out_dir / "top_diseases_bar.png",
    )

    meta = {
        "llm_jsonl": str(llm_jsonl),
        "rentrez_csv": str(rentrez_csv),
        "sample_json": str(sample_json),
        "n_pmids": len(pmids),
        "top_n": int(args.top_n),
        "require_positive": bool(args.require_positive),
        "llm_disease_mode": args.llm_disease_mode,
        "outputs": {
            "top_drugs_counts_csv": str(drug_csv),
            "top_diseases_counts_csv": str(dis_csv),
            "top_drugs_bar_png": str(out_dir / "top_drugs_bar.png"),
            "top_diseases_bar_png": str(out_dir / "top_diseases_bar.png"),
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {drug_csv}")
    print(f"Wrote: {dis_csv}")
    print(f"Wrote: {out_dir / 'top_drugs_bar.png'}")
    print(f"Wrote: {out_dir / 'top_diseases_bar.png'}")
    print(f"Wrote: {out_dir / 'run_metadata.json'}")
    print(f"Outputs folder: {out_dir}")


if __name__ == "__main__":
    main()
