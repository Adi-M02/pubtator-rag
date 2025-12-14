#!/usr/bin/env python3
import argparse, json, time, re
from pathlib import Path
from collections import Counter

import pandas as pd
import networkx as nx


def norm_basic(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def key_fn(mode: str):
    if mode == "raw":
        return lambda x: "" if x is None else str(x).strip()
    if mode == "norm_basic":
        return lambda x: norm_basic(x)
    raise ValueError(f"Unknown key mode: {mode}")


def ensure_flag(val, name: str):
    if val is None:
        raise SystemExit(
            f"Missing required choice: {name}. "
            f"Run with --help and explicitly set this option."
        )


def load_pubtator(pub_dir: Path):
    pub_dir = pub_dir.resolve()
    summ = sorted(pub_dir.glob("*_summary.csv"))
    art = sorted(pub_dir.glob("pipeline_*.json"))
    if not summ:
        raise SystemExit(f"No *_summary.csv found in {pub_dir}")
    if not art:
        raise SystemExit(f"No pipeline_*.json found in {pub_dir}")

    summ_path = max(summ, key=lambda p: p.stat().st_mtime)
    art_path = max(art, key=lambda p: p.stat().st_mtime)

    df = pd.read_csv(summ_path)
    need = {"drug_name", "disease_name", "pmid_count", "total_articles"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"PubTator summary missing columns: {sorted(miss)}")

    for c in ["pmid_count", "total_articles"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    with open(art_path, "r", encoding="utf-8") as f:
        artifact = json.load(f)

    return df, artifact, summ_path, art_path


def load_ctgov(ct_path: Path, disease_split: str | None):
    df = pd.read_csv(ct_path)

    need = {"drug", "nct_id", "disease", "first_submit"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"CT file missing columns: {sorted(miss)}")

    df["drug"] = df["drug"].astype(str)
    df["nct_id"] = df["nct_id"].astype(str)
    df["disease"] = df["disease"].astype(str)
    df["first_submit"] = pd.to_datetime(df["first_submit"], errors="coerce", utc=True)

    has_semicolon = df["disease"].str.contains(";", regex=False).any()
    if has_semicolon and not disease_split:
        raise SystemExit(
            "CT disease column contains ';' but you did not specify --ct-disease-split. "
            "Set it to ';' if you want to split multi-condition rows into multiple diseases, "
            "or set it to '' to keep the raw string."
        )

    if disease_split:
        rows = []
        for _, r in df.iterrows():
            parts = [p.strip() for p in str(r["disease"]).split(disease_split)]
            parts = [p for p in parts if p]
            if not parts:
                parts = [""]
            for d in parts:
                rows.append(
                    {
                        "drug": r["drug"],
                        "nct_id": r["nct_id"],
                        "title": r.get("title", ""),
                        "status": r.get("status", ""),
                        "phase": r.get("phase", ""),
                        "disease": d,
                        "first_submit": r["first_submit"],
                    }
                )
        df = pd.DataFrame(rows)

    return df


def pub_year_counts(artifact: dict) -> pd.DataFrame:
    rows = []
    for ind in artifact.get("indications", []) or []:
        drug_name = ind.get("drug_name", "")
        for ev in ind.get("evidence", []) or []:
            disease_name = ev.get("disease_name", "")
            for a in ev.get("articles", []) or []:
                dt = a.get("date")
                if isinstance(dt, str) and len(dt) >= 4 and dt[:4].isdigit():
                    rows.append((int(dt[:4]), drug_name, disease_name))
    if not rows:
        return pd.DataFrame(columns=["year", "n_articles", "n_drugs", "n_diseases"])

    tmp = pd.DataFrame(rows, columns=["year", "drug_name", "disease_name"])
    return (
        tmp.groupby("year")
        .agg(
            n_articles=("year", "size"),
            n_drugs=("drug_name", "nunique"),
            n_diseases=("disease_name", "nunique"),
        )
        .reset_index()
        .sort_values("year")
    )


def ct_year_counts(ct: pd.DataFrame) -> pd.DataFrame:
    tmp = ct.dropna(subset=["first_submit"]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=["year", "n_trials", "n_drugs", "n_diseases"])

    tmp["year"] = tmp["first_submit"].dt.year.astype(int)
    return (
        tmp.groupby("year")
        .agg(
            n_trials=("nct_id", "nunique"),
            n_drugs=("drug", "nunique"),
            n_diseases=("disease", "nunique"),
        )
        .reset_index()
        .sort_values("year")
    )


def build_bipartite_edges_pub(pub: pd.DataFrame, drug_key, disease_key):
    e = pub[["drug_name", "disease_name", "total_articles", "pmid_count"]].copy()
    e["drug_k"] = e["drug_name"].map(drug_key)
    e["dis_k"] = e["disease_name"].map(disease_key)
    e = e[(e["drug_k"] != "") & (e["dis_k"] != "")]
    edges = {(r["drug_k"], r["dis_k"]) for _, r in e.iterrows()}
    return edges, e


def build_bipartite_edges_ct(ct: pd.DataFrame, drug_key, disease_key):
    e = ct[["drug", "disease", "nct_id"]].copy()
    e["drug_k"] = e["drug"].map(drug_key)
    e["dis_k"] = e["disease"].map(disease_key)
    e = e[(e["drug_k"] != "") & (e["dis_k"] != "")]
    edges = {(r["drug_k"], r["dis_k"]) for _, r in e.iterrows()}
    return edges, e


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def bipartite_stats(edges: set[tuple[str, str]]):
    B = nx.Graph()
    drugs = {u for u, _ in edges}
    dis = {v for _, v in edges}
    B.add_nodes_from(drugs, bipartite="drug")
    B.add_nodes_from(dis, bipartite="disease")
    B.add_edges_from(edges)

    n = B.number_of_nodes()
    m = B.number_of_edges()
    possible = len(drugs) * len(dis)
    density = (m / possible) if possible else 0.0

    comps = list(nx.connected_components(B)) if n else []
    n_comp = len(comps)
    lcc = max((len(c) for c in comps), default=0)

    deg = dict(B.degree())
    drug_deg = [deg[x] for x in drugs] if drugs else []
    dis_deg = [deg[x] for x in dis] if dis else []

    def summ(xs):
        if not xs:
            return {"min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0.0}
        xs = sorted(xs)
        return {
            "min": int(xs[0]),
            "p50": int(xs[len(xs) // 2]),
            "p90": int(xs[int(0.9 * (len(xs) - 1))]),
            "max": int(xs[-1]),
            "mean": float(sum(xs) / len(xs)),
        }

    return {
        "n_nodes": int(n),
        "n_drugs": int(len(drugs)),
        "n_diseases": int(len(dis)),
        "n_edges": int(m),
        "density": float(density),
        "n_components": int(n_comp),
        "largest_component_size": int(lcc),
        "drug_degree_summary": summ(drug_deg),
        "disease_degree_summary": summ(dis_deg),
    }


def freq_tables_pub(pub_e: pd.DataFrame):
    d = (
        pub_e.groupby("dis_k")
        .agg(
            n_edges=("dis_k", "size"),
            n_drugs=("drug_k", "nunique"),
            total_articles_sum=("total_articles", "sum"),
            pmid_sum=("pmid_count", "sum"),
        )
        .reset_index()
        .sort_values(["n_drugs", "total_articles_sum", "n_edges"], ascending=False)
    )
    d["edge_share"] = d["n_edges"] / max(1, int(d["n_edges"].sum()))
    d["articles_share"] = d["total_articles_sum"] / max(1, int(d["total_articles_sum"].sum()))
    return d


def freq_tables_ct(ct_e: pd.DataFrame):
    d = (
        ct_e.groupby("dis_k")
        .agg(
            n_edges=("dis_k", "size"),
            n_drugs=("drug_k", "nunique"),
            n_trials=("nct_id", "nunique"),
        )
        .reset_index()
        .sort_values(["n_drugs", "n_trials", "n_edges"], ascending=False)
    )
    d["edge_share"] = d["n_edges"] / max(1, int(d["n_edges"].sum()))
    d["trial_share"] = d["n_trials"] / max(1, int(d["n_trials"].sum()))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pub-run-dir", type=Path, required=True)
    ap.add_argument("--ct-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)

    ap.add_argument("--drug-key-mode", choices=["raw", "norm_basic"], default=None)
    ap.add_argument("--disease-key-mode", choices=["raw", "norm_basic"], default=None)
    ap.add_argument("--ct-disease-split", type=str, default=None)

    args = ap.parse_args()
    ensure_flag(args.drug_key_mode, "--drug-key-mode")
    ensure_flag(args.disease_key_mode, "--disease-key-mode")

    drug_key = key_fn(args.drug_key_mode)
    disease_key = key_fn(args.disease_key_mode)

    pub, artifact, pub_csv, pub_json = load_pubtator(args.pub_run_dir)
    ct = load_ctgov(args.ct_path, disease_split=(args.ct_disease_split or None))

    pub_edges, pub_e = build_bipartite_edges_pub(pub, drug_key, disease_key)
    ct_edges, ct_e = build_bipartite_edges_ct(ct, drug_key, disease_key)

    pub_diseases = {disease_key(x) for x in pub["disease_name"].tolist()}
    ct_diseases = {disease_key(x) for x in ct["disease"].tolist()}
    pub_diseases.discard("")
    ct_diseases.discard("")

    shared_diseases = pub_diseases & ct_diseases
    pub_only_diseases = pub_diseases - ct_diseases
    ct_only_diseases = ct_diseases - pub_diseases

    shared_edges = pub_edges & ct_edges
    pub_only_edges = pub_edges - ct_edges
    ct_only_edges = ct_edges - pub_edges

    pub_year = pub_year_counts(artifact)
    ct_year = ct_year_counts(ct)

    pub_freq = freq_tables_pub(pub_e)
    ct_freq = freq_tables_ct(ct_e)

    stats = {
        "inputs": {
            "pub_summary_csv": str(pub_csv),
            "pub_artifact_json": str(pub_json),
            "ct_path": str(args.ct_path.resolve()),
            "drug_key_mode": args.drug_key_mode,
            "disease_key_mode": args.disease_key_mode,
            "ct_disease_split": args.ct_disease_split,
        },
        "set_overlap": {
            "pub_diseases": int(len(pub_diseases)),
            "ct_diseases": int(len(ct_diseases)),
            "shared_diseases": int(len(shared_diseases)),
            "pub_only_diseases": int(len(pub_only_diseases)),
            "ct_only_diseases": int(len(ct_only_diseases)),
            "disease_jaccard": float(jaccard(pub_diseases, ct_diseases)),
            "pub_edges": int(len(pub_edges)),
            "ct_edges": int(len(ct_edges)),
            "shared_edges": int(len(shared_edges)),
            "pub_only_edges": int(len(pub_only_edges)),
            "ct_only_edges": int(len(ct_only_edges)),
            "edge_jaccard": float(jaccard(pub_edges, ct_edges)),
        },
        "network_pubtator": bipartite_stats(pub_edges),
        "network_ctgov": bipartite_stats(ct_edges),
    }

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir or Path("outputs") / f"compare_pubtator_ctgov_{run_id}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "comparison_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    pd.DataFrame({"disease_key": sorted(shared_diseases)}).to_csv(out_dir / "shared_diseases.csv", index=False)
    pd.DataFrame({"disease_key": sorted(pub_only_diseases)}).to_csv(out_dir / "pub_only_diseases.csv", index=False)
    pd.DataFrame({"disease_key": sorted(ct_only_diseases)}).to_csv(out_dir / "ct_only_diseases.csv", index=False)

    pd.DataFrame(list(shared_edges), columns=["drug_key", "disease_key"]).to_csv(out_dir / "shared_edges.csv", index=False)
    pd.DataFrame(list(pub_only_edges), columns=["drug_key", "disease_key"]).to_csv(out_dir / "pub_only_edges.csv", index=False)
    pd.DataFrame(list(ct_only_edges), columns=["drug_key", "disease_key"]).to_csv(out_dir / "ct_only_edges.csv", index=False)

    pub_freq.to_csv(out_dir / "pubtator_disease_frequency.csv", index=False)
    ct_freq.to_csv(out_dir / "ctgov_disease_frequency.csv", index=False)

    pub_year.to_csv(out_dir / "pubtator_publication_year_counts.csv", index=False)
    ct_year.to_csv(out_dir / "ctgov_trial_first_submit_year_counts.csv", index=False)

    print(f"Wrote outputs to: {out_dir}")
    print("Key files:")
    print(f"  - {out_dir / 'comparison_stats.json'}")
    print(f"  - {out_dir / 'pubtator_disease_frequency.csv'}")
    print(f"  - {out_dir / 'ctgov_disease_frequency.csv'}")
    print(f"  - {out_dir / 'pubtator_publication_year_counts.csv'}")
    print(f"  - {out_dir / 'ctgov_trial_first_submit_year_counts.csv'}")


if __name__ == "__main__":
    main()
