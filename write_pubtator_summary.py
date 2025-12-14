#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd


def _find_one(run_dir: Path, pat: str) -> Path:
    hits = sorted(run_dir.glob(pat))
    if not hits:
        raise FileNotFoundError(f"No files match {pat} in {run_dir}")
    if len(hits) > 1:
        # prefer the most recently modified
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def load_inputs(run_dir: Path, summary_csv: Path | None, artifact_json: Path | None):
    run_dir = run_dir.resolve()
    summary_csv = summary_csv or _find_one(run_dir, "*_summary.csv")
    artifact_json = artifact_json or _find_one(run_dir, "pipeline_*.json")
    df = pd.read_csv(summary_csv)

    needed = {"drug_name", "drug_id", "disease_name", "disease_id", "pmid_count", "total_articles"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {summary_csv.name}: {sorted(missing)}")

    with open(artifact_json, "r", encoding="utf-8") as f:
        art = json.load(f)

    for c in ["pmid_count", "total_articles"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["drug_name"] = df["drug_name"].astype(str)
    df["drug_id"] = df["drug_id"].astype(str)
    df["disease_name"] = df["disease_name"].astype(str)
    df["disease_id"] = df["disease_id"].astype(str)

    return df, art, summary_csv, artifact_json


def publication_year_counts(artifact: dict) -> pd.DataFrame:
    rows = []
    for ind in artifact.get("indications", []):
        dname = ind.get("drug_name", "")
        dids = {ev.get("disease_id"): ev for ev in ind.get("evidence", [])}
        for disease_id, ev in dids.items():
            for a in ev.get("articles", []) or []:
                dt = a.get("date")
                if not dt or not isinstance(dt, str):
                    continue
                y = dt[:4]
                if len(y) == 4 and y.isdigit():
                    rows.append((int(y), dname, str(disease_id)))

    if not rows:
        return pd.DataFrame(columns=["year", "n_articles", "n_drug_names", "n_diseases"])

    tmp = pd.DataFrame(rows, columns=["year", "drug_name", "disease_id"])
    out = (
        tmp.groupby("year")
        .agg(
            n_articles=("year", "size"),
            n_drug_names=("drug_name", "nunique"),
            n_diseases=("disease_id", "nunique"),
        )
        .reset_index()
        .sort_values("year")
    )
    return out


def build_outputs(df: pd.DataFrame, artifact: dict):
    edges = df.drop_duplicates(subset=["drug_id", "disease_id"]).copy()

    n_edges = len(edges)
    drugs_by_name = edges["drug_name"].nunique()
    drugs_by_id = edges["drug_id"].nunique()
    diseases = edges["disease_id"].nunique()

    edge_pmid_sum = int(edges["pmid_count"].sum())
    edge_total_articles_sum = int(edges["total_articles"].sum())

    disease = (
        edges.groupby(["disease_id", "disease_name"])
        .agg(
            n_edges=("drug_id", "size"),
            n_drug_names=("drug_name", "nunique"),
            n_drug_ids=("drug_id", "nunique"),
            pmid_count_sum=("pmid_count", "sum"),
            total_articles_sum=("total_articles", "sum"),
            pmid_count_median=("pmid_count", "median"),
            total_articles_median=("total_articles", "median"),
        )
        .reset_index()
    )

    disease["edge_share"] = disease["n_edges"] / max(1, n_edges)
    disease["drug_name_share"] = disease["n_drug_names"] / max(1, drugs_by_name)
    disease["drug_id_share"] = disease["n_drug_ids"] / max(1, drugs_by_id)
    disease["pmid_share"] = disease["pmid_count_sum"] / max(1, edge_pmid_sum)
    disease["total_articles_share"] = disease["total_articles_sum"] / max(1, edge_total_articles_sum)

    disease = disease.sort_values(["n_drug_names", "total_articles_sum", "n_edges"], ascending=False)

    drug = (
        edges.groupby(["drug_id", "drug_name"])
        .agg(
            n_diseases=("disease_id", "nunique"),
            n_edges=("disease_id", "size"),
            pmid_count_sum=("pmid_count", "sum"),
            total_articles_sum=("total_articles", "sum"),
            total_articles_median=("total_articles", "median"),
        )
        .reset_index()
        .sort_values(["n_diseases", "total_articles_sum"], ascending=False)
    )

    unresolved = 0
    dropped_no_rel = len(artifact.get("dropped_no_relations", []) or [])
    hardcoded_drugs = artifact.get("drugs", []) or []
    drug_entities = artifact.get("drug_entities", []) or []
    resolved_drug_names = {d.get("drug_name") for d in drug_entities if d.get("entity_ids")}

    unresolved = len(set(hardcoded_drugs) - resolved_drug_names)

    overview = {
        "run_id": artifact.get("run_id"),
        "started_at": artifact.get("started_at"),
        "input_hardcoded_drugs": len(hardcoded_drugs),
        "resolved_drug_names": int(len(resolved_drug_names)),
        "unresolved_drug_names": int(unresolved),
        "drug_ids_total": int(drugs_by_id),
        "drug_names_total": int(drugs_by_name),
        "diseases_total": int(diseases),
        "edges_total": int(n_edges),
        "edges_with_pmids": int((edges["pmid_count"] > 0).sum()),
        "pmid_count_sum_observed": int(edge_pmid_sum),
        "total_articles_sum_reported": int(edge_total_articles_sum),
        "dropped_no_treat_relations": int(dropped_no_rel),
    }

    years = publication_year_counts(artifact)

    return overview, disease, drug, years


def write_report_snippet(out_dir: Path, overview: dict, disease: pd.DataFrame, years: pd.DataFrame):
    top = disease.head(10).copy()
    lines = []
    lines.append("### PubTator treatment relations summary\n")
    lines.append(
        f"- Run ID: {overview.get('run_id')}\n"
        f"- Drugs (input hardcoded list): {overview.get('input_hardcoded_drugs')}\n"
        f"- Drugs resolved (by name): {overview.get('resolved_drug_names')} (unresolved: {overview.get('unresolved_drug_names')})\n"
        f"- Unique drug identifiers (PubTator CHEMICAL IDs): {overview.get('drug_ids_total')}\n"
        f"- Unique diseases: {overview.get('diseases_total')}\n"
        f"- Unique drug–disease edges: {overview.get('edges_total')}\n"
        f"- Edges with at least 1 retrieved PMID: {overview.get('edges_with_pmids')}\n"
    )
    lines.append("\nTop diseases by number of distinct drugs linked:\n")
    for _, r in top.iterrows():
        lines.append(
            f"- {r['disease_name']} ({r['disease_id']}): "
            f"{int(r['n_drug_names'])} drugs, edge share={r['edge_share']:.3f}, "
            f"total_articles_sum={int(r['total_articles_sum'])}\n"
        )

    if len(years):
        y0, y1 = int(years["year"].min()), int(years["year"].max())
        lines.append(f"\nPublication years observed in retrieved evidence dates: {y0} to {y1}\n")

    (out_dir / "report_snippet.md").write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--summary-csv", type=Path, default=None)
    ap.add_argument("--artifact-json", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    df, artifact, summary_csv, artifact_json = load_inputs(args.run_dir, args.summary_csv, args.artifact_json)

    out_dir = args.out_dir or (args.run_dir / "analysis_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    overview, disease, drug, years = build_outputs(df, artifact)

    (out_dir / "run_overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    disease.to_csv(out_dir / "disease_frequency.csv", index=False)
    drug.to_csv(out_dir / "drug_summary.csv", index=False)
    years.to_csv(out_dir / "pub_year_counts.csv", index=False)

    write_report_snippet(out_dir, overview, disease, years)

    print(f"Wrote: {out_dir / 'run_overview.json'}")
    print(f"Wrote: {out_dir / 'disease_frequency.csv'}")
    print(f"Wrote: {out_dir / 'drug_summary.csv'}")
    print(f"Wrote: {out_dir / 'pub_year_counts.csv'}")
    print(f"Wrote: {out_dir / 'report_snippet.md'}")
    print(f"Inputs: {summary_csv} | {artifact_json}")


if __name__ == "__main__":
    main()
