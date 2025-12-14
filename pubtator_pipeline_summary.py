#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

OUTDIR = Path("outputs")
RUN_ID = "20251120_191408"
RUN_DIR = OUTDIR / f"pipeline_{RUN_ID}"


def load_run():
    if not RUN_DIR.exists():
        raise SystemExit(f"Run directory not found: {RUN_DIR}")

    json_path = RUN_DIR / f"pipeline_{RUN_ID}.json"
    summary_csv = RUN_DIR / f"pipeline_{RUN_ID}_summary.csv"
    stage3_csv = RUN_DIR / f"pipeline_{RUN_ID}_relations_stage3.csv"

    for p in [json_path, summary_csv, stage3_csv]:
        if not p.exists():
            raise SystemExit(f"Expected file not found: {p}")

    with open(json_path, "r", encoding="utf-8") as f:
        art = json.load(f)

    df_summary = pd.read_csv(summary_csv)
    df_stage3 = pd.read_csv(stage3_csv)

    return RUN_ID, art, df_summary, df_stage3


def extract_pmids_and_years(artifact):
    pmids = []
    years = []
    for ind in artifact.get("indications", []):
        for ev in ind.get("evidence", []):
            for art in ev.get("articles", []):
                pm = str(art.get("pmid") or "").strip()
                if not pm:
                    continue
                pmids.append(pm)
                dt = art.get("date")
                if not dt:
                    continue
                try:
                    year = int(str(dt)[:4])
                    years.append(year)
                except Exception:
                    continue
    return pmids, years


def build_pmid_relation_stats(artifact, run_id: str, run_dir: Path) -> pd.DataFrame:
    """
    Build pmid-level relation statistics:

      - n_pairs: number of distinct (drug_id, disease_id) pairs for this pmid
      - n_drugs: number of distinct drugs
      - n_diseases: number of distinct diseases
      - has_one_drug_multi_diseases: at least one drug with >1 disease in this pmid
      - has_one_disease_multi_drugs: at least one disease with >1 drug in this pmid
    """
    pmid_to_pairs = defaultdict(set)

    for ind in artifact.get("indications", []):
        drug_id = ind.get("drug_id")
        for ev in ind.get("evidence", []):
            disease_id = ev.get("disease_id")
            pmids = [str(p) for p in ev.get("pmids", [])]
            for pm in pmids:
                if not pm:
                    continue
                pmid_to_pairs[pm].add((drug_id, disease_id))

    records = []
    for pmid, pairs in pmid_to_pairs.items():
        drugs = {d for d, _ in pairs}
        diseases = {s for _, s in pairs}

        # For this pmid, does any single drug connect to multiple diseases?
        has_one_drug_multi_diseases = False
        for d in drugs:
            dis_for_d = {s for (d2, s) in pairs if d2 == d}
            if len(dis_for_d) > 1:
                has_one_drug_multi_diseases = True
                break

        # For this pmid, does any single disease connect to multiple drugs?
        has_one_disease_multi_drugs = False
        for s in diseases:
            drugs_for_s = {d for (d, s2) in pairs if s2 == s}
            if len(drugs_for_s) > 1:
                has_one_disease_multi_drugs = True
                break

        records.append(
            {
                "pmid": pmid,
                "n_pairs": len(pairs),
                "n_drugs": len(drugs),
                "n_diseases": len(diseases),
                "has_one_drug_multi_diseases": has_one_drug_multi_diseases,
                "has_one_disease_multi_drugs": has_one_disease_multi_drugs,
            }
        )

    df_pmids = pd.DataFrame(records)

    out_csv = run_dir / f"pipeline_{run_id}_pmid_relations.csv"
    df_pmids.to_csv(out_csv, index=False)

    return df_pmids


def print_global_summary(run_id, art, df_summary, df_stage3, top_n: int = 10, top_pairs: int = 20):
    print(f"Run ID: {run_id}")
    print(f"Started at: {art.get('started_at')}")
    print()

    drugs_cfg = art.get("drugs", [])
    drug_entities = art.get("drug_entities", [])
    indications = art.get("indications", [])
    dropped_no_rel = art.get("dropped_no_relations", [])

    resolved_drug_names = {de["drug_name"] for de in drug_entities}
    unresolved = sorted(set(drugs_cfg) - resolved_drug_names)

    print("Configured drugs")
    print(f"  Total configured drugs: {len(drugs_cfg)}")
    print(f"  Drugs with at least one CHEMICAL ID: {len(resolved_drug_names)}")
    print(f"  Drugs with no CHEMICAL ID: {len(unresolved)}")
    if unresolved:
        print("  Unresolved drug names (first 10):")
        for name in unresolved[:10]:
            print(f"    - {name}")
    print()

    print("Stage 3 relations (drug to disease)")
    print(f"  Stage 3 indication entries: {len(indications)}")
    print(f"  Drug IDs with no treat relations: {len(dropped_no_rel)}")
    print()

    df_summary["pmid_count"] = df_summary["pmid_count"].fillna(0).astype(int)
    df_summary["total_articles"] = df_summary["total_articles"].fillna(0).astype(int)

    n_pairs = len(df_summary)
    n_drug_ids = df_summary["drug_id"].nunique()
    n_disease_ids = df_summary["disease_id"].nunique()
    n_with_pmids = (df_summary["pmid_count"] > 0).sum()
    n_zero_pmids = n_pairs - n_with_pmids
    total_pmid_occ = int(df_summary["pmid_count"].sum())

    all_pmids, years = extract_pmids_and_years(art)
    unique_pmids = len(set(all_pmids))

    print("Drug disease pairs with PubMed evidence")
    print(f"  Total drug disease pairs: {n_pairs}")
    print(f"  Unique drug IDs: {n_drug_ids}")
    print(f"  Unique disease IDs: {n_disease_ids}")
    print(f"  Pairs with at least one PMID: {n_with_pmids}")
    print(f"  Pairs with zero PMIDs: {n_zero_pmids}")
    print(f"  Total PMID occurrences across all pairs: {total_pmid_occ}")
    print(f"  Unique PMIDs across all pairs: {unique_pmids}")
    print()

    pmid_stats = df_summary["pmid_count"].describe()
    print("PMID count per drug disease pair")
    print(f"  Mean:   {pmid_stats['mean']:.2f}")
    print(f"  Median: {pmid_stats['50%']:.0f}")
    print(f"  Min:    {pmid_stats['min']:.0f}")
    print(f"  Max:    {pmid_stats['max']:.0f}")
    print()

    if years:
        year_counts = Counter(years)
        print("Publication years for retrieved articles")
        print(f"  Year range: {min(years)} to {max(years)}")
        top_years = sorted(year_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        print("  Top years by article count (up to 10):")
        for y, c in top_years:
            print(f"    {y}: {c} articles")
        print()
    else:
        print("No article dates available.")
        print()

    buckets = [
        ("pmid_count == 0", df_summary["pmid_count"] == 0),
        ("pmid_count == 1", df_summary["pmid_count"] == 1),
        ("pmid_count 2-4", df_summary["pmid_count"].between(2, 4)),
        ("pmid_count 5-9", df_summary["pmid_count"].between(5, 9)),
        ("pmid_count 10-49", df_summary["pmid_count"].between(10, 49)),
        ("pmid_count >= 50", df_summary["pmid_count"] >= 50),
    ]
    print("Drug disease pair buckets by PMID count")
    for label, mask in buckets:
        print(f"  {label}: {int(mask.sum())} pairs")
    print()

    print(f"Top {top_n} drugs by total PMIDs across all diseases")
    g_drug = (
        df_summary.groupby(["drug_name", "drug_id"])["pmid_count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    for (dname, did), n in g_drug.items():
        print(f"  {dname} ({did}): {int(n)} PMIDs")
    print()

    print(f"Top {top_n} diseases by total PMIDs across all drugs")
    g_dis = (
        df_summary.groupby(["disease_name", "disease_id"])["pmid_count"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    for (sname, sid), n in g_dis.items():
        print(f"  {sname} ({sid}): {int(n)} PMIDs")
    print()

    print(f"Top {top_pairs} individual drug disease pairs by pmid_count")
    df_pairs = df_summary.sort_values("pmid_count", ascending=False).head(top_pairs)
    for _, r in df_pairs.iterrows():
        print(
            f"  {r['drug_name']} ({r['drug_id']}) ~ "
            f"{r['disease_name']} ({r['disease_id']}): "
            f"{int(r['pmid_count'])} PMIDs, total_articles={int(r['total_articles'])}"
        )
    print()

    # -------- New: PMID-level relation multiplicity summary --------
    df_pmids = build_pmid_relation_stats(art, run_id, RUN_DIR)

    if not df_pmids.empty:
        total_pmid_rel = len(df_pmids)
        n_single_rel = int((df_pmids["n_pairs"] == 1).sum())
        n_multi_rel = int((df_pmids["n_pairs"] > 1).sum())

        n_one_drug_multi_dis = int(df_pmids["has_one_drug_multi_diseases"].sum())
        n_one_dis_multi_drugs = int(df_pmids["has_one_disease_multi_drugs"].sum())

        print("PMID-level treatment relation multiplicity")
        print(f"  PMIDs with at least one drug–disease relation: {total_pmid_rel}")
        print(f"  PMIDs with exactly one (drug,disease) pair:   {n_single_rel}")
        print(f"  PMIDs with >1 distinct (drug,disease) pairs:  {n_multi_rel}")
        print()
        print("  PMIDs where at least one drug links to multiple diseases:")
        print(f"    Count: {n_one_drug_multi_dis}")
        print("  PMIDs where at least one disease links to multiple drugs:")
        print(f"    Count: {n_one_dis_multi_drugs}")
        print()
        print(
            f"  Per-PMID relation CSV written to: "
            f"{(RUN_DIR / f'pipeline_{run_id}_pmid_relations.csv').name}"
        )
        print()
    else:
        print("No PMID-level relation data could be derived from the artifact.")
        print()


def query_by_drug(df_summary: pd.DataFrame, name: str):
    mask = df_summary["drug_name"].str.lower() == name.lower()
    df = df_summary[mask].copy()
    if df.empty:
        print(f"No rows found for drug_name == {name!r}")
        return
    df = df.sort_values("pmid_count", ascending=False)
    print(f"Drug level view for {name} (rows: {len(df)})")
    for _, row in df.iterrows():
        print(
            f"  {row['drug_name']} ({row['drug_id']}) -> "
            f"{row['disease_name']} ({row['disease_id']}): "
            f"{int(row['pmid_count'])} PMIDs, total_articles={int(row['total_articles'])}"
        )


def query_by_disease(df_summary: pd.DataFrame, name: str):
    mask = df_summary["disease_name"].str.lower() == name.lower()
    df = df_summary[mask].copy()
    if df.empty:
        print(f"No rows found for disease_name == {name!r}")
        return
    df = df.sort_values("pmid_count", ascending=False)
    print(f"Disease level view for {name} (rows: {len(df)})")
    for _, row in df.iterrows():
        print(
            f"  {row['disease_name']} ({row['disease_id']}) <- "
            f"{row['drug_name']} ({row['drug_id']}): "
            f"{int(row['pmid_count'])} PMIDs, total_articles={int(row['total_articles'])}"
        )


def main():
    ap = argparse.ArgumentParser(
        description="Summarize PubTator pipeline outputs and query by drug or disease."
    )
    ap.add_argument(
        "--drug",
        type=str,
        help="Filter view to a single drug_name (case insensitive).",
    )
    ap.add_argument(
        "--disease",
        type=str,
        help="Filter view to a single disease_name (case insensitive).",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=10,
        help="Top N drugs or diseases in global summary (default 10).",
    )
    ap.add_argument(
        "--top-pairs",
        type=int,
        default=20,
        help="Top N drug disease pairs in global summary (default 20).",
    )
    args = ap.parse_args()

    run_id, art, df_summary, df_stage3 = load_run()

    if not args.drug and not args.disease:
        print_global_summary(
            run_id,
            art,
            df_summary,
            df_stage3,
            top_n=args.top,
            top_pairs=args.top_pairs,
        )
    if args.drug:
        print()
        query_by_drug(df_summary, args.drug)
    if args.disease:
        print()
        query_by_disease(df_summary, args.disease)


if __name__ == "__main__":
    main()
