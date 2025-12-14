#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("comparisons/ctgov_vs_pubtator_networks_v2")
TABLE_DIR = OUT_DIR / "tables"


def ensure_table_dir():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def pretty_disease_label(s: str) -> str:
    s = str(s)
    if s.startswith("CT_UNMAPPED::"):
        return s.split("::", 1)[1]
    if s.startswith("@DISEASE_"):
        return s.split("@DISEASE_", 1)[1].replace("_", " ")
    return s


def write_tsv(df: pd.DataFrame, name: str):
    ensure_table_dir()
    path = TABLE_DIR / name
    df.to_csv(path, sep="\t", index=False)
    print(f"Wrote {path}")


def safe_read_csv(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"{label} file not found, skipping.")
        return None
    if path.stat().st_size == 0:
        print(f"{label} file is empty, skipping.")
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"{label} file has no parsable columns, skipping.")
        return None
    if df.empty:
        print(f"{label} file has zero rows after parsing, skipping.")
        return None
    return df


def build_per_drug_alignment_tables():
    path = OUT_DIR / "per_drug_disease_alignment.csv"
    if not path.exists():
        print("per_drug_disease_alignment.csv not found, skipping alignment tables.")
        return

    df = pd.read_csv(path)
    df = df[df["n_union_diseases"] > 0].copy()
    if df.empty:
        print("No rows with non empty union in alignment table.")
        return

    df["jaccard_rounded"] = df["jaccard_disease_alignment"].round(4)

    df["drug_label"] = df["example_ct_drug"].where(
        df["example_ct_drug"].notna() & (df["example_ct_drug"].astype(str) != ""),
        df["example_pubtator_drug"],
    ).fillna("")

    cols = [
        "drug_label",
        "example_ct_drug",
        "example_pubtator_drug",
        "n_ct_diseases",
        "n_pubtator_diseases",
        "n_shared_diseases",
        "n_union_diseases",
        "jaccard_rounded",
    ]

    top_aligned = df.sort_values(
        ["jaccard_disease_alignment", "n_shared_diseases"],
        ascending=[False, False],
    ).head(20)[cols]
    write_tsv(top_aligned, "per_drug_alignment_top20_high_jaccard.txt")

    df_div = df[(df["n_ct_diseases"] > 0) & (df["n_pubtator_diseases"] > 0)].copy()
    top_divergent = df_div.sort_values(
        ["jaccard_disease_alignment", "n_union_diseases"],
        ascending=[True, False],
    ).head(20)[cols]
    write_tsv(top_divergent, "per_drug_alignment_top20_low_jaccard.txt")


def build_network_comparison_table():
    summ_path = OUT_DIR / "network_comparison_summary.json"
    if not summ_path.exists():
        print("network_comparison_summary.json not found, skipping network table.")
        return

    data = json.loads(summ_path.read_text(encoding="utf-8"))
    ct = data["graphs"]["ct_graph"]
    pt = data["graphs"]["pubtator_graph"]

    rows = []

    def add_metric(name, key, subkey=None):
        if subkey is None:
            ct_val = ct.get(key)
            pt_val = pt.get(key)
        else:
            ct_val = ct.get(key, {}).get(subkey)
            pt_val = pt.get(key, {}).get(subkey)
        rows.append({"metric": name, "ct_value": ct_val, "pubtator_value": pt_val})

    add_metric("number_of_nodes", "n_nodes")
    add_metric("number_of_edges", "n_edges")
    add_metric("number_of_drug_nodes", "n_drugs")
    add_metric("number_of_disease_nodes", "n_diseases")
    add_metric("density_undirected", "density_undirected")
    add_metric("number_of_components", "n_components")
    add_metric("largest_component_size", "largest_component_size")
    add_metric("largest_component_fraction", "largest_component_fraction")
    add_metric("largest_component_avg_path_length", "largest_component_avg_path_length")
    add_metric("largest_component_diameter", "largest_component_diameter")
    add_metric("degree_min", "degree_stats", "min")
    add_metric("degree_mean", "degree_stats", "mean")
    add_metric("degree_max", "degree_stats", "max")
    add_metric("weighted_degree_min", "weighted_degree_stats", "min")
    add_metric("weighted_degree_mean", "weighted_degree_stats", "mean")
    add_metric("weighted_degree_max", "weighted_degree_stats", "max")

    df = pd.DataFrame(rows)
    for col in ["ct_value", "pubtator_value"]:
        df[col] = df[col].apply(
            lambda x: round(x, 4) if isinstance(x, (float, int)) else x
        )

    write_tsv(df, "network_comparison_table.txt")


def build_disease_shared_drug_tables():
    ct_pairs_path = OUT_DIR / "ct_disease_shared_drugs_top_pairs.csv"
    pt_pairs_path = OUT_DIR / "pubtator_disease_shared_drugs_top_pairs.csv"

    # CT disease shared drugs with human readable disease and drug names
    ct_pairs = safe_read_csv(ct_pairs_path, "CT disease shared drugs")
    ct_edges = safe_read_csv(OUT_DIR / "ct_edges_canonical.csv", "CT canonical edges")

    if ct_pairs is not None and ct_edges is not None:
        ct_pairs = ct_pairs.head(20).copy()

        # build mapping from disease_id -> set of drug names
        edges = ct_edges[["disease_concept_id", "drug"]].dropna().copy()
        edges["disease_concept_id"] = edges["disease_concept_id"].astype(str)
        edges["drug"] = edges["drug"].astype(str)

        dis_to_drugs = (
            edges.groupby("disease_concept_id")["drug"].apply(lambda s: set(s)).to_dict()
        )

        human_rows = []
        for _, row in ct_pairs.iterrows():
            dis_a = str(row["disease_a"])
            dis_b = str(row["disease_b"])
            shared_count = int(row["shared_drugs"])

            drugs_a = dis_to_drugs.get(dis_a, set())
            drugs_b = dis_to_drugs.get(dis_b, set())
            shared_names = sorted(drugs_a & drugs_b)

            shared_str = "; ".join(shared_names)

            human_rows.append(
                {
                    "disease_a_label": pretty_disease_label(dis_a),
                    "disease_b_label": pretty_disease_label(dis_b),
                    "disease_a_id": dis_a,
                    "disease_b_id": dis_b,
                    "shared_drugs_count": shared_count,
                    "shared_drug_names": shared_str,
                }
            )

        ct_out = pd.DataFrame(human_rows)
        write_tsv(ct_out, "ct_disease_shared_drugs_top20.txt")
    else:
        print("Skipping CT disease shared drug table (missing pairs or edges).")

    # PubTator disease shared drugs (keep structure, may or may not have data)
    pt_pairs = safe_read_csv(pt_pairs_path, "PubTator disease shared drugs")
    pt_edges = safe_read_csv(
        OUT_DIR / "pubtator_edges_canonical_used.csv", "PubTator canonical edges"
    )

    if pt_pairs is not None and pt_edges is not None:
        pt_pairs = pt_pairs.head(20).copy()

        edges = pt_edges[["disease_concept_id", "drug"]].dropna().copy()
        edges["disease_concept_id"] = edges["disease_concept_id"].astype(str)
        edges["drug"] = edges["drug"].astype(str)

        dis_to_drugs = (
            edges.groupby("disease_concept_id")["drug"].apply(lambda s: set(s)).to_dict()
        )

        human_rows = []
        for _, row in pt_pairs.iterrows():
            dis_a = str(row["disease_a"])
            dis_b = str(row["disease_b"])
            shared_count = int(row["shared_drugs"])

            drugs_a = dis_to_drugs.get(dis_a, set())
            drugs_b = dis_to_drugs.get(dis_b, set())
            shared_names = sorted(drugs_a & drugs_b)

            shared_str = "; ".join(shared_names)

            human_rows.append(
                {
                    "disease_a_label": pretty_disease_label(dis_a),
                    "disease_b_label": pretty_disease_label(dis_b),
                    "disease_a_id": dis_a,
                    "disease_b_id": dis_b,
                    "shared_drugs_count": shared_count,
                    "shared_drug_names": shared_str,
                }
            )

        pt_out = pd.DataFrame(human_rows)
        write_tsv(pt_out, "pubtator_disease_shared_drugs_top20.txt")
    else:
        print("Skipping PubTator disease shared drug table (missing pairs or edges).")


def main():
    ensure_table_dir()
    build_per_drug_alignment_tables()
    build_network_comparison_table()
    build_disease_shared_drug_tables()
    print(f"Tables written under: {TABLE_DIR}")


if __name__ == "__main__":
    main()
