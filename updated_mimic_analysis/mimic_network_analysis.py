#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

APP_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = APP_DIR.parent / "bigquery_outputs" / "mimic_48_1_10.csv"
OUT_DIR = APP_DIR / "mimic_figures"


def load_mimic_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["drug_id"] = df["src"]
    df["disease_id"] = df["dst"]
    df["drug_label"] = df["drug_id"].str.replace(r"^drug:", "", regex=True)
    if "icd_long_title" in df.columns:
        df["disease_label"] = df["icd_long_title"]
    else:
        df["disease_label"] = df["disease_id"]
    if "weight_admissions" in df.columns:
        df["weight"] = df["weight_admissions"].astype(float)
    else:
        df["weight"] = 1.0
    return df[
        [
            "drug_id",
            "disease_id",
            "drug_label",
            "disease_label",
            "weight",
        ]
    ]


def build_bipartite_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in df.iterrows():
        d = row["drug_id"]
        dis = row["disease_id"]
        if d not in G:
            G.add_node(d, type="drug", label=row["drug_label"])
        if dis not in G:
            G.add_node(dis, type="disease", label=row["disease_label"])
        w = float(row["weight"])
        if G.has_edge(d, dis):
            G[d][dis]["weight"] += w
        else:
            G.add_edge(d, dis, weight=w)
    return G


def degree_summaries(df: pd.DataFrame):
    drug_deg = (
        df.groupby(["drug_id", "drug_label"])["disease_id"]
        .nunique()
        .reset_index(name="n_diseases")
    )
    disease_deg = (
        df.groupby(["disease_id", "disease_label"])["drug_id"]
        .nunique()
        .reset_index(name="n_drugs")
    )
    return drug_deg, disease_deg


def quantiles(s: pd.Series):
    if s.empty:
        return dict(mean=np.nan, median=np.nan, p90=np.nan, max=np.nan)
    return dict(
        mean=float(s.mean()),
        median=float(s.quantile(0.5)),
        p90=float(s.quantile(0.9)),
        max=int(s.max()),
    )


def component_stats(G: nx.Graph):
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes == 0:
        return dict(
            n_nodes=0,
            n_edges=0,
            n_components=0,
            lcc_size=0,
            lcc_frac_nodes=0.0,
            lcc_edges=0,
            avg_path_len=np.nan,
            diameter=np.nan,
        )
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    n_components = len(comps)
    lcc_nodes = comps[0]
    G_lcc = G.subgraph(lcc_nodes).copy()
    lcc_size = G_lcc.number_of_nodes()
    lcc_edges = G_lcc.number_of_edges()
    lcc_frac = lcc_size / n_nodes if n_nodes else 0.0
    try:
        avg_path = nx.average_shortest_path_length(G_lcc)
    except Exception:
        avg_path = np.nan
    try:
        diam = nx.diameter(G_lcc)
    except Exception:
        diam = np.nan
    return dict(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_components=n_components,
        lcc_size=lcc_size,
        lcc_frac_nodes=lcc_frac,
        lcc_edges=lcc_edges,
        avg_path_len=float(avg_path) if avg_path == avg_path else np.nan,
        diameter=float(diam) if diam == diam else np.nan,
    )


def degree_bins(series: pd.Series):
    if series.empty:
        return dict(deg1=0, deg2_5=0, deg_gt5=0)
    deg1 = int((series == 1).sum())
    deg2_5 = int(((series >= 2) & (series <= 5)).sum())
    deg_gt5 = int((series > 5).sum())
    return dict(deg1=deg1, deg2_5=deg2_5, deg_gt5=deg_gt5)


def print_summary(
    df: pd.DataFrame,
    drug_deg: pd.DataFrame,
    disease_deg: pd.DataFrame,
    comp: dict,
):
    n_drugs = drug_deg.shape[0]
    n_diseases = disease_deg.shape[0]
    n_edges = df[["drug_id", "disease_id"]].drop_duplicates().shape[0]
    density = n_edges / (n_drugs * n_diseases) if n_drugs and n_diseases else 0.0

    drug_stats = quantiles(drug_deg["n_diseases"])
    dis_stats = quantiles(disease_deg["n_drugs"])

    drug_bins = degree_bins(drug_deg["n_diseases"])
    dis_bins = degree_bins(disease_deg["n_drugs"])

    top_drugs = (
        drug_deg.sort_values("n_diseases", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    top_diseases = (
        disease_deg.sort_values("n_drugs", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    print("")
    print("MIMIC drug-disease treatment network summary")
    print("============================================")
    print(
        f"Nodes: {n_drugs} drugs and {n_diseases} diseases "
        f"(total {comp['n_nodes']} nodes)."
    )
    print(
        f"Edges: {n_edges} unique drug-disease pairs "
        f"(network density {density:.6f})."
    )
    print("")
    print("Degree statistics for drugs (number of diseases per drug):")
    print(
        f"  median {drug_stats['median']:.2f}, "
        f"mean {drug_stats['mean']:.2f}, "
        f"90th percentile {drug_stats['p90']:.2f}, "
        f"max {drug_stats['max']}."
    )
    print(
        f"  degree 1: {drug_bins['deg1']}, "
        f"degree 2-5: {drug_bins['deg2_5']}, "
        f"degree >5: {drug_bins['deg_gt5']}."
    )
    print("")
    print("Degree statistics for diseases (number of drugs per disease):")
    print(
        f"  median {dis_stats['median']:.2f}, "
        f"mean {dis_stats['mean']:.2f}, "
        f"90th percentile {dis_stats['p90']:.2f}, "
        f"max {dis_stats['max']}."
    )
    print(
        f"  degree 1: {dis_bins['deg1']}, "
        f"degree 2-5: {dis_bins['deg2_5']}, "
        f"degree >5: {dis_bins['deg_gt5']}."
    )
    print("")
    print("Connected component structure:")
    print(
        f"  {comp['n_components']} connected components; "
        f"largest component has {comp['lcc_size']} nodes "
        f"({comp['lcc_frac_nodes']*100:.1f}% of all nodes) "
        f"and {comp['lcc_edges']} edges."
    )
    if comp["avg_path_len"] == comp["avg_path_len"]:
        print(
            f"  average shortest path length in largest component "
            f"{comp['avg_path_len']:.3f}, diameter {int(comp['diameter'])}."
        )
    else:
        print("  average shortest path length and diameter could not be computed.")
    print("")
    print("Top 5 drugs by number of linked diseases:")
    for _, row in top_drugs.iterrows():
        print(
            f"  {row['drug_label']} ({row['drug_id']}): "
            f"{row['n_diseases']} diseases."
        )
    print("")
    print("Top 5 diseases by number of linked drugs:")
    for _, row in top_diseases.iterrows():
        print(
            f"  {row['disease_label']} ({row['disease_id']}): "
            f"{row['n_drugs']} drugs."
        )
    print("")
    print("CSV outputs written under mimic_figures/ for degrees and basic stats.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to mimic_48_1_10.csv",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_mimic_edges(args.input)
    drug_deg, disease_deg = degree_summaries(df)
    G = build_bipartite_graph(df)
    comp = component_stats(G)

    # save basic tables
    drug_deg.to_csv(OUT_DIR / "mimic_drug_degrees.csv", index=False)
    disease_deg.to_csv(OUT_DIR / "mimic_disease_degrees.csv", index=False)

    stats_row = {
        "n_drugs": drug_deg.shape[0],
        "n_diseases": disease_deg.shape[0],
        "n_edges": df[["drug_id", "disease_id"]].drop_duplicates().shape[0],
        "density": (
            df[["drug_id", "disease_id"]].drop_duplicates().shape[0]
            / (drug_deg.shape[0] * disease_deg.shape[0])
            if drug_deg.shape[0] and disease_deg.shape[0]
            else np.nan
        ),
        **comp,
    }
    pd.DataFrame([stats_row]).to_csv(
        OUT_DIR / "mimic_network_basic_stats.csv", index=False
    )

    print_summary(df, drug_deg, disease_deg, comp)


if __name__ == "__main__":
    main()
