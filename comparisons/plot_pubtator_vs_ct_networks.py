#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = Path("comparisons/ctgov_vs_pubtator_networks_v2")
PLOT_DIR = OUT_DIR / "plots"


def ensure_plot_dir():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_hubs():
    ct_drugs = pd.read_csv(OUT_DIR / "ct_hubs_drugs.csv")
    pt_drugs = pd.read_csv(OUT_DIR / "pubtator_hubs_drugs.csv")
    ct_dis = pd.read_csv(OUT_DIR / "ct_hubs_diseases.csv")
    pt_dis = pd.read_csv(OUT_DIR / "pubtator_hubs_diseases.csv")
    return ct_drugs, pt_drugs, ct_dis, pt_dis


def pretty_disease_label(s: str) -> str:
    s = str(s)
    if s.startswith("CT_UNMAPPED::"):
        return s.split("::", 1)[1]
    if s.startswith("@DISEASE_"):
        return s.split("@DISEASE_", 1)[1].replace("_", " ")
    return s


def build_union_top(df_ct, df_pt, metric_col, name_col="node", topn=20):
    top_ct = df_ct.sort_values(metric_col, ascending=False).head(topn)
    top_pt = df_pt.sort_values(metric_col, ascending=False).head(topn)

    labels_ct = list(top_ct[name_col].astype(str))
    labels_pt = list(top_pt[name_col].astype(str))
    # preserve order, remove duplicates
    labels_union = list(dict.fromkeys(labels_ct + labels_pt))

    ct_vals = {r[name_col]: r[metric_col] for _, r in top_ct.iterrows()}
    pt_vals = {r[name_col]: r[metric_col] for _, r in top_pt.iterrows()}

    ct_series = np.array([ct_vals.get(lbl, 0.0) for lbl in labels_union], float)
    pt_series = np.array([pt_vals.get(lbl, 0.0) for lbl in labels_union], float)

    return labels_union, ct_series, pt_series


def plot_combined_bars(labels, ct_vals, pt_vals, title, ylabel, filename):
    # normalize within each source so shapes are comparable
    ct_norm = ct_vals.copy()
    pt_norm = pt_vals.copy()
    if ct_norm.max() > 0:
        ct_norm = ct_norm / ct_norm.max()
    if pt_norm.max() > 0:
        pt_norm = pt_norm / pt_norm.max()

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.4), 6))

    ax.bar(x - width / 2, ct_norm, width, label="ClinicalTrials.gov", color="blue")
    ax.bar(x + width / 2, pt_norm, width, label="PubTator", color="red")

    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel + " (normalized per source)", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)

    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_drugs_and_diseases():
    ct_drugs, pt_drugs, ct_dis, pt_dis = load_hubs()

    # readable labels for drugs
    ct_drugs["label"] = ct_drugs["node"].astype(str)
    pt_drugs["label"] = pt_drugs["node"].astype(str)

    # readable labels for diseases, stripping prefixes and underscores
    ct_dis["label"] = ct_dis["node"].apply(pretty_disease_label)
    pt_dis["label"] = pt_dis["node"].apply(pretty_disease_label)

    # 1) Top drugs by number of diseases (degree)
    labels, ct_vals, pt_vals = build_union_top(
        ct_drugs, pt_drugs, metric_col="degree", name_col="label", topn=20
    )
    plot_combined_bars(
        labels,
        ct_vals,
        pt_vals,
        title="Top 20 drugs per dataset by number of associated diseases",
        ylabel="Number of diseases (degree)",
        filename="top_drugs_by_degree_combined_normalized.png",
    )

    # 2) Top drugs by weighted degree
    labels, ct_vals, pt_vals = build_union_top(
        ct_drugs, pt_drugs, metric_col="weighted_degree", name_col="label", topn=20
    )
    plot_combined_bars(
        labels,
        ct_vals,
        pt_vals,
        title="Top 20 drugs per dataset by weighted degree",
        ylabel="Weighted degree (trials or articles)",
        filename="top_drugs_by_weighted_degree_combined_normalized.png",
    )

    # 3) Top diseases by number of drugs (degree)
    labels, ct_vals, pt_vals = build_union_top(
        ct_dis, pt_dis, metric_col="degree", name_col="label", topn=20
    )
    plot_combined_bars(
        labels,
        ct_vals,
        pt_vals,
        title="Top 20 diseases per dataset by number of associated drugs",
        ylabel="Number of drugs (degree)",
        filename="top_diseases_by_degree_combined_normalized.png",
    )

    # 4) Top diseases by weighted degree
    labels, ct_vals, pt_vals = build_union_top(
        ct_dis, pt_dis, metric_col="weighted_degree", name_col="label", topn=20
    )
    plot_combined_bars(
        labels,
        ct_vals,
        pt_vals,
        title="Top 20 diseases per dataset by weighted degree",
        ylabel="Weighted degree (trials or articles)",
        filename="top_diseases_by_weighted_degree_combined_normalized.png",
    )


def plot_jaccard_alignment():
    path = OUT_DIR / "per_drug_disease_alignment.csv"
    if not path.exists():
        return

    df = pd.read_csv(path).copy()
    df = df[df["n_union_diseases"] > 0].copy()
    if df.empty:
        return

    # Histogram of Jaccard similarity
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["jaccard_disease_alignment"], bins=50, color="steelblue")
    ax.set_title("Distribution of Jaccard alignment across drugs", fontsize=14)
    ax.set_xlabel("Jaccard similarity of CT vs PubTator disease sets", fontsize=12)
    ax.set_ylabel("Number of drugs", fontsize=12)
    plt.tight_layout()
    fig.savefig(
        PLOT_DIR / "jaccard_alignment_histogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Scatter: CT diseases vs PubTator diseases, colored by Jaccard
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        df["n_ct_diseases"],
        df["n_pubtator_diseases"],
        c=df["jaccard_disease_alignment"],
        cmap="viridis",
        alpha=0.8,
    )

    ax.set_title("CT vs PubTator disease counts per drug", fontsize=14)
    ax.set_xlabel("Number of CT diseases per drug", fontsize=12)
    ax.set_ylabel("Number of PubTator diseases per drug", fontsize=12)

    cb = plt.colorbar(sc)
    cb.set_label("Jaccard similarity", fontsize=12)

    plt.tight_layout()
    fig.savefig(
        PLOT_DIR / "jaccard_alignment_scatter.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def build_network_comparison_table():
    summ_path = OUT_DIR / "network_comparison_summary.json"
    if not summ_path.exists():
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
    df.to_csv(OUT_DIR / "network_comparison_table.csv", index=False)
    print(df.to_string(index=False))


def main():
    ensure_plot_dir()
    plot_top_drugs_and_diseases()
    plot_jaccard_alignment()
    build_network_comparison_table()
    print(f"Plots written to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
