#!/usr/bin/env python3
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
INPUT_PATH = APP_DIR.parent / "bigquery_outputs" / "mimic_48_1_10.csv"

OUT_DIR = APP_DIR / "mimic_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDGES_CSV = OUT_DIR / "drug_drug_edges_mimic_48_1_10.csv"
NODES_CSV = OUT_DIR / "drug_drug_nodes_mimic_48_1_10.csv"
FIG_PNG = OUT_DIR / "drug_drug_co_treatment_top20_mimic_48_1_10.png"


def load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["drug_id"] = df["src"]
    df["disease_id"] = df["dst"]
    df["drug_name"] = df["drug_id"].str.replace(r"^drug:", "", regex=True)
    df["disease_name"] = df["icd_long_title"]
    return df


def build_drug_drug_network(df: pd.DataFrame) -> nx.Graph:
    total_by_drug = (
        df.groupby("drug_id")["weight_admissions"]
        .sum()
        .to_dict()
    )

    edges = {}
    drug_names = {}

    for _, row in df[["drug_id", "drug_name"]].drop_duplicates().iterrows():
        drug_names[row["drug_id"]] = row["drug_name"]

    grouped = df.groupby(["disease_id", "disease_name"])

    for (_, _), g in grouped:
        if len(g) < 2:
            continue
        rows = g[["drug_id", "drug_name", "weight_admissions"]].values
        for (d1, n1, w1), (d2, n2, w2) in itertools.combinations(rows, 2):
            if d1 == d2:
                continue
            key = tuple(sorted([d1, d2]))
            contrib_min = float(min(w1, w2))
            if key not in edges:
                edges[key] = {"sum_min": 0.0, "shared_diseases": 0}
            edges[key]["sum_min"] += contrib_min
            edges[key]["shared_diseases"] += 1

    G = nx.Graph()
    for d_id, name in drug_names.items():
        G.add_node(d_id, label=name)

    for (d1, d2), info in edges.items():
        sum_min = info["sum_min"]
        total_i = float(total_by_drug.get(d1, 0.0))
        total_j = float(total_by_drug.get(d2, 0.0))
        denom = total_i + total_j - sum_min
        w_jaccard = sum_min / denom if denom > 0 else 0.0

        G.add_edge(
            d1,
            d2,
            weight_admissions=sum_min,
            shared_diseases=info["shared_diseases"],
            weighted_jaccard=w_jaccard,
        )

    return G


def summarize_network(G: nx.Graph):
    print(f"Total drugs in network: {G.number_of_nodes()}")
    print(f"Total drug-drug edges: {G.number_of_edges()}")

    strengths = {}
    for n in G.nodes():
        strengths[n] = sum(
            d.get("weight_admissions", 0.0)
            for _, _, d in G.edges(n, data=True)
        )

    summary_rows = []
    for n in G.nodes():
        summary_rows.append(
            {
                "drug_id": n,
                "drug_name": G.nodes[n].get("label", n),
                "degree": G.degree(n),
                "strength_admissions": strengths[n],
            }
        )
    df_nodes = pd.DataFrame(summary_rows)

    edge_rows = []
    for u, v, d in G.edges(data=True):
        edge_rows.append(
            {
                "drug1_id": u,
                "drug1_name": G.nodes[u].get("label", u),
                "drug2_id": v,
                "drug2_name": G.nodes[v].get("label", v),
                "weight_admissions": d.get("weight_admissions", 0.0),
                "shared_diseases": d.get("shared_diseases", 0),
                "weighted_jaccard": d.get("weighted_jaccard", 0.0),
            }
        )
    df_edges = pd.DataFrame(edge_rows)

    df_nodes.to_csv(NODES_CSV, index=False)
    df_edges.to_csv(EDGES_CSV, index=False)
    print(f"Saved node table to {NODES_CSV}")
    print(f"Saved edge table to {EDGES_CSV}")

    df_nodes_sorted = df_nodes.sort_values(
        ["strength_admissions", "degree"], ascending=[False, False]
    )

    print("\nTop 10 drugs by co-treatment strength:")
    for _, row in df_nodes_sorted.head(10).iterrows():
        print(
            f"  {row['drug_name']}: degree={row['degree']}, "
            f"strength={row['strength_admissions']:.1f}"
        )

    df_edges_by_weight = df_edges.sort_values(
        "weight_admissions", ascending=False
    )

    print("\nTop 10 drug-drug edges by admissions-weighted co-treatment:")
    for _, row in df_edges_by_weight.head(10).iterrows():
        print(
            f"  {row['drug1_name']} - {row['drug2_name']}: "
            f"weight_admissions={row['weight_admissions']:.1f}, "
            f"shared_diseases={row['shared_diseases']}, "
            f"weighted_jaccard={row['weighted_jaccard']:.3f}"
        )

    min_shared = 50
    df_edges_jaccard = (
        df_edges[df_edges["shared_diseases"] >= min_shared]
        .sort_values("weighted_jaccard", ascending=False)
    )

    print(
        f"\nTop 10 drug-drug edges by weighted Jaccard "
        f"(shared_diseases >= {min_shared}):"
    )
    for _, row in df_edges_jaccard.head(10).iterrows():
        print(
            f"  {row['drug1_name']} - {row['drug2_name']}: "
            f"Jw={row['weighted_jaccard']:.3f}, "
            f"weight_admissions={row['weight_admissions']:.1f}, "
            f"shared_diseases={row['shared_diseases']}"
        )

    return df_nodes_sorted, df_edges_by_weight


def plot_top20_network(G: nx.Graph, df_nodes_sorted: pd.DataFrame, top_k: int = 20):
    top_drugs = list(df_nodes_sorted.head(top_k)["drug_id"])
    H = G.subgraph(top_drugs).copy()

    if H.number_of_nodes() == 0:
        print("No nodes for top-k plot.")
        return

    pos = nx.spring_layout(H, seed=42, k=0.4, iterations=300)

    weights = [d.get("weight_admissions", 1.0) for _, _, d in H.edges(data=True)]
    if weights:
        w_min, w_max = min(weights), max(weights)
        if w_max == w_min:
            widths = [1.5 for _ in weights]
        else:
            widths = [
                0.3 + 3.0 * (w - w_min) / (w_max - w_min) for w in weights
            ]
    else:
        widths = []

    strengths = {
        n: sum(
            d.get("weight_admissions", 0.0)
            for _, _, d in H.edges(n, data=True)
        )
        for n in H.nodes()
    }
    s_vals = np.array(list(strengths.values()))
    if len(s_vals) > 0:
        s_min, s_max = float(s_vals.min()), float(s_vals.max())
        if s_max == s_min:
            sizes = [150 for _ in H.nodes()]
        else:
            sizes = [
                150 + 250 * (strengths[n] - s_min) / (s_max - s_min)
                for n in H.nodes()
            ]
    else:
        sizes = [150 for _ in H.nodes()]

    fig, ax = plt.subplots(figsize=(10, 4))

    nx.draw_networkx_edges(
        H,
        pos,
        ax=ax,
        width=widths,
        alpha=0.25,
        edge_color="gray",
    )

    nx.draw_networkx_nodes(
        H,
        pos,
        ax=ax,
        node_color="#f58634",
        node_size=sizes,
        alpha=0.9,
    )

    labels = {n: H.nodes[n].get("label", n) for n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8, ax=ax)

    ax.set_axis_off()
    ax.set_title("Drug drug co-treatment network (MIMIC, top 20 drugs)")
    plt.tight_layout()
    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved top-20 drug co-treatment figure to {FIG_PNG}")


def main():
    df = load_edges(INPUT_PATH)
    G = build_drug_drug_network(df)
    df_nodes_sorted, _ = summarize_network(G)
    plot_top20_network(G, df_nodes_sorted, top_k=20)


if __name__ == "__main__":
    main()
