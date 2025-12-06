#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.algorithms import bipartite, community


APP_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = APP_DIR.parent / "bigquery_outputs" / "mimic_48_1_10.csv"
OUT_DIR = APP_DIR / "mimic_figures"

FIG_DRUG_DISEASE = OUT_DIR / "drug_disease_network_mimic_48_1_10.png"
GRAPH_DRUG_DISEASE = OUT_DIR / "drug_disease_network_mimic_48_1_10.graphml"

FIG_DISEASE_BACKBONE = OUT_DIR / "disease_backbone_mimic_48_1_10.png"
GRAPH_DISEASE_BACKBONE = OUT_DIR / "disease_backbone_mimic_48_1_10.graphml"


def load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["drug_id"] = df["src"]
    df["disease_id"] = df["dst"]
    df["drug_name"] = df["drug_id"].str.replace(r"^drug:", "", regex=True)
    df["disease_name"] = df["icd_long_title"]
    return df


def build_bipartite_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, row in df.iterrows():
        d_id = row["drug_id"]
        dis_id = row["disease_id"]

        if d_id not in G:
            G.add_node(d_id, type="drug", label=row["drug_name"])
        if dis_id not in G:
            G.add_node(dis_id, type="disease", label=row["disease_name"])

        if G.has_edge(d_id, dis_id):
            G[d_id][dis_id]["weight_admissions"] += row["weight_admissions"]
            G[d_id][dis_id]["unique_patients"] += row["unique_patients"]
        else:
            G.add_edge(
                d_id,
                dis_id,
                weight_admissions=float(row["weight_admissions"]),
                unique_patients=float(row["unique_patients"]),
            )
    return G


def largest_component(H: nx.Graph) -> nx.Graph:
    if H.number_of_nodes() == 0:
        return H
    comp = max(nx.connected_components(H), key=len)
    return H.subgraph(comp).copy()


def draw_drug_disease_network(
    G: nx.Graph,
    out_png: Path,
    out_graphml: Path,
    top_k_drugs: int = 25,
    top_k_diseases: int = 40,
    min_edge_weight: float = 10.0,
) -> None:
    H = G.copy()
    H.remove_edges_from(
        [
            (u, v)
            for u, v, d in H.edges(data=True)
            if d.get("weight_admissions", 0.0) < min_edge_weight
        ]
    )
    H.remove_nodes_from([n for n, d in H.degree() if d == 0])
    H = largest_component(H)

    if H.number_of_nodes() == 0:
        print("Drug disease graph empty after filtering")
        return

    drugs = [n for n, d in H.nodes(data=True) if d.get("type") == "drug"]
    diseases = [n for n, d in H.nodes(data=True) if d.get("type") == "disease"]

    deg = dict(H.degree())
    top_drugs = sorted(drugs, key=lambda n: deg.get(n, 0), reverse=True)[:top_k_drugs]
    top_diseases = sorted(diseases, key=lambda n: deg.get(n, 0), reverse=True)[
        :top_k_diseases
    ]

    core_nodes = set(top_drugs) | set(top_diseases)
    for n in list(core_nodes):
        core_nodes.update(H.neighbors(n))

    H_core = H.subgraph(core_nodes).copy()
    H_core = largest_component(H_core)

    if H_core.number_of_nodes() == 0:
        print("Core drug disease graph empty")
        return

    drugs_core = [n for n, d in H_core.nodes(data=True) if d.get("type") == "drug"]
    diseases_core = [n for n, d in H_core.nodes(data=True) if d.get("type") == "disease"]

    pos = nx.spring_layout(H_core, seed=42, k=0.4, iterations=300)

    weights = [d.get("weight_admissions", 1.0) for _, _, d in H_core.edges(data=True)]
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

    fig, ax = plt.subplots(figsize=(8, 6))

    nx.draw_networkx_edges(
        H_core,
        pos,
        ax=ax,
        width=widths,
        alpha=0.2,
        edge_color="gray",
    )

    nx.draw_networkx_nodes(
        H_core,
        pos,
        nodelist=diseases_core,
        node_color="red",
        node_size=40,
        alpha=0.8,
        ax=ax,
        label="Disease",
    )
    nx.draw_networkx_nodes(
        H_core,
        pos,
        nodelist=drugs_core,
        node_color="blue",
        node_size=70,
        alpha=0.9,
        ax=ax,
        label="Drug",
    )

    deg_core = dict(H_core.degree())
    label_nodes = sorted(H_core.nodes(), key=lambda n: deg_core[n], reverse=True)[:15]
    labels = {n: H_core.nodes[n].get("label", n) for n in label_nodes}

    nx.draw_networkx_labels(H_core, pos, labels=labels, font_size=7, ax=ax)

    ax.set_axis_off()
    ax.set_title("Drug disease treatment network (MIMIC, 48h, primary dx)")
    ax.legend(scatterpoints=1, loc="upper right")
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    nx.write_graphml(H_core, out_graphml)
    print(f"Saved drug disease network to {out_png} and {out_graphml}")


def draw_disease_backbone(
    G: nx.Graph,
    out_png: Path,
    out_graphml: Path,
    min_shared_drugs: int = 3,
) -> None:
    diseases = [n for n, d in G.nodes(data=True) if d.get("type") == "disease"]
    if not diseases:
        print("No disease nodes found")
        return

    P = bipartite.weighted_projected_graph(G, diseases)

    P.remove_edges_from(
        [(u, v) for u, v, d in P.edges(data=True) if d.get("weight", 0) < min_shared_drugs]
    )
    P.remove_nodes_from([n for n, d in P.degree() if d == 0])

    P = largest_component(P)
    if P.number_of_nodes() == 0:
        print("Disease projection empty after filtering")
        return

    B = nx.maximum_spanning_tree(P, weight="weight")
    B = largest_component(B)

    comms = list(community.greedy_modularity_communities(B))
    comm_map = {}
    for i, c in enumerate(comms):
        for n in c:
            comm_map[n] = i

    num_comms = max(comm_map.values()) + 1 if comm_map else 1
    cmap = plt.cm.get_cmap("tab10", num_comms)
    node_colors = [cmap(comm_map.get(n, 0)) for n in B.nodes()]

    deg = dict(B.degree())
    node_sizes = [60 + 20 * deg[n] for n in B.nodes()]

    pos = nx.spring_layout(B, seed=42, k=0.5, iterations=400)

    fig, ax = plt.subplots(figsize=(8, 6))

    nx.draw_networkx_edges(
        B,
        pos,
        ax=ax,
        width=1.0,
        alpha=0.4,
        edge_color="gray",
    )
    nx.draw_networkx_nodes(
        B,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
    )

    sorted_nodes = sorted(B.nodes(), key=lambda n: deg[n], reverse=True)
    label_nodes = set(sorted_nodes[:20])
    labels = {
        n: G.nodes[n].get("label", n)
        for n in label_nodes
        if n in G.nodes
    }
    nx.draw_networkx_labels(B, pos, labels=labels, font_size=7, ax=ax)

    ax.set_axis_off()
    ax.set_title(
        "Disease backbone based on shared treatments (MIMIC, 48h, primary dx)"
    )
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    nx.write_graphml(B, out_graphml)
    print(f"Saved disease backbone to {out_png} and {out_graphml}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to mimic_48_1_10.csv",
    )
    parser.add_argument(
        "--min-edge-weight",
        type=float,
        default=10.0,
        help="Minimum admissions per edge for the bipartite plot",
    )
    parser.add_argument(
        "--min-shared-drugs",
        type=int,
        default=3,
        help="Minimum shared drugs for disease backbone edges",
    )
    args = parser.parse_args()

    df = load_edges(args.input)
    G = build_bipartite_graph(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    draw_drug_disease_network(
        G,
        FIG_DRUG_DISEASE,
        GRAPH_DRUG_DISEASE,
        min_edge_weight=args.min_edge_weight,
    )

    draw_disease_backbone(
        G,
        FIG_DISEASE_BACKBONE,
        GRAPH_DISEASE_BACKBONE,
        min_shared_drugs=args.min_shared_drugs,
    )


if __name__ == "__main__":
    main()
