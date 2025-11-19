#!/usr/bin/env python3
import argparse, pathlib
from collections import defaultdict, Counter

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np

def load_graph(run_dir: pathlib.Path):
    nodes = pd.read_csv(run_dir / "nodes.csv")
    edges = pd.read_csv(run_dir / "edges.csv")
    edges["key"] = list(zip(edges["node_u"], edges["node_v"]))
    edges = edges.groupby("key", as_index=False).agg({
        "node_u":"first","node_v":"first","relation":"first",
        "pmid_count":"sum","total_articles":"sum","pmids":"first","chem_ids":"first"
    })
    return nodes, edges

def top15_diseases_by_total_pubs(nodes, edges):
    type_map = dict(zip(nodes["id"], nodes["type"]))
    pubs = defaultdict(int)
    for _, r in edges.iterrows():
        u, v = r["node_u"], r["node_v"]
        w = int(r.get("total_articles", 0))
        if type_map.get(u) == "disease": pubs[u] += w
        if type_map.get(v) == "disease": pubs[v] += w
    df = pd.DataFrame([(k, v) for k, v in pubs.items()], columns=["disease_id","total_articles"])
    df["label"] = df["disease_id"].apply(lambda s: s.split("_",1)[1].replace("_"," ") if isinstance(s,str) and s.startswith("@") else s)
    return df.sort_values("total_articles", ascending=False).head(15)

def top15_drugs_by_pubs(nodes, edges):
    drug_ids = set(nodes.loc[nodes["type"]=="drug","id"])
    pubsum = defaultdict(int)
    for _, r in edges.iterrows():
        u, v = r["node_u"], r["node_v"]
        w = int(r.get("total_articles", 0))
        if u in drug_ids: pubsum[u] += w
        if v in drug_ids: pubsum[v] += w
    df = pd.DataFrame([(k, v) for k, v in pubsum.items()], columns=["drug_name","total_articles"])
    df["label"] = df["drug_name"]
    return df.sort_values("total_articles", ascending=False).head(15)

def scatter_drug_degree_vs_pubs(nodes, edges):
    drug_ids = set(nodes.loc[nodes["type"]=="drug","id"])
    deg = Counter(); pubs = defaultdict(int)
    for _, r in edges.iterrows():
        u, v = r["node_u"], r["node_v"]
        w = int(r.get("total_articles", 0))
        if u in drug_ids: deg[u]+=1; pubs[u]+=w
        if v in drug_ids: deg[v]+=1; pubs[v]+=w
    df = pd.DataFrame([(d, deg[d], pubs[d]) for d in drug_ids], columns=["drug_name","unique_diseases","total_articles"])
    return df

def draw_network(nodes, edges, outpath, spread=1.8, seed=42):
    G = nx.Graph()
    for _, r in nodes.iterrows():
        G.add_node(r["id"], label=(r["label"] if "label" in nodes.columns else r["id"]), type=r["type"])
    for _, r in edges.iterrows():
        u, v = r["node_u"], r["node_v"]
        w = int(r.get("total_articles", 1)) if not pd.isna(r.get("total_articles", None)) else 1
        if u in G and v in G:
            G.add_edge(u, v, total_articles=w)

    n = max(G.number_of_nodes(), 1)
    k = spread / np.sqrt(n)
    pos = nx.spring_layout(G, seed=seed, k=k, iterations=300)

    drug_nodes = [n for n,d in G.nodes(data=True) if d.get("type")=="drug"]
    disease_nodes = [n for n,d in G.nodes(data=True) if d.get("type")=="disease"]

    weights = [G[u][v].get("total_articles",1) for u,v in G.edges()]
    if weights:
        lo, hi = min(weights), max(weights)
        widths = [1.5 if hi==lo else 0.4 + 4.6*(w-lo)/(hi-lo) for w in weights]
    else:
        widths = []

    plt.figure(figsize=(14,9))
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.28)
    nx.draw_networkx_nodes(G, pos, nodelist=disease_nodes, node_size=22, node_color="red", alpha=0.85)
    nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_size=360, node_color="#3A78B4", alpha=0.95)
    labels = {n:n for n in drug_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    plt.axis("off"); plt.tight_layout()
    plt.savefig(outpath, dpi=250); plt.close()

def bubblesize(series, min_size=60, max_size=360):
    s = series.fillna(0).astype(float)
    if len(s)==0: return s
    lo, hi = s.min(), s.max()
    if hi==lo: return pd.Series([ (min_size+max_size)/2 ]*len(s), index=s.index)
    return min_size + (s - lo) * (max_size - min_size) / (hi - lo)

def add_size_legend(ax, sizes, title="total_trials"):
    if len(sizes)==0: return
    vals = np.linspace(sizes.min(), sizes.max(), 4)
    handles = []
    for v in vals:
        handles.append(plt.scatter([], [], s=bubblesize(pd.Series([v]))[0], edgecolors="none"))
    ax.legend(handles, [f"{int(v)}" for v in vals], scatterpoints=1, frameon=False, title=title, loc="upper right")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--outdir", type=str, default="figures")
    ap.add_argument("--spread", type=float, default=1.8)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    out_dir = (run_dir / args.outdir); out_dir.mkdir(parents=True, exist_ok=True)

    nodes, edges = load_graph(run_dir)

    # Top diseases by total publications
    dis_df = top15_diseases_by_total_pubs(nodes, edges)
    plt.figure(figsize=(10,6))
    plt.barh(dis_df["label"][::-1], dis_df["total_articles"][::-1], color="#C23B22")
    plt.xlabel("Total publications"); plt.ylabel("Disease")
    plt.title("Top 15 diseases by total publications")
    plt.tight_layout(); plt.savefig(out_dir / "top15_diseases_by_total_pubs.png", dpi=200); plt.close()

    # Top drugs by publications
    drug_df = top15_drugs_by_pubs(nodes, edges)
    plt.figure(figsize=(10,6))
    plt.barh(drug_df["label"][::-1], drug_df["total_articles"][::-1], color="#3A78B4")
    plt.xlabel("Total publications"); plt.ylabel("Drug")
    plt.title("Top 10 drugs by total publications")
    plt.tight_layout(); plt.savefig(out_dir / "top15_drugs_by_publications.png", dpi=200); plt.close()

    # Scatter with viridis and bubble sizes
    scat = scatter_drug_degree_vs_pubs(nodes, edges)
    sizes = bubblesize(scat["total_articles"], min_size=80, max_size=420)
    cmap = mpl.cm.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(11,6.5))
    sc = ax.scatter(
        scat["unique_diseases"], scat["total_articles"],
        c=scat["unique_diseases"], cmap=cmap, s=sizes, alpha=0.9, edgecolors="black", linewidths=0.4
    )
    for _, r in scat.iterrows():
        ax.text(r["unique_diseases"]+0.12, r["total_articles"], r["drug_name"], fontsize=9)

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("unique_diseases")

    add_size_legend(ax, scat["total_articles"], title="total_trials")

    ax.set_xlabel("Unique Diseases")
    ax.set_ylabel("Total Trials")
    ax.set_title("Drug Breadth vs Total Positive Trials (PubMed)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "drug_scatter_unique_vs_publications.png", dpi=220)
    plt.close(fig)

    # Network with more spread
    draw_network(nodes, edges, out_dir / "network_drug_disease.png", spread=args.spread)

if __name__ == "__main__":
    main()
