#!/usr/bin/env python3
# mimic_graph_analytics.py
# Usage: python mimic_graph_analytics.py RUN_DIR [--outdir figures] [--spread 2.0] [--min_weight 0]
import os, math, argparse, pathlib, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

RED = "red"
STEEL = "steelblue"

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def read_inputs(run_dir: pathlib.Path):
    nodes = pd.read_csv(run_dir / "nodes.csv")
    edges = pd.read_csv(run_dir / "edges.csv")
    for c in ["id","type","label"]:
        if c not in nodes.columns: raise ValueError(f"nodes.csv missing {c}")
    for c in ["src","dst","weight_admissions","unique_patients"]:
        if c not in edges.columns: raise ValueError(f"edges.csv missing {c}")
    # numeric cleanup
    for c in ["weight_admissions","unique_patients","p_disease_given_drug","p_drug_given_disease"]:
        if c in edges.columns: edges[c] = pd.to_numeric(edges[c], errors="coerce").fillna(0)
    # de-dup potential duplicate (src,dst)
    agg = {"weight_admissions":"sum","unique_patients":"sum"}
    for c in ["p_disease_given_drug","p_drug_given_disease"]:
        if c in edges.columns: agg[c] = "mean"
    edges = edges.groupby(["src","dst"], as_index=False).agg(agg)
    return nodes, edges

def bubblesize(vals, min_size=80, max_size=420, vmin=None, vmax=None):
    v = np.asarray(pd.Series(vals).astype(float))
    if v.size == 0: return v
    if vmin is None: vmin = np.nanmin(v)
    if vmax is None: vmax = np.nanmax(v)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return np.full_like(v, (min_size + max_size) / 2.0, dtype=float)
    s = (v - vmin) / (vmax - vmin)
    return min_size + s * (max_size - min_size)

def add_size_legend(ax, values, title, min_size=80, max_size=420):
    vals = [v for v in values if np.isfinite(v)]
    if not vals: return None
    vals = [min(vals), np.median(vals), max(vals)]
    handles = [
        ax.scatter([], [], s=bubblesize([v], min_size, max_size, min(vals), max(vals))[0],
                   facecolors="none", edgecolors="gray", linewidths=1.0, label=str(int(v)))
        for v in vals
    ]
    leg = ax.legend(handles=handles, title=title, loc="lower right", frameon=True)
    # compatibility across matplotlib versions
    hlist = getattr(leg, "legendHandles", None) or getattr(leg, "legend_handles", None) or []
    for h in hlist: 
        try: h.set_alpha(1.0)
        except Exception: pass
    return leg

def scale_to_range(x, lo=0.4, hi=4.6):
    x = np.asarray(pd.Series(x).astype(float))
    if x.size == 0: return x
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
        return np.full_like(x, (lo + hi) / 2.0, dtype=float)
    return lo + (x - mn) * (hi - lo) / (mx - mn)

def wrap_labels(labels, width=32):
    return ['\n'.join(textwrap.wrap(str(l), width=width, break_long_words=False)) for l in labels]

def barh_plot(labels, values, color, xlabel, outpath):
    # widen and pad left to fit wrapped labels
    h = max(5, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(12, h), constrained_layout=False)
    ax.barh(labels, values, color=color)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    fig.subplots_adjust(left=0.30, right=0.98, top=0.98, bottom=0.08)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--spread", type=float, default=2.0)
    ap.add_argument("--min_weight", type=float, default=0.0)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    outdir = run_dir / args.outdir
    ensure_dir(outdir)

    nodes, edges = read_inputs(run_dir)
    type_map = dict(zip(nodes["id"], nodes["type"]))
    label_map = dict(zip(nodes["id"], nodes["label"]))

    use_patients = str(os.getenv("MIMIC_USE_UNIQUE_PATIENTS", "0")).strip().lower() in {"1","true","yes"}
    metric_col = "unique_patients" if use_patients else "weight_admissions"
    metric_title = "unique_patients" if use_patients else "total_admissions"

    # Undirected view for aggregations
    eL = edges.copy(); eL["node"] = eL["src"]; eL["nbr"] = eL["dst"]
    eR = edges.copy(); eR["node"] = eR["dst"]; eR["nbr"] = eR["src"]
    ei = pd.concat([eL[["node","nbr",metric_col]], eR[["node","nbr",metric_col]]], ignore_index=True)
    ei["type"] = ei["node"].map(type_map)
    ei["nbr_type"] = ei["nbr"].map(type_map)

    # 1) Top 15 diseases by total admissions
    dis = (ei[(ei["type"]=="disease") & (ei["nbr_type"]=="drug")]
           .groupby("node", as_index=False)[metric_col].sum()
           .sort_values(metric_col, ascending=False).head(15))
    if dis.empty:
        print("No diseases found; skipping disease bar chart.")
    else:
        dis["label"] = dis["node"].map(label_map).fillna(dis["node"])
        lab = wrap_labels(dis["label"], width=32)
        out = outdir/"top15_diseases_by_total_admissions.png"
        barh_plot(lab, dis[metric_col], RED, metric_title, out)
        print(out)

    # 2) Top 15 drugs by total admissions
    dr = (ei[(ei["type"]=="drug") & (ei["nbr_type"]=="disease")]
          .groupby("node", as_index=False)[metric_col].sum()
          .sort_values(metric_col, ascending=False).head(15))
    if dr.empty:
        print("No drugs found; skipping drug bar chart.")
    else:
        dr["label"] = dr["node"].map(label_map).fillna(dr["node"])
        lab = wrap_labels(dr["label"], width=22)
        out = outdir/"top15_drugs_by_total_admissions.png"
        barh_plot(lab, dr[metric_col], STEEL, metric_title, out)
        print(out)

    # 3) Drug breadth vs total admissions (scatter)
    breadth = (ei[(ei["type"]=="drug") & (ei["nbr_type"]=="disease")]
               .groupby("node")["nbr"].nunique().rename("unique_diseases"))
    totals  = (ei[(ei["type"]=="drug") & (ei["nbr_type"]=="disease")]
               .groupby("node")[metric_col].sum().rename(metric_title))
    df_sc = pd.concat([breadth, totals], axis=1).dropna().reset_index()
    if df_sc.empty:
        print("No drug breadth data; skipping scatter.")
    else:
        df_sc["label"] = df_sc["node"].map(label_map).fillna(df_sc["node"])
        sizes = bubblesize(df_sc[metric_title], 80, 420)
        fig, ax = plt.subplots(figsize=(10.5, 7.5))
        sc = ax.scatter(df_sc["unique_diseases"], df_sc[metric_title],
                        s=sizes, c=df_sc["unique_diseases"], cmap="viridis",
                        alpha=0.85, edgecolors="k", linewidths=0.3)
        for _, r in df_sc.iterrows():
            ax.annotate(r["label"], (r["unique_diseases"], r[metric_title]),
                        textcoords="offset points", xytext=(4,3), fontsize=8)
        cb = plt.colorbar(sc, ax=ax); cb.set_label("unique_diseases")
        add_size_legend(ax, list(df_sc[metric_title]), title=metric_title)
        ax.set_xlabel("unique_diseases"); ax.set_ylabel(metric_title)
        plt.tight_layout()
        out = outdir/"drug_scatter_unique_vs_total_admissions.png"
        fig.savefig(out, dpi=220); plt.close(fig)
        print(out)

    # 4) Network plot (undirected)
    G = nx.Graph()
    for _, r in nodes.iterrows():
        G.add_node(r["id"], type=r["type"], label=r["label"])
    plot_edges = edges[edges[metric_col] >= float(args.min_weight)].copy()
    for _, r in plot_edges.iterrows():
        u, v, w = r["src"], r["dst"], float(r[metric_col])
        if u not in G: G.add_node(u, type=type_map.get(u,""), label=label_map.get(u,u))
        if v not in G: G.add_node(v, type=type_map.get(v,""), label=label_map.get(v,v))
        if G.has_edge(u,v): G[u][v]["weight"] += w
        else: G.add_edge(u, v, weight=w)

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print("No edges/nodes for network after filtering; skipping network plot.")
    else:
        n = G.number_of_nodes()
        k = args.spread / math.sqrt(max(n,1))
        pos = nx.spring_layout(G, seed=42, k=k, iterations=300)

        widths = scale_to_range([G[u][v]["weight"] for u,v in G.edges()], 0.4, 4.6)
        drugs = [n for n,d in G.nodes(data=True) if d.get("type")=="drug"]
        diseases = [n for n,d in G.nodes(data=True) if d.get("type")=="disease"]

        fig, ax = plt.subplots(figsize=(12, 9))
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="gray", alpha=0.28)
        nx.draw_networkx_nodes(G, pos, nodelist=diseases, node_color=RED, node_size=22, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=drugs, node_color=STEEL, node_size=360, ax=ax)
        nx.draw_networkx_labels(G, pos, labels={n:G.nodes[n]["label"] for n in drugs}, font_size=10, ax=ax)
        ax.axis("off"); plt.tight_layout()
        out = outdir/"network_drug_disease.png"
        fig.savefig(out, dpi=260); plt.close(fig)
        print(out)

if __name__ == "__main__":
    main()
