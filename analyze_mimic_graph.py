#!/usr/bin/env python3
import argparse, time, pathlib, pandas as pd, networkx as nx
from itertools import combinations
from collections import defaultdict

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edges", default="bigquery_tables/edges.csv")
    p.add_argument("--nodes", default="bigquery_tables/nodes.csv")
    p.add_argument("--outdir", default=None)
    p.add_argument("--min_shared", type=int, default=2)     # for projections
    p.add_argument("--topk", type=int, default=10)
    args = p.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = pathlib.Path(args.outdir or f"outputs/mimic_analysis_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(args.edges)
    nodes = pd.read_csv(args.nodes)
    need = {"src","dst","weight_admissions"}
    assert need.issubset(edges.columns), f"edges.csv needs {need}"
    nodes = nodes.set_index("id")
    label = nodes["label"].to_dict()
    ntype  = nodes["type"].to_dict()

    # Clean types
    if "unique_patients" not in edges: edges["unique_patients"]=0
    edges["weight_admissions"] = edges["weight_admissions"].fillna(0).astype(int)
    edges["unique_patients"]   = edges["unique_patients"].fillna(0).astype(int)

    # Summaries
    dis_sum = (edges.groupby("dst")
               .agg(distinct_drugs=("src","nunique"),
                    admissions=("weight_admissions","sum"),
                    patients=("unique_patients","sum"))
               .reset_index())
    dis_sum["disease_label"] = dis_sum["dst"].map(label)
    dis_sum.sort_values(["distinct_drugs","admissions"], ascending=[False,False]) \
           .to_csv(outdir/"summary_diseases_by_frequency.csv", index=False)

    drug_sum = (edges.groupby("src")
                .agg(distinct_diseases=("dst","nunique"),
                     total_admissions=("weight_admissions","sum"),
                     total_patients=("unique_patients","sum"))
                .reset_index())
    drug_sum["drug_label"] = drug_sum["src"].map(label)
    drug_sum.sort_values("total_admissions", ascending=False) \
            .to_csv(outdir/"summary_drugs_by_admissions.csv", index=False)

    # Top-K lists
    cols_extra = [c for c in ["p_disease_given_drug","p_drug_given_disease"] if c in edges.columns]
    td = (edges.sort_values(["src","weight_admissions"], ascending=[True,False])
          .groupby("src").head(args.topk)
          .assign(disease_label=lambda d: d["dst"].map(label)))
    td[["src","disease_label","dst","weight_admissions","unique_patients",*cols_extra]] \
        .to_csv(outdir/"top_diseases_per_drug.csv", index=False)

    tj = (edges.sort_values(["dst","weight_admissions"], ascending=[True,False])
          .groupby("dst").head(args.topk)
          .assign(drug_label=lambda d: d["src"].map(label)))
    tj[["dst","drug_label","src","weight_admissions","unique_patients",*cols_extra]] \
        .to_csv(outdir/"top_drugs_per_disease.csv", index=False)

    # Bipartite graph (drug→disease)
    G = nx.DiGraph()
    for nid, row in nodes.iterrows():
        G.add_node(nid, type=row["type"], label=row["label"],
                   aliases=row.get("aliases",""), source_ids=row.get("source_ids",""))
    for _, r in edges.iterrows():
        G.add_edge(r["src"], r["dst"], relation="treat",
                   weight_admissions=int(r["weight_admissions"]),
                   unique_patients=int(r["unique_patients"]),
                   **({k: float(r[k]) for k in cols_extra} if cols_extra else {}))
    nx.write_graphml(G, outdir/"graph.graphml")

    # Disease–disease projection (shared drugs)
    deg_d = edges.groupby("dst")["src"].nunique().to_dict()
    dd_counts = defaultdict(int)
    for _, grp in edges.groupby("src"):
        ds = sorted(grp["dst"].unique())
        for a,b in combinations(ds,2): dd_counts[(a,b)] += 1
    dd_rows = []
    for (a,b), inter in dd_counts.items():
        ua, ub = deg_d.get(a,0), deg_d.get(b,0)
        j = inter / (ua + ub - inter)
        if inter >= args.min_shared:
            dd_rows.append({"src":a,"dst":b,"shared_drugs":inter,"jaccard":j,
                            "src_deg_drugs":ua,"dst_deg_drugs":ub,
                            "src_label":label.get(a,""),"dst_label":label.get(b,"")})
    dd = pd.DataFrame(dd_rows).sort_values(["shared_drugs","jaccard"], ascending=[False,False])
    dd.to_csv(outdir/"disease_disease_projection.csv", index=False)
    H = nx.Graph()
    for _, r in dd.iterrows():
        H.add_edge(r["src"], r["dst"], shared_drugs=int(r["shared_drugs"]), jaccard=float(r["jaccard"]))
        H.nodes[r["src"]]["label"]=label.get(r["src"],"")
        H.nodes[r["dst"]]["label"]=label.get(r["dst"],"")
        H.nodes[r["src"]]["type"]="disease"; H.nodes[r["dst"]]["type"]="disease"
    nx.write_graphml(H, outdir/"disease_disease_projection.graphml")

    # Drug–drug projection (shared diseases)
    deg_g = edges.groupby("src")["dst"].nunique().to_dict()
    gg_counts = defaultdict(int)
    for _, grp in edges.groupby("dst"):
        gs = sorted(grp["src"].unique())
        for a,b in combinations(gs,2): gg_counts[(a,b)] += 1
    gg_rows = []
    for (a,b), inter in gg_counts.items():
        ua, ub = deg_g.get(a,0), deg_g.get(b,0)
        j = inter / (ua + ub - inter)
        if inter >= args.min_shared:
            gg_rows.append({"src":a,"dst":b,"shared_diseases":inter,"jaccard":j,
                            "src_deg_diseases":ua,"dst_deg_diseases":ub,
                            "src_label":label.get(a,""),"dst_label":label.get(b,"")})
    gg = pd.DataFrame(gg_rows).sort_values(["shared_diseases","jaccard"], ascending=[False,False])
    gg.to_csv(outdir/"drug_drug_projection.csv", index=False)
    K = nx.Graph()
    for _, r in gg.iterrows():
        K.add_edge(r["src"], r["dst"], shared_diseases=int(r["shared_diseases"]), jaccard=float(r["jaccard"]))
        K.nodes[r["src"]]["label"]=label.get(r["src"],""); K.nodes[r["src"]]["type"]="drug"
        K.nodes[r["dst"]]["label"]=label.get(r["dst"],""); K.nodes[r["dst"]]["type"]="drug"
    nx.write_graphml(K, outdir/"drug_drug_projection.graphml")

    # README
    (outdir/"README.txt").write_text(
f"""Inputs: {args.edges}, {args.nodes}
Outputs:
- graph.graphml (drug→disease), disease_disease_projection.graphml, drug_drug_projection.graphml
- summary_diseases_by_frequency.csv, summary_drugs_by_admissions.csv
- top_diseases_per_drug.csv, top_drugs_per_disease.csv
- disease_disease_projection.csv (shared_drugs ≥ {args.min_shared}, Jaccard), drug_drug_projection.csv (shared_diseases ≥ {args.min_shared})
Notes: weights=admission counts; patients=unique subject counts; Jaccard on neighbor sets.
""")
    print(f"Wrote {outdir}")

if __name__ == "__main__":
    main()
