#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import networkx as nx


CT_PATH = Path("ct_manual_filter/ctgov_manual_filtered_final.csv")
PT_EDGES_PATH = Path(
    "outputs/pipeline_20251120_191408/pipeline_20251120_191408_summary.csv"
)
OUT_DIR = Path("comparisons/ctgov_vs_pubtator_networks_v2")


def norm_basic(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    out = []
    prev_space = False
    for ch in s:
        ok = ("a" <= ch <= "z") or ("0" <= ch <= "9")
        if ok:
            out.append(ch)
            prev_space = False
        else:
            if not prev_space:
                out.append(" ")
                prev_space = True
    return " ".join("".join(out).split())


def singularize_token(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("ses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def token_key(s: str) -> tuple:
    toks = norm_basic(s).split()
    toks = [singularize_token(t) for t in toks if t]
    c = Counter(toks)
    return tuple(sorted(c.items()))


def build_pubtator_disease_index(pt_edges: pd.DataFrame):
    exact_idx = defaultdict(list)
    token_idx = defaultdict(list)
    for _, r in pt_edges.iterrows():
        lab = str(r["disease_name"])
        did = str(r["disease_id"])
        kn = norm_basic(lab)
        kt = token_key(lab)
        exact_idx[kn].append((lab, did))
        token_idx[kt].append((lab, did))
    return exact_idx, token_idx


def resolve_ct_disease_to_pubtator(d: str, exact_idx, token_idx, block_norm: set):
    d0 = "" if d is None else str(d).strip()
    kn = norm_basic(d0)
    if not kn or kn in block_norm:
        return {"status": "blocked_non_disease", "mapped_label": "", "mapped_id": ""}

    kt = token_key(d0)

    exact = exact_idx.get(kn, [])
    if len(exact) == 1:
        lab, did = exact[0]
        return {"status": "matched_exact", "mapped_label": lab, "mapped_id": did}
    if len(exact) > 1:
        return {"status": "ambiguous", "mapped_label": "", "mapped_id": ""}

    tok = token_idx.get(kt, [])
    if len(tok) == 1:
        lab, did = tok[0]
        return {"status": "matched_token_set", "mapped_label": lab, "mapped_id": did}
    if len(tok) > 1:
        return {"status": "ambiguous", "mapped_label": "", "mapped_id": ""}

    return {"status": "unmatched", "mapped_label": "", "mapped_id": ""}


def build_bipartite_graph(edges_df: pd.DataFrame, drug_col: str, disease_col: str, weight_col: str):
    G = nx.Graph()
    for _, r in edges_df.iterrows():
        d = f"drug::{r[drug_col]}"
        s = f"disease::{r[disease_col]}"
        w = float(r[weight_col])
        G.add_node(d, bipartite="drug")
        G.add_node(s, bipartite="disease")
        if G.has_edge(d, s):
            G[d][s]["weight"] += w
        else:
            G.add_edge(d, s, weight=w)
    return G


def graph_summary(G: nx.Graph):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    drugs = [u for u, a in G.nodes(data=True) if a.get("bipartite") == "drug"]
    diseases = [u for u, a in G.nodes(data=True) if a.get("bipartite") == "disease"]
    largest_cc = max(nx.connected_components(G), key=len) if n else set()
    G_lcc = G.subgraph(largest_cc).copy() if largest_cc else nx.Graph()

    if G_lcc.number_of_nodes() > 1:
        try:
            avg_path = nx.average_shortest_path_length(G_lcc)
        except Exception:
            avg_path = None
        try:
            diameter = nx.diameter(G_lcc)
        except Exception:
            diameter = None
    else:
        avg_path = None
        diameter = None

    degs = [G.degree(u) for u in G.nodes()]
    wdegs = [
        sum(float(G[u][v].get("weight", 0.0)) for v in G.neighbors(u))
        for u in G.nodes()
    ]

    def stats(x):
        if not x:
            return {"min": 0, "max": 0, "mean": 0.0}
        return {
            "min": int(min(x)),
            "max": int(max(x)),
            "mean": float(sum(x) / len(x)),
        }

    return {
        "n_nodes": int(n),
        "n_edges": int(m),
        "n_drugs": int(len(drugs)),
        "n_diseases": int(len(diseases)),
        "density_undirected": float((2.0 * m) / (n * (n - 1))) if n > 1 else 0.0,
        "n_components": int(nx.number_connected_components(G)) if n else 0,
        "largest_component_size": int(len(largest_cc)),
        "largest_component_fraction": float(len(largest_cc) / n) if n else 0.0,
        "largest_component_avg_path_length": avg_path,
        "largest_component_diameter": diameter,
        "degree_stats": stats(degs),
        "weighted_degree_stats": stats(wdegs),
    }


def top_nodes_by_degree(G: nx.Graph, kind_prefix: str, topn: int = 50):
    rows = []
    for u in G.nodes():
        if not u.startswith(kind_prefix):
            continue
        deg = G.degree(u)
        wdeg = sum(float(G[u][v].get("weight", 0.0)) for v in G.neighbors(u))
        rows.append(
            {
                "node": u.split("::", 1)[1],
                "degree": int(deg),
                "weighted_degree": float(wdeg),
            }
        )
    rows.sort(key=lambda r: (r["weighted_degree"], r["degree"]), reverse=True)
    return pd.DataFrame(rows[:topn])


def disease_projection_top_pairs(
    edges_df: pd.DataFrame,
    disease_col: str,
    drug_col: str,
    top_diseases: int = 250,
    top_pairs: int = 1000,
):
    # Rank diseases by total edge weight only, which is enough to select a core set
    deg = (
        edges_df.groupby(disease_col)["weight"]
        .sum()
        .reset_index(name="w")
        .sort_values("w", ascending=False)
    )

    keep = set(deg.head(top_diseases)[disease_col].astype(str).tolist())
    sub = edges_df[edges_df[disease_col].astype(str).isin(keep)].copy()

    # Build mapping drug -> list of diseases it is linked to (within the kept set)
    drug_to_dis = defaultdict(list)
    for _, r in sub.iterrows():
        drug_to_dis[str(r[drug_col])].append(str(r[disease_col]))

    # Count disease pairs that share at least one drug
    pair_counts = Counter()
    for dis_list in drug_to_dis.values():
        dis_list = sorted(set(dis_list))
        L = len(dis_list)
        for i in range(L):
            for j in range(i + 1, L):
                pair_counts[(dis_list[i], dis_list[j])] += 1

    rows = [
        {"disease_a": a, "disease_b": b, "shared_drugs": int(c)}
        for (a, b), c in pair_counts.most_common(top_pairs)
    ]
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # PubTator edges from pipeline summary
    pt = pd.read_csv(PT_EDGES_PATH)
    need_pt = {
        "drug_name",
        "drug_id",
        "disease_name",
        "disease_id",
        "pmid_count",
        "total_articles",
    }
    miss_pt = need_pt - set(pt.columns)
    if miss_pt:
        raise SystemExit(f"PubTator edges missing columns: {sorted(miss_pt)}")

    pt["drug_name"] = pt["drug_name"].astype(str)
    pt["drug_id"] = pt["drug_id"].astype(str)
    pt["disease_name"] = pt["disease_name"].astype(str)
    pt["disease_id"] = pt["disease_id"].astype(str)
    pt["pmid_count"] = pd.to_numeric(pt["pmid_count"], errors="coerce").fillna(0)
    pt["total_articles"] = pd.to_numeric(pt["total_articles"], errors="coerce").fillna(0)
    pt["drug_key"] = pt["drug_name"].map(norm_basic)

    exact_idx, token_idx = build_pubtator_disease_index(pt)

    # CT data
    ct0 = pd.read_csv(CT_PATH)
    need_ct = {"drug", "nct_id", "title", "status", "phase", "disease", "first_submit"}
    miss_ct = need_ct - set(ct0.columns)
    if miss_ct:
        raise SystemExit(f"CT file missing columns: {sorted(miss_ct)}")

    ct0["drug"] = ct0["drug"].astype(str)
    ct0["drug_key"] = ct0["drug"].map(norm_basic)
    ct0["nct_id"] = ct0["nct_id"].astype(str)
    ct0["first_submit"] = pd.to_datetime(ct0["first_submit"], errors="coerce", utc=True)
    ct0["year"] = ct0["first_submit"].dt.year.astype("Int64")

    rows = []
    for _, r in ct0.iterrows():
        parts = [p.strip() for p in str(r["disease"]).split(";")]
        parts = [p for p in parts if p]
        if not parts:
            rows.append({**r.to_dict(), "disease_raw": ""})
        else:
            for d in parts:
                rr = r.to_dict()
                rr["disease_raw"] = d
                rows.append(rr)
    ct_long = pd.DataFrame(rows)
    ct_long["disease_raw"] = ct_long["disease_raw"].astype(str)

    block = {norm_basic(x) for x in ["healthy", "healthy volunteer", "healthy volunteers"]}

    uniq_ct_dis = sorted(set(ct_long["disease_raw"].astype(str).tolist()))
    res_map = {}
    stats = Counter()
    for d in uniq_ct_dis:
        out = resolve_ct_disease_to_pubtator(d, exact_idx, token_idx, block)
        res_map[d] = out
        stats[out["status"]] += 1

    ct_long["match_status"] = ct_long["disease_raw"].map(
        lambda x: res_map.get(x, {}).get("status", "unmatched")
    )
    ct_long["disease_mapped_label"] = ct_long["disease_raw"].map(
        lambda x: res_map.get(x, {}).get("mapped_label", "")
    )
    ct_long["disease_mapped_id"] = ct_long["disease_raw"].map(
        lambda x: res_map.get(x, {}).get("mapped_id", "")
    )

    excluded_mask = ct_long["match_status"].isin(
        ["ambiguous", "blocked_non_disease"]
    )
    ct_used = ct_long[~excluded_mask].copy()

    def ct_concept_id(row):
        mid = str(row["disease_mapped_id"] or "")
        if mid:
            return mid
        return "CT_UNMAPPED::" + norm_basic(row["disease_raw"])

    def ct_concept_label(row):
        ml = str(row["disease_mapped_label"] or "")
        if ml:
            return ml
        return str(row["disease_raw"])

    ct_used["disease_concept_id"] = ct_used.apply(ct_concept_id, axis=1)
    ct_used["disease_concept_label"] = ct_used.apply(ct_concept_label, axis=1)

    # CT edges: drug_key x disease_concept_id
    ct_edges = (
        ct_used.groupby(
            ["drug_key", "drug", "disease_concept_id", "disease_concept_label"],
            dropna=False,
        )
        .agg(
            n_trials=("nct_id", "nunique"),
            n_rows=("nct_id", "size"),
        )
        .reset_index()
    )
    ct_edges["weight"] = ct_edges["n_trials"].astype(float)

    # PubTator edges in canonical format
    pt_edges = pt.rename(
        columns={
            "drug_name": "drug",
            "disease_id": "disease_concept_id",
            "disease_name": "disease_concept_label",
            "total_articles": "weight",
        }
    ).copy()
    pt_edges["weight"] = pt_edges["weight"].astype(float)

    # Per drug alignment of disease sets
    pt_by_drug = (
        pt_edges.groupby("drug_key")["disease_concept_id"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )
    ct_by_drug = (
        ct_edges.groupby("drug_key")["disease_concept_id"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

    drugs_all = sorted(set(pt_by_drug.keys()) | set(ct_by_drug.keys()))
    align_rows = []
    for dk in drugs_all:
        s_pt = pt_by_drug.get(dk, set())
        s_ct = ct_by_drug.get(dk, set())
        inter = s_pt & s_ct
        uni = s_pt | s_ct
        jac = (len(inter) / len(uni)) if len(uni) else 0.0
        example_ct = (
            ct_edges[ct_edges["drug_key"] == dk]["drug"]
            .dropna()
            .astype(str)
            .head(1)
            .tolist()
        )
        example_pt = (
            pt_edges[pt_edges["drug_key"] == dk]["drug"]
            .dropna()
            .astype(str)
            .head(1)
            .tolist()
        )
        align_rows.append(
            {
                "drug_key": dk,
                "example_ct_drug": example_ct[0] if example_ct else "",
                "example_pubtator_drug": example_pt[0] if example_pt else "",
                "n_ct_diseases": int(len(s_ct)),
                "n_pubtator_diseases": int(len(s_pt)),
                "n_shared_diseases": int(len(inter)),
                "n_union_diseases": int(len(uni)),
                "jaccard_disease_alignment": float(jac),
            }
        )

    align_df = pd.DataFrame(align_rows).sort_values(
        ["jaccard_disease_alignment", "n_shared_diseases"], ascending=False
    )
    align_df.to_csv(OUT_DIR / "per_drug_disease_alignment.csv", index=False)

    # Bipartite graphs and summaries
    G_ct = build_bipartite_graph(ct_edges, "drug_key", "disease_concept_id", "weight")
    G_pt = build_bipartite_graph(pt_edges, "drug_key", "disease_concept_id", "weight")

    summ_ct = graph_summary(G_ct)
    summ_pt = graph_summary(G_pt)

    hubs_ct_drug = top_nodes_by_degree(G_ct, "drug::", 50)
    hubs_pt_drug = top_nodes_by_degree(G_pt, "drug::", 50)
    hubs_ct_dis = top_nodes_by_degree(G_ct, "disease::", 50)
    hubs_pt_dis = top_nodes_by_degree(G_pt, "disease::", 50)

    hubs_ct_drug.to_csv(OUT_DIR / "ct_hubs_drugs.csv", index=False)
    hubs_pt_drug.to_csv(OUT_DIR / "pubtator_hubs_drugs.csv", index=False)
    hubs_ct_dis.to_csv(OUT_DIR / "ct_hubs_diseases.csv", index=False)
    hubs_pt_dis.to_csv(OUT_DIR / "pubtator_hubs_diseases.csv", index=False)

    # Disease projection networks for shared drugs
    ct_proj = ct_edges.rename(
        columns={"drug_key": "drug_id", "disease_concept_id": "disease_id"}
    ).copy()
    pt_proj = pt_edges.rename(
        columns={"drug_key": "drug_id", "disease_concept_id": "disease_id"}
    ).copy()

    ct_pairs = disease_projection_top_pairs(
        ct_proj, "disease_id", "drug_id", top_diseases=250, top_pairs=1000
    )
    pt_pairs = disease_projection_top_pairs(
        pt_proj, "disease_id", "drug_id", top_diseases=250, top_pairs=1000
    )

    ct_pairs.to_csv(OUT_DIR / "ct_disease_shared_drugs_top_pairs.csv", index=False)
    pt_pairs.to_csv(
        OUT_DIR / "pubtator_disease_shared_drugs_top_pairs.csv", index=False
    )

    # Resolution table for CT diseases
    res_tbl = []
    for d, o in res_map.items():
        res_tbl.append(
            {
                "ct_disease_raw": d,
                "match_status": o["status"],
                "mapped_pubtator_label": o.get("mapped_label", ""),
                "mapped_pubtator_id": o.get("mapped_id", ""),
            }
        )
    pd.DataFrame(res_tbl).to_csv(
        OUT_DIR / "ct_to_pubtator_disease_resolution.csv", index=False
    )

    # Summary json
    align_nonempty = align_df[align_df["n_union_diseases"] > 0]
    mean_jac = (
        float(align_nonempty["jaccard_disease_alignment"].mean())
        if len(align_nonempty)
        else 0.0
    )

    summ = {
        "paths": {
            "ct_path": str(CT_PATH),
            "pub_edges_path": str(PT_EDGES_PATH),
            "out_dir": str(OUT_DIR),
        },
        "ct_mapping": {
            "match_status_counts": {k: int(v) for k, v in stats.items()},
            "n_unique_ct_disease_strings": int(len(uniq_ct_dis)),
        },
        "graphs": {
            "ct_graph": summ_ct,
            "pubtator_graph": summ_pt,
        },
        "alignment": {
            "n_drugs_with_any_alignment": int(
                align_nonempty["drug_key"].nunique()
            ),
            "mean_jaccard_over_drugs_with_union": mean_jac,
            "alignment_csv": "per_drug_disease_alignment.csv",
        },
        "outputs": {
            "ct_hubs_drugs_csv": "ct_hubs_drugs.csv",
            "pubtator_hubs_drugs_csv": "pubtator_hubs_drugs.csv",
            "ct_hubs_diseases_csv": "ct_hubs_diseases.csv",
            "pubtator_hubs_diseases_csv": "pubtator_hubs_diseases.csv",
            "ct_disease_shared_drugs_top_pairs_csv": "ct_disease_shared_drugs_top_pairs.csv",
            "pubtator_disease_shared_drugs_top_pairs_csv": "pubtator_disease_shared_drugs_top_pairs.csv",
        },
        "weights": {
            "ct_edge_weight": "n_trials (unique NCT ids) per drug_key and disease_concept_id",
            "pubtator_edge_weight": "total_articles per drug_key and disease_id",
        },
    }

    (OUT_DIR / "network_comparison_summary.json").write_text(
        json.dumps(summ, indent=2), encoding="utf-8"
    )

    print(f"Wrote outputs under: {OUT_DIR}")


if __name__ == "__main__":
    main()
