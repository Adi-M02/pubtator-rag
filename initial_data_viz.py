#!/usr/bin/env python3
# three_layer_entities_aligned.py
# Disease -> Entity IDs -> Drugs with right-side aligned per-entity counts.
# Edge width (entity->drug) ∝ total_articles (0 if none). Includes all drug_ids.

import argparse, json, math, re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

# ---------- helpers ----------
def load_data(p: Path) -> Dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def ensure_outdir(run_id: str) -> Path:
    out = Path(f"outputs/graphs_{run_id}")
    out.mkdir(parents=True, exist_ok=True)
    return out

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "disease"

def clean_chem_label(s: str | None) -> str:
    s = s or "UNKNOWN"
    if s.startswith("@CHEMICAL_"): s = s[len("@CHEMICAL_"):]
    return s.replace("_", " ")

def crop_middle(s: str, max_chars: int) -> str:
    if len(s) <= max_chars: return s
    keep = max_chars - 3
    left = keep // 2
    right = keep - left
    return s[:left] + "..." + s[-right:]

def layered_positions_3(layers: List[List[str]], x_gap=2.5, y_pad=0.06) -> Dict[str, Tuple[float, float]]:
    pos: Dict[str, Tuple[float, float]] = {}
    for xi, nodes in enumerate(layers):
        m = max(1, len(nodes))
        ys = [0.5] if m == 1 else [y_pad + i*(1 - 2*y_pad)/(m - 1) for i in range(m)]
        for yi, n in zip(ys, nodes):
            pos[n] = (xi * x_gap, 1 - yi)
    return pos

def scale_widths(values: List[int], min_w=1.6, max_w=10.0) -> tuple[Dict[int, float], float]:
    if not values: return {}, min_w * 0.6
    pos_vals = [v for v in values if v > 0]
    zero_width = min_w * 0.6
    if not pos_vals: return {}, zero_width
    vmin, vmax = min(pos_vals), max(pos_vals)
    if vmax == vmin: return {v: (min_w + max_w)/2 for v in pos_vals}, zero_width
    lvmin, lvmax = math.log1p(vmin), math.log1p(vmax)
    def f(v: int) -> float:
        return min_w + (max_w - min_w) * ((math.log1p(v) - lvmin) / (lvmax - lvmin))
    return {v: f(v) for v in pos_vals}, zero_width

# ---------- data extraction ----------
def collect_single_disease_entity_specific(data: Dict, disease_name: str):
    # entities listed for disease
    d2e = {d["disease_name"]: d.get("entity_ids", []) for d in data.get("disease_entities", [])}
    entity_ids = d2e.get(disease_name, []) or [disease_name]

    # per-entity drug weights
    weights: Dict[str, Dict[str, int]] = {eid: {} for eid in entity_ids}
    seen_order: List[str] = []

    for t in data.get("treatments", []):
        if t.get("disease_name") != disease_name: continue
        eid = t.get("disease_id") or disease_name
        if eid not in weights: weights[eid] = {}
        if eid not in seen_order: seen_order.append(eid)

        for did in t.get("drug_ids", []) or []:
            lbl = clean_chem_label(did)
            weights[eid].setdefault(lbl, 0)

        for ev in t.get("evidence", []) or []:
            lbl = clean_chem_label(ev.get("drug_name") or ev.get("drug_id"))
            w = int(ev.get("total_articles") or len(ev.get("pmids", [])) or 0)
            weights[eid][lbl] = weights[eid].get(lbl, 0) + w

    if seen_order:
        entity_ids = seen_order

    # union of drugs and sort by max per-entity weight then name
    drug_set = {d for per in weights.values() for d in per.keys()}
    def max_w(drug: str) -> int: return max((weights[e].get(drug, 0) for e in entity_ids), default=0)
    drugs = sorted(drug_set, key=lambda d: (-max_w(d), d.lower()))
    return entity_ids, drugs, weights

# ---------- draw ----------
def draw_three_layer_entities_aligned(
    disease_name: str,
    entity_ids: List[str],
    drugs: List[str],
    weights: Dict[str, Dict[str, int]],
    out_png: Path,
    *,
    width_in=20.0, height_in=12.0, dpi=300,
    font_node=12, font_count=12,
    hide_zero_counts=False, col_gap=0.9,
    show_column_headers=True, header_max_chars=28
):
    # layers
    L0, L1, L2 = [disease_name], entity_ids, drugs
    layers = [L0, L1, L2]

    # graph
    G = nx.DiGraph()
    for layer in layers:
        for n in layer: G.add_node(n)

    for eid in L1:
        G.add_edge(disease_name, eid, kind="disease->entity", weight=1)

    ed_edges, ed_vals = [], []
    for eid in L1:
        per = weights.get(eid, {})
        for d in L2:
            w = int(per.get(d, 0))
            G.add_edge(eid, d, kind="entity->drug", weight=w)
            ed_edges.append((eid, d)); ed_vals.append(w)

    # layout
    pos = layered_positions_3(layers, x_gap=2.6, y_pad=0.06)

    # figure
    plt.figure(figsize=(width_in, height_in), dpi=dpi)
    ax = plt.gca(); ax.set_facecolor("white")

    # nodes
    colors = {0: "#4C78A8", 1: "#F58518", 2: "#54A24B"}
    sizes  = {0: 3600,      1: 1700,      2: 1300}
    shapes = {0: "s",       1: "o",       2: "o"}
    for xi, layer in enumerate(layers):
        nx.draw_networkx_nodes(
            G, pos, nodelist=layer,
            node_color=colors[xi], node_size=sizes[xi], node_shape=shapes[xi],
            linewidths=1.0, edgecolors="white"
        )

    nx.draw_networkx_labels(G, pos, font_size=font_node)

    # edges split: zero vs positive weights for clean styling
    de_edges = [(u, v) for u, v, a in G.edges(data=True) if a.get("kind") == "disease->entity"]
    pos_edges = [(u, v) for u, v in ed_edges if G[u][v]["weight"] > 0]
    zero_edges = [(u, v) for u, v in ed_edges if G[u][v]["weight"] == 0]
    width_map, zero_w = scale_widths([G[u][v]["weight"] for u, v in pos_edges], min_w=1.6, max_w=10.0)

    nx.draw_networkx_edges(
        G, pos, edgelist=de_edges, width=2.0, alpha=0.25,
        arrows=True, arrowstyle="-|>", arrowsize=16, edge_color="#6BAED6"
    )
    if pos_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=pos_edges,
            width=[width_map[G[u][v]["weight"]] for u, v in pos_edges],
            alpha=0.75, arrows=True, arrowstyle="-|>", arrowsize=12,
            edge_color="#666666", connectionstyle="arc3,rad=0.18"
        )
    if zero_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=zero_edges,
            width=[zero_w for _ in zero_edges],
            alpha=0.25, arrows=True, arrowstyle="-|>", arrowsize=10,
            edge_color="#BDBDBD", connectionstyle="arc3,rad=0.18"
        )

    # ----- right-side aligned numbers -----
    labels, pos_map = {}, {}
    if L2:
        drug_x = pos[L2[0]][0]
        base_x = drug_x + 0.40  # a bit more room so counts are clear
        for j, eid in enumerate(L1):
            xj = base_x + j * col_gap
            # column header
            if show_column_headers:
                hk = f"HEAD::{eid}"
                labels[hk] = crop_middle(eid, header_max_chars)
                pos_map[hk] = (xj, 1.03)
            # per-row counts
            for d in L2:
                v = int(weights.get(eid, {}).get(d, 0))
                if hide_zero_counts and v == 0: continue
                k = f"{eid}::{d}"
                labels[k] = f"{v:,}"
                pos_map[k] = (xj, pos[d][1])

        nx.draw_networkx_labels(
            G, pos_map, labels=labels,
            font_size=font_count, font_color="#333333",
            horizontalalignment="right", verticalalignment="center"
        )

    # keep counts in frame
    min_x = min(p[0] for p in pos.values())
    max_counts_x = (pos[L2[0]][0] if L2 else 2.6*2) + 0.40 + max(0, (len(L1)-1))*col_gap + 0.2
    ax.set_xlim(min_x - 0.3, max_counts_x)
    ax.set_ylim(-0.05, 1.08)

    plt.title(
        f"{disease_name} → Entity IDs → Drugs  (entity→drug width ∝ total_articles; 0 = listed, no evidence)",
        fontsize=font_node + 2
    )
    plt.axis("off"); plt.tight_layout()
    plt.savefig(out_png); plt.close()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Three-layer disease→entity IDs→drugs with right-side aligned per-entity counts.")
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--disease", type=str, required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--width", type=float, default=20.0)
    ap.add_argument("--height", type=float, default=12.0)
    ap.add_argument("--font-node", type=int, default=12)
    ap.add_argument("--font-count", type=int, default=12)
    ap.add_argument("--hide-zero-counts", type=int, default=0)
    ap.add_argument("--col-gap", type=float, default=0.9)
    ap.add_argument("--show-column-headers", type=int, default=1)
    ap.add_argument("--header-max-chars", type=int, default=28)
    args = ap.parse_args()

    data = load_data(args.json)
    outdir = ensure_outdir(data.get("run_id", "run"))

    entity_ids, drugs, weights = collect_single_disease_entity_specific(data, args.disease)

    out_png = outdir / f"three_layer_entities_aligned_{safe_name(args.disease)}.png"
    draw_three_layer_entities_aligned(
        disease_name=args.disease,
        entity_ids=entity_ids,
        drugs=drugs,
        weights=weights,
        out_png=out_png,
        width_in=args.width, height_in=args.height, dpi=args.dpi,
        font_node=args.font_node, font_count=args.font_count,
        hide_zero_counts=bool(args.hide_zero_counts),
        col_gap=args.col_gap,
        show_column_headers=bool(args.show_column_headers),
        header_max_chars=args.header_max_chars,
    )
    print(f"Wrote {out_png}")

if __name__ == "__main__":
    main()
