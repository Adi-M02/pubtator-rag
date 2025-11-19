#!/usr/bin/env python3
import sys, json, csv, pathlib
from typing import Dict, Any, List, Tuple, Set

try:
    import networkx as nx
except Exception:
    nx = None

def _pretty(eid: str) -> str:
    return eid.split("_", 1)[1].replace("_", " ") if isinstance(eid, str) and eid.startswith("@") and "_" in eid else str(eid)

def load_artifact(p: pathlib.Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def build_graph(artifact: Dict[str, Any]):
    # Undirected graph
    G = nx.Graph() if nx else None

    # We will collapse by drug_name (the human input/LLM result), not by chemical_id.
    # Aggregate across all indications (each indication corresponds to one chemical_id).
    nodes: Dict[str, Dict[str, Any]] = {}  # node_id -> attrs
    # key: (drug_name, disease_id)
    edges: Dict[Tuple[str, str], Dict[str, Any]] = {}

    indications: List[Dict[str, Any]] = artifact.get("indications", [])
    for ind in indications:
        drug_name = ind["drug_name"]                     # collapse key
        drug_node_id = drug_name                         # node id = the name itself
        drug_label = drug_name                           # label = name for readability

        # ensure drug node
        if drug_node_id not in nodes:
            nodes[drug_node_id] = {"id": drug_node_id, "label": drug_label, "type": "drug"}

        for ev in ind.get("evidence", []):
            dis_id = ev["disease_id"]
            dis_label = _pretty(dis_id)

            # ensure disease node
            if dis_id not in nodes:
                nodes[dis_id] = {"id": dis_id, "label": dis_label, "type": "disease"}

            key = (drug_node_id, dis_id)
            if key not in edges:
                edges[key] = {
                    "u": drug_node_id,
                    "v": dis_id,
                    "relation": "treat",
                    "pmids": set(),                 # unique union across all chem_ids for this drug
                    "pmid_count": 0,
                    "total_articles": 0,            # sum across chem_ids for this pair
                    "chem_ids": set(),              # which chemical_ids contributed
                }

            # aggregate evidence
            pmids_here: List[str] = [str(p) for p in ev.get("pmids", []) if p]
            edges[key]["pmids"].update(pmids_here)
            edges[key]["pmid_count"] = len(edges[key]["pmids"])
            try:
                edges[key]["total_articles"] += int(ev.get("total_articles", 0))
            except Exception:
                pass

            # record which chemical_id this indication came from (for traceability)
            chem_id = ind.get("drug_id")
            if chem_id:
                edges[key]["chem_ids"].add(str(chem_id))

    # Decide output folder
    run_dir = pathlib.Path(artifact.get("run_dir") or ".")
    if not run_dir.exists():
        run_dir = pathlib.Path(".")
    nodes_csv = run_dir / "nodes.csv"
    edges_csv = run_dir / "edges.csv"
    graphml_path = run_dir / "graph.graphml"

    # Write nodes.csv
    with open(nodes_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "type"])
        for n in nodes.values():
            w.writerow([n["id"], n["label"], n["type"]])

    # Write edges.csv (undirected)
    with open(edges_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "node_u", "node_v", "relation",
            "pmid_count", "total_articles",
            "pmids", "chem_ids"
        ])
        for e in edges.values():
            pmids_str = "|".join(sorted(e["pmids"]))
            chem_ids_str = "|".join(sorted(e["chem_ids"]))
            w.writerow([
                e["u"], e["v"], e["relation"],
                e["pmid_count"], e["total_articles"],
                pmids_str, chem_ids_str
            ])

    # Optional GraphML
    if nx:
        for n in nodes.values():
            G.add_node(n["id"], label=n["label"], type=n["type"])
        for e in edges.values():
            G.add_edge(
                e["u"], e["v"],
                relation=e["relation"],
                pmid_count=e["pmid_count"],
                total_articles=e["total_articles"],
                pmids="|".join(sorted(e["pmids"])),
                chem_ids="|".join(sorted(e["chem_ids"])),
            )
        nx.write_graphml(G, graphml_path)

    return {
        "nodes_csv": str(nodes_csv),
        "edges_csv": str(edges_csv),
        "graphml": str(graphml_path) if nx else None
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: build_graph_from_json.py <pipeline_json_path>")
        sys.exit(1)
    json_path = pathlib.Path(sys.argv[1]).resolve()
    artifact = load_artifact(json_path)
    # Ensure outputs land in the run folder
    artifact["run_dir"] = json_path.parent.as_posix()
    out = build_graph(artifact)
    print(out)

if __name__ == "__main__":
    main()
