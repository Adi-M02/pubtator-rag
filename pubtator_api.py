#!/usr/bin/env python3
"""
pubtator_simple_csv.py

Outputs 3 CSVs in --outdir:
  1) disease_entities.csv        # entity IDs returned for the disease query
  2) treat_relations.csv         # related entity IDs via type='treat' (all types)
  3) search_results.csv          # PMIDs for CHEMICAL–DISEASE treat pairs

Usage:
  python pubtator_simple_csv.py --disease "breast cancer" --outdir out
"""

import argparse
import csv
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
HEADERS = {"User-Agent": "adi-pubtator3-demo/mini/1.0"}
SLEEP_SEC = 0.4  # stay under 3 req/sec

def _sleep():
    time.sleep(SLEEP_SEC)

def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.75 * (attempt + 1))
    raise RuntimeError("unreachable")

def nice_name(eid: Optional[str]) -> str:
    if not eid or "@" not in eid: return eid or ""
    tail = eid.split("@", 1)[1]
    parts = tail.split("_", 1)
    return (parts[1] if len(parts) == 2 else tail).replace("_", " ")

def etype(eid: Optional[str]) -> str:
    if not eid or "@" not in eid: return "unknown"
    return eid.split("@", 1)[1].split("_", 1)[0].lower()

def autocomplete_diseases(query: str, limit: int) -> List[Tuple[str, str]]:
    data = _get(f"{BASE}/entity/autocomplete/", {"query": query, "concept": "DISEASE", "limit": limit})
    _sleep()
    out: List[Tuple[str, str]] = []
    if isinstance(data, list):
        for item in data:
            eid = item.get("_id") or item.get("entity_id") or item.get("id")
            if eid and eid.startswith("@DISEASE_"):
                out.append((eid, item.get("name") or nice_name(eid)))
    # dedup keep order
    seen = set(); uniq = []
    for eid, nm in out:
        if eid not in seen:
            uniq.append((eid, nm)); seen.add(eid)
    return uniq

def find_related_treat(e1: str) -> List[Tuple[str, str, str]]:
    data = _get(f"{BASE}/relations", {"e1": e1, "type": "treat"})
    _sleep()
    items = data.get("results") if isinstance(data, dict) else (data if isinstance(data, list) else [])
    out: List[Tuple[str, str, str]] = []
    for obj in (items or []):
        if isinstance(obj, str):
            rid = obj; out.append((rid, nice_name(rid), etype(rid)))
        elif isinstance(obj, dict):
            rid = obj.get("_id") or obj.get("entity_id") or obj.get("id") or obj.get("e2") or obj.get("target")
            if rid:
                out.append((rid, obj.get("name") or nice_name(rid), obj.get("type") or etype(rid)))
    # dedup by id
    seen = set(); uniq = []
    for rid, rname, rtype in out:
        if rid not in seen:
            uniq.append((rid, rname, rtype)); seen.add(rid)
    return uniq

def search_pmids_for_treat_pair(chem_id: str, disease_id: str, page: int, cap: int) -> List[int]:
    # try relation form first
    q1 = f"relations:treat|{chem_id}|{disease_id}"
    data = _get(f"{BASE}/search/", {"text": q1, "page": page})
    _sleep()
    results = data.get("results", data if isinstance(data, list) else [])
    pmids: List[int] = []
    for rec in (results or []):
        pmid = rec.get("pmid") if isinstance(rec, dict) else None
        if isinstance(pmid, int):
            pmids.append(pmid)
        elif isinstance(pmid, str) and pmid.isdigit():
            pmids.append(int(pmid))
        if len(pmids) >= cap: return pmids
    if pmids:
        return pmids
    # fallback to AND query
    q2 = f"{chem_id} AND {disease_id}"
    data = _get(f"{BASE}/search/", {"text": q2, "page": page})
    _sleep()
    results = data.get("results", data if isinstance(data, list) else [])
    for rec in (results or []):
        pmid = rec.get("pmid") if isinstance(rec, dict) else None
        if isinstance(pmid, int):
            pmids.append(pmid)
        elif isinstance(pmid, str) and pmid.isdigit():
            pmids.append(int(pmid))
        if len(pmids) >= cap: break
    return pmids

def write_csv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", default="breast cancer")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--candidates", type=int, default=15)
    ap.add_argument("--pmids-per-pair", type=int, default=20)
    ap.add_argument("--page", type=int, default=1)
    args = ap.parse_args()

    # 1) disease entity IDs
    diseases = autocomplete_diseases(args.disease, limit=args.candidates)
    write_csv(
        os.path.join(args.outdir, "disease_entities.csv"),
        ["disease_query", "disease_entity_id", "disease_name"],
        [[args.disease, did, dname] for did, dname in diseases],
    )
    if not diseases:
        print("No disease entity IDs found.")
        return
    print(f"Saved disease_entities.csv with {len(diseases)} rows")

    # 2) treat relations (all types)
    rel_rows: List[List[Any]] = []
    for did, dname in diseases:
        related = find_related_treat(did)
        for rid, rname, rtype in related:
            rel_rows.append([did, dname, rid, rname, rtype])
    write_csv(
        os.path.join(args.outdir, "treat_relations.csv"),
        ["disease_entity_id", "disease_name", "related_entity_id", "related_name", "related_type"],
        rel_rows,
    )
    print(f"Saved treat_relations.csv with {len(rel_rows)} rows")

    # 3) search results for CHEMICAL–DISEASE treat pairs
    search_rows: List[List[Any]] = []
    for did, dname in diseases:
        related = find_related_treat(did)
        for rid, rname, rtype in related:
            if rtype != "chemical":
                continue
            pmids = search_pmids_for_treat_pair(rid, did, page=args.page, cap=args.pmids_per_pair)
            for pmid in pmids:
                search_rows.append([did, dname, rid, rname, pmid])
    write_csv(
        os.path.join(args.outdir, "search_results.csv"),
        ["disease_entity_id", "disease_name", "chemical_entity_id", "chemical_name", "pmid"],
        search_rows,
    )
    print(f"Saved search_results.csv with {len(search_rows)} rows")
    print("Done.")

if __name__ == "__main__":
    main()
