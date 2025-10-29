#!/usr/bin/env python3
import os, json, time, logging, pathlib
from datetime import datetime, timedelta
from typing import Dict, List, Any

os.environ["OLLAMA_HOST"] = "http://localhost:11438"
import ollama

from pubtator_api import (
    pubtator_entity_autocomplete,
    treatment_drugs_for_disease,
    search_treatment_evidence,
)

MODEL = os.getenv("OLLAMA_MODEL", "llama3.3:latest")
OUTDIR = pathlib.Path("outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
LOG_PATH = OUTDIR / f"pipeline_{RUN_ID}.log"
JSON_PATH = OUTDIR / f"pipeline_{RUN_ID}.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, encoding="utf-8")]
)
log = logging.getLogger("pipeline")

START_TS = time.time()
START_STR = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _uniq(xs): return list(dict.fromkeys(xs))
def _pretty(eid: str) -> str:
    return eid.split("_", 1)[1].replace("_", " ") if isinstance(eid, str) and eid.startswith("@") and "_" in eid else str(eid)

def ask_llm_for_diseases(n=10) -> List[str]:
    msg = [
        {"role": "system", "content": "Output JSON only."},
        {"role": "user", "content": f"List exactly {n} distinct human diseases by name. JSON: {{\"diseases\":[\"...\"]}}."}
    ]
    out = ollama.chat(model=MODEL, messages=msg, options={"temperature": 0}).get("message", {}).get("content", "")
    try:
        data = json.loads(out)
        return _uniq([d.strip() for d in data.get("diseases", []) if isinstance(d, str) and d.strip()])[:n]
    except Exception:
        return []

def main():
    log.info(f"Run start: {START_STR}")

    log.info("Stage 1: asking LLM for 10 diseases")
    diseases = ask_llm_for_diseases(10)
    log.info(f"  diseases: {diseases}")

    log.info("\nStage 2: resolving to PubTator DISEASE IDs (≤10 per disease)")
    disease_entities: List[Dict[str, Any]] = []
    for d in diseases:
        name_to_id, total_count = pubtator_entity_autocomplete(d, concept="DISEASE", limit=10)
        ids = _uniq(list(name_to_id.values()))[:10]
        disease_entities.append({"disease_name": d, "entity_ids": ids, "total_entity_ids": int(total_count)})
        log.info(f"  {d}: {ids} (total={total_count})")

    log.info("\nStage 3: fetching treating CHEMICALs (top 10 by publications)")
    treatments: List[Dict[str, Any]] = []
    for de in disease_entities:
        dname = de["disease_name"]
        for did in de["entity_ids"]:
            rel_map, drug_total = treatment_drugs_for_disease(did, relation_type="treat", limit=10)
            drugs = rel_map.get(did, [])
            log.info(f"  {dname} ({did}): {len(drugs)} drugs (total={drug_total})")
            treatments.append({
                "disease_name": dname,
                "disease_id": did,
                "drug_ids": drugs,
                "total_drug_entities": int(drug_total),
                "evidence": []
            })

    log.info("\nStage 4: fetching PMIDs per disease–drug")
    for t in treatments:
        did = t["disease_id"]
        for cid in t["drug_ids"]:
            results, article_total = search_treatment_evidence(disease_id=did, chemical_id=cid)
            pmids = _uniq([str(r.get("pmid")) for r in results if r.get("pmid")])
            t["evidence"].append({
                "drug_id": cid,
                "drug_name": _pretty(cid),
                "pmids": pmids,
                "total_articles": int(article_total),
            })
            head = ", ".join(pmids[:10]) + ("..." if len(pmids) > 10 else "")
            log.info(f"  {_pretty(did)} ~ {_pretty(cid)}: {len(pmids)} PMIDs (total={article_total}) {head if head else ''}")
            if article_total == 0 and not pmids:
                log.info(f"    note: no evidence returned for pair (possible invalid IDs or too-generic terms).")

    log.info("\nChat-style outputs:")
    for t in treatments:
        for ev in t["evidence"]:
            if ev["pmids"]:
                log.info(f"{t['disease_name']} is treated by {ev['drug_name']} and these PMIDs show the relation: {', '.join(ev['pmids'])}")

    artifact = {
        "run_id": RUN_ID,
        "diseases": diseases,
        "disease_entities": disease_entities,
        "treatments": treatments,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    log.info(f"\nSaved JSON: {JSON_PATH}")
    log.info(f"Saved log:  {LOG_PATH}")

    end_ts = time.time()
    end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = timedelta(seconds=round(end_ts - START_TS))
    log.info(f"Run end:   {end_str} | Total runtime: {elapsed}")

if __name__ == "__main__":
    main()
