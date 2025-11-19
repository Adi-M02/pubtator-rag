#!/usr/bin/env python3
import os, json, time, logging, pathlib, csv, re
from datetime import datetime, timedelta
from typing import Dict, List, Any

from pubtator_api import (
    pubtator_entity_autocomplete,
    treatment_diseases_for_drug,
    search_treatment_evidence,
)

# ---- caps (keep your 200 limits) ----
MAX_CHEM_IDS   = int(os.getenv("MAX_CHEM_IDS", "200"))
MAX_DISEASES   = int(os.getenv("MAX_DISEASES", "200"))
MAX_PMIDS     = 10

OUTDIR = pathlib.Path("outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTDIR / f"pipeline_{RUN_ID}"; RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RUN_DIR / f"pipeline_{RUN_ID}.log"
JSON_PATH = RUN_DIR / f"pipeline_{RUN_ID}.json"
CSV_PATH  = RUN_DIR / f"pipeline_{RUN_ID}_summary.csv"

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
def _nrm(s: str) -> str: return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()
def _chem_id_for(name: str) -> str: return "@CHEMICAL_" + re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')

_BAD_TOKENS = {"sulfone","glucuronide","metabolite","hydroxy","methyl","oxide","phosphate","lactate","acetate","nitrate","sulfate","salt"}

def resolve_chemical_ids(drug_name: str, limit: int = MAX_CHEM_IDS):
    # Avoid passing an oversized 'limit' to PubTator; fetch once, clamp client-side.
    m1, t1 = pubtator_entity_autocomplete(drug_name, concept="CHEMICAL", limit=None)
    if not m1:
        m1, t1 = pubtator_entity_autocomplete(drug_name.lower(), concept="CHEMICAL", limit=None)
        if not m1: return [], 0

    items = list(m1.items())
    qn = _nrm(drug_name); base_id = _chem_id_for(drug_name)

    pref = []
    for k,v in items:
        if _nrm(k) == qn and (k,v) not in pref: pref.append((k,v))
    for k,v in items:
        if v == base_id and (k,v) not in pref: pref.append((k,v))
    for k,v in items:
        kn = _nrm(k)
        if qn in kn and not any(b in kn for b in _BAD_TOKENS) and (k,v) not in pref:
            pref.append((k,v))
    for kv in items:
        if kv not in pref: pref.append(kv)

    ids = [v for _,v in pref][:limit]
    return ids, t1

# Hardcoded drugs (use this list; LLM disabled)
HARDCODED_DRUGS = [
    "metformin",
    "atenolol",
    "simvastatin",
    "lisinopril",
    "amlodipine",
    "omeprazole",
    "furosemide",
    "warfarin",
    "levofloxacin",
    "glyburide",
]

def main():
    log.info(f"Run start: {START_STR}")

    # Stage 1: hardcoded drugs
    log.info("Stage 1: using hardcoded drug list")
    drugs_items = [{"drug": d, "example_disease": ""} for d in HARDCODED_DRUGS]
    drugs = [d["drug"] for d in drugs_items]
    log.info(f"  drugs: {drugs}")

    # Stage 2: CHEMICAL IDs
    log.info(f"\nStage 2: resolving to PubTator CHEMICAL IDs (≤{MAX_CHEM_IDS} per drug)")
    drug_entities: List[Dict[str, Any]] = []
    for it in drugs_items:
        name = it["drug"]; ex_dis = it.get("example_disease","")
        ids, total = resolve_chemical_ids(name, limit=MAX_CHEM_IDS)
        if not ids:
            log.info(f"  drop (unresolved): {name}")
            continue
        drug_entities.append({
            "drug_name": name,
            "example_disease": ex_dis,
            "entity_ids": ids,
            "total_entity_ids": int(total)
        })
        preview = ", ".join(ids[:3]) + ("..." if len(ids) > 3 else "")
        log.info(f"  {name}: {len(ids)} IDs (total={total}) -> {preview}")

    # Stage 3: relations
    log.info(f"\nStage 3: fetching treated DISEASEs (top {MAX_DISEASES} by publications)")
    indications: List[Dict[str, Any]] = []
    dropped_no_rel: List[Dict[str, Any]] = []
    for de in drug_entities:
        dname = de["drug_name"]; ex_dis = de["example_disease"]
        for cid in de["entity_ids"]:
            rel_map, dis_total = treatment_diseases_for_drug(cid, relation_type="treat", limit=MAX_DISEASES)
            diseases = rel_map.get(cid, [])
            if dis_total == 0 or not diseases:
                dropped_no_rel.append({"drug_name": dname, "drug_id": cid})
                log.info(f"  drop (no treat relations): {dname} ({cid})")
                continue
            log.info(f"  {dname} ({cid}): {len(diseases)} diseases (total={dis_total})")
            indications.append({
                "drug_name": dname,
                "example_disease": ex_dis,
                "drug_id": cid,
                "disease_ids": diseases[:MAX_DISEASES],
                "total_disease_entities": int(dis_total),
                "evidence": []
            })

    # Stage 4: evidence (paged, 10 PMIDs per page)
    log.info(f"\nStage 4: fetching PMIDs per drug–disease (up to {MAX_PMIDS} per pair across pages)")
    for ind in indications:
        cid = ind["drug_id"]
        for did in ind["disease_ids"]:
            all_pmids, total_articles, page = [], 0, 1
            while True:
                results, count = search_treatment_evidence(disease_id=did, chemical_id=cid, page=page)
                if page == 1: total_articles = int(count)
                pmids_page = [str(r.get("pmid")) for r in results if r.get("pmid")]
                all_pmids.extend(pmids_page)
                if not pmids_page or len(all_pmids) >= MAX_PMIDS or (page * 10) >= total_articles:
                    break
                page += 1
            all_pmids = _uniq(all_pmids)[:MAX_PMIDS]
            ind["evidence"].append({
                "disease_id": did,
                "disease_name": _pretty(did),
                "pmids": all_pmids,
                "total_articles": int(total_articles),
                "pages_fetched": page
            })
            head = ", ".join(all_pmids[:10]) + ("..." if len(all_pmids) > 10 else "")
            log.info(f"  {_pretty(cid)} ~ {_pretty(did)}: {len(all_pmids)} PMIDs (total={total_articles}) {head if head else ''}")

    # Chat-style log
    log.info("\nChat-style outputs:")
    for ind in indications:
        for ev in ind["evidence"]:
            if ev["pmids"]:
                log.info(f"{ind['drug_name']} treats {ev['disease_name']}. PMIDs: {', '.join(ev['pmids'])}")

    artifact = {
        "run_id": RUN_ID,
        "drugs": [d["drug"] for d in drugs_items],
        "drug_entities": drug_entities,
        "indications": indications,
        "dropped_no_relations": dropped_no_rel,
        "started_at": START_STR,
        "hardcoded_drugs": True,
        "limits": {
            "max_chem_ids_per_drug": MAX_CHEM_IDS,
            "max_diseases_per_entity": MAX_DISEASES,
            "max_pmids_per_pair": MAX_PMIDS
        }
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["drug_name","drug_id","disease_name","disease_id","pmid_count","total_articles"])
        for ind in indications:
            for ev in ind["evidence"]:
                w.writerow([ind["drug_name"], ind["drug_id"], ev["disease_name"], ev["disease_id"],
                            len(ev["pmids"]), ev["total_articles"]])

    end_ts = time.time()
    end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = timedelta(seconds=round(end_ts - START_TS))
    log.info(f"\nSaved JSON: {JSON_PATH}")
    log.info(f"Saved CSV:  {CSV_PATH}")
    log.info(f"Saved log:  {LOG_PATH}")
    log.info(f"Run end:   {end_str} | Total runtime: {elapsed}")

if __name__ == "__main__":
    main()
