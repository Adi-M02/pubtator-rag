#!/usr/bin/env python3
import os, json, time, logging, pathlib, csv, re
from datetime import datetime, timedelta
from typing import Dict, List, Any

# os.environ.setdefault("OLLAMA_HOST", "http://localhost:11438")
# import ollama
# MODEL = os.getenv("OLLAMA_MODEL", "llama3.3:latest")

from pubtator_api import (
    pubtator_entity_autocomplete,
    treatment_diseases_for_drug,
    search_treatment_evidence,
)

OUTDIR = pathlib.Path("outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTDIR / f"pipeline_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
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
    if isinstance(eid, str) and eid.startswith("@") and "_" in eid:
        return eid.split("_", 1)[1].replace("_", " ")
    return str(eid)

def _nrm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def _chem_id_for(name: str) -> str:
    return "@CHEMICAL_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

HARDCODED_DRUGS = ["Atorvastatin", "Metformin", "Levothyroxine", "Lisinopril", "Amlodipine", "Metoprolol", "Albuterol", "Losartan", "Gabapentin", "Omeprazole", "Sertraline", "Rosuvastatin", "Pantoprazole", "Escitalopram", "Dextroamphetamine", "Hydrochlorothiazide", "Bupropion", "Fluoxetine", "Semaglutide", "Montelukast", "Trazodone", "Simvastatin", "Amoxicillin", "Tamsulosin", "Hydrocodone", "Fluticasone", "Meloxicam", "Apixaban", "Furosemide", "Insulin Glargine", "Duloxetine", "Ibuprofen", "Famotidine", "Empagliflozin", "Carvedilol", "Tramadol", "Alprazolam", "Prednisone", "Hydroxyzine", "Buspirone", "Clopidogrel", "Glipizide", "Citalopram", "Potassium Chloride", "Allopurinol", "Aspirin", "Cyclobenzaprine", "Ergocalciferol", "Oxycodone", "Methylphenidate", "Venlafaxine", "Spironolactone", "Ondansetron", "Zolpidem", "Cetirizine", "Estradiol", "Pravastatin", "Lamotrigine", "Quetiapine", "Salmeterol", "Clonazepam", "Dulaglutide", "Azithromycin", "Clavulanate", "Latanoprost", "Cholecalciferol", "Propranolol", "Ezetimibe", "Topiramate", "Paroxetine", "Diclofenac", "Formoterol", "Atenolol", "Lisdexamfetamine", "Doxycycline", "Pregabalin", "Norethindrone", "Glimepiride", "Tizanidine", "Clonidine", "Fenofibrate", "Insulin Lispro", "Valsartan", "Cephalexin", "Baclofen", "Rivaroxaban", "Ferrous Sulfate", "Amitriptyline", "Finasteride", "Dapagliflozin", "Folic Acid", "Aripiprazole", "Olmesartan", "Norgestimate", "Valacyclovir", "Mirtazapine", "Lorazepam", "Levetiracetam", "Insulin Aspart", "Naproxen", "Cyanocobalamin", "Loratadine", "Diltiazem", "Sumatriptan", "Triamcinolone", "Hydralazine", "Tirzepatide", "Celecoxib", "Acetaminophen", "Alendronate", "Oxybutynin", "Triamterene", "Warfarin", "Progesterone", "Vilanterol", "Testosterone", "Nifedipine", "Methocarbamol", "Benzonatate", "Sitagliptin", "Chlorthalidone", "Isosorbide", "Donepezil", "Dexmethylphenidate", "Sulfamethoxazole", "Clobetasol", "Methotrexate", "Hydroxychloroquine", "Lovastatin", "Pioglitazone", "Irbesartan", "Methylprednisolone", "Ethinyl Estradiol", "Meclizine", "Levonorgestrel", "Ketoconazole", "Thyroid", "Azelastine", "Nitrofurantoin", "Adalimumab", "Memantine", "Prednisolone", "Esomeprazole", "Docusate", "Clindamycin", "Acyclovir", "Sildenafil", "Ciprofloxacin", "Levocetirizine", "Valproate" ]

def resolve_chemical_ids(drug_name: str, limit: int = 10):
    m1, t1 = pubtator_entity_autocomplete(drug_name, concept="CHEMICAL", limit=limit)
    if not m1:
        m1, t1 = pubtator_entity_autocomplete(drug_name.lower(), concept="CHEMICAL", limit=limit)
        if not m1:
            return [], 0

    items = list(m1.items())
    qn = _nrm(drug_name)
    base_id = _chem_id_for(drug_name)

    pref = []

    for k, v in items:
        if _nrm(k) == qn and (k, v) not in pref:
            pref.append((k, v))

    for k, v in items:
        if v == base_id and (k, v) not in pref:
            pref.append((k, v))

    for k, v in items:
        if qn in _nrm(k) and (k, v) not in pref:
            pref.append((k, v))

    for kv in items:
        if kv not in pref:
            pref.append(kv)

    ids = [v for _, v in pref][:limit]
    return ids, t1

# def ask_llm_for_drugs(n=10) -> List[Dict[str, str]]:
#     resp = ollama.chat(
#         model=MODEL,
#         messages=[
#             {"role": "system", "content": "Output JSON only."},
#             {
#                 "role": "user",
#                 "content": (
#                     f"List exactly {n} distinct prescription-only generic drugs that are commonly prescribed "
#                     f"and primarily treat diseases with severe implications. For each, include one example "
#                     f"disease it treats. Return JSON exactly as "
#                     f'{{"drugs":[{{"drug":"...","example_disease":"..."}}]}}. '
#                     f"No brand names. No OTC. Examples formatted like: albuterol / bronchitis."
#                 ),
#             },
#         ],
#         format="json",
#         options={"temperature": 0},
#     )
#     txt = (resp.get("message", {}) or {}).get("content", "").strip()
#     if not txt:
#         raise RuntimeError("LLM returned empty content")
#     try:
#         data = json.loads(txt)
#     except json.JSONDecodeError:
#         s = txt.strip()
#         i, j = s.find("{"), s.rfind("}")
#         if i == -1 or j == -1 or j <= i:
#             raise
#         data = json.loads(s[i : j + 1])

#     items = []
#     for it in data.get("drugs", []):
#         d = (it.get("drug") or "").strip()
#         ex = (it.get("example_disease") or "").strip()
#         if d:
#             items.append({"drug": d, "example_disease": ex})

#     seen, uniq = set(), []
#     for obj in items:
#         key = json.dumps(obj, sort_keys=True)
#         if key not in seen:
#             seen.add(key)
#             uniq.append(obj)
#     if len(uniq) != n:
#         raise RuntimeError(f"LLM did not return exactly {n} items, got {len(uniq)}")
#     return uniq[:n]

def main():
    log.info(f"Run start: {START_STR}")

    log.info("Stage 1: using hardcoded drug list")
    drugs_items = [{"drug": d, "example_disease": ""} for d in HARDCODED_DRUGS]
    drugs = [d["drug"] for d in drugs_items]
    log.info(f"  drugs: {drugs}")

    log.info("\nStage 2: resolving to PubTator CHEMICAL IDs (≤10 per drug)")
    drug_entities: List[Dict[str, Any]] = []
    for it in drugs_items:
        name = it["drug"]
        ex_dis = it.get("example_disease", "")
        ids, total = resolve_chemical_ids(name, limit=10)
        if not ids:
            log.info(f"  drop (unresolved): {name}")
            continue
        drug_entities.append(
            {
                "drug_name": name,
                "example_disease": ex_dis,
                "entity_ids": ids,
                "total_entity_ids": int(total),
            }
        )
        log.info(f"  {name}: {ids} (total={total})")

    log.info("\nStage 3: fetching treated DISEASEs (top 25 by publications)")
    indications: List[Dict[str, Any]] = []
    dropped_no_rel: List[Dict[str, Any]] = []
    for de in drug_entities:
        dname = de["drug_name"]
        ex_dis = de["example_disease"]
        for cid in de["entity_ids"]:
            rel_map, dis_total = treatment_diseases_for_drug(
                cid, relation_type="treat", limit=25
            )
            diseases = rel_map.get(cid, [])
            if dis_total == 0 or not diseases:
                dropped_no_rel.append({"drug_name": dname, "drug_id": cid})
                log.info(f"  drop (no treat relations): {dname} ({cid})")
                continue
            log.info(f"  {dname} ({cid}): {len(diseases)} diseases (total={dis_total})")
            indications.append(
                {
                    "drug_name": dname,
                    "example_disease": ex_dis,
                    "drug_id": cid,
                    "disease_ids": diseases,
                    "total_disease_entities": int(dis_total),
                    "evidence": [],
                }
            )

    log.info("\nStage 4: fetching PMIDs per drug disease (first page)")
    for ind in indications:
        cid = ind["drug_id"]
        for did in ind["disease_ids"]:
            try:
                results, article_total = search_treatment_evidence(
                    disease_id=did,
                    chemical_id=cid,
                    page=1,
                )
            except Exception as e:
                log.warning(
                    f"search_treatment_evidence failed for {cid} ~ {did}: {e}. "
                    f"Recording zero PMIDs."
                )
                results, article_total = [], 0

            pmids = _uniq([str(r.get("pmid")) for r in results if r.get("pmid")])[:50]
            ind["evidence"].append(
                {
                    "disease_id": did,
                    "disease_name": _pretty(did),
                    "pmids": pmids,
                    "total_articles": int(article_total),
                }
            )
            head = ", ".join(pmids[:10]) + ("..." if len(pmids) > 10 else "")
            log.info(
                f"  {_pretty(cid)} ~ {_pretty(did)}: {len(pmids)} PMIDs "
                f"(total={article_total}) {head if head else ''}"
            )

    log.info("\nChat-style outputs:")
    for ind in indications:
        for ev in ind["evidence"]:
            if ev["pmids"]:
                log.info(
                    f"{ind['drug_name']} treats {ev['disease_name']}. "
                    f"PMIDs: {', '.join(ev['pmids'])}"
                )

    artifact = {
        "run_id": RUN_ID,
        "drugs": [d["drug"] for d in drugs_items],
        "drug_entities": drug_entities,
        "indications": indications,
        "dropped_no_relations": dropped_no_rel,
        "started_at": START_STR,
        "hardcoded_drugs": True,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "drug_name",
                "drug_id",
                "disease_name",
                "disease_id",
                "pmid_count",
                "total_articles",
            ]
        )
        for ind in indications:
            for ev in ind["evidence"]:
                w.writerow(
                    [
                        ind["drug_name"],
                        ind["drug_id"],
                        ev["disease_name"],
                        ev["disease_id"],
                        len(ev["pmids"]),
                        ev["total_articles"],
                    ]
                )

    end_ts = time.time()
    end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = timedelta(seconds=round(end_ts - START_TS))
    log.info(f"\nSaved JSON: {JSON_PATH}")
    log.info(f"Saved CSV:  {CSV_PATH}")
    log.info(f"Saved log:  {LOG_PATH}")
    log.info(f"Run end:   {end_str} | Total runtime: {elapsed}")

if __name__ == "__main__":
    main()
