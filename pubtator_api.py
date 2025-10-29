# pubtator_api.py
import time, requests
from typing import Any, Dict, List, Tuple, Optional

BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"

# --- simple 2 RPS limiter + shared session + retries ---
_SESSION = requests.Session()
_MIN_INTERVAL = 0.5  # 2 requests/second
_last_ts = 0.0

def _throttle():
    global _last_ts
    dt = time.time() - _last_ts
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)

def _get(path: str, params: Dict[str, Any], timeout: float, retries: int = 2) -> requests.Response:
    url = f"{BASE}{path}"
    for i in range(retries + 1):
        _throttle()
        r = _SESSION.get(url, params=params, timeout=timeout)
        _last_ts = time.time()
        if r.status_code in (429, 500, 502, 503, 504) and i < retries:
            time.sleep(0.5 * (2 ** i))
            continue
        return r  # let caller decide on raise_for_status
    return r

def pubtator_entity_autocomplete(
    query: str,
    concept: Optional[str] = None,
    limit: Optional[int] = None,
    timeout: float = 15.0,
    strict: bool = False,
) -> Tuple[Dict[str, str], int]:
    """({name:id}, total_count)."""
    def to_list(d: Any) -> list:
        if isinstance(d, list): return d
        if isinstance(d, dict):
            if isinstance(d.get("results"), list): return d["results"]
            if isinstance(d.get("data"), list):    return d["data"]
        return []

    base_params: Dict[str, Any] = {"query": query}
    if concept: base_params["concept"] = concept

    r1 = _get("/entity/autocomplete/", base_params, timeout)
    try:
        r1.raise_for_status()
    except requests.HTTPError:
        if strict: raise
        return {}, 0

    d1 = r1.json()
    total_count = len(to_list(d1))

    if limit is not None:
        params2 = dict(base_params); params2["limit"] = int(limit)
        r2 = _get("/entity/autocomplete/", params2, timeout)
        try:
            r2.raise_for_status()
        except requests.HTTPError:
            if strict: raise
            return {}, total_count
        data = to_list(r2.json())
    else:
        data = to_list(d1)

    out: Dict[str, str] = {}
    for it in data:
        name = it.get("label") or it.get("name") or it.get("text")
        ent_id = it.get("id") or it.get("identifier") or it.get("entity_id") or it.get("_id")
        if name and ent_id: out[name] = ent_id
    return out, total_count

def treatment_drugs_for_disease(
    disease_id: str,
    relation_type: str = "treat",
    limit: int = 10,
    timeout: float = 15.0,
    strict: bool = False,
) -> Tuple[Dict[str, List[str]], int]:
    """({disease_id:[@CHEMICAL_*...]}, total_unique_chemicals)."""
    # PubTator expects lowercase entity type here
    params = {"e1": disease_id, "type": relation_type, "e2": "chemical"}
    r = _get("/relations", params, timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        if strict: raise
        return {disease_id: []}, 0

    data = r.json()
    if not isinstance(data, list):
        return {disease_id: []}, 0

    items = [it for it in data if it.get("target") == disease_id and it.get("source")]
    items.sort(key=lambda it: it.get("publications", 0), reverse=True)

    seen, chem_ids = set(), []
    for it in items:
        src = it["source"]
        if src not in seen:
            seen.add(src); chem_ids.append(src)
            if limit and len(chem_ids) >= limit: break

    count = len({it["source"] for it in items})
    return {disease_id: chem_ids[:limit] if limit else chem_ids}, count

def search_treatment_evidence(
    disease_id: str,
    chemical_id: str,
    page: int = 1,
    page_size: Optional[int] = None,
    timeout: float = 15.0,
    strict: bool = False,
) -> Tuple[List[Dict], int]:
    """(results, total_count) for relations:ANY|chemical_id|disease_id."""
    q = f"relations:ANY|{chemical_id}|{disease_id}"
    params: Dict[str, Any] = {"text": q, "page": page}
    if page_size is not None: params["page_size"] = page_size
    r = _get("/search/", params, timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        # common 400 causes: invalid IDs like @CHEMICAL_Steroids or too-generic disease terms
        if strict: raise
        return [], 0
    j = r.json()
    return j.get("results", []), int(j.get("count", 0))


if __name__ == "__main__":
    # Example 1: map disease names -> IDs
    disease_map = pubtator_entity_autocomplete("breast cancer", concept="Disease", limit=10)
    print("Disease name -> ID mapping:")
    print(json.dumps(disease_map, indent=2))

    # Example 2: chemicals that treat this disease (normalized + raw)
    # for name, disease_id in disease_map.items():
    #     print(f"\nChemicals that treat {name} ({disease_id}):")
    #     out = treatment_drugs_for_disease(disease_id=disease_id, limit=5)
    #     print(f"\nChemicals that treat {name} ({disease_id}):")
    #     print(json.dumps(out, indent=2))
    #     time.sleep(0.5)

    #Example 3: evidence for a specific disease-chemical pair
    print(json.dumps(
        search_treatment_evidence(
            disease_id="@DISEASE_Triple_Negative_Breast_Neoplasms",  # breast cancer
            chemical_id="@CHEMICAL_Paclitaxel"
        ),indent=2
    ))