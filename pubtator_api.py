# pubtator_api.py
import time, requests, random
from typing import Any, Dict, List, Tuple, Optional

BASE = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"

_SESSION = requests.Session()
_MIN_INTERVAL = 1.0 / 3.0  # 3 requests/second
_last_ts = 0.0

def _throttle():
    global _last_ts
    dt = time.time() - _last_ts
    if dt < _MIN_INTERVAL:
        time.sleep((_MIN_INTERVAL - dt) + 0.05 * random.random())

def _get(path: str, params: Dict[str, Any], timeout: float, retries: int = 3) -> requests.Response:
    url = f"{BASE}{path}"
    backoff = 0.5
    for i in range(retries + 1):
        try:
            _throttle()
            r = _SESSION.get(url, params=params, timeout=timeout)
            _last_ts = time.time()
            if r.status_code in (429, 500, 502, 503, 504) and i < retries:
                time.sleep(backoff); backoff *= 2
                continue
            return r
        except requests.RequestException:
            if i < retries:
                time.sleep(backoff); backoff *= 2
                continue
            raise

def _same_id(a, b) -> bool:
    return str(a or "").lower() == str(b or "").lower()

def pubtator_entity_autocomplete(
    query: str,
    concept: Optional[str] = None,
    limit: Optional[int] = None,
    timeout: float = 15.0,
    strict: bool = False,
) -> Tuple[Dict[str, str], int]:
    """Return ({name:id}, total_count)."""

    def to_list(d: Any) -> List[Dict[str, Any]]:
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
        if name and ent_id:
            out[name] = ent_id
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

def treatment_diseases_for_drug(
    chemical_id: str,
    relation_type: str = "treat",
    limit: Optional[int] = None,
    timeout: float = 15.0,
    strict: bool = False,
) -> Tuple[Dict[str, List[str]], int]:
    """({chemical_id:[@DISEASE_*...]}, total_unique_diseases)"""
    params = {"e1": chemical_id, "type": relation_type, "e2": "disease"}  # e1 EXACT, e2 lowercase
    r = _get("/relations", params, timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        if strict: raise
        return {chemical_id: []}, 0

    data = r.json()
    if not isinstance(data, list):
        return {chemical_id: []}, 0

    def _same_id(a, b): return str(a or "").lower() == str(b or "").lower()

    # For chemical e1, rows have source==chemical_id and target==@DISEASE_*
    items = [it for it in data
             if _same_id(it.get("source"), chemical_id)
             and isinstance(it.get("target"), str)
             and it["target"].lower().startswith("@disease_")]

    items.sort(key=lambda it: it.get("publications", 0), reverse=True)

    seen, dis_ids = set(), []
    for it in items:
        tgt = it["target"].strip()
        k = tgt.lower()
        if k not in seen:
            seen.add(k)
            dis_ids.append(tgt)
            if limit and len(dis_ids) >= limit:
                break

    total_unique = len({it["target"].lower() for it in items})
    return {chemical_id: dis_ids}, total_unique

def search_treatment_evidence(
    disease_id: str,
    chemical_id: str,
    page: int = 1,
    timeout: float = 15.0,
    strict: bool = False,
) -> Tuple[List[Dict], int]:
    """
    Return (results, total_count) for relations:ANY|chemical_id|disease_id.

    Aggregates results across up to 10 pages starting at `page`,
    capped at 100 results total, preserving API order.
    """
    q = f"relations:ANY|{chemical_id}|{disease_id}"

    # First page (keep original error semantics)
    params: Dict[str, Any] = {"text": q, "page": page}
    r = _get("/search/", params, timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        if strict:
            raise
        return [], 0

    j = r.json()
    first_results = j.get("results", []) or []
    total_count = int(j.get("count", 0))

    all_results: List[Dict] = list(first_results)
    page_size = len(first_results)

    # If nothing on the first page, nothing more to do
    if page_size == 0 or total_count <= page_size:
        return all_results[:100], total_count

    # Compute maximum number of pages available
    max_pages = (total_count + page_size - 1) // page_size
    last_page = min(page + 9, max_pages)  # at most 10 pages total

    # Fetch additional pages
    for p in range(page + 1, last_page + 1):
        if len(all_results) >= 100:
            break
        params = {"text": q, "page": p}
        r = _get("/search/", params, timeout)
        try:
            r.raise_for_status()
        except requests.HTTPError:
            if strict:
                raise
            break
        j = r.json()
        results_p = j.get("results", []) or []
        if not results_p:
            break
        all_results.extend(results_p)

    return all_results[:100], total_count