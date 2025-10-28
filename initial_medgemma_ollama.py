#!/usr/bin/env python3
import os, json, time
from typing import Dict, List, Any, Optional
import requests
import ollama

os.environ["OLLAMA_HOST"] = "http://localhost:11348"
MODEL = os.getenv("OLLAMA_MODEL", "llama3.3:latest")
PUBTATOR = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"

# --- PubTator client (polite defaults) ---
class PT:
    def __init__(self, ua="pubtator-demo/0.1 (contact: you@example.com)", rps=3.0, timeout=30, retries=2, backoff=0.6):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": ua, "Accept": "application/json"})
        self.min_dt = 1.0 / max(1e-6, rps)
        self.last = 0.0
        self.timeout, self.retries, self.backoff = timeout, retries, backoff

    def _throttle(self):
        dt = time.time() - self.last
        if dt < self.min_dt: time.sleep(self.min_dt - dt)

    def _get(self, path, params=None):
        url = f"{PUBTATOR}{path}"
        for i in range(self.retries + 1):
            self._throttle()
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
                self.last = time.time()
                if r.status_code in (429, 500, 502, 503, 504) and i < self.retries:
                    time.sleep(self.backoff * (2 ** i)); continue
                r.raise_for_status()
                return r
            except requests.RequestException:
                if i >= self.retries: raise
                time.sleep(self.backoff * (2 ** i))
        raise RuntimeError("unreachable")

    def entity_autocomplete(self, q: str, concept="DISEASE", limit=6):
        p = {"query": q, "limit": limit}
        if concept: p["concept"] = concept
        return self._get("/entity/autocomplete/", p).json()

    def relations(self, e1: str, rel="treat", e2="chemical", limit=8):
        rows = self._get("/relations", {"e1": e1, "type": rel, "e2": e2}).json()
        rows = rows.get("results", []) if isinstance(rows, dict) else rows
        return rows[:limit]

    def search_rel(self, e1: str, e2: str, rel="treat", pages=1):
        q = f"relations:{rel}|{e1}|{e2}"
        out = []
        for page in range(1, pages + 1):
            data = self._get("/search/", {"text": q, "page": page}).json()
            res = data.get("results", [])
            if not res: break
            out += res
        return {"query": q, "results": out}

pt = PT()

# --- Tool functions ---
def tool_find_entity_id(a: Dict[str, Any]) -> Dict[str, Any]:
    return {"results": pt.entity_autocomplete(a.get("query",""), "DISEASE", int(a.get("limit",6)))}

def tool_find_related_entities(a: Dict[str, Any]) -> Dict[str, Any]:
    rows = pt.relations(a["e1"], a.get("relation_type","treat"), a.get("e2","chemical"), int(a.get("limit",8)))
    return {"results": rows}

def tool_export_relevant_search_results(a: Dict[str, Any]) -> Dict[str, Any]:
    data = pt.search_rel(a["e1"], a["e2"], a.get("relation_type","treat"), int(a.get("max_pages",1)))
    pmids = [str(r.get("pmid")) for r in data.get("results", []) if r.get("pmid")]
    return {"pmids": list(dict.fromkeys(pmids))[:20], "query": data.get("query")}

TOOLS = {
    "find_entity_id": tool_find_entity_id,
    "find_related_entities": tool_find_related_entities,
    "export_relevant_search_results": tool_export_relevant_search_results,
}

# --- LLM helpers ---
def llm(messages: List[Dict[str,str]], temp=0) -> str:
    resp = ollama.chat(model=MODEL, messages=messages, options={"temperature": temp})
    return resp.get("message",{}).get("content","")

def tool_loop(system: str, user: str, max_turns=20) -> Dict[str, Any]:
    msgs = [{"role":"system","content":system},{"role":"user","content":user}]
    for _ in range(max_turns):
        out = llm(msgs)
        try: obj = json.loads(out)
        except json.JSONDecodeError:
            msgs += [{"role":"assistant","content":out},{"role":"user","content":"Return one JSON object only."}]
            continue
        tool, args = obj.get("tool"), obj.get("arguments",{})
        if tool == "final_answer": return args
        if tool not in TOOLS:
            msgs += [{"role":"assistant","content":out},{"role":"user","content":"Use listed tools only."}]
            continue
        try: result = TOOLS[tool](args)
        except Exception as e: result = {"error": str(e)}
        msgs += [{"role":"assistant","content":out},{"role":"tool","content":json.dumps({"tool":tool,"result":result})}]
    return {}

# --- Stage 1: 15 diseases from LLM ---
def get_15_diseases() -> List[str]:
    out = llm([
        {"role":"system","content":"Output JSON only."},
        {"role":"user","content":"List exactly 15 distinct human diseases. JSON: {\"diseases\":[\"...\"]}. No categories/symptoms."}
    ])
    data = json.loads(out)
    uniq, seen = [], set()
    for d in data.get("diseases", []):
        k = d.strip()
        if k and k.lower() not in seen:
            seen.add(k.lower()); uniq.append(k)
        if len(uniq) == 15: break
    return uniq

# --- Stage 2: disease -> @DISEASE_* ---
def map_diseases_to_ids(diseases: List[str]) -> Dict[str,str]:
    system = ("Call find_entity_id to normalize disease names to PubTator DISEASE IDs (@DISEASE_*). "
              "Return final_answer {\"mapping\": {\"Name\":\"@DISEASE_*\",...}}. JSON only.")
    user = json.dumps({"diseases": diseases})
    return tool_loop(system, user).get("mapping", {})

# --- Stage 3: treating chemicals per disease ---
def find_treating_drugs(dmap: Dict[str,str], n=3) -> List[Dict[str,str]]:
    system = ("Use find_related_entities with type=treat, e2=chemical. "
              "Return final_answer {\"pairs\":[{\"disease\":\"...\",\"disease_id\":\"@DISEASE_*\",\"drug\":\"...\",\"drug_id\":\"@CHEMICAL_*\"},...]}.")
    user = json.dumps({"mapping": dmap, "N": n})
    return tool_loop(system, user).get("pairs", [])

# --- Stage 4: PMIDs per pair ---
def fetch_pmids_for_pairs(pairs: List[Dict[str,str]], k=10) -> List[Dict[str,Any]]:
    system = ("Use export_relevant_search_results with relation_type=treat, e1=@CHEMICAL_*, e2=@DISEASE_*. "
              "Return final_answer {\"evidence\":[{\"disease\":\"...\",\"disease_id\":\"@DISEASE_*\",\"drug\":\"...\",\"drug_id\":\"@CHEMICAL_*\",\"pmids\":[...]}]}.")
    user = json.dumps({"pairs": pairs, "max_pmids": k})
    ev = tool_loop(system, user).get("evidence", [])
    for row in ev: row["pmids"] = row.get("pmids", [])[:k]
    return ev

def main():
    diseases = get_15_diseases()
    print("Stage 1:", diseases, "\n")
    dmap = map_diseases_to_ids(diseases)
    print("Stage 2:")
    for k,v in dmap.items(): print(f"  {k} -> {v}")
    print()
    pairs = find_treating_drugs(dmap, n=3)
    print(f"Stage 3 ({len(pairs)} pairs):")
    for p in pairs:
        print(f"  {p.get('disease')} ({p.get('disease_id')}) ~ {p.get('drug')} ({p.get('drug_id')})")
    print()
    evidence = fetch_pmids_for_pairs(pairs, k=10)
    print("Stage 4:")
    for r in evidence:
        print(f"Disease {r.get('disease')} with entity id {r.get('disease_id')} is treated by drug {r.get('drug')} which has entity id {r.get('drug_id')} and these are the PMID results which show this:")
        print("  PMIDs:", ", ".join(r.get("pmids", [])) or "(none)")
    print("\nDone.")

if __name__ == "__main__":
    main()
