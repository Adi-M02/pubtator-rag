#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
import ollama
from tqdm import tqdm


DEFAULT_SAMPLE_RUN_TAG = "shared_pmids_eval_stratified"

EXTRACT_SCHEMA = {
    "type": "object",
    "required": ["drugs", "disease_concepts", "treats"],
    "properties": {
        "drugs": {"type": "array", "items": {"type": "string"}},
        "disease_concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["canonical", "aliases"],
                "properties": {
                    "canonical": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "treats": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["drug", "disease_canonical"],
                "properties": {
                    "drug": {"type": "string"},
                    "disease_canonical": {"type": "string"},
                },
            },
        },
    },
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def el_text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def fetch_title_abstract(pmid: str, email: str = "") -> dict:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    if email:
        params["email"] = email

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    title = el_text(root.find(".//ArticleTitle")) or "(No title found.)"

    abs_els = root.findall(".//AbstractText")
    parts = []
    for a in abs_els:
        label = (a.attrib.get("Label") or "").strip()
        txt = el_text(a)
        if not txt:
            continue
        parts.append(f"{label}: {txt}" if label else txt)

    abstract = "\n\n".join(parts).strip()
    if not abstract:
        abstract = "(No abstract found.)"

    return {"title": title, "abstract": abstract}


def safe_json_from_text(s: str) -> dict | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def extract_abbrev_pairs(text: str) -> dict[str, str]:
    text = text or ""
    out: dict[str, str] = {}
    # allow digits in acronym like AD04, TIA, ALD, AUD
    for m in re.finditer(r"([A-Za-z][A-Za-z \-]{3,80}?)\s*\(([A-Z0-9]{2,12})\)", text):
        long_form = m.group(1).strip()
        acr = m.group(2).strip()
        if long_form and acr:
            out[acr] = long_form
    return out


def enrich_disease_aliases(obj: dict | None, title: str, abstract: str) -> dict | None:
    if not isinstance(obj, dict):
        return obj

    dcs = obj.get("disease_concepts", [])
    if not isinstance(dcs, list):
        return obj

    abbrev = {}
    abbrev.update(extract_abbrev_pairs(title))
    abbrev.update(extract_abbrev_pairs(abstract))

    for dc in dcs:
        if not isinstance(dc, dict):
            continue

        canonical = (dc.get("canonical") or "").strip()
        aliases = dc.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []

        alias_set = {str(a).strip() for a in aliases if str(a).strip()}
        if canonical:
            alias_set.add(canonical)

        for acr, long_form in abbrev.items():
            if acr in alias_set and long_form not in alias_set:
                alias_set.add(long_form)
            if long_form in alias_set and acr not in alias_set:
                alias_set.add(acr)

        dc["canonical"] = canonical
        dc["aliases"] = sorted(alias_set, key=lambda x: x.lower())

    obj["disease_concepts"] = dcs
    return obj


def prompt_for(title: str, abstract: str) -> str:
    return (
        "Extract drug-to-disease (or condition) treatment relationships from the title and abstract.\n\n"
        "Rules:\n"
        "- Be generous: list all drugs used as therapies, regimens, or assigned treatments.\n"
        "- Be generous for diseases/conditions: list the primary treated disease and any other treated conditions mentioned.\n"
        "- Prefer generic drug names (but include codes like AD04 as an alias if present).\n"
        "- For diseases: return a canonical form plus multiple aliases/restatements.\n"
        "- If an abbreviation appears (example: 'alcohol use disorder (AUD)' or 'transient ischemic attack (TIA)'), include BOTH forms as aliases.\n"
        "- If you know common formal headings (example: 'Ischemic Attack, Transient'), include them as aliases too.\n"
        "- Output must match the provided JSON schema.\n\n"
        f"TITLE:\n{title}\n\nABSTRACT:\n{abstract}\n"
    )


def sanitize_ollama_host(host: str) -> str:
    h = (host or "").strip()
    if not h:
        return "http://localhost:11434"
    if not h.startswith(("http://", "https://")):
        h = "http://" + h
    for suffix in ("/api/chat", "/api/generate", "/api"):
        if h.endswith(suffix):
            h = h[: -len(suffix)]
            break
    h = h.rstrip("/")
    u = urlparse(h)
    if not u.scheme or not u.netloc:
        return "http://localhost:11434"
    return h


def resp_content(resp) -> str:
    if resp is None:
        return ""
    try:
        if isinstance(resp, dict):
            return ((resp.get("message") or {}) or {}).get("content") or ""
    except Exception:
        pass
    try:
        return (resp.message.content or "").strip()
    except Exception:
        pass
    try:
        return (resp["message"]["content"] or "").strip()
    except Exception:
        return ""


def ollama_extract(
    client: "ollama.Client",
    model: str,
    title: str,
    abstract: str,
    temperature: float,
    seed: int | None,
) -> tuple[dict | None, str, str]:
    msgs = [
        {
            "role": "system",
            "content": "You are a biomedical information extraction assistant. Respond with JSON that matches the schema.",
        },
        {"role": "user", "content": prompt_for(title, abstract)},
    ]
    opts = {"temperature": float(temperature)}
    if seed is not None:
        opts["seed"] = int(seed)

    try:
        resp = client.chat(model=model, messages=msgs, options=opts, format=EXTRACT_SCHEMA)
        text = resp_content(resp)
        obj = safe_json_from_text(text)
        if isinstance(obj, dict):
            return obj, text, ""
        return None, text, "json_parse_failed"
    except Exception as e:
        return None, "", f"ollama_error: {type(e).__name__}: {e}"


def parse_pmids_arg(pmids_arg: str) -> list[str]:
    if not pmids_arg:
        return []
    p = Path(pmids_arg)
    if p.exists():
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
        return out
    return [x.strip() for x in pmids_arg.split(",") if x.strip()]


def load_sample_pmids_from_file(sample_path: Path) -> list[str]:
    obj = json.loads(sample_path.read_text(encoding="utf-8"))
    pmids = obj.get("pmids", [])
    if not isinstance(pmids, list) or not pmids:
        raise ValueError(f"No PMIDs found in sample file: {sample_path}")
    return [str(x).strip() for x in pmids if str(x).strip()]


def find_sample_json(script_dir: Path, sample_run_tag: str) -> Path | None:
    project_dir = script_dir.parent
    name = f"{sample_run_tag}_sample_pmids.json"
    candidates = [
        script_dir / "eval_results" / name,
        project_dir / "eval_results" / name,
        project_dir / "manual_evaluation" / "eval_results" / name,
        project_dir / "manual_evaluation" / "manual_evaluation" / "eval_results" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_done_pmids(jsonl_path: Path) -> set[str]:
    if not jsonl_path.exists():
        return set()
    done = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pmid = str(obj.get("pmid", "")).strip()
                if pmid:
                    done.add(pmid)
            except Exception:
                continue
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmids", type=str, default="", help="Comma-separated PMIDs or path to a text file. If empty, loads saved sample PMIDs.")
    ap.add_argument("--sample_run_tag", type=str, default=DEFAULT_SAMPLE_RUN_TAG)
    ap.add_argument("--sample_json", type=str, default="", help="Explicit path to *_sample_pmids.json. Overrides --sample_run_tag.")
    ap.add_argument("--ollama_host", type=str, default=os.environ.get("OLLAMA_HOST", ""), help="Ollama base URL, e.g. http://server:11434")
    ap.add_argument("--model", type=str, default="llama3.3:latest")
    ap.add_argument("--email", type=str, default="")
    ap.add_argument("--run_tag", type=str, default="llama3_3_latest_extract_temp0")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tqdm", action="store_true")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent

    out_dir = script_dir / "eval_results" / "llm_extractions" / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "llm_outputs.jsonl"
    out_meta = out_dir / "run_metadata.json"

    pmids = parse_pmids_arg(args.pmids)

    if not pmids:
        if args.sample_json:
            sample_path = Path(args.sample_json)
            if not sample_path.is_absolute():
                sample_path = (script_dir.parent / sample_path).resolve()
            if not sample_path.exists():
                raise FileNotFoundError(f"--sample_json not found: {sample_path}")
        else:
            sample_path = find_sample_json(script_dir, args.sample_run_tag)
            if sample_path is None:
                expected = f"{args.sample_run_tag}_sample_pmids.json"
                raise FileNotFoundError(
                    "Sample PMID file not found.\n"
                    f"Expected filename: {expected}\n"
                    "Fix options:\n"
                    "1) Pass --sample_json path/to/<run_tag>_sample_pmids.json\n"
                    "2) Pass --sample_run_tag to match the Streamlit run_tag\n"
                    "3) Pass --pmids explicitly"
                )
        pmids = load_sample_pmids_from_file(sample_path)

    if not pmids:
        raise SystemExit("No PMIDs available.")

    if args.overwrite and out_jsonl.exists():
        out_jsonl.unlink(missing_ok=True)

    done_pmids = set() if args.overwrite else load_done_pmids(out_jsonl)
    remaining_pmids = [p for p in pmids if p not in done_pmids]

    host = sanitize_ollama_host(args.ollama_host) if args.ollama_host else "http://localhost:11434"
    client = ollama.Client(host=host)

    meta = {
        "run_tag": args.run_tag,
        "ollama_host": host,
        "model": args.model,
        "temperature": float(args.temperature),
        "seed": int(args.seed),
        "n_requested": len(pmids),
        "n_already_done": len(done_pmids),
        "n_remaining": len(remaining_pmids),
        "started_at_utc": now_utc_iso(),
        "sample_run_tag": args.sample_run_tag if (not args.pmids and not args.sample_json) else "",
        "sample_json": str(args.sample_json) if args.sample_json else "",
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    iterator = remaining_pmids
    if args.tqdm:
        iterator = tqdm(remaining_pmids, total=len(remaining_pmids), desc="PMIDs", unit="pmid")

    n_written = 0
    for pmid in iterator:
        rec = fetch_title_abstract(pmid, email=args.email)
        title, abstract = rec["title"], rec["abstract"]

        if args.sleep and args.sleep > 0:
            time.sleep(float(args.sleep))

        obj, raw, err = ollama_extract(
            client,
            args.model,
            title,
            abstract,
            temperature=float(args.temperature),
            seed=int(args.seed) if args.seed is not None else None,
        )
        obj = enrich_disease_aliases(obj, title, abstract)

        row = {
            "pmid": pmid,
            "fetched_title": title,
            "fetched_abstract": abstract,
            "ollama_host": host,
            "model": args.model,
            "temperature": float(args.temperature),
            "seed": int(args.seed),
            "llm_json": obj,
            "llm_raw": raw,
            "llm_error": err,
            "saved_at_utc": now_utc_iso(),
        }

        with out_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        n_written += 1

    meta["finished_at_utc"] = now_utc_iso()
    meta["n_written"] = n_written
    meta["n_total_done_after"] = len(done_pmids) + n_written
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_meta}")


if __name__ == "__main__":
    main()
