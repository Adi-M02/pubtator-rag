#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


NA_SET = {"", "NA", "N/A", "NAN", "NONE", "NULL"}


def norm_na(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.upper() in NA_SET else s


def norm_key(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_json_cell(cell: str):
    s = norm_na(cell)
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"(\[.*\]|\{.*\})", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def extract_abbrev_pairs(text: str) -> dict[str, str]:
    text = text or ""
    out: dict[str, str] = {}
    for m in re.finditer(r"([A-Za-z][A-Za-z \-]{3,160}?)\s*\(([A-Z]{2,10})\)", text):
        long_form = m.group(1).strip()
        acr = m.group(2).strip()
        if long_form and acr:
            out[acr] = long_form
    return out


def naive_singular(tok: str) -> str:
    if len(tok) <= 3:
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("sses"):
        return tok
    if tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


DEFAULT_SALT_TOKENS = {
    "hydrochloride",
    "chloride",
    "sulfate",
    "sulphate",
    "sodium",
    "potassium",
    "phosphate",
    "acetate",
    "succinate",
    "tartrate",
    "maleate",
    "besylate",
    "mesylate",
    "citrate",
    "fumarate",
    "bromide",
}

DEFAULT_DRUG_DROP = {
    "mg",
    "mcg",
    "g",
    "kg",
    "ml",
    "l",
    "iu",
    "unit",
    "units",
    "percent",
    "%",
    "dose",
    "doses",
    "bid",
    "tid",
    "qid",
    "qd",
    "daily",
    "weekly",
    "monthly",
    "twice",
    "once",
    "iv",
    "po",
    "im",
    "sc",
    "sq",
    "oral",
    "intravenous",
    "subcutaneous",
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "solution",
    "suspension",
    "injection",
    "sr",
    "xr",
    "er",
    "dr",
    "cr",
    "extended",
    "release",
    "placebo",
}

DEFAULT_DISEASE_DROP = {
    "disease",
    "diseases",
    "disorder",
    "disorders",
    "condition",
    "conditions",
    "syndrome",
    "syndromes",
    "infection",
    "infections",
}


@dataclass(frozen=True)
class MatchConfig:
    name: str
    order_agnostic: bool = False
    singularize_disease: bool = False
    singularize_drug: bool = False
    drop_drug_dose_tokens: bool = False
    strip_drug_salts: bool = False
    drop_disease_generic_tokens: bool = False
    expand_abbrev: bool = False
    synonyms_json: str = ""


@dataclass
class CanonTools:
    synonyms_map: dict[str, str]
    drug_drop: set[str]
    salt_tokens: set[str]
    disease_drop: set[str]


@dataclass
class RecallCounts:
    matched_truth: int = 0
    total_truth: int = 0

    @property
    def recall(self):
        return (self.matched_truth / self.total_truth) if self.total_truth else None

    @property
    def fn(self) -> int:
        return max(0, self.total_truth - self.matched_truth)


def load_synonyms_map(path: str) -> dict[str, str]:
    p = Path(path)
    if not path or not p.exists():
        return {}
    obj = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    if not isinstance(obj, dict):
        raise ValueError("synonyms_json must be a JSON dict")
    for k, v in obj.items():
        if isinstance(v, list):
            canon = norm_key(k)
            if canon:
                out[canon] = canon
            for s in v:
                ss = norm_key(str(s))
                if ss and canon:
                    out[ss] = canon
        elif isinstance(v, str):
            kk = norm_key(k)
            vv = norm_key(v)
            if kk and vv:
                out[kk] = vv
    return out


def tokenize_basic(s: str) -> list[str]:
    s = (s or "").lower().strip()
    if not s:
        return []
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    toks = [t for t in s.split(" ") if t]
    toks = [t for t in toks if not re.fullmatch(r"\d+(?:\.\d+)?", t)]
    return toks


def canon_variants(
    s: str,
    kind: str,
    cfg: MatchConfig,
    tools: CanonTools,
    abbrev_map: dict[str, str] | None,
) -> set[str]:
    raw = norm_na(s)
    if not raw:
        return set()

    base_key = norm_key(raw)
    if tools.synonyms_map:
        mapped = tools.synonyms_map.get(base_key, "")
        if mapped:
            raw = mapped

    toks = tokenize_basic(raw)

    if kind == "drug":
        if cfg.drop_drug_dose_tokens:
            toks = [t for t in toks if t not in tools.drug_drop]
        if cfg.strip_drug_salts:
            toks = [t for t in toks if t not in tools.salt_tokens]
        if cfg.singularize_drug:
            toks = [naive_singular(t) for t in toks]
    elif kind == "disease":
        if cfg.singularize_disease:
            toks = [naive_singular(t) for t in toks]
        if cfg.drop_disease_generic_tokens:
            toks = [t for t in toks if t not in tools.disease_drop]

    toks = [t for t in toks if t]
    if not toks:
        return set()

    def mk(ts: list[str]) -> str:
        return " ".join(sorted(ts) if cfg.order_agnostic else ts).strip()

    forms = {mk(toks)}

    if cfg.expand_abbrev and kind == "disease" and abbrev_map:
        acr_to_long = {norm_key(k): v for k, v in abbrev_map.items() if norm_key(k) and norm_na(v)}
        long_to_acr = {norm_key(v): k for k, v in abbrev_map.items() if norm_key(v) and norm_na(k)}

        def prep_long(long_form: str) -> list[str]:
            lt = tokenize_basic(long_form)
            if cfg.singularize_disease:
                lt = [naive_singular(t) for t in lt]
            if cfg.drop_disease_generic_tokens:
                lt = [t for t in lt if t not in tools.disease_drop]
            return [t for t in lt if t]

        expanded = set()
        for form in list(forms):
            ftoks = form.split()

            for acr, long_form in acr_to_long.items():
                if acr in ftoks:
                    ltoks = prep_long(long_form)
                    if ltoks:
                        rep = []
                        for t in ftoks:
                            if t == acr:
                                rep.extend(ltoks)
                            else:
                                rep.append(t)
                        expanded.add(mk(rep))

            for long_k, acr in long_to_acr.items():
                long_toks = long_k.split()
                if not long_toks:
                    continue
                i = 0
                rep = []
                while i < len(ftoks):
                    if ftoks[i : i + len(long_toks)] == long_toks:
                        rep.append(norm_key(acr))
                        i += len(long_toks)
                    else:
                        rep.append(ftoks[i])
                        i += 1
                expanded.add(mk(rep))

        forms |= {f for f in expanded if f}

    return {f for f in forms if f}


def load_llm_jsonl(path: Path) -> dict[str, dict]:
    by_pmid: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pmid = norm_na(obj.get("pmid", ""))
            if pmid:
                by_pmid[pmid] = obj
    return by_pmid


def dedupe_strings(xs: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for x in xs:
        x = norm_na(x)
        k = norm_key(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def truth_from_claims_json(cell: str) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    arr = parse_json_cell(cell)
    drugs: list[str] = []
    diseases: list[str] = []
    pairs: list[tuple[str, str]] = []
    if not isinstance(arr, list):
        return [], [], []
    for x in arr:
        if not isinstance(x, dict):
            continue
        dr = norm_na(x.get("drug", ""))
        di = norm_na(x.get("disease", ""))
        if dr:
            drugs.append(dr)
        if di:
            diseases.append(di)
        if dr and di:
            pairs.append((dr, di))

    drugs = dedupe_strings(drugs)
    diseases = dedupe_strings(diseases)

    seen = set()
    pairs_out = []
    for dr, di in pairs:
        k = (norm_key(dr), norm_key(di))
        if not k[0] or not k[1] or k in seen:
            continue
        seen.add(k)
        pairs_out.append((dr, di))
    return drugs, diseases, pairs_out


def pred_from_llm_row(llm_row: dict) -> tuple[list[str], list[str], list[tuple[str, set[str]]]]:
    llm_json = llm_row.get("llm_json")
    if not isinstance(llm_json, dict):
        return [], [], []

    drugs = llm_json.get("drugs", [])
    drugs = drugs if isinstance(drugs, list) else []
    drugs = dedupe_strings([norm_na(x) for x in drugs])

    disease_concepts = llm_json.get("disease_concepts", [])
    disease_concepts = disease_concepts if isinstance(disease_concepts, list) else []

    canon_to_alias: dict[str, set[str]] = {}
    diseases_flat: list[str] = []
    for dc in disease_concepts:
        if not isinstance(dc, dict):
            continue
        canon = norm_na(dc.get("canonical", ""))
        aliases = dc.get("aliases", [])
        aliases = aliases if isinstance(aliases, list) else []
        s = set()
        if canon:
            s.add(canon)
        for a in aliases:
            aa = norm_na(a)
            if aa:
                s.add(aa)
        if canon:
            canon_to_alias[canon] = s
        diseases_flat.extend(list(s))

    diseases_flat = dedupe_strings(diseases_flat)

    treats = llm_json.get("treats", [])
    treats = treats if isinstance(treats, list) else []
    pairs: list[tuple[str, set[str]]] = []
    for t in treats:
        if not isinstance(t, dict):
            continue
        dr = norm_na(t.get("drug", ""))
        di_c = norm_na(t.get("disease_canonical", ""))
        if not dr or not di_c:
            continue
        alias_set = canon_to_alias.get(di_c, {di_c})
        pairs.append((dr, set(alias_set)))

    return drugs, diseases_flat, pairs


def match_items_recall(
    truth_items: list[str],
    pred_items: list[str],
    kind: str,
    cfg: MatchConfig,
    tools: CanonTools,
    abbrev_map: dict[str, str],
) -> tuple[RecallCounts, list[str]]:
    truth_vars = [(t, canon_variants(t, kind, cfg, tools, abbrev_map)) for t in truth_items]
    pred_vars = [(p, canon_variants(p, kind, cfg, tools, abbrev_map)) for p in pred_items]

    truth_vars = [(t, s) for t, s in truth_vars if s]
    pred_vars = [(p, s) for p, s in pred_vars if s]

    matched = 0
    missed = []
    for t_raw, tset in truth_vars:
        if any(tset & pset for _, pset in pred_vars):
            matched += 1
        else:
            missed.append(t_raw)

    counts = RecallCounts(matched_truth=matched, total_truth=len(truth_vars))
    return counts, missed


def match_pairs_recall(
    truth_pairs: list[tuple[str, str]],
    pred_pairs: list[tuple[str, set[str]]],
    cfg: MatchConfig,
    tools: CanonTools,
    abbrev_map: dict[str, str],
) -> tuple[RecallCounts, list[tuple[str, str]]]:
    tv = []
    for dr, di in truth_pairs:
        dset = canon_variants(dr, "drug", cfg, tools, abbrev_map)
        iset = canon_variants(di, "disease", cfg, tools, abbrev_map)
        if dset and iset:
            tv.append((dr, di, dset, iset))

    pv = []
    for pdr, pdis_set in pred_pairs:
        dset = canon_variants(pdr, "drug", cfg, tools, abbrev_map)
        isets = []
        for s in pdis_set:
            iset = canon_variants(s, "disease", cfg, tools, abbrev_map)
            if iset:
                isets.append(iset)
        if dset and isets:
            pv.append((pdr, pdis_set, dset, isets))

    matched = 0
    missed = []
    for tdr, tdi, t_dset, t_iset in tv:
        ok = False
        for _, _, p_dset, p_isets in pv:
            if not (t_dset & p_dset):
                continue
            if any(t_iset & pis for pis in p_isets):
                ok = True
                break
        if ok:
            matched += 1
        else:
            missed.append((tdr, tdi))

    counts = RecallCounts(matched_truth=matched, total_truth=len(tv))
    return counts, missed


def default_sweep(synonyms_json: str = "") -> list[MatchConfig]:
    base = dict(synonyms_json=synonyms_json or "")
    return [
        MatchConfig(name="00_base", **base),
        MatchConfig(name="01_order", order_agnostic=True, **base),
        MatchConfig(name="02_order_sing_dis", order_agnostic=True, singularize_disease=True, **base),
        MatchConfig(
            name="03_add_drug_dose",
            order_agnostic=True,
            singularize_disease=True,
            drop_drug_dose_tokens=True,
            **base,
        ),
        MatchConfig(
            name="04_add_salts",
            order_agnostic=True,
            singularize_disease=True,
            drop_drug_dose_tokens=True,
            strip_drug_salts=True,
            **base,
        ),
        MatchConfig(
            name="05_add_dis_generic_drop",
            order_agnostic=True,
            singularize_disease=True,
            drop_drug_dose_tokens=True,
            strip_drug_salts=True,
            drop_disease_generic_tokens=True,
            **base,
        ),
        MatchConfig(
            name="06_add_abbrev",
            order_agnostic=True,
            singularize_disease=True,
            drop_drug_dose_tokens=True,
            strip_drug_salts=True,
            drop_disease_generic_tokens=True,
            expand_abbrev=True,
            **base,
        ),
        MatchConfig(
            name="07_all_plus_sing_drug",
            order_agnostic=True,
            singularize_disease=True,
            singularize_drug=True,
            drop_drug_dose_tokens=True,
            strip_drug_salts=True,
            drop_disease_generic_tokens=True,
            expand_abbrev=True,
            **base,
        ),
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_jsonl", required=True)
    ap.add_argument("--manual_csv", required=True)
    ap.add_argument("--out_tag", default="llm_recall_only")
    ap.add_argument("--out_root", default="", help="Default: <parent_of_this_file>/output")
    ap.add_argument("--max_examples", type=int, default=200)
    ap.add_argument("--truth_sources", default="pubtator,rentrez,union")

    ap.add_argument("--run_single", action="store_true")
    ap.add_argument("--config_name", default="single")
    ap.add_argument("--order_agnostic", action="store_true")
    ap.add_argument("--singularize_disease", action="store_true")
    ap.add_argument("--singularize_drug", action="store_true")
    ap.add_argument("--drop_drug_dose_tokens", action="store_true")
    ap.add_argument("--strip_drug_salts", action="store_true")
    ap.add_argument("--drop_disease_generic_tokens", action="store_true")
    ap.add_argument("--expand_abbrev", action="store_true")
    ap.add_argument("--synonyms_json", default="")

    args = ap.parse_args()

    llm_path = Path(args.llm_jsonl)
    manual_path = Path(args.manual_csv)
    if not llm_path.exists():
        raise FileNotFoundError(llm_path)
    if not manual_path.exists():
        raise FileNotFoundError(manual_path)

    script_dir = Path(__file__).resolve().parent
    out_root = Path(args.out_root) if args.out_root else (script_dir.parent / "output")
    out_dir = out_root / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    tools = CanonTools(
        synonyms_map=load_synonyms_map(args.synonyms_json) if args.synonyms_json else {},
        drug_drop=set(DEFAULT_DRUG_DROP),
        salt_tokens=set(DEFAULT_SALT_TOKENS),
        disease_drop=set(DEFAULT_DISEASE_DROP),
    )

    llm_by_pmid = load_llm_jsonl(llm_path)
    df = pd.read_csv(manual_path, dtype=str).fillna("")
    if "pmid" not in df.columns:
        raise ValueError(f"manual_csv missing 'pmid' column. Found: {list(df.columns)}")
    if "pubtator_true_claims_json" not in df.columns or "rentrez_true_claims_json" not in df.columns:
        raise ValueError(
            "manual_csv missing required truth columns: pubtator_true_claims_json and/or rentrez_true_claims_json"
        )

    pmids_manual = [norm_na(x) for x in df["pmid"].tolist() if norm_na(x)]
    pmids_overlap = [p for p in pmids_manual if p in llm_by_pmid]

    truth_sources = [x.strip() for x in args.truth_sources.split(",") if x.strip()]
    for s in truth_sources:
        if s not in {"pubtator", "rentrez", "union"}:
            raise ValueError(f"Invalid truth source: {s}")

    coverage = {
        "manual_pmids": len(pmids_manual),
        "llm_pmids": len(llm_by_pmid),
        "pmids_overlap": len(pmids_overlap),
        "pmids_missing_llm": len(pmids_manual) - len(pmids_overlap),
        "truth_sources": truth_sources,
        "llm_jsonl": str(llm_path),
        "manual_csv": str(manual_path),
        "notes": "Recall is computed against human-verified truth sets in pubtator_true_claims_json and rentrez_true_claims_json only.",
    }
    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    if args.run_single:
        cfgs = [
            MatchConfig(
                name=args.config_name,
                order_agnostic=args.order_agnostic,
                singularize_disease=args.singularize_disease,
                singularize_drug=args.singularize_drug,
                drop_drug_dose_tokens=args.drop_drug_dose_tokens,
                strip_drug_salts=args.strip_drug_salts,
                drop_disease_generic_tokens=args.drop_disease_generic_tokens,
                expand_abbrev=args.expand_abbrev,
                synonyms_json=args.synonyms_json or "",
            )
        ]
    else:
        cfgs = default_sweep(synonyms_json=args.synonyms_json or "")

    sweep_rows = []

    for cfg in cfgs:
        cfg_out = out_dir / f"config_{cfg.name}"
        cfg_out.mkdir(parents=True, exist_ok=True)
        (cfg_out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

        summary_rows = []
        by_pmid_rows = []

        missed_drug_rows = []
        missed_dis_rows = []
        missed_pair_rows = []

        for src in truth_sources:
            tot_dr = RecallCounts()
            tot_di = RecallCounts()
            tot_pa = RecallCounts()

            for _, r in df.iterrows():
                pmid = norm_na(r.get("pmid", ""))
                if not pmid or pmid not in llm_by_pmid:
                    continue

                llm_row = llm_by_pmid[pmid]
                p_drugs, p_dis_flat, p_pairs = pred_from_llm_row(llm_row)

                title = llm_row.get("fetched_title", "") or ""
                abstract = llm_row.get("fetched_abstract", "") or ""
                abbrev_map = {}
                abbrev_map.update(extract_abbrev_pairs(title))
                abbrev_map.update(extract_abbrev_pairs(abstract))

                pub_dr, pub_di, pub_pa = truth_from_claims_json(r.get("pubtator_true_claims_json", ""))
                ren_dr, ren_di, ren_pa = truth_from_claims_json(r.get("rentrez_true_claims_json", ""))

                if src == "pubtator":
                    t_drugs, t_dis, t_pairs = pub_dr, pub_di, pub_pa
                elif src == "rentrez":
                    t_drugs, t_dis, t_pairs = ren_dr, ren_di, ren_pa
                else:
                    t_drugs = dedupe_strings(pub_dr + ren_dr)
                    t_dis = dedupe_strings(pub_di + ren_di)
                    seen = set()
                    t_pairs = []
                    for a, b in pub_pa + ren_pa:
                        k = (norm_key(a), norm_key(b))
                        if not k[0] or not k[1] or k in seen:
                            continue
                        seen.add(k)
                        t_pairs.append((a, b))

                dr_counts, dr_missed = match_items_recall(t_drugs, p_drugs, "drug", cfg, tools, abbrev_map)
                di_counts, di_missed = match_items_recall(t_dis, p_dis_flat, "disease", cfg, tools, abbrev_map)
                pa_counts, pa_missed = match_pairs_recall(t_pairs, p_pairs, cfg, tools, abbrev_map)

                tot_dr.matched_truth += dr_counts.matched_truth
                tot_dr.total_truth += dr_counts.total_truth
                tot_di.matched_truth += di_counts.matched_truth
                tot_di.total_truth += di_counts.total_truth
                tot_pa.matched_truth += pa_counts.matched_truth
                tot_pa.total_truth += pa_counts.total_truth

                by_pmid_rows.append(
                    {
                        "config": cfg.name,
                        "truth_source": src,
                        "pmid": pmid,
                        "drugs_recall": dr_counts.recall,
                        "drugs_total_truth": dr_counts.total_truth,
                        "drugs_fn": dr_counts.fn,
                        "diseases_recall": di_counts.recall,
                        "diseases_total_truth": di_counts.total_truth,
                        "diseases_fn": di_counts.fn,
                        "pairs_recall": pa_counts.recall,
                        "pairs_total_truth": pa_counts.total_truth,
                        "pairs_fn": pa_counts.fn,
                    }
                )

                if len(missed_drug_rows) < args.max_examples:
                    for x in dr_missed[: max(0, args.max_examples - len(missed_drug_rows))]:
                        missed_drug_rows.append({"config": cfg.name, "truth_source": src, "pmid": pmid, "missed_drug": x})

                if len(missed_dis_rows) < args.max_examples:
                    for x in di_missed[: max(0, args.max_examples - len(missed_dis_rows))]:
                        missed_dis_rows.append({"config": cfg.name, "truth_source": src, "pmid": pmid, "missed_disease": x})

                if len(missed_pair_rows) < args.max_examples:
                    for d, di in pa_missed[: max(0, args.max_examples - len(missed_pair_rows))]:
                        missed_pair_rows.append(
                            {"config": cfg.name, "truth_source": src, "pmid": pmid, "missed_drug": d, "missed_disease": di}
                        )

            def add_summary(metric: str, counts: RecallCounts):
                summary_rows.append(
                    {
                        "config": cfg.name,
                        "truth_source": src,
                        "metric": metric,
                        "matched_truth": counts.matched_truth,
                        "total_truth": counts.total_truth,
                        "recall": counts.recall,
                        "fn": counts.fn,
                    }
                )

            add_summary("drugs", tot_dr)
            add_summary("diseases", tot_di)
            add_summary("pairs", tot_pa)

        write_csv(cfg_out / "summary.csv", summary_rows)
        write_csv(cfg_out / "by_pmid.csv", by_pmid_rows)

        if missed_drug_rows:
            write_csv(cfg_out / "samples_missed_drugs.csv", missed_drug_rows)
        if missed_dis_rows:
            write_csv(cfg_out / "samples_missed_diseases.csv", missed_dis_rows)
        if missed_pair_rows:
            write_csv(cfg_out / "samples_missed_pairs.csv", missed_pair_rows)

        sweep_rows.extend(summary_rows)

    write_csv(out_dir / "sweep_summary.csv", sweep_rows)

    print(f"Wrote outputs under: {out_dir}")
    print("Key file: sweep_summary.csv")


if __name__ == "__main__":
    main()
