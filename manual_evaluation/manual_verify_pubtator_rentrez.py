#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import random

import pandas as pd
import requests
import streamlit as st
import xml.etree.ElementTree as ET


APP_DIR = Path(__file__).resolve().parent
OUT_DIR = APP_DIR / "eval_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Claim:
    dataset: str
    drug: str
    disease: str
    disease_id: str | None = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.upper() in {"NA", "N/A", "NAN", "NONE", "NULL"}:
        return ""
    return s


def claim_key_pub(c: Claim) -> str:
    did = c.disease_id or ""
    return f"{c.drug}|||{c.disease}|||{did}"


def claim_key_ren(c: Claim) -> str:
    return f"{c.drug}|||{c.disease}"


def format_pub_key(k: str) -> str:
    parts = k.split("|||")
    drug = parts[0] if len(parts) > 0 else ""
    dis = parts[1] if len(parts) > 1 else ""
    did = parts[2] if len(parts) > 2 else ""
    return f"{drug} treats {dis}" + (f" ({did})" if did else "")


def format_ren_key(k: str) -> str:
    parts = k.split("|||")
    drug = parts[0] if len(parts) > 0 else ""
    dis = parts[1] if len(parts) > 1 else ""
    return f"{drug} treats {dis}"


def claims_to_json(claims: list[Claim]) -> str:
    return json.dumps(
        [{"drug": c.drug, "disease": c.disease, "disease_id": c.disease_id} for c in claims],
        ensure_ascii=False,
    )


def json_to_pub_keys(s: str) -> list[str]:
    if not s:
        return []
    try:
        arr = json.loads(s)
        out = []
        for x in arr:
            drug = (x.get("drug") or "").strip()
            dis = (x.get("disease") or "").strip()
            did = (x.get("disease_id") or "").strip()
            if drug and dis:
                out.append(f"{drug}|||{dis}|||{did}")
        return out
    except Exception:
        return []


def json_to_ren_keys(s: str) -> list[str]:
    if not s:
        return []
    try:
        arr = json.loads(s)
        out = []
        for x in arr:
            drug = (x.get("drug") or "").strip()
            dis = (x.get("disease") or "").strip()
            if drug and dis:
                out.append(f"{drug}|||{dis}")
        return out
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def load_pubtator_min(pub_path: str) -> pd.DataFrame:
    df = pd.read_csv(pub_path, dtype=str, usecols=["drug_name", "disease_name", "disease_id", "pmid"])
    df = df.copy()
    df["pmid"] = df["pmid"].map(_norm_str)
    df["drug_name"] = df["drug_name"].map(_norm_str)
    df["disease_name"] = df["disease_name"].map(_norm_str)
    df["disease_id"] = df["disease_id"].map(_norm_str)
    df = df[(df["pmid"] != "") & (df["drug_name"] != "") & (df["disease_name"] != "")]
    return df


@st.cache_data(show_spinner=False)
def load_rentrez_min(ren_path: str, require_positive: bool) -> pd.DataFrame:
    df = pd.read_csv(ren_path, dtype=str)
    df = df.copy()
    if "pmid" not in df.columns or "drug" not in df.columns or "disease" not in df.columns:
        raise ValueError(f"Rentrez CSV missing required columns. Found: {list(df.columns)}")

    df["pmid"] = df["pmid"].map(_norm_str)
    df["drug"] = df["drug"].map(_norm_str)
    df["disease"] = df["disease"].map(_norm_str)

    if require_positive and "positive" in df.columns:
        pos = df["positive"].fillna("").astype(str).str.strip().str.upper()
        df = df[pos.isin({"TRUE", "T", "1", "YES", "Y"})]

    df = df[(df["pmid"] != "") & (df["drug"] != "") & (df["disease"] != "")]
    return df[["pmid", "drug", "disease"]]


@st.cache_data(show_spinner=False)
def build_items_cached(pub_path: str, ren_path: str, require_positive: bool) -> list[dict]:
    pub_df = load_pubtator_min(pub_path)
    ren_df = load_rentrez_min(ren_path, require_positive=require_positive)

    pub_pmids = set(pub_df["pmid"].unique())
    ren_pmids = set(ren_df["pmid"].unique())
    shared = sorted(pub_pmids & ren_pmids, key=lambda x: int(x) if str(x).isdigit() else x)
    if not shared:
        return []

    shared_set = set(shared)
    pub_df = pub_df[pub_df["pmid"].isin(shared_set)].copy()
    ren_df = ren_df[ren_df["pmid"].isin(shared_set)].copy()

    pub_g = pub_df.groupby("pmid", sort=False)
    ren_g = ren_df.groupby("pmid", sort=False)

    items = []
    for pmid in shared:
        pub_claims: list[Claim] = []
        if pmid in pub_g.groups:
            sub = pub_g.get_group(pmid).drop_duplicates(subset=["drug_name", "disease_name", "disease_id"])
            for _, r in sub.iterrows():
                pub_claims.append(
                    Claim(
                        dataset="PubTator",
                        drug=_norm_str(r["drug_name"]),
                        disease=_norm_str(r["disease_name"]),
                        disease_id=_norm_str(r.get("disease_id", "")) or None,
                    )
                )

        ren_claims: list[Claim] = []
        if pmid in ren_g.groups:
            sub = ren_g.get_group(pmid).drop_duplicates(subset=["drug", "disease"])
            for _, r in sub.iterrows():
                ren_claims.append(
                    Claim(
                        dataset="Rentrez",
                        drug=_norm_str(r["drug"]),
                        disease=_norm_str(r["disease"]),
                        disease_id=None,
                    )
                )

        items.append(
            {
                "pmid": pmid,
                "pubtator_claims": pub_claims,
                "rentrez_claims": ren_claims,
                "n_pub": len(pub_claims),
                "n_ren": len(ren_claims),
            }
        )
    return items


def _el_text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


@st.cache_data(show_spinner=False)
def fetch_title_abstract(pmid: str, email: str) -> dict:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    if email:
        params["email"] = email

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    title_el = root.find(".//ArticleTitle")
    title = _el_text(title_el)

    abs_els = root.findall(".//AbstractText")
    abstract_parts = []
    for a in abs_els:
        label = a.attrib.get("Label", "").strip()
        txt = _el_text(a)
        if not txt:
            continue
        abstract_parts.append(f"{label}: {txt}" if label else txt)

    abstract = "\n\n".join(abstract_parts).strip()
    if not abstract:
        abstract = "(No abstract found in PubMed record.)"

    return {"title": title or "(No title found in PubMed record.)", "abstract": abstract}


EVAL_COLS_CANONICAL = [
    "saved_at_utc",
    "annotator",
    "pmid",
    "group",
    "pubtator_agree",
    "rentrez_agree",
    "pubtator_true_claims_json",
    "rentrez_true_claims_json",
    "manual_drug",
    "manual_disease",
    "any_relation",
    "extracted_drug",
    "extracted_disease",
    "overall_notes",
]


def ensure_eval_csv_schema(path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in EVAL_COLS_CANONICAL if c not in df.columns]
    if not missing:
        return
    for c in missing:
        df[c] = ""
    df.to_csv(path, index=False)


def load_existing_pmid_eval(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    out = {}
    for _, r in df.iterrows():
        pmid = r.get("pmid", "")
        if not pmid:
            continue
        out[pmid] = {
            "group": r.get("group", ""),
            "pubtator_agree": r.get("pubtator_agree", ""),
            "rentrez_agree": r.get("rentrez_agree", ""),
            "any_relation": r.get("any_relation", ""),
            "extracted_drug": r.get("extracted_drug", ""),
            "extracted_disease": r.get("extracted_disease", ""),
            "overall_notes": r.get("overall_notes", ""),
            "saved_at_utc": r.get("saved_at_utc", ""),
            "annotator": r.get("annotator", ""),
            "pubtator_true_claims_json": r.get("pubtator_true_claims_json", ""),
            "rentrez_true_claims_json": r.get("rentrez_true_claims_json", ""),
            "manual_drug": r.get("manual_drug", ""),
            "manual_disease": r.get("manual_disease", ""),
        }
    return out


def write_pmid_eval(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    for c in EVAL_COLS_CANONICAL:
        if c not in df.columns:
            df[c] = ""
    df = df[EVAL_COLS_CANONICAL]
    df.to_csv(path, index=False)


def compute_precision_from_pmid(rows: list[dict], dataset_key: str) -> dict:
    decided = [r for r in rows if r.get(dataset_key, "") in {"Yes", "No"}]
    yes = sum(1 for r in decided if r.get(dataset_key, "") == "Yes")
    total = len(decided)
    unclear = sum(1 for r in rows if r.get(dataset_key, "") == "Unclear")
    precision = (yes / total) if total else None
    return {"yes": yes, "decided": total, "unclear": unclear, "precision": precision}


def sample_pmids(items: list[dict], n_single: int, n_multi: int, seed: int, sample_path: Path) -> list[str]:
    if sample_path.exists():
        obj = json.loads(sample_path.read_text(encoding="utf-8"))
        pmids = obj.get("pmids", [])
        if isinstance(pmids, list) and pmids:
            return [str(x) for x in pmids]

    singles = [it["pmid"] for it in items if it["n_pub"] == 1 and it["n_ren"] == 1]
    multis = [it["pmid"] for it in items if not (it["n_pub"] == 1 and it["n_ren"] == 1)]

    rng = random.Random(seed)
    rng.shuffle(singles)
    rng.shuffle(multis)

    take_single = singles[: min(n_single, len(singles))]
    take_multi = multis[: min(n_multi, len(multis))]

    pmids = take_single + take_multi
    sample_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "n_single": n_single,
                "n_multi": n_multi,
                "single_available": len(singles),
                "multi_available": len(multis),
                "pmids": pmids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pmids


def fmt_prec(x):
    return f"{x:.3f}" if isinstance(x, float) else "NA"


st.set_page_config(page_title="Shared PMID treatment evaluation", layout="wide")
st.title("Shared PMID drug treats disease evaluation (75 single, 75 multi)")

with st.sidebar:
    st.header("Inputs")
    default_pub = "pubtator_outputs/drug_disease_pmid_long.csv"
    default_ren = "rentrez_data/pubmed_results.csv"

    pub_path = Path(st.text_input("PubTator CSV path", value=default_pub))
    ren_path = Path(st.text_input("Rentrez CSV path", value=default_ren))

    require_positive = st.checkbox("Rentrez: require positive == TRUE (if column exists)", value=True)
    annotator = st.text_input("Annotator name", value="")

    st.divider()
    st.header("Sampling")
    n_single = st.number_input("Number of single-claim PMIDs", min_value=1, max_value=500, value=75, step=1)
    n_multi = st.number_input("Number of multi-claim PMIDs", min_value=1, max_value=500, value=75, step=1)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)

    st.divider()
    st.header("PubMed fetch")
    email = st.text_input("NCBI email (recommended)", value="")

    st.divider()
    st.header("Outputs")
    run_tag = st.text_input("Run tag", value="shared_pmids_eval_stratified").strip()
    reset_sample = st.checkbox("Resample (overwrite saved sample list)", value=False)

if not pub_path.exists():
    st.error(f"PubTator file not found: {pub_path}")
    st.stop()
if not ren_path.exists():
    st.error(f"Rentrez file not found: {ren_path}")
    st.stop()

pmid_eval_csv = OUT_DIR / f"{run_tag}_pmid_eval.csv"
sample_json = OUT_DIR / f"{run_tag}_sample_pmids.json"

if reset_sample and sample_json.exists():
    sample_json.unlink(missing_ok=True)

ensure_eval_csv_schema(pmid_eval_csv)

try:
    items_all = build_items_cached(str(pub_path), str(ren_path), bool(require_positive))
except Exception as e:
    st.error(str(e))
    st.stop()

sample_pmids_list = sample_pmids(items_all, int(n_single), int(n_multi), int(seed), sample_json)
sample_set = set(sample_pmids_list)

items = [it for it in items_all if it["pmid"] in sample_set]
items = sorted(items, key=lambda it: sample_pmids_list.index(it["pmid"]))

existing_eval = load_existing_pmid_eval(pmid_eval_csv)
completed_pmids = set(existing_eval.keys()) & {it["pmid"] for it in items}
remaining_pmids = [it["pmid"] for it in items if it["pmid"] not in completed_pmids]

st.caption(
    f"Annotating {len(items):,} sampled PMIDs. Completed {len(completed_pmids):,}. Remaining {len(remaining_pmids):,}."
)

colA, colB, colC, colD = st.columns(4)
colA.metric("Sampled PMIDs", f"{len(items):,}")
colB.metric("Completed", f"{len(completed_pmids):,}")
colC.metric("Remaining", f"{len(remaining_pmids):,}")
colD.metric("Sample file", sample_json.name)

if not items:
    st.error("No sampled PMIDs available. Check inputs or resample.")
    st.stop()

if "idx" not in st.session_state:
    st.session_state.idx = 0
if "jump_pmid" not in st.session_state:
    st.session_state.jump_pmid = ""

nav1, nav2, nav3, nav4 = st.columns([1, 1, 2, 3])
with nav1:
    if st.button("Back", use_container_width=True) and st.session_state.idx > 0:
        st.session_state.idx -= 1
with nav2:
    if st.button("Next", use_container_width=True) and st.session_state.idx < len(items) - 1:
        st.session_state.idx += 1
with nav3:
    st.session_state.jump_pmid = st.text_input("Jump to PMID", value=st.session_state.jump_pmid).strip()
    if st.button("Go", use_container_width=True) and st.session_state.jump_pmid:
        pmids = [it["pmid"] for it in items]
        if st.session_state.jump_pmid in pmids:
            st.session_state.idx = pmids.index(st.session_state.jump_pmid)
        else:
            st.warning("PMID not in sampled set.")
with nav4:
    show_only_remaining = st.checkbox("Focus on remaining PMIDs", value=False)
    if show_only_remaining and remaining_pmids:
        current_pmid = items[st.session_state.idx]["pmid"]
        if current_pmid in completed_pmids:
            pmids = [it["pmid"] for it in items]
            st.session_state.idx = pmids.index(remaining_pmids[0])

it = items[st.session_state.idx]
pmid = it["pmid"]
group = "single" if (it["n_pub"] == 1 and it["n_ren"] == 1) else "multi"
is_completed = pmid in completed_pmids

left, right = st.columns([2, 1])

with left:
    st.subheader(f"PMID: {pmid} | group: {group} {'| completed' if is_completed else ''}")

    manual_mode = st.checkbox(
        "Manual title/abstract entry (skip PubMed fetch)",
        value=False,
        key=f"manual_mode_{pmid}",
    )

    title = ""
    abstract = ""
    fetch_err = None
    if manual_mode:
        title = st.text_input("Title", value="", key=f"title_{pmid}")
        abstract = st.text_area("Abstract", value="", height=260, key=f"abstract_{pmid}")
    else:
        try:
            rec = fetch_title_abstract(pmid, email=email)
            title = rec["title"]
            abstract = rec["abstract"]
        except Exception as e:
            fetch_err = str(e)

    if fetch_err:
        st.error(f"Failed to fetch PubMed record: {fetch_err}")
        st.info("Enable manual mode above to paste title and abstract.")
    else:
        st.markdown("**Title**")
        st.write(title)
        st.markdown("**Abstract**")
        st.write(abstract)

with right:
    st.subheader("Evaluation")

    pub_claims: list[Claim] = it["pubtator_claims"]
    ren_claims: list[Claim] = it["rentrez_claims"]

    st.markdown("### PubTator says")
    if not pub_claims:
        st.write("(No PubTator claims for this PMID.)")
    else:
        for i, c in enumerate(pub_claims, start=1):
            extra = f" ({c.disease_id})" if c.disease_id else ""
            st.write(f"{i}. {c.drug} treats {c.disease}{extra}")

    st.markdown("### Rentrez says")
    if not ren_claims:
        st.write("(No Rentrez claims for this PMID.)")
    else:
        for i, c in enumerate(ren_claims, start=1):
            st.write(f"{i}. {c.drug} treats {c.disease}")

    prev = existing_eval.get(pmid, {})
    any_relation_opts = ["", "Yes", "No", "Unclear"]

    prev_pub_keys = json_to_pub_keys(prev.get("pubtator_true_claims_json", ""))
    prev_ren_keys = json_to_ren_keys(prev.get("rentrez_true_claims_json", ""))

    pub_options = [claim_key_pub(c) for c in pub_claims]
    ren_options = [claim_key_ren(c) for c in ren_claims]

    with st.form(key=f"eval_form_{pmid}", clear_on_submit=False):
        st.markdown("### Mark the relationships supported by the title and abstract")
        pub_selected = st.multiselect(
            "PubTator supported claims (can be multiple)",
            options=pub_options,
            default=[k for k in prev_pub_keys if k in pub_options],
            format_func=format_pub_key,
            key=f"pub_selected_{pmid}",
        )
        ren_selected = st.multiselect(
            "Rentrez supported claims (can be multiple)",
            options=ren_options,
            default=[k for k in prev_ren_keys if k in ren_options],
            format_func=format_ren_key,
            key=f"ren_selected_{pmid}",
        )

        st.divider()
        st.markdown("### Optional manual label (saved even if you accept claims above)")
        manual_drug = st.text_input("Manual drug (optional)", value=prev.get("manual_drug", ""), key=f"manual_drug_{pmid}")
        manual_disease = st.text_input("Manual disease (optional)", value=prev.get("manual_disease", ""), key=f"manual_disease_{pmid}")

        st.divider()
        st.markdown("### If neither source is correct")
        st.caption("Fill this section only if you selected no claims from both PubTator and Rentrez.")
        any_relation = st.selectbox(
            "Is there any drug treats disease relationship in the title/abstract?",
            options=any_relation_opts,
            index=any_relation_opts.index(prev.get("any_relation", "")) if prev.get("any_relation", "") in any_relation_opts else 0,
            key=f"any_relation_{pmid}",
        )
        extracted_drug = st.text_input("If yes, drug", value=prev.get("extracted_drug", ""), key=f"extracted_drug_{pmid}")
        extracted_disease = st.text_input("If yes, disease", value=prev.get("extracted_disease", ""), key=f"extracted_disease_{pmid}")

        overall_notes = st.text_area("Notes (optional)", value=prev.get("overall_notes", ""), height=90, key=f"overall_notes_{pmid}")

        save_col1, save_col2 = st.columns([1, 1])
        with save_col1:
            do_save = st.form_submit_button("Save", use_container_width=True)
        with save_col2:
            do_save_next = st.form_submit_button("Save and Next", use_container_width=True)

if do_save or do_save_next:
    if not annotator.strip():
        st.warning("Annotator name is empty. Saving will continue, but it is recommended to fill it in.")

    def derive_agree(selected: list[str], options: list[str]) -> str:
        if not options:
            return "Unclear"
        return "Yes" if len(selected) > 0 else "No"

    pub_agree = derive_agree(pub_selected, pub_options)
    ren_agree = derive_agree(ren_selected, ren_options)

    pub_sel_set = set(pub_selected)
    ren_sel_set = set(ren_selected)

    pub_true_claims = [c for c in pub_claims if claim_key_pub(c) in pub_sel_set]
    ren_true_claims = [c for c in ren_claims if claim_key_ren(c) in ren_sel_set]

    if pub_selected or ren_selected:
        any_relation_to_save = ""
        extracted_drug_to_save = ""
        extracted_disease_to_save = ""
    else:
        any_relation_to_save = any_relation
        extracted_drug_to_save = (extracted_drug or "").strip()
        extracted_disease_to_save = (extracted_disease or "").strip()

    existing_eval[pmid] = {
        "saved_at_utc": _now_utc_iso(),
        "annotator": annotator.strip(),
        "pmid": pmid,
        "group": group,
        "pubtator_agree": pub_agree,
        "rentrez_agree": ren_agree,
        "pubtator_true_claims_json": claims_to_json(pub_true_claims),
        "rentrez_true_claims_json": claims_to_json(ren_true_claims),
        "manual_drug": (manual_drug or "").strip(),
        "manual_disease": (manual_disease or "").strip(),
        "any_relation": any_relation_to_save,
        "extracted_drug": extracted_drug_to_save,
        "extracted_disease": extracted_disease_to_save,
        "overall_notes": (overall_notes or "").strip(),
    }

    rows = []
    for pm, v in existing_eval.items():
        rows.append(
            {
                "saved_at_utc": v.get("saved_at_utc", ""),
                "annotator": v.get("annotator", ""),
                "pmid": pm,
                "group": v.get("group", ""),
                "pubtator_agree": v.get("pubtator_agree", ""),
                "rentrez_agree": v.get("rentrez_agree", ""),
                "pubtator_true_claims_json": v.get("pubtator_true_claims_json", ""),
                "rentrez_true_claims_json": v.get("rentrez_true_claims_json", ""),
                "manual_drug": v.get("manual_drug", ""),
                "manual_disease": v.get("manual_disease", ""),
                "any_relation": v.get("any_relation", ""),
                "extracted_drug": v.get("extracted_drug", ""),
                "extracted_disease": v.get("extracted_disease", ""),
                "overall_notes": v.get("overall_notes", ""),
            }
        )

    rows = sorted(rows, key=lambda r: (r["group"], int(r["pmid"]) if str(r["pmid"]).isdigit() else r["pmid"]))
    write_pmid_eval(pmid_eval_csv, rows)
    ensure_eval_csv_schema(pmid_eval_csv)

    st.success(f"Saved to {pmid_eval_csv.name} in {OUT_DIR}")

    if do_save_next and st.session_state.idx < len(items) - 1:
        st.session_state.idx += 1
        st.rerun()

if len(completed_pmids) == len(items) and len(items) > 0:
    st.divider()
    st.subheader("Evaluation metrics (end of sample)")

    eval_rows_live = pd.read_csv(pmid_eval_csv, dtype=str).fillna("").to_dict(orient="records")
    pub_s = compute_precision_from_pmid(eval_rows_live, "pubtator_agree")
    ren_s = compute_precision_from_pmid(eval_rows_live, "rentrez_agree")

    sum_df = pd.DataFrame(
        [
            {
                "dataset": "PubTator",
                "precision_like": fmt_prec(pub_s["precision"]),
                "yes": pub_s["yes"],
                "decided": pub_s["decided"],
                "unclear": pub_s["unclear"],
            },
            {
                "dataset": "Rentrez",
                "precision_like": fmt_prec(ren_s["precision"]),
                "yes": ren_s["yes"],
                "decided": ren_s["decided"],
                "unclear": ren_s["unclear"],
            },
        ]
    )
    st.dataframe(sum_df, use_container_width=True)
    st.caption(f"Outputs folder: {OUT_DIR} | Sample list: {sample_json.name}")
else:
    st.caption("Metrics will appear automatically after all sampled PMIDs are completed.")
