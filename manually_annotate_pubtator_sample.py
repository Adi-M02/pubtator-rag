#!/usr/bin/env python3
import json, random
from pathlib import Path
from collections import defaultdict

import pandas as pd
import streamlit as st
from Bio import Entrez

# ------------- Config -------------
OUTDIR = Path("outputs")
RUN_ID = "20251120_191408"
RUN_DIR = OUTDIR / f"pipeline_{RUN_ID}"

NCBI_EMAIL = "your_email@example.com"  # set this
NCBI_API_KEY = None  # optional

Entrez.email = NCBI_EMAIL
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY


# ------------- Core loading -------------

def load_artifacts():
    if not RUN_DIR.exists():
        raise FileNotFoundError(f"Run directory not found: {RUN_DIR}")

    json_path = RUN_DIR / f"pipeline_{RUN_ID}.json"
    summary_csv = RUN_DIR / f"pipeline_{RUN_ID}_summary.csv"
    pmid_rel_csv = RUN_DIR / f"pipeline_{RUN_ID}_pmid_relations.csv"

    for p in [json_path, summary_csv, pmid_rel_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    with open(json_path, "r", encoding="utf-8") as f:
        art = json.load(f)

    df_summary = pd.read_csv(summary_csv)
    df_summary["pmid_count"] = df_summary["pmid_count"].fillna(0).astype(int)

    df_pmid_rel = pd.read_csv(pmid_rel_csv)
    df_pmid_rel["pmid"] = df_pmid_rel["pmid"].astype(str)

    return RUN_ID, art, df_summary, df_pmid_rel


# ------------- Vocab builders -------------

def build_drug_vocab(artifact: dict) -> list[str]:
    drugs = artifact.get("drugs", [])
    return sorted(set(drugs))


def build_disease_vocab(df_summary: pd.DataFrame):
    df_d = df_summary[["disease_name", "disease_id"]].drop_duplicates()
    df_d = df_d.sort_values(["disease_name", "disease_id"])
    labels = []
    for _, r in df_d.iterrows():
        name = str(r["disease_name"])
        did = str(r["disease_id"])
        label = f"{name} [{did}]"
        labels.append((label, name, did))
    return labels


# ------------- PubTator pair mapping -------------

def build_pmid_to_pairs(artifact: dict, df_summary: pd.DataFrame) -> dict[str, list[dict]]:
    df_s = df_summary.copy()
    df_s["drug_id"] = df_s["drug_id"].astype(str)
    df_s["disease_id"] = df_s["disease_id"].astype(str)

    pair_to_pmid_count: dict[tuple[str, str], int] = {}
    for _, r in df_s.iterrows():
        key = (str(r["drug_id"]), str(r["disease_id"]))
        pair_to_pmid_count[key] = int(r["pmid_count"])

    pmid_to_year: dict[str, int | None] = {}

    for ind in artifact.get("indications", []):
        for ev in ind.get("evidence", []):
            for art in ev.get("articles", []):
                pm = str(art.get("pmid") or "").strip()
                if not pm:
                    continue
                date = art.get("date")
                year = None
                if date:
                    try:
                        year = int(str(date)[:4])
                    except Exception:
                        year = None
                if pm not in pmid_to_year and year is not None:
                    pmid_to_year[pm] = year

    pmid_to_pairs_map: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)

    for ind in artifact.get("indications", []):
        dname = ind.get("drug_name")
        did = str(ind.get("drug_id"))
        for ev in ind.get("evidence", []):
            dis_name = ev.get("disease_name")
            dis_id = str(ev.get("disease_id"))
            pmids = [str(p) for p in ev.get("pmids", [])]
            key = (did, dis_id)
            for pm in pmids:
                if not pm:
                    continue
                if key not in pmid_to_pairs_map[pm]:
                    pmid_to_pairs_map[pm][key] = {
                        "drug_name": dname,
                        "drug_id": did,
                        "disease_name": dis_name,
                        "disease_id": dis_id,
                        "pub_year": pmid_to_year.get(pm),
                        "pmid_count_pair": pair_to_pmid_count.get(key, 0),
                    }

    pmid_to_pairs: dict[str, list[dict]] = {}
    for pm, pairs_dict in pmid_to_pairs_map.items():
        pmid_to_pairs[pm] = list(pairs_dict.values())

    return pmid_to_pairs


# ------------- PubMed fetching (batched) -------------

@st.cache_data(show_spinner=True)
def fetch_batch_abstracts(pmids: list[str]) -> dict[str, tuple[str, str]]:
    if not pmids:
        return {}

    results: dict[str, tuple[str, str]] = {}
    chunk_size = 50

    for i in range(0, len(pmids), chunk_size):
        chunk = pmids[i : i + chunk_size]

        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(chunk),
                rettype="abstract",
                retmode="xml",
            )
            records = Entrez.read(handle)
            handle.close()
        except Exception:
            for pm in chunk:
                results.setdefault(pm, ("", ""))
            continue

        for art in records.get("PubmedArticle", []):
            try:
                pmid_field = art["MedlineCitation"]["PMID"]
                pmid_val = str(pmid_field)
            except Exception:
                continue

            article = art["MedlineCitation"]["Article"]
            title = str(article.get("ArticleTitle", ""))
            abs_obj = article.get("Abstract", {})
            abs_parts = abs_obj.get("AbstractText", [])
            abstract = " ".join(str(x) for x in abs_parts)

            results[pmid_val] = (title, abstract)

        for pm in chunk:
            results.setdefault(pm, ("", ""))

    return results


# ------------- Sampling at PMID level -------------

def sample_pmids_for_annotation(
    artifact: dict,
    df_summary: pd.DataFrame,
    df_pmid_rel: pd.DataFrame,
    n_single: int,
    n_multi: int,
    seed: int = 0,
) -> pd.DataFrame:
    rng = random.Random(seed)

    pmid_to_pairs = build_pmid_to_pairs(artifact, df_summary)

    df_pmid = df_pmid_rel.copy()
    df_pmid["pmid"] = df_pmid["pmid"].astype(str)
    df_pmid = df_pmid[df_pmid["pmid"].isin(pmid_to_pairs.keys())]

    df_single = df_pmid[df_pmid["n_pairs"] == 1]
    df_multi = df_pmid[df_pmid["n_pairs"] > 1]

    n_single_eff = min(n_single, len(df_single))
    n_multi_eff = min(n_multi, len(df_multi))

    df_single_s = df_single.sample(n=n_single_eff, random_state=rng.randint(0, 10**9))
    df_multi_s = df_multi.sample(n=n_multi_eff, random_state=rng.randint(0, 10**9))

    triple_rows = []

    def add_rows(sub_df: pd.DataFrame, group_label: str):
        for _, r in sub_df.iterrows():
            pmid = str(r["pmid"])
            pairs = pmid_to_pairs.get(pmid, [])
            if not pairs:
                continue

            pairs_json = json.dumps(pairs, ensure_ascii=False)
            primary_pair = pairs[0]
            pub_year = primary_pair.get("pub_year")

            triple_rows.append(
                {
                    "group": group_label,
                    "pmid": pmid,
                    "pub_year": pub_year,
                    "drug_name_pt": primary_pair["drug_name"],
                    "drug_id_pt": primary_pair["drug_id"],
                    "disease_name_pt": primary_pair["disease_name"],
                    "disease_id_pt": primary_pair["disease_id"],
                    "pmid_count_pair": int(primary_pair.get("pmid_count_pair", 0)),
                    "n_pairs_pmid": int(r["n_pairs"]),
                    "n_drugs_pmid": int(r["n_drugs"]),
                    "n_diseases_pmid": int(r["n_diseases"]),
                    "pmid_has_one_drug_multi_diseases": bool(
                        r["has_one_drug_multi_diseases"]
                    ),
                    "pmid_has_one_disease_multi_drugs": bool(
                        r["has_one_disease_multi_drugs"]
                    ),
                    "pt_pairs_json": pairs_json,
                }
            )

    add_rows(df_single_s, "pmid_single_rel")
    add_rows(df_multi_s, "pmid_multi_rel")

    rng.shuffle(triple_rows)
    for i, row in enumerate(triple_rows):
        row["row_id"] = i

    df_sample = pd.DataFrame(triple_rows)
    return df_sample


# ------------- Annotation setup -------------

def load_or_init_sample(
    n_single: int,
    n_multi: int,
    seed: int,
) -> tuple[str, pd.DataFrame, Path]:
    rid, artifact, df_summary, df_pmid_rel = load_artifacts()
    ann_csv = RUN_DIR / f"pubtator_manual_annotations_{rid}.csv"

    if ann_csv.exists():
        df = pd.read_csv(ann_csv)

        # Ensure all expected columns exist
        for col in [
            "annotator_id",
            "pubtator_agree",
            "pt_pair_agree_json",
            "other_relations",
            "other_drug_name",
            "other_disease_name",
            "other_disease_id",
            "other_drug_free_text",
            "other_disease_free_text",
        ]:
            if col not in df.columns:
                df[col] = ""
        return rid, df, ann_csv

    df_sample = sample_pmids_for_annotation(
        artifact=artifact,
        df_summary=df_summary,
        df_pmid_rel=df_pmid_rel,
        n_single=n_single,
        n_multi=n_multi,
        seed=seed,
    )

    if df_sample.empty:
        raise RuntimeError(
            "Sampling produced an empty set of PMIDs. "
            "Check that the pmid_relations CSV and JSON artifact are consistent."
        )

    st.write("Sample size (rows):", len(df_sample))
    st.write("Unique PMIDs:", df_sample["pmid"].nunique())

    pmid_list = [str(p) for p in df_sample["pmid"]]
    pmid_to_text = fetch_batch_abstracts(pmid_list)

    titles, abstracts = [], []
    for pmid in pmid_list:
        t, a = pmid_to_text.get(pmid, ("", ""))
        titles.append(t)
        abstracts.append(a)

    df_sample["title"] = titles
    df_sample["abstract"] = abstracts

    df_sample["annotator_id"] = ""
    df_sample["pubtator_agree"] = ""
    df_sample["pt_pair_agree_json"] = ""
    df_sample["other_relations"] = ""
    df_sample["other_drug_name"] = ""
    df_sample["other_disease_name"] = ""
    df_sample["other_disease_id"] = ""
    df_sample["other_drug_free_text"] = ""
    df_sample["other_disease_free_text"] = ""

    df_sample.to_csv(ann_csv, index=False)
    return rid, df_sample, ann_csv


def update_row(df: pd.DataFrame, idx: int, values: dict):
    for k, v in values.items():
        df.at[idx, k] = v


# ------------- Streamlit app -------------

def main():
    st.title("PubTator manual treatment relation evaluation")

    if not RUN_DIR.exists():
        st.error(f"Hardcoded run directory does not exist: {RUN_DIR}")
        return

    n_single = st.sidebar.number_input(
        "Single-relation PMIDs to sample", 1, 500, 75
    )
    n_multi = st.sidebar.number_input(
        "Multi-relation PMIDs to sample", 1, 500, 75
    )
    seed = st.sidebar.number_input("Sampling random seed", 0, 10_000, 0)
    annotator_default = st.sidebar.text_input("Annotator ID", "")

    if "sample_df" not in st.session_state:
        if st.button("Load or generate sample"):
            rid, df_sample, ann_csv = load_or_init_sample(n_single, n_multi, seed)
            st.session_state["run_id"] = rid
            st.session_state["annotation_csv"] = str(ann_csv)
            st.session_state["sample_df"] = df_sample
            st.session_state["current_idx"] = 0
    else:
        st.success("Sample loaded")

    if "sample_df" not in st.session_state:
        st.info("Click 'Load or generate sample' to begin.")
        return

    df = st.session_state["sample_df"]
    ann_csv_path = Path(st.session_state["annotation_csv"])

    if "drug_vocab" not in st.session_state or "disease_vocab" not in st.session_state:
        _, artifact, df_summary, _ = load_artifacts()
        st.session_state["drug_vocab"] = build_drug_vocab(artifact)
        st.session_state["disease_vocab"] = build_disease_vocab(df_summary)

    drugs = st.session_state["drug_vocab"]
    disease_entries = st.session_state["disease_vocab"]

    total_items = len(df)
    completed = df["pt_pair_agree_json"].astype(str).str.len().gt(0).sum()
    st.write(f"Progress: {completed} of {total_items} items annotated")

    col_a, col_b = st.columns(2)
    with col_a:
        idx = st.session_state.get("current_idx", 0)
        idx = st.number_input("Current item index", 0, total_items - 1, idx, step=1)
        st.session_state["current_idx"] = idx
    with col_b:
        st.write(f"Showing item {idx + 1} of {total_items}")

    row = df.iloc[idx]

    st.markdown(
        f"**PMID:** {row['pmid']}  "
        f"[Open on PubMed](https://pubmed.ncbi.nlm.nih.gov/{row['pmid']}/)"
    )

    try:
        pairs = json.loads(row.get("pt_pairs_json", "[]") or "[]")
    except json.JSONDecodeError:
        pairs = []

    n_pairs = int(row.get("n_pairs_pmid", len(pairs) or 0))

    if n_pairs <= 0 or not pairs:
        st.warning("No PubTator candidate relations were stored for this PMID.")
    else:
        if n_pairs == 1:
            p0 = pairs[0]
            st.markdown(
                f"**PubTator claimed relation:** "
                f"{p0['drug_name']} treats {p0['disease_name']}"
            )
        else:
            st.markdown("**PubTator claimed relations in this article:**")
            for i, p in enumerate(pairs):
                st.markdown(
                    f"- Relation {i + 1}: {p['drug_name']} treats {p['disease_name']}"
                )

    if pd.notna(row.get("pub_year")):
        try:
            st.markdown(f"**Publication year:** {int(row['pub_year'])}")
        except Exception:
            pass

    st.markdown("### PubTator relation evaluation")

    st.markdown("Article title")
    st.write(row["title"] or "(no title found)")
    st.markdown("Abstract")
    st.write(row["abstract"] or "(no abstract found)")

    prev_pair_agree = {}
    if isinstance(row.get("pt_pair_agree_json"), str) and row["pt_pair_agree_json"]:
        try:
            prev_pair_agree = json.loads(row["pt_pair_agree_json"])
        except json.JSONDecodeError:
            prev_pair_agree = {}

    pair_agree_answers = {}
    options_agree = ["", "yes", "no", "unclear"]

    if n_pairs > 0 and pairs:
        if n_pairs == 1:
            p0 = pairs[0]
            prev_ans = prev_pair_agree.get("0", "")
            try:
                default_idx = options_agree.index(prev_ans)
            except ValueError:
                default_idx = 0

            ans0 = st.radio(
                f"Does the title and abstract support the PubTator claimed relation "
                f"{p0['drug_name']} treats {p0['disease_name']}?",
                options=options_agree,
                index=default_idx,
                key=f"ptagree_{idx}_0",
                format_func=lambda x: "(select)" if x == "" else x,
            )
            pair_agree_answers["0"] = ans0
            pubtator_agree_value = ans0
        else:
            st.markdown(
                "For each PubTator relation below, indicate whether the title and abstract "
                "support that specific treatment relation."
            )
            pubtator_agree_value = row["pubtator_agree"] if isinstance(
                row.get("pubtator_agree"), str
            ) else ""

            for i, p in enumerate(pairs):
                key = str(i)
                prev_ans = prev_pair_agree.get(key, "")
                try:
                    default_idx = options_agree.index(prev_ans)
                except ValueError:
                    default_idx = 0
                ans_i = st.radio(
                    f"Relation {i + 1}: {p['drug_name']} treats {p['disease_name']}",
                    options=options_agree,
                    index=default_idx,
                    key=f"ptagree_{idx}_{i}",
                    format_func=lambda x: "(select)" if x == "" else x,
                )
                pair_agree_answers[key] = ans_i
    else:
        pubtator_agree_value = row["pubtator_agree"] if isinstance(
            row.get("pubtator_agree"), str
        ) else ""

    other_opts = ["", "yes", "no", "unclear"]
    current_other = row["other_relations"] if isinstance(
        row.get("other_relations"), str
    ) else ""
    try:
        other_default_idx = other_opts.index(current_other)
    except ValueError:
        other_default_idx = 0

    other_relations_value = current_other

    if n_pairs == 1 and n_pairs > 0 and pairs:
        ans0 = pair_agree_answers.get("0", "")
        if ans0 == "yes":
            other_relations_value = st.radio(
                "Beyond this relation, does the abstract clearly describe any "
                "other drug disease treatment relationships?",
                options=other_opts,
                index=other_default_idx,
                key=f"other_rel_{idx}",
                format_func=lambda x: "(select)" if x == "" else x,
            )
    elif n_pairs > 1:
        other_relations_value = st.radio(
            "Beyond the relations listed above, does the abstract clearly describe any "
            "additional drug disease treatment relationships?",
            options=other_opts,
            index=other_default_idx,
            key=f"other_rel_{idx}",
            format_func=lambda x: "(select)" if x == "" else x,
        )

    other_drug_prev = row["other_drug_name"] if isinstance(
        row.get("other_drug_name"), str
    ) else ""
    other_did_prev = row["other_disease_id"] if isinstance(
        row.get("other_disease_id"), str
    ) else ""
    other_drug_free_prev = row["other_drug_free_text"] if isinstance(
        row.get("other_drug_free_text"), str
    ) else ""
    other_disease_free_prev = row["other_disease_free_text"] if isinstance(
        row.get("other_disease_free_text"), str
    ) else ""

    other_drug_selected = ""
    other_disease_name_selected = ""
    other_disease_id_selected = ""
    other_drug_free = other_drug_free_prev
    other_disease_free = other_disease_free_prev

    if other_relations_value == "yes":
        st.markdown("#### Additional drug disease relation")

        drug_options = ["(none / not in 150)"] + drugs
        if other_drug_prev and other_drug_prev in drugs:
            drug_index = drug_options.index(other_drug_prev)
        elif other_drug_prev == "(none / not in 150)":
            drug_index = 0
        else:
            drug_index = 0

        sel_other_drug = st.selectbox(
            "Select the additional treatment drug (if in configured list)",
            options=drug_options,
            index=drug_index,
            key=f"other_drug_{idx}",
        )
        if sel_other_drug != "(none / not in 150)":
            other_drug_selected = sel_other_drug
        else:
            other_drug_selected = "(none / not in 150)"

        other_drug_free = st.text_input(
            "Or type the additional drug name if it is not in the list",
            value=other_drug_free_prev,
            key=f"other_drug_free_{idx}",
        )

        disease_search = st.text_input(
            "Filter diseases by name or ID for the additional relation",
            value="",
            key=f"other_disease_search_{idx}",
        )

        filtered_labels = [
            label
            for (label, name, did) in disease_entries
            if disease_search.lower() in label.lower()
        ]
        if not filtered_labels:
            filtered_labels = ["(no matches)"]

        disease_options = ["(none / not in list)"] + filtered_labels

        if other_did_prev:
            current_label = None
            for label, name, did in disease_entries:
                if did == other_did_prev:
                    current_label = label
                    break
            if current_label and current_label in disease_options:
                dis_index = disease_options.index(current_label)
            else:
                dis_index = 0
        else:
            dis_index = 0

        sel_other_dis = st.selectbox(
            "Select the additional disease or condition (if in list)",
            options=disease_options,
            index=dis_index,
            key=f"other_disease_{idx}",
        )

        if sel_other_dis not in ("(none / not in list)", "(no matches)"):
            for label, name, did in disease_entries:
                if label == sel_other_dis:
                    other_disease_name_selected = name
                    other_disease_id_selected = did
                    break
        else:
            other_disease_name_selected = "(none / not in list)"
            other_disease_id_selected = ""

        other_disease_free = st.text_input(
            "Or type the additional disease or condition name if it is not in the list",
            value=other_disease_free_prev,
            key=f"other_disease_free_{idx}",
        )

    ann_id_current = row["annotator_id"] if isinstance(row["annotator_id"], str) else ""
    annotator_id = st.text_input(
        "Annotator ID for this item",
        value=ann_id_current or annotator_default,
        key=f"annotator_id_{idx}",
    )

    col_prev, col_save, col_next = st.columns(3)
    with col_prev:
        if st.button("Previous", disabled=(idx == 0)):
            st.session_state["current_idx"] = max(0, idx - 1)
    with col_next:
        if st.button("Next", disabled=(idx == total_items - 1)):
            st.session_state["current_idx"] = min(total_items - 1, idx + 1)
    with col_save:
        if st.button("Save annotation"):
            values = {
                "annotator_id": annotator_id,
                "other_relations": other_relations_value,
                "pt_pair_agree_json": json.dumps(pair_agree_answers, ensure_ascii=False),
            }

            if n_pairs == 1 and "0" in pair_agree_answers:
                values["pubtator_agree"] = pair_agree_answers["0"]
            else:
                values["pubtator_agree"] = row["pubtator_agree"] if isinstance(
                    row.get("pubtator_agree"), str
                ) else ""

            if other_relations_value == "yes":
                values["other_drug_name"] = other_drug_selected or ""
                values["other_disease_name"] = other_disease_name_selected
                values["other_disease_id"] = other_disease_id_selected
                values["other_drug_free_text"] = other_drug_free
                values["other_disease_free_text"] = other_disease_free
            else:
                values["other_drug_name"] = ""
                values["other_disease_name"] = ""
                values["other_disease_id"] = ""
                values["other_drug_free_text"] = ""
                values["other_disease_free_text"] = ""

            update_row(df, idx, values)
            df.to_csv(ann_csv_path, index=False)
            st.session_state["sample_df"] = df
            st.success("Annotation saved")

            if idx < total_items - 1:
                st.session_state["current_idx"] = idx + 1


if __name__ == "__main__":
    main()
