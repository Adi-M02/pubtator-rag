#!/usr/bin/env python3
import argparse, json, re, time
from pathlib import Path
from collections import Counter

import pandas as pd
from tqdm import tqdm

from pubtator_api import pubtator_entity_autocomplete_list_fast

ORIG_COLS = ["drug", "nct_id", "title", "status", "phase", "disease", "first_submit"]


def norm_basic(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def singularize_token(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("ses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def token_multiset(s: str, singularize: bool) -> Counter:
    toks = norm_basic(s).split()
    if singularize:
        toks = [singularize_token(t) for t in toks]
    toks = [t for t in toks if t]
    return Counter(toks)


def is_blocked_non_disease(d: str, block: set[str]) -> bool:
    return norm_basic(d) in block or norm_basic(d) == ""


def atomic_write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def _safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _sanitize_cache_entry(x: dict) -> dict:
    if not isinstance(x, dict):
        x = {}

    top5 = x.get("top5", [])
    if top5 is None:
        top5 = []
    if isinstance(top5, str):
        try:
            top5 = json.loads(top5)
        except Exception:
            top5 = []
    if not isinstance(top5, list):
        top5 = []

    return {
        "best_label": "" if x.get("best_label") is None else str(x.get("best_label", "")),
        "best_id": "" if x.get("best_id") is None else str(x.get("best_id", "")),
        "status": "" if x.get("status") is None else str(x.get("status", "")),
        "top5": top5,
        "n_candidates_returned": _safe_int(x.get("n_candidates_returned"), 0),
        "extra_tokens": _safe_int(x.get("extra_tokens"), 0),
        "shared_tokens": _safe_int(x.get("shared_tokens"), 0),
        "flag_ambiguous": bool(x.get("flag_ambiguous", False)),
        "flag_too_specific": bool(x.get("flag_too_specific", False)),
        "flag_low_conf": bool(x.get("flag_low_conf", False)),
        "reject_reason": "" if x.get("reject_reason") is None else str(x.get("reject_reason", "")),
    }


def resolve_best_candidate(
    query: str,
    candidates: list[tuple[str, str]],
    singularize: bool,
    max_extra_tokens: int,
) -> dict:
    qn = norm_basic(query)
    q_ms = token_multiset(query, singularize=singularize)
    q_len = sum(q_ms.values())

    scored = []
    for lab, eid in candidates:
        ln = norm_basic(lab)
        l_ms = token_multiset(lab, singularize=singularize)

        extra = sum((l_ms - q_ms).values())
        shared = sum((l_ms & q_ms).values())

        if ln == qn and qn:
            match_type = "exact"
            score = 1.0
        elif q_len > 0 and l_ms == q_ms:
            match_type = "token_set_exact"
            score = 0.99
        elif shared > 0 and q_len > 0:
            match_type = "partial_overlap"
            score = 0.5 + 0.49 * (shared / max(q_len, 1))
        else:
            match_type = "no_overlap"
            score = 0.0

        scored.append(
            {
                "label": lab,
                "id": eid,
                "score": float(score),
                "match_type": match_type,
                "extra_tokens": int(extra),
                "shared_tokens": int(shared),
            }
        )

    scored.sort(key=lambda r: (r["score"], -r["shared_tokens"], -r["extra_tokens"]), reverse=True)
    best = scored[0]
    tie = len(scored) > 1 and scored[1]["score"] == best["score"] and best["score"] > 0

    flag_amb = bool(tie)
    flag_too_spec = best["extra_tokens"] > max_extra_tokens
    flag_low_conf = best["score"] == 0.0

    status = best["match_type"]
    if flag_amb:
        status = "ambiguous_top_score_tie"
    elif flag_too_spec:
        status = "too_specific"
    elif flag_low_conf:
        status = "low_conf_best"

    return {
        "best_label": best["label"],
        "best_id": best["id"],
        "status": status,
        "top5": [(r["label"], r["id"]) for r in scored[:5]],
        "n_candidates_returned": len(candidates),
        "extra_tokens": int(best["extra_tokens"]),
        "shared_tokens": int(best["shared_tokens"]),
        "flag_ambiguous": flag_amb,
        "flag_too_specific": flag_too_spec,
        "flag_low_conf": flag_low_conf,
        "reject_reason": "",
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ct-in", type=Path, required=True)

    ap.add_argument("--outdir", type=Path, default=Path("ct_filtering_testing"))

    ap.add_argument("--split", type=str, default=";")
    ap.add_argument("--singularize", action="store_true")

    ap.add_argument("--max-extra-tokens", type=int, required=True)
    ap.add_argument("--min-shared-tokens", type=int, default=1)

    ap.add_argument("--use-default-blocklist", action="store_true")
    ap.add_argument("--blocklist", type=Path, default=None)

    ap.add_argument("--top-review", type=int, default=50)

    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--flush-seconds", type=float, default=60.0)

    ap.add_argument("--cache-in", type=Path, default=None)
    ap.add_argument("--cache-out", type=Path, default=None)

    ap.add_argument("--limit", type=int, default=10)

    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    verified_any_path = outdir / "ctgov_verified_long_any.csv"
    thrown_any_path = outdir / "ctgov_thrown_out_long_any.csv"
    verified_min_path = outdir / f"ctgov_verified_long_minoverlap{args.min_shared_tokens}.csv"
    thrown_min_path = outdir / f"ctgov_thrown_out_long_minoverlap{args.min_shared_tokens}.csv"
    top_any_path = outdir / "ctgov_thrown_out_top_any.csv"
    top_min_path = outdir / f"ctgov_thrown_out_top_minoverlap{args.min_shared_tokens}.csv"
    disease_table_path = outdir / "ctgov_disease_resolution_table.csv"
    summary_path = outdir / "ctgov_filter_summary.json"

    cache_in = args.cache_in
    if cache_in is None:
        cache_in = outdir / "ctgov_resolution_cache.json"

    cache_out = args.cache_out
    if cache_out is None:
        cache_out = outdir / "ctgov_resolution_cache.json"

    df = pd.read_csv(args.ct_in)
    df["_row_id"] = range(len(df))

    miss = set(ORIG_COLS) - set(df.columns)
    if miss:
        raise SystemExit(f"CT file missing columns: {sorted(miss)}")

    df["first_submit"] = pd.to_datetime(df["first_submit"], errors="coerce", utc=True)

    block = set()
    if args.use_default_blocklist:
        block |= {"healthy", "healthy volunteer", "healthy volunteers"}
    if args.blocklist and args.blocklist.exists():
        block |= {
            norm_basic(x)
            for x in args.blocklist.read_text(encoding="utf-8").splitlines()
            if norm_basic(x)
        }

    rows = []
    for _, r in df.iterrows():
        parts = [p.strip() for p in str(r["disease"]).split(args.split)]
        parts = [p for p in parts if p]
        if not parts:
            parts = [""]

        for d in parts:
            rr = r.to_dict()
            rr["disease_raw"] = d
            rows.append(rr)

    ct = pd.DataFrame(rows)

    cache = {}
    if cache_in.exists():
        try:
            cache = json.loads(cache_in.read_text(encoding="utf-8"))
            if not isinstance(cache, dict):
                cache = {}
        except Exception:
            cache = {}

    cache = {str(k): _sanitize_cache_entry(v) for k, v in cache.items()}

    resolved_rows = []
    dirty = 0
    last_flush = time.time()

    n_blocked = 0
    n_cached = 0
    n_queried = 0
    n_no_candidates = 0

    it = tqdm(ct["disease_raw"].astype(str).tolist(), desc="Resolving diseases", unit="row")
    for d in it:
        d0 = str(d).strip()

        if is_blocked_non_disease(d0, block):
            n_blocked += 1
            out = {
                "best_label": "",
                "best_id": "",
                "status": "blocked_non_disease",
                "top5": [],
                "n_candidates_returned": 0,
                "extra_tokens": 0,
                "shared_tokens": 0,
                "flag_ambiguous": False,
                "flag_too_specific": False,
                "flag_low_conf": False,
                "reject_reason": "blocked_non_disease",
            }
            resolved_rows.append(out)
        elif d0 in cache:
            n_cached += 1
            out = _sanitize_cache_entry(cache[d0])
            cache[d0] = out
            resolved_rows.append(out)
        else:
            n_queried += 1
            cand = pubtator_entity_autocomplete_list_fast(d0, concept="DISEASE", limit=args.limit)
            if not cand:
                n_no_candidates += 1
                out = {
                    "best_label": "",
                    "best_id": "",
                    "status": "unresolved",
                    "top5": [],
                    "n_candidates_returned": 0,
                    "extra_tokens": 0,
                    "shared_tokens": 0,
                    "flag_ambiguous": False,
                    "flag_too_specific": False,
                    "flag_low_conf": False,
                    "reject_reason": "no_candidates",
                }
            else:
                out = resolve_best_candidate(
                    query=d0,
                    candidates=cand,
                    singularize=args.singularize,
                    max_extra_tokens=args.max_extra_tokens,
                )

            out = _sanitize_cache_entry(out)
            cache[d0] = out
            resolved_rows.append(out)
            dirty += 1

        it.set_postfix(
            blocked=n_blocked,
            cached=n_cached,
            queried=n_queried,
            no_candidates=n_no_candidates,
            cache_size=len(cache),
        )

        now = time.time()
        if dirty >= args.flush_every or (dirty > 0 and (now - last_flush) >= args.flush_seconds):
            atomic_write_json(cache_out, cache)
            dirty = 0
            last_flush = now

    atomic_write_json(cache_out, cache)

    ct["disease_resolved_label"] = [x.get("best_label", "") for x in resolved_rows]
    ct["disease_resolved_id"] = [x.get("best_id", "") for x in resolved_rows]
    ct["resolve_status"] = [x.get("status", "") for x in resolved_rows]
    ct["reject_reason"] = [x.get("reject_reason", "") for x in resolved_rows]
    ct["flag_ambiguous"] = [bool(x.get("flag_ambiguous", False)) for x in resolved_rows]
    ct["flag_too_specific"] = [bool(x.get("flag_too_specific", False)) for x in resolved_rows]
    ct["flag_low_conf"] = [bool(x.get("flag_low_conf", False)) for x in resolved_rows]
    ct["top5_candidates"] = [json.dumps(x.get("top5", [])) for x in resolved_rows]

    ct["extra_tokens"] = (
        pd.to_numeric(pd.Series([x.get("extra_tokens") for x in resolved_rows]), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    ct["shared_tokens"] = (
        pd.to_numeric(pd.Series([x.get("shared_tokens") for x in resolved_rows]), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    ct["n_candidates_returned"] = (
        pd.to_numeric(pd.Series([x.get("n_candidates_returned") for x in resolved_rows]), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    ct["accept_any"] = (ct["reject_reason"] == "") & (ct["disease_resolved_id"] != "")
    ct["accept_minoverlap"] = ct["accept_any"] & (ct["shared_tokens"] >= int(args.min_shared_tokens))

    ct["reject_reason_minoverlap"] = ""
    ct.loc[~ct["accept_any"], "reject_reason_minoverlap"] = ct.loc[~ct["accept_any"], "reject_reason"]
    ct.loc[ct["accept_any"] & ~ct["accept_minoverlap"], "reject_reason_minoverlap"] = "min_overlap_excluded"

    verified_any = ct[ct["accept_any"]].copy()
    thrown_any = ct[~ct["accept_any"]].copy()

    verified_min = ct[ct["accept_minoverlap"]].copy()
    thrown_min = ct[~ct["accept_minoverlap"]].copy()

    verified_any.to_csv(verified_any_path, index=False)
    thrown_any.to_csv(thrown_any_path, index=False)
    verified_min.to_csv(verified_min_path, index=False)
    thrown_min.to_csv(thrown_min_path, index=False)

    top_any = (
        thrown_any.groupby(["disease_raw", "reject_reason", "resolve_status"])
        .agg(n_trials=("nct_id", "nunique"), n_rows=("disease_raw", "size"))
        .reset_index()
        .sort_values(["n_trials", "n_rows"], ascending=False)
        .head(args.top_review)
    )
    top_any.to_csv(top_any_path, index=False)

    top_min = (
        thrown_min.groupby(["disease_raw", "reject_reason_minoverlap", "resolve_status"])
        .agg(n_trials=("nct_id", "nunique"), n_rows=("disease_raw", "size"))
        .reset_index()
        .sort_values(["n_trials", "n_rows"], ascending=False)
        .head(args.top_review)
    )
    top_min.to_csv(top_min_path, index=False)

    disease_table = (
        ct.groupby(["disease_raw", "disease_resolved_label", "disease_resolved_id", "resolve_status"])
        .agg(
            n_rows=("disease_raw", "size"),
            n_trials=("nct_id", "nunique"),
            accept_any=("accept_any", "max"),
            accept_minoverlap=("accept_minoverlap", "max"),
            shared_tokens=("shared_tokens", "max"),
            extra_tokens=("extra_tokens", "max"),
            flag_ambiguous=("flag_ambiguous", "max"),
            flag_too_specific=("flag_too_specific", "max"),
            flag_low_conf=("flag_low_conf", "max"),
            n_candidates_returned=("n_candidates_returned", "max"),
            reject_reason=("reject_reason", lambda s: s.iloc[0] if len(s) else ""),
        )
        .reset_index()
        .sort_values(["n_trials", "n_rows"], ascending=False)
    )
    disease_table.to_csv(disease_table_path, index=False)

    summary = {
        "inputs": {
            "ct_in": str(args.ct_in),
            "split": args.split,
            "singularize": bool(args.singularize),
            "max_extra_tokens": int(args.max_extra_tokens),
            "min_shared_tokens": int(args.min_shared_tokens),
            "limit": int(args.limit),
        },
        "cache": {
            "cache_in": str(cache_in),
            "cache_out": str(cache_out),
            "cache_size": int(len(cache)),
        },
        "counts": {
            "ct_rows_total_after_split": int(len(ct)),
            "blocked_rows": int(n_blocked),
            "no_candidates_rows": int(n_no_candidates),
            "verified_any_rows": int(len(verified_any)),
            "thrown_any_rows": int(len(thrown_any)),
            "verified_minoverlap_rows": int(len(verified_min)),
            "thrown_minoverlap_rows": int(len(thrown_min)),
        },
        "outputs": {
            "verified_any": str(verified_any_path),
            "thrown_any": str(thrown_any_path),
            "verified_minoverlap": str(verified_min_path),
            "thrown_minoverlap": str(thrown_min_path),
            "top_any": str(top_any_path),
            "top_minoverlap": str(top_min_path),
            "disease_table": str(disease_table_path),
        },
    }
    atomic_write_json(summary_path, summary)

    print("CT filtering summary")
    print(f"  outdir: {outdir}")
    print(f"  rows after split: {len(ct)}")
    print(f"  blocked: {n_blocked}")
    print(f"  no candidates: {n_no_candidates}")
    print(f"  verified_any: {len(verified_any)}")
    print(f"  verified_minoverlap: {len(verified_min)}")
    print(f"  wrote cache_out: {cache_out}")
    print(f"  wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
