#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def sniff_sep(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        line = f.readline()
    return "\t" if "\t" in line else ","


def parse_def(path: Path):
    m = re.match(r"mim(?:ic|c)_(\d+)_(all|\d+)_10\.csv$", path.name)
    if not m:
        return None
    return int(m.group(1)), str(m.group(2))


def norm_drug(src: str) -> str:
    s = "" if src is None else str(src).strip()
    if s.startswith("drug:"):
        s = s.split("drug:", 1)[1]
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def norm_title(t: str) -> str:
    s = "" if t is None else str(t).strip()
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def load_table(path: Path) -> pd.DataFrame:
    sep = sniff_sep(path)
    return pd.read_csv(path, sep=sep)


def agg_counts(df: pd.DataFrame, weight_col: str):
    d = df.copy()
    d["drug_name"] = d["src"].map(norm_drug)
    d["disease_title"] = d["icd_long_title"].map(norm_title)

    drug_counts = (
        d.groupby("drug_name", dropna=True)[weight_col]
        .sum()
        .sort_values(ascending=False)
    )
    dis_counts = (
        d[d["disease_title"] != ""]
        .groupby("disease_title", dropna=True)[weight_col]
        .sum()
        .sort_values(ascending=False)
    )
    return drug_counts, dis_counts


def topk(series: pd.Series, k: int) -> list[str]:
    return list(series.head(k).index)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def label_def(los: int, diag: str) -> str:
    return f"{los}_dx{diag}"


def heatmap(df_piv: pd.DataFrame, outpath: Path, title: str):
    plt.figure(figsize=(8, max(3.0, 0.35 * len(df_piv))))
    ax = plt.gca()
    im = ax.imshow(df_piv.values, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(df_piv.shape[1]))
    ax.set_xticklabels([str(c) for c in df_piv.columns])
    ax.set_yticks(range(df_piv.shape[0]))
    ax.set_yticklabels(df_piv.index.tolist())

    ax.set_xlabel("Top-K")
    ax.set_title(title)

    for i in range(df_piv.shape[0]):
        for j in range(df_piv.shape[1]):
            v = df_piv.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def diff_table(base_ser: pd.Series, cur_ser: pd.Series, k: int) -> pd.DataFrame:
    base_top = topk(base_ser, k)
    cur_top = topk(cur_ser, k)

    base_rank = {x: i + 1 for i, x in enumerate(base_top)}
    cur_rank = {x: i + 1 for i, x in enumerate(cur_top)}

    base_set, cur_set = set(base_top), set(cur_top)

    rows = []
    for x in sorted(base_set - cur_set):
        rows.append(
            dict(item=x, status="dropped", base_rank=base_rank[x], cur_rank=None,
                 base_weight=float(base_ser.loc[x]), cur_weight=None, rank_delta=None)
        )
    for x in sorted(cur_set - base_set):
        rows.append(
            dict(item=x, status="added", base_rank=None, cur_rank=cur_rank[x],
                 base_weight=None, cur_weight=float(cur_ser.loc[x]), rank_delta=None)
        )
    for x in sorted(base_set & cur_set):
        bd, cd = base_rank[x], cur_rank[x]
        rows.append(
            dict(item=x, status="shared", base_rank=bd, cur_rank=cd,
                 base_weight=float(base_ser.loc[x]), cur_weight=float(cur_ser.loc[x]),
                 rank_delta=cd - bd)
        )

    out = pd.DataFrame(rows)
    order = {"shared": 0, "dropped": 1, "added": 2}
    out["status_order"] = out["status"].map(order)
    out = out.sort_values(["status_order", "status", "base_rank", "cur_rank", "item"]).drop(columns=["status_order"])
    return out


def presence_table(top_sets: dict[str, set], k: int) -> pd.DataFrame:
    counts = {}
    for _, s in top_sets.items():
        for x in s:
            counts[x] = counts.get(x, 0) + 1
    out = pd.DataFrame([dict(item=x, n_defs=n) for x, n in counts.items()])
    out = out.sort_values(["n_defs", "item"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "k", k)
    out.insert(1, "n_total_defs", len(top_sets))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--baseline", default="mimic_48_1_10.csv")
    ap.add_argument("--weight", choices=["unique_patients", "weight_admissions"], default="unique_patients")
    ap.add_argument("--k_list", default="10,20,50,100")
    ap.add_argument("--diff_k", type=int, default=20)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    baseline_path = in_dir / args.baseline
    if not baseline_path.exists():
        raise SystemExit(f"Baseline not found in --in_dir: {baseline_path}")

    out_dir = Path(__file__).resolve().parent / "mimic_stability"
    diffs_dir = out_dir / "diffs"
    out_dir.mkdir(parents=True, exist_ok=True)
    diffs_dir.mkdir(parents=True, exist_ok=True)

    ks = [int(x) for x in args.k_list.split(",") if x.strip()]
    ks = sorted(set([k for k in ks if k > 0]))

    files = []
    for p in sorted(in_dir.glob("mim*_10.csv")):
        info = parse_def(p)
        if info:
            files.append((p, info[0], info[1]))

    if not files:
        raise SystemExit(f"No matching input files found in {in_dir}")

    base_df = load_table(baseline_path)
    base_drug_ser, base_dis_ser = agg_counts(base_df, args.weight)

    results = []
    drug_top_sets_k = {k: {} for k in ks}
    dis_top_sets_k = {k: {} for k in ks}

    for path, los, diag in files:
        df = load_table(path)
        drug_ser, dis_ser = agg_counts(df, args.weight)

        def_id = label_def(los, diag)

        for k in ks:
            bd = set(topk(base_drug_ser, k))
            cd = set(topk(drug_ser, k))
            bdi = set(topk(base_dis_ser, k))
            cdi = set(topk(dis_ser, k))

            results.append(dict(def_id=def_id, los=los, diag=diag, metric="drugs", k=k,
                                jaccard=jaccard(cd, bd), overlap=len(cd & bd)))
            results.append(dict(def_id=def_id, los=los, diag=diag, metric="diseases", k=k,
                                jaccard=jaccard(cdi, bdi), overlap=len(cdi & bdi)))

            drug_top_sets_k[k][def_id] = cd
            dis_top_sets_k[k][def_id] = cdi

        if path.resolve() != baseline_path.resolve():
            dd = diff_table(base_drug_ser, drug_ser, args.diff_k)
            dd.to_csv(diffs_dir / f"drugs_diff_vs_baseline_{def_id}_top{args.diff_k}.csv", index=False)

            di = diff_table(base_dis_ser, dis_ser, args.diff_k)
            di.to_csv(diffs_dir / f"diseases_diff_vs_baseline_{def_id}_top{args.diff_k}.csv", index=False)

    res = pd.DataFrame(results)
    res = res.sort_values(["metric", "los", "diag", "k"]).reset_index(drop=True)
    res.to_csv(out_dir / f"metrics_long_{args.weight}.csv", index=False)

    base_info = parse_def(baseline_path)
    base_id = label_def(base_info[0], base_info[1])

    for metric in ["drugs", "diseases"]:
        sub = res[res["metric"] == metric].copy()
        sub = sub[sub["def_id"] != base_id]
        piv = sub.pivot_table(index="def_id", columns="k", values="jaccard", aggfunc="first")
        def def_sort_key(def_id: str):
            m = re.match(r"(\d+)_dx(all|\d+)$", str(def_id))
            if not m:
                return (999, 999)
            los = int(m.group(1))
            diag = m.group(2)
            diag_key = 99 if diag == "all" else int(diag)  # put "all" last
            return (los, diag_key)

        piv = piv.loc[sorted(piv.index, key=def_sort_key)]
        heatmap(
            piv,
            out_dir / f"heatmap_{metric}_jaccard_vs_baseline_{args.weight}.png",
            f"{metric.capitalize()} stability vs baseline {base_id} (Jaccard vs baseline, weight={args.weight})"
        )

    k0 = args.diff_k
    drug_presence = presence_table(drug_top_sets_k.get(k0, {}), k0)
    dis_presence = presence_table(dis_top_sets_k.get(k0, {}), k0)
    drug_presence.to_csv(out_dir / f"presence_drugs_top{k0}_{args.weight}.csv", index=False)
    dis_presence.to_csv(out_dir / f"presence_diseases_top{k0}_{args.weight}.csv", index=False)

    for metric, top_sets in [("drugs", drug_top_sets_k.get(k0, {})), ("diseases", dis_top_sets_k.get(k0, {}))]:
        if top_sets:
            inter = set.intersection(*top_sets.values()) if top_sets.values() else set()
            with (out_dir / f"core_{metric}_top{k0}_{args.weight}.txt").open("w") as f:
                f.write(f"Definitions: {len(top_sets)}\n")
                f.write(f"Intersection size: {len(inter)}\n")
                for x in sorted(inter):
                    f.write(x + "\n")

    print(f"Wrote outputs to: {out_dir}")
    print(f"Baseline: {baseline_path} (id={base_id})")
    print("Key files:")
    print(f"  {out_dir / f'metrics_long_{args.weight}.csv'}")
    print(f"  {out_dir / f'heatmap_drugs_jaccard_vs_baseline_{args.weight}.png'}")
    print(f"  {out_dir / f'heatmap_diseases_jaccard_vs_baseline_{args.weight}.png'}")
    print(f"  {out_dir / f'presence_drugs_top{k0}_{args.weight}.csv'}")
    print(f"  {out_dir / f'presence_diseases_top{k0}_{args.weight}.csv'}")
    print(f"  diffs/: per-definition added/dropped/shared (top{k0})")


if __name__ == "__main__":
    main()
