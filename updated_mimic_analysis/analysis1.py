#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
INPUT_PATH = APP_DIR.parent / "bigquery_outputs" / "mimic_48_1_10.csv"

OUT_DIR = APP_DIR / "mimic_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_DISEASE_TOP20 = OUT_DIR / "top20_diseases_mimic_48_1_10.csv"
FIG_DISEASE_TOP20 = OUT_DIR / "top20_diseases_mimic_48_1_10.png"

CSV_DRUG_TOP20 = OUT_DIR / "top20_drugs_mimic_48_1_10.csv"
FIG_DRUG_TOP20 = OUT_DIR / "top20_drugs_mimic_48_1_10.png"

CSV_DENSITY = OUT_DIR / "disease_density_mimic_48_1_10.csv"
FIG_DENSITY = OUT_DIR / "disease_density_mimic_48_1_10.png"

MAX_LABEL_LEN = 60
HIGH_DENSITY_THRESHOLD = 5  # diseases with >5 drugs are "high-density"

def truncate_label(text: str, max_len: int = MAX_LABEL_LEN) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

def compute_disease_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dst", "icd_long_title"], as_index=False)
        .agg(
            n_drugs=("src", "nunique"),
            total_admissions=("weight_admissions", "sum"),
        )
    )

def make_disease_top20_plot(agg: pd.DataFrame) -> None:
    top20 = (
        agg.sort_values(
            ["n_drugs", "total_admissions"],
            ascending=[False, False],
        )
        .head(20)
    )
    top20.to_csv(CSV_DISEASE_TOP20, index=False)

    plot_df = top20.sort_values("n_drugs", ascending=True).copy()
    plot_df["icd_label"] = plot_df["icd_long_title"].apply(truncate_label)
    y_pos = range(len(plot_df))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y_pos, plot_df["n_drugs"], color="#c95a5a", alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["icd_label"])
    ax.set_xlabel("Number of distinct drugs")
    ax.set_ylabel("Disease")
    ax.set_title("Top 20 Diseases (MIMIC, 48h, primary dx)")

    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DISEASE_TOP20, dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_drug_top20_plot(df: pd.DataFrame) -> None:
    df = df.copy()
    df["drug"] = df["src"].str.replace(r"^drug:", "", regex=True)

    agg = (
        df.groupby("drug", as_index=False)
        .agg(
            n_diseases=("dst", "nunique"),
            total_admissions=("weight_admissions", "sum"),
        )
    )

    top20 = (
        agg.sort_values(
            ["n_diseases", "total_admissions"],
            ascending=[False, False],
        )
        .head(20)
    )
    top20.to_csv(CSV_DRUG_TOP20, index=False)

    plot_df = top20.sort_values("n_diseases", ascending=True)
    y_pos = range(len(plot_df))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y_pos, plot_df["n_diseases"], alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["drug"])
    ax.set_xlabel("Unique diseases")
    ax.set_ylabel("Drug")
    ax.set_title("Top 20 Drugs by Number of Diseases Treated (MIMIC, 48h, primary dx)")

    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DRUG_TOP20, dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_disease_density_plot(agg: pd.DataFrame) -> None:
    agg.to_csv(CSV_DENSITY, index=False)

    low = agg[agg["n_drugs"] <= HIGH_DENSITY_THRESHOLD]
    high = agg[agg["n_drugs"] > HIGH_DENSITY_THRESHOLD]

    max_deg = agg["n_drugs"].max()
    bins = range(1, max_deg + 2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        low["n_drugs"],
        bins=bins,
        alpha=0.8,
        label="low-density",
        color="#4c72b0",
    )
    ax.hist(
        high["n_drugs"],
        bins=bins,
        alpha=0.8,
        label="high-density",
        color="#c44e52",
    )

    ax.set_xlabel("Number of drugs treating the disease")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of disease co-annotation density (MIMIC, 48h, primary dx)")
    ax.legend(title="category")

    plt.tight_layout()
    fig.savefig(FIG_DENSITY, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    df = pd.read_csv(INPUT_PATH)

    disease_stats = compute_disease_stats(df)
    make_disease_top20_plot(disease_stats)
    make_disease_density_plot(disease_stats)
    make_drug_top20_plot(df)

if __name__ == "__main__":
    main()
