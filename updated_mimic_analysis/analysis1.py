#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

# Input: BigQuery output
INPUT_PATH = APP_DIR.parent / "bigquery_outputs" / "mimic_48_1_10.csv"

# Outputs
OUT_DIR = APP_DIR / "mimic_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_DISEASE = OUT_DIR / "top20_diseases_mimic_48_1_10.csv"
FIG_DISEASE = OUT_DIR / "top20_diseases_mimic_48_1_10.png"

CSV_DRUG = OUT_DIR / "top20_drugs_mimic_48_1_10.csv"
FIG_DRUG = OUT_DIR / "top20_drugs_mimic_48_1_10.png"

MAX_LABEL_LEN = 60  # for long disease names

def truncate_label(text: str, max_len: int = MAX_LABEL_LEN) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

def make_disease_plot(df: pd.DataFrame) -> None:
    agg = (
        df.groupby(["dst", "icd_long_title"], as_index=False)
        .agg(
            n_drugs=("src", "nunique"),
            total_admissions=("weight_admissions", "sum"),
        )
    )

    top20 = (
        agg.sort_values(
            ["n_drugs", "total_admissions"],
            ascending=[False, False],
        )
        .head(20)
    )

    # Save table
    top20.to_csv(CSV_DISEASE, index=False)

    # Plot
    top20_plot = top20.sort_values("n_drugs", ascending=True).copy()
    top20_plot["icd_label"] = top20_plot["icd_long_title"].apply(truncate_label)

    y_pos = range(len(top20_plot))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y_pos, top20_plot["n_drugs"], color="#c95a5a", alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20_plot["icd_label"])

    ax.set_xlabel("Number of distinct drugs", fontsize=11)
    ax.set_ylabel("Disease", fontsize=11)
    ax.set_title("Top 20 Diseases (MIMIC, 48h, primary dx)", fontsize=13)

    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DISEASE, dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_drug_plot(df: pd.DataFrame) -> None:
    # Clean drug names: strip "drug:" prefix
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

    # Save table
    top20.to_csv(CSV_DRUG, index=False)

    # Plot
    top20_plot = top20.sort_values("n_diseases", ascending=True)

    y_pos = range(len(top20_plot))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y_pos, top20_plot["n_diseases"], alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20_plot["drug"])

    ax.set_xlabel("Unique diseases", fontsize=11)
    ax.set_ylabel("Drug", fontsize=11)
    ax.set_title("Top 20 Drugs by Number of Diseases Treated (MIMIC, 48h, primary dx)", fontsize=13)

    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DRUG, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    df = pd.read_csv(INPUT_PATH)

    make_disease_plot(df)
    make_drug_plot(df)

    print(f"Saved disease table to {CSV_DISEASE}")
    print(f"Saved disease figure to {FIG_DISEASE}")
    print(f"Saved drug table to {CSV_DRUG}")
    print(f"Saved drug figure to {FIG_DRUG}")

if __name__ == "__main__":
    main()
