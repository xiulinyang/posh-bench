

"""
Aggregate per-model benchmark result CSVs into two summary tables:
  1. subcategory_results.csv  — one row per (model, subcategory)
  2. category_results.csv     — one row per (model, main category), averaged
                                 over that category's subcategories

Expected layout:
    <base_dir>/
        blimp_results/results_gpt2_..._best.csv
        scamp_results/results_gpt2_..._best.csv
        zorro_results/results_gpt2_..._best.csv
        posh_results/results_gpt2_..._best.csv

Each results CSV is expected to look like:
    ,best
    island-adjunct,0.772
    island-complex-np,0.382
    ...

Filenames are expected to follow:
    results_gpt2_<arch>_<domain>_<train_size>_<vocab>_<seed>_best.csv
    e.g. results_gpt2_mini_baby_10Mf_32768_42_best.csv

A bare `results_gpt2_best.csv` (no arch/domain/size/seed) is treated as the
baseline GPT-2 checkpoint and given domain/train_size/seed = "NA".
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Category -> benchmark suite -> subcategory names, as provided.
# ---------------------------------------------------------------------------
category_dataset_map = {
    "Island": {
        "blimp": [
            "adjunct_island",
             "wh_island", "complex_NP_island",
        ],
        "scamp_plausible": [
            "complex_np_island", "wh_island", "adjunct_island",
        ],
        "zorro": [
            "island-effects-adjunct_island",
        ],
        "posh": ["island-adjunct", "island-complex-np", "island-subject", "island-wh"]
    },

    "Question Formation": {"posh": ["question-formation_or", "question-formation_rr", "question-formation_sr"]},
    "Wanna": {"posh": ["wanna"]},
    "Binding": {
        "blimp": [
            "principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
            "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3",
        ],
        "zorro": ["binding-principle_a"],
        "scamp_plausible": ["principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3", "principle_A_c_command"],
        "posh": ["principle_a_command", "principle_a_locality"]
    }
}

# Flatten to: {suite: {subcategory_name: category_name}}
SUBCAT_TO_CATEGORY = {}
for _category, _suites in category_dataset_map.items():
    for _suite, _subcats in _suites.items():
        SUBCAT_TO_CATEGORY.setdefault(_suite, {})
        for _sub in _subcats:
            SUBCAT_TO_CATEGORY[_suite][_sub] = _category

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
FILENAME_RE = re.compile(
    r'^results_gpt2_(?P<arch>[A-Za-z0-9]+)_(?P<domain>baby|wiki)_'
    r'(?P<train_size>\d+Mf?)_(?P<vocab>\d+)_(?P<seed>\d+)_best\.csv$'
)
BASELINE_RE = re.compile(r'^results_gpt2_best\.csv$')


def parse_filename(fname: str):
    """Return a dict of metadata for a results filename, or None if unrecognized."""
    m = FILENAME_RE.match(fname)
    if m:
        d = m.groupdict()
        return {
            "arch": d["arch"],
            "domain": d["domain"],
            "train_size": d["train_size"],
            "vocab": d["vocab"],
            "seed": d["seed"],
        }
    if BASELINE_RE.match(fname):
        return {
            "arch": "gpt2",
            "domain": "NA",
            "train_size": "NA",
            "vocab": "NA",
            "seed": "NA",
        }
    return None


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
def collect_results(base_dir, suites=("blimp", "scamp_plausible", "zorro", "posh")) -> pd.DataFrame:
    rows = []
    base_dir = Path(base_dir)

    for suite in suites:
        suite_dir = base_dir / f"{suite}_results"
        if not suite_dir.is_dir():
            continue

        subcat_map = SUBCAT_TO_CATEGORY.get(suite, {})
        if not subcat_map:
            continue

        for csv_path in sorted(suite_dir.glob("*.csv")):
            meta = parse_filename(csv_path.name)
            if meta is None:
                print(f"  [skip] unrecognized filename: {csv_path.name}")
                continue

            df = pd.read_csv(csv_path, index_col=0)
            if "best" not in df.columns:
                print(f"  [skip] {csv_path.name}: no 'best' column found")
                continue

            for subcat, category in subcat_map.items():
                if subcat not in df.index:
                    continue
                acc = df.loc[subcat, "best"]
                rows.append({
                    "benchmark_suite": suite,
                    "category": category,
                    "subcategory": subcat,
                    "arch": meta["arch"],
                    "domain": meta["domain"],
                    "train_size": meta["train_size"],
                    "seed": meta["seed"],
                    "accuracy": acc,
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark result CSVs into category-level and subcategory-level summaries."
    )
    parser.add_argument("base_dir", help="Directory containing <suite>_results/ subfolders")
    parser.add_argument("--subcat_out", default="subcategory_results.csv",
                         help="Output path for subcategory-level CSV")
    parser.add_argument("--category_out", default="category_results.csv",
                         help="Output path for main-category-level CSV")
    args = parser.parse_args()

    detailed = collect_results(args.base_dir)
    if detailed.empty:
        print("No matching results found — check base_dir and that "
              "<suite>_results/ subfolders exist.")
        return

    detailed.to_csv(args.subcat_out, index=False)
    print(f"Saved subcategory-level results to {args.subcat_out} ({len(detailed)} rows)")

    group_cols = ["benchmark_suite", "category", "arch", "domain", "train_size", "seed"]
    main_cat = (
        detailed.groupby(group_cols, as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "avg_accuracy"})
    )
    main_cat.to_csv(args.category_out, index=False)
    print(f"Saved main-category results to {args.category_out} ({len(main_cat)} rows)")


if __name__ == "__main__":
    main()