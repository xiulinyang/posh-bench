"""
Build BOTH the fine-grained (per-phenomenon) and category-level PoSH LaTeX
tables from posh_results/*.csv, and write them into a single .tex file.

Usage:
    python generate_posh_tables.py /path/to/posh_results --out posh_tables.tex

Significance methodology (both tables):
  - A chi-squared goodness-of-fit test against chance (50%) is run
    separately for EACH seed, and the MAX p-value across seeds is used to
    assign stars. Seeds are never pooled into one N, since that would
    treat repeated-measures observations on the same items across seeds
    as independent trials (pseudoreplication).
  - In the category-level table, a category's N per seed is the sum of
    its subcategories' pair counts (e.g. Binding: 2 * 500 = 1000). Pooling
    subcategories within a seed is valid (different items, not repeated
    measures) — only pooling across seeds is the problem.

Assumes each CSV was produced by the eval script's:
    pd.DataFrame({'best': {phenomenon: acc, ...}}).to_csv(...)
i.e. index = phenomenon, single column named 'best'.

Assumes \\scorecell{mean}{stars}, \\sd{std}, \\babyfiltered, and \\baby are
already defined in your LaTeX preamble.
"""

import argparse
import re
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import chisquare

# --- config ---------------------------------------------------------------

N_PER_PHENOMENON = 500  # sentence pairs per phenomenon, per single eval run

FNAME_RE = re.compile(
    r'results_gpt2(?:_(?P<model_size>mini|small))?_'
    r'(?P<corpus>baby|wiki)_(?P<data_size>\d+)(?P<variant>[Mm])(?P<f>f?)_'
    r'(?P<vocab>\d+)_(?P<seed>\d+)_best\.csv'
)

BASELINE_FNAME = 'results_gpt2_best.csv'
BASELINE_LABEL = 'gpt2'

CORPUS_MACRO = {
    'baby-f': r'\babyfiltered',
    'baby': r'\baby',
    'wiki': r'\textsc{wiki}',
}
CORPUS_ORDER = ['baby-f', 'baby', 'wiki']

# phenomenon (normalized, underscores) -> (category, display order within category)
# EDIT HERE if you add new phenomena.
CATEGORY_MAP = {
    'island_adjunct':        ('Island', 0),
    'island_complex_np':     ('Island', 1),
    'island_wh':               ('Island', 2),
    'question_formation_or':   ('Question Formation', 0),
    'question_formation_rr':   ('Question Formation', 1),
    'question_formation_sr':   ('Question Formation', 2),
    'principle_a_command':     ('Binding', 0),
    'principle_a_locality':    ('Binding', 1),
    'wanna':                     ('Wanna', 0),
}
CATEGORY_ORDER = ['Island', 'Question Formation', 'Binding', 'Wanna']

CATEGORY_PHENOMENA = defaultdict(list)
for _phe, (_cat, _order) in sorted(CATEGORY_MAP.items(), key=lambda kv: kv[1][1]):
    CATEGORY_PHENOMENA[_cat].append(_phe)


def normalize_phenomenon(name):
    return name.replace('-', '_')


def escape_latex(name):
    return normalize_phenomenon(name).replace('_', r'\_')


# --- parsing ----------------------------------------------------------

def parse_filename(fname):
    base = os.path.basename(fname)
    if base == BASELINE_FNAME:
        return BASELINE_LABEL, '0'
    m = FNAME_RE.search(base)
    if not m:
        return None
    d = m.groupdict()
    corpus = d['corpus'] + ('-f' if d['f'] == 'f' else '')
    size = int(d['data_size'])
    return (corpus, size), d['seed']


def load_results(folder):
    """condition_key -> phenomenon -> {seed: acc}"""
    data = defaultdict(lambda: defaultdict(dict))
    skipped = []
    for path in glob.glob(os.path.join(folder, '*.csv')):
        parsed = parse_filename(path)
        if parsed is None:
            skipped.append(path)
            continue
        condition, seed = parsed
        df = pd.read_csv(path, index_col=0)
        col = df.columns[0]  # 'best'
        for phenomenon, acc in df[col].items():
            data[condition][normalize_phenomenon(phenomenon)][seed] = acc
    if skipped:
        print(f"[note] skipped {len(skipped)} file(s) that didn't match the "
              f"naming pattern:")
        for s in skipped:
            print("   ", os.path.basename(s))
    return data


# --- stats --------------------------------------------------------------

def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


def per_seed_max_p(seed_correct_incorrect):
    """seed_correct_incorrect: list of (correct, incorrect) pairs, one per
    seed. Returns the max p-value across seeds' individual chi-squared
    tests (never pools seeds together)."""
    p_values = []
    for correct, incorrect in seed_correct_incorrect:
        total = correct + incorrect
        expected = [total * 0.5, total * 0.5]
        _, p = chisquare([correct, incorrect], f_exp=expected)
        p_values.append(p)
    return max(p_values)


def phenomenon_scorecell(seed_to_acc, n_per_seed=N_PER_PHENOMENON):
    """seed_to_acc: {seed: acc} for a single phenomenon."""
    accs = list(seed_to_acc.values())
    ci_pairs = []
    for acc in accs:
        correct = round(acc * n_per_seed)
        ci_pairs.append((correct, n_per_seed - correct))
    mean = np.mean(accs) * 100
    stars = significance_stars(per_seed_max_p(ci_pairs))
    cell = f"\\scorecell{{{mean:.1f}}}{{{stars}}}"
    if len(accs) > 1:
        std = np.std(accs, ddof=1) * 100
        cell += f"\\sd{{{std:.1f}}}"
    return cell


def category_scorecell(col_data, phenomena, n_per_phenomenon=N_PER_PHENOMENON):
    """col_data: phenomenon -> {seed: acc}. phenomena: subcategories that
    make up this category."""
    seed_sets = [set(col_data.get(p, {})) for p in phenomena]
    seeds = set.intersection(*seed_sets) if seed_sets and all(seed_sets) else set()
    if not seeds:
        return None

    n_per_seed_total = n_per_phenomenon * len(phenomena)
    seed_accs = []
    ci_pairs = []
    for seed in seeds:
        correct = sum(round(col_data[p][seed] * n_per_phenomenon) for p in phenomena)
        seed_accs.append(correct / n_per_seed_total)
        ci_pairs.append((correct, n_per_seed_total - correct))

    mean = np.mean(seed_accs) * 100
    stars = significance_stars(per_seed_max_p(ci_pairs))
    cell = f"\\scorecell{{{mean:.1f}}}{{{stars}}}"
    if len(seed_accs) > 1:
        std = np.std(seed_accs, ddof=1) * 100
        cell += f"\\sd{{{std:.1f}}}"
    return cell


def category_mean_accs(col_data, phenomena, n_per_phenomenon=N_PER_PHENOMENON):
    """Return list of per-seed pooled accuracies for a category (no stars/
    formatting) — used when building the Average row."""
    seed_sets = [set(col_data.get(p, {})) for p in phenomena]
    seeds = set.intersection(*seed_sets) if seed_sets and all(seed_sets) else set()
    if not seeds:
        return []
    n_per_seed_total = n_per_phenomenon * len(phenomena)
    seed_accs = []
    for seed in seeds:
        correct = sum(round(col_data[p][seed] * n_per_phenomenon) for p in phenomena)
        seed_accs.append(correct / n_per_seed_total)
    return seed_accs


# --- shared column layout -------------------------------------------------

def build_columns(data):
    sizes = sorted({k[1] for k in data if k != BASELINE_LABEL})
    columns = []
    size_groups = []
    for size in sizes:
        present = [c for c in CORPUS_ORDER if (c, size) in data]
        if present:
            size_groups.append((size, present))
            columns.extend((c, size) for c in present)
    if BASELINE_LABEL in data:
        columns.append(BASELINE_LABEL)
    return columns, size_groups


def header_lines(columns, size_groups, has_baseline, extra_leading_cols):
    """extra_leading_cols: number of non-data label columns before the data
    columns start (2 for Category+Phenomenon, 1 for Category-only)."""
    header1 = [" "] * extra_leading_cols
    col_pos = extra_leading_cols + 1
    cmidrules = []
    for size, present in size_groups:
        k = len(present)
        header1.append(r"\multicolumn{%d}{c}{\textbf{%dM}}" % (k, size))
        cmidrules.append(r"\cmidrule(lr){%d-%d}" % (col_pos, col_pos + k - 1))
        col_pos += k
    if has_baseline:
        header1.append(" ")
    row1 = " & ".join(header1) + r" \\"
    row2_cmid = "".join(cmidrules)
    return row1, row2_cmid


# --- fine-grained table ---------------------------------------------------

def build_finegrained_table(data):
    columns, size_groups = build_columns(data)
    has_baseline = BASELINE_LABEL in data
    n_cols = len(columns)

    phenomena_by_cat = defaultdict(list)
    for phe, (cat, order) in CATEGORY_MAP.items():
        phenomena_by_cat[cat].append((order, phe))
    for cat in phenomena_by_cat:
        phenomena_by_cat[cat].sort()

    lines = []
    lines.append(r"\begin{table*}[tbp]")
    lines.append(r"\centering")
    lines.append(r"\begin{adjustbox}{max width=1\textwidth}")
    lines.append(r"\begin{tabular}{ll*{%d}{l}}" % n_cols)
    lines.append(r"\toprule")

    row1, cmid = header_lines(columns, size_groups, has_baseline, extra_leading_cols=2)
    lines.append(row1)
    lines.append(cmid)

    header2 = [r"\textbf{Category}", r"\textbf{Phenomenon}"]
    for size, present in size_groups:
        header2.extend(CORPUS_MACRO[c] for c in present)
    if has_baseline:
        header2.append(r"\textbf{GPT2}")
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for cat in CATEGORY_ORDER:
        phes = [p for _, p in phenomena_by_cat.get(cat, [])]
        if not phes:
            continue
        for i, phe in enumerate(phes):
            row = []
            if i == 0:
                row.append(r"\multirow{%d}{*}{%s}" % (len(phes), cat))
            else:
                row.append(" ")
            row.append(escape_latex(phe))
            for col in columns:
                seed_to_acc = data.get(col, {}).get(phe, {})
                if not seed_to_acc:
                    row.append("--")
                    continue
                row.append(phenomenon_scorecell(seed_to_acc))
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")

    all_phes = list(CATEGORY_MAP.keys())
    avg_row = ["Average", " "]
    for col in columns:
        col_data = data.get(col, {})
        n_seeds = max((len(v) for v in col_data.values()), default=0)
        # align by seed id, not position
        all_seed_ids = set()
        for phe in all_phes:
            all_seed_ids |= set(col_data.get(phe, {}))
        per_seed_means = []
        for seed in all_seed_ids:
            vals = [col_data[phe][seed] for phe in all_phes
                    if phe in col_data and seed in col_data[phe]]
            if vals:
                per_seed_means.append(np.mean(vals))
        if not per_seed_means:
            avg_row.append("--")
            continue
        mean = np.mean(per_seed_means) * 100
        cell = f"\\scorecell{{{mean:.1f}}}{{}}"
        if len(per_seed_means) > 1:
            std = np.std(per_seed_means, ddof=1) * 100
            cell += f"\\sd{{{std:.1f}}}"
        avg_row.append(cell)
    lines.append(" & ".join(avg_row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\caption{Fine-grained PoSH results grouped by training "
                 r"size (columns) and phenomenon (rows). The last row "
                 r"reports the average across all PoSH phenomena (mean and "
                 r"standard deviation over seeds). Significance is "
                 r"assessed per seed against chance (50\%) with a $\chi^2$ "
                 r"goodness-of-fit test; a cell is starred only if every "
                 r"seed individually clears that threshold (seeds are not "
                 r"pooled). ({*}~$p < 0.05$, {**}~$p < 0.01$, "
                 r"{***}~$p < 0.001$).}")
    lines.append(r"\label{tab:finegrained}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# --- category-level table -------------------------------------------------

def build_category_table(data):
    columns, size_groups = build_columns(data)
    has_baseline = BASELINE_LABEL in data
    n_cols = len(columns)

    lines = []
    lines.append(r"\begin{table*}[tbp]")
    lines.append(r"\centering")
    lines.append(r"\begin{adjustbox}{max width=1\textwidth}")
    lines.append(r"\begin{tabular}{l*{%d}{l}}" % n_cols)
    lines.append(r"\toprule")

    row1, cmid = header_lines(columns, size_groups, has_baseline, extra_leading_cols=1)
    lines.append(row1)
    lines.append(cmid)

    header2 = [r"\textbf{Category}"]
    for size, present in size_groups:
        header2.extend(CORPUS_MACRO[c] for c in present)
    if has_baseline:
        header2.append(r"\textbf{GPT2}")
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for cat in CATEGORY_ORDER:
        phenomena = CATEGORY_PHENOMENA.get(cat, [])
        if not phenomena:
            continue
        row = [cat]
        for col in columns:
            col_data = data.get(col, {})
            cell = category_scorecell(col_data, phenomena)
            row.append(cell if cell is not None else "--")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\midrule")

    avg_row = ["Average"]
    for col in columns:
        col_data = data.get(col, {})
        cat_means = []
        for cat in CATEGORY_ORDER:
            phenomena = CATEGORY_PHENOMENA.get(cat, [])
            seed_accs = category_mean_accs(col_data, phenomena)
            if seed_accs:
                cat_means.append(np.mean(seed_accs))
        if not cat_means:
            avg_row.append("--")
            continue
        mean = np.mean(cat_means) * 100
        cell = f"\\scorecell{{{mean:.1f}}}{{}}"
        if len(cat_means) > 1:
            std = np.std(cat_means, ddof=1) * 100
            cell += f"\\sd{{{std:.1f}}}"
        avg_row.append(cell)
    lines.append(" & ".join(avg_row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\caption{Category-level PoSH results grouped by "
                 r"training size (columns). Each category's accuracy pools "
                 r"all of its subcategories' pairs within a seed (e.g. "
                 r"Binding: 2 subcategories $\times$ 500 pairs = "
                 r"1{,}000 pairs). Significance is assessed per seed "
                 r"against chance (50\%) with a $\chi^2$ goodness-of-fit "
                 r"test; a cell is starred only if every seed individually "
                 r"clears that threshold (seeds are not pooled, to avoid "
                 r"treating repeated-measures observations as independent "
                 r"trials). ({*}~$p < 0.05$, {**}~$p < 0.01$, "
                 r"{***}~$p < 0.001$).}")
    lines.append(r"\label{tab:category}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# --- main -----------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='path to posh_results folder')
    parser.add_argument('--out', default='posh_tables.tex')
    args = parser.parse_args()

    data = load_results(args.folder)
    if not data:
        raise SystemExit("No matching CSVs found — check the folder path "
                          "and filename pattern.")

    finegrained = build_finegrained_table(data)
    category = build_category_table(data)

    with open(args.out, 'w') as f:
        f.write(finegrained)
        f.write("\n\n")
        f.write(category)
        f.write("\n")

    print(f"\nWrote {args.out}\n")
    print(finegrained)
    print()
    print(category)