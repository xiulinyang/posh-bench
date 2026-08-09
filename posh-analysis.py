import argparse
import re
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import chisquare, combine_pvalues, false_discovery_control
from statsmodels.stats.multitest import multipletests

N_PER_PHENOMENON = 500
CORRECTION_METHOD = 'holm'  # 'bonferroni' or 'bh' or 'holm'
FDR_ALPHA = 0.05
MODEL='transformer'

if MODEL=='transformer':
    FNAME_RE = re.compile(
        r'results_gpt2(?:_(?P<model_size>mini|small))?_'
        r'(?P<corpus>baby|wiki)_(?P<data_size>\d+)(?P<variant>[Mm])(?P<f>f?)_'
        r'(?P<vocab>\d+)_(?P<seed>\d+)_best\.csv'
    )

    BASELINE_FNAME = 'results_gpt2_best.csv'
    BASELINE_LABEL = 'gpt2'
elif MODEL=='lstm':
    FNAME_RE = re.compile(
        r'results_LSTM(?:_(?P<model_size>medium|small|large))?_'
        r'(?P<corpus>baby|wiki)_(?P<data_size>\d+)(?P<variant>[Mm])(?P<f>f?)_'
        r'(?P<vocab>\d+)_(?P<seed>\d+)_best\.csv'
    )

    BASELINE_FNAME = 'results_LSTM_best.csv'
    BASELINE_LABEL = 'LSTM'

elif MODEL=='ngram':
    FNAME_RE = re.compile(
        r'results_(?P<corpus>baby|wiki)_(?P<data_size>\d+)(?P<variant>[Mm])(?P<f>f?)_'
        r'best\.csv'
    )

    BASELINE_FNAME = 'results_best.csv'
    BASELINE_LABEL = 'ngram'
else:
    raise ValueError(f"Unknown MODEL: {MODEL}")


CORPUS_MACRO = {
    'baby-f': r'\babyfiltered',
    'baby': r'\baby',
    'wiki': r'\textsc{wiki}',
}
CORPUS_ORDER = ['baby-f', 'baby', 'wiki']

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


def get_direction(accs, chance=0.5):
    directions = np.sign(np.array(accs) - chance)

    if np.all(directions >= 0):
        return 'positive'
    elif np.all(directions < 0):
        return 'negative'
    else:
        return 'mixed'




def normalize_phenomenon(name):
    return name.replace('-', '_')

def escape_latex(name):
    return normalize_phenomenon(name).replace('_', r'\_')

def parse_filename(fname):
    base = os.path.basename(fname)
    if base == BASELINE_FNAME:
        return BASELINE_LABEL, '0'
    m = FNAME_RE.search(base)
    if not m:
        return None
    d = m.groupdict()
    if MODEL=='ngram':
        d['seed']=0
    corpus = d['corpus'] + ('-f' if d['f'] == 'f' else '')
    size = int(d['data_size'])
    return (corpus, size), d['seed']


def load_results(folder):
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

def significance_stars(p_adj):
    if p_adj < 0.001:
        return '***'
    elif p_adj < 0.01:
        return '**'
    elif p_adj < 0.05:
        return '*'
    return ''


def per_seed_fisher_p(seed_correct_incorrect):
    p_values = []
    for correct, incorrect in seed_correct_incorrect:
        total = correct + incorrect
        expected = [total * 0.5, total * 0.5]
        _, p = chisquare([correct, incorrect], f_exp=expected)
        p_values.append(p)
    if len(p_values) == 1:
        return p_values[0]
    _, combined_p = combine_pvalues(p_values, method='fisher')
    return combined_p

def apply_correction_within_family(raw_p_by_key, family_of, method=CORRECTION_METHOD):
    families = defaultdict(list)
    for key in raw_p_by_key:
        families[family_of[key]].append(key)

    adjusted = {}
    for family_id, keys in families.items():
        ps = np.array([raw_p_by_key[k] for k in keys])
        if method == 'bonferroni':
            m = len(ps)
            adj = np.minimum(ps * m, 1.0)
        elif method == 'bh':
            adj = false_discovery_control(ps, method='bh')
        elif method == 'holm':
            _, adj, _, _ = multipletests(
                ps,
                alpha=FDR_ALPHA,
                method='holm'
            )
        else:
            raise ValueError(f"Unknown correction method: {method}")
        for k, a in zip(keys, adj):
            adjusted[k] = a
    return adjusted


def phenomenon_raw_stats(seed_to_acc, n_per_seed=N_PER_PHENOMENON):
    accs = list(seed_to_acc.values())
    ci_pairs = []
    for acc in accs:
        correct = round(acc * n_per_seed)
        ci_pairs.append((correct, n_per_seed - correct))
    mean = np.mean(accs) * 100
    raw_p = per_seed_fisher_p(ci_pairs)

    direction = get_direction(accs)
    if raw_p < 0.05 and direction =='mixed':
        raw_p = 1.0
    std = np.std(accs, ddof=1) * 100 if len(accs) > 1 else None
    return mean, std, raw_p


def category_raw_stats(col_data, phenomena, n_per_phenomenon=N_PER_PHENOMENON):
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
    raw_p = per_seed_fisher_p(ci_pairs)
    direction = get_direction(seed_accs)
    if raw_p<0.05 and direction == 'mixed':
        raw_p = 1.0
    std = np.std(seed_accs, ddof=1) * 100 if len(seed_accs) > 1 else None
    return mean, std, raw_p


def category_mean_accs(col_data, phenomena, n_per_phenomenon=N_PER_PHENOMENON):
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


def make_cell(mean, std, q):
    stars = significance_stars(q) if q is not None else ''
    cell = f"\\scorecell{{{mean:.1f}}}{{{stars}}}"
    if std is not None:
        cell += f"\\sd{{{std:.1f}}}"
    return cell


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


def build_finegrained_table(data):
    columns, size_groups = build_columns(data)
    has_baseline = BASELINE_LABEL in data
    n_cols = len(columns)

    phenomena_by_cat = defaultdict(list)
    for phe, (cat, order) in CATEGORY_MAP.items():
        phenomena_by_cat[cat].append((order, phe))
    for cat in phenomena_by_cat:
        phenomena_by_cat[cat].sort()

    raw_p_by_key = {}
    stats_by_key = {}
    family_of = {}
    for cat in CATEGORY_ORDER:
        phes = [p for _, p in phenomena_by_cat.get(cat, [])]
        for phe in phes:
            for col in columns:
                seed_to_acc = data.get(col, {}).get(phe, {})
                if not seed_to_acc:
                    continue
                mean, std, raw_p = phenomenon_raw_stats(seed_to_acc)
                key = (phe, col)
                stats_by_key[key] = (mean, std)
                raw_p_by_key[key] = raw_p
                family_of[key] = col  # family = training-data column

    # --- Pass 2: BH correction within each family (column) ---
    adjusted_q_by_key = apply_correction_within_family(raw_p_by_key, family_of)

    # --- Pass 3: build the table using corrected p/q-values for stars ---
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
                key = (phe, col)
                if key not in stats_by_key:
                    row.append("--")
                    continue
                mean, std = stats_by_key[key]
                q = adjusted_q_by_key[key]
                row.append(make_cell(mean, std, q))
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")

    all_phes = list(CATEGORY_MAP.keys())
    avg_row = ["Average", " "]
    for col in columns:
        col_data = data.get(col, {})
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
                 r"standard deviation over seeds). For each cell, a "
                 r"$\chi^2$ goodness-of-fit test against chance (50\%) is "
                 r"run per seed and the resulting $p$-values are combined "
                 r"via Fisher's method; the combined $p$-values are then "
                 r"corrected for multiple comparisons using the "
                 r"Bonferroni procedure, applied separately within "
                 r"each training-data condition (column), and stars are "
                 r"assigned based on the resulting adjusted $p$-values. "
                 r"({*}~$p < 0.05$, {**}~$p < 0.01$, {***}~$p < 0.001$, "
                 r"Bonferroni-corrected).}")
    lines.append(r"\label{tab:finegrained}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# --- category-level table -------------------------------------------------

def build_category_table(data):
    columns, size_groups = build_columns(data)
    has_baseline = BASELINE_LABEL in data
    n_cols = len(columns)

    # --- Pass 1: raw stats + raw p per (category, column) ---
    raw_p_by_key = {}
    stats_by_key = {}
    family_of = {}
    for cat in CATEGORY_ORDER:
        phenomena = CATEGORY_PHENOMENA.get(cat, [])
        if not phenomena:
            continue
        for col in columns:
            col_data = data.get(col, {})
            result = category_raw_stats(col_data, phenomena)
            if result is None:
                continue
            mean, std, raw_p = result
            key = (cat, col)
            stats_by_key[key] = (mean, std)
            raw_p_by_key[key] = raw_p
            family_of[key] = col  # family = training-data column

    # --- Pass 2: BH correction within each family (column) ---
    adjusted_q_by_key = apply_correction_within_family(raw_p_by_key, family_of)

    # --- Pass 3: build table ---
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
            key = (cat, col)
            if key not in stats_by_key:
                row.append("--")
                continue
            mean, std = stats_by_key[key]
            q = adjusted_q_by_key[key]
            row.append(make_cell(mean, std, q))
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
                 r"1{,}000 pairs). For each cell, a $\chi^2$ "
                 r"goodness-of-fit test against chance (50\%) is run per "
                 r"seed (seeds are never pooled, to avoid treating "
                 r"repeated-measures observations as independent trials) "
                 r"and the resulting $p$-values are combined via Fisher's "
                 r"method; the combined $p$-values are then corrected for "
                 r"multiple comparisons using the Bonferroni "
                 r"procedure, applied separately within each training-data "
                 r"condition (column), and stars are assigned based on the "
                 r"resulting adjusted $p$-values. "
                 r"({*}~$p < 0.05$, {**}~$p < 0.01$, {***}~$p < 0.001$, "
                 r"Bonferroni-corrected).}")
    lines.append(r"\label{tab:category}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='path to posh_results folder')

    args = parser.parse_args()

    data = load_results(args.folder)
    if not data:
        raise SystemExit("No matching CSVs found — check the folder path "
                          "and filename pattern.")

    finegrained = build_finegrained_table(data)
    category = build_category_table(data)
    out = f'posh_tables_{MODEL}.tex'
    with open(out, 'w') as f:
        f.write(finegrained)
        f.write("\n\n")
        f.write(category)
        f.write("\n")

    print(f"\nWrote {out}\n")
    print(finegrained)
    print()
    print(category)

    for p in [0.001, 0.005, 0.01, 0.02, 0.03, 0.04]:
        holm = multipletests(
            [p] * 8,
            alpha=0.05,
            method='holm'
        )[1][0]

        bh = false_discovery_control(
            np.array([p] * 8),
            method='bh'
        )[0]

        print(f"{p:.3f}  Holm={holm:.4f}  BH={bh:.4f}")