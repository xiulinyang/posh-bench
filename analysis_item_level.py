import pandas as pd
from glob import glob

category_dataset_map = {
    "Island": {
        "blimp": ["adjunct_island", "wh_island", "complex_NP_island"],
        "scamp": ["complex_np_island", "wh_island", "adjunct_island"],
        "zorro": ["island-effects-adjunct_island"],
        "posh": ["island-adjunct", "island-complex-np", "island-subject", "island-wh"]
    },
    "Question Formation": {
        "posh": ["question-formation_or", "question-formation_rr", "question-formation_sr"]
    },
    "Wanna": {
        "posh": ["wanna"]
    },
    "Binding": {
        "blimp": [
            "principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
            "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3"
        ],
        "zorro": ["binding-principle_a"],
        "scamp": [
            "principle_A_domain_1", "principle_A_domain_2",
            "principle_A_domain_3", "principle_A_c_command"
        ],
        "posh": ["principle_a_command", "principle_a_locality"]
    }
}

reverse_map = {
    dataset: category
    for category, sources in category_dataset_map.items()
    for source_list in sources.values()
    for dataset in source_list
}


def collect_benchmark_results(result_path):
    print(result_path)
    model_name = '_'.join(result_path.split('/')[-1].split('.')[0].split('_')[:6])
    phenomenon = '_'.join(result_path.split('/')[-1].split('.')[0].split('_')[6:])
    data_source = model_name.split('_')[2]
    filter = 'yes' if 'Mf' in model_name else 'no'
    data_size = model_name.split('_')[3][:-1] if model_name.split('_')[3][-1] == 'M' else model_name.split('_')[3][:-2]
    benchmark = result_path.split('/')[-2].split('_')[0]
    random_seed = model_name.split('_')[-1]
    category = reverse_map[phenomenon]
    df = pd.read_csv(result_path, sep='\t').to_dict(orient='records')

    sent1 = [x['Sentence2'] for x in df]
    correct = [x['Correct'] for x in df]
    return [{'model_name': model_name,
             'phenomenon': phenomenon,
             'category': category,
             'data_source': data_source,
             'data_size': data_size,
             'filter': filter,
             'benchmark': benchmark,
             'random_seed': random_seed,
             'sentence': s,
             'correct': c} for s, c in zip(sent1, correct)]


# zorro_result_foler = glob('zorro_item_results/*.tsv')
# blimp_results = glob('blimp_item_results/*.tsv')
# scamp_results = glob('scamp_item_results/*.tsv')
#
posh_results = glob('posh_item_results/*.tsv')

all_results = []

for result_file in posh_results:
    all_results.extend(collect_benchmark_results(result_file))


df_all = pd.DataFrame(all_results)
df_all.to_csv('stats/posh_item_level_results.csv', index=False)



