from minicons import scorer
import argparse
from huggingface_hub import list_repo_refs
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config
from glob import glob
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os


def read_data(data_path, dataset_name):
    test_set = {}

    if dataset_name in ['zorro']:
        phenomenon_paths = glob(f'{data_path}/*.txt')
        for p in tqdm(phenomenon_paths):
            phenomenon = p.split('/')[1].split('.')[0]
            if phenomenon in ['island-effects-adjunct_island', 'binding-principle_a']:
                sentences = Path(p).read_text().strip().split('\n')
                sent_pair = [(sentences[i], sentences[i+1])for i in range(len(sentences)) if i%2==0]
                test_set[phenomenon] = sent_pair
    elif dataset_name in ['posh']:
        phenomenon_paths = glob(f'{data_path}/*.txt')
        for p in tqdm(phenomenon_paths):
            phenomenon = p.split('/')[1].split('.')[0]
            sentences = Path(p).read_text().strip().split('\n')
            sent_pair = [(sentences[i], sentences[i+1])for i in range(len(sentences)) if i%2==0]
            test_set[phenomenon] = sent_pair
    elif dataset_name in ['blimp']:
        phenomenon_paths = glob(f'{data_path}/*.jsonl')
        for p in tqdm(phenomenon_paths):
            phenomenon_n = p.split('/')[1].split('.')[0]
            if phenomenon_n in ["principle_A_c_command", "principle_A_case_2", "principle_A_domain_2",
                                "principle_A_domain_3",
                                "adjunct_island", "wh_island", "complex_NP_island"]:
                phenomenon = pd.read_json(p, lines=True).to_dict(orient='records')
                sent_pair = [(x['sentence_bad'], x['sentence_good']) for x in phenomenon]
                test_set[phenomenon_n] = sent_pair
    elif dataset_name in ['scamp_plausible', 'scamp_implausible']:
        phenomenon_paths = glob(f'{data_path}/*.tsv')
        for p in tqdm(phenomenon_paths):
            phenomenon = p.split('/')[-1].split('.')[0]
            if phenomenon in ["complex_np_island", "wh_island", "adjunct_island",
                              "principle_A_domain_2", "principle_A_domain_3", "principle_A_c_command"]:
                sentences = Path(p).read_text().strip().split('\n')
                sent_pair = [(x.split('\t')[1], x.split('\t')[0]) for x in sentences]
                test_set[phenomenon] = sent_pair
    else:
        raise ValueError(f'{dataset_name} is not available! Please choose from the following: [blimp, babyberta, scamp_plausible, scamp_implausible, posh]')

    return test_set

def eval_sent_pair(ilm_model, tokenizer, test_set, output_path, model):
    results = {}
    for phe, sents in tqdm(test_set.items()):
        with open(f'{output_path}/{model}_{phe}.tsv', 'w') as f_out:
            f_out.write('sentence1\tPPL1\tSentence2\tPPL2\tCorrect\n')
            correct = 0
            for sent in sents:
                sent = list(sent)
                mean_surprisal0, mean_surprisal1 = ilm_model.sequence_score(sent, reduction=lambda x: -x.mean(0).item())

                if mean_surprisal0 > mean_surprisal1:
                    correct+=1
                    f_out.write(f'{sent[0]}\t{mean_surprisal0}\t{sent[1]}\t{mean_surprisal1}\tcorrect\n')
                else:
                    f_out.write(f'{sent[0]}\t{mean_surprisal0}\t{sent[1]}\t{mean_surprisal1}\tincorrect\n')
            acc = correct/len(sents)
            results[phe] = acc
            print(phe, acc)
    return results



if __name__ == '__main__':
    args = argparse.ArgumentParser('eval language models')
    args.add_argument('model_name', type=str, help='model name')
    args.add_argument('--eval_dataset', type=str, help='dataset name', default='posh')
    args = args.parse_args()
    dataset = args.eval_dataset
    os.makedirs(f'{dataset}_results', exist_ok=True)
    model_name = args.model_name
    refs = list_repo_refs(model_name, repo_type="model")
    if 'gpt2' in model_name:
        sep = '-'
    else:
        sep = '_'
    test = read_data(f'{dataset}', dataset)

    model_name_name = model_name.split('/')[-1]
    f_results = {}
    print(model_name)
    os.makedirs(f'{dataset}_item_results', exist_ok=True)
    output_f = f'{dataset}_item_results'
    ilm_model = scorer.IncrementalLMScorer(model_name, 'cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    acc = eval_sent_pair(ilm_model, tokenizer, test, output_f, model_name_name)
