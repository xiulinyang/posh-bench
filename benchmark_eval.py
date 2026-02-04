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

    if dataset_name in ['zorro', 'posh']:
        phenomenon_paths = glob(f'{data_path}/*.txt')
        for p in tqdm(phenomenon_paths):
            phenomenon = p.split('/')[1].split('.')[0]
            sentences = Path(p).read_text().strip().split('\n')
            if 'strict' in p:
                sent_pair = [(sentences[i], sentences[i+1], sentences[i+2], sentences[i+3], sentences[i+4], sentences[i+5])for i in range(len(sentences)) if i%6==0]
            else:
                sent_pair = [(sentences[i], sentences[i+1])for i in range(len(sentences)) if i%2==0]
            test_set[phenomenon] = sent_pair
    elif dataset_name in ['blimp']:
        phenomenon_paths = glob(f'{data_path}/*.jsonl')
        for p in tqdm(phenomenon_paths):
            phenomenon_n = p.split('/')[1].split('.')[0]

            phenomenon = pd.read_json(p, lines=True).to_dict(orient='records')
            sent_pair = [(x['sentence_bad'], x['sentence_good']) for x in phenomenon]
            test_set[phenomenon_n] = sent_pair
            print(len(test_set[phenomenon_n]))
    elif dataset_name in ['scamp_plausible', 'scamp_implausible']:
        phenomenon_paths = glob(f'{data_path}/*.tsv')
        for p in tqdm(phenomenon_paths):
            phenomenon = p.split('/')[-1].split('.')[0]

            sentences = Path(p).read_text().strip().split('\n')
            sent_pair = [(x.split('\t')[1], x.split('\t')[0]) for x in sentences]
            test_set[phenomenon] = sent_pair
    else:
        raise ValueError(f'{dataset_name} is not available! Please choose from the following: [blimp, babyberta, scamp_plausible, scamp_implausible, posh]')

    return test_set

def eval_sent_pair(ilm_model, tokenizer, test_set):
    results = {}
    distributions = {}
    for phe, sents in tqdm(test_set.items()):
        correct = 0
        distribution = []
        for sent in sents:
            sent = list(sent)
            if 'strict' in phe:
                num_tokens = [len(tokenizer.encode(sent[i],add_special_tokens=False)) for i in range(len(sent))]
                scores = ilm_model.sequence_score(sent, reduction=lambda x: -x.sum(0).item())
                ppls = [(i, x/y) for i, (x, y) in enumerate(zip(scores, num_tokens))]
                ppls = sorted(ppls, key=lambda x: x[1])
                if ppls[0][0] ==5:
                    correct += 1
            else:
                num_token0 = len(tokenizer.encode(sent[0],add_special_tokens=False))
                num_token1 = len(tokenizer.encode(sent[1],add_special_tokens=False))
                nll0, nll1 = ilm_model.sequence_score(sent, reduction=lambda x: -x.sum(0).item())
                ppl0 = nll0/num_token0
                ppl1 = nll1/num_token1
                distribution.append(f'{sent[0]}\t{ppl0}\t{sent[1]}\t{ppl1}')
                if ppl0 > ppl1:
                    correct+=1
        acc = correct/len(sents)
        results[phe] = acc
        distributions[phe] = '|||'.join(distribution)
        print(phe, acc)
    return results, distributions



if __name__ == '__main__':
    args = argparse.ArgumentParser('eval language models')
    args.add_argument('model_name', type=str, help='model name')
    args.add_argument('--best_checkpoint', action='store_true')
    args.add_argument('--eval_dataset', type=str, help='dataset name', default='posh')
    args = args.parse_args()
    dataset = args.eval_dataset
    os.makedirs(f'{dataset}_results', exist_ok=True)
    model_name = args.model_name
    best_checkpoint = args.best_checkpoint
    refs = list_repo_refs(model_name, repo_type="model")
    num_checkpoints = refs.branches
    if 'gpt2' in model_name:
        sep = '-'
    else:
        sep = '_'
    checkpoints = sorted([x.name for x in num_checkpoints if 'main' not in x.name], key=lambda x: int(x.split(sep)[-1]))
    test = read_data(f'{dataset}', dataset)

    model_name_name = model_name.split('/')[-1]
    f_results = {}
    if best_checkpoint:
        print(model_name)
        ilm_model = scorer.IncrementalLMScorer(model_name, 'cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        acc, dist = eval_sent_pair(ilm_model, tokenizer, test)
        f_results['best'] = acc
        pd.DataFrame(f_results).to_csv(f'{dataset}_results/results_{model_name_name}_best.csv')
        df_dist = pd.DataFrame.from_dict(dist, orient='index', columns=['distribution'])
        df_dist.index.name = 'phenomenon'
        df_dist.to_csv(f'{dataset}_results/distributions_{model_name_name}_best.csv')
    else:
        for checkpoint in checkpoints:
            results = {}
            print(model_name, checkpoint)
            ilm_model = scorer.IncrementalLMScorer(model_name, 'cuda',revision=checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            acc, dist = eval_sent_pair(ilm_model, tokenizer, test)
            results[checkpoint] = acc
            pd.DataFrame(results).to_csv(f'{dataset}_results/results_{model_name_name}_ckpt{checkpoint}.csv')
            df_dist = pd.DataFrame.from_dict(dist, orient='index', columns=['distribution'])
            df_dist.index.name = 'phenomenon'
            # df_dist.to_csv(f'{dataset}_results/distributions_ckpt{checkpoint}.csv')
