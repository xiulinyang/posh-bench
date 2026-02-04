import argparse, json, os, pathlib
from datasets import load_dataset, load_from_disk
from transformers import PreTrainedTokenizerFast
from clm.tokenization.byte_level_bpe import ByteLevelBPETokenizer
from clm.tokenization.sentencepiece_unigram import SentencePieceUnigramTokenizer
from tokenizers import Tokenizer
from clm.utils import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from tokenizers import AddedToken
import pyarrow.compute as pc

SPECIALS = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def load_json_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_fast_tokenizer(tok_file: str, tokenizer_type):
    print(tok_file)
    return PreTrainedTokenizerFast(
        tokenizer_file=tok_file,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        eos_token=EOS_TOKEN,
        bos_token=BOS_TOKEN,
        add_prefix_space=True if 'bpe' in tokenizer_type else False,
    )

def main(args):
    cfg = load_json_cfg(args.config)
    tok_cfg = cfg["tokenizer"]
    model_name = cfg["tokenizer"]['model_name']
    raw_data = cfg['tokenizer']['raw_data']
    tok_type = tok_cfg["type"]
    tokenizer_name = cfg["tokenizer"]['tokenizer_name']
    vocab_size = int(tok_cfg["vocab_size"])
    add_prefix_space = bool(tok_cfg.get("add_prefix_space", False))
    tok_train_file = tok_cfg.get("train_file")
    save_dir = tok_cfg.get("save_dir")
    tok_dir = f'models/{tokenizer_name}'
    print(save_dir)
    ensure_dir(save_dir)


    if not os.path.exists(tok_train_file):
        raise FileNotFoundError(f"train_file not found: {tok_train_file}")
    if not tokenizer_name:
        if tok_type == "bpe":
            base_tokenizer = ByteLevelBPETokenizer(add_prefix_space=True if add_prefix_space else False)
        elif tok_type== 'unigram':
            base_tokenizer = SentencePieceUnigramTokenizer()
        else:
            raise ValueError(f"Unsupported tokenizer type: {tok_type}")

        base_tokenizer.train(
            files=tok_train_file,
            vocab_size=vocab_size,
            special_tokens=SPECIALS,
        )
        tok_json_path = os.path.join(save_dir, "tokenizer.json")
        base_tokenizer.save(tok_json_path)
        base_tokenizer.save_model(save_dir)
        tokenizer = build_fast_tokenizer(tok_json_path, tok_type)
        eos_id = tokenizer.eos_token_id
    else:
        print('loaded a pretrained tokenizer!')
        tok_json_path = os.path.join(tok_dir, "tokenizer.json")
        tokenizer = build_fast_tokenizer(tok_json_path, 'bpe')
        eos_id = tokenizer.eos_token_id

    def encode(ex):
        ids = tokenizer(ex["text"]).input_ids
        ids.append(eos_id)
        return {"input_ids": ids}

    if len(tokenizer) != vocab_size:
        raise ValueError(f"Vocab mismatch: tokenizer len={len(tokenizer)} vs config vocab_size={vocab_size}")

    files = {}
    if tok_cfg.get("train_file"): files["train"] = tok_cfg["train_file"]
    if tok_cfg.get("validation_file"): files["validation"] = tok_cfg["validation_file"]

    raw_all = load_dataset("text", data_files=files, keep_linebreaks=True)
    encoded = raw_all.map(encode, batched=False, remove_columns=raw_all["train"].column_names, num_proc=20, desc="Encoding to IDs")
    tbl = encoded["train"].data
    arr = tbl.column("input_ids")
    total_tokens = pc.sum(pc.list_value_length(arr)).as_py()
    print(total_tokens)
    print('total tokens for the training split: ', total_tokens)
    tok_ds_dir = os.path.join(f'{raw_data}', model_name)
    ensure_dir(tok_ds_dir)
    encoded.save_to_disk(tok_ds_dir)
    tokenizer.save_pretrained(tok_json_path)
    print(f" Tokenizer + tokenized dataset saved under: {save_dir}")
    print(f" - tokenizer.json: {tok_json_path}")
    print(f" - tokenized ds:   {tok_ds_dir}")
    ds = load_from_disk(tok_ds_dir)
    print("Example:", ds["train"][0]["input_ids"][:50])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=str, required=True, help="Path to JSON config generated earlier")
    args = ap.parse_args()
    main(args)
