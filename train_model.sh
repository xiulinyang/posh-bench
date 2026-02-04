#!/bin/bash
source /scratch/xiulyang/anaconda3/etc/profile.d/conda.sh
dataset_size=$1
vocab_size=$2
model_type=$3
baby_or_wiki=$4
seed=42

WORK_DIR='/scratch/xiulyang/'
model_name="$model_type"_"$baby_or_wiki"_"$dataset_size"_"$vocab_size"_"$seed"

python generate_config.py \
  --dataset_size "$dataset_size" \
  --tokenizer_type "bpe" \
  --vocab "$vocab_size" \
  --model_type "$model_type" \
  --baby_or_wiki "$baby_or_wiki" \
  --pretokenized_file


python train_tokenizer.py -c configs/$model_type/${baby_or_wiki}_${dataset_size}_${vocab_size}_${seed}.json
python save_config.py -c configs/$model_type/${baby_or_wiki}_${dataset_size}_${vocab_size}_${seed}.json
python train_clm.py configs/$model_type/${baby_or_wiki}_${dataset_size}_${vocab_size}_${seed}.json

bash upload_models.sh "$model_name"
