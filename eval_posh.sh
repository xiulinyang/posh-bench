#!/bin/bash

MODEL_PATH=$1

python benchmark_eval.py "${MODEL_PATH}" --eval_dataset blimp
python benchmark_eval.py "${MODEL_PATH}" --eval_dataset posh
python benchmark_eval.py "${MODEL_PATH}" --eval_dataset zorro
python benchmark_eval.py "${MODEL_PATH}" --eval_dataset scamp_plausible