#!/bin/bash
python benchmark_eval.py xiulinyang/linear_only_0 --eval_dataset zorro
python benchmark_eval.py xiulinyang/linear_only_128 --eval_dataset zorro
python benchmark_eval.py xiulinyang/linear_only_96 --eval_dataset zorro

python benchmark_eval.py xiulinyang/dynamic_only_96 --eval_dataset zorro
python benchmark_eval.py xiulinyang/dynamic_only_128 --eval_dataset zorro
python benchmark_eval.py xiulinyang/dynamic_only_0 --eval_dataset zorro

python benchmark_eval.py xiulinyang/linear_dyck_96 --eval_dataset zorro
python benchmark_eval.py xiulinyang/linear_dyck_128 --eval_dataset zorro
python benchmark_eval.py xiulinyang/linear_dyck_0 --eval_dataset zorro

python benchmark_eval.py xiulinyang/dynamic_dyck_96 --eval_dataset zorro
python benchmark_eval.py xiulinyang/dynamic_dyck_128 --eval_dataset zorro
python benchmark_eval.py xiulinyang/dynamic_dyck_0 --eval_dataset zorro


python benchmark_eval.py xiulinyang/TPT_10Mf_128 --eval_dataset zorro
python benchmark_eval.py xiulinyang/TPT_10Mf_96 --eval_dataset zorro
python benchmark_eval.py xiulinyang/TPT_10Mf_0 --eval_dataset zorro


