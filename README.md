# posh-bench

This is the repository for the paper: A Unified Assessment of the Poverty of the Stimulus Argument for Neural Language Models
by Xiulin Yang, Arianna Bisazza, Nathan Schneider, and Ethan Gotlieb Wilcox

## Setup
To set up the environment, run:

```bash
conda create -n posh-bench python=3.8
conda activate posh-bench
pip install -r requirements.txt
pip install -e . --no-dependencies
```

## Experiments
To run the experiments, use the following command:

```bash
# train models
bash train_model.sh $dataset_size $vocab_size $model_type $baby_or_wiki # you can find the options available in ```generate_config.py```
# evaluate models
python benchmark_eval.py model_name --eval_dataset posh --best_checkpoint 
```

## Dataset
- Training data: it is stored in [OSF](https://osf.io/jht6y/overview)
- Evaluation data: different benchmarks are listed in different folders in this repository, e.g., posh: posh-bench



## Citation
Will come soon!