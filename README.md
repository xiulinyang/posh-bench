# posh-bench
This is the repository for the paper: A Unified Assessment of the Poverty of the Stimulus Argument for Neural Language Models

[//]: # (&#40;https://arxiv.org/abs/2602.09992#:~:text=According%20to%20the%20Poverty%20of,necessary%20to%20explain%20language%20learning.&#41;)

[//]: # (by Xiulin Yang, Arianna Bisazza, Nathan Schneider, and Ethan Gotlieb Wilcox)

## update
July 30, 2026: We manually reviewed the benchmark and revised examples that (i) contained words not included in the word list, (ii) contained unnatural collocations, or (iii) used verbs that can be either transitive or intransitive.


## Setup
To set up the environment, run:

```bash
conda create -n posh-bench python=3.11
conda activate posh-bench
pip install -r requirements.txt
pip install -e . --no-dependencies
```

## Experiments 1&2
To run the experiments, use the following command:

```bash

# train models
bash train_model.sh $dataset_size $vocab_size $model_type $baby_or_wiki # you can find the options available in ```generate_config.py```
# evaluate models
python benchmark_eval.py model_name --eval_dataset posh --best_checkpoint 
```
## Experiments 3

You can run experiment 3 using the code from the repositories for [dynamic locality bias](https://github.com/osekilab/CPLM) and [pre-pretraining](https://github.com/michahu/pre-pretraining).
(Customized code will be released upon acceptance to preserve anonymity during the review process.)
## Dataset
- Training data: it is stored in [OSF](https://osf.io/jht6y/overview)
- Evaluation data: different benchmarks are listed in different folders in this repository, e.g., posh: posh-bench



## Citation
```
@misc{yang2026unifiedassessmentpovertystimulus,
      title={A Unified Assessment of the Poverty of the Stimulus Argument for Neural Language Models}, 
      author={Xiulin Yang and Arianna Bisazza and Nathan Schneider and Ethan Gotlieb Wilcox},
      year={2026},
      eprint={2602.09992},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.09992}, 
}
```
