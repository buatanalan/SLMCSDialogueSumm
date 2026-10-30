# SLMCSDialogueSumm

Research code for training small language models (SLMs) to summarize customer-service style, task-oriented dialogues, using knowledge distillation from a large language model followed by contrastive learning.

## Based on

This repository is built on top of **[symbolic_distill_contrastive_summ](https://github.com/731935354/symbolic_distill_contrastive_summ)**, the code for the paper *Factual Dialogue Summarization via Learning from Large Language Models*. The original training methods are kept: MLE fine-tuning and sequence-level distillation, margin-based contrastive learning (`margin_contrast`), and pair-based contrastive learning (`pair_contrast`).

The original code is licensed under the Apache License 2.0. Its license is kept in [symbolic_distill_contrastive_summ/LICENSE](symbolic_distill_contrastive_summ/LICENSE), and the original README is in [symbolic_distill_contrastive_summ/README.md](symbolic_distill_contrastive_summ/README.md).

## What this repository adds

### 1. Decoder-only model support

The original code only supports encoder-decoder models (BART, Pegasus, T5). This repository adds training and evaluation for decoder-only (causal LM) models such as Qwen2.5 and Qwen3:

| Component | Addition |
|---|---|
| Fine-tuning | `CustomCausalLMTrainer` in [finetuning/trainer.py](symbolic_distill_contrastive_summ/finetuning/trainer.py); causal-LM data handling and Qwen tokenizer/padding setup in [finetuning/run_summarization.py](symbolic_distill_contrastive_summ/finetuning/run_summarization.py) |
| Margin contrast | `LlamaForContrastive`, `QwenForContrastive` in [margin_contrast_models.py](symbolic_distill_contrastive_summ/margin_contrast/margin_contrast_models.py); `CausalUnlikelihoodTrainer` in [margin_contrast_trainer.py](symbolic_distill_contrastive_summ/margin_contrast/margin_contrast_trainer.py) |
| Pair contrast | `LlamaForContrastive` in [pair_contrast_model.py](symbolic_distill_contrastive_summ/pair_contrast/pair_contrast_model.py); `LlamaContrastiveTrainer` in [pair_contrast_trainer.py](symbolic_distill_contrastive_summ/pair_contrast/pair_contrast_trainer.py) |
| Generation / evaluation | [evaluation/basic_eval.py](symbolic_distill_contrastive_summ/evaluation/basic_eval.py) loads both seq2seq and causal-LM checkpoints |

### 2. Updated library versions

The original code targets `transformers==4.30.2`. This repository was updated and tested with:

| Package | Version |
|---|---|
| Python | 3.10 |
| torch | 2.5.1 (CUDA 12.1) |
| transformers | 4.45.0 |
| accelerate | 1.13.0 |
| datasets | 4.8.4 |
| evaluate | 0.4.6 |
| rouge-score | 0.1.2 |
| nltk | 3.9.4 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| wandb | 0.25.1 |
| openai | 2.31.0 |
| python-dotenv | 1.2.2 |
| PyYAML | 6.0.3 |

### 3. Code supporting the research

**Dataset construction** ([generate_dataset/](generate_dataset/))
- [multiwoz_to_qiscus_format.py](generate_dataset/multiwoz_to_qiscus_format.py): converts MultiWOZ 2.2 dialogues into a customer-service chat format (`room_id`, `message_id`, `role`, `message`).
- [sampling.py](generate_dataset/sampling.py): selects 2,500 diverse dialogues with K-Means clustering (k=100) and k-Center Greedy sampling.
- [generate_dataset.py](generate_dataset/generate_dataset.py): generates teacher summaries with an LLM through the OpenRouter API, using the prompts in [prompt_formatted.json](generate_dataset/prompt_formatted.json) (structured *User Profile / Problem / Resolution*) and [prompt_unformatted.json](generate_dataset/prompt_unformatted.json) (single paragraph).
- [generate_negative.py](generate_dataset/generate_negative.py): rule-based hard negatives for contrastive training (removing booking references, generalizing contact numbers, mutating quantities).
- [format_dataset.py](generate_dataset/format_dataset.py), [merge_dataset_train.py](generate_dataset/merge_dataset_train.py): formatting and merging of the generated batches.

**Experiment runner** ([symbolic_distill_contrastive_summ/configs/](symbolic_distill_contrastive_summ/configs/))
- [run_experiment.py](symbolic_distill_contrastive_summ/run_experiment.py) runs a sequence of training stages (e.g. warm-up → fine-tuning → contrastive) defined in YAML. A *method* (`configs/methods/`) lists *experiments* (`configs/experiments/`), each combining a *technique* (`configs/technique/`), a model, and a dataset (`configs/datasets/`).
- [run_test.py](symbolic_distill_contrastive_summ/run_test.py) runs the evaluation scripts listed in `configs/test/` over the trained checkpoints.

**Evaluation metrics**
- [evaluation/align_score.py](symbolic_distill_contrastive_summ/evaluation/align_score.py), [evaluation/unieval.py](symbolic_distill_contrastive_summ/evaluation/unieval.py): factual consistency (from the original repository).
- [CSR.py](symbolic_distill_contrastive_summ/CSR.py): slot recall/precision/F1 against MultiWOZ slot values and Salient Content Ratio.
- [compute_format_metrics.py](symbolic_distill_contrastive_summ/compute_format_metrics.py): compliance with the structured summary format.
- [compute_abstractiveness_metrics.py](symbolic_distill_contrastive_summ/compute_abstractiveness_metrics.py): abstractiveness (novel n-grams, coverage, density).
- The notebooks in [symbolic_distill_contrastive_summ/](symbolic_distill_contrastive_summ/) contain the result analysis, statistical tests, and error analysis.

## Repository structure

```
.
├── generate_dataset/                  # data conversion, sampling, teacher summaries, negatives
├── symbolic_distill_contrastive_summ/ # training, evaluation, configs (based on the original repo)
│   ├── finetuning/                    # MLE / distillation fine-tuning
│   ├── margin_contrast/               # margin-based contrastive learning
│   ├── pair_contrast/                 # pair-based contrastive learning
│   ├── evaluation/                    # generation and metrics
│   └── configs/                       # YAML experiment definitions
├── AlignScore/  UniEval/  FineSurE-ACL24/  SumLLM/  multiwoz/   # external repositories (see Setup)
```

## Setup

1. Install the packages listed above, e.g.
   ```
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   pip install transformers==4.45.0 accelerate==1.13.0 datasets==4.8.4 evaluate==0.4.6 rouge-score==0.1.2 nltk==3.9.4 pandas==2.3.3 scikit-learn==1.7.2 wandb==0.25.1 openai==2.31.0 python-dotenv==1.2.2 PyYAML==6.0.3
   ```
2. The external folders are not included. Clone the ones you need into the repository root:
   ```
   git clone https://github.com/budzianowski/multiwoz.git        # dataset source
   git clone https://github.com/yuh-zha/AlignScore.git           # AlignScore metric
   git clone https://github.com/maszhongming/UniEval.git         # UniEval metric
   ```
   `evaluation/align_score.py` expects the checkpoint at `AlignScore/ckpt/AlignScore-large.ckpt`.
3. To generate teacher summaries, create a `.env` file with your OpenRouter key:
   ```
   OPENAI_API_KEY=<your OpenRouter key>
   ```

## Running experiments

Run from inside `symbolic_distill_contrastive_summ/`:

```
cd symbolic_distill_contrastive_summ
python run_experiment.py   # training, method set by method_num in the script
python run_test.py         # evaluation
```

The dataset configs in `configs/datasets/` point to `data/*.json`. Convert the `.jsonl` files from `generate_dataset/` with [data/convert_data.py](symbolic_distill_contrastive_summ/data/convert_data.py) first, for example:

```
python data/convert_data.py --input_file generate_dataset/all_dialogue_train_2000.jsonl --output_file all_dialogue_train_2000.json
```

## Data

All dialogues come from [MultiWOZ 2.2](https://github.com/budzianowski/multiwoz). The reference summaries were generated by an LLM and are not human-written.

## Acknowledgements

- [symbolic_distill_contrastive_summ](https://github.com/731935354/symbolic_distill_contrastive_summ), the base of this repository
- [MultiWOZ](https://github.com/budzianowski/multiwoz)
- [AlignScore](https://github.com/yuh-zha/AlignScore) and [UniEval](https://github.com/maszhongming/UniEval)
