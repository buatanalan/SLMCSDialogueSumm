#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Rongxin Zhu. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
import pathlib
import json

import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset
from filelock import FileLock

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForCausalLM,
    set_seed,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
# from transformers.utils.generic import send_example_telemetry
from transformers.utils.hub import is_offline_mode
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
from finetuning.trainer import CustomSeq2SeqTrainer, CustomCausalLMTrainer

check_min_version("4.30.0")  # current version: 4.30.2

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

load_dotenv()

try:
    nltk.data.find("tokenizers/punkt")
    nltk.download('punkt_tab')
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]

import torch

def debug_labels(dataset, tokenizer, num_samples=5):
    """Cek apakah labels valid sebelum training."""
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")
    print(f"Pad token id: {tokenizer.pad_token_id}")
    print("-" * 50)

    for i in range(min(num_samples, len(dataset))):
        labels = dataset[i]["labels"]
        labels_t = torch.tensor(labels)

        # mask -100 untuk cek range token asli
        valid = labels_t[labels_t != -100]

        print(f"[Sample {i}]")
        print(f"  panjang labels      : {len(labels)}")
        print(f"  jumlah -100 (ignore): {(labels_t == -100).sum().item()}")
        print(f"  jumlah token valid  : {len(valid)}")

        # === pengecekan masalah umum ===
        if len(valid) == 0:
            print("  ⚠️  SEMUA token = -100 → loss akan NaN/None!")

        if (valid >= vocab_size).any():
            bad = valid[valid >= vocab_size].tolist()
            print(f"  ⚠️  Token >= vocab_size: {bad[:10]}  → loss NaN!")

        if (valid < 0).any():
            bad = valid[valid < 0].tolist()
            print(f"  ⚠️  Token negatif (selain -100): {bad[:10]}  → loss NaN!")

        if tokenizer.pad_token_id in valid.tolist():
            print(f"  ⚠️  pad_token_id ({tokenizer.pad_token_id}) belum di-set -100!")

        print()


def debug_batch_forward(model, dataset, data_collator, device="cuda"):
    """Jalankan 1 batch lewat model, cek loss & logits NaN/inf."""
    batch = data_collator([dataset[0], dataset[1]])
    batch = {k: v.to(device) for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        out = model(**batch)

    print("=== Forward pass check ===")
    print(f"loss          : {out.loss.item()}")
    print(f"loss is NaN   : {torch.isnan(out.loss).item()}")
    print(f"loss is inf   : {torch.isinf(out.loss).item()}")
    print(f"logits NaN    : {torch.isnan(out.logits).any().item()}")
    print(f"logits inf    : {torch.isinf(out.logits).any().item()}")
    print(f"logits min/max: {out.logits.min().item():.3f} / {out.logits.max().item():.3f}")

    # cek input_ids juga di luar range
    ids = batch["input_ids"]
    ids = batch["input_ids"]
    vocab = model.config.vocab_size
    print(f"input_ids max : {ids.max().item()} (vocab={vocab})")
    if ids.max().item() >= vocab:
        print("  ⚠️  input_ids di luar vocab → forward error / NaN!")

def debug_backward(model, dataset, data_collator, device="cuda"):
    """Cek gradien NaN/inf setelah backward."""
    batch = data_collator([dataset[0], dataset[1]])
    batch = {k: v.to(device) for k, v in batch.items()}

    model.train()
    out = model(**batch)
    out.loss.backward()

    print("=== Backward pass check ===")
    print(f"loss: {out.loss.item()}")

    total_norm = 0.0
    nan_params = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                nan_params.append(name)
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    print(f"grad norm total: {total_norm:.3f}")
    if nan_params:
        print(f"⚠️  Param dengan grad NaN/inf ({len(nan_params)}):")
        for n in nan_params[:10]:
            print(f"     {n}")
    else:
        print("✅ Semua gradien normal")

    model.zero_grad()

def trace_nan(model):
    def make_hook(name):
        def hook(module, inp, out):
            def has_nan(x):
                return isinstance(x, torch.Tensor) and (torch.isnan(x).any() or torch.isinf(x).any())
            # cek input
            for t in inp:
                if has_nan(t):
                    print(f"⚠️ NaN/inf MASUK ke {name} ({module.__class__.__name__})")
                    return
            # cek output
            o = out[0] if isinstance(out, tuple) else out
            if has_nan(o):
                print(f"⚠️ NaN/inf KELUAR dari {name} ({module.__class__.__name__})")
        return hook
    for name, m in model.named_modules():
        m.register_forward_hook(make_hook(name))


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    use_lora: bool = field(
        default=False, metadata={"help": "Whether to use LoRA for parameter efficient finetuning."}
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help" : "trust remote code or not"}
    )
    overwrite_output_dir: bool = field(
        default=True, metadata={"help" : "overwrite"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    prompt_path: Optional[str] = field(
        default="prompt_unformatted",
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_summarization", model_args, data_args)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    print(model_args)

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train:
        if not getattr(model_args, "overwrite_output_dir", False):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
   
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if not config.is_encoder_decoder:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            # use_auth_token=True if model_args.use_auth_token else None,
        )

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=True if model_args.use_auth_token else None,
            trust_remote_code=model_args.trust_remote_code
        )
        if other_args.model_type == "qwen":
            # ============================================================
            # FIX: Tokenizer Qwen yang AMAN (tidak corrupt saat di-save)
            # ============================================================
            # BUG SEBELUMNYA:
            #   tokenizer.pad_token = "<|endoftext|>"  (string)
            # bisa menambah token sebagai "added_tokens" baru, lalu saat
            # save mid-training, tokenizer.json jadi corrupt (vocab=1 token).
            #
            # SOLUSI: assign via existing token ID, JANGAN tambah baru.
            # ============================================================
            tokenizer = AutoTokenizer.from_pretrained(
                other_args.model_name_or_path,
                trust_remote_code=True,
                use_fast=True,
            )
    
            # ----- Set pad_token via existing token (TIDAK menambah token) -----
            if tokenizer.pad_token_id is None:
                # Qwen2.5 base: <|endoftext|> ada di vocab dengan ID 151643.
                # Cek via convert_tokens_to_ids (return unk_token_id kalau tidak ada).
                endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
                if endoftext_id is not None and endoftext_id != tokenizer.unk_token_id:
                    # Set via attribute internal -- JANGAN pakai assignment string,
                    # supaya tidak memicu add_special_tokens.
                    tokenizer.pad_token_id = endoftext_id
                else:
                    # Fallback: pakai eos_token_id yang sudah ada
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                    # Sinkronkan pad_token string juga (aman karena eos_token sudah ada)
                    tokenizer.pad_token = tokenizer.eos_token
    
            # Causal LM dengan format [PAD..PAD, src, tgt] -> left pad source
            tokenizer.padding_side = "left"
    
            # ----- Verifikasi tokenizer SEHAT sebelum melanjutkan -----
            vocab_size = tokenizer.vocab_size
            tok_len = len(tokenizer)
            assert vocab_size > 1000, (
                f"Tokenizer corrupt! vocab_size={vocab_size}. "
                f"Cek model_name_or_path: {other_args.model_name_or_path}"
            )
            # Sanity: encode-decode round trip
            test_text = "Hello, this is a test."
            test_ids = tokenizer.encode(test_text, add_special_tokens=False)
            test_decoded = tokenizer.decode(test_ids)
            assert len(test_ids) > 1, f"Tokenizer broken: '{test_text}' -> {test_ids}"
    
            # ----- Load config & model -----
            config = AutoConfig.from_pretrained(
                other_args.model_name_or_path,
                trust_remote_code=True,
            )
            config.pad_token_id = tokenizer.pad_token_id
    
            model = QwenForContrastive.from_pretrained(
                other_args.model_name_or_path,
                config=config,
                trust_remote_code=True,
            )
            model.config.pad_token_id = tokenizer.pad_token_id
    
            # Sanity check
            print(f"[Qwen Setup]")
            print(f"  model_path       : {other_args.model_name_or_path}")
            print(f"  tokenizer vocab  : {vocab_size}")
            print(f"  len(tokenizer)   : {tok_len}")
            print(f"  model vocab      : {model.config.vocab_size}")
            print(f"  pad_token        : {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
            print(f"  eos_token        : {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
            print(f"  padding_side     : {tokenizer.padding_side}")
            print(f"  model class      : {type(model).__name__}")
            print(f"  test encode      : {test_ids[:5]}... -> {test_decoded!r}")
            assert "Contrastive" in type(model).__name__, (
                "FATAL: model bukan QwenForContrastive! "
                "Cek inherit class di margin_contrast_models.py"
            )

    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        # Apply LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM if model.config.is_encoder_decoder else TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    decoder_start_id = getattr(model.config, "decoder_start_token_id", None)

    if decoder_start_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if model.config.is_encoder_decoder:
            if isinstance(tokenizer, MBartTokenizer):
                model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
            else:
                model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )
    
    def preprocess_function(examples):
                
        # 1. Bersihkan data dari None
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])
        
        is_encoder_decoder = model.config.is_encoder_decoder

        # --- LOGIKA SEQ2SEQ (Encoder-Decoder seperti T5/BART) ---
        if is_encoder_decoder:
            
            inputs = [source_prefix + inp for inp in inputs]
    
            # 2. Tokenize dengan padding FIX (max_length)
            model_inputs = tokenizer(
                inputs, 
                max_length=data_args.max_source_length, 
                padding="max_length", 
                truncation=True
            )
        
            labels = tokenizer(
                text_target=targets, 
                max_length=max_target_length, 
                padding="max_length", 
                truncation=True
            )
        
            # 3. Handle ignore_pad_token_for_loss (-100)
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] 
                for label in labels["input_ids"]
            ]
        
            # 4. Bungkus menjadi 3D: [1, seq_len] untuk setiap sampel
            model_inputs["input_ids"] = [[seq] for seq in model_inputs["input_ids"]]
            model_inputs["attention_mask"] = [[seq] for seq in model_inputs["attention_mask"]]
            model_inputs["labels"] = [[seq] for seq in labels["input_ids"]]
        
            return model_inputs

        # --- LOGIKA CAUSAL LM (Decoder-only seperti Llama/Qwen) ---
        else:
            # FIX #6: HAPUS setup pad_token/padding_side di sini.
            # Setup sudah dilakukan di awal main() (sebelum dataset dibuat),
            # supaya konsisten antara training dan validation.
            # tokenizer.pad_token = tokenizer.eos_token        <- HAPUS
            # model.config.pad_token_id = model.config.eos_token_id <- HAPUS
            # tokenizer.padding_side = "left"                  <- HAPUS

            all_input_ids = []
            all_labels = []
            all_attention_masks = []
            prompt_path = pathlib.Path(__file__).parent.parent.parent / f'generate_dataset/{data_args.prompt_path}'

            with open(prompt_path, "r") as f:
                try:
                    data = json.loads(f.read())
                    instruct = data["prompt_positive"]
                except Exception as e:
                    print(f"Failed to load prompt !!! {e}")
                    instruct = "Summarize the following dialogue:\n"
            # Tentukan jatah masing-masing (misal total 512 = 384 + 128)
            max_src = data_args.max_source_length  # Jatah Input
            max_tgt = data_args.max_target_length  # Jatah Output
            
            for i in range(len(inputs)):
                # 1. Tokenisasi Input (Tanpa Padding)
                full_prompt = f"{instruct}{inputs[i]}\nSummary: "
                tokenized_src = tokenizer(
                    full_prompt,
                    max_length=max_src,
                    truncation=True,
                    add_special_tokens=True
                )["input_ids"]
            
                # 2. Tokenisasi Output (Tanpa Padding)
                tokenized_tgt = tokenizer(
                    targets[i] + tokenizer.eos_token,
                    max_length=max_tgt,
                    truncation=True,
                    add_special_tokens=False
                )["input_ids"]
            
                # --- PROSES PADDING ---
            
                # A. Input Block: Left Padding
                # [PAD, PAD, ..., Prompt, Dialogue]
                num_pad_src = max_src - len(tokenized_src)
                src_block = ([tokenizer.pad_token_id] * num_pad_src) + tokenized_src
                src_mask = ([0] * num_pad_src) + ([1] * len(tokenized_src))
            
                # B. Output Block: Right Padding
                # [Summary, EOS, PAD, PAD, ...]
                num_pad_tgt = max_tgt - len(tokenized_tgt)
                tgt_block = tokenized_tgt + ([tokenizer.pad_token_id] * num_pad_tgt)
                tgt_mask = ([1] * len(tokenized_tgt)) + ([0] * num_pad_tgt)
            
                # --- PENGGABUNGAN ---
            
                # Gabungkan menjadi satu sequence utuh
                input_ids = src_block + tgt_block
                attention_mask = src_mask + tgt_mask
                
                # Labels: Seluruh blok input adalah -100, 
                # di blok output hanya Summary yang asli, sisanya (pad) adalah -100
                labels = ([-100] * max_src) + tokenized_tgt + ([-100] * num_pad_tgt)

                all_input_ids.append(input_ids)
                all_labels.append(labels)
                all_attention_masks.append(attention_mask)
            
            return {
                "input_ids": all_input_ids,
                "labels": all_labels,
                "attention_mask": all_attention_masks
            }

    def preprocess_logits_for_metrics(logits, labels):
        """
        Logits yang datang biasanya berbentuk [batch_size, seq_length, vocab_size].
        Kita ubah langsung jadi [batch_size, seq_length] di level GPU.
        """
        if isinstance(logits, tuple):
            # Beberapa model mengembalikan tuple (logits, past_key_values)
            logits = logits[0]
        return logits.argmax(dim=-1)

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            
    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if model.config.is_encoder_decoder:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    else:
        data_collator = default_data_collator

    sample_input_ids = train_dataset['input_ids'][0]
    decoded_text = tokenizer.decode(sample_input_ids, skip_special_tokens=False)

    # Metric
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
    
        # 1. Ekstraksi prediksi
        if isinstance(preds, tuple):
            preds = preds[0]
        if len(preds.shape) == 3:
            preds = np.argmax(preds, axis=-1)
    
        # 2. Pengamanan Tipe Data (Menghindari OverflowError)
        # Ubah format ke int64 agar muat untuk vocab size yang besar (>50.000)
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
            
        preds = preds.astype(np.int64)
        labels = labels.astype(np.int64)
    
        # 3. Penggantian ignore_index (-100) dengan pad_token_id
        # Hindari proses iterasi manual (looping) yang memutus panjang array
        ignore_id = -100
        pad_id = tokenizer.pad_token_id
    
        # Array 2D akan tetap utuh, sehingga aman masuk ke batch_decode
        preds = np.where(preds != ignore_id, preds, pad_id)
        labels = np.where(labels != ignore_id, labels, pad_id)
    
        # 4. Decode ke teks
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        # 5. Post-processing dan Kalkulasi Metrik
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        
        # 6. Hitung panjang generasi
        prediction_lens = [np.count_nonzero(pred != pad_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        
        return result

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    training_args.save_safetensors = False

    # Initialize our Trainer
    if model.config.is_encoder_decoder :
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            # eval_dataset=eval_dataset if training_args.do_eval else None,
            # test_dataset=predict_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None
        )
    else:
        trainer = CustomCausalLMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            # eval_dataset=eval_dataset if training_args.do_eval else None,
            # test_dataset=predict_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics, # TAMBAHKAN INI
        )
        training_args.predict_with_generate = True
    debug_labels(train_dataset, tokenizer)
    debug_batch_forward(model, train_dataset, data_collator)
    debug_backward(model, train_dataset, data_collator)
    trace_nan(model)
    w = model.get_input_embeddings().weight
    print("embed awal nan:", torch.isnan(w).any().item())
    print("embed awal inf:", torch.isinf(w).any().item())
    print("embed min/max:", w.min().item(), w.max().item())
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_predict_results = trainer.predict(eval_dataset, metric_key_prefix="eval")
        metrics = eval_predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_predict_samples, len(eval_dataset))

        trainer.log_metrics("predict_eval", metrics)
        trainer.save_metrics("predict_eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        test_predict_results = trainer.predict(predict_dataset, metric_key_prefix="test")
        metrics = test_predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict_test", metrics)
        trainer.save_metrics("predict_test", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = test_predict_results.predictions
                if len(predictions.shape) == 3:
                    predictions = np.argmax(predictions, axis=-1)
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
