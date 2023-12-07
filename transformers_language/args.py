# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
import argparse

from transformers import MODEL_MAPPING, SchedulerType

from transformers_language.dataset_setups import DatasetSetups
from transformers_language.models.bert_attention import AttentionGateType
from transformers_language.models.softmax import SOFTMAX_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune/pre-train a transformers model on a " "MLM/CLM task"
    )

    # *** Options from example script ***

    #
    ## Base
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    #
    ## Data
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )

    #
    ## Model & tokenizer
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than "
            "this will be truncated."
        ),
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )

    #
    ## Task
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )

    #
    ## Training
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    #
    ## Regularization
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    #
    ## Saving/loading & logging
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' "
        "for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--extra_tb_stats",
        action="store_true",
        help="Whether to log extra scalars and histograms to TensorBoard.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            "The integration to report the results and logs to. Supported platforms are "
            '`"tensorboard"`, `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to '
            "report to all integrations. Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )

    # *** New options ***

    #
    ## Data
    parser.add_argument(
        "--dataset_setup",
        choices=DatasetSetups.list_names(),
        default=DatasetSetups.wikitext_103.name,
        help=f"The setup/preset of the datasets to use.",
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default="/local/mnt/workspace/.hf_data",
        help="Where to store data.",
    )
    parser.add_argument(
        "--train_percentage",
        type=int,
        default=None,
        help="Percentage of training set to use.",
    )
    parser.add_argument(
        "--validation_percentage",
        type=int,
        default=None,
        help="Percentage of validation set to use.",
    )

    #
    ## Model & tokenizer
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to a yaml file with model config modifications.",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default="/local/mnt/workspace/.hf_cache",
        help="Where to store models & tokenizers.",
    )

    #
    ## Training
    parser.add_argument(
        "--final_lr_fraction",
        type=float,
        default=0.0,
        help="Final LR as a fraction of the maximum LR (only for CLM).",
    )

    #
    ## Logging
    parser.add_argument(
        "--tqdm_update_interval",
        type=int,
        default=100,
        help="How often to update the progress bar. "
        "Note that using small value might generate large log files.",
    )

    #
    ## Regularization
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=None,
        help="Max gradient norm. If set to 0, no clipping will be applied.",
    )
    parser.add_argument(
        "--grad_norm_type",
        type=float,
        default=2.0,
        help="Norm type to use for gradient clipping.",
    )
    parser.add_argument(
        "--attn_dropout",
        type=float,
        default=None,
        help="Dropout rate to set for attention probs.",
    )
    parser.add_argument(
        "--hidden_dropout",
        type=float,
        default=None,
        help="Dropout rate to set for hidden states.",
    )

    #
    ## Logging
    parser.add_argument(
        "--tb_scalar_log_interval",
        type=int,
        default=1000,
        help="How often to log scalar stats of weights and activations to TensorBoard.",
    )
    parser.add_argument(
        "--tb_hist_log_interval",
        type=int,
        default=10000,
        help="How often to log histograms of weights and activations to TensorBoard.",
    )

    #
    ## Extra options
    parser.add_argument("--wd_LN_gamma", action="store_true")

    parser.add_argument(
        "--skip_attn",
        action="store_true",
        help="Skip attention (don't update the residual).",
    )

    parser.add_argument(
        "--attn_softmax",
        type=str,
        default="vanilla",
        help="Softmax variation to use in attention module.",
        choices=SOFTMAX_MAPPING.keys(),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="If specified, use clipped softmax gamma = -alpha / seq_length.",
    )
    parser.add_argument(
        "--attn_gate_type",
        type=str,
        default=AttentionGateType.none.name,
        help="The type of gating to use for the self-attention.",
        choices=AttentionGateType.list_names(),
    )
    parser.add_argument(
        "--attn_gate_init",
        type=float,
        default=0.5,
        help="init bias s.t. the gate prob is approx this value",
    )
    parser.add_argument(
        "--attn_gate_mlp",
        action="store_true",
        help="Use MLP instead of single linear layer to predict the gate.",
    )
    parser.add_argument(
        "--attn_gate_mlp2",
        action="store_true",
        help="Use bigger MLP instead of single linear layer to predict the gate.",
    )
    parser.add_argument(
        "--attn_gate_linear_all_features",
        action="store_true",
        help="Use Linear (d_model -> n_heads) instead of n_heads Linear's (d_head -> 1).",
    )

    #
    ## Quantization
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--est_num_batches", type=int, default=1)
    parser.add_argument("--n_bits", type=int, default=8)
    parser.add_argument("--n_bits_act", type=int, default=8)
    parser.add_argument("--no_weight_quant", action="store_true")
    parser.add_argument("--no_act_quant", action="store_true")
    parser.add_argument("--qmethod_acts", type=str, default="asymmetric_uniform")
    parser.add_argument("--ranges_weights", type=str, default="minmax")
    parser.add_argument("--ranges_acts", type=str, default="running_minmax")
    parser.add_argument(
        "--percentile", type=float, default=None, help="Percentile (in %) for range estimation."
    )
    parser.add_argument("--quant_setup", type=str, default="all")

    # Fine-tuning
    parser.add_argument("--fine_tuning", action="store_true")

    # Parse options
    args = parser.parse_args()

    return args
