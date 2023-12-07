#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation

You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
"""
import json
import logging
import math
import os
import random
import warnings
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from pprint import pformat

import datasets
import numpy as np
import torch
import transformers
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from timm.utils import AverageMeter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)

from transformers_language.args import parse_args
from transformers_language.dataset_setups import DatasetSetups
from transformers_language.models.opt_attention import (
    AttentionGateType,
    OPTAttentionWithExtras,
)
from transformers_language.models.softmax import SOFTMAX_MAPPING
from transformers_language.utils import count_params, kurtosis

logger = get_logger("run_clm")


MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# attach hooks for activation stats
def attach_act_hooks_for_eval(model):
    act_dict = OrderedDict()

    def _make_hook(name):
        def _hook(mod, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                inp = inp[0]
            if isinstance(out, tuple) and len(out) > 0:
                out = out[0]
            act_dict[name] = (inp, out)

        return _hook

    for name, module in model.named_modules():
        module.register_forward_hook(_make_hook(name))
    return act_dict


def attach_tb_act_hooks(model):
    act_dict = OrderedDict()

    def _make_hook(name):
        def _hook(mod, inp, out):
            act_dict[name] = out[0]

        return _hook

    for name, module in model.named_modules():
        module.register_forward_hook(_make_hook(name))
    return act_dict


def main():
    args = parse_args()

    # convert dataset setup to an enum
    dataset_setup = DatasetSetups[args.dataset_setup]

    # Initialize the accelerator. We will let the accelerator handle device placement for us in
    # this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up
    # all supported trackers in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs
    )
    accelerator.project_configuration.total_limit = 1
    accelerator.project_configuration.automatic_checkpoint_naming = True

    # log passed args
    logger.info(args)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # -----------------------------------------------------------------

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": args.model_cache_dir,
    }
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, **config_kwargs)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # Load model config changes from file, if provided
    if args.config_path is not None:
        logger.info(f"Loading model config changes from {args.config_path}")
        with open(args.config_path) as f:
            config_changes = yaml.safe_load(f)

        for key, value in config_changes.items():
            setattr(config, key, value)

    # Set dropout rates, if specified
    if args.attn_dropout is not None:
        logger.info(f"Setting attention dropout rate to {args.attn_dropout}")
        config.attention_probs_dropout_prob = args.attn_dropout

    if args.hidden_dropout is not None:
        logger.info(f"Setting hidden dropout rate to {args.hidden_dropout}")
        config.hidden_dropout_prob = args.hidden_dropout

    # Display config after changes
    logger.info("HuggingFace config after user changes:")
    logger.info(str(config))

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.model_cache_dir,
    }
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, **tokenizer_kwargs
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Load and prepare model
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            cache_dir=args.model_cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # >> replace self-attention module with ours
    # NOTE: currently assumes OPT
    for layer_idx in range(len(model.model.decoder.layers)):
        old_attn = model.model.decoder.layers[layer_idx].self_attn
        model.model.decoder.layers[layer_idx].self_attn = OPTAttentionWithExtras(
            embed_dim=old_attn.embed_dim,
            num_heads=old_attn.num_heads,
            dropout=old_attn.dropout,
            is_decoder=old_attn.is_decoder,
            bias=True,
            # new
            softmax_fn=SOFTMAX_MAPPING[args.attn_softmax],
            alpha=args.alpha,
            max_seq_length=args.block_size,
            skip_attn=args.skip_attn,
            attn_gate_type=AttentionGateType[args.attn_gate_type],
            attn_gate_init=args.attn_gate_init,
            attn_gate_mlp=args.attn_gate_mlp,
            attn_gate_mlp2=args.attn_gate_mlp2,
            attn_gate_linear_all_features=args.attn_gate_linear_all_features,
            fine_tuning=args.fine_tuning,
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Print model
    logger.info("Model:")
    logger.info(model)

    # Display num params
    n_embeddings = count_params(model.model.decoder.embed_tokens) + count_params(
        model.model.decoder.embed_positions
    )
    n_decoder = count_params(model.model.decoder.layers) + count_params(
        model.model.decoder.final_layer_norm
    )
    n_head = count_params(model.lm_head)
    logger.info(
        f"\nNumber of parameters:\n"
        f"\t* Embeddings:\t{n_embeddings}\n"
        f"\t* Decoder:\t{n_decoder}\n"
        f"\t* Head:\t{n_head}\n"
        f"\t= Total (pre-training):\t{n_embeddings + n_decoder + n_head}\n"
        f"\t= Total (decoder only):\t{n_embeddings + n_decoder}\n"
    )

    # -----------------------------------------------------------------

    # Get the datasets.
    # In distributed training, the load_dataset function guarantee that only one local process can
    # concurrently download the dataset.
    tokenized_book_wiki_path = (
        Path(args.data_cache_dir) / f"tokenized_book_wiki_OPT_{args.block_size}"
    )
    if dataset_setup == DatasetSetups.bookcorpus_and_wiki and tokenized_book_wiki_path.exists():
        accelerator.print(f"Loading tokenized dataset from {str(tokenized_book_wiki_path)}")

        tokenized_datasets = load_from_disk(str(tokenized_book_wiki_path))

    else:  # do tokenization
        train_split = (
            "train" if args.train_percentage is None else f"train[:{args.train_percentage}%]"
        )
        val_split = (
            "validation"
            if args.validation_percentage is None
            else f"validation[:{args.validation_percentage}%]"
        )

        if dataset_setup == DatasetSetups.wikitext_2:
            raw_datasets = DatasetDict()
            raw_datasets["train"] = load_dataset(
                "wikitext", "wikitext-2-raw-v1", cache_dir=args.data_cache_dir, split=train_split
            )
            raw_datasets["validation"] = load_dataset(
                "wikitext", "wikitext-2-raw-v1", cache_dir=args.data_cache_dir, split=val_split
            )

        elif dataset_setup == DatasetSetups.wikitext_103:
            raw_datasets = DatasetDict()
            raw_datasets["train"] = load_dataset(
                "wikitext", "wikitext-103-raw-v1", cache_dir=args.data_cache_dir, split=train_split
            )
            raw_datasets["validation"] = load_dataset(
                "wikitext", "wikitext-103-raw-v1", cache_dir=args.data_cache_dir, split=val_split
            )

        elif dataset_setup == DatasetSetups.bookcorpus_and_wiki:
            bookcorpus = load_dataset(
                "bookcorpus", cache_dir=args.data_cache_dir, split=train_split
            )

            wiki_train = load_dataset(
                "wiki40b", "en", cache_dir=args.data_cache_dir, split=train_split
            )
            wiki_val = load_dataset("wiki40b", "en", cache_dir=args.data_cache_dir, split=val_split)

            # only keep the 'text' column
            wiki_train = wiki_train.remove_columns(
                [c for c in wiki_train.column_names if c != "text"]
            )
            wiki_val = wiki_val.remove_columns(
                [col for col in wiki_val.column_names if col != "text"]
            )
            assert bookcorpus.features.type == wiki_train.features.type

            raw_datasets = DatasetDict()
            raw_datasets["train_book"] = bookcorpus
            raw_datasets["train_wiki"] = wiki_train
            raw_datasets["validation"] = wiki_val

        else:
            raise ValueError(f"Unknown dataset, {dataset_setup}")

        # Preprocessing the datasets.
        # Check sequence length
        if args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > 1024:
                logger.warning(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
            block_size = 1024
        else:
            if args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(args.block_size, tokenizer.model_max_length)

        # Tokenize all the texts.
        column_names = raw_datasets["validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        # YB: make the default bs for text pre-processing explicit
        tokenizer_map_batch_size = 1000
        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                batch_size=tokenizer_map_batch_size,
                writer_batch_size=tokenizer_map_batch_size,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        # Main data processing function that will concatenate all texts from our dataset and generate
        # chunks of max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of
            # this drop, you can customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            else:
                total_length = 0
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with accelerator.main_process_first():
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                batch_size=tokenizer_map_batch_size,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

        # <end elif: do tokenization>

    if dataset_setup == DatasetSetups.bookcorpus_and_wiki:
        train_dataset = concatenate_datasets(
            [tokenized_datasets["train_book"], tokenized_datasets["train_wiki"]]
        )
        eval_dataset = tokenized_datasets["validation"]
    else:
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]

    # Log a few random samples from the training set:
    if len(train_dataset) > 3:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.preprocessing_num_workers,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.preprocessing_num_workers,
    )

    # -----------------------------------------------------------------

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.

    if args.wd_LN_gamma:
        no_decay = ["bias"]
    else:
        no_decay = ["bias", "layer_norm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95)
    )  # <- as per OPT paper

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    w = args.num_warmup_steps / max(1.0, args.max_train_steps)
    eps = args.final_lr_fraction
    a = 1 / (1 - (1.0 - w) * eps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.num_warmup_steps * a),
        num_training_steps=int(args.max_train_steps * a),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("tb_logs", experiment_config)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last

        # Extract `checkpoint_{i}`
        training_difference = os.path.splitext(path)[0]

        # total number of completed optimizer steps (since the start of the training):
        completed_steps = int(training_difference.replace("checkpoint_", ""))
        logger.info(f"Resuming training from opt. step {completed_steps} ...")
        # total number of forward passes (since the start of the training):
        resume_step = completed_steps * args.gradient_accumulation_steps
        # compute starting epoch
        starting_epoch = resume_step // len(train_dataloader)
        # number of forward passes (since the start of the current epoch):
        resume_step -= starting_epoch * len(train_dataloader)
        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)

    if args.with_tracking and args.extra_tb_stats:
        act_dict = attach_tb_act_hooks(model)

    num_layers = len(model.model.decoder.layers)

    # ** Training loop **
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
            N_total_loss = len(train_dataloader)

        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == starting_epoch
                and resume_step is not None
                and step < resume_step
            ):
                N_total_loss -= 1
                del batch
                continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float().item()
                accelerator.backward(loss)

                # grad clipping
                if (
                    args.max_grad_norm is not None
                    and args.max_grad_norm > 0
                    and accelerator.sync_gradients
                ):
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        max_norm=args.max_grad_norm,
                        norm_type=args.grad_norm_type,
                    )

                optimizer.step()

                if not accelerator.optimizer_step_was_skipped:
                    # do not update LR if the grad update was skipped (because of overflow in grad
                    # computation cause by mixed-precision)
                    lr_scheduler.step()

                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                completed_steps += 1

                # update progress bar
                tqdm_update_interval = args.tqdm_update_interval
                if completed_steps % tqdm_update_interval == 0:
                    progress_bar.update(tqdm_update_interval)

                # log current LR
                accelerator.log(
                    {"learning_rate": lr_scheduler.get_last_lr()[0]}, step=completed_steps
                )

                # save model if needed
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)

                        # set save iteration to the number of steps completed
                        accelerator.project_configuration.iteration = completed_steps

                        # save states for model, optimizer, scheduler, scaler, RNG
                        accelerator.save_state(output_dir)

                # TB log scalars
                if (
                    args.with_tracking
                    and args.extra_tb_stats
                    and completed_steps % args.tb_scalar_log_interval == 0
                ):
                    # weights inf-norm
                    for name, module in model.named_modules():
                        if hasattr(module, "weight"):
                            w = module.weight
                            w_inf_norm = max(w.max().item(), -w.min().item())
                            accelerator.log(
                                {f"{name}.weight_inf_norm": w_inf_norm}, step=completed_steps
                            )

                    # act inf norm
                    for name, x in act_dict.items():
                        x_inf_norm = max(x.max().item(), -x.min().item())
                        accelerator.log({f"{name}.act_inf_norm": x_inf_norm}, step=completed_steps)

                # TB log histograms
                if (
                    args.with_tracking
                    and accelerator.is_main_process
                    and args.extra_tb_stats
                    and completed_steps % args.tb_hist_log_interval == 0
                ):
                    tb_writer = accelerator.trackers[0].writer

                    # weight histograms
                    for name, module in model.named_modules():
                        if hasattr(module, "weight"):
                            w = module.weight
                            try:
                                with warnings.catch_warnings():
                                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                                    tb_writer.add_histogram(
                                        f"{name}.weight_hist", w, global_step=completed_steps
                                    )
                            except:
                                logger.warn(
                                    f"Could not log weight histogram for {name} at step {completed_steps}"
                                )

                    # act histograms
                    for name, x in act_dict.items():
                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=DeprecationWarning)
                                tb_writer.add_histogram(
                                    f"{name}.act_hist", x, global_step=completed_steps
                                )
                        except:
                            logger.warn(
                                f"Could not log act histogram for {name} at step {completed_steps}"
                            )

            if completed_steps >= args.max_train_steps:
                break

        # ---------------------------------

        act_dict_eval = attach_act_hooks_for_eval(model)

        ACT_KEYS = [
            "model.decoder.final_layer_norm",
            *[f"model.decoder.layers.{j}" for j in range(num_layers)],
            *[f"model.decoder.layers.{j}.fc2" for j in range(num_layers)],
            *[f"model.decoder.layers.{j}.final_layer_norm" for j in range(num_layers)],
            *[f"model.decoder.layers.{j}.self_attn.out_proj" for j in range(num_layers)],
            *[f"model.decoder.layers.{j}.self_attn_layer_norm" for j in range(num_layers)],
        ]

        act_inf_norms = OrderedDict()
        act_kurtoses = OrderedDict()

        # ** Evaluation **
        model.eval()
        losses = []
        for batch_idx, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            loss_ = accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size))
            losses.append(loss_)

            # compute inf norms & kurtosis (>>>)
            if batch_idx <= 256:
                for name in ACT_KEYS:
                    if name in act_dict_eval:
                        x_inp, x_out = act_dict_eval[name]
                        x = x_out
                        x = x.view(x.size(0), -1)

                        # compute inf norm
                        inf_norms = x.norm(dim=1, p=np.inf)
                        if not name in act_inf_norms:
                            act_inf_norms[name] = AverageMeter()
                        for v in inf_norms:
                            act_inf_norms[name].update(v.item())

                        # compute kurtosis
                        kurt = kurtosis(x)
                        if not name in act_kurtoses:
                            act_kurtoses[name] = AverageMeter()
                        for v in kurt:
                            act_kurtoses[name].update(v.item())

            if batch_idx >= 1024:
                break

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            # log metrics
            log_metrics = {
                "perplexity": perplexity,
                "eval_loss": eval_loss,
                "epoch": epoch,
                "step": completed_steps,
            }
            if N_total_loss > 0 and total_loss > 0:
                log_metrics["train_loss"] = total_loss / N_total_loss
            accelerator.log(log_metrics, step=completed_steps)

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    # -----------

    # metrics
    metrics = OrderedDict([("perplexity", perplexity)])

    for name, v in act_inf_norms.items():
        metrics[name] = v.avg

    max_inf_norm = max(v.avg for v in act_inf_norms.values())
    max_ffn_inf_norm = max(v.avg for k, v in act_inf_norms.items() if ".fc" in k)
    max_layer_inf_norm = max(
        act_inf_norms[f"model.decoder.layers.{j}"].avg for j in range(num_layers)
    )

    avg_kurtosis = sum(v.avg for v in act_kurtoses.values()) / len(act_kurtoses.values())
    max_kurtosis = max(v.avg for v in act_kurtoses.values())
    max_kurtosis_layers = max(
        act_kurtoses[f"model.decoder.layers.{j}"].avg for j in range(num_layers)
    )

    metrics["max_inf_norm"] = max_inf_norm
    metrics["max_ffn_inf_norm"] = max_ffn_inf_norm
    metrics["max_layer_inf_norm"] = max_layer_inf_norm
    metrics["avg_kurtosis"] = avg_kurtosis
    metrics["max_kurtosis"] = max_kurtosis
    metrics["max_kurtosis_layers"] = max_kurtosis_layers

    logger.info(f"Max inf norm: {max_inf_norm:.1f}")
    logger.info(f"Max FFN inf norm: {max_ffn_inf_norm:.1f}")
    logger.info(f"Max layer inf norm: {max_layer_inf_norm:.1f}")
    logger.info(f"Avg Kurtosis: {avg_kurtosis:.2f}")
    logger.info(f"Max Kurtosis: {max_kurtosis:.1f}")
    logger.info(f"Max Kurtosis layers: {max_kurtosis_layers:.1f}")
    logger.info(f"\nAll metrics:\n{pformat(metrics)}")

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(metrics, f)


if __name__ == "__main__":
    main()
