#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
import json
import logging
import math
import os
import random
import warnings
from collections import OrderedDict
from itertools import chain
from pathlib import Path

import colored_traceback.auto # MZ: make error tracebacks colorful
import datasets
import torch
import transformers
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
)

from transformers_language.args import parse_args
from transformers_language.dataset_setups import DatasetSetups
from transformers_language.models.bert_attention import (
    AttentionGateType,
    BertSelfAttentionWithExtras,
)
from transformers_language.models.softmax import SOFTMAX_MAPPING
from transformers_language.utils import count_params

logger = get_logger("run_mlm")


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
        
        # MZ: Support WandB logging
        if args.report_to == 'wandb':
            import wandb
            wandb_run_name = args.config_name + '-' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            wandb.init(
                project=args.project_name,
                name=wandb_run_name,
                resume=args.resume_from_checkpoint,
                allow_val_change=True
            )
            os.environ['WANDB_LOG_MODEL'] = 'checkpoint' # upload model artifacts


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

    # Prepare HuggingFace config
    # In distributed training, the .from_pretrained methods guarantee that only one local process
    # can concurrently download model & vocab.
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
            accelerator.print(f"config: {key}={value}")
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
            "You are instantiating a new tokenizer from scratch. This is not supported by this "
            "script. You can do it from another script, save it, and load it from here, "
            "using --tokenizer_name."
        )

    # Load and prepare model
    if args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.model_cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    # >> replace Self-attention module with ours
    # NOTE: currently assumes BERT
    for layer_idx in range(len(model.bert.encoder.layer)):
        old_self = model.bert.encoder.layer[layer_idx].attention.self
        new_self = BertSelfAttentionWithExtras(
            config,
            softmax_fn=SOFTMAX_MAPPING[args.attn_softmax],
            alpha=args.alpha,
            skip_attn=args.skip_attn,
            attn_gate_type=AttentionGateType[args.attn_gate_type],
            attn_gate_init=args.attn_gate_init,
            attn_gate_mlp=args.attn_gate_mlp,
            attn_gate_mlp2=args.attn_gate_mlp2,
            attn_gate_linear_all_features=args.attn_gate_linear_all_features,
            fine_tuning=args.fine_tuning,
        )

        # copy loaded weights
        if args.model_name_or_path is not None:
            new_self.load_state_dict(old_self.state_dict(), strict=False)
        model.bert.encoder.layer[layer_idx].attention.self = new_self

    # Gating -> load the model again to load missing alpha
    if args.model_name_or_path is not None and args.attn_gate_type != "none":
        state_dict = torch.load(str(Path(args.model_name_or_path) / "pytorch_model.bin"))
        new_state_dict = {}
        for name, val in state_dict.items():
            if "alpha" in name:
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict, strict=False)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a
    # model from scratch on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]  # = vocab size
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Display num params
    n_embeddings = count_params(model.bert.embeddings)
    n_encoder = count_params(model.bert.encoder)
    n_head = count_params(model.cls)
    logger.info(
        f"\nNumber of parameters:\n"
        f"\t* Embeddings:\t{n_embeddings}\n"
        f"\t* Encoder:\t{n_encoder}\n"
        f"\t* Head:\t{n_head}\n"
        f"\t= Total (pre-training):\t{n_embeddings + n_encoder + n_head}\n"
        f"\t= Total (encoder):\t{n_embeddings + n_encoder}\n"
    )
    
    accelerator.print(f"{args.gradient_accumulation_steps} grad accum steps * {accelerator.state.num_processes} processes * {config.per_device_train_batch_size} batch size * {config.max_seq_length} max seq len")
    
    tokens_per_iter = (
        args.gradient_accumulation_steps * accelerator.state.num_processes * config.per_device_train_batch_size * config.max_seq_length
    )
    accelerator.print(f"tokens per iteration will be: {tokens_per_iter:,}")
    accelerator.print(
        f"breaks down as: {args.gradient_accumulation_steps} grad accum steps * {accelerator.state.num_processes} processes * {config.per_device_train_batch_size} batch size * {config.max_seq_length} max seq len"
    )

    # Get the datasets.
    # In distributed training, the load_dataset function guarantee that only one local process can
    # concurrently download the dataset.
    tokenized_book_wiki_path = (
        Path(args.data_cache_dir) / f"tokenized_book_wiki_{config.max_seq_length}"
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
        if config.max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
            if max_seq_length > 1024:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` "
                    f"({tokenizer.model_max_length}). Picking 1024 instead. You can change that "
                    f"default value by passing --max_seq_length xxx."
                )
                max_seq_length = 1024
        else:
            if config.max_seq_length > tokenizer.model_max_length:
                logger.warning(
                    f"The max_seq_length passed ({config.max_seq_length}) is larger than the maximum "
                    f"length for the model ({tokenizer.model_max_length}). Using "
                    f"max_seq_length={tokenizer.model_max_length}."
                )
            max_seq_length = min(config.max_seq_length, tokenizer.model_max_length)

        # Tokenize all the texts.
        # YB: removed line-by-line option as we'll likely never use it
        column_names = raw_datasets["validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        # ... we tokenize every text, then concatenate them together before splitting them in smaller
        # parts. We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling
        # (see below) is more efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

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
                desc="Running tokenizer on every text in dataset",
            )

        # Main data processing function that will concatenate all texts from our dataset and generate
        # chunks of max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of
            # this drop, you can customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts
        # throws away a remainder for each of those groups of 1,000 texts. You can adjust that
        # batch_size here but a higher value might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for
        # more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with accelerator.main_process_first():
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                batch_size=tokenizer_map_batch_size,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )
            
            
        def count_tokens(batch):
            return {'sum': [len(ids) for ids in batch]}
        
        # We can use .map to .reduce: https://github.com/huggingface/datasets/pull/5533#issuecomment-1440571658
        
        with accelerator.main_process_first():
            token_ds = tokenized_datasets.map(
                count_tokens,
                input_columns=['input_ids'],
                batched=True,
                num_proc=args.preprocessing_num_workers,
                keep_in_memory=True,
                desc=f'Count tokens'
            )
            print(token_ds.column_names)
            print(token_ds)
            print(token_ds.keys())
            print(token_ds.values())
            token_sum = sum(token_ds['sum'])
            accelerator.print(f"Total tokens: {token_sum}")
        
        if dataset_setup == DatasetSetups.bookcorpus_and_wiki:
            # Save the tokenizer's hard work
            tokenized_datasets.save_to_disk(str(tokenized_book_wiki_path))

        # <end elif: do tokenization>

    if dataset_setup == DatasetSetups.bookcorpus_and_wiki:
        train_dataset = concatenate_datasets(
            [tokenized_datasets["train_book"], tokenized_datasets["train_wiki"]]
        )
        eval_dataset = tokenized_datasets["validation"]
    else:
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]

    # Conditional for small test subsets
    if len(train_dataset) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=args.mlm_probability
    )

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config.per_device_train_batch_size,
        num_workers=args.preprocessing_num_workers,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=config.per_device_eval_batch_size,
        num_workers=args.preprocessing_num_workers,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    # LR Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may
    # have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        config.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

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
        accelerator.init_trackers(args.project_name, experiment_config)

    # Train!
    total_batch_size = (
        config.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = "
        f"{total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(config.max_train_steps), disable=not accelerator.is_local_main_process)
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
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

        # update the progress_bar if load from checkpoint
        progress_bar.update(starting_epoch * num_update_steps_per_epoch)
        completed_steps = starting_epoch * num_update_steps_per_epoch

    # attach hooks for activation stats (if needed)
    if args.with_tracking:
        act_dict = attach_tb_act_hooks(model)

    # store the value of the FFN magnitude (second to last layer)
    num_layers = len(model.bert.encoder.layer)
    ffn_inf_norm = None

    # ** Training loop **
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0

        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
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

            tqdm_update_interval = args.tqdm_update_interval
            if completed_steps % tqdm_update_interval == 0:
                progress_bar.update(tqdm_update_interval)

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            # TB log scalars
            if args.with_tracking and completed_steps % args.tb_scalar_log_interval == 0:
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

                # gate probs (if present)
                for layer_idx in range(len(model.bert.encoder.layer)):
                    self_attn_layer = model.bert.encoder.layer[layer_idx].attention.self
                    if self_attn_layer.last_gate_avg_prob is not None:
                        for head_idx in range(self_attn_layer.num_attention_heads):
                            gate_prob = self_attn_layer.last_gate_avg_prob[head_idx].item()
                            accelerator.log(
                                {f"layer{layer_idx}.head{head_idx}.avg_prob": gate_prob},
                                step=completed_steps,
                            )

            # TB log histograms
            if (
                args.with_tracking
                and accelerator.is_main_process
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

                # gate probs (if present)
                for layer_idx in range(len(model.bert.encoder.layer)):
                    self_attn_layer = model.bert.encoder.layer[layer_idx].attention.self
                    if self_attn_layer.last_gate_all_probs is not None:
                        for head_idx in range(self_attn_layer.num_attention_heads):
                            gate_prob_head = self_attn_layer.last_gate_all_probs[:, head_idx, ...]
                            try:
                                with warnings.catch_warnings():
                                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                                    tb_writer.add_histogram(
                                        f"layer{layer_idx}.head{head_idx}.probs",
                                        gate_prob_head,
                                        global_step=completed_steps,
                                    )
                            except:
                                logger.warn(
                                    f"Could not log act histogram for {name} at step {completed_steps}"
                                )

            if completed_steps >= config.max_train_steps:
                break

        # ** Evaluation **
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            loss_ = accelerator.gather_for_metrics(loss.repeat(config.per_device_eval_batch_size))
            losses.append(loss_)

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

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
                json.dump({"perplexity": perplexity, "ffn_inf_norm": ffn_inf_norm}, f)


if __name__ == "__main__":
    main()
