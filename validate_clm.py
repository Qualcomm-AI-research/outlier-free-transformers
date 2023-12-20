#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
import json
import logging
import math
import os
import random
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from pprint import pformat

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datetime import datetime
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
)

from quantization.quantizers import QMethods
from quantization.range_estimators import OptMethod, RangeEstimators
from transformers_language.args import parse_args
from transformers_language.dataset_setups import DatasetSetups
from transformers_language.models.opt_attention import (
    AttentionGateType,
    OPTAttentionWithExtras,
)
from transformers_language.models.quantized_opt import QuantizedOPTForCausalLM
from transformers_language.models.softmax import SOFTMAX_MAPPING
from transformers_language.quant_configs import get_quant_config
from transformers_language.utils import (
    count_params,
    kurtosis,
    pass_data_for_range_estimation,
    val_qparams,
)

logger = logging.getLogger("validate_clm")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    args = parse_args()
    logger.info(args)

    # convert dataset setup to an enum
    dataset_setup = DatasetSetups[args.dataset_setup]

    # Initialize the accelerator. We will let the accelerator handle device placement for us in
    # this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up
    # all supported trackers in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir
        
        # MZ: Support WandB logging
        if args.report_to == 'wandb':
            import wandb
            wandb_run_name = args.config_name + '-' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            wandb.init(
                project=args.project_name,
                name=wandb_run_name,
                resume=args.resume_from_checkpoint,
                allow_val_change=True,
            )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs
    )

    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

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

    # Display config after changes
    logger.info("HuggingFace config after user changes:")
    logger.info(str(config))

    # Load tokenizer
    tokenizer_kwargs = {
        # 'cache_dir': args.model_cache_dir,
    }
    if args.model_name_or_path:
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
        new_attn = OPTAttentionWithExtras(
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
        )
        # copy loaded weights
        new_attn.load_state_dict(old_attn.state_dict(), strict=False)
        model.model.decoder.layers[layer_idx].self_attn = new_attn

    # Gating -> load the model again to load missing alpha
    if args.attn_gate_type != "none":
        state_dict = torch.load(str(Path(args.model_name_or_path) / "pytorch_model.bin"))
        new_state_dict = {}
        for name, val in state_dict.items():
            if "alpha" in name:
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict, strict=False)

    # Print model
    logger.info("Model:")
    logger.info(model)

    # Display num params
    n_embeddings = count_params(model.model.decoder.embed_tokens) + count_params(
        model.model.decoder.embed_positions
    )
    n_decoder = count_params(model.model.decoder) - n_embeddings
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

    # (data_setup, block_size, train_percentage, validation_percentage) -> train_dirname
    pre_tokenized_path_map = OrderedDict(
        [
            ((DatasetSetups.wikitext_103, 512, None, None), ("tokenized_wikitext_103_OPT_512")),
            (
                (DatasetSetups.wikitext_103, 512, None, 10),
                ("tokenized_wikitext_103_OPT_512_val_10%"),
            ),
            (
                (DatasetSetups.wikitext_103, 512, 10, None),
                ("tokenized_wikitext_103_OPT_512_train_10%"),
            ),
            (
                (DatasetSetups.bookcorpus_and_wiki, 512, 1, 5),
                ("tokenized_book_wiki_OPT_512_train_1%_val_5%"),
            ),
            (
                (DatasetSetups.bookcorpus_and_wiki, 512, 1, 1),
                ("tokenized_book_wiki_OPT_512_train_1%_val_1%"),
            ),
        ]
    )
    for k, v in pre_tokenized_path_map.items():
        pre_tokenized_path_map[k] = Path(args.data_cache_dir) / v

    tokenized_configuration = (
        dataset_setup,
        args.block_size,
        args.train_percentage,
        args.validation_percentage,
    )
    pre_tokenized_path = pre_tokenized_path_map.get(tokenized_configuration, None)

    if pre_tokenized_path is not None and pre_tokenized_path.exists():
        pre_tokenized_path = str(pre_tokenized_path)

        accelerator.print(f"Loading pre-tokenized dataset from {pre_tokenized_path}")
        tokenized_datasets = load_from_disk(pre_tokenized_path)

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
        batch_size=config.per_device_train_batch_size,
        num_workers=args.preprocessing_num_workers,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=config.per_device_eval_batch_size,
        num_workers=args.preprocessing_num_workers,
    )

    # Prepare everything with our `accelerator`.
    model, train_dataloader, eval_dataloader = accelerator.prepare(
        model, train_dataloader, eval_dataloader
    )

    logger.info("FP model:")
    logger.info(model)

    #
    ## Quantize:
    #
    if args.quantize:
        click_config = get_quant_config()

        # override number of batches
        click_config.act_quant.num_batches = args.est_num_batches
        click_config.quant.n_bits = args.n_bits
        click_config.quant.n_bits_act = args.n_bits_act
        click_config.quant.quant_setup = args.quant_setup
        if args.no_weight_quant:
            click_config.quant.weight_quant = False
        if args.no_act_quant:
            click_config.quant.act_quant = False

        # use MSE for weights (ignore `args.ranges_weights`)
        # click_config.quant.weight_quant_method = RangeEstimators.current_minmax
        click_config.quant.weight_quant_method = RangeEstimators.MSE
        click_config.quant.weight_opt_method = OptMethod.grid

        # qmethod acts
        if args.qmethod_acts == "symmetric_uniform":
            click_config.quant.qmethod_act = QMethods.symmetric_uniform
        elif args.qmethod_acts == "asymmetric_uniform":
            click_config.quant.qmethod_act = QMethods.asymmetric_uniform
        else:
            raise NotImplementedError(f"Unknown qmethod_act setting, '{args.qmethod_acts}'")

        # Acts ranges
        if args.percentile is not None:
            click_config.act_quant.options["percentile"] = args.percentile

        if args.ranges_acts == "running_minmax":
            click_config.act_quant.quant_method = RangeEstimators.running_minmax

        elif args.ranges_acts == "MSE":
            click_config.act_quant.quant_method = RangeEstimators.MSE
            if args.qmethod_acts == "symmetric_uniform":
                click_config.act_quant.options = dict(opt_method=OptMethod.grid)
            elif args.qmethod_acts == "asymmetric_uniform":
                click_config.act_quant.options = dict(opt_method=OptMethod.golden_section)

        elif args.ranges_acts.startswith("L"):
            click_config.act_quant.quant_method = RangeEstimators.Lp
            p_norm = float(args.ranges_acts.replace("L", ""))
            options = dict(p_norm=p_norm)
            if args.qmethod_acts == "symmetric_uniform":
                options["opt_method"] = OptMethod.grid
            elif args.qmethod_acts == "asymmetric_uniform":
                options["opt_method"] = OptMethod.golden_section
            click_config.act_quant.options = options

        else:
            raise NotImplementedError(f"Unknown range estimation setting, '{args.ranges_acts}'")

        qparams = val_qparams(click_config)
        qparams["quant_dict"] = {}

        model = QuantizedOPTForCausalLM(model, **qparams)
        model.set_quant_state(
            weight_quant=click_config.quant.weight_quant, act_quant=click_config.quant.act_quant
        )

        logger.info("Quantized model:")
        logger.info(model)

        # Range estimation
        logger.info("** Estimate quantization ranges on training data **")
        pass_data_for_range_estimation(
            loader=train_dataloader,
            model=model,
            act_quant=click_config.quant.act_quant,
            max_num_batches=click_config.act_quant.num_batches,
        )
        model.fix_ranges()

        model.set_quant_state(
            weight_quant=click_config.quant.weight_quant, act_quant=click_config.quant.act_quant
        )

    # attach hooks for activation stats
    def attach_act_hooks(model):
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

    if args.quantize:
        act_dict = {}
    else:
        act_dict = attach_act_hooks(model)
    num_layers = len(model.model.decoder.layers)

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

    # -----------------------------------------------------------------

    # *** Evaluation ***
    model.eval()
    losses = []
    for batch_idx, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        loss_ = accelerator.gather_for_metrics(loss.repeat(config.per_device_eval_batch_size))
        losses.append(loss_)

        # compute inf norms
        if not args.quantize:
            for name in ACT_KEYS:
                if name in act_dict:
                    x_inp, x_out = act_dict[name]
                    x = x_out
                    x = x.view(x.size(0), -1)

                    # compute inf norm
                    inf_norms = x.norm(dim=1, p=np.inf)
                    if not name in act_inf_norms:
                        act_inf_norms[name] = AverageMeter()
                    for v in inf_norms:
                        act_inf_norms[name].update(v.item())

                    # compute kurtosis
                    if batch_idx <= 100:
                        kurt = kurtosis(x)
                        if not name in act_kurtoses:
                            act_kurtoses[name] = AverageMeter()
                        for v in kurt:
                            act_kurtoses[name].update(v.item())

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    logger.info(f"perplexity: {perplexity:.4f}")

    # metrics
    metrics = OrderedDict([("perplexity", perplexity)])

    if not args.quantize:
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
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    main()
