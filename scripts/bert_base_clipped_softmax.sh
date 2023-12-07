#!/bin/bash
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_mlm.py \
--with_tracking \
--report_to tensorboard \
--extra_tb_stats \
--seed 1000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 4 \
--data_cache_dir ~/.hf_data \
--model_cache_dir ~/.hf_cache \
--model_type bert \
--tokenizer_name bert-base-uncased \
--max_seq_length 128 \
--mlm_probability 0.15 \
--learning_rate 0.0001 \
--lr_scheduler_type linear \
--max_train_steps 1000000 \
--num_warmup_steps 10000 \
--per_device_train_batch_size 256 \
--per_device_eval_batch_size 256 \
--gradient_accumulation_steps 1 \
--max_grad_norm 1.0 \
--weight_decay 0.01 \
--config_name bert-base-uncased \
--checkpointing_steps 100000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 100000 \
--attn_softmax "clipped(-.025:1)" \
--output_dir output
