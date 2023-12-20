#!/bin/bash
accelerate launch --config_file accelerate_configs/1gpu_fp16.yaml run_clm.py \
--pad_to_max_length \
--wd_LN_gamma \
--with_tracking \
--report_to wandb \
--project_name quantizable_transformers \
--extra_tb_stats \
--seed 1000 \
--dataset_setup bookcorpus_and_wiki \
--preprocessing_num_workers 6 \
--data_cache_dir ~/.hf_data \
--model_cache_dir ~/.hf_cache \
--model_type opt \
--tokenizer_name facebook/opt-350m \
--max_seq_length 2048 \
--block_size 512 \
--learning_rate 0.0004 \
--lr_scheduler_type linear \
--max_train_steps 125000 \
--num_warmup_steps 2000 \
--per_device_train_batch_size 48 \
--per_device_eval_batch_size 48 \
--gradient_accumulation_steps 4 \
--max_grad_norm 1.0 \
--weight_decay 0.1 \
--checkpointing_steps 10000 \
--tb_scalar_log_interval 2000 \
--tb_hist_log_interval 10000 \
--config_path model_configs/opt-12L12H.yaml \
--alpha 12 \
--output_dir output
