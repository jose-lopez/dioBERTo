#!/usr/bin/env bash

# train tokenizer and get roberta config.json
python3.9 ./src/dioberto/train_tokenizer.py

# train lang model
python3.9 ./src/dioberto/run_language_modeling.py \
    --output_dir ./dioBERTo/model/dioberto-v1 \
    --model_type RobertaForMaskedLM \
    --mlm \
    --train_data_files "./dioBERTo/text/train/*" \
    --eval_data_file "./dioBERTo/text/validation/TLG5014.TXT-002.txt" \
    --config_name "./dioBERTo/model/" \
    --tokenizer_name "./dioBERTo/model/" \
    --do_train \
    --line_by_line \
    --overwrite_output_dir \
    --do_eval \
    --block_size 256 \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --save_total_limit 2 \
    --save_steps 2000 \
    --per_gpu_eval_batch_size 32 \
    --per_gpu_train_batch_size 32 \
    --seed 21
