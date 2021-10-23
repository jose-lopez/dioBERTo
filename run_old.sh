#!/usr/bin/env bash

# train tokenizer and get roberta config.json
python3.9 ./train_tokenizer_roberta.py

# train lang model
python3.9 ./run_language_modeling.py \
    --output_dir ./dioBERTo/model/weights \
    --model_type RobertaForMaskedLM \
    --mlm \
    --train_data_files "./dioBERTo/text/train/*" \
    --eval_data_file "./dioBERTo/text/validation/eval.txt" \
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
    --logging_steps 500 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 100.0 \
    --per_gpu_eval_batch_size 64 \
    --per_gpu_train_batch_size 64 \
    --evaluation_strategy "steps" \
    --seed 21

# ls -l ./dioBERTo/model/weights/runs
# ls -l ./dioBERTo/model/weights/eval_results_lm.txt
# cp ./dioBERTo/model/weights/eval_results_lm.txt ./dioBERTo/model/weights/runs

# Visualizing the model's metrics
#tensorboard dev upload --logdir ./dioBERTo/model/weights/runs
