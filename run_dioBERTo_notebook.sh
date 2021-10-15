#!/usr/bin/env bash

git clone -b v2.5.0  https://github.com/huggingface/transformers.git

# train tokenizer and get roberta config.json
python3.9 ./train_tokenizer_notebook.py

# train lang model
python3.9 ./training_dioBERTo_notebook.py

# Visualizing the model's metrics
# tensorboard dev upload --logdir ./dioBERTo/model/weights/runs
