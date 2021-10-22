#!/usr/bin/env bash

# train tokenizer and get roberta config.json
python3.9 ./train_tokenizer_roberta.py

# train lang model
python3.9 ./training_dioBERTo_roberta.py

# Visualizing the model's metrics
# tensorboard dev upload --logdir ./dioBERTo/model/weights/runs
