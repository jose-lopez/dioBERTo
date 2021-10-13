#!/usr/bin/env bash

# train tokenizer and get roberta config.json
python3.9 ./train_tokenizer.py

# train lang model
python3.9 ./training_dioBERTo.py

# Visualizing the model's metrics
# tensorboard dev upload --logdir ./dioBERTo/model/weights/runs
