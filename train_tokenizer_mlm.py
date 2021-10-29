# -*- coding: utf-8 -*-

from tokenizers import ByteLevelBPETokenizer
from os import path
import json
import shutil
import os
from transformers import RobertaConfig
from transformers import RobertaTokenizer
from transformers import RobertaForMaskedLM
from pathlib import Path
import setuptools
import simplejson as json

if __name__ == '__main__':
    
            
    corpus = "dioBERTo/text/roberta.txt"

    lm_data_dir = "dioBERTo/text/train/"
    eval_data_dir = "dioBERTo/text/validation/"
    
    with open(corpus, 'r', encoding="utf8") as f:
        lines = f.readlines()    
        
    train_split = 0.97
    train_data_size = int(len(lines)*train_split)
    
    with open(os.path.join(lm_data_dir,'train.txt') , 'w') as f:
        for item in lines[:train_data_size]:
            f.write(item)
    
    with open(os.path.join(eval_data_dir,'eval.txt') , 'w') as f:
        for item in lines[train_data_size:]:
            f.write(item)    
    
    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=corpus, vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    
    model_path = "dioBERTo/model/"
    weights_dir =  model_path + "weights"
    
    if path.exists(model_path):
        shutil.rmtree('dioBERTo/model')

    Path(weights_dir).mkdir(parents=True)
    
    tokenizer.save_model(model_path)
    
    # Setting the config for the model
    config = {
	"architectures": ["RobertaForMaskedLM"],
 	 "attention_probs_dropout_prob": 0.1,
 	 "bos_token_id": 0,
 	 "eos_token_id": 2,
	 "hidden_act": "gelu",
	 "hidden_dropout_prob": 0.1,
	 "hidden_size": 768,
	 "initializer_range": 0.02,
	 "intermediate_size": 3072,
	 "layer_norm_eps": 1e-05,
 	 "max_position_embeddings": 514,
 	 "model_type": "roberta",
 	 "num_attention_heads": 12,
 	 "num_hidden_layers": 6,
 	 "pad_token_id": 1,
 	 "type_vocab_size": 1,
 	 "vocab_size": 52000
    }
    with open(model_path + "config.json", 'w') as fp:
        json.dump(config, fp)
