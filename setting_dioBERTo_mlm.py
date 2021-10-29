'''
Created on 13 oct. 2021

@author: jose-lopez
'''
import os
from datasets import load_dataset
from transformers import RobertaConfig
from transformers import RobertaTokenizer
from transformers import RobertaForMaskedLM
from pathlib import Path
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import setuptools
import simplejson as json

def tokenize_function(example):
    return tokenizer(example["text"], max_length=512, truncation=True)

# Setting the config for the model
config = {
"architectures": ["RobertaForMaskedLM"],
"model_type": "roberta",
"attention_probs_dropout_prob": 0.1,
"hidden_act": "gelu",
"hidden_dropout_prob": 0.3,
"hidden_size": 144,
"initializer_range": 0.02,
"num_attention_heads": 12,
"num_hidden_layers": 12,
"vocab_size": 52_000,
"intermediate_size": 256,
"max_position_embeddings": 512
}
with open('./model/model/' + "config.json", 'w') as fp:
    json.dump(config, fp)

tokenizer = RobertaTokenizer.from_pretrained('./model/model/', max_length=512,  truncation=True)

model = RobertaForMaskedLM(config=config)

os.environ["CUDA_LAUNCH_BLOCKING"]='1'  #Makes for easier debugging (just in case)

tokenizer.save_model('./model/model/')
