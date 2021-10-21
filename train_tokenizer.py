'''
Created on 5 oct. 2021

@author: jose-lopez
'''

from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
from os import path
import json
import shutil

if __name__ == '__main__':    
    
    txt_files_dir = "dioBERTo/text"
    
    paths = [str(x) for x in Path(txt_files_dir).glob("**/*.txt")]
    print(paths)

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
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
    with open(model_path + "config.json", 'w') as fp:
        json.dump(config, fp)


    
    
    
        
    

    
    
