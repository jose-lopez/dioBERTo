'''
Created on 5 oct. 2021

@author: jose-lopez
'''

from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
from os import path
import json

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
    
    if not path.exists(model_path):
        Path(weights_dir).mkdir(parents=True)
        
    # Saving the tokenizer files    
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
      "num_hidden_layers": 12,
      "pad_token_id": 1,
      "type_vocab_size": 1,
      "vocab_size": 50265
    }
    with open(model_path + "config.json", 'w') as fp:
        json.dump(config, fp)


    
    
    
        
    

    
    