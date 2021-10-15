'''
Created on 5 oct. 2021

@author: jose-lopez
'''

from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
from os import path
import shutil

if __name__ == '__main__':    
    
    txt_files_dir = "dioBERTo/text"
    lm_data_dir = "dioBERTo/text/train"
    
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
    weights_dir =  model_path + "dioberto/weights"
        
    if path.exists(model_path):
        shutil.rmtree(model_path)
        
    Path(model_path).mkdir(parents=True)
    Path(weights_dir).mkdir(parents=True) 
        
    tokenizer.save(model_path, "dioberto")
    
    shutil.move("dioBERTo/model/dioberto-merges.txt", "dioBERTo/model/merges.txt")
    shutil.move("dioBERTo/model/dioberto-vocab.json", "dioBERTo/model/vocab.json")

    
    
    
    
    