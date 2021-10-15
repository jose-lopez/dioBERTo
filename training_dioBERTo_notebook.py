'''
Created on 13 oct. 2021

@author: jose-lopez
'''
import os
from transformers import RobertaConfig
from transformers import RobertaTokenizer
from transformers import RobertaForMaskedLM
import setuptools
import json

if __name__ == '__main__':   

    lm_data_dir = "dioBERTo/text/train"
    
    train_path = os.path.join(lm_data_dir,"train.txt")
    eval_path = os.path.join(lm_data_dir,"eval.txt")
    
    tokenizer = RobertaTokenizer.from_pretrained('dioBERTo/models', max_length=512, truncation=True)
       
    config = {
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
    with open("dioBERTo/models/dioberto/config.json", 'w') as fp:
        json.dump(config, fp)
        
    model = RobertaForMaskedLM(config=config)
        
    #Setting environment variables
    os.environ["train_path"] = train_path
    os.environ["eval_path"] = eval_path
    os.environ["CUDA_LAUNCH_BLOCKING"]='1'  #Makes for easier debugging (just in case)
    weights_dir = "dioBERTo/models/dioberto/weights"    
    
    cmd = '''python /content/transformers/examples/run_language_modeling.py --output_dir {0}  \
    --model_type roberta \
    --mlm \
    --train_data_file {1} \
    --eval_data_file {2} \
    --config_name /content/dioBERTo/dioBERTo/models/dioberto/ \
    --tokenizer_name /content/dioBERTo/dioBERTo/models/ \
    --do_train \
    --line_by_line \
    --overwrite_output_dir \
    --do_eval \
    --block_size 256 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --save_total_limit 2 \
    --save_steps 2000 \
    --logging_steps 500 \
    --per_gpu_eval_batch_size 32 \
    --per_gpu_train_batch_size 32 \
    --evaluate_during_training \
    --seed 42 \
    '''.format(weights_dir, train_path, eval_path)
    
    status = os.system(cmd)
    
    print(status)
    
    
    
    
    
    
    
    

    