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

def tokenize_function(example):
    return tokenizer(example["text"], max_length=512, truncation=True)

config = RobertaConfig(
 vocab_size=52_000,
 max_position_embeddings=512,
 num_attention_heads=12,
 num_hidden_layers=12,
 type_vocab_size=1,
)

tokenizer = RobertaTokenizer.from_pretrained('./dioBERTo/model/', max_length=512,  truncation=True)

model = RobertaForMaskedLM(config=config)

train_path="./dioBERTo/text/train"
validation_path="./dioBERTo/text/validation"
test_path="./dioBERTo/text/test"

train_files = [str(x) for x in Path(train_path).glob("**/*.txt")]
validation_files = [str(x) for x in Path(validation_path).glob("**/*.txt")]
test_files = [str(x) for x in Path(test_path).glob("**/*.txt")]

raw_datasets = load_dataset('text', data_files={'train': train_files,'test': test_files, 'validation': validation_files })
print(raw_datasets)

raw_train_dataset = raw_datasets['test']
print(raw_train_dataset[3])

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

weigths_dir = './dioBERTo/model/weigths'
if not os.path.exists(weigths_dir):
    os.makedirs(weigths_dir)
    
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
print(tokenized_datasets)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
 output_dir='./dioBERTo/model/weigths',
 overwrite_output_dir=True,
 num_train_epochs=1,
 per_device_train_batch_size=4,
 save_steps=10_000,
 save_total_limit=2
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["test"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
    
os.environ["CUDA_LAUNCH_BLOCKING"]='1'  #Makes for easier debugging (just in case)

trainer.train()
    