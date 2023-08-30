import torch
import logging
# from Python_projects.TextGeneration.lib.CVE2CoA_functions import func_savejson
from functions import group_texts, read2list
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import random
from tokenizers import ByteLevelBPETokenizer
import os

def func_savejson(DICT, fname):
    import json
    with open(fname, 'w', encoding='"iso-8859-1"') as fout:
        json.dump(DICT, fout)
try:

    # get index of currently selected device
    logging.warning(torch.cuda.current_device()) # returns 0 in my case


    # get number of GPUs available
    logging.warning(torch.cuda.device_count()) # returns 1 in my case


    # get the name of the device
    logging.warning(torch.cuda.get_device_name(0)) 
except:
    pass
q="s"

class Config:
    if q=="s":
        model_name = "./gpt2/checkpoint-427500"
        batch = 8
    else:
        model_name = "gpt2-large"
        batch = 14
    # tokenizer = "./models/SecureGPTTokenizer"
    tokenizer = "gpt2"
    train_data = "./data/SecureBERT_Dataset_2023_2.txt"
    # train_data = "dataset_512_uncased_m.txt"
    epochs = 5
    shuffle = False
    train_tokenizer = False
    

if Config.shuffle:
    fname = Config.train_data
    logging.warning("Data shuffle is ON!\nReading the dataset to a list")
    with open(Config.train_data, "r") as f:
        data = f.readlines()
    data = [d for d in data if d not in [""," ","\n"]]

    logging.warning("Shuffling data")
    random.shuffle(data)
    logging.warning(f"Saving new data to {fname}")
    with open(fname, "w") as f:
        f.writelines(data)
    Config.train_data = fname
    del data

# if Config.train_tokenizer or not os.listdir(Config.tokenizer):
#     logging.warning("Training new tokenizer")
# #  tokenizer = GPT2TokenizerFast.from_pretrained(Config.tokenizer)
#     tokenizer = ByteLevelBPETokenizer()
#     # Customize training
#     data_chunk_addresses = [Config.train_data]
#     tokenizer.train(files=data_chunk_addresses, vocab_size=50257, show_progress=True, min_frequency=8,
#                     special_tokens=["<s>",
#                                     "<pad>",
#                                     "</s>",
#                                     "<sep>",
#                                     "<unk>",
#                                     "<mask>",
#                                     ])

#     tokenizer.save_model(Config.tokenizer)
#     del tokenizer
# else:
#     logging.warning("Chose to not train new tokenizer")

# tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import  DataCollatorForLanguageModeling
class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, block_size):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.file = open(filepath, 'r', encoding='utf-8')

    def __len__(self):
        self.file.seek(0)
        lines = 0
        buf_size = 1024 * 1024
        read_f = self.file.read
        buf = read_f(buf_size)
        while buf:
            lines += buf.count('\n')
            buf = read_f(buf_size)
        self.size = lines
        return self.size

    def __getitem__(self, i):
        line = self.file.readline()
        if not line:
            self.file.seek(0)
            line = self.file.readline()
        tokenized_text = self.tokenizer.encode(line, add_special_tokens=True, truncation=True, max_length=self.block_size)
        return {'input_ids': torch.tensor(tokenized_text)}
    

filepath = Config.train_data
logging.warning(f"Load Tokenizer")
tokenizer = GPT2Tokenizer.from_pretrained(Config.tokenizer)
tokenizer.pad_token = tokenizer.eos_token
block_size = 512
dataset = TextDataset(filepath, tokenizer, block_size)

logging.warning(f"Load Model")

model = GPT2LMHeadModel.from_pretrained(Config.model_name)
batch_size = Config.batch
epochs = Config.epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.warning(f"Model is running on {device}")

def print_training_loss(trainer, model, **kwargs):
    if trainer.state.global_step % 500 == 0:
        print(f"Step: {trainer.state.global_step} | Training loss: {trainer.state.metrics['loss']}")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,return_tensors="pt"
)
training_args = TrainingArguments(
    output_dir=f"./models/gpt2",
    overwrite_output_dir=True,
    num_train_epochs=Config.epochs,
    per_device_train_batch_size=Config.batch,
    weight_decay=0.01,
    logging_dir='./logs',
    save_steps=7500,
    save_total_limit=2,
    logging_steps=7500,
    learning_rate=5e-05,
    adam_beta2=0.99,
    # adam_epsilon=1e-08,
    warmup_steps=10000,
    resume_from_checkpoint=True

)

# create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    # callbacks=[print_training_loss]
)

# train model
trainer.train()
