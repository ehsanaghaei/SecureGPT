# https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/language_modeling.ipynb#scrollTo=FX3tVGtTXKMp

import torch
import logging
from functions import group_texts, read2list
import torch
from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPT2LMHeadModel, AutoConfig
from transformers import GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pandas as pd
try:

    # get index of currently selected device
    logging.warning(torch.cuda.current_device()) # returns 0 in my case


    # get number of GPUs available
    logging.warning(torch.cuda.device_count()) # returns 1 in my case


    # get the name of the device
    logging.warning(torch.cuda.get_device_name(0)) 
except:
    pass

class Config:
    model_name = "gpt2-large"
    tokenizer = "gpt2-large"
    train_data = "/users/eaghaei/Python_projects/RoBERTa/data/dataset_512_uncased_4.txt"
    # train_data = "/users/eaghaei/Python_projects/TextGeneration/dataset_512_uncased_m.txt"
    epochs = 8
    context_length = 256
    batch = 12

#  tokenizer = GPT2TokenizerFast.from_pretrained(Config.tokenizer)


# tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, DataCollatorForLanguageModeling

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

def train(dataset, model, tokenizer, batch_size, epochs, device):
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator, num_workers=8, shuffle=True)
    
    for epoch in range(epochs):
        logging.warning(f"Epoch: {epoch+1}")
        for idx, batch in enumerate(dataloader):
            inputs = batch['input_ids'].to(device)
            outputs = model(inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if idx % 1000 == 0:
                logging.warning(f"Epoch: {epoch+1}, Batch {idx} Loss:  {loss.item()}")

        # Save model after each epoch
        torch.save(model.state_dict(), f"/users/eaghaei/Python_projects/TextGeneration/models/Securegpt2-large/gpt2_finetuned_epoch{epoch}.pth")
        try:
            model.save_pretrained(f"/users/eaghaei/Python_projects/TextGeneration/lib/models/Securegpt2-large/securegpt_epoch{epoch}")
        except:
            logging.warning("** Could not save the pre-trained model")

#         try:
#             model.save_pretrained(f"/users/eaghaei/Python_projects/TextGeneration/lib/models/Securegpt2/securegpt_epoch{epoch}")
# `       except:
#             logging.warning("** Could not save the pre-trained model")

# define parameters
filepath = Config.train_data
logging.warning(f"Load Tokenizer")
tokenizer = GPT2Tokenizer.from_pretrained(Config.tokenizer)
tokenizer.pad_token = tokenizer.eos_token
block_size = 512
logging.warning(f"Load Model")
model = GPT2LMHeadModel.from_pretrained(Config.model_name)
batch_size = Config.batch
epochs = Config.epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.warning(f"Model is running on {device}")

# create dataset
logging.warning(f"Create Dataset")
dataset = TextDataset(filepath, tokenizer, block_size)

# train model
logging.warning(f"Start Training")
train(dataset, model, tokenizer, batch_size, epochs, device)


from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Load the fine-tuned model
# model_path = './model_gpt2_finetuned_epoch1'  # Adjust the path and filename to match where your model is saved
# tokenizer = GPT2Tokenizer.from_pretrained(model_path)
# model = GPT2LMHeadModel.from_pretrained(model_path)

# # Define the text generation pipeline
# generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# # Use the model to generate text
# prompt = "Once upon a time"
# generated_text = generator(prompt, max_length=100, temperature=0.7)

# print(generated_text[0]['generated_text'])