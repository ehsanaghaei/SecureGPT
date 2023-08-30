import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig, AdamW, TrainingArguments, Trainer, RobertaForCausalLM, LineByLineTextDataset
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaForMaskedLM, RobertaTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import logging
import random


class Config:
    model_name = "/users/eaghaei/Python_projects/RoBERTa/SecureBERT"
    tokenizer = "/users/eaghaei/Python_projects/RoBERTa/SecureBERT"
    train_data = "/users/eaghaei/Python_projects/RoBERTa/data/dataset_512_uncased_4.txt"

# logging.warning("Reading Training data")
# with open(Config.train_data, 'r') as file:
#     texts = file.readlines()
logging.warning("Done with reading training data")

logging.warning("Load tokenizer")
tokenizer = RobertaTokenizer.from_pretrained(Config.tokenizer)
logging.warning("Load model")
model = RobertaForCausalLM.from_pretrained(Config.model_name)

logging.warning("Create Dataset")
dataset = TextDataset(tokenizer, "/users/eaghaei/Python_projects/RoBERTa/data/dataset_512_uncased_4.txt", 512)
# dataset = TextDataset(tokenizer, "/media/ea/SSD2/Projects/CVE2TTP/data/sample_LM_data.txt", 512)
random.shuffle(dataset.examples)

logging.warning("Set up training args")
training_args = TrainingArguments(
    output_dir="./models/textgeneration",
    overwrite_output_dir=True,
    num_train_epochs=8,
    per_device_train_batch_size=48,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)
logging.warning("Set up data collator")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)



logging.warning("Set up trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Step 6: Train the RoBERTa text generation model
logging.warning("Model training")
trainer.train()
