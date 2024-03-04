import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from LstmTraining.download_dataset import DownloadDataset
from torch.nn.utils.rnn import pad_sequence
DATASET_NAME = "billsum"
TOKENIZER = "bert-base-uncased"


class LstmTokenizer:
    def __init__(
        self,
        dataset=DownloadDataset(DATASET_NAME).download_dataset(),
        tokenizer=AutoTokenizer.from_pretrained(TOKENIZER),
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def tokenize_function(self, raw_input):
        model_inputs = self.tokenizer(
            raw_input["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                raw_input["summary"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
        model_inputs["labels"] = labels["input_ids"].squeeze() 
        #model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def generate_tokenized_dataset(self):
        tokenized_datasets = self.dataset.map(self.tokenize_function, batched=True)
        return tokenized_datasets  # returns tokenized data
