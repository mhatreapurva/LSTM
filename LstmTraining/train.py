import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from LstmModel.lstm import LSTM
from LstmSummarizer.lstm_summarizer import Summarizer
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

class LstmTrainer:
    def __init__(
        self, tokenizer_obj, hidden_size=20, num_layers=5, num_epochs=10
    ) -> None:
        self.tokenizer = tokenizer_obj
        self.input_size = self.tokenizer.tokenizer.model_max_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = self.tokenizer.tokenizer.vocab_size
        self.model = Summarizer(
            lstm_input_size=self.input_size,
            lstm_hidden_size=self.hidden_size,
            lstm_num_layers=self.num_layers,
            summary_output_size=self.output_size,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset = self.tokenizer.generate_tokenized_dataset()['train']
        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, collate_fn=self.collate_fn)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.tokenizer.pad_token_id)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.num_epochs = num_epochs
    
    def collate_fn(self, batch):
        # Convert input_ids and labels to tensors if they are not already
        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item['labels'], dtype=torch.long) for item in batch]

        # Pad the sequences
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.tokenizer.pad_token_id)

        return {"input_ids": input_ids_padded, "labels": labels_padded}

    def train_lstm(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            loop = tqdm(self.train_loader, leave=True)
            breakpoint()
            for batch in loop:
                self.optimizer.zero_grad()
                # Extract inputs and labels from batch and move to device
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                # Forward pass
                outputs = self.model(input_ids)

                # Calculate loss and perform a backward pass
                loss = self.criterion(
                    outputs.view(-1, self.tokenizer.vocab_size), labels.view(-1)
                )
                loss.backward()
                self.optimizer.step()

                # Update progress bar
                loop.set_description(f"Epoch {epoch+1}/{self.num_epochs}")
                loop.set_postfix(loss=loss.item())
