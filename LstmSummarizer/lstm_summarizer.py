import torch.nn as nn
from LstmModel.lstm.LSTM import LSTM

# Assuming you have the LSTM and LSTMCell classes defined as in your code

class Summarizer(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, summary_output_size):
        super(Summarizer, self).__init__()
        self.lstm = LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers)
        self.linear = nn.Linear(lstm_hidden_size, summary_output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Assume x is a tensor of shape (batch_size, seq_length, input_size)
        h, c = self.lstm(x)

        # Use the final hidden state as the input for the summary generation
        summary = self.linear(h[-1])  # Assuming h[-1] is the final hidden state
        summary = self.softmax(summary)

        return summary
