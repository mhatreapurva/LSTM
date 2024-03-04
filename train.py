import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import YourDatasetClass  # Define your dataset class
from LstmModel import LSTM, LSTMCell  # Your LSTM model

# Define hyperparameters
input_size =  # Your input size
hidden_size =  # Your hidden size
num_layers =  # Your number of layers
learning_rate = 0.001
num_epochs =  # Your number of epochs
batch_size =  # Your batch size

# Create dataset and dataloader
dataset = YourDatasetClass()  # Initialize your dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = LSTM(input_size, hidden_size, num_layers)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_data, target = batch  # Adjust this according to your dataset structure
        input_data, target = input_data.float(), target.float()  # Convert to float if needed
        h, c = model(input_data)
        loss = criterion(h, target)  # Adjust this according to your loss function
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss}")

# Save the trained model
torch.save(model.state_dict(), "lstm_model.pth")
