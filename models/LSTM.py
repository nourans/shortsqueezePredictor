import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Data Preparation
np.random.seed(42)
torch.manual_seed(42)

# Simulated data
time = np.arange(0, 100)
sentiment = np.sin(time / 10) + np.random.normal(scale=0.1, size=100)  # Sentiment data
price = np.cumsum(np.random.choice([-1, 1], size=100) * (1 + 0.1 * sentiment))  # Stock price

# Combine data into features
data = np.stack([sentiment, price], axis=1)

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length, 1]  # Predicting stock price
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

SEQ_LENGTH = 10  # Number of time steps in each sequence
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # We only need the final hidden state
        hidden = hidden[-1]  # Get the output from the last LSTM layer
        output = self.fc(hidden)
        return output

# Hyperparameters
INPUT_SIZE = 2  # Sentiment and price
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1  # Predicting the next price
NUM_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 100

# Initialize model, loss function, and optimizer
model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# Denormalize predictions for interpretability
y_test_denorm = scaler.inverse_transform(
    np.hstack((np.zeros((y_test.shape[0], 1)), y_test.numpy()))
)[:, 1]
test_predictions_denorm = scaler.inverse_transform(
    np.hstack((np.zeros((test_predictions.shape[0], 1)), test_predictions.numpy()))
)[:, 1]

# Plot results
import matplotlib.pyplot as plt
plt.plot(y_test_denorm, label="True Prices")
plt.plot(test_predictions_denorm, label="Predicted Prices")
plt.legend()
plt.title("LSTM Stock Price Prediction")
plt.show()
