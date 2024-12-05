import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow_addons.layers import TCN

# Load Stock Data
with open('alphaVanData_AMC_full_daily.json', 'r') as stock_file:
    stock_data = json.load(stock_file)

# Convert stock data to DataFrame
stock_df = pd.DataFrame.from_dict(stock_data["Time Series (Daily)"], orient="index")
stock_df = stock_df.sort_index()  # Ensure chronological order
stock_df = stock_df.astype(float)  # Convert to float for calculations
stock_df = stock_df[["4. close", "5. volume"]]  # Use Close and Volume

# Normalize Stock Data
stock_scaler = MinMaxScaler()
stock_df_scaled = stock_scaler.fit_transform(stock_df)

# Load Social Media Data
with open('redditData.json', 'r') as reddit_file:
    reddit_data = json.load(reddit_file)

# ____________________________________________________________________________
# Convert Reddit JSON to DataFrame
# FILL OUT: Replace with logic to handle your `redditData.json` format
reddit_df = pd.DataFrame(reddit_data)
# ____________________________________________________________________________


# Parse dates and preprocess Reddit Data
reddit_df["date"] = pd.to_datetime(reddit_df["date"])  # Ensure date format
reddit_df = reddit_df.sort_values("date")  # Sort by date
reddit_df["sentiment_score"] = reddit_df["upvotes"] * reddit_df["sentiment"]  # Example aggregation
social_features = reddit_df.groupby("date")[["mentions", "sentiment_score"]].sum()

# Merge Stock and Social Media Data
stock_df.index = pd.to_datetime(stock_df.index)  # Ensure stock dates are datetime
combined_df = pd.merge(stock_df, social_features, left_index=True, right_index=True, how="inner")

# Create Sequences for the Model
sequence_length = 30  # Number of historical days in each sequence
price_increase_threshold = 0.2  # 20% price increase threshold for short squeeze. WE SAID 50%
volume_increase_threshold = 2  # Volume doubling threshold for short squeeze
X, y = [], []

for i in range(len(combined_df) - sequence_length - 1):
    # Extract sequence
    X.append(combined_df.iloc[i:i + sequence_length].values)
    
    # Define target: Short squeeze based on price and volume thresholds
    current_price = combined_df.iloc[i + sequence_length - 1, 0]
    next_price = combined_df.iloc[i + sequence_length, 0]
    current_volume = combined_df.iloc[i + sequence_length - 1, 1]
    next_volume = combined_df.iloc[i + sequence_length, 1]
    
    price_increase = (next_price - current_price) / current_price
    volume_increase = next_volume / current_volume
    
    # Label: 1 if short squeeze, 0 otherwise
    y.append(1 if price_increase > price_increase_threshold and volume_increase > volume_increase_threshold else 0)

X, y = np.array(X), np.array(y)

# Split Data into Train/Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the TCN Model
model = Sequential([
    TCN(input_shape=(sequence_length, X.shape[2]), return_sequences=False),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification: short squeeze or not
])

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,  # FILL OUT: Adjust the number of epochs based on your resources
    batch_size=32,  # FILL OUT: Adjust batch size based on data size and memory
    verbose=1
)

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save the Model and Scalers
model.save('short_squeeze_tcn_model.h5')
import joblib
joblib.dump(stock_scaler, 'stock_data_scaler.pkl')

# Predict on Test Set
predictions = model.predict(X_test)

# Visualize Results (optional)
import matplotlib.pyplot as plt

# Training and Validation Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Predictions vs True Labels
plt.plot(y_test, label='True Labels')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()