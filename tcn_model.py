import json
import re

import numpy as np
import pandas as pd
from keras.src.layers import Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tcn import TCN


# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

################### GME Data #####################

# Load GME Reddit data
reddit_file_path = 'WSB_data/gamestop.csv'
reddit_data = pd.read_csv(reddit_file_path)

# Drop unnecessary columns from the Reddit data
reddit_cleaned = reddit_data.drop(columns=['username', 'URL', 'body'], errors='ignore')

# Filter for posts mentioning the stock (e.g., "GME") in the title or selftext
reddit_cleaned = reddit_cleaned[
    reddit_cleaned['title'].str.contains(r'\bGME\b', flags=re.IGNORECASE, regex=True, na=False)]

# Add sentiment analysis
reddit_cleaned['sentiment'] = reddit_cleaned['title'].apply(
    lambda x: analyzer.polarity_scores(x)['compound']  # Use 'compound' as overall sentiment score
)

# Convert the 'date' column to datetime format
reddit_cleaned['date'] = pd.to_datetime(reddit_cleaned['date'], errors='coerce')

# Drop rows with invalid dates
reddit_cleaned = reddit_cleaned.dropna(subset=['date'])

# Group by date and calculate daily metrics
reddit_grouped = reddit_cleaned.groupby(reddit_cleaned['date'].dt.date).agg(
    mention_count=('score', 'count'),  # Total number of posts (mention count)
    average_score=('score', 'mean'),  # Average score of posts per day
    average_sentiment=('sentiment', 'mean')  # Average sentiment
).reset_index()

# Convert date back to datetime format
reddit_grouped['date'] = pd.to_datetime(reddit_grouped['date'])

# Load GME stock data
stock_file_path = 'stock_data/GME_DataProcessed.json'
with open(stock_file_path, 'r') as json_file:
    stock_data = json.load(json_file)

stock_df = pd.DataFrame(stock_data, columns=['date', 'price', 'volume'])
stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
stock_df = stock_df.dropna(subset=['date']).drop_duplicates(subset=['date'])

# Merge GME Reddit and stock data
combined_data = pd.merge(reddit_grouped, stock_df, on='date', how='inner')

# Calculate percentage change in price
combined_data['price_change_pct'] = combined_data['price'].pct_change()

# Calculate rolling average of volume
combined_data['volume_avg_7d'] = combined_data['volume'].rolling(window=7, min_periods=1).mean()

# Define thresholds for short squeeze
price_spike_threshold = 0.2  # 20% price increase
volume_spike_threshold = 2  # Volume 2x the 7-day average

# Identify short squeezes
combined_data['squeeze_label'] = (
        (combined_data['price_change_pct'] > price_spike_threshold) &
        (combined_data['volume'] > volume_spike_threshold * combined_data['volume_avg_7d'])
).astype(int)

# Drop intermediate columns
combined_data = combined_data.drop(columns=['price_change_pct', 'volume_avg_7d'], errors='ignore')

# Normalize GME data
features_to_normalize = ['mention_count', 'average_score', 'average_sentiment', 'price', 'volume']
scaler = StandardScaler()
normalized_features = scaler.fit_transform(combined_data[features_to_normalize])
normalized_data = pd.DataFrame(normalized_features, columns=features_to_normalize)
normalized_data['date'] = combined_data['date']
normalized_data['squeeze_label'] = combined_data['squeeze_label']

# Prepare GME sequences
sequence_length = 30
features = ['mention_count', 'average_score', 'average_sentiment', 'price', 'volume']
target = 'squeeze_label'

X = []
y = []
for i in range(len(normalized_data) - sequence_length):
    X.append(normalized_data[features].iloc[i:i + sequence_length].values)
    y.append(normalized_data[target].iloc[i + sequence_length])

X = np.array(X)
y = np.array(y)

############# Train on GME Data ##################

# Use entire GME dataset for training
X_train, y_train = X, y

# Build TCN model
model = Sequential([
    TCN(input_shape=(sequence_length, X.shape[2]), return_sequences=False),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',  # Automatically balance classes
    classes=np.unique(y_train),  # Classes in the training set
    y=y_train  # Labels for the training set
)

# Convert to dictionary format for Keras
class_weights_dict = dict(enumerate(class_weights))

print("Class Weights:", class_weights_dict)

# Train the model with class weights
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    verbose=1,
    class_weight=class_weights_dict  # Use class weights here
)

################### AMC Data #####################

# Load AMC Reddit data
amc_reddit_file_path = 'WSB_data/amc.csv'
amc_reddit_data = pd.read_csv(amc_reddit_file_path)

# Preprocess AMC Reddit data
amc_reddit_cleaned = amc_reddit_data.drop(columns=['username', 'URL', 'body'], errors='ignore')
amc_reddit_cleaned = amc_reddit_cleaned[
    amc_reddit_cleaned['title'].str.contains(r'\bAMC\b', flags=re.IGNORECASE, regex=True, na=False)]
amc_reddit_cleaned['sentiment'] = amc_reddit_cleaned['title'].apply(
    lambda x: analyzer.polarity_scores(x)['compound']
)
amc_reddit_cleaned['date'] = pd.to_datetime(amc_reddit_cleaned['date'], errors='coerce')
amc_reddit_cleaned = amc_reddit_cleaned.dropna(subset=['date'])

amc_reddit_grouped = amc_reddit_cleaned.groupby(
    amc_reddit_cleaned['date'].dt.date
).agg(
    mention_count=('score', 'count'),
    average_score=('score', 'mean'),
    average_sentiment=('sentiment', 'mean')
).reset_index()

amc_reddit_grouped['date'] = pd.to_datetime(amc_reddit_grouped['date'])

# Load GameStop stock data
amc_stock_file_path = 'stock_data/AMC_DataProcessed.json'
with open(amc_stock_file_path, 'r') as json_file:
    gamestop_stock_data = json.load(json_file)

amc_stock_df = pd.DataFrame(gamestop_stock_data, columns=['date', 'price', 'volume'])
amc_stock_df['date'] = pd.to_datetime(amc_stock_df['date'], errors='coerce')
amc_stock_df = amc_stock_df.dropna(subset=['date']).drop_duplicates(subset=['date'])

# Merge AMC Reddit and stock data
amc_combined_data = pd.merge(amc_reddit_grouped, amc_stock_df, on='date', how='inner')

# Calculate percentage change and rolling average
amc_combined_data['price_change_pct'] = amc_combined_data['price'].pct_change()
amc_combined_data['volume_avg_7d'] = amc_combined_data['volume'].rolling(window=7, min_periods=1).mean()
amc_combined_data['squeeze_label'] = (
        (amc_combined_data['price_change_pct'] > price_spike_threshold) &
        (amc_combined_data['volume'] > volume_spike_threshold * amc_combined_data['volume_avg_7d'])
).astype(int)
amc_combined_data = amc_combined_data.drop(columns=['price_change_pct', 'volume_avg_7d'], errors='ignore')

# Normalize AMC data
amc_normalized_features = scaler.transform(amc_combined_data[features_to_normalize])
amc_normalized_data = pd.DataFrame(amc_normalized_features, columns=features_to_normalize)
amc_normalized_data['date'] = amc_combined_data['date']
amc_normalized_data['squeeze_label'] = amc_combined_data['squeeze_label']

# Prepare AMC sequences
X_amc = []
y_amc = []

for i in range(len(amc_normalized_data) - sequence_length):
    X_amc.append(amc_normalized_data[features].iloc[i:i + sequence_length].values)
    y_amc.append(amc_normalized_data[target].iloc[i + sequence_length])

X_amc = np.array(X_amc)
y_amc = np.array(y_amc)

################### Test on AMC Data #####################

# Predict and evaluate on AMC data
amc_predictions = model.predict(X_amc)
binary_amc_predictions = (amc_predictions > 0.3).astype(int)

amc_accuracy = np.mean(binary_amc_predictions.flatten() == y_amc)
print(f"GameStop Test Accuracy: {amc_accuracy}")

# Plot True vs Predicted Labels for AMC
plt.figure(figsize=(12, 6))
plt.plot(y_amc, label='True Labels', alpha=0.8)
plt.plot(binary_amc_predictions, label='Predicted Labels', alpha=0.8, linestyle='--')
plt.title('True vs Predicted Labels for AMC')
plt.xlabel('Test Data Points')
plt.ylabel('Labels')
plt.legend()
plt.grid(True)
plt.show()
