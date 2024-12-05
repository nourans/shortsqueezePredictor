import numpy as np
import pandas as pd
import json
import re
import torch
#from preprocessStocks import extractData() # maybe... for combining datasets for each stock in one total dataset
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset



# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

reddit_file_path = 'WSB_data/amc.csv'
reddit_data = pd.read_csv(reddit_file_path)

# CHECKPOINT: inspect the first few rows of the Reddit data
# print("Reddit Data Preview:")
# print(reddit_data.head())

# # CHECKPOINT: Check Reddit data structure
# print("\nReddit Data Info:")
# print(reddit_data.info())

# # CHECKPOINT: Check for missing values in Reddit data
# print("\nMissing Values in Reddit Data:")
# print(reddit_data.isnull().sum())

# Drop unnecessary columns from the Reddit data (e.g., usernames, URLs)
reddit_cleaned = reddit_data.drop(columns=['username', 'URL', 'body'], errors='ignore')

# Drop unnecessary columns from the Reddit data (e.g., usernames, URLs)
reddit_cleaned = reddit_data.drop(columns=['username', 'reddit_url', 'other_links'], errors='ignore')

# Filter for posts mentioning the stock (e.g., "AMC") in the title or selftext
reddit_cleaned = reddit_cleaned[reddit_cleaned['title'].str.contains(r'\bAMC\b', flags=re.IGNORECASE, regex=True, na=False)]

# Add sentiment analysis for each post (textblob or VADER)
# reddit_cleaned['sentiment'] = reddit_cleaned['title'].apply(lambda x: TextBlob(x).sentiment.polarity) # textblob
# VADER
reddit_cleaned['sentiment'] = reddit_cleaned['title'].apply(
    lambda x: analyzer.polarity_scores(x)['compound']  # Use 'compound' as overall sentiment score
)

# Convert the 'date' column to datetime format
reddit_cleaned['date'] = pd.to_datetime(reddit_cleaned['date'], errors='coerce')

# Check for invalid dates
if reddit_cleaned['date'].isnull().any():
    print("Warning: Some rows have invalid dates and will be dropped.")
    reddit_cleaned = reddit_cleaned.dropna(subset=['date'])

# Group by date and calculate daily metrics
reddit_grouped = reddit_cleaned.groupby(reddit_cleaned['date'].dt.date).agg(
    mention_count=('score', 'count'),    # Total number of posts (mention count)
    average_score=('score', 'mean'),     # Average score of posts per day
    average_sentiment=('sentiment', 'mean')  # Average sentiment
).reset_index()

# Convert date back to datetime format
reddit_grouped['date'] = pd.to_datetime(reddit_grouped['date'])

# # CHECKPOINT: Print so far
# print("\nCleaned Reddit Data Preview:")
# print(reddit_grouped.head(50))


################### Stocks #####################

stock_file_path = 'AMC_DataProcessed.json'
with open(stock_file_path, 'r') as json_file:
    stock_data = json.load(json_file)

# Convert the stock data to a pandas DataFrame
stock_df = pd.DataFrame(stock_data, columns=['date', 'price', 'volume'])

# Convert 'date' to datetime
stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')

# # CHECKPINT: Inspect the data
# print(stock_df.head())
# print(stock_df.info())

# Check for missing or invalid rows
if stock_df['date'].isnull().any():
    print("Warning: Some stock data rows have invalid dates and will be dropped.")
    stock_df = stock_df.dropna(subset=['date'])

# Ensure no duplicate dates
stock_df = stock_df.drop_duplicates(subset=['date'])

# # CHECKPOINT: Check for other anomalies 
# print(stock_df.describe())

# Ensure both stock_df and reddit_grouped are indexed by 'date'
stock_df['date'] = pd.to_datetime(stock_df['date'])
reddit_grouped['date'] = pd.to_datetime(reddit_grouped['date'])

###################### Merged #######################

# Merge Reddit and stock data
combined_data = pd.merge(reddit_grouped, stock_df, on='date', how='inner')

# # CHECKPOINT: Inspect the combined dataset
# print(combined_data.tail(50))

# STEP 5

# Calculate percentage change in price
combined_data['price_change_pct'] = combined_data['price'].pct_change()

# Calculate rolling average of volume (e.g., over 7 days)
combined_data['volume_avg_7d'] = combined_data['volume'].rolling(window=7, min_periods=1).mean()

# Define thresholds for short squeeze
price_spike_threshold = 0.2  # 20% price increase
volume_spike_threshold = 2  # Volume 2x the 7-day average

# Identify short squeezes
combined_data['squeeze_label'] = (
    (combined_data['price_change_pct'] > price_spike_threshold) &
    (combined_data['volume'] > volume_spike_threshold * combined_data['volume_avg_7d'])
).astype(int)

# Drop intermediate columns if not needed
combined_data = combined_data.drop(columns=['price_change_pct', 'volume_avg_7d'], errors='ignore')

# # CHECKPOINT: inspect the updated data
# print("\nData with Squeeze Labels:")
# print(combined_data.head(100))
# print("\n data BEFORE normalization\n")
# print(combined_data.describe())

############### normalization 3################
# Select features to normalize
features_to_normalize = ['mention_count', 'average_score', 'average_sentiment', 'price', 'volume']

# Z-Score normalization
scaler = StandardScaler()
normalized_features = scaler.fit_transform(combined_data[features_to_normalize])

# Create a new DataFrame with normalized features
normalized_data = pd.DataFrame(normalized_features, columns=features_to_normalize)

# Add the 'date' and 'squeeze_label' columns back to the DataFrame
normalized_data['date'] = combined_data['date']
normalized_data['squeeze_label'] = combined_data['squeeze_label']

# # CHECKPOINT: inspect mean and std of normalized features and then normalized data
# import numpy as np
# print("Means:", np.mean(normalized_features, axis=0))  # Should be close to 0
# print("Standard Deviations:", np.std(normalized_features, axis=0))  # Should be close to 1
# print("\nNormalized Data Preview:")
# print(normalized_data.head(5))


########### sequencing ###############
# Define sequence length
sequence_length = 30

# Select features and target
features = ['mention_count', 'average_score', 'average_sentiment', 'price', 'volume']
target = 'squeeze_label'

# Prepare the feature matrix and target vector
X = []  # Input sequences
y = []  # Corresponding targets

# Iterate over the dataset using a sliding window
for i in range(len(normalized_data) - sequence_length):
    # Extract the sequence of features
    X.append(normalized_data[features].iloc[i:i + sequence_length].values)
    
    # Extract the target for the next day
    y.append(normalized_data[target].iloc[i + sequence_length])

# Convert to NumPy arrays for machine learning models
X = np.array(X)
y = np.array(y)

# CHECKPOINT: inspect the shapes
print("Input shape (X):", X.shape)  # Should be (num_sequences, sequence_length, num_features)
print("Target shape (y):", y.shape)  # Should be (num_sequences,)


############# train-test split ##################

# Split data into training, validation, and test sets based on time
train_cutoff = int(len(X) * 0.7)  # 70% for training
val_cutoff = int(len(X) * 0.85)  # 15% for validation, 15% for testing

# Training set
X_train = X[:train_cutoff]
y_train = y[:train_cutoff]

# Validation set
X_val = X[train_cutoff:val_cutoff]
y_val = y[train_cutoff:val_cutoff]

# Test set
X_test = X[val_cutoff:]
y_test = y[val_cutoff:]

# # CHECKPOINT: inspect the shapes
# print("Training set shape (X_train, y_train):", X_train.shape, y_train.shape)
# print("Validation set shape (X_val, y_val):", X_val.shape, y_val.shape)
# print("Test set shape (X_test, y_test):", X_test.shape, y_test.shape)

############# tensor conversion ##################

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create TensorDatasets for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders for the model
batch_size = 32  # Define the batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# CHECKPOINT: ispect one batch
for X_batch, y_batch in train_loader:
    print("Batch shape (X, y):", X_batch.shape, y_batch.shape)
    break

