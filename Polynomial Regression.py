import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import KFold

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

################### Load and Process Data #####################

# Load GME Reddit data
reddit_file_path = 'WSB_data/gamestop.csv'
reddit_data = pd.read_csv(reddit_file_path)

# Filter for posts mentioning GME and calculate sentiment
reddit_cleaned = reddit_data[
    reddit_data['title'].str.contains(r'\bGME\b', flags=re.IGNORECASE, regex=True, na=False)]
reddit_cleaned['sentiment'] = reddit_cleaned['title'].apply(
    lambda x: analyzer.polarity_scores(x)['compound']
)
reddit_cleaned['date'] = pd.to_datetime(reddit_cleaned['date'], errors='coerce')
reddit_cleaned = reddit_cleaned.dropna(subset=['date'])

# Group by date
reddit_grouped = reddit_cleaned.groupby(reddit_cleaned['date'].dt.date).agg(
    mention_count=('score', 'count'),
    average_score=('score', 'mean'),
    average_sentiment=('sentiment', 'mean')
).reset_index()
reddit_grouped['date'] = pd.to_datetime(reddit_grouped['date'])

# Load and process GME stock data
stock_file_path = 'stock_data/GME_DataProcessed.json'
with open(stock_file_path, 'r') as json_file:
    stock_data = json.load(json_file)

stock_df = pd.DataFrame(stock_data, columns=['date', 'price', 'volume'])
stock_df['date'] = pd.to_datetime(stock_df['date'], errors='coerce')
stock_df = stock_df.dropna(subset=['date']).drop_duplicates(subset=['date'])

# Merge GME Reddit and stock data
combined_data = pd.merge(reddit_grouped, stock_df, on='date', how='inner')

# Normalize features
features_to_normalize = ['mention_count', 'average_score', 'average_sentiment', 'price', 'volume']
scaler = StandardScaler()
normalized_features = scaler.fit_transform(combined_data[features_to_normalize])
normalized_data = pd.DataFrame(normalized_features, columns=features_to_normalize)
normalized_data['date'] = combined_data['date']
normalized_data['squeeze_label'] = (
        (combined_data['price'].pct_change() > 0.2) &
        (combined_data['volume'] > 2 * combined_data['volume'].rolling(window=7, min_periods=1).mean())
).astype(int)

################### Aggregate Data #####################

sequence_length = 30
features = ['mention_count', 'average_score', 'average_sentiment', 'price', 'volume']
target = 'squeeze_label'

aggregated_features = []
targets = []

for i in range(len(normalized_data) - sequence_length):
    sequence = normalized_data[features].iloc[i:i + sequence_length]
    aggregated_features.append({
        'mention_mean': sequence['mention_count'].mean(),
        'mention_std': sequence['mention_count'].std(),
        'score_mean': sequence['average_score'].mean(),
        'sentiment_mean': sequence['average_sentiment'].mean(),
        'price_mean': sequence['price'].mean(),
        'volume_mean': sequence['volume'].mean()
    })
    targets.append(normalized_data[target].iloc[i + sequence_length])

aggregated_df = pd.DataFrame(aggregated_features)
y = np.array(targets)

################### Find Optimal Degree Using Cross-Validation #####################


# Train-test split
x_train, x_test, y_train, y_test = train_test_split(aggregated_df, y, test_size=0.2, random_state=42)

# Cross-validation to find the best degree
best_degree = None
best_score = -np.inf
results = {}

for degree in range(1, 6):  # Test polynomial degrees from 1 to 5
    pipeline = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, x_train, y_train, scoring='r2', cv=kf)
    mean_score = cv_scores.mean()
    results[degree] = mean_score

    if mean_score > best_score:
        best_score = mean_score
        best_degree = degree

# Display cross-validation results
print("Cross-Validation Results:")
for degree, score in results.items():
    print(f"Degree {degree}: Mean R2 = {score:.4f}")

print(f"\nBest Polynomial Degree: {best_degree} with Mean R2 = {best_score:.4f}")

################### Train Polynomial Regression with Best Degree #####################

pipeline = make_pipeline(PolynomialFeatures(best_degree), LinearRegression())
pipeline.fit(x_train, y_train)

# Predict on test data
y_pred = pipeline.predict(x_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
print(f"R-squared on test data: {r2}")

################### Visualize Results #####################

# Plot one feature vs. predictions for simplicity
plt.scatter(x_test['price_mean'], y_test, label='Actual', color='blue')
plt.scatter(x_test['price_mean'], y_pred, label='Predicted', color='red')
plt.xlabel('Price Mean')
plt.ylabel('Target')
plt.title(f"Polynomial Regression Results (Degree {best_degree})")
plt.legend()
plt.show()
