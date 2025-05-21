import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('tweets.csv')

# Check for missing values and drop them
data = data.dropna(subset=['tweet_text'])

# Function to calculate sentiment
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply sentiment function
data['sentiment'] = data['tweet_text'].apply(get_sentiment)

# Categorize sentiment
data['sentiment_category'] = pd.cut(data['sentiment'], 
                                     bins=[-1, -0.1, 0.1, 1], 
                                     labels=['Negative', 'Neutral', 'Positive'])

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment_category', data=data, palette='viridis')
plt.title('Sentiment Distribution of Tweets')
plt.xlabel('Sentiment Category')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()

# Visualize sentiment over time (if you have a timestamp column)
# Assuming there's a 'timestamp' column in the dataset
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Resample to daily frequency and calculate mean sentiment
daily_sentiment = data['sentiment'].resample('D').mean()

plt.figure(figsize=(12, 6))
daily_sentiment.plot()
plt.title('Average Daily Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.grid()
plt.show()
