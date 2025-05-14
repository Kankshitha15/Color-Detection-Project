#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the dataset
df = pd.read_csv('twitter_data.csv')

# Perform sentiment analysis using VADER
analyzer = SentimentIntensityAnalyzer()
df['sentiment_scores'] = df['text'].apply(lambda x: analyzer.polarity_scores(x))
df['compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])
df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x >= 0 else 'negative')

# Visualize sentiment distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Explore sentiment over time
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.strftime('%b')
plt.figure(figsize=(12, 6))
sns.countplot(x='month', hue='sentiment', data=df)
plt.title('Sentiment Over Time')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.show()

# Analyze sentiment by hashtags
plt.figure(figsize=(12, 6))
df.groupby('hashtags')['sentiment'].value_counts().unstack().plot(kind='bar')
plt.title('Sentiment by Hashtags')
plt.xlabel('Hashtags')
plt.ylabel('Count')
plt.show()

# Identify top positive and negative tweets
top_positive = df.nlargest(5, 'compound')
top_negative = df.nsmallest(5, 'compound')

print('Top 5 Positive Tweets:')
for i, row in top_positive.iterrows():
    print(f"Text: {row['text']}\nSentiment Score: {row['compound']:.2f}")
    print('-' * 50)

print('\nTop 5 Negative Tweets:')
for i, row in top_negative.iterrows():
    print(f"Text: {row['text']}\nSentiment Score: {row['compound']:.2f}")
    print('-' * 50)

