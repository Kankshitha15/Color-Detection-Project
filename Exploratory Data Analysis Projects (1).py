#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Explore the dataset
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize the distribution of numerical features
plt.figure(figsize=(12, 6))
df.hist(bins=20, figsize=(20, 15))
plt.suptitle('Histogram of Numerical Features')
plt.show()

# Visualize the correlation between features
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='YlOrRd')
plt.title('Correlation Heatmap')
plt.show()

# Identify outliers using box plots
plt.figure(figsize=(12, 6))
df.boxplot()
plt.title('Box Plots of Numerical Features')
plt.show()

# Explore the relationship between numerical and categorical features
plt.figure(figsize=(12, 6))
for col in df.select_dtypes(include=['object']).columns:
    sns.countplot(x=col, data=df)
    plt.title(f'Count Plot of {col}')
    plt.show()

# Perform feature engineering
df['new_feature'] = df['existing_feature1'] + df['existing_feature2']

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X = df.drop('target_variable', axis=1)
y = df['target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a machine learning model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model's performance
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))

