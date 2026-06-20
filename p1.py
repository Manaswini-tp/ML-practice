import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('housing.csv')
df.head()
df.describe().T
df.nunique()
df.isnull().sum()

df['total_bedrooms'].median()
df['total_bedrooms'].fillna(df['total_bedrooms'].median, inplace=True)

Numerical = df.select_dtypes(include=[np.number]).columns

for col in Numerical:
  plt.figure(figsize=(10,6))
  df[col].plot(kind='hist', bins=60, color='brown', edgecolor='black')
  plt.ylabel('Frequency')
  plt.show()

for col in Numerical:
  plt.figure(figsize=(10,6))
  sns.boxplot(x=df[col], color='pink')
  plt.show()
