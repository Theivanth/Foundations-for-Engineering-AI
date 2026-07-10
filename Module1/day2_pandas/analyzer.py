import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('environment_data.csv')
print("Environmental Data:", df.describe())

df = df.fillna(df.mean())

danger = df[df['AQI'] > 100]
print("Days with AQI above 100:")
print(danger)

plt.bar(df['Day'], df['AQI'])
plt.xlabel('Day')
plt.ylabel('Air Quality Index (AQI)')
plt.title('Air Quality Index Over Time')
plt.show()

