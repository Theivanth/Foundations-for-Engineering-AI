import numpy as np

temps = np.array([34,30,31,32,33,35,36,40,38,37,35])
Mean = np.mean(temps)
Max = np.max(temps)
Min = np.min(temps)
std = np.std(temps)

print("The Environmental temperaturre data is:")
print("Mean:", Mean)
print("Max:", Max)
print("Min:", Min)
print("Standard Deviation:", round(std, 2))



