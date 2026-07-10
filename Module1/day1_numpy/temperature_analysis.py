import numpy as np
temps = np.array([30,31,32,33,35,36,40,38,37,35])
temp_mean = np.mean(temps)
temp_max = np.max(temps)
temp_min = np.min(temps)
temp_std = np.std(temps)

print("The Environmental temperature data is:")
print("Mean:", temp_mean)
print("Max:", temp_max)
print("Min:", temp_min)
print("Standard Deviation:", round(temp_std, 2))

threshold_high = temp_mean + temp_std
print("The threshold for high temperature is:", round(threshold_high, 2), "degrees Celsius.")

threshold_low = temp_mean - temp_std
print("The threshold for low temperature is:", round(threshold_low, 2), "degrees Celsius.")

