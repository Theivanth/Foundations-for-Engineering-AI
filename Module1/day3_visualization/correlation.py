import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_analysis(df, save_path="outputs/correlation_matrix.png"):
    corr_matrix = df[["Temperature", "Humidity", "AQI"]].corr()
    print("\nCorrelation Matrix:\n")
    print(corr_matrix)

    # Ensure output directory exists
    dirpath = os.path.dirname(save_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Environmental Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
