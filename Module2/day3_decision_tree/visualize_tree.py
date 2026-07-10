import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder


def visualize(csv_path='machine_data.csv', model_path='tree_model.pkl'):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    obj_cols = df.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].apply(lambda col: col.str.strip())

    encoder = LabelEncoder()
    df['Status'] = encoder.fit_transform(df['Status'])

    x = df[['Temperature', 'Vibration', 'Pressure', 'RPM']]

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    plt.figure(figsize=(12, 8))
    plot_tree(model, feature_names=x.columns, class_names=encoder.classes_, filled=True)
    plt.savefig('tree.png')
    plt.close()

    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
    min_temp = int(df['Temperature'].min())
    max_temp = int(df['Temperature'].max())

    # Aggregate into 5-degree bins for clearer counts on small datasets
    bin_step = 5
    start_bin = min_temp - (min_temp % bin_step)
    bins = list(range(start_bin, max_temp + bin_step, bin_step))

    temps = df['Temperature'].dropna().values
    counts, edges = np.histogram(temps, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]

    plt.figure()
    plt.bar(centers, counts, width=widths, edgecolor='black', align='center')
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature')
    plt.ylabel('Count')
    plt.xticks(bins)
    import matplotlib.ticker as mticker
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig('temperature_hist.png')
    plt.close()

    corr = df.corr()
    print(corr)

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=45)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_sns.png')
    plt.close()

if __name__ == '__main__':
    visualize()
