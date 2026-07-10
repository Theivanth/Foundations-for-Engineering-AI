import matplotlib.pyplot as plt


def plot_status_distribution(df):
    """
    Display the AQI category distribution.
    """

    counts = df["Status"].value_counts()

    plt.bar(counts.index, counts.values)

    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("AQI Status Distribution")

    plt.show()