import matplotlib.pyplot as plt
def plt_temperature(df):
    # Assuming 'Temperature' is a column in the DataFrame
    plt.figure(figsize=(10, 6))
    plt.plot(df['Temperature'], label='Temperature', color='blue')
    plt.title('Temperature Trend')
    plt.xlabel('Day')
    plt.ylabel('Temperature')
    plt.grid(True)
    plt.savefig('outputs/temperature_trend.png')  # Save the plot as an image file
    plt.show()

def plt_aqi(df):
    # Assuming 'AQI' is a column in the DataFrame
    plt.figure(figsize=(10, 6))
    plt.plot(df['AQI'], label='Air Quality Index', color='red')
    plt.title('Air Quality Index Trend')
    plt.xlabel('Day')
    plt.ylabel('AQI')
    plt.grid(True)
    plt.savefig('outputs/aqi_trend.png')  # Save the plot as an image file
    plt.show()

def correlation_analysis(df):
    correlation = df.corr()
    print("\nCorrelation Matrix:\n", correlation)
    plt.figure(figsize=(12, 8))
    plt.colorbar(plt.imshow(correlation, cmap='coolwarm'), label='Correlation Coefficient')
    plt.imshow(correlation, cmap = 'coolwarm')
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.tight_layout()
    plt.savefig('outputs/correlation_matrix.png')  # Save the plot as an image file
    plt.show()  
