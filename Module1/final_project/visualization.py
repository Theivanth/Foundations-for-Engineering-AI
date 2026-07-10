import matplotlib.pyplot as plt
from preprocessing import data

# Visualization functions
def plt_temperature(data):
    plt.plot(data['Date'], data['Temperature_C'], 'o-')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Trend')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plt_aqi(data):
    plt.plot(data['Date'], data['AQI'], 'o-')
    plt.xlabel('Date')
    plt.ylabel('Air Quality Index')
    plt.title('Air Quality Index Trend')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plt_humidity(data):
    plt.plot(data['Date'], data['Humidity_Percent'], 'o-')
    plt.xlabel('Date')
    plt.ylabel('Humidity (%)')
    plt.title('Humidity Trend')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()








