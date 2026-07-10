import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_temperature(df):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(df["Day"], df["Temperature"], marker='o')
    plt.title("Temperature Trend")
    plt.xlabel("Day")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.savefig("outputs/temperature_trend.png")
    plt.close()

def plot_aqi(df):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(df["Day"], df["AQI"])
    plt.title("AQI Monitoring")
    plt.xlabel("Day")
    plt.ylabel("AQI")
    plt.savefig("outputs/aqi_monitoring.png")
    plt.close()

def plot_humidity(df):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(df["Day"], df["Humidity"], marker='s')
    plt.title("Humidity Trend")
    plt.xlabel("Day")
    plt.ylabel("Humidity (%)")
    plt.grid(True)
    plt.savefig("outputs/humidity_trend.png")
    plt.close()


    





