import os
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_FILE = "environment_data.csv"

st.set_page_config(page_title="Environmental Dashboard", layout="wide")

st.title("Environmental Dashboard")

@st.cache_data
def load_data(path=DATA_FILE):
    return pd.read_csv(path)

df = load_data()

st.sidebar.header("Filters")
day_min = int(df['Day'].min())
day_max = int(df['Day'].max())
days = st.sidebar.slider("Day range", min_value=day_min, max_value=day_max, value=(day_min, day_max))
df_filtered = df[(df['Day'] >= days[0]) & (df['Day'] <= days[1])]
# Outlier threshold control (multiples of std deviation)
threshold = st.sidebar.slider("Outlier threshold (× std)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)


st.subheader("Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Avg Temperature (°C)", f"{df_filtered['Temperature'].mean():.2f}", f"σ={df_filtered['Temperature'].std():.2f}")
col2.metric("Avg AQI", f"{df_filtered['AQI'].mean():.2f}", f"max={df_filtered['AQI'].max():.0f}")
col3.metric("Avg Humidity (%)", f"{df_filtered['Humidity'].mean():.2f}", f"σ={df_filtered['Humidity'].std():.2f}")

st.subheader("Data")
st.dataframe(df_filtered.reset_index(drop=True))

# Outlier detection based on threshold
st.subheader("Outlier Detection")
avg_temp = df_filtered['Temperature'].mean()
std_temp = df_filtered['Temperature'].std()
outliers = df_filtered[df_filtered['Temperature'] - avg_temp > threshold * std_temp]
st.write(f"Threshold: {threshold} × std ({std_temp:.2f}) — Avg temp: {avg_temp:.2f} °C")
if not outliers.empty:
    st.dataframe(outliers.reset_index(drop=True))
else:
    st.write("No outliers detected for the selected threshold and day range.")

st.subheader("Plots")
with st.container():
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=df_filtered, x='Day', y='Temperature', marker='o', ax=ax1)
    ax1.set_title('Temperature Trend')
    st.pyplot(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df_filtered, x='Day', y='AQI', palette='magma', ax=ax2)
    ax2.set_title('AQI Monitoring')
    st.pyplot(fig2)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=df_filtered, x='Day', y='Humidity', marker='s', ax=ax3)
    ax3.set_title('Humidity Trend')
    st.pyplot(fig3)
    plt.close(fig3)

st.subheader('Correlation Matrix')
fig4, ax4 = plt.subplots(figsize=(6, 5))
corr = df_filtered[['Temperature', 'Humidity', 'AQI']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax4)
ax4.set_title('Environmental Correlation')
st.pyplot(fig4)
plt.close(fig4)

st.markdown("---")
st.write("Generated with Streamlit — use the sidebar to filter days.")
