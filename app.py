# Streamlit App for Customer Segmentation with Interface
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler

# Page setup
st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("🛍️ Customer Segmentation using K-Means")

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

df = load_data()
st.subheader("🔍 Raw Data Preview")
st.dataframe(df.head())

# Preprocess
df_clean = df.copy()
df_clean['Gender'] = df_clean['Gender'].map({'Male': 0, 'Female': 1})
df_clean.drop('CustomerID', axis=1, inplace=True)
features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df_clean[features]

# Load scaler and model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Scale and predict clusters
X_scaled = scaler.transform(X.iloc[:, 1:])  # Exclude Gender if not scaled
df_clean['Cluster'] = kmeans.predict(X_scaled)

# Show clustered data
st.subheader("📊 Clustered Customer Data")
st.dataframe(df_clean)

# Cluster Visualization
st.subheader("🖼️ Cluster Visualization")
fig, ax = plt.subplots()
sns.scatterplot(data=df_clean, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster", palette="Set2", ax=ax)
plt.title("Customer Segments")
st.pyplot(fig)

# Interface to predict cluster of new customer
st.subheader("🧠 Predict Cluster for New Customer")

gender_input = st.selectbox("Gender", options=["Male", "Female"])
gender = 0 if gender_input == "Male" else 1
age = st.number_input("Age", min_value=1, max_value=100, value=30)
income = st.number_input("Annual Income (k$)", min_value=0.0, value=70.0)
score = st.number_input("Spending Score (1-100)", min_value=0.0, max_value=100.0, value=50.0)

if st.button("Predict Cluster"):
    new_data = np.array([[gender, age, income, score]])
    scaled_input = scaler.transform(new_data[:, 1:])  # Use only Age, Income, Score
    cluster_id = kmeans.predict(scaled_input)[0]

    st.success(f"✅ The customer belongs to **Cluster {cluster_id}**")

    # Optional: Add segment meaning
    segment_messages = {
        0: "🟡 Cluster 0: Average income and average spending. Average Customers.",
        1: "🟢 Cluster 1: High income, high spending. Premium Customers.",
        2: "🔵 Cluster 2: Low income, low spending. Budget Friendly Customers.",
        3: "🟠 Cluster 3: High income but low spending. Cautious Customers.",
        4: "🔴 Cluster 4: Low income but high spending. Impulsive or trend-driven Customers."
    }
    st.info(segment_messages.get(cluster_id, "Unknown segment"))
