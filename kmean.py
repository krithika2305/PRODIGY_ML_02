import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

def predict_customer_segment(gender, age, income, score):
    with open("scaler.pkl", "rb") as s_file:
        scaler = pickle.load(s_file)
    with open("kmeans_model.pkl", "rb") as m_file:
        model = pickle.load(m_file)

    new_data = np.array([[gender, age, income, score]])

    scaled_data = scaler.transform(new_data[:, 1:])  

    cluster = model.predict(scaled_data)[0]
    return cluster
print("\n🧾 Enter new customer details:")

gender_input = input("Gender (Male/Female): ").strip().capitalize()
gender = 0 if gender_input == "Male" else 1

age = int(input("Age: "))
income = float(input("Annual Income (k$): "))
score = float(input("Spending Score (1-100): "))

predicted_cluster = predict_customer_segment(gender, age, income, score)

print(f"\n📊 The customer belongs to Cluster: {predicted_cluster}")
cluster_labels = {
    0: "High income, low spending",
    1: "Low income, high spending",
    2: "High income, high spending",
    3: "Low income, low spending",
    4: "Moderate income and spending"
}
print(f"📌 Segment: {cluster_labels.get(predicted_cluster, 'Unknown')}")
