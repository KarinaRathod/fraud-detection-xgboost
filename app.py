import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier

st.set_page_config(page_title="AI Fraud Detection System", layout="wide")

# -----------------------------
# Train Model (Cached)
# -----------------------------
@st.cache_resource
def train_model():
    data = pd.read_csv("creditcard.csv")

    X = data.drop("Class", axis=1)
    y = data["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, accuracy, data, cm


model, scaler, accuracy, data, cm = train_model()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Fraud Prediction", "Model Performance"]
)

st.title("💳 AI Powered Credit Card Fraud Detection System")

# -----------------------------
# Dashboard Page
# -----------------------------
if option == "Dashboard":

    st.subheader("📊 Fraud Analytics Overview")

    col1, col2 = st.columns(2)

    total_transactions = len(data)
    fraud_cases = len(data[data["Class"] == 1])
    legit_cases = len(data[data["Class"] == 0])

    col1.metric("Total Transactions", total_transactions)
    col2.metric("Fraud Cases", fraud_cases)

    fig, ax = plt.subplots()
    sns.countplot(x=data["Class"])
    ax.set_xticklabels(["Legitimate", "Fraud"])
    ax.set_title("Fraud vs Legitimate Distribution")
    st.pyplot(fig)

# -----------------------------
# Prediction Page
# -----------------------------
elif option == "Fraud Prediction":

    st.subheader("🔍 Check Transaction")

    input_data = []
    for i in range(30):
        value = st.number_input(f"Feature {i+1}", value=0.0)
        input_data.append(value)

    if st.button("Predict Transaction"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction[0] == 1:
            st.error("🚨 Fraudulent Transaction Detected!")
            st.write(f"Fraud Probability: {probability:.2%}")
        else:
            st.success("✅ Legitimate Transaction")
            st.write(f"Fraud Probability: {probability:.2%}")

# -----------------------------
# Model Performance Page
# -----------------------------
elif option == "Model Performance":

    st.subheader("📈 Model Evaluation")

    st.metric("Model Accuracy", f"{accuracy:.4f}")

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix")
    st.pyplot(fig2)

    st.text("Classification Report:")
    st.text(classification_report(
        train_model()[3]["Class"], 
        train_model()[3]["Class"]
    ))
