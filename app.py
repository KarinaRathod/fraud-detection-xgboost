import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from xgboost import XGBClassifier

# --- Page Config ---
st.set_page_config(page_title="Sentinel AI | Fraud Detection", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a modern look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .footer { position: fixed; bottom: 0; width: 100%; text-align: center; color: gray; font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- Data & Model Functions ---
@st.cache_data
def load_data():
    # Using a subset if the file is massive, or load full
    data = pd.read_csv("creditcard.csv")
    return data

@st.cache_resource
def train_pipeline(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = XGBClassifier(
        n_estimators=100, # Reduced for speed in demo
        max_depth=4,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    
    # Pre-calculate performance metrics
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, scaler, report, cm, X.columns

# Initialize
try:
    df = load_data()
    model, scaler, report, cm, feature_names = train_pipeline(df)
except FileNotFoundError:
    st.error("⚠️ 'creditcard.csv' not found. Please ensure the dataset is in the root directory.")
    st.stop()

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/shield-with-check.png", width=80)
    st.title("Sentinel AI")
    st.markdown("---")
    menu = st.radio("Navigation", ["Dashboard", "Real-time Prediction", "Model Health"])
    st.markdown("---")
    st.info("System Status: **Active**")

# --- Dashboard ---
if menu == "Dashboard":
    st.title("📊 Fraud Analytics Dashboard")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Scanned", f"{len(df):,}")
    m2.metric("Fraud Detected", len(df[df.Class==1]), delta="Imbalanced", delta_color="inverse")
    m3.metric("Precision", f"{report['1']['precision']:.2%}")
    m4.metric("Recall (Sensitivity)", f"{report['1']['recall']:.2%}")

    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Transaction Distribution")
        fig_pie = px.pie(df, names='Class', values='Amount', 
                         color='Class', color_discrete_map={0:'#2ecc71', 1:'#e74c3c'},
                         hole=0.4)
        fig_pie.update_layout(showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Transaction Volume vs Amount")
        # Sample for visualization performance
        fig_scatter = px.scatter(df.sample(2000), x="Time", y="Amount", color="Class",
                                 color_continuous_scale=['#2ecc71', '#e74c3c'], opacity=0.5)
    
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- Prediction Page ---
elif menu == "Real-time Prediction":
    st.title("🔍 Transaction Verification")
    
    st.markdown("""
    Use this panel to verify specific transactions. You can manually enter values or 
    **pull a random sample** from the database to see the AI in action.
    """)

    if st.button("🎲 Load Random Sample from Dataset"):
        sample = df.sample(1)
        st.session_state.inputs = sample.drop("Class", axis=1).values.flatten().tolist()
    
    if "inputs" not in st.session_state:
        st.session_state.inputs = [0.0] * 30

    with st.expander("Modify Transaction Features", expanded=True):
        cols = st.columns(4)
        updated_values = []
        for i, name in enumerate(feature_names):
            val = cols[i % 4].number_input(f"{name}", value=float(st.session_state.inputs[i]))
            updated_values.append(val)

    if st.button("Analyze Transaction", type="primary"):
        input_arr = np.array(updated_values).reshape(1, -1)
        scaled_input = scaler.transform(input_arr)
        
        prob = model.predict_proba(scaled_input)[0][1]
        pred = 1 if prob > 0.5 else 0

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if pred == 1:
                st.error("### 🚨 HIGH RISK")
                st.write(f"**Fraud Probability:** {prob:.2%}")
            else:
                st.success("### ✅ LOW RISK")
                st.write(f"**Fraud Probability:** {prob:.2%}")
        
        with res_col2:
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e74c3c" if pred == 1 else "#2ecc71"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d4efdf"},
                        {'range': [30, 70], 'color': "#fcf3cf"},
                        {'range': [70, 100], 'color': "#fadbd8"}]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

# --- Model Health ---
elif menu == "Model Health":
    st.title("📈 Model Performance Metrics")
    
    tab1, tab2 = st.tabs(["Confusion Matrix", "Feature Importance"])
    
    with tab1:
        fig_cm = px.imshow(cm, text_auto=True, 
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Legit', 'Fraud'], y=['Legit', 'Fraud'],
                           color_continuous_scale='Blues')
        st.plotly_chart(fig_cm)
        
    with tab2:
        # Get feature importance from XGBoost
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False).head(10)
        
        fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h',
                         title="Top 10 Predictors of Fraud",
                         color='Importance', color_continuous_scale='Reds')
        st.plotly_chart(fig_imp)

st.markdown('<div class="footer">Sentinel AI Fraud Engine v2.1 | Powered by XGBoost</div>', unsafe_allow_html=True)