import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
with open("../models/best_model_pipeline.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

with open("../models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ----------------------------
# Title
# ----------------------------
st.title("📊 Customer Churn Prediction System")
st.markdown("Predict whether a customer is likely to **churn** or **stay** based on customer profile and behavior.")

st.markdown("---")

# ----------------------------
# Sidebar Info
# ----------------------------
st.sidebar.header("Project Info")
st.sidebar.write("""
This application predicts customer churn using a trained machine learning pipeline.

### Models Compared:
- Logistic Regression
- KNN
- Decision Tree
- Naive Bayes
- SVM
- Random Forest

### Output:
- Churn / Stay prediction
- Churn probability score
""")

# ----------------------------
# Input Layout
# ----------------------------
st.subheader("Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Income = st.number_input("Income", min_value=0, value=50000)
    SpendingScore = st.slider("Spending Score", 1, 100, 50)
    PurchaseAmount = st.number_input("Purchase Amount", min_value=0, value=1000)
    ProductCategory = st.selectbox("Product Category", ["Electronics", "Clothing", "Home", "Beauty", "Sports"])

with col2:
    PaymentMethod = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "UPI", "Cash", "Net Banking"])
    City = st.text_input("City", "Bangalore")
    State = st.text_input("State", "Karnataka")
    Country = st.text_input("Country", "India")
    IsActive = st.selectbox("Is Active", ["Yes", "No"])
    Returns = st.number_input("Returns", min_value=0, value=0)

with col3:
    DiscountUsed = st.number_input("Discount Used", min_value=0, value=0)
    ReviewScore = st.slider("Review Score", 1, 5, 3)
    Browser = st.selectbox("Browser", ["Chrome", "Firefox", "Safari", "Edge"])
    Device = st.selectbox("Device", ["Mobile", "Desktop", "Tablet"])
    SessionTime = st.number_input("Session Time", min_value=0, value=100)
    DaysSinceLastPurchase = st.number_input("Days Since Last Purchase", min_value=0, value=30)

# ----------------------------
# Create Input DataFrame
# ----------------------------
input_data = pd.DataFrame([{
    'Age': Age,
    'Gender': Gender,
    'Income': Income,
    'SpendingScore': SpendingScore,
    'PurchaseAmount': PurchaseAmount,
    'ProductCategory': ProductCategory,
    'PaymentMethod': PaymentMethod,
    'City': City,
    'State': State,
    'Country': Country,
    'IsActive': IsActive,
    'Returns': Returns,
    'DiscountUsed': DiscountUsed,
    'ReviewScore': ReviewScore,
    'Browser': Browser,
    'Device': Device,
    'SessionTime': SessionTime,
    'DaysSinceLastPurchase': DaysSinceLastPurchase
}])

input_data = input_data[feature_columns]

# ----------------------------
# Prediction
# ----------------------------
st.markdown("---")

if st.button("🔍 Predict Churn", use_container_width=True):
    prediction = model_pipeline.predict(input_data)[0]
    probability = model_pipeline.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ This customer is likely to CHURN")
    else:
        st.success("✅ This customer is likely to STAY")

    st.metric("Churn Probability", f"{probability*100:.2f}%")

    st.markdown("### Input Summary")
    st.dataframe(input_data)

    st.markdown("### Business Interpretation")
    if probability >= 0.75:
        st.warning("High churn risk. Immediate retention action is recommended.")
    elif probability >= 0.50:
        st.info("Moderate churn risk. Consider customer engagement strategies.")
    else:
        st.success("Low churn risk. Customer retention is currently stable.")
