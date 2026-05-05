import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ML Prediction App", page_icon="🤖", layout="wide")

# ---------------- LOAD MODELS ----------------
titanic_model = joblib.load('../models/titanic_best_model.pkl')
breast_model = joblib.load('../models/breast_cancer_best_model.pkl')

# ---------------- TITLE ----------------
st.title("🤖 Classification Prediction App")
st.write("Predict using the **best trained model** for **Titanic** and **Breast Cancer** datasets.")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Best Model Information")
st.sidebar.write("### 🚢 Titanic")
st.sidebar.success("Best Model: SVM")
st.sidebar.info("Accuracy: 82.52%")

st.sidebar.write("### 🧬 Breast Cancer")
st.sidebar.success("Best Model: SVM")
st.sidebar.info("Accuracy: 98.25%")

# ---------------- DATASET SELECT ----------------
dataset = st.selectbox("📂 Select Dataset", ["Titanic", "Breast Cancer"])

# ==========================================================
# TITANIC SECTION
# ==========================================================
if dataset == "Titanic":
    st.header("🚢 Titanic Survival Prediction")
    st.write("Enter passenger details to predict whether the passenger survived or not.")

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
        sex = st.selectbox("Sex", ["Male", "Female"])
        age = st.slider("Age", 1, 80, 25)
        sibsp = st.slider("Siblings / Spouses Aboard", 0, 8, 0)

    with col2:
        parch = st.slider("Parents / Children Aboard", 0, 6, 0)
        fare = st.slider("Fare", 0.0, 600.0, 50.0)
        embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

    # Encoding
    sex_encoded = 1 if sex == "Male" else 0
    embarked_map = {"C": 0, "Q": 1, "S": 2}
    embarked_encoded = embarked_map[embarked]

    # Prepare input
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

    # Predict
    if st.button("🚀 Predict Titanic Survival"):
        prediction = titanic_model.predict(input_data)

        if prediction[0] == 1:
            st.success("✅ Passenger **Survived**")
            st.balloons()
        else:
            st.error("❌ Passenger **Did Not Survive**")

        # Show input summary
        st.subheader("📋 Input Summary")
        df_input = pd.DataFrame({
            "Feature": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
            "Value": [pclass, sex, age, sibsp, parch, fare, embarked]
        })
        st.dataframe(df_input, use_container_width=True)

# ==========================================================
# BREAST CANCER SECTION
# ==========================================================
else:
    st.header("🧬 Breast Cancer Prediction")
    st.write("Enter the medical feature values to predict whether the tumor is **Malignant** or **Benign**.")

    feature_names = [
        "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
        "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
        "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
        "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
        "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
        "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
    ]

    default_values = [
        14.0, 20.0, 90.0, 700.0, 0.10,
        0.10, 0.10, 0.05, 0.18, 0.06,
        0.40, 1.20, 2.80, 40.0, 0.007,
        0.02, 0.03, 0.01, 0.02, 0.003,
        16.0, 25.0, 105.0, 850.0, 0.13,
        0.25, 0.27, 0.11, 0.29, 0.08
    ]

    features = []
    cols = st.columns(3)

    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            value = st.number_input(feature, value=float(default_values[i]))
            features.append(value)

    input_data = np.array([features])

    # Predict
    if st.button("🚀 Predict Breast Cancer"):
        prediction = breast_model.predict(input_data)

        # sklearn breast cancer:
        # 0 = malignant, 1 = benign
        if prediction[0] == 0:
            st.error("⚠️ Prediction: **Malignant**")
        else:
            st.success("✅ Prediction: **Benign**")
            st.balloons()

        # Show input summary
        st.subheader("📋 Input Summary")
        df_input = pd.DataFrame({
            "Feature": feature_names,
            "Value": features
        })
        st.dataframe(df_input, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Made with ❤️ using Streamlit + Scikit-learn")