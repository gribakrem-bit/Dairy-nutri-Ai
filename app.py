import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="DairyNutri AI",
    page_icon="🥛"
)

st.title("🥛 DairyNutri AI")
st.write("AI-based nutritional and quality analysis of dairy products.")

# Load dataset
df = pd.read_csv("data/dairy_products.csv")

features = [
    "Energy_kcal",
    "Fat_g",
    "Protein_g",
    "Carbohydrate_g",
    "Calcium_mg",
    "Sodium_mg",
    "Water_g"
]

X = df[features]
y = df["Category"]

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

st.header("🔬 Dairy Product Analysis")

st.write("Enter the nutritional composition:")

energy = st.number_input("Energy (kcal)", min_value=0.0, value=60.0)
fat = st.number_input("Fat (g)", min_value=0.0, value=3.0)
protein = st.number_input("Protein (g)", min_value=0.0, value=3.0)
carbohydrate = st.number_input(
    "Carbohydrates (g)",
    min_value=0.0,
    value=5.0
)
calcium = st.number_input("Calcium (mg)", min_value=0.0, value=120.0)
sodium = st.number_input("Sodium (mg)", min_value=0.0, value=50.0)
water = st.number_input("Water (g)", min_value=0.0, value=88.0)

if st.button("🔍 Analyze Product"):

    input_data = pd.DataFrame([[
        energy,
        fat,
        protein,
        carbohydrate,
        calcium,
        sodium,
        water
    ]], columns=features)

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted dairy category: **{prediction}**")

    st.subheader("Nutritional Profile")

    st.dataframe(input_data)
