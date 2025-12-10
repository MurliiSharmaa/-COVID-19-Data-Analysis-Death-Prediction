import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ------------------------------
# Load Data & Model
# ------------------------------
df = pd.read_csv("Latest Covid-19 India Status.csv")

model = joblib.load("covid_deaths_prediction_model.pkl")
le = joblib.load("state_label_encoder.pkl")

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(
    page_title="COVID-19 Dashboard & Predictor",
    page_icon="🦠",
    layout="wide"
)

# Custom CSS for Beautiful UI
st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .title {
            text-align:center;
            color:#ff4b4b;
            font-size:40px;
            font-weight:700;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<p class='title'>🦠 COVID-19 Analytics Dashboard + Prediction</p>", unsafe_allow_html=True)
st.markdown("### A Complete Data Analytics + Machine Learning Based Application")

# ------------------------------
# Sidebar Filters
# ------------------------------
st.sidebar.header("📌 Filters")

state_filter = st.sidebar.selectbox("Select State", ["All"] + list(df["State/UTs"].unique()))
min_cases = st.sidebar.slider("Minimum Total Cases", 0, int(df["Total Cases"].max()), 0)

# Apply Filters
filtered_df = df.copy()
if state_filter != "All":
    filtered_df = filtered_df[filtered_df["State/UTs"] == state_filter]

filtered_df = filtered_df[filtered_df["Total Cases"] >= min_cases]

# ------------------------------
# KPI Cards
# ------------------------------
st.subheader("📊 Key Insights")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Cases", f"{filtered_df['Total Cases'].sum():,}")
col2.metric("Active Cases", f"{filtered_df['Active'].sum():,}")
col3.metric("Total Deaths", f"{filtered_df['Deaths'].sum():,}")
col4.metric("Total Discharged", f"{filtered_df['Discharged'].sum():,}")

# ------------------------------
# Charts Section
# ------------------------------

st.subheader("📈 Data Visualizations")

colA, colB = st.columns(2)

# 1. Total Cases Bar Chart
with colA:
    fig1 = px.bar(
        filtered_df,
        x="State/UTs",
        y="Total Cases",
        title="Total Cases by State",
        color="Total Cases"
    )
    st.plotly_chart(fig1, use_container_width=True)

# 2. Death Ratio Pie Chart
with colB:
    fig2 = px.pie(
        filtered_df,
        names="State/UTs",
        values="Deaths",
        title="Deaths Distribution by State"
    )
    st.plotly_chart(fig2, use_container_width=True)

# 3. Line Chart – Active vs. Discharged
st.subheader("📉 Active vs Discharged Trend")
fig3 = px.line(
    filtered_df,
    x="State/UTs",
    y=["Active", "Discharged"],
    markers=True,
    title="Active vs Discharged Cases"
)
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------
# Prediction Section
# ------------------------------

st.header("🔮 COVID-19 Death Prediction (Machine Learning)")

state = st.selectbox("State / UT", le.classes_)
total_cases = st.number_input("Total Cases", min_value=0.0)
active = st.number_input("Active Cases", min_value=0.0)
discharged = st.number_input("Discharged", min_value=0.0)
active_ratio = st.number_input("Active Ratio (%)", min_value=0.0)
discharge_ratio = st.number_input("Discharge Ratio (%)", min_value=0.0)
death_ratio = st.number_input("Death Ratio (%)", min_value=0.0)
population = st.number_input("Population", min_value=0.0)

if st.button("🔍 Predict Deaths", use_container_width=True):
    # Encode state
    state_encoded = le.transform([state])[0]

    # Prepare features
    features = np.array([[state_encoded, total_cases, active, discharged,
                          active_ratio, discharge_ratio, death_ratio, population]])

    # Predict
    prediction = model.predict(features)[0]

    st.success(f"🧮 Predicted Deaths: **{round(prediction, 2)}**")


# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed by <b>Murli Sharma</b> | M.Sc. Big Data Analytics</p>",
    unsafe_allow_html=True
)
