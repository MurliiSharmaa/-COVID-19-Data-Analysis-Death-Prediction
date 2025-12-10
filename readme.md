🦠 COVID-19 Data Analysis & Death Prediction
📊 Machine Learning + Streamlit Dashboard

This project analyzes India’s COVID-19 state-wise data and builds a predictive ML model to estimate the number of COVID-19 deaths based on Active Cases, Discharged Patients, and Total Cases.
It also includes a beautiful interactive Streamlit Dashboard for visualization and real-time prediction.

🚀 Project Overview

COVID-19 created a massive challenge worldwide. Understanding the relationship between total cases, active cases, recoveries, deaths, and population helps in:

✔ Predicting the severity of the outbreak
✔ Understanding patterns among states
✔ Supporting early decision-making
✔ Visualizing trends and ratios

This project performs Exploratory Data Analysis (EDA), calculates important metrics like Active Ratio, Discharge Ratio, and Death Ratio, and trains a Machine Learning regression model to predict deaths.

📁 Dataset Description

The dataset includes the following columns:

Column	Description
State/UTs	Indian State or Union Territory
Total Cases	Total confirmed COVID-19 cases
Active	Total active cases
Discharged	Total recovered patients
Deaths	Total deaths
Active Ratio	(Active / Total Cases) × 100
Discharge Ratio	(Discharged / Total Cases) × 100
Death Ratio	(Deaths / Total Cases) × 100
Population	Population of the State/UT
🧠 Machine Learning Model
🎯 Goal

Predict the number of COVID-19 deaths using the available numerical features.

📌 Independent (Feature) Variables

These columns are used for prediction:

Total Cases

Active

Discharged

Population

🎯 Dependent (Target) Variable

Deaths

🏆 Model Used

✔ Random Forest Regressor

Why Random Forest?

Handles non-linear data

High accuracy

Resistant to overfitting

Works extremely well on small datasets

📊 Streamlit Dashboard Features
👉 1. Data Visualization

Bar Chart (Active, Discharged, Deaths per State)

Pie Chart

Line Trends

Metric Boxes

Interactive Filters

👉 2. Prediction Panel

Enter the following manually:

Total Cases

Active Cases

Discharged

Population

The model predicts:

🔮 Expected number of deaths

👉 3. Model Extraction

The model is saved as:

model.pkl
label_encoder.pkl


Loaded directly in the Streamlit UI.

📂 Project Structure
📁 COVID-19-Prediction
│── app.py                # Streamlit Dashboard
│── model_build.py        # Model training
│── model.pkl             # Saved Machine Learning Model
│── label_encoder.pkl     # Saved encoder for state names
│── data.csv              # Dataset
│── README.md             # Documentation
│── requirements.txt      # Libraries list

🛠 Installation & Running the Dashboard
1️⃣ Clone the Repository
git clone https://github.com/your-username/covid19-prediction.git
cd covid19-prediction

2️⃣ Install Required Packages
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app.py

📈 Model Training Code (Short Overview)
df = pd.read_csv("data.csv")

X = df[['Total Cases', 'Active', 'Discharged', 'Population']]
y = df['Deaths']

model = RandomForestRegressor()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

🌐 Streamlit UI Code (Short Preview)
model = pickle.load(open("model.pkl", "rb"))

st.title("COVID-19 Death Prediction Dashboard")

total = st.number_input("Total Cases")
active = st.number_input("Active Cases")
discharged = st.number_input("Discharged")
population = st.number_input("Population")

if st.button("Predict Deaths"):
    val = model.predict([[total, active, discharged, population]])
    st.success(f"Predicted Deaths: {int(val[0])}")

🎨 Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn (Random Forest)

Streamlit (Dashboard)

📘 Learnings & Outcomes

Through this project, I learned:

Handling real-world COVID-19 data

Ratio calculation & statistical analysis

Feature engineering

Machine learning model development

Deployment using Streamlit

Creating interactive dashboards

⭐ Future Enhancements

🟡 Add time-series forecasting
🟡 Use LSTM/ARIMA models
🟡 Add state maps & geospatial plots
🟡 Deploy on cloud (Streamlit Cloud / AWS)

🤝 Contribution

Contributions are welcome!
Feel free to fork this repo and submit a pull request.

📬 Contact

Author: Murli Sharma
📧 Email: murli.analyst@gmail.com

📍 Ahmedabad, Gujarat