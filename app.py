import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Toggle theme setting
theme_mode = st.toggle("🌞 / 🌜 Toggle Light/Dark Mode")

# CSS + JS for animated background and theme toggle
st.markdown(f"""
    <style>
    body {{
        background: linear-gradient(-45deg, {'#121212, #1c1c1c, #2c2c2c, #3d3d3d' if theme_mode else '#ee7752, #e73c7e, #23a6d5, #23d5ab'});
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: {'white' if theme_mode else 'black'};
    }}

    @keyframes gradientBG {{
        0% {{background-position: 0% 50%;}}
        50% {{background-position: 100% 50%;}}
        100% {{background-position: 0% 50%;}}
    }}

    .main {{
        font-family: 'Segoe UI', sans-serif;
    }}

    .alert {{
        padding: 20px;
        margin-top: 20px;
        border-radius: 5px;
        font-size: 18px;
        text-align: center;
        font-weight: bold;
    }}

    .green {{background-color: #4CAF50; color: white;}}
    .red {{background-color: #f44336; color: white;}}

    div.stButton > button:first-child {{
        background: linear-gradient(135deg, #00f2fe, #4facfe);
        border: none;
        border-radius: 12px;
        color: white;
        padding: 0.75em 2em;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px #4facfe, 0 0 20px #00f2fe, 0 0 30px #4facfe;
        animation: pulse 2s infinite;
        text-shadow: 0 0 5px #fff, 0 0 10px #00f2fe, 0 0 15px #4facfe;
    }}

    div.stButton > button:first-child:hover {{
        background: linear-gradient(135deg, #43e97b, #38f9d7);
        box-shadow: 0 0 20px #43e97b, 0 0 30px #38f9d7;
        transform: scale(1.05);
        cursor: pointer;
    }}

    @keyframes pulse {{
        0% {{
            box-shadow: 0 0 10px #4facfe, 0 0 20px #00f2fe;
        }}
        50% {{
            box-shadow: 0 0 20px #4facfe, 0 0 30px #00f2fe;
        }}
        100% {{
            box-shadow: 0 0 10px #4facfe, 0 0 20px #00f2fe;
        }}
    }}
    </style>
""", unsafe_allow_html=True)

# Glowing Title
st.markdown(f"""
<h1 style='text-align: center;
           font-size: 3em;
           color: {"#fff" if theme_mode else "#111"};
           text-shadow: 0 0 5px #fff, 0 0 10px #00f2fe, 0 0 15px #4facfe;'>
    🫀 Heart Disease Prediction App
</h1>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart_disease_final_filled.csv")

df = load_data()
target_column = "Output"

if target_column not in df.columns:
    st.error(f"Target column '{target_column}' not found.")
else:
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    @st.cache_resource
    def train_model():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        return model, accuracy

    model, accuracy = train_model()

    st.markdown(f"### 📈 Model Accuracy: `{accuracy*100:.2f}%`")

    st.markdown("### 🧠 Input Patient Information:")
    user_input = {}
    for col in X.columns:
        if df[col].dtype in ['float64', 'int64']:
            user_input[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        else:
            user_input[col] = st.selectbox(f"{col}", df[col].unique())

    input_df = pd.DataFrame([user_input])

    if st.button("🔍 Predict"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.subheader("🧮 Prediction Confidence")
        fig1, ax1 = plt.subplots()
        ax1.bar(["No Heart Disease", "Heart Disease"], proba, color=["green", "red"])
        ax1.set_ylabel("Probability")
        ax1.set_ylim(0, 1)
        st.pyplot(fig1)

        if prediction == 1:
            st.markdown(
                f'<div class="alert red">⚠️ The model predicts <b>Heart Disease</b><br>Confidence: {proba[1]:.2f}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="alert green">✅ The model predicts <b>No Heart Disease</b><br>Confidence: {proba[0]:.2f}</div>',
                unsafe_allow_html=True)

    st.subheader("📊 Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)
