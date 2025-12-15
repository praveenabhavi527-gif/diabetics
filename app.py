import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS (The "Design" Part)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to inject into the app
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .result-positive {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ef9a9a;
    }
    .result-negative {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & MODEL TRAINING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Attempt to load local file first, otherwise load from URL (Pima Indians Dataset)
    try:
        # Standard Pima dataset URL
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        column_names = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age', 'class']
        df = pd.read_csv(url, names=column_names)
        
        # In the dataset, class 1 is tested_positive, 0 is tested_negative
        # We map it to match your screenshot's output format
        df['class'] = df['class'].map({1: 'tested_positive', 0: 'tested_negative'})
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

@st.cache_resource
def train_model(data):
    if data is None:
        return None, 0
    
    # Feature Selection (matching your script)
    ind = data[['age', 'mass', 'insu', 'plas']]
    dep = data['class']
    
    # Model Training
    Logr = LogisticRegression(max_iter=1000) # Increased max_iter for safety
    Logr.fit(ind, dep)
    
    # Calculate Accuracy
    predictions = Logr.predict(ind)
    acc = accuracy_score(dep, predictions)
    
    return Logr, acc

model, accuracy = train_model(df)

# -----------------------------------------------------------------------------
# 3. SIDEBAR (User Inputs)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🩺 Patient Data")
    st.write("Enter the details below:")
    
    # Using sliders and number inputs for better UX
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=23)
    mass = st.number_input("Body Mass Index (mass)", min_value=0.0, max_value=70.0, value=30.0)
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=99)
    plasma = st.number_input("Plasma Glucose (plas)", min_value=0, max_value=300, value=100)
    
    st.markdown("---")
    predict_btn = st.button("Analyze Result")

# -----------------------------------------------------------------------------
# 4. MAIN INTERFACE
# -----------------------------------------------------------------------------
st.title("Diabetes Prediction System")
st.markdown("This application uses **Logistic Regression** to predict the likelihood of diabetes based on clinical parameters.")

col1, col2 = st.columns([2, 1])

with col1:
    if predict_btn:
        if model:
            # Prepare input data as DataFrame to avoid the "valid feature names" warning
            input_data = pd.DataFrame(
                [[age, mass, insulin, plasma]], 
                columns=['age', 'mass', 'insu', 'plas']
            )
            
            # Prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)
            
            # Display Logic
            if prediction == 'tested_positive':
                prob_score = probability[0][1] * 100
                st.markdown(f"""
                    <div class="result-card result-positive">
                        <h2>Result: Tested Positive</h2>
                        <p>The model predicts a high likelihood of diabetes.</p>
                        <h1>{prob_score:.1f}% Probability</h1>
                    </div>
                """, unsafe_allow_html=True)
            else:
                prob_score = probability[0][0] * 100
                st.markdown(f"""
                    <div class="result-card result-negative">
                        <h2>Result: Tested Negative</h2>
                        <p>The model predicts a low likelihood of diabetes.</p>
                        <h1>{prob_score:.1f}% Confidence</h1>
                    </div>
                """, unsafe_allow_html=True)
                
        else:
            st.error("Model could not be loaded.")
    else:
        st.info("👈 Please enter patient data in the sidebar and click 'Analyze Result'.")

with col2:
    st.subheader("Model Performance")
    if model:
        st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")
        st.progress(accuracy)
        st.caption("Training Data: Pima Indians Diabetes Database")
        
        st.write("### Features Used:")
        st.code("['age', 'mass', 'insu', 'plas']", language="python")
