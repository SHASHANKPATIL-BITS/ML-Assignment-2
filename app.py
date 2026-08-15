import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Heart Disease Prediction & Model Evaluation")
st.write("Upload your test data to evaluate the performance of different Machine Learning models.")

# 1. Dataset upload option (CSV)
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

# 2. Model selection dropdown
model_name = st.selectbox(
    "Select Machine Learning Model", 
    ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"]
)

if uploaded_file is not None:
    # Load the uploaded dataset
    test_data = pd.read_csv(uploaded_file)
    
    st.write("### Data Preview")
    st.dataframe(test_data.head())
    
    # Check if target column exists
    if 'target' in test_data.columns:
        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']
        
        # Load the Scaler
        try:
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            X_test_scaled = scaler.transform(X_test)
        except FileNotFoundError:
            st.error("Error: 'scaler.pkl' not found. Please ensure it is in the repository.")
            st.stop()
            
        # Load the selected model
        model_filename = f"model/{model_name.replace(' ', '_')}.pkl"
        try:
            with open(model_filename, 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError:
            st.error(f"Error: Model file '{model_filename}' not found.")
            st.stop()
            
        # Make Predictions
        y_pred = model.predict(X_test_scaled)
        
        # 3. Display of evaluation metrics
        st.write(f"### Evaluation Metrics for {model_name}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
        
        # 4. Confusion matrix and classification report
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))
        
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        st.pyplot(fig)
        
    else:
        st.error("The uploaded CSV must contain a 'target' column to evaluate the model.")