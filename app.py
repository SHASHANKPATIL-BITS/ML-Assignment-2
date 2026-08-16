import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef, 
    confusion_matrix, 
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

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
            
        # Makes Predictions and Probability Estimates
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculates all 6 required metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # 3. Displays all 6 evaluation metrics
        st.write(f"### Evaluation Metrics for {model_name}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")
        
        # 4. Confusion matrix and classification report
        st.write("---")
        st.write("### Classification Report")
        
        # Convert report to dictionary, then to a DataFrame
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        
        # UI ENHANCEMENT 1: Capitalize and clean column names
        report_df.columns = ["Precision", "Recall", "F1 Score", "Support"]
        
        # UI ENHANCEMENT 2: Make row names (index) human-readable
        index_mapping = {
            '0': 'Class 0 (No Disease)',
            '1': 'Class 1 (Disease)',
            'accuracy': 'Overall Accuracy',
            'macro avg': 'Macro Avg',
            'weighted avg': 'Weighted Avg'
        }
        report_df.rename(index=index_mapping, inplace=True)
        
        # UI ENHANCEMENT 3 & 4: Format decimal places and center align
        styled_df = report_df.style.format({
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "Support": "{:.0f}"  # Whole numbers for patient counts!
        }).set_properties(**{'text-align': 'center'})
        
        # Display the upgraded table stretched nicely across the container
        st.dataframe(styled_df, use_container_width=True)
        
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        st.pyplot(fig)
        
    else:
        st.error("The uploaded CSV must contain a 'target' column to evaluate the model.")
