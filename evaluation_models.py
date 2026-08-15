import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef
)

def main():
    print("Loading dataset and preprocessing...")
    
    # 1. Load the data (Ensure 'heart.csv' is in the same folder)
    try:
        df = pd.read_csv('heart.csv')
    except FileNotFoundError:
        print("Error: 'heart.csv' not found. Please upload it to the same directory.")
        return

    X = df.drop(columns=['target'])
    y = df['target']

    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Initialize the 5 mandatory models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "kNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    # 5. Train and Evaluate
    print("Training models and calculating metrics...\n")
    results_list = []
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] 
        
        # Calculate the 6 mandatory metrics
        results_list.append({
            "ML Model Name": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "AUC": round(roc_auc_score(y_test, y_prob), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
        })

    # 6. Display results for the screenshot
    results_df = pd.DataFrame(results_list)
    print("-" * 75)
    print("BITS Virtual Lab Execution Successful! Final Metrics:")
    print("-" * 75)
    print(results_df.to_string(index=False))
    print("-" * 75)

if __name__ == "__main__":
    main()