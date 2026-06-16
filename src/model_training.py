import pandas as pd
import numpy as np
import sqlite3
import os
import time
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Ensure directories exist
os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "figures")), exist_ok=True)
os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")), exist_ok=True)

def train_and_evaluate():
    print("--- Phase 5: Model Training, Evaluation & Tuning ---\n")
    
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cleaned", "nids_data.db"))
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "figures"))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    
    # 1. Load Data
    print("Loading data from SQLite...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM network_traffic", conn)
    conn.close()
    
    # Drop non-predictive or multi-class target columns
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'attack_cat' in df.columns:
        df = df.drop(columns=['attack_cat'])
        
    # 2. Preprocessing
    print("Preprocessing data (Encoding and Scaling)...")
    # Frequency encoding for high cardinality categoricals
    categorical_cols = ['proto', 'service', 'state']
    for col in categorical_cols:
        if col in df.columns:
            freq_encoding = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq_encoding)
            
    X = df.drop(columns=['label']).fillna(0)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Scale numerical data robustly
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Maintain column names for feature importance later
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # 3. Base Models
    print("\nTraining Base Models...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1)
    }
    
    results = []
    trained_models = {}
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'Training Time (s)': training_time
        })
        trained_models[name] = model
        
        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'cm_{name.replace(" ", "_").lower()}.png'))
        plt.close()
        
        # ROC Curve Accumulation
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')
        
    # Save combined ROC
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'roc_curve_combined.png'))
    plt.close()
    
    results_df = pd.DataFrame(results)
    print("\n--- Model Comparison Table ---")
    print(results_df.to_string(index=False))
    
    # 4. Hyperparameter Tuning (LightGBM)
    print("\n--- Hyperparameter Tuning (LightGBM) ---")
    lgbm_base = LGBMClassifier(random_state=42, n_jobs=-1)
    
    param_distributions = {
        'num_leaves': [31, 50, 100],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 500],
        'max_depth': [-1, 10, 20]
    }
    
    random_search = RandomizedSearchCV(
        estimator=lgbm_base,
        param_distributions=param_distributions,
        n_iter=10,
        scoring='f1',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    random_search.fit(X_train_scaled, y_train)
    tune_time = time.time() - start_time
    
    best_lgbm = random_search.best_estimator_
    y_pred_tuned = best_lgbm.predict(X_test_scaled)
    y_prob_tuned = best_lgbm.predict_proba(X_test_scaled)[:, 1]
    
    f1_tuned = f1_score(y_test, y_pred_tuned)
    auc_tuned = roc_auc_score(y_test, y_prob_tuned)
    
    base_lgbm_idx = results_df[results_df['Model'] == 'LightGBM'].index[0]
    base_f1 = results_df.loc[base_lgbm_idx, 'F1-Score']
    base_auc = results_df.loc[base_lgbm_idx, 'ROC-AUC']
    
    print(f"Tuning Time: {tune_time:.2f} seconds")
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Before Tuning F1: {base_f1:.4f} | After Tuning F1: {f1_tuned:.4f} | Diff: {f1_tuned - base_f1:+.4f}")
    print(f"Before Tuning AUC: {base_auc:.4f} | After Tuning AUC: {auc_tuned:.4f} | Diff: {auc_tuned - base_auc:+.4f}")
    
    # 5. Feature Importance
    print("\nExtracting Feature Importance from Tuned LightGBM...")
    importances = best_lgbm.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Top 15
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=X.columns[indices], palette='viridis')
    plt.title('Top 15 Feature Importances (Tuned LightGBM)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'feature_importance.png'))
    plt.close()
    
    # 6. Save Best Model
    model_filepath = os.path.join(models_dir, 'best_nids_model.joblib')
    pipeline_to_save = {
        'scaler': scaler,
        'model': best_lgbm,
        'categorical_cols': categorical_cols,
        'features': X.columns.tolist()
    }
    joblib.dump(pipeline_to_save, model_filepath)
    print(f"\nBest model pipeline successfully saved to {model_filepath}")

if __name__ == '__main__':
    train_and_evaluate()
