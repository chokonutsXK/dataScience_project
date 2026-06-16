import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text = """\
# Phase 5: Model Training, Evaluation & Tuning
This notebook implements the machine learning pipeline, including data preparation, base model comparison, hyperparameter tuning, and model interpretability.
"""

code_import = """\
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import joblib

# Ensure plots are displayed inline
%matplotlib inline
sns.set_theme(style="whitegrid")
"""

code_prep = """\
# 1. Data Loading & Preprocessing
db_path = '../data/cleaned/nids_data.db'
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM network_traffic", conn)
conn.close()

# Drop identifiers and multi-class target
df = df.drop(columns=['id', 'attack_cat'], errors='ignore')

# Frequency Encoding for categorical variables
for col in ['proto', 'service', 'state']:
    if col in df.columns:
        freq_encoding = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq_encoding)
        
X = df.drop(columns=['label']).fillna(0)
y = df['label']

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Robust Scaling
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
print(f"Training set shape: {X_train_scaled.shape}")
"""

code_models = """\
# 2. Base Model Training & Evaluation
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1)
}

results = []
trained_models = {}

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
        'Time (s)': training_time
    })
    trained_models[name] = model

results_df = pd.DataFrame(results)
display(results_df)
"""

code_roc = """\
# 3. ROC Curve Comparison
plt.figure(figsize=(10, 8))

for name, model in trained_models.items():
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()
"""

code_tuning = """\
# 4. Hyperparameter Tuning on Best Model (LightGBM)
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

print("Starting RandomizedSearchCV...")
random_search.fit(X_train_scaled, y_train)

best_lgbm = random_search.best_estimator_
y_pred_tuned = best_lgbm.predict(X_test_scaled)
y_prob_tuned = best_lgbm.predict_proba(X_test_scaled)[:, 1]

f1_tuned = f1_score(y_test, y_pred_tuned)
auc_tuned = roc_auc_score(y_test, y_prob_tuned)

base_lgbm_idx = results_df[results_df['Model'] == 'LightGBM'].index[0]
base_f1 = results_df.loc[base_lgbm_idx, 'F1-Score']

print(f"Best Parameters: {random_search.best_params_}")
print(f"Before Tuning F1: {base_f1:.4f} | After Tuning F1: {f1_tuned:.4f} | Diff: {f1_tuned - base_f1:+.4f}")
"""

code_importance = """\
# 5. Feature Importance Analysis
importances = best_lgbm.feature_importances_
indices = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X.columns[indices], palette='viridis')
plt.title('Top 15 Feature Importances (Tuned LightGBM)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
"""

code_save = """\
# 6. Save Best Model
pipeline_to_save = {
    'scaler': scaler,
    'model': best_lgbm,
    'features': X.columns.tolist()
}
joblib.dump(pipeline_to_save, '../models/best_nids_model.joblib')
print("Model saved successfully to ../models/best_nids_model.joblib")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code_import),
    nbf.v4.new_code_cell(code_prep),
    nbf.v4.new_code_cell(code_models),
    nbf.v4.new_code_cell(code_roc),
    nbf.v4.new_code_cell(code_tuning),
    nbf.v4.new_code_cell(code_importance),
    nbf.v4.new_code_cell(code_save)
]

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "notebooks"), exist_ok=True)
notebook_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "03_Model_Training.ipynb")

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook successfully created at {notebook_path}")
