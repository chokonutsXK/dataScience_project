import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text = """\
# Phase 3: Feature Engineering & Data Management
This notebook covers the creation of new features, feature selection, SQLite database storage, and analytical SQL queries.
"""

code_import = """\
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure plots are displayed inline
%matplotlib inline
sns.set_theme(style="whitegrid")
"""

code_load = """\
# Load and apply basic Phase 2 cleaning
data_path = '../data/raw/UNSW_NB15_training-set.csv'
df = pd.read_csv(data_path)
df.replace('-', np.nan, inplace=True)
df['service'] = df['service'].fillna('unknown')

print(f"Initial Dataset shape: {df.shape}")
"""

code_feature_eng = """\
# 1. Feature Engineering (Creating 3 new features)

# Feature 1: Total Network Bytes
df['total_network_bytes'] = df['sbytes'] + df['dbytes']

# Feature 2: Total Network Packets
df['total_network_packets'] = df['spkts'] + df['dpkts']

# Feature 3: Bytes per Packet
df['bytes_per_packet'] = df['total_network_bytes'] / (df['total_network_packets'] + 1e-9)

print("Created 3 new features: 'total_network_bytes', 'total_network_packets', 'bytes_per_packet'")
"""

code_baseline = """\
# 1.5 Baseline Metric Comparison
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

print("--- Baseline Metric Comparison ---")
orig_num_features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['label', 'id', 'total_network_bytes', 'total_network_packets', 'bytes_per_packet']]
X_orig = df[orig_num_features].fillna(0)
y = df['label']

X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_orig, y, test_size=0.3, random_state=42)

dt_base = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_base.fit(X_train_orig, y_train)
f1_base = f1_score(y_test, dt_base.predict(X_test_orig))
print(f"Baseline F1-Score (Original Numeric Features): {f1_base:.4f}")

new_features = ['total_network_bytes', 'total_network_packets', 'bytes_per_packet']
X_new = df[orig_num_features + new_features].fillna(0)
X_train_new, X_test_new, _, _ = train_test_split(X_new, y, test_size=0.3, random_state=42)

dt_new = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_new.fit(X_train_new, y_train)
f1_new = f1_score(y_test, dt_new.predict(X_test_new))
print(f"Engineered F1-Score (With 3 New Features): {f1_new:.4f}")
print(f"Absolute Improvement in F1-Score: {f1_new - f1_base:.4f}")
"""

code_feature_sel = """\
# 2. Feature Selection Report

# Calculate correlation with label
num_df = df.select_dtypes(include=[np.number])
corr_with_label = num_df.corr()['label'].sort_values(ascending=False)

print("Top 10 Positively Correlated Features:")
display(corr_with_label.head(10))

print("\\nTop 5 Negatively Correlated Features:")
display(corr_with_label.tail(5))

# Drop highly collinear base features since we created aggregates
features_to_drop = ['sbytes', 'dbytes', 'spkts', 'dpkts']
df_selected = df.drop(columns=features_to_drop)

print(f"\\nDropped {len(features_to_drop)} features to reduce multicollinearity: {features_to_drop}")
print(f"Final Dataset shape after feature selection: {df_selected.shape}")
"""

code_sqlite = """\
# 3. Storing Cleaned Dataset in SQLite
db_path = '../data/cleaned/nids_data.db'
os.makedirs(os.path.dirname(db_path), exist_ok=True)

conn = sqlite3.connect(db_path)
df_selected.to_sql('network_traffic', conn, if_exists='replace', index=False)
print(f"Dataset successfully saved to SQLite database at: {db_path}")
"""

code_sql1 = """\
# 4. Analytical SQL Queries

# Query 1: Count of normal vs. attack connections
query1 = \"\"\"
    SELECT label, COUNT(*) as count 
    FROM network_traffic 
    GROUP BY label
\"\"\"
print("Query 1: Distribution of Labels")
display(pd.read_sql_query(query1, conn))
"""

code_sql2 = """\
# Query 2: Top 3 most common protocols used in attack traffic
query2 = \"\"\"
    SELECT proto, COUNT(*) as attack_count 
    FROM network_traffic 
    WHERE label = 1 
    GROUP BY proto 
    ORDER BY attack_count DESC 
    LIMIT 3
\"\"\"
print("Query 2: Top 3 Attack Protocols")
display(pd.read_sql_query(query2, conn))
"""

code_sql3 = """\
# Query 3: Average duration grouped by attack_cat
query3 = \"\"\"
    SELECT attack_cat, AVG(dur) as avg_duration 
    FROM network_traffic 
    WHERE label = 1 
    GROUP BY attack_cat 
    ORDER BY avg_duration DESC
\"\"\"
print("Query 3: Average Duration by Attack Category")
display(pd.read_sql_query(query3, conn))
"""

code_sql4 = """\
# Query 4: Custom Analytical Query - Volumetric profiling
query4 = \"\"\"
    SELECT attack_cat, 
           AVG(bytes_per_packet) as avg_bytes_per_packet,
           SUM(total_network_bytes) as total_volume
    FROM network_traffic 
    WHERE label = 1 
    GROUP BY attack_cat 
    ORDER BY avg_bytes_per_packet DESC
\"\"\"
print("Query 4 (Custom): Volumetric Profiling by Attack Category")
display(pd.read_sql_query(query4, conn))

# Close the connection
conn.close()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code_import),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_feature_eng),
    nbf.v4.new_code_cell(code_baseline),
    nbf.v4.new_code_cell(code_feature_sel),
    nbf.v4.new_code_cell(code_sqlite),
    nbf.v4.new_code_cell(code_sql1),
    nbf.v4.new_code_cell(code_sql2),
    nbf.v4.new_code_cell(code_sql3),
    nbf.v4.new_code_cell(code_sql4)
]

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "notebooks"), exist_ok=True)
notebook_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "02_Feature_Engineering.ipynb")

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook successfully created at {notebook_path}")
