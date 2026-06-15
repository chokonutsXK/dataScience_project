import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text = """\
# Phase 2: Exploratory Data Analysis & Data Wrangling
This notebook performs the required EDA for the UNSW-NB15 dataset, including dealing with missing values, outlier detection, and generating 8+ visualisations.
"""

code_import = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure plots are displayed inline
%matplotlib inline
sns.set_theme(style="whitegrid")
"""

code_load = """\
# Load dataset
data_path = '../data/raw/UNSW_NB15_training-set.csv'
df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape}")
"""

code_audit = """\
# 1. Data Audit & Cleaning
# The dataset represents missing values as '-' in string columns
df.replace('-', np.nan, inplace=True)
null_counts = df.isnull().sum()
print("Missing values after replacing '-':")
print(null_counts[null_counts > 0])

# Impute missing 'service' values with 'unknown'
df['service'] = df['service'].fillna('unknown')
"""

code_outliers = """\
# 2. Outlier Detection
def calculate_iqr_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

for col in ['sbytes', 'dbytes', 'dur']:
    num_outliers, lb, ub = calculate_iqr_outliers(col)
    print(f"{col}: {num_outliers} outliers (Bounds: {lb:.2f} - {ub:.2f})")
"""

code_vis1 = """\
# 3. Visualizations

# Plot 1: Target class imbalance
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label', hue='label', palette='viridis', legend=False)
plt.title('Distribution of Target Label (0=Normal, 1=Attack)')
plt.show()
"""

code_vis2 = """\
# Plot 2: Attack Categories
plt.figure(figsize=(10, 5))
sns.countplot(data=df[df['label'] == 1], y='attack_cat', hue='attack_cat', order=df[df['label'] == 1]['attack_cat'].value_counts().index, palette='magma', legend=False)
plt.title('Distribution of Attack Categories')
plt.show()
"""

code_vis3 = """\
# Plot 3: Boxplots of packets (log scale)
plt.figure(figsize=(10, 4))
sns.boxplot(data=df, x='label', y='spkts', hue='label', palette='viridis', legend=False)
plt.yscale('log')
plt.title('Source Packets by Label (Log Scale)')
plt.show()
"""

code_vis4 = """\
# Plot 4: Boxplots of bytes (log scale)
plt.figure(figsize=(10, 4))
sns.boxplot(data=df, x='label', y='sbytes', hue='label', palette='viridis', legend=False)
plt.yscale('log')
plt.title('Source Bytes by Label (Log Scale)')
plt.show()
"""

code_vis5 = """\
# Plot 5: Top 10 Protocols
top_protos = df['proto'].value_counts().nlargest(10).index
plt.figure(figsize=(10, 5))
sns.countplot(data=df[df['proto'].isin(top_protos)], y='proto', hue='label', palette='viridis')
plt.title('Top 10 Protocols by Target Label')
plt.show()
"""

code_vis6 = """\
# Plot 6: Services
plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='service', hue='label', order=df['service'].value_counts().index, palette='viridis')
plt.title('Service Types by Target Label')
plt.show()
"""

code_vis7 = """\
# Plot 7: Duration by Attack vs Normal
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='dur', hue='label', fill=True, common_norm=False, palette='viridis')
plt.xscale('log')
plt.title('Distribution of Flow Duration (Log Scale)')
plt.show()
"""

code_vis8 = """\
# Plot 8: Correlation Heatmap (Top 15 correlated with label)
num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr()
top_corr_features = corr.index[abs(corr["label"]) > 0.2]
plt.figure(figsize=(12, 10))
sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap (Features highly correlated with Label)')
plt.show()
"""

code_summary = """\
# Summary
print("Top correlated features with label:")
print(corr['label'].sort_values(ascending=False).head(10))
print(corr['label'].sort_values(ascending=False).tail(5))
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code_import),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_audit),
    nbf.v4.new_code_cell(code_outliers),
    nbf.v4.new_code_cell(code_vis1),
    nbf.v4.new_code_cell(code_vis2),
    nbf.v4.new_code_cell(code_vis3),
    nbf.v4.new_code_cell(code_vis4),
    nbf.v4.new_code_cell(code_vis5),
    nbf.v4.new_code_cell(code_vis6),
    nbf.v4.new_code_cell(code_vis7),
    nbf.v4.new_code_cell(code_vis8),
    nbf.v4.new_code_cell(code_summary)
]

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "notebooks"), exist_ok=True)
notebook_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "01_EDA_and_Wrangling.ipynb")

with open(notebook_path, 'w') as f:
    nbf.write(nb, f)
    
print(f"Notebook successfully created at {notebook_path}")
