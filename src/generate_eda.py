import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import os
import seaborn as sns

def perform_eda_and_save_plots(data_path, output_dir):
    print("--- Starting Phase 2 EDA ---")
    df = pd.read_csv(data_path)
    
    # 1. Data Audit & Cleaning
    # Replace '-' with NaN
    df.replace('-', np.nan, inplace=True)
    null_counts = df.isnull().sum()
    print("Missing values after replacing '-':")
    print(null_counts[null_counts > 0])
    
    # Impute missing values
    df['service'].fillna('unknown', inplace=True)
    
    # 2. Outlier Detection (IQR on sbytes and dbytes)
    def calculate_iqr_outliers(column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return len(outliers), lower_bound, upper_bound
    
    print("\nOutlier Detection (IQR):")
    for col in ['sbytes', 'dbytes', 'dur']:
        num_outliers, lb, ub = calculate_iqr_outliers(col)
        print(f"{col}: {num_outliers} outliers (Bounds: {lb:.2f} - {ub:.2f})")
    
    # 3. Generating 8+ Visualizations
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Target class imbalance
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='label', palette='viridis')
    plt.title('Distribution of Target Label (0=Normal, 1=Attack)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_label_distribution.png'))
    plt.close()
    
    # Plot 2: Attack Categories
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df[df['label'] == 1], y='attack_cat', order=df[df['label'] == 1]['attack_cat'].value_counts().index, palette='magma')
    plt.title('Distribution of Attack Categories')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_attack_cat_distribution.png'))
    plt.close()
    
    # Plot 3: Boxplots of packets (log scale)
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df, x='label', y='spkts', palette='viridis')
    plt.yscale('log')
    plt.title('Source Packets by Label (Log Scale)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_spkts_boxplot.png'))
    plt.close()
    
    # Plot 4: Boxplots of bytes (log scale)
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df, x='label', y='sbytes', palette='viridis')
    plt.yscale('log')
    plt.title('Source Bytes by Label (Log Scale)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_sbytes_boxplot.png'))
    plt.close()

    # Plot 5: Top 10 Protocols
    top_protos = df['proto'].value_counts().nlargest(10).index
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df[df['proto'].isin(top_protos)], y='proto', hue='label', palette='viridis')
    plt.title('Top 10 Protocols by Target Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_top_protocols.png'))
    plt.close()

    # Plot 6: Services
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, y='service', hue='label', order=df['service'].value_counts().index, palette='viridis')
    plt.title('Service Types by Target Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_services_distribution.png'))
    plt.close()
    
    # Plot 7: Duration by Attack vs Normal
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x='dur', hue='label', fill=True, common_norm=False, palette='viridis')
    plt.xscale('log')
    plt.title('Distribution of Flow Duration (Log Scale)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_duration_kde.png'))
    plt.close()

    # Plot 8: Correlation Heatmap (Top 15 correlated with label)
    # Select numeric columns
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    top_corr_features = corr.index[abs(corr["label"]) > 0.2]
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap (Features highly correlated with Label)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_correlation_heatmap.png'))
    plt.close()

    print("\nVisualizations generated and saved to:", output_dir)
    print("Top correlated features with label:")
    print(corr['label'].sort_values(ascending=False).head(10))
    print(corr['label'].sort_values(ascending=False).tail(5))

if __name__ == "__main__":
    raw_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "UNSW_NB15_training-set.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
    perform_eda_and_save_plots(raw_data_path, output_dir)
