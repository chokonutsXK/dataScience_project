import pandas as pd
import numpy as np
import sqlite3
import os

def feature_engineering_pipeline(data_path, db_path):
    print("--- Starting Phase 3: Feature Engineering & Data Management ---")
    
    # Load dataset
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Apply Phase 2 basic cleaning to ensure consistency
    df.replace('-', np.nan, inplace=True)
    df['service'] = df['service'].fillna('unknown')
    
    # ---------------------------------------------------------
    # 1. Feature Engineering (3 New Features)
    # ---------------------------------------------------------
    print("\n--- 1. Creating Engineered Features ---")
    
    # Feature 1: Total Network Bytes
    # Motivation: Captures the total volume of data exchanged.
    df['total_network_bytes'] = df['sbytes'] + df['dbytes']
    print("Created 'total_network_bytes' = sbytes + dbytes")
    
    # Feature 2: Total Network Packets
    # Motivation: Captures total interaction level.
    df['total_network_packets'] = df['spkts'] + df['dpkts']
    print("Created 'total_network_packets' = spkts + dpkts")
    
    # Feature 3: Bytes per Packet
    # Motivation: Attackers often use unusual packet sizes.
    # Add a small epsilon to avoid division by zero
    df['bytes_per_packet'] = df['total_network_bytes'] / (df['total_network_packets'] + 1e-9)
    print("Created 'bytes_per_packet' = total_network_bytes / total_network_packets")

    # ---------------------------------------------------------
    # Baseline Metric Comparison
    # ---------------------------------------------------------
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    print("\n--- Baseline Metric Comparison ---")
    # Original numeric features excluding 'label' and 'id'
    orig_num_features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['label', 'id', 'total_network_bytes', 'total_network_packets', 'bytes_per_packet']]
    
    X_orig = df[orig_num_features].fillna(0)
    y = df['label']
    
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_orig, y, test_size=0.3, random_state=42)
    
    dt_base = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_base.fit(X_train_orig, y_train)
    f1_base = f1_score(y_test, dt_base.predict(X_test_orig))
    print(f"Baseline F1-Score (Original Numeric Features): {f1_base:.4f}")
    
    # New features
    new_features = ['total_network_bytes', 'total_network_packets', 'bytes_per_packet']
    X_new = df[orig_num_features + new_features].fillna(0)
    X_train_new, X_test_new, _, _ = train_test_split(X_new, y, test_size=0.3, random_state=42)
    
    dt_new = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_new.fit(X_train_new, y_train)
    f1_new = f1_score(y_test, dt_new.predict(X_test_new))
    print(f"Engineered F1-Score (With 3 New Features): {f1_new:.4f}")
    
    improvement = f1_new - f1_base
    print(f"Absolute Improvement in F1-Score: {improvement:.4f}")

    # ---------------------------------------------------------
    # 2. Feature Selection Report
    # ---------------------------------------------------------
    print("\n--- 2. Feature Selection Report ---")
    
    # Calculate correlation with the target label for numeric features
    num_df = df.select_dtypes(include=[np.number])
    corr_with_label = num_df.corr()['label'].sort_values(ascending=False)
    
    print("\nTop 10 Most Important Features (Positive Correlation):")
    print(corr_with_label.head(10))
    
    print("\nTop 5 Least Important Features (Negative Correlation):")
    print(corr_with_label.tail(5))

    # Identify features to drop based on multicollinearity with our new features
    # Since we created total_network_bytes, keeping sbytes and dbytes might cause multicollinearity.
    features_to_drop = ['sbytes', 'dbytes', 'spkts', 'dpkts']
    df_selected = df.drop(columns=features_to_drop)
    
    print(f"\nDropped {len(features_to_drop)} features to reduce multicollinearity:")
    print(features_to_drop)
    print(f"Dataset shape after feature selection: {df_selected.shape}")

    # ---------------------------------------------------------
    # 3. Data Management: SQLite Database
    # ---------------------------------------------------------
    print("\n--- 3. Storing Cleaned Dataset in SQLite ---")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    df_selected.to_sql('network_traffic', conn, if_exists='replace', index=False)
    print(f"Dataset successfully saved to table 'network_traffic' in {db_path}")

    # ---------------------------------------------------------
    # 4. SQL Queries (Analytical Questions)
    # ---------------------------------------------------------
    print("\n--- 4. Analytical SQL Queries ---")
    
    # Query 1: Count of normal vs. attack connections
    query1 = """
        SELECT label, COUNT(*) as count 
        FROM network_traffic 
        GROUP BY label
    """
    print("\nQuery 1: Count of Normal (0) vs Attack (1) connections")
    print(pd.read_sql_query(query1, conn))

    # Query 2: Top 3 most common protocols used in attack traffic
    query2 = """
        SELECT proto, COUNT(*) as attack_count 
        FROM network_traffic 
        WHERE label = 1 
        GROUP BY proto 
        ORDER BY attack_count DESC 
        LIMIT 3
    """
    print("\nQuery 2: Top 3 protocols used in attack traffic")
    print(pd.read_sql_query(query2, conn))

    # Query 3: Average duration grouped by attack_cat
    query3 = """
        SELECT attack_cat, AVG(dur) as avg_duration 
        FROM network_traffic 
        WHERE label = 1 
        GROUP BY attack_cat 
        ORDER BY avg_duration DESC
    """
    print("\nQuery 3: Average duration by attack category")
    print(pd.read_sql_query(query3, conn))

    # Query 4: Custom Analytical Query - Volumetric profiling
    query4 = """
        SELECT attack_cat, 
               AVG(bytes_per_packet) as avg_bytes_per_packet,
               SUM(total_network_bytes) as total_volume
        FROM network_traffic 
        WHERE label = 1 
        GROUP BY attack_cat 
        ORDER BY avg_bytes_per_packet DESC
    """
    print("\nQuery 4 (Custom): Volumetric profiling (Avg Bytes/Packet and Total Volume) by attack category")
    print(pd.read_sql_query(query4, conn))

    conn.close()
    print("\n--- Phase 3 Pipeline Completed ---")

if __name__ == "__main__":
    raw_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "UNSW_NB15_training-set.csv"))
    sqlite_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cleaned", "nids_data.db"))
    
    feature_engineering_pipeline(raw_data_path, sqlite_db_path)
