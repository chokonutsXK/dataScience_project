import pandas as pd
import os

def load_and_describe_data(file_path):
    """
    Loads a dataset and prints its basic statistics, fulfilling Phase 1 requirements.
    """
    print(f"--- Loading data from {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please ensure the UNSW-NB15 dataset is downloaded and placed in the data/raw/ directory.")
        return None

    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!\n")
        
        print("1. Shape of the dataset (Rows, Columns):")
        print(df.shape)
        print("\n" + "-"*50 + "\n")
        
        print("2. Data Types (dtypes):")
        print(df.dtypes)
        print("\n" + "-"*50 + "\n")
        
        print("3. Null Counts per Column:")
        null_counts = df.isnull().sum()
        print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "No missing values found.")
        print("\n" + "-"*50 + "\n")
        
        print("4. Basic Statistics (Numerical Columns):")
        print(df.describe())
        print("\n" + "-"*50 + "\n")
        
        return df
    
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

if __name__ == "__main__":
    # Pointing to the expected dataset file in data/raw/
    # If your file is named differently, update the filename below:
    raw_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "UNSW_NB15_training-set.csv")
    
    # Resolve the absolute path for cleaner output
    raw_data_path = os.path.abspath(raw_data_path)
    
    df = load_and_describe_data(raw_data_path)
