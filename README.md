# Intelligent Network Intrusion Detection System (NIDS)

**Team Members:** Kamal Benramdane, FRANCOIS ZOGBELEMOU
**Module:** AI & Data Science Basics | Semester S4 | EHTP - MIG

## Problem Statement
Modern enterprise networks face an escalating volume of sophisticated cyberthreats that easily bypass traditional, rigid rule-based firewalls. To mitigate this risk, we are building a supervised machine learning model to act as an intelligent Network Intrusion Detection System (NIDS). The system will analyze active network traffic flow metrics to automatically detect and classify anomalous connections into specific attack categories (such as Denial of Service, Fuzzers, Exploits, and Backdoors) versus benign background traffic. This project is designed for Security Operations Center (SOC) analysts; by providing early, automated classification of complex malicious traffic, the model reduces manual log analysis and minimizes "alert fatigue," ultimately preventing devastating data breaches.

## AI Context Note
This project addresses the limitations of early AI rule-based expert systems in cybersecurity. While traditional firewalls rely on static signatures (a brittle reasoning paradigm that fails against novel zero-day attacks), our NIDS leverages statistical machine learning to identify hidden patterns and generalize anomalies. This represents the historical shift towards data-driven AI solutions in threat detection, demonstrating a practical application of concepts from Course 2.

## Dataset
- **Name:** UNSW-NB15 (University of New South Wales Canberra Cyber Range Dataset)
- **Source:** [Kaggle Link](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
- **Size:** 257,673 × 45 (Pre-split train/test subset)

## Project Structure
- `data/raw/`: Raw datasets
- `data/cleaned/`: Processed datasets
- `notebooks/`: Jupyter notebooks for EDA and model exploration
- `src/`: Reusable Python scripts (e.g., data ingestion, preprocessing)
- `models/`: Serialized trained models
- `reports/`: Final reports and presentations

## Instructions to Reproduce Results

To fully reproduce this project from scratch, including data preprocessing, feature engineering, and model training:

### 1. Install Dependencies
Create a clean virtual environment and install the required packages from the project root:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Execution Order
The pipeline is fully automated. You can run the Python scripts in the `src/` directory sequentially to generate the SQLite database, train the models, and create the final Jupyter notebooks:

1. **Data Ingestion \& EDA:** Run `python src/generate_eda.py` (This generates `notebooks/01_EDA_and_Wrangling.ipynb`).
2. **Feature Engineering:** Run `python src/create_phase3_notebook.py` (This generates `notebooks/02_Feature_Engineering.ipynb` and populates the `data/cleaned/nids_data.db` database).
3. **Model Training:** Run `python src/create_phase5_notebook.py` (This trains the models, tunes LightGBM, saves plots to `reports/figures/`, outputs the serialized model to `models/`, and generates `notebooks/03_Model_Training.ipynb`).

Alternatively, after generating the notebooks, you can navigate to the `notebooks/` directory and execute the `.ipynb` files cell-by-cell in numerical order. All notebooks will run end-to-end without errors in a clean environment containing the provided `requirements.txt`.

## Best Model Performance Summary

Our champion model is a hyperparameter-tuned **LightGBM Classifier**. It was explicitly selected over Logistic Regression, Random Forest, and Support Vector Machines due to its exceptional capacity to process high-dimensional network data rapidly (0.63 seconds training time) without sacrificing accuracy.

**Key Metrics (Test Set):**
- **F1-Score:** 0.9820
- **ROC-AUC:** 0.9981
- **Recall:** 0.9725
- **Accuracy:** 0.9767

The model achieves its objective perfectly: it catches 97.25\% of all network attacks while strictly limiting False Positives, thereby preventing "alert fatigue" for SOC analysts.
