# Credit Card Fraud Detection

## Project Overview
This project identifies fraudulent credit card transactions using **unsupervised anomaly detection** and **supervised machine learning** techniques.  
It compares **Isolation Forest** and **Local Outlier Factor (LOF)** with **XGBoost** on a highly imbalanced dataset.

**Tools & Technologies:**  
Python, Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn, Jupyter Notebook

---

## Dataset
- **Source:** Kaggle – [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)  
- **Features:** `V1`–`V28`, `Amount`, `Class` (0 = legitimate, 1 = fraud)  
- **Imbalance:** Fraud transactions <1% of total

---

## Project Steps
1. **Data Loading & Preprocessing**  
   - Load dataset  
   - Handle missing values, scale features

2. **Unsupervised Anomaly Detection**  
   - Isolation Forest  
   - Local Outlier Factor (LOF)  
   - Evaluate using confusion matrix and classification report

3. **Supervised Learning**  
   - Train XGBoost classifier  
   - Evaluate using precision, recall, F1-score, confusion matrix, ROC-AUC

4. **Performance Comparison**  
   - Compare unsupervised vs supervised models

5. **(Optional) Deployment**  
   - Streamlit/Flask dashboard for user input and prediction

---

## Usage

- Open `Credit_Card_Fraud_Detection.ipynb` in Jupyter Notebook or Google Colab

- Run all cells to:
  1. Load and preprocess data
  2. Train models (Isolation Forest, LOF, XGBoost)
  3. Evaluate and visualize results

- Optional: Run `app.py` for Streamlit dashboard (if deployed)

---

## Installation
```bash
# Clone repository
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

---


## Results

| Model | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
|-------|-----------------|----------------|-----------------|
| XGBoost | 0.87 | 0.85 | 0.86 |
| Isolation Forest | 0.40 | 0.23 | 0.30 |
| LOF (sample) | 0.001 | 1.0 | 0.003 |

**Observation:** XGBoost outperforms unsupervised models for fraud detection.

---

## Visualizations

- ROC Curve, Precision-Recall Curve
- Confusion Matrices
- Performance comparison bar plot

---

## Future Work

- Deploy interactive dashboard for real-time predictions
- Experiment with other supervised models (LightGBM, CatBoost)
- Tune hyperparameters to improve recall on fraud class

---

## Author

**Mukesh Sunku**  
- GitHub: [https://github.com/<mukeshsunku721>](https://github.com/<mukeshsunku721>)  
- Email: `<mukeshrugs@gmail.com>`
