# Customer 360 Intelligence - ML Workshop (NileConnect)

End-to-end machine learning project on a synthetic telecom dataset (3,000 customers):
churn classification, 12-month revenue regression, customer segmentation, PCA,
anomaly detection and a final Customer 360 table.

Full results, tables, recommendations and challenge answers: see [REPORT.md](REPORT.md).

## Project structure
```
ML_Workshop_Customer360/
├── data/customer_360_ml_workshop.csv
├── src/
│   ├── common.py                 # shared paths, column lists, helpers
│   ├── part_a_eda.py             # Part A  - EDA
│   ├── part_b_preprocessing.py   # Part B  - preprocessing pipeline
│   ├── part_c_classification.py  # Part C  - churn models, tuning, plots
│   ├── part_d_regression.py      # Part D  - revenue models
│   ├── part_e_clustering.py      # Part E  - K-Means, Agglomerative, DBSCAN
│   ├── part_f_pca_anomalies.py   # Part F  - PCA, Isolation Forest
│   ├── part_g_customer360.py     # Part G  - final Customer 360 table
│   └── main.py                   # runs everything
├── outputs/                      # plots and CSVs (including customer_360_predictions.csv)
├── notebooks/
├── REPORT.md
└── requirements.txt
```

## Setup
Python 3.7 with the pinned versions in `requirements.txt`:
```bash
conda create -n ml-workshop python=3.7.16 -y
conda activate ml-workshop
pip install -r requirements.txt
python check_environment.py
```

## Run
```bash
cd src
python main.py              # everything
python part_c_classification.py   # or one part at a time
```
In Google Colab, upload all `src/*.py` files to `/content` and the CSV to `/content/data/`.

## Key results
- Churn: ROC-AUC about 0.75 for the best models; month-to-month contract, tenure and satisfaction are the top predictors.
- Revenue: Gradient Boosting, R2 = 0.897, MAE about 369 EGP.
- Segments: 4 K-Means segments (High-Value At-Risk, Loyal, New Low-Spend, New High-Value).
- Retention priority: 325 customers, 38.2% actual churn, about 1.95M EGP of predicted revenue.
