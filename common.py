"""Shared settings and helpers used by every part of the workshop (Python 3.7 compatible)."""
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
try:
    get_ipython()            # running in Jupyter / Colab -> keep inline plots
except NameError:
    matplotlib.use("Agg")    # running as a .py script -> save figures only
import matplotlib.pyplot as plt
import pandas as pd

RANDOM_STATE = 42

try:
    # Running as a script: project root is one level above src/
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Running in a notebook (no __file__): use the current working folder
    BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_360_ml_workshop.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

ID_COL = "CustomerID"
CLASS_TARGET = "Churn"
REG_TARGET = "Future12MRevenueEGP"

NUMERIC_COLS = ["Age", "TenureMonths", "MonthlyUsageGB", "MonthlyChargeEGP",
                "NumServices", "SupportCalls6M", "LatePayments12M",
                "SatisfactionScore"]
CATEGORICAL_COLS = ["Region", "ContractType", "InternetType",
                    "PaperlessBilling", "AutoPay"]
SEGMENT_COLS = ["TenureMonths", "MonthlyUsageGB", "MonthlyChargeEGP",
                "NumServices", "SupportCalls6M", "LatePayments12M",
                "SatisfactionScore"]


def load_data():
    return pd.read_csv(DATA_PATH)


def save_fig(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print("Saved:", path)
