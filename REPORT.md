# Customer 360 Intelligence - Project Report

All numbers come from `python src/main.py` (random_state=42). Plots and CSVs are in `outputs/`.

## 1. EDA summary (Part A)
1. 3,000 customers, 16 columns. `CustomerID` is an identifier and is never used as a feature.
2. Churn is imbalanced: 544 Yes (18.1%) vs 2,456 No.
3. Missing values: `InternetType` 45, `MonthlyUsageGB` 75, `SatisfactionScore` 60 (all under 2.5%), handled by imputation.
4. Churn by contract: month-to-month 26.6%, one-year 8.7%, two-year 5.1%.
5. Churn by AutoPay: without 23.4%, with 14.0%.
6. `Future12MRevenueEGP` is excluded when predicting churn (target leakage: it is measured after the period and depends on whether the customer stays).

## 2. Churn classification (Parts B and C)
Preprocessing lives in a `Pipeline` (median/most-frequent imputation, scaling, one-hot with `handle_unknown="ignore"`), fitted on training data only. Stratified 80/20 split.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.813 | 0.459 | 0.156 | 0.233 | 0.764 |
| Naive Bayes | 0.750 | 0.374 | 0.560 | 0.449 | 0.761 |
| Gradient Boosting | 0.808 | 0.412 | 0.128 | 0.196 | 0.732 |
| Random Forest | 0.820 | 0.519 | 0.128 | 0.206 | 0.725 |
| SVM | 0.820 | 0.522 | 0.110 | 0.182 | 0.708 |
| Decision Tree | 0.802 | 0.400 | 0.183 | 0.252 | 0.694 |
| KNN | 0.822 | 0.562 | 0.083 | 0.144 | 0.665 |

- Accuracy is misleading: always predicting "No" scores about 0.82. KNN has the best accuracy and the worst recall (0.083).
- Tuned Random Forest (`GridSearchCV`, 5-fold, ROC-AUC): `max_depth=8, min_samples_leaf=20, max_features=sqrt, n_estimators=500`, CV ROC-AUC 0.758, test ROC-AUC 0.753.
- Top predictors: month-to-month contract (0.174), tenure (0.146), satisfaction (0.123), monthly charge (0.087), usage (0.058).
- Figures: `classification_confusion_matrices_all.png`, `classification_roc_all.png`, `classification_feature_importance.png`.

Threshold analysis (tuned RF, test set):

| Threshold | Precision | Recall | F1 | Flagged |
|---|---|---|---|---|
| 0.20 | 0.340 | 0.734 | 0.465 | 235 |
| 0.25 | 0.358 | 0.532 | 0.428 | 162 |
| 0.35 | 0.415 | 0.202 | 0.272 | 53 |
| 0.50 | 1.000 | 0.009 | 0.018 | 1 |

All models reach a ROC-AUC of about 0.75, which looks like the ceiling of this synthetic data.

## 3. Revenue regression (Part D)

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Gradient Boosting | 368.6 | 465.0 | 0.897 |
| Ridge | 378.4 | 474.9 | 0.892 |
| Linear Regression | 378.4 | 474.9 | 0.892 |
| Lasso | 378.7 | 475.1 | 0.892 |
| Random Forest | 386.4 | 489.8 | 0.885 |
| Decision Tree | 419.1 | 534.7 | 0.864 |
| KNN Regressor | 575.9 | 732.6 | 0.744 |

Gradient Boosting is best; linear models are nearly as good, so revenue is mostly driven by simple linear relationships. Plot: `regression_actual_vs_predicted.png`.

## 4. Segmentation (Part E)
K-Means was tested for k=2..8 (`kmeans_selection.png/csv`). Silhouette is highest at k=2 (0.179) but that only splits high vs low spend, and every silhouette is below 0.2 (overlapping groups). I selected **k=4** as the smallest k that gives distinct, actionable segments.

| Segment | Size | Tenure | Charge (EGP) | Late payments | Churn | Revenue (EGP) |
|---|---|---|---|---|---|---|
| High-Value At-Risk | 518 | 21.6 | 526 | 2.69 | 25.7% | 5,588 |
| Loyal | 462 | 53.8 | 472 | 1.04 | 8.2% | 5,383 |
| New Low-Spend | 1,041 | 19.8 | 379 | 0.85 | 17.4% | 4,196 |
| New High-Value | 979 | 20.4 | 589 | 0.59 | 19.6% | 6,470 |

| Algorithm | Clusters | Silhouette | Notes |
|---|---|---|---|
| K-Means | 4 | 0.146 | clearest segments |
| Agglomerative (Ward) | 4 | 0.082 | partly agrees with K-Means (ARI 0.39) |
| DBSCAN | 1 | n/a | 49 noise points; no density-separated groups |

DBSCAN label `-1` = noise: a point not dense enough to join any cluster. It is not a segment.
A high silhouette alone does not make a cluster useful; it must also be stable, interpretable and actionable.

## 5. PCA and anomalies (Part F)
- PC1 explains 26.2% and PC2 15.0% of the variance (41.2% together). PC1 reflects spend and number of services; PC2 mixes usage, late payments and satisfaction. Plot: `pca_segments.png`.
- Isolation Forest (contamination 3%) flagged 90 customers; the 15 most unusual are in `top_anomalies.csv`, each with the feature that makes it extreme. Anomalies churn at 22.2% vs 18.0% for normal customers.
- Anomalies are statistically unusual behaviour, not fraud. They need business review (heavy legitimate users, billing errors, data issues).

## 6. Customer 360 output (Part G)
`outputs/customer_360_predictions.csv` has CustomerID, ChurnProbability, PredictedRevenue12M_EGP, Segment, AnomalyFlag and RetentionPriority for all 3,000 customers. Probabilities and revenue are out-of-fold (5-fold CV), so no customer is scored by a model that trained on them.

**Retention-priority rule:** `ChurnProbability >= 0.25` AND `PredictedRevenue12M >= median (5,314 EGP)`. The threshold balances recall (about 53%) and precision (about 36%); the revenue condition spends the retention budget on customers worth keeping.

Result: 325 priority customers (10.8%), actual churn 38.2% inside the group vs 18.1% overall, about 1.95 million EGP of predicted revenue. They are mostly New High-Value (236) and High-Value At-Risk (76).

## 7. Business recommendations
1. Move month-to-month customers to one- or two-year contracts with a discount: churn is 26.6% vs 5.1% for two-year.
2. Push AutoPay enrolment (churn 14.0% with AutoPay vs 23.4% without).
3. Run a targeted retention campaign on the 325 priority customers instead of all customers.
4. Build an onboarding programme for New High-Value customers (979 customers, 19.6% churn, highest revenue); tenure is a top churn predictor.
5. Fix billing and payment friction for High-Value At-Risk customers (2.7 late payments on average, 25.7% churn).

## 8. Challenge questions
1. **Threshold 0.50 to 0.35 or 0.70:** lowering it flags more customers and raises recall but lowers precision; raising it does the opposite. With our model, recall falls from 0.73 (0.20) to 0.20 (0.35).
2. **Recall over precision:** when losing a customer costs far more than a retention offer.
3. **Why drop CustomerID:** it is a unique label with no real signal; the model would memorise it and not generalise.
4. **Why one-hot encoding:** models need numbers, and plain integer codes would imply a false order between categories.
5. **Scale-sensitive:** KNN, SVM, logistic/linear models, K-Means, PCA, DBSCAN. Trees and forests are not.
6. **High silhouette = useful cluster?** No. It must also be stable, interpretable and actionable.
7. **Drift monitoring:** track input distributions (PSI), predicted probability distribution, and live AUC/recall once true churn is known; retrain on a schedule or when drift passes a threshold.
8. **Risks of automatic offers:** wasted discounts on false positives, unfair treatment of groups, customers learning to game the system, and no human check. Keep a human in the loop and measure with a control group.
