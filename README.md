# Credit Card Fraud Detection (Imbalanced Classification)

Detecting fraudulent credit card transactions when fraud is only **0.17%** of the data. The project covers EDA, leakage-safe splitting and scaling, imbalance handling (class weights, SMOTE, undersampling), model comparison, a cost-based decision threshold, and a Power BI dashboard for exploring the results.

## Results at a glance

On a held-out test set (56,746 transactions, 95 frauds), the final XGBoost model with a cost-optimised threshold:

| Metric | Value |
|---|---|
| Frauds caught | **79 of 95 (83% recall)** |
| Precision | 0.54 (about 0.8 false alarms per fraud caught) |
| Transactions flagged for review | **0.26%** (146 of 56,746) |
| PR-AUC | 0.82 |
| Total cost vs no model* | **1,734 vs 9,500 (82% lower)** |

\*Illustrative cost assumption: a missed fraud costs 50x a false alarm (100 vs 2 cost units). These are assumptions, not real business data.

## Dataset

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). European cardholder transactions over two days: 284,807 rows, 492 frauds (0.172%). Features `V1`-`V28` are PCA components (anonymised); `Time`, `Amount` and `Class` are original.

The data file is **not included** in this repo (too large for GitHub). Download `creditcard.csv` from Kaggle and place it in the project folder.

> Dal Pozzolo, Caelen, Johnson and Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE CIDM, 2015.

## Approach

| Notebook | What it does |
|---|---|
| `1_eda_split.ipynb` | EDA (data quality, class imbalance, amount and time patterns, feature-fraud correlations), then removes duplicates, makes a stratified 80/20 split and scales with RobustScaler fitted on train only |
| `2_base_model.ipynb` | Logistic Regression with class weights, 5-fold stratified cross-validation |
| `3_model_threshold.ipynb` | Compares SMOTE, undersampling, Random Forest and XGBoost; threshold tuning |
| `4_final_eval.ipynb` | Locks the threshold on validation, scores once on the test set, feature importance |
| `5_powerbi_export.ipynb` | Rebuilds the test rows with original amounts and hours, attaches the model's predictions, and exports the tables used by the dashboard |

## Dashboard

A three-page Power BI dashboard (`fraud_dashboard.pbix`) built on the model results.

1. **Fraud overview:** fraud rate by hour of day and by amount bucket, using all 283,726 transactions.
2. **Model performance:** a threshold selector showing frauds caught and missed, false alarms, precision, recall and cost at any threshold, plus the confusion matrix at the chosen threshold (0.33).
3. **Alert review:** the flagged and missed transactions with their real amounts, hours and fraud probabilities.

![Fraud overview](images/1.png)
![Model performance](images/2.png)
![Alert review](images/3.png)

Notes: hours are relative to the first transaction, not clock time. The model pages use the test set only (56,746 transactions, 95 frauds). Cost figures are illustrative assumptions.

## Key findings

- **Accuracy is meaningless here.** Predicting "not fraud" every time is 99.8% accurate and catches nothing, so models were judged on precision, recall and PR-AUC.
- **Duplicates:** 1,081 duplicate rows (19 fraud) were removed to avoid train/test leakage, leaving 283,726 rows and 473 frauds.
- **Fraud tends to be smaller:** median amount 9.25 for fraud vs 22 for legitimate transactions.
- **Time pattern:** fraud rate peaked near 1.7% in one hour of the day (about 10x the overall rate), measured relative to the first transaction.
- **Threshold tuning mattered most.** The baseline at the default 0.5 threshold had precision 0.06 and recall 0.92. At a tuned threshold the same type of model reached precision about 0.88 and recall about 0.82 on validation (measured on different evaluation splits, so treat the comparison as indicative).
- **Model comparison (validation PR-AUC):**

| Model | PR-AUC |
|---|---|
| XGBoost | 0.882 |
| Random Forest | 0.862 |
| Logistic + SMOTE | 0.787 |
| Logistic (class weights) | 0.787 |
| Logistic + undersampling | 0.490 |

- **SMOTE did not beat plain class weights**, and undersampling was clearly worst because it discards over 99% of legitimate rows.
- Top features by importance: V14, V10, V12.

## Limitations

- Only **95 frauds in the test set**, so recall is an estimate (roughly 74% to 89% plausible range). Differences of a few frauds are within noise.
- The cost-based threshold (0.33) was only marginally cheaper than the default 0.5 on test (about 7%), and it was chosen on a validation set with around 95 frauds. The dashboard's cost curve is fairly flat across a wide range of thresholds, so the exact choice matters less than having a sensible one.
- Features are PCA-anonymised, so results can't be explained in real-world terms and there is little room for feature engineering.
- Two days of data, so no check for drift over time.
- Cost figures are illustrative assumptions. Every fraud is treated as costing the same, regardless of its amount.

## How to run

```bash
pip install -r requirements.txt
```

1. Download `creditcard.csv` from Kaggle into the project folder.
2. Run the notebooks in order (`1` to `5`). Notebook 1 creates `train.csv` and `test.csv`, notebooks 2 to 4 use them, and notebook 5 builds the data for the dashboard.

## Next steps

- SHAP analysis to explain individual predictions
- A cost model that uses each transaction's own amount instead of a flat cost per missed fraud

## Author

Rutvik Randive | [LinkedIn](https://www.linkedin.com/in/rutvik-randive-ab08a5232/) | [Portfolio projects on GitHub](https://github.com/samael-420)
