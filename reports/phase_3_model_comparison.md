# Phase 3 Model Comparison

## Experiment design

- Models: Logistic Regression, Random Forest, and XGBoost.
- Train period: before 2018-05-01 (71,124 orders).
- Test period: on/after 2018-05-01 (25,352 orders).
- Train late rate: 8.79%.
- Test late rate: 6.21%.
- Decision threshold: 0.50 for every model.
- MLflow tracking store: `sqlite:///C:/Users/gimen/AI-Delivery-Risk-Intelligence-Platform/mlflow.db`.

The same chronological test set was used for every model. A random split was
not used because deployment means predicting genuinely future orders.

## Class imbalance

Late orders are the minority class. Logistic Regression and Random Forest use
balanced class weights; XGBoost uses the training ratio of on-time to late
orders as `scale_pos_weight`. Accuracy is not a selection metric because a
model can appear accurate by predicting almost everything as on-time.

## Test metrics

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | False positives | False negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.092 | 0.840 | 0.167 | 0.698 | 0.124 | 12,988 | 252 |
| random_forest | 0.112 | 0.081 | 0.094 | 0.607 | 0.088 | 1,004 | 1,448 |
| xgboost | 0.124 | 0.166 | 0.142 | 0.650 | 0.102 | 1,855 | 1,313 |

Metric meanings:

- Precision: among risk alerts, the fraction that were truly late.
- Recall: among truly late orders, the fraction the model caught.
- F1: a balance between precision and recall.
- ROC-AUC: ranking quality across classification thresholds.
- PR-AUC: minority-class ranking quality, sensitive to false alerts.

## Best model

**logistic_regression** was selected by highest F1, with recall and then
precision used as tie-breakers. F1 matches the business need to catch delays
without overwhelming operations with false alarms.

## Feature importance

Because the overall winner is not necessarily tree-based, this table reports
importance from the best-performing tree candidate: **xgboost**.
Its transformed one-hot columns were aggregated back to the original business
features. These importances do not explain the Logistic Regression coefficients.

| Feature | Importance |
|---|---:|
| product_category_name | 0.3278 |
| customer_state | 0.3104 |
| seller_state | 0.1161 |
| order_month | 0.0323 |
| payment_type | 0.0231 |
| seller_count | 0.0197 |
| seller_historical_avg_review_score | 0.0186 |
| seller_customer_distance_km | 0.0177 |
| seller_historical_late_rate | 0.0174 |
| category_historical_late_rate | 0.0164 |

The chart is saved at `reports/phase_3_feature_importance.png`.

## Known limitations

- Distance is straight-line haversine distance, not carrier-route distance.
- Olist is historical Brazilian marketplace data and may not generalize.
- Class weighting improves recall but can create many false positives.
- Feature importance shows model reliance, not causal impact.
- Aggregated one-hot importance can favor high-cardinality categories.
- No threshold tuning or extensive hyperparameter search was performed.

## Recommended Phase 4

Add SHAP explainability, a reusable prediction explanation function, and the
first scoped tools: `predict_delay_risk(order_id)`, `explain_risk(order_id)`,
and `get_seller_history(seller_id)`.
