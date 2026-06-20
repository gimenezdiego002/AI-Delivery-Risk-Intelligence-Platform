# Phase 2 Baseline Summary

## Objective

Predict whether an Olist order will arrive after its estimated delivery date,
using only information available when the order is placed.

## Added features

- Seller-to-customer haversine distance based on median coordinates per zip
  prefix. This is straight-line distance and does not represent a driving or
  carrier route.
- Seller historical order volume.
- Seller historical late-delivery rate.
- Seller historical average review score.
- Product-category historical late-delivery rate.
- Product-category historical average delivery time.

Historical outcome features become available only after a prior delivery was
completed. Review values become available only after review creation. This
prevents the current order and future events from leaking into its features.

## Missing feature values

- 476 orders lack distance because at least one zip prefix has no valid match.
- 5,276 orders have no earlier completed seller outcome.
- 5,424 orders have no earlier seller review.
- 1,727 orders have no earlier completed category outcome.

These values remain missing and are handled by the fitted preprocessing
pipeline. They are not silently replaced with zero.

## Evaluation design

- Training orders: 71,124 orders placed before 2018-05-01.
- Test orders: 25,352 orders placed on or after 2018-05-01.
- Test late-delivery rate: 6.21%.

A chronological split represents deployment on future orders. A random split
would mix time periods and can hide temporal drift.

## Logistic Regression results

| Metric | Result |
|---|---:|
| Accuracy | 0.478 |
| Precision | 0.092 |
| Recall | 0.840 |
| F1 | 0.167 |
| True negatives | 10,789 |
| False positives | 12,988 |
| False negatives | 252 |
| True positives | 1,323 |

The class-balanced baseline detects 84% of truly late orders, but only 9.2% of
its late-risk alerts are correct. Accuracy is not a useful headline metric here
because only 6.21% of test orders are late; predicting every order as on-time
would produce high accuracy while detecting no delivery risk.

## Phase 3 recommendation

Compare Random Forest and XGBoost against this fixed chronological baseline.
Track parameters, precision, recall, F1, confusion matrices, and artifacts with
MLflow. Do not change the test period while comparing models.
