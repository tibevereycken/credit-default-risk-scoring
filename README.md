# Credit Risk Scoring & Default Prediction Engine

A machine learning pipeline for corporate credit scoring and default risk prediction under severe class imbalance, benchmarked across decision trees, random forests, and gradient-boosted ensembles.

## Business Context & Challenge
Predicting default risk (`IN_DEFAULT_FLG`) over a 24-month horizon for commercial borrowers. The primary quantitative challenges include:
- **Extreme Class Imbalance:** Defaults represent only ~4.39% of historical observations.
- **High-Dimensional Categoricals:** Geographical and sector features requiring robust encoding without overfitting.
- **Asymmetric Cost of Errors:** In commercial lending, false negatives (granting credit to a defaulting client) incur substantially higher costs than false positives.

## Methodology & Pipeline Architecture

1. **Stratified Partitioning:** Train, validation, and test splits generated via stratified sampling to prevent distribution shift and data leakage.
2. **Feature Engineering & Preprocessing:**
   - **Weight of Evidence (WoE) Encoding:** Transforms high-cardinality categorical variables.
   - **Robust Scaling:** Standardizes skewed continuous indicators using median and interquartile ranges (IQR) to neutralize financial outliers.
   - **Imputation:** Median replacement for non-normal continuous features; mode replacement for categorical features.
3. **Model Ensembles & Cost-Sensitive Learning:**
   - Class-weighted Tree Classifiers (`min_samples_leaf` constraints)
   - Random Forests with balanced bagging
   - LightGBM and Extreme Gradient Boosting (XGBoost) configured with dynamic `scale_pos_weight` to penalize false negatives.

## Quickstart

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)tibevereycken/credit-default-risk-scoring.git
cd credit-default-risk-scoring

# Install dependencies
pip install -r requirements.txt

# Execute the pipeline
python main.py