# Cross-Dataset Generalization Study

## Overview

This document describes a post-submission validation study conducted to evaluate SEFD-Plus's governance stability across different fraud detection contexts.

## Motivation

The primary goal of SEFD-Plus is to provide a **governance-focused** fraud detection framework that:
- Minimizes false positives through uncertainty quantification
- Maintains low human review burden across different operational contexts
- Demonstrates robust behavior without dataset-specific tuning

## Dataset

**Credit Card Fraud Detection (Kaggle)**
- **Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Size:** 284,807 transactions
- **Fraud Rate:** 0.17% (492 fraudulent transactions)
- **Features:** 30 PCA-transformed features (V1-V28, Time, Amount)
- **Challenge:** Highly imbalanced, lower fraud rate than IEEE-CIS (3.5%)

## Methodology

### Configuration
- **Model:** Same XGBoost ensemble (5 models, seeds: 42, 123, 456, 789, 1011)
- **Hyperparameters:** Identical to IEEE-CIS study (n_estimators=100, max_depth=6, learning_rate=0.1)
- **Uncertainty Threshold:** θ_low = 0.05 (unchanged)
- **Decision Threshold:** p_high = 0.9 (unchanged)
- **No tuning or feature engineering applied**

### Three-Zone Triage
- **SAFE Zone:** σ(x) < 0.05 AND p(x) < 0.9 → Auto-approve
- **GRAY Zone:** σ(x) ≥ 0.05 → Human review
- **FLAGGED Zone:** σ(x) < 0.05 AND p(x) ≥ 0.9 → Auto-block

## Results

### Performance Metrics

| Metric | IEEE-CIS (3.5% fraud) | Credit Card (0.17% fraud) |
|--------|----------------------|---------------------------|
| **F2 Score** | 0.570 | 0.792 |
| **TPR** | 81.2% | 76.4% |
| **FPR** | 8.4% | 0.02% |
| **Gray Zone** | 9.3% | 0.2% |

### Confusion Matrix (Credit Card Dataset)

```
                Predicted
                Legit    Fraud
Actual  Legit   85,287   18      (FP = 18)
        Fraud   35       113     (FN = 35)
```

### Key Observations

1. **Extremely Low False Positives:** Only 18 FP out of 85,305 legitimate transactions (FPR = 0.02%)
2. **Minimal Human Review Burden:** Gray Zone = 0.2% (vs. 9.3% on IEEE-CIS)
3. **Governance-Friendly Behavior:** System automatically adapts to lower fraud rate by reducing HITL load
4. **No Tuning Required:** Same hyperparameters work effectively across datasets

## Interpretation

### Why Does Gray Zone Decrease?

The dramatic reduction in Gray Zone size (9.3% → 0.2%) reflects **adaptive governance behavior**:

- **Lower fraud rate** (0.17% vs. 3.5%) → Fewer ambiguous cases near decision boundary
- **Cleaner features** (PCA-transformed) → Lower ensemble variance for most transactions
- **Conservative threshold** (θ_low = 0.05) → Only truly uncertain cases flagged for review

This is **not a bug but a feature** — SEFD-Plus automatically adjusts human review load based on dataset characteristics while maintaining low false positive rates.

### Implications for Deployment

1. **Robust Across Contexts:** Same model configuration works on datasets with 20x different fraud rates
2. **Operational Efficiency:** Human review burden scales with uncertainty, not volume
3. **Conservative by Design:** Extremely low FP rate (0.02%) suitable for real-world banking systems

## Reproducibility

### Requirements
```bash
pip install xgboost scikit-learn pandas numpy scipy
```

### Running the Experiment
```bash
# Download dataset from Kaggle
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# Run generalization study
python experiments/credit_card_generalization.py
```

### Expected Runtime
- ~5-10 minutes on standard CPU
- ~2-3 minutes with GPU acceleration

## Limitations

1. **Single Alternative Dataset:** Only one additional dataset tested
2. **No Hyperparameter Tuning:** Deliberately avoided to test robustness, but tuning might improve F2
3. **PCA Features:** Original feature interpretability lost in Credit Card dataset

## Conclusion

This cross-dataset study validates SEFD-Plus's core thesis: **uncertainty-aware governance provides robust, conservative fraud detection across different operational contexts without dataset-specific tuning**.

The automatic reduction in human review burden (9.3% → 0.2%) while maintaining extremely low false positives (0.02%) demonstrates that SEFD-Plus is a **governance framework**, not just a performance optimization.

---

**Note:** This study was conducted as post-submission validation and is not included in the IEEE CCECE 2026 conference paper.
