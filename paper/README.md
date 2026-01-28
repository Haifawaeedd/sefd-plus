# Paper: SEFD-Plus

## 📄 Submitted Manuscript

**File:** `2692212270-manuscript.pdf`

**Title:** SEFD-Plus: Uncertainty-Aware Fraud Detection with Human-in-the-Loop Governance

**Conference:** IEEE Canadian Conference on Electrical and Computer Engineering (CCECE) 2026

**Paper ID:** 2692212270

**Track:** Machine Learning, Data Analytics and Artificial Intelligence

---

## 👤 Author Information

**Author:** Haifaa Owayed  
**Email:** howay035@uottawa.ca  
**Alternative Email:** haifawaeed2015@gmail.com  
**Institution:** University of Ottawa  
**Department:** Electrical Engineering and Computer Science

---

## 📊 Key Results (from submitted paper)

### Performance Metrics

| Metric | Baseline | SEFD-Plus | Improvement |
|--------|----------|-----------|-------------|
| **False Positive Rate (FPR)** | 10.42% | 8.41% | **-19.3%** |
| **True Positive Rate (TPR)** | 79.05% | 81.16% | **+2.11%** |
| **F2 Score** | 0.5157 | 0.5701 | **+0.0544** |
| **False Positives** | 17,818 | 13,041 | **-4,777** |

### Statistical Significance

- **Fisher's Exact Test:** p < 10⁻⁸⁵
- **Confidence Level:** 95%
- **Bootstrap Iterations:** 1,000

### Gray Zone Analysis

- **Size:** 16,476 transactions (9.3% of test set)
- **Fraud Rate:** 3.3%
- **Enrichment Factor:** 0.95x

### Cost-Benefit Analysis

- **Annual Baseline Cost:** $2,787,660
- **Annual SEFD-Plus Cost:** $1,829,120
- **Annual Savings:** $958,540 (34.4% reduction)

---

## 🎯 Abstract

Fraud detection systems face a fundamental trade-off between automation and accuracy. Traditional binary classifiers (approve/reject) struggle with uncertain cases, leading to high false positive rates that frustrate legitimate customers. This paper introduces SEFD-Plus, a novel framework that combines uncertainty quantification with human-in-the-loop governance through a "Gray Zone" mechanism.

SEFD-Plus uses an ensemble of XGBoost models to quantify prediction uncertainty and routes uncertain transactions to human review. Evaluated on the IEEE-CIS Fraud Detection dataset (590,540 transactions), SEFD-Plus achieves a 19.3% reduction in false positives while improving true positive rate by 2.11 percentage points, with statistical significance (p < 10⁻⁸⁵). The system routes only 9.3% of transactions to human review, resulting in $958,540 annual cost savings (34.4% reduction).

---

## 🔬 Methodology

### 1. Baseline System

- **Model:** Single XGBoost classifier
- **Threshold:** 0.5 (standard practice)
- **Decision:** Binary (approve/reject)

### 2. SEFD-Plus System

- **Ensemble:** 5 XGBoost models (different random seeds)
- **Uncertainty:** Ensemble variance (epistemic uncertainty)
- **Gray Zone Thresholds:**
  - θ_low = 0.05 (SAFE zone - automatic approval)
  - θ_high = 0.95 (FLAGGED zone - automatic rejection)
  - [θ_low, θ_high] (GRAY zone - human review)

### 3. Dataset

- **Name:** IEEE-CIS Fraud Detection
- **Source:** Kaggle
- **Total Transactions:** 590,540
- **Test Set:** 177,162 (30%)
- **Fraud Rate:** 3.5%

---

## 📈 Contributions

1. **Novel Gray Zone Mechanism:** Routes uncertain cases to human review instead of forcing binary decisions

2. **Uncertainty Quantification:** Uses ensemble variance to measure prediction confidence

3. **Cost-Sensitive Framework:** Balances automation benefits with human oversight costs

4. **Empirical Validation:** Demonstrates significant improvements on real-world dataset with rigorous statistical testing

5. **Reproducibility:** Complete code and methodology provided for verification

---

## 📚 Citation

```bibtex
@inproceedings{owayed2026sefdplus,
  title={SEFD-Plus: Uncertainty-Aware Fraud Detection with Human-in-the-Loop Governance},
  author={Owayed, Haifaa},
  booktitle={IEEE Canadian Conference on Electrical and Computer Engineering (CCECE)},
  year={2026},
  organization={IEEE},
  note={Paper ID: 2692212270}
}
```

---

## 🔗 Related Materials

- **Reproducibility Notebook:** `../notebooks/SEFD_Plus_Reproducibility.ipynb`
- **Source Code:** Available in this repository
- **Dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
- **GitHub Repository:** https://github.com/Haifawaeedd/sefd-plus

---

## 📧 Contact

For questions about the paper or methodology:

**Email:** howay035@uottawa.ca  
**Alternative:** haifawaeed2015@gmail.com

For code or reproducibility issues:

**GitHub Issues:** https://github.com/Haifawaeedd/sefd-plus/issues

---

## 📅 Timeline

- **Submission Date:** January 2026
- **Conference Date:** IEEE CCECE 2026
- **Paper ID:** 2692212270
- **Status:** Under Review

---

## ⚖️ License

The paper content is copyright © 2026 IEEE. All rights reserved.

The accompanying code is released under MIT License (see `../LICENSE`).
