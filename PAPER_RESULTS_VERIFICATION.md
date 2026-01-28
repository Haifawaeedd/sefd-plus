# Paper Results Verification Document

**Source:** 2692212270-manuscript.pdf (Submitted to IEEE CCECE 2026)

---

## 📊 EXACT RESULTS FROM SUBMITTED PAPER

### A. Classification Performance (Table in Section V.A)

| Metric | Baseline | SEFD-Plus | 95% CI | p-value |
|--------|----------|-----------|---------|---------|
| **TPR** | 79.1% | 81.2% | [80.0%, 82.3%] | < 10⁻⁴ |
| **FPR** | 10.4% | 8.4% | [8.3%, 8.5%] | < 10⁻⁸⁵ |
| **F2** | 0.516 | 0.570 | [0.561, 0.578] | < 10⁻¹² |

### B. Confusion Matrices

**Baseline XGBoost:**
- TP: 4,901
- FP: 17,818
- TN: 153,144
- FN: 1,299
- FPR: 10.42%
- TPR: 79.1%

**SEFD-Plus (Automated):**
- TP: 4,589
- FP: 13,041
- TN: 157,921
- FN: 1,611
- FPR: 8.41%
- TPR: 81.2%

### C. Improvements

- **FP Reduction:** 19.3% (17,818 → 13,041 false positives)
- **TP Improvement:** 2.1% (4,913 → 5,045 true positives)

**NOTE:** There's a discrepancy in TP numbers:
- Table shows: 4,901 (baseline) and 4,589 (SEFD-Plus)
- Text shows: 4,913 → 5,045

Need to verify which is correct!

### D. Gray Zone Analysis (Table in Section V.B)

| Zone | Count | % | Fraud Rate | Enrichment |
|------|-------|---|------------|------------|
| **SAFE** | 144,149 | 81.4% | 2.8% | 0.80x |
| **GRAY** | 16,476 | 9.3% | 3.3% | 0.95x |
| **FLAGGED** | 16,537 | 9.3% | 31.7% | 9.06x |

**Total test transactions:** 177,162

### E. Cost-Benefit Analysis (Table in Section V.C)

**For 1M transactions/year (3.5% fraud rate):**

| Cost Component | Baseline | SEFD-Plus | Savings |
|----------------|----------|-----------|---------|
| False Positives ($100) | $1,781,800 | $1,304,100 | $477,700 |
| False Negatives ($500) | $649,500 | $338,500 | $311,000 |
| Human Review ($20) | $356,360 | $186,520 | $169,840 |
| **Total** | **$2,787,660** | **$1,829,120** | **$958,540** |

**ROI:** 34.4% cost reduction

---

## 🔧 CONFIGURATION FROM PAPER

### Dataset (Section IV)
- **Name:** IEEE-CIS Fraud Detection
- **Total transactions:** 590,540
- **Fraud rate:** 3.5%
- **Train/test split:** 70/30
- **Test transactions:** 177,162

### Baseline (Section IV)
- **Model:** XGBoost
- **Hyperparameters:**
  - n_estimators=100
  - max_depth=6
  - learning_rate=0.1
  - scale_pos_weight=27.6

### SEFD-Plus Configuration (Section IV)
- **Ensemble size:** 5 models
- **Seeds:** 42, 123, 456, 789, 1011
- **θ_low:** 0.05
- **θ_high:** Not explicitly stated (need to infer from results)

### Metrics (Section IV)
- TPR (True Positive Rate)
- FPR (False Positive Rate)
- F2 score
- Gray Zone size
- Enrichment
- Cost-benefit analysis

### Statistical Tests (Section IV)
- Bootstrap 95% CI (1000 samples)
- Fisher's exact test for FPR
- Permutation test for TPR

---

## ⚠️ DISCREPANCIES TO INVESTIGATE

### 1. True Positive Numbers
- **Confusion matrix shows:** TP_baseline = 4,901, TP_sefdplus = 4,589
- **Text shows:** 4,913 → 5,045
- **Which is correct?**

### 2. θ_high Value
- Paper states θ_low = 0.05
- θ_high is not explicitly stated
- Need to infer from FLAGGED zone definition

### 3. Cost Parameters
- Paper shows $100 for FP, $500 for FN, $20 for review
- But earlier context mentioned $25, $100, $5
- **Which is correct for the paper?**

---

## ✅ VERIFICATION CHECKLIST

- [ ] Verify TP/FP/TN/FN numbers match confusion matrices
- [ ] Verify FPR/TPR calculations from confusion matrices
- [ ] Verify F2 score calculation
- [ ] Verify Gray Zone percentages sum to 100%
- [ ] Verify cost calculations with stated parameters
- [ ] Verify ensemble seeds match paper
- [ ] Verify θ_low = 0.05 in code
- [ ] Determine correct θ_high value
- [ ] Verify statistical test implementations

---

## 🎯 KEY FORMULAS FROM PAPER

### Trust Score (Section III.B)
```
Trust(x) = 1 - p(x)  if σ(x) < θ_low
         = GRAY      if σ(x) ≥ θ_low
```

### Three-Zone Triage (Section III.A)
```
SAFE:    σ(x) < θ_low AND p(x) < 0.9 → Auto-approve
GRAY:    σ(x) ≥ θ_low → Human review
FLAGGED: σ(x) < θ_low AND p(x) ≥ 0.9 → Auto-block
```

### Uncertainty Quantification (Section III.A)
```
σ²(x) = Var(p₁(x), ..., p₅(x))
```

---

## 📝 NOTES

1. The paper uses **ensemble variance** for uncertainty, not standard deviation
2. Gray Zone is based on **uncertainty threshold**, not probability
3. FLAGGED zone requires both low uncertainty AND high probability
4. Cost analysis assumes 1M transactions/year
5. Statistical significance is very strong (p < 10⁻⁸⁵)

---

**Last Updated:** January 28, 2026  
**Verified By:** Automated extraction from submitted paper
