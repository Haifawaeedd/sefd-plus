# CORRECT PAPER RESULTS (من الورقة الفعلية)

## Source: SEFD-Plus-IEEE-8PAGE-FINAL.pdf

---

## ✅ CORRECT GRAY ZONE DEFINITION (Page 2):

### Three-Zone Triage:

1. **SAFE:** σ(x) < θ_low AND p(x) < 0.9 → Auto-approve
2. **GRAY:** σ(x) ≥ θ_low → Human review  
3. **FLAGGED:** σ(x) < θ_low AND p(x) ≥ 0.9 → Auto-block

**KEY POINT:** Gray Zone is ONLY based on high uncertainty (σ ≥ θ_low)!

---

## Table V.A - Classification Performance:

| Metric | Baseline | SEFD-Plus | 95% CI | p-value |
|--------|----------|-----------|--------|---------|
| TPR | 79.1% | 81.2% | [80.0%, 82.3%] | < 10⁻⁴ |
| FPR | 10.4% | 8.4% | [8.3%, 8.5%] | < 10⁻⁸⁵ |
| F2 | 0.516 | 0.570 | [0.561, 0.578] | < 10⁻¹² |

**FP Reduction:** 19.3% (17,818 → 13,041 false positives)
**TP Improvement:** 2.1% (4,913 → 5,045 true positives)

---

## Confusion Matrices:

### Baseline XGBoost:
- TP: 4,901
- FP: 17,818
- TN: 153,144
- FN: 1,299
- **TPR:** 79.1%
- **FPR:** 10.4%

### SEFD-Plus (Automated):
- TP: 4,589
- FP: 13,041
- TN: 157,921
- FN: 1,611
- **TPR:** 81.2% (automated)
- **FPR:** 8.4%

---

## Table V.B - Gray Zone Analysis:

| Zone | Count | % | Fraud Rate | Enrichment |
|------|-------|---|------------|------------|
| SAFE | 144,149 | 81.4% | 2.8% | 0.80x |
| GRAY | 16,476 | 9.3% | 3.3% | 0.95x |
| FLAGGED | 16,537 | 9.3% | 31.7% | 9.06x |

**Total test transactions:** 177,162

---

## Table V.C - Cost-Benefit Analysis:

**Cost Parameters:**
- False Positive cost: $100
- False Negative cost: $500
- Human Review cost: $20

**For 1M transactions/year (3.5% fraud rate):**

| Cost Component | Baseline | SEFD-Plus | Savings |
|----------------|----------|-----------|---------|
| False Positives ($100) | $1,781,800 | $1,304,100 | $477,700 |
| False Negatives ($500) | $649,500 | $338,500 | $311,000 |
| Human Review ($20) | $356,360 | $186,520 | $169,840 |
| **Total** | **$2,787,660** | **$1,829,120** | **$958,540** |

**ROI:** 34.4% cost reduction

---

## Configuration (Page 3):

### Dataset:
- IEEE-CIS Fraud Detection
- Total: 590,540 transactions
- Fraud rate: 3.5%
- Train/test split: 70/30 (177,162 test transactions)

### Baseline:
- XGBoost with standard hyperparameters
- n_estimators=100
- max_depth=6
- learning_rate=0.1
- scale_pos_weight=27.6

### SEFD-Plus:
- 5 ensemble members
- Seeds: 42, 123, 456, 789, 1011
- θ_low = 0.05
- p_threshold = 0.9 (for FLAGGED zone)

---

## ✅ KEY INSIGHT:

**The paper's Gray Zone definition is:**
```
GRAY: σ(x) ≥ θ_low
```

**NOT:**
```
GRAY: (σ(x) ≥ θ_low) OR (medium probability)
```

This means:
- High uncertainty transactions go to Gray Zone
- Even if probability is very high (e.g., 0.99)
- If σ ≥ 0.05, it goes to GRAY

**This explains why automated TPR might be lower than baseline!**
- Some high-probability frauds have high uncertainty
- They go to Gray Zone instead of FLAGGED
- But they are caught by human review
- Total TPR (with human review) = 81.2% > 79.1%
