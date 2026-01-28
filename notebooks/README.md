# SEFD-Plus Reproducibility Notebooks

This directory contains Jupyter notebooks for reproducing all results from the paper.

---

## 📓 Available Notebooks

### 1. `SEFD_Plus_COMPLETE.ipynb` ⭐ **RECOMMENDED**

**Complete notebook with all corrections and enhancements.**

**What's included:**
- ✅ All results from paper (Table V.A, V.B, V.C)
- ✅ Corrected cost parameters ($100/$500/$20)
- ✅ Clarified TP/FP/FN numbers
- ✅ Ablation study (different θ_low values)
- ✅ Calibration analysis (reliability curve)
- ✅ SHAP feature importance (Gray Zone)
- ✅ Statistical significance tests
- ✅ Bootstrap confidence intervals
- ✅ Cost-benefit analysis
- ✅ Ready for Google Colab

**Expected runtime:** ~60 minutes on CPU

**Open in Colab:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haifawaeedd/sefd-plus/blob/main/notebooks/SEFD_Plus_COMPLETE.ipynb)

---

### 2. `SEFD_Plus_Reproducibility.ipynb`

**Original basic notebook (for reference).**

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Click the "Open in Colab" badge above
2. Upload `train_transaction.csv` from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)
3. Run all cells (Runtime → Run all)
4. Results will match paper within ±1%

### Option 2: Local Jupyter

```bash
# Install dependencies
pip install -r ../requirements.txt

# Start Jupyter
jupyter notebook SEFD_Plus_COMPLETE.ipynb

# Upload dataset and run
```

---

## 📊 Expected Results

| Metric | Baseline | SEFD-Plus | Paper Value | Match |
|--------|----------|-----------|-------------|-------|
| TPR | 79.1% | 81.2% | 81.2% | ✅ |
| FPR | 10.4% | 8.4% | 8.4% | ✅ |
| F2 | 0.516 | 0.570 | 0.570 | ✅ |
| Gray Zone | - | 9.3% | 9.3% | ✅ |
| Savings | - | $958,540 | $958,540 | ✅ |

---

## 🔧 Configuration

All hyperparameters match the paper:

```python
# Ensemble
ENSEMBLE_SIZE = 5
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1011]

# Thresholds
SIGMA_THRESHOLD = 0.05  # θ_low
PROB_THRESHOLD = 0.9    # p_threshold

# Costs (from paper Table V.C)
COST_FALSE_POSITIVE = 100
COST_FALSE_NEGATIVE = 500
COST_HUMAN_REVIEW = 20
```

---

## ⚠️ Important Notes

1. **Random Seed:** Fixed to 42 for reproducibility
2. **Library Versions:** See `requirements.txt`
3. **Dataset:** IEEE-CIS (590,540 transactions)
4. **Runtime:** ~60 min on CPU, ~15 min on GPU
5. **Memory:** ~4GB RAM required

---

## 📧 Support

For questions:
- **Author:** Haifaa Owayed
- **Email:** howay035@uottawa.ca
- **GitHub Issues:** https://github.com/Haifawaeedd/sefd-plus/issues

---

## 📚 Citation

```bibtex
@inproceedings{owayed2026sefdplus,
  title={SEFD-Plus: Uncertainty-Aware Fraud Detection with Human-in-the-Loop Governance},
  author={Owayed, Haifaa},
  booktitle={IEEE CCECE},
  year={2026},
  note={Paper ID: 2692212270}
}
```
