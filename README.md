# SEFD-Plus: Reproducibility Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Conference](https://img.shields.io/badge/IEEE-CCECE%202026-green.svg)](https://ccece2026.ieee.ca/)

This repository contains the reproducibility package for the paper:

**"SEFD-Plus: Uncertainty-Aware Fraud Detection with Human-in-the-Loop Governance"**

Submitted to **IEEE CCECE 2026** (Paper ID: 2692212270)

---

## 📋 Overview

SEFD-Plus is a novel fraud detection system that combines:
- **Uncertainty Quantification** using ensemble learning
- **Gray Zone Mechanism** for routing uncertain cases to human review
- **Cost-Sensitive Decision Making** balancing automation and human oversight

### Key Results

| Metric | Baseline | SEFD-Plus | Improvement |
|--------|----------|-----------|-------------|
| **False Positive Rate** | 10.42% | 8.41% | **-19.3%** |
| **True Positive Rate** | 79.05% | 81.16% | **+2.11%** |
| **F2 Score** | 0.5157 | 0.5701 | **+0.0544** |
| **Annual Cost Savings** | - | - | **$958,540** |
| **Gray Zone** | 0% | 9.3% | Human review |

**Statistical Significance:** p < 10⁻⁸⁵ (Fisher's exact test)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended
- ~2GB disk space for dataset

### Installation

```bash
# Clone the repository
git clone https://github.com/Haifawaeedd/sefd-plus.git
cd sefd-plus

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Download the **IEEE-CIS Fraud Detection** dataset from Kaggle:

1. Go to: https://www.kaggle.com/c/ieee-fraud-detection/data
2. Download `train_transaction.csv` (~470 MB)
3. Place it in the `data/` directory

```bash
# Directory structure
data/
  └── train_transaction.csv
```

### Running the Notebook

```bash
# Launch Jupyter Notebook
jupyter notebook notebooks/SEFD_Plus_Reproducibility.ipynb
```

Or use **Google Colab** (no local setup required):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haifawaeedd/sefd-plus/blob/main/notebooks/SEFD_Plus_Reproducibility.ipynb)

**Expected Runtime:** ~1 hour (on standard CPU)

---

## 📁 Repository Structure

```
sefd-plus-reproducibility/
├── paper/
│   ├── 2692212270-manuscript.pdf          # Submitted paper (IEEE CCECE 2026)
│   └── README.md                           # Paper information and citation
├── notebooks/
│   └── SEFD_Plus_Reproducibility.ipynb    # Main reproducibility notebook
├── data/
│   └── README.md                           # Dataset download instructions
├── results/
│   ├── sefd_plus_results.json             # Experimental results (generated)
│   └── models/                             # Trained models (generated)
├── docs/
│   ├── METHODOLOGY.md                      # Detailed methodology
│   ├── RESULTS.md                          # Complete results analysis
│   └── FAQ.md                              # Frequently asked questions
├── requirements.txt                        # Python dependencies
├── LICENSE                                 # MIT License
└── README.md                               # This file
```

---

## 🔬 Methodology

### 1. Baseline Model

- **Algorithm:** XGBoost (single model)
- **Threshold:** 0.5 (standard practice)
- **Decision:** Binary (approve/reject)

### 2. SEFD-Plus System

- **Ensemble:** 5 XGBoost models with different random seeds
- **Uncertainty:** Ensemble variance (epistemic uncertainty)
- **Gray Zone Thresholds:**
  - θ_low = 0.05 (SAFE zone)
  - θ_high = 0.95 (FLAGGED zone)
- **Decision:** Ternary (approve/review/reject)

### 3. Evaluation Metrics

- **True Positive Rate (TPR):** Fraud detection rate
- **False Positive Rate (FPR):** Legitimate transactions incorrectly flagged
- **F2 Score:** Weighted harmonic mean (emphasizes recall)
- **Cost Analysis:** Financial impact calculation

### 4. Statistical Testing

- **Fisher's Exact Test:** For false positive reduction significance
- **Bootstrap Confidence Intervals:** 95% CI with 1,000 iterations
- **Significance Level:** α = 0.05

---

## 📊 Results

### Performance Comparison

```
Baseline:
  TPR: 79.05% [78.23%, 79.87%]
  FPR: 10.42% [10.21%, 10.63%]
  F2:  0.5157 [0.5089, 0.5225]

SEFD-Plus:
  TPR: 81.16% [80.38%, 81.94%]
  FPR:  8.41% [8.22%, 8.60%]
  F2:  0.5701 [0.5635, 0.5767]
```

### Gray Zone Analysis

- **Size:** 16,476 transactions (9.3% of test set)
- **Fraud Rate:** 3.3% (vs. 3.5% overall)
- **Enrichment:** 0.95x (slightly below average)
- **Purpose:** Route uncertain cases to human experts

### Cost-Benefit Analysis

**Assumptions:**
- False Positive Cost: $25 (customer friction)
- False Negative Cost: $100 (fraud loss)
- Human Review Cost: $5 (manual inspection)

**Annual Costs (365 days):**
- Baseline: $2,787,660
- SEFD-Plus: $1,829,120
- **Savings: $958,540 (34.4% reduction)**

---

## 🔧 Configuration

Key parameters can be modified in the notebook:

```python
# Gray Zone thresholds
THETA_LOW = 0.05   # Lower threshold
THETA_HIGH = 0.95  # Upper threshold

# Ensemble configuration
N_MODELS = 5
ENSEMBLE_SEEDS = [42, 123, 456, 789, 2024]

# Cost parameters
COST_FALSE_POSITIVE = 25
COST_FALSE_NEGATIVE = 100
COST_HUMAN_REVIEW = 5
```

---

## 📖 Documentation

- **[METHODOLOGY.md](docs/METHODOLOGY.md):** Detailed technical methodology
- **[RESULTS.md](docs/RESULTS.md):** Complete experimental results
- **[FAQ.md](docs/FAQ.md):** Frequently asked questions

---

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory Error**

```bash
# Reduce dataset size for testing
df = df.sample(n=100000, random_state=42)
```

**2. Slow Training**

```bash
# Use GPU acceleration (if available)
XGBOOST_PARAMS['device'] = 'cuda'
```

**3. Missing Dataset**

```bash
# Download from Kaggle using kaggle-api
pip install kaggle
kaggle competitions download -c ieee-fraud-detection
```

---

## 📝 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@inproceedings{owayed2026sefdplus,
  title={SEFD-Plus: Uncertainty-Aware Fraud Detection with Human-in-the-Loop Governance},
  author={Owayed, Haifaa},
  booktitle={IEEE Canadian Conference on Electrical and Computer Engineering (CCECE)},
  year={2026},
  organization={IEEE}
}
```

---

## 📧 Contact

**Author:** Haifaa Owayed  
**Email:** howay035@uottawa.ca  
**Institution:** University of Ottawa  
**Conference:** IEEE CCECE 2026  
**Paper ID:** 2692212270

For questions or issues:
- Open a GitHub issue
- Email the author directly
- Check the [FAQ](docs/FAQ.md)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset:** IEEE Computational Intelligence Society (IEEE-CIS)
- **Conference:** IEEE CCECE 2026 organizing committee
- **Institution:** University of Ottawa
- **Reviewers:** Anonymous reviewers for their valuable feedback

---

## 🔗 Links

- **Paper:** IEEE CCECE 2026 (Paper ID: 2692212270)
- **Dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
- **XGBoost:** [Official Documentation](https://xgboost.readthedocs.io/)
- **Scikit-learn:** [Official Documentation](https://scikit-learn.org/)

---

## 📊 Reproducibility Checklist

- [x] Complete source code provided
- [x] Dataset publicly available
- [x] Dependencies specified (requirements.txt)
- [x] Random seeds fixed for reproducibility
- [x] Hyperparameters documented
- [x] Statistical tests included
- [x] Confidence intervals calculated
- [x] Results verified against paper
- [x] Comprehensive documentation
- [x] MIT License for open access

---

**Last Updated:** January 2026  
**Version:** 1.0.0  
**Status:** ✅ Ready for Review
