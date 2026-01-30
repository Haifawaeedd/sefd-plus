# SEFD-Plus: Uncertainty-Aware Fraud Detection with Human-in-the-Loop Governance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Conference](https://img.shields.io/badge/Conference-IEEE%20CCECE%202026-green.svg)](https://ccece2026.ieee.ca/)

SEFD-Plus is a governance-focused fraud detection framework that integrates ensemble-based uncertainty quantification with human-in-the-loop triage to reduce false positives while maintaining detection accuracy.

**📄 Paper:** "SEFD-Plus: Uncertainty-Aware Fraud Detection with Human-in-the-Loop Governance"  
**🎓 Conference:** IEEE Canadian Conference on Electrical and Computer Engineering (CCECE) 2026  
**👤 Author:** Haifaa Owayed

---

## 🎯 Key Features

- **Ensemble-Based Uncertainty Quantification**: Uses XGBoost ensemble (5 models) to generate calibrated fraud probabilities and epistemic uncertainty estimates
- **Three-Zone Triage**: Routes transactions to SAFE (auto-approve), GRAY (human review), or FLAGGED (auto-block) based on uncertainty
- **Cost-Sensitive Policy**: Balances fraud losses, false positive costs, and human review overhead
- **Governance-First Design**: Phased deployment with shadow monitoring, approval gates, and rollback mechanisms
- **Reproducible Research**: Complete code, hyperparameters, and experimental setup

---

## 📊 Results

Evaluation on **IEEE-CIS Fraud Detection dataset** (177,162 test transactions, 3.5% fraud rate):

| Metric | Baseline | SEFD-Plus | Improvement |
|--------|----------|-----------|-------------|
| **False Positive Rate (FPR)** | 10.42% | 8.4% | **-19.3%** (p < 10⁻⁸⁵) |
| **True Positive Rate (TPR)** | 79.1% | 81.2% | **+2.1%** (p < 10⁻⁴) |
| **F2 Score** | 0.516 | 0.570 | **+10.5%** |
| **HITL Load** | - | 9.3% | - |
| **Annual Savings** | - | **$815,540** | (1M txn/year) |

**Statistical Significance:** All improvements are highly significant (p < 10⁻⁴) with bootstrap 95% confidence intervals.

**Note:** Prior commit message contained a formatting typo; actual annual savings are **$815,540** as documented throughout this repository and the accompanying paper.


---


## 🌍 Cross-Dataset Generalization

To validate SEFD-Plus's governance stability across different fraud detection contexts, we conducted a post-submission study on the **Credit Card Fraud Detection dataset** (Kaggle).

### Why This Matters

SEFD-Plus is designed as a **governance framework**, not a dataset-specific optimization. This study demonstrates:
- Robust behavior without retraining or tuning
- Automatic adaptation of human review burden
- Conservative false positive control across contexts

### Generalization Study Results

| Dataset | Fraud Rate | F2 Score | Gray Zone (HITL) | False Positives |
|---------|------------|----------|------------------|-----------------|
| **IEEE-CIS** | 3.5% | 0.570 | 9.3% | 8.4% FPR |
| **Credit Card (Kaggle)** | 0.17% | 0.792 | 0.2% | 0.02% FPR |

**Key Observations:**
- ✅ **20x lower fraud rate** → System automatically reduces human review burden (9.3% → 0.2%)
- ✅ **Extremely low false positives** → Only 18 FP out of 85,305 legitimate transactions
- ✅ **No tuning required** → Same hyperparameters work effectively across datasets
- ✅ **Governance-friendly** → Conservative behavior maintained in both contexts

### Interpretation

The dramatic reduction in Gray Zone size (9.3% → 0.2%) is **not a bug but a feature**:
- Lower fraud rate → Fewer ambiguous cases near decision boundary
- Cleaner features (PCA-transformed) → Lower ensemble variance
- Conservative threshold (θ_low = 0.05) → Only truly uncertain cases flagged

This validates SEFD-Plus's core thesis: **uncertainty-aware governance provides robust fraud detection without dataset-specific tuning**.

📄 **Detailed Documentation:** [docs/generalization.md](docs/generalization.md)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Haifawaeedd/sefd-plus.git
cd sefd-plus

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.sefd_plus import SEFDPlus

# Initialize SEFD-Plus
sefd = SEFDPlus(
    n_models=5,           # Ensemble size
    theta_low=0.05,       # Uncertainty threshold
    fraud_threshold=0.9   # Fraud probability threshold
)

# Train on your data
sefd.fit(X_train, y_train)

# Assign transactions to zones
zones, probs, uncertainties = sefd.assign_zones(X_test)

# zones: 0=SAFE, 1=GRAY (human review), 2=FLAGGED
```

### Run Experiments

Reproduce paper results:

```bash
# Download IEEE-CIS dataset from Kaggle
# https://www.kaggle.com/c/ieee-fraud-detection

# Run experiments
python experiments/run_experiments.py \
    --data_path data/ieee_cis_fraud.csv \
    --theta_low 0.05 \
    --output_dir results
```

---

## 📁 Repository Structure

```
sefd-plus/
├── paper/
│   ├── SEFD-Plus-WITH-GITHUB-LINK.pdf    # IEEE CCECE 2026 paper
│   └── SEFD-Plus-WITH-GITHUB-LINK.docx   # Editable version
├── figures/
│   ├── Figure1_Confusion_Matrices.png    # Performance comparison
│   ├── Figure2_Zone_Distribution.png     # Transaction distribution
│   ├── Figure3_Cost_Comparison.png       # Cost-benefit analysis
│   ├── Figure4_System_Architecture.png   # System pipeline
│   └── Figure5_Performance_Metrics.png   # Metrics visualization
├── src/
│   └── sefd_plus.py                      # Core SEFD-Plus implementation
├── experiments/
│   └── run_experiments.py                # Reproduce paper results
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
├── data/
│   └── README.md                         # Dataset instructions
├── tests/
│   └── test_sefd_plus.py                 # Unit tests
├── requirements.txt                      # Python dependencies
├── LICENSE                               # MIT License
└── README.md                             # This file
```

---

## 🔬 Methodology

### System Architecture

SEFD-Plus processes transactions through a five-stage pipeline:

1. **Feature Engineering**: Transform raw transaction data into 339 features
2. **Fraud Probability Estimation**: XGBoost ensemble (5 models) generates fraud probabilities
3. **Uncertainty Quantification**: Compute prediction variance across ensemble members
4. **Uncertainty-Based Triage**: Assign transactions to three zones:
   - **SAFE**: σ(x) < θ_low → Auto-approve
   - **GRAY**: σ(x) ≥ θ_low → Human review
   - **FLAGGED**: p(x) > 0.9 AND σ(x) < θ_low → Auto-block
5. **Human Review**: Present GRAY zone transactions with SHAP explanations

![System Architecture](figures/Figure4_System_Architecture.png)

### Hyperparameters

**XGBoost Configuration:**

```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 27.6  # Based on 3.5% fraud rate
}
```

**Ensemble Configuration:**

- Number of models: 5
- Random seeds: {42, 123, 456, 789, 1011}
- Uncertainty threshold (θ_low): 0.05

---

## 📊 Dataset

**IEEE-CIS Fraud Detection Dataset**

- **Source**: [Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection)
- **Size**: 590,540 transactions (6 months)
- **Fraud Rate**: 3.5%
- **Features**: Transaction amount, card metadata, device ID, temporal patterns
- **Train/Test Split**: 70/30 (413,378 train, 177,162 test)

---

## 🧪 Reproducibility

All experiments are fully reproducible:

### Computational Environment

- **Hardware**: NVIDIA V100 GPU (32GB), Intel Xeon CPU (16 cores), 128GB RAM
- **Software**: Python 3.11, XGBoost 1.7.0, NumPy 1.24, Pandas 2.0, Scikit-learn 1.3
- **Training Time**: ~15 minutes for 5 ensemble members
- **Inference Time**: 10,000 transactions/second

### Statistical Tests

- **Bootstrap CI**: 1000 samples, stratified sampling, 95% confidence
- **Fisher's Exact Test**: For FPR reduction significance
- **Permutation Test**: For TPR improvement significance

---

## 📈 Cost-Benefit Analysis

For a merchant processing **1M transactions/year**:

| Cost Component | Baseline | SEFD-Plus | Savings |
|----------------|----------|-----------|---------|
| False Positives ($100 each) | $1,781,800 | $1,304,100 | $477,700 |
| False Negatives ($500 each) | $649,500 | $338,500 | $311,000 |
| Human Review ($20 each) | $356,360 | $329,520 | $26,840 |
| **Total** | **$2,787,660** | **$1,972,120** | **$815,540** |

**ROI: 29.2% cost reduction**

![Cost Comparison](figures/Figure3_Cost_Comparison.png)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 Citation

If you use SEFD-Plus in your research, please cite:

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- IEEE-CIS for providing the fraud detection dataset
- Kaggle for hosting the competition
- XGBoost team for the excellent gradient boosting library
- IEEE CCECE 2026 reviewers for valuable feedback

---

## 📧 Contact

**Haifaa Owayed**

- 📧 Email: haifawaeed2015@gmail.com
- 🎓 University: howay035@uottawa.ca
- 🔗 LinkedIn: [linkedin.com/in/haifaa-owayed-765297132](https://linkedin.com/in/haifaa-owayed-765297132)
- 🐙 GitHub: [@Haifawaeedd](https://github.com/Haifawaeedd)

---

## 🔗 Related Work

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [IEEE-CIS Fraud Detection Dataset](https://www.kaggle.com/c/ieee-fraud-detection)
- [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap)
- [Uncertainty Quantification in ML](https://arxiv.org/abs/1906.02530)

---

⭐ **If you find this work useful, please consider starring the repository!**

---

## 📊 Three-Zone Triage Results

| Zone | Count | Percentage | Fraud Rate | Enrichment | Action |
|------|-------|------------|------------|------------|--------|
| **SAFE** | 144,149 | 81.4% | 2.8% | 0.80x | Auto-approve |
| **GRAY** | 16,476 | 9.3% | 3.3% | 0.95x | Human review |
| **FLAGGED** | 16,537 | 9.3% | 31.7% | 9.06x | Auto-block |

![Zone Distribution](figures/Figure2_Zone_Distribution.png)

---

## 📈 Performance Comparison

![Confusion Matrices](figures/Figure1_Confusion_Matrices.png)

![Performance Metrics](figures/Figure5_Performance_Metrics.png)

---

**Last Updated:** January 2026  
**Paper Status:** Submitted to IEEE CCECE 2026  
**Code Status:** Available with complete implementation
