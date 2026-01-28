# Dataset Instructions

## IEEE-CIS Fraud Detection Dataset

This reproducibility package requires the **IEEE-CIS Fraud Detection** dataset from Kaggle.

---

## 📥 Download Instructions

### Method 1: Manual Download (Recommended)

1. **Create a Kaggle account** (if you don't have one):
   - Go to: https://www.kaggle.com/
   - Sign up for free

2. **Accept competition rules**:
   - Go to: https://www.kaggle.com/c/ieee-fraud-detection
   - Click "Join Competition" and accept the rules

3. **Download the dataset**:
   - Go to the Data tab: https://www.kaggle.com/c/ieee-fraud-detection/data
   - Download `train_transaction.csv` (~470 MB)

4. **Place the file in this directory**:
   ```
   data/
     └── train_transaction.csv
   ```

---

### Method 2: Kaggle API (Advanced)

```bash
# Install Kaggle API
pip install kaggle

# Setup API credentials
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/

# Download dataset
kaggle competitions download -c ieee-fraud-detection

# Extract train_transaction.csv
unzip ieee-fraud-detection.zip train_transaction.csv -d data/
```

---

## 📊 Dataset Information

- **Name:** IEEE-CIS Fraud Detection
- **Source:** Kaggle Competition
- **File:** train_transaction.csv
- **Size:** ~470 MB (compressed), ~1.2 GB (uncompressed)
- **Transactions:** 590,540
- **Features:** 394 columns
- **Target:** isFraud (binary: 0 = legitimate, 1 = fraud)
- **Fraud Rate:** ~3.5%

---

## 🔒 Data Privacy

This dataset is provided by IEEE Computational Intelligence Society (IEEE-CIS) and Vesta Corporation for research purposes. It contains anonymized transaction data with:

- No personally identifiable information (PII)
- Anonymized features (V1-V339)
- Synthetic transaction IDs
- Masked card information

**Usage:** Research and educational purposes only, as per Kaggle competition rules.

---

## ✅ Verification

After downloading, verify the file:

```bash
# Check file exists
ls -lh data/train_transaction.csv

# Check file size (~470 MB compressed, ~1.2 GB uncompressed)
# Expected: 590,540 rows × 394 columns

# Quick verification in Python
python -c "import pandas as pd; df = pd.read_csv('data/train_transaction.csv'); print(f'Shape: {df.shape}'); print(f'Fraud rate: {df.isFraud.mean():.2%}')"
```

**Expected Output:**
```
Shape: (590540, 394)
Fraud rate: 3.50%
```

---

## 🚨 Important Notes

1. **Do NOT commit the dataset to Git** (it's too large and violates Kaggle terms)
2. The `.gitignore` file already excludes CSV files
3. Each user must download the dataset individually from Kaggle
4. The dataset is required for reproducibility but not included in the repository

---

## 📧 Issues

If you encounter download issues:

1. **Check Kaggle account status** - Ensure you're logged in
2. **Accept competition rules** - Required before downloading
3. **Check internet connection** - Large file download
4. **Try alternative browser** - Some browsers may have issues
5. **Contact Kaggle support** - For dataset-specific issues

For code-related issues, contact: howay035@uottawa.ca

---

## 🔗 Links

- **Competition:** https://www.kaggle.com/c/ieee-fraud-detection
- **Data Page:** https://www.kaggle.com/c/ieee-fraud-detection/data
- **Discussion:** https://www.kaggle.com/c/ieee-fraud-detection/discussion
- **Kaggle API:** https://github.com/Kaggle/kaggle-api
