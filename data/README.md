# Dataset Instructions

## IEEE-CIS Fraud Detection Dataset

The SEFD-Plus experiments use the **IEEE-CIS Fraud Detection Dataset** from Kaggle.

### Download Instructions

1. **Create Kaggle Account:**
   - Go to [kaggle.com](https://www.kaggle.com/)
   - Sign up or log in

2. **Download Dataset:**
   - Visit: https://www.kaggle.com/c/ieee-fraud-detection/data
   - Click "Download All"
   - Extract files to this directory

3. **Expected Files:**
   ```
   data/
   ├── train_transaction.csv
   ├── train_identity.csv
   ├── test_transaction.csv
   └── test_identity.csv
   ```

### Dataset Statistics

- **Total Transactions:** 590,540
- **Training Set:** 413,378 transactions (70%)
- **Test Set:** 177,162 transactions (30%)
- **Fraud Rate:** 3.5%
- **Time Period:** 6 months
- **Features:** 
  - Transaction: 394 features
  - Identity: 41 features

### Feature Categories

1. **Transaction Features:**
   - `TransactionDT`: Time delta from reference datetime
   - `TransactionAmt`: Transaction amount
   - `ProductCD`: Product code
   - `card1-card6`: Card information
   - `addr1-addr2`: Address information
   - `dist1-dist2`: Distance information
   - `P_emaildomain`, `R_emaildomain`: Email domains
   - `C1-C14`: Counting features
   - `D1-D15`: Time delta features
   - `M1-M9`: Match features
   - `V1-V339`: Vesta engineered features

2. **Identity Features:**
   - `id_01-id_11`: Identity information
   - `id_12-id_38`: Digital signature features
   - `DeviceType`: Device type
   - `DeviceInfo`: Device information

### Preprocessing

The dataset requires minimal preprocessing:
- Missing values are handled by XGBoost natively
- Features are standardized (zero mean, unit variance)
- No feature engineering required (Vesta features are pre-engineered)

### Citation

If you use this dataset, please cite:

```
IEEE Computational Intelligence Society
Vesta Corporation
"IEEE-CIS Fraud Detection"
Kaggle Competition, 2019
https://www.kaggle.com/c/ieee-fraud-detection
```

### License

The dataset is provided under Kaggle's competition terms.
Please review: https://www.kaggle.com/c/ieee-fraud-detection/rules

---

**Note:** Due to size constraints, the dataset files are not included in this repository.
Please download them from Kaggle as instructed above.
