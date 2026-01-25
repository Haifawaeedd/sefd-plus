"""
SEFD-Plus: Uncertainty-Aware Fraud Detection with Human-in-the-Loop Governance

This module implements the core SEFD-Plus framework for fraud detection with
uncertainty quantification and three-zone triage (SAFE, GRAY, FLAGGED).

Paper: "SEFD-Plus: Uncertainty-Aware Fraud Detection with Human-in-the-Loop Governance"
Conference: IEEE CCECE 2026
Author: Haifaa Owayed
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SEFDPlus:
    """
    SEFD-Plus: Ensemble-based fraud detection with uncertainty-aware triage.
    
    The system uses XGBoost ensemble to generate fraud probabilities and
    uncertainty estimates, then routes transactions to three zones:
    - SAFE: Low uncertainty, low fraud probability → Auto-approve
    - GRAY: High uncertainty → Human review
    - FLAGGED: Low uncertainty, high fraud probability → Auto-block
    
    Attributes:
        n_models (int): Number of ensemble members (default: 5)
        random_seeds (List[int]): Random seeds for ensemble diversity
        theta_low (float): Uncertainty threshold for Gray Zone
        fraud_threshold (float): Fraud probability threshold for FLAGGED
        models (List): Trained XGBoost models
        scaler: Feature scaler
    """
    
    def __init__(
        self,
        n_models: int = 5,
        random_seeds: List[int] = None,
        theta_low: float = 0.05,
        fraud_threshold: float = 0.9,
        xgb_params: Dict = None
    ):
        """
        Initialize SEFD-Plus framework.
        
        Args:
            n_models: Number of ensemble members
            random_seeds: List of random seeds for ensemble diversity
            theta_low: Uncertainty threshold for Gray Zone
            fraud_threshold: Probability threshold for FLAGGED zone
            xgb_params: XGBoost hyperparameters (optional)
        """
        self.n_models = n_models
        self.random_seeds = random_seeds or [42, 123, 456, 789, 1011]
        self.theta_low = theta_low
        self.fraud_threshold = fraud_threshold
        
        # Default XGBoost parameters
        self.xgb_params = xgb_params or {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'scale_pos_weight': 27.6  # Adjust based on fraud rate
        }
        
        self.models = []
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SEFDPlus':
        """
        Train ensemble of XGBoost models.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (0=legitimate, 1=fraud)
            
        Returns:
            self: Fitted model
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train ensemble members
        print(f"Training {self.n_models} ensemble members...")
        for i, seed in enumerate(self.random_seeds[:self.n_models]):
            print(f"  Training model {i+1}/{self.n_models} (seed={seed})...")
            
            model = xgb.XGBClassifier(**self.xgb_params, random_state=seed)
            model.fit(X_scaled, y)
            self.models.append(model)
            
        print("✅ Training complete!")
        return self
    
    def predict_with_uncertainty(
        self, 
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fraud probabilities and uncertainty estimates.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            mean_probs: Mean fraud probability across ensemble
            uncertainties: Prediction variance (epistemic uncertainty)
        """
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all ensemble members
        predictions = np.array([
            model.predict_proba(X_scaled)[:, 1]
            for model in self.models
        ])
        
        # Compute mean and variance
        mean_probs = predictions.mean(axis=0)
        uncertainties = predictions.var(axis=0)
        
        return mean_probs, uncertainties
    
    def assign_zones(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assign transactions to three zones based on uncertainty and probability.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            zones: Zone assignments (0=SAFE, 1=GRAY, 2=FLAGGED)
            probs: Fraud probabilities
            uncertainties: Uncertainty estimates
        """
        probs, uncertainties = self.predict_with_uncertainty(X)
        
        # Initialize zones
        zones = np.zeros(len(X), dtype=int)
        
        # Assign zones based on triage policy
        # SAFE: Low uncertainty AND low fraud probability
        safe_mask = uncertainties < self.theta_low
        zones[safe_mask] = 0
        
        # FLAGGED: Low uncertainty AND high fraud probability
        flagged_mask = (uncertainties < self.theta_low) & (probs > self.fraud_threshold)
        zones[flagged_mask] = 2
        
        # GRAY: High uncertainty (requires human review)
        gray_mask = uncertainties >= self.theta_low
        zones[gray_mask] = 1
        
        return zones, probs, uncertainties
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict:
        """
        Evaluate SEFD-Plus performance on test set.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        zones, probs, uncertainties = self.assign_zones(X)
        
        # Compute metrics for each zone
        safe_mask = zones == 0
        gray_mask = zones == 1
        flagged_mask = zones == 2
        
        # Overall metrics (automated decisions only)
        auto_mask = safe_mask | flagged_mask
        auto_preds = (probs[auto_mask] > 0.5).astype(int)
        auto_true = y[auto_mask]
        
        tp = ((auto_preds == 1) & (auto_true == 1)).sum()
        fp = ((auto_preds == 1) & (auto_true == 0)).sum()
        tn = ((auto_preds == 0) & (auto_true == 0)).sum()
        fn = ((auto_preds == 0) & (auto_true == 1)).sum()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Gray Zone metrics
        gray_fraud_rate = y[gray_mask].mean() if gray_mask.sum() > 0 else 0
        overall_fraud_rate = y.mean()
        gray_enrichment = gray_fraud_rate / overall_fraud_rate if overall_fraud_rate > 0 else 0
        
        metrics = {
            'tpr': tpr,
            'fpr': fpr,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'safe_count': int(safe_mask.sum()),
            'gray_count': int(gray_mask.sum()),
            'flagged_count': int(flagged_mask.sum()),
            'gray_fraud_rate': gray_fraud_rate,
            'overall_fraud_rate': overall_fraud_rate,
            'gray_enrichment': gray_enrichment,
            'hitl_load': gray_mask.mean()
        }
        
        return metrics


def load_ieee_cis_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess IEEE-CIS fraud detection dataset.
    
    Args:
        data_path: Path to dataset CSV file
        
    Returns:
        X_train, X_test, y_train, y_test: Train/test splits
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Separate features and labels
    X = df.drop('isFraud', axis=1).values
    y = df['isFraud'].values
    
    # Train/test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"✅ Data loaded:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    print(f"   Fraud rate: {y.mean()*100:.2f}%")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    print("SEFD-Plus: Uncertainty-Aware Fraud Detection")
    print("=" * 60)
    
    # Load data
    # X_train, X_test, y_train, y_test = load_ieee_cis_data('data/fraud_data.csv')
    
    # Initialize SEFD-Plus
    sefd = SEFDPlus(
        n_models=5,
        theta_low=0.05,
        fraud_threshold=0.9
    )
    
    # Train
    # sefd.fit(X_train, y_train)
    
    # Evaluate
    # metrics = sefd.evaluate(X_test, y_test)
    # print("\nEvaluation Results:")
    # print(f"  TPR: {metrics['tpr']*100:.2f}%")
    # print(f"  FPR: {metrics['fpr']*100:.2f}%")
    # print(f"  HITL Load: {metrics['hitl_load']*100:.2f}%")
    # print(f"  Gray Zone Enrichment: {metrics['gray_enrichment']:.2f}x")
    
    print("\n✅ SEFD-Plus module loaded successfully!")
