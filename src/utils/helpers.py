"""Utility functions for causal inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    
    # Set additional random seeds if packages are available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def check_data_quality(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> dict[str, Union[bool, str, int]]:
    """
    Check data quality for causal inference.
    
    Args:
        X: Covariates matrix
        treatment: Treatment indicator
        outcome: Outcome variable
        feature_names: Names of features
        
    Returns:
        Dictionary with data quality checks
    """
    checks = {}
    
    # Basic shape checks
    checks["consistent_lengths"] = len(X) == len(treatment) == len(outcome)
    checks["n_samples"] = len(X)
    checks["n_features"] = X.shape[1]
    
    # Treatment checks
    checks["binary_treatment"] = np.all(np.isin(treatment, [0, 1]))
    checks["treatment_rate"] = float(np.mean(treatment))
    checks["has_treated"] = np.sum(treatment) > 0
    checks["has_control"] = np.sum(treatment == 0) > 0
    
    # Outcome checks
    checks["finite_outcome"] = np.all(np.isfinite(outcome))
    checks["outcome_range"] = (float(np.min(outcome)), float(np.max(outcome)))
    
    # Feature checks
    checks["finite_features"] = np.all(np.isfinite(X))
    checks["no_constant_features"] = np.all(np.std(X, axis=0) > 1e-10)
    
    # Missing data checks
    checks["no_missing_X"] = not np.any(np.isnan(X))
    checks["no_missing_treatment"] = not np.any(np.isnan(treatment))
    checks["no_missing_outcome"] = not np.any(np.isnan(outcome))
    
    # Data quality score
    quality_score = sum([
        checks["consistent_lengths"],
        checks["binary_treatment"],
        checks["has_treated"],
        checks["has_control"],
        checks["finite_outcome"],
        checks["finite_features"],
        checks["no_constant_features"],
        checks["no_missing_X"],
        checks["no_missing_treatment"],
        checks["no_missing_outcome"],
    ]) / 10
    
    checks["quality_score"] = quality_score
    
    if quality_score < 0.8:
        checks["quality_warning"] = "Data quality issues detected"
    else:
        checks["quality_warning"] = "Data quality looks good"
    
    return checks


def preprocess_data(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    standardize: bool = True,
    remove_outliers: bool = False,
    outlier_threshold: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess data for causal inference.
    
    Args:
        X: Covariates matrix
        treatment: Treatment indicator
        outcome: Outcome variable
        standardize: Whether to standardize features
        remove_outliers: Whether to remove outliers
        outlier_threshold: Threshold for outlier detection
        
    Returns:
        Tuple of preprocessed (X, treatment, outcome)
    """
    X_processed = X.copy()
    treatment_processed = treatment.copy()
    outcome_processed = outcome.copy()
    
    # Remove outliers if requested
    if remove_outliers:
        # Remove outliers based on outcome
        outcome_z_scores = np.abs((outcome_processed - np.mean(outcome_processed)) / np.std(outcome_processed))
        outlier_mask = outcome_z_scores < outlier_threshold
        
        X_processed = X_processed[outlier_mask]
        treatment_processed = treatment_processed[outlier_mask]
        outcome_processed = outcome_processed[outlier_mask]
    
    # Standardize features
    if standardize:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_processed)
    
    return X_processed, treatment_processed, outcome_processed


def create_train_test_split(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    test_size: float = 0.2,
    stratify: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train-test split for causal inference.
    
    Args:
        X: Covariates matrix
        treatment: Treatment indicator
        outcome: Outcome variable
        test_size: Proportion of data for testing
        stratify: Variable to stratify on
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, treatment_train, treatment_test, outcome_train, outcome_test)
    """
    stratify_var = stratify if stratify is not None else treatment
    
    X_train, X_test, treatment_train, treatment_test, outcome_train, outcome_test = train_test_split(
        X, treatment, outcome,
        test_size=test_size,
        stratify=stratify_var,
        random_state=random_state,
    )
    
    return X_train, X_test, treatment_train, treatment_test, outcome_train, outcome_test


def calculate_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    treatment_rate: float = 0.5,
) -> int:
    """
    Calculate required sample size for detecting treatment effect.
    
    Args:
        effect_size: Expected treatment effect size
        power: Statistical power (1 - beta)
        alpha: Significance level
        treatment_rate: Proportion of treated units
        
    Returns:
        Required sample size
    """
    from scipy.stats import norm
    
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    # Sample size formula for two-sample t-test
    n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    
    # Adjust for treatment rate
    n_total = n_per_group / (treatment_rate * (1 - treatment_rate))
    
    return int(np.ceil(n_total))


def bootstrap_sample(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate bootstrap samples.
    
    Args:
        X: Covariates matrix
        treatment: Treatment indicator
        outcome: Outcome variable
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed
        
    Returns:
        List of bootstrap samples
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    bootstrap_samples = []
    n_samples = len(X)
    
    for _ in range(n_bootstrap):
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        X_bootstrap = X[bootstrap_indices]
        treatment_bootstrap = treatment[bootstrap_indices]
        outcome_bootstrap = outcome[bootstrap_indices]
        
        bootstrap_samples.append((X_bootstrap, treatment_bootstrap, outcome_bootstrap))
    
    return bootstrap_samples


def calculate_effect_size(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    method: str = "cohen_d",
) -> float:
    """
    Calculate effect size between treated and control groups.
    
    Args:
        treated_outcomes: Outcomes for treated units
        control_outcomes: Outcomes for control units
        method: Method for calculating effect size
        
    Returns:
        Effect size
    """
    if method == "cohen_d":
        # Cohen's d
        pooled_std = np.sqrt(
            ((len(treated_outcomes) - 1) * np.var(treated_outcomes, ddof=1) +
             (len(control_outcomes) - 1) * np.var(control_outcomes, ddof=1)) /
            (len(treated_outcomes) + len(control_outcomes) - 2)
        )
        
        if pooled_std == 0:
            return 0
        
        effect_size = (np.mean(treated_outcomes) - np.mean(control_outcomes)) / pooled_std
        
    elif method == "mean_difference":
        # Simple mean difference
        effect_size = np.mean(treated_outcomes) - np.mean(control_outcomes)
        
    else:
        raise ValueError(f"Unknown effect size method: {method}")
    
    return effect_size


def validate_assumptions(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity_scores: Optional[np.ndarray] = None,
) -> dict[str, Union[bool, float, str]]:
    """
    Validate causal inference assumptions.
    
    Args:
        X: Covariates matrix
        treatment: Treatment indicator
        outcome: Outcome variable
        propensity_scores: Estimated propensity scores
        
    Returns:
        Dictionary with assumption validation results
    """
    validation = {}
    
    # Unconfoundedness assumption (cannot be directly tested)
    validation["unconfoundedness"] = "Cannot be directly tested - requires domain knowledge"
    
    # Overlap assumption
    if propensity_scores is not None:
        treated_ps = propensity_scores[treatment == 1]
        control_ps = propensity_scores[treatment == 0]
        
        min_treated = np.min(treated_ps)
        max_treated = np.max(treated_ps)
        min_control = np.min(control_ps)
        max_control = np.max(control_ps)
        
        overlap_range = min(max_treated, max_control) - max(min_treated, min_control)
        validation["overlap_range"] = overlap_range
        validation["has_overlap"] = overlap_range > 0.1
        
        # Propensity score distribution
        validation["ps_min"] = float(np.min(propensity_scores))
        validation["ps_max"] = float(np.max(propensity_scores))
        validation["ps_mean"] = float(np.mean(propensity_scores))
    
    # SUTVA assumption (cannot be directly tested)
    validation["sutva"] = "Cannot be directly tested - requires domain knowledge"
    
    # Balance checks
    balance_stats = calculate_balance_metrics(X, treatment)
    max_std_diff = np.max(np.abs(balance_stats["std_diff"]))
    validation["max_std_diff"] = max_std_diff
    validation["balanced"] = max_std_diff < 0.1
    
    return validation


def calculate_balance_metrics(
    X: np.ndarray,
    treatment: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculate covariate balance metrics.
    
    Args:
        X: Covariates matrix
        treatment: Treatment indicator
        feature_names: Names of features
        
    Returns:
        DataFrame with balance metrics
    """
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(X.shape[1])]
    
    balance_stats = []
    
    for i, feature_name in enumerate(feature_names):
        treated_values = X[treatment == 1, i]
        control_values = X[treatment == 0, i]
        
        treated_mean = np.mean(treated_values)
        control_mean = np.mean(control_values)
        treated_std = np.std(treated_values)
        control_std = np.std(control_values)
        
        # Standardized difference
        pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
        std_diff = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Variance ratio
        var_ratio = treated_std**2 / control_std**2 if control_std > 0 else np.inf
        
        balance_stats.append({
            "feature": feature_name,
            "treated_mean": treated_mean,
            "control_mean": control_mean,
            "treated_std": treated_std,
            "control_std": control_std,
            "std_diff": std_diff,
            "var_ratio": var_ratio,
        })
    
    return pd.DataFrame(balance_stats)
