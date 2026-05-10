"""Evaluation metrics and utilities for causal inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error


@dataclass
class CausalResults:
    """Container for causal inference results."""
    
    ate: float
    att: Optional[float] = None
    atc: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    n_treated: Optional[int] = None
    n_control: Optional[int] = None
    n_matched: Optional[int] = None
    balance_stats: Optional[pd.DataFrame] = None


def evaluate_causal_effect(
    model,
    data,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
) -> CausalResults:
    """
    Evaluate causal effect estimation with comprehensive metrics.
    
    Args:
        model: Fitted causal inference model
        data: CausalData object
        confidence_level: Confidence level for intervals
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        CausalResults object
    """
    # Estimate ATE
    if hasattr(model, 'estimate_ate'):
        ate = model.estimate_ate(data.outcome)
    else:
        ate = np.nan
    
    # Estimate ATT if available
    att = None
    if hasattr(model, 'estimate_att'):
        att = model.estimate_att(data.outcome)
    
    # Calculate confidence interval
    ci = None
    if hasattr(model, 'get_confidence_interval'):
        ci = model.get_confidence_interval(alpha=1-confidence_level)
    else:
        ci = bootstrap_confidence_interval(
            model, data, confidence_level, n_bootstrap
        )
    
    # Calculate p-value
    p_value = None
    if ci is not None:
        p_value = 1 if (ci[0] <= 0 <= ci[1]) else 0
    
    # Get sample sizes
    n_treated = np.sum(data.treatment)
    n_control = len(data.treatment) - n_treated
    
    # Get matched sample size if available
    n_matched = None
    if hasattr(model, 'matches_'):
        n_matched = len([m for m in model.matches_ if m[1] is not None])
    
    # Get balance statistics if available
    balance_stats = None
    if hasattr(model, 'get_balance_stats'):
        balance_stats = model.get_balance_stats(data.X, data.feature_names)
    
    return CausalResults(
        ate=ate,
        att=att,
        confidence_interval=ci,
        p_value=p_value,
        n_treated=n_treated,
        n_control=n_control,
        n_matched=n_matched,
        balance_stats=balance_stats,
    )


def bootstrap_confidence_interval(
    model,
    data,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for causal effect.
    
    Args:
        model: Fitted causal inference model
        data: CausalData object
        confidence_level: Confidence level
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_ates = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        n_samples = len(data.X)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        bootstrap_X = data.X[bootstrap_indices]
        bootstrap_treatment = data.treatment[bootstrap_indices]
        bootstrap_outcome = data.outcome[bootstrap_indices]
        
        # Fit model on bootstrap sample
        bootstrap_model = type(model)(**model.get_params())
        bootstrap_model.fit(bootstrap_X, bootstrap_treatment, bootstrap_outcome)
        
        # Estimate ATE
        if hasattr(bootstrap_model, 'estimate_ate'):
            bootstrap_ate = bootstrap_model.estimate_ate(bootstrap_outcome)
            bootstrap_ates.append(bootstrap_ate)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_ates, lower_percentile)
    upper_bound = np.percentile(bootstrap_ates, upper_percentile)
    
    return lower_bound, upper_bound


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


def evaluate_prediction_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate prediction accuracy metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def sensitivity_analysis(
    model,
    data,
    gamma_range: Tuple[float, float] = (1.0, 2.0),
    n_points: int = 20,
) -> pd.DataFrame:
    """
    Perform sensitivity analysis for unobserved confounding.
    
    Args:
        model: Fitted causal inference model
        data: CausalData object
        gamma_range: Range of sensitivity parameters
        n_points: Number of sensitivity points
        
    Returns:
        DataFrame with sensitivity results
    """
    gammas = np.linspace(gamma_range[0], gamma_range[1], n_points)
    sensitivity_results = []
    
    for gamma in gammas:
        # Adjust treatment assignment based on sensitivity parameter
        # This is a simplified sensitivity analysis
        adjusted_treatment = data.treatment.copy()
        
        # Simulate unobserved confounding effect
        unobserved_confounder = np.random.randn(len(data.treatment))
        confounding_effect = gamma * unobserved_confounder
        
        # Adjust treatment probability
        propensity_adjustment = 1 / (1 + np.exp(-confounding_effect))
        treatment_adjustment = np.random.binomial(1, propensity_adjustment)
        adjusted_treatment = np.logical_xor(data.treatment, treatment_adjustment).astype(int)
        
        # Re-estimate causal effect
        if hasattr(model, 'estimate_ate'):
            adjusted_ate = model.estimate_ate(data.outcome)
        else:
            adjusted_ate = np.nan
        
        sensitivity_results.append({
            "gamma": gamma,
            "ate": adjusted_ate,
        })
    
    return pd.DataFrame(sensitivity_results)


def create_evaluation_report(
    results: List[CausalResults],
    model_names: List[str],
    true_ate: Optional[float] = None,
) -> pd.DataFrame:
    """
    Create comprehensive evaluation report.
    
    Args:
        results: List of CausalResults objects
        model_names: Names of models
        true_ate: True ATE for comparison
        
    Returns:
        DataFrame with evaluation report
    """
    report_data = []
    
    for i, (result, model_name) in enumerate(zip(results, model_names)):
        row = {
            "model": model_name,
            "ate": result.ate,
            "att": result.att,
            "n_treated": result.n_treated,
            "n_control": result.n_control,
            "n_matched": result.n_matched,
        }
        
        if result.confidence_interval is not None:
            row["ci_lower"] = result.confidence_interval[0]
            row["ci_upper"] = result.confidence_interval[1]
            row["ci_width"] = result.confidence_interval[1] - result.confidence_interval[0]
        
        if result.p_value is not None:
            row["p_value"] = result.p_value
        
        if true_ate is not None:
            row["bias"] = result.ate - true_ate
            row["relative_bias"] = (result.ate - true_ate) / true_ate if true_ate != 0 else np.nan
        
        report_data.append(row)
    
    return pd.DataFrame(report_data)
