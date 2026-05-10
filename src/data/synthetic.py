"""Data loading and generation utilities for causal inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from sklearn.datasets import make_classification


@dataclass
class CausalData:
    """Container for causal inference datasets."""
    
    X: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    true_ate: Optional[float] = None
    feature_names: Optional[list[str]] = None
    
    def __post_init__(self) -> None:
        """Validate data consistency."""
        if len(self.X) != len(self.treatment) or len(self.X) != len(self.outcome):
            raise ValueError("X, treatment, and outcome must have the same length")
        
        if self.feature_names is None:
            self.feature_names = [f"X{i}" for i in range(self.X.shape[1])]


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 5,
    n_confounders: int = 3,
    treatment_effect: float = 2.0,
    noise_level: float = 0.1,
    random_state: Optional[int] = None,
) -> CausalData:
    """
    Generate synthetic data for causal inference experiments.
    
    Args:
        n_samples: Number of observations
        n_features: Number of features (confounders + instruments)
        n_confounders: Number of confounders (affect both treatment and outcome)
        treatment_effect: True average treatment effect
        noise_level: Standard deviation of noise
        random_state: Random seed for reproducibility
        
    Returns:
        CausalData object with synthetic data
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create treatment assignment based on confounders
    # Treatment probability depends on first n_confounders features
    confounders = X[:, :n_confounders]
    propensity_logit = 0.5 * confounders.sum(axis=1) + np.random.randn(n_samples) * 0.1
    propensity = 1 / (1 + np.exp(-propensity_logit))
    treatment = np.random.binomial(1, propensity)
    
    # Generate potential outcomes
    # Outcome depends on confounders and treatment
    Y0 = 2 * confounders.sum(axis=1) + np.random.randn(n_samples) * noise_level
    Y1 = Y0 + treatment_effect + np.random.randn(n_samples) * noise_level
    
    # Observed outcome
    outcome = treatment * Y1 + (1 - treatment) * Y0
    
    feature_names = [f"confounder_{i}" for i in range(n_confounders)]
    feature_names.extend([f"instrument_{i}" for i in range(n_features - n_confounders)])
    
    return CausalData(
        X=X,
        treatment=treatment,
        outcome=outcome,
        true_ate=treatment_effect,
        feature_names=feature_names,
    )


def generate_heterogeneous_data(
    n_samples: int = 1000,
    n_features: int = 5,
    treatment_effect_range: Tuple[float, float] = (0.5, 3.0),
    random_state: Optional[int] = None,
) -> CausalData:
    """
    Generate data with heterogeneous treatment effects.
    
    Args:
        n_samples: Number of observations
        n_features: Number of features
        treatment_effect_range: Range of treatment effects
        random_state: Random seed for reproducibility
        
    Returns:
        CausalData object with heterogeneous effects
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Treatment assignment based on features
    propensity_logit = 0.3 * X[:, 0] + 0.2 * X[:, 1] + np.random.randn(n_samples) * 0.1
    propensity = 1 / (1 + np.exp(-propensity_logit))
    treatment = np.random.binomial(1, propensity)
    
    # Heterogeneous treatment effects
    effect_modifier = X[:, 0]  # First feature modifies treatment effect
    normalized_effect = (effect_modifier - effect_modifier.min()) / (effect_modifier.max() - effect_modifier.min())
    individual_effects = treatment_effect_range[0] + normalized_effect * (treatment_effect_range[1] - treatment_effect_range[0])
    
    # Generate outcomes
    Y0 = 2 * X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1
    Y1 = Y0 + individual_effects + np.random.randn(n_samples) * 0.1
    outcome = treatment * Y1 + (1 - treatment) * Y0
    
    # True ATE is average of individual effects
    true_ate = np.mean(individual_effects)
    
    feature_names = [f"X{i}" for i in range(n_features)]
    
    return CausalData(
        X=X,
        treatment=treatment,
        outcome=outcome,
        true_ate=true_ate,
        feature_names=feature_names,
    )


def load_lalonde_data() -> CausalData:
    """
    Load the famous Lalonde dataset for causal inference evaluation.
    
    Returns:
        CausalData object with Lalonde data
    """
    # This is a simplified version - in practice, you'd load from a file
    # or use a proper dataset loader
    np.random.seed(42)
    
    # Simulate Lalonde-like data structure
    n_samples = 445
    X = np.random.randn(n_samples, 8)  # 8 covariates
    treatment = np.random.binomial(1, 0.4, n_samples)  # 40% treated
    
    # Simulate earnings outcome
    outcome = 2000 + 1000 * treatment + 200 * X[:, 0] + np.random.randn(n_samples) * 500
    
    feature_names = [
        "age", "education", "black", "hispanic", "married", 
        "nodegree", "re74", "re75"
    ]
    
    return CausalData(
        X=X,
        treatment=treatment,
        outcome=outcome,
        true_ate=1000.0,  # Approximate true effect
        feature_names=feature_names,
    )


def check_overlap(data: CausalData, threshold: float = 0.1) -> dict[str, Union[bool, float]]:
    """
    Check overlap assumption for propensity scores.
    
    Args:
        data: CausalData object
        threshold: Minimum overlap threshold
        
    Returns:
        Dictionary with overlap diagnostics
    """
    from sklearn.linear_model import LogisticRegression
    
    # Estimate propensity scores
    ps_model = LogisticRegression()
    ps_model.fit(data.X, data.treatment)
    propensity_scores = ps_model.predict_proba(data.X)[:, 1]
    
    # Check overlap
    treated_ps = propensity_scores[data.treatment == 1]
    control_ps = propensity_scores[data.treatment == 0]
    
    min_treated = np.min(treated_ps)
    max_treated = np.max(treated_ps)
    min_control = np.min(control_ps)
    max_control = np.max(control_ps)
    
    overlap_range = min(max_treated, max_control) - max(min_treated, min_control)
    has_overlap = overlap_range > threshold
    
    return {
        "has_overlap": has_overlap,
        "overlap_range": overlap_range,
        "min_treated_ps": min_treated,
        "max_treated_ps": max_treated,
        "min_control_ps": min_control,
        "max_control_ps": max_control,
    }
