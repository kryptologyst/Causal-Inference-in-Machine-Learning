"""Inverse Probability Weighting implementation."""

from __future__ import annotations

import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class InverseProbabilityWeighting(BaseEstimator, RegressorMixin):
    """
    Inverse Probability Weighting for causal inference.
    
    Weights observations by the inverse of their propensity scores
    to create a pseudo-population where treatment assignment is random.
    """
    
    def __init__(
        self,
        propensity_model: Optional[BaseEstimator] = None,
        trim_weights: bool = True,
        weight_threshold: float = 0.1,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize Inverse Probability Weighting.
        
        Args:
            propensity_model: Model for estimating propensity scores
            trim_weights: Whether to trim extreme weights
            weight_threshold: Threshold for trimming weights
            random_state: Random seed for reproducibility
        """
        self.propensity_model = propensity_model or LogisticRegression(random_state=random_state)
        self.trim_weights = trim_weights
        self.weight_threshold = weight_threshold
        self.random_state = random_state
        
        self.propensity_scores_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: Optional[np.ndarray] = None) -> InverseProbabilityWeighting:
        """
        Fit the IPW model.
        
        Args:
            X: Covariates matrix
            treatment: Binary treatment indicator
            outcome: Outcome variable (optional, for convenience)
            
        Returns:
            Self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Estimate propensity scores
        self.propensity_model.fit(X, treatment)
        self.propensity_scores_ = self.propensity_model.predict_proba(X)[:, 1]
        
        # Calculate IPW weights
        self.weights_ = self._calculate_weights(treatment, self.propensity_scores_)
        
        self.is_fitted_ = True
        return self
    
    def _calculate_weights(self, treatment: np.ndarray, propensity_scores: np.ndarray) -> np.ndarray:
        """Calculate IPW weights."""
        # Stabilized IPW weights
        weights = np.zeros(len(treatment))
        
        # Weight for treated units: 1 / P(T=1|X)
        treated_mask = treatment == 1
        weights[treated_mask] = 1.0 / propensity_scores[treated_mask]
        
        # Weight for control units: 1 / P(T=0|X) = 1 / (1 - P(T=1|X))
        control_mask = treatment == 0
        weights[control_mask] = 1.0 / (1.0 - propensity_scores[control_mask])
        
        # Trim extreme weights if requested
        if self.trim_weights:
            weights = self._trim_weights(weights, treatment, propensity_scores)
        
        return weights
    
    def _trim_weights(self, weights: np.ndarray, treatment: np.ndarray, propensity_scores: np.ndarray) -> np.ndarray:
        """Trim extreme weights to reduce variance."""
        trimmed_weights = weights.copy()
        
        # Calculate trimming thresholds
        treated_mask = treatment == 1
        control_mask = treatment == 0
        
        if np.sum(treated_mask) > 0:
            treated_ps = propensity_scores[treated_mask]
            treated_threshold = np.percentile(treated_ps, self.weight_threshold * 100)
            treated_max_weight = 1.0 / treated_threshold
            
            treated_weights = trimmed_weights[treated_mask]
            trimmed_weights[treated_mask] = np.minimum(treated_weights, treated_max_weight)
        
        if np.sum(control_mask) > 0:
            control_ps = propensity_scores[control_mask]
            control_threshold = np.percentile(control_ps, (1 - self.weight_threshold) * 100)
            control_max_weight = 1.0 / (1.0 - control_threshold)
            
            control_weights = trimmed_weights[control_mask]
            trimmed_weights[control_mask] = np.minimum(control_weights, control_max_weight)
        
        return trimmed_weights
    
    def estimate_ate(self, outcome: np.ndarray) -> float:
        """
        Estimate Average Treatment Effect using IPW.
        
        Args:
            outcome: Outcome variable
            
        Returns:
            Estimated ATE
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before estimate_ate()")
        
        # Weighted average of outcomes
        weighted_outcome = np.sum(self.weights_ * outcome)
        total_weight = np.sum(self.weights_)
        
        if total_weight == 0:
            return np.nan
        
        # Calculate ATE using Hajek estimator
        treated_mask = self.weights_ > 0  # All units contribute
        treated_weighted_outcome = np.sum(self.weights_[treated_mask] * outcome[treated_mask])
        treated_total_weight = np.sum(self.weights_[treated_mask])
        
        control_mask = self.weights_ > 0
        control_weighted_outcome = np.sum(self.weights_[control_mask] * outcome[control_mask])
        control_total_weight = np.sum(self.weights_[control_mask])
        
        if treated_total_weight == 0 or control_total_weight == 0:
            return np.nan
        
        ate = (treated_weighted_outcome / treated_total_weight) - (control_weighted_outcome / control_total_weight)
        return ate
    
    def estimate_att(self, outcome: np.ndarray) -> float:
        """
        Estimate Average Treatment Effect on the Treated.
        
        Args:
            outcome: Outcome variable
            
        Returns:
            Estimated ATT
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before estimate_att()")
        
        treated_mask = self.weights_ > 0
        control_mask = self.weights_ > 0
        
        # Weighted average for treated units
        treated_outcomes = outcome[treated_mask]
        treated_weights = self.weights_[treated_mask]
        treated_mean = np.sum(treated_weights * treated_outcomes) / np.sum(treated_weights)
        
        # Weighted average for control units (using IPW weights)
        control_outcomes = outcome[control_mask]
        control_weights = self.weights_[control_mask]
        control_mean = np.sum(control_weights * control_outcomes) / np.sum(control_weights)
        
        att = treated_mean - control_mean
        return att
    
    def get_effective_sample_size(self) -> float:
        """
        Calculate effective sample size after weighting.
        
        Returns:
            Effective sample size
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before get_effective_sample_size()")
        
        # Effective sample size = (sum of weights)^2 / sum of weights^2
        total_weight = np.sum(self.weights_)
        sum_squared_weights = np.sum(self.weights_**2)
        
        if sum_squared_weights == 0:
            return 0
        
        effective_n = total_weight**2 / sum_squared_weights
        return effective_n
    
    def get_weight_statistics(self) -> dict[str, float]:
        """
        Get statistics about the IPW weights.
        
        Returns:
            Dictionary with weight statistics
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before get_weight_statistics()")
        
        return {
            "mean_weight": np.mean(self.weights_),
            "std_weight": np.std(self.weights_),
            "min_weight": np.min(self.weights_),
            "max_weight": np.max(self.weights_),
            "median_weight": np.median(self.weights_),
            "effective_sample_size": self.get_effective_sample_size(),
        }
