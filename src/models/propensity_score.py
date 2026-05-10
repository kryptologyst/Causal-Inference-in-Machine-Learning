"""Propensity Score Matching implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class PropensityScoreMatching:
    """
    Propensity Score Matching for causal inference.
    
    Matches treated and control units based on similar propensity scores
    to estimate causal effects from observational data.
    """
    
    def __init__(
        self,
        caliper: float = 0.1,
        n_neighbors: int = 1,
        replace: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize Propensity Score Matching.
        
        Args:
            caliper: Maximum distance for matching (in propensity score units)
            n_neighbors: Number of neighbors to match
            replace: Whether to allow replacement in matching
            random_state: Random seed for reproducibility
        """
        self.caliper = caliper
        self.n_neighbors = n_neighbors
        self.replace = replace
        self.random_state = random_state
        
        self.ps_model_: Optional[LogisticRegression] = None
        self.propensity_scores_: Optional[np.ndarray] = None
        self.matches_: Optional[list] = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: Optional[np.ndarray] = None) -> PropensityScoreMatching:
        """
        Fit the propensity score model and perform matching.
        
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
        self.ps_model_ = LogisticRegression(random_state=self.random_state)
        self.ps_model_.fit(X, treatment)
        self.propensity_scores_ = self.ps_model_.predict_proba(X)[:, 1]
        
        # Perform matching
        self.matches_ = self._perform_matching(treatment, self.propensity_scores_)
        self.is_fitted_ = True
        
        return self
    
    def _perform_matching(self, treatment: np.ndarray, propensity_scores: np.ndarray) -> list:
        """Perform propensity score matching."""
        treated_indices = np.where(treatment == 1)[0]
        control_indices = np.where(treatment == 0)[0]
        
        treated_ps = propensity_scores[treated_indices]
        control_ps = propensity_scores[control_indices]
        
        matches = []
        used_controls = set()
        
        for i, treated_idx in enumerate(treated_indices):
            treated_ps_val = treated_ps[i]
            
            # Find control units within caliper
            distances = np.abs(control_ps - treated_ps_val)
            valid_controls = np.where(distances <= self.caliper)[0]
            
            if len(valid_controls) == 0:
                # No match found within caliper
                matches.append((treated_idx, None))
                continue
            
            # Select best matches
            if not self.replace:
                # Remove already used controls
                valid_controls = [c for c in valid_controls if control_indices[c] not in used_controls]
            
            if len(valid_controls) == 0:
                matches.append((treated_idx, None))
                continue
            
            # Sort by distance and select n_neighbors
            sorted_indices = np.argsort(distances[valid_controls])
            selected_controls = sorted_indices[:self.n_neighbors]
            
            matched_controls = [control_indices[valid_controls[idx]] for idx in selected_controls]
            matches.append((treated_idx, matched_controls))
            
            if not self.replace:
                used_controls.update(matched_controls)
        
        return matches
    
    def estimate_ate(self, outcome: np.ndarray) -> float:
        """
        Estimate Average Treatment Effect.
        
        Args:
            outcome: Outcome variable
            
        Returns:
            Estimated ATE
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before estimate_ate()")
        
        treated_outcomes = []
        control_outcomes = []
        
        for treated_idx, control_indices in self.matches_:
            if control_indices is None:
                continue  # Skip unmatched units
            
            treated_outcomes.append(outcome[treated_idx])
            for control_idx in control_indices:
                control_outcomes.append(outcome[control_idx])
        
        if len(treated_outcomes) == 0:
            return np.nan
        
        ate = np.mean(treated_outcomes) - np.mean(control_outcomes)
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
        
        treated_outcomes = []
        matched_control_outcomes = []
        
        for treated_idx, control_indices in self.matches_:
            if control_indices is None:
                continue
            
            treated_outcomes.append(outcome[treated_idx])
            matched_control_outcomes.append(np.mean([outcome[c] for c in control_indices]))
        
        if len(treated_outcomes) == 0:
            return np.nan
        
        att = np.mean(treated_outcomes) - np.mean(matched_control_outcomes)
        return att
    
    def get_matched_data(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get matched dataset.
        
        Args:
            X: Covariates matrix
            treatment: Treatment indicator
            outcome: Outcome variable
            
        Returns:
            Tuple of (matched_X, matched_treatment, matched_outcome)
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before get_matched_data()")
        
        matched_indices = []
        
        for treated_idx, control_indices in self.matches_:
            if control_indices is None:
                continue
            
            matched_indices.append(treated_idx)
            matched_indices.extend(control_indices)
        
        matched_indices = np.array(matched_indices)
        
        return X[matched_indices], treatment[matched_indices], outcome[matched_indices]
    
    def get_balance_stats(self, X: np.ndarray, feature_names: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Calculate covariate balance statistics.
        
        Args:
            X: Covariates matrix
            feature_names: Names of features
            
        Returns:
            DataFrame with balance statistics
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before get_balance_stats()")
        
        matched_X, matched_treatment, _ = self.get_matched_data(X, np.zeros(len(X)), np.zeros(len(X)))
        
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(X.shape[1])]
        
        balance_stats = []
        
        for i, feature_name in enumerate(feature_names):
            treated_mean = np.mean(matched_X[matched_treatment == 1, i])
            control_mean = np.mean(matched_X[matched_treatment == 0, i])
            
            treated_std = np.std(matched_X[matched_treatment == 1, i])
            control_std = np.std(matched_X[matched_treatment == 0, i])
            
            # Standardized difference
            pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
            std_diff = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0
            
            balance_stats.append({
                "feature": feature_name,
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "treated_std": treated_std,
                "control_std": control_std,
                "std_diff": std_diff,
            })
        
        return pd.DataFrame(balance_stats)
