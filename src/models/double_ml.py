"""Double Machine Learning implementation."""

from __future__ import annotations

import numpy as np
from typing import Optional, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict


class DoubleMachineLearning(BaseEstimator, RegressorMixin):
    """
    Double Machine Learning for causal inference.
    
    Uses machine learning to estimate nuisance parameters (propensity scores
    and outcome models) to reduce bias in causal effect estimation.
    """
    
    def __init__(
        self,
        outcome_model: Optional[BaseEstimator] = None,
        propensity_model: Optional[BaseEstimator] = None,
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize Double Machine Learning.
        
        Args:
            outcome_model: Model for predicting outcomes
            propensity_model: Model for predicting propensity scores
            n_folds: Number of folds for cross-fitting
            random_state: Random seed for reproducibility
        """
        self.outcome_model = outcome_model or RandomForestRegressor(random_state=random_state)
        self.propensity_model = propensity_model or RandomForestRegressor(random_state=random_state)
        self.n_folds = n_folds
        self.random_state = random_state
        
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> DoubleMachineLearning:
        """
        Fit the Double ML model.
        
        Args:
            X: Covariates matrix
            treatment: Binary treatment indicator
            outcome: Outcome variable
            
        Returns:
            Self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = len(X)
        
        # Cross-fit outcome model
        outcome_pred = cross_val_predict(
            self.outcome_model, X, outcome, cv=self.n_folds, random_state=self.random_state
        )
        
        # Cross-fit propensity model
        propensity_pred = cross_val_predict(
            self.propensity_model, X, treatment, cv=self.n_folds, random_state=self.random_state
        )
        
        # Store residuals
        self.outcome_residuals_ = outcome - outcome_pred
        self.treatment_residuals_ = treatment - propensity_pred
        
        # Fit final models for prediction
        self.outcome_model.fit(X, outcome)
        self.propensity_model.fit(X, treatment)
        
        self.is_fitted_ = True
        return self
    
    def estimate_ate(self) -> float:
        """
        Estimate Average Treatment Effect using Double ML.
        
        Returns:
            Estimated ATE
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before estimate_ate()")
        
        # Calculate ATE using residuals
        numerator = np.mean(self.outcome_residuals_ * self.treatment_residuals_)
        denominator = np.mean(self.treatment_residuals_ * self.treatment_residuals_)
        
        if denominator == 0:
            return np.nan
        
        ate = numerator / denominator
        return ate
    
    def estimate_cate(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate Conditional Average Treatment Effects.
        
        Args:
            X: Covariates for which to estimate CATE
            
        Returns:
            Estimated CATE for each observation
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before estimate_cate()")
        
        # Predict outcomes under treatment and control
        outcome_treated = self.outcome_model.predict(X)
        outcome_control = self.outcome_model.predict(X)
        
        # Add treatment effect
        cate = outcome_treated - outcome_control
        return cate
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outcomes.
        
        Args:
            X: Covariates
            
        Returns:
            Predicted outcomes
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before predict()")
        
        return self.outcome_model.predict(X)
    
    def get_confidence_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        """
        Calculate confidence interval for ATE using bootstrap.
        
        Args:
            alpha: Significance level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before get_confidence_interval()")
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_ates = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            n_samples = len(self.outcome_residuals_)
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            boot_outcome_residuals = self.outcome_residuals_[bootstrap_indices]
            boot_treatment_residuals = self.treatment_residuals_[bootstrap_indices]
            
            # Calculate ATE
            numerator = np.mean(boot_outcome_residuals * boot_treatment_residuals)
            denominator = np.mean(boot_treatment_residuals * boot_treatment_residuals)
            
            if denominator != 0:
                bootstrap_ates.append(numerator / denominator)
        
        # Calculate confidence interval
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_ates, lower_percentile)
        upper_bound = np.percentile(bootstrap_ates, upper_percentile)
        
        return lower_bound, upper_bound
