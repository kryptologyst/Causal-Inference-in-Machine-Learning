"""Causal Forest implementation."""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


class CausalTree(BaseEstimator, RegressorMixin):
    """
    Causal Tree for estimating heterogeneous treatment effects.
    
    Uses a modified splitting criterion that maximizes the difference
    in treatment effects between child nodes.
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize Causal Tree.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples in leaf nodes
            random_state: Random seed for reproducibility
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.tree_ = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> CausalTree:
        """
        Fit the causal tree.
        
        Args:
            X: Covariates matrix
            treatment: Binary treatment indicator
            outcome: Outcome variable
            
        Returns:
            Self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Use custom splitting criterion
        self.tree_ = self._build_tree(X, treatment, outcome, depth=0)
        self.is_fitted_ = True
        
        return self
    
    def _build_tree(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, depth: int) -> dict:
        """Build causal tree recursively."""
        n_samples = len(X)
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split:
            return self._create_leaf(treatment, outcome)
        
        # Find best split
        best_split = self._find_best_split(X, treatment, outcome)
        
        if best_split is None:
            return self._create_leaf(treatment, outcome)
        
        feature_idx, threshold = best_split
        
        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Check minimum samples in each child
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return self._create_leaf(treatment, outcome)
        
        # Recursively build children
        left_tree = self._build_tree(
            X[left_mask], treatment[left_mask], outcome[left_mask], depth + 1
        )
        right_tree = self._build_tree(
            X[right_mask], treatment[right_mask], outcome[right_mask], depth + 1
        )
        
        return {
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree,
        }
    
    def _find_best_split(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> Optional[Tuple[int, float]]:
        """Find the best split using causal splitting criterion."""
        best_gain = 0
        best_split = None
        
        for feature_idx in range(X.shape[1]):
            unique_values = np.unique(X[:, feature_idx])
            
            for threshold in unique_values[:-1]:  # Exclude last value
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate causal splitting criterion
                gain = self._causal_split_gain(
                    treatment[left_mask], outcome[left_mask],
                    treatment[right_mask], outcome[right_mask]
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, threshold)
        
        return best_split
    
    def _causal_split_gain(self, t_left: np.ndarray, y_left: np.ndarray, 
                          t_right: np.ndarray, y_right: np.ndarray) -> float:
        """Calculate causal splitting criterion."""
        # Estimate treatment effects in each child
        effect_left = self._estimate_treatment_effect(t_left, y_left)
        effect_right = self._estimate_treatment_effect(t_right, y_right)
        
        # Causal splitting criterion: maximize difference in treatment effects
        gain = (effect_left - effect_right) ** 2
        
        return gain
    
    def _estimate_treatment_effect(self, treatment: np.ndarray, outcome: np.ndarray) -> float:
        """Estimate treatment effect in a node."""
        treated_mask = treatment == 1
        control_mask = treatment == 0
        
        if np.sum(treated_mask) == 0 or np.sum(control_mask) == 0:
            return 0
        
        treated_mean = np.mean(outcome[treated_mask])
        control_mean = np.mean(outcome[control_mask])
        
        return treated_mean - control_mean
    
    def _create_leaf(self, treatment: np.ndarray, outcome: np.ndarray) -> dict:
        """Create a leaf node."""
        effect = self._estimate_treatment_effect(treatment, outcome)
        return {'effect': effect, 'is_leaf': True}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict treatment effects.
        
        Args:
            X: Covariates
            
        Returns:
            Predicted treatment effects
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before predict()")
        
        predictions = np.zeros(len(X))
        
        for i, x in enumerate(X):
            predictions[i] = self._predict_single(x, self.tree_)
        
        return predictions
    
    def _predict_single(self, x: np.ndarray, node: dict) -> float:
        """Predict for a single observation."""
        if node.get('is_leaf', False):
            return node['effect']
        
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])


class CausalForest(BaseEstimator, RegressorMixin):
    """
    Causal Forest for estimating heterogeneous treatment effects.
    
    Ensemble of causal trees that provides robust estimates of
    conditional average treatment effects (CATE).
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        max_features: Optional[str] = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize Causal Forest.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples in leaf nodes
            max_features: Number of features to consider for splits
            bootstrap: Whether to use bootstrap sampling
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self.trees_: List[CausalTree] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> CausalForest:
        """
        Fit the causal forest.
        
        Args:
            X: Covariates matrix
            treatment: Binary treatment indicator
            outcome: Outcome variable
            
        Returns:
            Self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Determine number of features to consider
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, n_features)
        else:
            max_features = n_features
        
        # Build forest
        self.trees_ = []
        feature_counts = np.zeros(n_features)
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap = X[bootstrap_indices]
                treatment_bootstrap = treatment[bootstrap_indices]
                outcome_bootstrap = outcome[bootstrap_indices]
            else:
                X_bootstrap = X
                treatment_bootstrap = treatment
                outcome_bootstrap = outcome
            
            # Feature subsampling
            if max_features < n_features:
                feature_indices = np.random.choice(n_features, size=max_features, replace=False)
                X_bootstrap = X_bootstrap[:, feature_indices]
            else:
                feature_indices = np.arange(n_features)
            
            # Build tree
            tree = CausalTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i if self.random_state is not None else None,
            )
            
            tree.fit(X_bootstrap, treatment_bootstrap, outcome_bootstrap)
            self.trees_.append((tree, feature_indices))
            
            # Track feature importance
            feature_counts[feature_indices] += 1
        
        # Calculate feature importances
        self.feature_importances_ = feature_counts / np.sum(feature_counts)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict treatment effects.
        
        Args:
            X: Covariates
            
        Returns:
            Predicted treatment effects
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before predict()")
        
        predictions = np.zeros(len(X))
        
        for tree, feature_indices in self.trees_:
            X_subset = X[:, feature_indices]
            tree_predictions = tree.predict(X_subset)
            predictions += tree_predictions
        
        # Average predictions
        predictions /= len(self.trees_)
        
        return predictions
    
    def estimate_ate(self) -> float:
        """
        Estimate Average Treatment Effect.
        
        Returns:
            Estimated ATE
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before estimate_ate()")
        
        # ATE is the average of all predicted CATEs
        # This is a simplified implementation
        return np.mean(self.predict(np.zeros((1, len(self.feature_importances_)))))
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted_:
            raise ValueError("Must call fit() before get_feature_importance()")
        
        return self.feature_importances_
