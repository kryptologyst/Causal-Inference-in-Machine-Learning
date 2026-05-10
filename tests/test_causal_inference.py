"""Unit tests for causal inference methods."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.data.synthetic import generate_synthetic_data, generate_heterogeneous_data, check_overlap
from src.models.propensity_score import PropensityScoreMatching
from src.models.ipw import InverseProbabilityWeighting
from src.models.double_ml import DoubleMachineLearning
from src.models.causal_forest import CausalForest
from src.evaluation.metrics import evaluate_causal_effect, calculate_balance_metrics


class TestDataGeneration:
    """Test data generation functions."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        data = generate_synthetic_data(n_samples=100, n_features=3, random_state=42)
        
        assert len(data.X) == 100
        assert data.X.shape[1] == 3
        assert len(data.treatment) == 100
        assert len(data.outcome) == 100
        assert data.true_ate is not None
        assert len(data.feature_names) == 3
    
    def test_generate_heterogeneous_data(self):
        """Test heterogeneous data generation."""
        data = generate_heterogeneous_data(n_samples=100, n_features=3, random_state=42)
        
        assert len(data.X) == 100
        assert data.X.shape[1] == 3
        assert len(data.treatment) == 100
        assert len(data.outcome) == 100
        assert data.true_ate is not None
    
    def test_data_consistency(self):
        """Test data consistency."""
        data = generate_synthetic_data(n_samples=50, random_state=42)
        
        assert len(data.X) == len(data.treatment)
        assert len(data.X) == len(data.outcome)
        assert np.all(np.isin(data.treatment, [0, 1]))
    
    def test_check_overlap(self):
        """Test overlap checking."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        overlap_stats = check_overlap(data)
        
        assert "has_overlap" in overlap_stats
        assert "overlap_range" in overlap_stats
        assert isinstance(overlap_stats["has_overlap"], bool)


class TestPropensityScoreMatching:
    """Test Propensity Score Matching."""
    
    def test_psm_initialization(self):
        """Test PSM initialization."""
        psm = PropensityScoreMatching(caliper=0.1, n_neighbors=1)
        
        assert psm.caliper == 0.1
        assert psm.n_neighbors == 1
        assert not psm.is_fitted_
    
    def test_psm_fit(self):
        """Test PSM fitting."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        psm = PropensityScoreMatching(random_state=42)
        
        psm.fit(data.X, data.treatment, data.outcome)
        
        assert psm.is_fitted_
        assert psm.ps_model_ is not None
        assert psm.propensity_scores_ is not None
        assert psm.matches_ is not None
    
    def test_psm_estimate_ate(self):
        """Test ATE estimation."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        psm = PropensityScoreMatching(random_state=42)
        
        psm.fit(data.X, data.treatment, data.outcome)
        ate = psm.estimate_ate(data.outcome)
        
        assert isinstance(ate, float)
        assert not np.isnan(ate)
    
    def test_psm_balance_stats(self):
        """Test balance statistics."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        psm = PropensityScoreMatching(random_state=42)
        
        psm.fit(data.X, data.treatment, data.outcome)
        balance_stats = psm.get_balance_stats(data.X, data.feature_names)
        
        assert isinstance(balance_stats, pd.DataFrame)
        assert "feature" in balance_stats.columns
        assert "std_diff" in balance_stats.columns


class TestInverseProbabilityWeighting:
    """Test Inverse Probability Weighting."""
    
    def test_ipw_initialization(self):
        """Test IPW initialization."""
        ipw = InverseProbabilityWeighting(trim_weights=True, weight_threshold=0.1)
        
        assert ipw.trim_weights is True
        assert ipw.weight_threshold == 0.1
        assert not ipw.is_fitted_
    
    def test_ipw_fit(self):
        """Test IPW fitting."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        ipw = InverseProbabilityWeighting(random_state=42)
        
        ipw.fit(data.X, data.treatment, data.outcome)
        
        assert ipw.is_fitted_
        assert ipw.propensity_scores_ is not None
        assert ipw.weights_ is not None
    
    def test_ipw_estimate_ate(self):
        """Test ATE estimation."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        ipw = InverseProbabilityWeighting(random_state=42)
        
        ipw.fit(data.X, data.treatment, data.outcome)
        ate = ipw.estimate_ate(data.outcome)
        
        assert isinstance(ate, float)
        assert not np.isnan(ate)
    
    def test_ipw_weight_statistics(self):
        """Test weight statistics."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        ipw = InverseProbabilityWeighting(random_state=42)
        
        ipw.fit(data.X, data.treatment, data.outcome)
        weight_stats = ipw.get_weight_statistics()
        
        assert isinstance(weight_stats, dict)
        assert "mean_weight" in weight_stats
        assert "effective_sample_size" in weight_stats


class TestDoubleMachineLearning:
    """Test Double Machine Learning."""
    
    def test_double_ml_initialization(self):
        """Test Double ML initialization."""
        dml = DoubleMachineLearning(n_folds=5)
        
        assert dml.n_folds == 5
        assert not dml.is_fitted_
    
    def test_double_ml_fit(self):
        """Test Double ML fitting."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        dml = DoubleMachineLearning(random_state=42)
        
        dml.fit(data.X, data.treatment, data.outcome)
        
        assert dml.is_fitted_
        assert dml.outcome_residuals_ is not None
        assert dml.treatment_residuals_ is not None
    
    def test_double_ml_estimate_ate(self):
        """Test ATE estimation."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        dml = DoubleMachineLearning(random_state=42)
        
        dml.fit(data.X, data.treatment, data.outcome)
        ate = dml.estimate_ate()
        
        assert isinstance(ate, float)
        assert not np.isnan(ate)
    
    def test_double_ml_confidence_interval(self):
        """Test confidence interval calculation."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        dml = DoubleMachineLearning(random_state=42)
        
        dml.fit(data.X, data.treatment, data.outcome)
        ci = dml.get_confidence_interval()
        
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]


class TestCausalForest:
    """Test Causal Forest."""
    
    def test_causal_forest_initialization(self):
        """Test Causal Forest initialization."""
        cf = CausalForest(n_estimators=50, max_depth=5)
        
        assert cf.n_estimators == 50
        assert cf.max_depth == 5
        assert not cf.is_fitted_
    
    def test_causal_forest_fit(self):
        """Test Causal Forest fitting."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        cf = CausalForest(n_estimators=10, random_state=42)  # Small forest for testing
        
        cf.fit(data.X, data.treatment, data.outcome)
        
        assert cf.is_fitted_
        assert len(cf.trees_) == 10
        assert cf.feature_importances_ is not None
    
    def test_causal_forest_predict(self):
        """Test CATE prediction."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        cf = CausalForest(n_estimators=10, random_state=42)
        
        cf.fit(data.X, data.treatment, data.outcome)
        predictions = cf.predict(data.X)
        
        assert len(predictions) == len(data.X)
        assert not np.any(np.isnan(predictions))


class TestEvaluationMetrics:
    """Test evaluation metrics."""
    
    def test_evaluate_causal_effect(self):
        """Test causal effect evaluation."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        psm = PropensityScoreMatching(random_state=42)
        
        psm.fit(data.X, data.treatment, data.outcome)
        result = evaluate_causal_effect(psm, data)
        
        assert result.ate is not None
        assert result.n_treated is not None
        assert result.n_control is not None
    
    def test_calculate_balance_metrics(self):
        """Test balance metrics calculation."""
        data = generate_synthetic_data(n_samples=100, random_state=42)
        balance_stats = calculate_balance_metrics(
            data.X, data.treatment, data.feature_names
        )
        
        assert isinstance(balance_stats, pd.DataFrame)
        assert "feature" in balance_stats.columns
        assert "std_diff" in balance_stats.columns
        assert len(balance_stats) == data.X.shape[1]


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_experiment(self):
        """Test complete end-to-end experiment."""
        # Generate data
        data = generate_synthetic_data(n_samples=200, random_state=42)
        
        # Test multiple models
        models = {
            "PSM": PropensityScoreMatching(random_state=42),
            "IPW": InverseProbabilityWeighting(random_state=42),
            "Double ML": DoubleMachineLearning(random_state=42),
        }
        
        results = {}
        for name, model in models.items():
            model.fit(data.X, data.treatment, data.outcome)
            
            if hasattr(model, 'estimate_ate'):
                if name == "Double ML":
                    ate = model.estimate_ate()
                else:
                    ate = model.estimate_ate(data.outcome)
            else:
                ate = np.nan
            
            results[name] = ate
        
        # Check that all models produce reasonable estimates
        for name, ate in results.items():
            assert isinstance(ate, float)
            assert not np.isnan(ate)
            assert abs(ate) < 10  # Reasonable range for synthetic data
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        data1 = generate_synthetic_data(n_samples=100, random_state=42)
        data2 = generate_synthetic_data(n_samples=100, random_state=42)
        
        # Check that same seed produces same data
        assert np.allclose(data1.X, data2.X)
        assert np.allclose(data1.treatment, data2.treatment)
        assert np.allclose(data1.outcome, data2.outcome)
        
        # Test model reproducibility
        psm1 = PropensityScoreMatching(random_state=42)
        psm2 = PropensityScoreMatching(random_state=42)
        
        psm1.fit(data1.X, data1.treatment, data1.outcome)
        psm2.fit(data2.X, data2.treatment, data2.outcome)
        
        ate1 = psm1.estimate_ate(data1.outcome)
        ate2 = psm2.estimate_ate(data2.outcome)
        
        assert np.isclose(ate1, ate2, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
