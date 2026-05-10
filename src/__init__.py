"""Package initialization."""

__version__ = "0.1.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"
__description__ = "Modern causal inference methods for machine learning research and education"

# Import main classes for easy access
from .data.synthetic import generate_synthetic_data, generate_heterogeneous_data, CausalData
from .models.propensity_score import PropensityScoreMatching
from .models.ipw import InverseProbabilityWeighting
from .models.double_ml import DoubleMachineLearning
from .models.causal_forest import CausalForest
from .evaluation.metrics import evaluate_causal_effect, CausalResults
from .utils.helpers import set_random_seed, check_data_quality, validate_assumptions

__all__ = [
    "generate_synthetic_data",
    "generate_heterogeneous_data", 
    "CausalData",
    "PropensityScoreMatching",
    "InverseProbabilityWeighting",
    "DoubleMachineLearning",
    "CausalForest",
    "evaluate_causal_effect",
    "CausalResults",
    "set_random_seed",
    "check_data_quality",
    "validate_assumptions",
]
