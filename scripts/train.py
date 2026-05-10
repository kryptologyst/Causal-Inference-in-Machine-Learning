"""Main training and evaluation script for causal inference."""

from __future__ import annotations

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from src.data.synthetic import generate_synthetic_data, generate_heterogeneous_data, load_lalonde_data
from src.models.propensity_score import PropensityScoreMatching
from src.models.ipw import InverseProbabilityWeighting
from src.models.double_ml import DoubleMachineLearning
from src.models.causal_forest import CausalForest
from src.evaluation.metrics import evaluate_causal_effect, create_evaluation_report
from src.visualization.plots import plot_treatment_effects, create_interactive_plot


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('causal_inference.log'),
            logging.StreamHandler()
        ]
    )


def run_experiment(
    config: Dict,
    data_type: str = "synthetic",
    n_samples: int = 1000,
    n_features: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    Run a complete causal inference experiment.
    
    Args:
        config: Experiment configuration
        data_type: Type of data to generate
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random seed
        
    Returns:
        Dictionary with experiment results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment with {data_type} data")
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Generate data
    if data_type == "synthetic":
        data = generate_synthetic_data(
            n_samples=n_samples,
            n_features=n_features,
            treatment_effect=config.get("true_ate", 2.0),
            random_state=random_state,
        )
    elif data_type == "heterogeneous":
        data = generate_heterogeneous_data(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
        )
    elif data_type == "lalonde":
        data = load_lalonde_data()
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    logger.info(f"Generated data with {len(data.X)} samples and {data.X.shape[1]} features")
    logger.info(f"Treatment rate: {np.mean(data.treatment):.3f}")
    
    # Initialize models
    models = {}
    model_configs = config.get("models", {})
    
    if model_configs.get("psm", {}).get("enabled", True):
        models["PSM"] = PropensityScoreMatching(
            caliper=model_configs["psm"].get("caliper", 0.1),
            n_neighbors=model_configs["psm"].get("n_neighbors", 1),
            random_state=random_state,
        )
    
    if model_configs.get("ipw", {}).get("enabled", True):
        models["IPW"] = InverseProbabilityWeighting(
            trim_weights=model_configs["ipw"].get("trim_weights", True),
            weight_threshold=model_configs["ipw"].get("weight_threshold", 0.1),
            random_state=random_state,
        )
    
    if model_configs.get("double_ml", {}).get("enabled", True):
        models["Double ML"] = DoubleMachineLearning(
            n_folds=model_configs["double_ml"].get("n_folds", 5),
            random_state=random_state,
        )
    
    if model_configs.get("causal_forest", {}).get("enabled", True):
        models["Causal Forest"] = CausalForest(
            n_estimators=model_configs["causal_forest"].get("n_estimators", 100),
            max_depth=model_configs["causal_forest"].get("max_depth", None),
            random_state=random_state,
        )
    
    # Fit models and evaluate
    results = []
    model_names = []
    
    for name, model in models.items():
        logger.info(f"Fitting {name} model")
        
        try:
            # Fit model
            model.fit(data.X, data.treatment, data.outcome)
            
            # Evaluate
            result = evaluate_causal_effect(
                model, data,
                confidence_level=config.get("confidence_level", 0.95),
                n_bootstrap=config.get("n_bootstrap", 1000),
            )
            
            # Add model-specific results
            result_dict = {
                "ate": result.ate,
                "att": result.att,
                "confidence_interval": result.confidence_interval,
                "p_value": result.p_value,
                "n_treated": result.n_treated,
                "n_control": result.n_control,
                "n_matched": result.n_matched,
            }
            
            # Add model-specific metrics
            if hasattr(model, "get_weight_statistics"):
                weight_stats = model.get_weight_statistics()
                result_dict.update(weight_stats)
            
            if hasattr(model, "get_feature_importance"):
                feature_importance = model.get_feature_importance()
                result_dict["feature_importance"] = feature_importance.tolist()
            
            results.append(result_dict)
            model_names.append(name)
            
            logger.info(f"{name} - ATE: {result.ate:.4f}")
            
        except Exception as e:
            logger.error(f"Error fitting {name}: {str(e)}")
            continue
    
    # Create evaluation report
    evaluation_report = create_evaluation_report(
        results, model_names, true_ate=data.true_ate
    )
    
    logger.info("Experiment completed successfully")
    
    return {
        "data": {
            "n_samples": len(data.X),
            "n_features": data.X.shape[1],
            "treatment_rate": float(np.mean(data.treatment)),
            "true_ate": data.true_ate,
        },
        "results": results,
        "model_names": model_names,
        "evaluation_report": evaluation_report.to_dict("records"),
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Causal Inference Experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data-type", type=str, default="synthetic",
                       choices=["synthetic", "heterogeneous", "lalonde"],
                       help="Type of data to generate")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples")
    parser.add_argument("--n-features", type=int, default=5,
                       help="Number of features")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--output-dir", type=str, default="assets",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        config = {
            "true_ate": 2.0,
            "confidence_level": 0.95,
            "n_bootstrap": 1000,
            "models": {
                "psm": {"enabled": True, "caliper": 0.1},
                "ipw": {"enabled": True, "trim_weights": True},
                "double_ml": {"enabled": True, "n_folds": 5},
                "causal_forest": {"enabled": True, "n_estimators": 100},
            }
        }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Run experiment
    try:
        results = run_experiment(
            config=config,
            data_type=args.data_type,
            n_samples=args.n_samples,
            n_features=args.n_features,
            random_state=args.random_state,
        )
        
        # Save results
        results_file = output_dir / f"results_{args.data_type}_{args.random_state}.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info(f"Results saved to {results_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        print(f"Data: {args.data_type}")
        print(f"Samples: {results['data']['n_samples']}")
        print(f"Features: {results['data']['n_features']}")
        print(f"Treatment Rate: {results['data']['treatment_rate']:.3f}")
        print(f"True ATE: {results['data']['true_ate']:.3f}")
        print("\nModel Results:")
        
        for i, (result, name) in enumerate(zip(results['results'], results['model_names'])):
            ate = result.get('ate', np.nan)
            ci = result.get('confidence_interval', None)
            print(f"{name:15} ATE: {ate:8.4f}", end="")
            if ci is not None:
                print(f" CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            else:
                print()
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
