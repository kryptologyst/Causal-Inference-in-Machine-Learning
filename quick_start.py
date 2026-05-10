"""Quick start script for causal inference experiments."""

#!/usr/bin/env python3
"""
Quick start script for causal inference experiments.

Usage:
    python quick_start.py --data-type synthetic --n-samples 1000
    python quick_start.py --data-type heterogeneous --n-samples 2000
    python quick_start.py --data-type lalonde
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.synthetic import generate_synthetic_data, generate_heterogeneous_data, load_lalonde_data
from src.models.propensity_score import PropensityScoreMatching
from src.models.ipw import InverseProbabilityWeighting
from src.models.double_ml import DoubleMachineLearning
from src.models.causal_forest import CausalForest
from src.evaluation.metrics import evaluate_causal_effect
from src.utils.helpers import set_random_seed


def main():
    """Main function for quick start."""
    parser = argparse.ArgumentParser(description="Quick Start Causal Inference")
    parser.add_argument("--data-type", type=str, default="synthetic",
                       choices=["synthetic", "heterogeneous", "lalonde"],
                       help="Type of data to generate")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples (ignored for lalonde)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--models", nargs="+", 
                       default=["psm", "ipw", "double_ml", "causal_forest"],
                       choices=["psm", "ipw", "double_ml", "causal_forest"],
                       help="Models to run")
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.random_state)
    
    print("🚀 Causal Inference Quick Start")
    print("=" * 40)
    print(f"Data Type: {args.data_type}")
    print(f"Random Seed: {args.random_state}")
    print(f"Models: {', '.join(args.models)}")
    print()
    
    # Generate data
    print("📊 Generating data...")
    if args.data_type == "synthetic":
        data = generate_synthetic_data(
            n_samples=args.n_samples,
            n_features=5,
            treatment_effect=2.0,
            random_state=args.random_state,
        )
    elif args.data_type == "heterogeneous":
        data = generate_heterogeneous_data(
            n_samples=args.n_samples,
            n_features=5,
            random_state=args.random_state,
        )
    elif args.data_type == "lalonde":
        data = load_lalonde_data()
    
    print(f"   Samples: {len(data.X)}")
    print(f"   Features: {data.X.shape[1]}")
    print(f"   Treatment Rate: {data.treatment.mean():.3f}")
    print(f"   True ATE: {data.true_ate:.3f}")
    print()
    
    # Initialize models
    models = {}
    if "psm" in args.models:
        models["PSM"] = PropensityScoreMatching(random_state=args.random_state)
    if "ipw" in args.models:
        models["IPW"] = InverseProbabilityWeighting(random_state=args.random_state)
    if "double_ml" in args.models:
        models["Double ML"] = DoubleMachineLearning(random_state=args.random_state)
    if "causal_forest" in args.models:
        models["Causal Forest"] = CausalForest(n_estimators=50, random_state=args.random_state)
    
    # Run experiments
    print("🤖 Running experiments...")
    results = {}
    
    for name, model in models.items():
        print(f"   Fitting {name}...")
        try:
            model.fit(data.X, data.treatment, data.outcome)
            
            if hasattr(model, 'estimate_ate'):
                if name == "Double ML":
                    ate = model.estimate_ate()
                else:
                    ate = model.estimate_ate(data.outcome)
            else:
                ate = np.nan
            
            results[name] = ate
            print(f"      ATE: {ate:.4f}")
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
            results[name] = np.nan
    
    # Print summary
    print("\n📋 RESULTS SUMMARY")
    print("=" * 40)
    print(f"True ATE: {data.true_ate:.4f}")
    print()
    
    for name, ate in results.items():
        if not np.isnan(ate):
            bias = ate - data.true_ate
            relative_bias = (bias / data.true_ate) * 100
            print(f"{name:15} ATE: {ate:8.4f} | Bias: {bias:8.4f} | Rel. Bias: {relative_bias:6.1f}%")
        else:
            print(f"{name:15} ATE: {'N/A':>8} | Bias: {'N/A':>8} | Rel. Bias: {'N/A':>6}")
    
    print("\n✅ Quick start completed!")
    print("\n⚠️  Remember: This is for research/educational purposes only.")
    print("   Not for production decisions without expert review.")


if __name__ == "__main__":
    main()
