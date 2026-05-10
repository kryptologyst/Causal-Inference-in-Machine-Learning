# Causal Inference in Machine Learning

A research-ready implementation of causal inference methods for machine learning, featuring classical and advanced techniques for estimating causal effects from observational data.

**Author:** [kryptologyst](https://github.com/kryptologyst)  
**GitHub:** https://github.com/kryptologyst

## ⚠️ Important Disclaimers

- **Research/Educational Purpose Only**: This project is designed for research and educational purposes
- **Not for Production Decisions**: Do not use these methods for critical production decisions without expert review
- **Causal Claims Require Domain Expertise**: Causal inference requires deep domain knowledge and careful validation
- **Observational Data Limitations**: Results from observational studies may not generalize to new contexts

## Overview

This project implements a comprehensive suite of causal inference methods, including:

- **Classical Methods**: Propensity Score Matching (PSM), Inverse Probability Weighting (IPW)
- **Advanced Methods**: Double Machine Learning (Double ML), Causal Forest, Meta-learners
- **Evaluation Framework**: Comprehensive metrics and visualization tools
- **Interactive Demo**: Web-based interface for exploring causal effects

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Causal-Inference-in-Machine-Learning.git
cd Causal-Inference-in-Machine-Learning

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.data.synthetic import generate_synthetic_data
from src.models.propensity_score import PropensityScoreMatching
from src.evaluation.metrics import evaluate_causal_effect

# Generate synthetic data
data = generate_synthetic_data(n_samples=1000, n_features=5)

# Fit propensity score matching
psm = PropensityScoreMatching()
psm.fit(data.X, data.treatment, data.outcome)

# Estimate causal effect
ate = psm.estimate_ate()
print(f"Average Treatment Effect: {ate:.4f}")

# Evaluate with confidence intervals
results = evaluate_causal_effect(psm, data)
print(results)
```

## Methods Implemented

### Classical Baselines
- **Propensity Score Matching (PSM)**: Match units with similar propensity scores
- **Inverse Probability Weighting (IPW)**: Weight observations by inverse propensity
- **Regression Adjustment**: Control for confounders via regression

### Advanced Methods
- **Double Machine Learning**: Use ML to estimate nuisance parameters
- **Causal Forest**: Tree-based method for heterogeneous treatment effects
- **Meta-learners**: S/T/R-learners for flexible causal effect estimation
- **Instrumental Variables**: Two-stage least squares for endogeneity

## Evaluation Metrics

- **Average Treatment Effect (ATE)**: Overall causal effect
- **Conditional Average Treatment Effect (CATE)**: Effect for specific subgroups
- **Confidence Intervals**: Bootstrap and analytical confidence intervals
- **Balance Diagnostics**: Covariate balance after matching/weighting
- **Sensitivity Analysis**: Robustness to unobserved confounding

## Interactive Demo

Launch the interactive demo to explore causal effects:

```bash
# Streamlit demo
streamlit run demo/streamlit_app.py

# Gradio demo
python demo/gradio_app.py
```

The demo allows you to:
- Upload your own datasets
- Select causal inference methods
- Visualize treatment effects
- Explore sensitivity analysis
- Download results and plots

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data loading and generation
│   ├── models/            # Causal inference methods
│   ├── evaluation/        # Metrics and evaluation
│   ├── visualization/     # Plotting utilities
│   └── utils/             # Helper functions
├── configs/               # Configuration files
├── data/                  # Data storage
│   ├── raw/              # Raw datasets
│   └── processed/        # Processed datasets
├── assets/               # Generated plots and results
├── tests/                # Unit tests
├── demo/                 # Interactive demos
├── notebooks/            # Jupyter notebooks
└── scripts/              # Training and evaluation scripts
```

## Research Applications

This toolkit is suitable for:
- **Policy Evaluation**: Estimating policy impacts from observational data
- **Medical Research**: Analyzing treatment effects in healthcare
- **Economics**: Understanding causal relationships in economic data
- **Social Science**: Studying intervention effects in social contexts
- **Business Analytics**: Measuring marketing campaign effectiveness

## Key Concepts

### Causal Inference
The process of determining whether a relationship between variables is causal rather than correlational.

### Treatment Effects
- **ATE**: Average Treatment Effect across the population
- **ATT**: Average Treatment Effect on the Treated
- **ATC**: Average Treatment Effect on the Control

### Confounding
Variables that affect both treatment assignment and outcomes, potentially biasing causal estimates.

### Identification Assumptions
- **Unconfoundedness**: No unobserved confounders
- **Overlap**: All units have positive probability of treatment
- **SUTVA**: Stable Unit Treatment Value Assumption

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/
ruff check src/ tests/
```

### Type Checking
```bash
mypy src/
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to the main repository.
# Causal-Inference-in-Machine-Learning
