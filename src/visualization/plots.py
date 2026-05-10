"""Visualization utilities for causal inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple, Union


def plot_propensity_scores(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    title: str = "Propensity Score Distribution",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot propensity score distributions for treated and control groups.
    
    Args:
        propensity_scores: Estimated propensity scores
        treatment: Treatment indicator
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    treated_ps = propensity_scores[treatment == 1]
    control_ps = propensity_scores[treatment == 0]
    
    plt.hist(treated_ps, bins=30, alpha=0.7, label="Treated", color='blue', density=True)
    plt.hist(control_ps, bins=30, alpha=0.7, label="Control", color='red', density=True)
    
    plt.xlabel("Propensity Score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_balance_before_after(
    balance_stats: pd.DataFrame,
    title: str = "Covariate Balance",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot covariate balance before and after matching/weighting.
    
    Args:
        balance_stats: DataFrame with balance statistics
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    features = balance_stats['feature']
    std_diffs = balance_stats['std_diff']
    
    colors = ['red' if abs(diff) > 0.1 else 'green' for diff in std_diffs]
    
    plt.barh(features, std_diffs, color=colors, alpha=0.7)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='Balance threshold')
    plt.axvline(x=-0.1, color='red', linestyle='--', alpha=0.5)
    
    plt.xlabel("Standardized Difference")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_treatment_effects(
    results: List[dict],
    model_names: List[str],
    true_ate: Optional[float] = None,
    title: str = "Treatment Effect Estimates",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot treatment effect estimates with confidence intervals.
    
    Args:
        results: List of result dictionaries
        model_names: Names of models
        true_ate: True ATE for comparison
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    y_positions = np.arange(len(model_names))
    
    for i, (result, model_name) in enumerate(zip(results, model_names)):
        ate = result.get('ate', np.nan)
        ci = result.get('confidence_interval', None)
        
        # Plot point estimate
        plt.scatter(ate, y_positions[i], s=100, alpha=0.8, label=model_name)
        
        # Plot confidence interval
        if ci is not None:
            plt.plot([ci[0], ci[1]], [y_positions[i], y_positions[i]], 
                    color='black', alpha=0.7, linewidth=2)
            plt.plot([ci[0], ci[0]], [y_positions[i]-0.1, y_positions[i]+0.1], 
                    color='black', alpha=0.7, linewidth=2)
            plt.plot([ci[1], ci[1]], [y_positions[i]-0.1, y_positions[i]+0.1], 
                    color='black', alpha=0.7, linewidth=2)
    
    # Plot true ATE if available
    if true_ate is not None:
        plt.axvline(x=true_ate, color='red', linestyle='--', alpha=0.7, 
                   label=f'True ATE: {true_ate:.3f}')
    
    plt.yticks(y_positions, model_names)
    plt.xlabel("Average Treatment Effect")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_heterogeneous_effects(
    X: np.ndarray,
    cate_estimates: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "Heterogeneous Treatment Effects",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot heterogeneous treatment effects across different subgroups.
    
    Args:
        X: Covariates matrix
        cate_estimates: Conditional average treatment effects
        feature_names: Names of features
        title: Plot title
        save_path: Path to save the plot
    """
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(X.shape[1])]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(min(4, X.shape[1])):
        ax = axes[i]
        
        # Scatter plot of CATE vs feature
        scatter = ax.scatter(X[:, i], cate_estimates, alpha=0.6, c=cate_estimates, cmap='viridis')
        
        # Add trend line
        z = np.polyfit(X[:, i], cate_estimates, 1)
        p = np.poly1d(z)
        ax.plot(X[:, i], p(X[:, i]), "r--", alpha=0.8)
        
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel("Treatment Effect")
        ax.set_title(f"CATE vs {feature_names[i]}")
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label="Treatment Effect")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_interactive_plot(
    results: List[dict],
    model_names: List[str],
    true_ate: Optional[float] = None,
    title: str = "Interactive Treatment Effect Estimates",
) -> go.Figure:
    """
    Create interactive plot using Plotly.
    
    Args:
        results: List of result dictionaries
        model_names: Names of models
        true_ate: True ATE for comparison
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for i, (result, model_name) in enumerate(zip(results, model_names)):
        ate = result.get('ate', np.nan)
        ci = result.get('confidence_interval', None)
        
        # Add point estimate
        fig.add_trace(go.Scatter(
            x=[ate],
            y=[i],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name=model_name,
            text=f"ATE: {ate:.3f}",
            hovertemplate=f"<b>{model_name}</b><br>ATE: {ate:.3f}<extra></extra>"
        ))
        
        # Add confidence interval
        if ci is not None:
            fig.add_trace(go.Scatter(
                x=[ci[0], ci[1]],
                y=[i, i],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add true ATE line
    if true_ate is not None:
        fig.add_vline(
            x=true_ate,
            line_dash="dash",
            line_color="red",
            annotation_text=f"True ATE: {true_ate:.3f}"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Average Treatment Effect",
        yaxis_title="Model",
        yaxis=dict(tickmode='array', tickvals=list(range(len(model_names))), ticktext=model_names),
        height=400,
        showlegend=True
    )
    
    return fig


def plot_sensitivity_analysis(
    sensitivity_results: pd.DataFrame,
    title: str = "Sensitivity Analysis",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot sensitivity analysis results.
    
    Args:
        sensitivity_results: DataFrame with sensitivity analysis results
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(sensitivity_results['gamma'], sensitivity_results['ate'], 
             marker='o', linewidth=2, markersize=6)
    
    plt.xlabel("Sensitivity Parameter (γ)")
    plt.ylabel("Estimated ATE")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add confidence band
    if 'ci_lower' in sensitivity_results.columns and 'ci_upper' in sensitivity_results.columns:
        plt.fill_between(
            sensitivity_results['gamma'],
            sensitivity_results['ci_lower'],
            sensitivity_results['ci_upper'],
            alpha=0.3,
            label='95% Confidence Interval'
        )
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_evaluation_dashboard(
    results: List[dict],
    model_names: List[str],
    balance_stats: Optional[pd.DataFrame] = None,
    true_ate: Optional[float] = None,
) -> go.Figure:
    """
    Create comprehensive evaluation dashboard.
    
    Args:
        results: List of result dictionaries
        model_names: Names of models
        balance_stats: Covariate balance statistics
        true_ate: True ATE for comparison
        
    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Treatment Effects", "Model Comparison", 
                       "Balance Statistics", "Sensitivity Analysis"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Treatment effects plot
    for i, (result, model_name) in enumerate(zip(results, model_names)):
        ate = result.get('ate', np.nan)
        ci = result.get('confidence_interval', None)
        
        fig.add_trace(
            go.Scatter(x=[ate], y=[i], mode='markers', name=model_name),
            row=1, col=1
        )
        
        if ci is not None:
            fig.add_trace(
                go.Scatter(x=[ci[0], ci[1]], y=[i, i], mode='lines', 
                          line=dict(color='black'), showlegend=False),
                row=1, col=1
            )
    
    # Add true ATE
    if true_ate is not None:
        fig.add_vline(x=true_ate, line_dash="dash", line_color="red", row=1, col=1)
    
    # Balance statistics plot
    if balance_stats is not None:
        fig.add_trace(
            go.Bar(x=balance_stats['feature'], y=balance_stats['std_diff'],
                   name='Standardized Difference'),
            row=2, col=1
        )
    
    fig.update_layout(height=800, title_text="Causal Inference Evaluation Dashboard")
    
    return fig
