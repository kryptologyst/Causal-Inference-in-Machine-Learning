"""Streamlit demo application for causal inference."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from pathlib import Path

from src.data.synthetic import generate_synthetic_data, generate_heterogeneous_data
from src.models.propensity_score import PropensityScoreMatching
from src.models.ipw import InverseProbabilityWeighting
from src.models.double_ml import DoubleMachineLearning
from src.models.causal_forest import CausalForest
from src.evaluation.metrics import evaluate_causal_effect, calculate_balance_metrics
from src.visualization.plots import plot_propensity_scores, plot_balance_before_after


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Causal Inference Demo",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🔬 Causal Inference in Machine Learning")
    st.markdown("""
    **Author:** [kryptologyst](https://github.com/kryptologyst)  
    **GitHub:** https://github.com/kryptologyst
    
    This interactive demo showcases various causal inference methods for estimating treatment effects from observational data.
    """)
    
    # Safety disclaimer
    st.warning("""
    ⚠️ **Important Disclaimers:**
    - This is a research/educational tool only
    - Not for production decisions without expert review
    - Causal claims require domain expertise and careful validation
    - Results may not generalize to new contexts
    """)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Experiment Controls")
    
    # Data generation options
    st.sidebar.subheader("📊 Data Generation")
    data_type = st.sidebar.selectbox(
        "Data Type",
        ["Synthetic", "Heterogeneous Effects"],
        help="Choose the type of synthetic data to generate"
    )
    
    n_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Number of observations in the dataset"
    )
    
    n_features = st.sidebar.slider(
        "Number of Features",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        help="Number of covariates"
    )
    
    true_ate = st.sidebar.slider(
        "True Treatment Effect",
        min_value=0.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="True average treatment effect (for synthetic data)"
    )
    
    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=10000,
        value=42,
        help="Random seed for reproducibility"
    )
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    models_to_run = {
        "Propensity Score Matching": st.sidebar.checkbox("PSM", value=True),
        "Inverse Probability Weighting": st.sidebar.checkbox("IPW", value=True),
        "Double Machine Learning": st.sidebar.checkbox("Double ML", value=True),
        "Causal Forest": st.sidebar.checkbox("Causal Forest", value=True),
    }
    
    # Generate data
    if st.sidebar.button("🚀 Run Experiment", type="primary"):
        with st.spinner("Generating data and running experiments..."):
            # Generate data
            if data_type == "Synthetic":
                data = generate_synthetic_data(
                    n_samples=n_samples,
                    n_features=n_features,
                    treatment_effect=true_ate,
                    random_state=random_seed,
                )
            else:  # Heterogeneous Effects
                data = generate_heterogeneous_data(
                    n_samples=n_samples,
                    n_features=n_features,
                    random_state=random_seed,
                )
            
            # Store data in session state
            st.session_state.data = data
            st.session_state.results = {}
            st.session_state.models = {}
            
            # Run selected models
            for model_name, enabled in models_to_run.items():
                if enabled:
                    try:
                        if model_name == "Propensity Score Matching":
                            model = PropensityScoreMatching(random_state=random_seed)
                        elif model_name == "Inverse Probability Weighting":
                            model = InverseProbabilityWeighting(random_state=random_seed)
                        elif model_name == "Double Machine Learning":
                            model = DoubleMachineLearning(random_state=random_seed)
                        elif model_name == "Causal Forest":
                            model = CausalForest(random_state=random_seed)
                        
                        # Fit model
                        model.fit(data.X, data.treatment, data.outcome)
                        
                        # Evaluate
                        result = evaluate_causal_effect(model, data)
                        
                        # Store results
                        st.session_state.models[model_name] = model
                        st.session_state.results[model_name] = result
                        
                    except Exception as e:
                        st.error(f"Error running {model_name}: {str(e)}")
    
    # Display results
    if hasattr(st.session_state, 'data') and st.session_state.data is not None:
        data = st.session_state.data
        
        # Data summary
        st.header("📈 Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Samples", len(data.X))
        with col2:
            st.metric("Features", data.X.shape[1])
        with col3:
            st.metric("Treatment Rate", f"{np.mean(data.treatment):.3f}")
        with col4:
            st.metric("True ATE", f"{data.true_ate:.3f}")
        
        # Results comparison
        if st.session_state.results:
            st.header("🎯 Treatment Effect Estimates")
            
            # Create results table
            results_data = []
            for model_name, result in st.session_state.results.items():
                ci = result.confidence_interval
                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "N/A"
                
                results_data.append({
                    "Model": model_name,
                    "ATE": f"{result.ate:.4f}",
                    "ATT": f"{result.att:.4f}" if result.att else "N/A",
                    "95% CI": ci_str,
                    "P-value": f"{result.p_value:.4f}" if result.p_value else "N/A",
                    "N Treated": result.n_treated,
                    "N Control": result.n_control,
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Interactive plot
            st.subheader("📊 Interactive Results Plot")
            
            fig = go.Figure()
            
            for i, (model_name, result) in enumerate(st.session_state.results.items()):
                ate = result.ate
                ci = result.confidence_interval
                
                # Add point estimate
                fig.add_trace(go.Scatter(
                    x=[ate],
                    y=[i],
                    mode='markers',
                    marker=dict(size=15, color='blue'),
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
                        line=dict(color='black', width=3),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            # Add true ATE line
            fig.add_vline(
                x=data.true_ate,
                line_dash="dash",
                line_color="red",
                annotation_text=f"True ATE: {data.true_ate:.3f}"
            )
            
            fig.update_layout(
                title="Treatment Effect Estimates with Confidence Intervals",
                xaxis_title="Average Treatment Effect",
                yaxis_title="Model",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(st.session_state.results))),
                    ticktext=list(st.session_state.results.keys())
                ),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed analysis tabs
            st.header("🔍 Detailed Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Propensity Scores", "Balance", "Model Details", "Sensitivity"])
            
            with tab1:
                st.subheader("Propensity Score Distribution")
                
                # Calculate propensity scores for visualization
                from sklearn.linear_model import LogisticRegression
                ps_model = LogisticRegression()
                ps_model.fit(data.X, data.treatment)
                propensity_scores = ps_model.predict_proba(data.X)[:, 1]
                
                # Create histogram
                fig = go.Figure()
                
                treated_ps = propensity_scores[data.treatment == 1]
                control_ps = propensity_scores[data.treatment == 0]
                
                fig.add_trace(go.Histogram(
                    x=treated_ps,
                    name="Treated",
                    opacity=0.7,
                    nbinsx=30
                ))
                
                fig.add_trace(go.Histogram(
                    x=control_ps,
                    name="Control",
                    opacity=0.7,
                    nbinsx=30
                ))
                
                fig.update_layout(
                    title="Propensity Score Distribution",
                    xaxis_title="Propensity Score",
                    yaxis_title="Frequency",
                    barmode='overlay'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Covariate Balance")
                
                # Calculate balance metrics
                balance_stats = calculate_balance_metrics(
                    data.X, data.treatment, data.feature_names
                )
                
                # Display balance table
                st.dataframe(balance_stats, use_container_width=True)
                
                # Balance plot
                fig = go.Figure()
                
                colors = ['red' if abs(diff) > 0.1 else 'green' 
                         for diff in balance_stats['std_diff']]
                
                fig.add_trace(go.Bar(
                    x=balance_stats['std_diff'],
                    y=balance_stats['feature'],
                    orientation='h',
                    marker_color=colors,
                    name='Standardized Difference'
                ))
                
                fig.add_vline(x=0.1, line_dash="dash", line_color="red")
                fig.add_vline(x=-0.1, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    title="Covariate Balance (Standardized Differences)",
                    xaxis_title="Standardized Difference",
                    yaxis_title="Feature",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Model-Specific Details")
                
                for model_name, model in st.session_state.models.items():
                    with st.expander(f"{model_name} Details"):
                        if hasattr(model, "get_weight_statistics"):
                            weight_stats = model.get_weight_statistics()
                            st.write("**Weight Statistics:**")
                            for key, value in weight_stats.items():
                                st.write(f"- {key}: {value:.4f}")
                        
                        if hasattr(model, "get_feature_importance"):
                            feature_importance = model.get_feature_importance()
                            st.write("**Feature Importance:**")
                            for i, importance in enumerate(feature_importance):
                                st.write(f"- {data.feature_names[i]}: {importance:.4f}")
            
            with tab4:
                st.subheader("Sensitivity Analysis")
                st.info("Sensitivity analysis helps assess robustness to unobserved confounding.")
                
                # Simple sensitivity analysis
                gamma_values = np.linspace(1.0, 2.0, 10)
                sensitivity_results = []
                
                for gamma in gamma_values:
                    # Simulate sensitivity parameter effect
                    adjusted_treatment = data.treatment.copy()
                    unobserved_confounder = np.random.randn(len(data.treatment))
                    confounding_effect = gamma * unobserved_confounder
                    
                    # Adjust treatment probability
                    propensity_adjustment = 1 / (1 + np.exp(-confounding_effect))
                    treatment_adjustment = np.random.binomial(1, propensity_adjustment)
                    adjusted_treatment = np.logical_xor(data.treatment, treatment_adjustment).astype(int)
                    
                    # Re-estimate with adjusted treatment
                    try:
                        model = PropensityScoreMatching(random_state=random_seed)
                        model.fit(data.X, adjusted_treatment, data.outcome)
                        adjusted_ate = model.estimate_ate(data.outcome)
                        sensitivity_results.append({"gamma": gamma, "ate": adjusted_ate})
                    except:
                        sensitivity_results.append({"gamma": gamma, "ate": np.nan})
                
                sensitivity_df = pd.DataFrame(sensitivity_results)
                
                # Plot sensitivity analysis
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sensitivity_df['gamma'],
                    y=sensitivity_df['ate'],
                    mode='lines+markers',
                    name='Estimated ATE',
                    line=dict(width=3)
                ))
                
                fig.add_hline(y=data.true_ate, line_dash="dash", line_color="red",
                             annotation_text=f"True ATE: {data.true_ate:.3f}")
                
                fig.update_layout(
                    title="Sensitivity Analysis",
                    xaxis_title="Sensitivity Parameter (γ)",
                    yaxis_title="Estimated ATE",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👈 Use the sidebar controls to configure and run an experiment.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This tool is for research and educational purposes only. 
    Causal inference requires careful consideration of assumptions and domain expertise.
    """)

if __name__ == "__main__":
    main()
