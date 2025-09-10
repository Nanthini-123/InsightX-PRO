# modules/interpretability.py
import streamlit as st

try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def run_shap_explain(model, X_train):
    """
    Run SHAP interpretability on a trained model.
    Works with tree-based models (RandomForest, XGBoost, LightGBM, etc.)
    """
    if not SHAP_AVAILABLE:
        st.warning("⚠️ SHAP is not installed. Run `pip install shap` to enable interpretability.")
        return

    try:
        # Pick a SHAP explainer (TreeExplainer works well for RandomForest/XGBoost)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        st.subheader("🔍 SHAP Interpretability")
        st.markdown("Feature contribution to predictions:")

        # Summary plot
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_train, show=False)
        st.pyplot(fig)

        # Bar plot of feature importance
        fig2, ax2 = plt.subplots()
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"SHAP explainability failed: {e}")
