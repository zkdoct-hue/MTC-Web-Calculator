# ==============================================================================
# Final MTC Fine-Gray Web Calculator
# ==============================================================================
# Clinical translation of the locked parsimonious Fine-Gray competing-risk model:
#   Age + T stage + N stage + M stage + Surgery + Radiotherapy
#
# Prediction target:
#   cumulative incidence of MTC-specific death at 3, 5, and 10 years,
#   accounting for death from other causes as a competing event.
#
# IMPORTANT:
#   This app does NOT use the old Elastic-net/Cox approximation.
#   Exact predictions are read from a 192-profile lookup table exported directly
#   from the locked QHScrnomo Fine-Gray model in R.
# ==============================================================================

from pathlib import Path

import pandas as pd
import streamlit as st


# ------------------------------------------------------------------------------
# 1. Page configuration
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="MTC Competing-Risk Calculator",
    page_icon="⚕️",
    layout="wide",
)


# ------------------------------------------------------------------------------
# 2. Load exact Fine-Gray lookup
# ------------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
LOOKUP_FILE = APP_DIR / "MTC_FineGray_WebCalculator_Lookup.csv"
META_FILE = APP_DIR / "MTC_FineGray_WebCalculator_Metadata.csv"


@st.cache_data
def load_lookup():
    if not LOOKUP_FILE.exists():
        raise FileNotFoundError(
            "MTC_FineGray_WebCalculator_Lookup.csv was not found. "
            "Run the supplied R export script first, then upload the generated "
            "CSV file to the same GitHub repository as web_calculator.py."
        )

    df = pd.read_csv(LOOKUP_FILE, dtype={
        "Profile_key": str,
        "Age": str,
        "T_stage": str,
        "N_stage": str,
        "M_stage": str,
        "Surgery": str,
        "Radiation": str,
    })

    required = {
        "Profile_key",
        "Age", "T_stage", "N_stage", "M_stage", "Surgery", "Radiation",
        "Risk_36m", "Risk_60m", "Risk_120m",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "The lookup CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if len(df) != 192:
        raise ValueError(
            f"Expected 192 predictor profiles in the lookup table; found {len(df)}."
        )

    if df["Profile_key"].duplicated().any():
        raise ValueError("Duplicate Profile_key values were found in the lookup table.")

    for c in ["Risk_36m", "Risk_60m", "Risk_120m"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
        if ((df[c] < 0) | (df[c] > 1)).any():
            raise ValueError(f"{c} contains values outside [0, 1].")

    return df


try:
    lookup = load_lookup()
except Exception as exc:
    st.error("The prediction table could not be loaded.")
    st.code(str(exc))
    st.stop()


# ------------------------------------------------------------------------------
# 3. Display helpers
# ------------------------------------------------------------------------------

AGE_OPTIONS = ["<55", "55-69", ">=70"]
T_OPTIONS = ["T1", "T2", "T3", "T4"]
N_OPTIONS = ["N0", "N1"]
M_OPTIONS = ["M0", "M1"]
YES_NO = ["No", "Yes"]

AGE_LABELS = {
    "<55": "<55 years",
    "55-69": "55–69 years",
    ">=70": "≥70 years",
}

YES_NO_LABELS = {
    "No": "No",
    "Yes": "Yes",
}


def build_profile_key(age, t_stage, n_stage, m_stage, surgery, radiation):
    return "|".join([
        age,
        t_stage,
        n_stage,
        m_stage,
        surgery,
        radiation,
    ])


def pct(x):
    value = 100.0 * float(x)
    # Avoid displaying model estimates extremely close to 1.0 as absolute certainty.
    if value >= 99.95:
        return ">99.9%"
    return f"{value:.1f}%"


# ------------------------------------------------------------------------------
# 4. Sidebar: final six-variable clinical model
# ------------------------------------------------------------------------------

st.sidebar.markdown("### Patient characteristics")

age = st.sidebar.selectbox(
    "Age",
    AGE_OPTIONS,
    format_func=lambda x: AGE_LABELS[x],
)

t_stage = st.sidebar.selectbox(
    "T stage",
    T_OPTIONS,
)

n_stage = st.sidebar.selectbox(
    "N stage",
    N_OPTIONS,
)

m_stage = st.sidebar.selectbox(
    "M stage",
    M_OPTIONS,
)

surgery = st.sidebar.selectbox(
    "Surgery",
    YES_NO,
    format_func=lambda x: YES_NO_LABELS[x],
)

radiation = st.sidebar.selectbox(
    "Radiotherapy",
    YES_NO,
    format_func=lambda x: YES_NO_LABELS[x],
)

calculate = st.sidebar.button(
    "Calculate risk",
    type="primary",
    use_container_width=True,
)


# ------------------------------------------------------------------------------
# 5. Main page
# ------------------------------------------------------------------------------

st.title(
    "Web-Based Competing-Risk Prediction Tool for "
    "Medullary Thyroid Carcinoma–Specific Mortality"
)

st.caption(
    "Predictions are derived from the final parsimonious Fine–Gray model and "
    "represent the cumulative incidence of MTC-specific death while accounting "
    "for death from other causes as a competing event."
)


# ------------------------------------------------------------------------------
# 6. Prediction
# ------------------------------------------------------------------------------

if calculate:
    profile_key = build_profile_key(
        age,
        t_stage,
        n_stage,
        m_stage,
        surgery,
        radiation,
    )

    row = lookup.loc[lookup["Profile_key"] == profile_key]

    if len(row) != 1:
        st.error(
            "The selected patient profile could not be matched uniquely in the "
            "prediction lookup table."
        )
        st.stop()

    r = row.iloc[0]

    st.subheader("Predicted MTC-specific mortality")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "3-year risk",
            pct(r["Risk_36m"]),
        )

    with col2:
        st.metric(
            "5-year risk",
            pct(r["Risk_60m"]),
        )

    with col3:
        st.metric(
            "10-year risk",
            pct(r["Risk_120m"]),
        )

    st.markdown("#### Selected profile")

    profile_df = pd.DataFrame({
        "Characteristic": [
            "Age",
            "T stage",
            "N stage",
            "M stage",
            "Surgery",
            "Radiotherapy",
        ],
        "Value": [
            AGE_LABELS[age],
            t_stage,
            n_stage,
            m_stage,
            surgery,
            radiation,
        ],
    })

    st.dataframe(
        profile_df,
        hide_index=True,
        use_container_width=True,
    )

    st.info(
        "The reported values are cumulative incidences of MTC-specific death, "
        "not overall survival probabilities."
    )

else:
    st.markdown(
        "Select the six patient characteristics in the sidebar and click "
        "**Calculate risk**."
    )


# ------------------------------------------------------------------------------
# 7. Model information
# ------------------------------------------------------------------------------

with st.expander("Model information"):
    st.markdown(
        """
        **Final model:** Fine–Gray competing-risk regression  
        **Predictors:** Age, T stage, N stage, M stage, surgery, and radiotherapy  
        **Prediction horizons:** 3, 5, and 10 years  
        **Development cohort:** 1,521 patients  
        **Held-out internal validation cohort:** 651 patients  
        **Competing-risk C-index:** 0.863  
        **Time-dependent AUC:** 0.919 at 3 years, 0.929 at 5 years, and 0.858 at 10 years

        The web tool uses exact predictions exported from the locked final
        Fine–Gray model for all 192 possible combinations of the six categorical
        predictors.
        """
    )

    if META_FILE.exists():
        try:
            metadata = pd.read_csv(META_FILE)
            with st.popover("Technical model metadata"):
                st.dataframe(
                    metadata,
                    hide_index=True,
                    use_container_width=True,
                )
        except Exception:
            pass


# ------------------------------------------------------------------------------
# 8. Disclaimer
# ------------------------------------------------------------------------------

st.divider()

st.caption(
    "For research and educational purposes only. This calculator is based on "
    "retrospective SEER data and has undergone held-out internal validation but "
    "not independent external validation. It is not intended to replace "
    "individualized clinical judgment."
)
