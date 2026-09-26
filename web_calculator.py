# ==============================================================================
# Final MTC Fine-Gray Web Calculator — enhanced interpretation version
# ==============================================================================
# Clinical translation of the locked parsimonious Fine-Gray competing-risk model:
#   Age + T stage + N stage + M stage + Surgery + Radiotherapy
#
# Prediction target:
#   cumulative incidence of MTC-specific death at 3, 5, and 10 years,
#   accounting for death from other causes as a competing event.
#
# IMPORTANT:
#   1) Exact predictions are read from the 192-profile lookup table exported
#      directly from the locked final Fine-Gray model in R.
#   2) The model-interpretation plot below uses the same Fine-Gray model
#      coefficients/SHRs. It does NOT display Elastic-net SHAP values.
# ==============================================================================

from pathlib import Path

import pandas as pd
import streamlit as st


# ------------------------------------------------------------------------------
# 1. Page configuration
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="MTC-Specific Mortality Risk Calculator",
    page_icon="⚕️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        background: rgba(248, 249, 251, 0.75);
        border: 1px solid rgba(49, 51, 63, 0.12);
        padding: 0.9rem 1rem;
        border-radius: 0.7rem;
    }
    div[data-testid="stMetricLabel"] {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 2. Load exact Fine-Gray lookup + coefficient audit
# ------------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
LOOKUP_FILE = APP_DIR / "MTC_FineGray_WebCalculator_Lookup.csv"
META_FILE = APP_DIR / "MTC_FineGray_WebCalculator_Metadata.csv"
AUDIT_FILE = APP_DIR / "MTC_FineGray_WebCalculator_Coefficient_Audit.csv"


@st.cache_data
def load_lookup():
    if not LOOKUP_FILE.exists():
        raise FileNotFoundError(
            "MTC_FineGray_WebCalculator_Lookup.csv was not found. "
            "Upload it to the same repository as web_calculator.py."
        )

    df = pd.read_csv(
        LOOKUP_FILE,
        dtype={
            "Profile_key": str,
            "Age": str,
            "T_stage": str,
            "N_stage": str,
            "M_stage": str,
            "Surgery": str,
            "Radiation": str,
        },
    )

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

    if (df["Risk_36m"] > df["Risk_60m"]).any() or (df["Risk_60m"] > df["Risk_120m"]).any():
        raise ValueError("Non-monotonic cumulative-incidence predictions were found.")

    return df


@st.cache_data
def load_audit():
    if not AUDIT_FILE.exists():
        raise FileNotFoundError(
            "MTC_FineGray_WebCalculator_Coefficient_Audit.csv was not found. "
            "Upload it to the same repository as web_calculator.py."
        )

    df = pd.read_csv(AUDIT_FILE)
    required = {"Term", "Coefficient", "SHR"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "The coefficient audit CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["Coefficient"] = pd.to_numeric(df["Coefficient"], errors="raise")
    df["SHR"] = pd.to_numeric(df["SHR"], errors="raise")
    return df


try:
    lookup = load_lookup()
    audit = load_audit()
except Exception as exc:
    st.error("The calculator files could not be loaded.")
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

TERM_LABELS = {
    "Age=55-69": "Age 55–69 vs <55 years",
    "Age=>=70": "Age ≥70 vs <55 years",
    "T_stage=T2": "T2 vs T1",
    "T_stage=T3": "T3 vs T1",
    "T_stage=T4": "T4 vs T1",
    "N_stage=N1": "N1 vs N0",
    "M_stage=M1": "M1 vs M0",
    "Surgery=Yes": "Surgery: Yes vs No",
    "Radiation=Yes": "Radiotherapy: Yes vs No",
}

TERM_ORDER = [
    "Age=55-69", "Age=>=70",
    "T_stage=T2", "T_stage=T3", "T_stage=T4",
    "N_stage=N1", "M_stage=M1",
    "Surgery=Yes", "Radiation=Yes",
]


REFERENCE_TEXT = {
    "Age": "<55 years",
    "T stage": "T1",
    "N stage": "N0",
    "M stage": "M0",
    "Surgery": "No",
    "Radiotherapy": "No",
}


def build_profile_key(age, t_stage, n_stage, m_stage, surgery, radiation):
    return "|".join([age, t_stage, n_stage, m_stage, surgery, radiation])


def pct(x):
    value = 100.0 * float(x)
    if value >= 99.95:
        return ">99.9%"
    return f"{value:.1f}%"


def selected_model_terms(age, t_stage, n_stage, m_stage, surgery, radiation):
    selected = set()
    if age == "55-69":
        selected.add("Age=55-69")
    elif age == ">=70":
        selected.add("Age=>=70")

    if t_stage != "T1":
        selected.add(f"T_stage={t_stage}")
    if n_stage == "N1":
        selected.add("N_stage=N1")
    if m_stage == "M1":
        selected.add("M_stage=M1")
    if surgery == "Yes":
        selected.add("Surgery=Yes")
    if radiation == "Yes":
        selected.add("Radiation=Yes")
    return selected


def make_interpretation_data(selected_terms):
    df = audit.copy()
    df["Display"] = df["Term"].map(TERM_LABELS).fillna(df["Term"])
    df["Order"] = df["Term"].map({term: i for i, term in enumerate(TERM_ORDER)})
    df["Selected"] = df["Term"].isin(selected_terms)
    df["SHR_label"] = df["SHR"].map(lambda x: f"{x:.2f}")
    return df.sort_values("Order")


# ------------------------------------------------------------------------------
# 4. Sidebar
# ------------------------------------------------------------------------------

st.sidebar.markdown("### Patient characteristics")

age = st.sidebar.selectbox("Age", AGE_OPTIONS, format_func=lambda x: AGE_LABELS[x])
t_stage = st.sidebar.selectbox("T stage", T_OPTIONS)
n_stage = st.sidebar.selectbox("N stage", N_OPTIONS)
m_stage = st.sidebar.selectbox("M stage", M_OPTIONS)
surgery = st.sidebar.selectbox("Surgery", YES_NO)
radiation = st.sidebar.selectbox("Radiotherapy", YES_NO)

calculate = st.sidebar.button(
    "Calculate risk",
    type="primary",
    use_container_width=True,
)

st.sidebar.caption(
    "Predictions are based on the final six-variable Fine–Gray model."
)


# ------------------------------------------------------------------------------
# 5. Header
# ------------------------------------------------------------------------------

st.title("MTC-Specific Mortality Risk Calculator")

st.markdown(
    "**Fine–Gray competing-risk prediction model for individualized 3-, 5-, "
    "and 10-year cumulative incidence of medullary thyroid carcinoma "
    "(MTC)-specific death.**"
)

st.caption(
    "Death from causes other than MTC is treated as a competing event. "
    "Predictions are generated from the locked final six-variable Fine–Gray model."
)


# ------------------------------------------------------------------------------
# 6. Prediction + patient-specific display
# ------------------------------------------------------------------------------

if calculate:
    profile_key = build_profile_key(
        age, t_stage, n_stage, m_stage, surgery, radiation
    )
    row = lookup.loc[lookup["Profile_key"] == profile_key]

    if len(row) != 1:
        st.error(
            "The selected patient profile could not be matched uniquely in the "
            "prediction lookup table."
        )
        st.stop()

    r = row.iloc[0]
    risk36 = float(r["Risk_36m"])
    risk60 = float(r["Risk_60m"])
    risk120 = float(r["Risk_120m"])

    st.subheader("Predicted MTC-specific mortality")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("3-year risk", pct(risk36))
    with col2:
        st.metric("5-year risk", pct(risk60))
    with col3:
        st.metric("10-year risk", pct(risk120))

    st.caption(
        "Values are cumulative incidences of MTC-specific death, not overall survival probabilities."
    )

    st.markdown("#### Risk across prediction horizons")
    risk_df = pd.DataFrame(
        {
            "Horizon": ["3 years", "5 years", "10 years"],
            "Risk": [100 * risk36, 100 * risk60, 100 * risk120],
            "Order": [1, 2, 3],
        }
    )
    risk_df["RiskLabel"] = risk_df["Risk"].map(lambda x: f"{x:.1f}%")

    risk_spec = {
        "height": 260,
        "layer": [
            {
                "mark": {"type": "bar", "cornerRadiusTopLeft": 5, "cornerRadiusTopRight": 5},
                "encoding": {
                    "x": {
                        "field": "Horizon",
                        "type": "ordinal",
                        "sort": ["3 years", "5 years", "10 years"],
                        "axis": {"title": None, "labelAngle": 0},
                    },
                    "y": {
                        "field": "Risk",
                        "type": "quantitative",
                        "axis": {"title": "Cumulative incidence (%)"},
                        "scale": {"zero": True},
                    },
                    "color": {"value": "#4C78A8"},
                    "tooltip": [
                        {"field": "Horizon", "type": "nominal", "title": "Horizon"},
                        {"field": "Risk", "type": "quantitative", "title": "Risk (%)", "format": ".1f"},
                    ],
                },
            },
            {
                "mark": {"type": "text", "dy": -10, "fontSize": 14, "fontWeight": "bold"},
                "encoding": {
                    "x": {
                        "field": "Horizon",
                        "type": "ordinal",
                        "sort": ["3 years", "5 years", "10 years"],
                    },
                    "y": {"field": "Risk", "type": "quantitative"},
                    "text": {"field": "RiskLabel", "type": "nominal"},
                    "color": {"value": "#2B2D42"},
                },
            },
        ],
    }
    st.vega_lite_chart(risk_df, risk_spec, use_container_width=True)
    st.caption(
        "Only the 3-, 5-, and 10-year horizons shown above are model prediction outputs."
    )

    st.markdown("#### Selected profile")
    profile_df = pd.DataFrame(
        {
            "Characteristic": [
                "Age", "T stage", "N stage", "M stage", "Surgery", "Radiotherapy"
            ],
            "Value": [
                AGE_LABELS[age], t_stage, n_stage, m_stage, surgery, radiation
            ],
        }
    )
    st.dataframe(profile_df, hide_index=True, use_container_width=True)

    st.markdown("### Model interpretation")
    st.caption(
        "The figure below summarizes predictor associations from the same final Fine–Gray model used by the calculator. "
        "Points are subdistribution hazard ratio (SHR) estimates relative to the reference category; selected non-reference "
        "levels for this patient are highlighted. These are prognostic associations, not causal effects."
    )

    selected_terms = selected_model_terms(
        age, t_stage, n_stage, m_stage, surgery, radiation
    )
    interpret_df = make_interpretation_data(selected_terms)

    interp_left, interp_right = st.columns([1.55, 0.85], gap="large")

    with interp_left:
        shr_spec = {
            "height": 360,
            "layer": [
                {
                    "mark": {"type": "rule", "strokeDash": [5, 4], "color": "#A0A0A0"},
                    "encoding": {"x": {"datum": 1}},
                },
                {
                    "mark": {"type": "point", "filled": True, "size": 110},
                    "encoding": {
                        "x": {
                            "field": "SHR",
                            "type": "quantitative",
                            "scale": {"type": "log", "domain": [0.3, 7.5]},
                            "axis": {
                                "title": "Subdistribution hazard ratio (log scale)",
                                "values": [0.5, 1, 2, 4, 6],
                            },
                        },
                        "y": {
                            "field": "Display",
                            "type": "nominal",
                            "sort": {"field": "Order", "order": "ascending"},
                            "axis": {"title": None, "labelLimit": 230},
                        },
                        "color": {
                            "condition": {"test": "datum.Selected === true", "value": "#D9534F"},
                            "value": "#59636E",
                        },
                        "tooltip": [
                            {"field": "Display", "type": "nominal", "title": "Predictor level"},
                            {"field": "SHR", "type": "quantitative", "title": "SHR", "format": ".2f"},
                        ],
                    },
                },
                {
                    "mark": {"type": "text", "align": "left", "dx": 8, "fontSize": 12},
                    "encoding": {
                        "x": {"field": "SHR", "type": "quantitative", "scale": {"type": "log", "domain": [0.3, 7.5]}},
                        "y": {
                            "field": "Display",
                            "type": "nominal",
                            "sort": {"field": "Order", "order": "ascending"},
                        },
                        "text": {"field": "SHR_label", "type": "nominal"},
                        "color": {"value": "#333333"},
                    },
                },
            ],
        }
        st.vega_lite_chart(interpret_df, shr_spec, use_container_width=True)
        st.caption(
            "SHR = 1 denotes the reference level. Point estimates are shown; confidence intervals are not displayed in this web export."
        )

    with interp_right:
        st.markdown("**Reference categories**")
        ref_df = pd.DataFrame(
            {"Predictor": list(REFERENCE_TEXT.keys()), "Reference": list(REFERENCE_TEXT.values())}
        )
        st.dataframe(ref_df, hide_index=True, use_container_width=True)

        st.markdown("**Selected non-reference levels**")
        active = interpret_df.loc[interpret_df["Selected"], ["Display", "SHR"]].copy()
        if len(active):
            active["SHR"] = active["SHR"].map(lambda x: f"{x:.2f}")
            active.columns = ["Selected level", "SHR"]
            st.dataframe(active, hide_index=True, use_container_width=True)
        else:
            st.caption("All selected levels are reference categories in the model.")

    st.warning(
        "Surgery and radiotherapy are prognostic predictors in this retrospective model. "
        "Changing these selections does not estimate the causal benefit or harm of treatment "
        "and should not be used to make treatment decisions."
    )

else:
    st.info(
        "Select the six patient characteristics in the sidebar and click **Calculate risk**."
    )


# ------------------------------------------------------------------------------
# 7. Model information
# ------------------------------------------------------------------------------

with st.expander("Model information and validation performance"):
    perf1, perf2, perf3, perf4 = st.columns(4)
    with perf1:
        st.metric("C-index", "0.863")
    with perf2:
        st.metric("3-year AUC", "0.919")
    with perf3:
        st.metric("5-year AUC", "0.929")
    with perf4:
        st.metric("10-year AUC", "0.858")

    st.markdown(
        """
        **Final model:** Fine–Gray competing-risk regression  
        **Predictors:** Age, T stage, N stage, M stage, surgery, and radiotherapy  
        **Prediction horizons:** 3, 5, and 10 years  
        **Development cohort:** 1,521 patients  
        **Held-out internal validation cohort:** 651 patients

        The calculator uses exact predictions exported from the locked final Fine–Gray
        model for all 192 possible combinations of the six categorical predictors.
        """
    )

    if META_FILE.exists():
        try:
            metadata = pd.read_csv(META_FILE)
            with st.popover("Technical model metadata"):
                st.dataframe(metadata, hide_index=True, use_container_width=True)
        except Exception:
            pass


# ------------------------------------------------------------------------------
# 8. Disclaimer
# ------------------------------------------------------------------------------

st.divider()

st.caption(
    "For research and educational purposes only. This calculator was developed "
    "from retrospective SEER data and has undergone held-out internal temporal "
    "validation, but not independent external validation. The predictions are "
    "prognostic associations and are not intended to estimate treatment effects, "
    "replace individualized clinical judgment, or direct patient management."
)
