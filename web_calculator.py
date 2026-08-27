# ==============================================================================
# Polished MTC Fine-Gray Web Calculator — v4
# ==============================================================================
# Locked final model:
#   Age + T stage + N stage + M stage + Surgery + Radiotherapy
#
# Prediction target:
#   3-, 5-, and 10-year cumulative incidence of MTC-specific death,
#   accounting for other-cause death as a competing event.
#
# Scientific principle:
#   The app reads exact predictions exported from the locked QHScrnomo
#   Fine-Gray model. No Cox approximation, no SHAP from another model,
#   and no unvalidated low/medium/high risk thresholds are used.
# ==============================================================================

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


# ------------------------------------------------------------------------------
# 1. Page configuration
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="MTC-Specific Mortality Risk Calculator",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------------------
# 2. Visual styling
# ------------------------------------------------------------------------------

st.markdown(
    """
    <style>
        :root {
            --ink: #172033;
            --muted: #667085;
            --line: #E6EAF0;
            --panel: #F7F9FC;
            --accent: #315C7C;
            --accent-soft: #EAF1F6;
            --accent-dark: #24465F;
            --success-soft: #EEF6F3;
        }

        .block-container {
            max-width: 1220px;
            padding-top: 2.0rem;
            padding-bottom: 3.5rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.4rem;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: -0.02em;
        }

        .hero {
            padding: 0.2rem 0 1.2rem 0;
        }

        .eyebrow {
            display: inline-block;
            padding: 0.32rem 0.68rem;
            margin-bottom: 0.78rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent-dark);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.035em;
        }

        .hero-title {
            margin: 0;
            max-width: 940px;
            font-size: clamp(2rem, 4vw, 3.15rem);
            line-height: 1.08;
            font-weight: 760;
            color: var(--ink);
            letter-spacing: -0.035em;
        }

        .hero-subtitle {
            max-width: 980px;
            margin-top: 0.75rem;
            color: var(--muted);
            font-size: 1.01rem;
            line-height: 1.62;
        }

        .section-kicker {
            margin-top: 0.2rem;
            margin-bottom: 0.18rem;
            color: var(--accent);
            font-size: 0.77rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .section-title {
            margin-top: 0;
            margin-bottom: 0.85rem;
            font-size: 1.45rem;
            line-height: 1.25;
            font-weight: 720;
            color: var(--ink);
        }

        .risk-card {
            min-height: 142px;
            padding: 1.20rem 1.25rem 1.08rem 1.25rem;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: #FFFFFF;
            box-shadow: 0 5px 18px rgba(23, 32, 51, 0.045);
        }

        .risk-horizon {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.055em;
            text-transform: uppercase;
        }

        .risk-value {
            margin-top: 0.36rem;
            color: var(--ink);
            font-size: 2.18rem;
            font-weight: 760;
            letter-spacing: -0.035em;
            line-height: 1.10;
        }

        .risk-label {
            margin-top: 0.42rem;
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.35;
        }

        .profile-card {
            min-height: 86px;
            padding: 0.88rem 1.0rem;
            border: 1px solid var(--line);
            border-radius: 13px;
            background: var(--panel);
        }

        .profile-label {
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 650;
        }

        .profile-value {
            margin-top: 0.26rem;
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 700;
        }

        .validation-card {
            min-height: 92px;
            padding: 0.85rem 0.95rem;
            border: 1px solid var(--line);
            border-radius: 13px;
            background: #FFFFFF;
        }

        .validation-label {
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 650;
        }

        .validation-value {
            margin-top: 0.22rem;
            color: var(--ink);
            font-size: 1.38rem;
            font-weight: 750;
            line-height: 1.15;
        }

        .soft-note {
            padding: 0.92rem 1.05rem;
            border-left: 3px solid var(--accent);
            border-radius: 10px;
            background: var(--accent-soft);
            color: #334155;
            font-size: 0.88rem;
            line-height: 1.52;
        }

        .empty-state {
            margin-top: 1.25rem;
            padding: 2.0rem 1.4rem;
            border: 1px dashed #C8D1DC;
            border-radius: 16px;
            background: #FAFBFD;
            text-align: center;
            color: var(--muted);
        }

        .empty-state strong {
            display: block;
            margin-bottom: 0.32rem;
            color: var(--ink);
            font-size: 1.03rem;
        }

        .footer-note {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.5;
        }

        div[data-testid="stFormSubmitButton"] button {
            width: 100%;
            border-radius: 10px;
            border: 1px solid var(--accent-dark);
            background: var(--accent);
            color: white;
            font-weight: 700;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            border-color: var(--accent-dark);
            background: var(--accent-dark);
            color: white;
        }

        div[data-testid="stSelectbox"] > div > div {
            border-radius: 10px;
        }

        hr {
            border-color: var(--line);
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 1.2rem;
            }
            .hero-title {
                font-size: 2.1rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 3. Exact Fine-Gray prediction table
# ------------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
LOOKUP_FILE = APP_DIR / "MTC_FineGray_WebCalculator_Lookup.csv"
META_FILE = APP_DIR / "MTC_FineGray_WebCalculator_Metadata.csv"


@st.cache_data
def load_lookup():
    if not LOOKUP_FILE.exists():
        raise FileNotFoundError(
            "MTC_FineGray_WebCalculator_Lookup.csv was not found."
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
        "Age",
        "T_stage",
        "N_stage",
        "M_stage",
        "Surgery",
        "Radiation",
        "Risk_36m",
        "Risk_60m",
        "Risk_120m",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "The lookup CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if len(df) != 192:
        raise ValueError(
            f"Expected 192 predictor profiles; found {len(df)}."
        )

    if df["Profile_key"].duplicated().any():
        raise ValueError("Duplicate Profile_key values were found.")

    for col in ["Risk_36m", "Risk_60m", "Risk_120m"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if ((df[col] < 0) | (df[col] > 1)).any():
            raise ValueError(f"{col} contains values outside [0, 1].")

    return df


try:
    lookup = load_lookup()
except Exception as exc:
    st.error("The prediction table could not be loaded.")
    st.code(str(exc))
    st.stop()


# ------------------------------------------------------------------------------
# 4. Helpers
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


def build_profile_key(age, t_stage, n_stage, m_stage, surgery, radiation):
    return "|".join(
        [age, t_stage, n_stage, m_stage, surgery, radiation]
    )


def pct(value):
    value = 100.0 * float(value)
    if value >= 99.95:
        return ">99.9%"
    return f"{value:.1f}%"


def profile_card(label, value):
    return f"""
    <div class="profile-card">
        <div class="profile-label">{label}</div>
        <div class="profile-value">{value}</div>
    </div>
    """


def validation_card(label, value):
    return f"""
    <div class="validation-card">
        <div class="validation-label">{label}</div>
        <div class="validation-value">{value}</div>
    </div>
    """


# ------------------------------------------------------------------------------
# 5. Sidebar inputs
# ------------------------------------------------------------------------------

st.sidebar.markdown("## Patient characteristics")
st.sidebar.caption(
    "Enter the six variables used in the final parsimonious Fine–Gray model."
)

with st.sidebar.form("patient_input_form"):
    age = st.selectbox(
        "Age",
        AGE_OPTIONS,
        format_func=lambda x: AGE_LABELS[x],
    )
    t_stage = st.selectbox("T stage", T_OPTIONS)
    n_stage = st.selectbox("N stage", N_OPTIONS)
    m_stage = st.selectbox("M stage", M_OPTIONS)
    surgery = st.selectbox("Surgery", YES_NO)
    radiation = st.selectbox("Radiotherapy", YES_NO)

    submitted = st.form_submit_button(
        "Calculate risk",
        type="primary",
        use_container_width=True,
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Prediction horizons: 3, 5, and 10 years. "
    "Outcome: cumulative incidence of MTC-specific death."
)

if submitted:
    st.session_state["mtc_last_profile"] = {
        "age": age,
        "t_stage": t_stage,
        "n_stage": n_stage,
        "m_stage": m_stage,
        "surgery": surgery,
        "radiation": radiation,
    }


# ------------------------------------------------------------------------------
# 6. Hero
# ------------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">FINE–GRAY COMPETING-RISK MODEL</div>
        <h1 class="hero-title">MTC-Specific Mortality Risk Calculator</h1>
        <div class="hero-subtitle">
            A web-based prediction tool for medullary thyroid carcinoma.
            Estimates are cumulative incidences of MTC-specific death and
            explicitly account for death from other causes as a competing event.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 7. Prediction result
# ------------------------------------------------------------------------------

profile = st.session_state.get("mtc_last_profile")

if profile is None:
    st.markdown(
        """
        <div class="empty-state">
            <strong>Ready to calculate an individualized prediction</strong>
            Select the patient characteristics in the sidebar and choose
            <b>Calculate risk</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    profile_key = build_profile_key(
        profile["age"],
        profile["t_stage"],
        profile["n_stage"],
        profile["m_stage"],
        profile["surgery"],
        profile["radiation"],
    )

    row = lookup.loc[lookup["Profile_key"] == profile_key]

    if len(row) != 1:
        st.error(
            "The selected patient profile could not be matched uniquely."
        )
        st.stop()

    r = row.iloc[0]

    st.markdown(
        '<div class="section-kicker">Individual prediction</div>'
        '<div class="section-title">Predicted MTC-specific mortality</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="medium")

    risk_specs = [
        ("3-year risk", r["Risk_36m"]),
        ("5-year risk", r["Risk_60m"]),
        ("10-year risk", r["Risk_120m"]),
    ]

    for col, (label, value) in zip([c1, c2, c3], risk_specs):
        with col:
            st.markdown(
                f"""
                <div class="risk-card">
                    <div class="risk-horizon">{label}</div>
                    <div class="risk-value">{pct(value)}</div>
                    <div class="risk-label">
                        Cumulative incidence of MTC-specific death
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    left, right = st.columns([1.32, 1.0], gap="large")

    # --------------------------------------------------------------------------
    # Exact 3/5/10-year risk visualization
    # --------------------------------------------------------------------------
    with left:
        st.markdown(
            '<div class="section-kicker">Prediction horizons</div>'
            '<div class="section-title">Risk at 3, 5, and 10 years</div>',
            unsafe_allow_html=True,
        )

        chart_df = pd.DataFrame(
            {
                "Horizon": ["3 years", "5 years", "10 years"],
                "Order": [3, 5, 10],
                "Risk": [
                    float(r["Risk_36m"]),
                    float(r["Risk_60m"]),
                    float(r["Risk_120m"]),
                ],
                "Label": [
                    pct(r["Risk_36m"]),
                    pct(r["Risk_60m"]),
                    pct(r["Risk_120m"]),
                ],
            }
        )

        max_risk = chart_df["Risk"].max()
        chart_top = min(
            1.0,
            max(0.05, max_risk * 1.22 + 0.008)
        )

        bars = (
            alt.Chart(chart_df)
            .mark_bar(
                cornerRadiusTopLeft=7,
                cornerRadiusTopRight=7,
                size=54,
                color="#315C7C",
            )
            .encode(
                x=alt.X(
                    "Horizon:N",
                    sort=["3 years", "5 years", "10 years"],
                    title=None,
                    axis=alt.Axis(
                        labelAngle=0,
                        labelColor="#475467",
                        labelFontSize=12,
                        ticks=False,
                        domain=False,
                    ),
                ),
                y=alt.Y(
                    "Risk:Q",
                    title="Predicted cumulative incidence",
                    scale=alt.Scale(domain=[0, chart_top]),
                    axis=alt.Axis(
                        format=".0%",
                        grid=True,
                        gridColor="#EEF1F5",
                        domain=False,
                        tickColor="#D8DEE8",
                        labelColor="#667085",
                        titleColor="#475467",
                        titlePadding=14,
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Horizon:N", title="Horizon"),
                    alt.Tooltip("Risk:Q", title="Risk", format=".2%"),
                ],
            )
        )

        labels = (
            alt.Chart(chart_df)
            .mark_text(
                dy=-12,
                fontSize=13,
                fontWeight=700,
                color="#172033",
            )
            .encode(
                x=alt.X(
                    "Horizon:N",
                    sort=["3 years", "5 years", "10 years"],
                ),
                y="Risk:Q",
                text="Label:N",
            )
        )

        chart = (
            (bars + labels)
            .properties(height=310)
            .configure_view(strokeWidth=0)
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown(
            """
            <div class="soft-note">
                The chart displays the model's exact predictions at the three
                prespecified horizons (3, 5, and 10 years). No low-, medium-,
                or high-risk thresholds are imposed.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # Selected clinical profile
    # --------------------------------------------------------------------------
    with right:
        st.markdown(
            '<div class="section-kicker">Current inputs</div>'
            '<div class="section-title">Selected clinical profile</div>',
            unsafe_allow_html=True,
        )

        pcols = st.columns(2, gap="small")
        profile_items = [
            ("Age", AGE_LABELS[profile["age"]]),
            ("T stage", profile["t_stage"]),
            ("N stage", profile["n_stage"]),
            ("M stage", profile["m_stage"]),
            ("Surgery", profile["surgery"]),
            ("Radiotherapy", profile["radiation"]),
        ]

        for idx, (label, value) in enumerate(profile_items):
            with pcols[idx % 2]:
                st.markdown(
                    profile_card(label, value),
                    unsafe_allow_html=True,
                )
                st.write("")

        st.markdown(
            """
            <div class="soft-note">
                These values are cumulative incidences of MTC-specific death,
                not overall-survival probabilities and not estimates of
                treatment benefit.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------------------
# 8. Model validation summary
# ------------------------------------------------------------------------------

st.write("")
st.divider()
st.markdown(
    '<div class="section-kicker">Model validation</div>'
    '<div class="section-title">Held-out internal validation</div>',
    unsafe_allow_html=True,
)

v1, v2, v3, v4 = st.columns(4, gap="small")

validation_specs = [
    ("Competing-risk C-index", "0.863"),
    ("3-year AUC", "0.919"),
    ("5-year AUC", "0.929"),
    ("10-year AUC", "0.858"),
]

for col, (label, value) in zip([v1, v2, v3, v4], validation_specs):
    with col:
        st.markdown(
            validation_card(label, value),
            unsafe_allow_html=True,
        )

st.caption(
    "Development cohort: 1,521 patients. Held-out internal validation cohort: "
    "651 patients. Time-dependent AUCs are reported at the prespecified "
    "3-, 5-, and 10-year horizons."
)


# ------------------------------------------------------------------------------
# 9. Model information
# ------------------------------------------------------------------------------

with st.expander("Model details and technical information"):
    st.markdown(
        """
        **Final model:** Fine–Gray competing-risk regression  
        **Predictors:** Age, T stage, N stage, M stage, surgery, and radiotherapy  
        **Prediction horizons:** 36, 60, and 120 months  
        **Prediction target:** Cumulative incidence of MTC-specific death  
        **Competing event:** Death from other causes  

        The application uses exact predictions exported from the locked final
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
# 10. Disclaimer
# ------------------------------------------------------------------------------

st.divider()
st.markdown(
    """
    <div class="footer-note">
        <b>Research-use disclaimer.</b> This calculator was developed from
        retrospective SEER data and has undergone held-out internal validation,
        but not independent external validation. It is intended for research
        and educational use and should not replace individualized clinical
        judgment.
    </div>
    """,
    unsafe_allow_html=True,
)
