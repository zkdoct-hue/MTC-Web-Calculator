# ==============================================================================
# MTC Fine-Gray Web Calculator — Polished v5
# ==============================================================================
# Locked final model:
#   Age + T stage + N stage + M stage + Surgery + Radiotherapy
#
# Outputs:
#   Exact 3-, 5-, and 10-year cumulative incidence of MTC-specific death,
#   accounting for other-cause death as a competing event.
#
# Interpretation:
#   Individual contributions are exact X*beta components of the same locked
#   Fine-Gray model, relative to the reference categories. They are NOT SHAP
#   values and should not be interpreted causally.
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
# 2. Styling
# ------------------------------------------------------------------------------

st.markdown(
    """
    <style>
        :root {
            --ink: #172033;
            --muted: #667085;
            --line: #E5E9F0;
            --panel: #F7F9FC;
            --accent: #315C7C;
            --accent-dark: #24465F;
            --accent-soft: #EAF1F6;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.6rem;
            padding-bottom: 3.2rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--line);
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: -0.02em;
        }

        .hero {
            padding: 0.1rem 0 1.0rem 0;
        }

        .eyebrow {
            display: inline-block;
            padding: 0.30rem 0.68rem;
            margin-bottom: 0.66rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent-dark);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.045em;
        }

        .hero-title {
            margin: 0;
            font-size: clamp(2rem, 3.6vw, 2.9rem);
            line-height: 1.08;
            font-weight: 780;
            color: var(--ink);
            letter-spacing: -0.035em;
        }

        .hero-subtitle {
            margin-top: 0.62rem;
            max-width: 1000px;
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.58;
        }

        .section-kicker {
            margin-top: 0.05rem;
            margin-bottom: 0.14rem;
            color: var(--accent);
            font-size: 0.74rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .section-title {
            margin-top: 0;
            margin-bottom: 0.75rem;
            font-size: 1.35rem;
            line-height: 1.25;
            font-weight: 740;
            color: var(--ink);
        }

        .risk-card {
            min-height: 132px;
            padding: 1.08rem 1.15rem 1rem 1.15rem;
            border: 1px solid var(--line);
            border-radius: 15px;
            background: #FFFFFF;
            box-shadow: 0 4px 16px rgba(23, 32, 51, 0.04);
        }

        .risk-horizon {
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 740;
            text-transform: uppercase;
            letter-spacing: 0.055em;
        }

        .risk-value {
            margin-top: 0.28rem;
            color: var(--ink);
            font-size: 2.05rem;
            font-weight: 790;
            letter-spacing: -0.035em;
            line-height: 1.10;
        }

        .risk-label {
            margin-top: 0.34rem;
            color: var(--muted);
            font-size: 0.80rem;
            line-height: 1.35;
        }

        .profile-chip {
            min-height: 69px;
            padding: 0.70rem 0.82rem;
            border: 1px solid var(--line);
            border-radius: 11px;
            background: var(--panel);
        }

        .profile-label {
            color: var(--muted);
            font-size: 0.70rem;
            font-weight: 650;
        }

        .profile-value {
            margin-top: 0.18rem;
            color: var(--ink);
            font-size: 0.96rem;
            font-weight: 720;
        }

        .soft-note {
            padding: 0.80rem 0.92rem;
            border-left: 3px solid var(--accent);
            border-radius: 9px;
            background: var(--accent-soft);
            color: #334155;
            font-size: 0.82rem;
            line-height: 1.48;
        }

        .empty-state {
            margin-top: 1.1rem;
            padding: 1.8rem 1.2rem;
            border: 1px dashed #C8D1DC;
            border-radius: 15px;
            background: #FAFBFD;
            text-align: center;
            color: var(--muted);
        }

        .empty-state strong {
            display: block;
            margin-bottom: 0.28rem;
            color: var(--ink);
            font-size: 1.02rem;
        }

        .footer-note {
            color: var(--muted);
            font-size: 0.77rem;
            line-height: 1.48;
        }

        div[data-testid="stFormSubmitButton"] button {
            width: 100%;
            border-radius: 10px;
            border: 1px solid var(--accent-dark);
            background: var(--accent);
            color: white;
            font-weight: 720;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background: var(--accent-dark);
            border-color: var(--accent-dark);
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
                padding-top: 1.0rem;
            }
            .hero-title {
                font-size: 2.0rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 3. Load verified locked-model exports
# ------------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
LOOKUP_FILE = APP_DIR / "MTC_FineGray_WebCalculator_Lookup.csv"
META_FILE = APP_DIR / "MTC_FineGray_WebCalculator_Metadata.csv"


@st.cache_data
def load_inputs():
    if not LOOKUP_FILE.exists():
        raise FileNotFoundError(
            "MTC_FineGray_WebCalculator_Lookup.csv was not found."
        )
    if not META_FILE.exists():
        raise FileNotFoundError(
            "MTC_FineGray_WebCalculator_Metadata.csv was not found."
        )

    lookup = pd.read_csv(
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
    meta = pd.read_csv(META_FILE)

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
        "Linear_predictor",
    }
    missing = required.difference(lookup.columns)
    if missing:
        raise ValueError(
            "The lookup CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if len(lookup) != 192:
        raise ValueError(
            f"Expected 192 predictor profiles; found {len(lookup)}."
        )

    if lookup["Profile_key"].duplicated().any():
        raise ValueError("Duplicate Profile_key values were found.")

    for col in ["Risk_36m", "Risk_60m", "Risk_120m", "Linear_predictor"]:
        lookup[col] = pd.to_numeric(lookup[col], errors="raise")

    if ((lookup[["Risk_36m", "Risk_60m", "Risk_120m"]] < 0) |
        (lookup[["Risk_36m", "Risk_60m", "Risk_120m"]] > 1)).any().any():
        raise ValueError("Predicted risks outside [0, 1] were found.")

    meta_dict = dict(zip(meta["Item"].astype(str), meta["Value"]))

    # Exact coefficients exported from the same locked model.
    coef = {
        item.replace("Coefficient_", "", 1): float(value)
        for item, value in meta_dict.items()
        if item.startswith("Coefficient_")
    }

    expected_coef_terms = {
        "Age=55-69",
        "Age=>=70",
        "T_stage=T2",
        "T_stage=T3",
        "T_stage=T4",
        "N_stage=N1",
        "M_stage=M1",
        "Surgery=Yes",
        "Radiation=Yes",
    }

    if set(coef) != expected_coef_terms:
        raise ValueError(
            "Coefficient metadata does not match the locked six-variable model."
        )

    return lookup, meta, coef


try:
    lookup, metadata, COEF = load_inputs()
except Exception as exc:
    st.error("The locked Fine–Gray model exports could not be loaded.")
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


def pct(value):
    value = 100.0 * float(value)
    if value >= 99.95:
        return ">99.9%"
    return f"{value:.1f}%"


def profile_key(age, t_stage, n_stage, m_stage, surgery, radiation):
    return "|".join(
        [age, t_stage, n_stage, m_stage, surgery, radiation]
    )


def chip(label, value):
    return f"""
    <div class="profile-chip">
        <div class="profile-label">{label}</div>
        <div class="profile-value">{value}</div>
    </div>
    """


def individual_contributions(p):
    # Exact X*beta components on the Fine–Gray linear-predictor scale.
    rows = [
        {
            "Variable": "Age",
            "Level": AGE_LABELS[p["age"]],
            "Contribution": (
                0.0 if p["age"] == "<55"
                else COEF[f"Age={p['age']}"]
            ),
        },
        {
            "Variable": "T stage",
            "Level": p["t_stage"],
            "Contribution": (
                0.0 if p["t_stage"] == "T1"
                else COEF[f"T_stage={p['t_stage']}"]
            ),
        },
        {
            "Variable": "N stage",
            "Level": p["n_stage"],
            "Contribution": (
                0.0 if p["n_stage"] == "N0"
                else COEF["N_stage=N1"]
            ),
        },
        {
            "Variable": "M stage",
            "Level": p["m_stage"],
            "Contribution": (
                0.0 if p["m_stage"] == "M0"
                else COEF["M_stage=M1"]
            ),
        },
        {
            "Variable": "Surgery",
            "Level": p["surgery"],
            "Contribution": (
                0.0 if p["surgery"] == "No"
                else COEF["Surgery=Yes"]
            ),
        },
        {
            "Variable": "Radiotherapy",
            "Level": p["radiation"],
            "Contribution": (
                0.0 if p["radiation"] == "No"
                else COEF["Radiation=Yes"]
            ),
        },
    ]

    df = pd.DataFrame(rows)
    df["Display"] = df["Variable"] + "  ·  " + df["Level"]
    df["Direction"] = df["Contribution"].apply(
        lambda x: "Higher model score" if x > 1e-12
        else ("Lower model score" if x < -1e-12 else "Reference category")
    )
    df["Contribution_label"] = df["Contribution"].map(lambda x: f"{x:+.2f}")
    return df


# ------------------------------------------------------------------------------
# 5. Sidebar
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

if submitted:
    st.session_state["mtc_profile"] = {
        "age": age,
        "t_stage": t_stage,
        "n_stage": n_stage,
        "m_stage": m_stage,
        "surgery": surgery,
        "radiation": radiation,
    }

st.sidebar.markdown("---")
st.sidebar.caption(
    "Outcome: cumulative incidence of MTC-specific death. "
    "Other-cause death is treated as a competing event."
)


# ------------------------------------------------------------------------------
# 6. Hero
# ------------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">FINE–GRAY COMPETING-RISK MODEL</div>
        <h1 class="hero-title">MTC-Specific Mortality Risk Calculator</h1>
        <div class="hero-subtitle">
            Individualized 3-, 5-, and 10-year predictions for medullary
            thyroid carcinoma. The model explicitly accounts for death from
            other causes as a competing event.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 7. Results
# ------------------------------------------------------------------------------

p = st.session_state.get("mtc_profile")

if p is None:
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
    key = profile_key(
        p["age"], p["t_stage"], p["n_stage"],
        p["m_stage"], p["surgery"], p["radiation"]
    )

    row = lookup.loc[lookup["Profile_key"] == key]

    if len(row) != 1:
        st.error("The selected patient profile could not be matched uniquely.")
        st.stop()

    r = row.iloc[0]

    # ---- Risk cards -----------------------------------------------------------
    st.markdown(
        '<div class="section-kicker">Individual prediction</div>'
        '<div class="section-title">Predicted MTC-specific mortality</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="medium")

    for col, label, val in [
        (c1, "3-year risk", r["Risk_36m"]),
        (c2, "5-year risk", r["Risk_60m"]),
        (c3, "10-year risk", r["Risk_120m"]),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="risk-card">
                    <div class="risk-horizon">{label}</div>
                    <div class="risk-value">{pct(val)}</div>
                    <div class="risk-label">
                        Cumulative incidence of MTC-specific death
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # ---- Main interactive visualization area ---------------------------------
    left, right = st.columns([1.0, 1.12], gap="large")

    with left:
        st.markdown(
            '<div class="section-kicker">Prediction horizons</div>'
            '<div class="section-title">Risk across prespecified horizons</div>',
            unsafe_allow_html=True,
        )

        risk_df = pd.DataFrame(
            {
                "Years": [3, 5, 10],
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

        max_risk = risk_df["Risk"].max()
        y_top = min(1.0, max(0.05, max_risk * 1.24 + 0.008))

        line = (
            alt.Chart(risk_df)
            .mark_line(
                point=alt.OverlayMarkDef(
                    filled=True,
                    size=95,
                    color="#315C7C",
                ),
                strokeWidth=3,
                color="#315C7C",
            )
            .encode(
                x=alt.X(
                    "Years:Q",
                    title="Prediction horizon (years)",
                    scale=alt.Scale(domain=[2.5, 10.5]),
                    axis=alt.Axis(
                        values=[3, 5, 10],
                        format="d",
                        grid=False,
                        domain=False,
                        labelColor="#475467",
                        titleColor="#475467",
                        titlePadding=12,
                    ),
                ),
                y=alt.Y(
                    "Risk:Q",
                    title="Predicted cumulative incidence",
                    scale=alt.Scale(domain=[0, y_top]),
                    axis=alt.Axis(
                        format=".0%",
                        grid=True,
                        gridColor="#EEF1F5",
                        domain=False,
                        labelColor="#667085",
                        titleColor="#475467",
                        titlePadding=12,
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Years:Q", title="Horizon", format=".0f"),
                    alt.Tooltip("Risk:Q", title="Predicted risk", format=".2%"),
                ],
            )
        )

        labels = (
            alt.Chart(risk_df)
            .mark_text(
                dy=-14,
                fontSize=13,
                fontWeight=700,
                color="#172033",
            )
            .encode(
                x="Years:Q",
                y="Risk:Q",
                text="Label:N",
            )
        )

        st.altair_chart(
            (line + labels)
            .properties(height=315)
            .configure_view(strokeWidth=0),
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="soft-note">
                The line connects the model's three prespecified prediction
                horizons for visualization; it should not be interpreted as
                an estimated continuous risk curve between those time points.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="section-kicker">Individual model explanation</div>'
            '<div class="section-title">Predictor contributions to the Fine–Gray score</div>',
            unsafe_allow_html=True,
        )

        contrib = individual_contributions(p)

        # Verify exact decomposition against the exported lookup linear predictor.
        contribution_sum = float(contrib["Contribution"].sum())
        locked_lp = float(r["Linear_predictor"])
        if abs(contribution_sum - locked_lp) > 1e-8:
            st.error(
                "Internal model-audit check failed: predictor contributions do "
                "not reproduce the locked-model linear predictor."
            )
            st.stop()

        max_abs = max(0.35, float(contrib["Contribution"].abs().max()) * 1.25)

        contribution_chart = (
            alt.Chart(contrib)
            .mark_bar(
                cornerRadiusEnd=5,
                size=24,
            )
            .encode(
                y=alt.Y(
                    "Display:N",
                    sort=alt.SortField(
                        field="Contribution",
                        order="descending",
                    ),
                    title=None,
                    axis=alt.Axis(
                        labelLimit=190,
                        labelColor="#344054",
                        labelFontSize=11,
                        ticks=False,
                        domain=False,
                    ),
                ),
                x=alt.X(
                    "Contribution:Q",
                    title="Contribution to Fine–Gray linear predictor (Xβ)",
                    scale=alt.Scale(domain=[-max_abs, max_abs]),
                    axis=alt.Axis(
                        grid=True,
                        gridColor="#EEF1F5",
                        domain=False,
                        labelColor="#667085",
                        titleColor="#475467",
                        titlePadding=12,
                    ),
                ),
                color=alt.Color(
                    "Direction:N",
                    scale=alt.Scale(
                        domain=[
                            "Higher model score",
                            "Lower model score",
                            "Reference category",
                        ],
                        range=[
                            "#9A4F46",
                            "#3E6C8A",
                            "#B7C0CC",
                        ],
                    ),
                    legend=alt.Legend(
                        title=None,
                        orient="bottom",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("Variable:N", title="Variable"),
                    alt.Tooltip("Level:N", title="Selected level"),
                    alt.Tooltip(
                        "Contribution:Q",
                        title="Xβ contribution",
                        format="+.3f",
                    ),
                ],
            )
        )

        zero_line = (
            alt.Chart(pd.DataFrame({"x": [0]}))
            .mark_rule(
                color="#475467",
                strokeWidth=1.2,
            )
            .encode(x="x:Q")
        )

        contrib_labels = (
            alt.Chart(contrib)
            .mark_text(
                align="left",
                baseline="middle",
                dx=6,
                fontSize=11,
                fontWeight=650,
                color="#172033",
            )
            .encode(
                y=alt.Y(
                    "Display:N",
                    sort=alt.SortField(
                        field="Contribution",
                        order="descending",
                    ),
                ),
                x="Contribution:Q",
                text="Contribution_label:N",
            )
        )

        st.altair_chart(
            (contribution_chart + zero_line + contrib_labels)
            .properties(height=315)
            .configure_view(strokeWidth=0),
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="soft-note">
                Contributions are exact components of the same final Fine–Gray
                model on the linear-predictor scale, relative to the reference
                categories. Positive values increase the model score; negative
                values decrease it. Treatment-related contributions are
                predictive associations and must not be interpreted causally.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Compact selected profile --------------------------------------------
    st.write("")
    st.markdown(
        '<div class="section-kicker">Current inputs</div>'
        '<div class="section-title">Selected clinical profile</div>',
        unsafe_allow_html=True,
    )

    profile_cols = st.columns(6, gap="small")
    profile_items = [
        ("Age", AGE_LABELS[p["age"]]),
        ("T stage", p["t_stage"]),
        ("N stage", p["n_stage"]),
        ("M stage", p["m_stage"]),
        ("Surgery", p["surgery"]),
        ("Radiotherapy", p["radiation"]),
    ]

    for col, (label, value) in zip(profile_cols, profile_items):
        with col:
            st.markdown(chip(label, value), unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 8. Collapsible model-performance section
# ------------------------------------------------------------------------------

st.write("")
st.divider()

with st.expander("Model performance & held-out internal validation"):
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Competing-risk C-index", "0.863")
    p2.metric("3-year AUC", "0.919")
    p3.metric("5-year AUC", "0.929")
    p4.metric("10-year AUC", "0.858")

    st.caption(
        "Development cohort: 1,521 patients. Held-out internal validation cohort: "
        "651 patients. Time-dependent AUCs are reported at 3, 5, and 10 years."
    )


# ------------------------------------------------------------------------------
# 9. Collapsible technical section
# ------------------------------------------------------------------------------

with st.expander("Model details & technical information"):
    st.markdown(
        """
        **Final model:** Fine–Gray competing-risk regression  
        **Predictors:** Age, T stage, N stage, M stage, surgery, and radiotherapy  
        **Prediction horizons:** 36, 60, and 120 months  
        **Prediction target:** Cumulative incidence of MTC-specific death  
        **Competing event:** Death from other causes  

        The calculator uses exact predictions exported from the locked final
        Fine–Gray model for all 192 possible predictor combinations.

        The individual contribution plot is not a SHAP plot. It displays the
        exact selected-level coefficient contribution to the Fine–Gray linear
        predictor and is internally checked against the exported locked-model
        linear predictor.
        """
    )

    with st.popover("Technical model metadata"):
        st.dataframe(
            metadata,
            hide_index=True,
            use_container_width=True,
        )


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
