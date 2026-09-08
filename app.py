"""
Lab-4: Applied Statistical Modeling & Interactive Web Dashboard
Part 3 — Interactive Streamlit Dashboard
Dataset: Restaurant Tipping Behavior (tips dataset)

Run with:
    streamlit run app.py
"""

import streamlit as st
from background import apply_background

st.set_page_config(
    page_title="Tipping Behavior Dashboard",
    layout="wide"
)

apply_background()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from scipy import stats
from statsmodels.stats.stattools import jarque_bera, omni_normtest


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Tipping Behavior Dashboard",
    layout="wide"
)

ALPHA = 0.05

NUM_COLS = [
    "total_bill",
    "tip",
    "size"
]

CAT_COLS = [
    "sex",
    "smoker",
    "day",
    "time"
]

MODEL_FORMULA = "tip ~ total_bill + size + smoker + sex + day + time"


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    """Load and prepare the tips dataset."""

    data = pd.read_csv("data/tips.csv")

    data["day"] = pd.Categorical(
        data["day"],
        categories=["Thur", "Fri", "Sat", "Sun"],
        ordered=True
    )

    return data


# =============================================================================
# MODEL
# =============================================================================

@st.cache_resource
def fit_model(data):
    """Fit the OLS regression model."""

    return smf.ols(
        MODEL_FORMULA,
        data=data
    ).fit()


# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

def descriptive_stats(data):
    """Calculate descriptive statistics."""

    desc = data[NUM_COLS].agg(
        ["mean", "median", "std"]
    ).T

    desc["IQR"] = (
        data[NUM_COLS].quantile(0.75)
        - data[NUM_COLS].quantile(0.25)
    )

    desc["skewness"] = data[NUM_COLS].skew()
    desc["kurtosis"] = data[NUM_COLS].kurt()

    return desc.round(3)


# =============================================================================
# RESIDUAL DIAGNOSTICS
# =============================================================================

def residual_diagnostic_figs(resid, fitted_vals):
    """
    Return residual-vs-fitted and Q-Q plot figures.
    """

    # -------------------------------------------------------------------------
    # Residuals vs Fitted
    # -------------------------------------------------------------------------

    fig_resid = px.scatter(
        x=fitted_vals,
        y=resid,
        labels={
            "x": "Fitted values",
            "y": "Residuals"
        },
        title="Residuals vs Fitted (Linearity / Homoscedasticity)"
    )

    fig_resid.add_hline(
        y=0,
        line_dash="dash",
        line_color="red"
    )

    # -------------------------------------------------------------------------
    # Q-Q Plot
    # -------------------------------------------------------------------------

    qq = sm.ProbPlot(resid)

    theo_q = qq.theoretical_quantiles
    samp_q = qq.sample_quantiles

    slope, intercept = np.polyfit(
        theo_q,
        samp_q,
        1
    )

    line_vals = np.array([
        theo_q.min(),
        theo_q.max()
    ])

    fig_qq = go.Figure()

    fig_qq.add_trace(
        go.Scatter(
            x=theo_q,
            y=samp_q,
            mode="markers",
            name="Residuals"
        )
    )

    fig_qq.add_trace(
        go.Scatter(
            x=line_vals,
            y=slope * line_vals + intercept,
            mode="lines",
            name="Reference",
            line=dict(
                color="red",
                dash="dash"
            )
        )
    )

    fig_qq.update_layout(
        title="Q-Q Plot of Residuals",
        xaxis_title="Theoretical quantiles",
        yaxis_title="Sample quantiles"
    )

    return fig_resid, fig_qq


# =============================================================================
# LOAD DATA AND MODEL
# =============================================================================

df = load_data()

model = fit_model(df)


# =============================================================================
# HEADER
# =============================================================================

st.title(
    "Restaurant Tipping Behavior — Statistical Dashboard"
)

st.caption(
    "Lab-4 · Applied Statistical Modeling & Interactive Web Dashboard · "
    "`tips` dataset (244 records)"
)


# =============================================================================
# SIDEBAR — DATA EXPLORATION FILTERS
# =============================================================================

st.sidebar.header("Data Exploration Filters")

st.sidebar.caption(
    "These filters affect only the Data Exploration tab."
)

bill_min = float(df["total_bill"].min())
bill_max = float(df["total_bill"].max())

size_min = int(df["size"].min())
size_max = int(df["size"].max())


with st.sidebar.form("filter_form"):

    bill_range = st.slider(
        "Total bill range ($)",
        min_value=bill_min,
        max_value=bill_max,
        value=(bill_min, bill_max),
        step=0.5
    )

    size_range = st.slider(
        "Party size range",
        min_value=size_min,
        max_value=size_max,
        value=(size_min, size_max),
        step=1
    )

    day_sel = st.multiselect(
        "Day",
        options=list(df["day"].cat.categories),
        default=[]
    )

    time_sel = st.multiselect(
        "Time",
        options=sorted(df["time"].unique()),
        default=[]
    )

    smoker_sel = st.multiselect(
        "Smoker",
        options=sorted(df["smoker"].unique()),
        default=[]
    )

    sex_sel = st.multiselect(
        "Sex",
        options=sorted(df["sex"].unique()),
        default=[]
    )

    st.form_submit_button(
        "Apply Filters",
        width="stretch"
    )


# =============================================================================
# FILTER FUNCTION
# =============================================================================

def match_filter(series, selected):
    """Return True for all rows if no filter is selected."""

    if not selected:
        return pd.Series(
            True,
            index=series.index
        )

    return series.isin(selected)


filtered = df[
    df["total_bill"].between(
        bill_range[0],
        bill_range[1]
    )
    &
    df["size"].between(
        size_range[0],
        size_range[1]
    )
    &
    match_filter(df["day"], day_sel)
    &
    match_filter(df["time"], time_sel)
    &
    match_filter(df["smoker"], smoker_sel)
    &
    match_filter(df["sex"], sex_sel)
]


# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs(
    [
        "Data Exploration",
        "Hypothesis Testing Lab",
        "Live Prediction & Diagnostics"
    ]
)


# =============================================================================
# TAB 1 — DATA EXPLORATION
# =============================================================================

with tab1:

    st.subheader("Summary Statistics")

    if filtered.empty:

        st.warning(
            "No records match the selected filters."
        )

    else:

        st.dataframe(
            descriptive_stats(filtered),
            width="stretch"
        )

        # ---------------------------------------------------------------------
        # Distribution and Scatter Plot
        # ---------------------------------------------------------------------

        col_a, col_b = st.columns(2)

        with col_a:

            metric_col = st.selectbox(
                "Distribution of:",
                NUM_COLS,
                index=1,
                key="tab1_metric"
            )

            fig_hist = px.histogram(
                filtered,
                x=metric_col,
                nbins=25,
                marginal="box",
                color_discrete_sequence=["#4C78A8"],
                title=f"Distribution of {metric_col}"
            )

            st.plotly_chart(
                fig_hist,
                width="stretch",
                key="tab1_hist"
            )

        with col_b:

            color_by = st.selectbox(
                "Color scatter by:",
                CAT_COLS,
                index=1,
                key="tab1_color"
            )

            fig_scatter = px.scatter(
                filtered,
                x="total_bill",
                y="tip",
                color=color_by,
                size="size",
                trendline="ols",
                title="Tip vs Total Bill"
            )

            st.plotly_chart(
                fig_scatter,
                width="stretch",
                key="tab1_scatter"
            )

        # ---------------------------------------------------------------------
        # Correlation Matrix
        # ---------------------------------------------------------------------

        st.subheader("Correlation Matrix")

        corr = filtered[NUM_COLS].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix"
        )

        st.plotly_chart(
            fig_corr,
            width="stretch",
            key="tab1_corr"
        )

        # ---------------------------------------------------------------------
        # Raw Data
        # ---------------------------------------------------------------------

        with st.expander("View filtered raw data"):

            st.dataframe(
                filtered,
                width="stretch"
            )


# =============================================================================
# TAB 2 — HYPOTHESIS TESTING LAB
# =============================================================================

with tab2:

    st.subheader("Hypothesis Testing Lab")

    st.write(
        "Run a hypothesis test using the full dataset. "
    )

    test_family = st.radio(
        "Test type",
        [
            "Compare a numeric metric across groups",
            "Association between two categorical variables"
        ],
        horizontal=True,
        key="test_family"
    )

    # =========================================================================
    # NUMERIC VS GROUP
    # =========================================================================

    if test_family == "Compare a numeric metric across groups":

        col1, col2 = st.columns(2)

        metric = col1.selectbox(
            "Numeric metric",
            NUM_COLS,
            index=1,
            key="ht_metric"
        )

        group_var = col2.selectbox(
            "Group by (categorical)",
            CAT_COLS,
            index=1,
            key="ht_group"
        )

        groups = {
            level: df.loc[
                df[group_var] == level,
                metric
            ].dropna()
            for level in df[group_var].unique()
        }

        groups = {
            key: value
            for key, value in groups.items()
            if len(value) > 2
        }

        n_groups = len(groups)

        if n_groups < 2:

            st.error(
                "At least two groups are required."
            )

        else:

            st.markdown(
                f"""
                **H0:** Mean `{metric}` is equal across all
                levels of `{group_var}`.

                **H1:** At least one group's mean `{metric}` differs.
                """
            )

            # -----------------------------------------------------------------
            # Shapiro-Wilk Normality Test
            # -----------------------------------------------------------------

            st.markdown(
                "**Normality check (Shapiro-Wilk)**"
            )

            sw_rows = []

            for level, values in groups.items():

                sw = stats.shapiro(values)

                sw_rows.append(
                    {
                        "group": level,
                        "n": len(values),
                        "statistic": round(
                            sw.statistic,
                            4
                        ),
                        "p_value": round(
                            sw.pvalue,
                            4
                        )
                    }
                )

            sw_df = pd.DataFrame(sw_rows)

            st.dataframe(
                sw_df,
                width="stretch",
                hide_index=True
            )

            all_normal = (
                sw_df["p_value"] > ALPHA
            ).all()

            # -----------------------------------------------------------------
            # Levene's Test
            # -----------------------------------------------------------------

            levene_stat, levene_p = stats.levene(
                *groups.values()
            )

            st.markdown(
                f"**Levene's test (equal variance):** "
                f"statistic = {levene_stat:.4f}, "
                f"p = {levene_p:.4f}"
            )

            equal_var = levene_p > ALPHA

            # -----------------------------------------------------------------
            # Choose Appropriate Test
            # -----------------------------------------------------------------

            if n_groups == 2:

                (name_a, vals_a), (
                    name_b,
                    vals_b
                ) = list(groups.items())

                if all_normal:

                    stat, p = stats.ttest_ind(
                        vals_a,
                        vals_b,
                        equal_var=equal_var
                    )

                    test_name = (
                        f"Two-sample t-test "
                        f"(equal_var={equal_var})"
                    )

                else:

                    stat, p = stats.mannwhitneyu(
                        vals_a,
                        vals_b,
                        alternative="two-sided"
                    )

                    test_name = "Mann-Whitney U test"

            else:

                if all_normal:

                    stat, p = stats.f_oneway(
                        *groups.values()
                    )

                    test_name = "One-Way ANOVA"

                else:

                    stat, p = stats.kruskal(
                        *groups.values()
                    )

                    test_name = "Kruskal-Wallis H test"

            # -----------------------------------------------------------------
            # Conclusion
            # -----------------------------------------------------------------

            if p < ALPHA:

                conclusion = (
                    "Reject H0 — statistically significant difference"
                )

            else:

                conclusion = (
                    "Fail to Reject H0 — no significant difference"
                )

            st.markdown("---")

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Test used",
                test_name
            )

            m2.metric(
                "Statistic",
                f"{stat:.4f}"
            )

            m3.metric(
                "p-value",
                f"{p:.4f}"
            )

            st.markdown(
                f"### Conclusion (α = {ALPHA}): {conclusion}"
            )

            # -----------------------------------------------------------------
            # Box Plot
            # -----------------------------------------------------------------

            fig_box = px.box(
                df,
                x=group_var,
                y=metric,
                color=group_var,
                points="all",
                title=f"{metric} by {group_var}"
            )

            st.plotly_chart(
                fig_box,
                width="stretch",
                key="tab2_box"
            )

    # =========================================================================
    # CHI-SQUARE TEST
    # =========================================================================

    else:

        col1, col2 = st.columns(2)

        var_a = col1.selectbox(
            "Categorical variable A",
            CAT_COLS,
            index=1,
            key="chi_a"
        )

        var_b = col2.selectbox(
            "Categorical variable B",
            CAT_COLS,
            index=2,
            key="chi_b"
        )

        if var_a == var_b:

            st.warning(
                "Pick two different variables."
            )

        else:

            st.markdown(
                f"""
                **H0:** `{var_a}` and `{var_b}` are independent.

                **H1:** `{var_a}` and `{var_b}` are associated.
                """
            )

            # -----------------------------------------------------------------
            # Contingency Table
            # -----------------------------------------------------------------

            contingency = pd.crosstab(
                df[var_a],
                df[var_b]
            )

            st.markdown(
                "**Contingency table**"
            )

            st.dataframe(
                contingency,
                width="stretch"
            )

            # -----------------------------------------------------------------
            # Chi-Square Test
            # -----------------------------------------------------------------

            chi2, p, dof, expected = stats.chi2_contingency(
                contingency
            )

            if p < ALPHA:

                conclusion = (
                    "Reject H0 — variables are significantly associated"
                )

            else:

                conclusion = (
                    "Fail to Reject H0 — no significant association"
                )

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Chi-square statistic",
                f"{chi2:.4f}"
            )

            m2.metric(
                "Degrees of freedom",
                dof
            )

            m3.metric(
                "p-value",
                f"{p:.4f}"
            )

            st.markdown(
                f"### Conclusion (α = {ALPHA}): {conclusion}"
            )

            # -----------------------------------------------------------------
            # Bar Chart
            # -----------------------------------------------------------------

            fig_bar = px.bar(
                contingency.reset_index().melt(
                    id_vars=var_a
                ),
                x=var_a,
                y="value",
                color=var_b,
                barmode="group",
                title=f"{var_a} vs {var_b} counts"
            )

            st.plotly_chart(
                fig_bar,
                width="stretch",
                key="tab2_bar"
            )


# =============================================================================
# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
# =============================================================================

with tab3:

    st.subheader(
        "Model: tip ~ total_bill + size + smoker + sex + day + time (OLS)"
    )

    # -------------------------------------------------------------------------
    # Model Performance
    # -------------------------------------------------------------------------

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "R-squared",
        f"{model.rsquared:.4f}"
    )

    m2.metric(
        "Adjusted R-squared",
        f"{model.rsquared_adj:.4f}"
    )

    m3.metric(
        "Observations",
        int(model.nobs)
    )

    # -------------------------------------------------------------------------
    # Prediction Inputs
    # -------------------------------------------------------------------------

    st.subheader("Live Tip Prediction")

    p1, p2, p3 = st.columns(3)

    with p1:

        in_bill = st.slider(
            "Total bill ($)",
            min_value=bill_min,
            max_value=bill_max,
            value=float(df["total_bill"].median()),
            step=0.5,
            key="pred_bill"
        )

        in_size = st.slider(
            "Party size",
            min_value=size_min,
            max_value=size_max,
            value=int(df["size"].median()),
            step=1,
            key="pred_size"
        )

    with p2:

        in_smoker = st.selectbox(
            "Smoker",
            sorted(df["smoker"].unique()),
            key="pred_smoker"
        )

        in_sex = st.selectbox(
            "Sex",
            sorted(df["sex"].unique()),
            key="pred_sex"
        )

    with p3:

        in_day = st.selectbox(
            "Day",
            list(df["day"].cat.categories),
            key="pred_day"
        )

        in_time = st.selectbox(
            "Time",
            sorted(df["time"].unique()),
            key="pred_time"
        )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    new_point = pd.DataFrame(
        [
            {
                "total_bill": in_bill,
                "size": in_size,
                "smoker": in_smoker,
                "sex": in_sex,
                "day": in_day,
                "time": in_time
            }
        ]
    )

    pred = model.get_prediction(
        new_point
    ).summary_frame(
        alpha=0.05
    )

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Predicted tip",
        f"${pred['mean'].iloc[0]:.2f}"
    )

    r2.metric(
        "95% PI lower",
        f"${pred['obs_ci_lower'].iloc[0]:.2f}"
    )

    r3.metric(
        "95% PI upper",
        f"${pred['obs_ci_upper'].iloc[0]:.2f}"
    )

    # -------------------------------------------------------------------------
    # Residual Diagnostics
    # -------------------------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Residual Diagnostics (Gauss-Markov checks, full dataset)"
    )

    fitted_vals = model.fittedvalues
    resid = model.resid

    fig_resid, fig_qq = residual_diagnostic_figs(
        resid,
        fitted_vals
    )

    d1, d2 = st.columns(2)

    with d1:

        st.plotly_chart(
            fig_resid,
            width="stretch",
            key="tab3_resid"
        )

    with d2:

        st.plotly_chart(
            fig_qq,
            width="stretch",
            key="tab3_qq"
        )

    # -------------------------------------------------------------------------
    # Normality Tests
    # -------------------------------------------------------------------------

    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(
        resid
    )

    omni_stat, omni_p = omni_normtest(
        resid
    )

    j1, j2 = st.columns(2)

    j1.metric(
        "Jarque-Bera p-value",
        f"{jb_p:.4g}"
    )

    j2.metric(
        "Omnibus p-value",
        f"{omni_p:.4g}"
    )

    if jb_p < ALPHA:

        st.caption(
            "Residuals show significant deviation from normality "
            "(JB test) — consistent with the right skew visible "
            "in the Q-Q plot."
        )

    else:

        st.caption(
            "Residuals are consistent with normality (JB test)."
        )