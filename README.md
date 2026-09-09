# Lab-4: Applied Statistical Modeling & Interactive Web Dashboard

This project is deployed and can be accessed at:-
https://restaurant-tips-202618050.streamlit.app/

**Dataset:** Restaurant Tipping Behavior (`tips`) — 244 records, 7 features
(`total_bill`, `tip`, `sex`, `smoker`, `day`, `time`, `size`).

## Repository Structure

```
DS602_Lab4_202618050/
├── app.py             
├── requirements.txt
├── background.py (UI theme module)
├── data/
│   └── tips.csv
└── README.md
```

## How to Run

### Local (VS Code / terminal)

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py            # opens at http://localhost:8501
```

### Google Colab (fallback)

```python
!pip install streamlit
%%writefile app.py
# ...paste app.py contents...
!streamlit run app.py & npx localtunnel --port 8501
```

## Dataset Summary

| Column | Type | Description |
|---|---|---|
| `total_bill` | float | Total bill amount ($) |
| `tip` | float | Tip amount ($) |
| `sex` | categorical | Payer's sex |
| `smoker` | categorical | Whether the party included a smoker |
| `day` | categorical | Day of week (Thur/Fri/Sat/Sun) |
| `time` | categorical | Lunch or Dinner |
| `size` | int | Party size |

## App Tabs

1. **Analysis Report (Parts 1 & 2)**:
   descriptive stats, distribution/scatter/correlation plots, the two
   required hypothesis tests, the OLS model summary, VIF, and residual
   diagnostics, all computed live from the data (not hardcoded).
2. **Data Exploration** — sidebar filters (bill/size range, day, time,
   smoker, sex) driving reactive Plotly charts and summary stats.
3. **Hypothesis Testing Lab** — pick any numeric metric and grouping
   variable to auto-run Shapiro-Wilk, Levene's, and the appropriate
   t-test / Mann-Whitney / ANOVA / Kruskal-Wallis test, or run a
   Chi-Square test of association between two categorical variables.
4. **Live Prediction & Diagnostics** — enter bill/size/smoker/sex/day/time
   to get a live tip prediction with a 95% prediction interval, plus the
   full model's residual diagnostic plots.

## Statistical Findings

**Hypothesis Test 1 — Smokers vs Non-Smokers (tip amount).** Shapiro-Wilk
rejected normality for at least one group, so a Mann-Whitney U test was used
(U = 7163.0, p = 0.792). **Fail to reject H0** — no statistically significant
difference in tip amount between smokers and non-smokers at α = 0.05.

**Hypothesis Test 2 — One-Way ANOVA (tip across day).** F = 1.672, p = 0.174.
**Fail to reject H0** — mean tip does not differ significantly across
Thursday, Friday, Saturday, and Sunday.

**OLS Regression** — `tip ~ total_bill + size + smoker + sex + day + time`:
R² = 0.470, Adjusted R² = 0.452. `total_bill` is by far the strongest,
most significant predictor of `tip`; the categorical predictors (`smoker`,
`sex`, `day`, `time`) and `size` add comparatively little explanatory power
once `total_bill` is in the model.

**Diagnostics.** VIF for the continuous predictors is low (no serious
multicollinearity). The residuals-vs-fitted plot shows mild
heteroscedasticity (spread widens with fitted values), and the Q-Q plot /
Jarque-Bera test (p < 0.001) indicate right-skewed, non-normal residuals —
so the constant-variance and normality Gauss-Markov assumptions are only
partially met. A log-transform of `tip` and/or `total_bill` would likely
improve the fit.


