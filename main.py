import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# JIE42403 SCIENTIFIC VISUALISATION
# ============================================================

st.set_page_config(
    page_title="District-Level Analysis of Poverty, Population Demographics and Teacher Availability in Malaysia",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = Path(__file__).parent / "processed_data"

PALETTE = {
    "navy": "#24445C",
    "teal": "#397F91",
    "sage": "#79AFA8",
    "gold": "#D6A66B",
    "slate": "#8795A5",
    "ink": "#263442",
    "muted": "#718096",
    "grid": "#E5EAF0",
    "white": "#FFFFFF",
    "coral": "#B85C50",
    "purple": "#81749B"
}

st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1450px;}
    h1, h2, h3 {color: #24445C;}
    div[data-testid="stMetric"] {
        background: #F5F8FA; border: 1px solid #E5EAF0;
        padding: 14px 16px; border-radius: 10px;
    }
    .insight {
        background: #F5F8FA; border-left: 4px solid #397F91;
        padding: 14px 17px; border-radius: 5px; margin: 8px 0 18px 0;
    }
    .small-note {color: #718096; font-size: 0.88rem;}
</style>
""", unsafe_allow_html=True)


# -------------------- Data loading and validation --------------------

REQUIRED_FILES = {
    "poverty": "poverty_processed.csv",
    "population": "population_total.csv",
    "ethnicity": "population_ethnicity.csv",
    "age": "population_age.csv",
    "teachers": "teachers_processed.csv",
    "integrated": "integrated_2024.csv",
    "education": "education_features.csv"
}

@st.cache_data
def load_data():
    data = {}
    for key, filename in REQUIRED_FILES.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        data[key] = pd.read_csv(path)

    for df in data.values():
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "year" not in df.columns and "date" in df.columns:
            df["year"] = df["date"].dt.year

    # Keep records with a valid year where year is used.
    return data

try:
    DATA = load_data()
except Exception as exc:
    st.error(f"Unable to load data: {exc}")
    st.info("Place all seven processed CSV files inside a folder named 'processed_data' beside main.py.")
    st.stop()

poverty = DATA["poverty"].copy()
population = DATA["population"].copy()
ethnicity = DATA["ethnicity"].copy()
age = DATA["age"].copy()
education = DATA["education"].copy()

# Validate columns needed by the application.
required_columns = {
    "poverty": {"state", "district", "year", "poverty_absolute", "poverty_relative"},
    "population": {"state", "district", "year", "population"},
    "ethnicity": {"state", "district", "year", "ethnicity", "population"},
    "age": {"state", "district", "year", "age", "population"},
    "education": {"state", "district", "year", "poverty_absolute", "teachers_per_1000"}
}
for key, cols in required_columns.items():
    missing = cols - set(DATA[key].columns)
    if missing:
        st.error(f"{REQUIRED_FILES[key]} is missing required columns: {', '.join(sorted(missing))}")
        st.stop()

# Numeric conversion avoids plotting strings as categorical values.
for df, cols in [
    (poverty, ["poverty_absolute", "poverty_relative"]),
    (population, ["population"]),
    (ethnicity, ["population"]),
    (age, ["population"]),
    (education, ["poverty_absolute", "poverty_relative", "population", "teachers_per_1000"])
]:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# -------------------- Chart helpers --------------------

def apply_style(fig, height=520):
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Arial, sans-serif", size=12, color=PALETTE["ink"]),
        title=dict(font=dict(size=19, color=PALETTE["navy"]), x=0.02, xanchor="left"),
        paper_bgcolor=PALETTE["white"],
        plot_bgcolor=PALETTE["white"],
        margin=dict(l=65, r=55, t=125, b=90),
        hoverlabel=dict(bgcolor="white", bordercolor=PALETTE["slate"],
                        font=dict(color=PALETTE["ink"], size=12)),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.08,
            xanchor="left", x=0, title_text="",
            tracegroupgap=8
        ),
        hovermode="closest"
    )
    fig.update_xaxes(
        showline=True, linecolor=PALETTE["slate"], linewidth=1,
        showgrid=False, zeroline=False, automargin=True,
        title_font=dict(size=13), tickfont=dict(size=11)
    )
    fig.update_yaxes(
        showline=False, showgrid=True, gridcolor=PALETTE["grid"],
        gridwidth=1, zeroline=False, automargin=True,
        title_font=dict(size=13), tickfont=dict(size=11)
    )
    return fig


def show_chart(fig, key, height=520):
    fig = apply_style(fig, height)

    # Override the global top legend for the quadrant chart only.
    # This is applied AFTER apply_style(), which otherwise resets the legend
    # position and causes it to overlap the chart title at narrow widths.
    if key == "district_priority_quadrant":
        fig.update_layout(
            title=dict(
                font=dict(size=19, color=PALETTE["navy"]),
                x=0.02, xanchor="left",
                y=0.98, yanchor="top"
            ),
            legend=dict(
                orientation="h",
                x=0, xanchor="left",
                y=-0.28, yanchor="top",
                title_text="",
                font=dict(size=10),
                tracegroupgap=6
            ),
            margin=dict(l=70, r=45, t=105, b=155)
        )

    st.plotly_chart(
        fig, use_container_width=True, key=key,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
            "doubleClick": "reset",
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png", "filename": key,
                "height": 900, "width": 1600, "scale": 2
            }
        }
    )


def chart_guide(text):
    """Explain chart axes, labels, colours or categories without changing chart interaction."""
    st.caption(f"**How to read this chart:** {text}")


def insight(text):
    # Markdown is not parsed inside a raw HTML <div>, so convert **bold** to HTML.
    rendered_text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", str(text))
    st.markdown(
        f'<div class="insight"><b>Interpretation</b><br>{rendered_text}</div>',
        unsafe_allow_html=True
    )


def pearson(x, y):
    pair = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 2 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan
    return pair["x"].corr(pair["y"])


def add_linear_trend(fig, df, xcol, ycol):
    pair = df[[xcol, ycol]].dropna()
    if len(pair) >= 2 and pair[xcol].nunique() > 1:
        slope, intercept = np.polyfit(pair[xcol], pair[ycol], 1)
        xs = np.linspace(pair[xcol].min(), pair[xcol].max(), 100)
        fig.add_trace(go.Scatter(
            x=xs, y=slope * xs + intercept, mode="lines",
            line=dict(color=PALETTE["gold"], width=3),
            name="Linear trend", hoverinfo="skip"
        ))
    return fig


def filter_state(df, selected):
    if not selected:
        return df.iloc[0:0].copy()
    return df[df["state"].isin(selected)].copy()


def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


# -------------------- Header and navigation --------------------

st.title("Malaysian District Data Visualisation")
st.markdown(
    "Exploring **household poverty**, **population demographics** and "
    "**teacher availability** across Malaysian districts."
)
st.caption("JIE42403 Scientific Visualisation | By Arissa Aida Kamilia Binti Aris (s23a0149)")

states = sorted(set(poverty["state"].dropna()) & set(population["state"].dropna())
                & set(education["state"].dropna()))
with st.sidebar:
    st.title("Dashboard controls")
    page = st.radio(
        "Select research objective",
        [
            "1. Household Poverty Patterns",
            "2. Population Distribution",
            "3. Poverty & Teacher Availability"
        ]
    )
    selected_states = st.multiselect("Filter by state", states, default=states)
    st.caption("All summary values and charts respond to this state filter.")
    st.divider()
    st.markdown("**Visualisation principles**")
    st.caption("Clarity · Accuracy · Relevance · Aesthetics")

if not selected_states:
    st.warning("Select at least one state from the sidebar to display the analysis.")
    st.stop()


# ============================================================
# OBJECTIVE 1 — POVERTY
# ============================================================

if page == "1. Household Poverty Patterns":
    st.header("Objective 1: Analyse Household Poverty Patterns")
    st.write(
        "Examine absolute and relative poverty rates across Malaysian districts, "
        "compare changes over time and identify districts with higher poverty levels."
    )

    p = filter_state(poverty, selected_states)
    p24 = p[p["year"] == 2024].dropna(
        subset=["poverty_absolute", "poverty_relative"]
    ).copy()

    if p24.empty:
        st.warning("No 2024 poverty records are available for the selected states.")
        st.stop()

    top_record = p24.loc[p24["poverty_absolute"].idxmax()]
    metric_row([
        ("Mean absolute poverty", f"{p24['poverty_absolute'].mean():.2f}%", "2024 selected district records"),
        ("Mean relative poverty", f"{p24['poverty_relative'].mean():.2f}%", "2024 selected district records"),
        ("Districts represented", f"{p24['district'].nunique():,}", "Unique districts in 2024")
    ])
    st.info(
        f"**Key summary:** The highest 2024 absolute poverty rate in the selected "
        f"records is **{top_record['district']}, {top_record['state']} "
        f"({top_record['poverty_absolute']:.1f}%)**."
    )

    st.subheader("1.1 Poverty trend across consistently observed districts")
    years = [2019, 2022, 2024]
    trend = p[p["year"].isin(years)].copy()
    trend["district_key"] = trend["state"].astype(str) + "|" + trend["district"].astype(str)
    counts = trend.groupby("district_key")["year"].nunique()
    common_keys = counts[counts == len(years)].index
    trend = trend[trend["district_key"].isin(common_keys)]
    trend_summary = trend.groupby("year", as_index=False).agg(
        absolute=("poverty_absolute", "mean"),
        relative=("poverty_relative", "mean")
    ).melt(id_vars="year", var_name="Poverty measure", value_name="Mean rate (%)")
    trend_summary["Poverty measure"] = trend_summary["Poverty measure"].map(
        {"absolute": "Absolute poverty", "relative": "Relative poverty"}
    )
    if not trend_summary.empty:
        fig = px.line(
            trend_summary, x="year", y="Mean rate (%)", color="Poverty measure",
            markers=True, color_discrete_map={
                "Absolute poverty": PALETTE["navy"],
                "Relative poverty": PALETTE["gold"]
            }, title="Average poverty rates (same district coverage)"
        )
        fig.update_traces(line_width=3, marker_size=10)
        fig.update_xaxes(tickmode="array", tickvals=years, ticktext=[str(y) for y in years])
        fig.update_layout(yaxis_title="Mean poverty rate (%)", xaxis_title="Year")
        show_chart(fig, "poverty_trend")
        chart_guide('X-axis = year; Y-axis = mean poverty rate (%). The two lines compare absolute and relative poverty across the same consistently observed districts.')
        wide = trend_summary.pivot(index="year", columns="Poverty measure", values="Mean rate (%)")
        if len(wide) >= 2:
            start, end = wide.index.min(), wide.index.max()
            da = wide.loc[end, "Absolute poverty"] - wide.loc[start, "Absolute poverty"]
            dr = wide.loc[end, "Relative poverty"] - wide.loc[start, "Relative poverty"]
            insight(
                f"Among **{len(common_keys)} consistently observed districts**, "
                f"mean absolute poverty changed by **{da:+.2f} percentage points** "
                f"and mean relative poverty by **{dr:+.2f} percentage points** "
                f"from {start} to {end}. Percentage-point changes describe the "
                f"difference between rates, not percentage growth."
            )
    else:
        st.warning("There are not enough matched district records to display the trend.")

    st.subheader("1.2 Ten districts with the highest absolute poverty (2024)")
    top10 = p24.nlargest(10, "poverty_absolute").sort_values("poverty_absolute", ascending=True)
    fig = px.bar(
        top10, x="poverty_absolute", y="district", orientation="h",
        text="poverty_absolute", color_discrete_sequence=[PALETTE["teal"]],
        hover_data={"state": True, "poverty_absolute": ":.1f"},
        title="Highest absolute poverty rates by district"
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%", textposition="outside",
        marker_line_color=PALETTE["navy"], marker_line_width=0.5,
        cliponaxis=False
    )
    fig.update_layout(
        xaxis_title="Absolute poverty rate (%)", yaxis_title="District",
        showlegend=False, margin=dict(l=100, r=90, t=80, b=60)
    )
    fig.update_xaxes(range=[0, max(1, top10["poverty_absolute"].max() * 1.2)])
    show_chart(fig, "top10_poverty", height=560)
    chart_guide('Each horizontal bar represents one district. Longer bars and higher percentages indicate higher absolute poverty; hover over a bar to view its state and exact rate.')
    highest = top10.iloc[-1]
    lowest_top10 = top10.iloc[0]
    insight(
        f"**{highest['district']} ({highest['state']})** has the highest rate "
        f"among the ten districts shown at **{highest['poverty_absolute']:.1f}%**. "
        f"The difference between the first and tenth positions is "
        f"**{highest['poverty_absolute'] - lowest_top10['poverty_absolute']:.1f} "
        f"percentage points**. This ranking identifies where rates are highest; "
        f"it does not explain the causes of poverty."
    )

    st.subheader("1.3 Relationship between absolute and relative poverty (2024)")
    scatter = p24.dropna(subset=["poverty_absolute", "poverty_relative"]).copy()
    fig = px.scatter(
        scatter, x="poverty_absolute", y="poverty_relative",
        hover_name="district", hover_data={"state": True},
        color_discrete_sequence=[PALETTE["teal"]],
        title="District-level relationship between poverty measures"
    )
    fig.update_traces(marker=dict(size=9, opacity=0.7,
                                  line=dict(width=0.6, color=PALETTE["navy"])))
    fig = add_linear_trend(fig, scatter, "poverty_absolute", "poverty_relative")
    fig.update_layout(
        xaxis_title="Absolute poverty rate (%)",
        yaxis_title="Relative poverty rate (%)",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )
    show_chart(fig, "poverty_relationship")
    chart_guide('X-axis = absolute poverty rate (%); Y-axis = relative poverty rate (%). Each dot is a district, and the trend line summarises the overall linear relationship. Hover to identify the district and state.')
    r = pearson(scatter["poverty_absolute"], scatter["poverty_relative"])
    r_text = "not estimable from the selected records" if pd.isna(r) else f"**{r:.2f}**"
    insight(
        f"The Pearson correlation between absolute and relative poverty is {r_text}. "
        f"A positive coefficient indicates that districts with higher absolute "
        f"poverty also tend to have higher relative poverty. Points far from the "
        f"trend line may have different poverty profiles. Correlation does not "
        f"establish causation."
    )


# ============================================================
# OBJECTIVE 2 — POPULATION
# ============================================================

elif page == "2. Population Distribution":
    st.header("Objective 2: Examine Population Distribution and Demographics")
    st.write(
        "Explore reported population changes and the demographic composition "
        "of Malaysian districts by ethnic group and age category."
    )

    pop = filter_state(population, selected_states)
    eth = filter_state(ethnicity, selected_states)
    age_data = filter_state(age, selected_states)

    pop24 = pop[pop["year"] == 2024].dropna(subset=["population"]).copy()
    eth24 = eth[eth["year"] == 2024].dropna(subset=["population"]).copy()
    age24 = age_data[age_data["year"] == 2024].dropna(subset=["population"]).copy()

    if pop24.empty:
        st.warning("No 2024 population records are available for the selected states.")
        st.stop()

    # Population total is taken only from population_total.csv, never by
    # summing age/ethnicity breakdowns (which would risk double counting).
    total_pop = pop24.groupby(["state", "district"], as_index=False)["population"].first()["population"].sum()
    district_count = pop24[["state", "district"]].drop_duplicates().shape[0]
    age_count = age24["age"].nunique() if not age24.empty else 0

    metric_row([
        ("Reported population (2024)", f"{total_pop:,.1f}k", "Sum of unique district totals; source unit is thousands"),
        ("Districts represented", f"{district_count:,}", "Unique state-district pairs"),
        ("Age categories", f"{age_count:,}", "Categories in the 2024 age dataset")
    ])
    st.info(
        "**Data note:** Population values are reported in thousands. "
        "The overall population total is calculated from `population_total.csv`, "
        "not by adding the age or ethnicity breakdowns."
    )

    st.subheader("2.1 Reported population trend (2020–2026)")
    pop_trend = pop.groupby("year", as_index=False).agg(
        population=("population", "sum"),
        districts=("district", "nunique")
    ).dropna(subset=["year"]).sort_values("year")
    fig = px.line(
        pop_trend, x="year", y="population", markers=True,
        color_discrete_sequence=[PALETTE["navy"]],
        title="Total reported population by year"
    )
    fig.update_traces(line_width=3, marker=dict(size=9, color=PALETTE["gold"]))
    fig.update_layout(
        xaxis_title="Year", yaxis_title="Reported population (thousands)",
        xaxis=dict(tickmode="linear", dtick=1)
    )
    show_chart(fig, "population_trend")
    chart_guide('X-axis = year; Y-axis = reported population in thousands. The line shows the sum of district totals available in each year; the number of represented districts may vary.')
    if len(pop_trend) >= 2:
        a, b = pop_trend.iloc[0], pop_trend.iloc[-1]
        pct = ((b["population"] - a["population"]) / a["population"] * 100) if a["population"] else np.nan
        insight(
            f"Reported population changed from **{a['population']:,.1f} thousand** "
            f"({int(a['districts'])} districts) in {int(a['year'])} to "
            f"**{b['population']:,.1f} thousand** ({int(b['districts'])} districts) "
            f"in {int(b['year'])}, a calculated change of **{pct:+.2f}%**. "
            f"District coverage changes across years, so the trend may reflect "
            f"both population change and differences in available records."
        )

    st.subheader("2.2 Population composition by ethnic group (2024)")
    labels = {
        "bumi_malay": "Bumiputera Malay",
        "bumi_other": "Other Bumiputera",
        "chinese": "Chinese",
        "indian": "Indian",
        "other_citizen": "Other Citizens",
        "other_noncitizen": "Other Non-Citizens",
        "overall": "Overall"
    }
    eth_summary = eth24.groupby("ethnicity", as_index=False)["population"].sum()
    eth_summary = eth_summary[eth_summary["ethnicity"] != "overall"].copy()
    eth_summary["Ethnic group"] = eth_summary["ethnicity"].map(labels).fillna(eth_summary["ethnicity"])
    eth_total = eth_summary["population"].sum()
    eth_summary["share"] = eth_summary["population"] / eth_total * 100 if eth_total else 0
    eth_summary = eth_summary.sort_values("share", ascending=True)
    fig = px.bar(
        eth_summary, x="share", y="Ethnic group", orientation="h",
        text="share", color_discrete_sequence=[PALETTE["teal"]],
        custom_data=["population"],
        title="Share of reported population by ethnic group"
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%", textposition="outside",
        marker_line_color=PALETTE["navy"], marker_line_width=0.5,
        cliponaxis=False,
        hovertemplate="%{y}<br>Population: %{customdata[0]:,.1f} thousand"
                      "<br>Share: %{x:.2f}%<extra></extra>"
    )
    fig.update_layout(
        xaxis_title="Share of reported population (%)",
        yaxis_title="Ethnic group", showlegend=False,
        margin=dict(l=150, r=80, t=80, b=60)
    )
    fig.update_xaxes(range=[0, max(1, eth_summary["share"].max() * 1.2)], ticksuffix="%")
    show_chart(fig, "ethnicity_share", height=480)
    chart_guide("Y-axis = ethnic group; X-axis = percentage share of the selected dataset's reported ethnic-group population. Longer bars indicate a larger share; hover to see the population count and percentage.")
    if not eth_summary.empty:
        lead = eth_summary.iloc[-1]
        insight(
            f"**{lead['Ethnic group']}** is the largest category in the selected "
            f"2024 records, representing **{lead['share']:.2f}%** of the summed "
            f"ethnic-group population. These shares describe the coverage of the "
            f"selected dataset and should not be generalised beyond its coverage."
        )

    st.subheader("2.3 Population distribution by age group (2024)")
    age_summary = age24.groupby("age", as_index=False)["population"].sum()
    age_order = [
        "0-4", "5-9", "10-14", "15-19", "20-24", "25-29",
        "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
        "60-64", "65-69", "70-74", "75-79", "80-84", "85+"
    ]
    age_summary["age"] = pd.Categorical(age_summary["age"], categories=age_order, ordered=True)
    age_summary = age_summary.sort_values("age").dropna(subset=["age"])
    fig = px.bar(
        age_summary, x="population", y="age", orientation="h",
        color_discrete_sequence=[PALETTE["navy"]],
        title="Reported population across age groups"
    )
    fig.update_traces(
        marker_line_color=PALETTE["navy"], marker_line_width=0.4,
        hovertemplate="Age group: %{y}<br>Population: %{x:,.1f} thousand<extra></extra>"
    )
    fig.update_layout(
        xaxis_title="Population (thousands)", yaxis_title="Age group",
        showlegend=False, height=650, margin=dict(l=90, r=40, t=80, b=60)
    )
    show_chart(fig, "age_distribution", height=650)
    chart_guide('Y-axis = age category in chronological order; X-axis = reported population in thousands. Longer bars indicate more people in that age group.')
    if not age_summary.empty:
        peak = age_summary.loc[age_summary["population"].idxmax()]
        insight(
            f"The largest age category is **{peak['age']}**, with "
            f"**{peak['population']:,.1f} thousand** people represented in the "
            f"selected records. Keeping age groups in chronological order makes "
            f"the demographic profile easier to interpret."
        )


# ============================================================
# OBJECTIVE 3 — POVERTY AND TEACHER AVAILABILITY
# ============================================================

elif page == "3. Poverty & Teacher Availability":
    st.header("Objective 3: Poverty and Relative Teacher Availability")
    st.write(
        "Identify districts with higher poverty and comparatively lower "
        "teacher availability using the available 2024 integrated district data."
    )

    edu = filter_state(education, selected_states)
    edu24 = edu[edu["year"] == 2024].dropna(
        subset=["poverty_absolute", "teachers_per_1000"]
    ).copy()

    if edu24.empty:
        st.warning("No integrated 2024 records are available for the selected states.")
        st.stop()

    mean_poverty = edu24["poverty_absolute"].mean()
    mean_teacher = edu24["teachers_per_1000"].mean()
    priority_mask = (
        (edu24["poverty_absolute"] > mean_poverty)
        & (edu24["teachers_per_1000"] < mean_teacher)
    )
    priority_n = int(priority_mask.sum())

    metric_row([
        ("Mean absolute poverty", f"{mean_poverty:.2f}%", "Selected integrated district sample"),
        ("Mean teachers per 1,000", f"{mean_teacher:.2f}", "Teachers divided by population in thousands"),
        ("Potential priority districts", f"{priority_n}", "Above-mean poverty and below-mean teacher indicator")
    ])
    st.info(
        f"**Key summary:** {priority_n} of {len(edu24)} selected districts meet the "
        f"relative screening criteria: above-mean absolute poverty and below-mean "
        f"teachers per 1,000 residents."
    )
    st.warning(
        "Interpretation limitation: teachers per 1,000 residents is a population-adjusted "
        "teacher availability indicator. It is not a student–teacher ratio and does not "
        "prove teacher shortages."
    )

    st.subheader("3.1 Association between poverty and teacher availability")
    fig = px.scatter(
        edu24, x="teachers_per_1000", y="poverty_absolute",
        hover_name="district", hover_data={
            "state": True, "population": ":,.1f",
            "total_teachers": ":,.0f" if "total_teachers" in edu24.columns else False
        },
        color_discrete_sequence=[PALETTE["teal"]],
        title="Absolute poverty and teachers per 1,000 residents"
    )
    fig.update_traces(marker=dict(size=9, opacity=0.72,
                                  line=dict(width=0.6, color=PALETTE["navy"])))
    fig = add_linear_trend(fig, edu24, "teachers_per_1000", "poverty_absolute")
    fig.update_layout(
        xaxis_title="Teachers per 1,000 residents",
        yaxis_title="Absolute poverty rate (%)"
    )
    show_chart(fig, "poverty_teacher_association")
    chart_guide('X-axis = teachers per 1,000 residents; Y-axis = absolute poverty rate (%). Each dot is a district. A lower X-value means fewer teachers relative to the resident population, not necessarily a shortage.')
    r = pearson(edu24["teachers_per_1000"], edu24["poverty_absolute"])
    r_text = "not estimable" if pd.isna(r) else f"**{r:.2f}**"
    insight(
        f"The Pearson correlation is {r_text}. This describes the direction and "
        f"strength of the linear association in the selected districts only. "
        f"It does not demonstrate that teacher availability causes poverty or "
        f"that poverty causes teacher availability."
    )

    st.subheader("3.2 Ten districts with the lowest teacher availability indicator")
    lowest = edu24.nsmallest(10, "teachers_per_1000").sort_values("teachers_per_1000", ascending=True)
    fig = px.bar(
        lowest, x="teachers_per_1000", y="district", orientation="h",
        text="teachers_per_1000", hover_data={"state": True},
        color_discrete_sequence=[PALETTE["sage"]],
        title="Lowest teachers per 1,000 residents"
    )
    fig.update_traces(
        texttemplate="%{text:.2f}", textposition="outside",
        marker_line_color=PALETTE["navy"], marker_line_width=0.5,
        cliponaxis=False
    )
    fig.update_layout(
        xaxis_title="Teachers per 1,000 residents", yaxis_title="District",
        showlegend=False, margin=dict(l=110, r=85, t=80, b=60)
    )
    fig.update_xaxes(range=[0, max(1, lowest["teachers_per_1000"].max() * 1.2)])
    show_chart(fig, "lowest_teacher_availability", height=560)
    chart_guide('Each bar represents one of the ten districts with the lowest indicator. X-axis = teachers per 1,000 residents; shorter bars indicate a lower population-adjusted teacher indicator.')
    low = lowest.iloc[0]
    insight(
        f"**{low['district']}, {low['state']}** has the lowest indicator among the "
        f"ten districts shown (**{low['teachers_per_1000']:.2f} teachers per 1,000 "
        f"residents**). This comparison does not account for student enrolment, "
        f"school distribution, teacher workload or subject specialisation."
    )

    st.subheader("3.3 District priority classification (quadrant analysis)")
    quadrant = edu24.copy()
    conditions = [
        (quadrant["poverty_absolute"] > mean_poverty) & (quadrant["teachers_per_1000"] < mean_teacher),
        (quadrant["poverty_absolute"] > mean_poverty) & (quadrant["teachers_per_1000"] >= mean_teacher),
        (quadrant["poverty_absolute"] <= mean_poverty) & (quadrant["teachers_per_1000"] < mean_teacher),
        (quadrant["poverty_absolute"] <= mean_poverty) & (quadrant["teachers_per_1000"] >= mean_teacher)
    ]
    groups = [
        "Higher poverty · Lower teacher indicator",
        "Higher poverty · Higher teacher indicator",
        "Lower poverty · Lower teacher indicator",
        "Lower poverty · Higher teacher indicator"
    ]
    quadrant["Priority group"] = np.select(conditions, groups, default="Unclassified")
    group_colors = {
        groups[0]: PALETTE["coral"],
        groups[1]: PALETTE["gold"],
        groups[2]: PALETTE["slate"],
        groups[3]: PALETTE["sage"]
    }
    fig = px.scatter(
        quadrant, x="teachers_per_1000", y="poverty_absolute",
        color="Priority group", color_discrete_map=group_colors,
        hover_name="district", hover_data={"state": True},
        title="Districts classified by sample mean thresholds"
    )
    fig.update_traces(marker=dict(size=10, opacity=0.82,
                                  line=dict(width=0.6, color="white")))
    fig.add_hline(y=mean_poverty, line_dash="dash", line_color=PALETTE["navy"],
                  annotation_text=f"Mean poverty: {mean_poverty:.1f}%")
    fig.add_vline(x=mean_teacher, line_dash="dash", line_color=PALETTE["navy"],
                  annotation_text=f"Mean teacher indicator: {mean_teacher:.1f}")
    fig.update_layout(
        xaxis_title="Teachers per 1,000 residents",
        yaxis_title="Absolute poverty rate (%)",
        legend=dict(
            orientation="h", y=-0.30, x=0, xanchor="left",
            yanchor="top", tracegroupgap=8
        ),
        margin=dict(l=70, r=45, t=125, b=165)
    )
    show_chart(fig, "district_priority_quadrant", height=600)
    chart_guide('X-axis = teachers per 1,000 residents; Y-axis = absolute poverty rate (%). Dashed lines mark the selected-sample means. Each colour identifies a quadrant: higher/lower poverty crossed with higher/lower teacher indicator. Hover to identify a district and state.')

    counts = quadrant["Priority group"].value_counts()
    st.markdown("**District counts by quadrant**")
    count_cols = st.columns(4)
    for col, group in zip(count_cols, groups):
        col.metric(group, int(counts.get(group, 0)))
    insight(
        f"The higher-poverty/lower-teacher-indicator quadrant contains **{priority_n} "
        f"districts**. The cut-offs are the means of the selected district sample; "
        f"changing the state filter changes the thresholds and may change district "
        f"classification. These are screening categories for further investigation, "
        f"not evidence of an actual teacher shortage."
    )

    st.subheader("Potential priority districts")
    priority_table = quadrant[quadrant["Priority group"] == groups[0]][
        ["state", "district", "poverty_absolute", "teachers_per_1000"]
    ].sort_values("poverty_absolute", ascending=False).rename(columns={
        "state": "State",
        "district": "District",
        "poverty_absolute": "Absolute poverty (%)",
        "teachers_per_1000": "Teachers per 1,000 residents"
    })
    st.dataframe(priority_table, use_container_width=True, hide_index=True)


# -------------------- Footer and methodological notes --------------------

st.divider()
with st.expander("Data definitions and methodological limitations"):
    st.markdown("""
    - **Poverty:** Absolute and relative poverty rates are percentages from the processed poverty dataset.
    - **Population:** Population values are expressed in thousands ('000). Population totals are taken from the dedicated total-population dataset.
    - **Population coverage:** The number of districts represented can vary by year; interpret year-to-year totals with this limitation in mind.
    - **Teacher indicator:** Teachers per 1,000 residents is calculated as total teachers divided by population measured in thousands. It is not a student–teacher ratio.
    - **Integrated analysis:** The education analysis uses the available matched 2024 district records. Poverty/population and teacher reporting dates may not be identical.
    - **Quadrant thresholds:** Mean values are descriptive thresholds within the currently selected district sample, not official policy thresholds.
    - **Association:** Correlation and trend lines describe association, not causation.
    """)

st.caption("JIE42403 Scientific Visualisation Assignment | Malaysian District Data")
