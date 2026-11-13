import os

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="European Bank Churn Analytics", page_icon="🏦", layout="wide")

DATA_PATH = "European_Bank.csv"


# ---------------------------------------------------------------- data
@st.cache_data
def load_data(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    df["Churn"] = df["Exited"].map({0: "Retained", 1: "Churned"})
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0, 29, 39, 49, 59, 120],
        labels=["<30", "30-39", "40-49", "50-59", "60+"],
    )
    df["CreditBand"] = pd.cut(
        df["CreditScore"], bins=[0, 579, 669, 739, 799, 900],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"],
    )
    df["ZeroBalance"] = (df["Balance"] == 0).map({True: "Zero balance", False: "Has balance"})
    return df


def churn_rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = (
        df.groupby(col, observed=True)["Exited"]
        .agg(Customers="count", Churned="sum", ChurnRate="mean")
        .reset_index()
    )
    out["ChurnRate"] = out["ChurnRate"] * 100
    return out


def rate_bar(df: pd.DataFrame, col: str, title: str):
    d = churn_rate_by(df, col)
    fig = px.bar(
        d, x=col, y="ChurnRate", text=d["ChurnRate"].round(1).astype(str) + "%",
        hover_data=["Customers", "Churned"], title=title,
        labels={"ChurnRate": "Churn rate (%)"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_range=[0, max(d["ChurnRate"].max() * 1.2, 5)])
    return fig


# ---------------------------------------------------------------- load
st.title("🏦 European Bank: Customer Segmentation & Churn Analytics")

if os.path.exists(DATA_PATH):
    raw = load_data(DATA_PATH)
else:
    up = st.file_uploader("Upload European_Bank.csv", type="csv")
    if up is None:
        st.info("Place `European_Bank.csv` next to `app.py` or upload it above.")
        st.stop()
    raw = load_data(up)

# ---------------------------------------------------------------- sidebar filters
st.sidebar.header("Filters")
geo = st.sidebar.multiselect("Geography", sorted(raw["Geography"].unique()), default=sorted(raw["Geography"].unique()))
gender = st.sidebar.multiselect("Gender", sorted(raw["Gender"].unique()), default=sorted(raw["Gender"].unique()))
age_min, age_max = int(raw["Age"].min()), int(raw["Age"].max())
age = st.sidebar.slider("Age", age_min, age_max, (age_min, age_max))
cs_min, cs_max = int(raw["CreditScore"].min()), int(raw["CreditScore"].max())
credit = st.sidebar.slider("Credit score", cs_min, cs_max, (cs_min, cs_max))
products = st.sidebar.multiselect(
    "Number of products", sorted(raw["NumOfProducts"].unique()), default=sorted(raw["NumOfProducts"].unique())
)
active = st.sidebar.selectbox("Member status", ["All", "Active", "Inactive"])

df = raw[
    raw["Geography"].isin(geo)
    & raw["Gender"].isin(gender)
    & raw["Age"].between(*age)
    & raw["CreditScore"].between(*credit)
    & raw["NumOfProducts"].isin(products)
]
if active != "All":
    df = df[df["IsActiveMember"] == (1 if active == "Active" else 0)]

if df.empty:
    st.warning("No customers match the current filters.")
    st.stop()

# ---------------------------------------------------------------- KPIs
overall = raw["Exited"].mean() * 100
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Customers", f"{len(df):,}")
c2.metric("Churned", f"{int(df['Exited'].sum()):,}")
c3.metric("Churn rate", f"{df['Exited'].mean() * 100:.1f}%", f"{df['Exited'].mean() * 100 - overall:+.1f} pts vs overall")
c4.metric("Avg balance", f"€{df['Balance'].mean():,.0f}")
c5.metric("Balance at risk", f"€{df.loc[df['Exited'] == 1, 'Balance'].sum() / 1e6:,.1f}M")

tab_over, tab_demo, tab_fin, tab_seg, tab_data = st.tabs(
    ["Overview", "Demographics", "Financial profile", "Segmentation", "Data"]
)

# ---------------------------------------------------------------- overview
with tab_over:
    a, b = st.columns(2)
    a.plotly_chart(rate_bar(df, "Geography", "Churn rate by geography"), use_container_width=True)
    b.plotly_chart(rate_bar(df, "NumOfProducts", "Churn rate by number of products"), use_container_width=True)

    a, b = st.columns(2)
    d = df.assign(Member=df["IsActiveMember"].map({1: "Active", 0: "Inactive"}))
    a.plotly_chart(rate_bar(d, "Member", "Churn rate by member status"), use_container_width=True)
    d = df.assign(Card=df["HasCrCard"].map({1: "Has card", 0: "No card"}))
    b.plotly_chart(rate_bar(d, "Card", "Churn rate by credit card ownership"), use_container_width=True)

    st.subheader("Churn heatmap: Geography × Age group")
    heat = df.pivot_table(index="Geography", columns="AgeGroup", values="Exited", aggfunc="mean", observed=True) * 100
    fig = px.imshow(heat.round(1), text_auto=True, color_continuous_scale="Reds", aspect="auto",
                    labels=dict(color="Churn %"))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- demographics
with tab_demo:
    a, b = st.columns(2)
    a.plotly_chart(rate_bar(df, "AgeGroup", "Churn rate by age group"), use_container_width=True)
    b.plotly_chart(rate_bar(df, "Gender", "Churn rate by gender"), use_container_width=True)

    a, b = st.columns(2)
    fig = px.histogram(df, x="Age", color="Churn", barmode="overlay", nbins=40, opacity=0.65,
                       title="Age distribution: churned vs retained",
                       color_discrete_map={"Churned": "#d62728", "Retained": "#1f77b4"})
    a.plotly_chart(fig, use_container_width=True)
    b.plotly_chart(rate_bar(df, "Tenure", "Churn rate by tenure (years)"), use_container_width=True)

# ---------------------------------------------------------------- financial
with tab_fin:
    a, b = st.columns(2)
    a.plotly_chart(rate_bar(df, "CreditBand", "Churn rate by credit score band"), use_container_width=True)
    b.plotly_chart(rate_bar(df, "ZeroBalance", "Churn rate: zero vs positive balance"), use_container_width=True)

    a, b = st.columns(2)
    fig = px.box(df, x="Churn", y="Balance", color="Churn", title="Balance by churn status",
                 color_discrete_map={"Churned": "#d62728", "Retained": "#1f77b4"})
    a.plotly_chart(fig, use_container_width=True)
    fig = px.box(df, x="Churn", y="EstimatedSalary", color="Churn", title="Estimated salary by churn status",
                 color_discrete_map={"Churned": "#d62728", "Retained": "#1f77b4"})
    b.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation with churn")
    num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
                "IsActiveMember", "EstimatedSalary", "Exited"]
    corr = df[num_cols].corr()["Exited"].drop("Exited").sort_values()
    fig = px.bar(corr, orientation="h", labels={"value": "Correlation with Exited", "index": ""})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- segmentation
with tab_seg:
    st.caption("KMeans on standardized customer attributes (runs on the filtered data).")
    features = st.multiselect(
        "Features to cluster on",
        ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary"],
        default=["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "IsActiveMember"],
    )
    k = st.slider("Number of segments (k)", 2, 8, 4)

    if len(features) < 2 or len(df) < k:
        st.warning("Select at least 2 features and ensure enough customers for k segments.")
    else:
        X = StandardScaler().fit_transform(df[features])
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        seg = df.assign(Segment=("Segment " + pd.Series(labels + 1, index=df.index).astype(str)))

        summary = (
            seg.groupby("Segment")
            .agg(Customers=("CustomerId", "count"), ChurnRate=("Exited", "mean"),
                 AvgAge=("Age", "mean"), AvgCredit=("CreditScore", "mean"),
                 AvgBalance=("Balance", "mean"), AvgSalary=("EstimatedSalary", "mean"),
                 AvgProducts=("NumOfProducts", "mean"), ActiveShare=("IsActiveMember", "mean"))
            .reset_index()
        )
        summary["ChurnRate"] *= 100
        summary["ActiveShare"] *= 100
        summary = summary.sort_values("ChurnRate", ascending=False)

        st.dataframe(
            summary.style.format({
                "ChurnRate": "{:.1f}%", "AvgAge": "{:.1f}", "AvgCredit": "{:.0f}",
                "AvgBalance": "€{:,.0f}", "AvgSalary": "€{:,.0f}",
                "AvgProducts": "{:.2f}", "ActiveShare": "{:.0f}%",
            }).background_gradient(subset=["ChurnRate"], cmap="Reds"),
            use_container_width=True, hide_index=True,
        )

        a, b = st.columns(2)
        fig = px.bar(summary, x="Segment", y="ChurnRate", title="Churn rate by segment",
                     labels={"ChurnRate": "Churn rate (%)"})
        a.plotly_chart(fig, use_container_width=True)
        fig = px.scatter(seg.sample(min(len(seg), 3000), random_state=1), x="Age", y="Balance",
                         color="Segment", symbol="Churn", opacity=0.6, title="Segments: Age vs Balance")
        b.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- data
with tab_data:
    st.dataframe(df.drop(columns=["Churn", "AgeGroup", "CreditBand", "ZeroBalance"]), use_container_width=True)
    st.download_button(
        "Download filtered data (CSV)",
        df.drop(columns=["Churn", "AgeGroup", "CreditBand", "ZeroBalance"]).to_csv(index=False).encode("utf-8"),
        file_name="european_bank_filtered.csv",
        mime="text/csv",
    )
