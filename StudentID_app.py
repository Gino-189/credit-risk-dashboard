import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

st.set_page_config(layout="wide")

# LOADING DATA
@st.cache_data
def load_data():
    data = fetch_openml(name="credit-g", version=1, as_frame=True)
    return data.frame

df = load_data()

# PREPROCESSING
df_clean = df.replace("unknown", np.nan).drop_duplicates()

df_clean["credit_amount"] = pd.to_numeric(df_clean["credit_amount"])
df_clean["duration"] = pd.to_numeric(df_clean["duration"])

# OUTLIERS (IQR)
Q1 = df_clean["credit_amount"].quantile(0.25)
Q3 = df_clean["credit_amount"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean["credit_amount_capped"] = df_clean["credit_amount"].clip(lower, upper)

# FEATURE ENGINEERING
df_clean["age_group"] = pd.cut(
    df_clean["age"],
    bins=[18, 30, 50, 100],
    labels=["Young", "Adult", "Senior"]
)

# REGEX FEATURE
df_clean["purpose_group"] = df_clean["purpose"].str.replace(
    r"(new car|used car|radio/tv|furniture/equipment|domestic appliance)",
    "consumer_goods",
    regex=True
)

# SIDEBAR FILTERS
st.sidebar.title("Filters")

duration_range = st.sidebar.slider(
    "Duration",
    int(df_clean["duration"].min()),
    int(df_clean["duration"].max()),
    (6, 36)
)

credit_range = st.sidebar.slider(
    "Credit Amount",
    int(df_clean["credit_amount"].min()),
    int(df_clean["credit_amount"].max()),
    (1000, 10000)
)

age_range = st.sidebar.slider(
    "Age",
    int(df_clean["age"].min()),
    int(df_clean["age"].max()),
    (25, 60)
)

purpose_list = st.sidebar.multiselect(
    "Purpose",
    options=df_clean["purpose"].unique(),
    default=df_clean["purpose"].unique()[:3]
)

# REGEX FILTER
purpose_group_filter = st.sidebar.multiselect(
    "Purpose Group (Regex)",
    options=df_clean["purpose_group"].unique(),
    default=df_clean["purpose_group"].unique()
)

use_capped = st.sidebar.checkbox("Use Capped Data (Outlier Control)", value=True)

# APPLYING FILTERS
df_clean["credit_used"] = np.where(
    use_capped,
    df_clean["credit_amount_capped"],
    df_clean["credit_amount"]
)

filtered_df = df_clean[
    (df_clean["duration"].between(*duration_range)) &
    (df_clean["credit_used"].between(*credit_range)) &
    (df_clean["age"].between(*age_range)) &
    (df_clean["purpose"].isin(purpose_list)) &
    (df_clean["purpose_group"].isin(purpose_group_filter))
]

# DRILL-DOWN
housing_filter = st.sidebar.selectbox(
    "Housing (Drill-down)",
    options=["All"] + list(df_clean["housing"].unique())
)

if housing_filter != "All":
    drill_df = filtered_df[filtered_df["housing"] == housing_filter]
else:
    drill_df = filtered_df.copy()

# DASHBOARD TITLE 
st.markdown(
    "<h1 style='text-align: center;'>Credit Risk Analysis Dashboard</h1>",
    unsafe_allow_html=True
)

# KPIs
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", len(filtered_df))

col2.metric(
    "Average Credit (€)",
    round(filtered_df["credit_used"].mean(), 2) if len(filtered_df) > 0 else 0
)

col3.metric(
    "Bad Credit (%)",
    round((filtered_df["class"] == "bad").mean() * 100, 2) if len(filtered_df) > 0 else 0
)

col4.metric(
    "Top Loan Category",
    filtered_df["purpose_group"].mode()[0] if len(filtered_df) > 0 else "N/A"
)

st.markdown(
    "Filters dynamically adjust the dataset, enabling focused analysis of specific customer segments."
)

# CHARTS
plt.style.use("ggplot")

col1, col2 = st.columns(2)

# Credit Distribution
with col1:
    fig, ax = plt.subplots()
    ax.hist(filtered_df["credit_used"], bins=20, color="orange", edgecolor="black")
    ax.set_title("Distribution of Credit Amounts (Right-Skew Indicates Few High Loans)")
    ax.set_xlabel("Credit Amount")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# Class Distribution
with col2:
    class_counts = filtered_df["class"].value_counts()
    fig, ax = plt.subplots()
    bars = ax.bar(class_counts.index, class_counts.values, color=["green", "red"])

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, int(height),
                ha='center', va='bottom')

    ax.set_title("Proportion of Good vs Bad Credit Customers")
    st.pyplot(fig)

# Purpose Group vs Credit
fig, ax = plt.subplots()
avg_credit = filtered_df.groupby("purpose_group")["credit_used"].mean()

bars = ax.bar(avg_credit.index, avg_credit.values, color="skyblue")

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height, f"{height:.0f}",
            ha='center', va='bottom')

plt.xticks(rotation=45)
ax.set_title("Average Credit by Loan Category (Consumer Goods Dominate)")
st.pyplot(fig)

# Age Distribution
fig, ax = plt.subplots()
ax.hist(filtered_df["age"], bins=20, color="purple", edgecolor="black")
ax.set_title("Age Distribution of Borrowers (Majority in Working Age)")
st.pyplot(fig)

# DRILL-DOWN TABLE
st.subheader("Detailed Segment Analysis (Drill-down View)")

st.dataframe(
    drill_df[["purpose", "purpose_group", "credit_used", "class", "housing"]].head(50)
)

# LSEPI FOOTER 
st.markdown("---")

st.markdown("### LSEPI Considerations")

st.markdown("""
**Ethical Consideration:**  
Credit risk classification inherently carries the risk of reinforcing bias, particularly when customers are labelled as "good" or "bad". Such classifications can oversimplify complex financial behaviour and may disadvantage certain groups if interpreted without context.  
**Mitigation:** This dashboard is designed strictly for exploratory analysis and explicitly avoids automated decision-making. Clear explanations are provided to emphasise that patterns observed are descriptive, not predictive, reducing the risk of misuse.

**Professional Consideration:**  
Misinterpretation of statistical outputs—such as relying solely on mean values in skewed distributions—can lead to flawed conclusions and poor business decisions. Additionally, users may overlook how filtering choices influence results.  
**Mitigation:** The dashboard incorporates multiple statistical views (mean, distribution, grouped analysis) and uses descriptive chart titles to guide interpretation. It also highlights that results are sensitive to filters, encouraging responsible and informed analysis.
""")