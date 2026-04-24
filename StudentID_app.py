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
    df = data.frame
    return df

df = load_data()

# PREPROCESSING
df_clean = df.replace("unknown", np.nan)
df_clean = df_clean.drop_duplicates()

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

df_clean["purpose_group"] = df_clean["purpose"].str.replace(
    r"(new car|used car|radio/tv|furniture/equipment|domestic appliance)",
    "consumer_goods",
    regex=True
)

# SIDEBAR (FILTERS)
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

use_capped = st.sidebar.checkbox("Use Capped Data", value=True)

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
    (df_clean["purpose"].isin(purpose_list))
]

# Drill-down
housing_filter = st.sidebar.selectbox(
    "Housing (Drill-down)",
    options=["All"] + list(df_clean["housing"].unique())
)

if housing_filter != "All":
    drill_df = filtered_df[filtered_df["housing"] == housing_filter]
else:
    drill_df = filtered_df.copy()

# MAIN DASHBOARD
st.title("Credit Risk Dashboard")

# KPIs
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(filtered_df))
col2.metric("Avg Credit", round(filtered_df["credit_used"].mean(), 2))
col3.metric("Bad Credit %", round((filtered_df["class"]=="bad").mean()*100, 2))

st.markdown("Filtering reduces dataset size and allows focused segment analysis.")

# CHARTS
plt.style.use("ggplot")

col1, col2 = st.columns(2)

# Credit Distribution
with col1:
    fig, ax = plt.subplots()
    ax.hist(filtered_df["credit_used"], bins=20, color="orange", edgecolor="black")
    ax.set_title("Credit Distribution")
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

    ax.set_title("Class Distribution")
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
ax.set_title("Avg Credit by Purpose Group")
st.pyplot(fig)

# Age Distribution
fig, ax = plt.subplots()
ax.hist(filtered_df["age"], bins=20, color="purple", edgecolor="black")
ax.set_title("Age Distribution")
st.pyplot(fig)

# DRILL-DOWN TABLE
st.subheader("Drill-down Data")
st.dataframe(drill_df.head(50))

# LSEPI FOOTER
st.markdown("---")

st.markdown("### LSEPI Considerations")

st.markdown("""
**Ethical:**  
There is a risk of biased interpretation of credit data, particularly when grouping customers into 'good' or 'bad'.  
Mitigation: The dashboard avoids automated decisions and clearly states that results are descriptive only.

**Professional:**  
Misinterpretation of statistical measures such as mean vs median may lead to incorrect conclusions.  
Mitigation: The dashboard includes explanations and uses both mean and median where appropriate.
""")