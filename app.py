import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="FraudShield | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS - HYBRID THEME
# --------------------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F4F7FC;
    color: #172033;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #101B35;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

/* Sidebar radio buttons */
[data-testid="stSidebar"] .stRadio label {
    color: #E5ECF8;
}

/* Headings */
h1, h2, h3 {
    color: #172033;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E4EAF3;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(20, 40, 80, 0.05);
}

/* Form containers */
[data-testid="stForm"] {
    background-color: #FFFFFF;
    padding: 22px;
    border: 1px solid #E4EAF3;
    border-radius: 12px;
}

/* Buttons */
.stButton > button {
    background-color: #087E8B;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #066570;
    color: white;
    border: none;
}

/* Info boxes */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATA AND MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    preprocessor = joblib.load("models/preprocessor.pkl")

    dt_model = joblib.load("models/decision_tree.pkl")

    rf_model = joblib.load("models/random_forest.pkl")

    kmeans = joblib.load("models/k_means.pkl")

    processed_data = joblib.load("models/processed_data.pkl")

    return (
        preprocessor,
        dt_model,
        rf_model,
        kmeans,
        processed_data
    )


@st.cache_data
def load_data():

    df = pd.read_csv("data/fraud_data_engineered.csv")

    return df


try:

    (
        preprocessor,
        dt_model,
        rf_model,
        kmeans,
        processed_data
    ) = load_models()

    df = load_data()

    X_train, X_test, y_train, y_test = processed_data

except FileNotFoundError as error:

    st.error(f"Required project file not found: {error}")

    st.info(
        "Make sure you have run preprocessing.py and train_model.py "
        "and that the required files are saved in the correct folders."
    )

    st.stop()


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def create_chart_layout(fig, height=350):

    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            color="#172033",
            family="Arial"
        ),
        margin=dict(
            l=20,
            r=20,
            t=45,
            b=20
        )
    )

    return fig


def get_model_comparison():

    models = {
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }

    results = []

    for name, model in models.items():

        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            )
        })

    return pd.DataFrame(results)


def get_cluster_analysis():

    clusters = kmeans.predict(X_train)

    cluster_data = pd.DataFrame({
        "Cluster": clusters,
        "Actual Class": np.array(y_train)
    })

    cluster_summary = cluster_data.groupby("Cluster").agg(
        Total_Transactions=("Actual Class", "count"),
        Fraud_Cases=("Actual Class", "sum")
    ).reset_index()

    cluster_summary["Fraud_Percentage"] = (
        cluster_summary["Fraud_Cases"]
        / cluster_summary["Total_Transactions"]
    ) * 100

    return cluster_data, cluster_summary


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("# 🛡️ FraudShield")

    st.caption("ONLINE TRANSACTION SECURITY")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Fraud Detection",
            "K-Means Analysis"
        ]
    )

    st.divider()

    st.markdown("### Project Information")

    st.caption("Machine Learning Project")

    st.caption("Algorithms")

    st.markdown("""
    - Decision Tree
    - Random Forest
    - K-Means
    """)

    st.divider()

    st.caption("FraudShield | 2026")


# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    st.title("🛡️ FraudShield Dashboard")

    st.write(
        "Monitor online shopping transactions and explore "
        "fraud patterns using machine learning."
    )

    st.divider()

    # KPI VALUES

    total_transactions = len(df)

    fraud_cases = int(df["class"].sum())

    normal_cases = total_transactions - fraud_cases

    fraud_percentage = (fraud_cases / total_transactions) * 100

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Fraud Cases",
        f"{fraud_cases:,}"
    )

    col3.metric(
        "Normal Transactions",
        f"{normal_cases:,}"
    )

    col4.metric(
        "Fraud Rate",
        f"{fraud_percentage:.2f}%"
    )

    st.divider()

    # TRANSACTION DISTRIBUTION

    st.subheader("Transaction Distribution")

    col1, col2 = st.columns(2)

    with col1:

        distribution = df["class"].value_counts().sort_index()

        pie_data = pd.DataFrame({
            "Transaction Type": [
                "Normal",
                "Fraud"
            ],
            "Count": [
                distribution.get(0, 0),
                distribution.get(1, 0)
            ]
        })

        fig = px.pie(
            pie_data,
            names="Transaction Type",
            values="Count",
            hole=0.60,
            color="Transaction Type",
            color_discrete_map={
                "Normal": "#087E8B",
                "Fraud": "#F05D5E"
            },
            title="Normal vs Fraud Transactions"
        )

        fig.update_traces(
            textinfo="percent+label",
            textposition="inside"
        )

        st.plotly_chart(
            create_chart_layout(fig),
            use_container_width=True
        )

   
    with col2:

        source_data = (
            df.groupby("source")["class"]
            .mean()
            .reset_index()
        )

        # Create Fraud Rate column
        source_data["Fraud Rate"] = source_data["class"] * 100

        fig = px.bar(
            source_data,
            x="source",
            y="Fraud Rate",
            color="Fraud Rate",
            color_continuous_scale="Teal",
            title="Fraud Rate by Traffic Source",
            labels={
                "source": "Traffic Source",
                "Fraud Rate": "Fraud Rate (%)"
            },
            text_auto=".2f"
        )

        st.plotly_chart(
            create_chart_layout(fig),
            use_container_width=True
        )

    st.divider()

    # COUNTRY FRAUD ANALYSIS

    st.subheader("Geographical Fraud Analysis")

    country_data = (
        df.groupby("country")
        .agg(
            Transactions=("class", "count"),
            Fraud_Cases=("class", "sum")
        )
        .reset_index()
    )

    country_data["Fraud Rate"] = (
        country_data["Fraud_Cases"]
        / country_data["Transactions"]
    ) * 100

    country_data = country_data[
        country_data["country"] != "Unknown"
    ]

    col1, col2 = st.columns([1.5, 1])

    with col1:

        map_data = country_data[
            country_data["Transactions"] >= 100
        ]

        if len(map_data) > 0:

            fig = px.choropleth(
                map_data,
                locations="country",
                locationmode="country names",
                color="Fraud Rate",
                hover_name="country",
                hover_data=[
                    "Transactions",
                    "Fraud_Cases"
                ],
                color_continuous_scale="Teal",
                title="Fraud Rate Across Countries",
                labels={
                    "Fraud Rate": "Fraud Rate (%)"
                }
            )

            st.plotly_chart(
                create_chart_layout(fig, 450),
                use_container_width=True
            )

        else:

            st.info("Country map data is not available.")

    with col2:

        top_countries = country_data.sort_values(
            "Fraud Rate",
            ascending=False
        ).head(10)

        fig = px.bar(
            top_countries.sort_values("Fraud Rate"),
            x="Fraud Rate",
            y="country",
            orientation="h",
            color="Fraud Rate",
            color_continuous_scale="Teal",
            title="Top 10 Countries by Fraud Rate",
            labels={
                "country": "Country",
                "Fraud Rate": "Fraud Rate (%)"
            },
            text_auto=".2f"
        )

        st.plotly_chart(
            create_chart_layout(fig, 450),
            use_container_width=True
        )

    st.divider()

    # MODEL COMPARISON

    st.subheader("Machine Learning Model Comparison")

    comparison = get_model_comparison()

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    metric_choice = st.selectbox(
        "Select a metric to compare",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    )

    fig = px.bar(
        comparison,
        x="Model",
        y=metric_choice,
        color="Model",
        barmode="group",
        text=metric_choice,
        title=f"{metric_choice} Comparison",
        color_discrete_sequence=[
            "#087E8B",
            "#7357C8"
        ]
    )

    fig.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside"
    )

    fig.update_yaxes(
        tickformat=".0%",
        range=[0, 1.05]
    )

    st.plotly_chart(
        create_chart_layout(fig),
        use_container_width=True
    )


# ==================================================
# PAGE 2: FRAUD DETECTION
# ==================================================

elif page == "Fraud Detection":

    st.title("🔍 Transaction Fraud Scanner")

    st.write(
        "Enter transaction details below to check whether "
        "a transaction appears suspicious."
    )

    st.divider()

    # DATASET VALUES FOR DROPDOWNS

    sources = sorted(df["source"].dropna().unique().tolist())

    browsers = sorted(df["browser"].dropna().unique().tolist())

    countries = sorted(df["country"].dropna().unique().tolist())

    # TRANSACTION FORM

    st.subheader("Transaction Information")

    with st.form("fraud_detection_form"):

        col1, col2 = st.columns(2)

        with col1:

            purchase_value = st.number_input(
                "Purchase Value",
                min_value=0.0,
                value=50.0,
                step=1.0
            )

            source = st.selectbox(
                "Traffic Source",
                sources
            )

            browser = st.selectbox(
                "Browser",
                browsers
            )

            sex = st.selectbox(
                "Sex",
                sorted(df["sex"].dropna().unique().tolist())
            )

        with col2:

            age = st.number_input(
                "Customer Age",
                min_value=1,
                max_value=100,
                value=25
            )

            account_age = st.number_input(
                "Account Age (Hours)",
                min_value=0.0,
                value=100.0,
                step=1.0
            )

            country = st.selectbox(
                "Country",
                countries
            )

            purchase_hour = st.slider(
                "Purchase Hour",
                min_value=0,
                max_value=23,
                value=12
            )

        selected_model = st.selectbox(
            "Select Machine Learning Model",
            [
                "Random Forest",
                "Decision Tree"
            ]
        )

        submit = st.form_submit_button(
            "Analyze Transaction",
            use_container_width=True
        )

    # PREDICTION

    if submit:

        input_data = pd.DataFrame([{
            "purchase_value": purchase_value,
            "source": source,
            "browser": browser,
            "sex": sex,
            "age": age,
            "account_age": account_age,
            "country": country,
            "purchase_hour": purchase_hour
        }])

        # Apply saved preprocessing

        processed_input = preprocessor.transform(input_data)

        # Select model

        if selected_model == "Random Forest":

            model = rf_model

        else:

            model = dt_model

        prediction = model.predict(processed_input)[0]

        probability = model.predict_proba(
            processed_input
        )[0][1]

        fraud_probability = probability * 100

        st.divider()

        st.subheader("Transaction Analysis Result")

        col1, col2 = st.columns([1, 1])

        with col1:

            if prediction == 1:

                st.error("⚠️ Suspicious Transaction Detected")

                st.write(
                    "The selected model classified this transaction "
                    "as potentially fraudulent."
                )

            else:

                st.success("✅ Transaction Classified as Normal")

                st.write(
                    "The selected model classified this transaction "
                    "as normal."
                )

            st.metric(
                "Fraud Probability",
                f"{fraud_probability:.2f}%"
            )

            st.caption(
                "This is the model's estimated probability, "
                "not a guarantee that fraud has or has not occurred."
            )

        with col2:

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=fraud_probability,
                    number={
                        "suffix": "%",
                        "font": {
                            "size": 35,
                            "color": "#172033"
                        }
                    },
                    title={
                        "text": "Fraud Probability",
                        "font": {
                            "size": 18,
                            "color": "#172033"
                        }
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "bar": {
                            "color": "#7357C8"
                        },
                        "steps": [
                            {
                                "range": [0, 30],
                                "color": "#D7F2E3"
                            },
                            {
                                "range": [30, 60],
                                "color": "#FFF0C2"
                            },
                            {
                                "range": [60, 100],
                                "color": "#FAD6D6"
                            }
                        ]
                    }
                )
            )

            st.plotly_chart(
                create_chart_layout(fig, 300),
                use_container_width=True
            )

        st.info(
            f"Model used: {selected_model} | "
            "Classification uses the model's default prediction threshold."
        )


# ==================================================
# PAGE 3: K-MEANS ANALYSIS
# ==================================================

elif page == "K-Means Analysis":

    st.title("📊 K-Means Fraud Pattern Analysis")

    st.write(
        "Explore transaction clusters and compare the "
        "fraud distribution found in each cluster."
    )

    st.divider()

    # CLUSTER ANALYSIS

    cluster_data, cluster_summary = get_cluster_analysis()

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            cluster_summary,
            names="Cluster",
            values="Total_Transactions",
            hole=0.60,
            color="Cluster",
            title="Transaction Distribution by Cluster",
            color_discrete_sequence=[
                "#087E8B",
                "#7357C8"
            ]
        )

        fig.update_traces(
            textinfo="percent+label"
        )

        st.plotly_chart(
            create_chart_layout(fig),
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            cluster_summary,
            x="Cluster",
            y="Fraud_Percentage",
            color="Cluster",
            title="Fraud Percentage in Each Cluster",
            labels={
                "Fraud_Percentage": "Fraud Percentage (%)",
                "Cluster": "Cluster"
            },
            text_auto=".2f",
            color_discrete_sequence=[
                "#087E8B",
                "#7357C8"
            ]
        )

        st.plotly_chart(
            create_chart_layout(fig),
            use_container_width=True
        )

    st.divider()

    st.subheader("Cluster Statistics")

    display_summary = cluster_summary.rename(
        columns={
            "Total_Transactions": "Total Transactions",
            "Fraud_Cases": "Fraud Cases",
            "Fraud_Percentage": "Fraud Percentage"
        }
    )

    st.dataframe(
        display_summary.style.format({
            "Fraud Percentage": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "K-Means is an unsupervised algorithm. It creates clusters "
        "based on transaction features. Actual class labels are used "
        "here only to analyze the fraud percentage in each cluster."
    )

    st.divider()

    # NEW TRANSACTION CLUSTER

    st.subheader("Assign a Transaction to a Cluster")

    st.write(
        "Enter transaction details to find which existing "
        "K-Means cluster the transaction belongs to."
    )

    sources = sorted(df["source"].dropna().unique().tolist())

    browsers = sorted(df["browser"].dropna().unique().tolist())

    countries = sorted(df["country"].dropna().unique().tolist())

    with st.form("kmeans_form"):

        col1, col2 = st.columns(2)

        with col1:

            purchase_value = st.number_input(
                "Purchase Value",
                min_value=0.0,
                value=50.0,
                step=1.0,
                key="km_purchase"
            )

            source = st.selectbox(
                "Traffic Source",
                sources,
                key="km_source"
            )

            browser = st.selectbox(
                "Browser",
                browsers,
                key="km_browser"
            )

            sex = st.selectbox(
                "Sex",
                sorted(df["sex"].dropna().unique().tolist()),
                key="km_sex"
            )

        with col2:

            age = st.number_input(
                "Customer Age",
                min_value=1,
                max_value=100,
                value=25,
                key="km_age"
            )

            account_age = st.number_input(
                "Account Age (Hours)",
                min_value=0.0,
                value=100.0,
                step=1.0,
                key="km_account"
            )

            country = st.selectbox(
                "Country",
                countries,
                key="km_country"
            )

            purchase_hour = st.slider(
                "Purchase Hour",
                min_value=0,
                max_value=23,
                value=12,
                key="km_hour"
            )

        cluster_submit = st.form_submit_button(
            "Find Transaction Cluster",
            use_container_width=True
        )

    if cluster_submit:

        input_data = pd.DataFrame([{
            "purchase_value": purchase_value,
            "source": source,
            "browser": browser,
            "sex": sex,
            "age": age,
            "account_age": account_age,
            "country": country,
            "purchase_hour": purchase_hour
        }])

        processed_input = preprocessor.transform(input_data)

        assigned_cluster = int(
            kmeans.predict(processed_input)[0]
        )

        cluster_info = cluster_summary[
            cluster_summary["Cluster"] == assigned_cluster
        ].iloc[0]

        st.divider()

        st.success(
            f"This transaction belongs to Cluster {assigned_cluster}."
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Cluster",
            f"Cluster {assigned_cluster}"
        )

        col2.metric(
            "Historical Fraud Rate",
            f"{cluster_info['Fraud_Percentage']:.2f}%"
        )

        st.caption(
            "This percentage describes historical transactions "
            "in the assigned cluster. It is not the individual "
            "transaction's fraud probability."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "FraudShield | Online Shopping Fraud & Suspicious Transaction Detection"
)