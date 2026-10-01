import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats

from src.data_loader import load_dataset
from src.data_profiler import get_dataset_profile, get_numeric_summary
from src.data_cleaner import (
    remove_columns,
    remove_duplicate_rows,
    remove_missing_rows,
    get_cleaning_summary,
)
from src.plot_engine import (
    create_histogram,
    create_box_plot,
    create_violin_plot,
    create_ecdf,
    create_probability_plot,
    fit_distribution,
    create_pdf_cdf_plot,
    create_log_transform,
    create_log_transform_plot,
    create_distribution_comparison,
    create_fit_plot,
    create_qq_plot,
)
from src.demo_datasets import DEMO_DATASETS, load_demo_dataset
from src.spatial_analysis import (
    detect_coordinate_columns,
    suggest_value_column,
    spatial_quality_report,
    generate_demo_spatial_dataset,
    prepare_spatial_data,
    create_spatial_map,
    compute_variogram_cloud,
    calculate_experimental_variogram,
    fit_variogram_model,
    create_variogram_plot,
    idw_interpolate,
    ordinary_kriging_predict,
    create_prediction_grid,
    create_interpolation_map,
    cross_validate_idw,
    cross_validate_kriging,
    summarize_cross_validation,
    create_validation_plot,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GeoStatix",
    page_icon="🌍",
    layout="wide",
)


# ============================================================
# RELATIONSHIP ANALYSIS HELPERS
# ============================================================

def prepare_xy(df, x_name, y_name):
    """Return clean numeric X/Y arrays for relationship analysis."""
    data = df[[x_name, y_name]].copy()
    data[x_name] = pd.to_numeric(data[x_name], errors="coerce")
    data[y_name] = pd.to_numeric(data[y_name], errors="coerce")
    data = data.dropna()

    if len(data) < 3:
        return None, None

    x = data[x_name].to_numpy(dtype=float)
    y = data[y_name].to_numpy(dtype=float)

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return None, None

    return x, y


def create_scatter_relationship_plot(x, y, x_name, y_name, show_regression=True):
    """Create an interactive scatter plot with optional linear regression."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Observed",
        )
    )

    if show_regression:
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = intercept + slope * x_line

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Linear Regression",
            )
        )

    fig.update_layout(
        title=f"{y_name} vs {x_name}",
        xaxis_title=x_name,
        yaxis_title=y_name,
    )

    return fig


def correlation_results(x, y):
    """Calculate Pearson, Spearman and Kendall correlation statistics."""
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    kendall_r, kendall_p = stats.kendalltau(x, y)

    return pd.DataFrame(
        [
            {
                "Method": "Pearson",
                "Correlation": pearson_r,
                "p-value": pearson_p,
            },
            {
                "Method": "Spearman",
                "Correlation": spearman_r,
                "p-value": spearman_p,
            },
            {
                "Method": "Kendall",
                "Correlation": kendall_r,
                "p-value": kendall_p,
            },
        ]
    )


def regression_results(x, y):
    """Calculate linear regression and diagnostic metrics."""
    result = stats.linregress(x, y)
    slope = result.slope
    intercept = result.intercept
    r_squared = result.rvalue ** 2
    predictions = intercept + slope * x
    residuals = y - predictions

    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    residual_std = (
        float(np.std(residuals, ddof=1))
        if len(residuals) > 1
        else 0.0
    )

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "p_value": result.pvalue,
        "predictions": predictions,
        "residuals": residuals,
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "residual_std": residual_std,
    }


def create_residual_plot(x, y, regression):
    """Create residuals versus observed/predicted X values."""
    predictions = regression["predictions"]
    residuals = regression["residuals"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=residuals,
            mode="markers",
            name="Residuals",
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="Zero residual",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="Regression Residual Plot",
        xaxis_title="X Variable",
        yaxis_title="Residual",
    )

    return fig


def create_predicted_observed_plot(y, predictions, y_name):
    """Create predicted-versus-observed plot with a 1:1 reference line."""
    minimum = float(min(y.min(), predictions.min()))
    maximum = float(max(y.max(), predictions.max()))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y,
            y=predictions,
            mode="markers",
            name="Predictions",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[minimum, maximum],
            y=[minimum, maximum],
            mode="lines",
            name="1:1 Reference",
        )
    )

    fig.update_layout(
        title="Predicted vs Observed",
        xaxis_title=f"Observed {y_name}",
        yaxis_title=f"Predicted {y_name}",
    )

    return fig


def create_correlation_heatmap(df, method="pearson"):
    """Create a correlation heatmap from all numeric variables."""
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return None

    matrix = numeric_df.corr(method=method)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.columns,
            text=np.round(matrix.values, 2),
            texttemplate="%{text}",
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )

    fig.update_layout(
        title=f"{method.title()} Correlation Heatmap",
        xaxis_title="Variables",
        yaxis_title="Variables",
    )

    return fig


def create_pairwise_correlation_table(df):
    """Create a screening table for every numeric-variable pair."""
    numeric_columns = list(
        df.select_dtypes(include="number").columns
    )

    rows = []

    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            x_name = numeric_columns[i]
            y_name = numeric_columns[j]
            x, y = prepare_xy(df, x_name, y_name)

            if x is None or y is None:
                continue

            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)

            rows.append(
                {
                    "Variable 1": x_name,
                    "Variable 2": y_name,
                    "Observations": len(x),
                    "Pearson r": pearson_r,
                    "Pearson p-value": pearson_p,
                    "Spearman ρ": spearman_r,
                    "Spearman p-value": spearman_p,
                }
            )

    if not rows:
        return None

    return pd.DataFrame(rows)


def find_coordinate_columns(df):
    """Try to identify common X/Y coordinate columns."""
    columns = list(df.columns)
    normalized = {
        col: str(col).strip().lower().replace(" ", "_")
        for col in columns
    }

    x_candidates = [
        "x", "easting", "east", "longitude", "lon", "utm_x", "x_coordinate"
    ]
    y_candidates = [
        "y", "northing", "north", "latitude", "lat", "utm_y", "y_coordinate"
    ]

    x_col = None
    y_col = None

    for col, norm in normalized.items():
        if x_col is None and norm in x_candidates:
            x_col = col
        if y_col is None and norm in y_candidates:
            y_col = col

    return x_col, y_col


# ============================================================
# HEADER / ONBOARDING
# ============================================================

st.title("🌍 GeoStatix")
st.subheader("Interactive Statistical & Geostatistical Analytics Platform")
st.write(
    """
    Explore geological and geostatistical data through profiling,
    distribution diagnostics, relationship analysis and spatial workflows.
    Start with a built-in demo or bring your own CSV/XLSX dataset.
    """
)


# ============================================================
# DATA SOURCE
# ============================================================

source_mode = st.radio(
    "How would you like to start?",
    ["🧪 Try a built-in demo", "📂 Upload my dataset"],
    horizontal=True,
    key="data_source_mode",
)

df = None
data_source_label = None
uploaded_file = None

if source_mode == "🧪 Try a built-in demo":
    st.info(
        "No dataset needed. These examples are synthetic, reproducible and generated locally, "
        "so you can test GeoStatix immediately without downloading anything."
    )

    demo_names = list(DEMO_DATASETS.keys())

    selected_demo = st.selectbox(
        "Choose a demo dataset",
        demo_names,
        key="selected_demo_dataset",
    )

    demo_config = DEMO_DATASETS[selected_demo]

    st.caption(demo_config["description"])

    active_demo = selected_demo

    df = load_demo_dataset(active_demo)

    data_source_label = f"Built-in demo: {active_demo}"

    st.success(
        f"Loaded **{active_demo}** — "
        f"{len(df):,} rows × {len(df.columns):,} columns."
    )

    st.download_button(
        "⬇️ Download this demo as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"{active_demo.lower().replace(' ', '_').replace('+', 'plus')}.csv"
        ),
        mime="text/csv",
        key="download_active_demo",
    )

else:
    uploaded_file = st.file_uploader(
        "📂 Upload your dataset",
        type=["csv", "xlsx"],
        help=(
            "Supported formats: CSV and Excel (.xlsx). "
            "GeoStatix does not require a specific column naming convention."
        ),
    )

    if uploaded_file is not None:
        try:
            df = load_dataset(uploaded_file)

            data_source_label = f"Uploaded: {uploaded_file.name}"

            st.success(
                f"Successfully loaded: {uploaded_file.name}"
            )

        except Exception as e:
            st.error(
                f"Could not load the uploaded dataset: {e}"
            )

# Keep compatibility with the existing spatial-analysis workflow.
use_demo = source_mode == "🧪 Try a built-in demo"


# ============================================================
# PUBLIC DATA SOURCES
# ============================================================

with st.expander("🌐 Find public geological / geostatistical data", expanded=False):
    st.write(
        "If you do not have your own data, these public sources provide geological, "
        "drillhole, geochemical or geometallurgical datasets that you can download "
        "and then upload to GeoStatix."
    )

    public_sources = [
        (
            "GeoMet dataset — Zenodo",
            "Curated geometallurgical data; `drillholes.csv` contains chemical analyses and geospatial coordinates.",
            "https://zenodo.org/records/7051975",
        ),
        (
            "New Brunswick Drillhole Dataset",
            "Government drillhole locations with CSV/GeoJSON resources; Open Government Licence — New Brunswick.",
            "https://open.canada.ca/data/en/dataset/8d55cc9a-3caf-5dd3-e7ca-4bb0633f9671",
        ),
        (
            "Alberta Geological Survey — Drillhole Data",
            "More than 5,300 drillholes with collar, interval and assay tables in a downloadable package.",
            "https://ags.aer.ca/publications/all-publications/dig-2024-0022",
        ),
        (
            "South Australia SARIG — Drillhole & Geochemistry",
            "Search spatially and download drillhole/geochemistry data packages in CSV format.",
            "https://energymining.sa.gov.au/industry/geological-survey/products-and-services/map-viewers-databases-and-services/mineral-drilling-and-geochemistry",
        ),
        (
            "Western Australia — Mineral Exploration Drillholes",
            "Large public drillhole database with collar data and, where available, assays, geology and surveys.",
            "https://www.wa.gov.au/service/natural-resources/mineral-resources/access-company-mineral-drillhole-and-surface-geochemistry-database",
        ),
        (
            "Kaggle — Geology Dataset Search",
            "Community-hosted datasets; check each dataset's licence and structure before reuse.",
            "https://www.kaggle.com/datasets?search=geology",
        ),
    ]

    for title, description, url in public_sources:
        st.markdown(f"**{title}** — {description}")
        st.markdown(f"[Open dataset/source ↗]({url})")

    st.caption(
        "Tip: GeoStatix works best when the downloaded file contains numeric variables and, "
        "for spatial analysis, explicit X/Y, Easting/Northing or longitude/latitude columns. "
        "Always check the source licence, units and metadata before using external data."
    )


# ============================================================
# DATASET PROCESSING
# ============================================================

if df is not None:
    try:
        original_df = df.copy()
        profile = get_dataset_profile(df)

        # Clear a previous cleaning result when the underlying dataset changes.
        source_signature = (data_source_label, tuple(df.columns), len(df))
        if st.session_state.get("active_source_signature") != source_signature:
            st.session_state.pop("cleaned_df", None)
            st.session_state.pop("cleaning_summary", None)
            st.session_state["active_source_signature"] = source_signature

        if data_source_label:
            st.caption(f"**Current data source:** {data_source_label}")

        # ====================================================
        # DATASET OVERVIEW
        # ====================================================

        st.subheader("📊 Dataset Overview")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Rows", profile["rows"])

        with col2:
            st.metric("Columns", profile["columns"])

        with col3:
            st.metric(
                "Numeric Columns",
                len(profile["numeric_columns"]),
            )

        with col4:
            st.metric(
                "Categorical Columns",
                len(profile["categorical_columns"]),
            )

        with col5:
            st.metric("Missing Cells", profile["missing_cells"])

        # ====================================================
        # DATA QUALITY
        # ====================================================

        st.subheader("🔍 Data Quality Report")

        if profile["missing_cells"] == 0:
            st.success("✓ No missing values detected.")
        else:
            st.warning(
                f"⚠ {profile['missing_cells']} missing cells detected."
            )

        if profile["duplicate_rows"] == 0:
            st.success("✓ No duplicate rows detected.")
        else:
            st.warning(
                f"⚠ {profile['duplicate_rows']} duplicate rows detected."
            )

        if profile["unnamed_columns"]:
            st.warning(
                "⚠ Potential unnamed/index columns detected: "
                + ", ".join(profile["unnamed_columns"])
            )

        if profile["constant_columns"]:
            st.warning(
                "⚠ Constant numeric columns detected: "
                + ", ".join(profile["constant_columns"])
            )

        if profile["log_columns"]:
            st.info(
                "ℹ Possible log-transformed variables detected: "
                + ", ".join(profile["log_columns"])
            )

        # ====================================================
        # COLUMN CLASSIFICATION
        # ====================================================

        st.subheader("🧩 Column Classification")

        classification_col1, classification_col2 = st.columns(2)

        with classification_col1:
            st.write("**🔢 Numeric Variables**")
            for column in profile["numeric_columns"]:
                st.write(f"• {column}")

        with classification_col2:
            st.write("**🔤 Categorical Variables**")
            for column in profile["categorical_columns"]:
                st.write(f"• {column}")

        # ====================================================
        # DATA CLEANING
        # ====================================================

        st.subheader("🧹 Data Cleaning & Review")
        st.write(
            """
            GeoStatix does not automatically delete suspicious geological
            data. Review the detected issues and choose what you want to remove.
            """
        )

        suggested_columns = list(
            dict.fromkeys(
                profile["unnamed_columns"] + profile["constant_columns"]
            )
        )

        cleaning_options = st.multiselect(
            "Select columns to remove",
            options=df.columns.tolist(),
            default=[c for c in suggested_columns if c in df.columns],
        )

        col1, col2 = st.columns(2)

        with col1:
            remove_duplicates = st.checkbox(
                "Remove duplicate rows",
                value=False,
            )

        with col2:
            remove_missing = st.checkbox(
                "Remove rows containing missing values",
                value=False,
            )

        if st.button("🧹 Apply Cleaning", type="primary"):
            cleaned_df = remove_columns(df, cleaning_options)

            if remove_duplicates:
                cleaned_df = remove_duplicate_rows(cleaned_df)

            if remove_missing:
                cleaned_df = remove_missing_rows(cleaned_df)

            st.session_state["cleaned_df"] = cleaned_df
            st.session_state["cleaning_summary"] = get_cleaning_summary(
                df,
                cleaned_df,
            )

            st.success("Cleaning applied successfully.")

        # ====================================================
        # CLEANED DATASET
        # ====================================================

        if "cleaned_df" in st.session_state:
            cleaned_df = st.session_state["cleaned_df"]
            summary = st.session_state["cleaning_summary"]

            st.subheader("📋 Cleaning Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Original Rows", summary["original_rows"])

            with col2:
                st.metric("Remaining Rows", summary["remaining_rows"])

            with col3:
                st.metric("Original Columns", summary["original_columns"])

            with col4:
                st.metric("Remaining Columns", summary["remaining_columns"])
        else:
            cleaned_df = df.copy()

        # ====================================================
        # VARIABLE SELECTION
        # ====================================================

        st.subheader("🎯 Select Variable for Analysis")

        numeric_columns = cleaned_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            selected_variable = st.selectbox(
                "Choose a numeric variable",
                numeric_columns,
                key="selected_variable",
            )

            selected_data = pd.to_numeric(
                cleaned_df[selected_variable],
                errors="coerce",
            ).dropna()

            st.write(
                f"**Selected variable:** `{selected_variable}`"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Observations", len(selected_data))

            with col2:
                st.metric("Mean", f"{selected_data.mean():.4f}")

            with col3:
                st.metric("Median", f"{selected_data.median():.4f}")

            with col4:
                st.metric("Std Dev", f"{selected_data.std():.4f}")

            # ============================================================
            # DISTRIBUTION ANALYSIS
            # ============================================================

            st.subheader("📊 Distribution Analysis")

            plot_type = st.selectbox(
                "Choose an analysis",
                [
                    "Distribution Diagnostic",
                    "Histogram",
                    "Box Plot",
                    "Violin Plot",
                    "Empirical CDF",
                    "Normal Distribution Fit",
                    "Lognormal Distribution Fit",
                    "Normal Q-Q Plot",
                    "Lognormal Q-Q Plot",
                    "Normal Probability Plot",
                    "Lognormal Probability Plot",
                    "PDF & CDF",
                    "Log Transformation",
                    "Distribution Comparison",
                ],
                key="distribution_analysis",
            )

            if plot_type == "Distribution Diagnostic":
                st.write("### Distribution Diagnostic")
                st.caption(
                    "A compact diagnostic view combining distribution shape, fitted-distribution statistics, "
                    "and visual checks. Use these results together rather than as an automatic distribution selector."
                )

                # Clean numeric series for diagnostics.
                diagnostic_data = pd.to_numeric(selected_data, errors="coerce").dropna()

                if len(diagnostic_data) < 3:
                    st.warning("At least 3 valid observations are required for distribution diagnostics.")
                elif diagnostic_data.nunique() < 2:
                    st.warning("Distribution diagnostics require at least two distinct values.")
                else:
                    skewness = stats.skew(diagnostic_data, bias=False)
                    kurtosis = stats.kurtosis(diagnostic_data, fisher=True, bias=False)

                    st.write("#### Shape Statistics")
                    shape1, shape2, shape3, shape4 = st.columns(4)
                    with shape1:
                        st.metric("Mean", f"{diagnostic_data.mean():.5g}")
                    with shape2:
                        st.metric("Median", f"{diagnostic_data.median():.5g}")
                    with shape3:
                        st.metric("Skewness", f"{skewness:.5g}")
                    with shape4:
                        st.metric("Kurtosis", f"{kurtosis:.5g}")

                    st.write("#### Normal Fit")
                    normal_result = fit_distribution(
                        diagnostic_data,
                        distribution="normal",
                    )

                    if normal_result:
                        n1, n2, n3 = st.columns(3)
                        with n1:
                            st.metric("KS Statistic", f"{normal_result['KS Statistic']:.5g}")
                        with n2:
                            st.metric("KS p-value", f"{normal_result['KS p-value']:.6g}")
                        with n3:
                            st.metric("AIC", f"{normal_result['AIC']:.2f}")
                        st.caption("Fitted parameters")
                        st.json(normal_result["Parameters"])
                    else:
                        st.info("Normal distribution fit could not be calculated for this variable.")

                    st.write("#### Lognormal Fit")
                    if (diagnostic_data <= 0).any():
                        st.info(
                            "Not available — lognormal fitting requires strictly positive values; "
                            "zero or negative values were detected."
                        )
                    else:
                        lognormal_result = fit_distribution(
                            diagnostic_data,
                            distribution="lognormal",
                        )
                        if lognormal_result:
                            l1, l2, l3 = st.columns(3)
                            with l1:
                                st.metric("KS Statistic", f"{lognormal_result['KS Statistic']:.5g}")
                            with l2:
                                st.metric("KS p-value", f"{lognormal_result['KS p-value']:.6g}")
                            with l3:
                                st.metric("AIC", f"{lognormal_result['AIC']:.2f}")
                            st.caption("Fitted parameters")
                            st.json(lognormal_result["Parameters"])
                        else:
                            st.info("Lognormal distribution fit could not be calculated for this variable.")

                    st.write("#### Visual Diagnostics")
                    visual_tabs = st.tabs([
                        "Histogram + Normal",
                        "Normal Q-Q",
                        "Normal Probability",
                        "Lognormal Diagnostics",
                    ])

                    with visual_tabs[0]:
                        fig = create_fit_plot(
                            diagnostic_data,
                            distribution="normal",
                        )
                        if fig is not None:
                            st.plotly_chart(fig, width="stretch")

                    with visual_tabs[1]:
                        fig = create_qq_plot(
                            diagnostic_data,
                            distribution="normal",
                        )
                        if fig is not None:
                            st.plotly_chart(fig, width="stretch")

                    with visual_tabs[2]:
                        fig = create_probability_plot(
                            diagnostic_data,
                            distribution="normal",
                        )
                        if fig is not None:
                            st.plotly_chart(fig, width="stretch")

                    with visual_tabs[3]:
                        if (diagnostic_data <= 0).any():
                            st.info(
                                "Lognormal visual diagnostics are unavailable because zero or negative values are present."
                            )
                        else:
                            fig = create_fit_plot(
                                diagnostic_data,
                                distribution="lognormal",
                            )
                            if fig is not None:
                                st.plotly_chart(fig, width="stretch")

                            q1, q2 = st.columns(2)
                            with q1:
                                qq_fig = create_qq_plot(
                                    diagnostic_data,
                                    distribution="lognormal",
                                )
                                if qq_fig is not None:
                                    st.plotly_chart(qq_fig, width="stretch")
                            with q2:
                                prob_fig = create_probability_plot(
                                    diagnostic_data,
                                    distribution="lognormal",
                                )
                                if prob_fig is not None:
                                    st.plotly_chart(prob_fig, width="stretch")

                    st.info(
                        "Interpret KS statistics, p-values, AIC, skewness/kurtosis and the plots together. "
                        "A fitted distribution is a statistical approximation and does not by itself establish "
                        "geological validity or prove that one distribution is universally preferable."
                    )

            elif plot_type == "Histogram":
                bins = st.slider(
                    "Number of bins",
                    min_value=5,
                    max_value=100,
                    value=20,
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    show_kde = st.checkbox("Show KDE", value=True)

                with col2:
                    show_mean = st.checkbox("Show Mean", value=True)

                with col3:
                    show_median = st.checkbox("Show Median", value=True)

                fig = create_histogram(
                    selected_data,
                    bins=bins,
                    show_kde=show_kde,
                    show_mean=show_mean,
                    show_median=show_median,
                )
                st.plotly_chart(fig, width="stretch")

            elif plot_type == "Box Plot":
                fig = create_box_plot(selected_data)
                st.plotly_chart(fig, width="stretch")

            elif plot_type == "Violin Plot":
                fig = create_violin_plot(selected_data)
                st.plotly_chart(fig, width="stretch")

            elif plot_type == "Empirical CDF":
                fig = create_ecdf(selected_data)
                st.plotly_chart(fig, width="stretch")

            elif plot_type == "Normal Distribution Fit":
                fig = create_fit_plot(selected_data, distribution="normal")

                if fig is not None:
                    st.plotly_chart(fig, width="stretch")

                    result = fit_distribution(
                        selected_data,
                        distribution="normal",
                    )

                    if result:
                        st.write("### Normal Fit Diagnostics")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "KS Statistic",
                                f"{result['KS Statistic']:.4f}",
                            )

                        with col2:
                            st.metric(
                                "KS p-value",
                                f"{result['KS p-value']:.6f}",
                            )

                        with col3:
                            st.metric(
                                "AIC",
                                f"{result['AIC']:.2f}",
                            )

                        st.write("Fitted parameters")
                        st.json(result["Parameters"])

            elif plot_type == "Lognormal Distribution Fit":
                if (selected_data <= 0).any():
                    st.warning(
                        "Lognormal fitting requires all values to be strictly greater than zero."
                    )
                else:
                    fig = create_fit_plot(
                        selected_data,
                        distribution="lognormal",
                    )

                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")

                        result = fit_distribution(
                            selected_data,
                            distribution="lognormal",
                        )

                        if result:
                            st.write("### Lognormal Fit Diagnostics")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric(
                                    "KS Statistic",
                                    f"{result['KS Statistic']:.4f}",
                                )

                            with col2:
                                st.metric(
                                    "KS p-value",
                                    f"{result['KS p-value']:.6f}",
                                )

                            with col3:
                                st.metric(
                                    "AIC",
                                    f"{result['AIC']:.2f}",
                                )

                            st.write("Fitted parameters")
                            st.json(result["Parameters"])

            elif plot_type == "Normal Q-Q Plot":
                fig = create_qq_plot(
                    selected_data,
                    distribution="normal",
                )
                if fig is not None:
                    st.plotly_chart(fig, width="stretch")

            elif plot_type == "Lognormal Q-Q Plot":
                if (selected_data <= 0).any():
                    st.warning(
                        "Lognormal Q-Q analysis requires strictly positive values."
                    )
                else:
                    fig = create_qq_plot(
                        selected_data,
                        distribution="lognormal",
                    )
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")

            elif plot_type == "Normal Probability Plot":
                fig = create_probability_plot(
                    selected_data,
                    distribution="normal",
                )
                if fig is not None:
                    st.plotly_chart(fig, width="stretch")

            elif plot_type == "Lognormal Probability Plot":
                if (selected_data <= 0).any():
                    st.warning(
                        "Lognormal probability analysis requires strictly positive values."
                    )
                else:
                    fig = create_probability_plot(
                        selected_data,
                        distribution="lognormal",
                    )
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")

            elif plot_type == "PDF & CDF":
                distribution = st.radio(
                    "Distribution",
                    ["Normal", "Lognormal"],
                    horizontal=True,
                )

                if (
                    distribution == "Lognormal"
                    and (selected_data <= 0).any()
                ):
                    st.warning(
                        "Lognormal analysis requires strictly positive values."
                    )
                else:
                    fig = create_pdf_cdf_plot(
                        selected_data,
                        distribution=distribution.lower(),
                    )
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")

            elif plot_type == "Log Transformation":
                if (selected_data <= 0).any():
                    st.warning(
                        "ln(X) cannot be calculated because this variable contains zero or negative values."
                    )
                else:
                    transformed = create_log_transform(selected_data)

                    st.write("### Log-Transformed Statistics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Original Mean",
                            f"{selected_data.mean():.4f}",
                        )

                    with col2:
                        st.metric(
                            "Original Std Dev",
                            f"{selected_data.std():.4f}",
                        )

                    with col3:
                        st.metric(
                            "ln(X) Mean",
                            f"{transformed.mean():.4f}",
                        )

                    with col4:
                        st.metric(
                            "ln(X) Std Dev",
                            f"{transformed.std():.4f}",
                        )

                    fig = create_log_transform_plot(selected_data)
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch")

                    transformed_df = pd.DataFrame(
                        {
                            selected_variable: selected_data.values,
                            f"ln({selected_variable})": transformed.values,
                        }
                    )

                    st.write("### Original vs Log-Transformed Data")
                    st.dataframe(
                        transformed_df,
                        width="stretch",
                        hide_index=True,
                    )

                    csv = transformed_df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Log-Transformed Data",
                        csv,
                        file_name="log_transformed_data.csv",
                        mime="text/csv",
                    )

            elif plot_type == "Distribution Comparison":
                comparison = create_distribution_comparison(selected_data)

                st.write("### Distribution Fit Comparison")

                if comparison.empty:
                    st.info(
                        "No valid distribution fits could be calculated for this variable."
                    )
                else:
                    st.dataframe(
                        comparison,
                        width="stretch",
                        hide_index=True,
                    )

                    st.info(
                        """
                        Diagnostic note: KS statistics, p-values and AIC are
                        distribution-fitting diagnostics. Interpret them alongside
                        geological knowledge and graphical diagnostics rather than
                        using them as an automatic decision rule.
                        """
                    )

            # ============================================================
            # RELATIONSHIP ANALYSIS
            # ============================================================

            st.subheader("🔗 Relationship Analysis")
            st.write(
                "Explore relationships between two numeric variables using scatter plots, correlation analysis and regression."
            )

            if len(numeric_columns) >= 2:
                relationship_col1, relationship_col2 = st.columns(2)

                with relationship_col1:
                    x_variable = st.selectbox(
                        "Select X variable",
                        numeric_columns,
                        key="relationship_x",
                    )

                with relationship_col2:
                    default_y_index = (
                        1 if numeric_columns[0] == x_variable and len(numeric_columns) > 1 else 0
                    )
                    y_variable = st.selectbox(
                        "Select Y variable",
                        numeric_columns,
                        index=default_y_index,
                        key="relationship_y",
                    )

                x_data, y_data = prepare_xy(
                    cleaned_df,
                    x_variable,
                    y_variable,
                )

                if x_data is None or y_data is None:
                    st.warning(
                        "The selected variables need at least 3 paired observations and must not be constant."
                    )
                else:
                    relationship_analysis = st.selectbox(
                        "Choose a relationship analysis",
                        [
                            "Scatter Plot",
                            "Correlation Analysis",
                            "Correlation Heatmap",
                            "Pairwise Correlation Screening",
                            "Linear Regression",
                            "Residual Plot",
                            "Predicted vs Observed",
                        ],
                        key="relationship_analysis",
                    )

                    if relationship_analysis == "Scatter Plot":
                        show_regression = st.checkbox(
                            "Show regression line",
                            value=True,
                        )

                        fig = create_scatter_relationship_plot(
                            x_data,
                            y_data,
                            x_variable,
                            y_variable,
                            show_regression=show_regression,
                        )
                        st.plotly_chart(fig, width="stretch")

                    elif relationship_analysis == "Correlation Analysis":
                        st.write("### Correlation Results")
                        correlation_df = correlation_results(
                            x_data,
                            y_data,
                        )
                        st.dataframe(
                            correlation_df,
                            width="stretch",
                            hide_index=True,
                        )
                        st.info(
                            "Pearson measures linear association. Spearman measures monotonic association using ranks. Kendall measures rank-based concordance."
                        )

                    elif relationship_analysis == "Correlation Heatmap":
                        method = st.radio(
                            "Correlation method",
                            ["Pearson", "Spearman"],
                            horizontal=True,
                            key="heatmap_method",
                        )

                        fig = create_correlation_heatmap(
                            cleaned_df,
                            method=method.lower(),
                        )

                        if fig is not None:
                            st.plotly_chart(fig, width="stretch")
                        else:
                            st.info(
                                "At least two numeric variables are required for a correlation heatmap."
                            )

                    elif relationship_analysis == "Pairwise Correlation Screening":
                        st.write("### Pairwise Correlation Screening")
                        pairwise_table = create_pairwise_correlation_table(
                            cleaned_df
                        )

                        if pairwise_table is None:
                            st.info(
                                "No suitable numeric variable pairs were found."
                            )
                        else:
                            st.dataframe(
                                pairwise_table,
                                width="stretch",
                                hide_index=True,
                            )
                            st.info(
                                "Use this table as a screening tool. Correlation does not establish causation and should be interpreted with geological context and graphical diagnostics."
                            )

                    elif relationship_analysis == "Linear Regression":
                        regression = regression_results(
                            x_data,
                            y_data,
                        )

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "Slope",
                                f"{regression['slope']:.5f}",
                            )

                        with col2:
                            st.metric(
                                "Intercept",
                                f"{regression['intercept']:.5f}",
                            )

                        with col3:
                            st.metric(
                                "R²",
                                f"{regression['r_squared']:.4f}",
                            )

                        with col4:
                            st.metric(
                                "p-value",
                                f"{regression['p_value']:.6f}",
                            )

                        st.write("### Regression Equation")
                        st.code(
                            f"{y_variable} = {regression['slope']:.5f} × {x_variable} + {regression['intercept']:.5f}"
                        )

                        st.write("### Regression Error Metrics")
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                        with metric_col1:
                            st.metric("MAE", f"{regression['mae']:.6f}")

                        with metric_col2:
                            st.metric("RMSE", f"{regression['rmse']:.6f}")

                        with metric_col3:
                            st.metric("MSE", f"{regression['mse']:.6f}")

                        with metric_col4:
                            st.metric(
                                "Residual Std Dev",
                                f"{regression['residual_std']:.6f}",
                            )

                        fig = create_scatter_relationship_plot(
                            x_data,
                            y_data,
                            x_variable,
                            y_variable,
                            show_regression=True,
                        )
                        st.plotly_chart(fig, width="stretch")

                    elif relationship_analysis == "Residual Plot":
                        regression = regression_results(
                            x_data,
                            y_data,
                        )

                        st.write("### Regression Residual Plot")
                        fig = create_residual_plot(
                            x_data,
                            y_data,
                            regression,
                        )
                        st.plotly_chart(fig, width="stretch")

                        st.info(
                            "A residual plot helps assess whether errors are centered around zero and whether systematic patterns remain after fitting the linear model."
                        )

                    elif relationship_analysis == "Predicted vs Observed":
                        regression = regression_results(
                            x_data,
                            y_data,
                        )

                        st.write("### Predicted vs Observed")
                        fig = create_predicted_observed_plot(
                            y_data,
                            regression["predictions"],
                            y_variable,
                        )
                        st.plotly_chart(fig, width="stretch")

                        st.info(
                            "Points closer to the 1:1 reference line indicate closer agreement between observed and regression-predicted values."
                        )

            else:
                st.info(
                    "At least two numeric variables are required for relationship analysis."
                )

            # ============================================================
            # SPATIAL / GEOSTATISTICAL ANALYSIS
            # ============================================================

            st.subheader("🌍 Spatial & Geostatistical Analysis")
            st.write(
                "Analyze spatial sample locations, value patterns, variograms and interpolation. "
                "GeoStatix uses coordinate-name detection when possible and never assumes that the first numeric columns are coordinates."
            )

            # A built-in reproducible dataset makes the spatial module testable even
            # when the uploaded file is not spatial (for example, a grade-only table).
            demo_col1, demo_col2 = st.columns([3, 1])
            with demo_col1:
                st.caption("No spatial dataset? Use the built-in demo to test maps, variograms and interpolation.")
            with demo_col2:
                use_demo = st.checkbox("Use demo spatial data", key="use_spatial_demo_v4")

            spatial_source_df = generate_demo_spatial_dataset() if use_demo else cleaned_df

            if use_demo:
                st.success("Using GeoStatix demo spatial dataset: 200 synthetic samples with Easting, Northing and Gold_Grade.")

            spatial_numeric_columns = spatial_source_df.select_dtypes(
                include="number"
            ).columns.tolist()

            if len(spatial_numeric_columns) >= 2:
                detected = detect_coordinate_columns(spatial_source_df)
                detected_x = detected.get("x") if detected else None
                detected_y = detected.get("y") if detected else None

                if detected_x is not None and detected_y is not None:
                    st.caption(
                        f"Detected **{detected.get('mode', 'spatial')}** coordinates: "
                        f"X = `{detected_x}`, Y = `{detected_y}` ({detected.get('confidence', 'unknown')} confidence). Verify before interpretation."
                    )
                else:
                    st.warning(
                        "No reliable coordinate columns were detected. GeoStatix will not automatically treat the first numeric columns as X/Y. "
                        "Select real spatial coordinates manually, or enable the demo dataset."
                    )

                # Streamlit selectbox state persists across uploads. Use a dataset-specific
                # widget key so an old X/Y selection cannot leak into a new dataset.
                if use_demo:
                    source_signature = "demo"
                else:
                    source_signature = (
    f"{spatial_source_df.shape}|"
    f"{'|'.join(map(str, spatial_source_df.columns))}"
)
                widget_token = str(abs(hash(source_signature)))

                spatial_col1, spatial_col2 = st.columns(2)
                x_options = spatial_numeric_columns

                with spatial_col1:
                    x_index = x_options.index(detected_x) if detected_x in x_options else None
                    spatial_x = st.selectbox(
                        "X / Easting / Longitude",
                        x_options,
                        index=x_index,
                        placeholder="Select X coordinate...",
                        key=f"spatial_x_v5_{widget_token}",
                    )

                # IMPORTANT: Y options exclude the CURRENTLY selected X.
                # This prevents X == Y even when the user changes X manually.
                y_options = [c for c in spatial_numeric_columns if c != spatial_x]

                with spatial_col2:
                    valid_y_default = detected_y if detected_y in y_options else None
                    y_index = y_options.index(valid_y_default) if valid_y_default is not None else None
                    spatial_y = st.selectbox(
                        "Y / Northing / Latitude",
                        y_options,
                        index=y_index,
                        placeholder="Select Y coordinate...",
                        key=f"spatial_y_v5_{widget_token}",
                    )

                value_candidates = [
                    c for c in spatial_numeric_columns
                    if c not in {spatial_x, spatial_y}
                ]
                suggested_value = suggest_value_column(spatial_source_df, spatial_x, spatial_y)

                # Prefer a spatial value suggestion over the globally selected
                # variable when that variable looks like an ID/index.
                if suggested_value in value_candidates:
                    default_value = suggested_value
                elif selected_variable in value_candidates:
                    default_value = selected_variable
                else:
                    default_value = None

                value_options = ["None"] + value_candidates
                value_index = value_options.index(default_value) if default_value in value_options else 0
                spatial_value = st.selectbox(
                    "Value variable for spatial analysis",
                    value_options,
                    index=value_index,
                    key=f"spatial_value_v5_{widget_token}",
                )
                spatial_value_column = None if spatial_value == "None" else spatial_value

                if suggested_value is not None and suggested_value in value_candidates:
                    st.caption(f"Suggested value variable: `{suggested_value}` — based on column-name/data heuristics; you can override it.")

                if spatial_x is not None and spatial_y is not None:
                    quality = spatial_quality_report(
                        spatial_source_df, spatial_x, spatial_y, spatial_value_column
                    )
                    if quality is not None:
                        with st.expander("🔎 Spatial data quality checks", expanded=False):
                            q1, q2, q3, q4 = st.columns(4)
                            with q1:
                                st.metric("Valid rows", quality["valid_coordinate_rows"])
                            with q2:
                                st.metric("Unique locations", quality["unique_locations"])
                            with q3:
                                st.metric("Duplicate-location rows", quality["duplicate_location_rows"])
                            with q4:
                                if spatial_value_column is not None:
                                    st.metric("Value range", f"{quality.get('value_range', 0):.5g}")
                                else:
                                    st.metric("X/Y ranges", f"{quality['x_range']:.4g} / {quality['y_range']:.4g}")
                            if quality["duplicate_location_rows"] > 0:
                                st.warning("Duplicate coordinate locations are present. Review whether they represent legitimate repeated samples or require compositing before interpreting variograms/interpolation.")
                            if spatial_value_column is not None and quality.get("value_constant", False):
                                st.warning("The selected value is constant. Variogram and interpolation results will contain little or no spatial information.")

                spatial_analysis = st.selectbox(
                    "Choose a spatial analysis",
                    [
                        "Spatial Sample Map",
                        "Spatial Value Map",
                        "Variogram Cloud",
                        "Experimental Variogram",
                        "IDW Interpolation",
                        "Ordinary Kriging",
                        "Cross-Validation",
                    ],
                    key="spatial_analysis_v4",
                )

                if spatial_x is None or spatial_y is None:
                    st.info(
                        "Select real X and Y coordinate columns to enable spatial analysis. "
                        "For datasets without locations, enable the demo spatial dataset above."
                    )
                elif spatial_x == spatial_y:
                    st.error("X and Y coordinates must be different columns.")
                elif spatial_analysis == "Spatial Sample Map":
                    spatial_data = prepare_spatial_data(spatial_source_df, spatial_x, spatial_y)
                    if spatial_data is None:
                        st.warning("At least 3 valid, distinct coordinate pairs are required.")
                    else:
                        st.metric("Spatial Samples", len(spatial_data))
                        fig = create_spatial_map(spatial_source_df, spatial_x, spatial_y)
                        st.plotly_chart(fig, width="stretch")

                elif spatial_analysis == "Spatial Value Map":
                    if spatial_value_column is None:
                        st.info("Select a numeric value variable different from X and Y.")
                    else:
                        spatial_data = prepare_spatial_data(spatial_source_df, spatial_x, spatial_y, spatial_value_column)
                        if spatial_data is None:
                            st.warning("At least 3 valid coordinate/value observations are required.")
                        else:
                            map_col1, map_col2, map_col3 = st.columns(3)
                            with map_col1:
                                st.metric("Samples", len(spatial_data))
                            with map_col2:
                                st.metric("Minimum", f"{spatial_data[spatial_value_column].min():.5g}")
                            with map_col3:
                                st.metric("Maximum", f"{spatial_data[spatial_value_column].max():.5g}")
                            fig = create_spatial_map(spatial_source_df, spatial_x, spatial_y, spatial_value_column)
                            st.plotly_chart(fig, width="stretch")

                elif spatial_analysis in ["Variogram Cloud", "Experimental Variogram"]:
                    if spatial_value_column is None:
                        st.info("Select a value variable different from X and Y to calculate a variogram.")
                    else:
                        spatial_data = prepare_spatial_data(spatial_source_df, spatial_x, spatial_y, spatial_value_column)
                        if spatial_data is None or len(spatial_data) < 4:
                            st.warning("At least 4 valid spatial observations are recommended for variogram analysis.")
                        else:
                            total_possible_pairs = len(spatial_data) * (len(spatial_data) - 1) // 2
                            max_pair_cap = min(100000, total_possible_pairs)
                            max_pairs_default = min(50000, total_possible_pairs)
                            if total_possible_pairs < 1000:
                                max_pairs = total_possible_pairs
                            else:
                                max_pairs = st.slider(
                                    "Maximum point pairs",
                                    min_value=1000,
                                    max_value=max_pair_cap,
                                    value=max_pairs_default,
                                    step=1000,
                                    key="variogram_max_pairs_v4",
                                )

                            cloud = compute_variogram_cloud(
                                spatial_data, spatial_x, spatial_y, spatial_value_column, max_pairs=max_pairs
                            )

                            if cloud is None or cloud.empty:
                                st.warning("A variogram cloud could not be calculated from the selected data.")
                            elif spatial_analysis == "Variogram Cloud":
                                st.write("### Variogram Cloud")
                                st.caption("Each point represents a sample pair; semivariance is ½(zᵢ − zⱼ)².")
                                cloud_plot = cloud.sample(10000, random_state=42) if len(cloud) > 10000 else cloud
                                fig = go.Figure(
                                    go.Scattergl(
                                        x=cloud_plot["Distance"],
                                        y=cloud_plot["Semivariance"],
                                        mode="markers",
                                        marker=dict(size=4, opacity=0.45),
                                        name="Pairs",
                                    )
                                )
                                fig.update_layout(
                                    title="Variogram Cloud",
                                    xaxis_title="Pair Distance",
                                    yaxis_title="Semivariance γ(h)",
                                    template="plotly_dark",
                                )
                                st.plotly_chart(fig, width="stretch")
                                st.dataframe(cloud.head(1000), width="stretch", hide_index=True)
                            else:
                                lag_col1, lag_col2 = st.columns(2)
                                with lag_col1:
                                    n_lags = st.slider("Number of lag classes", 5, 30, 12, key="variogram_lags_v4")
                                with lag_col2:
                                    max_distance_default = float(cloud["Distance"].quantile(0.75))
                                    max_distance = st.number_input(
                                        "Maximum lag distance",
                                        min_value=float(cloud["Distance"].min()),
                                        max_value=float(cloud["Distance"].max()),
                                        value=max_distance_default,
                                        key="variogram_max_distance_v4",
                                    )

                                experimental = calculate_experimental_variogram(cloud, n_lags=n_lags, max_distance=max_distance)
                                if experimental is None or len(experimental) < 3:
                                    st.warning("Not enough populated lag classes to build an experimental variogram.")
                                else:
                                    model_choice = st.selectbox(
                                        "Variogram model",
                                        ["None", "Spherical", "Exponential", "Gaussian"],
                                        key="variogram_model_v4",
                                    )
                                    fitted = fit_variogram_model(experimental, model_choice) if model_choice != "None" else None
                                    fig = create_variogram_plot(experimental, fitted=fitted, show_cloud=cloud if st.checkbox("Overlay variogram cloud", key="overlay_cloud_v4") else None)
                                    st.plotly_chart(fig, width="stretch")
                                    st.write("### Experimental Variogram Data")
                                    st.dataframe(experimental, width="stretch", hide_index=True)

                                    if fitted is not None:
                                        model_col1, model_col2, model_col3, model_col4 = st.columns(4)
                                        with model_col1:
                                            st.metric("Nugget", f"{fitted['nugget']:.6g}")
                                        with model_col2:
                                            st.metric("Partial Sill", f"{fitted['sill']:.6g}")
                                        with model_col3:
                                            st.metric("Range", f"{fitted['range']:.6g}")
                                        with model_col4:
                                            st.metric("Model RMSE", f"{fitted['rmse']:.6g}")

                elif spatial_analysis == "Cross-Validation":
                    if spatial_value_column is None:
                        st.info("Select a numeric value variable different from X and Y before validation.")
                    else:
                        spatial_data = prepare_spatial_data(
                            spatial_source_df, spatial_x, spatial_y, spatial_value_column
                        )
                        if spatial_data is None or len(spatial_data) < 5:
                            st.warning("At least 5 valid spatial observations are required for cross-validation.")
                        else:
                            st.write("### Spatial Cross-Validation")
                            st.caption(
                                "Each validation point is temporarily removed from the training data and predicted from the remaining samples. "
                                "The fitted Kriging variogram is held fixed during validation."
                            )

                            cv_col1, cv_col2 = st.columns(2)
                            with cv_col1:
                                cv_method = st.selectbox(
                                    "Validation method",
                                    ["Compare IDW and Kriging", "IDW", "Ordinary Kriging"],
                                    key="cv_method_v5",
                                )
                            with cv_col2:
                                max_cv = st.slider(
                                    "Validation points",
                                    min_value=5,
                                    max_value=len(spatial_data),
                                    value=len(spatial_data),
                                    key="cv_points_v5",
                                )

                            idw_result = None
                            kriging_result = None
                            metrics_rows = []

                            if cv_method in ["Compare IDW and Kriging", "IDW"]:
                                idw_col1, idw_col2 = st.columns(2)
                                with idw_col1:
                                    cv_power = st.slider("IDW power", 0.5, 5.0, 2.0, 0.5, key="cv_idw_power_v5")
                                with idw_col2:
                                    cv_neighbors = st.slider(
                                        "IDW maximum neighbors",
                                        3,
                                        min(50, len(spatial_data)),
                                        min(20, len(spatial_data)),
                                        key="cv_idw_neighbors_v5",
                                    )
                                idw_result = cross_validate_idw(
                                    spatial_data, spatial_x, spatial_y, spatial_value_column,
                                    power=cv_power, max_neighbors=cv_neighbors,
                                    max_validation_points=max_cv,
                                )
                                idw_metrics = summarize_cross_validation(idw_result)
                                if idw_metrics is not None:
                                    metrics_rows.append({"Method": "IDW", **idw_metrics})

                            if cv_method in ["Compare IDW and Kriging", "Ordinary Kriging"]:
                                krig_col1, krig_col2 = st.columns(2)
                                with krig_col1:
                                    cv_model = st.selectbox(
                                        "Kriging variogram model",
                                        ["Spherical", "Exponential", "Gaussian"],
                                        key="cv_kriging_model_v5",
                                    )
                                with krig_col2:
                                    cv_krig_neighbors = st.slider(
                                        "Kriging neighborhood",
                                        3,
                                        min(50, len(spatial_data)),
                                        min(20, len(spatial_data)),
                                        key="cv_kriging_neighbors_v5",
                                    )

                                total_pairs = len(spatial_data) * (len(spatial_data) - 1) // 2
                                cv_cloud = compute_variogram_cloud(
                                    spatial_data, spatial_x, spatial_y, spatial_value_column,
                                    max_pairs=min(50000, total_pairs),
                                )
                                cv_experimental = calculate_experimental_variogram(cv_cloud, n_lags=12) if cv_cloud is not None else None
                                cv_fitted = fit_variogram_model(cv_experimental, cv_model) if cv_experimental is not None else None

                                if cv_fitted is None:
                                    st.warning("A variogram model could not be fitted for Kriging validation.")
                                else:
                                    kriging_result = cross_validate_kriging(
                                        spatial_data, spatial_x, spatial_y, spatial_value_column,
                                        cv_fitted, max_neighbors=cv_krig_neighbors,
                                        max_validation_points=max_cv,
                                    )
                                    krig_metrics = summarize_cross_validation(kriging_result)
                                    if krig_metrics is not None:
                                        metrics_rows.append({"Method": "Ordinary Kriging", **krig_metrics})

                            if metrics_rows:
                                st.caption("Validation note: IDW is validated by leaving each target out. Kriging keeps the fitted variogram model fixed during holdout predictions, so this evaluates prediction conditional on that fitted model rather than refitting the variogram inside every fold.")
                                metrics_df = pd.DataFrame(metrics_rows)
                                st.write("### Validation Metrics")
                                st.dataframe(
                                    metrics_df.style.format({
                                        "MAE": "{:.6g}",
                                        "RMSE": "{:.6g}",
                                        "Mean Error": "{:.6g}",
                                        "R²": "{:.4f}",
                                    }),
                                    width="stretch",
                                    hide_index=True,
                                )

                                results_for_plot = {}
                                if idw_result is not None:
                                    results_for_plot["IDW"] = idw_result
                                if kriging_result is not None:
                                    results_for_plot["Ordinary Kriging"] = kriging_result
                                validation_fig = create_validation_plot(results_for_plot)
                                if validation_fig is not None:
                                    st.plotly_chart(validation_fig, width="stretch")

                                with st.expander("📖 How to read the validation results", expanded=False):
                                    st.markdown("- **MAE:** average absolute prediction error; lower is better.\n- **RMSE:** penalizes larger errors more strongly; lower is better.\n- **Mean Error:** average observed − predicted residual; values near zero indicate lower average bias.\n- **R²:** agreement measure that should be interpreted alongside MAE/RMSE and spatial diagnostics.\n- **1:1 plot:** points closer to the diagonal indicate closer observed/predicted agreement.")

                                st.write("### Validation Residuals")
                                residual_frames = []
                                for method_name, result in results_for_plot.items():
                                    if result is not None:
                                        temp = result.copy()
                                        temp.insert(0, "Method", method_name)
                                        residual_frames.append(temp)
                                if residual_frames:
                                    st.dataframe(
                                        pd.concat(residual_frames, ignore_index=True),
                                        width="stretch",
                                        hide_index=True,
                                    )

                elif spatial_analysis in ["IDW Interpolation", "Ordinary Kriging"]:
                    if spatial_value_column is None:
                        st.info("Select a numeric value variable different from X and Y before interpolation.")
                    else:
                        spatial_data = prepare_spatial_data(spatial_source_df, spatial_x, spatial_y, spatial_value_column)
                        if spatial_data is None or len(spatial_data) < 5:
                            st.warning("At least 5 valid spatial observations are recommended for interpolation.")
                        else:
                            grid_size = st.slider("Prediction grid resolution", 25, 100, 60, 5, key="prediction_grid_v4")
                            grid = create_prediction_grid(spatial_data, spatial_x, spatial_y, grid_size=grid_size)

                            if spatial_analysis == "IDW Interpolation":
                                power = st.slider("IDW power", 0.5, 5.0, 2.0, 0.5, key="idw_power_v4")
                                neighbors = st.slider("Maximum neighbors", 3, min(50, len(spatial_data)), min(20, len(spatial_data)), key="idw_neighbors_v4")
                                surface = idw_interpolate(
                                    spatial_data, spatial_x, spatial_y, spatial_value_column,
                                    grid[0], grid[1], power=power, max_neighbors=neighbors
                                )
                                fig = create_interpolation_map(
                                    grid[0], grid[1], surface, spatial_x, spatial_y,
                                    f"IDW Prediction — {spatial_value_column}", spatial_value_column, spatial_data
                                )
                                st.plotly_chart(fig, width="stretch")
                                st.info("IDW is a distance-weighted deterministic interpolator. It does not provide a model-based uncertainty estimate.")

                            else:
                                st.info("Ordinary Kriging uses the fitted experimental variogram and a local neighborhood of samples.")
                                total_possible_pairs = len(spatial_data) * (len(spatial_data) - 1) // 2
                                cloud = compute_variogram_cloud(spatial_data, spatial_x, spatial_y, spatial_value_column, max_pairs=min(50000, total_possible_pairs))
                                experimental = calculate_experimental_variogram(cloud, n_lags=12) if cloud is not None else None
                                model_choice = st.selectbox(
                                    "Kriging variogram model",
                                    ["Spherical", "Exponential", "Gaussian"],
                                    key="kriging_model_v4",
                                )
                                fitted = fit_variogram_model(experimental, model_choice) if experimental is not None else None
                                if fitted is None:
                                    st.warning("A variogram model could not be fitted. Adjust the data or use the Experimental Variogram view to inspect the variogram first.")
                                else:
                                    neighbors = st.slider("Kriging neighborhood", 3, min(50, len(spatial_data)), min(20, len(spatial_data)), key="kriging_neighbors_v4")
                                    result = ordinary_kriging_predict(
                                        spatial_data, spatial_x, spatial_y, spatial_value_column,
                                        grid[0], grid[1], fitted, max_neighbors=neighbors
                                    )
                                    if result is None:
                                        st.warning("Kriging could not be calculated for the selected data.")
                                    else:
                                        pred_col, var_col = st.columns(2)
                                        with pred_col:
                                            prediction_fig = create_interpolation_map(
                                                grid[0], grid[1], result["prediction"], spatial_x, spatial_y,
                                                f"Ordinary Kriging Prediction — {spatial_value_column}", spatial_value_column, spatial_data
                                            )
                                            st.plotly_chart(prediction_fig, width="stretch")
                                        with var_col:
                                            variance_fig = create_interpolation_map(
                                                grid[0], grid[1], result["variance"], spatial_x, spatial_y,
                                                "Kriging Variance / Uncertainty", "Kriging Variance", spatial_data
                                            )
                                            st.plotly_chart(variance_fig, width="stretch")
                                        st.caption(
                                            f"Fitted {fitted['model_name']} variogram — nugget={fitted['nugget']:.5g}, partial sill={fitted['sill']:.5g}, range={fitted['range']:.5g}, fit RMSE={fitted['rmse']:.5g}."
                                        )
                                        with st.expander("📖 Variogram & Kriging parameter guide", expanded=False):
                                            st.markdown("- **Nugget:** modeled short-scale variability and/or measurement error.\n- **Partial sill:** structured variance contribution represented by the fitted model.\n- **Range:** characteristic distance over which modeled spatial dependence approaches the sill.\n- **Kriging variance:** model-based uncertainty from spatial configuration and the fitted variogram; it is not the same as prediction error.\n- **Neighborhood:** limits how many nearby samples contribute to each local estimate.")
            else:
                st.info("At least two numeric variables are required for spatial analysis.")

            # ============================================================
            # NUMERIC SUMMARY
            # ============================================================

            st.subheader("📈 Numeric Statistical Summary")

            numeric_summary = get_numeric_summary(cleaned_df)

            if not numeric_summary.empty:
                st.dataframe(
                    numeric_summary,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No numeric variables are available.")

            # ============================================================
            # DATA PREVIEW
            # ============================================================

            st.subheader("👀 Data Preview")

            st.dataframe(
                cleaned_df.head(10),
                width="stretch",
                hide_index=True,
            )

        else:
            st.warning(
                "No numeric variables are available for analysis."
            )

    except Exception as e:
        st.error(
            f"❌ An error occurred while processing the dataset: {e}"
        )
