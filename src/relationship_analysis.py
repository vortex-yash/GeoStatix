import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy import stats


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def calculate_correlations(x, y):
    """
    Calculate Pearson, Spearman and Kendall correlations.
    """

    data = pd.DataFrame({
        "x": pd.Series(x),
        "y": pd.Series(y)
    }).dropna()

    if len(data) < 3:
        return None

    x_clean = data["x"]
    y_clean = data["y"]

    pearson_r, pearson_p = stats.pearsonr(
        x_clean,
        y_clean
    )

    spearman_r, spearman_p = stats.spearmanr(
        x_clean,
        y_clean
    )

    kendall_tau, kendall_p = stats.kendalltau(
        x_clean,
        y_clean
    )

    return {
        "Pearson": {
            "Correlation": pearson_r,
            "p-value": pearson_p
        },
        "Spearman": {
            "Correlation": spearman_r,
            "p-value": spearman_p
        },
        "Kendall": {
            "Correlation": kendall_tau,
            "p-value": kendall_p
        }
    }


# ============================================================
# LINEAR REGRESSION
# ============================================================

def calculate_regression(x, y):
    """
    Calculate simple linear regression.
    """

    data = pd.DataFrame({
        "x": pd.Series(x),
        "y": pd.Series(y)
    }).dropna()

    if len(data) < 3:
        return None

    x_clean = data["x"].values
    y_clean = data["y"].values

    if np.std(x_clean) == 0:
        return None

    result = stats.linregress(
        x_clean,
        y_clean
    )

    slope = result.slope
    intercept = result.intercept
    r_value = result.rvalue
    p_value = result.pvalue
    std_err = result.stderr

    r_squared = r_value ** 2

    predictions = (
        slope * x_clean
        + intercept
    )

    residuals = (
        y_clean
        - predictions
    )

    return {
        "slope": slope,
        "intercept": intercept,
        "r": r_value,
        "r_squared": r_squared,
        "p_value": p_value,
        "std_error": std_err,
        "predictions": predictions,
        "residuals": residuals,
        "x": x_clean,
        "y": y_clean
    }


# ============================================================
# SCATTER PLOT
# ============================================================

def create_scatter_plot(
    x,
    y,
    x_name,
    y_name,
    add_regression=True
):
    """
    Create an interactive scatter plot.
    """

    data = pd.DataFrame({
        "x": pd.Series(x),
        "y": pd.Series(y)
    }).dropna()

    if len(data) < 2:
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers",
            name="Observed"
        )
    )

    if add_regression and len(data) >= 3:

        regression = calculate_regression(
            data["x"],
            data["y"]
        )

        if regression is not None:

            x_line = np.linspace(
                data["x"].min(),
                data["x"].max(),
                200
            )

            y_line = (
                regression["slope"]
                * x_line
                + regression["intercept"]
            )

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Linear Regression"
                )
            )

    fig.update_layout(
        title=f"{y_name} vs {x_name}",
        xaxis_title=x_name,
        yaxis_title=y_name
    )

    return fig


# ============================================================
# RESIDUAL PLOT
# ============================================================

def create_residual_plot(
    x,
    y,
    x_name
):
    """
    Plot regression residuals against predictor values.
    """

    regression = calculate_regression(
        x,
        y
    )

    if regression is None:
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=regression["x"],
            y=regression["residuals"],
            mode="markers",
            name="Residuals"
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="Zero residual"
    )

    fig.update_layout(
        title="Regression Residual Plot",
        xaxis_title=x_name,
        yaxis_title="Residual"
    )

    return fig


# ============================================================
# CORRELATION TABLE
# ============================================================

def create_correlation_table(x, y):

    correlations = calculate_correlations(
        x,
        y
    )

    if correlations is None:
        return None

    rows = []

    for method, values in correlations.items():

        rows.append({
            "Method": method,
            "Correlation": values["Correlation"],
            "p-value": values["p-value"]
        })

    return pd.DataFrame(rows)


# ============================================================
# REGRESSION SUMMARY
# ============================================================

def create_regression_summary(x, y):

    regression = calculate_regression(
        x,
        y
    )

    if regression is None:
        return None

    return {
        "Slope": regression["slope"],
        "Intercept": regression["intercept"],
        "R": regression["r"],
        "R²": regression["r_squared"],
        "p-value": regression["p_value"],
        "Standard Error": regression["std_error"],
        "Observations": len(regression["x"])
    }

# ============================================================
# CORRELATION HEATMAP
# ============================================================

def create_correlation_heatmap(df, method="pearson"):
    """
    Create a correlation heatmap for all numeric variables.
    """

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] < 2:
        return None

    correlation_matrix = numeric_df.corr(
        method=method
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            text=np.round(
                correlation_matrix.values,
                2
            ),
            texttemplate="%{text}",
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title="Correlation"
            )
        )
    )

    fig.update_layout(
        title=f"{method.title()} Correlation Heatmap",
        xaxis_title="Variables",
        yaxis_title="Variables"
    )

    return fig


# ============================================================
# PAIRWISE CORRELATION SCREENING
# ============================================================

def create_pairwise_correlation_table(df):
    """
    Calculate pairwise Pearson and Spearman correlations
    for all numeric variables.
    """

    numeric_columns = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    if len(numeric_columns) < 2:
        return None

    rows = []

    for i in range(len(numeric_columns)):

        for j in range(i + 1, len(numeric_columns)):

            x_name = numeric_columns[i]
            y_name = numeric_columns[j]

            data = df[
                [x_name, y_name]
            ].dropna()

            if len(data) < 3:
                continue

            x = data[x_name]
            y = data[y_name]

            if x.nunique() < 2 or y.nunique() < 2:
                continue

            pearson_r, pearson_p = stats.pearsonr(
                x,
                y
            )

            spearman_r, spearman_p = stats.spearmanr(
                x,
                y
            )

            rows.append({
                "Variable 1": x_name,
                "Variable 2": y_name,
                "Observations": len(data),
                "Pearson r": pearson_r,
                "Pearson p-value": pearson_p,
                "Spearman ρ": spearman_r,
                "Spearman p-value": spearman_p
            })

    if not rows:
        return None

    result = pd.DataFrame(rows)

    return result


# ============================================================
# REGRESSION ERROR METRICS
# ============================================================

def calculate_regression_metrics(x, y):
    """
    Calculate additional regression error diagnostics.
    """

    regression = calculate_regression(
        x,
        y
    )

    if regression is None:
        return None

    observed = regression["y"]
    predicted = regression["predictions"]
    residuals = regression["residuals"]

    mse = np.mean(
        residuals ** 2
    )

    rmse = np.sqrt(
        mse
    )

    mae = np.mean(
        np.abs(residuals)
    )

    residual_std = np.std(
        residuals,
        ddof=1
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "Residual Std Dev": residual_std
    }


# ============================================================
# PREDICTED VS OBSERVED
# ============================================================

def create_predicted_vs_observed_plot(
    x,
    y,
    x_name,
    y_name
):
    """
    Plot observed values against regression predictions.
    """

    regression = calculate_regression(
        x,
        y
    )

    if regression is None:
        return None

    observed = regression["y"]
    predicted = regression["predictions"]

    min_value = min(
        observed.min(),
        predicted.min()
    )

    max_value = max(
        observed.max(),
        predicted.max()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=observed,
            y=predicted,
            mode="markers",
            name="Predictions"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="1:1 Reference"
        )
    )

    fig.update_layout(
        title="Predicted vs Observed",
        xaxis_title=f"Observed {y_name}",
        yaxis_title=f"Predicted {y_name}"
    )

    return fig