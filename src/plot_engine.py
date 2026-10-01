import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


# ============================================================
# BASIC DISTRIBUTION PLOTS
# ============================================================

def create_histogram(
    data,
    bins=20,
    show_kde=True,
    show_mean=True,
    show_median=True
):
    """
    Create an interactive histogram with optional KDE,
    mean and median lines.
    """

    data = pd.Series(data).dropna()

    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=bins,
            histnorm=None,
            name="Histogram",
            opacity=0.75
        )
    )

    # KDE
    if show_kde and len(data) > 1 and data.nunique() > 1:

        x = np.linspace(
            data.min(),
            data.max(),
            300
        )

        kde = stats.gaussian_kde(data)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=kde(x) * len(data) *
                (data.max() - data.min()) / bins,
                mode="lines",
                name="KDE"
            )
        )

    # Mean
    if show_mean:

        fig.add_vline(
            x=data.mean(),
            line_dash="dash",
            annotation_text="Mean"
        )

    # Median
    if show_median:

        fig.add_vline(
            x=data.median(),
            line_dash="dot",
            annotation_text="Median"
        )

    fig.update_layout(
        title="Histogram",
        xaxis_title="Value",
        yaxis_title="Frequency",
        bargap=0.05
    )

    return fig


def create_box_plot(data):

    data = pd.Series(data).dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=data,
            name="Distribution",
            boxpoints="outliers"
        )
    )

    fig.update_layout(
        title="Box Plot",
        yaxis_title="Value"
    )

    return fig


def create_violin_plot(data):

    data = pd.Series(data).dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Violin(
            y=data,
            name="Distribution",
            box_visible=True,
            meanline_visible=True,
            points="outliers"
        )
    )

    fig.update_layout(
        title="Violin Plot",
        yaxis_title="Value"
    )

    return fig


# ============================================================
# EMPIRICAL CDF
# ============================================================

def create_ecdf(data):

    data = np.sort(
        pd.Series(data).dropna().values
    )

    y = np.arange(
        1,
        len(data) + 1
    ) / len(data)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data,
            y=y,
            mode="lines",
            name="ECDF"
        )
    )

    fig.update_layout(
        title="Empirical Cumulative Distribution Function",
        xaxis_title="Value",
        yaxis_title="Cumulative Probability",
        yaxis=dict(
            range=[0, 1]
        )
    )

    return fig


# ============================================================
# NORMAL DISTRIBUTION
# ============================================================

def create_normal_fit(data):

    data = pd.Series(data).dropna()

    mean = data.mean()
    std = data.std()

    if std == 0:
        return None

    x = np.linspace(
        data.min(),
        data.max(),
        300
    )

    pdf = stats.norm.pdf(
        x,
        mean,
        std
    )

    fig = go.Figure()

    # Histogram density
    fig.add_trace(
        go.Histogram(
            x=data,
            histnorm="probability density",
            nbinsx=20,
            name="Observed",
            opacity=0.65
        )
    )

    # Normal PDF
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode="lines",
            name="Fitted Normal PDF"
        )
    )

    fig.update_layout(
        title="Normal Distribution Fit",
        xaxis_title="Value",
        yaxis_title="Probability Density"
    )

    return fig


# ============================================================
# LOGNORMAL DISTRIBUTION
# ============================================================

def create_lognormal_fit(data):

    data = pd.Series(data).dropna()

    # Lognormal requires strictly positive values
    if (data <= 0).any():
        return None

    if data.nunique() <= 1:
        return None

    shape, loc, scale = stats.lognorm.fit(
        data,
        floc=0
    )

    x = np.linspace(
        data.min(),
        data.max(),
        300
    )

    pdf = stats.lognorm.pdf(
        x,
        shape,
        loc=loc,
        scale=scale
    )

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=data,
            histnorm="probability density",
            nbinsx=20,
            name="Observed",
            opacity=0.65
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode="lines",
            name="Fitted Lognormal PDF"
        )
    )

    fig.update_layout(
        title="Lognormal Distribution Fit",
        xaxis_title="Value",
        yaxis_title="Probability Density"
    )

    return fig


# ============================================================
# Q-Q PLOT
# ============================================================

def create_qq_plot(data, distribution="normal"):

    data = pd.Series(data).dropna()

    if len(data) < 3:
        return None

    sorted_data = np.sort(data)

    n = len(sorted_data)

    probabilities = (
        np.arange(1, n + 1) - 0.5
    ) / n

    if distribution == "normal":

        theoretical = stats.norm.ppf(
            probabilities
        )

        sample_mean = data.mean()
        sample_std = data.std()

        if sample_std == 0:
            return None

        theoretical_values = (
            sample_mean
            + theoretical * sample_std
        )

        title = "Normal Q-Q Plot"
        x_title = "Theoretical Normal Quantiles"

    else:

        if (data <= 0).any():
            return None

        shape, loc, scale = stats.lognorm.fit(
            data,
            floc=0
        )

        theoretical_values = stats.lognorm.ppf(
            probabilities,
            shape,
            loc=loc,
            scale=scale
        )

        title = "Lognormal Q-Q Plot"
        x_title = "Theoretical Lognormal Quantiles"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=theoretical_values,
            y=sorted_data,
            mode="markers",
            name="Observed"
        )
    )

    line_min = min(
        theoretical_values.min(),
        sorted_data.min()
    )

    line_max = max(
        theoretical_values.max(),
        sorted_data.max()
    )

    fig.add_trace(
        go.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode="lines",
            name="Reference"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Observed Quantiles"
    )

    return fig


# ============================================================
# PROBABILITY PLOT
# ============================================================

def create_probability_plot(
    data,
    distribution="normal"
):

    data = pd.Series(data).dropna()

    if len(data) < 3:
        return None

    if distribution == "normal":

        theoretical, ordered = stats.probplot(
            data,
            dist="norm",
            fit=False
        )

        title = "Normal Probability Plot"

    else:

        if (data <= 0).any():
            return None

        shape, loc, scale = stats.lognorm.fit(
            data,
            floc=0
        )

        probabilities = (
            np.arange(1, len(data) + 1)
            - 0.5
        ) / len(data)

        theoretical = stats.lognorm.ppf(
            probabilities,
            shape,
            loc=loc,
            scale=scale
        )

        ordered = np.sort(data)

        title = "Lognormal Probability Plot"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=ordered,
            mode="markers",
            name="Observed"
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Observed Values"
    )

    return fig

# ============================================================
# DISTRIBUTION ANALYSIS V2
# ============================================================

def fit_distribution(data, distribution="normal"):
    """
    Fit a probability distribution and calculate
    goodness-of-fit diagnostics.
    """

    data = pd.Series(data).dropna()

    if len(data) < 3:
        return None

    if distribution == "normal":

        mean = data.mean()
        std = data.std()

        if std == 0:
            return None

        params = {
            "Mean": mean,
            "Std Dev": std
        }

        # Log likelihood
        log_likelihood = np.sum(
            stats.norm.logpdf(
                data,
                loc=mean,
                scale=std
            )
        )

        # Explicit CDF function
        def fitted_cdf(x):
            return stats.norm.cdf(
                x,
                loc=mean,
                scale=std
            )

        # KS test
        ks_stat, ks_pvalue = stats.kstest(
            data,
            fitted_cdf
        )

        num_parameters = 2

        distribution_name = "Normal"

    elif distribution == "lognormal":

        # Lognormal requires strictly positive values
        if (data <= 0).any():
            return None

        shape, loc, scale = stats.lognorm.fit(
            data,
            floc=0
        )

        params = {
            "Shape": shape,
            "Location": loc,
            "Scale": scale
        }

        # Log likelihood
        log_likelihood = np.sum(
            stats.lognorm.logpdf(
                data,
                s=shape,
                loc=loc,
                scale=scale
            )
        )

        # Explicit CDF function
        def fitted_cdf(x):
            return stats.lognorm.cdf(
                x,
                s=shape,
                loc=loc,
                scale=scale
            )

        # KS test
        ks_stat, ks_pvalue = stats.kstest(
            data,
            fitted_cdf
        )

        num_parameters = 3

        distribution_name = "Lognormal"

    else:

        return None

    n = len(data)

    aic = (
        2 * num_parameters
        - 2 * log_likelihood
    )

    return {
        "Distribution": distribution_name,
        "Parameters": params,
        "KS Statistic": ks_stat,
        "KS p-value": ks_pvalue,
        "Log Likelihood": log_likelihood,
        "AIC": aic
    }

def create_pdf_cdf_plot(
    data,
    distribution="normal"
):
    """
    Create a combined PDF and CDF visualization.
    """

    data = pd.Series(data).dropna()

    if len(data) < 3:
        return None

    if distribution == "normal":

        mean = data.mean()
        std = data.std()

        if std == 0:
            return None

        x = np.linspace(
            data.min(),
            data.max(),
            300
        )

        pdf = stats.norm.pdf(
            x,
            mean,
            std
        )

        cdf = stats.norm.cdf(
            x,
            mean,
            std
        )

        distribution_name = "Normal"

    elif distribution == "lognormal":

        if (data <= 0).any():
            return None

        shape, loc, scale = stats.lognorm.fit(
            data,
            floc=0
        )

        x = np.linspace(
            data.min(),
            data.max(),
            300
        )

        pdf = stats.lognorm.pdf(
            x,
            shape,
            loc=loc,
            scale=scale
        )

        cdf = stats.lognorm.cdf(
            x,
            shape,
            loc=loc,
            scale=scale
        )

        distribution_name = "Lognormal"

    else:

        return None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode="lines",
            name="PDF"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=cdf,
            mode="lines",
            name="CDF",
            yaxis="y2"
        )
    )

    fig.update_layout(
        title=f"{distribution_name} PDF & CDF",
        xaxis_title="Value",
        yaxis=dict(
            title="Probability Density"
        ),
        yaxis2=dict(
            title="Cumulative Probability",
            overlaying="y",
            side="right",
            range=[0, 1]
        )
    )

    return fig


def create_log_transform(data):

    data = pd.Series(data).dropna()

    if (data <= 0).any():
        return None

    transformed = np.log(data)

    return transformed


def create_log_transform_plot(data):

    transformed = create_log_transform(data)

    if transformed is None:
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=transformed,
            histnorm="probability density",
            nbinsx=20,
            name="ln(X)",
            opacity=0.7
        )
    )

    mean = transformed.mean()

    std = transformed.std()

    if std > 0:

        x = np.linspace(
            transformed.min(),
            transformed.max(),
            300
        )

        pdf = stats.norm.pdf(
            x,
            mean,
            std
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=pdf,
                mode="lines",
                name="Normal Fit"
            )
        )

    fig.update_layout(
        title="Log-Transformed Distribution: ln(X)",
        xaxis_title="ln(X)",
        yaxis_title="Probability Density"
    )

    return fig


def create_distribution_comparison(data):

    normal = fit_distribution(
        data,
        "normal"
    )

    lognormal = fit_distribution(
        data,
        "lognormal"
    )

    rows = []

    if normal is not None:

        rows.append({
            "Distribution": "Normal",
            "KS Statistic": normal["KS Statistic"],
            "KS p-value": normal["KS p-value"],
            "AIC": normal["AIC"]
        })

    if lognormal is not None:

        rows.append({
            "Distribution": "Lognormal",
            "KS Statistic": lognormal["KS Statistic"],
            "KS p-value": lognormal["KS p-value"],
            "AIC": lognormal["AIC"]
        })

    return pd.DataFrame(rows)


def create_fit_plot(
    data,
    distribution="normal"
):

    data = pd.Series(data).dropna()

    if len(data) < 3:
        return None

    if distribution == "normal":

        mean = data.mean()

        std = data.std()

        if std == 0:
            return None

        x = np.linspace(
            data.min(),
            data.max(),
            300
        )

        pdf = stats.norm.pdf(
            x,
            mean,
            std
        )

        name = "Normal"

    elif distribution == "lognormal":

        if (data <= 0).any():
            return None

        shape, loc, scale = stats.lognorm.fit(
            data,
            floc=0
        )

        x = np.linspace(
            data.min(),
            data.max(),
            300
        )

        pdf = stats.lognorm.pdf(
            x,
            shape,
            loc=loc,
            scale=scale
        )

        name = "Lognormal"

    else:

        return None

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=data,
            histnorm="probability density",
            nbinsx=20,
            name="Observed",
            opacity=0.65
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode="lines",
            name=f"{name} PDF"
        )
    )

    fig.update_layout(
        title=f"{name} Distribution Fit",
        xaxis_title="Value",
        yaxis_title="Probability Density"
    )

    return fig