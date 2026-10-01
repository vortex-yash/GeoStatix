import pandas as pd
import numpy as np


def get_dataset_profile(df):
    """
    Generate a comprehensive profile of the dataset.
    """

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    missing_by_column = df.isna().sum()

    duplicate_rows = int(
        df.duplicated().sum()
    )

    # Detect columns created by Excel/index exports
    unnamed_columns = [
        column
        for column in df.columns
        if str(column).strip().lower().startswith("unnamed")
    ]

    # Detect numeric columns containing only one unique value
    constant_columns = [
        column
        for column in numeric_columns
        if df[column].nunique(dropna=True) <= 1
    ]

    # Count zeros
    zero_counts = {}

    for column in numeric_columns:
        zero_counts[column] = int(
            (df[column] == 0).sum()
        )

    # Count negative values
    negative_counts = {}

    for column in numeric_columns:
        negative_counts[column] = int(
            (df[column] < 0).sum()
        )

    # Detect likely log-transformed variables
    log_columns = []

    for column in df.columns:

        column_name = str(column).lower()

        if (
            "ln " in column_name
            or column_name.startswith("ln_")
            or column_name.startswith("ln")
            or "log" in column_name
            or "natural log" in column_name
        ):
            log_columns.append(column)

    return {
        "rows": len(df),

        "columns": len(df.columns),

        "numeric_columns": numeric_columns,

        "categorical_columns": categorical_columns,

        "missing_cells": int(
            df.isna().sum().sum()
        ),

        "missing_by_column": missing_by_column,

        "duplicate_rows": duplicate_rows,

        "unnamed_columns": unnamed_columns,

        "constant_columns": constant_columns,

        "zero_counts": zero_counts,

        "negative_counts": negative_counts,

        "log_columns": log_columns,
    }


def get_numeric_summary(df):
    """
    Generate statistical summary for numeric variables.
    """

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.empty:
        return pd.DataFrame()

    rows = []

    for column in numeric_df.columns:

        series = numeric_df[column].dropna()

        if series.empty:
            continue

        mean = series.mean()

        std = series.std()

        # Coefficient of variation
        # is only meaningful for a positive mean.
        if mean > 0:
            cv = std / mean
        else:
            cv = np.nan

        rows.append({
            "Variable": column,

            "Count": len(series),

            "Mean": mean,

            "Median": series.median(),

            "Std Dev": std,

            "Variance": series.var(),

            "Minimum": series.min(),

            "Maximum": series.max(),

            "Range": (
                series.max()
                - series.min()
            ),

            "CV": cv,

            "Skewness": series.skew(),

            "Kurtosis": series.kurt(),

            "Q1": series.quantile(0.25),

            "Q3": series.quantile(0.75),

            "IQR": (
                series.quantile(0.75)
                - series.quantile(0.25)
            ),

            "Zeros": int(
                (series == 0).sum()
            ),

            "Negative Values": int(
                (series < 0).sum()
            ),
        })

    return pd.DataFrame(rows)