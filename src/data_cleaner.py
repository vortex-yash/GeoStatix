import pandas as pd


def remove_columns(df, columns_to_remove):
    """
    Remove user-selected columns from the dataset.
    """

    if not columns_to_remove:
        return df.copy()

    return df.drop(
        columns=columns_to_remove,
        errors="ignore"
    )


def remove_duplicate_rows(df):
    """
    Remove completely duplicated rows.
    """

    return df.drop_duplicates().reset_index(drop=True)


def remove_missing_rows(df):
    """
    Remove rows containing missing values.
    """

    return df.dropna().reset_index(drop=True)


def get_cleaning_summary(original_df, cleaned_df):
    """
    Compare original and cleaned datasets.
    """

    return {
        "original_rows": len(original_df),
        "remaining_rows": len(cleaned_df),
        "rows_removed": (
            len(original_df) - len(cleaned_df)
        ),
        "original_columns": len(original_df.columns),
        "remaining_columns": len(cleaned_df.columns),
        "columns_removed": (
            len(original_df.columns)
            - len(cleaned_df.columns)
        ),
    }