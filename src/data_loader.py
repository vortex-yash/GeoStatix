import pandas as pd


def load_dataset(uploaded_file):
    """
    Load a CSV or Excel file into a pandas DataFrame.

    Parameters
    ----------
    uploaded_file : Streamlit UploadedFile
        File uploaded through the Streamlit interface.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    file_name = uploaded_file.name.lower()

    try:

        # CSV files
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        # Excel files
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        else:
            raise ValueError(
                "Unsupported file format. Please upload a CSV or XLSX file."
            )

        return df

    except Exception as e:
        raise ValueError(f"Could not read the dataset: {e}")