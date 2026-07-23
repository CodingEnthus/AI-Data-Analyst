from langchain_core.tools import tool
import pandas as pd
import os
import difflib
from difflib import SequenceMatcher
import data_store
@tool
def load_dataset(file_path:str):
    """
    Load a CSV or Excel dataset into memory.

    Args:
        file_path: Path to the dataset file.

    Returns:
        Success or failure message.
    """
    try:
        extensions=os.path.splitext(file_path)[1].lower()
        if(extensions==".csv"):
                data_store.df=pd.read_csv(file_path)
        elif(extensions==".xlsx"):
           data_store.df=pd.read_excel(file_path)
        else: 
            return "Unsupported file. Please upload csv or excel file."
        return f"""Dataset Uploaded successfully.
        Rows: {data_store.df.shape[0]}
        Columns: {data_store.df.shape[1]}

        Available Columns:
        {", ".join(data_store.df.columns)}
        """
    except Exception as e:
        return f"Error loading dataset: {e}"
@tool
def get_dataset_info():
      """
    Retrieve basic information about the currently loaded dataset.

    Returns:
        A dictionary containing:
        - Number of rows
        - Number of columns
        - Column names
        - Data types of each column
        - Number of missing values in each column

        If no dataset is loaded, returns an error message.
    """

      if(data_store.df is None):
          return "No Dataset Loaded"
      return f""" 
            Rows:{data_store.df.shape[0]}
            Column:{data_store.df.shape[1]}
            Column_names:{", ".join(data_store.df.columns)}
            Data_types:{data_store.df.dtypes.to_string()}
            Missing_columns:{data_store.df.isna().sum().to_string()}
            """