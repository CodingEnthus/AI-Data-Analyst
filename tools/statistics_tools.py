from langchain_core.tools import tool
import pandas as pd
from tools import helper_tools
import data_store
@tool
def calculate_statistics(column_name:str,operation:str):
     """
        Calculate a statistical measure on a specified column
        of the loaded dataset.
        Supported operations:
        mean, median, sum, min, max, count, std, variance.
    """
     operation=operation.lower()
     if(data_store.df is None):
            return "Dataset not loaded"
     operations = {
        "mean": lambda s: s.mean(),
        "median": lambda s: s.median(),
        "sum": lambda s: s.sum(),
        "min": lambda s: s.min(),
        "max": lambda s: s.max(),
        "count": lambda s: s.count(),
        "std": lambda s: s.std(),
        "variance": lambda s: s.var(),
        "mode": lambda s: s.mode().tolist(),
        "nunique": lambda s: s.nunique()
    }
     numeric_operations = {
        "mean",
        "median",
        "sum",
        "min",
        "max",
        "std",
        "variance"
    }
     if(operation.lower() not in operations):
          return f"Unsupported Operation:{operation}"
     match=helper_tools.find_best_matching_column.invoke({"column_name":column_name})
     if(not match["success"] or match["confidence"]<0.8):
          return (
                f"No suitable column found for '{column_name}'. "
                "Please check the column name or upload a different dataset."
           )
     matched_column=match["matched_column"]
     if(not pd.api.types.is_numeric_dtype(data_store.df[matched_column]) and operation in numeric_operations):
          return (
        f"Operation '{operation}' is not supported "
        f"for non-numeric column '{matched_column}'."
    )
     result=operations[operation](data_store.df[matched_column])
     return{
          "success":True,
          "column":matched_column,
          "operation":operation,
          "result":result
     }