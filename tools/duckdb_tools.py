from langchain_core.tools import tool
import duckdb
import pandas as pd
import re
import data_store


def _format_dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Format pandas dataframe into a markdown table without external dependencies."""
    preview_df = df.head(max_rows)
    headers = [str(col) for col in preview_df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for _, row in preview_df.iterrows():
        row_vals = []
        for val in row:
            if pd.isna(val):
                row_vals.append("NULL")
            elif isinstance(val, float) and val.is_integer():
                row_vals.append(str(int(val)))
            elif isinstance(val, float):
                row_vals.append(f"{val:.4f}".rstrip("0").rstrip("."))
            else:
                row_vals.append(str(val).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(row_vals) + " |")
    return "\n".join(lines)


@tool
def query_dataset_with_sql(query: str) -> str:
    """
    Execute a read-only SQL query on the loaded dataset using DuckDB.
    The loaded dataset is available in SQL queries as table 'df' or 'dataset'.

    Use this tool for:
    - Complex filtering, sorting, and slicing records (e.g., WHERE, ORDER BY, LIMIT)
    - Multi-column aggregations and GROUP BY queries (e.g., AVG, SUM, COUNT per group)
    - Window functions, top-N queries, subqueries, and CTEs (WITH clauses)
    - Direct user SQL requests

    Args:
        query: A SQL SELECT statement. Examples:
               - "SELECT Pclass, COUNT(*) as count, AVG(Age) as avg_age FROM df GROUP BY Pclass"
               - "SELECT Name, Age, Fare FROM df WHERE Survived = 1 ORDER BY Fare DESC LIMIT 5"

    Returns:
        Formatted markdown table of query results or an informative error message.
    """
    if data_store.df is None:
        return "Error: No dataset loaded. Please ask the user to upload or load a dataset first."

    cleaned_query = query.strip()

    # Safety check: allow only read-only statements
    disallowed_keywords = [
        r"\bDROP\b", r"\bDELETE\b", r"\bINSERT\b", r"\bUPDATE\b",
        r"\bALTER\b", r"\bCREATE\b", r"\bATTACH\b", r"\bDETACH\b",
        r"\bCOPY\b", r"\bEXPORT\b", r"\bIMPORT\b", r"\bPRAGMA\b"
    ]
    for pattern in disallowed_keywords:
        if re.search(pattern, cleaned_query, re.IGNORECASE):
            return f"Security Error: Modifying statements or administrative commands are not allowed. Only read-only SELECT queries are supported."

    try:
        # Create an in-memory connection and register the DataFrame
        con = duckdb.connect(database=":memory:")
        con.register("df", data_store.df)
        con.register("dataset", data_store.df)

        result_df = con.execute(cleaned_query).fetchdf()

        if result_df.empty:
            return "Query executed successfully, but returned 0 rows."

        total_rows = len(result_df)
        max_display_rows = 50
        table_md = _format_dataframe_to_markdown(result_df, max_rows=max_display_rows)

        if total_rows > max_display_rows:
            return (
                f"**Query Results (showing first {max_display_rows} of {total_rows:,} rows):**\n\n"
                f"{table_md}\n\n"
                f"*Tip: Use `LIMIT` or aggregation if you need a smaller summary.*"
            )
        else:
            return f"**Query Results ({total_rows} row{'s' if total_rows != 1 else ''}):**\n\n{table_md}"

    except Exception as e:
        available_cols = ", ".join(list(data_store.df.columns))
        return (
            f"DuckDB SQL Error: {str(e)}\n\n"
            f"Available columns in dataset: {available_cols}"
        )
