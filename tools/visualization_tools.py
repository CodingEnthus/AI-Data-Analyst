from langchain_core.tools import tool
import plotly.express as px
import pandas as pd

import data_store

from tools.helper_tools import (
    validate_chart_column,
    save_chart,
)


@tool
def create_bar_chart(x_column: str, y_column: str):
    """
    Create a bar chart using two dataset columns.

    Args:
        x_column: Column for the X-axis.
        y_column: Numeric column for the Y-axis.

    Returns:
        Metadata about the generated chart or an error.
    """

    if data_store.df is None:
        return {
            "success": False,
            "message": "No dataset loaded."
        }

    validation = validate_chart_column(
        x_column,
        y_column
    )

    if not validation["success"]:
        return validation

    matched_x = validation["matched_x"]
    matched_y = validation["matched_y"]

    if not pd.api.types.is_numeric_dtype(
        data_store.df[matched_y]
    ):
        return {
            "success": False,
            "message": f"'{matched_y}' must be numeric."
        }

    fig = px.bar(
        data_store.df,
        x=matched_x,
        y=matched_y,
        title=f"{matched_y} by {matched_x}"
    )

    chart_path = save_chart(
        fig,
        "bar"
    )

    return {
        "success": True,
        "chart_type": "bar",
        "x_column": matched_x,
        "y_column": matched_y,
        "chart_path": chart_path
    }
@tool
def create_line_chart(x_column: str, y_column: str):
    """
    Create a line chart using two dataset columns.

    Args:
        x_column: Column for the X-axis.
        y_column: Numeric column for the Y-axis.

    Returns:
        Metadata about the generated chart or an error.
    """

    if data_store.df is None:
        return {
            "success": False,
            "message": "No dataset loaded."
        }

    validation = validate_chart_column(
        x_column,
        y_column
    )

    if not validation["success"]:
        return validation

    matched_x = validation["matched_x"]
    matched_y = validation["matched_y"]

    if not pd.api.types.is_numeric_dtype(
        data_store.df[matched_y]
    ):
        return {
            "success": False,
            "message": f"'{matched_y}' must be numeric."
        }

    fig = px.line(
        data_store.df,
        x=matched_x,
        y=matched_y,
        title=f"{matched_y} by {matched_x}"
    )

    chart_path = save_chart(
        fig,
        "line"
    )

    return {
        "success": True,
        "chart_type": "line",
        "x_column": matched_x,
        "y_column": matched_y,
        "chart_path": chart_path
    }


@tool
def create_scatter_plot(x_column: str, y_column: str):
    """
    Create a scatter plot using two numeric dataset columns.

    Args:
        x_column: Numeric column for the X-axis.
        y_column: Numeric column for the Y-axis.

    Returns:
        Metadata about the generated chart or an error.
    """

    if data_store.df is None:
        return {
            "success": False,
            "message": "No dataset loaded."
        }

    validation = validate_chart_column(
        x_column,
        y_column
    )

    if not validation["success"]:
        return validation

    matched_x = validation["matched_x"]
    matched_y = validation["matched_y"]

    if not pd.api.types.is_numeric_dtype(
        data_store.df[matched_x]
    ):
        return {
            "success": False,
            "message": f"'{matched_x}' must be numeric."
        }

    if not pd.api.types.is_numeric_dtype(
        data_store.df[matched_y]
    ):
        return {
            "success": False,
            "message": f"'{matched_y}' must be numeric."
        }

    fig = px.scatter(
        data_store.df,
        x=matched_x,
        y=matched_y,
        title=f"{matched_y} vs {matched_x}"
    )

    chart_path = save_chart(
        fig,
        "scatter"
    )

    return {
        "success": True,
        "chart_type": "scatter",
        "x_column": matched_x,
        "y_column": matched_y,
        "chart_path": chart_path
    }


@tool
def create_histogram(x_column: str):
    """
    Create a histogram using one numeric dataset column.

    Args:
        x_column: Numeric column for the X-axis.

    Returns:
        Metadata about the generated chart or an error.
    """

    if data_store.df is None:
        return {
            "success": False,
            "message": "No dataset loaded."
        }

    validation = validate_chart_column(
        x_column
    )

    if not validation["success"]:
        return validation

    matched_x = validation["matched_x"]

    if not pd.api.types.is_numeric_dtype(
        data_store.df[matched_x]
    ):
        return {
            "success": False,
            "message": f"'{matched_x}' must be numeric."
        }

    fig = px.histogram(
        data_store.df,
        x=matched_x,
        title=f"Distribution of {matched_x}"
    )

    chart_path = save_chart(
        fig,
        "histogram"
    )

    return {
        "success": True,
        "chart_type": "histogram",
        "x_column": matched_x,
        "chart_path": chart_path
    }
@tool
def create_pie_chart(label_column: str, value_column: str):
    """
    Create a pie chart using one categorical column and one numeric column.

    Args:
        label_column: Column containing category labels.
        value_column: Numeric column containing values.

    Returns:
        Metadata about the generated chart or an error.
    """

    if data_store.df is None:
        return {
            "success": False,
            "message": "No dataset loaded."
        }

    validation = validate_chart_column(
        label_column,
        value_column
    )

    if not validation["success"]:
        return validation

    matched_label = validation["matched_x"]
    matched_value = validation["matched_y"]

    if not pd.api.types.is_numeric_dtype(
        data_store.df[matched_value]
    ):
        return {
            "success": False,
            "message": f"'{matched_value}' must be numeric."
        }

    # Aggregate duplicate categories
    plot_df = (
        data_store.df
        .groupby(matched_label, as_index=False)[matched_value]
        .sum()
    )

    fig = px.pie(
        plot_df,
        names=matched_label,
        values=matched_value,
        title=f"{matched_value} by {matched_label}"
    )

    chart_path = save_chart(
        fig,
        "pie"
    )

    return {
        "success": True,
        "chart_type": "pie",
        "label_column": matched_label,
        "value_column": matched_value,
        "chart_path": chart_path
    }


@tool
def create_heatmap():
    """
    Create a correlation heatmap using all numeric columns in the dataset.

    Returns:
        Metadata about the generated heatmap or an error.
    """

    if data_store.df is None:
        return {
            "success": False,
            "message": "No dataset loaded."
        }

    numeric_df = data_store.df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return {
            "success": False,
            "message": (
                "At least two numeric columns are required "
                "to create a heatmap."
            )
        }

    correlation_matrix = numeric_df.corr()

    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

    chart_path = save_chart(
        fig,
        "heatmap"
    )

    return {
        "success": True,
        "chart_type": "heatmap",
        "columns": list(numeric_df.columns),
        "chart_path": chart_path
    }


@tool
def create_box_plot(numeric_column: str, category_column: str = ""):
    """
    Create a box plot to visualize the distribution and outliers of a numeric column,
    optionally grouped by a categorical column.

    Args:
        numeric_column: The numeric column to analyze for distributions/outliers (e.g., 'Age', 'Fare').
        category_column: Optional categorical column to group boxes by (e.g., 'Pclass', 'Sex').

    Returns:
        Metadata about the generated chart including 'chart_path' or an error message.
    """
    if data_store.df is None:
        return {
            "success": False,
            "message": "No dataset loaded."
        }

    category_col = category_column.strip() if category_column else None

    if category_col:
        validation = validate_chart_column(category_col, numeric_column)
        if not validation["success"]:
            return validation
        matched_x = validation["matched_x"]
        matched_y = validation["matched_y"]
    else:
        validation = validate_chart_column(numeric_column)
        if not validation["success"]:
            return validation
        matched_x = None
        matched_y = validation["matched_x"]

    # If swapped, auto-swap them
    if matched_x and not pd.api.types.is_numeric_dtype(data_store.df[matched_y]) and pd.api.types.is_numeric_dtype(data_store.df[matched_x]):
        matched_x, matched_y = matched_y, matched_x

    if not pd.api.types.is_numeric_dtype(data_store.df[matched_y]):
        return {
            "success": False,
            "message": f"'{matched_y}' must be a numeric column for a box plot."
        }

    if matched_x:
        fig = px.box(
            data_store.df,
            x=matched_x,
            y=matched_y,
            points="outliers",
            title=f"Box Plot of {matched_y} by {matched_x}"
        )
    else:
        fig = px.box(
            data_store.df,
            y=matched_y,
            points="outliers",
            title=f"Box Plot of {matched_y}"
        )

    chart_path = save_chart(fig, "box")

    return {
        "success": True,
        "chart_type": "box",
        "numeric_column": matched_y,
        "category_column": matched_x,
        "chart_path": chart_path
    }


@tool
def create_violin_plot(numeric_column: str, category_column: str = ""):
    """
    Create a violin plot to display the probability density and distribution of a numeric column,
    optionally grouped by a categorical column.

    Args:
        numeric_column: The numeric column to analyze (e.g., 'Age', 'Fare').
        category_column: Optional categorical column to group violins by (e.g., 'Pclass', 'Sex').

    Returns:
        Metadata about the generated chart including 'chart_path' or an error message.
    """
    if data_store.df is None:
        return {
            "success": False,
            "message": "No dataset loaded."
        }

    category_col = category_column.strip() if category_column else None

    if category_col:
        validation = validate_chart_column(category_col, numeric_column)
        if not validation["success"]:
            return validation
        matched_x = validation["matched_x"]
        matched_y = validation["matched_y"]
    else:
        validation = validate_chart_column(numeric_column)
        if not validation["success"]:
            return validation
        matched_x = None
        matched_y = validation["matched_x"]

    # If swapped, auto-swap them
    if matched_x and not pd.api.types.is_numeric_dtype(data_store.df[matched_y]) and pd.api.types.is_numeric_dtype(data_store.df[matched_x]):
        matched_x, matched_y = matched_y, matched_x

    if not pd.api.types.is_numeric_dtype(data_store.df[matched_y]):
        return {
            "success": False,
            "message": f"'{matched_y}' must be a numeric column for a violin plot."
        }

    if matched_x:
        fig = px.violin(
            data_store.df,
            x=matched_x,
            y=matched_y,
            box=True,
            points="all",
            title=f"Violin Plot of {matched_y} by {matched_x}"
        )
    else:
        fig = px.violin(
            data_store.df,
            y=matched_y,
            box=True,
            points="all",
            title=f"Violin Plot of {matched_y}"
        )

    chart_path = save_chart(fig, "violin")

    return {
        "success": True,
        "chart_type": "violin",
        "numeric_column": matched_y,
        "category_column": matched_x,
        "chart_path": chart_path
    }

