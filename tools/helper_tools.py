from langchain_core.tools import tool
import difflib
from difflib import SequenceMatcher
import data_store
import os
import uuid


@tool
def find_best_matching_column(column_name: str):
    """
    Find the closest matching column in the loaded dataset.

    Returns:
        {
            "success": bool,
            "requested_column": str,
            "matched_column": str | None,
            "confidence": float
        }
    """

    if data_store.df is None:
        return {
            "success": False,
            "message": "No dataset loaded."
        }

    original_columns = list(data_store.df.columns)

    # Mapping lowercase -> original names
    column_map = {col.lower(): col for col in original_columns}

    matches = difflib.get_close_matches(
        column_name.lower(),
        column_map.keys(),
        n=1,
        cutoff=0.6
    )

    if not matches:
        return {
            "success": False,
            "requested_column": column_name,
            "matched_column": None,
            "confidence": 0.0
        }

    matched_lower = matches[0]
    matched_original = column_map[matched_lower]

    confidence = SequenceMatcher(
        None,
        column_name.lower(),
        matched_lower
    ).ratio()

    return {
        "success": True,
        "requested_column": column_name,
        "matched_column": matched_original,
        "confidence": round(confidence, 2)
    }


def validate_chart_column(x_column: str, y_column: str = None):
    """
    Validate and resolve chart column names.

    Args:
        x_column: X-axis column or the only column required.
        y_column: Optional Y-axis column.

    Returns:
        Dictionary containing matched column names or an error.
    """

    x_match = find_best_matching_column.invoke({
        "column_name": x_column
    })

    if (
        not x_match["success"] or
        x_match["confidence"] < 0.8
    ):
        return {
            "success": False,
            "message": f"Could not find a suitable column for '{x_column}'."
        }

    matched_x = x_match["matched_column"]

    # Single-column chart (Histogram, Pie)
    if y_column is None:
        return {
            "success": True,
            "requested_x": x_column,
            "matched_x": matched_x,
            "x_confidence": x_match["confidence"]
        }

    y_match = find_best_matching_column.invoke({
        "column_name": y_column
    })

    if (
        not y_match["success"] or
        y_match["confidence"] < 0.8
    ):
        return {
            "success": False,
            "message": f"Could not find a suitable column for '{y_column}'."
        }

    matched_y = y_match["matched_column"]

    return {
        "success": True,
        "requested_x": x_column,
        "requested_y": y_column,
        "matched_x": matched_x,
        "matched_y": matched_y,
        "x_confidence": x_match["confidence"],
        "y_confidence": y_match["confidence"]
    }


def save_chart(fig, chart_type: str):
    """
    Save a Plotly chart to outputs/charts.

    Args:
        fig: Plotly figure object.
        chart_type: bar, line, scatter, histogram, pie, etc.

    Returns:
        Path to the saved HTML file.
    """

    output_dir = "outputs/charts"

    os.makedirs(output_dir, exist_ok=True)

    filename = f"{chart_type}_{uuid.uuid4().hex}.html"

    chart_path = os.path.join(output_dir, filename)

    fig.write_html(chart_path)

    return chart_path