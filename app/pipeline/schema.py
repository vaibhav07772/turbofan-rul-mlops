"""Pandera schema for NASA C-MAPSS Turbofan data.

NASA C-MAPSS format (space-separated, no header):
    unit cycle op_setting_1 op_setting_2 op_setting_3 sensor_1 ... sensor_21

For training data, we add a derived 'RUL' column.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

# Sensor columns in NASA C-MAPSS FD001
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
BASE_COLS = ["unit", "cycle"] + OP_SETTING_COLS + SENSOR_COLS


def build_schema(with_rul: bool = True) -> DataFrameSchema:
    """Build a Pandera schema. Includes RUL column if with_rul=True."""
    cols = {
        "unit": Column(
            pa.Int,
            nullable=False,
            checks=Check.ge(1),
            description="Engine unit ID",
        ),
        "cycle": Column(
            pa.Int,
            nullable=False,
            checks=Check.ge(1),
            description="Time cycle (flight number)",
        ),
    }

    # Operating settings (continuous, small range)
    for col in OP_SETTING_COLS:
        cols[col] = Column(pa.Float, nullable=False, coerce=True)

    # Sensors (numeric, non-null)
    for col in SENSOR_COLS:
        cols[col] = Column(pa.Float, nullable=False, coerce=True)

    if with_rul:
        cols["RUL"] = Column(
            pa.Int,
            nullable=False,
            checks=Check.ge(0),
            description="Remaining Useful Life in cycles",
        )

    return DataFrameSchema(columns=cols, strict=False, coerce=True)


class SchemaValidationError(ValueError):
    """Raised when the dataset fails schema validation."""


def validate_schema(data: pd.DataFrame, with_rul: bool = True) -> pd.DataFrame:
    """Validate a Turbofan dataframe. Returns validated df or raises."""
    schema = build_schema(with_rul=with_rul)
    try:
        return schema.validate(data, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failure = exc.failure_cases
        if failure is None or failure.empty:
            raise SchemaValidationError(str(exc)) from exc

        problems = []
        for _, row in failure.head(8).iterrows():
            col = row.get("column", "?")
            check = row.get("check", "?")
            problems.append(f"  • column '{col}' failed: {check}")
        summary = "Dataset schema validation failed:\n" + "\n".join(problems)
        if len(failure) > 8:
            summary += f"\n  ... and {len(failure) - 8} more issues."
        raise SchemaValidationError(summary) from exc
