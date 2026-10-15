"""Small shared validators for the smoke task's fixed schema."""
from math import isfinite


def finite_number(value: object, name: str) -> None:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
