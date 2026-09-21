"""Guard for numpy.einsum on new-style dtypes (numpy/numpy#32671)."""
from .core import (
    BugDetectionResult,
    detect_einsum_newstyle_dtype_bug,
    is_new_style_dtype,
    naive_einsum,
    safe_einsum,
)

__all__ = [
    "BugDetectionResult",
    "detect_einsum_newstyle_dtype_bug",
    "is_new_style_dtype",
    "naive_einsum",
    "safe_einsum",
]
