"""Public configuration API.

The refactored grouped configuration lives in ``config_refactored`` while this
module preserves the original ``lww_transport.config`` import path.
"""

from .config_refactored import (
    ComputeParams,
    DiscretizationParams,
    GeometryParams,
    LWWConfig,
    MaterialParams,
    OperatingConditions,
    SolverParams,
    format_config_summary,
    print_config_summary,
    save_config_summary,
)

__all__ = [
    "ComputeParams",
    "DiscretizationParams",
    "GeometryParams",
    "LWWConfig",
    "MaterialParams",
    "OperatingConditions",
    "SolverParams",
    "format_config_summary",
    "print_config_summary",
    "save_config_summary",
]
