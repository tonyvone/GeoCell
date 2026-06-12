"""GeoCell Field: a geometry-native epistemic memory engine.

Memory is modeled as a particle field. Each memory (a GeoCell) has a
position in a high-dimensional space, a semantic radius, provenance,
and a belief status. Evidence exerts force: corroboration attracts,
contradiction repels. Trust flows along support edges. Beliefs are
revised, superseded, and retracted as new memories arrive.
"""

from geocell.cell import (
    ACTIVE,
    CONTESTED,
    CONTRADICTION,
    GeoCell,
    RETRACTED,
    SUPERSEDED,
    SUPPORT,
    UNKNOWN,
)
from geocell.field import GeoCellField

__version__ = "0.5.0"

__all__ = [
    "GeoCell",
    "GeoCellField",
    "SUPPORT",
    "UNKNOWN",
    "CONTRADICTION",
    "ACTIVE",
    "CONTESTED",
    "SUPERSEDED",
    "RETRACTED",
    "__version__",
]
