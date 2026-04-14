"""Query-adaptive chunk ranking.

Public API re-exported from sub-modules:

- ``apply_query_boost`` — symbol-definition and NL stem boosting (boosting.py)
- ``diverse_topk``      — file-penalty + saturation-decay top-k selection (selection.py)
- ``resolve_alpha``     — query-type-adaptive alpha blending weight (boosting.py)
"""

from semble.ranking.boosting import apply_query_boost, resolve_alpha
from semble.ranking.selection import diverse_topk

__all__ = ["apply_query_boost", "diverse_topk", "resolve_alpha"]
