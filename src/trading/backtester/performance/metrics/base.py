# metrics/base.py

from functools import cached_property

from .core_stats import CoreStats


class MetricsGroup:
    """
    Base for all metrics groups (ReturnMetrics, RiskMetrics, CostMetrics,
    TradeMetrics).

    Every public @property or @cached_property defined on a subclass is
    automatically included in to_dict()/summary() output. Prefix a name
    with an underscore to keep it internal (e.g. a shared intermediate
    computation feeding several public metrics) and exclude it from export.
    """
    _exported: tuple[str, ...] = ()

    def __init__(self, core: CoreStats):

        self.core = core


    def to_dict(self) -> dict[str, float]:
        """
        Export all public property / cached_property metrics on this group.

        Returns
        -------
        dict[str, float]
            Sorted mapping of metric name → computed value. Underscore-prefixed
            attributes are excluded.
        """
        cls = type(self)
        names = [
            name for name, attr in vars(cls).items()
            if isinstance(attr, (property, cached_property))
            and not name.startswith("_")
        ]
        return {name: getattr(self, name) for name in sorted(names)}
