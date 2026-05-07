from src.server.selectors.base import BaseSelector, ClientResourceProfile
from src.server.selectors.profile import ResourceProfiler
from src.server.selectors.oort_selector import OortSelector
from src.server.selectors.lyapunov_selector import LyapunovSelector
from src.server.selectors.fedbalancer_selector import FedBalancerSelector

SELECTOR_REGISTRY = {
    "none": None,
    "oort": OortSelector,
    "lyapunov": LyapunovSelector,
    "fedbalancer": FedBalancerSelector,
}


def build_selector(name: str, **kwargs):
    name = (name or "none").lower()
    if name == "none" or name not in SELECTOR_REGISTRY:
        return None
    cls = SELECTOR_REGISTRY[name]
    if cls is None:
        return None
    return cls(**kwargs)


__all__ = [
    "BaseSelector",
    "ClientResourceProfile",
    "ResourceProfiler",
    "OortSelector",
    "LyapunovSelector",
    "FedBalancerSelector",
    "build_selector",
    "SELECTOR_REGISTRY",
]
