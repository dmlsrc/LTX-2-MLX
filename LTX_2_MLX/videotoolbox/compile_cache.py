"""Bounded module-level caching for mx.compile'd forwards.

Every net memoizes its compiled forward in a module-level dict keyed by
id(params) (plus config), and the compiled closure keeps that checkpoint
alive. Left unbounded, those dicts retain every checkpoint ever constructed
in the process -- an A/B loop constructing several processors accumulates
them all in memory. A small FIFO bound keeps the live configs cached while
letting dead checkpoints be collected; evicting a still-live entry only
costs a one-time re-trace on its next use.

id-reuse is safe under eviction: an entry's closure pins its params dict
alive, so a params id can only be recycled after the entries holding it are
evicted -- at which point the lookup misses and recompiles fresh.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Per-cache entry bound. The busiest caches hold several keys per checkpoint
# (basicvsrpp: 4 deform modules; the resblock cache: ~7 prefixes), so 16
# comfortably covers a couple of concurrently-live checkpoints.
_CAP = 16


def cached(cache: dict, key: Any, make: Callable[[], Any]) -> Any:
    """Return cache[key], building it with make() on a miss; FIFO-evict past _CAP."""
    fn = cache.get(key)
    if fn is None:
        fn = make()
        cache[key] = fn
        while len(cache) > _CAP:
            cache.pop(next(iter(cache)))   # oldest insertion first
    return fn
