"""Copying objects that hold API clients, caches and limits (#211).

``copy.deepcopy`` of an LLM or embeddings used to fail: instructor's clients
recurse in ``__getattr__`` on an object ``copy`` made without ``__init__``,
and a module (``LiteLLMEmbeddings.litellm``) can't be copied at all. And a
copy that kept the original's cache wrappers would run its calls on the
original.

``copy_sharing`` copies an object's ``__dict__`` but *shares* what connects
to a provider or is meant to be shared (clients, caches, rate limits,
modules), and leaves out attributes the object rebuilds for the copy.
``CachedCopyable`` is the mixin ``InstructorLLM`` and ``BaseRagasEmbedding``
build on: per-instance cache wrappers, rebuilt for each copy.
"""

from __future__ import annotations

import copy
import types
import typing as t

from ragas.cache import cache_identity, cacher

T = t.TypeVar("T")


def copy_sharing(
    obj: T,
    *,
    shared: t.AbstractSet[str],
    rebuilt: t.AbstractSet[str] = frozenset(),
    memo: t.Optional[t.Dict[int, t.Any]] = None,
) -> T:
    """A copy of ``obj`` (deep when ``memo`` is given, else shallow).

    Attributes named in ``shared``, and any module, are the same objects in
    the copy. Attributes named in ``rebuilt`` are left out, for the caller
    to set on the copy.
    """
    cls = type(obj)
    new = cls.__new__(cls)
    if memo is not None:
        memo[id(obj)] = new  # objects that refer back to obj
    for name, value in obj.__dict__.items():
        if name in rebuilt:
            continue
        if memo is None or name in shared or isinstance(value, types.ModuleType):
            new.__dict__[name] = value
        else:
            new.__dict__[name] = copy.deepcopy(value, memo)
    return new


class CachedCopyable:
    """Cache wrappers and copying for an LLM or embeddings (#72, #211).

    A subclass names the methods to route through ``self.cache`` in
    ``_CACHED_METHODS`` and what its copies share in ``_SHARED_ON_COPY``,
    and calls ``_wrap_with_cache()`` once ``cache`` and whatever
    ``cache_identity`` reads are set.
    """

    cache: t.Optional[t.Any] = None
    # Methods routed through ``cache``, wrapped per instance.
    _CACHED_METHODS: t.ClassVar[t.Tuple[str, ...]] = ()
    # Shared by copies, never copied: what connects to the provider, and the
    # limits and stores meant to be shared. Modules are always shared.
    _SHARED_ON_COPY: t.ClassVar[t.FrozenSet[str]] = frozenset()

    def _wrap_with_cache(self) -> None:
        """Route ``_CACHED_METHODS`` through ``cache``, if set.

        The wrappers are bound to this object, so a copy gets its own (see
        ``__copy__``): with the original's, the copy's calls would run on
        the original, with its settings.
        """
        if self.cache is None:
            return
        cls = type(self)
        for name in self._CACHED_METHODS:
            method = getattr(cls, name).__get__(self, cls)
            wrapped = cacher(
                cache_backend=self.cache, identity=lambda: cache_identity(self)
            )(method)
            setattr(self, name, wrapped)

    def _copied(self, memo: t.Optional[t.Dict[int, t.Any]]) -> t.Self:
        new = copy_sharing(
            self,
            shared=self._SHARED_ON_COPY,
            rebuilt=frozenset(self._CACHED_METHODS),
            memo=memo,
        )
        new._wrap_with_cache()
        return new

    def __copy__(self) -> t.Self:
        return self._copied(None)

    def __deepcopy__(self, memo: t.Dict[int, t.Any]) -> t.Self:
        # instructor's clients can't be deep-copied: their __getattr__
        # recurses on an object that copy made without __init__ (#211)
        return self._copied(memo)
