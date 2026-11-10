"""Copying objects that hold API clients, caches and limits (#211).

``copy.deepcopy`` of an LLM or embeddings used to fail: instructor's clients
recurse in ``__getattr__`` on an object ``copy`` made without ``__init__``,
and a module (``LiteLLMEmbeddings.litellm``) can't be copied at all. And a
copy that kept the original's cache wrappers would run its calls on the
original.

``copy_sharing`` copies an object's ``__dict__`` but *shares* what connects
to a provider or is meant to be shared (clients, caches, rate limits,
modules), and leaves out attributes the object rebuilds for the copy.
"""

from __future__ import annotations

import copy
import types
import typing as t

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
