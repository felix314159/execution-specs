"""Listings of all EIPs for Amsterdam fork."""

import importlib
import inspect
import pkgutil
import re

__all__ = ["AmsterdamEIPs"]

_prefix = __name__ + "."
_amsterdam_eips = []

for _importer, _modname, _ispkg in pkgutil.iter_modules(
    __path__, prefix=_prefix
):
    if _ispkg or not re.search(r"\.eip_\d+$", _modname):
        continue

    _module = importlib.import_module(_modname)

    for _name, _obj in inspect.getmembers(_module, inspect.isclass):
        if re.match(r"^EIP\d+$", _name) and _obj.__module__ == _modname:
            _amsterdam_eips.append(_obj)

_amsterdam_eips.sort(key=lambda cls: int(cls.__name__[3:]))


class _AmsterdamEIPs:
    """Expand to the currently available Amsterdam EIP mixins."""

    def __mro_entries__(self, bases: tuple[type, ...]) -> tuple[type, ...]:
        del bases
        return tuple(_amsterdam_eips)


AmsterdamEIPs = _AmsterdamEIPs()

del _importer, _ispkg, _modname, _module, _name, _obj, _prefix
