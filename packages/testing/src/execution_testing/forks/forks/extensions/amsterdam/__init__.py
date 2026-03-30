"""
Amsterdam extension discovery helpers.

Each Amsterdam EIP branch can contribute its own module under this package
without editing the shared Amsterdam fork registry in ``forks.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from importlib import import_module
from pkgutil import iter_modules

from execution_testing.forks.gas_costs import GasCosts
from execution_testing.vm import OpcodeBase, Opcodes


def _extension_modules() -> Iterator[object]:
    """Yield Amsterdam extension modules in a deterministic order."""
    package_name = __name__
    discovered_modules = sorted(
        iter_modules(__path__),
        key=lambda info: info.name,
    )
    for module_info in discovered_modules:
        if module_info.name.startswith("_"):
            continue
        yield import_module(f"{package_name}.{module_info.name}")


def valid_opcodes() -> list[Opcodes]:
    """Return valid opcode additions from all Amsterdam extension modules."""
    opcodes: list[Opcodes] = []
    for module in _extension_modules():
        opcodes.extend(getattr(module, "VALID_OPCODES", ()))
    return opcodes


def opcode_gas_map(
    gas_costs: GasCosts,
) -> dict[OpcodeBase, int | Callable[[OpcodeBase], int]]:
    """Return opcode gas overrides from all Amsterdam extension modules."""
    gas_map: dict[OpcodeBase, int | Callable[[OpcodeBase], int]] = {}
    for module in _extension_modules():
        module_opcode_gas_map = getattr(module, "opcode_gas_map", None)
        if module_opcode_gas_map is None:
            continue
        gas_map.update(module_opcode_gas_map(gas_costs))
    return gas_map
