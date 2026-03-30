"""Tests for Amsterdam extension aggregation."""

from typing import cast

import pytest

from execution_testing.vm import OpcodeBase

from ..forks import forks as forks_module
from ..forks.forks import Amsterdam


def test_amsterdam_valid_opcodes_include_extensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Amsterdam should prepend dynamically discovered extension opcodes."""
    sentinel_opcode = cast(OpcodeBase, object())
    monkeypatch.setattr(
        forks_module,
        "amsterdam_extension_valid_opcodes",
        lambda: [sentinel_opcode],
    )

    valid_opcodes = Amsterdam.valid_opcodes()

    assert valid_opcodes[0] is sentinel_opcode
    assert sentinel_opcode in valid_opcodes


def test_amsterdam_opcode_gas_map_includes_extensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Amsterdam should merge dynamically discovered gas overrides."""
    sentinel_opcode = cast(OpcodeBase, object())
    expected_gas_cost = Amsterdam.gas_costs().GAS_BASE
    monkeypatch.setattr(
        forks_module,
        "amsterdam_extension_opcode_gas_map",
        lambda gas_costs: {sentinel_opcode: gas_costs.GAS_BASE},
    )

    opcode_gas_map = Amsterdam.opcode_gas_map()

    assert opcode_gas_map[sentinel_opcode] == expected_gas_cost
