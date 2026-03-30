"""Amsterdam opcode extensions for EIP-7843."""

from __future__ import annotations

from collections.abc import Callable

from execution_testing.forks.gas_costs import GasCosts
from execution_testing.vm import OpcodeBase, Opcodes

VALID_OPCODES = [Opcodes.SLOTNUM]


def opcode_gas_map(
    gas_costs: GasCosts,
) -> dict[OpcodeBase, int | Callable[[OpcodeBase], int]]:
    """Return opcode gas overrides introduced by EIP-7843."""
    return {
        Opcodes.SLOTNUM: gas_costs.GAS_BASE,
    }
