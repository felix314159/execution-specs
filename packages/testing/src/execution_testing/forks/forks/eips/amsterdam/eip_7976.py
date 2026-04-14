"""
EIP-7976: Increase Calldata Floor Cost.

Increase the calldata floor cost to 64/64 gas per byte to reduce maximum block
size.

https://eips.ethereum.org/EIPS/eip-7976
"""

from dataclasses import replace
from typing import List

from execution_testing.base_types import AccessList, Bytes
from execution_testing.base_types.conversions import BytesConvertible

from ....base_fork import BaseFork, TransactionDataFloorCostCalculator
from ....gas_costs import GasCosts


class EIP7976(BaseFork):
    """EIP-7976 class."""

    @classmethod
    def gas_costs(cls) -> GasCosts:
        """Transaction data floor token cost is increased from 10 to 16."""
        return replace(
            super(EIP7976, cls).gas_costs(),
            GAS_TX_DATA_TOKEN_FLOOR=16,
        )

    @classmethod
    def transaction_data_floor_cost_calculator(
        cls,
    ) -> TransactionDataFloorCostCalculator:
        """
        The data floor uses floor tokens based on calldata bytes:
        ``4 * bytes`` (64/64 per byte), not EIP-7623 calldata tokens.

        When EIP-7981 is also active, access list tokens are included
        in the floor (via the EIP-7981 token counting method).
        """
        gas_costs = cls.gas_costs()
        eip_7981_active = cls.is_eip_enabled(eip_number=7981)

        def fn(
            *,
            data: BytesConvertible,
            access_list: List[AccessList] | None = None,
        ) -> int:
            floor_tokens = len(Bytes(data)) * 4
            if eip_7981_active and access_list:
                floor_tokens += cls._access_list_token_count(  # type: ignore[attr-defined]
                    access_list
                )
            return (
                floor_tokens * gas_costs.GAS_TX_DATA_TOKEN_FLOOR
                + gas_costs.GAS_TX_BASE
            )

        return fn
