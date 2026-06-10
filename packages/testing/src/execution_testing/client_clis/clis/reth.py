"""Reth execution client transition and fixture consumer tools."""

import json
import re
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from execution_testing.exceptions import (
    BlockException,
    ExceptionMapper,
    TransactionException,
)
from execution_testing.fixtures import (
    BlockchainFixture,
    FixtureFormat,
    StateFixture,
)

from ..file_utils import dump_files_to_directory
from ..fixture_consumer_tool import FixtureConsumerTool


class RevmeFixtureConsumer(
    FixtureConsumerTool,
    fixture_formats=[StateFixture, BlockchainFixture],
):
    """revm `revme` implementation of the fixture consumer."""

    default_binary = Path("revme")
    detect_binary_pattern = re.compile(r"^Usage: revme <COMMAND>")
    version_flag = "--help"
    cached_version: Optional[str] = None
    trace: bool

    def __init__(
        self,
        binary: Optional[Path] = None,
        trace: bool = False,
    ):
        """Initialize the RevmeFixtureConsumer class."""
        self.binary = binary if binary else self.default_binary
        self.trace = trace
        self._info_metadata: Optional[Dict[str, Any]] = {}

    def _run_command(self, command: List[str]) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise Exception("Command failed with non-zero status.") from e
        except Exception as e:
            raise Exception("Unexpected exception calling revme tool.") from e

    def _consume_debug_dump(
        self,
        command: List[str],
        result: subprocess.CompletedProcess,
        fixture_path: Path,
        debug_output_path: Path,
    ) -> None:
        assert all(isinstance(x, str) for x in command), (
            f"Not all elements of command list are strings: {command}"
        )
        debug_command = command.copy()
        debug_fixture_path = str(debug_output_path / "fixtures.json")
        debug_command[-1] = debug_fixture_path
        consume_direct_call = " ".join(
            shlex.quote(arg) for arg in debug_command
        )
        consume_direct_script = textwrap.dedent(
            f"""\
            #!/bin/bash
            {consume_direct_call}
            """
        )
        dump_files_to_directory(
            debug_output_path,
            {
                "consume_direct_args.py": debug_command,
                "consume_direct_returncode.txt": result.returncode,
                "consume_direct_stdout.txt": result.stdout,
                "consume_direct_stderr.txt": result.stderr,
                "consume_direct.sh+x": consume_direct_script,
            },
        )
        shutil.copyfile(fixture_path, debug_fixture_path)

    @staticmethod
    def _json_records(output: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
        return records

    @staticmethod
    def _skip_if_known_issue(
        records: List[Dict[str, Any]], fixture_path: Path
    ) -> None:
        """
        Skip the test if revme reported the fixture as skipped.

        revme emits a record like
        ``{"file": ..., "status": "skipped", "reason": "known_issue"}``
        (with no ``test`` field) for fixtures it deliberately does not
        execute, e.g. blob fixtures revm's `btest` runner does not yet
        support. These are not failures, so surface them as pytest skips.
        """
        for record in records:
            if record.get("status") != "skipped":
                continue
            record_file = record.get("file")
            if record_file is None or Path(record_file).name == (
                fixture_path.name
            ):
                pytest.skip(
                    f"revme skipped fixture: "
                    f"{record.get('reason', 'unknown reason')}"
                )

    def consume_state_test(
        self,
        fixture_path: Path,
        fixture_name: Optional[str] = None,
        debug_output_path: Optional[Path] = None,
    ) -> None:
        """Consume a state test fixture via `revme statetest`."""
        assert fixture_name, "Fixture name must be provided for revme."
        command = [
            str(self.binary),
            "statetest",
            "--json-outcome",
            "--omit-progress",
            str(fixture_path),
        ]
        result = self._run_command(command)
        if debug_output_path:
            self._consume_debug_dump(
                command, result, fixture_path, debug_output_path
            )
        records = self._json_records(result.stderr + "\n" + result.stdout)
        self._skip_if_known_issue(records, fixture_path)
        test_results = [
            record for record in records if record.get("test") == fixture_name
        ]
        assert len(test_results) < 2, (
            f"Multiple test results for {fixture_name}"
        )
        if result.returncode != 0 and not test_results:
            raise Exception(
                f"revme statetest exited with non-zero exit code "
                f"({result.returncode}).\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}\n"
                f"{chr(32).join(command)}"
            )
        assert len(test_results) == 1, (
            f"Test result for {fixture_name} missing"
        )
        assert test_results[0].get("pass") is True, (
            f"State test failed: {test_results[0].get('errorMsg', '')}"
        )

    def consume_blockchain_test(
        self,
        fixture_path: Path,
        fixture_name: Optional[str] = None,
        debug_output_path: Optional[Path] = None,
    ) -> None:
        """Consume a blockchain test fixture via `revme btest`."""
        assert fixture_name, "Fixture name must be provided for revme."
        command = [
            str(self.binary),
            "btest",
            "--json",
            "--omit-progress",
            str(fixture_path),
        ]
        result = self._run_command(command)
        if debug_output_path:
            self._consume_debug_dump(
                command, result, fixture_path, debug_output_path
            )
        records = self._json_records(result.stdout + "\n" + result.stderr)
        self._skip_if_known_issue(records, fixture_path)
        terminal_statuses = {
            "passed",
            "failed",
            "unexpected_success",
            "unexpected_failure",
        }
        test_results = [
            record
            for record in records
            if record.get("test") == fixture_name
            and record.get("status") in terminal_statuses
        ]
        assert len(test_results) < 2, (
            f"Multiple test results for {fixture_name}"
        )
        if result.returncode != 0 and not test_results:
            raise Exception(
                f"revme btest exited with non-zero exit code "
                f"({result.returncode}).\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}\n"
                f"{chr(32).join(command)}"
            )
        assert len(test_results) == 1, (
            f"Test result for {fixture_name} missing"
        )
        assert test_results[0].get("status") == "passed", (
            f"Blockchain test failed: {test_results[0]}"
        )

    def consume_fixture(
        self,
        fixture_format: FixtureFormat,
        fixture_path: Path,
        fixture_name: Optional[str] = None,
        debug_output_path: Optional[Path] = None,
    ) -> None:
        """Execute the appropriate revme fixture consumer."""
        if fixture_format == BlockchainFixture:
            self.consume_blockchain_test(
                fixture_path=fixture_path,
                fixture_name=fixture_name,
                debug_output_path=debug_output_path,
            )
        elif fixture_format == StateFixture:
            self.consume_state_test(
                fixture_path=fixture_path,
                fixture_name=fixture_name,
                debug_output_path=debug_output_path,
            )
        else:
            raise Exception(
                f"Fixture format {fixture_format.format_name} "
                f"not supported by {self.binary}"
            )


class RethExceptionMapper(ExceptionMapper):
    """Reth exception mapper."""

    mapping_substring = {
        TransactionException.SENDER_NOT_EOA: (
            "reject transactions from senders with deployed code"
        ),
        TransactionException.INSUFFICIENT_ACCOUNT_FUNDS: "lack of funds",
        TransactionException.INITCODE_SIZE_EXCEEDED: (
            "create initcode size limit"
        ),
        TransactionException.INSUFFICIENT_MAX_FEE_PER_GAS: (
            "gas price is less than basefee"
        ),
        TransactionException.PRIORITY_GREATER_THAN_MAX_FEE_PER_GAS: (
            "priority fee is greater than max fee"
        ),
        TransactionException.GASLIMIT_PRICE_PRODUCT_OVERFLOW: "overflow",
        TransactionException.TYPE_3_TX_CONTRACT_CREATION: "unexpected length",
        TransactionException.TYPE_3_TX_WITH_FULL_BLOBS: "unexpected list",
        TransactionException.TYPE_3_TX_INVALID_BLOB_VERSIONED_HASH: (
            "blob version not supported"
        ),
        TransactionException.TYPE_3_TX_ZERO_BLOBS: "empty blobs",
        TransactionException.TYPE_4_EMPTY_AUTHORIZATION_LIST: (
            "empty authorization list"
        ),
        TransactionException.TYPE_4_TX_CONTRACT_CREATION: "unexpected length",
        TransactionException.TYPE_4_TX_PRE_FORK: (
            "eip 7702 transactions present in pre-prague payload"
        ),
        BlockException.INVALID_REQUESTS: "mismatched block requests hash",
        BlockException.INVALID_RECEIPTS_ROOT: "receipt root mismatch",
        BlockException.INVALID_STATE_ROOT: "mismatched block state root",
        BlockException.INVALID_BLOCK_HASH: "block hash mismatch",
        BlockException.INVALID_GAS_USED: "block gas used mismatch",
        BlockException.RLP_BLOCK_LIMIT_EXCEEDED: "block is too large: ",
        BlockException.INVALID_BASEFEE_PER_GAS: "block base fee mismatch",
        BlockException.EXTRA_DATA_TOO_BIG: "invalid payload extra data",
        BlockException.INVALID_LOG_BLOOM: "header bloom filter mismatch",
    }
    mapping_regex = {
        TransactionException.NONCE_MISMATCH_TOO_LOW: (
            r"nonce \d+ too low, expected \d+"
        ),
        TransactionException.NONCE_MISMATCH_TOO_HIGH: (
            r"nonce \d+ too high, expected \d+"
        ),
        TransactionException.INSUFFICIENT_MAX_FEE_PER_BLOB_GAS: (
            r"blob gas price \(\d+\) is greater than "
            r"max fee per blob gas \(\d+\)"
        ),
        TransactionException.INTRINSIC_GAS_TOO_LOW: (
            r"call gas cost \(\d+\) exceeds the gas limit \(\d+\)|"
            r"gas floor \(\d+\) exceeds the gas limit \(\d+\)"
        ),
        TransactionException.INTRINSIC_GAS_BELOW_FLOOR_GAS_COST: (
            r"gas floor \(\d+\) exceeds the gas limit \(\d+\)"
        ),
        TransactionException.TYPE_3_TX_MAX_BLOB_GAS_ALLOWANCE_EXCEEDED: (
            r"blob gas used \d+ exceeds maximum allowance \d+"
        ),
        TransactionException.TYPE_3_TX_BLOB_COUNT_EXCEEDED: (
            r"too many blobs, have \d+, max \d+"
        ),
        TransactionException.TYPE_3_TX_PRE_FORK: (
            r"blob transactions present in pre-cancun payload|empty blobs"
        ),
        TransactionException.GAS_ALLOWANCE_EXCEEDED: (
            r"transaction gas limit \w+ is more than blocks available gas \w+|"
            r"caller gas limit exceeds the block gas limit"
        ),
        TransactionException.GAS_LIMIT_EXCEEDS_MAXIMUM: (
            r"transaction gas limit.*is greater than the cap"
        ),
        BlockException.SYSTEM_CONTRACT_CALL_FAILED: (
            r"failed to apply .* requests contract call"
        ),
        BlockException.INCORRECT_BLOB_GAS_USED: (
            r"blob gas used mismatch|"
            r"blob gas used \d+ is not a multiple of blob gas per blob"
        ),
        BlockException.INCORRECT_EXCESS_BLOB_GAS: (
            r"excess blob gas \d+ is not a multiple of blob gas per blob|"
            r"invalid excess blob gas"
        ),
        BlockException.INVALID_GAS_USED_ABOVE_LIMIT: (
            r"block used gas \(\d+\) is greater than gas limit \(\d+\)"
        ),
        BlockException.INVALID_GASLIMIT: (
            r"child gas_limit \d+ max .* is .*|"
            r"child gas_limit \d+ is below the max allowed decrease .*|"
            r"child gas limit \d+ is below the minimum allowed limit"
        ),
        BlockException.INVALID_BLOCK_TIMESTAMP_OLDER_THAN_PARENT: (
            r"block timestamp \d+ is in the past compared to "
            r"the parent timestamp \d+"
        ),
        BlockException.INVALID_BLOCK_NUMBER: (
            r"block number \d+ does not match parent block number \d+"
        ),
        BlockException.GAS_USED_OVERFLOW: (
            r"transaction gas limit \w+ is more than blocks available gas \w+"
        ),
        # BAL Exceptions
        BlockException.INVALID_BAL_HASH: (r"block access list hash mismatch"),
        BlockException.INVALID_BLOCK_ACCESS_LIST: (
            r"block access list hash mismatch|"
            r"BAL rejection: FinalHashMismatch"
        ),
        BlockException.INCORRECT_BLOCK_FORMAT: (
            r"block access list hash mismatch|"
            r"BAL rejection: FinalHashMismatch"
        ),
        # Reth does not validate the sizes or offsets of the deposit
        # contract logs. As a workaround we have set
        # INVALID_DEPOSIT_EVENT_LAYOUT equal to INVALID_REQUESTS.
        #
        # Although this is out of spec, it is understood that this
        # will not cause an issue so long as the mainnet/testnet
        # deposit contracts don't change.
        #
        # The offsets are checked second and the sizes are checked
        # third within the `is_valid_deposit_event_data` function:
        # https://eips.ethereum.org/EIPS/eip-6110#block-validity
        #
        # EELS definition for `is_valid_deposit_event_data`:
        # https://github.com/ethereum/execution-specs/blob/5ddb904fa7ba27daeff423e78466744c51e8cb6a/src/ethereum/forks/prague/requests.py#L51
        BlockException.INVALID_DEPOSIT_EVENT_LAYOUT: (
            r"failed to decode deposit requests from receipts|"
            r"mismatched block requests hash"
        ),
    }
