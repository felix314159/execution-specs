"""Nimbus Transition tool interface."""

import json
import re
import shlex
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

from execution_testing.exceptions import (
    BlockException,
    ExceptionBase,
    ExceptionMapper,
    TransactionException,
)
from execution_testing.fixtures import (
    BlockchainFixture,
    FixtureFormat,
)
from execution_testing.forks import Fork

from ..file_utils import dump_files_to_directory
from ..fixture_consumer_tool import FixtureConsumerTool
from ..transition_tool import TransitionTool


class NimbusFixtureConsumer(
    FixtureConsumerTool,
    fixture_formats=[BlockchainFixture],
):
    """Nimbus EEST blockchain fixture consumer."""

    default_binary = Path("eest_blockchain")
    detect_binary_pattern = re.compile(
        r"^Usage: .*eest_blockchain .*vector\.json"
    )
    version_flag = ""
    cached_version: Optional[str] = None
    trace: bool

    def __init__(
        self,
        binary: Optional[Path] = None,
        trace: bool = False,
    ):
        """Initialize the Nimbus fixture consumer."""
        self.binary = binary if binary else self.default_binary
        self.trace = trace

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
            raise Exception(
                "Unexpected exception calling Nimbus fixture consumer."
            ) from e

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

    def _filtered_fixture_path(
        self,
        fixture_path: Path,
        fixture_name: Optional[str],
    ) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as temporary_file:
            filtered_fixture_path = Path(temporary_file.name)
            if fixture_name is None:
                temporary_file.write(fixture_path.read_text())
                return filtered_fixture_path

            fixture = json.loads(fixture_path.read_text())
            if fixture_name not in fixture:
                filtered_fixture_path.unlink(missing_ok=True)
                raise Exception(
                    f"Fixture {fixture_name} not found in {fixture_path}"
                )
            json.dump({fixture_name: fixture[fixture_name]}, temporary_file)
            return filtered_fixture_path

    def consume_blockchain_test(
        self,
        fixture_path: Path,
        fixture_name: Optional[str] = None,
        debug_output_path: Optional[Path] = None,
    ) -> None:
        """Consume a blockchain test fixture via Nimbus `eest_blockchain`."""
        filtered_fixture_path = self._filtered_fixture_path(
            fixture_path, fixture_name
        )
        command = [str(self.binary), str(filtered_fixture_path)]
        try:
            result = self._run_command(command)
            if debug_output_path:
                self._consume_debug_dump(
                    command, result, fixture_path, debug_output_path
                )
            if result.returncode != 0:
                raise Exception(
                    f"Nimbus eest_blockchain exited with non-zero exit code "
                    f"({result.returncode}).\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}\n"
                    f"{chr(32).join(command)}"
                )
        finally:
            filtered_fixture_path.unlink(missing_ok=True)

    def consume_fixture(
        self,
        fixture_format: FixtureFormat,
        fixture_path: Path,
        fixture_name: Optional[str] = None,
        debug_output_path: Optional[Path] = None,
    ) -> None:
        """Execute the Nimbus blockchain fixture consumer."""
        if fixture_format == BlockchainFixture:
            self.consume_blockchain_test(
                fixture_path=fixture_path,
                fixture_name=fixture_name,
                debug_output_path=debug_output_path,
            )
        else:
            raise Exception(
                f"Fixture format {fixture_format.format_name} "
                f"not supported by {self.binary}"
            )


class NimbusTransitionTool(TransitionTool):
    """Nimbus `evm` Transition tool interface wrapper class."""

    default_binary = Path("t8n")
    detect_binary_pattern = re.compile(r"^Nimbus-t8n\b")
    version_flag: str = "--version"

    binary: Path
    cached_version: Optional[str] = None
    trace: bool

    def __init__(
        self,
        *,
        binary: Optional[Path] = None,
        trace: bool = False,
    ):
        """Initialize the Nimbus Transition tool interface."""
        super().__init__(
            exception_mapper=NimbusExceptionMapper(),
            binary=binary,
            trace=trace,
        )
        args = [str(self.binary), "--help"]
        try:
            result = subprocess.run(args, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise Exception(
                f"evm process unexpectedly returned "
                f"a non-zero status code: {e}."
            ) from e
        except Exception as e:
            raise Exception(
                f"Unexpected exception calling evm tool: {e}."
            ) from e
        self.help_string = result.stdout

    def version(self) -> str:
        """Get `evm` binary version."""
        if self.cached_version is None:
            self.cached_version = re.sub(
                r"\x1b\[0m", "", super().version()
            ).strip()

        return self.cached_version

    def is_fork_supported(self, fork: Fork) -> bool:
        """
        Return True if the fork is supported by the tool.

        If the fork is a transition fork, we want to check the fork it
        transitions to.
        """
        return fork.transition_tool_name() in self.help_string


class NimbusExceptionMapper(ExceptionMapper):
    """
    Translate between EEST exceptions and error strings returned by Nimbus.
    """

    mapping_substring: ClassVar[Dict[ExceptionBase, str]] = {
        TransactionException.TYPE_4_TX_CONTRACT_CREATION: (
            "set code transaction must not be a create transaction"
        ),
        TransactionException.INSUFFICIENT_ACCOUNT_FUNDS: (
            "invalid tx: not enough cash to send"
        ),
        TransactionException.TYPE_3_TX_MAX_BLOB_GAS_ALLOWANCE_EXCEEDED: (
            "would exceed maximum allowance"
        ),
        TransactionException.INSUFFICIENT_MAX_FEE_PER_BLOB_GAS: (
            "max fee per blob gas less than block blob gas fee"
        ),
        TransactionException.INSUFFICIENT_MAX_FEE_PER_GAS: (
            "max fee per gas less than block base fee"
        ),
        TransactionException.TYPE_3_TX_PRE_FORK: (
            "blob tx used but field env.ExcessBlobGas missing"
        ),
        TransactionException.TYPE_3_TX_INVALID_BLOB_VERSIONED_HASH: (
            "invalid tx: one of blobVersionedHash has invalid version"
        ),
        # TODO: temp solution until mapper for nimbus is fixed
        TransactionException.GAS_LIMIT_EXCEEDS_MAXIMUM: (
            "zero gasUsed but transactions present"
        ),
        # This message is the same as TYPE_3_TX_MAX_BLOB_GAS_ALLOWANCE_EXCEEDED
        TransactionException.TYPE_3_TX_BLOB_COUNT_EXCEEDED: (
            "exceeds maximum allowance"
        ),
        TransactionException.TYPE_3_TX_ZERO_BLOBS: (
            "blob transaction missing blob hashes"
        ),
        TransactionException.INTRINSIC_GAS_TOO_LOW: (
            "zero gasUsed but transactions present"
        ),
        TransactionException.INTRINSIC_GAS_BELOW_FLOOR_GAS_COST: (
            "intrinsic gas too low"
        ),
        TransactionException.INITCODE_SIZE_EXCEEDED: (
            "max initcode size exceeded"
        ),
        BlockException.RLP_BLOCK_LIMIT_EXCEEDED: (
            # TODO:
            "ExceededBlockSizeLimit: Exceeded block size limit"
        ),
        BlockException.INVALID_BASEFEE_PER_GAS: "invalid baseFee",
        BlockException.INVALID_BLOCK_NUMBER: (
            "Blocks must be numbered consecutively"
        ),
        BlockException.INVALID_BLOCK_TIMESTAMP_OLDER_THAN_PARENT: (
            "Invalid timestamp"
        ),
        BlockException.INVALID_GASLIMIT: "invalid gas limit",
        BlockException.INVALID_GAS_USED_ABOVE_LIMIT: (
            "gasUsed should be non negative and smaller or equal gasLimit"
        ),
        BlockException.INVALID_BLOCK_HASH: "blockhash mismatch",
        BlockException.INVALID_STATE_ROOT: "stateRoot mismatch",
        BlockException.INVALID_RECEIPTS_ROOT: "receiptRoot mismatch",
        BlockException.INVALID_LOG_BLOOM: "bloom mismatch",
    }
    mapping_regex: ClassVar[Dict[ExceptionBase, str]] = {}
