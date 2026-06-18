"""Hyperledger Besu Transition tool frontend."""

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import textwrap
from functools import cache
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set

import pytest
import requests

from execution_testing.exceptions import (
    BlockException,
    ExceptionBase,
    ExceptionMapper,
    TransactionException,
)
from execution_testing.fixtures import (
    BlockchainFixture,
    FixtureFormat,
    StateFixture,
)
from execution_testing.forks import Fork

from ..cli_types import TransitionToolOutput
from ..ethereum_cli import EthereumCLI
from ..fixture_consumer_tool import FixtureConsumerTool
from ..transition_tool import (
    Profiler,
    TransitionTool,
    dump_files_to_directory,
    model_dump_config,
)

BESU_BIN_DETECT_PATTERN = re.compile(r"^Besu evm .*$")


class BesuEvmTool(EthereumCLI):
    """Besu `evmtool` base class."""

    default_binary = Path("evmtool")
    detect_binary_pattern = BESU_BIN_DETECT_PATTERN
    cached_version: Optional[str] = None
    trace: bool

    def __init__(
        self,
        binary: Optional[Path] = None,
        trace: bool = False,
    ):
        """Initialize the BesuEvmTool class."""
        self.binary = binary if binary else self.default_binary
        self.trace = trace

    def _run_command(
        self, command: List[str], stdin_input: Optional[str] = None
    ) -> subprocess.CompletedProcess:
        """
        Run a command and return the result.

        ``stdin_input`` is fed to the process' standard input; Besu's
        ``state-test``/``block-test`` read newline-separated fixture file
        paths from stdin when given no positional arguments, which lets a
        single JVM process consume an arbitrary number of files without
        hitting the command line length limit.
        """
        try:
            return subprocess.run(
                command,
                input=stdin_input,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise Exception("Command failed with non-zero status.") from e
        except Exception as e:
            raise Exception("Unexpected exception calling evmtool.") from e

    def _consume_debug_dump(
        self,
        command: List[str],
        result: subprocess.CompletedProcess,
        fixture_path: Path,
        debug_output_path: Path,
    ) -> None:
        """Dump debug output for a consume command."""
        assert all(isinstance(x, str) for x in command), (
            f"Not all elements of 'command' list are strings: {command}"
        )
        assert len(command) > 0

        debug_fixture_path = str(debug_output_path / "fixtures.json")
        command[-1] = debug_fixture_path

        consume_direct_call = " ".join(shlex.quote(arg) for arg in command)

        consume_direct_script = textwrap.dedent(
            f"""\
            #!/bin/bash
            {consume_direct_call}
            """
        )
        dump_files_to_directory(
            debug_output_path,
            {
                "consume_direct_args.py": command,
                "consume_direct_returncode.txt": result.returncode,
                "consume_direct_stdout.txt": result.stdout,
                "consume_direct_stderr.txt": result.stderr,
                "consume_direct.sh+x": consume_direct_script,
            },
        )
        shutil.copyfile(fixture_path, debug_fixture_path)


class BesuTransitionTool(TransitionTool):
    """Besu EvmTool Transition tool frontend wrapper class."""

    default_binary = Path("evm")
    detect_binary_pattern = BESU_BIN_DETECT_PATTERN
    binary: Path
    cached_version: Optional[str] = None
    trace: bool
    process: Optional[subprocess.Popen] = None
    server_url: str
    besu_trace_dir: Optional[tempfile.TemporaryDirectory]

    supports_xdist: ClassVar[bool] = False

    def __init__(
        self,
        *,
        binary: Optional[Path] = None,
        trace: bool = False,
    ):
        """Initialize the BesuTransitionTool class."""
        super().__init__(
            exception_mapper=BesuExceptionMapper(), binary=binary, trace=trace
        )
        args = [str(self.binary), "t8n", "--help"]
        try:
            result = subprocess.run(args, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise Exception(
                "evm process unexpectedly returned a non-zero status "
                f"code: {e}."
            ) from e
        except Exception as e:
            raise Exception(
                f"Unexpected exception calling evm tool: {e}."
            ) from e
        self.help_string = result.stdout
        self.besu_trace_dir = (
            tempfile.TemporaryDirectory() if self.trace else None
        )

    def start_server(self) -> None:
        """
        Start the t8n-server process, extract the port, and leave it
        running for future reuse.
        """
        args = [
            str(self.binary),
            "t8n-server",
            "--port=0",  # OS assigned server port
        ]

        if self.trace:
            args.append("--trace")
            if self.besu_trace_dir:
                args.append(f"--output.basedir={self.besu_trace_dir.name}")

        self.process = subprocess.Popen(
            args=args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        while True:
            if self.process.stdout is None:
                raise Exception("Failed starting Besu subprocess")
            line = str(self.process.stdout.readline())

            if not line or "Failed to start transition server" in line:
                raise Exception("Failed starting Besu subprocess\n" + line)
            if "Transition server listening on" in line:
                match = re.search(
                    "Transition server listening on (\\d+)", line
                )
                if match:
                    port = match.group(1)
                    self.server_url = f"http://localhost:{port}/"
                    break

    def shutdown(self) -> None:
        """Stop the t8n-server process if it was started."""
        if self.process:
            self.process.kill()
        if self.besu_trace_dir:
            self.besu_trace_dir.cleanup()

    def _evaluate(
        self,
        *,
        transition_tool_data: TransitionTool.TransitionToolData,
        debug_output_path: Path | None,
        slow_request: bool,
        profiler: Profiler,
    ) -> TransitionToolOutput:
        """Execute `evm t8n` with the specified arguments."""
        del slow_request, profiler

        if not self.process:
            self.start_server()

        input_json = transition_tool_data.to_input().model_dump(
            mode="json", **model_dump_config
        )

        state_json = {
            "fork": transition_tool_data.fork_name,
            "chainid": transition_tool_data.chain_id,
            "reward": transition_tool_data.reward,
        }

        post_data = {"state": state_json, "input": input_json}

        if debug_output_path:
            post_data_string = json.dumps(post_data, indent=4)
            additional_indent = " " * 16  # for pretty indentation in t8n.sh
            indented_post_data_string = "{\n" + "\n".join(
                additional_indent + line
                for line in post_data_string[1:].splitlines()
            )
            t8n_script = textwrap.dedent(
                f"""\
                #!/bin/bash
                # Use $1 as t8n-server port if provided, else default to 3000
                PORT=${{1:-3000}}
                curl http://localhost:${{PORT}}/ -X POST \\
                -H "Content-Type: application/json" \\
                --data '{indented_post_data_string}'
                """
            )
            dump_files_to_directory(
                debug_output_path,
                {
                    "state.json": state_json,
                    "input/alloc.json": input_json["alloc"],
                    "input/env.json": input_json["env"],
                    "input/txs.json": input_json["txs"],
                    "t8n.sh+x": t8n_script,
                },
            )

        response = requests.post(self.server_url, json=post_data, timeout=5)
        # exception visible in pytest failure output
        response.raise_for_status()
        output: TransitionToolOutput = TransitionToolOutput.model_validate(
            response.json(),
            context={"exception_mapper": self.exception_mapper},
        )

        if debug_output_path:
            dump_files_to_directory(
                debug_output_path,
                {
                    "response.txt": response.text,
                    "status_code.txt": response.status_code,
                    "time_elapsed_seconds.txt": (
                        response.elapsed.total_seconds()
                    ),
                },
            )

        if response.status_code != 200:
            raise Exception(
                f"t8n-server returned status code {response.status_code}, "
                f"response: {response.text}"
            )

        if debug_output_path:
            dump_files_to_directory(
                debug_output_path,
                {
                    "output/alloc.json": output.alloc.raw,
                    "output/result.json": output.result.model_dump(
                        mode="json", **model_dump_config
                    ),
                    "output/txs.rlp": str(output.body),
                },
            )

        if self.trace and self.besu_trace_dir:
            self.collect_traces(
                output.result.receipts, self.besu_trace_dir, debug_output_path
            )
            for i, r in enumerate(output.result.receipts):
                trace_file_name = f"trace-{i}-{r.transaction_hash}.jsonl"
                os.remove(
                    os.path.join(self.besu_trace_dir.name, trace_file_name)
                )

        return output

    def is_fork_supported(self, fork: Fork) -> bool:
        """Return True if the fork is supported by the tool."""
        return fork.transition_tool_name() in self.help_string


class BesuExceptionMapper(ExceptionMapper):
    """Translate between EEST exceptions and error strings returned by Besu."""

    mapping_substring: ClassVar[Dict[ExceptionBase, str]] = {
        TransactionException.NONCE_IS_MAX: "invalid Nonce must be less than",
        TransactionException.INSUFFICIENT_MAX_FEE_PER_BLOB_GAS: (
            "transaction invalid tx max fee per blob gas less than "
            "block blob gas fee"
        ),
        TransactionException.GASLIMIT_PRICE_PRODUCT_OVERFLOW: (
            "invalid Upfront gas cost cannot exceed 2^256 Wei"
        ),
        TransactionException.INSUFFICIENT_MAX_FEE_PER_GAS: (
            "transaction invalid gasPrice is less than the current BaseFee"
        ),
        BlockException.GAS_USED_OVERFLOW: "provided gas insufficient",
        TransactionException.GAS_ALLOWANCE_EXCEEDED: (
            "provided gas insufficient"
        ),
        TransactionException.PRIORITY_GREATER_THAN_MAX_FEE_PER_GAS: (
            "transaction invalid max priority fee per gas cannot be greater "
            "than max fee per gas"
        ),
        TransactionException.TYPE_3_TX_INVALID_BLOB_VERSIONED_HASH: (
            "Invalid versionedHash"
        ),
        TransactionException.TYPE_3_TX_CONTRACT_CREATION: (
            "transaction invalid transaction blob transactions must have "
            "a to address"
        ),
        TransactionException.TYPE_3_TX_WITH_FULL_BLOBS: (
            "Failed to decode transactions from block parameter"
        ),
        TransactionException.TYPE_3_TX_ZERO_BLOBS: (
            "Failed to decode transactions from block parameter"
        ),
        TransactionException.TYPE_3_TX_PRE_FORK: (
            "Transaction type BLOB is invalid, accepted transaction types are"
        ),
        TransactionException.TYPE_4_EMPTY_AUTHORIZATION_LIST: (
            "transaction invalid transaction code delegation transactions "
            "must have a non-empty code delegation list"
        ),
        TransactionException.TYPE_4_TX_CONTRACT_CREATION: (
            "transaction invalid transaction code delegation transactions "
            "must have a to address"
        ),
        TransactionException.TYPE_4_TX_PRE_FORK: (
            "transaction invalid Transaction type DELEGATE_CODE is invalid"
        ),
        BlockException.RLP_STRUCTURES_ENCODING: (
            "Failed to decode transactions from block parameter"
        ),
        BlockException.INCORRECT_EXCESS_BLOB_GAS: (
            "Payload excessBlobGas does not match calculated excessBlobGas"
        ),
        BlockException.BLOB_GAS_USED_ABOVE_LIMIT: (
            "Payload BlobGasUsed does not match calculated BlobGasUsed"
        ),
        BlockException.INCORRECT_BLOB_GAS_USED: (
            "Payload BlobGasUsed does not match calculated BlobGasUsed"
        ),
        BlockException.INVALID_GAS_USED_ABOVE_LIMIT: (
            "Header validation failed (FULL)"
        ),
        BlockException.INVALID_GASLIMIT: "Header validation failed (FULL)",
        BlockException.EXTRA_DATA_TOO_BIG: "Header validation failed (FULL)",
        BlockException.INVALID_BLOCK_NUMBER: (
            "Header validation failed (FULL)"
        ),
        BlockException.INVALID_BASEFEE_PER_GAS: (
            "Header validation failed (FULL)"
        ),
        BlockException.INVALID_BLOCK_TIMESTAMP_OLDER_THAN_PARENT: (
            "block timestamp not greater than parent"
        ),
        BlockException.INVALID_LOG_BLOOM: (
            "failed to validate output of imported block"
        ),
        BlockException.INVALID_RECEIPTS_ROOT: (
            "failed to validate output of imported block"
        ),
        BlockException.INVALID_STATE_ROOT: (
            "World State Root does not match expected value"
        ),
    }
    mapping_regex = {
        BlockException.INVALID_REQUESTS: (
            r"Invalid execution requests|Requests hash mismatch, "
            r"calculated: 0x[0-9a-f]+ header: 0x[0-9a-f]+"
        ),
        BlockException.INVALID_BLOCK_HASH: (
            r"Computed block hash 0x[0-9a-f]+ does not match block "
            r"hash parameter 0x[0-9a-f]+"
        ),
        BlockException.SYSTEM_CONTRACT_CALL_FAILED: (
            r"System call halted|"
            r"System call did not execute to completion"
        ),
        BlockException.SYSTEM_CONTRACT_EMPTY: (
            r"(Invalid system call, no code at address)|"
            r"(Invalid system call address:)"
        ),
        BlockException.INVALID_DEPOSIT_EVENT_LAYOUT: (
            r"Invalid (amount|index|pubKey|signature|withdrawalCred) "
            r"(offset|size): expected (\d+), but got (-?\d+)|"
            r"Invalid deposit log length\. Must be \d+ bytes, "
            r"but is \d+ bytes"
        ),
        BlockException.RLP_BLOCK_LIMIT_EXCEEDED: (
            r"Block size of \d+ bytes exceeds limit of \d+ bytes"
        ),
        TransactionException.INITCODE_SIZE_EXCEEDED: (
            r"transaction invalid Initcode size of \d+ exceeds "
            r"maximum size of \d+"
        ),
        TransactionException.INSUFFICIENT_ACCOUNT_FUNDS: (
            r"transaction invalid transaction up-front cost 0x[0-9a-f]+ "
            r"exceeds transaction sender account balance 0x[0-9a-f]+"
        ),
        TransactionException.INTRINSIC_GAS_TOO_LOW: (
            r"transaction invalid intrinsic gas cost \d+"
            r"(?: \(regular \d+ \+ state \d+\))? "
            r"exceeds gas limit \d+"
        ),
        TransactionException.INTRINSIC_GAS_BELOW_FLOOR_GAS_COST: (
            r"transaction invalid intrinsic gas cost \d+"
            r"(?: \(regular \d+ \+ state \d+\))? "
            r"exceeds gas limit \d+"
        ),
        TransactionException.SENDER_NOT_EOA: (
            r"transaction invalid Sender 0x[0-9a-f]+ has deployed code "
            r"and so is not authorized to send transactions"
        ),
        TransactionException.NONCE_MISMATCH_TOO_LOW: (
            r"transaction invalid transaction nonce \d+ "
            r"below sender account nonce \d+"
        ),
        TransactionException.NONCE_MISMATCH_TOO_HIGH: (
            r"transaction invalid transaction nonce \d+ "
            r"does not match sender account nonce \d+"
        ),
        TransactionException.GAS_LIMIT_EXCEEDS_MAXIMUM: (
            r"transaction invalid Transaction gas limit "
            r"must be at most \d+"
        ),
        TransactionException.TYPE_3_TX_MAX_BLOB_GAS_ALLOWANCE_EXCEEDED: (
            r"Blob transaction 0x[0-9a-f]+ exceeds "
            r"block blob gas limit: \d+ > \d+"
        ),
        TransactionException.TYPE_3_TX_BLOB_COUNT_EXCEEDED: (
            r"Blob transaction has too many blobs: \d+|"
            r"Invalid Blob Count: \d+"
        ),
        # BAL Exceptions
        BlockException.INVALID_BAL_HASH: (
            r"Block access list hash mismatch, "
            r"calculated:\s*(0x[a-f0-9]+)\s+header:\s*(0x[a-f0-9]+)"
        ),
        BlockException.BLOCK_ACCESS_LIST_GAS_LIMIT_EXCEEDED: (
            r"Block access list validation failed for block 0x[a-f0-9]+"
        ),
        BlockException.INVALID_BLOCK_ACCESS_LIST: (
            r"Block access list hash mismatch, "
            r"calculated:\s*(0x[a-f0-9]+)\s+header:\s*(0x[a-f0-9]+)|"
            r"Block access list validation failed for block 0x[a-f0-9]+"
        ),
        BlockException.INCORRECT_BLOCK_FORMAT: (
            r"Block access list hash mismatch, "
            r"calculated:\s*(0x[a-f0-9]+)\s+header:\s*(0x[a-f0-9]+)|"
            r"Block access list validation failed for block 0x[a-f0-9]+"
        ),
    }


class BesuFixtureConsumer(
    BesuEvmTool,
    FixtureConsumerTool,
    fixture_formats=[StateFixture, BlockchainFixture],
):
    """
    Besu's implementation of the fixture consumer.

    Besu's ``evmtool`` pays a large fixed JVM/initialization cost (~2.5s:
    loading the KZG trusted setup, building the reference-test protocol
    schedules, ...) on every process launch, while the marginal cost of an
    extra test within an already-running process is small. Invoking the tool
    once per fixture file (or, worse, once per test) therefore makes Besu
    orders of magnitude slower than clients with negligible startup cost.

    To avoid this, fixtures are consumed in *batches*: a single ``state-test``
    or ``block-test`` process is given many fixture files at once (their paths
    fed via stdin so the command line length limit is never hit) and the
    per-test results are cached. The consume-direct plugin assigns each fixture
    file to a batch *group* and pins all of a group's tests to the same xdist
    worker (``--dist loadgroup``); the first test of a group to run triggers
    the batch for the whole group, and every other test in it is then a cache
    lookup. Each test is fully isolated inside the Besu process (state-test
    copies the initial world state per spec; block-test builds a fresh
    blockchain per test), so batching does not leak state between tests.
    """

    batch_capable: ClassVar[bool] = True

    def __init__(
        self,
        binary: Optional[Path] = None,
        trace: bool = False,
    ):
        """Initialize the Besu fixture consumer and its batch state."""
        super().__init__(binary=binary, trace=trace)
        # group name -> fixture files assigned to it, per format.
        self._state_group_files: Dict[str, List[Path]] = {}
        self._blockchain_group_files: Dict[str, List[Path]] = {}
        # absolute fixture path -> its batch group name.
        self._file_group: Dict[Path, str] = {}
        # groups whose batch has already been executed, per format.
        self._state_batched_groups: Set[str] = set()
        self._blockchain_batched_groups: Set[str] = set()
        # cached results, accumulated across batched groups.
        self._state_results: Dict[str, List[Dict[str, Any]]] = {}
        self._blockchain_ran: Set[str] = set()
        self._blockchain_failed: Dict[str, str] = {}

    def register_batch_group(
        self,
        group: str,
        fixture_format: FixtureFormat,
        fixture_path: Path,
    ) -> None:
        """
        Assign a fixture file to a batch group.

        Called by the consume-direct plugin during collection for every
        selected fixture file, so that when the first test of ``group`` runs
        the consumer knows the full set of files to batch in one process.
        """
        fixture_path = Path(fixture_path)
        self._file_group[fixture_path] = group
        if fixture_format == StateFixture:
            self._state_group_files.setdefault(group, [])
            if fixture_path not in self._state_group_files[group]:
                self._state_group_files[group].append(fixture_path)
        elif fixture_format == BlockchainFixture:
            self._blockchain_group_files.setdefault(group, [])
            if fixture_path not in self._blockchain_group_files[group]:
                self._blockchain_group_files[group].append(fixture_path)

    @property
    def _batching_enabled(self) -> bool:
        """True if any fixture files have been registered for batching."""
        return bool(self._file_group)

    def _run_batch(
        self, subcommand: str, files: List[Path]
    ) -> subprocess.CompletedProcess:
        """
        Run ``evmtool <subcommand>`` over many fixture files in one process.

        File paths are fed via stdin (one per line); Besu reads them when no
        positional file arguments are given, avoiding the command line length
        limit for very large batches.
        """
        command = [str(self.binary), subcommand]
        stdin_input = "".join(f"{path}\n" for path in files)
        result = self._run_command(command, stdin_input=stdin_input)
        if result.returncode != 0:
            raise Exception(
                f"Unexpected exit code running batched {subcommand}:\n"
                f"{' '.join(command)} (over {len(files)} files)\n\n"
                f"Error:\n{result.stderr}"
            )
        for load_error in ("File content error", "File not found"):
            if load_error in result.stdout:
                raise Exception(
                    f"Besu could not load a fixture in the batch "
                    f"({load_error}):\n{result.stdout}\n{result.stderr}"
                )
        return result

    def _ensure_state_group_batched(self, fixture_path: Path) -> None:
        """Run (once) the state-test batch for ``fixture_path``'s group."""
        group = self._file_group[Path(fixture_path)]
        if group in self._state_batched_groups:
            return
        files = self._state_group_files.get(group, [])
        result = self._run_batch("state-test", files)
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise Exception(
                    f"Failed to parse Besu state-test output as JSON.\n"
                    f"Offending line:\n{line}\n\nError: {e}"
                ) from e
            if "test" in entry and "name" not in entry:
                entry["name"] = entry["test"]
            self._state_results.setdefault(entry["name"], []).append(entry)
        self._state_batched_groups.add(group)

    def _ensure_blockchain_group_batched(self, fixture_path: Path) -> None:
        """Run (once) the block-test batch for ``fixture_path``'s group."""
        group = self._file_group[Path(fixture_path)]
        if group in self._blockchain_batched_groups:
            return
        files = self._blockchain_group_files.get(group, [])
        result = self._run_batch("block-test", files)
        in_failures = False
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if in_failures:
                # The failure list ends at the trailing "===" separator.
                if line.startswith("="):
                    in_failures = False
                    continue
                if line.startswith("- "):
                    name, _, reason = line[2:].partition(": ")
                    self._blockchain_failed[name] = reason
                continue
            if line == "Failed tests:":
                in_failures = True
            elif line.startswith("Running iteration"):
                continue
            elif line.startswith("Running "):
                self._blockchain_ran.add(line[len("Running ") :])
        self._blockchain_batched_groups.add(group)

    def consume_blockchain_test(
        self,
        fixture_path: Path,
        fixture_name: Optional[str] = None,
        debug_output_path: Optional[Path] = None,
    ) -> None:
        """
        Consume a single blockchain test.

        When batching is active (the normal consume-direct path), the result
        comes from a single ``block-test`` process shared by the fixture's
        whole batch group. Otherwise Besu's ``evmtool block-test`` is invoked
        for this file alone, using ``--test-name`` to select the fixture.
        """
        if self._batching_enabled and debug_output_path is None:
            assert fixture_name, "batched blockchain tests require a name"
            self._ensure_blockchain_group_batched(fixture_path)
            if fixture_name in self._blockchain_failed:
                raise Exception(
                    f"Blockchain test failed: {fixture_name}: "
                    f"{self._blockchain_failed[fixture_name]}"
                )
            if fixture_name not in self._blockchain_ran:
                raise AssertionError(
                    f"Besu ran no blockchain test for {fixture_name} "
                    f"(vacuous pass)"
                )
            return

        subcommand = "block-test"
        subcommand_options: List[str] = []
        if debug_output_path:
            subcommand_options += ["--json"]

        if fixture_name:
            subcommand_options += [
                "--test-name",
                fixture_name,
            ]

        command = (
            [str(self.binary)]
            + [subcommand]
            + subcommand_options
            + [str(fixture_path)]
        )

        result = self._run_command(command)

        if debug_output_path:
            self._consume_debug_dump(
                command, result, fixture_path, debug_output_path
            )

        if result.returncode != 0:
            raise Exception(
                f"Unexpected exit code:\n{' '.join(command)}\n\n"
                f"Error:\n{result.stderr}"
            )

        # Besu reports fixture load/parse problems (e.g. an unrecognized
        # field in the fixture, or a missing file) on stdout while still
        # exiting 0. Treat those as failures so a fixture that could not be
        # parsed is not silently reported as a pass.
        stdout = result.stdout
        for load_error in ("File content error", "File not found"):
            if load_error in stdout:
                raise Exception(
                    f"Besu could not load the fixture "
                    f"({load_error}):\n{stdout}\n{result.stderr}"
                )

        # Besu prints a "TEST SUMMARY" with `Passed`/`Failed` counts. Require
        # the summary to be present, at least one test to have run, and no
        # failures — otherwise a filter that matched nothing (or output we
        # failed to recognize) would be a silent vacuous pass.
        passed_match = re.search(r"Passed:\s+(\d+)", stdout)
        failed_match = re.search(r"Failed:\s+(\d+)", stdout)
        if passed_match is None or failed_match is None:
            raise Exception(
                f"Could not find Besu test summary; the blockchain test "
                f"may not have run:\n{stdout}\n{result.stderr}"
            )
        if int(failed_match.group(1)) > 0:
            raise Exception(f"Blockchain test failed:\n{stdout}")
        if int(passed_match.group(1)) < 1:
            raise Exception(
                f"Besu ran no blockchain tests (vacuous pass):\n{stdout}"
            )

    @staticmethod
    @cache
    def _load_state_test_fixture_file(fixture_path: Path) -> Dict[str, Any]:
        """Load (and cache) the raw JSON of a state test fixture file."""
        with open(fixture_path) as f:
            fixtures: Dict[str, Any] = json.load(f)
        return fixtures

    def _expected_post_for_result(
        self, fixture_path: Path, result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Return the fixture post entry matching a Besu state-test result.

        The result's fork and data/gas/value indexes select the post
        entry of the named test within the fixture file.
        """
        fixture = self._load_state_test_fixture_file(fixture_path).get(
            result["name"]
        )
        if fixture is None:
            return None
        result_indexes = (
            result.get("d", 0),
            result.get("g", 0),
            result.get("v", 0),
        )
        for post in fixture.get("post", {}).get(result.get("fork"), []):
            post_indexes = post.get("indexes", {})
            if result_indexes == (
                post_indexes.get("data", 0),
                post_indexes.get("gas", 0),
                post_indexes.get("value", 0),
            ):
                return post
        return None

    def _assert_state_test_result(
        self, fixture_path: Path, result: Dict[str, Any]
    ) -> None:
        """
        Assert a single Besu state-test result against the fixture.

        Besu's ``state-test`` reports ``pass: false`` together with a
        ``validationError`` when it rejects a transaction, even when the
        fixture expects exactly that rejection (``expectException``).
        Such a result is a pass if the reported state root also matches
        the fixture's post state (a rejected transaction must not mutate
        state).
        """
        post = self._expected_post_for_result(fixture_path, result)
        expect_exception = post.get("expectException") if post else None
        if not post or expect_exception is None:
            assert result["pass"], (
                f"State test failed: {result.get('error', 'unknown error')}"
            )
            return
        # Besu has two distinct ways of reporting a rejected transaction:
        #
        # 1. When the transaction can be parsed and processed, the rejection
        #    surfaces as ``pass: false`` together with a ``validationError``.
        # 2. When the transaction parameters are so malformed that Besu's
        #    reference-test layer cannot even build a ``Transaction`` (e.g.
        #    empty/invalid blob versioned hashes, a type-3 tx before Cancun),
        #    Besu takes its ``transaction == null`` branch and reports
        #    ``pass: true`` with ``validationError: "Transaction had
        #    out-of-bounds parameters"`` (``pass`` is set to whether an
        #    exception was expected).
        #
        # The processed path always reports ``pass: false`` when an exception
        # is expected, so a ``pass: true`` here can only come from the second
        # path and is itself confirmation that the transaction was rejected.
        if result["pass"]:
            return
        assert result.get("validationError"), (
            f"State test failed: expected the transaction to be "
            f"rejected ({expect_exception}), but it was accepted"
        )
        reported_root = str(result.get("stateRoot", "")).lower()
        expected_root = str(post["hash"]).lower()
        assert reported_root == expected_root, (
            f"State test failed: transaction was rejected as expected "
            f"({expect_exception}), but the reported post state root "
            f"{reported_root} does not match the expected "
            f"{expected_root}"
        )

    @cache  # noqa
    def consume_state_test_file(
        self,
        fixture_path: Path,
        debug_output_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        Consume an entire state test file.

        Besu's ``evmtool state-test`` outputs one JSON object per
        line (NDJSON) with a ``test`` field instead of ``name``.
        This method normalizes the output to match the expected
        format.
        """
        subcommand = "state-test"
        subcommand_options: List[str] = []
        if debug_output_path:
            subcommand_options += ["--json"]

        command = (
            [str(self.binary)]
            + [subcommand]
            + subcommand_options
            + [str(fixture_path)]
        )
        result = self._run_command(command)

        if debug_output_path:
            self._consume_debug_dump(
                command, result, fixture_path, debug_output_path
            )

        if result.returncode != 0:
            raise Exception(
                f"Unexpected exit code:\n{' '.join(command)}\n\n"
                f"Error:\n{result.stderr}"
            )

        # Parse NDJSON output, normalize "test" -> "name"
        results: List[Dict[str, Any]] = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "test" in entry and "name" not in entry:
                    entry["name"] = entry["test"]
                results.append(entry)
            except json.JSONDecodeError as e:
                raise Exception(
                    f"Failed to parse Besu state-test output as JSON.\n"
                    f"Offending line:\n{line}\n\n"
                    f"Error: {e}"
                ) from e
        return results

    def _handle_missing_state_test_result(
        self, fixture_path: Path, fixture_name: str
    ) -> None:
        """
        Handle a fixture for which Besu emitted no state-test result.

        Besu's ``state-test`` runner deliberately skips a test when the
        transaction gas limit exceeds the gas still available in the block
        (``StateTestSubCommand``): that allowance check lives in the block
        importer rather than the transaction processor, so it cannot be
        expressed in a single-transaction state test. Such fixtures expect a
        ``GAS_ALLOWANCE_EXCEEDED`` rejection; surface them as skips rather
        than failures. Any other missing result is a genuine error.
        """
        fixture = self._load_state_test_fixture_file(fixture_path).get(
            fixture_name
        )
        expected_exceptions = {
            str(post.get("expectException"))
            for posts in (fixture or {}).get("post", {}).values()
            for post in posts
            if post.get("expectException")
        }
        if expected_exceptions and all(
            "GAS_ALLOWANCE_EXCEEDED" in exception
            for exception in expected_exceptions
        ):
            pytest.skip(
                "Besu state-test runner skips block-level gas allowance "
                f"checks ({', '.join(sorted(expected_exceptions))}); "
                "not expressible as a state test"
            )
        raise AssertionError(f"Test result for {fixture_name} missing")

    def consume_state_test(
        self,
        fixture_path: Path,
        fixture_name: Optional[str] = None,
        debug_output_path: Optional[Path] = None,
    ) -> None:
        """
        Consume a single state test.

        When batching is active (the normal consume-direct path), the result
        comes from a single ``state-test`` process shared by the fixture's
        whole batch group. Otherwise (e.g. when dumping debug output) the file
        is run on its own via ``consume_state_test_file``.
        """
        if self._batching_enabled and debug_output_path is None:
            self._ensure_state_group_batched(fixture_path)
            assert fixture_name, "batched state tests require a fixture name"
            test_result = self._state_results.get(fixture_name, [])
            assert len(test_result) < 2, (
                f"Multiple test results for {fixture_name}"
            )
            if len(test_result) == 0:
                self._handle_missing_state_test_result(
                    fixture_path, fixture_name
                )
                return
            self._assert_state_test_result(fixture_path, test_result[0])
            return

        file_results = self.consume_state_test_file(
            fixture_path=fixture_path,
            debug_output_path=debug_output_path,
        )
        if fixture_name:
            test_result = [
                r for r in file_results if r["name"] == fixture_name
            ]
            assert len(test_result) < 2, (
                f"Multiple test results for {fixture_name}"
            )
            if len(test_result) == 0:
                self._handle_missing_state_test_result(
                    fixture_path, fixture_name
                )
                return
            self._assert_state_test_result(fixture_path, test_result[0])
        else:
            errors = []
            for r in file_results:
                try:
                    self._assert_state_test_result(fixture_path, r)
                except AssertionError as e:
                    errors.append(f"{r['name']}: {e}")
            if errors:
                raise Exception("State test failed: \n" + "\n".join(errors))

    def consume_fixture(
        self,
        fixture_format: FixtureFormat,
        fixture_path: Path,
        fixture_name: Optional[str] = None,
        debug_output_path: Optional[Path] = None,
    ) -> None:
        """
        Execute the appropriate Besu fixture consumer for the
        fixture at ``fixture_path``.
        """
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
