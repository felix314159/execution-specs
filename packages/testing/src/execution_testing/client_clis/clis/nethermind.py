"""Interfaces for Nethermind CLIs."""

import json
import re
import shlex
import subprocess
import textwrap
from functools import cache
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

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

from ..ethereum_cli import EthereumCLI
from ..file_utils import dump_files_to_directory
from ..fixture_consumer_tool import FixtureConsumerTool


class Nethtest(EthereumCLI):
    """Nethermind `nethtest` binary base class."""

    default_binary = Path("nethtest")
    # new pattern allows e.g. '1.2.3', in the past that was denied
    detect_binary_pattern = re.compile(
        r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?(\+[a-f0-9]{40})?$"
    )
    version_flag: str = "--version"
    cached_version: Optional[str] = None

    def __init__(
        self,
        binary: Path,
        trace: bool = False,
        exception_mapper: ExceptionMapper | None = None,
    ):
        """Initialize the Nethtest class."""
        self.binary = binary
        self.trace = trace
        # TODO: Implement NethermindExceptionMapper
        self.exception_mapper = exception_mapper if exception_mapper else None

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
            raise Exception("Unexpected exception calling evm tool.") from e

    def _consume_debug_dump(
        self,
        command: Tuple[str, ...],
        result: subprocess.CompletedProcess,
        debug_output_path: Path,
    ) -> None:
        # our assumption is that each command element is a string
        assert all(isinstance(x, str) for x in command), (
            f"Not all elements of 'command' list are strings: {command}"
        )

        # ensure that flags with spaces are wrapped in double-quotes
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

    @cache  # noqa
    def help(self, subcommand: str | None = None) -> str:
        """Return the help string, optionally for a subcommand."""
        help_command = [str(self.binary)]
        if subcommand:
            help_command.append(subcommand)
        help_command.append("--help")
        return self._run_command(help_command).stdout


class NethtestFixtureConsumer(
    Nethtest,
    FixtureConsumerTool,
    fixture_formats=[StateFixture, BlockchainFixture],
):
    """
    Nethermind implementation of the fixture consumer.

    `nethtest` is a .NET binary that pays a large fixed startup cost (~2.6s:
    CLR/JIT warm-up plus spec/genesis setup) on every process launch, while
    the marginal cost of an extra test in an already-running process is small.
    Invoking it once per fixture file (state) or once per test (blockchain)
    therefore makes Nethermind orders of magnitude slower than clients with
    negligible startup (e.g. ~62s vs ~5s for 23 tests, single-threaded).

    To avoid this, fixtures are consumed in *batches*. `nethtest --stdin`
    reads fixture-file paths one per line from stdin and runs each in the same
    process until stdin closes, so a single process consumes many files. The
    consume-direct plugin assigns each fixture file to a batch *group* and
    pins all of a group's tests to one xdist worker (`--dist loadgroup`); the
    first test of a group triggers the batch for the whole group, and every
    other test in it is then a cache lookup. State and blockchain tests are
    batched separately because the test kind is a process-level flag
    (`--blockTest`). This mirrors the Besu fixture consumer.
    """

    batch_capable: ClassVar[bool] = True

    def __init__(
        self,
        binary: Path,
        trace: bool = False,
        exception_mapper: ExceptionMapper | None = None,
    ):
        """Initialize the Nethermind fixture consumer and its batch state."""
        super().__init__(
            binary=binary, trace=trace, exception_mapper=exception_mapper
        )
        # group name -> fixture files assigned to it, per format.
        self._state_group_files: Dict[str, List[Path]] = {}
        self._blockchain_group_files: Dict[str, List[Path]] = {}
        # absolute fixture path -> its batch group name.
        self._file_group: Dict[Path, str] = {}
        # groups whose batch has already been executed, per format.
        self._state_batched_groups: Set[str] = set()
        self._blockchain_batched_groups: Set[str] = set()
        # cached results, accumulated across batched groups.
        self._state_results_by_name: Dict[str, Dict[str, Any]] = {}
        self._state_stderr: str = ""
        self._blockchain_statuses: Dict[str, str] = {}

    def register_batch_group(
        self,
        group: str,
        fixture_format: FixtureFormat,
        fixture_path: Path,
    ) -> None:
        """
        Assign a fixture file to a batch group.

        Called by the consume-direct plugin during collection for every
        selected fixture file, so that when the first test of `group` runs
        the consumer knows the full set of files to batch in one process.
        """
        fixture_path = Path(fixture_path)
        self._file_group[fixture_path] = group
        if fixture_format is StateFixture:
            self._state_group_files.setdefault(group, [])
            if fixture_path not in self._state_group_files[group]:
                self._state_group_files[group].append(fixture_path)
        elif fixture_format is BlockchainFixture:
            self._blockchain_group_files.setdefault(group, [])
            if fixture_path not in self._blockchain_group_files[group]:
                self._blockchain_group_files[group].append(fixture_path)

    @property
    def _batching_enabled(self) -> bool:
        """True if any fixture files have been registered for batching."""
        return bool(self._file_group)

    def _run_stdin_batch(
        self, files: List[Path], *, block_test: bool
    ) -> subprocess.CompletedProcess:
        """
        Run `nethtest --stdin` over many fixture files in one process.

        File paths are fed via stdin (one per line); `nethtest` runs each and
        loops until stdin closes, amortizing the fixed startup cost over the
        whole batch and avoiding the command line length limit.
        """
        command = [str(self.binary), "--stdin"]
        if block_test:
            command.append("--blockTest")
        stdin_input = "".join(f"{path}\n" for path in files)
        # Note: a non-zero exit is not raised here. `nethtest` processes the
        # piped files sequentially and can throw an unhandled exception on a
        # single bad fixture (e.g. a blockchain post-state mismatch), aborting
        # the batch. The verdicts printed before that point are still valid, so
        # the caller parses what it can and falls back to per-test execution
        # for any fixture the batch did not report on.
        return subprocess.run(
            command,
            input=stdin_input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    @staticmethod
    def _parse_concatenated_state_results(
        stdout: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse `nethtest --stdin` state output into a flat list of results.

        With `--stdin` the runner prints one JSON array per input file, and
        the arrays are concatenated back-to-back on stdout (e.g. `]​[`), so the
        whole stream is not a single JSON document. Decode the arrays one at a
        time and flatten their entries.
        """
        decoder = json.JSONDecoder()
        results: List[Dict[str, Any]] = []
        index = 0
        length = len(stdout)
        while index < length:
            # Skip the whitespace between consecutive JSON arrays.
            while index < length and stdout[index].isspace():
                index += 1
            if index >= length:
                break
            try:
                array, end = decoder.raw_decode(stdout, index)
            except json.JSONDecodeError:
                # A trailing incomplete array means the batch aborted partway
                # (e.g. nethtest threw on a bad fixture). Keep the complete
                # results parsed so far; the rest fall back to per-test runs.
                break
            if isinstance(array, list):
                results.extend(array)
            index = end
        return results

    def _ensure_state_group_batched(self, fixture_path: Path) -> None:
        """Run (once) the state-test batch for `fixture_path`'s group."""
        group = self._file_group[Path(fixture_path)]
        if group in self._state_batched_groups:
            return
        files = self._state_group_files.get(group, [])
        result = self._run_stdin_batch(files, block_test=False)
        self._state_stderr = result.stderr
        for entry in self._parse_concatenated_state_results(result.stdout):
            self._state_results_by_name[entry["name"]] = entry
        self._state_batched_groups.add(group)

    def _ensure_blockchain_group_batched(self, fixture_path: Path) -> None:
        """Run (once) the block-test batch for `fixture_path`'s group."""
        group = self._file_group[Path(fixture_path)]
        if group in self._blockchain_batched_groups:
            return
        files = self._blockchain_group_files.get(group, [])
        result = self._run_stdin_batch(files, block_test=True)
        self._blockchain_statuses.update(
            self._parse_blocktest_statuses(result.stdout)
        )
        self._blockchain_batched_groups.add(group)

    def _consume_single_unbatched(
        self,
        fixture_format: FixtureFormat,
        fixture_path: Path,
        fixture_name: str,
    ) -> None:
        """
        Run one fixture on its own via the non-batched per-file/per-test path.

        Used as a fallback for a fixture the batch did not report on (because
        `nethtest` threw on it or on an earlier file and aborted the batch),
        so each such test still gets an accurate, isolated verdict.
        """
        command = self._build_command_with_options(
            fixture_format, fixture_path, fixture_name, None
        )
        if fixture_format is BlockchainFixture:
            self.consume_blockchain_test(
                command=command,
                fixture_path=fixture_path,
                fixture_name=fixture_name,
            )
        else:
            self.consume_state_test(
                command=command,
                fixture_path=fixture_path,
                fixture_name=fixture_name,
            )

    def _consume_state_test_batched(
        self, fixture_path: Path, fixture_name: str
    ) -> None:
        """Assert a single state test from its batched group result."""
        self._ensure_state_group_batched(fixture_path)
        nethtest_suffix = "_d0g0v0_"
        short_fixture_name = self._nethtest_state_test_name(fixture_name)
        entry = self._state_results_by_name.get(
            short_fixture_name + nethtest_suffix
        )
        if entry is None:
            # A result under a different data index means the test uses the
            # multi-data ethereum/tests format, which is not supported (yet);
            # fail loudly rather than silently pass the wrong index.
            assert not any(
                name.startswith(short_fixture_name + "_d")
                for name in self._state_results_by_name
            ), (
                "consume direct with nethtest doesn't support the "
                "multi-data statetest format used in ethereum/tests (yet)"
            )
            # The batch aborted before reporting this test; run it alone.
            self._consume_single_unbatched(
                StateFixture, fixture_path, fixture_name
            )
            return
        assert entry["pass"], (
            f"State test '{fixture_name}' failed, "
            f"available stderr:\n {self._state_stderr}"
        )

    def _consume_blockchain_test_batched(
        self, fixture_path: Path, fixture_name: str
    ) -> None:
        """Assert a single blockchain test from its batched group result."""
        self._ensure_blockchain_group_batched(fixture_path)
        short_fixture_name = fixture_name.rsplit("::", maxsplit=1)[-1]
        status = self._blockchain_statuses.get(short_fixture_name)
        if status is None:
            # The batch aborted before reporting this test (nethtest throws on
            # some invalid blocks rather than printing a verdict); run it alone
            # so it gets an accurate pass/fail instead of a vacuous result.
            self._consume_single_unbatched(
                BlockchainFixture, fixture_path, fixture_name
            )
            return
        if status != "PASS":
            raise Exception(f"Blockchain test '{short_fixture_name}' failed.")

    def _build_command_with_options(
        self,
        fixture_format: FixtureFormat,
        fixture_path: Path,
        fixture_name: Optional[str] = None,
        debug_output_path: Optional[Path] = None,
    ) -> Tuple[str, ...]:
        assert fixture_name, "Fixture name must be provided for nethtest."
        command = [str(self.binary)]
        if fixture_format is BlockchainFixture:
            # nethtest names blockchain tests with the short (post-`::`) name
            # only, and matches `--filter` as `^(<filter>)` against it. Passing
            # the full `path::name` would match nothing, so nethtest would run
            # zero tests and exit 0 — a silent (vacuous) pass.
            short_fixture_name = fixture_name.rsplit("::", maxsplit=1)[-1]
            command += [
                "--blockTest",
                "--filter",
                f"{re.escape(short_fixture_name)}",
            ]
        elif fixture_format is StateFixture:
            # TODO: consider using `--filter` here to readily access traces
            # from the output
            pass  # no additional options needed
        else:
            raise Exception(
                f"Fixture format {fixture_format.format_name} "
                f"not supported by {self.binary}"
            )
        command += ["--input", str(fixture_path)]
        if debug_output_path:
            command += ["--trace"]
        return tuple(command)

    @cache  # noqa
    def consume_state_test_file(
        self,
        fixture_path: Path,
        command: Tuple[str, ...],
        debug_output_path: Optional[Path] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Consume an entire state test file.

        The `evm statetest` will always execute all the tests contained in a
        file without the possibility of selecting a single test, so this
        function is cached in order to only call the command once and
        `consume_state_test` can simply select the result that was requested.
        """
        del fixture_path
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if debug_output_path:
            self._consume_debug_dump(command, result, debug_output_path)

        if result.returncode != 0:
            raise Exception(
                f"Unexpected exit code:\n{' '.join(command)}\n\n"
                f"Error:\n{result.stderr}"
            )

        try:
            result_json = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise Exception(
                f"Failed to parse JSON output on stdout from nethtest:\n"
                f"{result.stdout}"
            ) from e

        if not isinstance(result_json, list):
            raise Exception(
                f"Unexpected result from evm statetest: {result_json}"
            )
        return result_json, result.stderr

    @staticmethod
    def _nethtest_state_test_name(fixture_name: str) -> str:
        """
        Reproduce the name `nethtest` derives from a state test fixture id.

        `nethtest` builds the per-result `name` from the fixture id by
        first taking the path basename (the substring after the last `/`)
        and then, if that still contains `.py::`, the substring after it.
        This drops the directory and module-file prefix while keeping any
        pytest class segment (`Class::method`).

        Mirroring this derivation here (rather than naively stripping the
        module path with `rsplit("::")`) lets us match results both for
        tests grouped under a class and for tests whose parametrization id
        itself contains a `/` (e.g. the EIP-7951 wycheproof vectors, which
        reference files like `wycheproof/...test.json`) — `nethtest` names
        both differently from the bare post-`::` name.
        """
        name = fixture_name.rsplit("/", maxsplit=1)[-1]
        if ".py::" in name:
            name = name.split(".py::", maxsplit=1)[-1]
        return name

    def consume_state_test(
        self,
        command: Tuple[str, ...],
        fixture_path: Path,
        fixture_name: Optional[str] = None,
        debug_output_path: Optional[Path] = None,
    ) -> None:
        """
        Consume a single state test.

        Uses the cached result from `consume_state_test_file` in order to not
        call the command every time and select a single result from there.
        """
        file_results, stderr = self.consume_state_test_file(
            fixture_path=fixture_path,
            command=command,
            debug_output_path=debug_output_path,
        )

        if fixture_name:
            # TODO: this check is too fragile; extend for ethereum/tests?
            nethtest_suffix = "_d0g0v0_"
            short_fixture_name = self._nethtest_state_test_name(fixture_name)
            assert all(
                test_result["name"].endswith(nethtest_suffix)
                for test_result in file_results
            ), (
                "consume direct with nethtest doesn't support the "
                "multi-data statetest format used in ethereum/tests (yet)"
            )
            test_result = [
                test_result
                for test_result in file_results
                if test_result["name"].removesuffix(nethtest_suffix)
                == short_fixture_name
            ]
            assert len(test_result) < 2, (
                f"Multiple test results for {fixture_name}"
            )
            assert len(test_result) == 1, (
                f"Test result for {fixture_name} missing"
            )
            assert test_result[0]["pass"], (
                f"State test '{fixture_name}' failed, "
                f"available stderr:\n {stderr}"
            )
        else:
            if any(not test_result["pass"] for test_result in file_results):
                exception_text = "State test failed: \n" + "\n".join(
                    f"{test_result['name']}: " + test_result["error"]
                    for test_result in file_results
                    if not test_result["pass"]
                )
                raise Exception(exception_text)

    @staticmethod
    def _parse_blocktest_statuses(stdout: str) -> Dict[str, str]:
        """
        Map each executed blockchain test name to its `PASS`/`FAIL` verdict.

        nethtest prints one line per executed test: the (short) test name
        left-padded to a fixed width, followed by `PASS` or `FAIL`. Any
        leftover ANSI color codes are stripped before parsing.
        """
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        statuses: Dict[str, str] = {}
        for raw_line in stdout.splitlines():
            line = ansi_escape.sub("", raw_line).rstrip()
            for status in ("PASS", "FAIL"):
                if line.endswith(status):
                    name = line[: -len(status)].strip()
                    if name:
                        statuses[name] = status
                    break
        return statuses

    def consume_blockchain_test(
        self,
        command: Tuple[str, ...],
        fixture_path: Path,
        fixture_name: Optional[str] = None,
        debug_output_path: Optional[Path] = None,
    ) -> None:
        """Execute the the fixture at `fixture_path` via `nethtest`."""
        del fixture_path
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if debug_output_path:
            self._consume_debug_dump(command, result, debug_output_path)

        # A non-zero exit code signals a hard failure (e.g. nethtest throws an
        # unhandled assertion on a post-state mismatch before printing a
        # verdict), so surface it directly.
        if result.returncode != 0:
            raise Exception(
                f"nethtest exited with non-zero exit code "
                f"({result.returncode}).\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}\n"
                f"{' '.join(command)}"
            )

        # A zero exit code is not sufficient: nethtest exits 0 both when the
        # `--filter` matches no test (running nothing) and when a test runs but
        # reports `FAIL` (e.g. an invalid block that was wrongly accepted).
        # Parse the per-test verdict to reject both cases.
        statuses = self._parse_blocktest_statuses(result.stdout)
        if fixture_name is not None:
            short_fixture_name = fixture_name.rsplit("::", maxsplit=1)[-1]
            assert short_fixture_name in statuses, (
                f"nethtest ran no blockchain test matching "
                f"'{short_fixture_name}' (filter matched nothing).\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}\n"
                f"{' '.join(command)}"
            )
            if statuses[short_fixture_name] != "PASS":
                raise Exception(
                    f"Blockchain test '{short_fixture_name}' failed.\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
        else:
            if not statuses:
                raise Exception(
                    f"nethtest ran no blockchain tests.\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
            failed = [
                name for name, status in statuses.items() if status != "PASS"
            ]
            if failed:
                raise Exception(
                    "Blockchain test(s) failed: "
                    + ", ".join(failed)
                    + f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )

    def consume_fixture(
        self,
        fixture_format: FixtureFormat,
        fixture_path: Path,
        fixture_name: Optional[str] = None,
        debug_output_path: Optional[Path] = None,
    ) -> None:
        """
        Execute the appropriate nethtest fixture consumer for the fixture at
        `fixture_path`.
        """
        # Batched path (the normal consume-direct run): the result comes from a
        # single `nethtest --stdin` process shared by the fixture's whole batch
        # group. Skipped when dumping debug output (`--trace` per file) so a
        # single failing fixture can still be investigated in isolation.
        if self._batching_enabled and debug_output_path is None:
            assert fixture_name, "Fixture name must be provided for nethtest."
            if fixture_format is BlockchainFixture:
                self._consume_blockchain_test_batched(
                    fixture_path, fixture_name
                )
            elif fixture_format is StateFixture:
                self._consume_state_test_batched(fixture_path, fixture_name)
            else:
                raise Exception(
                    f"Fixture format {fixture_format.format_name} "
                    f"not supported by {self.binary}"
                )
            return

        command = self._build_command_with_options(
            fixture_format, fixture_path, fixture_name, debug_output_path
        )
        if fixture_format == BlockchainFixture:
            self.consume_blockchain_test(
                command=command,
                fixture_path=fixture_path,
                fixture_name=fixture_name,
                debug_output_path=debug_output_path,
            )
        elif fixture_format == StateFixture:
            self.consume_state_test(
                command=command,
                fixture_path=fixture_path,
                fixture_name=fixture_name,
                debug_output_path=debug_output_path,
            )
        else:
            raise Exception(
                f"Fixture format {fixture_format.format_name} "
                f"not supported by {self.binary}"
            )


class NethermindExceptionMapper(ExceptionMapper):
    """Nethermind exception mapper."""

    mapping_substring = {
        TransactionException.SENDER_NOT_EOA: "sender has deployed code",
        TransactionException.INTRINSIC_GAS_TOO_LOW: "intrinsic gas too low",
        TransactionException.INTRINSIC_GAS_BELOW_FLOOR_GAS_COST: (
            "intrinsic gas too low"
        ),
        TransactionException.INSUFFICIENT_MAX_FEE_PER_GAS: (
            "miner premium is negative"
        ),
        TransactionException.PRIORITY_GREATER_THAN_MAX_FEE_PER_GAS: (
            "InvalidMaxPriorityFeePerGas: Cannot be higher than maxFeePerGas"
        ),
        TransactionException.GAS_ALLOWANCE_EXCEEDED: (
            "Block gas limit exceeded"
        ),
        TransactionException.NONCE_IS_MAX: "NonceTooHigh",
        TransactionException.INITCODE_SIZE_EXCEEDED: (
            "max initcode size exceeded"
        ),
        TransactionException.NONCE_MISMATCH_TOO_LOW: (
            "transaction nonce is too low"
        ),
        TransactionException.NONCE_MISMATCH_TOO_HIGH: (
            "transaction nonce is too high"
        ),
        TransactionException.INSUFFICIENT_MAX_FEE_PER_BLOB_GAS: (
            "InsufficientMaxFeePerBlobGas: Not enough to cover blob gas fee"
        ),
        TransactionException.TYPE_1_TX_PRE_FORK: (
            "InvalidTxType: Transaction type in Custom is not supported"
        ),
        TransactionException.TYPE_2_TX_PRE_FORK: (
            "InvalidTxType: Transaction type in Custom is not supported"
        ),
        TransactionException.TYPE_3_TX_PRE_FORK: (
            "InvalidTxType: Transaction type in Custom is not supported"
        ),
        TransactionException.TYPE_3_TX_ZERO_BLOBS: (
            "blob transaction must have at least 1 blob"
        ),
        TransactionException.TYPE_3_TX_INVALID_BLOB_VERSIONED_HASH: (
            "InvalidBlobVersionedHashVersion: Blob version not supported"
        ),
        TransactionException.TYPE_3_TX_CONTRACT_CREATION: (
            "blob transaction of type create"
        ),
        TransactionException.TYPE_4_EMPTY_AUTHORIZATION_LIST: (
            "EIP-7702 transaction with empty auth list"
        ),
        TransactionException.TYPE_4_TX_CONTRACT_CREATION: (
            "EIP-7702 transaction cannot be used to create contract"
        ),
        TransactionException.TYPE_4_TX_PRE_FORK: (
            "InvalidTxType: Transaction type in Custom is not supported"
        ),
        BlockException.INCORRECT_BLOB_GAS_USED: (
            "HeaderBlobGasMismatch: "
            "Blob gas in header does not match calculated"
        ),
        BlockException.INVALID_REQUESTS: (
            "InvalidRequestsHash: Requests hash mismatch in block"
        ),
        BlockException.INVALID_GAS_USED_ABOVE_LIMIT: (
            "ExceededGasLimit: Gas used exceeds gas limit."
        ),
        BlockException.RLP_BLOCK_LIMIT_EXCEEDED: (
            "ExceededBlockSizeLimit: Exceeded block size limit"
        ),
        BlockException.INVALID_DEPOSIT_EVENT_LAYOUT: (
            "DepositsInvalid: Invalid deposit event layout:"
        ),
        BlockException.INVALID_BASEFEE_PER_GAS: (
            "InvalidBaseFeePerGas: Does not match calculated"
        ),
        BlockException.INVALID_BLOCK_TIMESTAMP_OLDER_THAN_PARENT: (
            "InvalidTimestamp: "
            "Timestamp in header cannot be lower than ancestor"
        ),
        BlockException.INVALID_BLOCK_NUMBER: (
            "InvalidBlockNumber: Block number does not match the parent"
        ),
        BlockException.EXTRA_DATA_TOO_BIG: (
            "InvalidExtraData: Extra data in header is not valid"
        ),
        BlockException.INVALID_GASLIMIT: (
            "InvalidGasLimit: Gas limit is not correct"
        ),
        BlockException.INVALID_RECEIPTS_ROOT: (
            "InvalidReceiptsRoot: Receipts root in header does not match"
        ),
        BlockException.INVALID_LOG_BLOOM: (
            "InvalidLogsBloom: Logs bloom in header does not match"
        ),
        BlockException.INVALID_STATE_ROOT: (
            "InvalidStateRoot: State root in header does not match"
        ),
        BlockException.GAS_USED_OVERFLOW: ("Block gas limit exceeded"),
        BlockException.BLOCK_ACCESS_LIST_GAS_LIMIT_EXCEEDED: (
            "BlockAccessListGasLimitExceeded:"
        ),
    }
    mapping_regex = {
        TransactionException.INSUFFICIENT_ACCOUNT_FUNDS: (
            r"insufficient sender balance|"
            r"insufficient MaxFeePerGas for sender balance"
            r"|insufficient funds for gas \* price \+ value"
            r"|insufficient funds for transfer|insufficient funds for gas"
        ),
        TransactionException.INSUFFICIENT_MAX_FEE_PER_GAS: (
            r"max fee per gas less than block base fee"
        ),
        TransactionException.NONCE_MISMATCH_TOO_LOW: (r"nonce too low"),
        TransactionException.NONCE_MISMATCH_TOO_HIGH: (r"nonce too high"),
        TransactionException.TYPE_3_TX_WITH_FULL_BLOBS: (
            r"Transaction \d+ is not valid"
        ),
        TransactionException.TYPE_3_TX_MAX_BLOB_GAS_ALLOWANCE_EXCEEDED: (
            r"BlockBlobGasExceeded: A block cannot have more than "
            r"\d+ blob gas, blobs count \d+, blobs gas used: \d+"
        ),
        TransactionException.TYPE_3_TX_BLOB_COUNT_EXCEEDED: (
            r"BlobTxGasLimitExceeded: Transaction's totalDataGas=\d+ "
            r"exceeded MaxBlobGas per transaction=\d+"
        ),
        TransactionException.GAS_LIMIT_EXCEEDS_MAXIMUM: (
            r"TxGasLimitCapExceeded:"
        ),
        BlockException.INCORRECT_EXCESS_BLOB_GAS: (
            r"HeaderExcessBlobGasMismatch: Excess blob gas in header "
            r"does not match calculated|Overflow in excess blob gas"
        ),
        BlockException.INVALID_BLOCK_HASH: (
            r"Invalid block hash 0x[0-9a-f]+ does not match "
            r"calculated hash 0x[0-9a-f]+"
        ),
        BlockException.SYSTEM_CONTRACT_EMPTY: (
            r"(Withdrawals|Consolidations)Empty: Contract is not deployed\."
        ),
        BlockException.SYSTEM_CONTRACT_CALL_FAILED: (
            r"(Withdrawals|Consolidations)Failed: Contract execution failed\."
        ),
        # BAL Exceptions — specific exceptions have unique patterns, but
        # INVALID_BLOCK_ACCESS_LIST and INCORRECT_BLOCK_FORMAT intentionally
        # overlap because the test framework requires `want in got` matching.
        # BAL Exceptions
        BlockException.INVALID_BAL_HASH: (r"InvalidBlockLevelAccessListHash:"),
        BlockException.INVALID_BLOCK_ACCESS_LIST: (
            r"InvalidBlockLevelAccessListHash:"
            r"|InvalidBlockLevelAccessList:"
            r"|BlockLevelAccessListIndexOutOfRange:"
            r"|could not be parsed as a block: "
            r"Error decoding block access list:"
            r"|Error decoding block access list:"
        ),
        BlockException.INCORRECT_BLOCK_FORMAT: (
            r"could not be parsed as a block: "
            r"Error decoding block access list:"
            r"|Error decoding block access list:"
        ),
        TransactionException.GAS_ALLOWANCE_EXCEEDED: (
            r"TxGasLimitCapExceeded:"
            r"|BlockAccessListGasLimitExceeded:"
        ),
    }
