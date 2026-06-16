"""CLI entry point for the `consume` pytest-based command."""

import functools
import os
from pathlib import Path
from typing import Any, Callable, List

import click
from rich.console import Console

from .base import (
    ArgumentProcessor,
    PytestCommand,
    PytestExecution,
    common_pytest_options,
)
from .processors import (
    ConsumeCommandProcessor,
    HelpFlagsProcessor,
    HiveEnvironmentProcessor,
)

# Below this many selected tests, the Docker path runs serially: starting an
# xdist worker (and its per-worker container + resident shell) costs on the
# order of ~15s of fixed overhead, which only pays off once there is enough
# work to spread across the cores. Measured crossover on a 16-core machine sat
# around ~4–5k tests (e.g. ~2k tests: ~10s serial vs ~19s parallel; ~51k
# tests: ~240s serial vs ~70s parallel), so the threshold is set near it. The
# decision is logged, so it stays transparent and easy to retune.
DOCKER_SERIAL_TEST_THRESHOLD = 4000


def uses_docker_backend(args: List[str]) -> bool:
    """Return True if the args select the Docker client-image backend."""
    return any(
        arg == "--docker.client-branches"
        or arg.startswith("--docker.client-branches=")
        for arg in args
    )


def has_explicit_parallelism(args: List[str]) -> bool:
    """Return True if the user already chose a worker count."""
    return any(
        arg in ("-n", "--numprocesses")
        or arg.startswith("-n=")
        or arg.startswith("--numprocesses=")
        for arg in args
    )


def collects_no_tests(args: List[str]) -> bool:
    """Return True for runs that never execute tests (so workers are moot)."""
    return any(
        arg in ("--collect-only", "--co", "--docker.build-only")
        for arg in args
    )


class ConsumeDirectCommand(PytestCommand):
    """
    ``consume direct`` command that auto-selects xdist parallelism for Docker.

    The Docker backend parallelizes near-linearly (one container + resident
    shell per worker), but starting those workers has a fixed cost that only
    pays off with enough tests. When the Docker backend is used and the user
    did not pass ``-n``, this counts the selected tests with a quick in-process
    ``--collect-only`` pass, then runs serially or with ``-n auto`` accordingly
    — logging the count and the decision. The local ``--bin`` path and any
    explicit ``-n`` are left untouched.
    """

    def create_executions(
        self, pytest_args: List[str]
    ) -> List[PytestExecution]:
        """Build the execution(s), choosing parallelism for Docker runs."""
        processed_args = self.process_arguments(pytest_args)
        if (
            uses_docker_backend(processed_args)
            and not has_explicit_parallelism(processed_args)
            and not collects_no_tests(processed_args)
        ):
            processed_args = self._with_parallelism(processed_args)
        return [
            PytestExecution(
                config_file=self.config_path,
                command_logic_test_paths=self.test_args,
                args=processed_args,
                allowed_exit_codes=self.allowed_exit_codes,
            )
        ]

    def _with_parallelism(self, args: List[str]) -> List[str]:
        """Count the selected tests and append the chosen ``-n`` setting."""
        console = Console(stderr=True, highlight=False)
        count = self.runner.count_selected_tests(
            PytestExecution(
                config_file=self.config_path,
                command_logic_test_paths=self.test_args,
                args=args,
            )
        )
        if count is None:
            console.print(
                "[yellow]consume direct: could not pre-count tests; "
                "leaving the run serial.[/yellow]"
            )
            return args + ["-n", "0"]
        if count <= DOCKER_SERIAL_TEST_THRESHOLD:
            console.print(
                f"[bold]consume direct:[/bold] {count} tests collected "
                f"(≤ {DOCKER_SERIAL_TEST_THRESHOLD}) → running "
                "[bold]serially[/bold]; xdist worker startup would cost more "
                "than it saves at this size."
            )
            return args + ["-n", "0"]
        cores = os.cpu_count() or 1
        console.print(
            f"[bold]consume direct:[/bold] {count} tests collected "
            f"(> {DOCKER_SERIAL_TEST_THRESHOLD}) → running in "
            f"[bold]parallel[/bold] with `-n auto` (up to {cores} workers) to "
            "use all cores."
        )
        return args + ["-n", "auto"]


def create_consume_command(
    *,
    command_logic_test_paths: List[Path],
    is_hive: bool = False,
    command_name: str = "",
) -> PytestCommand:
    """Initialize consume command with paths and processors."""
    processors: List[ArgumentProcessor] = [HelpFlagsProcessor("consume")]

    if is_hive:
        processors.extend(
            [
                HiveEnvironmentProcessor(command_name=command_name),
                ConsumeCommandProcessor(is_hive=True),
            ]
        )
    else:
        processors.append(ConsumeCommandProcessor(is_hive=False))

    # `consume direct` auto-selects parallelism for the Docker backend; other
    # non-hive entry points (e.g. `cache`) keep the plain command.
    command_class = (
        ConsumeDirectCommand if command_name == "direct" else PytestCommand
    )
    return command_class(
        config_file="pytest-consume.ini",
        argument_processors=processors,
        command_logic_test_paths=command_logic_test_paths,
    )


def get_command_logic_test_paths(command_name: str) -> List[Path]:
    """Determine the command paths based on the command name and hive flag."""
    base_path = Path("cli/pytest_commands/plugins/consume")
    if command_name in ["engine", "enginex", "rlp"]:
        test_command = "engine" if command_name == "enginex" else command_name
        command_logic_test_paths = [
            base_path
            / "simulators"
            / "simulator_logic"
            / f"test_via_{test_command}.py"
        ]
    elif command_name == "sync":
        command_logic_test_paths = [
            base_path / "simulators" / "simulator_logic" / "test_via_sync.py"
        ]
    elif command_name == "direct":
        command_logic_test_paths = [
            base_path / "direct" / "test_via_direct.py"
        ]
    else:
        raise ValueError(f"Unexpected command: {command_name}.")
    return command_logic_test_paths


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def consume() -> None:
    """Consume command to aid client consumption of test fixtures."""
    pass


def consume_command(
    is_hive: bool = False,
) -> Callable[[Callable[..., Any]], click.Command]:
    """Generate a consume sub-command."""

    def decorator(func: Callable[..., Any]) -> click.Command:
        command_name = func.__name__
        command_help = func.__doc__
        command_logic_test_paths = get_command_logic_test_paths(command_name)

        @consume.command(
            name=command_name,
            help=command_help,
            context_settings={"ignore_unknown_options": True},
        )
        @common_pytest_options
        @functools.wraps(func)
        def command(pytest_args: List[str], **kwargs: Any) -> None:
            del kwargs

            consume_cmd = create_consume_command(
                command_logic_test_paths=command_logic_test_paths,
                is_hive=is_hive,
                command_name=command_name,
            )
            consume_cmd.execute(list(pytest_args))

        return command

    return decorator


@consume_command(is_hive=False)
def direct() -> None:
    """Clients consume directly via the `blocktest` interface."""
    pass


@consume_command(is_hive=True)
def rlp() -> None:
    """Client consumes RLP-encoded blocks on startup."""
    pass


@consume_command(is_hive=True)
def engine() -> None:
    """Client consumes via the Engine API."""
    pass


@consume_command(is_hive=True)
def enginex() -> None:
    """Consume via Engine API with pre-alloc optimization."""
    pass


@consume_command(is_hive=True)
def sync() -> None:
    """Client consumes via the Engine API with sync testing."""
    pass


@consume.command(
    context_settings={"ignore_unknown_options": True},
)
@common_pytest_options
def cache(pytest_args: List[str], **kwargs: Any) -> None:
    """Consume command to cache test fixtures."""
    del kwargs

    cache_cmd = create_consume_command(
        command_logic_test_paths=[], is_hive=False
    )
    cache_cmd.execute(list(pytest_args))
