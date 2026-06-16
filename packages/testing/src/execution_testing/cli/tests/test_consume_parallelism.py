"""Tests for `consume direct`'s automatic Docker parallelism decision."""

from typing import List, Optional

import pytest

from ..pytest_commands.consume import (
    DOCKER_SERIAL_TEST_THRESHOLD,
    ConsumeDirectCommand,
    collects_no_tests,
    create_consume_command,
    has_explicit_parallelism,
    uses_docker_backend,
)


def test_uses_docker_backend() -> None:
    """Both flag forms are detected; the local `--bin` path is not."""
    assert uses_docker_backend(["--docker.client-branches=clients.yaml"])
    assert uses_docker_backend(["--docker.client-branches", "clients.yaml"])
    assert not uses_docker_backend(["--bin=/path/to/evm"])
    assert not uses_docker_backend([])


def test_has_explicit_parallelism() -> None:
    """All `-n`/`--numprocesses` spellings count as user-chosen."""
    for flag in (["-n", "4"], ["-n=4"], ["--numprocesses", "8"], ["-n", "0"]):
        assert has_explicit_parallelism(flag)
    assert not has_explicit_parallelism(["--docker.client-branches=x"])


def test_collects_no_tests() -> None:
    """Collection-only and build-only runs execute no tests."""
    assert collects_no_tests(["--collect-only"])
    assert collects_no_tests(["--co"])
    assert collects_no_tests(["--docker.build-only"])
    assert not collects_no_tests(["--docker.client-branches=x"])


@pytest.fixture
def command() -> ConsumeDirectCommand:
    """A `consume direct` command instance (the auto-parallel variant)."""
    cmd = create_consume_command(
        command_logic_test_paths=[], command_name="direct"
    )
    assert isinstance(cmd, ConsumeDirectCommand)
    return cmd


def _patch_count(
    command: ConsumeDirectCommand,
    monkeypatch: pytest.MonkeyPatch,
    count: Optional[int],
) -> None:
    """Stub the collection pre-count so no real pytest collection runs."""
    monkeypatch.setattr(
        command.runner,
        "count_selected_tests",
        lambda execution: count,
    )


@pytest.mark.parametrize(
    "count,expected_n",
    [
        (DOCKER_SERIAL_TEST_THRESHOLD, "0"),
        (DOCKER_SERIAL_TEST_THRESHOLD - 1, "0"),
        (DOCKER_SERIAL_TEST_THRESHOLD + 1, "auto"),
        (51037, "auto"),
    ],
)
def test_decision_by_count(
    command: ConsumeDirectCommand,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    expected_n: str,
) -> None:
    """Small Docker runs go serial; large ones go `-n auto`."""
    _patch_count(command, monkeypatch, count)
    args = command._with_parallelism(["--docker.client-branches=x"])
    assert args[-2:] == ["-n", expected_n]


def test_unknown_count_falls_back_to_serial(
    command: ConsumeDirectCommand, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the pre-count fails, the run stays serial rather than guessing."""
    _patch_count(command, monkeypatch, None)
    args = command._with_parallelism(["--docker.client-branches=x"])
    assert args[-2:] == ["-n", "0"]


def test_routing_only_counts_for_runnable_docker(
    command: ConsumeDirectCommand, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Only a runnable Docker invocation (Docker backend, no explicit `-n`, not
    collect-only) triggers the count-and-decide path; everything else is
    passed through untouched.
    """
    # Bypass the click-context-dependent processor pipeline.
    monkeypatch.setattr(command, "process_arguments", lambda args: list(args))
    counted: List[List[str]] = []

    def _record(execution: object) -> Optional[int]:
        counted.append(list(getattr(execution, "args")))
        return 10  # small -> would choose serial if reached

    monkeypatch.setattr(command.runner, "count_selected_tests", _record)

    def n_args(argv: List[str]) -> List[str]:
        (execution,) = command.create_executions(argv)
        return execution.args

    # Runnable Docker run -> counted, decision applied.
    assert n_args(["--docker.client-branches=x"])[-2:] == ["-n", "0"]
    assert len(counted) == 1
    # The local --bin path, an explicit -n, and collect-only are not counted.
    counted.clear()
    assert "-n" not in n_args(["--bin=/evm"])
    assert n_args(["--docker.client-branches=x", "-n", "4"]).count("-n") == 1
    n_args(["--docker.client-branches=x", "--collect-only"])
    assert counted == []
