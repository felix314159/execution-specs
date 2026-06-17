"""Tests for `consume direct`'s automatic Docker parallelism decision."""

from typing import List, Optional

import pytest

from ..pytest_commands.consume import (
    BESU_MAX_WORKERS,
    DOCKER_SERIAL_TEST_THRESHOLD,
    ConsumeDirectCommand,
    collects_no_tests,
    create_consume_command,
    has_explicit_parallelism,
    runs_besu_only,
    selected_docker_clients,
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


def test_selected_docker_clients() -> None:
    """Both flag forms parse to a lower-cased list; absence yields None."""
    assert selected_docker_clients(["--docker.client=besu"]) == ["besu"]
    assert selected_docker_clients(["--docker.client", "Besu"]) == ["besu"]
    assert selected_docker_clients(
        ["--docker.client=besu,go-ethereum"]
    ) == ["besu", "go-ethereum"]
    assert selected_docker_clients(["--docker.client-branches=x"]) is None


def test_runs_besu_only() -> None:
    """Only a Docker run selecting exactly Besu counts as Besu-only."""
    assert runs_besu_only(
        ["--docker.client-branches=x", "--docker.client=besu"]
    )
    # Mixed selections, other clients, and bare `--bin` are not Besu-only.
    assert not runs_besu_only(
        ["--docker.client-branches=x", "--docker.client=besu,erigon"]
    )
    assert not runs_besu_only(
        ["--docker.client-branches=x", "--docker.client=geth"]
    )
    # No `--docker.client` means every client runs, so not Besu-only.
    assert not runs_besu_only(["--docker.client-branches=x"])
    assert not runs_besu_only(["--bin=/evm", "--docker.client=besu"])


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


def test_besu_only_caps_auto_parallelism(
    command: ConsumeDirectCommand, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A large Besu-only run is capped to `-n BESU_MAX_WORKERS`, not auto."""
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    _patch_count(command, monkeypatch, 51037)
    args = command._with_parallelism(
        ["--docker.client-branches=x", "--docker.client=besu"],
        besu_only=True,
    )
    assert args[-2:] == ["-n", str(BESU_MAX_WORKERS)]


def test_besu_cap_never_exceeds_core_count(
    command: ConsumeDirectCommand, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On fewer cores than the cap, the cap follows the core count."""
    monkeypatch.setattr("os.cpu_count", lambda: 2)
    _patch_count(command, monkeypatch, 51037)
    args = command._with_parallelism(
        ["--docker.client-branches=x", "--docker.client=besu"],
        besu_only=True,
    )
    assert args[-2:] == ["-n", "2"]


def test_besu_only_small_run_still_serial(
    command: ConsumeDirectCommand, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Below the serial threshold, Besu-only runs still go serial."""
    _patch_count(command, monkeypatch, DOCKER_SERIAL_TEST_THRESHOLD - 1)
    args = command._with_parallelism(
        ["--docker.client-branches=x", "--docker.client=besu"],
        besu_only=True,
    )
    assert args[-2:] == ["-n", "0"]


@pytest.mark.parametrize(
    "given,expected",
    [
        (["-n", "16"], ["-n", str(BESU_MAX_WORKERS)]),
        (["-n=16"], [f"-n={BESU_MAX_WORKERS}"]),
        (
            ["--numprocesses", "auto"],
            ["--numprocesses", str(BESU_MAX_WORKERS)],
        ),
        (["-n", "2"], ["-n", "2"]),  # already under the cap: untouched
        (["-n", "0"], ["-n", "0"]),  # explicit serial: untouched
    ],
)
def test_cap_explicit_besu_parallelism(
    command: ConsumeDirectCommand,
    monkeypatch: pytest.MonkeyPatch,
    given: List[str],
    expected: List[str],
) -> None:
    """An explicit `-n` for a Besu-only run is clamped to the cap."""
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    base = ["--docker.client-branches=x", "--docker.client=besu"]
    out = command._cap_besu_parallelism(base + given)
    assert out == base + expected


def test_create_executions_enforces_besu_cap_on_explicit_n(
    command: ConsumeDirectCommand, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`create_executions` clamps an over-cap explicit `-n` for Besu."""
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    monkeypatch.setattr(command, "process_arguments", lambda args: list(args))
    (execution,) = command.create_executions(
        ["--docker.client-branches=x", "--docker.client=besu", "-n", "16"]
    )
    assert execution.args[-2:] == ["-n", str(BESU_MAX_WORKERS)]


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
