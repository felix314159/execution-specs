"""Tests for the default-parallelism processor of `consume direct`."""

from typing import List

import pytest

from ..pytest_commands.processors import DockerParallelismProcessor


@pytest.fixture
def processor() -> DockerParallelismProcessor:
    """Provide the processor under test."""
    return DockerParallelismProcessor()


def test_docker_defaults_to_n_auto(
    processor: DockerParallelismProcessor,
) -> None:
    """A Docker run with no worker count gets `-n auto` appended."""
    args = ["--docker.client-branches=clients.yaml", "--docker.client=geth"]
    assert processor.process_args(list(args)) == args + ["-n", "auto"]


def test_docker_with_space_separated_value(
    processor: DockerParallelismProcessor,
) -> None:
    """The space-separated form of the flag is recognized too."""
    args = ["--docker.client-branches", "clients.yaml"]
    assert processor.process_args(list(args))[-2:] == ["-n", "auto"]


def test_bin_path_is_left_untouched(
    processor: DockerParallelismProcessor,
) -> None:
    """The local `--bin` path keeps its (serial) behavior unchanged."""
    args = ["--bin=/path/to/evm"]
    assert processor.process_args(list(args)) == args


@pytest.mark.parametrize(
    "parallel_flag",
    [["-n", "4"], ["-n=4"], ["--numprocesses", "8"], ["-n", "0"]],
)
def test_explicit_worker_count_wins(
    processor: DockerParallelismProcessor, parallel_flag: List[str]
) -> None:
    """An explicit `-n`/`--numprocesses` is never overridden (incl. `-n 0`)."""
    args = ["--docker.client-branches=x", *parallel_flag]
    assert processor.process_args(list(args)) == args


@pytest.mark.parametrize(
    "skip_flag", ["--collect-only", "--co", "--docker.build-only"]
)
def test_no_workers_when_nothing_runs(
    processor: DockerParallelismProcessor, skip_flag: str
) -> None:
    """Runs that collect nothing do not pay for worker startup."""
    args = ["--docker.client-branches=x", skip_flag]
    assert processor.process_args(list(args)) == args
