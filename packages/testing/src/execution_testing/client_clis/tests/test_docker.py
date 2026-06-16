"""Tests for the consume-direct Docker bridge (no Docker required)."""

from pathlib import Path
from typing import List, Type

import pytest

from execution_testing.client_clis import (
    UnknownCLIError,
    docker_client_for_image,
    fixture_consumers_from_docker_image,
)
from execution_testing.client_clis.clis.besu import BesuFixtureConsumer
from execution_testing.client_clis.clis.erigon import ErigonFixtureConsumer
from execution_testing.client_clis.clis.evmone import (
    EvmOneBlockchainFixtureConsumer,
    EvmOneStateFixtureConsumer,
)
from execution_testing.client_clis.clis.geth import GethFixtureConsumer
from execution_testing.client_clis.clis.nethermind import (
    NethtestFixtureConsumer,
)
from execution_testing.client_clis.clis.nimbus import NimbusFixtureConsumer
from execution_testing.client_clis.clis.reth import RevmeFixtureConsumer
from execution_testing.client_clis.fixture_consumer_tool import (
    FixtureConsumerTool,
)


@pytest.mark.parametrize(
    "image,expected_classes",
    [
        ("steel/go-ethereum:master", [GethFixtureConsumer]),
        ("steel/erigon:main", [ErigonFixtureConsumer]),
        ("steel/nethermind:master", [NethtestFixtureConsumer]),
        ("steel/besu:main", [BesuFixtureConsumer]),
        ("steel/nimbus-el:master", [NimbusFixtureConsumer]),
        ("steel/reth:main", [RevmeFixtureConsumer]),
        (
            "steel/evmone:master",
            [EvmOneStateFixtureConsumer, EvmOneBlockchainFixtureConsumer],
        ),
        # Matching is case-insensitive and registry-agnostic.
        ("registry.example.com/GO-ETHEREUM:latest", [GethFixtureConsumer]),
    ],
)
def test_docker_client_for_image(
    image: str, expected_classes: List[Type[FixtureConsumerTool]]
) -> None:
    """The image name maps to the expected fixture consumer class(es)."""
    client = docker_client_for_image(image)
    assert client is not None
    assert [c for c, _ in client.consumers] == expected_classes


def test_docker_client_for_unknown_image() -> None:
    """An unrecognized image yields no client."""
    assert docker_client_for_image("docker.io/library/ubuntu:24.04") is None


def test_fixture_consumers_from_unknown_image_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building from an unrecognized image raises a clear error."""
    monkeypatch.setattr(
        "execution_testing.client_clis.docker.shutil.which",
        lambda _: "/usr/bin/docker",
    )
    with pytest.raises(UnknownCLIError):
        fixture_consumers_from_docker_image("some/unrelated-image:tag")


class _FakeServer:
    """Stand-in for `_CommandServer` that opens no real shell."""

    _next_port = iter(range(40000, 41000))

    def __init__(self, container_id: str) -> None:
        self.container_id = container_id
        self.port = next(self._next_port)
        self.scratch = Path("/tmp/fake-eest-scratch")

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _isolate_session_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Pretend ``docker`` is on PATH and the session backend starts cleanly.

    Records every container-start request and stubs the resident-shell command
    server, so the bridge takes its fast path without touching a real daemon.
    Also clears the per-process server cache so each test starts fresh.
    """
    monkeypatch.setattr(
        "execution_testing.client_clis.docker.shutil.which",
        lambda _: "/usr/bin/docker",
    )
    monkeypatch.setattr("execution_testing.client_clis.docker._servers", {})
    started: List[dict] = []

    def fake_start(image, mounts, *, run_as_host_user):  # type: ignore
        started.append(
            {
                "image": image,
                "mounts": list(mounts),
                "run_as_host_user": run_as_host_user,
            }
        )
        return f"fakecid-{len(started)}"

    monkeypatch.setattr(
        "execution_testing.client_clis.docker._start_container", fake_start
    )
    monkeypatch.setattr(
        "execution_testing.client_clis.docker._CommandServer", _FakeServer
    )
    monkeypatch.setattr(
        "execution_testing.client_clis.docker.atexit.register",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "execution_testing.client_clis.docker._started_requests",
        started,
        raising=False,
    )


def _start_requests() -> List[dict]:
    """Return the recorded container-start requests for the current test."""
    import execution_testing.client_clis.docker as docker_mod

    return docker_mod._started_requests  # type: ignore[attr-defined]


def test_fixture_consumers_from_docker_image_builds_wrapper() -> None:
    """
    The bridge builds one consumer per binary, each wired to an executable
    wrapper that dispatches the expected in-image binary into the session
    container's resident shell.
    """
    image = "steel/evmone:master"
    consumers = fixture_consumers_from_docker_image(image)
    assert [type(c) for c in consumers] == [
        EvmOneStateFixtureConsumer,
        EvmOneBlockchainFixtureConsumer,
    ]
    # One session container is started for the image, shared by both consumers.
    assert len(_start_requests()) == 1
    # evmone writes its report to a host file, so the container runs as the
    # host user.
    assert _start_requests()[0]["run_as_host_user"] is True

    for consumer in consumers:
        wrapper = Path(str(consumer.binary))
        assert wrapper.is_file()
        assert wrapper.stat().st_mode & 0o111, "wrapper must be executable"
        script = wrapper.read_text()
        # Fast path: the wrapper talks to the resident shell over loopback,
        # it does not spawn its own `docker run`/`docker exec`.
        assert "/dev/tcp/127.0.0.1/" in script
        assert "docker run" not in script
        assert getattr(consumer, "docker_image", None) == image

    # The two evmone consumers point at the two distinct in-image binaries.
    scripts = [Path(str(c.binary)).read_text() for c in consumers]
    assert any("evmone-statetest" in s for s in scripts)
    assert any("evmone-blockchaintest" in s for s in scripts)


def test_docker_session_container_not_host_user_by_default() -> None:
    """Clients writing only to stdout do not start a `--user` container."""
    (consumer,) = fixture_consumers_from_docker_image(
        "steel/go-ethereum:master"
    )
    assert _start_requests()[0]["run_as_host_user"] is False
    script = Path(str(consumer.binary)).read_text()
    assert "/dev/tcp/127.0.0.1/" in script
    assert "/gethvm" in script


def test_start_backend_false_starts_no_container() -> None:
    """
    The xdist controller (``start_backend=False``) builds the consumers for
    collection but starts no container, so only the workers do work.
    """
    (consumer,) = fixture_consumers_from_docker_image(
        "steel/go-ethereum:master", start_backend=False
    )
    # No container was started for the controller...
    assert _start_requests() == []
    # ...but the consumer still exists with the same image id, so collection
    # ids match the workers'.
    assert getattr(consumer, "docker_image", None) == "steel/go-ethereum:master"


def test_docker_falls_back_to_run_when_no_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    If a long-lived container cannot be started, each consumer falls back to
    an ephemeral `docker run` per invocation.
    """
    monkeypatch.setattr("execution_testing.client_clis.docker._servers", {})

    def failing_start(image, mounts, *, run_as_host_user):  # type: ignore
        raise RuntimeError("daemon refused detached container")

    monkeypatch.setattr(
        "execution_testing.client_clis.docker._start_container",
        failing_start,
    )
    (consumer,) = fixture_consumers_from_docker_image(
        "steel/go-ethereum:master"
    )
    script = Path(str(consumer.binary)).read_text()
    assert "docker run" in script
    # The offline fixture consumers never need the network on the fallback
    # path either.
    assert "--network=none" in script
    assert "/gethvm" in script
