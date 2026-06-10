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


def test_fixture_consumers_from_docker_image_builds_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The bridge builds one consumer per binary, each wired to an executable
    wrapper that runs the expected in-image binary via `docker run`.
    """
    monkeypatch.setattr(
        "execution_testing.client_clis.docker.shutil.which",
        lambda _: "/usr/bin/docker",
    )
    image = "steel/evmone:master"
    consumers = fixture_consumers_from_docker_image(image)
    assert [type(c) for c in consumers] == [
        EvmOneStateFixtureConsumer,
        EvmOneBlockchainFixtureConsumer,
    ]
    for consumer in consumers:
        wrapper = Path(str(consumer.binary))
        assert wrapper.is_file()
        assert wrapper.stat().st_mode & 0o111, "wrapper must be executable"
        script = wrapper.read_text()
        assert "docker run" in script
        # Fixture consumers are offline; skipping bridge setup is a large
        # chunk of each container's startup cost.
        assert "--network=none" in script
        assert image in script
        # evmone writes its report to a host file, so it runs as host user.
        assert '--user "$(id -u):$(id -g)"' in script
        assert getattr(consumer, "docker_image", None) == image

    # The two evmone consumers point at the two distinct in-image binaries.
    scripts = [Path(str(c.binary)).read_text() for c in consumers]
    assert any("evmone-statetest" in s for s in scripts)
    assert any("evmone-blockchaintest" in s for s in scripts)


def test_docker_wrapper_no_user_flag_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clients writing only to stdout do not get a `--user` flag."""
    monkeypatch.setattr(
        "execution_testing.client_clis.docker.shutil.which",
        lambda _: "/usr/bin/docker",
    )
    (consumer,) = fixture_consumers_from_docker_image(
        "steel/go-ethereum:master"
    )
    script = Path(str(consumer.binary)).read_text()
    assert "--user" not in script
    assert "/gethvm" in script
