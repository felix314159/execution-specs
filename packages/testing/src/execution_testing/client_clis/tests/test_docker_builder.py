"""Tests for the docker image builder (no Docker or network required)."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from execution_testing.client_clis.clis.docker import builder
from execution_testing.client_clis.clis.docker.builder import (
    DOCKER_DIR,
    DockerBuildError,
    build_clients,
    load_client_specs,
    planned_client_images,
    sanitize_docker_tag,
)

BUNDLED_CLIENTS_FILE = DOCKER_DIR / "clients.yaml"
LATEST_SHA = "b" * 40
BUILT_SHA = "c" * 40


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("master", "master"),
        ("main", "main"),
        ("feature/foo", "feature_foo"),
        ("v1.2.3", "v1.2.3"),
        (".hidden", "_.hidden"),
        ("-leading", "_-leading"),
    ],
)
def test_sanitize_docker_tag(ref: str, expected: str) -> None:
    """Branch/tag names become valid, stable Docker tags."""
    assert sanitize_docker_tag(ref) == expected


def test_pick_ref_sha_prefers_exact_branch() -> None:
    """An exact branch match wins over an unrelated `foo/<tag>` ref."""
    output = (
        "deadbeef\trefs/heads/kch/master\n"
        "1111111111111111111111111111111111111111\trefs/heads/master\n"
        "2222222222222222222222222222222222222222\trefs/tags/master\n"
    )
    assert builder._pick_ref_sha(output, "master") == "1" * 40


def test_pick_ref_sha_no_match() -> None:
    """A bare near-match (`foo/master`) alone is not accepted."""
    output = "deadbeef\trefs/heads/kch/master\n"
    assert builder._pick_ref_sha(output, "master") is None


def test_load_bundled_client_specs() -> None:
    """The bundled clients.yaml parses into the expected clients."""
    specs = load_client_specs(BUNDLED_CLIENTS_FILE)
    names = {s.client for s in specs}
    assert {"go-ethereum", "besu", "reth", "evmone"} <= names
    geth = next(s for s in specs if s.client == "go-ethereum")
    assert geth.github == "ethereum/go-ethereum"
    assert geth.tag == "master"


@dataclass
class Stub:
    """Captured side effects of a stubbed `build_clients` run."""

    builds: List[List[str]] = field(default_factory=list)
    resolves: List[Tuple[str, str]] = field(default_factory=list)


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> Stub:
    """
    Stub out network/docker calls, recording resolves and issued builds.

    ``_resolve_remote_sha`` returns ``LATEST_SHA`` and records each call (so a
    test can assert upstream was — or was not — consulted); ``_run_build``
    records the build command instead of running Docker.
    """
    captured = Stub()

    def fake_resolve(github: str, tag: str) -> str:
        captured.resolves.append((github, tag))
        return LATEST_SHA

    monkeypatch.setattr(builder, "_resolve_remote_sha", fake_resolve)
    monkeypatch.setattr(builder, "_commit_datetime", lambda *_: None)
    monkeypatch.setattr(
        builder,
        "_run_build",
        lambda args, *_: captured.builds.append(list(args)),
    )
    return captured


def test_no_image_builds_from_scratch(
    monkeypatch: pytest.MonkeyPatch, stub: Stub
) -> None:
    """No prior image → build from scratch with --no-cache."""
    monkeypatch.setattr(builder, "_image_label", lambda *_: None)
    results = build_clients(BUNDLED_CLIENTS_FILE, clients=["go-ethereum"])
    assert results[0].reused is False
    assert results[0].sha == LATEST_SHA
    assert len(stub.builds) == 1
    assert "--no-cache" in stub.builds[0]
    assert "steel/go-ethereum:master" in stub.builds[0]
    assert stub.resolves == [("ethereum/go-ethereum", "master")]


def test_existing_image_reused_without_upstream_check(
    monkeypatch: pytest.MonkeyPatch, stub: Stub
) -> None:
    """A prior image with no --nocache is reused without resolving upstream."""
    monkeypatch.setattr(builder, "_image_label", lambda *_: BUILT_SHA)
    results = build_clients(BUNDLED_CLIENTS_FILE, clients=["go-ethereum"])
    assert results[0].reused is True
    assert results[0].sha == BUILT_SHA
    assert stub.builds == []
    # The fast path must not consult upstream at all.
    assert stub.resolves == []


def test_nocache_same_commit_skips_rebuild(
    monkeypatch: pytest.MonkeyPatch, stub: Stub
) -> None:
    """--nocache + already at latest commit → no rebuild (pointless)."""
    monkeypatch.setattr(builder, "_image_label", lambda *_: LATEST_SHA)
    results = build_clients(
        BUNDLED_CLIENTS_FILE, clients=["go-ethereum"], force=True
    )
    assert results[0].reused is True
    assert results[0].sha == LATEST_SHA
    assert stub.builds == []
    # It did check upstream to confirm the commit is unchanged.
    assert stub.resolves == [("ethereum/go-ethereum", "master")]


def test_nocache_new_commit_rebuilds(
    monkeypatch: pytest.MonkeyPatch, stub: Stub
) -> None:
    """--nocache + upstream moved → rebuild from scratch with --no-cache."""
    monkeypatch.setattr(builder, "_image_label", lambda *_: BUILT_SHA)
    results = build_clients(
        BUNDLED_CLIENTS_FILE, clients=["go-ethereum"], force=True
    )
    assert results[0].reused is False
    assert results[0].sha == LATEST_SHA
    assert len(stub.builds) == 1
    assert "--no-cache" in stub.builds[0]


def test_planned_images_build_nothing(
    monkeypatch: pytest.MonkeyPatch, stub: Stub
) -> None:
    """
    `planned_client_images` resolves the image name only.

    The collect-only pre-count relies on this: it must never issue a build or
    consult upstream, even when an image is missing (the case that would
    otherwise force a from-scratch build).
    """
    monkeypatch.setattr(builder, "_image_label", lambda *_: None)
    results = planned_client_images(
        BUNDLED_CLIENTS_FILE, clients=["go-ethereum"]
    )
    assert [r.image for r in results] == ["steel/go-ethereum:master"]
    assert results[0].sha == ""
    assert results[0].commit_datetime is None
    # The whole point: no build issued and no upstream lookup, ever.
    assert stub.builds == []
    assert stub.resolves == []


def test_planned_images_subset_and_order() -> None:
    """A subset is honored and the config's order is preserved."""
    results = planned_client_images(
        BUNDLED_CLIENTS_FILE, clients=["besu", "go-ethereum"]
    )
    names = [r.client for r in results]
    assert set(names) == {"besu", "go-ethereum"}
    specs = [s.client for s in load_client_specs(BUNDLED_CLIENTS_FILE)]
    assert names == [n for n in specs if n in {"besu", "go-ethereum"}]


def test_planned_images_unknown_client_raises() -> None:
    """An unknown client name is rejected, as in `build_clients`."""
    with pytest.raises(DockerBuildError, match="not found"):
        planned_client_images(BUNDLED_CLIENTS_FILE, clients=["nope"])


COMMIT_DT = datetime(2026, 6, 11, 21, 11, tzinfo=timezone.utc)


def test_reuse_reads_commit_date_from_label(
    monkeypatch: pytest.MonkeyPatch, stub: Stub
) -> None:
    """Reuse reports the stamped commit date without any network lookup."""

    def fake_label(_image: str, label: str) -> Optional[str]:
        if label == builder.SOURCE_COMMIT_DATETIME_LABEL:
            return COMMIT_DT.isoformat()
        return BUILT_SHA

    calls: List[Tuple[str, str]] = []

    def record(github: str, sha: str) -> None:
        calls.append((github, sha))

    monkeypatch.setattr(builder, "_image_label", fake_label)
    monkeypatch.setattr(builder, "_commit_datetime", record)

    results = build_clients(BUNDLED_CLIENTS_FILE, clients=["go-ethereum"])
    assert results[0].commit_datetime == COMMIT_DT
    # The label spared us the (network-fragile) GitHub lookup entirely.
    assert calls == []
    assert stub.builds == []


def test_reuse_falls_back_to_network_when_label_missing(
    monkeypatch: pytest.MonkeyPatch, stub: Stub
) -> None:
    """Legacy images without the label still get a best-effort lookup."""

    def fake_label(_image: str, label: str) -> Optional[str]:
        if label == builder.SOURCE_COMMIT_DATETIME_LABEL:
            return None
        return BUILT_SHA

    monkeypatch.setattr(builder, "_image_label", fake_label)
    monkeypatch.setattr(builder, "_commit_datetime", lambda *_: COMMIT_DT)

    results = build_clients(BUNDLED_CLIENTS_FILE, clients=["go-ethereum"])
    assert results[0].commit_datetime == COMMIT_DT
    assert stub.builds == []


def test_build_stamps_commit_date_label(
    monkeypatch: pytest.MonkeyPatch, stub: Stub
) -> None:
    """A from-scratch build stamps the resolved commit date as a label."""
    monkeypatch.setattr(builder, "_image_label", lambda *_: None)
    monkeypatch.setattr(builder, "_commit_datetime", lambda *_: COMMIT_DT)

    build_clients(BUNDLED_CLIENTS_FILE, clients=["go-ethereum"])
    label = f"{builder.SOURCE_COMMIT_DATETIME_LABEL}={COMMIT_DT.isoformat()}"
    assert label in stub.builds[0]


def test_unknown_client_in_filter_raises() -> None:
    """Requesting a client not in the config is an error."""
    with pytest.raises(DockerBuildError):
        build_clients(BUNDLED_CLIENTS_FILE, clients=["not-a-client"])


def test_missing_clients_file_raises() -> None:
    """A missing config path is reported clearly."""
    with pytest.raises(DockerBuildError):
        load_client_specs(Path("/nonexistent/clients.yaml"))
