"""
Docker bridge for ``consume direct`` fixture consumers.

Instead of a local ``--bin`` path, a client can be selected by its Docker image
name (e.g. ``steel/go-ethereum:master``). The client tool is then executed
inside an ephemeral container created from that image.

The image name itself identifies the client: the published images follow a
standard ``<registry>/<client>:<tag>`` naming convention, and each client image
ships its fixture-consumer binary at a well-known path. No in-container probing
or version detection is therefore required — the image name is mapped directly
to the matching :class:`FixtureConsumerTool` subclass.

This module does not build, pull, or otherwise manage container lifecycles. It
assumes the referenced image is already available to the local Docker daemon
and simply shells out to ``docker run`` per invocation, bind-mounting the
fixture paths so the containerized tool reads them unchanged. Images are made
available beforehand by :mod:`.clis.docker.builder` (driven by the consume
``--docker.client-branches`` option).
"""

import atexit
import re
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Type

from execution_testing.logging import get_logger

from .clis.besu import BesuFixtureConsumer
from .clis.erigon import ErigonFixtureConsumer
from .clis.evmone import (
    EvmOneBlockchainFixtureConsumer,
    EvmOneStateFixtureConsumer,
)
from .clis.geth import GethFixtureConsumer
from .clis.nethermind import NethtestFixtureConsumer
from .clis.nimbus import NimbusFixtureConsumer
from .clis.reth import RevmeFixtureConsumer
from .ethereum_cli import CLINotFoundInPathError, UnknownCLIError
from .fixture_consumer_tool import FixtureConsumerTool

logger = get_logger(__name__)


class DockerClient(NamedTuple):
    """A client selectable by Docker image name."""

    # Substrings that, if present in the image name, identify this client.
    tokens: Tuple[str, ...]
    # The fixture consumer(s) to instantiate, paired with the absolute path of
    # the binary inside the image. A client may expose more than one binary
    # (e.g. evmone's separate state- and blockchain-test runners), in which
    # case one consumer is created per binary, mirroring repeated `--bin`.
    consumers: Tuple[Tuple[Type[FixtureConsumerTool], str], ...]
    # Whether to run the container as the host user (`--user`). Needed when the
    # tool writes an output file into a bind-mounted host directory (e.g.
    # evmone's gtest JSON report). Left False for tools that write only to
    # container-internal, root-owned paths (e.g. nethtest's `/neth/db`).
    run_as_host_user: bool = False


# Image-name token -> fixture consumer(s) and their in-image binary path.
DOCKER_CLIENTS: Tuple[DockerClient, ...] = (
    DockerClient(
        tokens=("go-ethereum", "go_ethereum", "geth"),
        consumers=((GethFixtureConsumer, "/gethvm"),),
    ),
    DockerClient(
        tokens=("erigon",),
        consumers=((ErigonFixtureConsumer, "/erigon_vm"),),
    ),
    DockerClient(
        tokens=("nethermind",),
        consumers=((NethtestFixtureConsumer, "/nethtest"),),
    ),
    DockerClient(
        tokens=("besu",),
        consumers=((BesuFixtureConsumer, "/besu-vm"),),
    ),
    DockerClient(
        tokens=("nimbus",),
        consumers=((NimbusFixtureConsumer, "/eest_blockchain"),),
    ),
    DockerClient(
        tokens=("reth", "revm"),
        consumers=((RevmeFixtureConsumer, "/revme"),),
    ),
    DockerClient(
        tokens=("evmone",),
        consumers=(
            (
                EvmOneStateFixtureConsumer,
                "/evmone/build/bin/evmone-statetest",
            ),
            (
                EvmOneBlockchainFixtureConsumer,
                "/evmone/build/bin/evmone-blockchaintest",
            ),
        ),
        # evmone writes its gtest JSON report to a host temp file.
        run_as_host_user=True,
    ),
)

# Wrapper script run for every binary invocation. Each path argument is
# resolved to an absolute path (the container's working directory differs from
# the host's, so relative paths would not resolve) and its directory is
# bind-mounted at the same location, so the containerized tool reads host
# fixture paths unchanged. The temp directory is always mounted too: some
# consumers hand the tool a temp output file (e.g. evmone's gtest JSON report)
# or a temp fixture, which the host side then reads back. The client binary is
# run as the container entrypoint.
#
# `--network=none` is passed because the fixture consumers are wholly offline
# (they read a JSON fixture, execute it in the EVM, and report the result):
# none of the supported clients need the network. Skipping the default bridge
# network setup — creating a veth pair, attaching it to the bridge, and tearing
# it all down again — removes roughly 40% of each container's startup cost,
# which dominates the per-invocation time since a fresh container is created
# for every fixture file.
_WRAPPER_TEMPLATE = """\
#!/usr/bin/env bash
# Auto-generated by the EEST consume-direct Docker bridge.
# Runs `{binary}` inside an ephemeral container from `{image}`.
set -euo pipefail
image={image_quoted}
binary={binary_quoted}
declare -a forwarded=()
declare -a mounts=()
declare -A seen=()
tmp_dir=$(realpath -- "${{TMPDIR:-/tmp}}")
seen[$tmp_dir]=1
mounts+=("-v" "$tmp_dir:$tmp_dir")
for arg in "$@"; do
  if [[ -e "$arg" ]]; then
    abs=$(realpath -- "$arg")
    if [[ -d "$abs" ]]; then
      mount_dir="$abs"
    else
      mount_dir=$(dirname -- "$abs")
    fi
    if [[ -z "${{seen[$mount_dir]:-}}" ]]; then
      seen[$mount_dir]=1
      mounts+=("-v" "$mount_dir:$mount_dir")
    fi
    forwarded+=("$abs")
  else
    forwarded+=("$arg")
  fi
done
exec docker run --rm --network=none {user_flag}"${{mounts[@]}}" \
  --entrypoint "$binary" "$image" ${{forwarded[@]+"${{forwarded[@]}}"}}
"""

_wrapper_dir: Optional[Path] = None


def _wrapper_directory() -> Path:
    """Return the session directory (created lazily) holding wrappers."""
    global _wrapper_dir
    if _wrapper_dir is None:
        _wrapper_dir = Path(
            tempfile.mkdtemp(prefix="eest-consume-direct-docker-")
        )
        atexit.register(shutil.rmtree, _wrapper_dir, ignore_errors=True)
    return _wrapper_dir


def _build_wrapper(
    image: str, binary: str, run_as_host_user: bool = False
) -> Path:
    """Create an executable wrapper invoking ``binary`` inside ``image``."""
    user_flag = '--user "$(id -u):$(id -g)" ' if run_as_host_user else ""
    script = _WRAPPER_TEMPLATE.format(
        image=image,
        binary=binary,
        image_quoted=shlex.quote(image),
        binary_quoted=shlex.quote(binary),
        user_flag=user_flag,
    )
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{image}-{binary}")
    wrapper_path = _wrapper_directory() / f"{safe_name}.sh"
    wrapper_path.write_text(script)
    wrapper_path.chmod(0o755)
    return wrapper_path


def docker_client_for_image(image: str) -> Optional[DockerClient]:
    """Return the :class:`DockerClient` whose token matches ``image``."""
    name = image.lower()
    for client in DOCKER_CLIENTS:
        if any(token in name for token in client.tokens):
            return client
    return None


def fixture_consumers_from_docker_image(
    image: str,
    *,
    trace: bool = False,
) -> List[FixtureConsumerTool]:
    """
    Build the fixture consumer(s) backed by a client's Docker image.

    The image name is mapped to a known client; one consumer is returned per
    binary that client exposes (usually one, two for evmone). Each consumer is
    wired to run its binary inside an ephemeral container via a wrapper script.
    """
    if shutil.which("docker") is None:
        raise CLINotFoundInPathError(
            message="`docker` was not found in the path", binary=Path("docker")
        )

    client = docker_client_for_image(image)
    if client is None:
        known = ", ".join(client.tokens[0] for client in DOCKER_CLIENTS)
        raise UnknownCLIError(
            f"Could not determine the client for Docker image '{image}'. "
            f"Recognized clients: {known}."
        )

    # Every concrete consumer accepts `trace`, but the abstract base's
    # signature does not declare it; pass it via kwargs as `from_binary_path`
    # does so the call type-checks.
    kwargs = {"trace": trace}
    consumers: List[FixtureConsumerTool] = []
    for consumer_class, binary in client.consumers:
        wrapper = _build_wrapper(image, binary, client.run_as_host_user)
        logger.debug(
            f"Docker bridge: {image} -> {consumer_class.__name__} "
            f"(binary {binary} via {wrapper})"
        )
        consumer = consumer_class(binary=wrapper, **kwargs)
        # Tag the instance so the test id reflects the image, not just the
        # consumer class (which is shared with the local `--bin` path).
        consumer.docker_image = image  # type: ignore[attr-defined]
        consumers.append(consumer)
    return consumers
