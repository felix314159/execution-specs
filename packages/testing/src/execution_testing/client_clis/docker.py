"""
Docker bridge for ``consume direct`` fixture consumers.

Instead of a local ``--bin`` path, a client can be selected by its Docker image
name (e.g. ``steel/go-ethereum:master``). The client tool is then executed
inside a container created from that image.

The image name itself identifies the client: the published images follow a
standard ``<registry>/<client>:<tag>`` naming convention, and each client image
ships its fixture-consumer binary at a well-known path. No in-container probing
or version detection is therefore required — the image name is mapped directly
to the matching :class:`FixtureConsumerTool` subclass.

Two execution strategies are supported, both driving the *same*
:class:`FixtureConsumerTool` subclasses via an auto-generated wrapper script
the consumer runs as if it were a local binary:

* **long-lived container + persistent shell (the fast path).** One detached
  container per image is started once for the whole session (``sleep
  infinity`` as its only process), with the fixture and temp directories
  bind-mounted up front. A single ``docker exec -i <container> bash`` shell is
  kept open inside it, and every fixture-file invocation is fed to that shell
  as a command rather than being launched with its own ``docker exec``. This
  matters because a per-call ``docker exec`` still pays the full docker
  CLI/daemon round-trip (~0.05s) — at thousands of fixture files that round-trip
  dominates, dwarfing the EVM work. Reusing one resident shell drops the
  per-call overhead by ~250x (to ~0.0002s), since it pays only the in-container
  process launch. A tiny per-invocation client (the wrapper) hands its argv to
  the shell over a loopback socket served by :class:`_CommandServer`, the
  command runs with stdout/stderr redirected to temp files on the shared temp
  mount, and the client relays them back with the command's exit status. The
  shell and container are torn down at interpreter exit.
* **ephemeral ``docker run`` per invocation (the fallback).** Used when a
  long-lived container cannot be started (e.g. the daemon refuses the detached
  ``sleep`` container). Each invocation creates and tears down its own
  container, resolving and bind-mounting only the paths it is handed.

Because ``docker exec`` cannot add bind mounts after a container is created,
the long-lived path mounts the whole fixtures directory (plus the temp
directory, and the dump directory when ``--dump-dir`` is set) at session start,
at their identical host paths, so the absolute paths the consumers pass resolve
inside the container unchanged.

This module does not build or pull images: it assumes the referenced image is
already available to the local Docker daemon. Images are made available
beforehand by :mod:`.clis.docker.builder` (driven by the consume
``--docker.client-branches`` option).
"""

import atexit
import itertools
import os
import re
import shlex
import shutil
import socketserver
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
)

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

# Label stamped on every session container so leaked ones (after a hard kill
# that skips the atexit cleanup) can be found and removed:
# `docker rm -f $(docker ps -aq --filter label=eest.consume-direct.container)`.
CONTAINER_LABEL = "eest.consume-direct.container"

# Persistent-shell client wrapper (the fast path). A fresh wrapper process runs
# per invocation (the consumers shell out to `self.binary`), but instead of its
# own `docker exec` it hands the command to the always-open shell in the
# session container via :class:`_CommandServer`'s loopback socket. To keep this
# hot path cheap — it runs once per fixture file, thousands of times — the
# wrapper forks as few helpers as possible: the server (host-side Python) does
# the path resolution, names and cleans up the stdout/stderr files, so the
# wrapper only sends `binary, args...`, reads back `exit_status, out, err`, and
# relays the captured output (a `cat` of stdout, plus stderr only when
# non-empty). The container runs as the right `--user`, so no per-call user
# flag is needed.
_WRAPPER_EXEC_TEMPLATE = """\
#!/usr/bin/env bash
# Auto-generated by the EEST consume-direct Docker bridge.
# Dispatches `{binary}` into the session container's persistent shell.
set -uo pipefail
port={port}
request={binary_quoted}
for arg in "$@"; do
  request+=$'\\t'"$arg"
done
exec 3<>"/dev/tcp/127.0.0.1/$port"
printf '%s\\n' "$request" >&3
IFS=$'\\t' read -r code out err <&3
exec 3>&- 3<&-
# Relay stdout/stderr with bash builtins only — `$(<file)` reads the file
# without forking a `cat`, which matters on this per-fixture-file hot path.
# The clients parse their output as JSON, so the trailing-newline trim that
# command substitution applies is immaterial.
[[ -s "$err" ]] && printf '%s' "$(<"$err")" >&2
[[ -s "$out" ]] && printf '%s' "$(<"$out")"
exit "${{code:-1}}"
"""

# `docker run` wrapper (the fallback, used when no session container could be
# started). Each path argument is resolved to an absolute path and its
# directory is bind-mounted at the same location, so the containerized tool
# reads host fixture paths unchanged. The temp directory is always mounted too:
# some consumers hand the tool a temp output file (e.g. evmone's gtest JSON
# report) or a temp fixture, which the host side then reads back. The client
# binary is run as the container entrypoint.
#
# `--network=none` is passed because the fixture consumers are wholly offline
# (they read a JSON fixture, execute it in the EVM, and report the result):
# none of the supported clients need the network. Skipping the default bridge
# network setup — creating a veth pair, attaching it to the bridge, and tearing
# it all down again — removes roughly 40% of each container's startup cost,
# which dominates the per-invocation time since a fresh container is created
# for every fixture file.
_WRAPPER_RUN_TEMPLATE = """\
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

# image -> the command server (container + resident shell) serving its calls.
_servers: Dict[str, "_CommandServer"] = {}


def _wrapper_directory() -> Path:
    """Return the session directory (created lazily) holding wrappers."""
    global _wrapper_dir
    if _wrapper_dir is None:
        _wrapper_dir = Path(
            tempfile.mkdtemp(prefix="eest-consume-direct-docker-")
        )
        atexit.register(shutil.rmtree, _wrapper_dir, ignore_errors=True)
    return _wrapper_dir


def _write_wrapper(name_seed: str, script: str) -> Path:
    """Write ``script`` to an executable wrapper file and return its path."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name_seed)
    wrapper_path = _wrapper_directory() / f"{safe_name}.sh"
    wrapper_path.write_text(script)
    wrapper_path.chmod(0o755)
    return wrapper_path


def _mount_dirs(
    fixtures_root: Optional[Path], dump_dir: Optional[Path]
) -> List[Path]:
    """
    Resolve the directories to bind-mount into the session container.

    The temp directory is always mounted (temp fixtures from stdin and tools'
    temp output files live there); the fixtures root and, when set, the dump
    directory are mounted so every fixture file and debug-output path resolves
    inside the container. Only existing directories are returned, de-duplicated
    while preserving order.
    """
    candidates = [Path(tempfile.gettempdir()), fixtures_root, dump_dir]
    seen = set()
    resolved: List[Path] = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = candidate.resolve()
        if path in seen or not path.is_dir():
            continue
        seen.add(path)
        resolved.append(path)
    return resolved


def _start_container(
    image: str, mounts: Sequence[Path], *, run_as_host_user: bool
) -> str:
    """
    Start a detached, long-lived container for ``image`` and return its id.

    The container runs only ``sleep infinity`` so it stays alive to serve
    ``docker exec``s; ``mounts`` are bind-mounted at their host paths, and
    ``--network=none`` skips the (unused) bridge setup. Raises on failure so
    the caller can fall back to the per-invocation ``docker run`` path.
    """
    name = "eest-consume-direct-" + re.sub(
        r"[^A-Za-z0-9_.-]", "_", f"{image}-{os.getpid()}"
    )
    args = [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--network=none",
        "--init",
        "--name",
        name,
        "--label",
        f"{CONTAINER_LABEL}=1",
    ]
    if run_as_host_user:
        args += ["--user", f"{os.getuid()}:{os.getgid()}"]
    for src in mounts:
        args += ["--volume", f"{src}:{src}"]
    args += ["--entrypoint", "sleep", image, "infinity"]
    result = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "docker run failed")
    return result.stdout.strip()


def _stop_container(container_id: str) -> None:
    """Force-remove a session container, ignoring any error."""
    subprocess.run(
        ["docker", "rm", "--force", container_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class _CommandServer:
    """
    A resident shell inside a session container, reachable over a socket.

    Holds one ``docker exec -i <container> bash`` open for the whole session
    and serves commands to the per-invocation wrapper clients on a loopback
    TCP port. Each request is a tab-separated ``binary, args...`` line; the
    server resolves path arguments to absolute host paths, names this call's
    stdout/stderr files on the shared scratch mount, runs the command in the
    shell with its output redirected there, and replies ``exit_status, out,
    err`` so the (deliberately fork-light) wrapper need only ``cat`` the
    output. A lock serializes the shell, which has a single input stream;
    ``consume`` runs its cases sequentially, so this is not a bottleneck.
    """

    def __init__(self, container_id: str) -> None:
        """Open the resident shell and start the loopback dispatch server."""
        self.container_id = container_id
        self._lock = threading.Lock()
        self._counter = itertools.count()
        # stdout/stderr files of the previously served call, removed when the
        # next one arrives: calls are serial (the prior wrapper has already
        # read and exited), so this bounds scratch to the live call's files
        # without the wrapper paying for an `rm`.
        self._previous: Optional[Tuple[str, str]] = None
        # World-writable, non-sticky scratch dir on the shared temp mount: the
        # container creates each command's stdout/stderr files here (owned by
        # the container user), and the host — owning this directory — can read
        # them back and unlink them regardless of who owns the files.
        self.scratch = Path(
            tempfile.mkdtemp(prefix="eest-consume-direct-io-")
        )
        self.scratch.chmod(0o777)
        self._shell = subprocess.Popen(
            ["docker", "exec", "-i", container_id, "bash"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._server = socketserver.ThreadingTCPServer(
            ("127.0.0.1", 0), self._handler_class()
        )
        self._server.daemon_threads = True
        self.port: int = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    def _run(self, tokens: Sequence[str]) -> Tuple[int, str, str]:
        """
        Run one command in the shell, returning ``(status, out, err)``.

        ``tokens`` is ``[binary, *args]``; arguments that name an existing
        host path are resolved to their absolute form (the container's working
        directory differs from the host's). The stdout/stderr files for this
        call are named here and their paths returned for the wrapper to read.
        """
        binary, *args = tokens
        resolved = [binary] + [
            os.path.realpath(arg) if os.path.exists(arg) else arg
            for arg in args
        ]
        index = next(self._counter)
        out_file = str(self.scratch / f"{index}.out")
        err_file = str(self.scratch / f"{index}.err")
        marker = f"__EEST_DONE_{index}__"
        command = " ".join(shlex.quote(token) for token in resolved)
        line = (
            f"{command} > {shlex.quote(out_file)} 2> {shlex.quote(err_file)}; "
            f"printf '{marker}:%s\\n' \"$?\"\n"
        )
        prefix = f"{marker}:".encode()
        with self._lock:
            assert self._shell.stdin is not None
            assert self._shell.stdout is not None
            # The previous call's wrapper has read and exited (calls are
            # serial), so its scratch files are safe to drop now.
            if self._previous is not None:
                for path in self._previous:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            self._previous = (out_file, err_file)
            self._shell.stdin.write(line.encode())
            self._shell.stdin.flush()
            while True:
                output = self._shell.stdout.readline()
                if not output:
                    raise RuntimeError("session shell closed unexpectedly")
                if output.startswith(prefix):
                    status = int(output[len(prefix):].strip() or 1)
                    return status, out_file, err_file

    def _handler_class(self) -> Type[socketserver.StreamRequestHandler]:
        server = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                raw = self.rfile.readline()
                if not raw:
                    return
                tokens = raw.decode().rstrip("\n").split("\t")
                try:
                    status, out_file, err_file = server._run(tokens)
                except Exception as exc:
                    logger.debug(
                        f"Docker bridge: command dispatch failed: {exc}"
                    )
                    status, out_file, err_file = 1, "", ""
                self.wfile.write(
                    f"{status}\t{out_file}\t{err_file}\n".encode()
                )

        return Handler

    def close(self) -> None:
        """Tear down the dispatch server, the resident shell, and scratch."""
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass
        try:
            if self._shell.stdin is not None:
                self._shell.stdin.close()
        except Exception:
            pass
        try:
            self._shell.terminate()
        except Exception:
            pass
        shutil.rmtree(self.scratch, ignore_errors=True)


def _shutdown_server(server: "_CommandServer") -> None:
    """Close a command server and force-remove its container."""
    server.close()
    _stop_container(server.container_id)


def _session_server(
    image: str, mounts: Sequence[Path], *, run_as_host_user: bool
) -> Optional["_CommandServer"]:
    """
    Return the command server for ``image``, starting it on first use.

    Starts the session container and opens its resident shell, caching the
    result per image and registering an :mod:`atexit` hook to tear both down.
    Returns None (and logs a warning) if either step fails, signalling the
    caller to fall back to a ``docker run`` per invocation.
    """
    if image in _servers:
        return _servers[image]
    try:
        container_id = _start_container(
            image, mounts, run_as_host_user=run_as_host_user
        )
    except Exception as exc:
        logger.warning(
            f"Docker bridge: could not start a long-lived container for "
            f"{image} ({exc}); falling back to per-invocation `docker run`."
        )
        return None
    try:
        server = _CommandServer(container_id)
    except Exception as exc:
        _stop_container(container_id)
        logger.warning(
            f"Docker bridge: could not open a resident shell in {image} "
            f"({exc}); falling back to per-invocation `docker run`."
        )
        return None
    _servers[image] = server
    atexit.register(_shutdown_server, server)
    logger.debug(
        f"Docker bridge: started session container {container_id[:12]} for "
        f"{image} with a resident shell on port {server.port} "
        f"(mounts: {', '.join(str(m) for m in mounts)})"
    )
    return server


def _build_exec_wrapper(
    server: "_CommandServer", binary: str, image: str
) -> Path:
    """Create a wrapper dispatching ``binary`` to ``server``'s shell."""
    script = _WRAPPER_EXEC_TEMPLATE.format(
        binary=binary,
        port=server.port,
        binary_quoted=shlex.quote(binary),
    )
    return _write_wrapper(f"{image}-{binary}-exec", script)


def _build_run_wrapper(
    image: str, binary: str, run_as_host_user: bool = False
) -> Path:
    """Create a wrapper that runs ``binary`` via an ephemeral ``docker run``."""
    user_flag = '--user "$(id -u):$(id -g)" ' if run_as_host_user else ""
    script = _WRAPPER_RUN_TEMPLATE.format(
        image=image,
        binary=binary,
        image_quoted=shlex.quote(image),
        binary_quoted=shlex.quote(binary),
        user_flag=user_flag,
    )
    return _write_wrapper(f"{image}-{binary}-run", script)


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
    fixtures_root: Optional[Path] = None,
    dump_dir: Optional[Path] = None,
    trace: bool = False,
) -> List[FixtureConsumerTool]:
    """
    Build the fixture consumer(s) backed by a client's Docker image.

    The image name is mapped to a known client; one consumer is returned per
    binary that client exposes (usually one, two for evmone). A single
    long-lived container is started for the image — bind-mounting
    ``fixtures_root`` (and ``dump_dir`` when set) alongside the temp directory
    — with a resident shell each consumer dispatches its binary into. If that
    container or shell cannot be started, each consumer falls back to an
    ephemeral ``docker run`` per invocation.
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

    mounts = _mount_dirs(fixtures_root, dump_dir)
    server = _session_server(
        image, mounts, run_as_host_user=client.run_as_host_user
    )

    # Every concrete consumer accepts `trace`, but the abstract base's
    # signature does not declare it; pass it via kwargs as `from_binary_path`
    # does so the call type-checks.
    kwargs = {"trace": trace}
    consumers: List[FixtureConsumerTool] = []
    for consumer_class, binary in client.consumers:
        if server is not None:
            wrapper = _build_exec_wrapper(server, binary, image)
            how = f"resident shell (port {server.port})"
        else:
            wrapper = _build_run_wrapper(
                image, binary, client.run_as_host_user
            )
            how = "docker run"
        logger.debug(
            f"Docker bridge: {image} -> {consumer_class.__name__} "
            f"(binary {binary} via {how}, wrapper {wrapper})"
        )
        consumer = consumer_class(binary=wrapper, **kwargs)
        # Tag the instance so the test id reflects the image, not just the
        # consumer class (which is shared with the local `--bin` path).
        consumer.docker_image = image  # type: ignore[attr-defined]
        consumers.append(consumer)
    return consumers
