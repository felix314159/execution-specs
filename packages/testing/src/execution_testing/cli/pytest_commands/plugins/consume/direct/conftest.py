"""
A pytest plugin that configures the consume command to act as a test runner for
"direct" client fixture consumer interfaces.

For example, via go-ethereum's `evm blocktest` or `evm statetest` commands.
"""

import json
import tempfile
import warnings
from pathlib import Path
from typing import Any, Generator, List, Optional

import pytest

from execution_testing.base_types import to_json
from execution_testing.cli.pytest_commands.plugins.consume.consume import (
    FixturesSource,
)
from execution_testing.client_clis.clis.docker import (
    BuildResult,
    DockerBuildError,
    build_clients,
    build_summary,
    load_client_specs,
)
from execution_testing.client_clis.clis.evmone import (
    EvmoneFixtureConsumerCommon,
)
from execution_testing.client_clis.docker import (
    docker_client_for_image,
    fixture_consumers_from_docker_image,
)
from execution_testing.client_clis.ethereum_cli import EthereumCLI
from execution_testing.client_clis.fixture_consumer_tool import (
    FixtureConsumerTool,
)
from execution_testing.fixtures import (
    BaseFixture,
    BlockchainFixture,
    StateFixture,
)
from execution_testing.fixtures.consume import (
    TestCaseIndexFile,
    TestCaseStream,
)
from execution_testing.fixtures.file import Fixtures
from execution_testing.logging import get_logger

logger = get_logger(__name__)


class CollectOnlyCLI(EthereumCLI):
    """A dummy CLI for use with `--collect-only`."""

    def __init__(self) -> None:  # noqa: D107
        pass


class CollectOnlyFixtureConsumer(
    FixtureConsumerTool,
    CollectOnlyCLI,
    fixture_formats=list(BaseFixture.formats.values()),
):
    """A dummy fixture consumer for use with `--collect-only`."""

    def consume_fixture(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        pass


def pytest_addoption(parser: pytest.Parser) -> None:  # noqa: D103
    consume_group = parser.getgroup(
        "consume_direct",
        "Arguments related to consuming fixtures via a client",
    )

    consume_group.addoption(
        "--bin",
        action="append",
        dest="fixture_consumer_bin",
        type=Path,
        default=[],
        help=(
            "Path to a geth evm executable that provides `blocktest` or "
            "`statetest`. Flag can be used multiple times to specify "
            "multiple fixture consumer binaries. Mutually exclusive with "
            "`--docker.client-branches`."
        ),
    )
    consume_group.addoption(
        "--docker.client-branches",
        action="store",
        dest="docker_client_branches",
        type=Path,
        default=None,
        help=(
            "Use client Docker images instead of local `--bin` binaries. "
            "Builds (or reuses) one image per client listed in the given "
            "`clients.yaml` branch list, then runs the containerized binary "
            "via `docker run`. Mutually exclusive with `--bin`."
        ),
    )
    consume_group.addoption(
        "--docker.client",
        action="store",
        dest="docker_client",
        default=None,
        help=(
            "Comma-separated subset of clients to build and run, restricting "
            "the `--docker.client-branches` clients.yaml to just these "
            "(e.g. `--docker.client go-ethereum,erigon`). Every name must "
            "appear in the clients.yaml, otherwise consume aborts. Only valid "
            "together with `--docker.client-branches`."
        ),
    )
    consume_group.addoption(
        "--docker.build-only",
        action="store_true",
        dest="docker_build_only",
        default=False,
        help=(
            "Build (or check) the `--docker.client-branches` client images "
            "and print a summary, then exit without collecting or running any "
            "tests. Only valid together with `--docker.client-branches`."
        ),
    )
    consume_group.addoption(
        "--docker.nocache",
        action="store_true",
        dest="docker_nocache",
        default=False,
        help=(
            "Resolve the latest upstream commit for every client and force a "
            "`--no-cache` rebuild of any image whose built commit differs. An "
            "image already built at the latest remote commit is reused as-is "
            "(a from-scratch rebuild of the identical commit is pointless). "
            "Loosely mirrors hive's `--docker.nocache`."
        ),
    )
    consume_group.addoption(
        "--docker.buildretries",
        action="store",
        dest="docker_buildretries",
        type=int,
        default=0,
        help=(
            "Retry a failed image build up to N more times (N+1 attempts), "
            "for transient fetch errors. Mirrors hive's "
            "`--docker.buildretries`. Default 0."
        ),
    )
    consume_group.addoption(
        "--docker.buildjobs",
        action="store",
        dest="docker_buildjobs",
        type=int,
        default=None,
        help=(
            "Cap the build-time parallelism of each client image build, "
            "passed to the Dockerfile as the `jobs` build-arg (e.g. "
            "`make -jN`, `cmake --parallel N`, `GOMAXPROCS=N`, "
            "`cargo --jobs N`). Builds still run one client at a time; this "
            "only bounds the cores used *within* a single build. Omit to use "
            "each build tool's default (typically all available cores)."
        ),
    )
    consume_group.addoption(
        "--traces",
        action="store_true",
        dest="consumer_collect_traces",
        default=False,
        help=(
            "Collect traces of the execution information from the fixture "
            "consumer tool."
        ),
    )
    debug_group = parser.getgroup("debug", "Arguments defining debug behavior")
    debug_group.addoption(
        "--dump-dir",
        action="store",
        dest="base_dump_dir",
        type=Path,
        default=None,
        help="Path to dump the fixture consumer tool debug output.",
    )


def pytest_configure(config: pytest.Config) -> None:  # noqa: D103
    config.supported_fixture_formats = [  # type: ignore[attr-defined]
        StateFixture,
        BlockchainFixture,
    ]
    trace = config.getoption("consumer_collect_traces")
    bin_paths = config.getoption("fixture_consumer_bin")
    clients_file = config.getoption("docker_client_branches")
    docker_client = config.getoption("docker_client")
    docker_build_only = config.getoption("docker_build_only")

    # You either run binaries you built yourself (`--bin`) or client Docker
    # images (`--docker.client-branches`), not both.
    if clients_file is not None and bin_paths:
        pytest.exit(
            "Cannot use `--bin` and `--docker.client-branches` together: "
            "choose either local binaries (`--bin`) or Docker images "
            "(`--docker.client-branches`)."
        )

    # `--docker.build-only` only makes sense when there are Docker images to
    # build in the first place.
    if docker_build_only and clients_file is None:
        pytest.exit(
            "`--docker.build-only` requires `--docker.client-branches`: there "
            "are no client images to build otherwise."
        )

    # `--docker.client` only selects a subset of the Docker clients.yaml, so it
    # is meaningless without `--docker.client-branches`.
    selected_clients = None
    if docker_client is not None:
        if clients_file is None:
            pytest.exit(
                "`--docker.client` requires `--docker.client-branches`: it "
                "selects a subset of that clients.yaml."
            )
        selected_clients = [
            name.strip() for name in docker_client.split(",") if name.strip()
        ]
        if not selected_clients:
            pytest.exit(
                "`--docker.client` was given without any client names "
                "(e.g. `--docker.client go-ethereum,erigon`)."
            )

    fixture_consumers = []
    for fixture_consumer_bin_path in bin_paths:
        fixture_consumers.append(
            FixtureConsumerTool.from_binary_path(
                binary_path=Path(fixture_consumer_bin_path),
                trace=trace,
            )
        )
    if clients_file is not None:
        try:
            build_results = _build_docker_clients(
                config, clients_file, selected=selected_clients
            )
        except DockerBuildError as exc:
            pytest.exit(
                f"Docker client image build failed: {exc}",
                returncode=pytest.ExitCode.INTERNAL_ERROR,
            )
        logger.info("\n" + build_summary(build_results))
        if docker_build_only:
            plural = "s" if len(build_results) != 1 else ""
            pytest.exit(
                f"Built/checked {len(build_results)} Docker client "
                f"image{plural}; `--docker.build-only` set, so no tests were "
                "collected or run.",
                returncode=0,
            )
        fixture_consumers.extend(
            _docker_fixture_consumers(
                build_results,
                fixtures_root=config.fixtures_source.path,  # type: ignore[attr-defined]
                dump_dir=config.getoption("base_dump_dir"),
                trace=trace,
            )
        )
    if config.option.markers:
        return
    elif not fixture_consumers and config.option.collectonly:
        warnings.warn(
            (
                "No fixture consumer binaries provided; using a dummy "
                "consumer for collect-only; all possible fixture formats "
                "will be collected. Specify fixture consumer(s) via `--bin` "
                "or `--docker.client-branches` to see collection results."
            ),
            stacklevel=1,
        )
        fixture_consumers = [CollectOnlyFixtureConsumer()]
    elif not fixture_consumers:
        pytest.exit(
            "No fixture consumer provided; please specify a binary path via "
            "`--bin` or client images via `--docker.client-branches`."
        )
    config.fixture_consumers = fixture_consumers  # type: ignore[attr-defined]


def _build_docker_clients(
    config: pytest.Config,
    clients_file: Path,
    *,
    selected: Optional[List[str]] = None,
) -> List[BuildResult]:
    """
    Build (or reuse) the client images in ``clients_file`` and return the
    per-image build results.

    ``selected`` (from ``--docker.client``) restricts the build to that subset
    of the config, preserving the config's order. Only clients with a known
    fixture-consumer mapping are built/used; others in the config are skipped
    with a logged note.
    """
    specs = load_client_specs(clients_file)

    if selected is not None:
        available = {spec.client for spec in specs}
        missing = [name for name in selected if name not in available]
        if missing:
            pytest.exit(
                f"--docker.client: client(s) {missing} not found in "
                f"{clients_file}. Available clients: {sorted(available)}."
            )

    consumable = []
    for spec in specs:
        if selected is not None and spec.client not in selected:
            continue
        if docker_client_for_image(spec.client) is None:
            logger.info(
                f"⊘ {spec.client}: no fixture consumer mapping — "
                f"skipping build"
            )
        else:
            consumable.append(spec.client)

    build_jobs = config.getoption("docker_buildjobs")
    if build_jobs is not None and build_jobs < 1:
        pytest.exit(
            "`--docker.buildjobs` must be a positive integer "
            f"(got {build_jobs}); omit it to use each build tool's default."
        )

    return build_clients(
        clients_file,
        clients=consumable,
        force=config.getoption("docker_nocache"),
        retries=config.getoption("docker_buildretries"),
        build_jobs=build_jobs,
    )


def _docker_fixture_consumers(
    results: List[BuildResult],
    *,
    fixtures_root: Optional[Path],
    dump_dir: Optional[Path],
    trace: bool,
) -> List[FixtureConsumerTool]:
    """Return the fixture consumers backed by the built client images."""
    consumers: List[FixtureConsumerTool] = []
    for result in results:
        consumers.extend(
            fixture_consumers_from_docker_image(
                result.image,
                fixtures_root=fixtures_root,
                dump_dir=dump_dir,
                trace=trace,
            )
        )
    return consumers


@pytest.fixture(scope="function")
def test_dump_dir(
    request: pytest.FixtureRequest, fixture_path: Path, fixture_name: str
) -> Path | None:
    """The directory to write evm debug output to."""
    base_dump_dir = request.config.getoption("base_dump_dir")
    if not base_dump_dir:
        return None
    if len(fixture_name) > 142:
        # ensure file name is not too long for eCryptFS
        fixture_name = fixture_name[:70] + "..." + fixture_name[-70:]
    return base_dump_dir / fixture_path.stem / fixture_name.replace("/", "-")


@pytest.fixture
def fixture_path(
    test_case: TestCaseIndexFile | TestCaseStream,
    fixtures_source: FixturesSource,
) -> Generator[Path, None, None]:
    """
    Path to the current JSON fixture file.

    If the fixture source is stdin, the fixture is written to a temporary json
    file.
    """
    if fixtures_source.is_stdin:
        assert isinstance(test_case, TestCaseStream)
        temp_dir = tempfile.TemporaryDirectory()
        fixture_path = (
            Path(temp_dir.name) / f"{test_case.id.replace('/', '_')}.json"
        )
        fixtures = Fixtures({test_case.id: test_case.fixture})
        with open(fixture_path, "w") as f:
            json.dump(to_json(fixtures), f, indent=4)
        yield fixture_path
        temp_dir.cleanup()
    else:
        assert isinstance(test_case, TestCaseIndexFile)
        yield fixtures_source.path / test_case.json_path


@pytest.fixture(scope="function")
def fixture_name(test_case: TestCaseIndexFile | TestCaseStream) -> str:
    """Name of the current fixture."""
    return test_case.id


def _fixture_consumer_id(fixture_consumer: FixtureConsumerTool) -> str:
    """Build a unique test id for a fixture consumer."""
    name = fixture_consumer.__class__.__name__
    image = getattr(fixture_consumer, "docker_image", None)
    if image:
        return f"{name}-docker-{image}"
    return name


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize test cases for every fixture consumer."""
    metafunc.parametrize(
        "fixture_consumer",
        (
            pytest.param(
                fixture_consumer, id=_fixture_consumer_id(fixture_consumer)
            )
            for fixture_consumer in metafunc.config.fixture_consumers  # type: ignore[attr-defined]
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    """
    Deselect consumer/test-case pairs whose fixture format the consumer
    does not support.

    ``fixture_consumer`` and ``test_case`` are parametrized independently
    (cartesian product), but not every consumer handles every format (e.g.
    Nimbus' ``eest_blockchain`` and evmone's ``evmone-blockchaintest`` only
    consume blockchain fixtures), so the unsupported combinations would
    otherwise fail at runtime instead of never running.

    Additionally, xfail consumer/test-case pairs whose fork the consumer's
    client does not yet implement (e.g. evmone on Amsterdam), so the known
    gap is tracked rather than reported as a spurious failure.
    """
    selected = []
    deselected = []
    for item in items:
        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec is not None else {}
        fixture_consumer = params.get("fixture_consumer")
        test_case = params.get("test_case")
        if (
            fixture_consumer is not None
            and test_case is not None
            and test_case.format not in fixture_consumer.fixture_formats
        ):
            deselected.append(item)
            continue
        selected.append(item)
        if (
            isinstance(fixture_consumer, EvmoneFixtureConsumerCommon)
            and test_case is not None
            and test_case.fork is not None
            and not fixture_consumer.is_fork_supported(test_case.fork)
        ):
            item.add_marker(
                pytest.mark.xfail(
                    reason=(
                        f"evmone does not yet support fork {test_case.fork}"
                    ),
                    strict=False,
                )
            )
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
