"""
Build (or reuse) per-client Docker images from a ``clients.yaml`` branch list.

For each client (keyed by its branch/tag from ``clients.yaml``):

* no image built yet for that branch → build it from scratch (``--no-cache``);
* an image exists and ``--docker.nocache`` was *not* passed → reuse it as-is,
  without re-checking upstream (the fast path);
* an image exists and ``--docker.nocache`` *was* passed → resolve the latest
  upstream commit and rebuild only if it differs from the built commit; an
  unchanged commit is reused, since a from-scratch rebuild of the identical
  commit is pointless.

In every case the branch, the commit the image runs, and that commit's age are
logged. The built commit is read back from the ``steel.source.sha`` image
label stamped on the image at build time.

Used by ``consume direct``'s ``--docker.client-branches`` flow to make the
client images on demand before the fixtures are consumed against them.
"""

import hashlib
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml

from execution_testing.logging import get_logger

logger = get_logger(__name__)

DOCKER_DIR = Path(__file__).parent

# Image namespace and the labels stamped on every built image: a fingerprint
# used to recognize and reuse a prior build, plus the source repo/branch/commit
# the image was built from.
IMAGE_PREFIX = "steel"
FINGERPRINT_LABEL = "steel.fingerprint"
SOURCE_GITHUB_LABEL = "steel.source.github"
SOURCE_TAG_LABEL = "steel.source.tag"
SOURCE_SHA_LABEL = "steel.source.sha"
# Commit author date stamped at build time (ISO 8601) so the reuse path can
# report it without a live GitHub API call. See _commit_datetime.
SOURCE_COMMIT_DATETIME_LABEL = "steel.source.commit_datetime"

# Signatures in buildx output that mark a *transient* failure — a DNS,
# registry, or network hiccup (e.g. failing to fetch a base image's auth
# token) rather than a genuine build error. These almost always clear on a
# second attempt, so they are retried even when the user did not raise
# ``--docker.buildretries`` above its default.
_TRANSIENT_BUILD_ERROR = re.compile(
    "|".join(
        (
            r"i/o timeout",
            r"failed to fetch",
            r"failed to authorize",
            r"failed to resolve",
            r"TLS handshake timeout",
            r"connection refused",
            r"connection reset",
            r"temporary failure in name resolution",
            r"no such host",
            r"context deadline exceeded",
            r"timeout exceeded while awaiting headers",
        )
    ),
    re.IGNORECASE,
)
# Minimum number of retries to grant a transient failure even when
# ``--docker.buildretries`` is 0 (the default).
_MIN_TRANSIENT_RETRIES = 2
# Linear backoff between attempts: attempt number × this many seconds.
_RETRY_BACKOFF_SECONDS = 5


@dataclass(frozen=True)
class ClientSpec:
    """A single entry from ``clients.yaml``."""

    client: str
    github: str
    tag: str


@dataclass(frozen=True)
class BuildResult:
    """Outcome of building (or reusing) one client image."""

    client: str
    image: str
    github: str
    tag: str
    sha: str
    reused: bool
    commit_datetime: Optional[datetime] = None


class BuildAction(Enum):
    """What :func:`build_clients` decides to do with one client image."""

    BUILD_NEW = "build"  # no local image for this branch yet
    REBUILD = "rebuild"  # forced freshness check and upstream has moved
    REUSE = "reuse"  # local image exists, upstream deliberately not checked
    REUSE_UP_TO_DATE = "up-to-date"  # forced, but already at the latest commit


@dataclass(frozen=True)
class BuildPlan:
    """
    A per-client decision resolved *before* any image is built.

    Holds the locally built commit (if any) and its age, the latest upstream
    commit and its age (resolved only when relevant), and the resulting
    :class:`BuildAction`, so the whole plan can be reported up front.
    """

    spec: ClientSpec
    dockerfile: Path
    image: str
    action: BuildAction
    built_sha: Optional[str]
    built_commit_datetime: Optional[datetime]
    remote_sha: Optional[str]
    remote_commit_datetime: Optional[datetime]


class DockerBuildError(Exception):
    """Raised when a client image cannot be resolved or built."""


def sanitize_docker_tag(ref: str) -> str:
    """
    Turn a git branch/tag into a valid Docker image tag.

    A Docker tag must match ``[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}``, but git refs
    may contain ``/`` and other illegal characters. Every illegal character is
    replaced with ``_``, a leading ``.`` or ``-`` is prefixed with ``_``, and
    the result is capped at 128 characters.
    """
    out = "".join(c if c.isalnum() or c in "_.-" else "_" for c in ref)
    if not out or out[0] in ".-":
        out = "_" + out
    return out[:128]


def load_client_specs(
    clients_file: Path,
) -> List[ClientSpec]:
    """Parse ``clients.yaml`` into a list of :class:`ClientSpec`."""
    if not clients_file.is_file():
        raise DockerBuildError(f"clients file not found: {clients_file}")
    raw = yaml.safe_load(clients_file.read_text()) or []
    specs: List[ClientSpec] = []
    for entry in raw:
        build_args = entry.get("build_args", {})
        specs.append(
            ClientSpec(
                client=entry["client"],
                github=build_args["github"],
                tag=str(build_args["tag"]),
            )
        )
    return specs


def _pick_ref_sha(ls_remote_output: str, tag: str) -> Optional[str]:
    """
    Return the SHA of the ref that *exactly* matches ``tag``.

    ``git ls-remote`` with a bare name like ``master`` also returns refs such
    as ``refs/heads/kch/master``; we deliberately accept only exact matches
    (branch, then annotated tag peeled to its commit, then lightweight tag) and
    refuse to guess a near-match.
    """
    heads = tag_peeled = tag_plain = None
    for line in ls_remote_output.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        sha, ref = parts
        if ref == f"refs/heads/{tag}":
            heads = sha
        elif ref == f"refs/tags/{tag}^{{}}":
            tag_peeled = sha
        elif ref == f"refs/tags/{tag}" and tag_plain is None:
            tag_plain = sha
    return heads or tag_peeled or tag_plain


def _resolve_remote_sha(github: str, tag: str) -> str:
    """Resolve ``tag`` to a 40-char commit SHA, or raise."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", f"https://github.com/{github}", tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
        sha = (
            _pick_ref_sha(result.stdout, tag)
            if result.returncode == 0
            else None
        )
    except Exception:
        sha = None
    if sha:
        return sha
    # Accept a literal commit-ish (a raw SHA given directly in the config).
    if len(tag) >= 7 and all(c in "0123456789abcdef" for c in tag.lower()):
        return tag
    raise DockerBuildError(
        f"'{tag}' is not an exact branch or tag at "
        f"https://github.com/{github} (refusing to guess a near-match); "
        f"fix the 'tag' field or check the remote is reachable."
    )


def _dockerfile_sha(dockerfile: Path) -> str:
    """Return the sha256 of the Dockerfile contents."""
    return hashlib.sha256(dockerfile.read_bytes()).hexdigest()


def _image_label(image: str, label: str) -> Optional[str]:
    """Return the value of ``label`` on ``image``, or None if absent."""
    try:
        result = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                image,
                "--format",
                f'{{{{ index .Config.Labels "{label}" }}}}',
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _commit_datetime(github: str, sha: str) -> Optional[datetime]:
    """
    Best-effort lookup of a commit's author date via the GitHub API.

    Returns None on any failure (no network, rate limited, missing
    ``requests``) so a build is never blocked on it. Honors
    ``GITHUB_TOKEN`` / ``GH_TOKEN`` to dodge rate limits.

    This is a best-effort, network-dependent fallback only: the reuse path
    prefers the commit date stamped on the image at build time (see
    ``SOURCE_COMMIT_DATETIME_LABEL``), so a transient API hiccup here no
    longer turns into an ``unknown`` for an already-built image.
    """
    try:
        import requests
    except Exception:
        logger.debug("commit date lookup skipped: requests unavailable")
        return None
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.get(
            f"https://api.github.com/repos/{github}/commits/{sha}",
            headers=headers,
            timeout=8,
        )
        response.raise_for_status()
        date_str = response.json()["commit"]["author"]["date"]
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception as exc:
        logger.debug(
            f"commit date lookup for {github}@{sha[:12]} failed: {exc}"
        )
        return None


def _label_commit_datetime(image: str) -> Optional[datetime]:
    """
    Read the commit author date stamped on ``image`` at build time.

    Returns None if the label is absent (e.g. an image built before the
    label existed) or unparsable, in which case callers fall back to a
    live :func:`_commit_datetime` lookup.
    """
    raw = _image_label(image, SOURCE_COMMIT_DATETIME_LABEL)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _humanize_timedelta(seconds: float) -> str:
    """Render a positive duration like ``6 days`` / ``3 hours``."""
    seconds = abs(int(seconds))
    for unit, size in (
        ("day", 86400),
        ("hour", 3600),
        ("minute", 60),
    ):
        if seconds >= size:
            value = seconds // size
            return f"{value} {unit}{'s' if value != 1 else ''}"
    return f"{seconds} second{'s' if seconds != 1 else ''}"


def _commit_suffix(commit_dt: Optional[datetime], now: datetime) -> str:
    """Annotate a commit line with its date and age, e.g. ``, 6 days ago``."""
    if commit_dt is None:
        return ""
    date_str = commit_dt.astimezone(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    age = _humanize_timedelta((now - commit_dt).total_seconds())
    return f", committed {date_str} ({age} ago)"


def _delta_phrase(
    prev_dt: Optional[datetime], new_dt: Optional[datetime]
) -> str:
    """Describe how much newer/older ``new_dt`` is than ``prev_dt``."""
    if prev_dt is None or new_dt is None:
        return ""
    diff = (new_dt - prev_dt).total_seconds()
    if diff == 0:
        return "same commit time"
    return f"{_humanize_timedelta(diff)} {'newer' if diff > 0 else 'older'}"


def _run_one_build(args: Sequence[str]) -> Tuple[int, bool]:
    """
    Run one ``docker buildx build`` invocation.

    Streams the build output to the terminal as it arrives (so the live
    ``--progress=plain`` output is preserved) while capturing it to classify
    the failure. Returns the process exit code and whether the output bears a
    transient-error signature (see :data:`_TRANSIENT_BUILD_ERROR`).
    """
    process = subprocess.Popen(
        list(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    captured: List[str] = []
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    returncode = process.wait()
    transient = returncode != 0 and bool(
        _TRANSIENT_BUILD_ERROR.search("".join(captured))
    )
    return returncode, transient


def _run_build(args: Sequence[str], image: str, retries: int) -> None:
    """
    Run ``docker buildx build``, retrying failures.

    A genuine build failure is retried ``retries`` times (the configured
    ``--docker.buildretries``). A *transient* failure — a DNS/registry/network
    hiccup — is retried at least :data:`_MIN_TRANSIENT_RETRIES` times even when
    ``retries`` is 0, since those almost always clear on a second attempt.
    Retries back off linearly.
    """
    attempt = 0
    while True:
        attempt += 1
        returncode, transient = _run_one_build(args)
        if returncode == 0:
            return
        extra = max(retries, _MIN_TRANSIENT_RETRIES) if transient else retries
        attempts = extra + 1
        if attempt >= attempts:
            raise DockerBuildError(
                f"{image} build failed after {attempt} attempt(s)"
                + (" (transient network error)" if transient else "")
            )
        delay = _RETRY_BACKOFF_SECONDS * attempt
        logger.warning(
            f"✗ {image} build failed (attempt {attempt}/{attempts})"
            f"{' — transient network error' if transient else ''} — "
            f"retrying in {delay}s"
        )
        time.sleep(delay)


def _result(
    spec: ClientSpec,
    image: str,
    sha: str,
    *,
    reused: bool,
    commit_dt: Optional[datetime],
) -> BuildResult:
    """Assemble a :class:`BuildResult` for ``spec``/``image``."""
    return BuildResult(
        client=spec.client,
        image=image,
        github=spec.github,
        tag=spec.tag,
        sha=sha,
        reused=reused,
        commit_datetime=commit_dt,
    )


def _build_image(
    spec: ClientSpec,
    dockerfile: Path,
    image: str,
    remote_sha: str,
    *,
    retries: int,
    build_jobs: Optional[int],
    commit_dt: Optional[datetime] = None,
) -> None:
    """
    Build ``image`` from scratch (always ``--no-cache``).

    A build is only ever issued when the client has no prior image for this
    branch, or when ``--no-cache`` was requested and upstream has actually
    moved — in both cases a from-scratch build is what is wanted. The
    ``steel.*`` source and fingerprint labels are stamped on the result,
    including the commit date so the reuse path needs no GitHub lookup.
    """
    fingerprint = f"{remote_sha}:{_dockerfile_sha(dockerfile)}"
    build_args = [
        "docker",
        "buildx",
        "build",
        "--no-cache",
        "--network=host",
        "--progress=plain",
        "--load",
        "--label",
        f"{FINGERPRINT_LABEL}={fingerprint}",
        "--label",
        f"{SOURCE_GITHUB_LABEL}={spec.github}",
        "--label",
        f"{SOURCE_TAG_LABEL}={spec.tag}",
    ]
    if commit_dt is not None:
        build_args += [
            "--label",
            f"{SOURCE_COMMIT_DATETIME_LABEL}={commit_dt.isoformat()}",
        ]
    build_args += [
        "--label",
        f"{SOURCE_SHA_LABEL}={remote_sha}",
        "--build-arg",
        f"github={spec.github}",
        "--build-arg",
        f"tag={spec.tag}",
    ]
    if build_jobs:
        build_args += ["--build-arg", f"jobs={build_jobs}"]
    build_args += [
        "-t",
        image,
        "-f",
        str(dockerfile),
        str(DOCKER_DIR),
    ]
    _run_build(build_args, image, retries)
    logger.debug(f"✓ {image} built")


def build_clients(
    clients_file: Path,
    *,
    clients: Optional[Sequence[str]] = None,
    force: bool = False,
    retries: int = 0,
    build_jobs: Optional[int] = None,
) -> List[BuildResult]:
    """
    Build or reuse the Docker images for the clients in ``clients_file``.

    Resolves a :class:`BuildPlan` for every targeted client first — reading
    the locally built commit and, when relevant, the latest upstream commit —
    logs the whole plan (see :func:`_log_build_plan`), and only then starts
    building. Per client (matched per branch/tag from the config):

    * **No prior image** for that branch → build it from scratch
      (``--no-cache``).
    * **Prior image exists, ``force`` not set** → reuse it as-is, without
      re-checking upstream (the fast path).
    * **Prior image exists, ``force`` set** (``--docker.nocache``) → resolve
      the latest upstream commit and only rebuild (``--no-cache``) if it
      differs from the built commit; an unchanged commit is reused, since a
      from-scratch rebuild of the identical commit is pointless.

    ``clients`` restricts the build to a subset of the config. Returns one
    :class:`BuildResult` per targeted client that has an image afterwards.
    """
    specs = load_client_specs(clients_file)
    spec_names = {spec.client for spec in specs}
    wanted = set(clients) if clients is not None else None
    if wanted is not None:
        missing = wanted - spec_names
        if missing:
            raise DockerBuildError(
                f"client(s) {sorted(missing)} not found in {clients_file}"
            )

    # Phase 1 — resolve the full plan before touching any image, so the user
    # sees what is local, what is upstream, and what will (not) be rebuilt.
    plans: List[BuildPlan] = []
    for spec in specs:
        if wanted is not None and spec.client not in wanted:
            continue
        dockerfile = DOCKER_DIR / f"Dockerfile.{spec.client}"
        if not dockerfile.is_file():
            if wanted is not None:
                raise DockerBuildError(
                    f"no Dockerfile for client '{spec.client}' "
                    f"(expected {dockerfile})"
                )
            logger.info(f"⊘ skipping {spec.client} (no {dockerfile.name})")
            continue
        plans.append(_plan_build(spec, dockerfile, force=force))

    _log_build_plan(plans, force=force, now=datetime.now(timezone.utc))

    # Phase 2 — execute the plan (the only place an actual build is issued).
    return [
        _execute_plan(plan, retries=retries, build_jobs=build_jobs)
        for plan in plans
    ]


def planned_client_images(
    clients_file: Path,
    *,
    clients: Optional[Sequence[str]] = None,
) -> List[BuildResult]:
    """
    Resolve each targeted client's image *name* without building anything.

    A ``--collect-only`` run (e.g. ``consume direct``'s parallelism pre-count)
    only needs the fixture-consumer objects to count and parametrize tests, and
    those are derived from the image *name* alone (``docker``'s
    ``fixture_consumers_from_docker_image`` with ``start_backend=False`` never
    inspects or requires a built image and never invokes a consumer). Skipping
    the ``git ls-remote`` freshness check and the ``docker buildx`` build keeps
    that pass fast and silent, and leaves the actual (possibly minutes-long)
    build to the real run, where its progress is shown to the user rather than
    swallowed by the quiet pre-count.

    ``clients`` restricts the result to a subset of the config. Only clients
    with a matching ``Dockerfile`` are included, as in :func:`build_clients`
    (which is what the real run uses). The returned results carry the image
    name only; their ``sha``/``commit_datetime`` are left empty because no
    build or upstream lookup is performed here.
    """
    specs = load_client_specs(clients_file)
    wanted = set(clients) if clients is not None else None
    if wanted is not None:
        missing = wanted - {spec.client for spec in specs}
        if missing:
            raise DockerBuildError(
                f"client(s) {sorted(missing)} not found in {clients_file}"
            )

    results: List[BuildResult] = []
    for spec in specs:
        if wanted is not None and spec.client not in wanted:
            continue
        dockerfile = DOCKER_DIR / f"Dockerfile.{spec.client}"
        if not dockerfile.is_file():
            if wanted is not None:
                raise DockerBuildError(
                    f"no Dockerfile for client '{spec.client}' "
                    f"(expected {dockerfile})"
                )
            continue
        image = f"{IMAGE_PREFIX}/{spec.client}:{sanitize_docker_tag(spec.tag)}"
        results.append(
            BuildResult(
                client=spec.client,
                image=image,
                github=spec.github,
                tag=spec.tag,
                sha="",
                reused=True,
                commit_datetime=None,
            )
        )
    return results


def _plan_build(
    spec: ClientSpec, dockerfile: Path, *, force: bool
) -> BuildPlan:
    """Resolve what to do for ``spec`` without building anything."""
    image = f"{IMAGE_PREFIX}/{spec.client}:{sanitize_docker_tag(spec.tag)}"
    built_sha = _image_label(image, SOURCE_SHA_LABEL)

    def plan(
        action: BuildAction,
        *,
        remote_sha: Optional[str] = None,
    ) -> BuildPlan:
        return BuildPlan(
            spec=spec,
            dockerfile=dockerfile,
            image=image,
            action=action,
            built_sha=built_sha,
            # Prefer the date stamped on the image at build time; only fall
            # back to a (network-fragile) GitHub lookup for legacy images
            # built before the label existed.
            built_commit_datetime=(
                _label_commit_datetime(image)
                or _commit_datetime(spec.github, built_sha)
                if built_sha is not None
                else None
            ),
            remote_sha=remote_sha,
            remote_commit_datetime=(
                _commit_datetime(spec.github, remote_sha)
                if remote_sha is not None
                else None
            ),
        )

    # No local image for this branch → must build from scratch.
    if built_sha is None:
        return plan(
            BuildAction.BUILD_NEW,
            remote_sha=_resolve_remote_sha(spec.github, spec.tag),
        )

    # Local image exists and freshness not forced → reuse without checking
    # upstream (the fast path).
    if not force:
        return plan(BuildAction.REUSE)

    # Forced freshness check → resolve upstream and rebuild only if it moved.
    remote_sha = _resolve_remote_sha(spec.github, spec.tag)
    action = (
        BuildAction.REUSE_UP_TO_DATE
        if remote_sha == built_sha
        else BuildAction.REBUILD
    )
    return plan(action, remote_sha=remote_sha)


def _plan_phrase(plan: BuildPlan) -> str:
    """One-line explanation of what ``plan`` will do and why."""
    if plan.action is BuildAction.BUILD_NEW:
        return "build from scratch — no local image for this branch yet"
    if plan.action is BuildAction.REUSE:
        return (
            "reuse local image — upstream not checked "
            "(pass --docker.nocache to force a freshness check)"
        )
    if plan.action is BuildAction.REUSE_UP_TO_DATE:
        return "reuse — already at the latest upstream commit, no rebuild"
    delta = _delta_phrase(
        plan.built_commit_datetime, plan.remote_commit_datetime
    )
    return "REBUILD --no-cache — upstream moved" + (
        f" ({delta})" if delta else ""
    )


def _log_build_plan(
    plans: Sequence[BuildPlan], *, force: bool, now: datetime
) -> None:
    """Log a per-client plan (local vs. upstream, decision) before building."""
    if not plans:
        return
    mode = (
        "--docker.nocache set — checking each branch against the latest "
        "upstream commit"
        if force
        else "reusing existing images where present "
        "(pass --docker.nocache to force a freshness check)"
    )
    lines = [f"Docker client image plan ({len(plans)}) — {mode}:"]
    for plan in plans:
        lines.append(f"  {plan.spec.client} ({plan.image}):")
        if plan.built_sha is None:
            lines.append("    local : none built yet")
        else:
            lines.append(
                f"    local : {plan.built_sha[:12]}"
                f"{_commit_suffix(plan.built_commit_datetime, now)}"
            )
        if plan.remote_sha is not None:
            lines.append(
                f"    remote: {plan.remote_sha[:12]}"
                f"{_commit_suffix(plan.remote_commit_datetime, now)}"
            )
        lines.append(f"    → {_plan_phrase(plan)}")

    to_build = [
        p.spec.client
        for p in plans
        if p.action in (BuildAction.BUILD_NEW, BuildAction.REBUILD)
    ]
    to_reuse = [
        p.spec.client
        for p in plans
        if p.action in (BuildAction.REUSE, BuildAction.REUSE_UP_TO_DATE)
    ]
    summary = f"  plan: {len(to_build)} to build"
    if to_build:
        summary += f" ({', '.join(to_build)})"
    summary += f", {len(to_reuse)} to reuse"
    if to_reuse:
        summary += f" ({', '.join(to_reuse)})"
    lines.append(summary)
    logger.info("\n".join(lines))


def _execute_plan(
    plan: BuildPlan, *, retries: int, build_jobs: Optional[int]
) -> BuildResult:
    """Carry out a single :class:`BuildPlan`, building only if it says to."""
    spec = plan.spec
    if plan.action in (BuildAction.BUILD_NEW, BuildAction.REBUILD):
        assert plan.remote_sha is not None
        verb = (
            "building"
            if plan.action is BuildAction.BUILD_NEW
            else "rebuilding"
        )
        logger.debug(f"→ {verb} {plan.image} [--no-cache]")
        _build_image(
            spec,
            plan.dockerfile,
            plan.image,
            plan.remote_sha,
            retries=retries,
            build_jobs=build_jobs,
            commit_dt=plan.remote_commit_datetime,
        )
        return _result(
            spec,
            plan.image,
            plan.remote_sha,
            reused=False,
            commit_dt=plan.remote_commit_datetime,
        )

    assert plan.built_sha is not None
    logger.debug(f"✓ {plan.image} reusing existing build")
    return _result(
        spec,
        plan.image,
        plan.built_sha,
        reused=True,
        commit_dt=plan.built_commit_datetime,
    )


def build_summary(results: Sequence[BuildResult]) -> str:
    """
    Render a human-readable table summarizing :func:`build_clients` results.

    One row per image: its image tag, the client, the branch/tag built, the
    commit the image runs, when that commit was authored and its age, and
    whether an existing build was reused or the image was (re)built.
    """
    now = datetime.now(timezone.utc)
    header = (
        "IMAGE",
        "CLIENT",
        "BRANCH",
        "COMMIT",
        "COMMITTED",
        "AGE",
        "STATUS",
    )
    rows = [header]
    for result in sorted(results, key=lambda r: r.client):
        if result.commit_datetime is not None:
            committed = result.commit_datetime.astimezone(
                timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC")
            age = f"{_humanize_timedelta((now - result.commit_datetime).total_seconds())} ago"  # noqa: E501
        else:
            committed = age = "unknown"
        rows.append(
            (
                result.image,
                result.client,
                result.tag,
                result.sha[:12],
                committed,
                age,
                "reused" if result.reused else "built",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        for row in rows
    ]
    plural = "s" if len(results) != 1 else ""
    title = f"Docker client image summary ({len(results)} image{plural}):"
    return "\n".join([title, *lines])
