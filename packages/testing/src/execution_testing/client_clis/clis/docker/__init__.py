"""
Docker image build assets and builder for client fixture consumers.

This package bundles the per-client ``Dockerfile.<client>`` images, the
``clients.yaml`` branch list, and a Python builder (:mod:`.builder`) that
builds or reuses those images.
"""

from .builder import (
    BuildResult,
    ClientSpec,
    DockerBuildError,
    build_clients,
    build_summary,
    load_client_specs,
    planned_client_images,
    sanitize_docker_tag,
)

__all__ = (
    "BuildResult",
    "ClientSpec",
    "DockerBuildError",
    "build_clients",
    "build_summary",
    "load_client_specs",
    "planned_client_images",
    "sanitize_docker_tag",
)
