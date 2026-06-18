"""
Pytest plugin to fix the test IDs for all pytest commands that use a
command-logic test file.
"""

from typing import List

import pytest


def pytest_collection_modifyitems(items: List[pytest.Item]) -> None:
    """
    Modify collected item names to remove the test runner function from the
    name.
    """
    for item in items:
        original_name = item.originalname  # type: ignore
        remove = f"{original_name}["
        if item.name.startswith(remove):
            item.name = item.name.removeprefix(remove)[:-1]
        if remove in item.nodeid:
            # Under `--dist loadgroup` xdist has already appended an
            # `@<group>` suffix to the nodeid (see xdist's remote.py). Split
            # it off before stripping the trailing `]` of the parametrize
            # bracket, otherwise the slice eats the last character of the
            # group name and collapses every test into a single scheduling
            # group (sending all of them to one worker). Mirror xdist's own
            # detection (a real suffix's `@` comes after the last `]`) so an
            # `@` inside a test id is left untouched.
            nodeid = item.nodeid
            suffix = ""
            at = nodeid.rfind("@")
            if at != -1 and at > nodeid.rfind("]"):
                suffix = nodeid[at:]
                nodeid = nodeid[:at]
            nodeid = nodeid[nodeid.index(remove) + len(remove) : -1]
            item._nodeid = nodeid + suffix
