"""Shared by the method library's tests: the repository's root, with agent/ importable as the image's /opt/agent is,
the longest first line a function's card may have, and the two fixtures the card checks read."""
import functools
import json
import os
import sys
from pathlib import Path

# numpy and scipy each start a BLAS thread pool at import; no test does linear algebra worth a thread.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
FIXTURES = ROOT / "tests" / "fixtures"
FIRST_LINE = 120


@functools.cache
def shelved() -> dict:
    """`before`: each function's module (its shelf) before the library was reorganised into its current modules."""
    return json.loads((FIXTURES / "qf-libshelves1.json").read_text(encoding="utf-8"))


def shelved_before(shelf: str) -> list[str]:
    """The functions `shelf` held before the reorganisation, wherever they live now."""
    return [name for name, was in shelved()["before"].items() if was == shelf]


@functools.cache
def folded() -> frozenset[str]:
    """The functions the reorganisation folded: in no module's `__all__`, still importable as `qf.<name>`."""
    return frozenset(json.loads((FIXTURES / "qf-folded.json").read_text(encoding="utf-8")))


def pytest_configure(config):
    for tier in ("always", "fast", "slow"):
        config.addinivalue_line("markers", f"{tier}: the test's tier")
