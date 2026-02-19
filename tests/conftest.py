import sys
import types
from pathlib import Path


def pytest_configure() -> None:
    # Avoid importing whar_datasets/__init__.py during tests (it pulls heavy optional deps).
    repo_root = Path(__file__).resolve().parents[1]
    package_root = repo_root / "src" / "whar_datasets"

    pkg = types.ModuleType("whar_datasets")
    pkg.__path__ = [str(package_root)]
    sys.modules["whar_datasets"] = pkg

