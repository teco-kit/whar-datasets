import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, List, Set, TypeVar

from whar_datasets.config.config import WHARConfig
from whar_datasets.utils.logging import logger

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class AbstractStep(ABC, Generic[InputT, OutputT]):
    """Template base class for all processing steps.

    The step lifecycle is intentionally uniform:
    1) check whether cached artifacts are still valid via hash comparison
    2) load and validate step inputs
    3) build step outputs
    4) persist outputs and refresh the step hash

    Generic parameters:
    - `InputT`: data consumed by `build_output`
    - `OutputT`: data produced by `build_output` and handled by cache I/O methods
    """

    def __init__(
        self,
        cfg: WHARConfig,
        hash_dir: Path,
        dependent_on: List["AbstractStep[Any, Any]"] | None = None,
    ) -> None:
        self.cfg = cfg
        self.hash_dir = hash_dir
        self.hash_name: str
        self.relevant_cfg_keys: Set[str] = set()
        self.relevant_values: List[str] = []
        self.dependent_on = dependent_on or []

    def run(self, force_recompute: bool) -> None:
        """Run the full lifecycle for this step."""
        logger.info("Forcing recompute") if force_recompute else None
        logger.info(f"Running {self.__class__.__name__}")

        # check wether an update is needed
        if self._check_hash() and not force_recompute:
            return None

        # pass or load input
        step_input = self.load_input()

        # check initial format for processing
        if not self.validate_input(step_input):
            raise ValueError("Input validation failed")

        # compute (and cache) output
        step_output = self.build_output(step_input)
        self.save_output(step_output)

        # compute and save hash
        step_hash = self._compute_hash()
        self._save_hash(step_hash)

    def _check_hash(self) -> bool:
        """Return True when the cached hash still matches current inputs/config."""
        logger.info(f"Checking hash for {self.__class__.__name__}")

        check = self._load_hash() == self._compute_hash()

        if check:
            logger.info("Hash is up to date")
        else:
            logger.info("Hash is not up to date")

        return check

    def _compute_hash(self) -> str:
        """Compute a deterministic hash from config, dependencies, and extra values."""
        # hash based on relevant part of own config
        sub_cfg_dict = self.cfg.model_dump(include=self.relevant_cfg_keys)
        sub_cfg_json = json.dumps(sub_cfg_dict, sort_keys=True, separators=(",", ":"))
        input_hash = hashlib.sha256(sub_cfg_json.encode("utf-8")).hexdigest()

        # collect dependency hashes
        dep_hashes = [dep._load_hash() for dep in self.dependent_on]

        # Combine own hash + dependency hashes
        combined_str = input_hash + "".join(dep_hashes) + "".join(self.relevant_values)
        final_hash = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()

        return final_hash

    def _save_hash(self, hash: str) -> None:
        """Persist the current step hash to disk."""
        self.hash_dir.mkdir(parents=True, exist_ok=True)

        hash_path = self.hash_dir / f"{self.hash_name}.txt"
        with open(hash_path, "w") as f:
            f.write(hash)

    def _load_hash(self) -> str:
        """Load the persisted hash or return an empty string when absent."""
        hash_path = self.hash_dir / f"{self.hash_name}.txt"

        if not hash_path.exists():
            return ""

        with open(hash_path, "r") as f:
            return f.read().strip()

    @abstractmethod
    def load_input(self) -> InputT:
        """Load/prepare all inputs required to execute this step."""
        pass

    @abstractmethod
    def validate_input(self, step_input: InputT) -> bool:
        """Validate that input artifacts have the expected format and content."""
        pass

    @abstractmethod
    def build_output(self, step_input: InputT) -> OutputT:
        """Execute the core transformation for this step."""
        pass

    @abstractmethod
    def save_output(self, step_output: OutputT) -> None:
        """Persist step output artifacts to cache/storage."""
        pass

    @abstractmethod
    def load_output(self) -> OutputT:
        """Load cached outputs produced by this step."""
        pass
