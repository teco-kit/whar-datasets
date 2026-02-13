import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Set, TypeAlias

from whar_datasets.config.config import WHARConfig
from whar_datasets.utils.logging import logger

base_type: TypeAlias = Any
result_type: TypeAlias = Any


class ProcessingStep(ABC):
    def __init__(
        self, cfg: WHARConfig, hash_dir: Path, dependent_on: List["ProcessingStep"] = []
    ):
        self.cfg = cfg
        self.hash_dir = hash_dir
        self.hash_name: str
        self.relevant_cfg_keys: Set[str] = set()
        self.relevant_values: List[str] = []
        self.dependent_on = dependent_on

    def compute_hash(self) -> str:
        # hash based on relevant part of own config
        sub_cfg_dict = self.cfg.model_dump(include=self.relevant_cfg_keys)
        sub_cfg_json = json.dumps(sub_cfg_dict, sort_keys=True, separators=(",", ":"))
        base_hash = hashlib.sha256(sub_cfg_json.encode("utf-8")).hexdigest()

        # collect dependency hashes
        dep_hashes = [dep.load_hash() for dep in self.dependent_on]

        # Combine own hash + dependency hashes
        combined_str = base_hash + "".join(dep_hashes) + "".join(self.relevant_values)
        final_hash = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()

        return final_hash

    def save_hash(self, hash: str) -> None:
        self.hash_dir.mkdir(parents=True, exist_ok=True)

        hash_path = self.hash_dir / f"{self.hash_name}.txt"
        with open(hash_path, "w") as f:
            f.write(hash)

    def load_hash(self) -> str:
        hash_path = self.hash_dir / f"{self.hash_name}.txt"

        if not hash_path.exists():
            return ""

        with open(hash_path, "r") as f:
            return f.read().strip()

    def check_hash(self) -> bool:
        logger.info(f"Checking hash for {self.__class__.__name__}")

        check = self.load_hash() == self.compute_hash()

        if check:
            logger.info("Hash is up to date")
        else:
            logger.info("Hash is not up to date")

        return check

    def run(self, force_recompute: bool) -> None:
        logger.info("Forcing recompute") if force_recompute else None
        logger.info(f"Running {self.__class__.__name__}")

        # check wether an update is needed
        if self.check_hash() and not force_recompute:
            return None

        # pass or load base
        base = self.get_base()

        # check initial format for processing
        if not self.check_initial_format(base):
            raise ValueError("Initial format check failed")

        # compute (and cache) results
        results = self.compute_results(base)
        self.save_results(results)

        # compute and save hash
        hash = self.compute_hash()
        self.save_hash(hash)

    @abstractmethod
    def get_base(self) -> base_type:
        pass

    @abstractmethod
    def check_initial_format(self, base: base_type) -> bool:
        pass

    @abstractmethod
    def compute_results(self, base: base_type) -> result_type:
        pass

    @abstractmethod
    def save_results(self, results: result_type) -> None:
        pass

    @abstractmethod
    def load_results(self) -> result_type:
        pass
