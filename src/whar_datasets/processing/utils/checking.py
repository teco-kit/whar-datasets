from pathlib import Path

from whar_datasets.utils.logging import logger


def check_windowing(cache_dir: Path, windows_dir: Path) -> bool:
    """Check whether required cached windowing artifacts are present."""
    logger.info("Checking windowing")

    if not windows_dir.exists():
        logger.warning(f"Windows directory not found at '{windows_dir}'.")
        return False

    if len(list(windows_dir.iterdir())) == 0:
        logger.warning(f"Windows directory '{windows_dir}' is empty.")
        return False

    window_df_path = cache_dir / "window_df.csv"

    if not window_df_path.exists():
        logger.warning(f"Window index file not found at '{window_df_path}'.")
        return False

    logger.info("Windowing exists.")
    return True
