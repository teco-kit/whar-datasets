import gzip
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path


def extract(file_path: Path, extract_dir: Path) -> None:
    # 1. Extract depending on file type
    if tarfile.is_tarfile(file_path):
        with tarfile.open(file_path) as tar:
            extract_dir.mkdir(parents=True, exist_ok=True)
            tar.extractall(extract_dir)
        file_path.unlink()

    elif zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path) as zipf:
            extract_dir.mkdir(parents=True, exist_ok=True)
            zipf.extractall(extract_dir)
        file_path.unlink()

    # 2b. Handle single-file .gz archives (e.g. *.csv.gz)
    elif file_path.suffix.lower() == ".gz":
        extract_dir.mkdir(parents=True, exist_ok=True)
        output_name = file_path.stem  # remove ".gz"
        output_path = extract_dir / output_name

        with gzip.open(file_path, "rb") as src, open(output_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        file_path.unlink()

    # 3. RAR Handling using 'unrar' ONLY
    elif file_path.suffix.lower() == ".rar":
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Command: unrar x file.rar /path/to/extract/ -y
            subprocess.run(
                ["unrar", "x", str(file_path), str(extract_dir), "-y"],
                check=True,
                stdout=subprocess.DEVNULL,  # Silence success output
                stderr=subprocess.PIPE,  # Capture errors if needed
            )
            file_path.unlink()
        except subprocess.CalledProcessError as e:
            print(f"!!! Error extracting {file_path}")
            print(f"unrar Output:\n{e.stderr.decode()}")
        except FileNotFoundError:
            print("!!! Error: 'unrar' command not found.")
            print("Please install it (e.g., 'sudo apt install unrar')")

    # 4. Recursively extract nested archives
    for root, _, files in os.walk(extract_dir):
        for file in files:
            nested_path = Path(root) / file

            # Check for RAR extension
            is_rar = nested_path.suffix.lower() == ".rar"
            is_gz = nested_path.suffix.lower() == ".gz"

            if (
                tarfile.is_tarfile(nested_path)
                or zipfile.is_zipfile(nested_path)
                or is_rar
                or is_gz
            ):
                nested_extract_dir = nested_path.with_suffix("")
                extract(nested_path, nested_extract_dir)
