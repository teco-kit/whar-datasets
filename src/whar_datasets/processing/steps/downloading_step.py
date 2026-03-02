import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any, List, Set, TypeAlias
from urllib.parse import parse_qs, urlparse

import requests
from dotenv import dotenv_values, find_dotenv

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.steps.abstract_step import AbstractStep
from whar_datasets.utils.logging import logger

base_type: TypeAlias = Any
result_type: TypeAlias = None


class DownloadingStep(AbstractStep):
    def __init__(
        self,
        cfg: WHARConfig,
        datasets_dir: Path,
        dataset_dir: Path,
        data_dir: Path,
    ):
        super().__init__(cfg, data_dir)

        self.datasets_dir = datasets_dir
        self.dataset_dir = dataset_dir
        self.data_dir = data_dir

        self.hash_name: str = "download_hash"
        self.relevant_cfg_keys: Set[str] = {"dataset_id", "download_url"}

    def get_base(self) -> base_type:
        return None

    def check_initial_format(self, base: base_type) -> bool:
        return True

    @staticmethod
    def _is_kaggle_api_download(url: str) -> bool:
        parsed = urlparse(url)
        return (
            parsed.netloc.lower() == "www.kaggle.com"
            and "/api/v1/datasets/download/" in parsed.path
        )

    @staticmethod
    def _parse_kaggle_dataset_url(url: str) -> tuple[str, str, str | None]:
        parsed = urlparse(url)
        parts = [part for part in parsed.path.split("/") if part]
        # Expected: /api/v1/datasets/download/{owner}/{dataset}
        if len(parts) < 6:
            raise RuntimeError(f"Unexpected Kaggle API URL format: '{url}'")
        owner_slug = parts[-2]
        dataset_slug = parts[-1]
        query_map = parse_qs(parsed.query)
        version = query_map.get("datasetVersionNumber", [None])[0]
        return owner_slug, dataset_slug, version

    def _resolve_kaggle_credentials(self) -> tuple[str, str] | None:
        dotenv_map: dict[str, str | None] = {}
        found_dotenv = find_dotenv(".env", usecwd=True)
        if found_dotenv:
            dotenv_map.update(dotenv_values(found_dotenv))
        fallback_dotenv = Path.cwd() / ".env"
        if fallback_dotenv.is_file():
            dotenv_map.update(dotenv_values(fallback_dotenv))

        def _pick(name: str) -> str | None:
            env_value = os.getenv(name)
            if env_value:
                return env_value.strip()
            dotenv_value = dotenv_map.get(name)
            if isinstance(dotenv_value, str) and dotenv_value.strip():
                return dotenv_value.strip()
            return None

        kaggle_username = _pick("KAGGLE_USERNAME") or _pick("KAGGLE_API_USERNAME")
        kaggle_key = _pick("KAGGLE_KEY")
        if kaggle_username and kaggle_key:
            return kaggle_username, kaggle_key

        api_token = _pick("KAGGLE_API_TOKEN")
        if not api_token:
            return None

        # Support token formats:
        # 1) "username:key"
        # 2) {"username":"...","key":"..."}
        if ":" in api_token:
            username, key = api_token.split(":", 1)
            username = username.strip()
            key = key.strip()
            if username and key:
                return username, key

        if api_token.startswith("{") and api_token.endswith("}"):
            try:
                token_json = json.loads(api_token)
            except json.JSONDecodeError:
                token_json = {}
            if isinstance(token_json, dict):
                username = str(token_json.get("username", "")).strip()
                key = str(token_json.get("key", "")).strip()
                if username and key:
                    return username, key

        # If username is provided separately, treat token as the key.
        if kaggle_username:
            return kaggle_username, api_token

        return None

    def _normalize_download_urls(self) -> List[str]:
        if isinstance(self.cfg.download_url, str):
            urls = [self.cfg.download_url]
        else:
            urls = list(self.cfg.download_url)

        normalized_urls = [
            url.strip() for url in urls if isinstance(url, str) and url.strip()
        ]
        if not normalized_urls:
            raise RuntimeError(
                f"Config '{self.cfg.dataset_id}' must define at least one non-empty download URL."
            )

        return normalized_urls

    @staticmethod
    def _filename_from_url(url: str) -> str:
        path_name = Path(urlparse(url).path).name
        return path_name if path_name else "download.bin"

    def _build_download_path(
        self,
        url: str,
        index: int,
        used_filenames: Set[str],
    ) -> Path:
        base_name = self._filename_from_url(url)
        candidate = base_name

        if candidate in used_filenames:
            stem = Path(base_name).stem
            suffix = Path(base_name).suffix
            candidate = f"{stem}_{index + 1}{suffix}"

        dedupe_idx = 2
        while candidate in used_filenames:
            stem = Path(base_name).stem
            suffix = Path(base_name).suffix
            candidate = f"{stem}_{index + 1}_{dedupe_idx}{suffix}"
            dedupe_idx += 1

        used_filenames.add(candidate)
        return self.data_dir / candidate

    def _download_kaggle_dataset(
        self,
        url: str,
        request_auth: tuple[str, str],
    ) -> None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        owner_slug, dataset_slug, version = self._parse_kaggle_dataset_url(url)
        kaggle_username, kaggle_key = request_auth

        # Kaggle client reads credentials from env during authenticate().
        prev_username = os.environ.get("KAGGLE_USERNAME")
        prev_key = os.environ.get("KAGGLE_KEY")
        os.environ["KAGGLE_USERNAME"] = kaggle_username
        os.environ["KAGGLE_KEY"] = kaggle_key

        try:
            api = KaggleApi()
            api.authenticate()
            if version is not None:
                logger.info(
                    "Kaggle datasetVersionNumber is present in URL but not supported "
                    "by this Kaggle client version; downloading latest accessible version."
                )
            api.dataset_download_files(
                f"{owner_slug}/{dataset_slug}",
                path=str(self.data_dir),
                force=False,
                quiet=False,
            )
        finally:
            if prev_username is None:
                os.environ.pop("KAGGLE_USERNAME", None)
            else:
                os.environ["KAGGLE_USERNAME"] = prev_username
            if prev_key is None:
                os.environ.pop("KAGGLE_KEY", None)
            else:
                os.environ["KAGGLE_KEY"] = prev_key

    def _download_single_url(self, url: str, file_path: Path) -> None:
        request_headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
        }
        is_kaggle_api_download = self._is_kaggle_api_download(url)
        request_auth: tuple[str, str] | None = None

        if is_kaggle_api_download:
            kaggle_creds = self._resolve_kaggle_credentials()
            if kaggle_creds is None:
                raise RuntimeError(
                    "Kaggle API download requires credentials. "
                    "Set either KAGGLE_USERNAME + KAGGLE_KEY, or "
                    "KAGGLE_USERNAME + KAGGLE_API_TOKEN (key-only), or "
                    "KAGGLE_API_TOKEN with format username:key/json in environment variables "
                    "or a discoverable .env file."
                )
            request_auth = kaggle_creds
            self._download_kaggle_dataset(url, request_auth)
            return

        should_fallback_to_playwright = False

        try:
            # First-pass plain HTTP download for direct file URLs.
            with requests.get(
                url,
                headers=request_headers,
                auth=request_auth,
                timeout=120,
                stream=True,
            ) as response:
                content_type = (response.headers.get("content-type") or "").lower()
                if response.ok and "text/html" not in content_type:
                    with file_path.open("wb") as f:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    total_size = file_path.stat().st_size
                    if total_size > 0:
                        logger.info(f"File size: {total_size / (1024 * 1024):.2f} MB")
                        logger.info(f"Downloaded to {file_path}")
                        return

                if is_kaggle_api_download:
                    error_preview = response.text[:300].replace("\n", " ")
                    raise RuntimeError(
                        "Kaggle API download failed. "
                        f"status={response.status_code}, content_type={content_type}, "
                        f"body_preview='{error_preview}'"
                    )

                should_fallback_to_playwright = True
                logger.info(
                    "Requests download returned HTML/empty content; falling back to Playwright",
                )
        except requests.RequestException as e:
            should_fallback_to_playwright = True
            logger.info(f"Requests download failed, falling back to Playwright: {e}")

        if is_kaggle_api_download:
            raise RuntimeError(
                "Kaggle API download failed via HTTP request. "
                "Playwright fallback is not supported for Kaggle API endpoints."
            )

        if not should_fallback_to_playwright:
            return

        async def download_async() -> None:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=request_headers["User-Agent"],
                    viewport={"width": 1920, "height": 1080},
                )
                page = await context.new_page()
                try:
                    # Fast path: works for direct file URLs and avoids waiting on page load events.
                    response = await context.request.get(url, timeout=120000)
                    content_type = (response.headers.get("content-type") or "").lower()
                    if response.ok and "text/html" not in content_type:
                        body = await response.body()
                        if body:
                            file_path.write_bytes(body)
                            total_size = file_path.stat().st_size
                            logger.info(
                                f"File size: {total_size / (1024 * 1024):.2f} MB"
                            )
                            logger.info(f"Downloaded to {file_path}")
                            return

                    logger.info(
                        "Direct request returned HTML/empty content; falling back to browser download flow"
                    )
                    async with page.expect_download(timeout=120000) as download_info:
                        await page.goto(
                            url,
                            wait_until="domcontentloaded",
                            timeout=120000,
                        )

                    download = await download_info.value
                    await download.save_as(file_path)
                    total_size = file_path.stat().st_size
                    logger.info(f"File size: {total_size / (1024 * 1024):.2f} MB")
                    logger.info(f"Downloaded to {file_path}")
                finally:
                    await context.close()
                    await browser.close()

        # Run in asyncio context
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(download_async())
            else:
                thread_error: list[BaseException] = []

                def _thread_runner() -> None:
                    try:
                        asyncio.run(download_async())
                    except BaseException as exc:  # noqa: BLE001
                        thread_error.append(exc)

                runner_thread = threading.Thread(target=_thread_runner, daemon=True)
                runner_thread.start()
                runner_thread.join()

                if thread_error:
                    raise thread_error[0]
        except Exception as e:
            logger.error(f"Async download failed: {e}")
            raise

    def compute_results(self, base: base_type) -> result_type:
        download_urls = self._normalize_download_urls()
        used_filenames: Set[str] = set()

        logger.info(f"Downloading {self.cfg.dataset_id} from {len(download_urls)} URL(s)")

        for index, url in enumerate(download_urls):
            file_path = self._build_download_path(url, index, used_filenames)
            logger.info(
                f"Downloading [{index + 1}/{len(download_urls)}]: {url} -> {file_path.name}"
            )
            self._download_single_url(url, file_path)

    def save_results(self, results: result_type) -> None:
        return None

    def load_results(self) -> result_type:
        return None
