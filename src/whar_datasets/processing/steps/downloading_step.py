import asyncio
import threading
from pathlib import Path
from typing import Any, Set, TypeAlias

import requests

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

    def compute_results(self, base: base_type) -> result_type:
        # Use filename to define file path
        filename = self.cfg.download_url.split("/")[-1]
        file_path = self.data_dir / filename

        # download file from url
        logger.info(f"Downloading {self.cfg.dataset_id}")
        request_headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
        }
        should_fallback_to_playwright = False

        try:
            # First-pass plain HTTP download for direct file URLs.
            with requests.get(
                self.cfg.download_url,
                headers=request_headers,
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
                should_fallback_to_playwright = True
                logger.info(
                    "Requests download returned HTML/empty content; falling back to Playwright",
                )
        except requests.RequestException as e:
            should_fallback_to_playwright = True
            logger.info(f"Requests download failed, falling back to Playwright: {e}")

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
                    response = await context.request.get(
                        self.cfg.download_url, timeout=120000
                    )
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
                            self.cfg.download_url,
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

    def save_results(self, results: result_type) -> None:
        return None

    def load_results(self) -> result_type:
        return None
