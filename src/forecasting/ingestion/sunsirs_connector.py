"""Viscose Staple Fiber (VSF) data connector from SunSirs.

Source page (human-readable): https://www.sunsirs.com/uk/prodetail-1057.html
Commodity: China VSF spot prices (RMB/ton)
Frequency: Daily

Important constraints:
- No synthetic/sample data generation.
- No artificial gap filling or backfilling.
- If scraping fails (no internet, blocked, or site layout changes), raise an error.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional
from urllib.parse import urljoin

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class SunSirsConnector:
    """Fetch Viscose Staple Fiber (VSF) daily prices from SunSirs.
    
    Attributes:
        commodity_id: SunSirs commodity ID (1057 for VSF)
        max_retries: Number of retries on network failure
        timeout_sec: HTTP request timeout in seconds
        verify_ssl: Whether to verify SSL certificates
    """

    commodity_id: int = 1057  # VSF commodity ID
    max_retries: int = 3
    timeout_sec: int = 15
    verify_ssl: bool = True
    base_url: str = "https://www.sunsirs.com"

    _DATE_RE: re.Pattern = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
    _NUM_RE: re.Pattern = re.compile(r"[-+]?\d+(?:\.\d+)?")

    @property
    def prodetail_url(self) -> str:
        return f"{self.base_url}/uk/prodetail-{self.commodity_id}.html"

    def __post_init__(self):
        """Validate configuration."""
        if self.commodity_id != 1057:
            logger.warning(
                f"Non-standard commodity_id {self.commodity_id} may not be VSF; "
                "1057 is China VSF spot benchmark"
            )

    def fetch_daily_prices(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_pages: int = 50,
    ) -> pd.DataFrame:
        """Fetch daily VSF spot prices from SunSirs (real data only).
        
        Args:
            start_date: ISO format date (YYYY-MM-DD). Default: 1 year ago.
            end_date: ISO format date (YYYY-MM-DD). Default: today.
            max_pages: Max number of HTML pages to try when paging through history.
        
        Returns:
            DataFrame with columns:
                - timestamp: datetime64[ns] (date)
                - price_rmb: float (RMB/ton)
        
        Raises:
            ConnectionError: If unable to reach SunSirs after max_retries
            ValueError: If response format is unexpected
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(
            "Fetching VSF daily prices from SunSirs "
            f"(commodity_id={self.commodity_id}) for {start_date} to {end_date}"
        )

        start_dt = pd.to_datetime(start_date).normalize()
        end_dt = pd.to_datetime(end_date).normalize()

        all_records: list[dict] = []
        seen_dates: set[str] = set()
        consecutive_no_new = 0

        for page in range(1, max_pages + 1):
            url = self._build_page_url(page)
            logger.debug(f"Fetching SunSirs page {page}: {url}")

            html = self._fetch_html(url)
            page_records = list(self._parse_prodetail_html(html))
            if not page_records:
                # No table found / layout changed / blocked
                if page == 1:
                    raise ValueError(
                        "SunSirs page fetched but no (date, price) rows were parsed. "
                        "Site layout may have changed or scraping is blocked."
                    )
                break

            new_count = 0
            for rec in page_records:
                if rec["timestamp"] not in seen_dates:
                    seen_dates.add(rec["timestamp"])
                    all_records.append(rec)
                    new_count += 1

            if new_count == 0:
                consecutive_no_new += 1
            else:
                consecutive_no_new = 0

            # Heuristic: if we see no new dates twice, stop paging.
            if consecutive_no_new >= 2:
                break

        if not all_records:
            raise ValueError(
                "No VSF price rows parsed from SunSirs. "
                "This environment may not have internet access, or the site is blocking scraping."
            )

        df = pd.DataFrame(all_records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["price_rmb"] = pd.to_numeric(df["price_rmb"], errors="coerce")
        df = df.dropna(subset=["timestamp", "price_rmb"]).copy()
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "Parsed VSF prices, but none fell within the requested date range "
                f"({start_date} to {end_date})."
            )

        logger.info(
            f"Fetched {len(df)} VSF daily records ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})"
        )
        return df

    def _build_page_url(self, page: int) -> str:
        """Build a URL for paging through the prodetail history.

        SunSirs may ignore unknown paging parameters; we detect progress by whether
        new (date, price) rows appear.
        """
        if page <= 1:
            return self.prodetail_url
        return f"{self.prodetail_url}?page={page}"

    def _fetch_html(self, url: str) -> str:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
        }

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    timeout=self.timeout_sec,
                    verify=self.verify_ssl,
                )
                resp.raise_for_status()
                return resp.text
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                logger.warning(f"SunSirs request failed (attempt {attempt}/{self.max_retries}): {exc}")
        raise ConnectionError(f"Failed to fetch SunSirs page after {self.max_retries} attempts: {last_exc}")

    def _parse_prodetail_html(self, html: str) -> Iterable[dict]:
        """Parse (date, price) rows from the prodetail HTML.

        The implementation is intentionally layout-tolerant:
        - scans all table rows
        - extracts the first YYYY-MM-DD as the date
        - extracts the first plausible numeric value as the price
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "beautifulsoup4 is required for SunSirs HTML parsing. "
                "Install with: pip install beautifulsoup4"
            ) from exc

        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return []

        records: list[dict] = []
        for table in tables:
            for row in table.find_all("tr"):
                cells = row.find_all(["td", "th"])
                if not cells:
                    continue
                cell_texts = [c.get_text(" ", strip=True) for c in cells]
                date_str = self._extract_date(cell_texts)
                if not date_str:
                    continue
                price = self._extract_price(cell_texts)
                if price is None:
                    continue
                records.append({"timestamp": date_str, "price_rmb": float(price), "source_url": self.prodetail_url})

        return records

    def _extract_date(self, texts: list[str]) -> Optional[str]:
        for text in texts:
            m = self._DATE_RE.search(text)
            if m:
                return m.group(0)
        return None

    def _extract_price(self, texts: list[str]) -> Optional[float]:
        # Exclude the date cell content when scanning for numeric values.
        for text in texts:
            if self._DATE_RE.search(text):
                continue
            normalized = text.replace(",", " ")
            m = self._NUM_RE.search(normalized)
            if not m:
                continue
            try:
                value = float(m.group(0))
            except ValueError:
                continue
            # Heuristic bounds for RMB/ton (avoid parsing unrelated numbers)
            if 100.0 <= value <= 200000.0:
                return value
        return None

    def save_daily_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """Save daily prices to CSV (real data only).

        Output schema:
        - timestamp (YYYY-MM-DD)
        - price_rmb (float)
        """
        df_out = df[["timestamp", "price_rmb"]].copy()
        df_out.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_out)} daily records to {output_path}")
