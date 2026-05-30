"""
commodity_mapper.py
-------------------
Item code prefix → material category mapping for Oracle inventory and
consumption data.

Mapping priority (highest to lowest):
    1. External CSV file  (commodity_mapping.csv)  — runtime-configurable
    2. Built-in PREFIX_RULES                        — ships with the codebase
    3. Description regex fallbacks                  — last resort only

Design principles:
- CommodityMapper is the primary entry point; it holds one prefix map.
- Built-in rules are the default so the system works out-of-box with no CSV.
- Swapping mappings for a new mill/product = one CSV edit, zero code changes.
- Description fallbacks fire ONLY when both prefix maps yield no match.
- All classification logic is pure (no I/O inside classify()).

CSV format (commodity_mapping.csv):
    item_code_prefix,category
    COT,Cotton
    ICOT,Cotton
    PSF,Fiber
    FIB,Fiber
    STF,Stretch Fiber
    VIS,Viscose
    VSF,Viscose
    CW,Cotton Waste

    Rules:
    - item_code_prefix is matched against the FIRST segment of the item code
      (everything before the first "-").  "COT" matches "COT-2024-001".
    - Rows are matched in file order; first match wins.
    - Lines starting with "#" are treated as comments and ignored.
    - Header row (item_code_prefix,category) is auto-detected and skipped.

Usage:
    # Default — uses built-in rules
    mapper = CommodityMapper.default()
    mapper.classify("COT-2024-001")           # → "Cotton"

    # From CSV — fully overrides built-in prefix rules
    mapper = CommodityMapper.from_csv("scripts/commodity_mapping.csv")
    mapper.classify("PSF-ABC", "Polyester Staple")  # → "Fiber"

    # Module-level shortcut (uses built-in rules, no CSV)
    from commodity_mapper import classify_item_code
    classify_item_code("VIS-001")             # → "Viscose"
"""

from __future__ import annotations

import csv
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Built-in prefix rules (fallback when no CSV is provided)
# ---------------------------------------------------------------------------
# Keys are UPPERCASE prefix segments (no dash needed — matching is done on the
# first "-"-delimited segment of the item code).
# Longer/more-specific keys should appear first to ensure correct priority when
# one prefix is a substring of another (e.g. "ICOT" before "COT").

DEFAULT_PREFIX_RULES: OrderedDict[str, str] = OrderedDict([
    ("ICOT",  "Cotton"),        # Imported cotton — must precede COT
    ("COT",   "Cotton"),        # Local / generic cotton
    ("PSF",   "Fiber"),         # Polyester staple fiber
    ("FIB",   "Fiber"),         # Generic fiber
    ("STF",   "Stretch Fiber"), # Stretch / elastane blends
    ("VIS",   "Viscose"),       # Viscose / rayon
    ("VSF",   "Viscose"),       # Viscose staple fiber
    ("CW",    "Cotton Waste"),  # Cotton waste / by-product
    ("FW",    "Fiber Waste"),   # Fiber waste
    ("CHM",   "Chemicals"),
    ("PKG",   "Packaging"),
])


# ---------------------------------------------------------------------------
# Description-based fallback rules
# ---------------------------------------------------------------------------
# These fire ONLY when neither the built-in rules nor the CSV produce a match.
# Patterns are evaluated against the lowercased item description.

_DESCRIPTION_FALLBACKS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bicot\b|\bimported\s+cotton\b"),              "Cotton"),
    (re.compile(r"\bcot(ton)?\b|\braw\s*cot\b"),                 "Cotton"),
    (re.compile(r"\bvisco(se)?\b|\brayon\b|\bvsf\b"),            "Viscose"),
    (re.compile(r"\bpsf\b|\bpoly(ester)?\s*(staple)?\b"),        "Fiber"),
    (re.compile(r"\bfib(re|er)\b"),                              "Fiber"),
    (re.compile(r"\bstretch\b|\belast(ane|ic)?\b|\bspandex\b"),  "Stretch Fiber"),
    (re.compile(r"\bwaste\b.*\bcot\b|\bcot\b.*\bwaste\b"),       "Cotton Waste"),
    (re.compile(r"\bwaste\b"),                                   "Fiber Waste"),
]

UNMAPPED = "Unmapped"


# ---------------------------------------------------------------------------
# CommodityMapper class
# ---------------------------------------------------------------------------

class CommodityMapper:
    """Classifies Oracle item codes into material categories.

    Instantiate via CommodityMapper.default() or CommodityMapper.from_csv().
    """

    def __init__(self, prefix_map: OrderedDict[str, str]):
        # Normalise all keys to uppercase for case-insensitive matching
        self._prefix_map: OrderedDict[str, str] = OrderedDict(
            (k.strip().upper(), v.strip()) for k, v in prefix_map.items()
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "CommodityMapper":
        """Return a mapper using the built-in DEFAULT_PREFIX_RULES."""
        return cls(OrderedDict(DEFAULT_PREFIX_RULES))

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "CommodityMapper":
        """Load prefix→category mappings from an external CSV file.

        The CSV must have columns: item_code_prefix, category
        (header row is auto-detected; comments starting with # are skipped).
        File order is preserved — first match wins at classification time.

        Raises:
            FileNotFoundError: if csv_path does not exist.
            ValueError: if the CSV has no recognisable prefix/category columns.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Mapping CSV not found: {csv_path}\n"
                "Create it with columns: item_code_prefix, category"
            )

        prefix_map: OrderedDict[str, str] = OrderedDict()

        with csv_path.open(newline="", encoding="utf-8-sig") as fh:
            # Sniff delimiter (handles comma and tab exports)
            sample = fh.read(4096)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            except csv.Error:
                dialect = csv.excel

            reader = csv.reader(fh, dialect)
            prefix_col: Optional[int] = None
            category_col: Optional[int] = None

            for row in reader:
                if not row or all(c.strip() == "" for c in row):
                    continue
                if row[0].strip().startswith("#"):
                    continue

                # Detect header row by column name presence
                if prefix_col is None:
                    normalised = [c.strip().lower().replace(" ", "_") for c in row]
                    for i, name in enumerate(normalised):
                        if "prefix" in name or name in ("item_code_prefix", "prefix", "code"):
                            prefix_col = i
                        if "category" in name or name in ("category", "cat", "type"):
                            category_col = i
                    if prefix_col is not None and category_col is not None:
                        continue  # consumed the header row, move to data

                    # If first row has no header keywords, assume col 0 = prefix, col 1 = category
                    if len(row) >= 2:
                        prefix_col = 0
                        category_col = 1
                        # This row IS data, not a header — fall through to store it

                if prefix_col is None or category_col is None:
                    continue

                try:
                    prefix = row[prefix_col].strip().upper()
                    category = row[category_col].strip()
                except IndexError:
                    continue

                if prefix and category:
                    prefix_map[prefix] = category

        if not prefix_map:
            raise ValueError(
                f"No valid mappings found in {csv_path}. "
                "Ensure it has columns: item_code_prefix, category"
            )

        return cls(prefix_map)

    @classmethod
    def from_csv_or_default(cls, csv_path: Optional[str | Path]) -> "CommodityMapper":
        """Load from CSV if path given and file exists; otherwise use built-in rules.

        This is the recommended constructor for production scripts — it lets callers
        pass --mapping-csv as an optional argument without conditional logic.
        """
        if csv_path is not None:
            p = Path(csv_path)
            if p.exists():
                return cls.from_csv(p)
            # File path was provided but file is missing — warn rather than crash
            import warnings
            warnings.warn(
                f"Mapping CSV not found at '{p}'; falling back to built-in rules.",
                stacklevel=2,
            )
        return cls.default()

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, item_code: object, description: object = None) -> str:
        """Return the material category for an Oracle item code.

        Matching order:
            1. First segment of item_code (before the first "-") against prefix_map.
            2. Full item_code string against prefix_map (covers codes without dashes).
            3. Description regex fallbacks (only when steps 1–2 both fail).

        Args:
            item_code:   Oracle item code string, e.g. "COT-2024-001".
            description: Optional description used only as a final fallback.

        Returns:
            Matched category string, or UNMAPPED.
        """
        code_str = _clean_str(item_code).upper()

        # Step 1: match on the leading segment before first "-"
        segment = code_str.split("-")[0] if "-" in code_str else code_str
        if segment and segment in self._prefix_map:
            return self._prefix_map[segment]

        # Step 2: full code against map (handles codes without dashes)
        for prefix, category in self._prefix_map.items():
            if code_str.startswith(prefix):
                return category

        # Step 3: description regex — only fires when prefix map gives nothing
        if description is not None:
            desc_str = _clean_str(description).lower()
            for pattern, category in _DESCRIPTION_FALLBACKS:
                if pattern.search(desc_str):
                    return category

        return UNMAPPED

    def list_categories(self) -> list[str]:
        """Return all category names known to this mapper (deduplicated)."""
        seen: list[str] = []
        for cat in self._prefix_map.values():
            if cat not in seen:
                seen.append(cat)
        if UNMAPPED not in seen:
            seen.append(UNMAPPED)
        return seen

    def prefix_count(self) -> int:
        """Number of prefix rules currently loaded."""
        return len(self._prefix_map)

    def __repr__(self) -> str:
        return f"CommodityMapper({self.prefix_count()} rules, categories={self.list_categories()})"


# ---------------------------------------------------------------------------
# Module-level convenience API (backward compatible)
# ---------------------------------------------------------------------------

# Default instance — uses built-in rules; replaced by load_default_from_csv()
_default_mapper: CommodityMapper = CommodityMapper.default()


def classify_item_code(item_code: object, description: object = None) -> str:
    """Module-level shortcut using the default built-in mapper.

    For CSV-driven classification, instantiate CommodityMapper.from_csv() directly
    and call mapper.classify() instead of this function.
    """
    return _default_mapper.classify(item_code, description)


def list_categories() -> list[str]:
    """Return all category names from the built-in default mapper."""
    return _default_mapper.list_categories()


def load_csv_mappings(csv_path: str | Path) -> OrderedDict[str, str]:
    """Load a commodity_mapping.csv and return the raw prefix→category dict.

    Useful for inspection or building a custom CommodityMapper:
        mapping = load_csv_mappings("scripts/commodity_mapping.csv")
        mapper  = CommodityMapper(mapping)
    """
    mapper = CommodityMapper.from_csv(csv_path)
    return OrderedDict(mapper._prefix_map)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_str(value: object) -> str:
    """Coerce any value to a stripped string; None/NaN → empty string."""
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return str(value).strip()
