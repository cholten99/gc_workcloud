#!/usr/bin/env python3
"""
Create a word-cloud image from the contents of all PDFs under a docs folder,
excluding words listed in a separate text file read at runtime.

Requirements:
  pip install wordcloud matplotlib pypdf

Notes:
- This extracts *selectable text*. Scanned PDFs that are just images will likely
  produce little/no text unless you OCR them first.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set, List, Tuple

from pypdf import PdfReader
from wordcloud import WordCloud


LOG = logging.getLogger("wordcloud_pdf")


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")  # keep simple English-ish tokens


@dataclass
class Config:
    docs_dir: Path
    exclude_file: Path
    output: Path
    width: int
    height: int
    background: str
    max_words: int
    min_word_length: int
    recursive: bool


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def read_exclude_words(path: Path) -> Set[str]:
    """
    Reads exclude words from a text file.
    Supports:
      - one word per line
      - blank lines
      - comments starting with #
    Case-insensitive (everything lowercased).
    """
    if not path.exists():
        raise FileNotFoundError(f"Exclude words file not found: {path}")

    words: Set[str] = set()
    for i, raw in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        # Allow comma-separated too, just in case
        parts = [p.strip() for p in line.split(",") if p.strip()]
        for p in parts:
            words.add(p.lower())

    LOG.info("Loaded %d excluded word(s) from %s", len(words), path)
    return words


def iter_pdf_files(docs_dir: Path, recursive: bool) -> Iterable[Path]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs folder not found: {docs_dir}")
    if not docs_dir.is_dir():
        raise NotADirectoryError(f"Docs path is not a directory: {docs_dir}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    yield from docs_dir.glob(pattern)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF using pypdf.
    Returns a possibly-empty string (e.g., scanned PDFs).
    """
    try:
        reader = PdfReader(str(pdf_path))
        pieces: List[str] = []
        for idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                if text:
                    pieces.append(text)
            except Exception as e:
                LOG.warning("Failed extracting text from %s page %d: %s", pdf_path, idx, e)
        return "\n".join(pieces)
    except Exception as e:
        LOG.error("Failed reading PDF %s: %s", pdf_path, e)
        return ""


def tokenize(text: str, min_word_length: int) -> List[str]:
    # Lowercase and extract word-ish tokens
    tokens = [m.group(0).lower() for m in WORD_RE.finditer(text)]
    if min_word_length > 1:
        tokens = [t for t in tokens if len(t) >= min_word_length]
    return tokens


def build_corpus_tokens(pdf_paths: Iterable[Path], exclude: Set[str], min_word_length: int) -> Tuple[List[str], List[Path]]:
    all_tokens: List[str] = []
    empty_pdfs: List[Path] = []

    for pdf in pdf_paths:
        LOG.info("Reading %s", pdf)
        text = extract_text_from_pdf(pdf)
        if not text.strip():
            empty_pdfs.append(pdf)
            continue

        tokens = tokenize(text, min_word_length=min_word_length)
        if exclude:
            tokens = [t for t in tokens if t not in exclude]
        all_tokens.extend(tokens)

    return all_tokens, empty_pdfs


def generate_wordcloud(tokens: List[str], cfg: Config) -> WordCloud:
    if not tokens:
        raise ValueError("No tokens available to generate a word cloud (did the PDFs contain extractable text?).")

    text_for_wc = " ".join(tokens)

    wc = WordCloud(
        width=cfg.width,
        height=cfg.height,
        background_color=cfg.background,
        max_words=cfg.max_words,
        collocations=False,  # reduces weird "New_York"-style pairings
    )
    wc.generate(text_for_wc)
    return wc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a word-cloud PNG from all PDFs in a docs folder, excluding words listed in a text file."
    )
    p.add_argument(
        "--docs-dir",
        default="docs",
        help="Folder containing PDFs (default: docs)",
    )
    p.add_argument(
        "--exclude-file",
        required=True,
        help="Path to a text file listing words to exclude (one per line, or comma-separated).",
    )
    p.add_argument(
        "--output",
        default="wordcloud.png",
        help="Output image path (default: wordcloud.png)",
    )
    p.add_argument("--width", type=int, default=1800, help="Wordcloud image width in pixels (default: 1800)")
    p.add_argument("--height", type=int, default=1000, help="Wordcloud image height in pixels (default: 1000)")
    p.add_argument("--background", default="white", help="Background colour (default: white)")
    p.add_argument("--max-words", type=int, default=200, help="Max words in the cloud (default: 200)")
    p.add_argument("--min-word-length", type=int, default=3, help="Minimum word length to keep (default: 3)")
    p.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search subfolders of docs-dir for PDFs.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    cfg = Config(
        docs_dir=Path(args.docs_dir),
        exclude_file=Path(args.exclude_file),
        output=Path(args.output),
        width=args.width,
        height=args.height,
        background=args.background,
        max_words=args.max_words,
        min_word_length=args.min_word_length,
        recursive=not args.no_recursive,
    )

    try:
        exclude = read_exclude_words(cfg.exclude_file)
        pdfs = sorted(iter_pdf_files(cfg.docs_dir, recursive=cfg.recursive))
        if not pdfs:
            LOG.error("No PDFs found in %s (recursive=%s)", cfg.docs_dir, cfg.recursive)
            return 2

        LOG.info("Found %d PDF(s) under %s", len(pdfs), cfg.docs_dir)

        tokens, empty_pdfs = build_corpus_tokens(pdfs, exclude, cfg.min_word_length)
        LOG.info("Collected %d token(s) after exclusions", len(tokens))

        if empty_pdfs:
            LOG.warning(
                "%d PDF(s) produced no extractable text (possibly scanned images). Example: %s",
                len(empty_pdfs),
                empty_pdfs[0],
            )

        wc = generate_wordcloud(tokens, cfg)
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        wc.to_file(str(cfg.output))
        LOG.info("Saved word cloud to %s", cfg.output)
        return 0

    except FileNotFoundError as e:
        LOG.error("%s", e)
        return 2
    except Exception as e:
        LOG.exception("Unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

