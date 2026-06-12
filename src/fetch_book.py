import re
import sys
import urllib.request
from pathlib import Path

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOK_DIR = PROJECT_ROOT / "data/book"
BASE_URL = "https://web.stanford.edu/~jurafsky/slp3/{}.pdf"

CHAPTERS = list(range(2, 25))

HEADER_FOOTER = re.compile(r"^\s*\d+\s+CHAPTER\s+\d+", re.IGNORECASE)
PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")


def fetch_pdf(chapter: int) -> bytes | None:
    url = BASE_URL.format(chapter)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        return data if data[:4] == b"%PDF" else None
    except Exception as exc:
        print(f"  ch{chapter}: skip ({exc})")
        return None


def extract_text(data: bytes) -> str:
    tmp = BOOK_DIR / "_tmp.pdf"
    tmp.write_bytes(data)
    reader = PdfReader(str(tmp))
    pages = [page.extract_text() or "" for page in reader.pages]
    tmp.unlink()

    lines = []
    for page in pages:
        for line in page.splitlines():
            line = line.strip()
            if not line or PAGE_NUMBER.match(line) or HEADER_FOOTER.match(line):
                continue
            lines.append(line)
    text = " ".join(lines)
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> None:
    BOOK_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for chapter in CHAPTERS:
        data = fetch_pdf(chapter)
        if data is None:
            continue
        text = extract_text(data)
        if len(text.split()) < 1000:
            print(f"  ch{chapter}: too short, skip")
            continue
        out = BOOK_DIR / f"slp3_ch{chapter:02d}.txt"
        out.write_text(text, encoding="utf-8")
        print(f"  ch{chapter}: {len(text.split()):,} words -> {out.name}")
        saved += 1
    print(f"\nSaved {saved} chapters to {BOOK_DIR}")


if __name__ == "__main__":
    main()
