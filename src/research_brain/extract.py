"""Turn source files into plain text and split long texts into chunks."""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".tex", ".rst", ".org", ".bib"}


def read_source(path: Path, use_marker: bool = False) -> tuple[str, str]:
    """Return (text, kind) for a Markdown, text, LaTeX, HTML or PDF file.

    PDFs use pypdf by default. Equations and tables survive much better with Marker
    (``pip install marker-pdf``); pass ``use_marker=True`` to use it when installed.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="replace"), suffix.lstrip(".")
    if suffix in {".html", ".htm"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
        raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
        text = html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))
        return re.sub(r"[ \t]+", " ", text), "html"
    if suffix == ".pdf":
        if use_marker and shutil.which("marker_single"):
            return _marker(path), "pdf"
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        # "## Page N" headings let RAG citations link straight to the page: [[source#Page 7]]
        pages = [f"## Page {i}\n\n{(p.extract_text() or '').strip()}" for i, p in enumerate(reader.pages, 1)]
        return "\n\n".join(pages), "pdf"
    raise ValueError(f"Unsupported file type: {suffix}. Convert it to Markdown or PDF first.")


def _marker(path: Path) -> str:
    with tempfile.TemporaryDirectory() as out:
        subprocess.run(["marker_single", str(path), "--output_dir", out], check=True,
                       capture_output=True)
        mds = sorted(Path(out).rglob("*.md"))
        if not mds:
            raise RuntimeError("Marker produced no Markdown output.")
        return mds[0].read_text(encoding="utf-8")


def guess_title(text: str, fallback: str) -> str:
    for line in text.splitlines()[:40]:
        m = re.match(r"^#\s+(.+)", line.strip())
        if m:
            return m.group(1).strip()
    return fallback.replace("_", " ").replace("-", " ").strip().title()


def chunk_text(text: str, size: int = 6000) -> list[str]:
    """Split on paragraph boundaries into chunks of at most ~``size`` characters."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf, length = [], [], 0
    for para in paragraphs:
        if length + len(para) > size and buf:
            chunks.append("\n\n".join(buf))
            buf, length = [], 0
        while len(para) > size:  # very long paragraph: hard split
            chunks.append(para[:size])
            para = para[size:]
        buf.append(para)
        length += len(para) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def head_and_tail(text: str, limit: int) -> str:
    """Keep the start (abstract, introduction) and the end (conclusions) of a long text."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.65)
    return text[:head] + "\n\n[... middle of the source omitted for length ...]\n\n" + text[-(limit - head):]
