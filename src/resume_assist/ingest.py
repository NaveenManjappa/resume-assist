"""Turn the resume PDF into draft Markdown, ready to be corrected by hand."""

import re
from pathlib import Path

from pypdf import PdfReader

SECTION_TITLE = re.compile(r"^[A-Z][A-Z ]+$")
DATE_RANGE = re.compile(r"^\d{2}/\d{4} - (\d{2}/\d{4}|Present)$")
BULLET = "•"


def extract_pages(pdf_path: Path) -> list[str]:
    """Extract the text of each page of a PDF."""
    return [page.extract_text() for page in PdfReader(pdf_path).pages]


def draft_markdown(pages: list[str]) -> str:
    """Convert the text of each PDF page into a single draft Markdown document."""
    blocks: list[str] = []
    seen_sections: set[str] = set()
    in_list = False
    for line in "\n".join(pages).splitlines():
        line = line.replace("\xa0", " ").strip()
        if not re.search(r"\w", line):
            continue  # blank, or a page-break artefact such as " / "
        if not blocks:
            blocks.append(f"# {line}")
            in_list = False
        elif SECTION_TITLE.match(line):
            if line not in seen_sections:  # later pages repeat the title
                blocks.append(f"## {line}")
                seen_sections.add(line)
            in_list = False
        elif line.startswith(BULLET):
            item = "- " + line.removeprefix(BULLET).strip()
            if in_list:
                blocks[-1] += "\n" + item
            else:
                blocks.append(item)
            in_list = True
        elif in_list and not DATE_RANGE.match(line):
            blocks[-1] = _join_wrapped(blocks[-1], line)
        else:
            blocks.append(line)
            in_list = False
    return "\n\n".join(blocks) + "\n"


def _join_wrapped(text: str, continuation: str) -> str:
    """Rejoin a line that the PDF wrapped, keeping hyphenated words whole."""
    if text.endswith("-"):
        return text + continuation
    return f"{text} {continuation}"
