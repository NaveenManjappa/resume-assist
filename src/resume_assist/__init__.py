import argparse
from pathlib import Path

from resume_assist.ingest import draft_markdown, extract_pages

# The draft is unredacted, so it goes in the gitignored private folder.
DEFAULT_DRAFT_PATH = Path("data/private/resume.draft.md")


def main() -> None:
    parser = argparse.ArgumentParser(prog="resume-assist")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="turn the resume PDF into draft Markdown")
    ingest_parser.add_argument("pdf", type=Path, help="path to the resume PDF")
    ingest_parser.add_argument("-o", "--output", type=Path, default=DEFAULT_DRAFT_PATH)

    args = parser.parse_args()
    if args.command == "ingest":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(draft_markdown(extract_pages(args.pdf)), encoding="utf-8")
        print(f"Wrote draft Markdown to {args.output}")
