import argparse
from pathlib import Path

from dotenv import load_dotenv

from resume_assist.ingest import draft_markdown, extract_pages
from resume_assist.openai_clients import OpenAIEmbedder
from resume_assist.rag import build_index, save_index

# The draft is unredacted, so it goes in the gitignored private folder.
DEFAULT_DRAFT_PATH = Path("data/private/resume.draft.md")
DEFAULT_PUBLIC_RESUME = Path("data/public/resume.md")
DEFAULT_PUBLIC_INDEX = Path("data/public/index")


def main() -> None:
    parser = argparse.ArgumentParser(prog="resume-assist")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="turn the resume PDF into draft Markdown")
    ingest_parser.add_argument("pdf", type=Path, help="path to the resume PDF")
    ingest_parser.add_argument("-o", "--output", type=Path, default=DEFAULT_DRAFT_PATH)

    build_parser = commands.add_parser("build-index", help="embed the public resume into a saved index")
    build_parser.add_argument("--resume", type=Path, default=DEFAULT_PUBLIC_RESUME)
    build_parser.add_argument("-o", "--output", type=Path, default=DEFAULT_PUBLIC_INDEX)

    args = parser.parse_args()
    if args.command == "ingest":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(draft_markdown(extract_pages(args.pdf)), encoding="utf-8")
        print(f"Wrote draft Markdown to {args.output}")
    elif args.command == "build-index":
        load_dotenv()
        index = build_index(args.resume.read_text(encoding="utf-8"), OpenAIEmbedder())
        save_index(index, args.output)
        print(f"Wrote {len(index.chunks)} chunks ({index.embedding_model}) to {args.output}")
