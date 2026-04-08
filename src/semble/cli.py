from __future__ import annotations

import argparse
import sys
from typing import cast

from model2vec import StaticModel

from semble.types import Encoder


def main() -> None:
    """Run the command-line interface."""
    from semble import SearchMode
    from semble.index import DEFAULT_MODEL_NAME

    parser = argparse.ArgumentParser(
        prog="semble",
        description="Instant local code search for agents.",
    )
    sub = parser.add_subparsers(dest="command")

    # search
    srch = sub.add_parser("search", help="Search a directory")
    srch.add_argument("query", help="Search query")
    srch.add_argument("path", help="Directory to search in")
    srch.add_argument("-k", "--top-k", type=int, default=5)
    srch.add_argument(
        "-m",
        "--mode",
        default=SearchMode.HYBRID.value,
        choices=[mode.value for mode in SearchMode],
    )
    srch.add_argument("--model", default=DEFAULT_MODEL_NAME)
    srch.add_argument("--include-docs", action="store_true")

    args = parser.parse_args()

    if args.command == "search":
        from semble import SembleIndex

        index = SembleIndex(model=cast(Encoder, StaticModel.from_pretrained(args.model)))
        stats = index.index_directory(args.path, include_docs=args.include_docs)
        print(
            f"Indexed {stats.total_files} files → {stats.total_chunks} chunks"
            f" (embedding: {stats.embedding_time_ms:.0f}ms)"
        )

        results = index.search(args.query, top_k=args.top_k, mode=args.mode)

        if not results:
            print("No results found.")
            return

        for i, r in enumerate(results):
            print(f"\n#{i + 1} [{r.source.value}] score={r.score:.3f}")
            print(f"  {r.chunk.location}")
            lines = r.chunk.content.strip().splitlines()[:3]
            for line in lines:
                print(f"  | {line}")
            if len(r.chunk.content.strip().splitlines()) > 3:
                print(f"  | ... ({len(r.chunk.content.splitlines())} lines total)")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
