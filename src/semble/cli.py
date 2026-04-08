from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Run the command-line interface."""
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
        default="hybrid",
        choices=["hybrid", "semantic", "bm25", "symbol"],
    )
    srch.add_argument("--model", default="Pringled/potion-code-16M")
    srch.add_argument("--include-docs", action="store_true")

    args = parser.parse_args()

    if args.command == "search":
        from semble import SembleIndex

        index = SembleIndex(model=args.model)
        stats = index.index_directory(args.path, include_docs=args.include_docs)
        print(
            f"Indexed {stats.total_files} files → {stats.total_chunks} chunks"
            f" ({stats.index_time_ms:.0f}ms, embedding: {stats.embedding_time_ms:.0f}ms)"
        )

        results = index.search(args.query, top_k=args.top_k, mode=args.mode)

        if not results:
            print("No results found.")
            return

        for i, r in enumerate(results):
            print(f"\n#{i + 1} [{r.source}] score={r.score:.3f}")
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
