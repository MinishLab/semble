"""CLI for semble."""

from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="semble",
        description="Instant local code search for agents.",
    )
    sub = parser.add_subparsers(dest="command")

    # index
    idx = sub.add_parser("index", help="Index a directory")
    idx.add_argument("path", help="Directory to index")
    idx.add_argument("--model", default="Pringled/potion-code-16M")
    idx.add_argument(
        "--include-docs", action="store_true", help="Also index docs (md, yaml, toml, json)"
    )

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

    if args.command == "index":
        from semble import SembleIndex

        index = SembleIndex(model_name=args.model)
        t0 = time.perf_counter()
        stats = index.index_directory(args.path, include_docs=args.include_docs)
        elapsed = (time.perf_counter() - t0) * 1000
        print(
            f"Indexed {stats.total_files} files → {stats.total_chunks} chunks"
            f" ({stats.total_symbols} symbols)"
        )
        print(f"Time: {elapsed:.0f}ms (embedding: {stats.embedding_time_ms:.0f}ms)")
        print(f"Languages: {stats.languages}")

    elif args.command == "search":
        from semble import SembleIndex

        index = SembleIndex(model_name=args.model)
        index.index_directory(args.path, include_docs=args.include_docs)
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
