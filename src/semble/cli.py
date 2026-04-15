import argparse
import asyncio
import sys


def main() -> None:
    """Entry point for the semble command-line tool."""
    parser = argparse.ArgumentParser(
        prog="semble",
        description="Instant local code search for agents.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Local directory or git URL to pre-index at startup (optional).",
    )
    parser.add_argument("--ref", default=None, help="Branch or tag to check out (git URLs only).")
    args = parser.parse_args()

    try:
        from semble.mcp import serve
    except ImportError:
        print(
            'MCP support requires the mcp extra: pip install "semble[mcp]"',
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(serve(args.path, ref=args.ref))


if __name__ == "__main__":
    main()
