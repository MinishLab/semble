import argparse
import asyncio
import sys


def main() -> None:
    """Entry point for the semble command-line tool."""
    parser = argparse.ArgumentParser(
        prog="semble",
        description="Instant local code search for agents.",
    )
    parser.add_argument("path", help="Directory to index and serve.")
    args = parser.parse_args()

    try:
        from semble.mcp import serve
    except ImportError:
        print(
            'MCP support requires the mcp extra: pip install "semble[mcp]"',
            file=sys.stderr,
        )
        sys.exit(1)

    asyncio.run(serve(args.path))


if __name__ == "__main__":
    main()
