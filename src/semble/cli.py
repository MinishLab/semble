import argparse
import asyncio

from semble.mcp import serve


def main() -> None:
    """Entry point for the semble command-line tool."""
    parser = argparse.ArgumentParser(
        prog="semble",
        description="Instant local code memory for agents.",
    )
    parser.add_argument("path", help="Directory to index and serve.")
    args = parser.parse_args()

    asyncio.run(serve(args.path))


if __name__ == "__main__":
    main()
