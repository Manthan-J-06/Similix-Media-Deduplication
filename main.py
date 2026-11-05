"""Entrypoint for running the SIMILIX server."""
import sys
from core.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("serve")
    main()
