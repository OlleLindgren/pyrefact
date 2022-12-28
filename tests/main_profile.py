#!/usr/bin/env python3
"""Maing script for running all tests."""
import io
import sys
import timeit


def main() -> int:
    """Run all scripts in the pyrefact/tests folder.

    Returns:
        int: Always returns 0.
    """
    stdout = io.StringIO()
    original_sys_stdout = sys.stdout
    try:
        sys.stdout = stdout
        dt = timeit.timeit("main.main()", "import main", number=10)
    finally:
        sys.stdout = original_sys_stdout
    print(f"Completed in {dt:.3f} seconds")

    return 0


if __name__ == "__main__":
    sys.exit(main())
