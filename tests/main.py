#!/usr/bin/env python3
"""Maing script for running all tests."""
import sys
from pathlib import Path


def main() -> int:
    """Run all scripts in the pyrefact/tests folder.

    Returns:
        int: Always returns 0. If unsuccessful, raises exceptions.
    """
    for filename in Path(__file__).parent.rglob("*.py"):
        if filename.name in {
            "testing_infra.py",
            "integration_test_cases.py",
            Path(__file__).name,
            "main_profile.py",
        }:
            continue
        module = __import__(filename.stem)
        if module.main() == 0:
            print("PASSED")
        else:
            print("FAILED")
            return 1

    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
