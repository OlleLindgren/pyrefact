#!/usr/bin/env python3
"""Maing script for running all tests."""
import sys
from pathlib import Path


def main() -> int:
    """Run all scripts in the pyrefact/tests folder.

    Returns:
        int: Always returns 0. If unsuccessful, raises exceptions.
    """
    return_codes = {}
    for filename in Path(__file__).parent.rglob("*.py"):
        if filename.name in {
            "testing_infra.py",
            "integration_test_cases.py",
            Path(__file__).name,
            "main_profile.py",
        }:
            continue
        module = __import__(filename.stem)
        relpath = str(filename.absolute().relative_to(Path.cwd()))
        try:
            return_codes[relpath] = module.main()
        except Exception as error:
            return_codes[relpath] = error

    if set(return_codes.values()) - {0}:
        print("Some tests failed")
        print(f"{'Test path':<50}   Return code")
        for test, return_code in return_codes.items():
            if return_code != 0:
                print(f"./{test:<50} {return_code}")
        print("FAILED")
        return 1

    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
