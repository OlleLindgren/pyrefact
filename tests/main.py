#!/usr/bin/env python3
"""Maing script for running all tests."""
import itertools
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "unit"))
sys.path.append(str(Path(__file__).parent / "integration"))

import testing_infra


def main() -> int:
    """Run all scripts in the pyrefact/tests folder.

    Returns:
        int: 0 if successful, otherwise 1.
    """
    return_codes = {}
    unit_tests = testing_infra.iter_unit_tests()
    integration_tests = testing_infra.iter_integration_tests()

    for filename in itertools.chain(unit_tests, integration_tests):
        module = __import__(filename.stem)
        relpath = str(filename.absolute().relative_to(Path.cwd()))
        try:
            return_codes[relpath] = module.main()
        except Exception as error:
            return_codes[relpath] = error

    if not set(return_codes.values()) - {0}:
        print("PASSED")
        return 0

    print("Some tests failed")
    print(f"{'Test path':<50}   Return code")
    for (test, return_code) in return_codes.items():
        if return_code != 0:
            print(f"./{test:<50} {return_code}")
    print("FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
