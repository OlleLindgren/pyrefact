#!/usr/bin/env python3
"""Maing script for running all tests."""
import argparse
import itertools
import logging
import sys
import traceback
from pathlib import Path
from typing import Sequence

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "unit"))
sys.path.append(str(Path(__file__).parent / "integration"))

import testing_infra

from pyrefact import logs as logger

logger.set_level(logging.DEBUG)


def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(args)


def main(args: Sequence[str]) -> int:
    """Run all scripts in the pyrefact/tests folder.

    Returns:
        int: 0 if successful, otherwise 1.
    """
    args = _parse_args(args)
    return_codes = {}
    unit_tests = testing_infra.iter_unit_tests()
    integration_tests = testing_infra.iter_integration_tests()

    for filename in itertools.chain(unit_tests, integration_tests):
        module = __import__(filename.stem)
        relpath = str(filename.absolute().relative_to(Path.cwd()))
        try:
            return_codes[relpath] = module.main()
        except Exception as error:
            if args.verbose:
                return_codes[relpath] = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            else:
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
    sys.exit(main(sys.argv[1:]))
