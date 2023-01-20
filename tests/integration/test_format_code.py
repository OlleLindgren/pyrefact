#!/usr/bin/env python3

import sys
from pathlib import Path

import pyrefact

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra
from integration_test_cases import INTEGRATION_TEST_CASES


def main() -> int:
    for source, expected_abstraction in INTEGRATION_TEST_CASES:
        processed_content = pyrefact.format_code(source)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
