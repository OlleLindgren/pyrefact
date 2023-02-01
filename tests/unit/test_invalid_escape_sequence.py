#!/usr/bin/env python3

import logging
import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (
            r"""
import re
print(re.findall("\d+", "1234x23"))
            """,
            r"""
import re
print(re.findall(r"\d+", "1234x23"))
            """,
        ),
        (
            r"""
import re
print(re.findall("\+", "1234+23"))
            """,
            r"""
import re
print(re.findall(r"\+", "1234+23"))
            """,
        ),
        (  # Watch out with f strings
            r"""
import re
print(re.findall(f"\d{'+'}", "1234x23"))
            """,
            r"""
import re
print(re.findall(f"\d{'+'}", "1234x23"))
            """,
        ),
    )

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    for source, expected_abstraction in test_cases:
        processed_content = fixes.invalid_escape_sequence(source)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
