#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import format_code

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (  # No ignore comment
            """
print(0)
if False:
    print(1)
        """,
            """
print(0)
        """,
        ),
        (  # Valid ignore comment
            """
print(0)
if False:  # pyrefact: ignore
    print(1)
        """,
            """
print(0)
if False:  # pyrefact: ignore
    print(1)
        """,
        ),
        (  # Valid ignore comment
            """
print(0)
if False:  # pyrefact: skip_file
    print(1)
        """,
            """
print(0)
if False:  # pyrefact: skip_file
    print(1)
        """,
        ),
        (  # Unrelated ignore comment
            """
print(0)
if False:  # type: ignore
    print(1)
        """,
            """
print(0)
        """,
        ),
        (  # Invalid ignore comment
            """
print(0)
if False:  # pyrefact: asdfdsas
    print(1)
        """,
            """
print(0)
        """,
        ),
        (  # Valid ignore comment with extra spaces
            """
print(0)
if False:#            pyrefact      :ignore   
    print(1)
        """,
            """
print(0)
if False:#            pyrefact      :ignore   
    print(1)
        """,
        ),
        (  # Valid ignore comment with extra spaces
            """
print(0)
if False:#            pyrefact      :skip_file   
    print(1)
        """,
            """
print(0)
if False:#            pyrefact      :skip_file   
    print(1)
        """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = format_code(source)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
