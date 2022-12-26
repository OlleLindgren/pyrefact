#!/usr/bin/env python3

import sys
from pathlib import Path

import pyrefact

sys.path.append(str(Path(__file__).parent))
import testing_infra
from integration_test_cases import INTEGRATION_TEST_CASES


def _add_indent(content: str, indent: int) -> str:
    return "".join(f"{' ' * indent}{line}" for line in content.splitlines(keepends=True))


def main() -> int:
    for content, expected_abstraction in INTEGRATION_TEST_CASES:
        processed_content = pyrefact.format_code(content)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    for content, expected_abstraction in INTEGRATION_TEST_CASES:
        for indent in (0, 2, 4, 6, 8, 12, 16):
            indented_content = _add_indent(content, indent)
            indented_expected_abstraction = _add_indent(expected_abstraction, indent)
            processed_content = pyrefact.format_code(indented_content)
            if not testing_infra.check_fixes_equal(
                processed_content, indented_expected_abstraction
            ):
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
