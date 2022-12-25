#!/usr/bin/env python3
import sys
import tempfile
from pathlib import Path

import pyrefact

sys.path.append(str(Path(__file__).parent))
import testing_infra
from integration_test_cases import INTEGRATION_TEST_CASES


def main() -> int:
    for content, expected_abstraction in INTEGRATION_TEST_CASES:
        with tempfile.NamedTemporaryFile() as temp:
            temp = temp.name
            with open(temp, "w", encoding="utf-8") as stream:
                stream.write(content)

            pyrefact.format_file(temp)

            with open(temp, "r", encoding="utf-8") as stream:
                processed_content = stream.read()

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
