#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
import numpy
    """,
        """
    """,
        ),
        (
            """
import numpy as np, pandas as pd
print(pd)
    """,
        """
import pandas as pd
print(pd)
    """,
        ),
        (
            """
from . import a, c
c(2)
    """,
        """
from . import c
c(2)
    """,
        ),
        (
            """
from ... import a, c
c(2)
    """,
        """
from ... import c
c(2)
    """,
        ),
        (
            """
from ....af.qwerty import a, b, c as d, q, w as f
print(a, b, d)
    """,
        """
from ....af.qwerty import a, b, c as d
print(a, b, d)
    """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = fixes.remove_unused_imports(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
