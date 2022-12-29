#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
sys.path.append(os.getcwd())
            """,
            """
import os
import sys

sys.path.append(os.getcwd())
            """,
        ),
        (
            """
functools.reduce(lambda x: x+y, [1, 2, 3])
            """,
            """
import functools

functools.reduce(lambda x: x+y, [1, 2, 3])
            """,
        ),
        (
            """
import scipy.stats

a, b = 1.25, 0.5
mean, var, skew, kurt = scipy.stats.norminvgauss.stats(a, b, moments='mvsk')
            """,
            """
import scipy.stats

a, b = 1.25, 0.5
mean, var, skew, kurt = scipy.stats.norminvgauss.stats(a, b, moments='mvsk')
            """,
        ),
        (
            """
print(wierdo_library.strange_function())
            """,
            """
print(wierdo_library.strange_function())
            """,
        ),
        (
            """
x = np.array()
z = pd.DataFrame()
            """,
            """
import numpy as np
import pandas as pd


x = np.array()
z = pd.DataFrame()
            """,
        ),
        (
            """
w = numpy.zeros(10, dtype=numpy.float32)
            """,
            """
import numpy

w = numpy.zeros(10, dtype=numpy.float32)
            """,
        ),
    )

    for content, expected_abstraction in test_cases:
        processed_content = fixes.add_missing_imports(content)
        processed_content = fixes.fix_isort(processed_content)  # Or the order will be random
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
