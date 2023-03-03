#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import performance_pandas

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (
            """
for _, x in df.iterrows():
    print(x["value"])
            """,
            """
for x in df.itertuples():
    print(x.value)
            """,
        ),
        (
            """
for _, x in df.iterrows():
    print(x.at["value"])
            """,
            """
for x in df.itertuples():
    print(x.value)
            """,
        ),
        (
            """
for _, x in df.iterrows():
    print(x.iat[9])
            """,
            """
for x in df.itertuples():
    print(x[9 + 1])
            """,
        ),
        (
            """
for _, x in df.iterrows():
    print(x.iat[9 + value - 1])
            """,
            """
for x in df.itertuples():
    print(x[9 + value - 1 + 1])
            """,
        ),
        (
            """
for _, x in df.iterrows():
    print(x)
            """,
            """
for _, x in df.iterrows():
    print(x)
            """,
        ),
        (
            """
for i, x in df.iterrows():
    print(x["value"])
            """,
            """
for i, x in df.iterrows():
    print(x["value"])
            """,
        ),
        (  # Anti-pattern attribute access of column
            """
for _, x in df.iterrows():
    print(x.value)
            """,
            """
for _, x in df.iterrows():
    print(x.value)
            """,
        ),
        (
            """
for _, x in df.iterrows():
    print(x["value"])
    y = 0
    y += x.at["qwerty"] ** x.iat[9]
    if y >= 199 and x.iat[13] + x.iat[8] > x["q"]:
        print(x["jk"] + x["e"])
            """,
            """
for x in df.itertuples():
    print(x.value)
    y = 0
    y += x.qwerty ** x[9 + 1]
    if y >= 199 and x[13 + 1] + x[8 + 1] > x.q:
        print(x.jk + x.e)
            """,
        ),
        (
            """
for _, x in df.iterrows():
    print(x["value"])
    y = 0
    y += x.at["qwerty"] ** x.iat[9]
    y += x.__getattr__(1)
            """,
            """
for _, x in df.iterrows():
    print(x["value"])
    y = 0
    y += x.at["qwerty"] ** x.iat[9]
    y += x.__getattr__(1)
            """,
        ),
        (
            """
for i, x in df.iterrows():
    x["value"] = 1
            """,
            """
for i, x in df.iterrows():
    x["value"] = 1
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = performance_pandas.replace_iterrows_itertuples(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
