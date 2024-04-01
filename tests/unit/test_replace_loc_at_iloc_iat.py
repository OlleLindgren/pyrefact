#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import performance_pandas

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
y = x.loc[1, 2]
        """,
        """
y = x.at[1, 2]
        """,
        ),
        (
            """
y = x.iloc[1, 2]
        """,
        """
y = x.iat[1, 2]
        """,
        ),
        (  # Series
            """
y = x.loc[1]
        """,
        """
y = x.at[1]
        """,
        ),
        (
            """
y = x.iloc[2]
        """,
        """
y = x.iat[2]
        """,
        ),
        (
            """
y = x.loc[1.13, "name_of_column"]
        """,
        """
y = x.at[(1.13, "name_of_column")]
        """,
        ),
        (
            """
y = x.loc[1.13, ["name_of_column", "name_of_other_column"]]
        """,
        """
y = x.loc[1.13, ["name_of_column", "name_of_other_column"]]
        """,
        ),
        (
            """
y = x.loc[1, var]
        """,
        """
y = x.loc[1, var]
        """,
        ),
        (
            """
y = x.iloc[[2, 3], 2]
        """,
        """
y = x.iloc[[2, 3], 2]
        """,
        ),
        (
            """
y = iloc[3, 2]
        """,
        """
y = iloc[3, 2]
        """,
        ),
        (
            """
y = loc[3, 2]
        """,
        """
y = loc[3, 2]
        """,
        ),
        (  # Series
            """
y = x.loc[var]
        """,
        """
y = x.loc[var]
        """,
        ),
        (  # Series
            """
y = x.iloc[[2, 3]]
        """,
        """
y = x.iloc[[2, 3]]
        """,
        ),
        (
            """
y = iloc[3, 2]
        """,
        """
y = iloc[3, 2]
        """,
        ),
        (
            """
y = loc[3, 2]
        """,
        """
y = loc[3, 2]
        """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = performance_pandas.replace_loc_at_iloc_iat(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
