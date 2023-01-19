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
"select column from table where value in ('xyz', 'zyx')"
            """,
            '''
"""
select column
from table
where value in ('xyz',
                'zyx')
"""
            ''',
        ),
        (
            """
f"select column from table where value in ({sql_injection_vulnerable_stuff})"
            """,
            '''
f"""
select column
from table
where value in ({sql_injection_vulnerable_stuff})
"""
            ''',
        ),
    )

    for content, expected_abstraction in test_cases:
        processed_content = fixes.format_inlined_sql(content)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
