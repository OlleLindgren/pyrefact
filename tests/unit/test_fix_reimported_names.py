#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import tracing

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
import os
import sys
        """,
        """
import os
import sys
        """,
        ),
        (
            """
from c import z
        """,
        """
from d import y as z
        """,
        ),
        (
            """
from d import sys
        """,
        """
import sys
        """,
        ),
        (
            """
from c import z
from b import x as k
from d import sys
        """,
        """
from d import x as k
from d import y as z
import sys
        """,
        ),
        (
            """
from c import z
from b import x as k
from d import sys
from e import hh
        """,
        """
from d import x as k
from d import y as z
import sys
from e import hh
        """,
        ),
        (  # This fix doesn't touch starred imports. They're fixed by fix_starred_imports
            """
from c import *
from e import *
        """,
        """
from c import *
from e import *
        """,
    ),)

    sys.path.append(str(Path(__file__).parents[1] / "integration" / "tracing_test_files"))
    for source, expected_abstraction in test_cases:
        processed_content = tracing.fix_reimported_names(source)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
