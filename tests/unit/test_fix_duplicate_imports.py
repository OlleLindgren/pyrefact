#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
from logging import info
from logging import warning
from logging import error, info, log
from logging import (
    warning,
    critical   ,
    error, error, error as error)
from logging import critical
        """,
        """
from logging import critical, error, info, log, warning
        """,
        ),
        (
            """
from logging import info
from numpy import ndarray
from logging import warning
from numpy import array
from logging import error as info, warning as error
        """,
        """
from logging import error as info, info, warning, warning as error
from numpy import array, ndarray
        """,
        ),
        (
            """
import logging
import logging
import logging, numpy, pandas as pd, os as sys, os as os
import pandas as pd, os, os, os
import os
        """,
        """
import logging
import numpy
import os
import os as sys
import pandas as pd
        """,
        ),
        (
            """
if foo():
    from spam import eggs
else:
    from spam import spam
        """,
            """
if foo():
    from spam import eggs
else:
    from spam import spam
        """,
        ),
        (
            """
from spam import eggs
print(10)
from spam import spam
        """,
        """
from spam import eggs
print(10)
from spam import spam
        """,
        ),
        (
            """
from spam import eggs
from spam import spam
        """,
        """
from spam import eggs, spam
        """,
        ),
        (
            """
import foo as foo
        """,
        """
import foo
        """,
        ),
        (
            """
import foo as food
        """,
        """
import foo as food
        """,
        ),
        (
            """
import foo.bar as bar, spam.eggs as eggs
        """,
        """
from foo import bar
from spam import eggs
        """,
        ),
        (
            """
import foo.bar as bars
        """,
        """
import foo.bar as bars
        """,
        ),
        (
            """
import a.b.c.d.e.f.g.h.i as i
        """,
        """
from a.b.c.d.e.f.g.h import i
        """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = fixes.fix_duplicate_imports(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
