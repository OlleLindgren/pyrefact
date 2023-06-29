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
import os
import re
            """,
            """
import os
import re
            """,
        ),
        (
            """
import os


import re
            """,
            """
import os
import re
            """,
        ),
        (
            """
def f():
    import os


    import re
            """,
            """
def f():
    import os
    import re
            """,
        ),
        (
            """
import os; import sys


import re
            """,
            """
import os; import sys
import re
            """,
        ),
        (
            """
import os

from re import findall
            """,
            """
import os
from re import findall
            """,
        ),
        (
            """
import os


import re
from pathlib import (
    Path,
    PurePath, PosixPath,
)
import numpy
import pandas as pd
            """,
            """
import os
import re
from pathlib import (
    Path,
    PurePath, PosixPath,
)

import numpy
import pandas as pd
            """,
        ),
        (
            """
import os

# Interesting comment
import re
import numpy
import pandas as pd
def foo():
    import os as re
    
    import re as os

    print(100)
import pandas
foo()
            """,
            """
import os

# Interesting comment
import re

import numpy
import pandas as pd


def foo():
    import os as re
    import re as os

    print(100)

import pandas

foo()
            """,
        ),
    )

    for source, expected_abstraction in test_cases:

        processed_content = fixes.fix_import_spacing(source)

        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction, clear_whitespace=False):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
