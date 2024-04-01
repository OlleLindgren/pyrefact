#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import tracing

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
from os import *

print(getcwd())
        """,
        """
from os import getcwd

print(getcwd())
        """,
        ),
        (
            """
from os import *
from pathlib import *
from sys import *

print(Path(getcwd()))
        """,
        """
from os import getcwd
from pathlib import Path

print(Path(getcwd()))
        """,
        ),
        (
            """
import time

print(time.time())
        """,
        """
import time

print(time.time())
        """,
        ),
        (
            """
from time import *
from os import *
from pathlib import *
from datetime import *

print(f'''
Working directory: {getcwd()}
Current time:      {datetime.now().isoformat()}
Created:           {datetime.datetime.utcfromtimestamp(stat(__file__).st_ctime)}
Last modified:     {datetime.datetime.utcfromtimestamp(stat(__file__).st_mtime)}
''')
        """,
        """
from os import getcwd, stat
from datetime import datetime

print(f'''
Working directory: {getcwd()}
Current time:      {datetime.now().isoformat()}
Created:           {datetime.datetime.utcfromtimestamp(stat(__file__).st_ctime)}
Last modified:     {datetime.datetime.utcfromtimestamp(stat(__file__).st_mtime)}
''')
        """,
        ),
        (  # Thise are reimported from other files, but fix_reimported_names fixes that instead.
            """
from c import *
from e import *

print(x)
print(z)
print(ww)
        """,
        """
from c import x, z
from e import ww

print(x)
print(z)
print(ww)
        """,
    ),)

    sys.path.append(str(Path(__file__).parents[1] / "integration" / "tracing_test_files"))
    for source, expected_abstraction in test_cases:
        processed_content = tracing.fix_starred_imports(source)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
