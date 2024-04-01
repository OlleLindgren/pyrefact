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
from logging import critical
from logging import critical, error, error, error, warning
from logging import error, info, log
from logging import info
from logging import warning
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
from logging import error as info, warning as error
from logging import info
from logging import warning
from numpy import array
from numpy import ndarray
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
import logging
import os
import logging, numpy, os, os as sys, pandas as pd
import os, os, os, pandas as pd
        """,
        ),
        (
            """
from __future__ import annotations
from __future__ import absolute_import
import os
import logging
from typing import List, Dict, Tuple
from typing import List, Dict, Tuple, Any, Union
from numpy import ndarray
import numpy as np
from .. import utils2
from re import findall
from . import utils
from .. import utils
from .. import utils1
from .utils import *
import re
        """,
        """
from __future__ import absolute_import
from __future__ import annotations
import logging
import os
import re
from re import findall
from typing import Any, Dict, List, Tuple, Union
from typing import Dict, List, Tuple

import numpy as np
from numpy import ndarray

from .. import utils
from .. import utils1
from .. import utils2
from . import utils
from .utils import *
        """,
        ),
        (
            """
import logging as logging
        """,
        """
import logging
        """,
        ),
        (
            """
from logging import info as info
        """,
        """
from logging import info
        """,
        ),
        (
            """
import logging as logging
from logging import info as info
        """,
        """
import logging
from logging import info
        """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = fixes.sort_imports(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
