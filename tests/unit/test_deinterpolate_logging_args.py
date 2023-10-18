#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = ((
        """
logging.info("interesting information: {value}".format(value=1337))
    """,
        """
logging.info("interesting information: {value}", value=1337)
    """,
        ),
        (
            """
logging.critical("interesting information: {}, {}, {}".format(13, 14, 15))
    """,
        """
logging.critical("interesting information: {}, {}, {}", 13, 14, 15)
    """,
        ),
        (
            """
logging.log(logging.INFO, 'interesting information: {}, {}, {}'.format(13, 14, 15))
    """,
        """
logging.log(logging.INFO, 'interesting information: {}, {}, {}', 13, 14, 15)
    """,
        ),
        (
            """
logging.log(25, f'interesting information: {value}')
    """,
        """
logging.log(25, 'interesting information: {}', value)
    """,
        ),
        (
            """
logging.warning(f"interesting information: {value}")
    """,
        """
logging.warning('interesting information: {}', value)
    """,
        ),
        (
            """
logging.error(f'interesting information: {value}')
    """,
        """
logging.error('interesting information: {}', value)
    """,
        ),
        (  # Too complex if additional args
            """
logging.error(f"interesting information: {value}", more_args)
    """,
        """
logging.error(f"interesting information: {value}", more_args)
    """,
        ),
        (
            """
logging.info(f"interesting information: {value}", foo=more_args)
    """,
        """
logging.info(f"interesting information: {value}", foo=more_args)
    """,
        ),
        (  # Too complex if additional args.
            """
logging.log(10, f"interesting information: {value}", more_args)
    """,
        """
logging.log(10, f"interesting information: {value}", more_args)
    """,
        ),
        (
            """
logging.log(10, f'interesting information: {value}', foo=more_args)
    """,
        """
logging.log(10, f'interesting information: {value}', foo=more_args)
    """,
        ),
        (
            """
logger.info("interesting information: {value}".format(value=1337))
    """,
        """
logger.info("interesting information: {value}", value=1337)
    """,
        ),
        (
            """
logger.critical('interesting information: {}, {}, {}'.format(13, 14, 15))
    """,
        """
logger.critical('interesting information: {}, {}, {}', 13, 14, 15)
    """,
        ),
        (
            """
log.log(logging.INFO, "interesting information: {}, {}, {}".format(13, 14, 15))
    """,
        """
log.log(logging.INFO, "interesting information: {}, {}, {}", 13, 14, 15)
    """,
    ),)

    for source, expected_abstraction in test_cases:
        processed_content = fixes.deinterpolate_logging_args(source)
        if not testing_infra.check_fixes_equal(
            processed_content, expected_abstraction, clear_paranthesises=True
        ):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
