#!/usr/bin/env python3
"""Maing script for running all tests."""

import cProfile
import pstats
import subprocess
import sys
import tempfile
from pathlib import Path

import pyrefact


def main() -> int:
    """Run all scripts in the pyrefact/tests folder.

    Returns:
        int: Always returns 0.
    """
    out_filename = Path.cwd() / "pyrefact_profiling.pstats"
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.check_call(
            ["git", "clone", "--depth=1", "https://github.com/numpy/numpy.git", tmpdir]
        )

        with cProfile.Profile() as profile:
            try:
                pyrefact.main([tmpdir])
            finally:
                with open(out_filename, "w") as stream:
                    stats = pstats.Stats(profile, stream=stream).sort_stats(
                        pstats.SortKey.CUMULATIVE
                    )
                    stats.dump_stats(out_filename)
                    print(f"Saved profiling to {out_filename}")
                    return 0


if __name__ == "__main__":
    sys.exit(main())
