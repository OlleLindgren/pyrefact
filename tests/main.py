import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run all scripts in the pyrefact/tests folder.

    Returns:
        int: Always returns 0. If unsuccessful, raises exceptions.
    """
    for filename in Path(__file__).parent.rglob("*.py"):
        if str(filename.absolute()) != __file__:
            print(f"Testing {filename.name}...")
            subprocess.check_call([sys.executable, filename], cwd=Path(__file__).parents[1])
            print("PASSED")

    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
