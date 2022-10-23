import subprocess
import sys
from pathlib import Path


def main() -> int:
    for filename in Path(__file__).parent.rglob("*.py"):
        if str(filename.absolute()) != __file__:
            print(f"Testing {filename.name}...")
            subprocess.check_call([sys.executable, filename])
            print("PASSED")

    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
