"""Primitive autocomplete logic"""

import re

_COMPLETIONS = {
    "def main(args: Sequence[str]) -> int:": """
def _parse_args(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    return parser.parse_args(args)

def main(args: Sequence[str]) -> int:
    args = _parse_args(args)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
    """,
    'with open(filename, "r", encoding="utf-8") as stream:': """
with open(filename, "r", encoding="utf-8") as stream:
        content = stream.read()
    """,
    'if __name__ == "__main__":': """
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
""",
}


def _match_pattern(line1, line2) -> bool:
    if line1.strip() == line2.strip():
        return False

    matching = 0
    non_matching = 0
    for ch1, ch2 in zip(line1, line2):
        if ch1 == ch2:
            matching += 1
        else:
            non_matching += 1

    return matching > 5 and non_matching < max(2, matching * 0.1)


def autocomplete(content: str) -> None:
    """Autocomplete a file according to known patterns.

    Args:
        filename (Path): File to run autocomplete on.

    """
    lines = content.splitlines(keepends=True)

    if not lines:
        return content

    new_lines = []

    n_completes = 0
    for line in lines:
        for pattern, replacement in _COMPLETIONS.items():
            if not _match_pattern(pattern, line):
                continue

            whitespace = re.findall("^ *", line)
            whitespace = whitespace[0] if whitespace else ""
            completion = []
            for new_line in ("\n" + whitespace).join(replacement.splitlines()):
                completion.append(new_line)
            new_lines.extend(completion)
            n_completes += 1
            print(f"Auto-completed {line.strip()} to:\n{''.join(completion)}")
            break
        else:
            new_lines.append(line)

    if lines[0] == "#!/":
        n_completes += 1
        lines[0] = "#!/usr/bin/env python3"

    if n_completes == 0:
        return content

    content = "".join(new_lines)
    return content
