# Contributing to Pyrefact

First off, thank you for considering contributing to Pyrefact!

## Where to Start

If you're looking to contribute but aren't sure where to start, check out the open issues on GitHub, especially those labeled as "good first issue". These are tasks that have been identified as a good place to start for new contributors.

## Ways to Contribute

There are many ways you can directly contribute to Pyrefact:

* Open issues about problems: If you find a problem with Pyrefact, open an issue for it! User feedback is super valuable, so any and all suggestions for how to continue Pyrefact's development are welcome.
* Fix bugs: Look through the GitHub issues for bugs. Anything tagged with "bug" is open to be fixed.
* Implement enhancements: Look through the GitHub issues for enhancements. This is a list of features that the community has requested.
* Improve documentation: Pyrefact could always use more documentation. Whether it's more comments in the code, more explanation in the readme, or better usage guides, helpful documentation is always a great contribution!

## Setting up a development environment

Setting up a development environment is relatively straightforward; all you really need is to clone the repo, pip install it as editable source, and start making changes. There are no extra requirements or such, all you need is pyrefact itself:

```bash
git clone https://github.com/OlleLindgren/pyrefact.git
cd pyrefact
pip install -e .
```

To run the tests that go into the automated CI, just run:

```bash
cd pyrefact
./tests/main.py
./tests/numpy.sh
```

It does happen that tests pass on some python versions but not on others. If this happens, you'll need to install pyrefact on whatever python version it was that failed in the automated CI, and debug locally. In general, if `./tests/main.py` passes on both python3.8 and python3.12, all tests will pass in the automated CI. Differences in behaviour are less common with recent pyrefact versions than they used to be though, so testing on one python version is normally enough.

## Code Contributions

Pyrefact is structured to apply a number of "fixes" in sequence after eacother, over and over until convergence or a maximum
number of iterations. Contributions to the fixes made by pyrefact are encouraged.

Although the fixes have some categorization, most of them reside in [fixes.py](pyrefact/fixes.py), so it makes sense to put new
ones here as well, although the file is getting longer than I would like.

Simple fixes can sometimes be implemented with a find-replace style pattern. This might look like the following:

```python
from pyrefact import processing

@processing.fix
def remove_redundant_import_aliases(source: str) -> str:
    find = "import {{something}} as {{something}}"
    replace = "import {{something}}"
    yield from processing.find_replace(source, find=find, replace=replace)
```

Here, a few things are going on. `find` is a pattern that find_replace() will search for, and `replace` is what will be put instead
when the find pattern is found. The double curly braces around `{{something}}` indicate that this is a wildcard, so this could
match any python AST. Since we use the same name for `{{something}}` in both places in `find`, it must be the same code in both
places. And, since we also put it in `replace`, the same code pattern matched as `something` in `find` will be put in `replace`.

Next, create [/tests/unit/test_remove_redundant_import_aliases.py](/tests/unit/test_remove_redundant_import_aliases.py): 

```python
#!/usr/bin/env python3

import sys
from pathlib import Path

from pyrefact import fixes

sys.path.append(str(Path(__file__).parents[1]))
import testing_infra


def main() -> int:
    test_cases = (
        (  # What it's intended to fix
            """
import z as z
            """,
            """
import z
            """,
        ),
        (  # What it should not touch
            """
import foo as bar
            """,
            """
import foo as bar
            """,
        ),
        (  # To test for conflicts, highlight limitations etc
            """
import b as b
import x as y
import z as z, k as k, h as kh
import math as math
import heapq as heapq
import math
import numpy, math
import math as numpy
            """,
            """
import b
import x as y
import z as z, k as k, h as kh
import math
import heapq
import math
import numpy, math
import math as numpy
            """,
        ),
    )

    for source, expected_abstraction in test_cases:
        processed_content = fixes.remove_redundant_import_aliases(source)
        if not testing_infra.check_fixes_equal(processed_content, expected_abstraction):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

```

The code in this test file is mostly boilerplate that we could copy from any other similar test file, but we need to replace the
test cases from wherever we copied it. In general, it's good to have:
* A test case where it should fix exactly one thing
  * This makes it clear what the function does, and what it should have done, but didn't, in case this test fails in the future.
* A test case where it should fix nothing
  * This can highlight the limits of the fix. E.g. what is out of scope and needs to be handled elsewhere.
* A test case where it should fix multiple problems
  * This would test for race conditions, which can be a problem.

Then, add your fix to either `_multi_run_fixes()` or `_single_run_fixes()` in [main.py](/pyrefact/main.py), depending on whether you want it "in the loop",
or "before the loop" of other fixes. Most of the fixes run in `_multi_run_fixes()`.

Finally, run [tests/main.py](/tests/main.py) and see that all tests passes. It will automatically find your new test file if you used the normal naming convention.
In the GitHub CI, this will run on Windows, MacOS and Linux, and on all python versions between 3.8 and 3.12, but if tests pass on both python3.8 and python3.12
it's uncommon for them to fail on any of the other versions. All tests must pass before code can be merged.

Here's how you can contribute code:

* Fork the Repo: Click the "Fork" button in the top right of the main repo page. This will create a copy of the repository that you can edit.
* Clone your fork: Use `git clone https://github.com/<your_username>/pyrefact.git` to clone your fork to your local machine.
* Create a branch: It's best practice to create a new branch for each new feature or bugfix. You can do this using `git checkout -b <branch-name>`.
* Make your changes: Make the changes you want to contribute.
* Commit your changes: Use `git commit -am "Your descriptive commit message"` to commit your changes. Stick to the present-tense, imperative-style commit messages.
* Push your changes: Use `git push origin <branch-name>` to push your changes to your fork. Check so that all automated tests pass.
* Create a pull request: Go to your fork on GitHub and click the "New Pull Request" button. Fill out the form and submit it!

## Code Style

The Pyrefact repo is formatted with itself, and periodically with [black-compact](https://github.com/OlleLindgren/black-compact), a fork of the regular black formatter.
What's more important than formatting, though, is that your commits don't include additional formatting beyond the actual changes you're making. So it's better that you format too little than too much.

Appropriate formatting can be accomplished by:
```bash
cd $(git rev-parse --show-toplevel)  # To the root of the repo
pip install git+https://github.com/OlleLindgren/black-compact@b2fe670fafa22fea0660feb037f0909a10358c76#egg=black
black ./pyrefact
```

## Questions?

If you have any questions, please don't hesitate to create an issue on GitHub. We'll do our best to get back to you as soon as possible!
