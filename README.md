# pyrefact
Automatic python refactoring, with the goal of simplifying complicated code, deleting dead code, and to some extent improve performance. 

It is strongly recommended that you version control or otherwise backup any code you run pyrefact on.

## Features

* Delete unused imports
* Move safe imports to toplevel
* Add missing imports by guessing what you probably wanted.
  * For example, if `Sequence` is used but never defined, it will insert `from typing import Sequence` at the top of the file.
* Rename variables, functions and classes with conventions.
* Delete unused functions, classes and variables.
* Remove most pointless simple statements.
* Remove branches of code that obviously do nothing useful.
* Remove unreachable code.
* Replace hardcoded inlined collections and comprehensions with set or generator equivalents in places where that would improve performance.
* Remove redundant chained calls involving sorted(), set(), tuple(), reversed() and list().
* Replace `sorted()[:n]` with `heapq.nsmallest`, replace `sorted()[0]` with `min`
* Rename unused variables to `_`
* Delete variables named `_`, unless where that would cause a syntax error.
* Move code into primitive functions.
* Remove duplicate function definitions.
* Remove redundant elif and else.
* Use is for comparisons to None, True and False instead of ==.
* Remove unused `self` and `cls` function arguments, and add `@staticmethod` or `@classmethod`.
* Move functions decorated with `@staticmethod` outside of their class namespaces.
* Run black and isort with --line-length=100. (Will be removed in a future release)
* More to come!

## Usage

```bash
pip install pyrefact
python -m pyrefact /path/to/filename.py --preserve /path/to/module/where/filename/is/used
```

When running, pyrefact will do a number of different fixes in sequence after eachother. The intention of how these fixes are ordered is not to create a pretty result in the end, but for the earlier steps to expose patterns that later steps of pyrefact can re-refactor. While this typically works well, it is also the case that some steps of pyrefact (especially when creating abstracted functions) have a way of creating new problems that the earlier steps would have solved. Therefore, my advice is that you manually re-run pyrefact until convergence.

## Prerequisites

Many features require `python>=3.9` to work, and pyrefact is not tested with earlier python versions. Some effort is made for it not to crash on 3.8, but most of the cool stuff is disabled.
