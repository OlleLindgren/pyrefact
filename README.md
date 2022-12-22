# pyrefact
Automatic python refactoring, with the goal of simplifying complicated code, deleting dead code, and to some extent improve performance.

It is strongly recommended that you version control or otherwise backup any code you run pyrefact on.

## Features

### Imports

* Delete unused imports
* Move safe imports to toplevel
* Add missing imports by guessing what you probably wanted.
  * For example, if `Sequence` is used but never defined, it will insert `from typing import Sequence` at the top of the file.

### Readability

* Rename variables, functions and classes with conventions.

### Removing dead and useless code

* Delete unused functions, classes and variables.
* Remove most pointless simple statements.
* Remove branches of code that obviously do nothing useful.
* Remove unreachable code.
* Rename unused variables to `_`
* Delete variables named `_`, unless where that would cause a syntax error.
* Remove redundant chained calls involving `sorted()`, `set()`, `tuple()`, `reversed()`, `iter()` and `list()`.
* Remove duplicate function definitions.
* Remove redundant elif and else.
* Remove unused `self` and `cls` function arguments, and add `@staticmethod` or `@classmethod`.
* Move functions decorated with `@staticmethod` outside of their class namespaces.

### Performance

* Replace hardcoded inlined collections and comprehensions with set or generator equivalents in places where that would improve performance.
* Replace `sorted()[:n]` with `heapq.nsmallest`, replace `sorted()[0]` with `min`
* Use is for comparisons to None, True and False instead of ==.

### Restructuring

* Move code into primitive functions.
* Invert `if`/`else` to put the smaller block first
* De-indent code with early `continue` and `return` statements

### Cleanup

* Run black and isort with --line-length=100. (Will be removed in a future release)

## Usage

The `--preserve` flag lets you define places where code is used. When this is set, pyrefact will try to keep these usages intact.
The `--safe` flag will entirely prevent pyrefact from renaming or removing code.

```bash
pip install pyrefact
python -m pyrefact /path/to/filename.py --preserve /path/to/module/where/filename/is/used
python -m pyrefact /path/to/filename.py --safe
```

## Prerequisites

Pyrefact requires `python>=3.9`.
