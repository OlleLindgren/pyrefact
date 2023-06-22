# pyrefact

Pyrefact does advanced python refactoring, with the goal of simplifying complicated code, deleting dead code, and improving performance.

Unlike emerging AI tools, pyrefact is entirely rule based and does not share your code with any third parties. Pyrefact can however break your code in some cases, and is not suitable for CI or other automated workflows.

## Features

### Readability

* Move common code in `if`/`else` blocks before or after the `if`/`else`.
* De-indent code with early `continue` and `return` statements.
* Simplify boolean expressions.
* Replace for loops with immediate `if` conditions with `filter`.
* Remove commented code.
* Move code into primitive functions.
* Replace loops that only fill up lists, dicts or sets with comprehensions.
* Invert `if`/`else` to put the smaller block first.
* Rename variables, functions and classes with conventions.
* Rewrite defaultdict-like use of dictionaries with `collections.defaultdict()`.
* Formats lines that are longer than 100 characters with `black`.
* Put overused constant expressions in variables.
* Remove redundancies and improve definitions of lists, sets, dicts.
* Use `is` instead of `==` for comparisons to `None`, `True` and `False`.

### Performance

* Replace `sum` comprehensions and for loops with constant expressions. The symbolic algebra tool [Sympy](https://github.com/sympy/sympy) is used under the hood.
* Replace hardcoded inlined collections and comprehensions with set or generator equivalents in places where that would improve performance.
* Replace `sorted()[:n]` with `heapq.nsmallest`, replace `sorted()[0]` with `min`.
* Replace matrix operation comprehensions with equivalent `np.matmul()` and `np.dot()` calls, for code that already depends on numpy.
* Replace pandas .loc[], .iloc[] and .iterrows() with .at[], .iat[] and .itertuples()/.index where appropriate.
* Move constant code in loops before the loops.
* De-interpolate interpolated logging calls.

### Removing dead and useless code

* Delete unused functions, classes and variables.
* Remove most pointless simple statements.
* Remove branches of code that obviously do nothing useful.
* Remove unreachable code.
* Rename unused variables to `_`.
* Delete variables named `_`, unless where that would cause a syntax error.
* Remove redundant chained calls involving `sorted()`, `set()`, `tuple()`, `reversed()`, `iter()` and `list()`.
* Remove duplicate function definitions.
* Remove redundant elif and else.
* Remove unused `self` and `cls` function arguments, and add `@staticmethod` or `@classmethod`.
* Move functions decorated with `@staticmethod` outside of their class namespaces.
* Simplify deterministic `if`, `elif` and `else` statements.

### Imports

* Delete unused imports.
* Refactor star imports (e.g. `from pathlib import *`) to normal imports (e.g. `from pathlib import Path`)
* Move builtin and otherwise safe imports to toplevel.
* Replace indirect imports with direct imports, in cases where a name is imported from a file that also imports that name. Exceptions exists for `__init__.py` files, and files that define `__all__`.
* Merge or remove duplicate or partially duplicate imports
* Break out stacked plain imports to individual lines
* Add missing imports by guessing what you probably wanted.
  * For example, if `Sequence` is used but never defined, it will insert `from typing import Sequence` at the top of the file.

### Cleanup

* Run [isort](https://pycqa.github.io/isort/) to organize imports.
* Run [black](https://black.readthedocs.io/en/stable/) on added code, modified code, and lines that are longer than 100 characters.

## Usage

The `--preserve` flag lets you define places where code is used. When this is set, pyrefact will try to keep these usages intact.
The `--safe` flag will entirely prevent pyrefact from renaming or removing code.
The `--from-stdin` flag will format code recieved from stdin, and output the result to stdout.

```bash
pip install pyrefact
pyrefact /path/to/filename.py --preserve /path/to/module/where/filename/is/used
pyrefact /path/to/filename.py --safe
cat /path/to/filename.py | pyrefact --from-stdin
```

It is possible to disable pyrefact for a given file by adding a comment with `pyrefact: skip_file` anywhere in the file, as done [here](tests/unit/test_trace_origin.py).

## Contributing

To contribute to Pyrefact, please view [CONTRIBUTING.md](/CONTRIBUTING.md)

## VS Code Extension

Pyrefact is also available as a VS Code extension, simply named `Pyrefact`. The extension allows you to use pyrefact as your formatter, similar to how other formatting extensions work. 

Pyrefact always runs with the `--safe` flag when used through the VS Code extension.

The extension is published through the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=olleln.pyrefact), and the source code is available at [pyrefact-vscode-extension](https://github.com/OlleLindgren/pyrefact-vscode-extension).

## Prerequisites

### CPython

Pyrefact requires `python>=3.8`, and is tested on CPython 3.8, 3.9, 3.10, 3.11 and 3.12. Pyrefact works best on `python>=3.9`.

### Pypy

Pyrefact is tested on Pypy3.9.
