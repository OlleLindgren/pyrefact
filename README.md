# pyrefact
Automatic python refactoring, with the goal of simplifying complicated code, deleting dead code, and to some extent improve performance.

It is strongly recommended that you version control or otherwise backup any code you run pyrefact on.

## Features

### Readability

* Invert `if`/`else` to put the smaller block first.
* De-indent code with early `continue` and `return` statements.
* Replace loops that only fill up lists or sets with comprehensions.
* Rename variables, functions and classes with conventions.
* Move code into primitive functions.

### Performance

* Replace `sum` comprehensions and for loops with constant expressions. The symbolic algebra tool [Sympy](https://github.com/sympy/sympy) is used under the hood.
* Replace hardcoded inlined collections and comprehensions with set or generator equivalents in places where that would improve performance.
* Replace `sorted()[:n]` with `heapq.nsmallest`, replace `sorted()[0]` with `min`
* Replace matrix operation comprehensions with equivalent `np.matmul()` and `np.dot()` calls, for code that already depends on numpy.
* Use `is` instead of `==` for comparisons to `None`, `True` and `False`.

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

### Imports

* Delete unused imports
* Move safe imports to toplevel
* Add missing imports by guessing what you probably wanted.
  * For example, if `Sequence` is used but never defined, it will insert `from typing import Sequence` at the top of the file.

### Cleanup

* Run black and isort with --line-length=100. (Will be removed in a future release)

## Usage

The `--preserve` flag lets you define places where code is used. When this is set, pyrefact will try to keep these usages intact.
The `--safe` flag will entirely prevent pyrefact from renaming or removing code.
The `--from-stdin` flag will format code recieved from stdin, and output the result to stdout.

```bash
pip install pyrefact
python -m pyrefact /path/to/filename.py --preserve /path/to/module/where/filename/is/used
python -m pyrefact /path/to/filename.py --safe
cat /path/to/filename.py | pyrefact --from-stdin
```

## VS Code Extension

Pyrefact is also available as a VS Code extension, simply named `Pyrefact`. The extension allows you to use pyrefact as your formatter, similar to how the Black or Autopep8 extensions work.

You can also use pyrefact as your default formatter and use the configurations for format on save etc. Pyrefact always runs with the `--safe` flag when used through the VS Code extension.

The repository for the extension can be found at [pyrefact-vscode-extension](https://github.com/OlleLindgren/pyrefact-vscode-extension).

## Prerequisites

### CPython

Pyrefact requires `python>=3.8`, and is tested on CPython 3.8, 3.9, 3.10 and 3.11. Pyrefact works best on `python>=3.9`.

### Pypy

Pyrefact is supported and tested on Pypy3.9. Pypy3.8 is not supported. Pyrefact is slightly faster on Pypy3.9 than on CPython3.9, but only for large repos as the JIT compile overhead is otherwise too large to make up.
