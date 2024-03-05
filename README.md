# pyrefact

Pyrefact does advanced python refactoring, with the goal of simplifying complicated code, deleting dead code, and improving performance. Pyrefact is entirely rule-based.

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

### Pattern-matching

Pyrefact supports pattern-matching analogous to Python's builtin `re` library. The functions `finditer`, `findall`, `sub`, `subn`, `search`, `match`, `fullmatch` and `compile` are implemented:
```python
>>> from pyrefact import pattern_matching
>>> source = """
... x = 1
... y = "asdf"
... """
>>> pattern = "x = 1"
>>> list(pattern_matching.finditer(pattern, source))
[Match(span=Range(start=1, end=6), source='\nx = 1\ny = "asdf"\n', groups=(<ast.Assign object at 0x1015f38e0>,))]
>>> pattern_matching.findall(pattern, source)
['x = 1']
>>> pattern_matching.sub(pattern, "x = 33 - x", source)
'\nx = 33 - x\ny = "asdf"\n'
>>> pattern_matching.search(pattern, source)
Match(span=Range(start=1, end=6), source='\nx = 1\ny = "asdf"\n', groups=(<ast.Assign object at 0x103acaf20>,))
```

Pattern-matching can also be used from the command-line:
```bash
python -m pyrefact.pattern_matching find "x = {{value}}" /path/to/filename.py
python -m pyrefact.pattern_matching replace "x = {{value}}" "x = 1 - {{value}} ** 3" /path/to/filename.py
```

## Installation

Pyrefact can be installed with pip, and works on Python 3.8 or newer:

```bash
pip install pyrefact
```

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

Pyrefact is tested on CPython 3.8, 3.9, 3.10, 3.11 and 3.12, and on Windows, MacOS and Linux. Pyrefact is also tested on PyPy3.10.

PyPy tests may be removed in the future, see https://github.com/OlleLindgren/pyrefact/issues/25. I will add tests for new CPython versions when they enter alpha, and remove tests when they become EOL, along with any special logic in place to support those versions.
