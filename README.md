# pyrefact
Automatic python refactoring, with the goal of simplifying complicated code, and deleting dead code.

## Features

* Run black and isort with --line-length=100.
* Delete unused imports
* Add missing imports by guessing what you probably wanted.
  * For example, if pylint reports an undefined variable for `Sequence`, it will insert `from typing import Sequence` at the top of the file.
* Rename variables, functions and classes with conventions.
* Delete unused functions, classes and variables.
* Remove most pointless simple statements.
* Remove branches of code that obviously do nothing useful.
* Remove unreachable code.
* Rename unused variables to `_`
* Delete variables named `_`, unless where that would cause a syntax error.
* More to come!

## Usage

```bash
pip install pyrefact
python -m pyrefact /path/to/filename.py --preserve /path/to/module/where/filename/is/used
```
