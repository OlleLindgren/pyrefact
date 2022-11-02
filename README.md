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
* Rename unused variables to _
* Delete variables named _, unless that would cause a syntax error.
* More to come!

The interactions between these features can be very powerful, since making one change may reveal that other changes can also be made. For big repos with lots of redundancy this can lead to big quantities of code being removed. Since pyrefact sometimes catches things that pylint would have missed, this can be the case even in professionally managed repos with a well-functioning CI system.

## Usage

```bash
pip install pyrefact
python -m pyrefact /path/to/filename.py --preserve /path/to/module/where/filename/is/used
```
