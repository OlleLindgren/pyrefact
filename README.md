# pyrefact
Automatic python refactoring

## Features

* Run black and isort with --line-length=100.
* Delete unused imports
* Add missing imports by guessing what you probably wanted.
  * For example, if pylint reports an undefined variable for `Sequence`, it will insert `from typing import Sequence` at the top of the file.
* Rename variables, functions and classes with conventions.
* Delete unused functions, classes and variables.
* Remove pointless string statements.
* More to come!

## Usage

```bash
pip install pyrefact
python -m pyrefact /path/to/filename.py
```

## The purpose of pyrefact

Pyrefact exists to do the sort of simple fixes to python files that are in my experience unquestionably good.

My ambition is not for it to work for all use cases, and for everyone everywhere, but if someone is willing to co-develop it into that state, that would be fantastic.
