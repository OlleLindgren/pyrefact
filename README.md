# pyrefact
Automatic python refactoring

## Features

* Run black and isort with --line-length=100.
* Delete unused imports
* Make educated guesses for missing imports
  * For example, if pylint reports an undefined variable for `Sequence`, it will insert `from typing import Sequence` at the top of the file.
* Rename lowercased static variables to uppercase. For example, `some_variable` would become `_SOME_VARIABLE`.
* More to come!

## Usage

```bash
git clone <pyrefact git url>
pip install -e ./pyrefact
python -m pyrefact ./path/to/filename.py ./path/to/directory
```

## The purpose of pyrefact

Pyrefact exists to do the sort of simple fixes to python files that are in my experience unquestionably good.

My ambition is not for it to work for all use cases, and for everyone everywhere, but if someone is willing to co-develop it into that state, that would be fantastic.
