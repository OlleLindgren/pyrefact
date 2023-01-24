"""Code relating to coding style"""
import re
from typing import Sequence

from pyrefact import parsing


def _list_words(name: str) -> Sequence[str]:
    return [
        match.group()
        for match in re.finditer(r"([A-Z]{2,}(?![a-z])|[A-Z]?[a-z]*)\d*", name)
        if match.end() > match.start()
    ]


def _make_snakecase(name: str, *, uppercase: bool = False) -> str:
    return "_".join(word.upper() if uppercase else word.lower() for word in _list_words(name))


def _make_camelcase(name: str) -> str:
    return "".join(word[0].upper() + word[1:].lower() for word in _list_words(name))


def rename_class(name: str, *, private: bool) -> str:
    name = re.sub("_{1,}", "_", name)
    if len(name) == 0:
        raise ValueError("Cannot rename empty name")

    name = _make_camelcase(name)

    if private and not parsing.is_private(name):
        return f"_{name}"
    if not private and parsing.is_private(name):
        return name[1:]

    return name


def rename_variable(variable: str, *, static: bool, private: bool) -> str:
    if variable == "_":
        return variable

    if variable.startswith("__") and variable.endswith("__"):
        return variable

    renamed_variable = _make_snakecase(variable, uppercase=static)

    if private and not parsing.is_private(renamed_variable):
        renamed_variable = f"_{renamed_variable}"
    if not private and parsing.is_private(renamed_variable):
        renamed_variable = renamed_variable.lstrip("_")

    if renamed_variable:
        return renamed_variable

    raise RuntimeError(f"Unable to find a replacement name for {variable}")
