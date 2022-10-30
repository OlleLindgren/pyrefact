ASSUMED_PACKAGES = frozenset(
    {
        "argparse",
        "collections",
        "configparser",
        "datetime",
        "Flask",
        "functools",
        "itertools",
        "json",
        "keras",
        "math",
        "matplotlib",
        "numpy",
        "os",
        "pandas",
        "re",
        "requests",
        "scipy",
        "setuptools",
        "shlex",
        "sklearn",
        "subprocess",
        "sys",
        "tensorflow",
        "time",
        "traceback",
        "urllib",
        "warnings",
    }
)


PACKAGE_ALIASES = {"pd": "pandas", "np": "numpy", "plt": "matplotlib.pyplot"}
ASSUMED_SOURCES = {
    "typing": frozenset(
        {
            "Callable",
            "Collection",
            "Dict",
            "Iterable",
            "List",
            "Literal",
            "Mapping",
            "NamedTuple",
            "Optional",
            "Sequence",
            "Tuple",
            "Union",
        }
    ),
    "pathlib": frozenset({"Path"}),
    "types": frozenset({"MappingProxyType"}),
}
