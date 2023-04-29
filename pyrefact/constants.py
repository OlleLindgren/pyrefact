import ast
import builtins
import keyword
import operator
import sys
import typing
from types import MappingProxyType

ASSUMED_PACKAGES = frozenset(
    (
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
    )
)

import pathlib
import types

PACKAGE_ALIASES = {"pd": "pandas", "np": "numpy", "plt": "matplotlib.pyplot"}
ASSUMED_SOURCES = {
    "typing": frozenset(typing.__all__),
    "pathlib": frozenset(pathlib.__all__),
    "types": frozenset(name for name in types.__all__ if name.endswith("Type")),
}
PYTHON_311_STDLIB = frozenset(
    (
        "string",
        "re",
        "difflib",
        "textwrap",
        "unicodedata",
        "stringprep",
        "readline",
        "rlcompleter",
        "struct",
        "codecs",
        "datetime",
        "zoneinfo",
        "calendar",
        "collections",
        "collections.abc",
        "heapq",
        "bisect",
        "array",
        "weakref",
        "types",
        "copy",
        "pprint",
        "reprlib",
        "enum",
        "graphlib",
        "numbers",
        "math",
        "cmath",
        "decimal",
        "fractions",
        "random",
        "statistics",
        "itertools",
        "functools",
        "operator",
        "pathlib",
        "os.path",
        "fileinput",
        "stat",
        "filecmp",
        "tempfile",
        "glob",
        "fnmatch",
        "linecache",
        "shutil",
        "pickle",
        "copyreg",
        "shelve",
        "marshal",
        "dbm",
        "sqlite3",
        "zlib",
        "gzip",
        "bz2",
        "lzma",
        "zipfile",
        "tarfile",
        "csv",
        "configparser",
        "tomllib",
        "netrc",
        "plistlib",
        "hashlib",
        "hmac",
        "secrets",
        "os",
        "io",
        "time",
        "argparse",
        "getopt",
        "logging",
        "logging.config",
        "logging.handlers",
        "getpass",
        "curses",
        "curses.textpad",
        "curses.ascii",
        "curses.panel",
        "platform",
        "errno",
        "ctypes",
        "threading",
        "multiprocessing",
        "multiprocessing.shared_memory",
        "concurrent.futures",
        "subprocess",
        "sched",
        "queue",
        "contextvars",
        "_thread",
        "asyncio",
        "socket",
        "ssl",
        "select",
        "selectors",
        "signal",
        "mmap",
        "email",
        "json",
        "mailbox",
        "mimetypes",
        "base64",
        "binascii",
        "quopri",
        "html",
        "html.parser",
        "html.entities",
        "xml.etree.ElementTree",
        "xml.dom",
        "xml.dom.minidom",
        "xml.dom.pulldom",
        "xml.sax",
        "xml.sax.handler",
        "xml.sax.saxutils",
        "xml.sax.xmlreader",
        "xml.parsers.expat",
        "webbrowser",
        "wsgiref",
        "urllib",
        "urllib.request",
        "urllib.response",
        "urllib.parse",
        "urllib.error",
        "urllib.robotparser",
        "http",
        "http.client",
        "ftplib",
        "poplib",
        "imaplib",
        "smtplib",
        "uuid",
        "socketserver",
        "http.server",
        "http.cookies",
        "http.cookiejar",
        "xmlrpc",
        "xmlrpc.client",
        "xmlrpc.server",
        "ipaddress",
        "wave",
        "colorsys",
        "gettext",
        "locale",
        "turtle",
        "cmd",
        "shlex",
        "tkinter",
        "tkinter.colorchooser",
        "tkinter.font",
        "tkinter.messagebox",
        "tkinter.scrolledtext",
        "tkinter.dnd",
        "tkinter.ttk",
        "tkinter.tix",
        "typing",
        "pydoc",
        "doctest",
        "unittest",
        "unittest.mock",
        "unittest.mock",
        "test",
        "test.support",
        "test.support.socket_helper",
        "test.support.script_helper",
        "test.support.bytecode_helper",
        "test.support.threading_helper",
        "test.support.os_helper",
        "test.support.import_helper",
        "test.support.warnings_helper",
        "bdb",
        "faulthandler",
        "pdb",
        "timeit",
        "trace",
        "tracemalloc",
        "distutils",
        "ensurepip",
        "venv",
        "zipapp",
        "sys",
        "sysconfig",
        "builtins",
        "__main__",
        "warnings",
        "dataclasses",
        "contextlib",
        "abc",
        "atexit",
        "traceback",
        "__future__",
        "gc",
        "inspect",
        "site",
        "code",
        "codeop",
        "zipimport",
        "pkgutil",
        "modulefinder",
        "runpy",
        "importlib",
        "importlib.resources",
        "importlib.resources.abc",
        "ast",
        "symtable",
        "token",
        "keyword",
        "tokenize",
        "tabnanny",
        "pyclbr",
        "py_compile",
        "compileall",
        "dis",
        "pickletools",
        "aifc",
        "asynchat",
        "asyncore",
        "audioop",
        "cgi",
        "cgitb",
        "chunk",
        "crypt",
        "imghdr",
        "imp",
        "mailcap",
        "msilib",
        "nis",
        "nntplib",
        "optparse",
        "ossaudiodev",
        "pipes",
        "smtpd",
        "sndhdr",
        "spwd",
        "sunau",
        "telnetlib",
        "uu",
        "xdrlib",
    )
)

BUILTIN_FUNCTIONS = frozenset(name for name in dir(builtins) if name != "_")
PYTHON_KEYWORDS = frozenset(keyword.kwlist)

PYTHON_VERSION = tuple(sys.version_info)

REVERSE_OPERATOR_MAPPING = MappingProxyType(
    {
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
        ast.Gt: ast.LtE,
        ast.Lt: ast.GtE,
        ast.GtE: ast.Lt,
        ast.LtE: ast.Gt,
        ast.In: ast.NotIn,
        ast.NotIn: ast.In,
        ast.Is: ast.IsNot,
        ast.IsNot: ast.Is,
    }
)
COMPARISON_OPERATORS = MappingProxyType(
    {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: lambda x, y: x in y,
        ast.NotIn: lambda x, y: x not in y,
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
        ast.MatMult: operator.matmul,
    }
)

MATH_FUNCTIONS = {"sum", "len"}
ITERATOR_FUNCTIONS = {
    "iter",
    "sorted",
    "list",
    "range",
    "map",
    "filter",
    "tuple",
    "reversed",
    "set",
}

AST_TYPES_WITH_BODY = (
    ast.If,
    ast.For,
    ast.With,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Module,
    ast.While,
)
AST_TYPES_WITH_ORELSE = (ast.If, ast.For, ast.While)
