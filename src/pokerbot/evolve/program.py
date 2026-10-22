"""Candidate programs: editable blocks, the frozen-region check, the AST allow-list, loading.

A candidate is a single Python source following a frozen template. ``check`` enforces
(EVOLUTION.md §4):
  - everything outside the EVOLVE blocks is byte-identical to the template;
  - imports only ``math``, ``numpy`` and ``pokerbot.evolve.api``;
  - no I/O, introspection or dynamic code (``open``, ``eval``, ``exec``, ``__import__``,
    ``getattr``, ``globals`` ..., numpy's file and native-code entry points), no dunder
    attribute access, no ``global``/``nonlocal``;
  - ``PARAMS`` maps names to ``param(default, low, high)``.
``load`` executes the source with a restricted set of builtins and returns its namespace.
"""

from __future__ import annotations

import ast
import builtins
import re
from pathlib import Path

TEMPLATES = Path(__file__).with_name("templates")
CLASS_OF = {
    "classifier_v1": "Classifier",
    "exploiter_leduc_v1": "Exploiter",
    "exploiter_hu_v1": "Exploiter",
}
# A block body is the whole lines between its marker line and the END line. Both are anchored at
# line starts, so an empty block can't run on into the next block's END marker.
_BLOCK = re.compile(
    r"(^[ \t]*# EVOLVE-BLOCK (\w+)[ \t]*\n)(.*?)(^[ \t]*# END-EVOLVE-BLOCK)", re.S | re.M
)
ALLOWED_IMPORTS = {"math", "numpy", "pokerbot.evolve.api"}
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview", "type",
    "object", "super", "classmethod", "staticmethod", "property", "dir", "id", "hash",
}  # fmt: skip
# numpy attributes that reach the file system or native code
FORBIDDEN_ATTRS = {
    "load", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile",
    "tofile", "memmap", "ctypeslib", "ctypes", "lib", "f2py", "distutils", "testing",
    "DataSource", "fromregex", "show_config", "dump", "dumps",
}  # fmt: skip
SAFE_BUILTINS = [
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance",
    "len", "list", "map", "max", "min", "pow", "range", "reversed", "round", "set", "sorted",
    "str", "sum", "tuple", "zip", "frozenset", "divmod", "Exception", "ValueError",
    "ZeroDivisionError", "ArithmeticError", "IndexError", "KeyError", "TypeError",
    "FloatingPointError", "OverflowError", "True", "False", "None", "NotImplemented",
    "__build_class__",
]  # fmt: skip


class ProgramError(Exception):
    """A candidate failed a gate; ``reason`` is a short machine-readable tag."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def template(name: str = "classifier_v1") -> str:
    return (TEMPLATES / f"{name}.py").read_text(encoding="utf-8")


def blocks(source: str) -> dict[str, str]:
    return {m.group(2): m.group(3) for m in _BLOCK.finditer(source)}


def frozen(source: str) -> str:
    """The source with every block body removed (what must match the template)."""
    return _BLOCK.sub(lambda m: m.group(1) + "<BLOCK>" + m.group(4), source)


def replace_block(source: str, name: str, body: str) -> str:
    if body and not body.endswith("\n"):
        body += "\n"
    for m in _BLOCK.finditer(source):
        if m.group(2) == name:
            return source[: m.start(3)] + body + source[m.end(3) :]
    raise ProgramError("no_block", name)


def check(source: str, template_name: str = "classifier_v1") -> None:
    if frozen(source) != frozen(template(template_name)):
        raise ProgramError("frozen_changed", "code outside EVOLVE blocks differs from the template")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ProgramError("syntax", str(e)) from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    raise ProgramError("import", a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_IMPORTS or node.level:
                raise ProgramError("import", str(node.module))
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ProgramError("forbidden_name", node.id)
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ProgramError("dunder", node.attr)
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            raise ProgramError("forbidden_attr", node.attr)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            raise ProgramError("global_state", "global/nonlocal statement")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise ProgramError("dunder", "string containing __")


def _builtins() -> dict:
    return {k: getattr(builtins, k) for k in SAFE_BUILTINS if hasattr(builtins, k)} | {
        "True": True, "False": False, "None": None,
    }  # fmt: skip


def load(source: str, template_name: str = "classifier_v1") -> dict:
    """Check and execute a candidate; returns its namespace (``Classifier``, ``PARAMS``...)."""
    check(source, template_name)
    import math

    import numpy

    from pokerbot.evolve import api

    allowed = {"math": math, "numpy": numpy, "pokerbot.evolve.api": api}

    def _import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name.startswith("numpy.") and level == 0:
            # numpy's C code imports its own submodules lazily through the caller's builtins;
            # candidates can't reach this (the AST check only allows the three imports).
            return builtins.__import__(name, globals, locals, fromlist, level)
        if name not in allowed:
            raise ImportError(name)
        if name == "pokerbot.evolve.api" and not fromlist:
            raise ImportError("use 'from pokerbot.evolve.api import ...'")
        return allowed[name]

    namespace = {"__builtins__": _builtins() | {"__import__": _import}, "__name__": "candidate"}
    exec(compile(source, "<candidate>", "exec"), namespace)  # noqa: S102 - checked source
    params = namespace.get("PARAMS")
    if not isinstance(params, dict) or not all(isinstance(v, api.Param) for v in params.values()):
        raise ProgramError("params", "PARAMS must map names to param(default, low, high)")
    cls = CLASS_OF.get(template_name, "Classifier")
    if cls not in namespace:
        raise ProgramError("contract", f"no {cls} class")
    return namespace


def defaults(namespace: dict) -> dict[str, float]:
    return {k: v.default for k, v in namespace["PARAMS"].items()}
