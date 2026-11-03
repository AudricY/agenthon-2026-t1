"""Whether a script uses the method library `qf` (agent/qf), read from its source alone: moved from
scripts/run_sweep_read.py, which N9's read counts with, so N11's gate (agent/n11) and the read count one way.

`qf_use(script)`: whether the script imports `qf` (`import qf`, `import qf.sub as x`, `from qf import f`, `from
qf.sub import f`), and the `qf` functions it calls: an attribute call on a name the import bound (`qf.f(...)`,
`x.f(...)`, `qf.sub.f(...)`), or a call of a name imported from it, under its own name. By the script's syntax tree; a
script that does not parse (cut, or broken) by its lines (`fallback`, lane lib8): the names its import lines bind, as
the tree's reading binds them, with a name `from qf import` binds a shelf's too (`from qf import rates`), then each
call on them: `qf.f(`, `qf.sub.f(`, `x.f(` and a name's own `f(`. Standard library only.
"""
from __future__ import annotations

import ast
import re
import warnings

QF_IMPORT = re.compile(r"^\s*(?:import\s+qf\b|from\s+qf(?:\.\w+)*\s+import\b)", re.M)   # a script that does not parse
IMPORT_LINE = re.compile(r"^[ \t]*import[ \t]+([^\n#;]+)", re.M)            # its import statements, one a line
FROM_LINE = re.compile(r"^[ \t]*from[ \t]+(qf(?:\.\w+)*)[ \t]+import[ \t]+(\([^)]*\)?|[^\n#;]+)", re.M)   # (a, b) too
DEFINED = re.compile(r"\b(def|class)[ \t]+\w+")         # a definition's name is no call of it


def qf_use(script: str) -> tuple[bool, set[str]]:
    """Whether `script` imports `qf`, and the `qf` functions it calls (above)."""
    try:
        with warnings.catch_warnings():            # a script's own invalid escapes are its business, not the log's
            warnings.simplefilter("ignore")
            tree = ast.parse(script)
    except (SyntaxError, ValueError):
        return fallback(script)
    modules, names = set(), {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.asname or "qf" for alias in node.names
                        if alias.name == "qf" or alias.name.startswith("qf.")}
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module and (
                node.module == "qf" or node.module.startswith("qf.")):
            names.update({alias.asname or alias.name: alias.name for alias in node.names})
    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in names:
            called.add(names[func.id])
        elif isinstance(func, ast.Attribute):
            head = func.value
            while isinstance(head, ast.Attribute):
                head = head.value
            if isinstance(head, ast.Name) and head.id in modules:
                called.add(func.attr)
    return bool(modules or names), called


def fallback(script: str) -> tuple[bool, set[str]]:
    """`qf_use` of a script that does not parse, by its lines (above): the modules its `import` lines bind to `qf` or a
    shelf (`qf` itself, or the name after `as`), and the names its `from qf... import` lines bind (a parenthesised list
    across lines too, cut or not); then each attribute call on one of those modules, or on a name `from qf import`
    binds, by its last name, and each call of a bound name by the name it imports, a call's bracket on its name's line.
    Where the script parses this counts every call the tree's reading counts but one whose bracket opens on a later
    line (`qf.f` then `(` below it) or written `qf . f(`; `qf.<name>(` counts wherever a qf import line stands."""
    modules, names, shelves = set(), {}, set()
    for line in IMPORT_LINE.finditer(script):
        for item in line.group(1).split(","):
            words = item.split()
            if words and (words[0] == "qf" or words[0].startswith("qf.")):
                modules.add(words[2] if len(words) == 3 and words[1] == "as" else "qf")
    for line in FROM_LINE.finditer(script):
        for item in line.group(2).strip().strip("()").split(","):
            words = item.split()
            if words and words[0].isidentifier():
                bound = words[2] if len(words) == 3 and words[1] == "as" else words[0]
                names[bound] = words[0]
                if line.group(1) == "qf":
                    shelves.add(bound)
    if not (modules or names or QF_IMPORT.search(script)):
        return False, set()
    text, called = DEFINED.sub(r"\1", script), set()
    heads = "|".join(map(re.escape, sorted(modules | shelves | {"qf"})))
    for match in re.finditer(rf"(?<![\w.])(?:{heads})((?:\.\w+)+)[ \t]*\(", text):
        called.add(match.group(1).rsplit(".", 1)[1])
    if names:
        for match in re.finditer(r"(?<![\w.])(" + "|".join(map(re.escape, sorted(names))) + r")[ \t]*\(", text):
            called.add(names[match.group(1)])
    return True, called
