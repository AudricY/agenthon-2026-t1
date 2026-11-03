"""A static check of a saved Python file (lane clock1 of 10-13; the owner's in707 and in709, agenthon-t2; reader
`time`, docs/research/time-1013.md section 3: 17 of the 154 failing runs a check at the write would have caught).

`AGENTHON_N11_LINT=on` (agent/n11/solve.py) reads each `.py` file a write saves (commands.Workspace.write) and adds its
findings to the write's answer, one line each, free of counts; they are advisory: the file is saved and runs as before.
Three checks, by the file's syntax tree alone:
- `name`: a name the file loads and never binds anywhere (an import, an assignment, a definition, an argument, a
  loop's, a comprehension's or a `with`'s target, an `except ... as`, a `global`), nor a builtin or a module's own
  dunder (`free`); a `*` import from a file beside it under work/ binds what that file exports, and none is said where
  the file imports `*` from anywhere else. Under the run's shim (`SHIM=on`) its standard imports
  (agent/n11/shim.py `STANDARD`) are not said: the shim imports them.
- `qf`: a `qf` name the library lacks (`qf.<name>`, `qf.<shelf>.<name>`, `from qf import <name>`, `import qf.<shelf>`,
  `from qf.<shelf> import <name>`), with the library's nearest names; a keyword a function's signature lacks; a
  keyword-only argument it requires and a call leaves out (none where the call passes `**`). The library is the index's
  (`library.Library`: each function's shelf and arguments as written) and the package's own imports (`known`).
- `column`: a string literal read as a column (a subscript's, or a `usecols`, `parse_dates`, `index_col`, `on`,
  `left_on`, `right_on`, `subset`, `by`, `id_vars` or `value_vars` argument's) that names an input table's column under
  another case, where no input names it as written and the file never makes it (a subscript it assigns, a dict's key, a
  rename's or a `columns`, `names` or `assign` argument's).
Lane lint2 adds two narrow findings apart (`check2`; `AGENTHON_N11_LINT2=on`, agent/n11/solve.py `LINT2`), each
inferred from the syntax tree with its scopes (`Scopes`), false findings costing more than misses:
- `shadowed`: an imported name a binding shadows where the file reads it as the import. In a function: a read after a
  binding whose type the syntax proves (a literal, a display, a builtin's result, a loop over `range`, a `with open`)
  and which lacks the attribute the read takes or a call, or a read before a binding that makes the name local
  (UnboundLocalError); at module level, a read after such a binding, there or in a function. Not said: another
  import, an alias or any untyped value, an import's fallback in an `except`, a parameter of the name, a `global`, a
  closure's read; a conditional binding outside the read's arm; a module binding after an earlier call.
- `unused`: a name bound from a configuration read (`Config`: a JSON, YAML or TOML load's key, by subscript or `get`,
  through the file's own loader, a nested table or a conversion; an argument parse's attribute) that nothing reads
  afterwards in its scope or a scope seeing it (including annotations, defaults and class-body execution). Not said:
  a file calling `locals`, `globals`, `vars`, `eval` or
  `exec` or reading a `__dict__`; a `_`-led name, one in `__all__` or named by a format field; a key read again; a
  name or key a file beside it names; a name rebound and read.
`FINDINGS` bounds the lines a write's answer carries (the harness's bound, never said), each check's apart. Standard
library only.
"""
from __future__ import annotations

import ast
import builtins
import csv
import difflib
import io
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

from n11 import library

KINDS = ("name", "qf", "column")
FINDINGS = 4                    # a write's lines, at most: the harness's bound, never said
LINE_CHARS = 220                # a line's characters, at most (a long signature cut at its end)
NEAREST = 3                     # the library's names said beside one it lacks, at most
BUILTINS = frozenset(dir(builtins))
DUNDERS = frozenset({"__file__", "__name__", "__doc__", "__builtins__", "__spec__", "__loader__", "__package__",
                     "__path__", "__annotations__", "__cached__", "__dict__", "__class__", "__debug__"})
COLUMN_ARGS = frozenset({"usecols", "parse_dates", "index_col", "on", "left_on", "right_on", "subset", "by", "id_vars",
                         "value_vars"})
MADE_ARGS = frozenset({"columns", "names"})     # a call's argument that makes columns (`DataFrame(columns=...)`)
TABLES = (".csv", ".tsv", ".txt", ".dat")
HEADER_BYTES = 65_536           # a table's head read for its columns
SAID = {
    "name": "lint: `{name}` is used in {file} and never defined or imported there",
    "qf_name": "lint: the library holds no `{said}`{near}",
    "qf_shelf": "lint: the library has no module `{said}`{near}; its functions are each `qf.<name>`",
    "qf_keyword": "lint: `{call}` takes no keyword `{keyword}`: {signature}",
    "qf_required": "lint: `{call}` requires the keyword `{keyword}`: {signature}",
    "column": "lint: `{literal!r}` names no input column as written; {file} has `{column}`",
}


@dataclass(frozen=True)
class Finding:
    kind: str       # one of KINDS
    line: str       # the observation's line


@dataclass(frozen=True)
class Known:
    """The library's names as a check reads them: each function's arguments as its `def` writes them (None where its
    signature is not read), each shelf's functions, and every other name the package binds."""
    functions: dict[str, str | None]
    shelves: dict[str, frozenset[str] | None]
    names: frozenset[str]

    def has(self, name: str) -> bool:
        return name in self.functions or name in self.shelves or name in self.names


def parsed(text: str) -> ast.Module | None:
    try:
        with warnings.catch_warnings():         # a script's invalid escapes are its business
            warnings.simplefilter("ignore")
            return ast.parse(text)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None


def bound(tree: ast.AST) -> set[str]:
    """Every name the tree binds anywhere, in any scope."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        elif type(node).__name__ in ("TypeVar", "ParamSpec", "TypeVarTuple"):
            names.add(node.name)
    return names


def starred(tree: ast.AST) -> list[str | None]:
    """The modules the tree imports `*` from (None for a relative one's)."""
    return [node.module if not node.level else None for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)]


def exported(tree: ast.Module) -> set[str] | None:
    """The names a module's `*` binds: its `__all__` where it lists one, else the names its own top level binds (not
    `_`-led); None where it imports `*` itself."""
    if starred(tree):
        return None
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__"
                                                for target in node.targets):
            try:
                return {str(item) for item in ast.literal_eval(node.value)}
            except ValueError:
                return None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.With, ast.If, ast.Try)):
            names.update(child.id for child in ast.walk(node) if isinstance(child, ast.Name)
                         and isinstance(child.ctx, ast.Store))
    return {name for name in names if not name.startswith("_")}


def free(tree: ast.AST, modules: dict[str, str] | None = None) -> list[str]:
    """The names the tree loads and never binds, nor a builtin or a module's dunder, in the order first loaded; a `*`
    import from one of `modules` (the files beside it, by module name) binds what that file's `*` exports; none where
    it imports `*` from anywhere else."""
    known = bound(tree) | BUILTINS | DUNDERS
    for module in starred(tree):
        text = (modules or {}).get(module) if module else None
        tree_ = parsed(text) if text is not None else None
        names = exported(tree_) if tree_ is not None else None
        if names is None:
            return []
        known |= names
    loaded = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)]
    loaded.sort(key=lambda node: (node.lineno, node.col_offset))
    return list(dict.fromkeys(node.id for node in loaded if node.id not in known))


def known(lib, package: Path | None = None) -> Known:
    """The names a check reads from the library (`library.Library`: its functions with their arguments, by shelf) and,
    where `package` is given, its `__init__.py`'s own imports (a function folded out of its shelf's index is still
    `qf.<name>`) and its modules (an import module such as `qf.bs`, its members unread)."""
    functions: dict[str, str | None] = {}
    shelves: dict[str, frozenset[str] | None] = {}
    for shelf in getattr(lib, "shelves", ()) or ():
        shelves[shelf.name] = frozenset(function.name for function in shelf.functions)
        for function in shelf.functions:
            functions.setdefault(function.name, function.arguments)
    names: set[str] = set()
    if package is not None and Path(package).is_dir():
        imported, _ = library.exported(Path(package) / "__init__.py")
        names |= imported
        for path in sorted(Path(package).glob("*.py")):
            if not path.stem.startswith("_"):
                shelves.setdefault(path.stem, None)
    return Known(functions, shelves, frozenset(names - set(functions)))


def signature(arguments: str) -> ast.arguments | None:
    found = parsed(f"def f({arguments}): pass")
    return found.body[0].args if found is not None else None


def near(name: str, choices) -> str:
    found = difflib.get_close_matches(name, sorted(choices), n=NEAREST, cutoff=0.6)
    return f" (nearest: {', '.join(f'`qf.{each}`' for each in found)})" if found else ""


class QfReader:
    """The file's `qf` names, bound and used: the modules (`qf` and its aliases), the shelves and the functions its
    imports bind."""

    def __init__(self, tree: ast.Module, lib: Known) -> None:
        self.tree, self.lib = tree, lib
        self.modules: set[str] = set()                  # names bound to `qf`
        self.shelves: dict[str, str] = {}               # name -> the shelf it is bound to
        self.functions: dict[str, str] = {}             # name -> the function it is bound to
        self.found: list[tuple[int, str, str]] = []     # (line, kind, line said)

    def say(self, node: ast.AST, kind: str, **values) -> None:
        self.found.append((getattr(node, "lineno", 0), "qf", SAID[kind].format(**values)[:LINE_CHARS]))

    def imports(self) -> None:
        lib = self.lib
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if parts[0] != "qf":
                        continue
                    if len(parts) > 1 and parts[1] not in lib.shelves:
                        self.say(node, "qf_shelf", said=".".join(parts[:2]), near=near(parts[1], lib.shelves))
                        continue
                    if alias.asname and len(parts) == 2:
                        self.shelves[alias.asname] = parts[1]
                    elif alias.asname is None or len(parts) == 1:
                        self.modules.add(alias.asname or "qf")
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module and \
                    node.module.split(".")[0] == "qf":
                parts = node.module.split(".")
                if len(parts) > 1 and parts[1] not in lib.shelves:
                    self.say(node, "qf_shelf", said=".".join(parts[:2]), near=near(parts[1], lib.shelves))
                    continue
                members = lib.shelves.get(parts[1]) if len(parts) > 1 else None
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    bound_as = alias.asname or alias.name
                    if len(parts) == 1 and alias.name in lib.shelves:
                        self.shelves[bound_as] = alias.name
                    elif len(parts) == 1 and not lib.has(alias.name):
                        self.say(node, "qf_name", said=f"qf.{alias.name}", near=near(alias.name, lib.functions))
                    elif len(parts) > 1 and members is not None and alias.name not in members:
                        self.say(node, "qf_name", said=f"{node.module}.{alias.name}",
                                 near=near(alias.name, lib.functions))
                    elif alias.name in lib.functions:
                        self.functions[bound_as] = alias.name

    def target(self, node: ast.AST) -> str | None:
        """The function a call's callee names (`qf.f`, `qf.shelf.f`, a shelf's alias `.f`, an imported `f`), said
        where the library lacks it; None where it names no `qf` function the library reads."""
        if isinstance(node, ast.Name):
            return self.functions.get(node.id)
        if not isinstance(node, ast.Attribute):
            return None
        head, chain = node, []
        while isinstance(head, ast.Attribute):
            chain.append(head.attr)
            head = head.value
        if not isinstance(head, ast.Name):
            return None
        chain.reverse()
        if head.id in self.modules:
            first = chain[0]
            if not self.lib.has(first):
                return None
            if first in self.lib.shelves and first not in self.lib.functions and len(chain) > 1:
                members = self.lib.shelves[first]
                return chain[1] if members is None or chain[1] in members else None
            return first if first in self.lib.functions else None
        if head.id in self.shelves:
            members = self.lib.shelves.get(self.shelves[head.id])
            return chain[0] if members is None or chain[0] in members else None
        return None

    def attributes(self) -> None:
        """Each `qf.<name>` and `qf.<shelf>.<name>` (or a shelf alias's `.<name>`) the library lacks, said once."""
        said: set[str] = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
                continue
            base, name = node.value.id, node.attr
            if base in self.modules and not self.lib.has(name):
                key = f"qf.{name}"
                if key not in said:
                    said.add(key)
                    self.say(node, "qf_name", said=key, near=near(name, self.lib.functions))
            elif base in self.shelves:
                members = self.lib.shelves.get(self.shelves[base])
                key = f"qf.{self.shelves[base]}.{name}"
                if members is not None and name not in members and key not in said:
                    said.add(key)
                    self.say(node, "qf_name", said=key, near=near(name, self.lib.functions))
        for node in ast.walk(self.tree):        # qf.<shelf>.<name> on the module itself
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute) and \
                    isinstance(node.value.value, ast.Name) and node.value.value.id in self.modules:
                shelf, name = node.value.attr, node.attr
                members = self.lib.shelves.get(shelf) if shelf not in self.lib.functions else None
                key = f"qf.{shelf}.{name}"
                if members is not None and name not in members and key not in said:
                    said.add(key)
                    self.say(node, "qf_name", said=key, near=near(name, self.lib.functions))

    def calls(self) -> None:
        """Each call of a library function whose keywords its signature lacks, or which leaves out a keyword-only
        argument the signature requires."""
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            name = self.target(node.func)
            arguments = self.lib.functions.get(name) if name else None
            args = signature(arguments) if arguments is not None else None
            if args is None:
                continue
            named = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
            given = {keyword.arg for keyword in node.keywords if keyword.arg is not None}
            spread = any(keyword.arg is None for keyword in node.keywords)
            shown = f"qf.{name}({arguments})"
            for keyword in node.keywords:
                if keyword.arg is not None and keyword.arg not in named and args.kwarg is None:
                    self.say(node, "qf_keyword", call=f"qf.{name}", keyword=keyword.arg, signature=shown)
            if not spread:
                for arg, default in zip(args.kwonlyargs, args.kw_defaults):
                    if default is None and arg.arg not in given:
                        self.say(node, "qf_required", call=f"qf.{name}", keyword=arg.arg, signature=shown)

    def read(self) -> list[tuple[int, str, str]]:
        self.imports()
        if self.modules or self.shelves or self.functions:
            self.attributes()
            self.calls()
        return self.found


def strings(node: ast.AST) -> list[str]:
    """The string constants a node holds: itself, or a list's, tuple's or set's items."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [item.value for item in node.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)]
    return []


def columns_read(tree: ast.AST) -> tuple[list[tuple[int, str]], set[str]]:
    """The literals the file reads as columns (line, literal), and those it makes (`MADE_ARGS`, a subscript it assigns,
    a dict's key, a rename's target, an `assign`'s keyword)."""
    read: list[tuple[int, str]] = []
    made: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            found = strings(node.slice)
            if isinstance(node.ctx, ast.Load):
                read += [(node.lineno, literal) for literal in found]
            else:
                made.update(found)
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in COLUMN_ARGS:
                    read += [(node.lineno, literal) for literal in strings(keyword.value)]
                elif keyword.arg in MADE_ARGS:
                    made.update(strings(keyword.value))
                    if isinstance(keyword.value, ast.Dict):
                        made.update(literal for value in keyword.value.values for literal in strings(value))
            if isinstance(node.func, ast.Attribute) and node.func.attr == "assign":
                made.update(keyword.arg for keyword in node.keywords if keyword.arg)
        elif isinstance(node, ast.Dict):
            made.update(literal for key in node.keys if key is not None for literal in strings(key))
    return read, made


def table_columns(path: Path) -> list[str]:
    """An input table's columns: a delimited text's header row, a parquet file's schema; [] for any other file or one
    that cannot be read."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".parquet", ".pqt"):
            import pyarrow.parquet as pq
            return list(pq.read_schema(path).names)
        if suffix not in TABLES:
            return []
        with path.open("rb") as handle:
            head = handle.read(HEADER_BYTES).decode("utf-8-sig", errors="replace")
        lines = [line for line in head.split("\n")[:20] if line.strip()]
        if len(lines) < 2:
            return []
        delimiter = max((",", "\t", ";", "|"), key=lambda each: (len({line.count(each) for line in lines[:-1]}) == 1,
                                                                 lines[0].count(each)))
        if not lines[0].count(delimiter):
            return []
        return [cell.strip() for cell in next(csv.reader([lines[0]], delimiter=delimiter))]
    except Exception:  # noqa: BLE001 - a file that cannot be read has no columns to check
        return []


def inputs_columns(data_dir: Path) -> dict[str, list[str]]:
    """Each input table under `data_dir` (by its name there) with its columns, the tables with none left out."""
    data_dir = Path(data_dir)
    found = {}
    for path in sorted(data_dir.rglob("*")) if data_dir.is_dir() else []:
        if path.is_file():
            names = table_columns(path)
            if names:
                found[path.relative_to(data_dir).as_posix()] = names
    return found


def check(text: str, named: str, lib: Known | None, columns: dict[str, list[str]], skip: frozenset[str] = frozenset(),
          modules: dict[str, str] | None = None) -> list[Finding]:
    """The findings over one file's text (`named`, as `work/<name>.py`), in the file's order, each kind's once a name:
    its undefined names but `skip` (the shim's; a `*` import from one of `modules`, the files beside it, read through
    it), its `qf` names and keywords against `lib` (none where None), its column literals against `columns` (each
    input table's)."""
    tree = parsed(text)
    if tree is None:
        return []
    found: list[tuple[int, str, str]] = []
    first: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            first[node.id] = min(first.get(node.id, node.lineno), node.lineno)
    found += [(first.get(name, 0), "name", SAID["name"].format(name=name, file=named))
              for name in free(tree, modules) if name not in skip]
    if lib is not None:
        found += QfReader(tree, lib).read()
    if columns:
        exact = {name for names in columns.values() for name in names}
        folded: dict[str, tuple[str, str]] = {}
        for file, names in columns.items():
            for name in names:
                folded.setdefault(name.lower(), (file, name))
        read, made = columns_read(tree)
        said: set[str] = set()
        for line, literal in read:
            hit = folded.get(literal.lower())
            if literal in exact or literal in made or literal in said or hit is None:
                continue
            said.add(literal)
            found.append((line, "column", SAID["column"].format(literal=literal, file=hit[0], column=hit[1])))
    lines: dict[str, Finding] = {}
    for _, kind, line in sorted(found, key=lambda each: each[0]):
        lines.setdefault(line, Finding(kind, line[:LINE_CHARS]))
    return list(lines.values())


# ---------------------------------------------------------------------------------------- lane lint2's two findings

KINDS2 = ("shadowed", "unused")
SHAPE_CHARS = 48                # a quoted line's characters, at most (its head kept)
SAID2 = {
    "shadowed": "lint: `{name}` is imported (`{imported}`) and rebound by `{bound}` {where}, so `{read}` {there} reads "
                "the rebound value, not the import",
    "unbound": "lint: `{name}` is imported (`{imported}`) and rebound by `{bound}` in `{function}`, which makes it "
               "local there, so `{read}` before that raises UnboundLocalError",
    "unused": "lint: nothing reads `{name}` after `{bound}`, so the key `{key}` of `{source}` has no effect",
}
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
LOOPS = (ast.For, ast.AsyncFor, ast.While)
FILES = (io.TextIOWrapper, io.BufferedReader, io.BufferedWriter, io.FileIO)
RESULTS = {"int": (int,), "float": (float,), "str": (str,), "bool": (bool,), "bytes": (bytes,), "list": (list,),
           "dict": (dict,), "set": (set,), "tuple": (tuple,), "frozenset": (frozenset,), "len": (int,), "ord": (int,),
           "chr": (str,), "repr": (str,), "sorted": (list,), "range": (range,), "enumerate": (enumerate,),
           "zip": (zip,), "open": FILES}       # a builtin's result's type, where the file never rebinds the builtin
LOADS = {"json": frozenset({"load", "loads"}), "yaml": frozenset({"safe_load", "load", "full_load", "unsafe_load"}),
         "tomllib": frozenset({"load", "loads"}), "toml": frozenset({"load", "loads"})}
PARSES = frozenset({"parse_args"})     # an argument parse's namespace, read by attribute
CONVERSIONS = frozenset({"int", "float", "str", "bool", "list", "tuple", "dict", "set", "frozenset", "complex",
                         "Decimal", "Fraction"})
CONVERSION_ATTRS = frozenset({"array", "asarray", "Timestamp", "to_datetime", "Timedelta", "to_timedelta", "float64",
                              "int64", "float32", "int32"})
OPAQUE = frozenset({"locals", "globals", "vars", "eval", "exec"})   # a file calling one reads names unseen


@dataclass
class Binding:
    name: str
    kind: str                   # `import`, `value` (an assignment's or a loop's or a `with`'s target), `def`, `param`
    node: ast.AST               # the statement (the alias's import, the assignment, the loop, the `with`, the def)
    at: tuple[int, int]         # where it takes effect: after the value it binds is evaluated
    types: tuple | None = None  # the bound value's possible types, where the syntax proves them
    value: ast.AST | None = None    # an assignment's value, for a single name target
    alias: ast.alias | None = None
    fallback: bool = False      # in an `except` of a `try` that imports the name: the import's fallback


def own(scope: ast.AST) -> list[ast.AST]:
    """The nodes a scope evaluates itself: its body's, not a nested function's, lambda's or class's (whose decorators,
    defaults and bases are the scope's, and whose definition binds its name here); a comprehension's but the names its
    own targets bind."""
    body = [scope.body] if isinstance(scope, ast.Lambda) else list(scope.body)
    found: list[ast.AST] = []
    stack: list[tuple[ast.AST, frozenset]] = [(node, frozenset()) for node in reversed(body)]
    while stack:
        node, hidden = stack.pop()
        if isinstance(node, ast.Name) and node.id in hidden:
            continue
        found.append(node)
        if isinstance(node, FUNCTIONS):
            arguments = node.args
            children = [*getattr(node, "decorator_list", ()), *arguments.defaults,
                        *(each for each in arguments.kw_defaults if each is not None),
                        *(arg.annotation for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                          *(each for each in (arguments.vararg, arguments.kwarg) if each is not None))
                          if arg.annotation is not None),
                        *(each for each in (getattr(node, "returns", None),) if each is not None)]
        elif isinstance(node, ast.ClassDef):
            children = [*node.decorator_list, *node.bases, *node.keywords]
        elif isinstance(node, COMPREHENSIONS):
            inner = hidden | {each.id for generator in node.generators for each in ast.walk(generator.target)
                              if isinstance(each, ast.Name)}
            pairs = [(node.generators[0].iter, hidden)]
            pairs += [(each, inner) for generator in node.generators[1:] for each in (generator.iter,)]
            pairs += [(each, inner) for generator in node.generators for each in generator.ifs]
            pairs += [(each, inner) for each in (getattr(node, "elt", None), getattr(node, "key", None),
                                                 getattr(node, "value", None)) if each is not None]
            stack.extend(reversed(pairs))
            continue
        else:
            children = list(ast.iter_child_nodes(node))
        stack.extend((child, hidden) for child in reversed(children))
    return found


def end(node: ast.AST) -> tuple[int, int]:
    return (getattr(node, "end_lineno", 0) or 0, getattr(node, "end_col_offset", 0) or 0)


def start(node: ast.AST) -> tuple[int, int]:
    return (getattr(node, "lineno", 0) or 0, getattr(node, "col_offset", 0) or 0)


def shape(lines: list[str], node: ast.AST) -> str:
    """A statement's first line as written, its head kept within `SHAPE_CHARS`."""
    row = getattr(node, "lineno", 0) - 1
    line = lines[row] if 0 <= row < len(lines) else ""
    if getattr(node, "end_lineno", None) == getattr(node, "lineno", None):
        line = line[:node.end_col_offset]
    line = line[getattr(node, "col_offset", 0):].strip()
    return line if len(line) <= SHAPE_CHARS else line[:SHAPE_CHARS - 3].rstrip() + "..."


def import_shape(node: ast.AST, alias: ast.alias) -> str:
    """The part of an import that binds one name: `from scipy.stats import t`, `import numpy as np`."""
    named = alias.name + (f" as {alias.asname}" if alias.asname else "")
    if isinstance(node, ast.ImportFrom):
        return f"from {'.' * node.level}{node.module or ''} import {named}"
    return f"import {named}"


class Scopes:
    """A file's scopes (the module, each function and lambda) with the names each binds, in the order they take
    effect, and the parent of every node."""

    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        self.parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        self.module_names = bound(tree)
        self.scopes = [tree, *(node for node in ast.walk(tree) if isinstance(node, FUNCTIONS))]
        self.classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]     # read, never a binder's
        self.nodes = {id(scope): own(scope) for scope in (*self.scopes, *self.classes)}
        self.binds = {id(scope): self.bindings(scope) for scope in self.scopes}
        self.declared = {id(scope): {name for node in self.nodes[id(scope)] if isinstance(node, (ast.Global,
                                                                                                    ast.Nonlocal))
                                     for name in node.names} for scope in self.scopes}

    def typed(self, value: ast.AST | None) -> tuple | None:
        """The types the syntax proves a value has: a literal's, a display's, a builtin's result where the file never
        rebinds the builtin; None for anything else."""
        if isinstance(value, ast.Constant):
            return (type(value.value),)
        if isinstance(value, ast.JoinedStr):
            return (str,)
        for kinds, made in (((ast.List, ast.ListComp), list), ((ast.Tuple,), tuple), ((ast.Dict, ast.DictComp), dict),
                            ((ast.Set, ast.SetComp), set)):
            if isinstance(value, kinds):
                return (made,)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in RESULTS and \
                value.func.id not in self.module_names:
            return RESULTS[value.func.id]
        return None

    def fallback(self, node: ast.AST, name: str) -> bool:
        """Whether `node` lies in an `except` of a `try` whose body imports `name` (an import's fallback)."""
        here = node
        while here in self.parent:
            up = self.parent[here]
            if isinstance(up, ast.ExceptHandler):
                attempt = self.parent.get(up)
                if isinstance(attempt, ast.Try) and any(
                        isinstance(each, (ast.Import, ast.ImportFrom)) and
                        any((alias.asname or alias.name.split(".")[0]) == name for alias in each.names)
                        for statement in attempt.body for each in ast.walk(statement)):
                    return True
            if isinstance(up, FUNCTIONS):
                return False
            here = up
        return False

    def bindings(self, scope: ast.AST) -> list[Binding]:
        found: list[Binding] = []
        if isinstance(scope, FUNCTIONS):
            arguments = scope.args
            for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                        *(each for each in (arguments.vararg, arguments.kwarg) if each is not None)):
                found.append(Binding(arg.arg, "param", scope, (0, 0)))
        for node in self.nodes[id(scope)]:
            if node is scope:
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name != "*":
                        found.append(Binding(alias.asname or alias.name.split(".")[0], "import", node, start(node),
                                             alias=alias))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.append(Binding(node.name, "def", node, start(node)))
            elif isinstance(node, ast.ExceptHandler) and node.name:
                found.append(Binding(node.name, "value", node, start(node)))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                found.append(self.stored(node))
        for each in found:
            each.fallback = each.kind == "value" and self.fallback(each.node, each.name)
        return sorted(found, key=lambda each: each.at)

    def stored(self, name: ast.Name) -> Binding:
        """A name's store as a binding: its statement, where it takes effect and what the syntax proves of its type."""
        up = self.parent.get(name)
        if isinstance(up, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and (
                up.targets == [name] if isinstance(up, ast.Assign) else up.target is name):
            value = None if isinstance(up, ast.AugAssign) else up.value
            return Binding(name.id, "value", up, end(up), None if value is None else self.typed(value), value)
        if isinstance(up, ast.NamedExpr):
            return Binding(name.id, "value", up, end(up), self.typed(up.value), up.value)
        if isinstance(up, (ast.For, ast.AsyncFor)) and up.target is name:
            called = up.iter.func.id if isinstance(up.iter, ast.Call) and isinstance(up.iter.func, ast.Name) else ""
            types = {"range": (int,), "enumerate": (tuple,), "zip": (tuple,)}.get(called) \
                if called not in self.module_names else None
            if isinstance(up.iter, ast.Constant) and isinstance(up.iter.value, str):
                types = (str,)
            return Binding(name.id, "value", up, end(up.iter), types)
        if isinstance(up, ast.withitem) and up.optional_vars is name:
            statement = self.parent.get(up, up)
            opened = isinstance(up.context_expr, ast.Call) and isinstance(up.context_expr.func, ast.Name) and \
                up.context_expr.func.id == "open" and "open" not in self.module_names
            return Binding(name.id, "value", statement, end(up.context_expr), FILES if opened else None)
        statement = name
        while statement in self.parent and not isinstance(statement, ast.stmt):
            statement = self.parent[statement]
        return Binding(name.id, "value", statement, end(statement))

    def reads(self, scope: ast.AST, name: str) -> list[ast.Name]:
        """The scope's own loads of `name`, in the file's order."""
        return sorted((node for node in self.nodes[id(scope)] if isinstance(node, ast.Name) and node.id == name
                       and isinstance(node.ctx, ast.Load)), key=start)

    def looped(self, node: ast.AST, scope: ast.AST, inside: list[ast.AST]) -> bool:
        """Whether a loop of `scope` holding `node` also holds one of `inside` (a binding an earlier pass made)."""
        here = node
        while here in self.parent and here is not scope:
            here = self.parent[here]
            if isinstance(here, LOOPS) and any(any(each is child for child in ast.walk(here)) for each in inside):
                return True
        return False

    def same_branch(self, binding: Binding, read: ast.AST, scope: ast.AST) -> bool:
        """A conditional binding only reaches reads within that same arm; other paths are unproven."""
        here = binding.node
        while here in self.parent and here is not scope:
            up = self.parent[here]
            if isinstance(up, (ast.If, ast.Try, ast.TryStar, ast.ExceptHandler, ast.Match, ast.match_case)):
                arm = next((value for _, value in ast.iter_fields(up)
                            if isinstance(value, list) and any(child is here for child in value)), None)
                if arm is not None and not any(child is read or inside(self, read, child) for child in arm):
                    return False
            here = up
        return True


def read_shape(scopes: Scopes, node: ast.Name) -> tuple[str, str] | None:
    """A load read as the import reads it: `name.attr` (("attr", attr)) or `name(...)` (("call", ""))."""
    up = scopes.parent.get(node)
    if isinstance(up, ast.Attribute) and up.value is node and isinstance(up.ctx, ast.Load):
        return "attr", up.attr
    if isinstance(up, ast.Call) and up.func is node:
        return "call", ""
    return None


def clashes(types: tuple, read: tuple[str, str]) -> bool:
    """Whether every type the syntax proves lacks what the read takes: the attribute, or a call."""
    kind, attr = read
    if kind == "attr":
        return all(not hasattr(each, attr) for each in types)
    return all("__call__" not in dir(each) for each in types)


def said_read(name: str, read: tuple[str, str] | None) -> str:
    if read is None:
        return name
    return f"{name}.{read[1]}" if read[0] == "attr" else f"{name}(...)"


def shadowed(tree: ast.Module, lines: list[str], scopes: Scopes | None = None) -> list[tuple[int, str, str]]:
    """Each imported name a binding shadows where the file then reads it as the import: in a function, a read after a
    binding whose type the syntax proves and which lacks what the read takes (`t.ppf` after `for t in range(...)`), or a
    read before a binding that makes the name local (UnboundLocalError); at module level, the same where a later
    module-level binding shadows a module import, read at module level after it or in a function that reads the
    module's. Not said: a rebinding by another import, an alias or any value the syntax does not type (`t = stats.t`,
    `t = t(5)` at module level), an import's fallback in an `except`, a parameter of the same name. Once a name a
    scope."""
    scopes = scopes or Scopes(tree)
    found: list[tuple[int, str, str]] = []
    module = scopes.binds[id(tree)]
    imported = {}
    for each in module:
        if each.kind == "import":
            imported.setdefault(each.name, each)
    for scope in scopes.scopes:
        binds = scopes.binds[id(scope)]
        names = {each.name for each in binds if each.kind == "value"}
        is_function = scope is not tree
        for name in sorted(names):
            if name in scopes.declared[id(scope)]:
                continue
            own_binds = [each for each in binds if each.name == name]
            if any(each.kind == "param" for each in own_binds):
                continue
            imports = [each for each in own_binds if each.kind == "import"] or \
                ([imported[name]] if is_function and name in imported else [])
            if not imports:
                continue
            values = [each for each in own_binds if each.kind != "import" and not each.fallback]
            if not values:
                continue
            source = import_shape(imports[0].node, imports[0].alias)
            where = f"in `{scope.name}`" if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) else \
                "in a lambda" if is_function else "in the module"
            for read in scopes.reads(scope, name):
                at = start(read)
                before = [each for each in own_binds if each.at < at]
                shape_ = read_shape(scopes, read)
                if not before:
                    if not is_function or own_binds[0].kind == "import" or not isinstance(scope, (
                            ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    later = [each for each in values if each.at >= at]
                    if not later or scopes.looped(read, scope, [each.node for each in own_binds]):
                        continue
                    found.append((read.lineno, "shadowed", SAID2["unbound"].format(
                        name=name, imported=source, bound=shape(lines, later[0].node), function=scope.name,
                        read=said_read(name, shape_))))
                    break
                reaching = before[-1]
                if reaching.kind != "value" or reaching.fallback or reaching.types is None or shape_ is None or \
                        not clashes(reaching.types, shape_) or not scopes.same_branch(reaching, read, scope):
                    continue
                found.append((read.lineno, "shadowed", SAID2["shadowed"].format(
                    name=name, imported=source, bound=shape(lines, reaching.node), where=where,
                    read=said_read(name, shape_), there="there")))
                break
    said = {line for _, _, line in found}
    last = {}
    for each in module:
        if each.name in imported:
            last[each.name] = each
    for name, reaching in last.items():     # a module import shadowed at module level, read by a function later
        if reaching.kind != "value" or reaching.fallback or reaching.types is None or reaching.at < imported[name].at:
            continue
        for scope in scopes.scopes[1:]:
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) or \
                    any(each.name == name for each in scopes.binds[id(scope)]) or \
                    any(start(read) < reaching.at for read in scopes.reads(tree, scope.name)):
                continue
            hit = next(((read, shape_) for read in scopes.reads(scope, name)
                        if (shape_ := read_shape(scopes, read)) is not None and clashes(reaching.types, shape_)
                        and scopes.same_branch(reaching, read, tree)), None)
            if hit is None:
                continue
            line = SAID2["shadowed"].format(name=name, imported=import_shape(imported[name].node,
                                                                              imported[name].alias),
                                            bound=shape(lines, reaching.node), where="in the module",
                                            read=said_read(name, hit[1]), there=f"in `{scope.name}`")
            if line not in said:
                said.add(line)
                found.append((hit[0].lineno, "shadowed", line))
            break
    return found


def peeled(value: ast.AST) -> ast.AST:
    """A value with the conversions that keep what it reads taken off: `float(x)`, `np.array(x)`, `-x`, `x * 252`."""
    while True:
        if isinstance(value, ast.Call) and value.args and (
                isinstance(value.func, ast.Name) and value.func.id in CONVERSIONS or
                isinstance(value.func, ast.Attribute) and value.func.attr in CONVERSION_ATTRS):
            value = value.args[0]
        elif isinstance(value, ast.UnaryOp):
            value = value.operand
        elif isinstance(value, ast.BinOp) and isinstance(value.right, ast.Constant):
            value = value.left
        elif isinstance(value, ast.BinOp) and isinstance(value.left, ast.Constant):
            value = value.right
        else:
            return value


class Config:
    """A file's configuration reads: the names its JSON, YAML and TOML loads and its argument parses bind (and its own
    functions that return a load), and each value read from one by key."""

    def __init__(self, tree: ast.Module, scopes: Scopes) -> None:
        self.scopes = scopes
        self.modules: dict[str, str] = {}               # `import json as j`: j -> json
        self.functions: set[str] = set()                # `from json import load`: load
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.modules.update({alias.asname or alias.name: alias.name for alias in node.names
                                     if alias.name in LOADS})
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module in LOADS:
                self.functions.update(alias.asname or alias.name for alias in node.names
                                      if alias.name in LOADS[node.module])
        self.loaders: set[str] = set()                  # the file's functions returning a load
        for _ in range(2):
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in self.loaders:
                    loaded = {each.name for each in scopes.binds[id(node)] if each.value is not None and
                              self.source(each.value) == "dict"}
                    if any(isinstance(each, ast.Return) and each.value is not None and (
                            self.source(each.value) == "dict" or isinstance(each.value, ast.Name) and
                            each.value.id in loaded) for each in scopes.nodes[id(node)]):
                        self.loaders.add(node.name)

    def source(self, value: ast.AST) -> str | None:
        """`dict` for a load, `args` for an argument parse, None for anything else."""
        if not isinstance(value, ast.Call):
            return None
        func = value.func
        if isinstance(func, ast.Attribute):
            if func.attr in PARSES:
                return "args"
            module = self.modules.get(func.value.id) if isinstance(func.value, ast.Name) else None
            return "dict" if module is not None and func.attr in LOADS[module] else None
        if isinstance(func, ast.Name) and (func.id in self.functions or func.id in self.loaders):
            return "dict"
        return None

    def objects(self, scope: ast.AST, outer: dict[str, str]) -> dict[str, str]:
        """The names a scope reads configuration from: those whose every binding there is a load, a parse or a value
        read from one (a nested table), and the module's it does not bind."""
        by_name: dict[str, list[Binding]] = {}
        for each in self.scopes.binds[id(scope)]:
            by_name.setdefault(each.name, []).append(each)
        found = {name: kind for name, kind in outer.items() if name not in by_name}
        for _ in range(3):
            for name, binds in by_name.items():
                if name in found or not all(each.kind == "value" and each.value is not None for each in binds):
                    continue
                kinds = {self.source(each.value) or ("dict" if self.read(each.value, found) else None)
                         for each in binds}
                if len(kinds) == 1 and None not in kinds:
                    found[name] = kinds.pop()
        return found

    def read(self, value: ast.AST, objects: dict[str, str]) -> tuple[str, str] | None:
        """The key a value reads from a configuration and the name it reads it from: `params["window"]`,
        `params.get("window", 20)`, `params["model"]["window"]`, `args.window`, through `peeled`'s conversions."""
        value = peeled(value)
        if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Constant) and \
                isinstance(value.slice.value, str):
            key, base = value.slice.value, value.value
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "get" and \
                value.args and isinstance(value.args[0], ast.Constant) and isinstance(value.args[0].value, str):
            key, base = value.args[0].value, value.func.value
        elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and \
                objects.get(value.value.id) == "args":
            return value.attr, value.value.id
        else:
            return None
        if isinstance(base, ast.Name):
            return (key, base.id) if objects.get(base.id) == "dict" else None
        inner = self.read(base, objects)
        return (key, inner[1]) if inner is not None else None


def used(scopes: Scopes, scope: ast.AST, name: str) -> bool:
    """Whether `name`, bound in `scope`, is read there or in a scope nested in it that does not bind it (a function, a
    lambda, a class's body): a load, an augmented assignment, a `del`."""
    def reads(each: ast.AST) -> bool:
        return any(isinstance(node, ast.Name) and node.id == name and not isinstance(node.ctx, ast.Store) or
                   isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == name
                   for node in scopes.nodes[id(each)])
    if reads(scope):
        return True
    for inner in (*scopes.scopes, *scopes.classes):
        if inner is scope or (scope is not scopes.tree and not inside(scopes, inner, scope)):
            continue
        if isinstance(inner, ast.ClassDef):
            bindings = [each.at for each in scopes.bindings(inner) if each.name == name]
            if any(not bindings or start(read) < min(bindings) for read in scopes.reads(inner, name)):
                return True
            continue
        binds = any(each.name == name for each in scopes.binds[id(inner)])
        if (not binds or name in scopes.declared.get(id(inner), ())) and reads(inner):
            return True
    return False


def inside(scopes: Scopes, node: ast.AST, scope: ast.AST) -> bool:
    here = node
    while here in scopes.parent:
        here = scopes.parent[here]
        if here is scope:
            return True
    return False


def formatted(name: str, strings: list[str]) -> bool:
    """Whether a string of the file names `name` as a format field (`{name}`, `{name:.2f}`, `%(name)s`): a value a
    template reads through `format(**...)` or `%`."""
    field = re.compile(r"\{" + re.escape(name) + r"[}!:.\[]|%\(" + re.escape(name) + r"\)")
    return any(field.search(each) for each in strings)


def unused(tree: ast.Module, lines: list[str], scopes: Scopes, others: dict[str, str]) -> list[tuple[int, str, str]]:
    """Each name bound from a configuration read (`Config.read`: a JSON, YAML or TOML load's key, an argument parse's
    attribute) that nothing reads afterwards: not its scope, nor a scope nested in it that sees it. Not said: a file
    that calls `locals`, `globals`, `vars`, `eval` or `exec` or reads a `__dict__`; a name opening with `_`, in
    `__all__`, named by a format field of any string, or named by a file beside it (`others`, which may import it); a
    key the file reads again elsewhere or a file beside it names; a name rebound and read later. Once a name a
    scope."""
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in OPAQUE or
           isinstance(node, ast.Attribute) and node.attr == "__dict__" for node in ast.walk(tree)):
        return []
    config = Config(tree, scopes)
    strings = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    attributes = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
    exported = exported_names(tree)
    module = config.objects(tree, {})
    found: list[tuple[int, str, str]] = []
    for scope in scopes.scopes:
        objects = module if scope is tree else config.objects(scope, module)
        if not objects:
            continue
        said: set[str] = set()
        for each in scopes.binds[id(scope)]:
            if each.kind != "value" or not isinstance(each.node, (ast.Assign, ast.AnnAssign)) or each.name in said:
                continue
            value = each.value if each.value is not None else element(each.node, each.name)
            hit = config.read(value, objects) if value is not None else None
            if hit is None:
                continue
            key, source = hit
            mentioned = re.compile(rf"(?<!\w){re.escape(each.name)}(?!\w)|(?<!\w){re.escape(key)}(?!\w)")
            if each.name.startswith("_") or each.name in exported or each.name in scopes.declared[id(scope)] or \
                    used(scopes, scope, each.name) or strings.count(key) + attributes.count(key) > 1 or \
                    formatted(each.name, strings) or any(mentioned.search(text) for text in others.values()):
                continue
            said.add(each.name)
            found.append((each.node.lineno, "unused", SAID2["unused"].format(
                name=each.name, key=key, source=source, bound=shape(lines, each.node))))
    return found


def element(statement: ast.AST, name: str) -> ast.AST | None:
    """The value a tuple assignment binds to `name` element by element (`a, b = x["a"], x["b"]`); None otherwise."""
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and \
            isinstance(statement.targets[0], ast.Tuple) and isinstance(statement.value, ast.Tuple) and \
            len(statement.targets[0].elts) == len(statement.value.elts):
        for target, value in zip(statement.targets[0].elts, statement.value.elts):
            if isinstance(target, ast.Name) and target.id == name:
                return value
    return None


def exported_names(tree: ast.Module) -> set[str]:
    """The names a module's `__all__` lists, where it lists them as literals."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__"
                                                for target in node.targets):
            return set(strings(node.value))
    return set()


def check2(text: str, modules: dict[str, str] | None = None) -> list[Finding]:
    """Lane lint2's findings over one file's text, in the file's order, each once: an imported name shadowed
    (`shadowed`) and a configuration value never read (`unused`); `modules`, the files beside it by module name, which
    may import its names or read its keys."""
    tree = parsed(text)
    if tree is None:
        return []
    lines = text.splitlines()
    scopes = Scopes(tree)
    found = shadowed(tree, lines, scopes) + unused(tree, lines, scopes, modules or {})
    said: dict[str, Finding] = {}
    for _, kind, line in sorted(found, key=lambda each: each[0]):
        said.setdefault(line, Finding(kind, line[:LINE_CHARS]))
    return list(said.values())
