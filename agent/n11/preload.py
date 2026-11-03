"""The run's shim, loaded before the House's script (lane clock1 of 10-13; agent/n11/shim.py writes this file as the
script's `sitecustomize` under `AGENTHON_N11_SHIM=on` and reads what it reports).

It reads its spec (the JSON file `N11_SHIM` names) and pre-empts four mechanical crashes, each where it would have
failed and nowhere else, each reported once to the spec's `report` file (one JSON line: `kind`, `what`):
- `paths`: a path in the unit image's forms (`/app/data/...`, `/app/environment/data/...`, `/work/...`, `/app/output`,
  a bare `/app/<file>`, `/app/<name>.py`, or relative from `/app`: `data/...`, `output/...`) whose root does not exist
  here is opened as the spec's table maps it (the data's real path, work/, the output); `open`, `os`'s file calls,
  `subprocess`'s arguments and working directory, and pandas' parquet readers and writers;
- `api`: the names the sandbox's versions removed, each run as the old call ran: `np.math`, `fillna(method=...)`, a
  removed frequency alias (`T`, `H`, `M`, ...), `scipy.stats.mvn.mvnun`; patched as each library is imported;
- `json`: `json.dump` and `json.dumps` write numpy and pandas scalars, arrays and timestamps as their plain values
  where no `default` or encoder of the script's own is given;
- `imports`: the standard names a file under work/ uses and never imports (the spec's `names`), imported for it.
Apart from the shim, the failure snapshot (lane snapshot1; agent/n11/snapshot.py writes its spec, the JSON file
`N11_SNAPSHOT` names, and reads what it writes): at an uncaught exception, after the interpreter's own hook has printed
the traceback, the state at the failure is written to the spec's `side` file, never to stdout or stderr (`snapshot`):
the exception's type and message; the innermost frame in a file under the spec's `work` with the source around its
line; the locals the failing statement reads (`STATEMENT_NAMES`), then the frame's others assigned nearest before the
line (`OTHER_NAMES`), each by its type and, for a frame, series, index, array, mapping, sequence or scalar, its shape or
length, dtype, index type and first and last labels, columns or keys, missing and infinite cells and a few values
(`described`); and where the traceback goes on into the library (a module `qf` or under it), that call's actual
arguments (`called`). A SyntaxError is left to the write's check; past `SECONDS` no further name is described.
Sampling traverses only the retained prefixes; past `CELLS` a frame's numeric copy and cell counts are skipped.
Standard library only at load; nothing here ever raises into the script.
"""
import ast
import builtins
import datetime
import io
import itertools
import json
import math
import os
import posixpath
import sys
import time

_open, _replace = io.open, os.replace     # the real ones, before the shim's paths wrap them (the snapshot's writes)


def _start():
    spec_path = os.environ.get("N11_SHIM", "")
    if not spec_path:
        return
    with io.open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    report_path, said = spec.get("report", ""), set()
    real_open, real_stat = io.open, os.stat

    def report(kind, what):
        if (kind, what) in said or not report_path:
            return
        said.add((kind, what))
        try:
            with real_open(report_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"kind": kind, "what": what}) + "\n")
        except Exception:  # noqa: BLE001 - a report that cannot be written loses the note, never the run
            pass

    kinds = set(spec.get("kinds", ()))
    if "paths" in kinds:
        _paths(spec, report, real_stat)
    if "api" in kinds:
        _api(report)
    if "json" in kinds:
        _json(report)
    if "imports" in kinds:
        _imports(spec.get("names", {}), report)


# ------------------------------------------------------------------------------------------------------- paths


def mapper(spec, exists, report):
    """The spec's path map as a function of one path (`_paths` installs it; an offline read replays it with an
    `exists` of its own): a path in the unit image's forms whose row's own path is missing (`exists`), as the table
    maps it, the instruction's `/app/<name>.py` as work/'s, then /app itself (`after`); any other path as it is."""
    table = sorted(((str(prefix), str(target)) for prefix, target in spec.get("table", ())),
                   key=lambda pair: -len(pair[0]))
    after = [(str(prefix), str(target)) for prefix, target in spec.get("after", ())]
    scripts, work = spec.get("scripts", ""), spec.get("work", "")

    def like(path, new):
        if isinstance(path, str):
            return new
        try:
            return type(path)(new)          # a pathlib path as the script made it
        except Exception:  # noqa: BLE001 - another os.PathLike (a DirEntry): the text
            return new

    def under(norm, rows):
        for prefix, target in rows:
            if norm == prefix or norm.startswith(prefix + "/"):
                return prefix, target
        return None

    def mapped(path):
        """`path` where it names a missing place in the unit image's forms, as the table maps it; else itself."""
        if isinstance(path, (int, bytes)) or not isinstance(path, (str, os.PathLike)):
            return path
        raw = os.fspath(path)
        if not isinstance(raw, str) or not raw:
            return path
        norm = posixpath.normpath(raw)
        hit = under(norm, table)
        if hit is not None:
            prefix, target = hit
            if exists(prefix) or prefix == target:
                return path
            report("paths", f"{prefix} -> {target}")
            return like(path, target + norm[len(prefix):])
        if scripts and norm.startswith(scripts) and norm.endswith(".py") and "/" not in norm[len(scripts):] \
                and not exists(norm):
            report("paths", f"{scripts}<name>.py -> {work}/<name>.py")
            return like(path, posixpath.join(work, norm[len(scripts):]))
        hit = under(norm, after)
        if hit is not None and not exists(hit[0]):
            prefix, target = hit
            report("paths", f"{prefix} -> {target}")
            return like(path, target + norm[len(prefix):])
        return path
    return mapped


def _paths(spec, report, real_stat):
    seen = {}

    def exists(path):
        absolute = path.startswith("/")
        if absolute and path in seen:
            return seen[path]
        try:
            real_stat(path)
            found = True
        except (OSError, ValueError):
            found = False
        if absolute:
            seen[path] = found
        return found

    mapped = mapper(spec, exists, report)

    def one(function, name):
        def wrapped(*args, **kwargs):
            if args:
                args = (mapped(args[0]), *args[1:])
            elif name in kwargs:
                kwargs[name] = mapped(kwargs[name])
            return function(*args, **kwargs)
        wrapped.__name__, wrapped.__doc__ = function.__name__, function.__doc__
        return wrapped

    def two(function):
        def wrapped(src, dst, *args, **kwargs):
            return function(mapped(src), mapped(dst), *args, **kwargs)
        wrapped.__name__, wrapped.__doc__ = function.__name__, function.__doc__
        return wrapped

    builtins.open = io.open = one(io.open, "file")
    for name in ("open", "stat", "lstat", "listdir", "scandir", "mkdir", "chdir", "remove", "unlink", "rmdir",
                 "access"):
        if hasattr(os, name):
            setattr(os, name, one(getattr(os, name), "path"))
    for name in ("rename", "replace"):
        setattr(os, name, two(getattr(os, name)))

    def popen(module):
        original = module.Popen.__init__

        def __init__(self, args, *rest, **kwargs):
            if isinstance(args, (list, tuple)):
                args = type(args)(mapped(arg) if isinstance(arg, str) and "/" in arg else arg for arg in args)
            if kwargs.get("cwd") is not None:
                kwargs["cwd"] = mapped(kwargs["cwd"])
            original(self, args, *rest, **kwargs)
        module.Popen.__init__ = __init__

    def parquet(module):
        read = getattr(module, "read_parquet", None)
        if read is not None:
            module.read_parquet = one(read, "path")
        frame = getattr(module, "DataFrame", None)
        write = getattr(frame, "to_parquet", None) if frame is not None else None
        if write is not None:
            def to_parquet(self, path=None, *args, **kwargs):
                return write(self, mapped(path), *args, **kwargs)
            frame.to_parquet = to_parquet

    _after("subprocess", popen)
    _after("pandas", parquet)


# --------------------------------------------------------------------------------------------------------- api


FREQUENCIES = {"T": "min", "S": "s", "L": "ms", "U": "us", "N": "ns", "H": "h", "BH": "bh", "CBH": "cbh", "M": "ME",
               "BM": "BME", "CBM": "CBME", "SM": "SME", "Q": "QE", "BQ": "BQE", "A": "YE", "Y": "YE", "BA": "BYE",
               "BY": "BYE"}


def _frequency(freq):
    """A frequency string in pandas 3's aliases: `5T` as `5min`, `M` as `ME`, `A-DEC` as `YE-DEC`; None where it
    names no alias pandas 3 removed."""
    text = freq.strip()
    number = len(text) - len(text.lstrip("0123456789"))
    head, unit = text[:number], text[number:]
    base, dash, anchor = unit.partition("-")
    if base not in FREQUENCIES:
        return None
    return head + FREQUENCIES[base] + dash + anchor


def _api(report):
    def numpy(module):
        original = module.__dict__.get("__getattr__")

        def __getattr__(name):
            if name == "math":
                report("api", "np.math|the `math` module")
                return math
            if original is not None:
                return original(name)
            raise AttributeError(f"module 'numpy' has no attribute {name!r}")
        module.__getattr__ = __getattr__

    def pandas(module):
        for cls in (module.DataFrame, module.Series):
            original = cls.fillna

            def fillna(self, value=None, *args, _original=original, **kwargs):
                method = kwargs.pop("method", None)
                if method is None:
                    return _original(self, value, *args, **kwargs)
                if value is not None:
                    raise ValueError("Cannot specify both 'value' and 'method'.")
                kwargs.pop("downcast", None)
                if method in ("ffill", "pad"):
                    filled = self.ffill(**kwargs)
                elif method in ("bfill", "backfill"):
                    filled = self.bfill(**kwargs)
                else:
                    raise ValueError(f"Invalid fill method. Expecting pad (ffill) or backfill (bfill). Got {method}")
                report("api", f"fillna(method={method!r})|`.{'ffill' if method in ('ffill', 'pad') else 'bfill'}()`")
                return filled
            cls.fillna = fillna
        cython = getattr(getattr(getattr(module, "_libs", None), "tslibs", None), "to_offset", None)
        if cython is None:
            return

        def to_offset(freq, *args, **kwargs):
            try:
                return cython(freq, *args, **kwargs)
            except ValueError:
                new = _frequency(freq) if isinstance(freq, str) else None
                if new is None or new == freq:
                    raise
                found = cython(new, *args, **kwargs)
                report("api", f"freq={freq!r}|{new!r}")
                return found
        for name, held in list(sys.modules.items()):
            if name.startswith("pandas") and held is not None and getattr(held, "to_offset", None) is cython:
                try:
                    held.to_offset = to_offset
                except (AttributeError, TypeError):
                    continue

    def mvnun(lower, upper, means, covar, maxpts=None, abseps=1e-6, releps=1e-6):
        """scipy's old `mvn.mvnun`: the multivariate normal's mass between `lower` and `upper`, averaged over the
        columns of `means` (one column: one mean), and its `inform` (0)."""
        import numpy as np
        from scipy.stats import multivariate_normal
        report("api", "scipy.stats.mvn.mvnun|`scipy.stats.multivariate_normal(mean, cov).cdf(upper, lower_limit=lower)`")
        means = np.asarray(means, dtype=float)
        columns = means.T if means.ndim == 2 else means[None, :]
        extra = {"maxpts": int(maxpts)} if maxpts else {}
        value = np.mean([multivariate_normal.cdf(upper, mean=column, cov=covar, allow_singular=True, abseps=abseps,
                                                 releps=releps, lower_limit=lower, **extra) for column in columns])
        return float(value), 0

    def stats(module):
        held = getattr(module, "mvn", None) if module.__name__ == "scipy.stats" else module
        if held is not None:
            try:
                held.mvnun = mvnun
            except (AttributeError, TypeError):
                pass

    _after("numpy", numpy)
    _after("pandas", pandas)
    _after("scipy.stats", stats)
    _after("scipy.stats.mvn", stats)


# -------------------------------------------------------------------------------------------------------- json


_NOTHING = object()


def _plain(value):
    """A numpy or pandas scalar, array or timestamp (or a datetime) as its plain JSON value; `_NOTHING` for any other."""
    kind = type(value)
    top, name = (kind.__module__ or "").split(".")[0], kind.__name__
    if top == "numpy":
        if name in ("datetime64", "timedelta64"):
            return str(value)
        if hasattr(value, "tolist"):
            return value.tolist()
    if top == "pandas":
        if name == "NaTType":
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if name == "Period":
            return str(value)
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    return _NOTHING


def _json(report):
    original = json.JSONEncoder.default

    def default(self, value):
        plain = _plain(value)
        if plain is _NOTHING:
            return original(self, value)
        report("json", type(value).__name__)
        return plain
    json.JSONEncoder.default = default


# ----------------------------------------------------------------------------------------------------- imports


def _imports(names, report):
    import importlib
    for name, where in names.items():
        module, _, attribute = str(where).partition(":")
        try:
            found = importlib.import_module(module)
            setattr(builtins, name, getattr(found, attribute) if attribute else found)
            report("imports", f"{name}|{where}")
        except Exception:  # noqa: BLE001 - a module the sandbox lacks: the NameError stands
            continue


# --------------------------------------------------------------------------------------------- the import hook


_HOOKS = {}


def _after(name, patch):
    """`patch(module)` once `name` is imported (now, where it already is), its own code run first."""
    held = sys.modules.get(name)
    if held is not None:
        try:
            patch(held)
        except Exception:  # noqa: BLE001
            pass
        return
    _HOOKS.setdefault(name, []).append(patch)
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


class _Finder:
    """A meta path finder for the hooked modules alone: the spec the other finders give, its loader's `exec_module`
    followed by the patches."""

    def find_spec(self, name, path=None, target=None):
        patches = _HOOKS.pop(name, None)
        if not patches:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        run = getattr(loader, "exec_module", None)
        if run is None:
            return spec

        def exec_module(module):
            run(module)
            for patch in patches:
                try:
                    patch(module)
                except Exception:  # noqa: BLE001 - a patch that fails leaves the library as it was
                    pass
        try:
            loader.exec_module = exec_module
        except (AttributeError, TypeError):
            pass
        return spec


# ---------------------------------------------------------------------------------------------------- snapshot


SNAPSHOT_ENV = "N11_SNAPSHOT"   # the snapshot's spec: `work`, the House's directory, and `side`, the file written
STATEMENT_NAMES = 8             # the locals the failing statement reads, described at most
OTHER_NAMES = 3                 # then the frame's others, the nearest assigned before the line first
SOURCE_BEFORE, SOURCE_AFTER = 2, 1  # the source lines around the failing statement
SOURCE_LINES = 8                # at most, the failing line kept among them
SOURCE_CHARS = 160              # a source line, at most
ARGUMENTS = 8                   # a library call's arguments described, at most
KEYS = 8                        # a frame's columns, a mapping's keys, at most
LABELS = 3                      # an index's first labels, and its last
SAMPLES = 3                     # the values shown of a series, an array, a sequence or a mapping
TEXT = 80                       # one value's text, at most
MESSAGE = 300                   # the exception's message, at most
CELLS = 10_000_000              # past this many cells, the missing and non-finite ones are not counted
SECONDS = 2.0                   # the hook's own time: past it, no further name is described
PLAIN = (int, float, complex, bool, str, bytes, type(None), datetime.date, datetime.time, datetime.timedelta)
LABELLED = ("Timestamp", "Period", "Timedelta")     # an index's labels of these said by their text, the type named


def _cut(text, limit=TEXT):
    text = str(text)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _shown(value):
    """A value's own text within `TEXT`: a label's or a number's repr (a float's to six figures), a time's text."""
    try:
        if type(value).__name__ in LABELLED:
            return _cut(str(value))
        numpy = _loaded("numpy")
        if isinstance(value, float) or numpy is not None and isinstance(value, numpy.floating):
            return format(float(value), ".6g")
        if numpy is not None and isinstance(value, (numpy.integer, numpy.bool_)):
            return repr(value.item())
        return _cut(repr(value))
    except Exception:  # noqa: BLE001 - a value whose repr fails is named by its type
        return f"<{type(value).__name__}>"


def _kind(cls):
    module = getattr(cls, "__module__", "") or ""
    return cls.__qualname__ if module in ("builtins", "__main__") else f"{module}.{cls.__qualname__}"


def _loaded(name):
    module = sys.modules.get(name)
    return module if module is not None and hasattr(module, "__dict__") else None


def _index(index):
    """An index's type, dtype, length and its first and last labels."""
    found = {"type": type(index).__name__, "dtype": str(getattr(index, "dtype", "")), "length": len(index)}
    found["first"] = [_shown(label) for label in list(index[:LABELS])]
    if len(index) > LABELS:
        found["last"] = [_shown(label) for label in list(index[max(LABELS, len(index) - LABELS):])]
    return found


def _cells(found, value, numbers):
    """`found` with the missing cells (pandas' `isna`, numpy's NaN) and the infinite numbers of `value` counted,
    within `CELLS`; `numbers`, its float or complex part as an array."""
    numpy = _loaded("numpy")
    size = int(getattr(value, "size", 0) or 0)
    if numpy is None or size > CELLS:
        return found
    try:
        if hasattr(value, "isna"):
            found["missing"] = int(numpy.asarray(value.isna()).sum())
        elif numbers is not None:
            found["missing"] = int(numpy.isnan(numbers).sum())
        if numbers is not None and numbers.size:
            found["infinite"] = int(numpy.isinf(numbers).sum())
    except Exception:  # noqa: BLE001
        pass
    return found


def described(value):
    """What the snapshot says of one value: its type and, by its kind, its shape or length, dtype, index, columns or
    keys, missing and infinite cells and a few values. A part that fails to read is left out."""
    found = {"type": type(value).__name__}
    pandas, numpy = _loaded("pandas"), _loaded("numpy")
    try:
        if pandas is not None and isinstance(value, pandas.DataFrame):
            rows, columns = value.shape
            found["shape"] = [rows, columns]
            found["columns"] = [f"{_shown(name)} {dtype}" for name, dtype in value.iloc[:1, :KEYS].dtypes.items()]
            found["index"] = _index(value.index)
            if value.size <= CELLS:
                floats = value.select_dtypes(include=["floating", "complex"])
                _cells(found, value, floats.to_numpy() if floats.shape[1] else None)
            if rows:
                found["first"] = [f"{_shown(name)}: {_shown(value.iloc[0, at])}"
                                  for at, name in enumerate(value.columns[:KEYS])]
        elif pandas is not None and isinstance(value, pandas.Series):
            found.update(length=len(value), dtype=str(value.dtype), index=_index(value.index))
            if value.name is not None:
                found["named"] = _shown(value.name)
            if value.size <= CELLS:
                _cells(found, value, value.to_numpy() if value.dtype.kind in "fc" else None)
            found["values"] = [_shown(item) for item in list(value.iloc[:SAMPLES])]
        elif pandas is not None and isinstance(value, pandas.Index):
            found.update(_index(value))
        elif numpy is not None and isinstance(value, numpy.ndarray):
            found.update(shape=list(value.shape), dtype=str(value.dtype))
            _cells(found, value, value if value.dtype.kind in "fc" else None)
            found["values"] = [_shown(item) for item in itertools.islice(value.flat, SAMPLES)]
        elif isinstance(value, dict):
            found["length"] = len(value)
            found["keys"] = [_shown(key) for key in itertools.islice(value, KEYS)]
            found["values"] = [f"{_shown(key)}: {_shown(item)}"
                               for key, item in itertools.islice(value.items(), SAMPLES)]
        elif isinstance(value, (list, tuple, set, frozenset)):
            found["length"] = len(value)
            items = itertools.islice(value, SAMPLES) if not isinstance(value, (set, frozenset)) else ()
            found["values"] = [_shown(item) for item in items]
        elif isinstance(value, PLAIN) or numpy is not None and isinstance(value, numpy.generic) \
                or type(value).__name__ in LABELLED:
            if isinstance(value, (str, bytes)):
                found["length"] = len(value)
            found["value"] = _shown(value)
        elif hasattr(type(value), "__len__"):
            found["length"] = len(value)
    except Exception:  # noqa: BLE001 - what was read stands
        pass
    return found


def _skipped(value):
    """A module, a function, a method or a class: no state of the run's to say."""
    import types
    return isinstance(value, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType,
                              type))


def _statement(tree, line):
    """The innermost statement holding `line`, and the parts the line can be in: a compound statement's header alone
    where the line is above its body."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and node.lineno <= line <= (node.end_lineno or node.lineno):
            if best is None or (node.end_lineno or node.lineno) - node.lineno <= \
                    (best.end_lineno or best.lineno) - best.lineno:
                best = node
    if best is None:
        return None, []
    body = getattr(best, "body", None)
    if isinstance(body, list) and body and line < body[0].lineno:
        parts = [value for name, value in ast.iter_fields(best)
                 if name not in ("body", "orelse", "handlers", "finalbody") and isinstance(value, ast.AST)]
        parts += [item for name, value in ast.iter_fields(best) if name in ("items", "decorator_list")
                  and isinstance(value, list) for item in value if isinstance(item, ast.AST)]
        return best, parts
    return best, [best]


def _reads(parts):
    """The names the parts read, in source order, once each."""
    found = {}
    for part in parts:
        for node in ast.walk(part):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                found.setdefault(node.id, (node.lineno, node.col_offset))
    return sorted(found, key=found.get)


def _scope(tree, code):
    """The definition the frame's code runs, else the module."""
    if code.co_name == "<module>":
        return tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == code.co_name and \
                code.co_firstlineno in (node.lineno, *(each.lineno for each in node.decorator_list)):
            return node
    return tree


def _assigned(scope, line):
    """The names the scope binds before `line`, the nearest first; a definition's parameters last."""
    found = {}
    for node in ast.walk(scope):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.AsyncFor, ast.With,
                             ast.AsyncWith)) and node.lineno < line:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if hasattr(node, "target") \
                else [item.optional_vars for item in node.items if item.optional_vars is not None]
            for target in targets:
                for each in ast.walk(target):
                    if isinstance(each, ast.Name):
                        found[each.id] = max(found.get(each.id, 0), node.lineno)
    names = sorted(found, key=lambda name: -found[name])
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = scope.args
        names += [arg.arg for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
                  if arg.arg not in found]
    return names


def _source(lines, line, first, last):
    """The source lines around the failing statement (`first` to `last`), the failing line among them."""
    start, end = max(1, first - SOURCE_BEFORE), min(len(lines), last + SOURCE_AFTER)
    if end - start + 1 > SOURCE_LINES:
        start = max(1, line - SOURCE_LINES // 2)
        end = min(len(lines), start + SOURCE_LINES - 1)
    while start < line and not lines[start - 1].strip():
        start += 1
    while end > line and not lines[end - 1].strip():
        end -= 1
    return [[number, _cut(lines[number - 1].rstrip("\n"), SOURCE_CHARS)] for number in range(start, end + 1)]


def _library(frame):
    name = frame.f_globals.get("__name__", "") or ""
    return name == "qf" or name.startswith("qf.")


def called(frame, known=None):
    """The library call a frame of the library's runs: its function (the guard's wrapped one where the frame is the
    guard's) and the actual arguments as its parameters hold them, the wrapped function's names given to the
    positional ones; an argument that is a local already described (`known`: its id to its name) named so."""
    code, held = frame.f_code, frame.f_locals
    inner = held.get("function") if "function" in code.co_freevars or "function" in code.co_varnames else None
    inner = inner if callable(inner) and str(getattr(inner, "__module__", "")).startswith("qf") else None
    name = getattr(inner, "__name__", None) or code.co_name
    named = []
    if inner is not None:
        try:
            import inspect
            named = list(inspect.signature(inner).parameters)
        except (TypeError, ValueError):
            named = []
    count = code.co_argcount + code.co_kwonlyargcount
    arguments = [(each, held[each]) for each in code.co_varnames[:count] if each in held]
    star = code.co_varnames[count] if code.co_flags & 0x04 else None
    starstar = code.co_varnames[count + bool(star)] if code.co_flags & 0x08 else None
    if star and star in held:
        arguments += [(named[k] if k < len(named) else f"{star}[{k}]", item) for k, item in enumerate(held[star])]
    if starstar and starstar in held:
        arguments += list(held[starstar].items())
    known = known or {}
    return {"function": f"qf.{name}", "arguments": [{"name": each, "same": known[id(item)]} if id(item) in known else
                                                    {"name": each, **described(item)}
                                                    for each, item in arguments[:ARGUMENTS]]}


def snapshot(error, tb, work):
    """The state at an uncaught exception (`error`, its traceback `tb`), the House's files under `work`."""
    started = time.monotonic()
    taken = {"error": {"type": _kind(type(error)), "message": _cut(str(error), MESSAGE)}}
    frames = []
    while tb is not None:
        frames.append((tb.tb_frame, tb.tb_lineno))
        tb = tb.tb_next
    root = os.path.realpath(work) + os.sep
    users = [k for k, (frame, _) in enumerate(frames) if os.path.realpath(frame.f_code.co_filename).startswith(root)]
    if not users:
        taken["skipped"] = "no frame under work/"
        return taken
    at = users[-1]
    frame, line = frames[at]
    path = os.path.realpath(frame.f_code.co_filename)
    with _open(path, encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()
    try:
        tree = ast.parse("\n".join(lines))
    except (SyntaxError, ValueError):
        tree = None
    node, parts = _statement(tree, line) if tree is not None else (None, [])
    first, last = (node.lineno, node.end_lineno or node.lineno) if node is not None and parts == [node] else \
        (line, line)
    taken["frame"] = {"file": "work/" + path[len(root):], "line": line, "function": frame.f_code.co_name,
                      "source": _source(lines, line, first, last) if lines else []}
    held, around = frame.f_locals, frame.f_globals
    names, known = [], {}
    for name in _reads(parts):
        if len(names) >= STATEMENT_NAMES or time.monotonic() - started > SECONDS:
            break
        value = held[name] if name in held else around[name] if name in around else None
        if (name in held or name in around) and not _skipped(value):
            names.append({"name": name, "read": "statement", **described(value)})
            if not isinstance(value, PLAIN):        # a small number or a short text is shared by unrelated names
                known.setdefault(id(value), name)
    said = {each["name"] for each in names} | set(_reads(parts))
    others = 0
    for name in _assigned(_scope(tree, frame.f_code), line) if tree is not None else list(held)[::-1]:
        if others >= OTHER_NAMES or time.monotonic() - started > SECONDS:
            break
        if name in said or name not in held or name.startswith("__") or _skipped(held[name]):
            continue
        names.append({"name": name, "read": "frame", **described(held[name])})
        if not isinstance(held[name], PLAIN):
            known.setdefault(id(held[name]), name)
        others += 1
    taken["names"] = names
    library = next((frame for frame, _ in frames[at + 1:] if _library(frame)), None)
    if library is not None and time.monotonic() - started <= SECONDS:
        taken["call"] = called(library, known)
    return taken


def _written(side, taken):
    """`taken` written whole to the side file: a temporary beside it, then moved."""
    with _open(side + ".part", "w", encoding="utf-8") as handle:
        handle.write(json.dumps(taken, default=str))
    _replace(side + ".part", side)


def _snapshot_start():
    """The hook installed where the spec is named: the interpreter's hook first; then a mark that the hook ran (the
    exception's type alone), so a hook that fails after it leaves the mark and the harness counts it dropped, where a
    side file absent means no uncaught exception reached the hook (a script that printed its own traceback); then the
    snapshot over the mark, a SyntaxError's left unwritten. Nothing raised."""
    spec_path = os.environ.get(SNAPSHOT_ENV, "")
    if not spec_path:
        return
    with _open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    work, side = str(spec["work"]), str(spec["side"])
    previous = sys.excepthook

    def hook(kind, error, tb):
        try:
            previous(kind, error, tb)
        finally:
            try:
                _written(side, {"raised": kind.__name__})
                _written(side, {"skipped": kind.__name__} if issubclass(kind, SyntaxError) else
                         snapshot(error, tb, work))
            except BaseException:  # noqa: BLE001 - a snapshot that fails is dropped; the run's exit stands
                pass
    sys.excepthook = hook


try:
    _start()
except Exception:  # noqa: BLE001 - the shim never costs the run
    pass
try:
    _snapshot_start()
except Exception:  # noqa: BLE001 - nor does the snapshot
    pass
_chained = os.environ.get("N11_SHIM_CHAIN", "")
if _chained:        # the interpreter's own sitecustomize, which this file's name hides, run after it
    try:
        with io.open(_chained, encoding="utf-8") as _handle:
            exec(compile(_handle.read(), _chained, "exec"), {"__name__": "sitecustomize", "__file__": _chained})
    except Exception:  # noqa: BLE001
        pass
