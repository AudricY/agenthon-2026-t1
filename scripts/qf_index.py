"""The method library's index as n11's system prompt shows it (agent/n11/library.py `Library.index`), written at
build time: the Dockerfile's `n11-index` stage runs it into /opt/agent/n11/index.txt. The index is a scan of whatever
agent/qf holds, so a new module appears in it with no edit here.

    python scripts/qf_index.py [--qf DIR] [--out FILE]

`--qf` the package (agent/qf by default); `--out` the file (stdout by default). The index is fitted in the unit the
agent's bounds are in (agent/n11/measure.py: the model's tokens where the tokenizer loads, as in the image's stage,
else bytes), and the script prints its tokens against `INDEX_TOKENS` (or its bytes against `INDEX_BYTES`), its bytes,
its functions and shelves to stderr.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from n11 import library, measure  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qf", type=Path, default=ROOT / "agent" / library.PACKAGE)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    found = library.scan(args.qf)
    text = found.index() + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    size, tokens = len(text.encode("utf-8")), measure.UNIT.tokenizer is not None
    said = (f"{library.measured(text):,} tokens of {library.INDEX_TOKENS:,} ({size:,} B)" if tokens else
            f"{size:,} B of {library.INDEX_BYTES:,}, in {measure.UNIT.said()}")
    print(f"qf_index: {said}; {found.whole(text.rstrip(chr(10)))} of {len(found.functions)} function(s) whole on "
          f"{len(found.shelves)} shelves", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
