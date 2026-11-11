## This repository
- `docs/architecture.md` maps the modules and defines the code's words. A module docstring opens with what the
  module does, who calls it and where to start; keep that header true when a caller changes.
- `agent/qf/` is the Final image's method library, byte for byte, and its docstrings build the model's prompt index.
  Leave it unchanged; `scripts/verify_image.sh` checks it, the index, the tokenizer and the packages against the image.
- The rest of `agent/` is the Final image's agent with the code its settings never run removed. A change there must
  leave every request it sends unchanged; the tag `final-image` holds the original.
- The reproducibility claim rests on `docker/base.lock` and the hashes in `scripts/fetch_vendor.sh`; change them only
  with a rebuild that `verify_image.sh` passes.
