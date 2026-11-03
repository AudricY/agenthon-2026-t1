## Behaviour
- self drive commits and auto push
- be disciplined when adding code comments.
- proactively maintain Markdown docs with progressive disclosure: breadth at the root, depth at the leaves.
- Optimize for long-term product and codebase health rather than minimizing the immediate diff.
	- Treat sensible cleanup/refactoring as an upfront investment, not inherently as a cost to avoid.

## This repository
- `agent/` is the Final image's `/opt/agent`, byte for byte. Leave it unchanged; `scripts/verify_image.sh` checks it
  against the image.
- The reproducibility claim rests on `docker/base.lock` and the hashes in `scripts/fetch_vendor.sh`; change them only
  with a rebuild that `verify_image.sh` passes.
