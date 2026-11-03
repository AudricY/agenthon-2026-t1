#!/bin/sh
# Compares an image built from this repository with the image submitted to the Final: every file under /opt/agent
# and the solve wrapper by sha256, then the Python packages. Exits nonzero when the agent's files differ.
#   scripts/verify_image.sh [BUILT_IMAGE]      # default agenthon-2026-t1:n11
set -eu
FINAL=ghcr.io/audricy/agenthon-t4@sha256:8dba17e4236d7407eda9937afd7d69d3b5850e3588244f5ceb09bdae92d3e83f
BUILT=${1:-agenthon-2026-t1:n11}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

docker pull --quiet --platform linux/amd64 "$FINAL" >/dev/null
for name in final built; do
    if [ "$name" = final ]; then image=$FINAL; else image=$BUILT; fi
    docker run --rm --network none --entrypoint sh "$image" -c \
        'cd /opt/agent && find . -type f | LC_ALL=C sort | xargs sha256sum && sha256sum /usr/local/bin/solve' \
        > "$WORK/$name.files"
    docker run --rm --network none --entrypoint pip "$image" freeze --all 2>/dev/null | LC_ALL=C sort \
        > "$WORK/$name.packages"
done

echo "agent files: $(wc -l < "$WORK/final.files") in the Final image, $(wc -l < "$WORK/built.files") built"
if diff -u "$WORK/final.files" "$WORK/built.files"; then
    echo "agent files: identical"
else
    echo "agent files: DIFFER" >&2
    exit 1
fi
if diff -u "$WORK/final.packages" "$WORK/built.packages"; then
    echo "python packages: identical"
else
    echo "python packages: differ (above); the kit's base image pins no versions, so a later build can resolve newer ones"
fi
