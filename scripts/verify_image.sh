#!/bin/sh
# Compares an image built from this repository with the image submitted to the Final. The method library
# (/opt/agent/qf), the library's index and the tokenizer that the prompts are built from, the solve wrapper and the
# Python packages must be identical. The agent's own code is the Final's with the code its settings never run removed,
# so its files are counted, not compared; the tag final-image holds the Final's code, and its own copy of this script
# compares every file. Exits nonzero when a file that must match differs.
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
        'cd /opt/agent && find qf -type f | LC_ALL=C sort | xargs sha256sum &&
         sha256sum n11/index.txt n11/tokenizer.json /usr/local/bin/solve' > "$WORK/$name.same"
    docker run --rm --network none --entrypoint sh "$image" -c \
        'cd /opt/agent && find . -type f -name "*.py" -not -path "./qf/*" | xargs cat | wc -l' > "$WORK/$name.lines"
    docker run --rm --network none --entrypoint pip "$image" freeze --all 2>/dev/null | LC_ALL=C sort \
        > "$WORK/$name.packages"
done

if diff -u "$WORK/final.same" "$WORK/built.same"; then
    echo "library, index, tokenizer and solve wrapper: identical ($(wc -l < "$WORK/built.same") files)"
else
    echo "library, index, tokenizer or solve wrapper: DIFFER" >&2
    exit 1
fi
if diff -u "$WORK/final.packages" "$WORK/built.packages"; then
    echo "python packages: identical"
else
    echo "python packages: differ (above); the kit's base image pins no versions, so a later build can resolve newer ones"
fi
echo "agent code: $(cat "$WORK/final.lines") lines of Python in the Final image, $(cat "$WORK/built.lines") built"
