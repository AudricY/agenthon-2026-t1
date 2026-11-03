# Pins the kit's sandbox base to the Python packages under the Final image (docker/base.lock, from its `pip freeze`).
# The kit's own Dockerfile asks only for minimum versions, so a fresh build resolves newer ones. Packages the lock
# does not list are removed.
ARG KIT_BASE
FROM ${KIT_BASE}
COPY docker/base.lock /tmp/base.lock
RUN pip install --no-cache-dir --no-deps -r /tmp/base.lock && \
    pip freeze --all | cut -d= -f1 | LC_ALL=C sort > /tmp/installed && \
    cut -d= -f1 /tmp/base.lock | LC_ALL=C sort > /tmp/wanted && \
    extras=$(LC_ALL=C comm -23 /tmp/installed /tmp/wanted) && \
    if [ -n "$extras" ]; then pip uninstall -y $extras; fi && \
    rm /tmp/base.lock /tmp/installed /tmp/wanted
