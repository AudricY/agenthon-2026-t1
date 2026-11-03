"""What every Track 1 agent shares (N0 first, L1 next): the unit read with its canaries out (`unit`, `canary`), the
data digest (`digest`), the path map (`paths`), the model client under the agent's own cap (`house`), the script
runner (`runner`), the desk's trace (`trace`, off in the image) and the guard (`guard`: the write lock, the watchdog, the close that scrubs the output tree, the
step log). Standard library only, so it runs in any image beside model.py; pyarrow and openpyxl are used where the
image has them.
"""
