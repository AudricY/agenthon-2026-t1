"""The parts that know nothing of the loop: reading the unit, running a script, checking and guarding the output.
agent/n11 calls these; docs/architecture.md maps them.

What it holds
-------------
What the agent shares across its parts: the unit read with its canaries out (`unit`, `canary`), the data digest
(`digest`), the path map (`paths`), the output checks (`contract`, `selfreport`), whether a script uses the method
library (`qfuse`), the model client under the agent's own cap (`house`), the script runner (`runner`) and the guard
(`guard`: the write lock, the watchdog, the close that scrubs the output tree, the step log). Standard library only,
beside model.py; pyarrow where the image has it.
"""
