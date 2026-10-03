"""Adds "tensorrt" to Keras' _PLUGGABLE_BACKENDS in a keras checkout.

Needed until keras-team/keras lists the backend. Usage:
    python allow_backend.py path/to/keras/src/backend/config.py
"""

import re
import sys

path = sys.argv[1]
src = open(path).read()
match = re.search(r"_PLUGGABLE_BACKENDS = frozenset\(\[(.*?)\]\)", src, re.S)
if match is None:
    sys.exit(f"_PLUGGABLE_BACKENDS not found in {path}")
if '"tensorrt"' not in match.group(1):
    names = match.group(1).rstrip().rstrip(",")
    src = src.replace(
        match.group(0),
        f'_PLUGGABLE_BACKENDS = frozenset([{names}, "tensorrt"])',
    )
    open(path, "w").write(src)
print("ok:", re.search(r"_PLUGGABLE_BACKENDS = .*", src).group(0))
