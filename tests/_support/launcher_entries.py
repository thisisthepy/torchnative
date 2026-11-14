"""Entrypoints for tests/distributed/test_launcher.py.

Their own module, not the suite: `multiprocessing`'s spawn context re-imports
the entrypoint's module in every child, and the suite imports the shim's test
helpers, which a child has no reason to load.
"""

import os
import time


def entry(scale):
    return {"rank": int(os.environ["RANK"]), "local": int(os.environ["LOCAL_RANK"]),
            "world": int(os.environ["WORLD_SIZE"]), "value": scale * int(os.environ["RANK"])}


def entry_fails(_):
    if os.environ["RANK"] == "1":
        raise ValueError("boom from rank 1")
    time.sleep(60)
