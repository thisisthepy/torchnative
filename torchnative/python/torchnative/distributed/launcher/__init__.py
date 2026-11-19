"""`torchnative.distributed.launcher`: the programmatic launcher (D7, S6.11).

`start_method="spawn"` starts one process per rank; `start_method="thread"`
starts one thread per rank in this process (D5, issue #281).
"""

from .api import (ChildFailedError, LaunchConfig,
                  LauncherRefused, elastic_launch, rank_info)

__all__ = ["ChildFailedError", "LaunchConfig",
           "LauncherRefused", "elastic_launch", "rank_info"]
