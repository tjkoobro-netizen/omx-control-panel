#!/usr/bin/env python3
"""lerobot-teleoperate with the OMX safe start (same arguments as lerobot-teleoperate).

The follower first lifts to the saved start pose, closes the gripper and then moves slowly to the
leader pose with limited motor output, instead of jumping there at full power. See omx_safe_start.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omx_safe_start  # noqa: E402

omx_safe_start.install()

from lerobot.scripts import lerobot_teleoperate  # noqa: E402

if __name__ == "__main__":
    lerobot_teleoperate.main()
