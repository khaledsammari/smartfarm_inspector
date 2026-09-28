"""Locate the config directory, whether running installed or from source.

setup.py installs the configs into share/smartfarm_inspector/config/, but a
module-relative default resolves to site-packages/config/ after install --
a directory that does not exist. The node then silently ran with empty crop
and inspection databases and no few-shot examples.

Resolution order:
  1. SMARTFARM_CONFIG_DIR environment variable
  2. the ament share directory (installed case)
  3. <repo>/config (source checkout case)
"""

from __future__ import annotations

import os
from pathlib import Path

SOURCE_CONFIG = Path(__file__).resolve().parent.parent / "config"


def config_dir() -> Path:
    """Return the first config directory that actually exists."""
    env = os.environ.get("SMARTFARM_CONFIG_DIR")
    if env and Path(env).is_dir():
        return Path(env)

    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory("smartfarm_inspector")) / "config"
        if share.is_dir():
            return share
    except Exception:  # noqa: BLE001 - not in a ROS environment, fine
        pass

    return SOURCE_CONFIG


def config_file(name: str) -> Path:
    return config_dir() / name
