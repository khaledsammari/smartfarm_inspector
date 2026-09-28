"""Setup for smartfarm_inspector."""

from glob import glob

from setuptools import find_packages, setup

PACKAGE_NAME = "smartfarm_inspector"

setup(
    name=PACKAGE_NAME,
    version="0.2.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE_NAME]),
        ("share/" + PACKAGE_NAME, ["package.xml"]),
        ("share/" + PACKAGE_NAME + "/config", glob("config/*.json") + glob("config/*.yaml")),
        ("share/" + PACKAGE_NAME + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="Smart Farm Team",
    maintainer_email="team@example.com",
    description="LLM -> ROS 2 mission bridge for autonomous farm inspection.",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "inspection_node = smartfarm.ros.inspection_node:main",
        ],
    },
)
