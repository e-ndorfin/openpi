"""Compatibility imports for LeRobot dataset format versions."""

try:
    # LeRobot >= 0.4 (dataset format v3.0).
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
except ImportError:
    # Legacy LeRobot (dataset format v2.1).
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

__all__ = ["LeRobotDataset", "LeRobotDatasetMetadata"]
