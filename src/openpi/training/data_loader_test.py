import dataclasses
import types

import jax
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _FakeTable:
    def __init__(self, names: list[str]):
        self._names = names
        self.iloc = self

    def __getitem__(self, index: int):
        return types.SimpleNamespace(name=self._names[index])


class _FakeHFDataset:
    def __getitem__(self, index: int) -> dict:
        return {
            "episode_index": torch.tensor(7),
            "index": torch.tensor(100 + index),
            "task_index": torch.tensor(0),
            "subtask_index": torch.tensor(0),
            "timestamp": torch.tensor(index / 30),
        }


class _FakeLeRobotDataset:
    def __init__(self):
        self.meta = types.SimpleNamespace(
            video_keys=["observation.images.camera"],
            tasks=_FakeTable(["insert the wire"]),
            subtasks=_FakeTable(["align the wire"]),
        )
        self.features = {"subtask_index": {}}
        self.delta_indices = {"action": [0, 1]}
        self.hf_dataset = _FakeHFDataset()
        self.loaded = False
        self.query_calls = []

    def _ensure_hf_dataset_loaded(self):
        self.loaded = True

    def _get_query_indices(self, absolute_index: int, episode_index: int):
        self.query_calls.append((absolute_index, episode_index))
        return {"action": [absolute_index, absolute_index + 1]}, {"action_is_pad": torch.tensor([False, True])}

    def _query_hf_dataset(self, query_indices: dict):
        return {"action": torch.tensor(query_indices["action"], dtype=torch.float32).unsqueeze(-1)}

    def _query_videos(self, *_args, **_kwargs):
        raise AssertionError("normalization must not query videos")

    def __len__(self):
        return 3


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_lerobot_no_video_dataset_preserves_non_video_item_semantics():
    source = _FakeLeRobotDataset()
    dataset = _data_loader.LeRobotNoVideoDataset(source)

    item = dataset[2]

    assert len(dataset) == 3
    assert source.loaded
    assert source.query_calls == [(102, 7)]
    torch.testing.assert_close(item["action"], torch.tensor([[102.0], [103.0]]))
    torch.testing.assert_close(item["action_is_pad"], torch.tensor([False, True]))
    torch.testing.assert_close(item["observation.images.camera"], torch.zeros((3, 1, 1)))
    assert item["task"] == "insert the wire"
    assert item["subtask"] == "align the wire"


def test_lerobot_no_video_dataset_supports_multiple_workers():
    dataset = _data_loader.LeRobotNoVideoDataset(_FakeLeRobotDataset())
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        multiprocessing_context="spawn",
    )

    batch = next(iter(loader))

    assert batch["action"].shape == (2, 2, 1)
    assert batch["observation.images.camera"].shape == (2, 3, 1, 1)


def test_create_torch_dataset_skips_video_only_when_requested(monkeypatch):
    metadata = types.SimpleNamespace(fps=30, tasks={})
    source = _FakeLeRobotDataset()
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda *_args, **_kwargs: metadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", lambda *_args, **_kwargs: source)
    data_config = _config.DataConfig(repo_id="test")

    regular_dataset = _data_loader.create_torch_dataset(data_config, action_horizon=2, model_config=None)
    norm_dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=2,
        model_config=None,
        skip_video_decoding=True,
    )

    assert regular_dataset is source
    assert isinstance(norm_dataset, _data_loader.LeRobotNoVideoDataset)


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
