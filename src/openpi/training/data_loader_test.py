import dataclasses
import random
from typing import ClassVar

import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


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


def test_episode_subset_preserves_action_boundaries(monkeypatch):
    class Metadata:
        fps = 30
        episodes: ClassVar = {0: {"length": 2}, 1: {"length": 3}, 2: {"length": 2}}

    class EpisodeDataset:
        episode_data_index: ClassVar = {"from": [0, 2, 5], "to": [2, 5, 7]}

        def __init__(self, repo_id, *, delta_timestamps):
            assert repo_id == "test"
            assert delta_timestamps == {"action": [0, 1 / 30, 2 / 30]}

        def __getitem__(self, index):
            episode = next(ep for ep, end in enumerate(self.episode_data_index["to"]) if index < end)
            end = self.episode_data_index["to"][episode]
            return {"episode": episode, "actions": [min(index + delta, end - 1) for delta in range(3)]}

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda _: Metadata())
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", EpisodeDataset)
    data_config = _config.DataConfig(repo_id="test", episodes=(0, 2), action_sequence_keys=("action",))
    dataset = _data_loader.create_torch_dataset(data_config, 3, pi0_config.Pi0Config())

    assert len(dataset) == 4
    assert [dataset[index]["episode"] for index in range(len(dataset))] == [0, 0, 2, 2]
    assert dataset[1]["actions"] == [1, 1, 1]
    assert dataset[2]["actions"] == [5, 6, 6]
    assert dataset[3]["actions"] == [6, 6, 6]


def test_fold_cloth_configs_share_training_settings():
    full = _config.get_config("pi05_fold_cloth_real_150")
    half = _config.get_config("pi05_fold_cloth_real_75")
    assert full.data.base_config.episodes == tuple(range(150))
    assert len(half.data.base_config.episodes) == len(set(half.data.base_config.episodes)) == 75
    assert set(half.data.base_config.episodes) < set(full.data.base_config.episodes)
    assert half.data.base_config.episodes == tuple(sorted(random.Random(42).sample(range(150), 75)))
    assert dataclasses.replace(full.data, base_config=half.data.base_config) == half.data
    assert full.assets_dirs != half.assets_dirs
    assert full.data.assets.assets_dir is None
    assert half.data.assets.assets_dir is None
    assert dataclasses.replace(full, name=half.name, data=half.data) == half
