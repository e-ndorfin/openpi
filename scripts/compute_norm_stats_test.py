import numpy as np

from scripts import compute_norm_stats


def test_select_norm_keys_drops_images_and_metadata():
    state = np.arange(3)
    actions = np.arange(6).reshape(2, 3)

    result = compute_norm_stats.SelectNormKeys()(
        {
            "state": state,
            "actions": actions,
            "image": {"camera": np.zeros((1, 1, 3))},
            "prompt": "insert the wire",
        }
    )

    assert result.keys() == {"state", "actions"}
    np.testing.assert_array_equal(result["state"], state)
    np.testing.assert_array_equal(result["actions"], actions)
