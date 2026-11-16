# Train cloth folding with 150 or 75 episodes

Run these commands from `openpi/` in the installed training environment. Set
`HF_LEROBOT_HOME` to the parent of the copied `fold_cloth_real/` directory.
For example, a copy at `/data/lerobot_data/fold_cloth_real` uses:

```bash
export HF_LEROBOT_HOME=/data/lerobot_data
```

The directory must contain the LeRobot v2.1 `meta/`, `data/`, and `videos/`
directories. Both configs read that same local copy.

Compute normalization separately for each selection, then train:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_fold_cloth_real_150
uv run scripts/train.py pi05_fold_cloth_real_150 --exp-name full150_seed42

uv run scripts/compute_norm_stats.py --config-name pi05_fold_cloth_real_75
uv run scripts/train.py pi05_fold_cloth_real_75 --exp-name random75_seed42
```

Both configs use pi0.5 full fine-tuning from
`gs://openpi-assets/checkpoints/pi05_base/params`, 20,000 optimizer steps, global
batch size 64, training seed 42, and two-device FSDP. On a single suitable GPU,
append `--fsdp-devices 1` to each training command. Keep any batch size or step
changes identical across the two runs.

The full selection is episode IDs 0 through 149. The smaller selection is 75
whole episodes sampled without replacement using `random.Random(42)`, then sorted.
The subset seed is fixed in the config independently of the training seed.
Inspect and save the exact IDs before a run:

```bash
uv run python - <<'PY'
from openpi.training.config import get_config
for name in ("pi05_fold_cloth_real_150", "pi05_fold_cloth_real_75"):
    print(name, list(get_config(name).data.base_config.episodes))
PY
```

Training and normalization use the same episode selection. Each config stores its
own statistics under `assets/<config-name>/fold_cloth_real/` and checkpoints under
`checkpoints/<config-name>/<exp-name>/`. The reader loads the full dataset and
selects frame indices for the chosen whole episodes. This preserves action-chunk
boundaries with the pinned LeRobot reader's original episode indexing.

The camera mapping uses `cam_high`, `cam_wrist_left`, and `cam_wrist_right`, with
the prompt "Fold the tablecloth in half." The configs use the existing ALOHA
joint/gripper adaptation and delta joint actions. Before training, verify that
the Trossen recordings use the expected joint order, signs, units, gripper ranges,
and absolute action representation. Robot compatibility has not been established
by these configs.

Compare autonomous task success under the same evaluation conditions. Equal
optimizer steps hold compute approximately fixed; the 75-episode run sees each
selected episode more often. A single subset and training seed provide an initial
comparison, not an estimate of variation across seeds.
