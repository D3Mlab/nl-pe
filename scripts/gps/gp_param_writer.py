import os
import sys
sys.path.append('src')
from pathlib import Path
import itertools
import yaml

# ============================================================
# Hardcoded experiment constants
# ============================================================

BASE_EXP_DIR = Path(
    "trials/gps/exact/inference_only_speed/batching_study/fast"
)

BASE_CONFIG = {
    "logging": {"level": "INFO"},
    "n_obs": 10,
    "n_unobs": 1000,
    "d": 1,
    "gt_func": "sin",
    "device": "cuda",
    "fast_pred": True,
    "inf_batch_size": None,
}

# ============================================================
# Grid definition
# The variable name MUST match the config key
# ============================================================

n_obs = [10000]
n_unobs = [100000, 1000000]
# You can add:
# d = [1, 5, 10]
d = [1000]
inf_batch_size = [100000,10000,1000]

GRID_PARAMS = {
    "n_obs": n_obs,
    "n_unobs": n_unobs,
    "d": d,
    "inf_batch_size": inf_batch_size,
}

# ============================================================
# Helper: build experiment directory path
# ============================================================

def build_exp_dir(base_dir, config):
    """s
    Path formula:
    <base>/<n_obs>obs/<n_unobs>unobs/d<d>
    """
    return (
        base_dir
        / f"{config['n_obs']}obs"
        / f"{config['n_unobs']}unobs"
        / f"d{config['d']}"
        / f"b{config['inf_batch_size']}"
    )

# ============================================================
# Main generation logic
# ============================================================

def main():
    # Cartesian product of grid values
    keys = GRID_PARAMS.keys()
    values = GRID_PARAMS.values()

    for combo in itertools.product(*values):
        # Start from base config
        config = BASE_CONFIG.copy()

        # Overwrite base config entries with grid values
        for k, v in zip(keys, combo):
            config[k] = v

        # Determine experiment directory
        exp_dir = build_exp_dir(BASE_EXP_DIR, config)

        # Ensure directory exists
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Write config.yaml
        config_path = exp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        print(f"Created: {config_path}")

    print("\n✅ All configs generated.")

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
