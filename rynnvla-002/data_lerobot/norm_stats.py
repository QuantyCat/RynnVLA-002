import os
import re
import ast
from functools import lru_cache

import numpy as np


DEFAULT_ACTION_MIN = np.array([
    -0.13845688,
    -0.17819679,
    -0.19286394,
    -0.17750373,
    -0.28787115,
     0.00000000,
], dtype=np.float32)

DEFAULT_ACTION_MAX = np.array([
    0.16925158,
    0.17430055,
    0.16337049,
    0.20580864,
    0.29125005,
    0.48936170,
], dtype=np.float32)

DEFAULT_STATE_MIN = np.array([
    -1.20063305,
    -1.75530255,
    -0.22938614,
    -0.27925268,
    -1.25049961,
     0.00673854,
], dtype=np.float32)

DEFAULT_STATE_MAX = np.array([
    1.13925886,
    0.42808515,
    1.68702376,
    1.50520265,
    1.23362172,
    0.48517519,
], dtype=np.float32)

ACTION_STATS_ENV = "RYNNVLA_ACTION_STATS_FILE"
STATE_STATS_ENV = "RYNNVLA_STATE_STATS_FILE"
ACTION_NORM_SCALES_ENV = "RYNNVLA_ACTION_NORM_SCALES"


def _parse_stats_file(path: str):
    mins = []
    maxs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "|" not in line:
                continue
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if len(nums) < 5:
                continue
            # Expected format per row:
            # Dim/维度 <idx> | <min> | <max> | <q01> | <q99>
            mins.append(float(nums[1]))
            maxs.append(float(nums[2]))
    if len(mins) != 6 or len(maxs) != 6:
        raise ValueError(f"Expected 6 min/max rows in stats file, got {len(mins)} rows: {path}")
    return np.array(mins, dtype=np.float32), np.array(maxs, dtype=np.float32)


def _load_stats(path_env: str, default_min: np.ndarray, default_max: np.ndarray, label: str):
    path = os.environ.get(path_env, "").strip()
    if not path:
        return default_min, default_max
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} stats file not found: {path}")
    return _parse_stats_file(path)


def _parse_scales(value: str, action_dim: int) -> np.ndarray:
    if not value.strip():
        return np.ones(action_dim, dtype=np.float32)
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = [part.strip() for part in value.split(",") if part.strip()]
    scales = np.asarray([float(part) for part in parsed], dtype=np.float32)
    if scales.shape != (action_dim,):
        raise ValueError(
            f"{ACTION_NORM_SCALES_ENV} must contain {action_dim} values, got {scales.size}"
        )
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError(f"{ACTION_NORM_SCALES_ENV} values must be finite and positive: {scales}")
    return scales


@lru_cache(maxsize=1)
def get_action_stats():
    return _load_stats(ACTION_STATS_ENV, DEFAULT_ACTION_MIN, DEFAULT_ACTION_MAX, "action")


@lru_cache(maxsize=1)
def get_state_stats():
    return _load_stats(STATE_STATS_ENV, DEFAULT_STATE_MIN, DEFAULT_STATE_MAX, "state")


@lru_cache(maxsize=1)
def get_action_norm_scales():
    action_dim = int(get_action_stats()[0].shape[0])
    return _parse_scales(os.environ.get(ACTION_NORM_SCALES_ENV, ""), action_dim)


@lru_cache(maxsize=1)
def get_action_zero_norm():
    min_values, max_values = get_action_stats()
    return 2.0 * (0.0 - min_values) / (max_values - min_values + 1e-8) - 1.0


def apply_action_norm_scales(norm_action):
    scales = get_action_norm_scales()
    if np.allclose(scales, 1.0):
        return norm_action
    zero_norm = get_action_zero_norm()
    scaled = zero_norm + (np.asarray(norm_action) - zero_norm) * scales
    return np.clip(scaled, -1.0, 1.0)


def invert_action_norm_scales(norm_action):
    scales = get_action_norm_scales()
    if np.allclose(scales, 1.0):
        return norm_action
    zero_norm = get_action_zero_norm()
    unscaled = zero_norm + (np.asarray(norm_action) - zero_norm) / scales
    return np.clip(unscaled, -1.0, 1.0)
