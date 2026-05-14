from __future__ import annotations

from pathlib import Path
import copy
import pickle

import numpy as np


MIRROR_XZ = np.diag([1.0, -1.0, 1.0])


def matrix_to_rotvec(mat):
    mat = np.asarray(mat, dtype=float)
    cos_angle = (np.trace(mat) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    if angle < 1e-8:
        return np.zeros(3, dtype=float)

    axis = np.array(
        [
            mat[2, 1] - mat[1, 2],
            mat[0, 2] - mat[2, 0],
            mat[1, 0] - mat[0, 1],
        ],
        dtype=float,
    ) / (2.0 * np.sin(angle))

    return axis * angle


def matrices_to_rotvec(mats):
    mats = np.asarray(mats, dtype=float)
    return np.asarray([matrix_to_rotvec(mat) for mat in mats.reshape((-1, 3, 3))], dtype=float).reshape(
        mats.shape[:-2] + (3,)
    )


def rotvec_to_matrix(rotvec):
    rotvec = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-8:
        return np.eye(3, dtype=float)

    axis = rotvec / angle
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c

    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def rotvecs_to_matrices(rotvecs):
    rotvecs = np.asarray(rotvecs, dtype=float)
    return np.asarray([rotvec_to_matrix(v) for v in rotvecs.reshape((-1, 3))], dtype=float).reshape(
        rotvecs.shape[:-1] + (3, 3)
    )


def pos_rot_to_tf(pos, rot):
    pos = np.asarray(pos, dtype=float)
    rot = np.asarray(rot, dtype=float)
    tf = np.broadcast_to(np.eye(4, dtype=float), pos.shape[:-1] + (4, 4)).copy()
    tf[..., :3, 3] = pos
    tf[..., :3, :3] = rot
    return tf


def episode_to_tf(episode):
    if "ee_tf" in episode:
        return np.asarray(episode["ee_tf"], dtype=float)

    if "ee_pos" not in episode or "ee_axis_angle" not in episode:
        raise KeyError("episode must contain either 'ee_tf' or both 'ee_pos' and 'ee_axis_angle'")

    rot = rotvecs_to_matrices(episode["ee_axis_angle"])
    return pos_rot_to_tf(episode["ee_pos"], rot)


def validate_tf(tf):
    tf = np.asarray(tf, dtype=float)
    if tf.shape[-2:] != (4, 4):
        raise ValueError(f"tf must have shape (..., 4, 4), got {tf.shape}")
    return tf


def mirror_xz_tf(tf):
    """Mirror a homogeneous transform sequence across the XZ plane.

    Position is reflected as y -> -y. Rotation is conjugated with the same
    reflection matrix, R' = M R M, so the output remains a valid SO(3) rotation.
    """
    tf = validate_tf(tf)
    out = tf.copy()
    out[..., :3, 3] = tf[..., :3, 3] @ MIRROR_XZ.T
    out[..., :3, :3] = np.einsum("ij,...jk,kl->...il", MIRROR_XZ, tf[..., :3, :3], MIRROR_XZ)
    return out


def reverse_tf(tf):
    """Reverse transform sequence order along the time dimension."""
    tf = validate_tf(tf)
    if tf.ndim < 3:
        raise ValueError("tf must include a time dimension, for example shape (T, 4, 4)")
    return np.flip(tf, axis=0).copy()


def tf_to_episode_fields(tf):
    tf = validate_tf(tf)
    return {
        "ee_tf": tf,
        "ee_pos": tf[..., :3, 3],
        "ee_axis_angle": matrices_to_rotvec(tf[..., :3, :3]),
    }


def _copy_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _copy_episode(episode):
    return {key: _copy_value(value) for key, value in episode.items()}


def _sequence_length(episode):
    if "t" in episode:
        return len(episode["t"])
    if "ee_tf" in episode:
        return len(episode["ee_tf"])
    if "ee_pos" in episode:
        return len(episode["ee_pos"])
    raise KeyError("cannot infer trajectory length from episode")


def _is_time_series(value, length):
    value = np.asarray(value) if isinstance(value, list) else value
    return isinstance(value, np.ndarray) and value.shape[:1] == (length,)


def reset_reversed_time(t):
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be a 1D array")
    return t[-1] - t[::-1]


def mirror_xz_episode(episode, keep_ee_tf=False):
    """Mirror one pkl episode across the XZ plane."""
    out = _copy_episode(episode)
    mirrored_tf = mirror_xz_tf(episode_to_tf(episode))
    fields = tf_to_episode_fields(mirrored_tf)

    pos_dtype = np.asarray(episode["ee_pos"]).dtype if "ee_pos" in episode else mirrored_tf.dtype
    axis_angle_dtype = (
        np.asarray(episode["ee_axis_angle"]).dtype if "ee_axis_angle" in episode else mirrored_tf.dtype
    )
    tf_dtype = np.asarray(episode["ee_tf"]).dtype if "ee_tf" in episode else mirrored_tf.dtype

    out["ee_pos"] = fields["ee_pos"].astype(pos_dtype, copy=False)
    out["ee_axis_angle"] = fields["ee_axis_angle"].astype(axis_angle_dtype, copy=False)
    if keep_ee_tf or "ee_tf" in episode:
        out["ee_tf"] = fields["ee_tf"].astype(tf_dtype, copy=False)
    return out


def reverse_episode(episode):
    """Reverse one pkl episode while keeping time increasing from 0."""
    length = _sequence_length(episode)
    out = {}

    for key, value in episode.items():
        if key == "t":
            out[key] = reset_reversed_time(value)
        elif _is_time_series(value, length):
            out[key] = np.flip(value, axis=0).copy()
        else:
            out[key] = _copy_value(value)

    return out


def augment_episode(episode, mirror=True, reverse=True, include_original=True):
    """Return augmented variants of one episode.

    The default output order is: original, mirrored, reversed.
    """
    variants = []
    if include_original:
        variants.append(_copy_episode(episode))
    if mirror:
        variants.append(mirror_xz_episode(episode))
    if reverse:
        variants.append(reverse_episode(episode))
    return variants


def augment_dataset(data, mirror=True, reverse=True, include_original=True):
    augmented = []
    for episode in data:
        augmented.extend(
            augment_episode(
                episode,
                mirror=mirror,
                reverse=reverse,
                include_original=include_original,
            )
        )
    return augmented


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def augment_pkl(
    input_path,
    output_path,
    *,
    mirror=True,
    reverse=True,
    include_original=True,
):
    data = load_pkl(input_path)
    augmented = augment_dataset(
        data,
        mirror=mirror,
        reverse=reverse,
        include_original=include_original,
    )
    save_pkl(augmented, output_path)
    return augmented
