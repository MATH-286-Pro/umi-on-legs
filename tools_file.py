import pickle
import numpy as np
import tools_file_motn


# pkl format
#   list: [{"t", "ee_pos", "ee_axis_angle", "gripper_width"}]

# tf format
#   list: [{"t", "tf", "gripper_width"}]


# ======= Read Function ======= #
def get_fps(data):
    t = data[0]["t"]

    t_frame = len(t)
    t_last  = t[-1]
    fps = round(t_frame / t_last)
    return fps 


# ======= Tool Function ======= #
def pos_rot_to_tf(pos, rot):
    pos = np.asarray(pos, dtype=float)
    rot = np.asarray(rot, dtype=float)

    if pos.shape[-1] != 3:
        raise ValueError(f"pos must have shape (..., 3), got {pos.shape}")
    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"rot must have shape (..., 3, 3), got {rot.shape}")

    tf = np.broadcast_to(np.eye(4), pos.shape[:-1] + (4, 4)).copy()
    tf[..., :3, :3] = rot
    tf[..., :3, 3] = pos
    return tf

def tf_to_pos_rot(tf):
    tf = np.asarray(tf, dtype=float)

    if tf.shape[-2:] != (4, 4):
        raise ValueError(f"tf must have shape (..., 4, 4), got {tf.shape}")

    pos = tf[..., :3, 3]
    rot = tf[..., :3, :3]
    return pos, rot


def axis_angle_to_rot(axis_angle):
    axis_angle = np.asarray(axis_angle, dtype=float)

    if axis_angle.shape[-1] != 3:
        raise ValueError(f"axis_angle must have shape (..., 3), got {axis_angle.shape}")

    angle = np.linalg.norm(axis_angle, axis=-1)
    safe_angle = np.where(angle < 1e-6, 1.0, angle)
    axis = axis_angle / safe_angle[..., None]
    x, y, z = np.moveaxis(axis, -1, 0)

    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    R = np.empty(axis_angle.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = c + x*x*C
    R[..., 0, 1] = x*y*C - z*s
    R[..., 0, 2] = x*z*C + y*s
    R[..., 1, 0] = y*x*C + z*s
    R[..., 1, 1] = c + y*y*C
    R[..., 1, 2] = y*z*C - x*s
    R[..., 2, 0] = z*x*C - y*s
    R[..., 2, 1] = z*y*C + x*s
    R[..., 2, 2] = c + z*z*C

    R[angle < 1e-6] = np.eye(3)
    return R


def rot_to_axis_angle(rot):
    rot = np.asarray(rot, dtype=float)

    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"rot must have shape (..., 3, 3), got {rot.shape}")

    m00 = rot[..., 0, 0]
    m01 = rot[..., 0, 1]
    m02 = rot[..., 0, 2]
    m10 = rot[..., 1, 0]
    m11 = rot[..., 1, 1]
    m12 = rot[..., 1, 2]
    m20 = rot[..., 2, 0]
    m21 = rot[..., 2, 1]
    m22 = rot[..., 2, 2]

    q_abs = np.sqrt(
        np.maximum(
            np.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                axis=-1,
            ),
            0.0,
        )
    )

    quat_candidates = np.stack(
        [
            np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m21 + m12], axis=-1),
            np.stack([m10 - m01, m02 + m20, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )
    quat_candidates = quat_candidates / (2.0 * np.maximum(q_abs[..., None], 1e-8))

    best = np.argmax(q_abs, axis=-1)
    quat = np.take_along_axis(quat_candidates, best[..., None, None], axis=-2)[..., 0, :]
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
    quat = np.where(quat[..., :1] < 0.0, -quat, quat)

    vector = quat[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, quat[..., 0])
    scale = angle / np.where(vector_norm < 1e-8, 1.0, vector_norm)
    scale = np.where(vector_norm < 1e-8, 2.0, scale)

    return vector * scale[..., None]


def _rot_to_quat(rot):
    rot = np.asarray(rot, dtype=float)

    if rot.shape[-2:] != (3, 3):
        raise ValueError(f"rot must have shape (..., 3, 3), got {rot.shape}")

    m00 = rot[..., 0, 0]
    m01 = rot[..., 0, 1]
    m02 = rot[..., 0, 2]
    m10 = rot[..., 1, 0]
    m11 = rot[..., 1, 1]
    m12 = rot[..., 1, 2]
    m20 = rot[..., 2, 0]
    m21 = rot[..., 2, 1]
    m22 = rot[..., 2, 2]

    q_abs = np.sqrt(
        np.maximum(
            np.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                axis=-1,
            ),
            0.0,
        )
    )

    quat_candidates = np.stack(
        [
            np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m21 + m12], axis=-1),
            np.stack([m10 - m01, m02 + m20, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )
    quat_candidates = quat_candidates / (2.0 * np.maximum(q_abs[..., None], 1e-8))

    best = np.argmax(q_abs, axis=-1)
    quat = np.take_along_axis(quat_candidates, best[..., None, None], axis=-2)[..., 0, :]
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
    return np.where(quat[..., :1] < 0.0, -quat, quat)


def _quat_to_rot(quat):
    quat = np.asarray(quat, dtype=float)

    if quat.shape[-1] != 4:
        raise ValueError(f"quat must have shape (..., 4), got {quat.shape}")

    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
    w, x, y, z = np.moveaxis(quat, -1, 0)

    rot = np.empty(quat.shape[:-1] + (3, 3), dtype=float)
    rot[..., 0, 0] = 1.0 - 2.0 * (y*y + z*z)
    rot[..., 0, 1] = 2.0 * (x*y - z*w)
    rot[..., 0, 2] = 2.0 * (x*z + y*w)
    rot[..., 1, 0] = 2.0 * (x*y + z*w)
    rot[..., 1, 1] = 1.0 - 2.0 * (x*x + z*z)
    rot[..., 1, 2] = 2.0 * (y*z - x*w)
    rot[..., 2, 0] = 2.0 * (x*z - y*w)
    rot[..., 2, 1] = 2.0 * (y*z + x*w)
    rot[..., 2, 2] = 1.0 - 2.0 * (x*x + y*y)
    return rot


def _interp_array(t, value, new_t):
    value = np.asarray(value, dtype=float)

    if value.shape[:1] != (len(t),):
        raise ValueError(f"value first dimension must match len(t), got {value.shape} and {len(t)}")

    flat = value.reshape(len(t), -1)
    out = np.empty((len(new_t), flat.shape[1]), dtype=float)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(new_t, t, flat[:, i])
    return out.reshape((len(new_t),) + value.shape[1:])


def _slerp_quat(t, quat, new_t):
    quat = np.asarray(quat, dtype=float)

    if quat.shape != (len(t), 4):
        raise ValueError(f"quat must have shape (len(t), 4), got {quat.shape}")

    quat = quat.copy()
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
    for i in range(1, len(quat)):
        if np.dot(quat[i - 1], quat[i]) < 0.0:
            quat[i] = -quat[i]

    if len(t) == 1:
        return np.repeat(quat, len(new_t), axis=0)

    idx = np.searchsorted(t, new_t, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 2)
    t0 = t[idx]
    t1 = t[idx + 1]
    alpha = (new_t - t0) / (t1 - t0)

    q0 = quat[idx]
    q1 = quat[idx + 1]
    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where(dot[..., None] < 0.0, -q1, q1)
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    linear = dot > 0.9995

    scale0 = np.empty_like(alpha)
    scale1 = np.empty_like(alpha)
    scale0[linear] = 1.0 - alpha[linear]
    scale1[linear] = alpha[linear]
    scale0[~linear] = np.sin((1.0 - alpha[~linear]) * theta[~linear]) / sin_theta[~linear]
    scale1[~linear] = np.sin(alpha[~linear] * theta[~linear]) / sin_theta[~linear]

    out = scale0[..., None] * q0 + scale1[..., None] * q1
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8)


# ======= Resampling for transform matrix ======= #
def tf_resample_tf(tf_list, target_fps, padding=False) -> list[dict]:

    current_fps = get_fps(tf_list)
    has_repeated_t = any(
        len(item["t"]) > 1 and np.any(np.diff(np.asarray(item["t"], dtype=float)) == 0.0)
        for item in tf_list
    )

    if current_fps == target_fps and not has_repeated_t and not padding:
        return tf_list
    
    else:
        out = []
        for item in tf_list:
            t = np.asarray(item["t"], dtype=float)
            tf = np.asarray(item["tf"], dtype=float)
            gripper_width = np.asarray(item["gripper_width"], dtype=float)

            if t.ndim != 1:
                raise ValueError(f"t must be a 1D array, got {t.shape}")
            if len(t) == 0:
                raise ValueError("t must contain at least one timestamp")
            if tf.shape != (len(t), 4, 4):
                raise ValueError(f"tf must have shape (len(t), 4, 4), got {tf.shape}")
            if len(t) > 1 and np.any(np.diff(t) < 0.0):
                raise ValueError("t must be non-decreasing")

            keep = np.r_[True, np.diff(t) > 0.0]
            t = t[keep]
            tf = tf[keep]
            gripper_width = gripper_width[keep]

            if len(t) == 1:
                out.append({
                    "t": t.copy(),
                    "tf": tf.copy(),
                    "gripper_width": gripper_width.copy(),
                })
                continue

            duration = t[-1] - t[0]
            frame_count = max(2, int(round(duration * target_fps)))
            new_t = np.linspace(t[0], t[-1], frame_count)

            pos = tf[..., :3, 3]
            quat = _rot_to_quat(tf[..., :3, :3])
            new_pos = _interp_array(t, pos, new_t)
            new_rot = _quat_to_rot(_slerp_quat(t, quat, new_t))
            new_tf = pos_rot_to_tf(new_pos, new_rot)
            new_gripper_width = _interp_array(t, gripper_width, new_t)

            out.append({
                "t": new_t,
                "tf": new_tf,
                "gripper_width": new_gripper_width,
            })

        # Padding
        if padding:
            max_len = max(len(item["t"]) for item in out)
            dt = 1.0 / target_fps

            for item in out:
                pad_len = max_len - len(item["t"])
                if pad_len == 0:
                    continue

                pad_t = item["t"][-1] + dt * np.arange(1, pad_len + 1)
                pad_tf = np.repeat(item["tf"][-1:], pad_len, axis=0)
                pad_gripper_width = np.repeat(item["gripper_width"][-1:], pad_len, axis=0)

                item["t"] = np.concatenate([item["t"], pad_t], axis=0)
                item["tf"] = np.concatenate([item["tf"], pad_tf], axis=0)
                item["gripper_width"] = np.concatenate([item["gripper_width"], pad_gripper_width], axis=0)

        return out


# ======= File Convert ======= #
def pkl_to_tf(path) -> list[dict]:
    pkl_list = pickle.load(open(path, 'rb'))
    tf_list = []
    
    for item in pkl_list:
        t = item["t"]
        pos = item["ee_pos"]
        axis_angle = item["ee_axis_angle"]
        gripper_width = item["gripper_width"]

        rot = axis_angle_to_rot(axis_angle)

        tf = pos_rot_to_tf(pos, rot)
        
        tf_list.append({
            "t": t, 
            "tf": tf,
            "gripper_width": gripper_width,
            })

    return tf_list

def motn_to_tf(path) -> list[dict]:
    return tools_file_motn.motn_to_tf(path)


def tf_to_traj(tf_list) -> list[dict]:
    pkl_list = []
    for item in tf_list:
        t = item["t"]
        tf = item["tf"]
        gripper_width = item["gripper_width"]

        pos, rot = tf_to_pos_rot(tf)
        axis_angle = rot_to_axis_angle(rot)

        pkl_list.append({
            "t": t, 
            "ee_pos": pos, 
            "ee_axis_angle": axis_angle,
            "gripper_width": gripper_width,
            })

    return pkl_list