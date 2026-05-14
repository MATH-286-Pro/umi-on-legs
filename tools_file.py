import pickle
import numpy as np



# pkl format
#   list: [{"t", "ee_pos", "ee_axis_angle", "gripper_width"}]

# tf format
#   list: [{"t", "tf"}]


# ======= 工具函数 ======= #
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


# ======= 转换函数 ======= #
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




def tf_to_pkl(tf_list) -> list[dict]:
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
