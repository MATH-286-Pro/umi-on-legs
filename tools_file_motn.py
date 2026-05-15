import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path


# ======== User Defined Settings ======== #
POSITION_SCALE = 2.8352 / 1000.0
EULER_ORDER = "XYZ"
CONVENTION = {
    "X": "-Y",
    "Y": "+Z",
    "Z": "-X",
}

# ======== Read Functions ======== #
def parse_fraction_time(text):
    values = [float(x) for x in text.split()]
    if len(values) == 1:
        return values[0]
    return values[0] / values[1]


def find_motn_node(root, node_name):
    node = root.find(f".//scenenode[@name='{node_name}']")
    if node is None:
        raise KeyError(f"Cannot find scenenode named {node_name!r}")
    return node


def child_param(parent, name):
    for child in parent:
        if child.tag == "parameter" and child.attrib.get("name") == name:
            return child
    raise KeyError(f"Cannot find parameter {name!r}")


def param_path(parent, names):
    current = parent
    for name in names:
        current = child_param(current, name)
    return current


def read_motn_curve(param):
    curve = param.find("curve")
    if curve is None:
        raise KeyError(f"Parameter {param.attrib.get('name')} has no curve")

    t = []
    values = []
    for keypoint in curve.findall("keypoint"):
        t.append(parse_fraction_time(keypoint.findtext("time")))
        values.append(float(keypoint.findtext("value")))

    return np.asarray(t, dtype=float), np.asarray(values, dtype=float)


def read_motn_xyz_curves(node, group_name):
    curves = []
    for axis in ["X", "Y", "Z"]:
        param = param_path(node, ["Properties", "Transform", group_name, axis])
        curves.append(read_motn_curve(param))

    t = curves[0][0]
    xyz = np.stack([curve[1] for curve in curves], axis=1)

    for curve_t, _ in curves[1:]:
        if not np.allclose(t, curve_t):
            raise ValueError(f"{group_name} X/Y/Z curve times do not match")

    return t - t[0], xyz


def load_motn(path, node_name="ARCamera"):
    root = ET.parse(path).getroot()
    node = find_motn_node(root, node_name)

    t_pos, pos   = read_motn_xyz_curves(node, "Position")
    t_rot, euler = read_motn_xyz_curves(node, "Rotation")

    if not np.allclose(t_pos, t_rot):
        raise ValueError("Position and Rotation curve times do not match")


    return {
        "source_type": "motn",
        "t": t_pos,
        "position_source": pos * POSITION_SCALE,
        "euler_source": euler,
        "default_invert_euler": False,
    }




# ======== Transform Functions ======== #
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


ROT_FUNCS = {"X": rot_x, "Y": rot_y, "Z": rot_z}


def euler_to_matrix(euler_xyz, order="ZYX"):
    x, y, z = euler_xyz
    angles = {"X": x, "Y": y, "Z": z}

    mat = np.eye(3)
    for axis in order:
        mat = ROT_FUNCS[axis](angles[axis]) @ mat
    return mat


def pos_rot_to_tf(pos, rot):
    pos = np.asarray(pos, dtype=float)
    rot = np.asarray(rot, dtype=float)
    tf = np.broadcast_to(np.eye(4), pos.shape[:-1] + (4, 4)).copy()
    tf[..., :3, :3] = rot
    tf[..., :3, 3] = pos
    return tf


def axis_mapping_to_tf(C):
    T = np.eye(4)
    T[:3, :3] = C
    return T


def yaw_from_matrix_z(R):
    return np.arctan2(R[1, 0], R[0, 0])


def rot_z_4(a):
    T = np.eye(4)
    T[:3, :3] = rot_z(a)
    return T


def zero_init_xy(Ts):
    init = np.eye(4)
    init[:2, 3] = Ts[0, :2, 3]
    return np.linalg.inv(init) @ Ts


def zero_init_yaw(Ts):
    init = rot_z_4(yaw_from_matrix_z(Ts[0, :3, :3]))
    return np.linalg.inv(init) @ Ts




# ======== Main Functions ======== #
def SHIFT_TF_VERSION(tf):

    # 创建变化矩阵
    R_SHIFT = np.array([[0, 0, -1],
                        [-1, 0, 0],
                        [0, 1, 0]])
    P_SHIFT = np.array([0, 0, 0])

    TF_SHIFT = np.eye(4)
    TF_SHIFT[:3, :3] = R_SHIFT
    TF_SHIFT[:3, 3] = P_SHIFT


    # 应用变化 #00ff00
    new_tf = TF_SHIFT @ tf @ np.linalg.inv(TF_SHIFT)

    return new_tf


def motn_to_tf(
        path: Path,
    ):
    
    raw = load_motn(path)
    pos   = raw["position_source"].copy()
    euler = raw["euler_source"].copy()

    invert_euler = raw["default_invert_euler"]

    euler_rad = euler.copy()


    if invert_euler:
        euler_rad = -euler_rad

    # motn original transform matrix
    R_source = np.asarray([euler_to_matrix(e, order=EULER_ORDER) for e in euler_rad])
    T_source = pos_rot_to_tf(pos, R_source)

    # Apply Convention Transform
    T_target = T_source.copy()
    T_target = SHIFT_TF_VERSION(T_target)
    T_target = zero_init_xy(T_target)
    T_target = zero_init_yaw(T_target)

    return {
        "t": raw["t"],
        "tf": T_target,
        "gripper_width": np.zeros_like(raw["t"]),
    }
