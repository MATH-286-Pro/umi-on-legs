from pathlib import Path
import pickle
import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

#=========================== Tool functions #===========================#

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def get_fps(data):
    t_max = data[0]["t"][-1]
    frame_max = len(data[0]["t"])
    return int(frame_max / t_max)


def get_dt(data):
    return data[0]["t"][-1] / len(data[0]["t"])


def get_keys(data_list):
    keys = set(data_list[0][0].keys())
    for data in data_list:
        for traj in data:
            keys = keys & set(traj.keys())

    preferred_order = ["t", "ee_pos", "ee_axis_angle", "gripper_width"]
    ordered_keys = [k for k in preferred_order if k in keys]
    ordered_keys += sorted(keys - set(ordered_keys))
    return ordered_keys




#=========================== Plot functions #===========================#
def plot_traj(data, random=True):

    fig = plt.figure(figsize=(15, 15))

    num_plots = min(9, len(data))
    step      = 200   # 每隔多少个点画一次姿态
    axis_len  = 0.02  # 姿态箭头长度

    if random:
        plot_indices = np.random.choice(len(data), size=num_plots, replace=False)
    else:
        plot_indices = np.arange(num_plots)

    for plot_idx in range(num_plots):
        ax = fig.add_subplot(3, 3, plot_idx + 1, projection='3d')
        data_idx = plot_indices[plot_idx]

        traj = data[data_idx]['ee_pos']
        axis_angle = data[data_idx]['ee_axis_angle']

        x = traj[:, 0]
        y = traj[:, 1]
        z = traj[:, 2]

        # 画轨迹
        ax.plot(x, y, z, linewidth=1.5)

        # 起点和终点
        ax.scatter(0,0,0, c='b', marker='*', s=40)
        ax.scatter(x[0], y[0], z[0], c='g', marker='o', s=40)
        ax.scatter(x[-1], y[-1], z[-1], c='r', marker='x', s=40)

        # 画姿态（这里只画局部 z 轴，图会更清楚）
        for i in range(0, len(traj), step):
            p = traj[i]
            rot = R.from_rotvec(axis_angle[i]).as_matrix()
            x_axis = rot[:, 0]
            y_axis = rot[:, 1]
            z_axis = rot[:, 2]

            ax.quiver(p[0], p[1], p[2], x_axis[0], x_axis[1], x_axis[2], length=axis_len, normalize=True, color='r')
            ax.quiver(p[0], p[1], p[2], y_axis[0], y_axis[1], y_axis[2], length=axis_len, normalize=True, color='g')
            ax.quiver(p[0], p[1], p[2], z_axis[0], z_axis[1], z_axis[2], length=axis_len, normalize=True, color='b')

        # 等比例坐标轴
        max_range = np.array([
            x.max() - x.min(),
            y.max() - y.min(),
            z.max() - z.min()
        ]).max() / 2.0

        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_title(f"Trajectory index: {data_idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()
