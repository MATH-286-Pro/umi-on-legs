import numpy as np
import matplotlib.pyplot as plt


def plot_3d(tf,
            arrow_length=0.1,
            interval: float = 0.1,
            view: str = "normal",):
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(tf[:, 0, 3], tf[:, 1, 3], tf[:, 2, 3], 'k-', label='Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 绘制 orientation
    for i in range(0, len(tf), int(len(tf) * interval)):

        T = tf[i]
        origin = T[:3, 3]
        x_axis = origin + T[:3, 0] * arrow_length
        y_axis = origin + T[:3, 1] * arrow_length
        z_axis = origin + T[:3, 2] * arrow_length
        ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r')
        ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g')
        ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b')

    match view.lower():
        case "normal":
            pass
        case "top":
            ax.view_init(elev=90, azim=-180)
        case "side":
            ax.view_init(elev=0, azim=-90)
        case "front":
            ax.view_init(elev=0, azim=0)
    
    xyz = tf[:, :3, 3]
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    center = mins + (maxs - mins) / 2
    radius = np.max(maxs - mins) / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.set_proj_type('ortho')

    plt.show()



def PLOT_3D(tf, 
            arrow_scale = 0.05, 
            interval: float = 0.1,
            view:str = "normal"):
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(tf[:, 0, 3], tf[:, 1, 3], tf[:, 2, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 绘制 orientation
    for i in range(0, len(tf), max(1, int(len(tf)*interval))):  # 每隔一定数量的帧绘制一个坐标轴
        T = tf[i]
        origin = T[:3, 3]
        x_axis = origin + T[:3, 0] * arrow_scale  # X轴，红色
        y_axis = origin + T[:3, 1] * arrow_scale  # Y轴，绿色
        z_axis = origin + T[:3, 2] * arrow_scale  # Z轴，蓝色
        ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r')
        ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g')
        ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b')

    match view.lower():
        case "normal":
            pass  # 使用默认视角
        case "top":
            # ax.view_init(elev=90, azim=-90)
            ax.view_init(elev=90, azim=-180)
        case "side":
            ax.view_init(elev=0, azim=-90)
        case "front":
            ax.view_init(elev=0, azim=0)


    xyz = tf[:, :3, 3]
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")

    plt.show()