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
    for i in range(0, len(tf), int(1 / interval)):
        origin = tf[i, :3, 3]
        x_axis = tf[i, :3, 0] * arrow_length
        y_axis = tf[i, :3, 1] * arrow_length
        z_axis = tf[i, :3, 2] * arrow_length

        ax.quiver(*origin, *x_axis, color='r', length=arrow_length)
        ax.quiver(*origin, *y_axis, color='g', length=arrow_length)
        ax.quiver(*origin, *z_axis, color='b', length=arrow_length)
    
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