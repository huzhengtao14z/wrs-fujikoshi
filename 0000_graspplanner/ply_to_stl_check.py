import os
import numpy as np
import pyvista as pv
import trimesh
import open3d as o3d
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import modeling.geometric_model as gm
import visualization.panda.world as wd
import modeling.collision_model as cm


def plt_to_stl(input_folder, ply_file, output_folder):
    ply_path = os.path.join(input_folder, ply_file)
    pcd = o3d.io.read_point_cloud(ply_path)

    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,
                                                                           o3d.utility.DoubleVector(
                                                                               [radius, radius * 2]))

    # create the triangular mesh with the vertices and faces from open3d
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                               vertex_normals=np.asarray(mesh.vertex_normals))

    trimesh.convex.is_convex(tri_mesh)

    # 保存为 .stl 文件
    stl_file = os.path.splitext(ply_file)[0] + '.stl'
    stl_path = os.path.join(output_folder, stl_file)
    tri_mesh.export(stl_path)
    print(f"successfully save stl file {stl_file}!")
    return tri_mesh


def scale_fit(mesh, threshold):
    bounding_box = mesh.bounding_box.bounds
    dimensions = bounding_box[1] - bounding_box[0]  # 计算长宽高
    # 如果任一维度大于 100 mm，则缩放模型
    max_dimension = max(dimensions)
    if max_dimension > threshold:
        scale_factor = threshold / max_dimension
        mesh.apply_scale(scale_factor)
        scale_factor = [scale_factor, scale_factor, scale_factor]
    else:
        scale_factor = [1, 1, 1]
    return scale_factor


def plot_stl_file(stl_file, png_file, gif_file):
    # 创建一个3D绘图对象
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # 加载STL文件
    stl = mesh.Mesh.from_file(stl_file)

    ax.add_collection3d(mplot3d.Axes3D.Ploy3DCollection(stl.vectors, color='lightgrey'))
    scale = stl.points.flatten
    ax.auto_scale_xyz(scale, scale, scale)
    plt.axis('off')
    ax.view_init(azim=0)
    plt.savefig(png_file)
    def update(i):
        print(i*36)
        ax.azim = i*36
        return ax
    anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=200)
    anim.save(gif_file, dpi=80, writer='imagemagick')
    

if __name__ == '__main__':
    base = wd.World(cam_pos=[20.01557, 6.37317, 10.88133], w=960,
                    h=540, lookat_pos=[0, 0, 0])
    # 定义文件夹路径
    folder_path = 'D:/qmc/wrs-qiu-master/0000_edge_check_deep_learning/data_utils/ABC_data'
    # 获取所有 .ply 文件
    # ply_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]
    ply_files = [f"{i:04d}.ply" for i in range(6411, 7168)]
    # 创建输出文件夹
    output_folder = 'D:/qmc/wrs-qiu-master/0000_edge_check_deep_learning/data_utils/ABC_data_stl'
    os.makedirs(output_folder, exist_ok=True)
    # for ply_file in ply_files:
    #     stl_mesh = plt_to_stl(folder_path, ply_file, output_folder)

    stl_folder = 'D:/qmc/wrs-qiu-master/0000_edge_check_deep_learning/data_utils/ABC_data_stl'
    png_folder = 'D:/qmc/wrs-qiu-master/0000_edge_check_deep_learning/data_utils/ABC_data_png'
    gif_folder = 'D:/qmc/wrs-qiu-master/0000_edge_check_deep_learning/data_utils/ABC_data_gif'
    stl_files = [f"{i:04d}.stl" for i in range(100)]
    # stl_files = ['0075.stl']
    position = [0, 0, 0]
    for i, stl_file in enumerate(stl_files):
        stl_path = os.path.join(stl_folder, stl_file)
        png_file = os.path.splitext(stl_file)[0] + '.png'
        gif_file = os.path.splitext(stl_file)[0] + '.gif'
        png_path = os.path.join(png_folder, png_file)
        gif_path = os.path.join(gif_folder, gif_file)
        stl_mesh = trimesh.load(stl_path)
        plot_stl_file(stl_path, png_path, gif_path)
        # stl_obj = cm.CollisionModel(stl_path)
        # stl_obj.get_scale()
        # stl_obj.set_scale()
        #
        # stl_scale_fit = scale_fit(stl_mesh, 100)
        # stl_obj.set_rgba([0.5, 0.5, 0.5, 1])
        # stl_obj.set_pos(position)
        # stl_obj.set_scale(stl_scale_fit)
        # stl_obj.attach_to(base)
        # position[0] = position[0]+100

    base.run()
