import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

# 读取点云数据
pcd = o3d.io.read_point_cloud("./abc_ply/0002.ply")

# 估计法向量
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))

# 获取法向量
normals = np.asarray(pcd.normals)

# 映射法向量到单位球面（高斯映射）
unit_sphere_points = normals / np.linalg.norm(normals, axis=1, keepdims=True)

# 使用DBSCAN进行聚类
clustering = DBSCAN(eps=0.05, min_samples=10).fit(unit_sphere_points)
labels = clustering.labels_

# 找出边缘点（法向量变化较大的点）
edge_points = np.where(labels == -1)[0]

# 提取边缘点
edge_pcd = pcd.select_by_index(edge_points)

# 可视化边缘点
o3d.visualization.draw_geometries([edge_pcd])
