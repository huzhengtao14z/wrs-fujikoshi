from stl import mesh
import numpy as np

# 读取STL文件
your_mesh = mesh.Mesh.from_file('/0000_graspplanner/stl/rubik.STL')

normals = np.cross(your_mesh.vectors[:, 1] - your_mesh.vectors[:, 0],
                   your_mesh.vectors[:, 2] - your_mesh.vectors[:, 0])
normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

from itertools import combinations
import math

# 定义阈值角度
angle_threshold = math.cos(math.radians(30))

# 存储锐角边
edges = []

for i, j in combinations(range(len(your_mesh.vectors)), 2):
    if np.dot(normals[i], normals[j]) < angle_threshold:
        for vi in your_mesh.vectors[i]:
            for vj in your_mesh.vectors[j]:
                if np.array_equal(vi, vj):
                    edges.append((vi, vj))

edges = np.array(edges)

from collections import defaultdict

# 统计每个顶点出现次数
vertex_count = defaultdict(int)

for edge in edges:
    vertex_count[tuple(edge[0])] += 1
    vertex_count[tuple(edge[1])] += 1

# 定义角点的阈值，可以调整
corner_threshold = 3

# 过滤出角点
corners = [vertex for vertex, count in vertex_count.items() if count >= corner_threshold]
corners = np.array(corners)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三角面
ax.add_collection3d(Poly3DCollection(your_mesh.vectors, alpha=.25, linewidths=1, edgecolors='k'))

# 绘制边缘特征
for edge in edges:
    ax.plot3D(*zip(*edge), color='r')

# 绘制角点特征
ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], color='b', s=100)

plt.show()
