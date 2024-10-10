import open3d as o3d
import os
import pickle
import numpy as np
import pyransac3d as pyrsc
import basis.robot_math as rm
import modeling.geometric_model as gm
import visualization.panda.world as wd
import modeling.collision_model as cm
import vision.depth_camera.pcd_data_adapter as vdda
from sklearn.cluster import DBSCAN


def edge_detection(point_cloud_np, normal_np, k=20, threshold=14, load=True, show=True):
    if load:
        with open(f'./debug_data/{objname}/edge_np.pickle', 'rb') as file1:
            edge_np, edge_normal_np, edge_index, k, threshold = pickle.load(file1)
            print('Edge detection parameter: ')
            print(f'radius: {k}, threshold: {threshold}')
        with open(f'./debug_data/{objname}/surface_np.pickle', 'rb') as file2:
            surface_np, surface_index = pickle.load(file2)
    else:
        # pcd_neighbor_normal_np_list = []
        point_cloud = vdda.nparray_to_o3dpcd(point_cloud_np)
        # point_cloud, ind = point_cloud.remove_radius_outlier(nb_points=10, radius=radius)
        # point_cloud_np = vdda.o3dpcd_to_parray(point_cloud)

        point_cloud_tree = o3d.geometry.KDTreeFlann(point_cloud)
        pcd_neighbor_np_list = []
        edge_np, surface_np = [], []
        edge_normal_np = []
        edge_index, surface_index = [], []
        for index, anchor in enumerate(point_cloud_np):
            [k, idx, _] = point_cloud_tree.search_knn_vector_3d(anchor, k)
            # pcd_neighbor_normal_np = np.asarray([pcd.normals[i] for i in idx])
            # pcd_neighbor_normal_np_list.append(pcd_neighbor_normal_np)
            pcd_neighbor_np = np.vstack([vdda.o3dpcd_to_parray(point_cloud)[idx[1:]], anchor])
            pcd_neighbor_normal_np = np.vstack([normal_np[idx[1:]], normal_np[index]])
            pcd_neighbor_np, pcd_neighbor_normal_np= cluster_input(anchor, pcd_neighbor_np, pcd_neighbor_normal_np, normal_cluster=True)
            if len(pcd_neighbor_np) == 1:
                print(f'{index} is outlier!')
                continue
            pcd_neighbor_np_list.append(pcd_neighbor_np)
            pcd_neighbor_average = pcd_neighbor_np.mean(axis=0)
            # X = pcd_neighbor_np - pcd_neighbor_average
            # H = np.dot(X.T, X)
            # eigen_vals, eigen_vecs = np.linalg.eig(H / (X.shape[0] - 1))
            # PC = eigen_vals / np.sum(eigen_vals, axis=0)
            # if np.min(PC) > threshold:
            #     is_surface = False
            #     edge_np.append(anchor)
            #     edge_normal_np.append(normal_np[index])
            #     edge_index.append(index)
            # else:
            #     is_surface = True
            #     surface_np.append(anchor)
            #     surface_index.append(index)
            distance_neighbor_center = []
            for i in range(len(pcd_neighbor_np)):
                distance_neighbor_center.append(np.linalg.norm(anchor - pcd_neighbor_np[i]))
            distance_neighbor_center_min = np.sort(np.array(distance_neighbor_center))[1]
            distance_query = np.linalg.norm(pcd_neighbor_average - anchor)
            if distance_query > (distance_neighbor_center_min * threshold):
                is_surface = False
                edge_np.append(anchor)
                edge_normal_np.append(normal_np[index])
                edge_index.append(index)
            else:
                is_surface = True
                surface_np.append(anchor)
                surface_index.append(index)
            # print(is_surface, index)
        edge_np = np.asarray(edge_np)
        edge_normal_np = np.asarray(edge_normal_np)
        surface_np = np.asarray(surface_np)
        os.makedirs(f"./debug_data/{objname}", exist_ok=True)
        with open(f'./debug_data/{objname}/edge_np.pickle', 'wb') as file1:
            pickle.dump((edge_np, edge_normal_np, edge_index, k, threshold), file1)
        with open(f'./debug_data/{objname}/surface_np.pickle', 'wb') as file2:
            pickle.dump((surface_np, surface_index), file2)
    if show:
        gm.gen_pointcloud(edge_np, [[191/255, 144/255, 0, 1]], 6).attach_to(base)
        # for i in range(len(edge_np)):
        #     gm.gen_sphere(pos=edge_np[i], radius=0.05, rgba=[1, 0, 0, 1]).attach_to(base)
        # gm.gen_pointcloud(surface_np, [[0, 1, 0, 0.5]], 3).attach_to(base)
        # for i in range(len(surface_np)):
        #     gm.gen_sphere(pos=surface_np[i], radius=0.05, rgba=[0, 1, 0, 1]).attach_to(base)
    return edge_np, edge_normal_np, surface_np, edge_index, surface_index


def edge_detection_normal(point_cloud_np, normal_np, k=20, threshold=45, load=True, show=True):
    if load:
        with open(f'./debug_data/{objname}/edge_np_normal.pickle', 'rb') as file1:
            edge_np, edge_normal_np, edge_index, k, threshold = pickle.load(file1)
            print('Edge detection parameter: ')
            print(f'radius: {k}, threshold: {threshold}')
        with open(f'./debug_data/{objname}/surface_np_normal.pickle', 'rb') as file2:
            surface_np, surface_index = pickle.load(file2)
    else:
        # pcd_neighbor_normal_np_list = []
        point_cloud = vdda.nparray_to_o3dpcd(point_cloud_np)
        # point_cloud, ind = point_cloud.remove_radius_outlier(nb_points=10, radius=radius)
        # point_cloud_np = vdda.o3dpcd_to_parray(point_cloud)

        point_cloud_tree = o3d.geometry.KDTreeFlann(point_cloud)
        pcd_neighbor_np_list = []
        edge_np, surface_np = [], []
        edge_normal_np = []
        edge_index, surface_index = [], []
        for index, anchor in enumerate(point_cloud_np):
            [k, idx, _] = point_cloud_tree.search_knn_vector_3d(anchor, k)
            # pcd_neighbor_normal_np = np.asarray([pcd.normals[i] for i in idx])
            # pcd_neighbor_normal_np_list.append(pcd_neighbor_normal_np)
            pcd_neighbor_np = np.vstack([vdda.o3dpcd_to_parray(point_cloud)[idx[1:]], anchor])
            pcd_neighbor_normal_np = np.vstack([normal_np[idx[1:]], normal_np[index]])
            pcd_neighbor_np, pcd_neighbor_normal_np = cluster_input(anchor, pcd_neighbor_np, pcd_neighbor_normal_np,
                                                                    normal_cluster=True)
            if len(pcd_neighbor_np) == 1:
                print(f'{index} is outlier!')
                continue
            pcd_neighbor_np_list.append(pcd_neighbor_np)

            angles = [np.arccos(np.clip(np.dot(normal_np[index], n), -1.0, 1.0)) for n in pcd_neighbor_normal_np]
            average_angle = np.mean(angles)
            average_angle_degrees = np.degrees(average_angle)
            if average_angle_degrees > threshold:
                is_surface = False
                edge_np.append(anchor)
                edge_normal_np.append(normal_np[index])
                edge_index.append(index)
            else:
                is_surface = True
                surface_np.append(anchor)
                surface_index.append(index)
            # print(is_surface, index)
        edge_np = np.asarray(edge_np)
        edge_normal_np = np.asarray(edge_normal_np)
        surface_np = np.asarray(surface_np)
        os.makedirs(f"./debug_data/{objname}", exist_ok=True)
        with open(f'./debug_data/{objname}/edge_np_normal.pickle', 'wb') as file1:
            pickle.dump((edge_np, edge_normal_np, edge_index, k, threshold), file1)
        with open(f'./debug_data/{objname}/surface_np_normal.pickle', 'wb') as file2:
            pickle.dump((surface_np, surface_index), file2)
    if show:
        gm.gen_pointcloud(edge_np, [[191 / 255, 144 / 255, 0, 1]], 6).attach_to(base)
        # for i in range(len(edge_np)):
        #     gm.gen_sphere(pos=edge_np[i], radius=0.05, rgba=[1, 0, 0, 1]).attach_to(base)
        # gm.gen_pointcloud(surface_np, [[0, 1, 0, 0.5]], 3).attach_to(base)
        # for i in range(len(surface_np)):
        #     gm.gen_sphere(pos=surface_np[i], radius=0.05, rgba=[0, 1, 0, 1]).attach_to(base)
    return edge_np, edge_normal_np, surface_np, edge_index, surface_index

def corner_check(edge_np, edge_normal_np, k, threshold, eps, load=False, show=True):
    if load:
        with open(f'./debug_data/{objname}/corner_np.pickle', 'rb') as file1:
            corner_np, k, threshold, eps = pickle.load(file1)
            print('Corner detection parameter: ')
            print(f'k: {k}, threshold: {threshold}, eps: {eps}')
    else:
        corner_np = []
        for index, anchor in enumerate(edge_np):
            edge_tree = o3d.geometry.KDTreeFlann(vdda.nparray_to_o3dpcd(edge_np))
            [k, idx, _] = edge_tree.search_knn_vector_3d(anchor, k)
            edge_neighbor_np = np.vstack([edge_np[idx[1:]], anchor])
            edge_neighbor_normal_np = np.vstack([edge_normal_np[idx[1:]], edge_normal_np[index]])
            edge_neighbor_np, edge_neighbor_normal_np = cluster_input(anchor, edge_neighbor_np, edge_neighbor_normal_np, normal_cluster=False)
            if len(edge_neighbor_np) == 1:
                print(f'{index} is outlier!')
                continue
            edge_neighbor_average = edge_neighbor_np.mean(axis=0)
            distance_neighbor_query_list = []
            X = edge_neighbor_np - edge_neighbor_average
            # R = 0.003
            # for i in range(len(edge_neighbor_np)):
            #     distance_neighbor_query = np.linalg.norm(anchor - edge_neighbor_np[i])
            #     distance_neighbor_query_list.append(distance_neighbor_query)
            #     X[i] = X[i] * (R - distance_neighbor_query)
            # distance_neighbor_query_mean = np.mean(np.array(distance_neighbor_query_list))
            # H = np.dot(X.T, X)
            # eigen_vals, eigen_vecs = np.linalg.eig(H / (distance_neighbor_query_mean))
            H = np.dot(X.T, X)
            eigen_vals, eigen_vecs = np.linalg.eig(H / (X.shape[0] - 1))
            PC = eigen_vals / np.sum(eigen_vals, axis=0)
            if np.max(PC) < threshold:
                is_vertex = True
                corner_np.append(anchor)
            else:
                is_vertex = False
            # print(is_vertex, index)

            # if threshold_l < eigen_vals[1] / eigen_vals[0] < threshold_u and threshold_l < eigen_vals[2] / eigen_vals[
            #     0] < threshold_u and threshold_l < eigen_vals[2] / eigen_vals[1] < threshold_u:
            #     is_vertex = True
            #     corner_np.append(anchor)
            # else:
            #     is_vertex = False
            # PC1 = eig
            # print(is_vertex)
        corner_np = np.asarray(corner_np)
        os.makedirs(f"./debug_data/{objname}", exist_ok=True)
        with open(f'./debug_data/{objname}/corner_np.pickle', 'wb') as file1:
            pickle.dump((corner_np, k, threshold, eps), file1)
    if show:
        for i in range(len(corner_np)):
            gm.gen_sphere(pos=corner_np[i], radius=0.005, rgba=[0, 0, 1, 0.5]).attach_to(base)
    return corner_np


def precision_check(objname, orig_pcd, result_edge_pnt_ind, result_surf_pnt_ind, radius, threshold, save=True):
    orig_pcd_colors = np.asarray(orig_pcd.colors)
    mask = np.all(orig_pcd_colors == 1, axis=1)
    orig_edge_pnt_ind = np.where(~mask)[0]
    orig_surf_pnt_ind = np.where(mask)[0]
    # result_edge_pnt_ind = []
    # for edge_pnt in edge_np:
    #     for idx, orig_pnt in enumerate(orig_np):
    #         if np.array_equal(edge_pnt, orig_pnt):
    #             result_edge_pnt_ind.append(idx)
    # orig_pnt_ind = np.arange(orig_np.shape[0])
    # result_surf_pnt_ind = np.setdiff1d(orig_pnt_ind, result_edge_pnt_ind)
    TP = np.intersect1d(orig_edge_pnt_ind, result_edge_pnt_ind)
    TN = np.intersect1d(orig_surf_pnt_ind, result_surf_pnt_ind)
    FP = np.intersect1d(orig_surf_pnt_ind, result_edge_pnt_ind)
    FN = np.intersect1d(orig_edge_pnt_ind, result_surf_pnt_ind)
    precision = len(TP) / (len(TP) + len(FP))
    recall = len(TP) / (len(TP) + len(FN))
    if save:
        result_file_path = f"./debug_data/{objname}/acc_result.pickle"
        if os.path.exists(result_file_path) and os.path.isfile(result_file_path):
            with open(f'./debug_data/{objname}/acc_result.pickle', 'rb') as file1:
                result_list = pickle.load(file1)
        else:
            result_list = []
        result_list.append(f"precision: {precision}, recall: {recall}, radius: {radius}, threshold: {threshold}")
        with open(f'./debug_data/{objname}/acc_result.pickle', 'wb') as file1:
            pickle.dump(result_list, file1)
        print(f"precision: {precision}, recall: {recall}, radius: {radius}, threshold: {threshold}")
    return precision, recall


def calculate_distance(orig_pcd, orig_np, result_edge_np, save=True):
    # 初始化对应关系
    orig_pcd_colors = np.asarray(orig_pcd.colors)
    mask = np.all(orig_pcd_colors == 1, axis=1)
    orig_edge_pnt_ind = np.where(~mask)[0]
    orig_edge_pnt = orig_np[orig_edge_pnt_ind]
    correspondences = []
    distance = []
    # 对于每个源点，找到在目标点云中最近的点
    for result_edge_pnt in result_edge_np:
        distances = np.linalg.norm(orig_edge_pnt - result_edge_pnt, axis=1)
        nearest_index = np.argmin(distances)
        min_distance = np.min(distances)
        distance.append(min_distance)
        correspondences.append((result_edge_pnt, orig_edge_pnt[nearest_index]))
    mean_distance = np.mean(np.array(distance))
    if save:
        result_file_path = f"./debug_data/{objname}/acc_result.pickle"
        if os.path.exists(result_file_path) and os.path.isfile(result_file_path):
            with open(f'./debug_data/{objname}/acc_result.pickle', 'rb') as file1:
                result_list = pickle.load(file1)
        else:
            result_list = []
        result_list.append(f"distance: {mean_distance}")
        with open(f'./debug_data/{objname}/acc_result.pickle', 'wb') as file1:
            pickle.dump(result_list, file1)
        print(f"distance: {mean_distance}")
    return mean_distance


def sample_points_around_center(center, radius=1.0, num_samples=2):
    samples = []
    for _ in range(num_samples):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius)
        x_offset = r * np.cos(angle)
        y_offset = r * np.sin(angle)
        z_offset = np.random.uniform(-radius, radius)
        new_point = center + np.array([x_offset, y_offset, z_offset])
        samples.append(new_point)
    return np.array(samples)


def extend_point_cloud(original_cloud, radius, num_samples):
    extended_cloud = original_cloud.copy()
    for point in original_cloud:
        new_points = sample_points_around_center(point, radius, num_samples)
        extended_cloud = np.vstack((extended_cloud, new_points))
    return np.array(extended_cloud)


def cluster_input(anchor, points, normals, normal_cluster=False):
    points_scale = scale_compute(points)
    dbscan_points = DBSCAN(eps=points_scale/2, min_samples=1)
    points_labels = dbscan_points.fit_predict(points)

    unique_labels = set(points_labels)
    unique_labels.discard(-1)
    if len(unique_labels) != 1:
        target_label = points_labels[np.where((points == anchor).all(axis=1))[0][0]]
        result_points = points[points_labels == target_label]
        result_normals = normals[points_labels == target_label]

    else:
        result_points = points
        result_normals = normals

    if normal_cluster:

        normals_scale = scale_compute(result_normals)
        dbscan_normals = DBSCAN(eps=0.5, min_samples=1)
        normals_labels = dbscan_normals.fit_predict(result_normals)
        unique_normals_labels = set(normals_labels)
        if len(unique_normals_labels) != 1:
            target_label = normals_labels[np.where((result_points == anchor).all(axis=1))[0][0]]
            result_points = result_points[normals_labels == target_label]
            result_normals = result_normals[normals_labels == target_label]
        else:
            result_points = result_points
            result_normals = result_normals

    return result_points, result_normals


def cluster_corner_result(corner, eps, min_samples, load=False, show=True):
    if load:
        with open(f'./debug_data/{objname}/corner_mean.pickle', 'rb') as file1:
            average_corner, eps, min_samples = pickle.load(file1)
            print('Average result parameter:')
            print(f'eps: {eps}, min_samples: {min_samples}')
    else:
        if len(corner) == 0:
            average_corner = np.array([])
        else:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(corner)
            unique_labels = set(labels)
            average_corner = []
            for label in unique_labels:
                cluster_points = corner[labels == label]
                cluster_mean = np.mean(cluster_points, axis=0)
                cluster_distance = np.linalg.norm(cluster_points - cluster_mean, axis=1)
                min_distance_index = np.argmin(cluster_distance)
                average_corner.append(cluster_points[min_distance_index])
            average_corner = np.asarray(average_corner)
        with open(f'./debug_data/{objname}/corner_mean.pickle', 'wb') as file1:
            pickle.dump((average_corner, eps, min_samples), file1)
    if show:
        for i in range(len(average_corner)):
            gm.gen_sphere(pos=average_corner[i], radius=0.005, rgba=[192/255, 0, 0, 1]).attach_to(base)
    return average_corner


def check_result(objname, radius, threshold):
    result_file_path1 = f'./debug_data/{objname}/edge_np_{radius}_{threshold}.pickle'
    result_file_path2 = f"./debug_data/{objname}/acc_result.pickle"
    if os.path.exists(result_file_path1) and os.path.exists(result_file_path2):
        with open(result_file_path1, 'rb') as file1:
            edge_np = pickle.load(file1)
        with open(result_file_path2, 'rb') as file2:
            result_list = pickle.load(file2)
        gm.gen_pointcloud(edge_np).attach_to(base)
        print("result list: \n")
        for each in result_list:
            print(each, end=',\n')
    else:
        print("can't find result file!")


def voxel_grid_filter(point_cloud, voxel_size):
    """
    使用体素网格采样点云数据

    参数:
        point_cloud: 原始点云数据（numpy.ndarray, shape: [N, 3]）
        voxel_size: 体素网格的大小（float）

    返回:
        采样后的点云（numpy.ndarray, shape: [M, 3]）
    """
    # 获取点云数据的最小和最大值
    min_bound = np.min(point_cloud, axis=0)
    max_bound = np.max(point_cloud, axis=0)

    # 计算每个点所在的体素
    voxel_indices = np.floor((point_cloud - min_bound) / voxel_size).astype(np.int32)

    # 使用字典存储每个体素中的点
    voxel_dict = {}
    for i, v in enumerate(voxel_indices):
        v = tuple(v)
        if v not in voxel_dict:
            voxel_dict[v] = []
        voxel_dict[v].append(i)

    # 从每个体素中选择一个点
    sampled_indices = [indices[0] for indices in voxel_dict.values()]
    sampled_point_cloud = point_cloud[sampled_indices]

    return sampled_point_cloud


def scale_compute(data):
    x_scale = np.min(data[:, 0]) - np.max(data[:, 0])
    y_scale = np.min(data[:, 1]) - np.max(data[:, 1])
    z_scale = np.min(data[:, 2]) - np.max(data[:, 2])
    max_scale = np.max(np.abs([x_scale, y_scale, z_scale]))
    if max_scale == 0:
        max_scale = 1
    return max_scale

def svd(data):
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    U, S, Vt = np.linalg.svd(centered_data)
    direction = Vt[0]
    return mean, direction

def fit_line_from_point(points, radius, threshold, maxIteration):
    pcd = vdda.nparray_to_o3dpcd(points)
    random_point = points[np.random.randint(0, len(points)-1)]
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    [k, idx, _] = kdtree.search_radius_vector_3d(random_point, radius)
    neighbor_points = points[idx]
    line_model = pyrsc.Line()
    A, B, inliers = line_model.fit(neighbor_points, threshold, maxIteration)
    inliers_points = neighbor_points[inliers]
    inliers_mean, inliers_direction = svd(inliers_points)
    t = [-2, 2]
    line_points = np.zeros((2, 3))
    line_points[0] = inliers_mean + inliers_direction * t[0]
    line_points[1] = inliers_mean + inliers_direction * t[1]
    gm.gen_sphere(random_point, radius, rgba=[0, 1, 0, 0.5]).attach_to(base)

    gm.gen_arrow(line_points[0], line_points[1], thickness=0.001, rgba=[1, 0, 0, 1]).attach_to(base)

if __name__ == '__main__':
    # base = wd.World(cam_pos=[0.02001557, 0.00637317, 0.01088133], w=960,
    #                     h=540, lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[20.01557, 6.37317, 10.88133], w=960,
    #                 h=540, lookat_pos=[0, 0, 0])
    base = wd.World(cam_pos=[0.0457471, -0.0103006, -0.342874], w=960,
                    h=540, lookat_pos=[0, 0, 0])

    # 批量计算
    objname_list = [f"{i:04d}" for i in range(7, 50)]
    simple_obj_list = ["0002", "0003", "0000", "0008", "0009", "0016", "0022", "0023", "0030", "0033",
                       "0037", "0044", "0046", "0047", "0048", "0051", "0054", "0067", "0082", "0088",
                       "0091", "0097", "0101", "0110", "0113", "0115", "0117", "0124", "0129", "0150",
                       "0154", "0156", "0161", "0162", "0168", "0188", "0194", "0213", "0237", "0247",
                       "0250", "0263", "0278", "0296", "0326", "0409", "0416", "0421", "0428", "0548",
                       "0564", "0631", "0632", "0667", "0690", "0703", "0747", "0808", "0989", "0911"]
    success_obj_list = ["0002", "0016", "0022", "0044"]
    failed_obj_list = ["0009", "0030", "0048"]
    no_corner_obj_list = ["0003", "0000", "0008", "0033", "0037", "0047", "0051"]
    no_edge_obj_list = ["0023", "0046"]
    complex_obj_list = ["0130", "0133", "0175", "0222", "0238", "0275", "0351", "0379", "0482", "0538",
                        "0535", "0559", "0584", "0637", "0653", "0845", "0924", "1146", "1132", "0040"]
    error_obj_list = ["0031", "0032", "0041", "0066", "0086", "0196", "0191", "0235", "0272", "0330",
                      "0412", "0407", "0500", "0604", "0600", "0628", "0642", "0969", "1291", "0847"]
    # for objname in simple_obj_list:
    for objname in ["bunny_s"]:
        objpath = f"./stl/{objname}.stl"
        collision_model = cm.CollisionModel(objpath)
        collision_model.set_rgba((0.7, 0.7, 0.7, 1))
        collision_model.attach_to(base)
        # base.run()

        Mesh = o3d.io.read_triangle_mesh(objpath)
        Mesh.compute_vertex_normals()
        pcd = Mesh.sample_points_poisson_disk(number_of_points=10000)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        # pcd = o3d.io.read_point_cloud(
        #     f'D:/qmc/wrs-qiu-master/0000_edge_check_deep_learning/data_utils/ABC_data/{objname}.ply')

        k, threshold = 30, 15
        # pcd, ind = pcd.remove_radius_outlier(nb_points=2, radius=radius)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        pcd_np = vdda.o3dpcd_to_parray(pcd)
        pcd_normal_np = np.asarray(pcd.normals)[:, :]
        pcd_sample = pcd.uniform_down_sample(3)
        pcd_sample_np = vdda.o3dpcd_to_parray(pcd_sample)
        # gm.gen_pointcloud(pcd_np).attach_to(base)
        # base.run()

        # edge_np, edge_normal_np, surface_np, edge_index, surface_index = edge_detection(pcd_np, pcd_normal_np, k, threshold, load=False, show=True)
        edge_np, edge_normal_np, surface_np, edge_index, surface_index = edge_detection_normal(pcd_np, pcd_normal_np, k, threshold, load=False, show=True)
        # sampled_surface_np = voxel_grid_filter(surface_np, 0.002)

        # precision, recall = precision_check(objname, pcd, edge_index, surface_index, radius, threshold, True)
        # distance = calculate_distance(pcd, pcd_np, edge_np, True)
        # gm.gen_pointcloud(edge_np, [[1, 0, 0, 1]], 3).attach_to(base)

        # extend_edge_np = extend_point_cloud(edge_np, 0.5, 2)
        # edge_np, surface_np = edge_detection(extend_edge_np, radius=1, threshold=2, load=False, show=False)
        # gm.gen_pointcloud(edge_np, [[191/255, 144/255, 0, 1]], 7).attach_to(base)
        # gm.gen_pointcloud(sampled_surface_np, [[46/255, 117/255, 182/255, 1]], 8).attach_to(base)
        # for i in range(len(surface_np)):
        #     gm.gen_sphere(surface_np[i], subdivisions=2, radius=0.001, rgba=[46/255, 117/255, 182/255, 9]).attach_to(base)
        # for i in range(len(edge_np)):
        #     gm.gen_sphere(edge_np[i], radius=0.001, rgba=[191/255, 144/255, 0, 1]).attach_to(base)
        # base.run()

        # fit_line_from_point(edge_np, radius=0.01, threshold=0.008, maxIteration=1000)
        base.run()

        # 利用标注edge检测corner
        orig_pcd_colors = np.asarray(pcd.colors)
        mask = np.all(orig_pcd_colors == 1, axis=1)
        straight_edge_ind = np.where(np.all(orig_pcd_colors == [1, 0, 0], axis=1))
        arc_edge_ind = np.where(np.all(orig_pcd_colors == [0, 1, 0], axis=1))
        orig_edge_pnt_ind = np.where(~mask)[0]
        orig_surf_pnt_ind = np.where(mask)[0]
        orig_edge_pnt = pcd_np[orig_edge_pnt_ind]
        orig_edge_normal = pcd_normal_np[orig_edge_pnt_ind]
        straight_edge_pnt = pcd_np[straight_edge_ind]
        gm.gen_pointcloud(orig_edge_pnt, rgbas=[[0, 1, 0, 1]]).attach_to(base)
        corner_np = corner_check(edge_np, edge_normal_np, k=20, threshold=0.60, eps=0.01, load=False, show=False)
        average_corner = cluster_corner_result(corner_np, eps=0.003, min_samples=1, load=False, show=True)
        # corner_np = corner_check(straight_edge_pnt, k=20, threshold=0.5, eps=1, load=True, show=True)
        # average_corner = cluster_corner_result(corner_np, eps=1, min_samples=1, load=True, show=True)
        print(f'{objname} complete!')


    base.run()
