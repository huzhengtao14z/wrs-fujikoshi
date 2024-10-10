import math
import pickle
import numpy as np
import pyransac3d as pyrsc
# pip3 install pyransac3d

import modeling.geometric_model as gm
import basis.robot_math as rm
import visualization.panda.world as wd
import modeling.collision_model as cm
import vision.depth_camera.pcd_data_adapter as vdda

import random

def svd(data):
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    U, S, Vt = np.linalg.svd(centered_data)
    direction = Vt[0]
    return mean, direction


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


def fit_lines(data, threshold, maxIteration, least_pnt_num, least_inlier_points=50, show=True, load=True):
    if load:
        with open("./fit_line_data/0002/inliers_points_list.pickle", 'rb') as file1:
            inliers_points_list = pickle.load(file1)
        with open("./fit_line_data/0002/line_points_list.pickle", 'rb') as file2:
            line_points_list = pickle.load(file2)
    else:
        A_list = []
        B_list = []
        inliers_points_list = []
        line_points_list = []
        fit_data = data
        line_model = pyrsc.Line()
        while len(fit_data) > least_pnt_num:
            A, B, inliers = line_model.fit(fit_data, threshold, maxIteration)
            if len(inliers) > least_inlier_points:
                A_list.append(A)
                B_list.append(B)
                inliers_points = fit_data[inliers]
                inliers_points_list.append(inliers_points)
                inliers_mean, inliers_direction = svd(fit_data[inliers])
                t = [-20, 20]
                line_points = np.zeros((2, 3))
                line_points[0] = inliers_mean + inliers_direction * t[0]
                line_points[1] = inliers_mean + inliers_direction * t[1]
                line_points_list.append(line_points)
            mask = np.ones(fit_data.shape[0], dtype=bool)
            mask[inliers] = False
            fit_data = fit_data[mask]
        with open("./fit_line_data/0002/inliers_points_list.pickle", 'wb') as file1:
            pickle.dump(inliers_points_list, file1)
        with open("./fit_line_data/0002/line_points_list.pickle", 'wb') as file2:
            pickle.dump(line_points_list, file2)

    if show:
        for i in range(len(inliers_points_list)):
            np.random.seed(0)
            gm.gen_pointcloud(inliers_points_list[i], rgbas=[np.random.rand(4, )]).attach_to(base)
            gm.gen_arrow(spos=np.array(line_points_list[i][0]), epos=np.array(line_points_list[i][1]), thickness=0.05,
                         rgba=np.random.rand(4, )).attach_to(base)

    return inliers_points_list, line_points_list


if __name__ == "__main__":
    base = wd.World(cam_pos=[20.01557, 6.37317, 10.88133], w=960,
                    h=540, lookat_pos=[0, 0, 0])

    with open(f'./debug_data/0002/edge_np.pickle', 'rb') as file1:
        edge_np = pickle.load(file1)
    X = edge_np

    extended_X = extend_point_cloud(X, radius=0.5, num_samples=2)

    inliers, line_points = fit_lines(extended_X, 0.5, 1000,20, 100, True, False)

    gm.gen_pointcloud(extended_X, rgbas=[[1, 0, 0, 1]]).attach_to(base)
    base.run()
