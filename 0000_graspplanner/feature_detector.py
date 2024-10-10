from sklearn.cluster import DBSCAN
import pyransac3d as pyrsc

import visualization.panda.world as wd
import modeling.collision_model as cm
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
import numpy as np
import basis.robot_math as rm
import modeling.geometric_model as gm
import os
import pickle
import basis.data_adapter as da
import itertools

import basis.trimesh as trimeshWan
import trimesh as trimesh
# import trimeshwraper as tw
import open3d as o3d
# import open3d.geometry as o3dg
import vision.depth_camera.pcd_data_adapter as vdda
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq85
import time


class FeatureDetector:
    def __init__(self, objpath, objname, show_sample_contact=True):
        gm.gen_frame(length=0.01, thickness=0.001).attach_to(base)
        self.objname = objname
        self.objpath = objpath
        self._get_pcd(self.objpath, sample="poisson", pcd_num=10000, neighbour=50, radius=0.05, save=True)
        # self.show_obj_pcd()
        self.show_obj_mesh(rgba=[0.5, 0.5, 0.5, 1])
        # base.run()
        self.pcd_sample, self.pcd_sample_np, self.pcd_normal_sample, self.pcd_normal_sample_np = self._sample_contact(
            show=False)
        print("ss")

        t_edge_start = time.time()
        # self.detect_edge(threshold=0.7, r=0.005, k=30, load=False, save=True, show=False)
        self.detect_edge(threshold=0.7, r=0.005, k=30, load=False, save=True, show=False)
        t_edge_end = time.time()
        self.show_edge_pnt()
        # self.show_surface_pnt()

        # self.fit_line_from_point(self.edge_pnt, radius=0.01, threshold=0.008, maxIteration=1000)
        # base.run()

        t_vertex_start = time.time()
        # self.detect_vertex(threshold=0.6, r=0.01, k=20, load=False, save=True, show=False)
        self.detect_vertex(threshold=0.6, r=0.01, k=20, load=False, save=True, show=False)
        self.show_surface_pnt()
        t_vertex_end = time.time()

        # self.cluster_vertex_result(eps=0.003, min_samples=1, load=False, save=True, show=False)
        self.cluster_vertex_result(eps=0.003, min_samples=1, load=False, save=True, show=False)
        self.show_vertex_pnt()
        # self.show_vertex_pnt()

        print(f"Edge detection: {t_edge_end - t_edge_start}")
        print(f"vertex detection: {t_vertex_end - t_vertex_start}")
        base.run()
        self.plan_poses()

    def show_obj_pcd(self):
        gm.gen_pointcloud(self.pcd_np).attach_to(base)

    def show_obj_mesh(self, rgba=[0.5, 0.5, 0.5, 1]):
        self.collision_model = cm.CollisionModel(objpath)
        self.collision_model.set_rgba(rgba)
        # self.collision_model.set_scale((2, 2, 2))
        self.collision_model.attach_to(base)

    def show_edge_pnt(self, radius=0.001, rgba=[191 / 255, 144 / 255, 0, 1]):
        for i in range(len(self.edge_pnt)):
            gm.gen_sphere(self.edge_pnt[i], subdivisions=2, radius=radius,
                          rgba=rgba).attach_to(base)

    def show_surface_pnt(self, radius=0.001, rgba=[46 / 255, 117 / 255, 182 / 255, 1]):
        for i in range(len(self.surface_pnt)):
            gm.gen_sphere(self.surface_pnt[i], subdivisions=2, radius=radius,
                          rgba=rgba).attach_to(base)

    def show_vertex_pnt(self, radius=0.001, rgba=[1, 0, 0, 1]):
        for i in range(len(self.vertex_pnt_clustered)):
            gm.gen_sphere(self.vertex_pnt_clustered[i], subdivisions=2, radius=radius,
                          rgba=rgba).attach_to(base)

    def _get_pcd(self, objpath, sample="poisson", pcd_num=10000, neighbour=50, radius=0.05, save=False):
        mesh = o3d.io.read_triangle_mesh(objpath)
        mesh.compute_vertex_normals()
        if sample == "poisson":
            pcd = mesh.sample_points_poisson_disk(number_of_points=pcd_num)
        elif sample == "uniform":
            pcd = mesh.sample_points_uniformly(number_of_points=pcd_num, use_triangle_normal=False)
        self.pcd, self.ind = pcd.remove_radius_outlier(nb_points=neighbour, radius=radius)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        # number_of_points=5000
        # nb_points=50, radius=0.05
        self.pcd_np = vdda.o3dpcd_to_parray(pcd)
        # self.pcd_normal = np.asarray(pcd.normals)[:, :]
        self.pcd_normal = []
        if save:
            o3d.io.write_point_cloud(f"debug_data/{self.objname}.pcd", self.pcd)
            # self.outputinfo(f"debug_data/{self.objname}/pcd.pickle", self.pcd)
            self.outputinfo(f"debug_data/{self.objname}/pcd_id.pickle", self.ind)

    def detect_edge(self, threshold=0.5, r=0.003, k=20, load=False, save=False, show=True):
        r = r
        if load:
            self.edge_pnt = self.importinfo(f"debug_data/{self.objname}/edge_pnt.pickle")
            self.edge_normal = self.importinfo(f"debug_data/{self.objname}/edge_normal.pickle")
            self.edge_id = self.importinfo(f"debug_data/{self.objname}/edge_id.pickle")
            self.surface_pnt = self.importinfo(f"debug_data/{self.objname}/surface_pnt.pickle")
            self.surface_normal = self.importinfo(f"debug_data/{self.objname}/surface_normal.pickle")
            self.surface_id = self.importinfo(f"debug_data/{self.objname}/surface_id.pickle")
            self.edge_detect_parameter = self.importinfo(f"debug_data/{self.objname}/edge_detect_parameter.pickle")
            self.pcd_normal = self.importinfo(f"debug_data/{self.objname}/pcd_normal.pickle")
            print('Edge detect parameter: ', self.edge_detect_parameter)
        else:
            self.edge_pnt = []
            self.edge_normal = []
            self.edge_id = []
            self.surface_pnt = []
            self.surface_normal = []
            self.surface_id = []
            self.edge_detect_parameter = {'threshold': threshold, 'r': r, 'k': k}
            # self.convex = []
            for i, anchor in enumerate(self.pcd_sample_np):
                anchor, anchor_normal, pcd_neighbor_np, pcd_neighbor_normal_np = self._get_neibour_detect(anchor,
                                                                                                          i,
                                                                                                          threshold=threshold,
                                                                                                          radius=r, k=k)

            self.surface_pnt = np.asarray(self.surface_pnt)
            self.surface_normal = np.asarray(self.surface_normal)
            self.surface_id = np.asarray(self.surface_id)
            self.edge_pnt = np.asarray(self.edge_pnt)
            self.edge_normal = np.asarray(self.edge_normal)
            self.edge_id = np.asarray(self.edge_id)
            self.pcd_normal = np.asarray(self.pcd_normal)

            for id, pnt in enumerate(self.edge_pnt):
                if id % 20 == 0:
                    gm.gen_arrow(spos=pnt, epos=pnt + 0.1 * self.edge_normal[id]).attach_to(base)

            if save:
                self.outputinfo(f"debug_data/{self.objname}/edge_pnt.pickle", self.edge_pnt)
                self.outputinfo(f"debug_data/{self.objname}/edge_normal.pickle", self.edge_normal)
                self.outputinfo(f"debug_data/{self.objname}/edge_id.pickle", self.edge_id)
                self.outputinfo(f"debug_data/{self.objname}/surface_pnt.pickle", self.surface_pnt)
                self.outputinfo(f"debug_data/{self.objname}/surface_normal.pickle", self.surface_normal)
                self.outputinfo(f"debug_data/{self.objname}/surface_id.pickle", self.surface_id)
                self.outputinfo(f"debug_data/{self.objname}/edge_detect_parameter.pickle", self.edge_detect_parameter)
                self.outputinfo(f"debug_data/{self.objname}/pcd_normal.pickle", self.pcd_normal)
        if show:
            gm.gen_pointcloud(self.edge_pnt, [[191 / 255, 144 / 255, 0, 1]], 6).attach_to(base)

    def outputinfo(self, name, data, write="wb"):
        with open(name, write) as file:
            pickle.dump(data, file)

    def importinfo(self, name):
        with open(name, "rb") as file:
            f = pickle.load(file)
        return f

    def detect_vertex(self, threshold=1.0, r=0.01, k=20, toggle_nb=False, load=False, save=False, show=False):
        r = r
        if load:
            self.vertex_pnt = self.importinfo(f"debug_data/{self.objname}/vertex_pnt.pickle")
            self.vertex_normal = self.importinfo(f"debug_data/{self.objname}/vertex_normal.pickle")
            self.vertex_id = self.importinfo(f"debug_data/{self.objname}/vertex_id.pickle")
            self.vertex_detect_parameter = self.importinfo(f"debug_data/{self.objname}/vertex_detect_parameter.pickle")
            print('Vertex detect parameter: ', self.vertex_detect_parameter)
        else:
            self.vertex_pnt = []
            self.vertex_normal = []
            self.vertex_id = []
            self.vertex_detect_parameter = {'threshold': threshold, 'r': r, 'k': k}

            for id_num, pnt in enumerate(self.edge_pnt):
                if toggle_nb:
                    range_ball_cm = cm.gen_sphere(pos=pnt, radius=r)
                    range_ball_cm.set_rgba((1, 0.3, 0.6, 0.2))
                    range_ball_cm.attach_to(base)

                self.edge_pcd_tree = o3d.geometry.KDTreeFlann(vdda.nparray_to_o3dpcd(np.asarray(self.edge_pnt)))
                # k, idx, _ = self.edge_pcd_tree.search_radius_vector_3d(pnt, r)
                _, idx, _ = self.edge_pcd_tree.search_knn_vector_3d(pnt, k)
                pcd_neighbor_normal_np = np.vstack([self.edge_normal[idx[1:]], self.edge_normal[id_num]])
                pcd_neighbor_np = np.vstack([self.edge_pnt[idx[1:]], pnt])
                pcd_neighbor_np, pcd_neighbor_normal_np = self.cluster_input(pnt, pcd_neighbor_np,
                                                                             pcd_neighbor_normal_np,
                                                                             normal_cluster=False)
                if len(pcd_neighbor_np) == 1:
                    print(f'{id_num} is outlier!')
                    continue

                anchor_normal = pcd_neighbor_normal_np.mean(axis=0)
                if self._is_vertex(pnt, pcd_neighbor_np, threshold):
                    self.vertex_pnt.append(pnt)
                    self.vertex_normal.append(anchor_normal)
                    self.vertex_id.append(self.edge_id[id_num])

            self.vertex_pnt = np.asarray(self.vertex_pnt)
            self.vertex_normal = np.asarray(self.vertex_normal)
            self.vertex_id = np.asarray(self.vertex_id)
            if save:
                self.outputinfo(f"debug_data/{self.objname}/vertex_pnt.pickle", self.vertex_pnt)
                self.outputinfo(f"debug_data/{self.objname}/vertex_normal.pickle", self.vertex_normal)
                self.outputinfo(f"debug_data/{self.objname}/vertex_id.pickle", self.vertex_id)
                self.outputinfo(f"debug_data/{self.objname}/vertex_detect_parameter.pickle",
                                self.vertex_detect_parameter)
        if show:
            for i in range(len(self.vertex_pnt)):
                gm.gen_sphere(pos=self.vertex_pnt[i], radius=0.005, rgba=[0, 0, 1, 0.5]).attach_to(base)

    def cluster_vertex_result(self, eps, min_samples, load=False, save=True, show=True):
        if load:
            self.vertex_pnt_clustered = self.importinfo(f"debug_data/{self.objname}/vertex_pnt_clustered.pickle")
            self.vertex_normal_clustered = self.importinfo(f"debug_data/{self.objname}/vertex_normal_clustered.pickle")
            self.vertex_id_clustered = self.importinfo(f"debug_data/{self.objname}/vertex_id_clustered.pickle")
            self.vertex_clustered_parameter = self.importinfo(
                f"debug_data/{self.objname}/vertex_clustered_parameter.pickle")
            print('Vertex clustered parameter: ', self.vertex_clustered_parameter)
        else:
            self.vertex_clustered_parameter = {'eps': eps, 'min_samples': min_samples}
            if len(self.vertex_pnt) == 0:
                self.vertex_pnt_clustered = np.asarray([])
                self.vertex_normal_clustered = np.asarray([])
                self.vertex_id_clustered = np.asarray([])
            else:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(self.vertex_pnt)
                unique_labels = set(labels)
                self.vertex_pnt_clustered = []
                self.vertex_normal_clustered = []
                self.vertex_id_clustered = []
                for label in unique_labels:
                    cluster_points_id = np.where(labels == label)[0]
                    cluster_points = self.vertex_pnt[cluster_points_id]
                    cluster_points_normal = self.vertex_normal[cluster_points_id]
                    cluster_mean = np.mean(cluster_points, axis=0)
                    cluster_distance = np.linalg.norm(cluster_points - cluster_mean, axis=1)
                    min_distance_index = np.argmin(cluster_distance)
                    corner_clustered_id = cluster_points_id[min_distance_index]
                    self.vertex_pnt_clustered.append(cluster_points[min_distance_index])
                    self.vertex_id_clustered.append(self.vertex_id[corner_clustered_id])
                    self.vertex_normal_clustered.append(cluster_points_normal[min_distance_index])
                self.vertex_pnt_clustered = np.asarray(self.vertex_pnt_clustered)
                self.vertex_id_clustered = np.asarray(self.vertex_id_clustered)
                self.vertex_normal_clustered = np.asarray(self.vertex_normal_clustered)
            if save:
                self.outputinfo(f"debug_data/{self.objname}/vertex_pnt_clustered.pickle", self.vertex_pnt_clustered)
                self.outputinfo(f"debug_data/{self.objname}/vertex_normal_clustered.pickle",
                                self.vertex_normal_clustered)
                self.outputinfo(f"debug_data/{self.objname}/vertex_id_clustered.pickle", self.vertex_id_clustered)
                self.outputinfo(f"debug_data/{self.objname}/vertex_clustered_parameter.pickle",
                                self.vertex_clustered_parameter)
        if show:
            for i in range(len(self.vertex_pnt_clustered)):
                gm.gen_sphere(pos=self.vertex_pnt_clustered[i], radius=0.002, rgba=[192 / 255, 0, 0, 1]).attach_to(base)

    def fit_line_from_point(self, points, radius, threshold, maxIteration):
        pcd = vdda.nparray_to_o3dpcd(points)
        random_point = points[np.random.randint(0, len(points) - 1)]
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        [k, idx, _] = kdtree.search_radius_vector_3d(random_point, radius)
        neighbor_points = points[idx]
        line_model = pyrsc.Line()
        A, B, inliers = line_model.fit(neighbor_points, threshold, maxIteration)
        inliers_points = neighbor_points[inliers]
        inliers_mean = np.mean(inliers_points, axis=0)
        center_points = inliers_points - inliers_mean
        _, _, Vt = np.linalg.svd(center_points)
        inliers_direction = Vt[0]
        t = [-2, 2]
        line_points = np.zeros((2, 3))
        line_points[0] = inliers_mean + inliers_direction * t[0]
        line_points[1] = inliers_mean + inliers_direction * t[1]
        gm.gen_sphere(random_point, radius, rgba=[0, 1, 0, 0.5]).attach_to(base)

        gm.gen_arrow(line_points[0], line_points[1], thickness=0.001, rgba=[1, 0, 0, 1]).attach_to(base)

    def _sample_contact(self, everynum=3, show=False):
        # self.pcd_sample = self.pcd.random_down_sample(rate)
        # pcd_sample = self.pcd.uniform_down_sample(everynum)
        pcd_sample = self.pcd
        pcd_normal_sample = self.pcd.normals
        pcd_normal_sample_np = np.asarray(pcd_normal_sample)
        pcd_sample_np = vdda.o3dpcd_to_parray(pcd_sample)
        if show:
            for pnt in pcd_sample_np:
                # edge_cm = cm.gen_sphere(pos=pnt, rgba=(1, 0, 0, 1), radius=0.001).attach_to(base)
                edge_cm = cm.gen_sphere(pos=pnt, rgba=(1, 0, 0, 0.3), radius=0.005)
                edge_cm.attach_to(base)
        return pcd_sample, pcd_sample_np, pcd_normal_sample, pcd_normal_sample_np

    def _get_neibour_detect(self, anchor, id_num, threshold, radius=0.0025, k=20, toggle_nb=False):
        if toggle_nb:
            range_ball_cm = cm.gen_sphere(pos=anchor, radius=radius)
            range_ball_cm.set_rgba((1, 0.3, 0.6, 0.2))
            range_ball_cm.attach_to(base)
        # k, idx, _ = self.pcd_tree.search_radius_vector_3d(anchor, radius)
        _, idx, _ = self.pcd_tree.search_knn_vector_3d(anchor, k)
        pcd_neighbor_normal_np_ori = np.vstack([self.pcd_normal_sample_np[idx[1:]], self.pcd_normal_sample_np[id_num]])
        pcd_neighbor_np = np.vstack([self.pcd_np[idx[1:]], anchor])
        pcd_neighbor_np, pcd_neighbor_normal_np = self.cluster_input(anchor, pcd_neighbor_np,
                                                                     pcd_neighbor_normal_np_ori,
                                                                     normal_cluster=True)
        if len(pcd_neighbor_np) == 1:
            print(f'{id_num} is outlier!')
        pcd_neighbor_np, pcd_neighbor_normal_np, is_surface = self._tune_sample(anchor, pcd_neighbor_np,
                                                                                pcd_neighbor_normal_np, threshold)
        anchor_normal = pcd_neighbor_normal_np_ori.mean(axis=0)
        if is_surface:
            self.surface_pnt.append(anchor)
            self.surface_normal.append(anchor_normal)
            self.surface_id.append(id_num)
        else:
            self.edge_pnt.append(anchor)
            self.edge_normal.append(anchor_normal)
            self.edge_id.append(id_num)
        self.pcd_normal.append(anchor_normal)

        return anchor, anchor_normal, pcd_neighbor_np, pcd_neighbor_normal_np

    def _is_surface(self, anchor, pcd_neighbor_np, threshold):
        if len(pcd_neighbor_np) == 1:
            is_surface = True
        else:
            datamean = pcd_neighbor_np.mean(axis=0)
            distance_neighbor_center = []
            for i in range(len(pcd_neighbor_np)):
                distance_neighbor_center.append(np.linalg.norm(anchor - pcd_neighbor_np[i]))
            distance_neighbor_center_min = np.sort(np.array(distance_neighbor_center))[1]
            distance_query = np.linalg.norm(datamean - anchor)
            if distance_query > (distance_neighbor_center_min * threshold):
                is_surface = False
            else:
                is_surface = True
        return is_surface

    def _is_vertex(self, anchor, pcd_neighbor_np, threshold):
        datamean = pcd_neighbor_np.mean(axis=0)
        eigen_vals, eigen_vecs = self.pca(pcd_neighbor_np - datamean)
        PC = eigen_vals / np.sum(eigen_vals, axis=0)
        if np.max(PC) < threshold:
            is_vertex = True
            print("----is vertex")
            print(PC)
        else:
            is_vertex = False
        return is_vertex

    def _tune_sample(self, anchor, pcd_neighbor_np, pcd_neighbor_normal_np, threshold=1.0):

        is_surface = self._is_surface(anchor, pcd_neighbor_np, threshold)

        return pcd_neighbor_np, pcd_neighbor_normal_np, is_surface

    def plan_poses(self, width_detect=0.005, length_detect=0.09, show_neighbor=True):
        for anchor in self.pcd_sample_np:
            anchor, anchor_normal, pcd_neighbor_np, pcd_neighbor_normal_np = self._get_neibour(anchor)
            # anchor_normal = pcd_neighbor_normal_np.mean(axis=0)
            if show_neighbor:
                # gm.gen_arrow(anchor, anchor + anchor_normal * 0.02, thickness=0.001).attach_to(base)
                for i, pnt in enumerate(pcd_neighbor_np):
                    edge_cm = cm.gen_sphere(pos=pnt, radius=0.0005)
                    edge_cm.set_rgba((0, 1, 0, 0.5))
                    edge_cm.attach_to(base)
                    # gm.gen_arrow(pnt, pnt + pcd_neighbor_normal_np[i] * 0.010, thickness=0.0004).attach_to(base)
            width_detect = width_detect
            length_detect = length_detect

            detect_ray = gm.gen_stick(anchor - anchor_normal * length_detect,
                                      anchor - anchor_normal * 0.05 * length_detect,
                                      thickness=width_detect * np.sin(np.pi * 20 / 180) * 2, sections=7,
                                      rgba=(0, 1, 0, 0.3))
            # detect_ray.attach_to(base)
            detect_ray_model = detect_ray.objtrm.export("detect_ray.stl")
            inside_points = []
            outside_points = []
            detect_mesh = trimesh.load_mesh("detect_ray.stl")
            checker = trimesh.base.ray.ray_pyembree.RayMeshIntersector(detect_mesh)
            for point in self.pcd.points:
                if checker.contains_points(points=[point]):
                    inside_points.append(point)
                else:
                    outside_points.append(point)
            # print(inside_points)
            if len(inside_points):
                pcd_in = vdda.nparray_to_o3dpcd(np.asarray(inside_points))
                labels = np.array(pcd_in.cluster_dbscan(eps=0.003, min_points=1, print_progress=False))
                num_cluster = labels.max() + 1
                # print(labels)
                print(num_cluster)
                clustered_pcd_opposit = [[] * num_cluster]
                try:
                    for index, label in enumerate(labels):
                        clustered_pcd_opposit[label].append(inside_points[index])
                except:
                    continue
                print(clustered_pcd_opposit)
                clustered_ave_pcd_opposit_np = np.asarray(clustered_pcd_opposit).mean(axis=1)
                print(clustered_ave_pcd_opposit_np)

                for i, pnt in enumerate(inside_points):
                    edge_cm = cm.gen_sphere(pos=pnt, radius=0.0005)
                    edge_cm.set_rgba((0, 1, 0, 0.5))
                    # edge_cm.attach_to(base)

                anchor_opposite = clustered_ave_pcd_opposit_np[0]
                opposite_range_ball_cm = cm.gen_sphere(pos=anchor_opposite, radius=0.005)
                opposite_range_ball_cm.set_rgba((1, 0.3, 0.6, 0.2))
                # opposite_range_ball_cm.attach_to(base)

                [k, idx, _] = self.pcd_tree.search_radius_vector_3d(anchor_opposite, 0.005)
                pcd_opposite_neighbor_normal_np = np.asarray([self.pcd.normals[i] for i in idx])
                pcd_opposite_neighbor_np = vdda.o3dpcd_to_parray(self.pcd)[idx[1:]]

                for i, pnt in enumerate(pcd_opposite_neighbor_np):
                    edge_cm = cm.gen_sphere(pos=pnt, radius=0.0005)
                    edge_cm.set_rgba((0, 0, 1, 0.5))
                    # edge_cm.attach_to(base)
                    # gm.gen_arrow(pnt, pnt + pcd_opposite_neighbor_normal_np[i] * 0.010, thickness=0.0004).attach_to(base)

                anchor_opposite_normal = pcd_opposite_neighbor_normal_np.mean(axis=0)
                # gm.gen_arrow(anchor_opposite, anchor_opposite + anchor_opposite_normal * 0.02, thickness=0.001).attach_to(
                #     base)
                if self._is_surface(pcd_neighbor_np):
                    angle_threshold = 40
                else:
                    angle_threshold = 40
                print("here", rm.angle_between_vectors(anchor_normal, -anchor_opposite_normal) * 180 / np.pi)
                if (rm.angle_between_vectors(anchor_normal, -anchor_opposite_normal) * 180 / np.pi) <= angle_threshold:
                    print("check")
                    gripper = rtq85.Robotiq85()
                    gripper_y = rm.unit_vector(anchor_normal - anchor_opposite)
                    # rm.angle_between_vectors(anchor_normal,np.array([1,0,0]))
                    # t_ar = rm.rotmat_between_vectors( np.array([1,0,0]), anchor_normal)
                    t_ar = rm.rotmat_between_vectors(np.array([0, 1, 0]), gripper_y)
                    jaw_width = np.linalg.norm(anchor - anchor_opposite) + 0.01
                    if jaw_width <= 0.085:
                        gripper.grip_at_with_jcpose((anchor + anchor_opposite) / 2, gl_jaw_center_rotmat=t_ar,
                                                    jaw_width=jaw_width)
                    else:
                        continue
                    for rotate_angle in np.arange(0, 360, 60):
                        print("t")
                        tmp_rotmat = rm.rotmat_from_axangle(gripper_y, np.pi * rotate_angle / 180)
                        rotmat = np.dot(tmp_rotmat, t_ar)
                        gripper.grip_at_with_jcpose((anchor + anchor_opposite) / 2, gl_jaw_center_rotmat=rotmat,
                                                    jaw_width=jaw_width)
                        if not gripper.is_mesh_collided([self.collision_model]):
                            # gripper.gen_meshmodel(toggle_tcpcs=False, rgba=(0,1,0,0.1)).attach_to(base)
                            gripper.gen_meshmodel(toggle_tcpcs=False).attach_to(base)
                            break

                else:
                    pass
            else:
                pass

    def pca(self, X):
        n, m = X.shape
        assert np.allclose(X.mean(axis=0), np.zeros(m))  # 确保X已经中心化，每个维度的均值为0
        eigen_vals, eigen_vecs = np.linalg.eig(np.dot(X.T, X) / (n - 1))
        return eigen_vals, eigen_vecs

    def cluster_input(self, anchor, points, normals, normal_cluster=False):
        points_scale = self.scale_compute(points)
        dbscan_points = DBSCAN(eps=points_scale / 2, min_samples=1)
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

            normals_scale = self.scale_compute(result_normals)
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

    def scale_compute(self, data):
        x_scale = np.min(data[:, 0]) - np.max(data[:, 0])
        y_scale = np.min(data[:, 1]) - np.max(data[:, 1])
        z_scale = np.min(data[:, 2]) - np.max(data[:, 2])
        max_scale = np.max(np.abs([x_scale, y_scale, z_scale]))
        if max_scale == 0:
            max_scale = 1
        return max_scale


if __name__ == '__main__':
    base = wd.World(cam_pos=[0.163647, 0.164754, 0.140649], w=960,
                    h=540, lookat_pos=[0, 0, 0])
    # base = wd.World(cam_pos=[287.343, 206.852, -204.502], w=960,
    #                 h=540, lookat_pos=[0, 0, 0])
    # gm.gen_frame().attach_to(base)
    this_dir, this_filename = os.path.split(__file__)
    # objpath = "kit_model_stl/CatSitting_800_tex.stl"
    # objpath = "kit_model_stl/CoffeeBox_800_tex.stl"
    # objpath = "test_obj/tetrahedron.stl"
    # objpath = "kit_model_stl/RedCup_800_tex.stl"
    # objpath = "test_obj/ratchet.stl"
    # objpath = "kit_model_stl/InstantSoup_800_tex.stl"
    # objname = "cupramen"
    # objname = "tetrahedron"
    objname = "arrow"
    # objname = 'hexagonal_prisms'
    # objname = 'Amicelli_800_tex'
    # objpath = f"kit_model_stl/{objname}.stl"

    objpath = f"test_obj/{objname}.stl"
    # objname = "CatSitting_800_tex"
    # objname = "CatLying_800_tex"
    # objname = "CoffeeBox_800_tex"
    # objpath = f"stl/{objname}.stl"
    graspplanner = FeatureDetector(objpath, objname)

    # gm.gen_frame().attach_to(base)
    base.run()


    def update(textNode, count, task):

        if textNode[0] is not None:
            textNode[0].detachNode()
            textNode[1].detachNode()
            textNode[2].detachNode()
        cam_pos = base.cam.getPos()
        textNode[0] = OnscreenText(
            text=str(cam_pos[0])[0:5],
            fg=(1, 0, 0, 1),
            pos=(1.0, 0.8),
            align=TextNode.ALeft)
        textNode[1] = OnscreenText(
            text=str(cam_pos[1])[0:5],
            fg=(0, 1, 0, 1),
            pos=(1.3, 0.8),
            align=TextNode.ALeft)
        textNode[2] = OnscreenText(
            text=str(cam_pos[2])[0:5],
            fg=(0, 0, 1, 1),
            pos=(1.6, 0.8),
            align=TextNode.ALeft)
        return task.again


    cam_view_text = OnscreenText(
        text="Camera View: ",
        fg=(0, 0, 0, 1),
        pos=(1.15, 0.9),
        align=TextNode.ALeft)
    testNode = [None, None, None]
    count = [0]
    taskMgr.doMethodLater(0.01, update, "addobject", extraArgs=[testNode, count],
                          appendTask=True)

    base.run()
