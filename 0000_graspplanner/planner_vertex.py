import copy
import math
import visualization.panda.world as wd
import modeling.collision_model as cm
# import humath as hm
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
from trimesh.sample import sample_surface
from panda3d.core import NodePath
# import trimeshwraper as tw
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq85
import robot_sim.end_effectors.gripper.robotiq140.robotiq140 as rtq140
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtqhe
import open3d as o3d
# import open3d.geometry as o3dg
import vision.depth_camera.pcd_data_adapter as vdda
import time
class GraspPlanner():
    def __init__(self, objpath, objname, show_sample_contact = True):
        #load collsion model
        self.objname = objname
        self.collision_model = cm.CollisionModel(objpath)
        self.collision_model.set_rgba((0.5, 0.5, 0.5, 0.5))
        # self.collision_model.set_scale((2, 2, 2))
        self.collision_model.attach_to(base)
        # base.run()
        #load mesh and down sample
        Mesh = o3d.io.read_triangle_mesh(objpath)
        Mesh.compute_vertex_normals()
        pcd = Mesh.sample_points_poisson_disk(number_of_points=10000)
        # pcd = Mesh.sample_points_uniformly(number_of_points=10000, use_triangle_normal=False)
        self.pcd, ind = pcd.remove_radius_outlier(nb_points=50, radius=0.05)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        # number_of_points=5000
        # nb_points=50, radius=0.05
        self.pcd_np = vdda.o3dpcd_to_parray(pcd)
        self.pcd_normal_np = np.asarray(pcd.normals)[:, :]
        # self.showobj_pcd()
        # base.run()

        self.pcd_sample, self.pcd_sample_np = self._sample_contact( everynum=2, show = False)
        t_edge_start = time.time()
        self.detect_edge(self.pcd_sample_np, threshold_u = 50, threshold_l=0.02, r = 0.005, load = False, show=True)

        t_edge_end = time.time()
        t_vertex_start = time.time()
        base.run()
        self.detect_vertex(threshold_u = 10, threshold_l=0.1, r = 0.01, load = False)
        t_vertex_end = time.time()
        print(f"Edge detection: {t_edge_end - t_edge_start}")
        print(f"vertex detection: {t_vertex_end - t_vertex_start}")
        # self.detect_edge(load=True)
        # self.detect_vertex(load=True)
        # t_grasp_start = time.time()
        # self.show_edge_pcd()
        # t_grasp_end = time.time()
        # print(f"grasp: {t_grasp_end - t_grasp_start}")
        base.run()
        self.plan_poses()

    def detect_edge(self, cloud_np, threshold_u = 50, threshold_l=0.02, r = 0.003, show_neighber=False, load = False, show = True):
        r = r
        if load:
            self.edge_pnt = self.importinfo(f"debug_data/{self.objname}/edge_pnt.pickle")
            self.edge_normal = self.importinfo(f"debug_data/{self.objname}/edge_pnt.pickle")
            self.surface_pnt = self.importinfo(f"debug_data/{self.objname}/edge_pnt.pickle")
            self.surface_normal = self.importinfo(f"debug_data/{self.objname}/edge_pnt.pickle")
        else:
            self.edge_pnt = []
            self.edge_normal = []
            self.surface_pnt = []
            self.surface_normal = []
            # self.convex = []
            for anchor in cloud_np:
                anchor, anchor_normal, pcd_neighber_np, pcd_neighber_normal_np = self._get_neibour_detect(anchor, threshold_u, threshold_l, radius=r, show=show)
                # anchor_normal = pcd_neighber_normal_np.mean(axis=0)
                if show_neighber:
                    # gm.gen_arrow(anchor, anchor + anchor_normal * 0.02, thickness=0.001).attach_to(base)
                    for i, pnt in enumerate(pcd_neighber_np):
                        edge_cm = cm.gen_sphere(pos=pnt, radius=r)
                        edge_cm.set_rgba((0, 1, 0, 0.5))
                        edge_cm.attach_to(base)
            self.outputinfo(f"debug_data/{self.objname}/edge_pnt.pickle", self.edge_pnt)
            self.outputinfo(f"debug_data/{self.objname}/edge_normal.pickle", self.edge_normal)
            self.outputinfo(f"debug_data/{self.objname}/surface_pnt.pickle", self.surface_pnt)
            self.outputinfo(f"debug_data/{self.objname}/surface_normal.pickle", self.surface_normal)

    def outputinfo(self, name, data):
        with open(name, "wb") as file:
            pickle.dump(data, file)
    def importinfo(self, name):
        with open(name, "rb") as file:
            f = pickle.load(file)
        return  f

    def showobj_pcd(self):
        gm.gen_pointcloud(self.pcd_np).attach_to(base)

    def detect_vertex(self, threshold_u = 10, threshold_l = 0.1, r = 0.01, load=False):
        r = r
        if load:
            self.vertex_pnt_clustered = self.importinfo(f"debug_data/{self.objname}/vertex_pnt_clustered.pickle")
        else:
            self.vertex_pnt = []
            # edge_pnt_pcd = vdda.nparray_to_o3dpcd(np.asarray(self.edge_pnt))
            # edge_pnt_pcd.remove_radius_outlier(20, 0.001)
            # edge_pnt_np = vdda.o3dpcd_to_parray(edge_pnt_pcd)
            # gm.gen_pointcloud(edge_pnt_np, pntsize=10).attach_to(base)

            for pnt in self.edge_pnt:
                # range_ball_cm = cm.gen_sphere(pos=pnt, radius=r )
                # range_ball_cm.set_rgba((1, 0.3, 0.6, 0.2))
                # range_ball_cm.attach_to(base)
                try:
                    self.edge_pcd_tree = o3d.geometry.KDTreeFlann(vdda.nparray_to_o3dpcd(np.asarray(self.edge_pnt)))
                    [k, idx, _] = self.edge_pcd_tree.search_radius_vector_3d(pnt, r )
                    pcd_neighber_normal_np = np.asarray([self.edge_normal[i] for i in idx])
                    # pcd_neighber_np = vdda.o3dpcd_to_parray(vdda.nparray_to_o3dpcd(np.asarray(self.edge_pnt)))[idx[1:]]
                    pcd_neighber_np = np.asarray(self.edge_pnt)[idx[:]]
                    anchor_normal = pcd_neighber_normal_np.mean(axis=0)
                    if self._is_vertex(pcd_neighber_np, threshold_u, threshold_l):
                        self.vertex_pnt.append(pnt)
                        cm.gen_sphere(pos=pnt, rgba=(1, 0, 0, 0.3), radius=r ).attach_to(base)
                except:
                    pass
            vertex_pnt_o3d = vdda.nparray_to_o3dpcd(np.asarray(self.vertex_pnt))
            labels = np.array(vertex_pnt_o3d.cluster_dbscan(eps=0.01, min_points=1, print_progress=True))
            max_label = labels.max()
            print(f"point cloud has {max_label + 1} clusters")
            print(labels)

            self.vertex_pnt_clustered = []
            unique_groups = np.unique(labels)
            for j in unique_groups:
                mid = []
                for i, item in enumerate(self.vertex_pnt):
                    if labels[i] == j:
                        mid.append(item)
                mean_mid = np.mean(np.asarray(mid), axis=0)
                self.vertex_pnt_clustered.append(mean_mid)
            print(self.vertex_pnt_clustered)
            # for item in self.vertex_pnt_clustered:
            #     gm.gen_sphere(pos=item, rgba=(1,0,0,1), radius=0.0025).attach_to(base)
            # for i, item in enumerate(self.vertex_pnt):
            #     # self.vertex_pnt_clustered[labels[i]]=np.append(self.vertex_pnt_clustered[label[i]], item )
            #     np.random.seed(abs(labels[i]))
            #     random_elements = np.random.rand(3)
            #     one_element = np.array([0.3])
            #     result_array = np.concatenate((random_elements, one_element))
            #     # cm.gen_sphere(pos=item, rgba=result_array, radius=0.005).attach_to(base)
            self.outputinfo(f"debug_data/{self.objname}/vertex_pnt_clustered.pickle", self.vertex_pnt_clustered)

    def show_edge_pcd(self):

        # shape = (max_label + 1, 1, 3)
        # self.vertex_pnt_clustered = np.zeros(shape)

        for item in self.vertex_pnt_clustered:
            cm.gen_sphere(pos=item, rgba=(1,0,0,1), radius=0.002).attach_to(base)
        # self.vertex_grasp()

    def vertex_grasp(self):

        combinations = list(itertools.combinations(self.vertex_pnt_clustered, 2))
        # combinations = list(itertools.combinations(list(vdda.o3dpcd_to_parray(vdda.nparray_to_o3dpcd(np.asarray(self.vertex_pnt)).uniform_down_sample(20))),2))
        gripper = rtq85.Robotiq85()
        # print(combinations)
        for pair in combinations:
            # cm.gen_sphere(pos=pair[0], rgba=(1, 0, 0, 1), radius=0.002).attach_to(base)
            # cm.gen_sphere(pos=pair[1], rgba=(1, 0, 0, 1), radius=0.002).attach_to(base)
            # gm.gen_stick(pair[0], pair[1], thickness=0.003, rgba=(1, 0,0,1)).attach_to(base)
            center = (pair[0]+pair[1])/2

            # gripper = rtq140.Robotiq140()
            gripper_y = rm.unit_vector(pair[0] - pair[1])
            t_ar = rm.rotmat_between_vectors(np.array([0, 1, 0]), gripper_y)
            if np.linalg.norm(pair[0] - pair[1], ord=2)>0.085:
                continue
            else:
                if np.linalg.norm(pair[0] - pair[1], ord=2)>0.045:
                    gripper.grip_at_with_jcpose(center, gl_jaw_center_rotmat=t_ar, jaw_width=np.linalg.norm(pair[0] - pair[1], ord=2))
                    # gripper.gen_meshmodel(rgba=(0,1,0,0.2)).attach_to(base)
                    # gripper.gen_meshmodel().attach_to(base)
                    if gripper.is_mesh_collided(objcm_list=[self.collision_model]):
                        # gripper.gen_meshmodel(rgba=(1,0,0,0.2)).attach_to(base)
                        continue
                    else:
                        [k, idx, _] = self.pcd_tree.search_radius_vector_3d(pair[0], 0.012)
                        contact_a_normal_np = np.asarray([self.pcd.normals[i] for i in idx])
                        contact_a_neighber_np = vdda.o3dpcd_to_parray(self.pcd)[idx[1:]]
                        for i, pnt in enumerate(contact_a_neighber_np):
                            range_ball_cm = cm.gen_sphere(pos=pnt, radius=0.001)
                            # print(abs(np.dot(pnt-pair[0], rm.unit_vector(pair[0] - pair[1]))))
                            if abs(np.dot(pair[0]-pnt, rm.unit_vector(pair[1] - pair[0])))<0.001 and np.dot(rm.unit_vector(pair[1] - pair[0]), contact_a_normal_np[i])<0:
                                range_ball_cm.set_rgba((1, 0.3, 0.6, 1))
                            else:
                                range_ball_cm.set_rgba((1, 0.3, 0.6, 0.05))
                            range_ball_cm.attach_to(base)

                        [k, idx, _] = self.pcd_tree.search_radius_vector_3d(pair[1], 0.012)
                        contact_b_normal_np = np.asarray([self.pcd.normals[i] for i in idx])
                        contact_b_neighber_np = vdda.o3dpcd_to_parray(self.pcd)[idx[1:]]
                        for i, pnt in enumerate(contact_b_neighber_np):
                            range_ball_cm = cm.gen_sphere(pos=pnt, radius=0.001)
                            # print(abs(np.dot(pnt - pair[1], rm.unit_vector(pair[0] - pair[1]))))
                            if abs(np.dot(pair[1]-pnt , rm.unit_vector(pair[0] - pair[1]))) < 0.001 and np.dot(rm.unit_vector(pair[0] - pair[1]), contact_b_normal_np[i])<0:
                                range_ball_cm.set_rgba((1, 0.3, 0.6, 1))
                            else:
                                range_ball_cm.set_rgba((1, 0.3, 0.6, 0.05))
                            range_ball_cm.attach_to(base)

                        gripper.gen_meshmodel().attach_to(base)
                        break
                        # gripper.gen_meshmodel(rgba=(0, 1, 0, 0.2)).attach_to(base)
                        # break


    def _sample_contact(self, everynum = 3, show = True):
        # self.pcd_sample = self.pcd.random_down_sample(rate)
        pcd_sample = self.pcd.uniform_down_sample(everynum)
        pcd_sample_np = vdda.o3dpcd_to_parray(pcd_sample)
        if show:
            for pnt in pcd_sample_np:
                # edge_cm = cm.gen_sphere(pos=pnt, rgba=(1, 0, 0, 1), radius=0.001).attach_to(base)
                edge_cm = cm.gen_sphere(pos=pnt, rgba=(1, 0, 0, 0.3), radius=0.005)
                # edge_cm.attach_to(base)
        return pcd_sample, pcd_sample_np

    def _get_neibour(self, anchor, radius= 0.0025,show=True):
        # anchor = pcd_sample_np[0]
        range_ball_cm = cm.gen_sphere(pos=anchor, radius=radius)
        range_ball_cm.set_rgba((1, 0.3, 0.6, 0.2))
        # range_ball_cm.attach_to(base)
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        [k, idx, _] = self.pcd_tree.search_radius_vector_3d(anchor, radius)
        pcd_neighber_normal_np = np.asarray([self.pcd.normals[i] for i in idx])
        pcd_neighber_np = vdda.o3dpcd_to_parray(self.pcd)[idx[1:]]
        anchor_normal = pcd_neighber_normal_np.mean(axis=0)
        if show:
            for i, pnt in enumerate(pcd_neighber_np):
                edge_cm = cm.gen_sphere(pos=pnt, radius=0.0005)
                edge_cm.set_rgba((0, 1, 0, 0.5))
                edge_cm.attach_to(base)
                # gm.gen_arrow(pnt, pnt + pcd_neighber_normal_np[i] * 0.010, thickness=0.0004).attach_to(base)
            # gm.gen_arrow(anchor, anchor + anchor_normal * 0.015, rgba=(1, 0, 0, 1), thickness=0.001).attach_to(
            #     base)
        # base.run()

        pcd_neighber_np, pcd_neighber_normal_np = self._tune_sample(pcd_neighber_np, pcd_neighber_normal_np)
        anchor = pcd_neighber_np.mean(axis=0)
        # anchor = anchor
        anchor_normal = pcd_neighber_normal_np.mean(axis=0)
        if show:
            for i, pnt in enumerate(pcd_neighber_np):
                edge_cm = cm.gen_sphere(pos=pnt, radius=0.0005)
                edge_cm.set_rgba((0,1,1, 0.5))
                edge_cm.attach_to(base)
                # gm.gen_arrow(pnt, pnt+ pcd_neighber_normal_np[i] * 0.01, rgba=(0,0,1,1), thickness=0.0004).attach_to(base)
            gm.gen_arrow(anchor, anchor + anchor_normal * 0.015, rgba=(0,1,1,1), thickness=0.001).attach_to(
                base)
        return anchor, anchor_normal, pcd_neighber_np, pcd_neighber_normal_np

    def _get_neibour_detect(self, anchor, threshold_u, threshold_l, radius= 0.0025,show=True):
        # range_ball_cm = cm.gen_sphere(pos=anchor, radius=radius)
        # range_ball_cm.set_rgba((1, 0.3, 0.6, 0.2))
        # range_ball_cm.attach_to(base)
        [k, idx, _] = self.pcd_tree.search_radius_vector_3d(anchor, radius)
        pcd_neighber_normal_np = np.asarray([self.pcd.normals[i] for i in idx])
        pcd_neighber_np = vdda.o3dpcd_to_parray(self.pcd)[idx[1:]]
        anchor_normal = pcd_neighber_normal_np.mean(axis=0)

        pcd_neighber_np, pcd_neighber_normal_np, is_surface = self._tune_sample(pcd_neighber_np, pcd_neighber_normal_np, threshold_u, threshold_l)
        # anchor = pcd_neighber_np.mean(axis=0)
        # anchor = anchor
        anchor_normal = pcd_neighber_normal_np.mean(axis=0)
        if is_surface:
            self.surface_pnt.append(anchor)
            self.surface_normal.append(anchor_normal)
        else:
            self.edge_pnt.append(anchor)
            self.edge_normal.append(anchor_normal)

        if show:
            # for i, pnt in enumerate(pcd_neighber_np):
            #     edge_cm = cm.gen_sphere(pos=pnt, radius=0.0005)
            #     edge_cm.set_rgba((0,1,1, 0.5))
            #     edge_cm.attach_to(base)
            #     # gm.gen_arrow(pnt, pnt+ pcd_neighber_normal_np[i] * 0.01, rgba=(0,0,1,1), thickness=0.0004).attach_to(base)
            # gm.gen_arrow(anchor, anchor + anchor_normal * 0.015, rgba=(0,1,1,1), thickness=0.001).attach_to(
            #     base)
            if is_surface:
                edge_cm = cm.gen_sphere(pos=anchor, radius=0.00051)
                edge_cm.set_rgba((0,1,1, 1))
                # edge_cm.attach_to(base)
                # gm.gen_arrow(pnt, pnt+ pcd_neighber_normal_np[i] * 0.01, rgba=(0,0,1,1), thickness=0.0004).attach_to(base)
                # gm.gen_arrow(anchor, anchor + anchor_normal * 0.015, rgba=(0,1,1,1), thickness=0.001).attach_to(base)
            else:
                edge_cm = cm.gen_sphere(pos=anchor, radius=0.00051)
                # edge_cm.set_rgba((1, 1, 0, 1))
                b = []
                for neighber in pcd_neighber_np:
                    b.append(neighber - anchor)
                if np.dot(np.asarray(b).mean(axis=0), anchor_normal) >0:
                    # edge_cm.set_rgba((247 / 255, 122 / 255, 132 / 255, 1))
                    edge_cm.set_rgba((1, 0, 0, 1))
                else:
                    edge_cm.set_rgba((247 / 255, 122 / 255, 132 / 255, 1))
                    # edge_cm.set_rgba((7 / 255, 122 / 255, 132 / 255, 1))
                edge_cm.attach_to(base)
                # gm.gen_arrow(pnt, pnt+ pcd_neighber_normal_np[i] * 0.01, rgba=(0,0,1,1), thickness=0.0004).attach_to(base)
                # gm.gen_arrow(anchor, anchor + anchor_normal * 0.015, rgba=(1, 1, 0, 1), thickness=0.001).attach_to(base)

        return anchor, anchor_normal, pcd_neighber_np, pcd_neighber_normal_np

    def _is_surface(self, pcd_neighber_np, threshold_u, threshold_l):
        datamean = pcd_neighber_np.mean(axis=0)
        eigen_vals, eigen_vecs = self.pca(pcd_neighber_np - datamean)
        # print("--------------")
        # print("eigen_vals", eigen_vals)
        if eigen_vals[0] / eigen_vals[1] < threshold_l or eigen_vals[0] / eigen_vals[1] > threshold_u:
            # print(eigen_vals)
            # print("it is surface, ratio:", eigen_vals[0] / eigen_vals[1])
            is_surface = True
        elif eigen_vals[2] / eigen_vals[1] < threshold_l or eigen_vals[2] / eigen_vals[1] > threshold_u:
            # print(eigen_vals)
            # print("it is surface, ratio:", eigen_vals[2] / eigen_vals[1])
            is_surface = True
        else:
            # print("it is edge, ratio:", eigen_vals[2] / eigen_vals[1])
            is_surface = False
        # print("--------------")
        return is_surface

    def _is_vertex(self, pcd_neighber_np, threshold_u=20, threshold_l=0.05):
        datamean = pcd_neighber_np.mean(axis=0)
        eigen_vals, eigen_vecs = self.pca(pcd_neighber_np - datamean)
        # if np.allclose(eigen_vals_normalized, reference, atol=1):
        #     is_vertex = False
        # else:
        #     is_vertex = True
        # lower = 0.051
        # higher = 20
        # lower = 0.1
        # higher = 10
        # lower = 0.01
        # higher = 20
        if threshold_l < eigen_vals[1] / eigen_vals[0] <threshold_u and  threshold_l < eigen_vals[2] / eigen_vals[0] < threshold_u and  threshold_l < eigen_vals[2] / eigen_vals[1] < threshold_u:
            is_vertex = True
            print("----is vertex")
            print(eigen_vals)
        else:
            # print("----Not vertex")
            # print(eigen_vals)
            is_vertex = False
        return is_vertex

    def _tune_sample(self, pcd_neighber_np, pcd_neighber_normal_np, threshold_u = 50, threshold_l = 0.02, searchmaxPCA = False):
        datamean = pcd_neighber_np.mean(axis=0)
        eigen_vals, eigen_vecs = self.pca(pcd_neighber_np - datamean)
        # print("ratio", eigen_vals[0] / eigen_vals[1])
        is_surface = self._is_surface(pcd_neighber_np, threshold_u, threshold_l)
        if is_surface == False:
            # print(eigen_vals)
            # print("start edge refine")
            if searchmaxPCA:
                i = 0
                ratio = eigen_vals[0] / eigen_vals[1]
                while i < 20:
                    i += 1
                    reselect = pcd_neighber_np[np.random.choice(range(len(pcd_neighber_np)))]
                    [k, idx, _] = self.pcd_tree.search_radius_vector_3d(reselect, 0.005)
                    tem_pcd_neighber_normal_np = np.asarray([self.pcd.normals[i] for i in idx])
                    tem_pcd_neighber_np = vdda.o3dpcd_to_parray(self.pcd)[idx[1:]]
                    datamean = tem_pcd_neighber_np.mean(axis=0)
                    eigen_vals, eigen_vecs = self.pca(tem_pcd_neighber_np - datamean)
                    if eigen_vals[0] / eigen_vals[1] > ratio:
                        pcd_neighber_np = tem_pcd_neighber_np
                        pcd_neighber_normal_np = tem_pcd_neighber_normal_np

            # if eigen_vals[0] / eigen_vals[1] < 0.01:
            #     # print(eigen_vals)
            #     # print("it is surface, ratio:", eigen_vals[0] / eigen_vals[1])
            #     is_surface = True
            # elif eigen_vals[2] / eigen_vals[1] < 0.01:
            #     # print(eigen_vals)
            #     # print("it is surface, ratio:", eigen_vals[2] / eigen_vals[1])
            #     is_surface = True
            # print(eigen_vals)
            # print("update ratio = ", eigen_vals[0] / eigen_vals[1])
            pass
        else:
            pass
            # print("it is surface, no fine tune")
        return pcd_neighber_np, pcd_neighber_normal_np, is_surface

    def plan_poses(self, width_detect=0.005, length_detect=0.09, show_neighber=True):
        for anchor in self.pcd_sample_np:
            anchor, anchor_normal, pcd_neighber_np, pcd_neighber_normal_np = self._get_neibour(anchor)
            # anchor_normal = pcd_neighber_normal_np.mean(axis=0)
            if show_neighber:
                # gm.gen_arrow(anchor, anchor + anchor_normal * 0.02, thickness=0.001).attach_to(base)
                for i, pnt in enumerate(pcd_neighber_np):
                    edge_cm = cm.gen_sphere(pos=pnt, radius=0.0005)
                    edge_cm.set_rgba((0, 1, 0, 0.5))
                    edge_cm.attach_to(base)
                    # gm.gen_arrow(pnt, pnt + pcd_neighber_normal_np[i] * 0.010, thickness=0.0004).attach_to(base)
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
                pcd_opposite_neighber_normal_np = np.asarray([self.pcd.normals[i] for i in idx])
                pcd_opposite_neighber_np = vdda.o3dpcd_to_parray(self.pcd)[idx[1:]]

                for i, pnt in enumerate(pcd_opposite_neighber_np):
                    edge_cm = cm.gen_sphere(pos=pnt, radius=0.0005)
                    edge_cm.set_rgba((0, 0, 1, 0.5))
                    # edge_cm.attach_to(base)
                    # gm.gen_arrow(pnt, pnt + pcd_opposite_neighber_normal_np[i] * 0.010, thickness=0.0004).attach_to(base)

                anchor_opposite_normal = pcd_opposite_neighber_normal_np.mean(axis=0)
                # gm.gen_arrow(anchor_opposite, anchor_opposite + anchor_opposite_normal * 0.02, thickness=0.001).attach_to(
                #     base)
                if self._is_surface(pcd_neighber_np):
                    angle_threshold = 40
                else:
                    angle_threshold = 40
                print("here",rm.angle_between_vectors(anchor_normal, -anchor_opposite_normal) * 180/np.pi)
                if (rm.angle_between_vectors(anchor_normal, -anchor_opposite_normal) * 180/np.pi) <= angle_threshold:
                    print("check")
                    gripper = rtq85.Robotiq85()
                    gripper_y = rm.unit_vector(anchor_normal - anchor_opposite)
                    # rm.angle_between_vectors(anchor_normal,np.array([1,0,0]))
                    # t_ar = rm.rotmat_between_vectors( np.array([1,0,0]), anchor_normal)
                    t_ar = rm.rotmat_between_vectors(np.array([0, 1, 0]), gripper_y)
                    jaw_width = np.linalg.norm(anchor - anchor_opposite) + 0.01
                    if jaw_width<=0.085:
                        gripper.grip_at_with_jcpose((anchor + anchor_opposite) / 2, gl_jaw_center_rotmat=t_ar, jaw_width=jaw_width)
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
        # Data matrix X, assumes 0-centered
        n, m = X.shape
        assert np.allclose(X.mean(axis=0), np.zeros(m))  # 确保X已经中心化，每个维度的均值为0
        eigen_vals, eigen_vecs = np.linalg.eig(np.dot(X.T, X) / (n - 1))
        return eigen_vals, eigen_vecs

    def sample_points_around_center(self, center, radius=1.0, num_samples=2):
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

    def extend_point_cloud(self, original_cloud, radius, num_samples):
        extended_cloud = original_cloud.copy()
        for point in original_cloud:
            new_points = self.sample_points_around_center(point, radius, num_samples)
            extended_cloud = np.vstack((extended_cloud, new_points))
        return np.array(extended_cloud)

if __name__ == '__main__':
    base = wd.World(cam_pos=[0.2001557, 0.0637317, 0.1088133], w=960,
                    h=540, lookat_pos=[0, 0, 0])
    # gm.gen_frame().attach_to(base)
    this_dir, this_filename = os.path.split(__file__)
    # objpath = "kit_model_stl/Amicelli_800_tex.stl"
    # objpath = "kit_model_stl/CatSitting_800_tex.stl"
    # objpath = "kit_model_stl/CoffeeBox_800_tex.stl"
    # objpath = "test_obj/tetrahedron.stl"
    # objpath = "kit_model_stl/RedCup_800_tex.stl"
    # objpath = "test_obj/ratchet.stl"
    # objpath = "kit_model_stl/InstantSoup_800_tex.stl"
    # objname = "cupramen"
    # objname = "tetrahedron"
    objname = "Amicelli_800_tex"
    # objpath = f"test_obj/{objname}.stl"
    # objname = "CatSitting_800_tex"
    # objname = "CatLying_800_tex"
    # objname = "CoffeeBox_800_tex"
    objpath = f"kit_model_stl/{objname}.stl"
    graspplanner = GraspPlanner(objpath, objname)

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