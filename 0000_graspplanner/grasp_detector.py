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
import robot_sim.end_effectors.gripper.dh60 as dh60
import time


class ContactDetector():
    def __init__(self, objpath, objname, show_sample_contact=True):
        self.objname = objname
        self.objpath = objpath

        self.load_feature()

        self.pcd_np = vdda.o3dpcd_to_parray(self.pcd)
        # self.pcd_normal = np.asarray(self.pcd.normals)

        self.show_obj_mesh(rgba=[0.5, 0.5, 0.5, 0.1])
        self.show_obj_pcd()
        # self.show_feature("vertex")
        self.surface_sampled_id, self.surface_sampled = self.sample_contact(self.surface_pnt, "uniform")
        self.edge_sampled_id, self.edge_sampled = self.sample_contact(self.edge_pnt, "voxel")
        self.corner_sampled_id, self.corner_sampled = self.sample_contact(self.vertex_pnt_clustered, "voxel")

        self.surface_normal_sampled = self.surface_normal[self.surface_sampled_id]
        self.edge_normal_sampled = self.edge_normal[self.edge_sampled_id]
        self.corner_normal_sampled = self.vertex_normal_clustered[self.corner_sampled_id]

        self.detect_scale = self.get_pcd_max_scale()
        # self.input_data_debug()
        # base.run()

        for pnt in self.surface_sampled:
            gm.gen_sphere(pnt, radius=0.003, rgba=[46 / 255, 117 / 255, 182 / 255, 1]).attach_to(base)
        for pnt in self.edge_sampled:
            gm.gen_sphere(pnt, radius=0.003, rgba=[191 / 255, 144 / 255, 0, 1]).attach_to(base)
        for pnt in self.corner_sampled:
            gm.gen_sphere(pnt, radius=0.003, rgba=[1, 0, 0, 1]).attach_to(base)

        # edge_detector = self.get_edge_detector()
        # surface_detector = self.get_surface_detector()
        # corner_detector = self.get_corner_detector()
        # e_anti_pnts, e_anti_pnts_id = self.find_antipodal(edge_detector, self.edge_sampled, self.edge_sampled_id)
        # e_anti_pnts, e_anti_pnts_id = self.find_antipodal(corner_detector, self.surface_sampled, self.surface_sampled_id)
        # e_anti_pnts, e_anti_pnts_id = self.find_antipodal(surface_detector, self.surface_sampled, self.surface_sampled_id)

        pair_pnts = []
        pair_id = []
        pair_normal = []

        self.pairing_contacts()

        base.run()
    def pairing_contacts(self):
        temp_pair_pnts = []
        temp_pair_pnts_id = []
        for i in self.vertex_id_clustered[self.corner_sampled_id]:
            corner_detector = self.get_corner_detector(i, debug=False)
            anchor_normal = self.pcd_normal[i]
            anchor = self.pcd_np[i]
            # gm.gen_sphere(pnt, radius=0.006, rgba=[1, 0, 0, 1]).attach_to(base)
            # gm.gen_arrow(anchor, anchor + 0.1 * rm.unit_vector(anchor_normal)).attach_to(base)
            cc_pnts, cc_pnts_id, cc_normal = self.find_antipodal(corner_detector, anchor, i, anchor_normal,self.corner_sampled,
                                                              self.corner_sampled_id, self.corner_normal_sampled)
            ce_pnts, ce_pnts_id, ce_normal = self.find_antipodal(corner_detector, anchor, i, anchor_normal, self.edge_sampled,
                                                              self.edge_sampled_id, self.edge_normal_sampled, threshold = 0.9)
            cf_pnts, cf_pnts_id, cf_normal = self.find_antipodal(corner_detector, anchor, i, anchor_normal, self.surface_sampled,
                                                              self.surface_sampled_id, self.surface_normal_sampled, threshold = 0.9)
            c_temp_pair_pnts = [item for item in cc_pnts+ce_pnts+cf_pnts]
            c_temp_pair_pnts_id = [item for item in cc_pnts_id + ce_pnts_id + cf_pnts_id]
            temp_pair_pnts.append(c_temp_pair_pnts)
            temp_pair_pnts_id.append(c_temp_pair_pnts_id)
            # for item in c_temp_pair_pnts:
            #     # gm.gen_sphere(item[2], radius=0.01)
            #     gm.gen_stick(item[0], item[1], rgba=[0,0,0,1], thickness=0.003).attach_to(base)
            # break

        for i in self.edge_id[self.edge_sampled_id][8:]:
            edge_detector = self.get_edge_detector(i, debug=False)
            anchor_normal = self.pcd_normal[i]
            anchor = self.pcd_np[i]
            # gm.gen_sphere(anchor, radius=0.006, rgba=[1, 0, 0, 1]).attach_to(base)
            # gm.gen_arrow(anchor, anchor + 0.1 * rm.unit_vector(anchor_normal)).attach_to(base)
            ee_pnts, ee_pnts_id, ee_normal = self.find_antipodal(edge_detector, anchor, i, anchor_normal,self.edge_sampled,
                                                              self.edge_sampled_id, self.edge_normal_sampled, threshold = 0.95, debug=False)
            ef_pnts, ef_pnts_id, ef_normal = self.find_antipodal(edge_detector, anchor, i, anchor_normal, self.surface_sampled,
                                                              self.surface_sampled_id, self.surface_normal_sampled, threshold = 0.95, debug=False)
            e_temp_pair_pnts = [item for item in ee_pnts+ef_pnts]
            e_temp_pair_pnts_id = [item for item in ee_pnts_id + ef_pnts_id]
            temp_pair_pnts.append(e_temp_pair_pnts)
            temp_pair_pnts_id.append(e_temp_pair_pnts_id)
            # for item in e_temp_pair_pnts:
            #     # gm.gen_sphere(item[2], radius=0.01)
            #     gm.gen_stick(item[0], item[1], rgba=[0,0,0,1], thickness=0.003).attach_to(base)
            # break

        for i in self.surface_id[self.surface_sampled_id][4:]:
            surface_detector = self.get_surface_detector(i, diameter = 0.01, debug=False)
            anchor_normal = self.pcd_normal[i]
            anchor = self.pcd_np[i]
            # gm.gen_sphere(anchor, radius=0.006, rgba=[0, 0, 1, 1]).attach_to(base)
            # gm.gen_arrow(anchor, anchor + 0.03 * rm.unit_vector(anchor_normal), thickness=0.001).attach_to(base)
            ff_pnts, ff_pnts_id, ff_normal = self.find_antipodal(surface_detector, anchor, i, anchor_normal, self.surface_sampled,
                                                              self.surface_sampled_id, self.surface_normal_sampled, threshold = 0.95, debug=False)
            f_temp_pair_pnts = ff_pnts
            f_temp_pair_pnts_id = ff_pnts_id
            temp_pair_pnts.append(f_temp_pair_pnts)
            temp_pair_pnts_id.append(f_temp_pair_pnts_id)
            # print(temp_pair_pnts)
            # for item in f_temp_pair_pnts:
            #     # gm.gen_sphere(item[2], radius=0.01)
            #     gm.gen_stick(item[0], item[1], rgba=[0,0,0,1], thickness=0.003).attach_to(base)
            # break

        pair_pnts = list(itertools.chain(*temp_pair_pnts))
        pair_pnts_id = list(itertools.chain(*temp_pair_pnts_id))
        for item in pair_pnts:
            # gm.gen_sphere(item[2], radius=0.01)
            gm.gen_stick(item[0], item[1], rgba=[0, 0, 0, 1], thickness=0.0005).attach_to(base)

        self.pair_pnts = pair_pnts
        self.pair_pnts_id = pair_pnts_id

    def input_data_debug(self):
        for i, pnt in enumerate(self.corner_sampled):
            gm.gen_sphere(pnt, radius=0.003, rgba=[1, 0, 0, 1]).attach_to(base)
            gm.gen_sphere(self.pcd_np[self.vertex_id_clustered[self.corner_sampled_id[i]]], radius=0.01, rgba=[1, 0, 0, 0.5]).attach_to(base)
            # gm.gen_sphere(self.vertex_pnt_clustered[self.corner_sampled_id[i]], radius=0.01, rgba=[1, 0, 0, 0.5]).attach_to(base)
            gm.gen_arrow(pnt, self.corner_normal_sampled[i] * 0.1 + pnt, rgba=[1, 0, 0, 0.5]).attach_to(base)
            break
        for i, pnt in enumerate(self.edge_sampled):
            gm.gen_sphere(pnt, radius=0.003, rgba=[0, 1, 0, 1]).attach_to(base)
            gm.gen_sphere(self.pcd_np[self.edge_id[self.edge_sampled_id[i]]], radius=0.01, rgba=[0, 1, 0, 0.5]).attach_to(base)
            # gm.gen_sphere(self.edge_pnt[self.edge_sampled_id[i]], radius=0.01,
            #               rgba=[0, 1, 0, 0.5]).attach_to(base)
            gm.gen_arrow(pnt, self.edge_normal_sampled[i] * 0.1 + pnt, rgba=[0, 1, 0, 0.5]).attach_to(base)
            break
        for i, pnt in enumerate(self.surface_sampled):
            gm.gen_sphere(pnt, radius=0.003, rgba=[0, 0, 1, 1]).attach_to(base)
            gm.gen_sphere(self.pcd_np[self.surface_id[self.surface_sampled_id[i]]], radius=0.01, rgba=[0, 0, 1, 0.5]).attach_to(base)
            # gm.gen_sphere(self.surface_pnt[self.surface_sampled_id[i]], radius=0.01,
            #               rgba=[0, 0, 1, 0.5]).attach_to(base)
            gm.gen_arrow(pnt, self.surface_normal_sampled[i] * 0.1 + pnt, rgba=[0, 0, 1, 0.5]).attach_to(base)
            break

    def find_antipodal(self, detector, anchor, anchor_id, anchor_normal, target_pnts, target_pnts_id, target_normal, threshold = 0.0, debug = False):
        checker = trimesh.base.ray.ray_pyembree.RayMeshIntersector(detector.objtrm)
        inside_points = []
        inside_points_id = []
        inside_normal = []
        # outside_points = []
        anti_points = []
        anti_points_id = []
        anti_normal = []
        inner_tf_list = checker.contains_points(target_pnts)
        for i, item in enumerate(inner_tf_list):
            if item:
                inside_points.append(target_pnts[i])
                inside_points_id.append(target_pnts_id[i])
                inside_normal.append(target_normal[i])
                if rm.unit_vector(target_normal[i]).dot(rm.unit_vector(anchor_normal)) < -threshold:
                    # print("ttt", rm.unit_vector(target_normal[i]).dot(rm.unit_vector(anchor_normal)))
                    anti_points.append([anchor, target_pnts[i]])
                    anti_points_id.append([anchor_id, target_pnts_id[i]])
                    anti_normal.append([anchor_normal, target_normal[i]])
        # print(inside_points)
        if debug:
            for i, pnt in enumerate(inside_points):
                gm.gen_sphere(pnt, radius=0.006, rgba=[1, 0, 0, 1]).attach_to(base)
                gm.gen_arrow(pnt, pnt+0.1*rm.unit_vector(inside_normal[i])).attach_to(base)
        return anti_points, anti_points_id, anti_normal

    def show_obj_pcd(self):
        gm.gen_pointcloud(self.pcd_np).attach_to(base)

    def sample_contact(self, points, method="voxel"):
        # points->pcd "just change name"
        if method == "voxel":
            min_bound, max_bound = self.get_pcd_bound()
            sampled_contact, contact_id, _ = vdda.nparray_to_o3dpcd(
                points).voxel_down_sample_and_trace(voxel_size=0.03,
                                                    min_bound=min_bound,
                                                    max_bound=max_bound,
                                                    approximate_class=False)
            pcd = vdda.nparray_to_o3dpcd(points)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            sample = vdda.o3dpcd_to_parray(sampled_contact)
            sampled_contact_id = []
            sampled_contact_pnt = []
            for pnt in sample:
                [k, idx, _] = pcd_tree.search_knn_vector_3d(pnt, 1)
                sampled_contact_pnt.append(points[idx[0]])
                sampled_contact_id.append(idx[0])
            return sampled_contact_id, sampled_contact_pnt

        elif method == "uniform":
            sampled_contact_id = []
            sampled_contact_pnt = []
            pcd = vdda.nparray_to_o3dpcd(points)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            sample = vdda.o3dpcd_to_parray(pcd.uniform_down_sample(every_k_points=300))
            for pnt in sample:
                [k, idx, _] = pcd_tree.search_knn_vector_3d(pnt, 1)
                sampled_contact_pnt.append(points[idx[0]])
                sampled_contact_id.append(idx[0])
            return sampled_contact_id, sampled_contact_pnt

    def get_surface_detector(self, surface_id=None, diameter=0.01, debug = False):
        if surface_id is None:
            surface_id = self.surface_sampled_id[np.random.randint(0, len(self.surface_sampled_id) - 1)]
        surface_pnt = self.pcd_np[surface_id]
        surface_normal = -self.pcd_normal[surface_id]
        pcd_scale = self.detect_scale
        detector = gm.gen_cylinder(radius=diameter, height= pcd_scale,
                                   homomat=rm.homomat_from_posrot(surface_pnt, rm.rotmat_from_normal(surface_normal)),
                                   rgba=(46 / 255, 117 / 255, 182 / 255, 0.3))
        detector.objtrm.export("tst.stl")
        d = gm.GeometricModel("tst.stl")
        if debug:
            detector.attach_to(base)
        return d

    def get_edge_detector(self, edge_id=None, radius=0.01, threshold=0.008, maxIteration=100, debug=False):
        if not edge_id:
            edge_id = self.edge_sampled_id[np.random.randint(0, len(self.edge_sampled_id) - 1)]
        edge_pnt = self.pcd_np[edge_id]
        edge_normal = -self.pcd_normal[edge_id]
        edge_normal_scale = np.linalg.norm(edge_normal)
        edge_normal_uniform = (1 / edge_normal_scale) * edge_normal
        tangential_direction = self.fit_line_from_point(edge_pnt, radius, threshold, maxIteration)
        detector = gm.gen_section(spos=edge_pnt, epos=edge_pnt + self.detect_scale * edge_normal_uniform,
                                  rgba=[191 / 255, 144 / 255, 0, 0.3],
                                  height_vec=tangential_direction, height=0.02, angle=90, section=8)
        if debug:
            detector.attach_to(base)
        return detector

    def get_corner_detector(self, corner_id=None, debug=False):
        if corner_id is None:
            corner_id = self.corner_sampled_id[np.random.randint(0, len(self.corner_sampled_id) - 1)]
        corner_pnt = self.pcd_np[corner_id]
        # corner_pnt = self.corner_sampled[corner_id]
        corner_normal = -self.pcd_normal[corner_id]
        # corner_normal = -self.corner_normal_sampled[corner_id]
        edge_normal_scale = np.linalg.norm(corner_normal)
        edge_normal_uniform = (1 / edge_normal_scale) * corner_normal
        detector = gm.gen_cone(spos=corner_pnt +  self.detect_scale * edge_normal_uniform, epos=corner_pnt, rgba=[1, 0, 0, 0.3], radius=self.detect_scale, sections=24)

        detector.objtrm.export("tst.stl")
        d = gm.GeometricModel("tst.stl")
        if debug:
            detector.attach_to(base)
        return d

        # surface, normal, position, gen_cylinder, diameter()variable
        # edge, normal, position, tangential direction, thickness, gen_section()
        # corner, normal, position, cone

    def load_feature(self):
        self.edge_pnt = self.importinfo(f"debug_data/{self.objname}/edge_pnt.pickle")
        self.edge_normal = self.importinfo(f"debug_data/{self.objname}/edge_normal.pickle")
        self.edge_id = self.importinfo(f"debug_data/{self.objname}/edge_id.pickle")

        self.surface_pnt = self.importinfo(f"debug_data/{self.objname}/surface_pnt.pickle")
        self.surface_normal = self.importinfo(f"debug_data/{self.objname}/surface_normal.pickle")
        self.surface_id = self.importinfo(f"debug_data/{self.objname}/surface_id.pickle")

        self.vertex_pnt_clustered = self.importinfo(f"debug_data/{self.objname}/vertex_pnt_clustered.pickle")
        self.vertex_normal_clustered = self.importinfo(f"debug_data/{self.objname}/vertex_normal_clustered.pickle")
        self.vertex_id_clustered = self.importinfo(f"debug_data/{self.objname}/vertex_id_clustered.pickle")

        self.pcd = o3d.io.read_point_cloud(f"debug_data/{self.objname}.pcd")
        self.pcd_id = self.importinfo(f"debug_data/{self.objname}/pcd_id.pickle")
        self.pcd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        self.pcd_normal = self.importinfo(f"debug_data/{self.objname}/pcd_normal.pickle")
        # print(self.pcd_id)
        # print(len(self.pcd_id))

        # self.edge_detect_parameter = self.importinfo(f"debug_data/{self.objname}/edge_detect_parameter.pickle")
        # print('Edge detect parameter: ', self.edge_detect_parameter)
        # self.vertex_clustered_parameter = self.importinfo(f"debug_data/{self.objname}/vertex_clustered_parameter.pickle")
        # print('Vertex clustered parameter: ', self.vertex_clustered_parameter)

    def show_obj_mesh(self, rgba=[0.5, 0.5, 0.5, 1]):
        self.collision_model = cm.CollisionModel(objpath)
        self.collision_model.set_rgba(rgba)
        # self.collision_model.set_scale((2, 2, 2))
        self.collision_model.attach_to(base)

    def show_feature(self, feature):

        if feature == "vertex":
            radius = 0.001
            rgba = [1, 0, 0, 1]
            for i in range(len(self.vertex_pnt_clustered)):
                gm.gen_sphere(self.vertex_pnt_clustered[i], subdivisions=2, radius=radius,
                              rgba=rgba).attach_to(base)
        elif feature == "edge":
            radius = 0.001
            rgba = [191 / 255, 144 / 255, 0, 1]
            for i in range(len(self.edge_pnt)):
                gm.gen_sphere(self.edge_pnt[i], subdivisions=2, radius=radius,
                              rgba=rgba).attach_to(base)
        elif feature == "surface":
            radius = 0.001
            rgba = [46 / 255, 117 / 255, 182 / 255, 1]
            for i in range(len(self.surface_pnt)):
                gm.gen_sphere(self.surface_pnt[i], subdivisions=2, radius=radius,
                              rgba=rgba).attach_to(base)

    def get_pcd_bound(self):
        min_bound = np.min(self.pcd_np, axis=0)
        max_bound = np.max(self.pcd_np, axis=0)
        return min_bound, max_bound

    def get_pcd_max_scale(self):
        x_scale = np.min(self.pcd_np[:, 0]) - np.max(self.pcd_np[:, 0])
        y_scale = np.min(self.pcd_np[:, 1]) - np.max(self.pcd_np[:, 1])
        z_scale = np.min(self.pcd_np[:, 2]) - np.max(self.pcd_np[:, 2])
        max_scale = np.max(np.abs([x_scale, y_scale, z_scale]))
        return max_scale

    def fit_line_from_point(self, point, radius, threshold, maxIteration):
        kdtree = o3d.geometry.KDTreeFlann(self.pcd)
        [k, idx, _] = kdtree.search_radius_vector_3d(point, radius)
        neighbor_points = self.pcd_np[idx]
        line_model = pyrsc.Line()
        A, B, inliers = line_model.fit(neighbor_points, threshold, maxIteration)
        inliers_points = neighbor_points[inliers]
        inliers_mean = np.mean(inliers_points, axis=0)
        center_points = inliers_points - inliers_mean
        _, _, Vt = np.linalg.svd(center_points)
        inliers_direction = Vt[0]
        # t = [-2, 2]
        # line_points = np.zeros((2, 3))
        # line_points[0] = inliers_mean + inliers_direction * t[0]
        # line_points[1] = inliers_mean + inliers_direction * t[1]

        # gm.gen_sphere(point, radius, rgba=[0, 1, 0, 0.5]).attach_to(base)

        # gm.gen_arrow(line_points[0], line_points[1], thickness=0.001, rgba=[1, 0, 0, 1]).attach_to(base)
        return inliers_direction

    def outputinfo(self, name, data):
        with open(name, "wb") as file:
            pickle.dump(data, file)

    def importinfo(self, name):
        with open(name, "rb") as file:
            f = pickle.load(file)
        return f


if __name__ == '__main__':
    base = wd.World(cam_pos=[2.001557 / 2, 0.637317 / 2, 1.088133 / 2], w=960,
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
    objname = "arrow"
    # objname = 'hexagonal_prisms'
    objpath = f"test_obj/{objname}.stl"
    # objname = "CatSitting_800_tex"
    # objname = "CatLying_800_tex"
    # objname = "CoffeeBox_800_tex"
    # objpath = f"stl/{objname}.stl"
    graspplanner = ContactDetector(objpath, objname)

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
