import copy
import math
# from keras.models import Sequential, Model, load_model
import visualization.panda.world as wd
import modeling.collision_model as cm
import humath as hm
# import hufunc as hf
import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yg
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as hnde
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
import numpy as np
import basis.robot_math as rm
import modeling.geometric_model as gm
import robot_sim.robots.ur3_dual.ur3_dual as ur3d
import robot_sim.robots.ur3e_dual.ur3e_dual as ur3ed
import robot_sim.robots.sda5f.sda5f as sda5
import motion.probabilistic.rrt_connect as rrtc
import manipulation.pick_place_planner as ppp
import os
import pickle
import basis.data_adapter as da
import slope
import Sptpolygoninfo as sinfo
import basis.trimesh as trimeshWan
import trimesh as trimesh
from trimesh.sample import sample_surface
from panda3d.core import NodePath
import trimeshwraper as tw
import grasping.planning.antipodal as gpa
import robot_sim.end_effectors.gripper.robotiq85.robotiq85 as rtq85
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rtqhe
import open3d as o3d
# import open3d.geometry as o3dg
import vision.depth_camera.pcd_data_adapter as vdda



if __name__ == '__main__':
    base = wd.World(cam_pos=[0.2001557, 0.0637317, 0.1088133], w=960,
                    h=540, lookat_pos=[0, 0, 0])
    # gm.gen_frame().attach_to(base)
    this_dir, this_filename = os.path.split(__file__)
    Mesh = o3d.io.read_triangle_mesh("kit_model_stl/Amicelli_800_tex.stl")
    collision_model = cm.CollisionModel("kit_model_stl/Amicelli_800_tex.stl")
    collision_model.set_rgba((0,1,1,1))
    # collision_model.attach_to(base)
    Mesh.compute_vertex_normals()
    # Mesh.compute_triangle_normals()


    pcd1 = Mesh.sample_points_poisson_disk(number_of_points=5000)
    # pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    # o3d.visualization.draw_geometries([pcd1])
    pcd1, ind = pcd1.remove_radius_outlier(nb_points=50, radius=0.05)
    pcd1_np = vdda.o3dpcd_to_parray(pcd1)
    pcd1_normal_np = np.asarray(pcd1.normals)[:, :]

    gm.gen_pointcloud(pcd1_np).attach_to(base)

    pcd_sample = pcd1.random_down_sample(0.005)
    # pcd_sample = pcd1.uniform_down_sample(2)

    pcd_sample_np = vdda.o3dpcd_to_parray(pcd_sample)

    # for i, pnt in enumerate(pcd1_np):
    #     edge_cm = cm.gen_sphere(pos=pnt, radius=0.0005)
    #     edge_cm.set_rgba((0, 1, 0, 0.5))
    #     edge_cm.attach_to(base)
    #     gm.gen_arrow(pnt, pnt + pcd1_normal_np[i] * 0.010, thickness=0.0002).attach_to(base)
    # base.run()
    # print(pcd_sample_np.shape)
    for pnt in pcd_sample_np:
        edge_cm = cm.gen_sphere(pos=pnt, rgba=(1,0,0,0.3), radius=0.005)
        edge_cm.attach_to(base)
        break

    # base.run()

    # for anchor in pcd_sample_np:
    anchor = pcd_sample_np[0]
    range_ball_cm = cm.gen_sphere(pos=anchor, radius=0.005)
    range_ball_cm.set_rgba((1, 0.3, 0.6, 0.2))
    range_ball_cm.attach_to(base)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd1)
    [k, idx, _] = pcd_tree.search_radius_vector_3d(anchor, 0.005)
    pcd1_neighber_normal_np = np.asarray([pcd1.normals[i] for i in idx])
    pcd1_neighber_np = vdda.o3dpcd_to_parray(pcd1)[idx[1:]]

    for i, pnt in enumerate(pcd1_neighber_np):
        edge_cm = cm.gen_sphere(pos=pnt, radius=0.0005)
        edge_cm.set_rgba((0, 1, 0, 0.5))
        edge_cm.attach_to(base)
        # gm.gen_arrow(pnt, pnt+ pcd1_neighber_normal_np[i] * 0.010, thickness=0.0004).attach_to(base)

    anchor_normal = pcd1_neighber_normal_np.mean(axis = 0)
    gm.gen_arrow(anchor, anchor + anchor_normal*0.015, thickness=0.001).attach_to(base)

    # base.run()
    width_detect = 0.005
    length_detect = 0.09
    gm.gen_cone(anchor - anchor_normal * length_detect,anchor, radius=length_detect*np.sin(np.pi*20/180), sections=7, rgba=(0,1,0,0.3)).attach_to(base)
    detect_ray = gm.gen_stick(anchor - anchor_normal * length_detect,anchor- anchor_normal * 0.01*length_detect, thickness=width_detect*np.sin(np.pi*20/180)*2, sections=7, rgba=(0, 1, 0, 0.3))
    # detect_ray.attach_to(base)
    detect_ray_model = detect_ray.objtrm.export("detect_ray.stl")
    detect_mesh = o3d.io.read_triangle_mesh("detect_ray.stl")
    mesh_tree = o3d.geometry.KDTreeFlann(detect_mesh)


    inside_points = []
    outside_points = []



    def is_point_inside_mesh(point, mesh):
        ray_origin = point
        ray_direction = np.array([0, 0, 1])
        intersections = mesh.ray.intersects_id(ray_origins=[ray_origin], ray_directions=[ray_direction])

        # If the number of intersections is odd, the point is inside the mesh
        return len(intersections) % 2 == 1

    detect_mesh = trimesh.load_mesh("detect_ray.stl")
    checker = trimesh.base.ray.ray_pyembree.RayMeshIntersector(detect_mesh)
    for point in pcd1.points:
        if checker.contains_points(points=[point]):
            inside_points.append(point)
        else:
            outside_points.append(point)
    pcd_in = vdda.nparray_to_o3dpcd(np.asarray(inside_points))
    labels = np.array(pcd_in.cluster_dbscan(eps=0.003, min_points=1, print_progress=False))
    num_cluster = labels.max()+1
    print(labels)
    print(num_cluster)
    clustered_pcd_opposit=[[]*num_cluster]
    for index, label in enumerate(labels):
        clustered_pcd_opposit[label].append(inside_points[index])
    print(clustered_pcd_opposit)
    clustered_ave_pcd_opposit_np = np.asarray(clustered_pcd_opposit).mean(axis = 1)
    print(clustered_ave_pcd_opposit_np)

    for i, pnt in enumerate(inside_points):
        edge_cm = cm.gen_sphere(pos=pnt, radius=0.0005)
        edge_cm.set_rgba((0, 1, 0, 0.5))
        edge_cm.attach_to(base)



    anchor_opposite = clustered_ave_pcd_opposit_np[0]
    opposite_range_ball_cm = cm.gen_sphere(pos=anchor_opposite, radius=0.005)
    opposite_range_ball_cm.set_rgba((1, 0.3, 0.6, 0.2))
    opposite_range_ball_cm.attach_to(base)
    # pcd_tree = o3d.geometry.KDTreeFlann(pcd1)
    [k, idx, _] = pcd_tree.search_radius_vector_3d(anchor_opposite, 0.005)
    pcd1_opposite_neighber_normal_np = np.asarray([pcd1.normals[i] for i in idx])
    pcd1_opposite_neighber_np = vdda.o3dpcd_to_parray(pcd1)[idx[1:]]

    for i, pnt in enumerate(pcd1_opposite_neighber_np):
        edge_cm = cm.gen_sphere(pos=pnt, radius=0.0005)
        edge_cm.set_rgba((0, 0, 1, 0.5))
        edge_cm.attach_to(base)
        # gm.gen_arrow(pnt, pnt + pcd1_opposite_neighber_normal_np[i] * 0.010, rgba=(0, 0, 1, 1),thickness=0.0004).attach_to(base)

    anchor_opposite_normal = pcd1_opposite_neighber_normal_np.mean(axis=0)
    gm.gen_arrow(anchor_opposite, anchor_opposite + anchor_opposite_normal * 0.015, rgba=(0, 0, 1, 1),thickness=0.001).attach_to(base)

    datamean = pcd1_neighber_np.mean(axis=0)

    def pca(X):
        # Data matrix X, assumes 0-centered
        n, m = X.shape
        assert np.allclose(X.mean(axis=0), np.zeros(m))  # 确保X已经中心化，每个维度的均值为0
        # Compute covariance matrix
        C = np.dot(X.T, X) / (n - 1)
        # Eigen decomposition
        eigen_vals, eigen_vecs = np.linalg.eig(C)
        # print(eigen_vals)
        # Project X onto PC space
        # X_pca = np.dot(X, eigen_vecs)
        return eigen_vals, eigen_vecs

    eigen_vals, eigen_vecs = pca(pcd1_neighber_np - datamean)
    # is_surface = np.allclose(eigen_vals[1], eigen_vals[0], rtol=0, atol=1e-01, equal_nan=False)
    print("ratio", eigen_vals[0] / eigen_vals[1])
    if eigen_vals[0]/eigen_vals[1] < 0.01:
        print("ratio", eigen_vals[0]/eigen_vals[0])
        is_surface = True
    else:
        is_surface = False
    print("eigen_vals", eigen_vals)
    print("is_surface", is_surface)

    # pcd1_neighber_normal_np = np.asarray([pcd1.normals[i] for i in idx])
    # pcd1_neighber_np = vdda.o3dpcd_to_parray(pcd1)[idx[1:]]
    #
    # for i, pnt in enumerate(pcd1_neighber_np):
    #     edge_cm = cm.gen_sphere(pos=pnt, radius=0.0005)
    #     edge_cm.set_rgba((0, 1, 0, 0.5))
    #     edge_cm.attach_to(base)
    #     gm.gen_arrow(pnt, pnt + pcd1_neighber_normal_np[i] * 0.010, thickness=0.0004).attach_to(base)
    if is_surface == False:
        i = 0
        ratio = eigen_vals[0] / eigen_vals[1]
        print("start edge refine")
        while i < 10:
            i+=1
            reselect = pcd1_neighber_np[np.random.choice(range(len(pcd1_neighber_np)))]
            [k, idx, _] = pcd_tree.search_radius_vector_3d(reselect, 0.005)
            tem_pcd1_neighber_normal_np = np.asarray([pcd1.normals[i] for i in idx])
            tem_pcd1_neighber_np = vdda.o3dpcd_to_parray(pcd1)[idx[1:]]
            datamean = tem_pcd1_neighber_np.mean(axis=0)
            eigen_vals, eigen_vecs = pca(tem_pcd1_neighber_np - datamean)
            if eigen_vals[0] / eigen_vals[1] > ratio:
                pcd1_neighber_np = tem_pcd1_neighber_np
                pcd1_neighber_normal_np  = tem_pcd1_neighber_normal_np
        print("update ratio = ", eigen_vals[0] / eigen_vals[1])
    else:
        print("it is surface, no fine tune")
    # print(pca(pcd1_neighber_np - datamean))
    for i, pnt in enumerate(pcd1_neighber_np):
        edge_cm = cm.gen_sphere(pos=pnt, radius=0.0007)
        edge_cm.set_rgba((0, 1, 1, 0.5))
        # edge_cm.attach_to(base)
        # gm.gen_arrow(pnt, pnt+ pcd1_neighber_normal_np[i] * 0.01, rgba=(0,0,1,1), thickness=0.0004).attach_to(base)
    anchor_normal = pcd1_neighber_normal_np.mean(axis=0)
    new_anchor = pcd1_neighber_np.mean(axis=0)
    # gm.gen_arrow(new_anchor, new_anchor + anchor_normal * 0.015, thickness=0.001, rgba=(0,1,1,1)).attach_to(base)

    base.run()
    # gripper = rtqhe.RobotiqHE()
    gripper = rtq85.Robotiq85()
    gripper_y = rm.unit_vector(anchor_normal-anchor_opposite)
    # rm.angle_between_vectors(anchor_normal,np.array([1,0,0]))
    # t_ar = rm.rotmat_between_vectors( np.array([1,0,0]), anchor_normal)
    t_ar = rm.rotmat_between_vectors(np.array([0,1, 0]), gripper_y)
    jaw_width = np.linalg.norm(anchor - anchor_opposite)+0.01
    gripper.grip_at_with_jcpose((anchor + anchor_opposite) / 2, gl_jaw_center_rotmat=t_ar, jaw_width=jaw_width)
    for rotate_angle in np.arange(0, 360, 360):
        print("t")
        tmp_rotmat = rm.rotmat_from_axangle(gripper_y, np.pi*rotate_angle/180)
        rotmat = np.dot(tmp_rotmat,t_ar)
        gripper.grip_at_with_jcpose((anchor + anchor_opposite) / 2, gl_jaw_center_rotmat=rotmat, jaw_width=jaw_width)
        # if not gripper.is_mesh_collided([collision_model]):
        #     gripper.gen_meshmodel(toggle_tcpcs=False).attach_to(base)

    gm.gen_frame().attach_to(base)
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