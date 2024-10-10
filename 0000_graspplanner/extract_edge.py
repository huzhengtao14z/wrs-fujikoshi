import utils.pcd_utils as pcdu
import numpy as np
import pickle
import open3d as o3d
import basis.o3dhelper as o3dh
import visualization.panda.world as wd
import modeling.collision_model as cm
import cv2
import basis.data_adapter as da
import modeling.geometric_model as gm
if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0])
    objpath = "kit_model_stl/Amicelli_800_tex.stl"
    Mesh = o3d.io.read_triangle_mesh(objpath)
    collision_model = cm.CollisionModel(objpath)
    collision_model.set_rgba((0.5, 0.5, 0.5, 1))
    collision_model.attach_to(base)
    Mesh.compute_vertex_normals()
    o3dpcd = Mesh.sample_points_poisson_disk(number_of_points=5000)
    # gm.gen_pointcloud(np.asarray(o3dpcd.points)).attach_to(base)
    # img, dep, pcd = pickle.load(open('../img/charger.pkl', 'rb'))
    # print(img)
    # cv2.imshow("hi", img)
    # base.run()
    # pcd = np.asarray(pcdu.remove_pcd_zeros(pcd))
    # o3dpcd = o3dh.nparray2o3dpcd(pcd)
    plane_model, inliers = o3dpcd.segment_plane(distance_threshold=0.0175,
                                                ransac_n=30,
                                                num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inliers_o3dpcd = o3dpcd.select_by_index(inliers, invert=True)
    outliers_o3dpcd = o3dpcd.select_by_index(inliers, invert=False)
    labels = np.array(inliers_o3dpcd.cluster_dbscan(eps=0.01, min_points=10, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    # plane_1_o3dpcd = o3dpcd.select_by_index(inliers)

    o3dpcd = o3dpcd.select_by_index(inliers, invert=True)

    pcd = np.asarray(o3dpcd.points)

    p_list, d_list, nrmls = pcdu.detect_edge(pcd, voxel_size=.005, toggledebug=True)

    for i, p in enumerate(p_list):
        if d_list[i]==0:
            gm.gen_sphere(p , 0.002).attach_to(base)
    pcdu.show_pcd(pcd)


    base.run()
