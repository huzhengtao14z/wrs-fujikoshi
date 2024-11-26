import copy
import math
# from keras.models import Sequential, Model, load_model
import visualization.panda.world as wd
import modeling.collision_model as cm

from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
import modeling.geometric_model as gm
import os
import open3d as o3d
import vision.depth_camera.pcd_data_adapter as vdda


if __name__ == '__main__':
    base = wd.World(cam_pos=[2.01557, 0.637317, 1.88133], w=960,
                    h=540, lookat_pos=[0, 0, 0])
    # gm.gen_frame().attach_to(base)
    this_dir, this_filename = os.path.split(__file__)
    obj = cm.CollisionModel('temp.stl')
    obj.set_rgba([1, 1, 0, 0.5])
    obj.attach_to(base)
    # obj.objtrm.export('kit_model_stl/Amicelli_800_tex.stl')
    # base.run()
    # obj.set_scale([1000, 1000,1000])
    # obj2 = cm.CollisionModel('edgenool.STL')
    # obj2.set_rgba([0, 1, 1, 1])
    # obj2.attach_to(base)
    # base.run()
    Mesh = o3d.io.read_triangle_mesh('temp.stl')
    # Mesh.compute_vertex_normals()
    pcd1 = Mesh.sample_points_poisson_disk(number_of_points=50)
    # pcd1, ind = pcd1.remove_radius_outlier(nb_points=15, radius=5)
    pcd1_np = vdda.o3dpcd_to_parray(pcd1)
    center1 = pcd1_np.mean(axis=0)

    # Mesh2 = o3d.io.read_triangle_mesh('cutplane-intersectionEdgenool-2.stl')
    # # Mesh.compute_vertex_normals()
    # pcd2 = Mesh2.sample_points_poisson_disk(number_of_points=50)
    # # pcd1, ind = pcd1.remove_radius_outlier(nb_points=15, radius=5)
    # pcd2_np = vdda.o3dpcd_to_parray(pcd2)
    # center2 = pcd2_np.mean(axis=0)
    pcd = o3d.io.read_point_cloud('temp.stl')
    # pcd = o3d.io.read_point_cloud('pairtial/pairtial_pc/Amicelli_800_tex0.ply')
    o3d.visualization.draw_geometries([pcd1])
    # pcd_list = vdda.o3dpcd_to_parray(pcd1_np)
    # kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(pcd1_np)
    gm.gen_sphere(center1, radius=1.5).attach_to(base)
    # gm.gen_sphere(center2, radius=1.5).attach_to(base)
    gm.gen_pointcloud(pcd1_np, pntsize=5).attach_to(base)
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