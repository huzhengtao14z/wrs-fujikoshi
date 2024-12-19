import copy
import math
# from keras.models import Sequential, Model, load_model
import visualization.panda.world as wd
import modeling.collision_model as cm
# import hufunc as hf
import humath
import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yg
import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as hnde
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
import numpy as np
import basis.robot_math as rm
import modeling.geometric_model as gm
import os
import pickle
import basis.data_adapter as da
# import slope
# import Sptpolygoninfo as sinfo
import basis.trimesh as trimeshWan
import trimesh as trimesh

from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from descartes import PolygonPatch
import shapely
from shapely.geometry import MultiPoint
from shapely.geometry import Point


def getFacetsCenter(obj_trimesh, facets):
    '''
    get the coordinate of large facet center in self.largefacetscenter
    get the normal vecter of large facet center in self.largefacet_normals
    get the vertices coordinate of large facet in self.largefacet_vertices
    :return:
    '''
    vertices = obj_trimesh.vertices
    faces = obj_trimesh.faces
    facets = facets
    smallfacesarea = obj_trimesh.area_faces
    smallface_normals = obj_trimesh.face_normals
    smallfacecenter = []
    # prelargeface = []
    smallfacectlist = []
    smallfacectlist_area = []
    largefacet_normals = []
    largefacet_vertices = []
    largefacetscenter = []
    for i, smallface in enumerate(faces):
        smallfacecenter.append(humath.centerPoint(np.array([vertices[smallface[0]],
                                                                 vertices[smallface[1]],
                                                                 vertices[smallface[2]]])))
    for facet in facets:
        b = []
        b_area = []
        temlargefaceVerticesid = []
        temlargefaceVertices = []
        for face in facet:
            b.append(smallfacecenter[face])
            b_area.append(smallfacesarea[face])
            temlargefaceVerticesid.extend(faces[face])
            # print("temlargefaceVerticesid", temlargefaceVerticesid)
        smallfacectlist.append(b)
        smallfacectlist_area.append(b_area)
        smallfacenomallist = [smallface_normals[facet[j]] for j in range(len(facet))]
        largefacet_normals.append(np.average(smallfacenomallist, axis=0))
        # self.largefacet_normals.append(self.smallface_normals[facet[0]]) #TODO an average normal
        temlargefaceVerticesid = list(set(temlargefaceVerticesid))  # remove repeating vertices ID
        for id in temlargefaceVerticesid:
            temlargefaceVertices.append(vertices[id])
        largefacet_vertices.append(temlargefaceVertices)
    for i, largeface in enumerate(smallfacectlist):
        largefacetscenter.append(humath.centerPointwithArea(largeface, smallfacectlist_area[i]))
    return largefacetscenter, largefacet_normals, largefacet_vertices

if __name__ == '__main__':
    base = wd.World(cam_pos=[0.501557, 0.137317, 0.48133], w=960,
                    h=540, lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    this_dir, this_filename = os.path.split(__file__)
    obj = cm.CollisionModel('objects/test_long_small.stl')
    obj.set_rgba([1, 1, 0, 0.5])
    obj.attach_to(base)
    obj_ch= obj.objtrm.convex_hull
    vertices = obj_ch.vertices
    faces = obj_ch.faces
    com = obj.objtrm.center_mass
    # base.run()
    convex_obj = trimeshWan.Trimesh(vertices=vertices, faces=faces)
    convex_obj_gm = gm.GeometricModel(convex_obj)
    convex_obj_gm.set_rgba([1, 1, 0, 1])
    convex_obj_gm.attach_to(base)
    facets, facetnormals, facetcurvatures = convex_obj.facets_noover(faceangle=0.99)
    # for id, item in enumerate(facets_center):
    #     gm.gen_sphere(item, radius=0.005).attach_to(base)
    #     gm.gen_arrow(item, item+facet_normal[id]*0.05).attach_to(base)

    facets_center, facet_normal, facet_vertices = getFacetsCenter(convex_obj, facets)
    pos_list = [np.array([0, 0, 0]) + center for center in facets_center]
    rotmat_list = [rm.rotmat_between_vectors(normal, np.array([0, 0, 1])) for normal in facet_normal]
    # for id, item in enumerate(facets_center):
    #     gm.gen_sphere(item, radius=0.005).attach_to(base)
    #     gm.gen_arrow(item, item+facet_normal[id]*0.05).attach_to(base)
    # base.run()
    facet_project_list = []
    com_project_list = []
    rgba_list = []
    for id, facet in enumerate(facet_vertices):
        facet_project = [rotmat_list[id].dot(vertex)[:2] for vertex in facet]
        facet_project_list.append(facet_project)
        com_project_list.append(rotmat_list[id].dot(com)[:2])

    for id, facet_project in enumerate(facet_project_list):
        contact_polygon = Polygon(facet_project)
        contact_patch = PolygonPatch(shapely.convex_hull(contact_polygon), fc='yellow', ec='black', alpha=0.5)
        fig = plt.figure(figsize=(5, 5), dpi=100)
        plt.axis('on')
        # plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_ylim(-0.1, 0.1)
        ax1.set_xlim(-0.1, 0.1)
        # ring_patch2 = PolygonPatch(contact_patch, color="yellow", alpha=0.5)
        ax1.add_patch(contact_patch)

        if shapely.within(Point(com_project_list[id][0], com_project_list[id][1]), contact_polygon):
            ax1.scatter(x=com_project_list[id][0], y=com_project_list[id][1], color="green")
            rgba_list.append([0,1,0,1])
        else:
            ax1.scatter(x=com_project_list[id][0], y=com_project_list[id][1], color="red")
            rgba_list.append([1, 0, 0, 1])
    plt.show()



    homo_list = [rm.homomat_from_posrot(pos_list[i], rotmat_list[i]) for i in range(len(pos_list))]
    homo_inv_list = [np.linalg.inv(homo) for homo in homo_list]

    displacement_list = list(np.linspace([0, 0, 0], [0.8, 0, 0], len(rotmat_list)))
    displacement_homo_list = [rm.homomat_from_posrot(displacement_list[i]).dot(homo_inv_list[i]) for i in range(len(displacement_list))]
    for id,homo in enumerate(displacement_homo_list):
        obj_copy = copy.deepcopy(obj)
        obj_copy.set_homomat(homo)
        obj_copy.set_rgba(rgba_list[id])
        obj_copy.attach_to(base)

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