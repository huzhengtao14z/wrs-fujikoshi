#-*-coding:utf-8-*-
import numpy as np
import basis.trimesh as trimesh
from panda3d.core import *
# import pandaplotutils.pandageom as pandageom
from hu import humath
import modeling.geometric_model as gm
import basis.data_adapter as da


def drawSurface(base,vertices, normal, faces, color):
    '''
    draw a surface using pandageom.packpandageom
    :param base:
    :param vertices:
    :param normal:
    :param faces:
    :param color:
    :return:
    '''
    geom = pandageom.packpandageom_fn(vertices=vertices,
                                      facenormals=normal,
                                      triangles=faces)
    node = GeomNode('piece')
    node.addGeom(geom)
    checknode = node.getGeom(0)
    print('----------------------------')
    print(checknode)
    print('----------------------------')
    star = NodePath('piece')
    star.attachNewNode(node)
    star.setColor(color)
    star.setTwoSided(True)
    star.reparentTo(base.render)

def drawSingleFaceSurface(base,vertices, faces, color):
    '''
    draw a surface using a calculated fourth point to creat a hull
    :param base:
    :param vertices:
    :param faces:
    :param color:
    :return:
    '''
    # print("faces in plotsurface",faces)

    surface_vertices = np.array([vertices[faces[0]], vertices[faces[1]], vertices[faces[2]]])
    surface = humath.centerPoftrangle(surface_vertices[0], surface_vertices[1], surface_vertices[2])
    surface = trimesh.Trimesh(surface)
    print("hu")
    surface = surface.convex_hull
    # surface = base.pg.trimeshtonp(surface)
    gm.GeometricModel(surface)
    surface.set_rgba(color)
    surface.attach_to(base)

def drawanySingleSurface(base,vertices,color):
    '''
    draw a surface using a calculated fourth point to creat a hull
    :param base:
    :param vertices:
    :param faces:
    :param color:
    :return:
    '''
    # print("faces in plotsurface",faces)

    surface_vertices = vertices
    # surface = humath.centerPoftrangle(surface_vertices[0][:3], surface_vertices[1][:3], surface_vertices[2][:3])
    surface = trimesh.Trimesh(surface_vertices)
    surface = surface.convex_hull
    surface = da.trimesh_to_nodepath(surface)
    surface.set_color(color)
    surface.reparentTo(base.render)

def monitor(textNode):
    from direct.gui.OnscreenText import OnscreenText
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


def viewmonitor():
    def update(textNode, task):
        monitor(textNode)
        return task.again

    textNode = [None, None, None]
    taskMgr.add(update, "addobject", extraArgs=[textNode], appendTask=True)