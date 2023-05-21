#-*-coding:utf-8-*-
import hufunc
import visualization.panda.world as wd
import basis.trimesh as trimesh
import modeling.collision_model as cm
import modeling.geometric_model as gm
from shapely.geometry import Polygon
import numpy as np
from modeling.dynamics.bullet import bdmodel as bdm
from panda3d.core import *
from direct.gui.OnscreenText import OnscreenText
import basis.data_adapter as da

class Slope():
    def __init__(self,z=-0.015,placement="shu",size=1.0, show=True):
        self.placement=placement
        self.size=size
        if self.placement=="shu":
            self.slopex = trimesh.primitives.Box(box_extents = (0.150,0.150,0.006))
            self.slopex = cm.CollisionModel(self.slopex)
            self.slopex.set_rgba((255/255,51/255,153/255, 1))
            self.slopex.set_pos((0, 0, z))
            self.slopex.set_rpy(roll = 0, pitch = -45, yaw = 0)
            # self.slopex.reparentTo(base.render)
            self.slopey = trimesh.primitives.Box( box_extents = (0.200,0.200,0.006))
            self.slopey = cm.CollisionModel(self.slopey)
            self.slopey.set_rgba((51/255,255/255,255/255, 1))
            self.slopey.set_pos((0, 0, z))
            self.slopey.set_rpy(roll = 45, pitch = 45, yaw = 0)
            # self.slopey.reparentTo(base.render)
            self.slopez = trimesh.primitives.Box( box_extents = (0.200,0.200,0.006))
            self.slopez = cm.CollisionModel(self.slopez)
            self.slopez.set_rgba((178/255,102/255,255/255, 1))
            self.slopez.set_pos(( 0,  0,  z))
            self.slopez.set_rpy(roll = -45, pitch = 45, yaw = 0)
            # self.slopez.reparentTo(base.render)
        if placement == "ping":
            s = self.size
            self.slopex = trimesh.primitives.Extrusion(
                extrude_polygon=Polygon([np.array([0, 0]), np.array([0.0707, -0.0707]) * s, np.array([0.0707, 0.0707]) * s]),
                extrude_transform=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.006], [0, 0, 0, 1]]),
                extrude_height=.006)
            self.slopex = cm.CollisionModel(self.slopex)
            self.slopex.set_rgba((51 / 255, 255 / 255, 255 / 255, 1))
            self.slopex.set_pos((0, 0, z))
            self.slopex.set_rpy(roll=0, pitch=np.radians(-54.74), yaw=0)

            self.slopey = trimesh.primitives.Extrusion(
                extrude_polygon=Polygon([np.array([0, 0.100]) * s, np.array([0, 0]), np.array([-0.100, 0]) * s]),
                extrude_transform=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.006], [0, 0, 0, 1]]),
                extrude_height=0.006)
            self.slopey = cm.CollisionModel(self.slopey)
            self.slopey.set_rgba((51 / 255, 255 / 255, 255 / 255, 1))
            self.slopey.set_pos((0, 0, z))
            self.slopey.set_rpy(roll=np.radians(45), pitch=np.radians(90-54.75), yaw=0)

            self.slopez = trimesh.primitives.Extrusion(
                extrude_polygon=Polygon([np.array([0, -0.100]) * s, np.array([0, 0]), np.array([-0.100, 0]) * s]),
                extrude_transform=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.006], [0, 0, 0, 1]]),
                extrude_height=0.006)
            self.slopez = cm.CollisionModel(self.slopez)
            self.slopez.set_rgba((51 / 255, 255 / 255, 255 / 255, 1))
            self.slopez.set_pos((0, 0, z))
            self.slopez.set_rpy(roll=np.radians(-45), pitch=np.radians(90 - 54.75), yaw=0)
            if show:
                self.slopex.attach_to(base)
                self.slopez.attach_to(base)
                self.slopey.attach_to(base)

    def getSlope(self):
        return [self.slopex,self.slopey,self.slopez]

    def getSlopeDym(self,mass=0, restitution=0, dynamic=True, friction=0.3):
        s=self.size

        if self.placement=="shu":
            self.slopex = trimesh.primitives.Extrusion(
                extrude_polygon=Polygon(
                    [np.array([0, 0]), np.array([0.0707 * s, -70.7 * s]), np.array([0.0707 * s, 0.0707 * s])]),
                extrude_transform=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -6], [0, 0, 0, 1]]),
                extrude_height=6)
            self.slopex = bdm.BDModel(self.slopex, mass=mass,restitution=restitution,dynamic=dynamic, friction=friction)
            self.slopex.set_rgba((255 / 255, 51 / 255, 153 / 255, 1))
            self.slopex.set_pos((0, 0, -0))
            self.slopex.set_rpy(roll=0, pitch=np.radians(-45), yaw=np.radians(180))

            self.slopey = trimesh.primitives.Extrusion(
                extrude_polygon=Polygon(
                    [np.array([56 * s, 0]), np.array([56 * s, 100 * s]), np.array([-50 * s, 100 * s]),
                     np.array([-50 * s, 0])]),
                extrude_transform=np.array([[1, 0, 0, -50 * s], [0, 1, 0, 0], [0, 0, 1, -6], [0, 0, 0, 1]]),
                extrude_height=6)
            self.slopey =bdm.BDModel(self.slopey, mass=mass,restitution=restitution,dynamic=dynamic, friction=friction)
            self.slopey.set_rgba((255 / 255, 51 / 255, 153 / 255, 1))
            self.slopey.set_pos((0, 0, 0))
            self.slopey.set_rpy(roll=np.radians(45), pitch=np.radians(45), yaw=np.radians(180))

            self.slopez = trimesh.primitives.Extrusion(
                extrude_polygon=Polygon(
                    [np.array([56 * s, 0]), np.array([56 * s, -100 * s]), np.array([-50 * s, -100 * s]),
                     np.array([-50 * s, 0])]),
                extrude_transform=np.array([[1, 0, 0, -50 * s], [0, 1, 0, 0], [0, 0, 1, -6], [0, 0, 0, 1]]),
                extrude_height=6)
            self.slopez = bdm.BDModel(self.slopez, mass=mass,restitution=restitution,dynamic=dynamic, friction=friction)
            self.slopez.set_rgba((255 / 255, 51 / 255, 153 / 255, 1))
            self.slopez.set_pos((0, 0, 0))
            self.slopez.set_rpy(roll=np.radians(-45), pitch=np.radians(45), yaw=np.radians(0))


        if self.placement=="ping":
            self.slopex = trimesh.primitives.Extrusion(
                extrude_polygon=Polygon(
                    [np.array([0, 0]) * s, np.array([.0707, -0.0707]) * s, np.array([0.0707, 0.0707]) * s]),
                extrude_transform=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.006], [0, 0, 0, 1]]),
                extrude_height=0.006)
            self.slopex = bdm.BDModel(self.slopex, mass=mass, restitution=restitution, dynamic=dynamic,
                                      friction=friction)
            self.slopex.set_rgba((255 / 255, 51 / 255, 153 / 255, 1))
            self.slopex.set_pos((0, 0, -0))
            self.slopex.set_rpy(roll=0, pitch=np.radians(-54.74), yaw=np.radians(180))

            self.slopey = trimesh.primitives.Extrusion(
                extrude_polygon=Polygon(
                    [np.array([0, .100]) * s, np.array([0, 0]) * s, np.array([-0.100, 0]) * s]),
                extrude_transform=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.006], [0, 0, 0, 1]]),
                extrude_height=0.006)
            self.slopey = bdm.BDModel(self.slopey, mass=mass, restitution=restitution, dynamic=dynamic,
                                      friction=friction)
            self.slopey.set_rgba((255 / 255, 51 / 255, 153 / 255, 1))
            self.slopey.set_pos((0, 0, 0))
            self.slopey.set_rpy(roll=np.radians(45), pitch=np.radians(90-54.75), yaw=np.radians(180))

            self.slopez = trimesh.primitives.Extrusion(
                extrude_polygon=Polygon(
                    [np.array([0, -.100]) * s, np.array([0, 0]) * s, np.array([-.100, 0]) * s]),
                extrude_transform=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.006], [0, 0, 0, 1]]),
                extrude_height=0.006)
            self.slopez = bdm.BDModel(self.slopez, mass=mass, restitution=restitution, dynamic=dynamic,
                                      friction=friction)
            self.slopez.set_rgba((255 / 255, 51 / 255, 153 / 255, 1))
            self.slopez.set_pos((0, 0, 0))
            self.slopez.set_rpy(roll=np.radians(-45), pitch=np.radians(90 - 54.75), yaw=np.radians(180))

        self.slopex.start_physics()
        self.slopey.start_physics()
        self.slopez.start_physics()
        # self.slopex.attach_to(base)
        # self.slopey.attach_to(base)
        # self.slopez.attach_to(base)
        return [self.slopex,self.slopey,self.slopez]

if __name__=="__main__":
    import os

    cam_view_text = OnscreenText(text="Camera View: ", fg=(0, 0, 0, 1), pos=(1.15, 0.9), align=TextNode.ALeft)
    base = wd.World(cam_pos=[0.600,-.600,0.800], w=960, h=540, lookat_pos=[0, 0, 0.1])
    gm.gen_frame().attach_to(base)
    slope = Slope(z=-0.050, placement="ping", size=0.7070)
    alist = slope.getSlope()
    # alist = slope.getSlopeDym()
    # alist[0].attach_to(base)
    # alist[1].attach_to(base)
    # alist[2].attach_to(base)
    # base.pggen.plotAxis(base.render, length=60, thickness = 3)
    this_dir, this_filename = os.path.split(__file__)

    slopeforshowpath = os.path.join(this_dir, "objects", "tc71.stl")
    slopeforshowpath = slopeforshowpath.replace('\\', '/')
    slopeforshow = cm.CollisionModel(slopeforshowpath)
    slopeforshow.set_scale((0.001,0.001,0.001))
    slopeforshow.set_rgba((.8, .6, .3, 0.5))
    slopeforshow.set_pos((0,0,0))
    # slopeforshow.attach_to(base)

    # slopeforshowpath = os.path.join(this_dir, "..", "objects", "tc71.stl")
    # slopeforshowpath = slopeforshowpath.replace('\\', '/')
    # slopeforshow = cm.CollisionModel(initor=slopeforshowpath)
    # slopeforshow.set_rgba(.8, .6, .3, 0.5)
    # slopeforshow.set_pos(x=0, y=-170, z=0)
    # slopeforshow.attach_to(base.render)
    #
    # slopeforshowpath = os.path.join(this_dir, "..", "objects", "tc50.stl")
    # slopeforshowpath = slopeforshowpath.replace('\\', '/')
    # slopeforshow = cm.CollisionModel(initor=slopeforshowpath)
    # slopeforshow.set_rgba(.8, .6, .3, 0.5)
    # slopeforshow.set_pos(x=0, y=-300, z=0)
    # slopeforshow.attach_to(base.render)

    # for i in range(4):
    #     slopeforshowpath = os.path.join(this_dir, "..", "objects", "tc50.stl")
    #     slopeforshowpath = slopeforshowpath.replace('\\', '/')
    #     slopeforshow = cm.CollisionModel(initor=slopeforshowpath)
    #     slopeforshow.set_rgba(.8, .6, .3, 1-0.2*i)
    #     slopeforshow.set_pos(x=0, y=0, z=0)
    #     slopeforshow.set_rpy(0, i*7, 0)
    #     # slopeforshow.attach_to(base.render)
    # for i in range(4):
    #     slopeforshowpath = os.path.join(this_dir, "..", "objects", "tc50.stl")
    #     slopeforshowpath = slopeforshowpath.replace('\\', '/')
    #     slopeforshow = cm.CollisionModel(initor=slopeforshowpath)
    #     slopeforshow.set_rgba(.8, .6, .3, 1-0.2*i)
    #     slopeforshow.set_pos(x=0, y=0, z=0)
    #     slopeforshow.set_rpy(0, -i*7, 0)
    #     # slopeforshow.attach_to(base.render)

    slopeforshowpath = os.path.join(this_dir, "objects", "tc141.stl")
    slopeforshowpath = slopeforshowpath.replace('\\', '/')
    slopeforshow = cm.CollisionModel(initor=slopeforshowpath)
    slopeforshow.set_scale((0.001,0.001,0.001))
    slopeforshow.set_rgba((.8, .6, .3, 1))
    # slopeforshow.set_pos((0, 220, -5*1.414))
    # slopeforshow.set_rpy(54.74,0,30)
    slopeforshow.attach_to(base.render)

    slopeforshowpath = os.path.join(this_dir, "objects", "tc71.stl")
    slopeforshowpath = slopeforshowpath.replace('\\', '/')
    slopeforshow = cm.CollisionModel(initor=slopeforshowpath)
    slopeforshow.set_scale((0.001,0.001,0.001))
    slopeforshow.set_rgba((.8, .6, .3, 1))
    slopeforshow.set_pos((0, 0.220, 0))
    # slopeforshow.set_rpy(54.74,0,30)
    slopeforshow.attach_to(base.render)

    slopeforshowpath = os.path.join(this_dir, "objects", "tc50.stl")
    slopeforshowpath = slopeforshowpath.replace('\\', '/')
    slopeforshow = cm.CollisionModel(initor=slopeforshowpath)
    slopeforshow.set_scale((0.001, 0.001, 0.001))
    slopeforshow.set_rgba((.8, .6, .3, 1))
    slopeforshow.set_pos((0, 0.120, 0))
    # slopeforshow.set_rpy(54.74,0,30)
    slopeforshow.attach_to(base.render)

    slopeforshowpath = os.path.join(this_dir, "objects", "tc100.stl")
    slopeforshowpath = slopeforshowpath.replace('\\', '/')
    slopeforshow = cm.CollisionModel(initor=slopeforshowpath)
    slopeforshow.set_scale((0.001, 0.001, 0.001))
    slopeforshow.set_rgba((.8, .6, .3, 1))
    slopeforshow.set_pos((0, -0.120, 0))
    # slopeforshow.set_rpy(54.74,0,30)
    slopeforshow.attach_to(base.render)

    hufunc.viewmonitor()
    base.run()
