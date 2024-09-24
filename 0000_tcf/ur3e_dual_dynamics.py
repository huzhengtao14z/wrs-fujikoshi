import robot_sim.robots.ur3e_dual.ur3e_dual as ur3ed
from panda3d.core import NodePath, TextNode
from direct.gui.OnscreenText import OnscreenText
import modeling.dynamics.bullet.bdmodel as bdm
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import basis.robot_math as rm
import numpy as np
import random
import pickle
import basis.data_adapter as da

def printonscreen(pos, words, color="black", a=1):
    node = NodePath("text")
    node.reparentTo(base.render)
    if color == "red":
        fg = (1, 0, 0, a)
    elif color == "green":
        fg = (0, 1, 0, a)
    elif color == "yellow":
        fg = (1, 1, 0, a)
    elif color == "navy":
        fg = (0, 0, 0.5, a)
    else:
        fg = (0, 0, 0, a)

    node = OnscreenText(
        text=words, pos=pos,
        scale=0.07,
        fg=fg,
        align=TextNode.ALeft, mayChange=1)
    return node

def geterrorconfiguration(errorrange_xyz,errorrange_rpy):
    errorrange_xyz = errorrange_xyz
    errorrange_rpy = errorrange_rpy
    x = random.uniform(-errorrange_xyz, errorrange_xyz)
    y = random.uniform(-errorrange_xyz, errorrange_xyz)
    z = random.uniform(-errorrange_xyz, errorrange_xyz)
    x_axis = random.uniform(-errorrange_rpy, errorrange_rpy)
    y_axis = random.uniform(-errorrange_rpy, errorrange_rpy)
    z_axis = random.uniform(-errorrange_rpy, errorrange_rpy)
    return [x, y, z, x_axis, y_axis, z_axis]

def computehomomatforcom():
    rotating=rm.rotmat_from_axangle(axis=(0,0,1),angle=np.radians(180))
    moving=np.array([.600,0,.780])
    objT =rm.homomat_from_posrot(moving, rotating)
    # objT=da.pdmat4_to_npmat4((p3du.npToMat4(rotating, moving))
    # objhomomat=np.dot(objT,p3du.mat4ToNp(RotMatnozero))
    return objT

def computehomomat(RotMatnozero,height,error):
    roterror_x = rm.rotmat_from_axangle(axis=(1, 0, 0), angle=np.radians(error[3]))
    roterror_y = rm.rotmat_from_axangle(axis=(0, 1, 0), angle=np.radians(error[4]))
    roterror_z = rm.rotmat_from_axangle(axis=(0, 0, 1), angle=np.radians(error[5]))
    roterror = rm.homomat_from_posrot(rot = np.dot(roterror_x, np.dot(roterror_y, roterror_z)))
    rotating=rm.rotmat_from_axangle(axis=(0,0,1),angle=np.radians(180))
    moving=np.array([0.600+error[0],0+error[1],height+error[2]])
    objT=rm.homomat_from_posrot(moving, rotating)
    objhomomat=np.dot(objT,np.dot(da.pdmat4_to_npmat4(RotMatnozero),roterror))
    return objhomomat

if __name__ == '__main__':
    import os

    this_dir, this_filename = os.path.split(__file__)
    base = wd.World(cam_pos=[10, 0, 5], lookat_pos=[0, 0, 1])
    base.setFrameRateMeter(True)
    gm.gen_frame().attach_to(base)

    restitution = 0.5
    heightlist = [0.001 * i for i in range(820, 860, 30)]
    errorrange_xyz = 0.003
    errorrange_rpy = np.radians(2)
    errorconfiguration = geterrorconfiguration(errorrange_xyz, errorrange_rpy)
    xyz_error_print = printonscreen(pos=(-1, 0.7, 0), words="Error range on xyz: " + str(errorrange_xyz) + " (mm)")
    rpy_error_print = printonscreen(pos=(-1, 0.6, 0), words="Error range on rpy: " + str(errorrange_rpy) + " (degree)")

    # obj_box = cm.gen_box(extent=[.2, 1, .3], rgba=[.3, 0, 0, 1])
    obj_box = cm.gen_sphere(radius=.5, rgba=[.3, 0, 0, 1])
    obj_bd_box = bdm.BDModel(obj_box, mass=.3, type="triangles")
    obj_bd_box.set_pos(np.array([.7, 0, 2.7]))
    obj_bd_box.start_physics()
    base.attach_internal_update_obj(obj_bd_box)

    plane1 = cm.gen_box([0.30000, 0.3000, .3000100], rm.homomat_from_posrot([.500, 0, -.050]))
    plane1bm = bdm.BDModel(objinit=plane1, mass=1)
    plane1bm.set_rgba((.5, .5, .5, 1))
    plane1bm.set_pos(np.array([0, 0, 1.000]))
    # plane1bm.start_physics()
    # base.attach_internal_update_obj(plane1bm)

    plane = cm.gen_box([5.0000, 5.000, .000100], rm.homomat_from_posrot([.500, 0, -.050]))
    planebm = bdm.BDModel(objinit=plane, mass=0)
    planebm.set_rgba((.5, .5, .5, 1))
    planebm.set_pos(np.array([0, 0, 0.600]))
    planebm.start_physics()
    base.attach_internal_update_obj(planebm)

    address = this_dir + "/PlacementData"
    # objname = "test_long"
    objname = "test_long_small"
    # objname = "yuanbox_small"
    # objname = "lofted"
    # objname = "bar"
    # objname = "angle"
    # objname = "wedge"
    # objname = "polysolid"
    # objname = "box10"
    with open(address + "/" + objname + "/" + "placementrotmat.pickle", "rb") as f:
        RotMat = pickle.load(f)
    with open(address + "/" + objname + "/" + "stablecklist.pickle", "rb") as f:
        stablecklist = pickle.load(f)
    with open(address + "/" + objname + "/" + "placementcom.pickle", "rb") as f:
        comlistall = pickle.load(f)

    RotMatnozero = [RotMat[i] for i in range(len(stablecklist)) if stablecklist[i] is True]
    RotMatnozeroID = [i for i in range(len(stablecklist)) if stablecklist[i] is True]
    comlist = [comlistall[i] for i in range(len(stablecklist)) if stablecklist[i] is True]
    objpath = os.path.join(this_dir, "object-1000", objname + ".stl")
    objpath = objpath.replace('\\', '/')
    objbm = []

    testobj = bdm.BDModel(objpath, mass=0, restitution=restitution, dynamic=True, friction=0, type="convex")
    testobj.set_scale((1, 1, 1))
    testobj.set_rgba((.3, .5, .7, 0.5))
    testobj.start_physics()
    objbm.append(testobj)

    base.attach_internal_update_obj(objbm[0])

    base.run()
