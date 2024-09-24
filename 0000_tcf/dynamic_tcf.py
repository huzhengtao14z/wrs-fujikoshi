# import robotsim.ur3dual.ur3dual as ur3dual
# from robotsim.ur3dual import ur3dualmesh
# from robotsim.ur3dual import ur3dualball
# import robotsim.ur3edual.ur3edual as ur3edual
# from robotsim.ur3edual import ur3edualmesh
# from robotsim.ur3edual import ur3edualball
# import envloader as el
import basis.data_adapter as da
from panda3d.core import *
import modeling.geometric_model as gm
import modeling.collision_model as cm
import visualization.panda.world as wd
import random
import robot_sim.end_effectors.grippers.robotiqhe.robotiqhe as rtqhe
import modeling.dynamics.bullet.bdmodel as bdm
import os
import basis.robot_math as rm
import basis.trimesh as trimesh
import numpy as np
import slope
from direct.gui.OnscreenText import OnscreenText
import pickle
import time as tm
from shapely.geometry import Polygon


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
    rotating = rm.rotmat_from_axangle(axis=(0, 0, 1), angle=np.radians(180))
    moving = np.array([.600, 0, .780])
    objT = rm.homomat_from_posrot(moving, rotating)
    # objT=da.pdmat4_to_npmat4((p3du.npToMat4(rotating, moving))
    # objhomomat=np.dot(objT,p3du.mat4ToNp(RotMatnozero))
    return objT


def computehomomat(RotMatnozero, height, error):
    roterror_x = rm.rotmat_from_axangle(axis=(1, 0, 0), angle=np.radians(error[3]))
    roterror_y = rm.rotmat_from_axangle(axis=(0, 1, 0), angle=np.radians(error[4]))
    roterror_z = rm.rotmat_from_axangle(axis=(0, 0, 1), angle=np.radians(error[5]))
    roterror = rm.homomat_from_posrot(rot=np.dot(roterror_x, np.dot(roterror_y, roterror_z)))
    rotating = rm.rotmat_from_axangle(axis=(0, 0, 1), angle=np.radians(180))
    moving = np.array([0.600 + error[0], 0 + error[1], height + error[2]])
    objT = rm.homomat_from_posrot(moving, rotating)
    objhomomat = np.dot(objT, np.dot(da.pdmat4_to_npmat4(RotMatnozero), roterror))
    return objhomomat

if __name__=="__main__":
    # base = pandactrl.World(camp=[100, 0, 3000], lookatp=[300, 0, 1000])
    this_dir, this_filename = os.path.split(__file__)
    base = wd.World(cam_pos=[1.60982, 1.42529, 1.44947], w=960, h=540, lookat_pos=[0.3, 0, 1])
    base.setFrameRateMeter(True)
    gm.gen_frame().attach_to(base)

    repetition = 5
    friction = 0.2
    restitution=0.5
    heightlist=[0.001*i for i in range(820,860,30)]
    errorrange_xyz = 0.003
    errorrange_rpy = np.radians(2)
    errorconfiguration=geterrorconfiguration(errorrange_xyz,errorrange_rpy)

    xyz_error_print=printonscreen(pos=(-1,0.7,0),words="Error range on xyz: "+str(errorrange_xyz)+" (m)")
    rpy_error_print = printonscreen(pos=(-1, 0.6, 0), words="Error range on rpy: " + str(np.degrees(errorrange_rpy))+" (degree)")

    address = this_dir+"/PlacementData"
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
    
    RotMatnozero=[RotMat[i] for i in range(len(stablecklist)) if stablecklist[i] is True]
    RotMatnozeroID=[i for i in range(len(stablecklist)) if stablecklist[i] is True]
    comlist = [comlistall[i] for i in range(len(stablecklist)) if stablecklist[i] is True]

    objpath = os.path.join(this_dir, "object-1000", objname+".stl")
    objpath = objpath.replace('\\', '/')
    objbm = []

    testobj = bdm.BDModel(objpath, mass=1, restitution=restitution, dynamic=True, friction=friction, type="triangles")
    # testobj.set_scale((0.001,0.001,0.001))
    testobj.set_rgba((.3, .5, .7, 0.5))
    testobj.start_physics()
    testobj.set_linearDamping(0.1)
    testobj.set_angularDamping(0.1)
    objbm.append(testobj)

    # table
    plane = cm.gen_box([5.0000, 5.000, .000100], rm.homomat_from_posrot([.500, 0, -.050]))
    planebm = bdm.BDModel(objinit = plane, mass = 0 , friction = friction)
    planebm.set_rgba((.5,.5,.5,1))
    planebm.set_pos(np.array([0,0,0.600]))
    planebm.start_physics()

    # slope
    slope=slope.Slope(z=0,placement="ping",size=0.5)
    slopemodels=slope.getSlopeDym(mass=0, restitution=restitution, dynamic=True, friction=friction)
    slopePos=[0.600,0,0.78]
    slopemodels[0].set_pos(slopePos)
    slopemodels[1].set_pos(slopePos)
    slopemodels[2].set_pos(slopePos)

    base.attach_internal_update_obj(objbm[0])
    base.attach_internal_update_obj(planebm)
    base.attach_internal_update_obj(slopemodels[0])
    base.attach_internal_update_obj(slopemodels[1])
    base.attach_internal_update_obj(slopemodels[2])

    # base.run()
    print("Placement No. 0")
    print("Height try No. 0")
    stablecklable = printonscreen(pos=(-1, 0.1, 0), words="Stability:")
    placementID = printonscreen(pos=(-1, 0.8, 0),
                                  words="Placement ID: ")

    def update(objbm,comlist,count,countforrot,time,objposnode,error,estimation,estimation_com,clock,task):
        # print(time[0])
        if time[0] == 1:
            clock[0] = tm.time()
            error[0] = geterrorconfiguration(errorrange_xyz, errorrange_rpy)
            # objbm[0] = bdm.BDModel(objpath, mass=1000, restitution=restitution,dynamic=True, friction=.3, shapetype="triangle")
            objbm[0].set_homomat(computehomomat(RotMatnozero[countforrot[0]], heightlist[count[0]], error=error[0]))
            objbm[0].set_linearVelocity(0)
            objbm[0].set_angularVelocity(0)
        if time[0] == 10:
            objbm[0].set_mass(1)
            # base.attach_internal_update_obj(objbm[0])
            objposnode[0] = printonscreen(pos=(-1,0.4,0),words= "Initial ObjPos (Real): x=" + str(0.600+np.round(error[0][0],decimals=5))+ "  " + "y=" + str(0+np.round(error[0][1],decimals=5))+ "  " + "z=" + str(heightlist[count[0]]+np.round(error[0][1],decimals=5)))
            objposnode[1] = printonscreen(pos=(-1, -0.2, 0), words="Repetition: "+str(count[1]+1) + "/"+str(repetition))
            objposnode[4] = printonscreen(pos=(-1, 0.3, 0), words="Initial ObjPos (Ideal): x=" + str(0.600) + "  " + "y=" + str(0) + "  " + "z=" + str(heightlist[count[0]]))
            objposnode[5] = printonscreen(pos=(-1, 0.2, 0),
                                          words="Rot error: x axis: " + str(np.round(error[0][3],decimals=5)) + "  " + "y axis: " + str(
                                              np.round(error[0][4], decimals=2)) + "  " + "z axis: " + str(np.rad2deg(np.round(error[0][5],decimals=5))) +"  (degree)")
            objposnode[6] = printonscreen(pos=(-0.5, 0.8, 0),
                                          words=str(countforrot[0]),color="navy")
        # estimation_com[1] = NodePath("normal")
        if time[0]==320:
            estimation_com[0] = NodePath("normal")
            gm.gen_sphere(pos=np.round(np.dot(computehomomatforcom(), comlist[countforrot[0]]), decimals=4)[:3],
                                  radius=.0030, rgba=(1, 0, 0, 0.5)).attach_to(estimation_com[0])
            gm.gen_sphere(pos=np.round(np.dot(objbm[0].get_homomat(), np.concatenate((objbm[0].cm.objtrm.center_mass, np.array([1])), axis=0)), decimals=4)[:3],
                                  radius=.0030, rgba=(0, 1, 0, 0.5)).attach_to(estimation_com[0])
            estimation_com[0].reparentTo(base.render)
        # estimation_com[0].reparentTo(base.render)
        if time[0]==399:
            if objposnode[3] is not None:
                objposnode[3].detachNode()
        if time[0]==400 :
            objposnode[0].detachNode()
            objposnode[1].detachNode()
            objposnode[4].detachNode()
            objposnode[5].detachNode()
            objposnode[6].detachNode()
            estimation_com[0].detachNode()
            # estimation_com[1].detachNode()
            objcom = np.round(
                np.dot(objbm[0].get_homomat(), np.concatenate((objbm[0].cm.objtrm.center_mass, np.array([1])), axis=0)), decimals=5)
            com = np.round(np.dot(computehomomatforcom(), comlist[countforrot[0]]), decimals=5)
            print(np.linalg.norm(objcom - com))
            # < 0.0001
            if objcom[0] <= com[0] + 0.0001 and objcom[0] >= com[0] - 0.0001 \
                    and objcom[1] <= com[1] + 0.0001 and objcom[1] >= com[1] - 0.0001 \
                    and objcom[2] <= com[2] + 0.0001 and objcom[2] >= com[2] - 0.0001:
                objposnode[3] = printonscreen(pos=(-0.65, 0.1, 0), words="Stable", color="green")
                criteria = 1
            else:
                objposnode[3] = printonscreen(pos=(-0.65, 0.1, 0), words="Unstable!!", color="red")
                criteria = 0
            estimation.append(
                [countforrot[0], RotMatnozeroID[countforrot[0]], error[0], [.600, 0, heightlist[count[0]]], objcom, criteria])

            objbm[0].set_rgba((.3, .5, .7, 0.5))
            clock[1]=tm.time()
            print(clock[1]-clock[0])
            if count[1] == repetition-1:
                count[1] = 0
                count[0] += 1
            else:
                count[1] += 1

            if count[0]==len(heightlist):
                countforrot[0]+=1
                # print("Placement No.",countforrot[0])
                count[0]=0
                count[1] = 0
            if countforrot[0]==len(RotMatnozero):
            # if countforrot[0] == len(RotMatnozero):
                name = objname + "_estimation_3.pickle"
                with open(name, "wb") as file:
                    pickle.dump(estimation, file)
            time[0] = 0

        # if base.inputmgr.keymap['space'] is True:
        #     objbm[0]=bdm.BDModel(objpath, mass=1, restitution=restitution, dynamic=True, friction=.2, type="triangles")
        #     error[0] = geterrorconfiguration(errorrange_xyz, errorrange_rpy)
        #     objbm[0].sethomomat(computehomomat(RotMatnozero[countforrot[0]], heightlist[count[0]], error=error[0]))
        #     objbm[0].setColor(.3, .5, .7, 0.2)
        #     base.attach_internal_update_obj(objbm[0])
        time[0]+=1
        return task.cont

    count=[0,0]
    time=[0]
    countforrot=[0]
    objposnode=[None,None,None,None,None,None,None]
    error=[errorconfiguration]
    estimation = []
    estimation_com = [None,None]
    clock=[None,None]
    objposnode[3] = printonscreen(pos=(-0.65, 0.1, 0), words="Under computing", color="navy")
    taskMgr.add(update, extraArgs=[objbm, comlist,count,countforrot,time,objposnode,error,estimation,estimation_com,clock], appendTask=True)
    base.run()