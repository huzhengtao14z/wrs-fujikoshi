import copy

import yaml
from panda3d.core import NodePath
import numpy as np

import os
import modeling.collision_model as cm

import basis.robot_math as rm
from trimesh.primitives import Box
import robot_sim.robots.ur3e_dual.ur3e_dual as ur3ed
root = os.path.abspath(os.path.dirname(__file__))
# bcdchecker = bch.MCMchecker(toggledebug=False)
import visualization.panda.world as wd

def loadEnv():

    #Table width: 120
    #Table long: 1080

    base = wd.World(cam_pos=[2, 1, 3], w=960,
                    h=540, lookat_pos=[0, 0, 1.1])
    # env = wf.Env(boundingradius=7.0)
    # env.reparentTo(base.render)
    # # obstacle = cm.CollisionModel(objinit=Box(box_extents=[289,700,30]))
    # # env.addchangableobs(base.render,obstacle,[780-289/2-34,-600+410-64+700/2,800],np.eye(3))
    # yellowcar = cm.CollisionModel(objinit=Box(box_extents=[900,450, 740]))
    # yellowcar.setColor(1,1,0,1)
    # env.addchangableobs(base.render, yellowcar, [180+900/2, 450/2+530, 740/2], np.eye(3))
    #
    # #-------------------- box
    # obstacle = cm.CollisionModel(objinit=Box(box_extents=[30,298, 194]))
    # env.addchangableobs(base.render, obstacle, [1080+30/2, -600+200 +298/2-20, 780+97], np.eye(3))
    #
    # obstacle = cm.CollisionModel(objinit=Box(box_extents=[60 , 298, 15]))
    # env.addchangableobs(base.render, obstacle, [1080, -600+200  +298/2-20, 780 + 130], np.eye(3))
    #
    # obstacle = cm.CollisionModel(objinit=Box(box_extents=[60,40 , 194]))
    # env.addchangableobs(base.render, obstacle, [1080 , -600 +200+105 +298/2, 780 + 97], np.eye(3))
    # #------------------
    # # nailbox = cm.CollisionModel(objinit=Box(box_extents=[210, 300, 23]))
    # # nailbox.setColor(0.3,0.2,0.6,1)
    # # env.addchangableobs(base.render, nailbox, [1080-300, -600+110+300/2, 780 + 23/2], np.eye(3))
    #
    # # motorbody = cm.CollisionModel(objinit=Box(box_extents=[60, 60, 100]))
    # # motorbody.setColor(0.1, 0.1, 0.1, 1)
    # # env.addchangableobs(base.render, motorbody, [600,-400,780+100/2], np.eye(3))
    #
    # # toolzone = cm.CollisionModel(objinit=Box(box_extents=[150, 300, 200]))
    # # toolzone.setColor(0.7, 0.7, 0.7, 1)
    # # env.addchangableobs(base.render, toolzone, [910+150/2, 350-300, 780 + 200 / 2], np.eye(3))
    # #--------------------------------------------------phonix
    # phoxicam = cm.CollisionModel(objinit=Box(box_extents=[200, 430, 100]))
    # phoxicam.setColor(0.3, 0.2, 0.2, 1)
    # env.addchangableobs(base.render, phoxicam, [100+50,0, 1360+175+90/2], np.eye(3))

    return base

def loadRbt():
    rbtball = ur3ed.UR3EDual()
    rbthi = ur3ed.UR3EDual()

    rbtmg = rbthi.gen_meshmodel()
    return rbthi, rbtmg