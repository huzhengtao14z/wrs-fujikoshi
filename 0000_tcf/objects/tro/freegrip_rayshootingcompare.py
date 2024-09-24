import os
import time
import numpy as np
from panda3d.core import Mat4
import pandaplotutils.pandactrl as pandactrl
import pandaplotutils.pandageom as pandageom
import environment.collisionmodel as cm
from manipulation.grip import freegripcontactpairs as fgcp
from database import dbcvt as dc
import environment.bulletcdhelper as bcd
import utiltools.robotmath as rm
import trimesh.sample as sample
from sklearn.neighbors import RadiusNeighborsClassifier
import environment.bulletcdhelper as bh
import manipulation.grip.freegrip_rayshooting as fgr
import pickle

if __name__=='__main__':
    import manipulation.grip.robotiq85.robotiq85 as hnd
    import environment.collisionmodel as cm
    from panda3d.core import GeomNode
    from panda3d.core import NodePath
    from panda3d.core import Vec4

    # base = pandactrl.World(camp=[700,300,700], lookatp=[0,0,100])
    base = pandactrl.World(camp=[700,-300,300], lookatp=[0,0,100])

    pathname = "./objects/"
    objname = ["housing", "planewheel", "tool2", "ttube", "sandpart", "planelowerbody", "planerearstay", "bunnysim"]
    type = ["160ray"]
    for objn in objname:
        objpath = pathname+objn+".stl"
        objcm = cm.CollisionModel(objinit = objpath)

        hndfa = hnd.Robotiq85Factory()
        hand = hndfa.genHand()
        freegriprstst = fgr.FreegripRS(objcm)

        variouscosts = []

        tic = time.time()
        freegriprstst.planGrasps(hand)
        toc = time.time()
        print("plan grasp cost", toc-tic)
        print("number of grasps planned", len(freegriprstst.griprotmats_planned))
        print("number of samples", len(freegriprstst.samplepnts_refcls))
        print("number of contact pairs", len(freegriprstst.contactpairs))
        variouscosts.append(toc-tic)
        variouscosts.append(len(freegriprstst.griprotmats_planned))
        variouscosts.append(len(freegriprstst.samplepnts_refcls))
        variouscosts.append(len(freegriprstst.contactpairs))

        with open('facets-' + objn + "-" + type[0] + "-variouscosts" + '.pickle', mode='wb') as f:
            pickle.dump(variouscosts, f)

    base.run()