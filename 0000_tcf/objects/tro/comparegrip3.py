
if __name__=='__main__':
    import pandaplotutils.pandactrl as pandactrl
    import pandaplotutils.pandageom as pandageom
    import manipulation.grip.robotiq85.robotiq85 as hnd
    import manipulation.grip.freegrip as fg
    import environment.collisionmodel as cm
    from panda3d.core import GeomNode
    from panda3d.core import NodePath
    from panda3d.core import Vec4
    import numpy as np
    import database.dbaccess as db
    import random
    import trimesh as tm
    import time
    import utiltools.robotmath as rm
    import matplotlib.pyplot as plt
    import environment.bulletcdhelper as bh
    import pickle
    import os

    # base = pandactrl.World(camp=[700,300,700], lookatp=[0,0,100])
    base = pandactrl.World(camp=[700,-300,300], lookatp=[0,0,100])

    mc = bh.MCMchecker()
    # objpath = "../objects/sandpart.stl"
    # objpath = "../objects/043screwdriver1000.stl"
    # objpath = "../objects/035drill1000.stl"
    # objpath = "../objects/043screwdriver1000.stl"
    # objpath = "../objects/025mug1000.stl"
    # objpath = "../objects/025mug3000.stl"
    # objpath = "../objects/025mug3000_tsdf.stl"
    # objpath = "../objects/035drill3000_tsdf.stl"
    # objpath = "../objects/072toyplanedrill3000_tsdf.stl"
    # objpath = "../objects/housing.stl"
    # objpath = "../objects/ttube.stl"
    # objpath = "./objects/bunny12356.stl"
    # objstpath = "./objects/bunny17432.stl"

    pathname = "./objects/"
    objname = "housing"
    objns = [94, 189, 379, 561, 724]
    stn = objns[-1]
    objstpath = pathname+objname+str(stn)+".stl"
    for objn in objns:
        objpath = pathname+objname+str(objn)+".stl"

        objcm = cm.CollisionModel(objinit = objpath)
        # objcm.reparentTo(base.render)
        objstcm = cm.CollisionModel(objinit = objstpath)

        hndfa = hnd.Robotiq85Factory()

        difflistall = []
        for i in range(10):
            hand= hndfa.genHand()
            freegriptst = fg.Freegrip(objpath, hand, faceangle = .9, segangle = .9, refine1min=5, togglebcdcdebug=True, useoverlap=True)

            tic = time.time()
            freegriptst.planGrasps()
            toc = time.time()
            print("plan grasp cost", toc-tic)

            ngrasp = 0
            for pfacets in freegriptst.griprotmats_planned:
                for gmats in pfacets:
                    ngrasp += len(gmats)
            nsamples = 0
            for j in range(len(freegriptst.facets)):
                for samples in freegriptst.objsamplepnts_refcls[j]:
                    nsamples += len(samples)
            ncpairs = 0
            for pfacets in freegriptst.gripcontactpairs:
                for cpairs in pfacets:
                    ncpairs += len(cpairs)
            print("number of grasps planned", ngrasp)
            print("number of samples", nsamples)
            print("number of contactpairs", ncpairs)

            difflist = []
            for j, freegriprotmat in enumerate(freegriptst.griprotmats_planned):
                print("repeating planning...", i, 10, "saving grasp...", j, len(freegriptst.griprotmats_planned))
                ct0, ct1 = freegriptst.gripcontacts_planned[j]
                ctn0, ctn1 = freegriptst.gripcontactnormals_planned[j]
                hitpos0, hitnrml0 = mc.isRayHitMeshClosest(ct0 + ctn0 * 10, ct0 - ctn0 * 100000, objstcm)
                hitpos1, hitnrml1 = mc.isRayHitMeshClosest(ct1 + ctn1 * 10, ct1 - ctn1 * 100000, objstcm)
                difflist.append((rm.degree_betweenvector(-hitnrml0, ctn0) + rm.degree_betweenvector(-hitnrml1, ctn1)) / 2)
                hand = hndfa.genHand()
                hand.setColor(1, 1, 1, .3)
                newpos = freegriprotmat.getRow3(3) - freegriprotmat.getRow3(2) * 0.0
                freegriprotmat.setRow(3, newpos)
                hand.setMat(pandamat4=freegriprotmat)
                hand.setjawwidth(freegriptst.gripjawwidth_planned[j])
                hand.reparentTo(base.render)
            difflistall.append(difflist)
        with open('diff-'+objname+str(objn)+"-"+str(stn)+'.pickle', mode='wb') as f:
            pickle.dump(difflistall, f)

    # for samples in freegriptst.objsamplepnts_refcls:
    #     for pnt in samples:
    #         base.pggen.plotSphere(base.render, pos=pnt, radius=4, rgba=[.7,0,0,1])

    # def update(freegriptst, mc, objstcm, task):
    #     if base.inputmgr.keymap['space'] is True:
    #         for i, pfacets in enumerate(freegriptst.gripcontactpairs):
    #             for j, contactpair in enumerate(pfacets):
    #                 cpt0 = contactpair[0]
    #                 nrmal0 = freegriptst.gripcontactpairnormals[i][j][0]
    #                 cpt1 = contactpair[1]
    #                 nrmal1 = freegriptst.gripcontactpairnormals[i][j][1]
    #                 base.pggen.plotSphere(base.render, pos=cpt0, radius=5, rgba=[.7,0,0,1])
    #                 base.pggen.plotArrow(base.render, spos=cpt0, epos=cpt0+nrmal0, length=10, thickness=2, rgba=[.7,0,0,1])
    #                 base.pggen.plotSphere(base.render, pos=cpt1, radius=5, rgba=[0,0,.7,1])
    #                 base.pggen.plotArrow(base.render, spos=cpt1, epos=cpt1+nrmal1, length=10, thickness=2, rgba=[0,0,.7,1])
    #         base.inputmgr.keymap['space'] = False
    #     if base.inputmgr.keymap['g'] is True:
    #         difflist = []
    #         for i, freegriprotmat in enumerate(freegriptst.griprotmats_planned):
    #             print(i, len(freegriptst.griprotmats_planned))
    #             ct0, ct1 = freegriptst.gripcontacts_planned[i]
    #             ctn0, ctn1 = freegriptst.gripcontactnormals_planned[i]
    #             hitpos0, hitnrml0 = mc.isRayHitMeshClosest(ct0+ctn0*10, ct0-ctn0*100000, objstcm)
    #             hitpos1, hitnrml1 = mc.isRayHitMeshClosest(ct1+ctn1*10, ct1-ctn1*100000, objstcm)
    #             difflist.append((rm.degree_betweenvector(-hitnrml0, ctn0)+rm.degree_betweenvector(-hitnrml1, ctn1))/2)
    #             hand = hndfa.genHand()
    #             hand.setColor(1,1,1,.3)
    #             newpos = freegriprotmat.getRow3(3)-freegriprotmat.getRow3(2)*0.0
    #             freegriprotmat.setRow(3, newpos)
    #             hand.setMat(pandamat4=freegriprotmat)
    #             hand.setjawwidth(freegriptst.gripjawwidth_planned[i])
    #             hand.reparentTo(base.render)
    #         with open('diff'+os.path.basename(objpath)+os.path.basename(objstpath)+'.pickle', mode='wb') as f:
    #             pickle.dump(difflist, f)
    #         base.inputmgr.keymap['g'] = False
    #     return task.again
    #
    # taskMgr.doMethodLater(0.05, update, "update",
    #                       extraArgs=[freegriptst, mc, objstcm],
    #                       appendTask=True)
    base.run()

