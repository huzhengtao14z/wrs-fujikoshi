
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
    objname = ["housing", "planewheel", "tool2", "ttube", "sandpart", "planelowerbody", "planerearstay", "bunnysim"]
    type = ["160over", "160noover"]
    for objn in objname:
        objpath = pathname+objn+".stl"
        objcm = cm.CollisionModel(objinit = objpath)
        hndfa = hnd.Robotiq85Factory()

        for i in range(2):
            hand= hndfa.genHand()
            if i == 0:
                freegriptst = fg.Freegrip(objpath, hand, faceangle = .9, segangle = .9, refine1min=5, fpairparallel=-0.9, togglebcdcdebug=True, useoverlap=True)
            if i == 1:
                freegriptst = fg.Freegrip(objpath, hand, faceangle = .9, segangle = .9, refine1min=5, fpairparallel=-0.9, togglebcdcdebug=True, useoverlap=False)

            # facetsizes = []
            # for j, faces in enumerate(freegriptst.facets):
            #     rgba = [np.random.random(), np.random.random(), np.random.random(), 1]
            #     # compute facet normal
            #     facetnormal = np.sum(freegriptst.objtrimesh.face_normals[faces], axis=0)
            #     facetnormal = facetnormal / np.linalg.norm(facetnormal)
            #     geom = pandageom.packpandageom_fn(freegriptst.objtrimesh.vertices +
            #                                       np.tile(0 * facetnormal,
            #                                               [freegriptst.objtrimesh.vertices.shape[0], 1]),
            #                                       freegriptst.objtrimesh.face_normals[faces],
            #                                       freegriptst.objtrimesh.faces[faces])
            #     node = GeomNode('piece')
            #     node.addGeom(geom)
            #     star = NodePath('piece')
            #     star.attachNewNode(node)
            #     star.setColor(Vec4(rgba[0], rgba[1], rgba[2], rgba[3]))
            #     star.setTwoSided(True)
            #     star.reparentTo(base.render)
            #
            #     facetsizes.append(
            #         tm.Trimesh(vertices=freegriptst.objtrimesh.vertices, faces=freegriptst.objtrimesh.faces[faces],
            #                    face_normals=freegriptst.objtrimesh.face_normals[faces]).area)
            # print("repeating segmentation...", i+1, 2, "saving facetsizes...")
            # with open('facets-'+objn+"-"+type[i]+'.pickle', mode='wb') as f:
            #     pickle.dump(facetsizes, f)

            variouscosts = []
            tic = time.time()
            freegriptst.planGrasps()
            toc = time.time()
            print("plan grasp cost", toc - tic)
            variouscosts.append(toc-tic)

            ngrasp = 0
            for pfacets in freegriptst.griprotmats_planned:
                for gmats in pfacets:
                    ngrasp += len(gmats)
            nsamples = 0
            for k in range(len(freegriptst.facets)):
                for samples in freegriptst.objsamplepnts_refcls[k]:
                    nsamples += len(samples)
            ncpairs = 0
            for pfacets in freegriptst.gripcontactpairs:
                for cpairs in pfacets:
                    ncpairs += len(cpairs)
            print("number of grasps planned", ngrasp)
            print("number of samples", nsamples)
            print("number of contactpairs", ncpairs)
            variouscosts.append(ngrasp)
            variouscosts.append(nsamples)
            variouscosts.append(ncpairs)
            with open('facets-'+objn+"-"+type[i]+"-variouscosts"+'.pickle', mode='wb') as f:
                pickle.dump(variouscosts, f)

    base.run()

