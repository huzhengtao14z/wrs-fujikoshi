#-*-coding:utf-8-*-
import itertools
import os

import os
import os.path
from direct.gui.OnscreenText import OnscreenText
import visualization.panda.world as wd
import numpy as np
import copy
import modeling.geometric_model as gm
import grasping.planning.antipodal as gpa
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletTriangleMesh
from panda3d.bullet import BulletTriangleMeshShape
from panda3d.bullet import BulletWorld
from panda3d.core import *
from shapely.geometry import Point
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from sklearn.neighbors import RadiusNeighborsClassifier
# import environment.collisionmodel as cm
import modeling.collision_model as cm
# import pandaplotutils.pandactrl as pandactrl
import trimesh.sample as sample
import sys
import basis.trimesh as trimesh
# from utiltools import robotmath
import pickle
import time
import humath
import hufunc
import random
import basis.data_adapter as da
import basis.robot_math as rm
import modeling._ode_cdhelper as mcd

from hu import humath
import stbchecker
import random
import freeholdcontactpairs as fh
# from environment import bulletcdhelper as cdchecker
import slope as slope
import os.path
import pickle
import robot_sim.end_effectors.grippers.robotiqhe.robotiqhe as rtqhe
import Sptpolygoninfo as sinfo



class stableholdplanner(fh.FreeholdContactpairs, slope.Slope):
    def __init__(self, objname, slopename, slopeforcd, slopeforcd_high, dataaddress, checkstablity=False):
        this_dir, this_filename = os.path.split(__file__)
        self.objname = objname
        self.objpath = os.path.join(this_dir, "objects", objname + ".stl")
        self.objpath = self.objpath.replace('\\', '/')
        super().__init__(self.objpath, faceangle=.98, segangle=.98, verticalthreshold=.995, useoverlap=True)

        self.dataaddress = dataaddress
        self.slopename = slopename
        self.slopeforshowpath = os.path.join(this_dir, "objects", slopename)
        self.slopeforshowpath = self.slopeforshowpath.replace('\\', '/')

        self.loadslopeforshow()
        self.slopelist = slopeforcd
        print("slopelist", self.slopelist)# for collision check
        self.slopelist_high = slopeforcd_high

        self.getFacetsCenter()
        self.planHoldpairs()
        self.getFacetsArea()
        self.getHoldingpairSptPnt()
        self.getRotMat()
        self.getplacementRotMat()
        self.collisionlist = self.checkcollision()

        if checkstablity == True:
            self.stablelist, self.realstableindex_nolp, self.stabilityindex_nolp, self.contactinfo, self.picnodepath = self.checkstablity(
                lpsolve=False)
            print("finish stability check")
            self.stabilityindex = [self.stabilityindex_nolp]
            print("start checkgrasp")
            self.checkgrasp()
            print("finish checkgrasp")

        else:
            self.stablelist = None
            self.stablelist_lp = None
            self.stabilityindex_lp = None

        self.cfreerotmat = []
        # self.__getonebasedrotmats()

    def loadslopeforshow(self):
        slopeforshow = cm.CollisionModel(initor=self.slopeforshowpath)
        slopeforshow.set_rgba((.8, .6, .3, 0.5))
        slopeforshow.set_scale((0.001, 0.001, 0.001))
        # slopeforshow.setPos(x=0, y=0, z=-30)
        slopeforshow.attach_to(base)

    def checkcollision(self):
        # checker = cdchecker.MCMchecker(toggledebug=True)
        temobj = cm.CollisionModel(initor=self.objpath)
        temobj.set_scale((0.001, 0.001, 0.001))
        collisionlist = []
        for i in range(len(self.pRotMat)):
            temobj.set_homomat(da.pdmat4_to_npmat4(self.pRotMat[i]))
            checkresult = temobj.is_mcdwith(self.slopelist)
            # checkresult = checker.isMeshMeshListCollided(temobj, self.slopelist)
            collisionlist.append(checkresult)
        return collisionlist

    def checkstablity(self,lpsolve,friction=True):
        sptPnt = sinfo.Sptvertices[self.slopename]
        sptrate = sinfo.Sptrate[self.slopename]
        com = self.pCoM
        stable_list = []
        stableindex_list = []
        contactinformation_list = []
        picnodepath = []
        checker = stbchecker.StableChecher()
        contactinformation = 0
        for i, objPnt in enumerate(self.pFaces):
            if lpsolve == False:
                stable, stablelist, contactinformation, intersectionvertices = checker.stablecheck_nodistribution(
                    objlist=objPnt,
                    sptPnt=sptPnt,
                    com=com[i],
                    collisionlist=self.collisionlist[i],
                    normallist=self.pNormals[i],
                    facetcenterlist=self.pFacetpairscenter[i])
                picnodepath.append(
                    checker.getcontactinfo(center=com[i], intersectionvertices=intersectionvertices, sptPnt=sptPnt, sptrate = sptrate, polygon=True,
                                           point=False, ID=i, name=self.objname, thisdir=self.dataaddress,
                                           show=True))
            else:
                if friction == True:
                    stable, stablelist = checker.stabilityindex(objlist=objPnt,
                                                                sptPnt=sptPnt,
                                                                center=com[i],
                                                                collisionlist=self.collisionlist[i],
                                                                normallist=self.pNormals[i],
                                                                facetcenterlist=self.pFacetpairscenter[i],
                                                                u=0.3)
                else:
                    stable, stablelist = checker.stabilityindex(objlist=objPnt,
                                                                sptPnt=sptPnt,
                                                                center=com[i],
                                                                collisionlist=self.collisionlist[i],
                                                                normallist=self.pNormals[i],
                                                                facetcenterlist=self.pFacetpairscenter[i],
                                                                u=0)
            stable_list.append(stable)
            stableindex_list.append(stablelist)
            contactinformation_list.append(contactinformation)

        normalstablityindex = []
        maxindex = max(stableindex_list)
        for index in stableindex_list:
            if maxindex == 0:
                normalstablityindex.append(0)
                continue
            normalstablityindex.append(index / maxindex)

        return stable_list, stableindex_list, normalstablityindex, contactinformation_list, picnodepath

    def checkgrasp(self):
        grasp = gpa.load_pickle_file(self.objname, None, "rtqhe.pickle")
        grasp_info_list = grasp
        print("number of grasp 1", len(grasp))
        hndfa = rtqhe.RobotiqHE(enable_cc= True)
        rtqHE = copy.deepcopy(hndfa)
        rtqHE.jaw_to(jawwidth=0.035)
        slope = self.slopelist_high
        obj = cm.CollisionModel(initor=self.objpath)
        obj.set_scale((0.001,0.001,0.001))
        cfreegrasp = []
        for i, rotmat4 in enumerate(self.pRotMat):
            temcfreegrasp = []
            if self.collisionlist[i] == False and self.stablelist[i] == True:
                obstaclelist = slope
                obj.set_homomat(da.pdmat4_to_npmat4(rotmat4))
                obstaclelist.append(obj)
                for grasp in grasp_info_list:
                    jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp
                    rotmatgraspnp = np.dot(da.pdmat4_to_npmat4(rotmat4), rm.homomat_from_posrot(hnd_pos, hnd_rotmat))
                    rotmat4panda = da.npmat4_to_pdmat4(rotmatgraspnp)
                    rtqHE.fix_to(pos= rotmatgraspnp[:3, 3], rotmat = rotmatgraspnp[:3, :3] ,jawwidth = jaw_width)
                    if rtqHE.is_mesh_collided(obstaclelist) == False:
                        temcfreegrasp.append([rotmat4panda, False])
                    else:
                        temcfreegrasp.append([rotmat4panda, True])
                        print("grasp in collision")
            cfreegrasp.append(temcfreegrasp)
            print("number of grasp 2", len(temcfreegrasp))
        self.cfreegrasppose = cfreegrasp

        # return stable_list, stableindex_list

    def outputvertices(self, address):
        print("outoutputvertices")
        name = "placmentVertices.pickle"
        with open(address+"/"+name, "wb") as file:
            pickle.dump(self.pVertices, file)

    def outputverticeswithface(self, address):
        print("outputverticeswithface")
        name = "placementVerticeswithface.pickle"
        with open(address+"/"+name, "wb") as file:
            pickle.dump(self.pFaces, file)

    def outputcom(self, address):
        print("outputcom")
        name = "placementcom.pickle"
        with open(address+"/"+name, "wb") as file:
            pickle.dump(self.pCoM, file)

    def outputcollisionlist(self, address):
        print("outputcollisionlist")
        name = "collisionlist.pickle"
        with open(address+"/"+name, "wb") as file:
            pickle.dump(self.collisionlist, file)

    def outputnormal(self, address):
        print("outputnormal")
        name = "placementnormal.pickle"
        with open(address+"/"+name, "wb") as file:
            pickle.dump(self.pNormals, file)

    def outputfacetcenter(self, address):
        print("outputfacetcenter")
        name = "placementfacetcenter.pickle"
        with open(address+"/"+name, "wb") as file:
            pickle.dump(self.pFacetpairscenter, file)

    def outputplacmentrotmat(self, address):
        print("outputplacmentrotmat")
        name = "placementrotmat.pickle"
        with open(address+"/"+name, "wb") as file:
            pickle.dump(self.pRotMat,file)

    def outputstability(self, address):
        print("outputstability")
        name = "stablecklist.pickle"
        with open(address+"/"+name, "wb") as file:
            pickle.dump(self.stablelist, file)
        name = "normalstablityindex.pickle"
        with open(address+"/"+name, "wb") as file:
            pickle.dump(self.stabilityindex_nolp, file)
        name = "realstablityindex.pickle"
        with open(address + "/" + name, "wb") as file:
            pickle.dump(self.realstableindex_nolp, file)

    def getcfreerotmat(self):
        for i in range(len(self.collisionlist)):
            if self.collisionlist[i] is False:
                self.cfreerotmat.append(self.pRotMat[i])
        return self.cfreerotmat

    def __rotmattoRPY(self):
        self.rotRPY = []
        for rotmat in self.pRotMat:
            rotmat = p3du.mat4ToNp(rotmat)
            rotmat33=rotmat[0:3,0:3]
            self.rotRPY.append(rm.rotmat_to_euler(rotmat33))

    def __onebasedrotmattoRPY(self):
        self.rotRPY = []
        for rotmat in self.pRotMat:
            rotmat = p3du.mat4ToNp(rotmat)
            rotmat33=rotmat[0:3,0:3]
            self.rotRPY.append(rm.rotmat_to_euler(rotmat33))

    def __getonebasedrotmats(self):
        self.onebasedrotmats = []
        self.onebasedrotRPY = []
        self.onebasedrotRPYdistance = []
        self.placementRotMat33=[]
        for rotmat in self.pRotMat:
            rotmat = p3du.mat4ToNp(rotmat)
            rotmat33 = rotmat[0:3, 0:3]
            self.placementRotMat33.append(rotmat33)
        for i,rotmat in enumerate(self.placementRotMat33):
            rotmat_ni= np.linalg.inv(rotmat)
            temonebasedrotmat = []
            temonebasedrotRPY = []
            temonebasedrotRPYdistance = []
            for j in range(0, len(self.placementRotMat33)):
                if j !=i and self.collisionlist[j]==False:#add the False adjustment if no considering about collision
                    temonebasedrotmat.append(np.dot(rotmat_ni,self.placementRotMat33[j]))
                    transRottoRPY=rm.rotmat_to_euler(np.dot(rotmat_ni,self.placementRotMat33[j]))
                    # temonebasedrotRPY.append(transRottoRPY)
                    if abs(transRottoRPY[0]+transRottoRPY[1]+transRottoRPY[2])>0.00001:
                        temonebasedrotRPY.append(transRottoRPY)
                        temonebasedrotRPYdistance.append(humath.distance(transRottoRPY,[0,0,0]))
            self.onebasedrotmats.append(temonebasedrotmat)
            self.onebasedrotRPY.append(temonebasedrotRPY)
            self.onebasedrotRPYdistance.append(min(temonebasedrotRPYdistance))
        print("self.onebasedrotmats",self.onebasedrotmats)
        print("self.onebasedrotRPY", self.onebasedrotRPY)
        print("self.onebasedrotRPYdistance", self.onebasedrotRPYdistance)

    def getrotmatsdiff(self):
        self.rotmatsdiff = []
        self.placementRotMat33=[]
        for rotmat in self.pRotMat:
            rotmat = p3du.mat4ToNp(rotmat)
            rotmat33 = rotmat[0:3, 0:3]
            self.placementRotMat33.append(rotmat33)
        for i,rotmat in enumerate(self.placementRotMat33):
            # rotmat_ni= np.linalg.inv(rotmat)
            # temonebasedrotmat = []
            # temonebasedrotRPY = []
            temrotmatsdiff = []
            for j in range(0, len(self.placementRotMat33)):
                if j !=i and self.collisionlist[j]==False:#add the False adjustment if no considering about collision
                    temtemrotmatsdiff=rm.axangle_between_rotmat(self.placementRotMat33[i],self.placementRotMat33[j])
                    if temtemrotmatsdiff>0.00001:
                        temrotmatsdiff.append(temtemrotmatsdiff)
        #     self.onebasedrotmats.append(temonebasedrotmat)
        #     self.onebasedrotRPY.append(temonebasedrotRPY)
            self.rotmatsdiff.append(min(temrotmatsdiff))
        # print("self.onebasedrotmats",self.onebasedrotmats)
        # print("self.onebasedrotRPY", self.onebasedrotRPY)
        # print("self.onebasedrotRPYdistance", self.onebasedrotRPYdistance)

    def showRPYcedixian(self):
        from matplotlib import pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        import matplotlib
        import math
        import matplotlib.ticker as ticker
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['font.size'] = 35
        fig = plt.figure()
        plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.21)
        ax = plt.gca()
        bwith = 2
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        plt.tick_params(axis='both', direction='in', length=6, width=2, labelsize=30, pad=15)
        ax.tick_params(pad=3)
        ax.set_xticks(np.arange(0, len(self.rotmatsdiff), int(math.ceil(len(self.rotmatsdiff)/10))))
        ax.set_yticks(np.arange(0, 120, 20))
        plt.xlabel("placment No.", labelpad=5)
        plt.ylabel("RPY distance", labelpad=5)
        ax.set_xlim(-1, len(self.rotmatsdiff))
        ax.set_ylim(0, 120)
        plt.bar(range(len(self.rotmatsdiff)), self.rotmatsdiff, label='boy', fc='y')

        plt.show()

    def showRPY(self, baserotid):
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib
        import matplotlib.ticker as ticker
        fig = plt.figure()
        ax = Axes3D(fig, rect=[0.0, 0.03, 1.0, 1.0])
        elev = 30
        azim = -53
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(pad=3)
        ax.set_xlim(-200, 200)
        ax.set_zlim(-200, 200)
        ax.set_ylim(-200, 200)
        ax.plot([0,0],[0,0],[0,300],c="r",linewidth=5)
        ax.plot([0,0],[0,300],[0,0],c="g",linewidth=5)
        ax.plot([0,300],[0,0],[0,0],c="b",linewidth=5)

        for otherrotRPY in self.onebasedrotRPY[baserotid]:
            ax.scatter(otherrotRPY[0], otherrotRPY[1], otherrotRPY[2],color="k",s=100)
        plt.show()

    def showRPYdistance(self):
        from matplotlib import pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        import matplotlib
        import math
        import matplotlib.ticker as ticker
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['font.size'] = 35
        fig = plt.figure()
        plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.21)
        ax = plt.gca()
        bwith = 2
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        plt.tick_params(axis='both', direction='in', length=6, width=2, labelsize=30, pad=15)
        ax.set_xticks(np.arange(0, len(self.onebasedrotRPYdistance), int(math.ceil(len(self.onebasedrotRPYdistance)/10))))
        ax.set_yticks(np.arange(0, 120, 20))
        plt.xlabel("placment No.", labelpad=5)
        plt.ylabel("RPY distance", labelpad=5)
        ax.tick_params(pad=3)
        ax.set_xlim(-1, len(self.onebasedrotRPYdistance))
        ax.set_ylim(0, 120)
        plt.bar(range(len(self.onebasedrotRPYdistance)), self.onebasedrotRPYdistance, label='boy', fc='y')
        plt.show()

    def showStabilityIndex(self):
        from matplotlib import pyplot as plt
        import matplotlib
        import math
        import matplotlib.ticker as ticker
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['font.size'] = 35
        fig = plt.figure()
        plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.21)
        ax = plt.gca()
        bwith = 2
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        plt.tick_params(axis='both', direction='in', length=6, width=2, labelsize=30, pad=15)
        ax.set_xticks(np.arange(0, len(self.stabilityindex), int(math.ceil(len(self.stabilityindex)/10))))
        # ax.set_yticks(np.arange(0, 0.02, 0.02/5))
        ax.set_yticks(np.arange(0, 1.2, 1.2 / 5))
        plt.xlabel("placment No.", labelpad=5)
        plt.ylabel("Stability Index", labelpad=5)
        ax.tick_params(pad=3)
        ax.set_xlim(-1, len(self.stabilityindex))
        # ax.set_ylim(0, 0.02)
        ax.set_ylim(0, 1.2)
        normalstablityindex=[]
        maxindex=max(self.stabilityindex)
        for stabilityindex in self.stabilityindex:
            normalstablityindex.append(stabilityindex/maxindex)
        # plt.bar(range(len(self.stabilityindex)), sorted(normalstablityindex,reverse=True), label='boy', fc='y')
        plt.bar(range(len(self.stabilityindex)), normalstablityindex, label='boy', fc='y')
        plt.show()

    def showfeatures(self, node, normal,placementfacetpairscenter, placementVertices, placementCoM, showCoM, showNormal, showVertices):
        np.random.seed(0)
        rgb=np.random.rand(3)
        # color = (random.random(),random.random(),random.random(),1)
        color = (rgb[0], rgb[1], rgb[2], 1)
        # length = random.randint(60,100)
        length = 0.050
        # self.__showSptPnt(self.holdingpairsSptPnt[i], color=color)
        if showNormal==True:
            for i in range(len(normal)):
                gm.gen_arrow(spos = np.array([placementfacetpairscenter[i][0],
                                                placementfacetpairscenter[i][1],
                                                placementfacetpairscenter[i][2]]),
                                            epos = np.array([placementfacetpairscenter[i][0]+length*normal[i][0],
                                             placementfacetpairscenter[i][1]+length*normal[i][1],
                                             placementfacetpairscenter[i][2]+length*normal[i][2]]), thickness=0.005, rgba = color).attach_to(node)
        if showCoM == True:
            gm.gen_sphere(pos=(placementCoM[0], placementCoM[1], placementCoM[2]), radius=.003, rgba=color).attach_to(node)
            gm.gen_arrow(spos=np.array([placementCoM[0], placementCoM[1], placementCoM[2]]),
                                        epos=np.array([placementCoM[0],
                                              placementCoM[1],
                                              placementCoM[2]-0.100]), thickness=0.001, rgba=color).attach_to(node)
        if showVertices == True:
            for vertices in placementVertices:
                color2 = (random.random(), random.random(), random.random(), 1)
                for i in range(len(vertices)):
                    gm.gen_sphere(pos = (vertices[i][0],vertices[i][1],vertices[i][2]), radius=.005, rgba = color2).attach_to(node)

    def animation(self):
        counter = [0]
        from direct.gui.OnscreenText import OnscreenText
        from direct.gui.OnscreenImage import OnscreenImage
        def update(objmnp, counter, testnode, normalnode, graspnode, textNPose, textNPose_lp, picnode, task):
            if base.inputmgr.keymap['space'] is True:
                hndfa = rtqhe.RobotiqHE(enable_cc=True)
                if counter[0] < len(self.pRotMat):
                    print("It's No.", counter[0], "placement")
                    print("Is it collision?", self.collisionlist[counter[0]])
                    if objmnp[0] is not None:
                        objmnp[0].detach()
                        normalnode[0].detachNode()
                        graspnode[0].detachNode()
                        textNPose[0].detachNode()
                        textNPose[1].detachNode()
                        # textNPose_lp[0].detachNode()
                    objmnp[0] = cm.CollisionModel(initor=self.objpath)
                    objmnp[0].set_scale((0.001,0.001,0.001))
                    objmnp[0].set_homomat(da.pdmat4_to_npmat4(self.pRotMat[counter[0]]))

                    normalnode[0] = NodePath("normal")
                    graspnode[0] = NodePath("graspnode")
                    textNPose[0] = NodePath("text")
                    textNPose[1] = NodePath("text")
                    textNPose_lp[0] = NodePath("text")
                    # graspnode[0].reparentTo(base.render)
                    normalnode[0].reparentTo(base.render)

                    for j in range(len(self.cfreegrasppose[counter[0]])):
                        if j%30 != 0:
                            if not self.cfreegrasppose[counter[0]][j][1]:
                                hndfa.gen_meshmodel(rgba=(0, 1, 0, 0.15)).attach_to(graspnode[0])
                            else:
                                continue
                        print("number of grasp 3", len(self.cfreegrasppose[counter[0]]))
                        hndfa.fix_to(pos=da.pdmat4_to_npmat4(self.cfreegrasppose[counter[0]][j][0])[:3, 3],
                                     rotmat=da.pdmat4_to_npmat4(self.cfreegrasppose[counter[0]][j][0])[:3, :3])
                        if self.cfreegrasppose[counter[0]][j][1]:
                            hndfa.gen_meshmodel(rgba=(1, 0, 0, 0.05)).attach_to(graspnode[0])
                        else:
                            hndfa.gen_meshmodel(rgba=(0, 1, 0, 0.15)).attach_to(graspnode[0])

                    graspnode[0].reparentTo(base.render)
                    self.showfeatures(node=normalnode[0], normal=self.pNormals[counter[0]],
                                          placementfacetpairscenter=self.pFacetpairscenter[counter[0]],
                                          placementVertices=self.pVertices[counter[0]],
                                          placementCoM=self.pCoM[counter[0]],
                                          showCoM=True,
                                          showVertices=False,
                                          showNormal=False
                                          )
                    if self.stablelist is not None:
                        print("stable?", self.stablelist[counter[0]])
                        # textNPose[0] = OnscreenText(
                        #     text="Stability (uniform) is:"+str(self.stabilityindex[0][counter[0]]), pos=(-.9, -.8, 0),
                        #     scale=0.1,
                        #     fg=(1., 0, 0, 1),
                        #     align=TextNode.ALeft, mayChange=1)
                        # textNPose_lp[0] = OnscreenText(
                        #     text="Stability (lp u=0.3) is:" + str(self.stabilityindex[1][counter[0]]), pos=(-.9, -.9, 0),
                        #     scale=0.1,
                        #     fg=(1., 0, 0, 1),
                        #     align=TextNode.ALeft, mayChange=1)
                    # checker = cdchecker.MCMchecker(toggledebug=True)
                    # checker.showMesh(objmnp[0])
                    if self.collisionlist[counter[0]]:
                        objmnp[0].set_rgba((1, 0, 0, 1))
                        # objmnp[0].set_rgba((0.1, 0.1, 0.1, 1))
                    else:
                        if self.stablelist[counter[0]]:
                            objmnp[0].set_rgba((0, 191 / 255, 1, 1))
                            objmnp[0].set_rgba((0.1, 0.1, 0.1, 0.5))
                        else:
                            objmnp[0].set_rgba((0.1, 0.1, 0.1, 1))
                    objmnp[0].attach_to(base)
                    counter[0] += 1
                else:
                    counter[0] = 0

            if base.inputmgr.keymap['w'] is True:
                if counter[0] < len(self.pRotMat):
                    print("It's No.", counter[0], "placement")
                    print("Is it collision?", self.collisionlist[counter[0]])
                    if objmnp[0] is not None:
                        objmnp[0].detach()
                        normalnode[0].detachNode()
                        graspnode[0].detachNode()
                        textNPose[0].detachNode()
                        textNPose[1].detachNode()
                        # picnode[0].detachNode()
                        # picnode[1].detachNode()
                    objmnp[0] = cm.CollisionModel(initor=self.objpath)
                    objmnp[0].set_scale((0.001, 0.001, 0.001))
                    objmnp[0].set_homomat(da.pdmat4_to_npmat4(self.pRotMat[counter[0]]))

                    normalnode[0] = NodePath("normal")
                    graspnode[0] = NodePath("graspnode")
                    textNPose[0] = NodePath("text")
                    textNPose[1] = NodePath("text")
                    graspnode[0].reparentTo(base.render)
                    normalnode[0].reparentTo(base.render)
                    # picnode[0] = NodePath("obj")
                    # picnode[0].setMat(da.npmat4_to_pdmat4(rm.homomat_from_posrot(rot=rm.rotmat_from_axangle([1, 0, 0], np.radians(-45)))))
                    # picnode[0].reparentTo(base.render)
                    # picnode[1] = OnscreenImage(picnode[0], pos=(150, 0, 0), scale=100, parent=testnode[0])
                    # image = self.picnodepath[counter[0]]
                    # picnode[1].setImage(image=image)
                    textNPose[1] = OnscreenText(
                        text="ID" + str([counter[0]]), pos=(-.9, -.6, 0),
                        scale=0.1,
                        fg=(0, 0, 0, 1),
                        align=TextNode.ALeft, mayChange=1)
                    # for j in range(len(self.cfreegrasppose[counter[0]])):
                    #     rtqHE = hndfa.genHand()
                    #     rtqHE.setMat(self.cfreegrasppose[counter[0]][j])
                    #     rtqHE.reparentTo(graspnode[0])
                    # self.showsingleNormal(node = normalnode[0], normal = self.placementNormals[counter[0]],
                    #                       placementfacetpairscenter = self.placementfacetpairscenter[counter[0]],
                    #                       placementVertices = self.placementVertices[counter[0]],
                    #                       placementCoM = self.placementCoM[counter[0]])
                    self.showfeatures(node=normalnode[0], normal=self.pNormals[counter[0]],
                                          placementfacetpairscenter=self.pFacetpairscenter[counter[0]],
                                          placementVertices=self.pVertices[counter[0]],
                                          placementCoM = self.pCoM[counter[0]],
                                          showCoM = True,
                                          showVertices = False,
                                          showNormal = False
                                          )
                    if self.stablelist is not None:
                        print("stable?", self.stablelist[counter[0]])
                        textNPose[0] = OnscreenText(
                            text="Stability (uniform) is:" + str(self.stabilityindex_nolp[counter[0]]),
                            pos=(-.9, -.8, 0),
                            scale=0.1,
                            fg=(1., 0, 0, 1),
                            align=TextNode.ALeft, mayChange=1)
                        # textNPose_lp[0] = OnscreenText(
                        #     text="Stability (lp u=0.3) is:" + str(self.stabilityindex[1][counter[0]]), pos=(-.9, -.9, 0),
                        #     scale=0.1,
                        #     fg=(1., 0, 0, 1),
                        #     align=TextNode.ALeft, mayChange=1)
                    # checker = cdchecker.MCMchecker(toggledebug=True)
                    # checker.showMesh(objmnp[0])
                    if self.collisionlist[counter[0]]:
                        objmnp[0].set_rgba((1, 0, 0, 0.5))
                    else:
                        if self.stablelist is not None:
                            if self.stablelist[counter[0]]:
                                objmnp[0].set_rgba((0, 191 / 255, 1, 0.5))
                                objmnp[0].set_rgba((0.1, 0.1, 0.1, 0.5))
                                # objmnp[0].set_rgba((1, 0, 0, 1))
                            else:
                                objmnp[0].set_rgba((0.1, 0.1, 0.1, 0.5))
                                # objmnp[0].set_rgba((1, 0, 0, 1))
                        else:
                            objmnp[0].set_rgba((0, 191 / 255, 1, 0.5))
                            objmnp[0].set_rgba((0.1, 0.1, 0.1, 0.5))
                            # objmnp[0].set_rgba((1, 0, 0, 1))
                    objmnp[0].attach_to(base)
                    counter[0] += 1
                else:
                    counter[0] = 0
            return task.again

        objmnp = [None]
        normalnode = [None]
        graspnode = [None]
        textNPose = [None, None]  # 1 stability 2 ID
        textNPose_lp = [None]
        testnode = [None]
        picnode = [None, None]
        taskMgr.doMethodLater(0.1, update, "update",
                              extraArgs=[objmnp, counter, testnode, normalnode, graspnode, textNPose, textNPose_lp,
                                         picnode],
                              appendTask=True)
        # taskMgr.add(showCamPos, "showCamPos")
        base.run()
if __name__=='__main__':
    base = wd.World(cam_pos=[0.752962, -0.653211, 0.562782], w=960, h=540, lookat_pos=[0, 0, 0.1])
    # gm.gen_frame(length=0.10, thickness=0.005).attach_to(base)
    # objname = "yuanbox_small
    # objname = "box20"
    # objname = "test_long"
    # objname = "lofted"
    # objname = "angle"
    # objname = "wedge"
    # objname = "bar"
    # objname = "polysolid"
    objname = "tjunction-s-c"
    # objname = "smallvbase"
    # objname = "tjunction-show"
    # objname = "housing-cal"
    # objname = "bracket-box"
    # objname = "longtube"
    # objname = "test_long_small"
    slopename = "tc100.stl"

    # slopeplacement = "shu"
    slopeplacement = "ping"
    this_dir, this_filename = os.path.split(__file__)
    slope_low = slope.Slope(z=-0.002, placement=slopeplacement, size=sinfo.Sloperate[slopename], show = False)
    slopeforcd = slope_low.getSlope()
    slope_high = slope.Slope(z=0, placement=slopeplacement, size=sinfo.Sloperate[slopename], show = False)
    slopeforcd_high = slope_high.getSlope()
    # The slope for rendering is different to the one for collision detection. the one for cd is better to be a little lower.

    # slopeforshowpath = os.path.join(this_dir, "objects", "bracket-box.stl")
    # slopeforshowpath = slopeforshowpath.replace('\\', '/')
    # slopeforshow = cm.CollisionModel(slopeforshowpath)
    # slopeforshow.set_scale((0.001, 0.001, 0.001))
    # slopeforshow.set_rgba((0.1, .1, .1, 0.5))
    # # slopeforshow.set_rpy(np.radians(0),np.radians(-90+54.74),np.radians(-45))
    # # import basis.robot_math as rm
    #
    # slopeforshow.set_rotmat(np.dot(rm.rotmat_from_axangle((0, 1, 0), np.radians(-54.74)),
    #                                rm.rotmat_from_axangle((0, 0, 1), np.radians(-45))))
    #
    # slopeforshow.attach_to(base)


    # this_dir, this_filename = os.path.split(__file__)
    dic = "PlacementData"
    dataaddress = os.path.join(this_dir, dic, objname)
    dataaddress = dataaddress.replace('\\', '/')
    if not os.path.exists(dataaddress):
        os.makedirs(dataaddress)
    else:
        pass
    stablehold = stableholdplanner(objname=objname, slopename=slopename, slopeforcd=slopeforcd,
                                   slopeforcd_high=slopeforcd_high, dataaddress=dataaddress, checkstablity=True)
    collisionjudge = stablehold.collisionlist

    print("there are/is", len(collisionjudge), "placement(s)")
    cfreerotmat = stablehold.getcfreerotmat()
    print("there are/is", len(cfreerotmat), "collision free placement(s)")
    print("there are/is:")
    print(cfreerotmat)
    print(stablehold.checkcollision())
    print("stablehold.checkstablity")

    stablehold.outputverticeswithface(dataaddress)
    stablehold.outputcom(dataaddress)
    stablehold.outputcollisionlist(dataaddress)
    stablehold.outputnormal(dataaddress)
    stablehold.outputfacetcenter(dataaddress)
    stablehold.outputplacmentrotmat(dataaddress)
    stablehold.outputstability(dataaddress)

    stablehold.animation()





