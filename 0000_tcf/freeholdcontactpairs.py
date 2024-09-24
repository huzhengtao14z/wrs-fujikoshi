#!/usr/bin/python

import itertools
import os
import os.path
from direct.gui.OnscreenText import OnscreenText
import visualization.panda.world as wd
import numpy as np
import copy
import modeling.geometric_model as gm

from panda3d.core import *
from shapely.geometry import Point
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from sklearn.neighbors import RadiusNeighborsClassifier
# import environment.collisionmodel as cm
import modeling.collision_model as cm
# import pandaplotutils.pandactrl as pandactrl
import basis.trimesh as trimesh
# from utiltools import robotmath
import pickle
import time
import humath
import hufunc
import random
import basis.data_adapter as da
import basis.robot_math as rm

class FreeholdContactpairs(object):

    def __init__(self, objpath, faceangle=.97, segangle=.97, refine1min=2, refine1max=30,
                 refine2radius=10, verticalthreshold=.995, useoverlap=True):
        """
        :param objpath: path of the object
        :param faceangle: The threhold angle for two triangles to be considered as co-planer
        :param segangle: The threhold angle for two facets to be considered as co-planer
        :param refine1min: The minimum distance between a contact point and the facet boundary
        :param refine1max: The maximum distance between a contact point and the facet boundary
        :param refine2radius: Size of the contact circle
        :param verticalthreshold: The threhold dot product for two facets to be considered as vertical
        :param useoverlap: for comparison, toggle it off if overlaped segments are needed

        author: weiwei, hu
        date: 20190525osaka, 20220408
        """

        self.objcm = cm.CollisionModel(initor=objpath, name=os.path.splitext(os.path.basename(objpath))[0])
        self.objcm.set_scale([0.001, 0.001, 0.001])
        self.objtrimesh = self.objcm.objtrm  #trimesh the STL files
        self.com = self.objtrimesh.center_mass
        # print("check faces","there are",len(self.objtrimesh.faces),self.objtrimesh.faces)
        # print("check facets", "there are",len(self.objtrimesh.facets()),self.objtrimesh.facets())
        self.vertices = self.objtrimesh.vertices
        self.smallfaces = self.objtrimesh.faces
        self.smallfacesarea = self.objtrimesh.area_faces
        self.smallface_normals = self.objtrimesh.face_normals
        self.faces = self.objtrimesh.faces
        self.verticalthreshold = verticalthreshold
        # self.objcm.objpdnp.com
        # generate facets
        tic = time.time()
        if useoverlap:  #(trimesh.facets_over) Compute facets using oversegmentation
            self.facets, self.facetnormals, self.facetcurvatures = self.objtrimesh.facets_over(face_angle=faceangle, seg_angle=segangle)
        else:
            self.facets, self.facetnormals, self.facetcurvatures = self.objtrimesh.facets_noover(faceangle = faceangle)
        toc = time.time()
        print("facet cost", toc -tic)
        #
        # # the sampled points and their normals
        # tic = time.time()
        # self.objsamplepnts = None
        # self.objsamplenrmls = None
        # self.sampleObjModel()
        # toc = time.time()
        # print("sampling cost", toc-tic)
        #
        # # the sampled points (bad samples removed)
        # tic = time.time()
        # # facet2dbdries saves the 2d boundaries of each facet
        # self.removeBadSamples(mindist=refine1min, maxdist=refine1max)
        # toc = time.time()
        # print("remove bad sample cost", toc-tic)
        #
        # # the sampled points (clustered)
        # tic = time.time()
        # self.clusterFacetSamplesRNN(reduceRadius=refine2radius)
        # toc = time.time()
        # print("cluster samples cost", toc-tic)



        self.largefacetsarea = []

        self.smallfacecenter=[]
        self.largefacetscenter = []

        self.largefacet_normals = []
        self.largefacet_vertices = []
        # plan contact pairs
        self.hpairs = []
        self.hpairsnum = None
        self.hpairsnormals = []
        self.hpairscenters = []
        self.hpairsvertices = []
        self.hpairsCoM = []
        self.hpairsfaces = []
        self.hpairsfaces_triple = []

        self.pRotMat = []
        self.pNormals = []
        self.pFacetpairscenter = []
        self.pVertices = []
        self.pCoM = []
        self.pFaces = []

        self.color = (random.random(), random.random(), random.random(), 1)
        tic = time.time()
        # self.planContactpairs(hmax, fpairparallel, objmass, bypasssoftfgr = bypasssoftfgr)
        toc = time.time()
        print("plan contact pairs cost", toc-tic)

        # for plot
        self.counter = 0
        # self.facetcolorarray = pandageom.randomColorArray(self.facets.shape[0], nonrandcolor = [.5,.5,.7,1])

    def getFacetsArea(self):
        '''
        get the area of large facet center in self.largefacetsarea
        :return:
        '''
        # for facet in self.objtrimesh.facets():  # no superimposed
        for facet in self.facets:
            facetarea=0
            for face in facet:
                facetarea += self.smallfacesarea[face]
            self.largefacetsarea.append(facetarea)

    def getFacetsCenter(self):
        '''
        get the coordinate of large facet center in self.largefacetscenter
        get the normal vecter of large facet center in self.largefacet_normals
        get the vertices coordinate of large facet in self.largefacet_vertices
        :return:
        '''
        smallfacectlist = []
        smallfacectlist_area = []
        for i, smallface in enumerate(self.faces):
            self.smallfacecenter.append(humath.centerPoint(np.array([self.vertices[smallface[0]],
                                                                 self.vertices[smallface[1]],
                                                                 self.vertices[smallface[2]]])))

        self.prelargeface = copy.deepcopy(self.facets)

        for facet in self.facets:
            print("hi")
            b = []
            b_area = []
            temlargefaceVerticesid = []
            temlargefaceVertices = []
            for face in facet:
                print(face)
                b.append(self.smallfacecenter[face])
                b_area.append(self.smallfacesarea[face])
                temlargefaceVerticesid.extend(self.smallfaces[face])
                print("temlargefaceVerticesid",temlargefaceVerticesid)
            smallfacectlist.append(b)
            smallfacectlist_area.append(b_area)
            smallfacenomallist = [self.smallface_normals[facet[j]] for j in range(len(facet))]
            self.largefacet_normals.append(np.average(smallfacenomallist, axis=0))
            # self.largefacet_normals.append(self.smallface_normals[facet[0]]) #TODO an average normal
            temlargefaceVerticesid = list(set(temlargefaceVerticesid)) #remove repeating vertices ID
            for id in temlargefaceVerticesid:
                temlargefaceVertices.append(self.vertices[id])
            self.largefacet_vertices.append(temlargefaceVertices)
        for i,largeface in enumerate(smallfacectlist):
            self.largefacetscenter.append(humath.centerPointwithArea(largeface,smallfacectlist_area[i]))

    def planHoldpairs(self):
        '''
        find the holding pairs considering a jig with 3 mutually perpendicular surface
        :param verticaljudge_lft: 
        :param verticaljudge_rgt: 
        
        :return: 
        
        author: weiwei, hu zhengtao
        date: 2020/04/03 osaka university
        '''

        # facetparis for update
        updatedholdingfacetpairs = []

        # find facets combinations (3)
        facets_num = self.facets.shape[0]
        self.temholdingpairs = list(itertools.combinations(range(facets_num), 3))

        # judge if they are  vertical
        judgepair = [] #list for storing Boolean, True for vertical
        for facetpair in self.temholdingpairs:
            judgevertical = []
            temp_vect_1 = np.cross(self.largefacet_normals[facetpair[0]], self.largefacet_normals[facetpair[1]])
            temp_dot =  abs(np.dot(temp_vect_1 , self.largefacet_normals[facetpair[2]]))
            verticaljudge_lft = -np.cos(np.pi * 0.5 * self.verticalthreshold)
            verticaljudge_rgt =  np.cos(np.pi * 0.5 * self.verticalthreshold)
            if temp_dot > 1 + verticaljudge_lft and temp_dot < 1 + verticaljudge_rgt:
                judgevertical.append(True)
            else:
                judgevertical.append(False)
            if all(judgevertical):
                print("This is vertical, the facets ID are", facetpair)
                judgepair.append(True)
                facet0pnts = self.largefacetscenter[facetpair[0]]
                facet1pnts = self.largefacetscenter[facetpair[1]]
                facet2pnts = self.largefacetscenter[facetpair[2]]

                facet0normal = self.largefacet_normals[facetpair[0]]
                facet1normal = self.largefacet_normals[facetpair[1]]
                facet2normal = self.largefacet_normals[facetpair[2]]

                facet0vertices = self.largefacet_vertices[facetpair[0]]
                facet1vertices = self.largefacet_vertices[facetpair[1]]
                facet2vertices = self.largefacet_vertices[facetpair[2]]

                self.hpairscenters.append([facet0pnts, facet1pnts, facet2pnts])
                self.hpairsnormals.append([np.array([facet0normal[0], facet0normal[1], facet0normal[2]]),
                                                   np.array([facet1normal[0], facet1normal[1], facet1normal[2]]),
                                                   np.array([facet2normal[0], facet2normal[1], facet2normal[2]])])
                self.hpairsvertices.append([facet0vertices,facet1vertices,facet2vertices])
                facesgroup_0 = self.facets[facetpair[0]]
                facesgroup_1 = self.facets[facetpair[1]]
                facesgroup_2 = self.facets[facetpair[2]]
                facesVerticesID_0 = [self.faces[facesID] for facesID in facesgroup_0]
                facesVerticesID_1 = [self.faces[facesID] for facesID in facesgroup_1]
                facesVerticesID_2 = [self.faces[facesID] for facesID in facesgroup_2]
                self.hpairsfaces.append([facesVerticesID_0, facesVerticesID_1, facesVerticesID_2])
                updatedholdingfacetpairs.append(facetpair)
            else:
                print("it is not")
                judgepair.append(False)
        # update the facet pairs
        self.hpairs = updatedholdingfacetpairs
        self.hpairsnum = len(self.hpairs)
        # print("Is there any vertical pair? =>",any(judgepair))
        # print(self.hpairsnum, "Available holding facet pairs, they are (facets ID)", self.hpairs)
        # print("faces in facets", self.hpairsfaces)
        # print("Planned facetpair center:", self.hpairscenters)
        # print("Planned facetpair normal", self.hpairsnormals)
        # print("Planned facetpair vertice", self.hpairsvertices)

    def showallNormal(self):

        for i in range(len(self.hpairs)):
            color = (random.random(),random.random(),random.random(),1)
            # length = random.randint(60,100)
            length = 0.05
            self.__showSptPnt(self.holdingpairsSptPnt[i], color=color)
            for j,facetpair in enumerate(self.hpairs[i]):
                gm.gen_arrow(spos=np.array([self.largefacetscenter[facetpair][0],
                                                            self.largefacetscenter[facetpair][1],
                                                            self.largefacetscenter[facetpair][2]]),
                                         epos=np.array([self.largefacetscenter[facetpair][0] + length *
                                               self.largefacet_normals[facetpair][0],
                                               self.largefacetscenter[facetpair][1] + length *
                                               self.largefacet_normals[facetpair][1],
                                               self.largefacetscenter[facetpair][2] + length *
                                               self.largefacet_normals[facetpair][2]]), thickness=0.01, rgba=color).attach_to(base)
    def showsingleNormal(self, num):
        color = (random.random(), random.random(), random.random(), 1)
        # length = random.randint(60,100)
        length = 0.05
        self.__showSptPnt(self.holdingpairsSptPnt[num], color=color)
        for i, facetpair in enumerate(self.hpairs[num]):
            gm.gen_arrow(spos=np.array([self.largefacetscenter[facetpair][0],
                                        self.largefacetscenter[facetpair][1],
                                        self.largefacetscenter[facetpair][2]]),
                         epos=([self.largefacetscenter[facetpair][0] + length *
                                self.largefacet_normals[facetpair][0],
                                self.largefacetscenter[facetpair][1] + length *
                                self.largefacet_normals[facetpair][1],
                                self.largefacetscenter[facetpair][2] + length *
                                self.largefacet_normals[facetpair][2]]), thickness=0.003, rgba=color).attach_to(base)
            for face in self.facets[facetpair]:
                self.drawSingleFaceSurface(base, vertices= self.vertices, faces = self.faces[face], color = color)

    def showallFaces(self):
        vertices = self.objtrimesh.vertices
        for face in self.objtrimesh.faces:
            self.drawSingleFaceSurface(base,vertices,face,color=(random.random(),random.random(),random.random(),1))

    def showbaseframe(self, length=0.10, thickness=0.005):
        gm.gen_frame(length=length, thickness=thickness).attach_to(base)

    def showlargeFacets(self):
        vertices = self.vertices
        for i,facets in enumerate(self.facets):
            color = (random.random(),random.random(),random.random(),0.3)
            for facet in facets:
                self.drawSingleFaceSurface(base, vertices,self.faces[facet],color=color)

    def drawSingleFaceSurface(self, base, vertices, faces, color):
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
        surface = surface.convex_hull
        surface = gm.GeometricModel(surface)
        surface.set_rgba(color)
        surface.attach_to(base)

    def showmultiface(self, base, num):
        length=0.02
        for i,facet in enumerate(self.pFaces[num]):
            color = (random.random(), random.random(), random.random(), 1)
            for j,face in enumerate(facet):
                # color = (random.random(), random.random(), random.random(), 1)
                gm.gen_sphere(face[0][0:3], radius=.005, rgba=color).attach_to(base)
                gm.gen_sphere(face[1][0:3], radius=.005, rgba=color).attach_to(base)
                gm.gen_sphere(face[2][0:3], radius=.005, rgba=color).attach_to(base)

                hufunc.drawanySingleSurface(base, vertices=np.array(face), color=color)

            gm.gen_arrow(spos=np.array([self.pFacetpairscenter[num][i][0],
                                                        self.pFacetpairscenter[num][i][1],
                                                        self.pFacetpairscenter[num][i][2]]),
                                     epos=np.array([self.pFacetpairscenter[num][i][0] + length *
                                           self.pNormals[num][i][0],
                                           self.pFacetpairscenter[num][i][1] + length *
                                           self.pNormals[num][i][1],
                                           self.pFacetpairscenter[num][i][2] + length *
                                           self.pNormals[num][i][2]]), thickness=0.005, rgba=color)

    def getHoldingpairSptPnt(self):
        self.holdingpairsSptPnt = []
        for i in range(len(self.hpairs)):
            surface_B = np.zeros(shape =(3))
            surface_A = []

            pnt_0=self.hpairscenters[i][0]
            pnt_1 = self.hpairscenters[i][1]
            pnt_2 = self.hpairscenters[i][2]

            normal_0 = self.hpairsnormals[i][0]
            normal_1 = self.hpairsnormals[i][1]
            normal_2 = self.hpairsnormals[i][2]

            surface_A.append(normal_0)
            surface_A.append(normal_1)
            surface_A.append(normal_2)
            surface_A = np.array(surface_A)

            surface_B[0] = np.dot(normal_0,pnt_0)
            surface_B[1] = np.dot(normal_1,pnt_1)
            surface_B[2] = np.dot(normal_2,pnt_2)
            SptPnt = np.linalg.solve(surface_A,surface_B)

            self.holdingpairsSptPnt.append(SptPnt)

    def __showSptPnt(self, SptPnt, color):
        gm.gen_sphere(pos=Vec3(SptPnt[0], SptPnt[1], SptPnt[2]), radius=20, rgba=color).attach_to(base)

    def showSptAxis(self, num):
        axisnode=NodePath("normal")
        axisnode.setMat(self.p[int(num*3)])
        gm.gen_frame(length=0.050).attach_to(axisnode)
        axisnode.reparentTo(base.render)

    def getRotMat(self):
        self.rotmat = []
        for i in range(len(self.hpairsnormals)):
            temrotmat = np.zeros(shape = (3,3))
            temrotmat[0] = -self.hpairsnormals[i][0]
            temrotmat[1] = -self.hpairsnormals[i][1]
            temrotmat[2] = -self.hpairsnormals[i][2]
            rgthandedrotmat, faceID = humath.getrighthanded(temrotmat,self.hpairsfaces[i])
            self.rotmat.append(rgthandedrotmat)
            self.hpairsfaces_triple.append(faceID)

    def getplacementRotMat(self):
        self.p=[]
        for i in range(len(self.rotmat)):
            for j in range(len(self.rotmat[i])):
                com = copy.deepcopy(self.com)
                holdingpairsnormals = self.hpairsnormals[i]
                holdingpairscenters = self.hpairscenters[i]
                temholdingpairsvertices = self.hpairsvertices[i]
                # sloperot = None
                a = self.rotmat[i][j]
                b = self.holdingpairsSptPnt[i]
                addvector = np.array([[1],[1],[1]])
                holdingpairscenters = np.concatenate((holdingpairscenters, addvector), axis=1)
                addvector_vertices = np.array([1])
                addvector_com = np.array([1])
                print(com)
                com = np.concatenate((np.array(com), addvector_com), axis=0)
                holdingpairsvertices = []
                for temvertices in temholdingpairsvertices:
                    tempverticeonSFC = []
                    for vertices in temvertices:
                        vertices = np.concatenate((vertices, addvector_vertices), axis=0)
                        tempverticeonSFC.append(vertices)
                    holdingpairsvertices.append(tempverticeonSFC)

                sloperot = rm.homomat_from_posrot(rot = np.dot(rm.rotmat_from_axangle([0, 1, 0], np.radians(-54.74)), rm.rotmat_from_axangle([0, 0, 1], np.radians(-45))))
                # sloperot = np.eye(4)
                # sloperot2 = da.npmat4_to_pdmat4(sloperot1)
                # sloperot3 = da.pdmat4_to_npmat4(sloperot2)
                p = da.npmat4_to_pdmat4(rm.homomat_from_posrot(b, a))
                self.p.append(p)
                p = da.pdmat4_to_npmat4(p)
                p = np.linalg.inv(p)
                p = np.dot(sloperot, p)

                placementNormals0 = np.dot(p[0:3,0:3], holdingpairsnormals[0].T).T
                placementNormals1 = np.dot(p[0:3,0:3], holdingpairsnormals[1].T).T
                placementNormals2 = np.dot(p[0:3,0:3], holdingpairsnormals[2].T).T
                placementfacetpairscenter0 = np.dot(p, holdingpairscenters[0].T).T
                placementfacetpairscenter1 = np.dot(p, holdingpairscenters[1].T).T
                placementfacetpairscenter2 = np.dot(p, holdingpairscenters[2].T).T

                #change CoM
                # x_e=5
                # y_e = 5
                # z_e = 5
                # CoMRotMat = np.array([[1,0,0,x_e],[0,1,0,y_e],[0,0,1,z_e],[0,0,0,1]])
                # com = np.dot(CoMRotMat,com.T)
                #-----------------------
                print(com.T)
                placementCom0 = np.dot(p,com.T).T
                dtemplacementVertices = []
                for vertices in holdingpairsvertices:
                    templacementVertices = []
                    for vertice in vertices:
                        placementfacetpairvertices = np.dot(p, vertice.T).T
                        templacementVertices.append(placementfacetpairvertices)
                    dtemplacementVertices.append(templacementVertices)

                ddtemplacementface = []
                for placement in self.hpairsfaces_triple[i][j]:
                    dtemplacementface = []
                    for face in placement:
                        templacementface = []
                        for faceverticeID in face:
                            faceverticePos = np.concatenate((self.vertices[faceverticeID],np.array([1])),axis = 0)
                            rotatedfaceveritce = np.dot(p, faceverticePos.T).T
                            templacementface.append(rotatedfaceveritce)  #every small face
                        dtemplacementface.append(templacementface) #every facet
                    ddtemplacementface.append(dtemplacementface) #every placement
                p = da.npmat4_to_pdmat4(p)
                self.pRotMat.append(p)
                self.pNormals.append([placementNormals0[0:3],placementNormals1[0:3],placementNormals2[0:3]])
                self.pFacetpairscenter.append([placementfacetpairscenter0[0:3],placementfacetpairscenter1[0:3],placementfacetpairscenter2[0:3]])
                self.pVertices.append(dtemplacementVertices)
                self.pCoM.append(placementCom0)
                self.pFaces.append(ddtemplacementface)
        # print("placementRotMat", self.pRotMat)
        # print("placementRotMat", self.pRotMat)
        # print("placementNormals", self.pNormals)
        # print("placementfacetpairscenter", self.pFacetpairscenter)
        # print("placementVertice", self.pVertices)
        # print("placementCoM", self.pCoM)
        # print("placementfaces", self.pFaces)

    def showplanedfacetvertices(self, num):
        color = (random.random(), random.random(), random.random(), 1)
        for vertice in self.hpairsvertices[num]:
            for i in vertice:
                gm.gen_sphere(pos=(i[0], i[1], i[2]), radius=0.005, rgba=color).attach_to(base)


if __name__=='__main__':

    base = wd.World(cam_pos=[0.600,.600,0], w=960, h=540, lookat_pos=[0, 0, 0.0])
    pos=[600,600,0]
    # textNPose = OnscreenText(
    #     text=str("Camera pos is: (%.3f,%.3f,%.3f)" % (pos[0], pos[1], pos[2])), pos=(-.9, -.9, 0),
    #     scale=0.1,
    #     fg=(1., 0, 0, 1),
    #     align=TextNode.ALeft, mayChange=1)

    this_dir, this_filename = os.path.split(__file__)
    objpath = os.path.join(this_dir, "objects", "box.STL")
    objpath = objpath.replace('\\', '/')  # Windows os needs this replacement
    freehold = FreeholdContactpairs(objpath)
    freehold.getFacetsCenter()
    freehold.planHoldpairs()
    freehold.getFacetsArea()
    freehold.getHoldingpairSptPnt()
    freehold.getRotMat()
    freehold.getplacementRotMat()

    freehold.showbaseframe()
    freehold.showlargeFacets()
    # freehold.showallFaces()
    freehold.showallNormal()
    # freehold.showsingleNormal(num = 1)
    # freehold.showSptAxis(num = 1)
    # freehold.showplanedfacetvertices(num =1)

    # checkobj = cm.CollisionModel(objpath)
    # checkobj.set_scale((0.001,0.001,0.001))
    # checkobj.set_homomat(da.pdmat4_to_npmat4(freehold.pRotMat[3]))
    # checkobj.attach_to(base)

    def update(textNode, task):
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
    taskMgr.add(update, "addobject", extraArgs=[testNode], appendTask=True)
    base.run()

