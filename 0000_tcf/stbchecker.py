#-*-coding:utf-8-*-
import numpy as np
# import shapely as shaplely
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from descartes import PolygonPatch
from shapely.geometry import MultiPoint
from shapely.geometry import Point
import utils as au
from hu import humath
# from hu import lpsolver as hulp
import os
import copy
import cv2
from Sptpolygoninfo import *

class StableChecher():
    def __init__(self):
        self.comfordebug = []
        self.forcefordebug = []

    def isStable(self, objlist, sptPnt, center):
        self.polygonlist1 = objlist
        self.polygonlist2 = sptPnt

        self.polygon_1 = MultiPoint(self.polygonlist1).convex_hull
        self.polygon_2 = MultiPoint(self.polygonlist2).convex_hull

        self.polygon_3 = self.polygon_1.intersection(self.polygon_2)

        print(self.polygon_3.area)
        p4 = Point(center)
        # polygon_4 = self.polygon_3.intersection(p4)
        if p4.within(self.polygon_3):
            return True
        else:
            return False
        # print(polygon_4.area)

    def isStable_threefacesep(self, objlist, sptPnt, center):
        self.polygonlist1 = objlist
        self.polygonlist2 = sptPnt

        self.polygon_2 = MultiPoint(self.polygonlist2).convex_hull

        polygon_effectivelist = []
        self.singlepolygon_effective = []
        self.singlepolygon = []
        self.singlepolygon_draw = []
        for i, polygon in enumerate(self.polygonlist1):
            polygon_effectivelist2 = []
            # singlepolygon = Polygon(polygon).convex_hull
            singlepolygon = MultiPoint(polygon).convex_hull
            self.singlepolygon.append(singlepolygon)
            self.singlepolygon_draw.append(Polygon(polygon))
            polygon_effective = self.polygon_2.intersection(singlepolygon)
            self.singlepolygon_effective.append(polygon_effective)
            print(i, polygon_effective.area)
            if polygon_effective.area ==0.0:
                print(i,polygon_effective.area)
                #######
                return False
            x_list,y_list=polygon_effective.exterior.xy
            for j in range(len(x_list)):
                polygon_effectivelist2.append([x_list[j],y_list[j]])
            polygon_effectivelist.append(polygon_effectivelist2)
        # print("polygon_effectivelist", polygon_effectivelist)
        polygon_effectivelistnogroup = []
        for things in polygon_effectivelist:
            for thing in things:
                polygon_effectivelistnogroup.append(thing)
        # print("polygon_effectivelistnogroup",polygon_effectivelistnogroup)

        self.polygon_1 = MultiPoint(polygon_effectivelistnogroup).convex_hull
        self.polygon_3 = self.polygon_1.intersection(self.polygon_2)
        p4 = Point(center)
        # polygon_4 = self.polygon_3.intersection(p4)
        if p4.within(self.polygon_3):
            return True
        else:
            return False

        # print(polygon_4.area)

    def stablecheck(self, objlist, sptPnt, center, collisionlist):
        self.polygonlist1 = objlist
        self.polygonlist2 = sptPnt
        self.collisionlist = collisionlist

        # self.polygon_effective = None
        # self.polygon_2 = MultiPoint(self.polygonlist2).convex_hull
        self.polygon_2 = Polygon(self.polygonlist2)
        polygon_effectivelist = []
        self.singlepolygon_effective = []
        self.polygon_effectivelist = []
        self.singlepolygon = []
        # self.singlepolygon_draw = []
        if collisionlist ==True:
            print("it is a collision placement")
            return False
        for i, polygon in enumerate(self.polygonlist1):
            singlefacetpolygon = []
            for faceinpolygon in polygon:
                points = []
                for point in faceinpolygon:
                    points.append(point[:2])
                singlefacetpolygon.append(self.polygon_2.intersection(Polygon(points)))
            onepolygon = singlefacetpolygon[0]
            for j in range(len(singlefacetpolygon)-1):
                onepolygon = singlefacetpolygon[j+1].union(onepolygon)
            if onepolygon.area == 0.0:
                return False
            print("onepolygon.exterior.xy",onepolygon.exterior.xy)
            self.polygon_effectivelist.append(onepolygon)

        self.polygon_effective = self.polygon_effectivelist[0]
        for i in range(len(self.polygon_effectivelist)-1):
            self.polygon_effective = self.polygon_effectivelist[i+1].union(self.polygon_effective)
        # np.array(data2)[:, :2]
        p4 = Point(center[:2])
        # polygon_4 = self.polygon_3.intersection(p4)
        if p4.within(self.polygon_effective):

            # self.stabilityindex()
            return True
        else:
            return False

    def stablecheck_nodistribution(self, objlist, sptPnt, com, collisionlist, normallist, facetcenterlist):
        self.polygonlist1 = objlist
        self.polygonlist2 = sptPnt
        # the polygon of obj is list1, the one of slope is list2

        self.collisionlist = collisionlist
        if collisionlist == True:
            # if in collison, it is regarded as unstable.
            print("it is a collision placement")
            return False, 0, None, None

        self.polygon_2 = Polygon(self.polygonlist2)
        self.singlepolygon_effective = []
        self.polygon_effectivelist = []
        self.intercVerticeslist = []
        self.singlepolygon = []

        self.intersectionvertices = []
        for polygon in self.polygonlist1:
            # add every polygon together
            singlefacetpolygon=Polygon(np.around(np.array(polygon)[0][:,:2], decimals=5))
            # print(np.around(np.array(polygon)[0][:,:2], decimals=2))
            for k in range(len(polygon)):
                try:
                    singlefacetpolygon = singlefacetpolygon.union(
                        Polygon(np.around(np.array(polygon)[k][:, :2], decimals=5)))
                except:
                    print("stop")
            onepolygon = self.polygon_2.intersection(singlefacetpolygon)

            if onepolygon.area == 0.0:
                print("unstable")
                return False, 0, None, None
            # print("onepolygon.exterior.xy",onepolygon.exterior.xy)
            try:
                self.intersectionvertices.append(np.array(onepolygon.exterior.xy).T[:-1])
            except:
                multipolygonsolve=[]
                for i in range(len(onepolygon)):
                    multipolygonsolve.append(np.array(onepolygon[i].exterior.xy).T[:-1])
                self.intersectionvertices.append(multipolygonsolve)
        # print("check here")
        surfaceequationlist=[humath.getsurfaceequation(normallist[i],facetcenterlist[i]) for i in range(3)]

        tempconvexandnormal = []
        for i in range(len(self.intersectionvertices)):
            convex = []
            for j in range(len(self.intersectionvertices[i])):
                try:
                    convex.append(humath.getpointfromonesurface(surfaceequationlist[i],
                                                                self.intersectionvertices[i][j][0],
                                                                self.intersectionvertices[i][j][1]))
                except:
                    patchnum = len(self.intersectionvertices[i])
                    for k in range(patchnum):
                        convex.append(humath.getpointfromonesurface(surfaceequationlist[i],
                                                                    self.intersectionvertices[i][k][j][0],
                                                                    self.intersectionvertices[i][k][j][1]))
                    # print("Aaa")
            tempconvexandnormal.append([np.array(convex), np.array(-normallist[i])])

        # with open("tempconvexandnormal.pickle", "wb") as file:
        #     pickle.dump(tempconvexandnormal, file)
        # with open("com.pickle", "wb") as file:
        #     pickle.dump(center, file)
        stability_index = self.checkStabilityAndClearance(tempconvexandnormal, com)
        if stability_index > 0:
            stable = True
        else:
            stable = False

        return stable, stability_index, tempconvexandnormal, self.intersectionvertices

    def stabilityindex(self, objlist, sptPnt, center, collisionlist, normallist, facetcenterlist, u=0.3,
                       dataoutput=False):
        self.polygonlist1 = objlist
        self.polygonlist2 = sptPnt
        self.collisionlist = collisionlist

        self.polygon_2 = Polygon(self.polygonlist2)
        self.singlepolygon_effective = []
        self.polygon_effectivelist = []
        self.intercVerticeslist = []
        self.singlepolygon = []

        if collisionlist ==True:
            print("it is a collision placement")
            return False , 0
        self.intersectionvertices=[]
        for polygon in self.polygonlist1:
            singlefacetpolygon=Polygon(np.around(np.array(polygon)[0][:,:2], decimals=5))
            # print(np.around(np.array(polygon)[0][:,:2], decimals=5))
            for k in range(len(polygon)):
                singlefacetpolygon=singlefacetpolygon.union(Polygon(np.around(np.array(polygon)[k][:,:2], decimals=5)))
            onepolygon = self.polygon_2.intersection(singlefacetpolygon)

            if onepolygon.area == 0.0:
                print("unstable")
                return False , 0
            # print("onepolygon.exterior.xy",onepolygon.exterior.xy)
            try:
                norepeat=humath.listnorepeat(np.around(np.array(onepolygon.exterior.xy), decimals=4).T[:-1])
                # self.intersectionvertices.append(np.around(np.array(onepolygon.exterior.xy),decimals=4).T[:-1])
                self.intersectionvertices.append(norepeat)
            except:
                multipolygonsolve=[]
                for i in range(len(onepolygon)):
                    norepeat=humath.listnorepeat(np.around(np.array(onepolygon[i].exterior.xy), decimals=4).T[:-1])
                    multipolygonsolve.append(np.around(norepeat))
                    # multipolygonsolve.append(np.around(np.array(onepolygon[i].exterior.xy),decimals=4).T[:-1])
                self.intersectionvertices.append(multipolygonsolve)


        surfaceequationlist=[np.around(humath.getsurfaceequation(normallist[i],facetcenterlist[i]),decimals=4) for i in range(3)]

        tempconvexandnormal = []
        for i in range(len(self.intersectionvertices)):
            convex = []
            for j in range(len(self.intersectionvertices[i])):
                try:
                    convex.append(humath.getpointfromonesurface(surfaceequationlist[i],
                                                                self.intersectionvertices[i][j][0],
                                                                self.intersectionvertices[i][j][1]))
                except:
                    print("Aaa")
            tempconvexandnormal.append([np.array(convex),np.array(-normallist[i])])
        if u!=0:
            self.forcefordebug.append(tempconvexandnormal)
            self.comfordebug.append(center)
        # address = "../manipulation_hu/grip/PlacementData/"
        # # objname = "test_long"
        # objname = "yuanbox"
        # with open(address+"/"+objname+"/"+"tempconvexandnormal.pickle", "wb") as file:
        #     pickle.dump(tempconvexandnormal, file)
        # with open(address+"/"+objname+"/"+"com.pickle", "wb") as file:
        #     pickle.dump(center, file)
        lpsolver=hulp.Lpsolver(G=100,com=center,convexandnormal=tempconvexandnormal,u=u)
        Fdistributed=lpsolver.getresultwithfacet()
        stability_index=self.checkStabilityFdistributed(tempconvexandnormal, center,Fdistributed,u=u)

        return True, stability_index

    def show(self, center, polygon=True, point=False):
        '''
        useless
        :param center:
        :param polygon:
        :param point:
        :return:
        '''
        if polygon:
            fig = plt.figure(figsize=(10, 5), dpi=90)

            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_ylim(-150, 150)
            ax1.set_xlim(-150, 150)

            # ring_patch1 = PolygonPatch(self.polygon_1, alpha=0.5)
            # ax1.add_patch(ring_patch1)
            plt.scatter(center[0], center[1], color="red", alpha=1,zorder=10)
            # plt.scatter(center[53][0], center[53][1], color="red", alpha=1, zorder=10)
            print("len(self.polygon_effectivelist)",len(self.polygon_effectivelist))
            print("self.polygon_effectivelist", self.polygon_effectivelist)
            if len(self.polygon_effectivelist)==1:
                ring_patch11 = PolygonPatch(self.polygon_effectivelist[0], color ="green", alpha=1)
                ax1.add_patch(ring_patch11)
            if len(self.polygon_effectivelist)==2:
                ring_patch12 = PolygonPatch(self.polygon_effectivelist[1], color="navy", alpha=1)
                ax1.add_patch(ring_patch12)
                ring_patch11 = PolygonPatch(self.polygon_effectivelist[0], color="green", alpha=1)
                ax1.add_patch(ring_patch11)
            if len(self.polygon_effectivelist)==3:

                ring_patch13 = PolygonPatch(self.polygon_effectivelist[2], color="blue", alpha=1)
                ax1.add_patch(ring_patch13)
                ring_patch12 = PolygonPatch(self.polygon_effectivelist[1], color="navy", alpha=1)
                ax1.add_patch(ring_patch12)
                ring_patch11 = PolygonPatch(self.polygon_effectivelist[0], color="green", alpha=1)
                ax1.add_patch(ring_patch11)

                # ring_patch14 = PolygonPatch(self.polygon_effective, color="red", alpha=0.5)
                # ax1.add_patch(ring_patch14)

            # ax2 = fig.add_subplot(1, 2, 1)
            ring_patch2 = PolygonPatch(self.polygon_2, color = "yellow", alpha=0.5)
            ax1.add_patch(ring_patch2)

            # ax2 = fig.add_subplot(1, 2, 1)
            # ring_patch3 = PolygonPatch(self.polygon_3, color="red", alpha=0.5)
            # ax1.add_patch(ring_patch3)
        if point:
            print("Show points")
        else:
            print("Don't show points")
        plt.show()

    def getcontactinfo(self, center, intersectionvertices, sptPnt, sptrate = 1, polygon=True, point=False, ID=0, name="none", thisdir="none",
                       show=True):
        ID = ID
        name = os.path.splitext(name)[0]
        thisdir = thisdir
        if polygon:
            size = 224
            img_gray = np.zeros([size, size, 1], np.uint8)
            img_spt = np.zeros([size, size, 1], np.uint8)
            img_gray[:, :, 0] = np.zeros([size, size]) + 255
            img_spt[:, :, 0] = np.zeros([size, size]) + 255

            # print("len(self.polygon_effectivelist)", len(self.intersectionvertices))
            # print("self.polygon_effectivelist", self.intersectionvertices)
            # draw Fixture
            # ring_patch2 = PolygonPatch(self.polygon_2, color="silver", alpha=0.5)
            # ax1.add_patch(ring_patch2)
            color = "black"
            self.intersection = intersectionvertices
            pointsforcv = copy.deepcopy(self.intersection)
            # sptPnt = np.array([[40.82 / 2, -70.7 / 2], [40.82 / 2, 70.7 / 2], [-81.65 / 2, 0]], dtype=np.int32)
            sptPnt = sptPnt.astype(int)
            sfc_sptcv = copy.deepcopy(sptPnt)

            for i in range(len(sfc_sptcv)):
                sfc_sptcv[i] = sfc_sptcv[i] * np.array([sptrate, -sptrate]) + np.array([size / 2, size / 2])
            cv2.fillPoly(img_gray, np.array([sfc_sptcv]), 240)

            # sprarray = copy.deepcopy(img_spt)
            grey = 200
            if self.intersection is not None:
                for i, surface in enumerate(self.intersection):
                    if type(surface) == list:
                        for k, patch in enumerate(surface):
                            for j, p in enumerate(patch):
                                pointsforcv[i][k][j] = p * np.array([sptrate, -sptrate]) + np.array([size / 2, size / 2])
                            cv2.fillPoly(img_gray, np.array([[pointsforcv[i][k]]], dtype=np.int32), grey)
                    else:
                        for j, p in enumerate(surface):
                            pointsforcv[i][j] = p * np.array([sptrate, -sptrate]) + np.array([size / 2, size / 2])
                        try:
                            cv2.fillPoly(img_gray, np.array([[pointsforcv[i]]], dtype=np.int32), grey)
                        except:
                            print("stop!")

                # if len(self.intersection) == 1:
                #     cv2.fillPoly(img_gray, np.array([[pointsforcv[0]]], dtype=np.int32), grey)
                #
                # if len(self.intersection) == 2:
                #     cv2.fillPoly(img_gray, np.array([[pointsforcv[1]]], dtype=np.int32), grey)
                #
                #     cv2.fillPoly(img_gray, np.array([[pointsforcv[0]]], dtype=np.int32), grey)
                #
                # if len(self.intersection) == 3:
                #     cv2.fillPoly(img_gray, np.array([[pointsforcv[2]]], dtype=np.int32), grey)
                #
                #     cv2.fillPoly(img_gray, np.array([[pointsforcv[1]]], dtype=np.int32), grey)
                #
                #     cv2.fillPoly(img_gray, np.array([[pointsforcv[0]]], dtype=np.int32), grey)

        # img_base = cv2.imread("base-calibration.png", cv2.IMREAD_GRAYSCALE)
        # img_base = img_base[:, :, np.newaxis]
        #
        # img_gray = img_gray * img_base + img_spt
        cv2.circle(img_gray, (int(center[0] * sptrate + (size / 2)), int((-center[1]) * sptrate + (size / 2))), radius=4,
                   color=0 + (center[2] * sptrate),
                   thickness=-1)
        # cv2.circle(img_gray, (0,0), radius=6, color=40,thickness=-1)
        if point:
            print("Show points")
        else:
            print("Don't show points")
        if show:
            plt.show()
        cv2.namedWindow('gray', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.imshow("gray", img_gray)
        outputpath = thisdir + "/picinfo"
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        else:
            pass
        cv2.imwrite(outputpath + "/" + name + "-" + str(ID) + '.png', img_gray, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # print("after")
        # cv2.waitKey(0)
        print("finish one pic")
        cv2.destroyAllWindows()
        return outputpath + "/" + name + "-" + str(ID) + '.png'

    def show_withintersection(self, center, polygon=True, point=False):
        if polygon:
            fig = plt.figure(figsize=(5, 5), dpi=100)
            plt.axis('on')
            plt.tight_layout(pad=0, w_pad=0, h_pad=0)
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_ylim(-50, 50)
            ax1.set_xlim(-50, 50)
            # ring_patch1 = PolygonPatch(self.polygon_1, alpha=0.5)
            # ax1.add_patch(ring_patch1)
            plt.scatter(center[0], center[1], color="red", alpha=1,zorder=10)
            # plt.scatter(center[53][0], center[53][1], color="red", alpha=1, zorder=10)
            print("len(self.polygon_effectivelist)",len(self.intersectionvertices))
            print("self.polygon_effectivelist", self.intersectionvertices)
            ring_patch2 = PolygonPatch(self.polygon_2, color="yellow", alpha=0.5)
            ax1.add_patch(ring_patch2)
            if len(self.intersectionvertices)==1:
                ring_patch11 = PolygonPatch(Polygon(self.intersectionvertices[0]), color ="green", alpha=1)
                ax1.add_patch(ring_patch11)
            if len(self.intersectionvertices)==2:
                ring_patch12 = PolygonPatch(Polygon(self.intersectionvertices[1]), color="navy", alpha=1)
                ax1.add_patch(ring_patch12)
                ring_patch11 = PolygonPatch(Polygon(self.intersectionvertices[0]), color="green", alpha=1)
                ax1.add_patch(ring_patch11)
            if len(self.intersectionvertices)==3:
                ring_patch13 = PolygonPatch(Polygon(self.intersectionvertices[2]), color="blue", alpha=1)
                ax1.add_patch(ring_patch13)
                ring_patch12 = PolygonPatch(Polygon(self.intersectionvertices[1]), color="navy", alpha=1)
                ax1.add_patch(ring_patch12)
                ring_patch11 = PolygonPatch(Polygon(self.intersectionvertices[0]), color="green", alpha=1)
                ax1.add_patch(ring_patch11)
                # ring_patch14 = PolygonPatch(self.polygon_effective, color="red", alpha=0.5)
                # ax1.add_patch(ring_patch14)
            # ax2 = fig.add_subplot(1, 2, 1)
            # ax2 = fig.add_subplot(1, 2, 1)
            # ring_patch3 = PolygonPatch(self.polygon_3, color="red", alpha=0.5)
            # ax1.add_patch(ring_patch3)
        if point:
            print("Show points")
        else:
            print("Don't show points")
        plt.show()

    def show_stablecheck(self, center, polygon=True, point=False):
        '''
        useless
        :param center:
        :param polygon:
        :param point:
        :return:
        '''
        if polygon:
            fig = plt.figure(figsize=(10, 5), dpi=90)

            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_ylim(-150, 150)
            ax1.set_xlim(-150, 150)

            # ring_patch1 = PolygonPatch(self.polygon_1, alpha=0.5)
            # ax1.add_patch(ring_patch1)
            plt.scatter(center[22][0], center[22][1], color="red", alpha=1,zorder=10)
            # plt.scatter(center[53][0], center[53][1], color="red", alpha=1, zorder=10)
            print("lenlenlen",len(self.singlepolygon))
            # singlepolygon = self.singlepolygon
            singlepolygon = self.singlepolygon_draw
            print("self.singlepolygon_draw", singlepolygon)
            if len(self.singlepolygon)==1:
                ring_patch11 = PolygonPatch(singlepolygon[0], color ="green", alpha=1)
                ax1.add_patch(ring_patch11)
            if len(self.singlepolygon)==2:
                ring_patch12 = PolygonPatch(singlepolygon[1], color="navy", alpha=1)
                ax1.add_patch(ring_patch12)
                ring_patch11 = PolygonPatch(singlepolygon[0], color="green", alpha=1)
                ax1.add_patch(ring_patch11)
            if len(self.singlepolygon)==3:
                ring_patch12 = PolygonPatch(singlepolygon[1], color="navy", alpha=1)
                ax1.add_patch(ring_patch12)
                ring_patch11 = PolygonPatch(singlepolygon[0], color="green", alpha=1)
                ax1.add_patch(ring_patch11)
                ring_patch13 = PolygonPatch(singlepolygon[2], color ="blue", alpha=1)
                ax1.add_patch(ring_patch13)

            # ax2 = fig.add_subplot(1, 2, 1)
            ring_patch2 = PolygonPatch(self.polygon_2, color = "yellow", alpha=0.5)
            ax1.add_patch(ring_patch2)

            # ax2 = fig.add_subplot(1, 2, 1)
            # ring_patch3 = PolygonPatch(self.polygon_3, color="red", alpha=0.5)
            # ax1.add_patch(ring_patch3)
        if point:
            # self.polygonlist1
            # self.polygonlist2
            print("Show points")
        else:
            print("Don't show points")
        plt.show()

    def checkStabilityAndClearance(self, convexandnormal, CoM):

        # get stability
        totalfrictioncone = []
        coneposition = []
        convexpoints = [points[0] for points in convexandnormal]
        for idx, points in enumerate(convexpoints):
            normal = convexandnormal[idx][1]
            for point in points:
                totalfrictioncone.append(au.genFrictionCone(position=point, normal=normal, show=False))
                coneposition.append(point)
        com = CoM[:3]
        totalfrictioncone = au.wrenchspace(frictioncone_list=totalfrictioncone,
                                           frictioncone_pos_list=coneposition,
                                           objcom=com,
                                           show=False)
        gravity = [[0, 0, -1, 0, 0, 0]]
        if au.in_hull(np.array([[0, 0, 0, 0, 0, 0]]), np.array(totalfrictioncone + gravity)):
            # stability_score = abs(au.minidistance_hull(np.array([[0, 0, 0, 0, 0, 0]]), np.array(totalfrictioncone + gravity)))[0]
            stability_score = 1
        else:
            stability_score = 0

        return stability_score

    def checkStabilityFdistributed(self, convexandnormal, CoM, Fdistributed,u=0.3):

        # get stability
        totalfrictioncone = []
        coneposition = []
        convexpoints = [points[0] for points in convexandnormal]
        for idx, points in enumerate(convexpoints):
            normal = convexandnormal[idx][1]
            for idx2, point in enumerate(points):
                totalfrictioncone.append(au.genFrictionCone(position=point, u=u, normal=normal, Fdistributed=Fdistributed[idx][idx2], show=False))
                coneposition.append(point)
        com = CoM[:3]
        totalfrictioncone = au.wrenchspace(frictioncone_list=totalfrictioncone,
                                           frictioncone_pos_list=coneposition,
                                           objcom=com,
                                           show=False)
        gravity = [[0, 0, -1, 0, 0, 0]]
        if au.in_hull(np.array([[0, 0, 0, 0, 0, 0]]), np.array(totalfrictioncone + gravity)):
            # stability_score = \
            # abs(au.minidistance_hull(np.array([[0, 0, 0, 0, 0, 0]]), np.array(totalfrictioncone + gravity)))[0]
            stability_score = 1
        else:
            stability_score = 0

        return stability_score

if __name__ == "__main__":
    import pickle
    import pandaplotutils.pandactrl as pandactrl
    from matplotlib import pyplot as plt
    import matplotlib
    import math

    def showstabilityindex(stabilityindex):
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
        ax.set_xticks(np.arange(0, len(stabilityindex), int(math.ceil(len(stabilityindex) / 10))))
        # ax.set_yticks(np.arange(0, 0.02, 0.02/5))
        ax.set_yticks(np.arange(0, 1.2, 1.2 / 5))
        plt.xlabel("placment No.", labelpad=5)
        plt.ylabel("Stability Index", labelpad=5)
        ax.tick_params(pad=3)
        ax.set_xlim(-1, len(stabilityindex))
        # ax.set_ylim(0, 0.02)
        ax.set_ylim(0, 1.2)
        normalstablityindex = []
        maxindex = max(stabilityindex)
        for stabilityindex in stabilityindex:
            normalstablityindex.append(stabilityindex / maxindex)
        # plt.bar(range(len(self.stabilityindex)), sorted(normalstablityindex,reverse=True), label='boy', fc='y')
        plt.bar(range(len(stabilityindex)), normalstablityindex, label='boy', fc='y')

    address = "../manipulation_hu/grip/PlacementData/"
    objname = "test_long"
    # objname = "yuanbox"
    # objname = "lofted"
    # objname = "bar"
    # objname = "angle"
    # objname = "wedge"
    objname = "box20"
    # objname = "polysolid"
    # objname = "test_long_small"
    with open(address+"/"+objname+"/"+"placementVerticeswithface.pickle", "rb") as f:
        data = pickle.load(f)
    with open(address+"/"+objname+"/"+"placementcom.pickle", "rb") as f:
        data2 = pickle.load(f)
    with open(address+"/"+objname+"/"+"collisionlist.pickle", "rb") as f:
        data3 = pickle.load(f)
    with open(address+"/"+objname+"/"+"placementnormal.pickle", "rb") as f:
        data4 = pickle.load(f)
    with open(address+"/"+objname+"/"+"placementfacetcenter.pickle", "rb") as f:
        data5 = pickle.load(f)

    objPnts = data
    checher = StableChecher()
    stablecklist = []
    stabilityindex = []
    stablecklist_lp= []
    stabilityindex_lp = []
    stablecklist_lp_nou = []
    stabilityindex_lp_nou = []
    comlist = data2
    # comlist = np.array(data2)[:,:2]
    collisionlist = data3
    normallist = data4
    facetcenterlist = data5
    # print("comlist", comlist)
    # sptPnt = [np.array([70.7/2, 50]), np.array([0, 0]), np.array([70.7/2, -50]), np.array([-50,0]), np.array([-14.64,-50]), np.array([-14.64,50])]
    # sptPnt = [np.array([50, 70.7]), np.array([0, 0]), np.array([50, -70.7]), np.array([-70.7, 0]),
    #           np.array([-20.7, -70.7]), np.array([-20.7, 70.7])]

    # sptPnt = [np.array([50, 70.7]), np.array([50, -70.7]),
    #           np.array([-20.7, -70.7]), np.array([-70.7, 0]), np.array([-20.7, 70.7])]
    # sptPnt = [np.array([40.82, 70.7]), np.array([40.82, -70.7]),
    #           np.array([-40.82, -70.7]), np.array([-81.65, 0]), np.array([-40.82, 70.7])]
    # sptPnt = [np.array([40.82, -70.7]),
    #           np.array([-81.65, 0]), np.array([40.82, 70.7])]
    # small tray corner
    sptPnt = [np.array([40.82 / 2, -70.7 / 2]),
              np.array([-81.65 / 2, 0]), np.array([40.82 / 2, 70.7 / 2])]
    objlist = []
    for i, placement in enumerate(objPnts):
        print(i)
        stable, stability = checher.stablecheck_2(objlist=placement, sptPnt=sptPnt, center=comlist[i],
                                                   collisionlist=collisionlist[i], normallist=normallist[i],
                                                   facetcenterlist=facetcenterlist[i])

        stable_lp, stability_lp = checher.stabilityindex(objlist=placement, sptPnt=sptPnt, center=comlist[i],
                                        collisionlist=collisionlist[i], normallist=normallist[i],
                                        facetcenterlist=facetcenterlist[i],u=0.3)

        stable_lp_nou, stability_lp_nou = checher.stabilityindex(objlist=placement, sptPnt=sptPnt, center=comlist[i],
                                                         collisionlist=collisionlist[i], normallist=normallist[i],
                                                         facetcenterlist=facetcenterlist[i], u=0)
        # checher.show_withintersection(center=comlist[i], polygon=True, point=False)
        # if i ==5:
        #     result = checher.stabilityindex(objlist=placement, sptPnt=sptPnt, center=comlist[i],collisionlist = collisionlist[i], normallist=normallist[i],facetcenterlist=facetcenterlist[i])
        #     checher.show_withintersection(center = comlist[i], polygon=True, point=False)
        stablecklist.append(stable)
        stabilityindex.append(stability)
        stablecklist_lp.append(stable_lp)
        stabilityindex_lp.append(stability_lp)
        stablecklist_lp_nou.append(stable_lp_nou)
        stabilityindex_lp_nou.append(stability_lp_nou)
        # a = [np.array([0, 0]), np.array([1, 1]), np.array([1, 0]), np.array([0.5,0.3]),np.array([0.5,0.1])]
    print(stablecklist)
    print("================")
    print(stabilityindex_lp_nou)

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
    ax.set_xticks(np.arange(0, len(stabilityindex), int(math.ceil(len(stabilityindex) / 10))))
    # ax.set_yticks(np.arange(0, 0.02, 0.02/5))
    ax.set_yticks(np.arange(0, 1.2, 1.2 / 5))
    plt.xlabel("Placment No.", labelpad=5)
    plt.ylabel("Stability criteria (Unitization)", labelpad=5)
    ax.tick_params(pad=3)
    ax.set_xlim(-1, len(stabilityindex))
    # ax.set_ylim(0, 0.02)
    ax.set_ylim(0, 1.2)


    maxindex = max(stabilityindex)
    maxindex_lp = max(stabilityindex_lp)
    maxindex_lp_nou = max(stabilityindex_lp_nou)
    normalization = False
    if normalization:
        normalstablityindex = []
        normalstablityindex_lp = []
        normalstablityindex_lp_nou = []
        for index in stabilityindex:
            normalstablityindex.append(index / maxindex)
        for index in stabilityindex_lp:
            normalstablityindex_lp.append(index / maxindex_lp)
        for i,index in enumerate(stabilityindex_lp_nou):
            normalstablityindex_lp_nou.append(index / maxindex_lp_nou)
    else:
        normalstablityindex = (np.array(stabilityindex) * 100).tolist()
        normalstablityindex_lp = (np.array(stabilityindex_lp) * 100).tolist()
        normalstablityindex_lp_nou = (np.array(stabilityindex_lp_nou) * 100).tolist()
        ax.set_ylim(0, max([maxindex * 100, maxindex_lp * 100, maxindex_lp_nou * 100]))
        ax.set_yticks(np.arange(0, 1.5 * max([maxindex * 100, maxindex_lp * 100, maxindex_lp_nou * 100]),
                                max([maxindex * 100, maxindex_lp * 100, maxindex_lp_nou * 100]) / 5))
        plt.ylabel("Stability criteria (/100)", labelpad=5)
    # plt.bar(range(len(self.stabilityindex)), sorted(normalstablityindex,reverse=True), label='boy', fc='y')
    dic = "../manipulation_hu/grip/PlacementData"
    name = "stablecklist.pickle"
    # with open(dic + "/" + objname + "/" + name, "wb") as file:
    #     pickle.dump(stablecklist, file)
    # name1 = "normalstablityindex.pickle"
    # with open(dic + "/" + objname + "/" + name1, "wb") as file:
    #     pickle.dump(normalstablityindex, file)
    # name2 = "normalstablityindex_lp.pickle"
    # with open(dic + "/" + objname + "/" + name2, "wb") as file:
    #     pickle.dump(normalstablityindex_lp, file)
    # name3 = "normalstablityindex_lp_nou.pickle"
    # with open(dic + "/" + objname + "/" + name3, "wb") as file:
    #     pickle.dump(normalstablityindex_lp_nou, file)
    # name4 = "forcefordebug.pickle"
    # with open(dic + "/" + objname + "/" + name4, "wb") as file:
    #     pickle.dump(checher.forcefordebug, file)
    # name5 = "comfordebug.pickle"
    # with open(dic + "/" + objname + "/" + name5, "wb") as file:
    #     pickle.dump(checher.comfordebug, file)

    nolp = plt.bar([i - 0.3 for i in range(len(stabilityindex))], normalstablityindex, width=0.3, label='boy',
                   color='tomato')
    lp = plt.bar([i for i in range(len(normalstablityindex_lp))], normalstablityindex_lp, width=0.3, label='boy',
                 color='tan')
    lp_nou = plt.bar([i + 0.3 for i in range(len(normalstablityindex_lp_nou))], normalstablityindex_lp_nou,
                     width=0.3, label='boy', color='teal')
    # plt.legend(handles=[nolp, lp], labels=['Uniform', 'LP u=0.3'], loc='best')
    plt.legend(handles=[nolp, lp, lp_nou], labels=['Uniform', 'LP u=0.3', "LP frictionless"], loc='best')

    plt.show()

    # base.run()