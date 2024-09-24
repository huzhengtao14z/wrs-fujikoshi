import numpy as np
# import pandaplotutils.pandageom as pg
import basis.robot_math as rm
import basis.trimesh as trimesh
import basis.trimesh.geometry as trigeom
# from utiltools import collisiondetection as cd
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
from panda3d.core import TransparencyAttrib
import itertools, functools
from scipy.spatial import ConvexHull,Delaunay
import basis.data_adapter as da
import random
import copy

collisionTempAssembled = {}
collisionTempMated = {}
def facetboundary(objtrimesh, facet, facetnormal):
    """
    compute a boundary polygon for facet
    assumptions:
    1. there is only one boundary
    2. the facet is convex

    :param objtrimesh: a datatype defined in trimesh
    :param facet: a data type defined in trimesh
    :param facetcenter and facetnormal used to compute the transform, see trimesh.geometry.plane_transform
    :return: [a list of 3d points, a shapely polygon, facetmat4 (the matrix that cvt the facet to 2d 4x4)]

    author: weiwei
    Chen Hao revised
    """
    facetp = None
    fff = False
    # use -facetnormal to let the it face downward

    if np.allclose(np.array(facetnormal),np.array([0,0,1]),atol=1e-4) or np.allclose(np.array(facetnormal),np.array([0,0,-1]),atol=1e-4):
        facetmat4 = np.eye(4)
    else:
        rotatedirection = np.dot(facetnormal,[0,0,1])
        if rotatedirection >1e-6:
            alignaxis = np.array([0,0,1])
        elif rotatedirection<-1e-6:
            alignaxis = np.array([0, 0, -1])
        else:
            alignaxis = np.array([0, 0, 1])
            # if np.array(facetnormal).sum(axis=0) >0:
            #     alignaxis = np.array([0, 0, 1])
            # else:
            #     alignaxis = np.array([0, 0, -1])
            fff = True
        rotationaxis = np.cross(facetnormal, alignaxis)
        rotationangle = rm.angle_between_vectors(facetnormal, alignaxis)
        facetmat4 = np.eye(4)
        # print(facetnormal)
        if fff:
            rotationangle = rotationangle if np.array(facetnormal).sum(axis=0) >0 else - rotationangle
        facetmat4[:3,:3] = rm.rodrigues(rotationaxis, np.rad2deg(rotationangle))
    facetmat4Inv = np.linalg.inv(np.matrix(facetmat4[0:3, 0:3]))

    for faceidx in facet[0]:
        vert0 = objtrimesh.vertices[objtrimesh.faces[faceidx][0]]
        vert1 = objtrimesh.vertices[objtrimesh.faces[faceidx][1]]
        vert2 = objtrimesh.vertices[objtrimesh.faces[faceidx][2]]
        vert0p = rm.homotransform(facetmat4, vert0)
        vert1p = rm.homotransform(facetmat4, vert1)
        vert2p = rm.homotransform(facetmat4, vert2)
        facep = Polygon([vert0p[:2], vert1p[:2], vert2p[:2]])
        if facetp is None:
            facetp = facep
            # facetp = facetp.buffer(0)
        else:
            # facep = facep.buffer(0)
            facetp = facetp.union(facep)
    verts2d = list(facetp.exterior.coords)
    vertsZ = vert2p[2]
    return [vertsZ, verts2d, facetmat4,facetmat4Inv]

def boundarydetect(objtrimesh, facetnormal):
    pointList = facetboundary(objtrimesh = objtrimesh, facet = objtrimesh.facets(), facetnormal = facetnormal)

    return [Polygon(pointList[1]),pointList[0],pointList[3]]

def drawIntersectionArea(polygon1,polygon2, Z, invMat):
    convexVetices= []
    intersectionPart = polygon1.intersection(polygon2)
    if not isinstance(intersectionPart,type(polygon1)):
        intersectionPart = intersectionPart[1]
    # print intersectionPart
    for intersection in list(intersectionPart.exterior.coords):
        convexVetices.append((np.dot(invMat,np.append(np.array(intersection),Z)).tolist())[0])
    return convexVetices,intersectionPart

def drawTheArea(polygon1,polygon2, Z, invMat, showthecontact = True):
    convexHull,intersection = drawIntersectionArea(polygon1=polygon1, polygon2=polygon2,Z=Z,invMat=invMat)

    if showthecontact:

        intersectionarea = trimesh.Trimesh(vertices=convexHull)
        intersectionarea_ch = intersectionarea.convex_hull
        intersectionarea_np = base.pg.trimeshtonp(intersectionarea_ch)
        intersectionarea_np.setColor(0,1,0,1)
        intersectionarea_np.reparentTo(base.render)
    return convexHull

#
def collisiondetect(bulletworld, object1, object2):
    collisionsolid1= cd.genBulletCDMeshMultiNp(object1,basenodepath=base.render,name="object1")
    collisionsolid2= cd.genBulletCDMeshMultiNp(object2,basenodepath=base.render, name="object2")
    bulletworld.attachRigidBody(collisionsolid1)
    bulletworld.attachRigidBody(collisionsolid2)
    result = bulletworld.contactTestPair(collisionsolid1,collisionsolid2)
    return result

# #  Version 2
def collisionFace(bulletworld, faceContainer_assembled = None, faceContainer_mated = None, debugtoggle = False, returnconvex = False, showthecontact = True):           #faceArray1 is collided, faceArray2 is origin
    collisionlist_mated = []
    collisionlist_assembled = []
    convexhull_normal = []

    for i in range(0, len(faceContainer_mated.facenplist)):
        boundaryface_mated,Z1,InvMat1 = boundarydetect(objtrimesh=faceContainer_mated.facemeshlist[i],
                                                   facetnormal=faceContainer_mated.facenormallist[i].copy())
        facenormal_mated = faceContainer_mated.facenormallist[i].copy()
        for j in range(0, len(faceContainer_assembled.facenplist)):
            result = None
            if faceContainer_assembled.facenplist[j].name in collisionTempAssembled:
                if faceContainer_mated.facenplist[i].name in collisionTempAssembled[faceContainer_assembled.facenplist[j].name]:
                    result = collisionTempAssembled[faceContainer_assembled.facenplist[j].name][faceContainer_mated.facenplist[i].name]

            elif faceContainer_mated.facenplist[i].name in collisionTempMated:
                if faceContainer_assembled.facenplist[j].name in collisionTempMated[faceContainer_mated.facenplist[i].name]:
                    result = collisionTempMated[faceContainer_mated.facenplist[i].name][faceContainer_assembled.facenplist[j].name]
            if result is None:
                result = collisiondetect(bulletworld, faceContainer_assembled.facenplist[j], faceContainer_mated.facenplist[i])        #Face collision Detection
                if faceContainer_assembled.facenplist[j].name in collisionTempAssembled:
                    collisionTempAssembled[faceContainer_assembled.facenplist[j].name][faceContainer_mated.facenplist[i].name] = result.getNumContacts()
                else:
                    collisionTempAssembled[faceContainer_assembled.facenplist[j].name] = {faceContainer_mated.facenplist[i].name:result.getNumContacts()}

                if faceContainer_mated.facenplist[i].name in collisionTempMated:
                    collisionTempMated[faceContainer_mated.facenplist[i].name][faceContainer_assembled.facenplist[j].name] = result.getNumContacts()
                else:
                    collisionTempMated[faceContainer_mated.facenplist[i].name] = {
                        faceContainer_assembled.facenplist[j].name: result.getNumContacts()}
                result = result.getNumContacts()

            if result:
                facenormal_assembled = faceContainer_assembled.facenormallist[j]
                if -1.1<np.dot(facenormal_mated,facenormal_assembled)<-0.9:
                    boundaryface_assembled,Z2,InvMat2 = boundarydetect(objtrimesh=faceContainer_assembled.facemeshlist[j],
                                                               facetnormal=faceContainer_assembled.facenormallist[j])
                    if debugtoggle:
                        fig = plt.figure(1, figsize=(5, 5), dpi=90)
                        plt.xlim(-100,100)
                        plt.ylim(-100,100)
                        x, y = Polygon(boundaryface_mated).exterior.xy
                        v, z = Polygon(boundaryface_assembled).exterior.xy
                        # q, b = boundaryface_mated.intersects(boundaryface_assembled).exterior.xy
                        ax = fig.add_subplot(111)
                        ax.plot(y, x, color='#6699cc', alpha=0.7,
                                linewidth=3, solid_capstyle='round', zorder=2)
                        ax.plot(z, v, color='#FF0000', alpha=0.7,
                                linewidth=3, solid_capstyle='round', zorder=2)
                        # ax.plot(q, b, color='#00CC00', alpha=0.7,
                        #         linewidth=3, solid_capstyle='round', zorder=2)
                        ax.set_title('Polygon')
                        plt.show()
                    if (boundaryface_mated.intersection(boundaryface_assembled).area>10 and boundaryface_mated.intersects(boundaryface_assembled)):
                        collisionlist_assembled.append(faceContainer_assembled.facenormallist[j].copy())
                        collisionlist_mated.append(faceContainer_mated.facenormallist[i].copy())
                        convexhull_normal.append([drawTheArea(polygon1=boundaryface_mated,polygon2=boundaryface_assembled,Z=Z2,invMat=InvMat2,showthecontact=showthecontact),facenormal_mated])

#    print faceN2
    if returnconvex:
        return collisionlist_mated, collisionlist_assembled,convexhull_normal
    else:
        return collisionlist_mated, collisionlist_assembled

def removeDuplicate(input):
    # check the duplicate
    input_tmp = []
    for component in input:
        if len(input) != 0:
            appendflag = True
            for i in input_tmp:
                if np.allclose(i, component):
                    appendflag = False
                    break
            if not appendflag:
                continue
        input_tmp.append(component)
    return input_tmp


def drawTheCovexHull(constraints, pos=[150, 150, 150]):
    print("The constraints are:",constraints)
    print("The len is",len(constraints))
    base.pggen.plotAxis(base.render, spos=np.array(pos), length=80, thickness=3)
    convexVetices = [np.array(pos)]
    convexVeticesNormal = []
    for i in range(0, len(constraints)):
        base.pggen.plotArrow(base.render, spos=pos,
                             epos=pos + constraints[i] * 40, thickness=4,
                             rgba=[0,0,0, 0.7])

        convexVetices.append((pos + constraints[i] * 40))
        convexVeticesNormal.append(constraints[i])
    trimeshNew = trimesh.Trimesh(vertices=np.array(convexVetices))
    try:
        convexHull = trimeshNew.convex_hull
        convexHullpd = pg.trimeshtonp(convexHull)
        convexHullpd.setColor(0,0,0,0.2 )
        convexHullpd.setTransparency(TransparencyAttrib.MAlpha)
        convexHullpd.reparentTo(base.render)
    except:
        print("Cannot generate the convex hull")
    bestDirection = np.array([0, 0, 0])
    for z in constraints:
        bestDirection = bestDirection + z
    bestDirection = rm.unit_vector(bestDirection)
    base.pggen.plotArrow(base.render, spos=pos,
                         epos=pos + bestDirection * 40, thickness=4,
                         rgba=[147.0 / 255, 112.0 / 255, 219.0 / 255, 0.7])
    print("The best value is ", bestDirection)

def scoreConvexHullOfSpaceOfContactNormals(constraints, show = False):
    uniqconstraints = removeDuplicate(constraints)
    vertices = np.array(uniqconstraints + [np.array([0, 0, 0])])
    try:
        hull = ConvexHull(vertices, qhull_options="QJ")
        equations = hull.equations
        # print("The surface equation is",equations)
        centerofconvexhull = np.sum(vertices[hull.vertices],axis=0)/len(hull.vertices)
        centerofconvexhull = np.append(centerofconvexhull,1)
        distancetoeachplane = []
        for eq in equations:
            distancetoeachplane.append(abs(np.dot(eq,centerofconvexhull)))
        # print("The distance array is ", distancetoeachplane)
        # print("The total distance is ", sum(distancetoeachplane))
        if show:
            convexhull = trimesh.Trimesh(vertices=vertices)
            convexHull = convexhull.convex_hull
            convexHullpd = pg.trimeshtonp(convexHull)
            convexHullpd.setColor(0,0,0,0.2 )
            convexHullpd.setTransparency(TransparencyAttrib.MAlpha)
            convexHullpd.setScale(100)
            convexHullpd.reparentTo(base.render)
        return sum(distancetoeachplane)
    except Exception as e:
        # print(uniqconstraints)
        if len(uniqconstraints) >1:
            return 0
        else:
            return 4

def specialMax(a,b,n):
    """

    :param a: 1 by 3 np array
    :param b: 1 by 3 np array
    :return:
    """
    crosspd = np.cross(a, b)
    if np.allclose(rm.unit_vector(crosspd),n):
        return a
    else:
        return b

def checkSymbol(a,b,n):
    crosspd = np.cross(a, b)
    if np.allclose(rm.unit_vector(crosspd), n,atol=0.00001) or np.allclose(crosspd, np.array([0,0,0]),atol=0.00001):
        return 1
    else:
        return -1

def scoreStabilityRefToContactNormals(constraints, show = False):
    uniqconstraints = removeDuplicate(constraints)
    print(uniqconstraints)
    if len(uniqconstraints) == 1:       # case 1
        return 10
    elif len(uniqconstraints) == 2:
        if np.allclose(np.rad2deg(rm.angle_between_vectors(uniqconstraints[0], uniqconstraints[1])),180):
            return 10                   # case 2
        else:
            return 10               # case 3
    elif len(uniqconstraints) > 2:
        vertices = np.array(uniqconstraints + [np.array([0, 0, 0])])
        try:
            hull = ConvexHull(vertices)
            numberOfFacets = len(hull.equations)
        except:
            hull = ConvexHull(vertices, qhull_options="QJ")
            numberOfFacets = 1
        if numberOfFacets == 1:
            normalofvector = hull.equations[0,:3]
            sidevector = functools.reduce(lambda a,b:specialMax(a,b,normalofvector), uniqconstraints)
            searchspace = uniqconstraints.copy()
            for indx in range(len(searchspace)):
                if np.isclose(searchspace[indx], sidevector).all():
                    break
            popid = indx
            sidevector = searchspace.pop(popid)
            result = [np.rad2deg(rm.angle_between_vectors(sidevector, vector)*checkSymbol(sidevector,vector,normalofvector))for vector in searchspace]
            # print(result)
            result= np.array(result)
            if len(np.where(result<-1e-6)[0]) > 0 and len(np.where(result>1e-6)[0]):
                return 2                            # case 4
            elif len(np.where((result<1e-6 ) & (result>-1e-6))[0]) > 0:
                return 3                            # case 5
            else:
                return 10                       # case 3
        else:
            if minidistance_hull(np.array([0,0,0]),hull) > 1e-6:
                return 0               # case 9
            else:
                opnormalpair = []
                opnormalpairnum = 0
                for vector1 in uniqconstraints[:-1]:
                    if any(np.array_equal(vector1, x) for x in opnormalpair):
                        continue
                    for vector2 in uniqconstraints:
                        if np.allclose(-vector1,vector2,atol=1e-6):
                            opnormalpair.append(vector1)
                            opnormalpair.append(vector2)
                            opnormalpairnum += 1
                            break
                if 2> opnormalpairnum > 0:
                    return 3           # case 7
                elif opnormalpairnum < 1:
                    return 10           # case 6
                elif opnormalpairnum >1:
                    return 1               # case 8
                # xyspace = []
                # for vector in uniqconstraints:
                #     xyspace = vector[:2]
                #     if not np.allclose([0,0],xyspace):
                #         xyspace.append(xyspace)
                # leftmostvector = functools.reduce(lambda a,b: a if np.cross(np.append(a,0),np.append(b,0))[2] >0 else b, xyspace)
                # searchspace = xyspace.copy()
                # leftmostvector = searchspace.pop(leftmostvector)
                # result = [np.rad2deg(rm.angle_between_vectors(leftmostvector, vector) *
                #                      1 if np.cross(np.append(leftmostvector, 0), np.append(vector, 0))[2] > 0 else -1)
                #                      - 180
                #           for vector in searchspace]
                #
                # if len(np.where(result < 0)) > 0 and len(np.where(result > 0)):
                #     return 1
                # elif len(np.where(result=0)) > 0:
                #     return 3  # case 5
                # else:
                #     return 10  # case 3

def genHexahedralVectors(coefficient,polygon = 6, normal = np.array([0,0,1]), Fdistributed=1):
    '''generate vectors for simulating the friction cone'''
    initalMatrix = np.zeros([polygon,3])
    # initVector =pg.rm.unit_vector(np.array([0,coefficient,1]))
    initVector =  Fdistributed*np.array([0, coefficient, 1])
    # tfMatrix = pg.trigeom.align_vectors(vector_start=np.array([0,0,1]),vector_end=-normal)

    tfMatrix = trigeom.align_vectors(vector_start=np.array([0, 0, 1]), vector_end=normal)
    initalMatrix[0,:] = rm.homomat_transform_points(tfMatrix,initVector)
    angle = 360.0/ polygon
    for i in range(1,polygon):
        rotMat = rm.rotmat_from_axangle([0,0,1],angle*i)
        initalMatrix[i,:] = rm.homomat_transform_points(tfMatrix,np.dot(rotMat,initVector))
    return initalMatrix

def plotFrictionCone(frictionConeVector,position):
    for i in frictionConeVector:
        base.pggen.plotArrow(base.render,spos = position, epos = position + i, length = 15, thickness = 1, rgba=np.array([0,0,1,1]))

def genFrictionCone(position,u=0.25,normal=np.array([0,0,1]), polygon=6,  Fdistributed=1,show = True):
    #Coulomb friction model
    #friction coefficient u
    fc_u = u
    frictionCone = genHexahedralVectors(coefficient=fc_u,polygon=polygon,normal=normal, Fdistributed= Fdistributed)
    if show:

        # base.pggen.plotSphere(base.render, pos=position, radius=5, rgba=[0, 1, 0, 1])
        plotFrictionCone(frictionCone,position)
    return frictionCone

def bi_minkowski_sum(frictioncone_a,frictioncone_b):
    combination = list(itertools.product(frictioncone_a,frictioncone_b))
    resultant = [(np.array(i[0])+np.array(i[1])).tolist() for i in combination]
    print("finish---",len(resultant))
    resultantnp = np.array(resultant)
    result = resultantnp[ConvexHull(resultantnp,qhull_options="QJ").vertices].tolist()
    return result

def wrenchspace(frictioncone_list,frictioncone_pos_list,objcom, show = False):
    wrench = []
    for idx, contact_pos in enumerate(frictioncone_pos_list):
        tmparray = []
        for force in frictioncone_list[idx]:
            tmparray.append(
                force.tolist() + np.cross((np.array(contact_pos) - objcom) / 1000, np.array(force)).tolist())
            color =[random.random() for i in range(3)] + [1]
            if show:
                base.pggen.plotArrow(base.render, spos=contact_pos, epos=np.array(contact_pos) + np.array(tmparray[-1][-3:]),
                                     length=15, thickness=1,
                                     rgba=color)

                base.pggen.plotArrow(base.render, spos=contact_pos,
                                     epos=np.array(contact_pos) + np.array(force),
                                     length=29, thickness=1,
                                     rgba=color)
            # base.pggen.plotSphere(base.render,pos=contact_pos,radius=20,rgba=[1,0,0,1])
        # minkov
        wrench.append(tmparray)
        #tase
        if show: base.pggen.plotArrow(base.render, spos=objcom, epos=objcom + (np.array(contact_pos) - objcom) / 1000,
                             length=10, thickness=1,
                             rgba=np.array([0, 0, 0, 1]))
    #minkov
    wrench = [element for group in wrench for element in group]

    # wrench = [element for group in wrench for element in group]
    result = wrench
    # result = functools.reduce(bi_minkowski_sum, wrench)
    return result

def forcespace(frictioncone_list):
    result = functools.reduce(bi_minkowski_sum, frictioncone_list)
    return result

def torquespace(frictioncone_list,frictioncone_pos_list,objcom):
    torque = []
    for idx, contact_pos in enumerate(frictioncone_pos_list):
        tmparray = []
        for force in frictioncone_list[idx]:
            tmparray.append(np.cross((np.array(contact_pos) - objcom) / 1000, np.array(force)).tolist())
            print(np.dot(np.array(force),np.array(tmparray[-1])))
            # print(np.array(contact_pos) + np.array(tmparray[-1]))
            base.pggen.plotArrow(base.render, spos=contact_pos, epos=np.array(contact_pos) + np.array(tmparray[-1]), length=15, thickness=1,
                                 rgba=np.array([1, 0, 1, 1]))
        base.pggen.plotArrow(base.render, spos=objcom, epos=objcom+(np.array(contact_pos) - objcom) / 1000,
                             length=np.linalg.norm(np.array(contact_pos) - objcom) , thickness=1,
                             rgba=np.array([0, 0, 0, 1]))
        # base.pggen.plotStick(base.render, spos=objcom, epos=np.array(contact_pos),thickness=1,
        #                      rgba=np.array([0, 0, 0, 1]))

        torque.append(tmparray)
    # torque = [element for group in torque for element in group]
    result = functools.reduce(bi_minkowski_sum, torque)
    return result

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull,qhull_options="QJ")
    return hull.find_simplex(p)>=0

def minidistance_hull(p, hull):
    if not isinstance(hull,ConvexHull):
        hull = ConvexHull(hull,qhull_options="QJ")
    return np.max(np.dot(hull.equations[:, :-1], p.T).T + hull.equations[:, -1], axis=-1)

# def in_hull(P,hull):
#     '''
#     Datermine if the list of points P lies inside the hull
#     :return: list
#     List of boolean where true means that the point is inside the convex hull
#     '''
#     if not isinstance(hull, ConvexHull):
#         hull = ConvexHull(hull,qhull_options="QJ")
#     A = hull.equations[:,0:-1]
#     b = np.transpose(np.array([hull.equations[:,-1]]))
#     return np.all((A @ np.transpose(P)) <= np.tile(-b,(1,len(P))),axis=0)

