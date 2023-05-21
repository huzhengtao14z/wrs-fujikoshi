#-*-coding:utf-8-*-
import numpy as np
import math
import itertools
def centerPoftrangle(a,b,c):
    '''
    :param a: first vertice of triangle
    :param b: second vertice of triangle
    :param c: third vertice of triangle
    :return: a array consists of those three vertice and a center point
    '''
    x_middle=(a[0]+b[0]+c[0])/3
    y_middle = (a[1] + b[1] + c[1])/3
    z_middle = (a[2] + b[2] + c[2])/3
    d=np.array([x_middle, y_middle, z_middle])
    surface=np.concatenate((a,b,c,d),axis=0)
    return surface

def centerPoint(points):
    '''
    '''
    vertices = []
    for point in points:
        vertices.append(point)
    verticesnp = np.array(vertices)
    verticesnpsum = np.sum(verticesnp, axis = 0)/len(vertices)
    center=np.array([verticesnpsum[0], verticesnpsum[1], verticesnpsum[2]])
    return center

def centerPointwithArea(points,area):
    '''
    '''
    vertices = []
    for i, point in enumerate(points):
        vertices.append(point * area[i])
    verticesnp = np.array(vertices)
    verticesnpsum = np.sum(verticesnp, axis=0) / sum(area)
    center = np.array([verticesnpsum[0], verticesnpsum[1], verticesnpsum[2]])
    return center

def creatlist(n,m):
    n = n
    m = m
    matrix = [None] * n
    for i in range(len(matrix)):
        matrix[i] = [0] * m
    return matrix

def getrighthanded(coordinate, holdingpair):
    coordinate=coordinate
    faces = []
    x_axis0 = coordinate[0]
    x_tempcheck = np.cross(coordinate[1],coordinate[2])
    x_check = np.dot(x_axis0,x_tempcheck)
    if x_check >0:
        y_axis0 = coordinate[1]
        z_axis0 = coordinate[2]
        # faces.append([holdingpair[0],holdingpair[1],holdingpair[2]])
        faces.append([holdingpair[0], holdingpair[1], holdingpair[2]])
    else:
        y_axis0 = coordinate[2]
        z_axis0 = coordinate[1]
        # faces.append([holdingpair[0], holdingpair[2], holdingpair[1]])
        faces.append([holdingpair[0], holdingpair[1], holdingpair[2]])
    coordinate0 = np.array([x_axis0,y_axis0,z_axis0])

    x_axis1 = coordinate[1]
    x_tempcheck = np.cross(coordinate[0], coordinate[2])
    x_check = np.dot(x_axis1, x_tempcheck)
    if x_check > 0:
        y_axis1 = coordinate[0]
        z_axis1 = coordinate[2]
        # faces.append([holdingpair[1], holdingpair[0], holdingpair[2]])
        faces.append([holdingpair[0], holdingpair[1], holdingpair[2]])
    else:
        y_axis1 = coordinate[2]
        z_axis1 = coordinate[0]
        # faces.append([holdingpair[0], holdingpair[2], holdingpair[1]])
        faces.append([holdingpair[0], holdingpair[1], holdingpair[2]])
    coordinate1 = np.array([x_axis1, y_axis1, z_axis1])

    x_axis2 = coordinate[2]
    x_tempcheck = np.cross(coordinate[0], coordinate[1])
    x_check = np.dot(x_axis2, x_tempcheck)
    if x_check > 0:
        y_axis2 = coordinate[0]
        z_axis2 = coordinate[1]
        # faces.append([holdingpair[2], holdingpair[0], holdingpair[1]])
        faces.append([holdingpair[0], holdingpair[1], holdingpair[2]])
    else:
        y_axis2 = coordinate[1]
        z_axis2 = coordinate[0]
        # faces.append([holdingpair[2], holdingpair[1], holdingpair[0]])
        faces.append([holdingpair[0], holdingpair[1], holdingpair[2]])
    coordinate2 = np.array([x_axis2, y_axis2, z_axis2])
    righthandedco = []
    righthandedco.append(coordinate0.T)
    righthandedco.append(coordinate1.T)
    righthandedco.append(coordinate2.T)
    # print("righthandedco",righthandedco)
    # faces.append([holdingpair[0], holdingpair[1], holdingpair[2]])
    return righthandedco,faces

def listnorepeat(list):
    '''
    remove the repeated items in a list, while keepthe original sequence.
    :param list:
    :return:
    '''
    list = list
    listnorepeat = []
    for i, index in enumerate(list):
        # print("listnorepeat", listnorepeat)
        # print("index", index)
        ad = []
        if i == 0:
            listnorepeat.append(index)
        else:
            for j in range(len(listnorepeat)):
                # if j!=i:
                if (index == listnorepeat[j]).all():
                    ad.append(False)
                else:
                    ad.append(True)

            if all(ad):
                listnorepeat.append(index)

    return listnorepeat
def distance(a,b):
    if len(a) ==2:
        distance = math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2))
        return distance
    if len(a) ==3:
        distance = math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2)+math.pow(a[2]-b[2], 2))
        return distance


def TSP(obj):
    number = len(obj)
    seq_number = np.arange(0,number,1).tolist()
    iter = itertools.permutations(seq_number, number)
    sequence = list(iter)
    temdistance = []
    for i in range(len(sequence)):
        temdist = 0
        for j in range(len(sequence[i])):
            if j <len(sequence[i])-1:
                temdist += distance(obj[sequence[i][j]],obj[sequence[i][j+1]])
            if j == len(sequence[i])-1:
                temdist += distance(obj[sequence[i][j]], obj[sequence[i][0]])
        temdistance.append(temdist)
    mindistance = temdistance.index(min(temdistance))
    minsquence = list(sequence[mindistance])
    outputobj = np.copy(obj)
    for i in range(outputobj.shape[0]):
        outputobj[i] = obj[minsquence[i]]
    # outputobj = outputobj.tolist()
    # print("outputobj",outputobj,type(outputobj))
    # print(minsquence,type(minsquence))
    return outputobj

def rowtocolume(array):
    n = array.shape[0]
    new = np.zeros((n,1))
    for i in range(n):
        new[i][0] = array[i]
    return new

def removeduplication(array, decimals):
    b = np.round(array, decimals=decimals).tolist()[:-1]
    c = []
    for i in b:
        if i not in c:
            c.append(i)
    return c

def getsurfaceequation(normal,pnt):
    '''
    generate a surface equation using a known normal and a point
    :param normal:
    :param pnt:
    :return: a vector shows the surface equation Ax+By+Cz+D=0
    A = surfaceequation[0];B = surfaceequation[1]L;C = surfaceequation[2];D = surfaceequation[3]
    '''
    pnt = pnt
    normal = normal
    D = -np.dot(normal, pnt)

    surfaceequation = [normal[0],normal[1],normal[2],D]
    return surfaceequation

def getsurfacefrom3pnt(p):
    '''
    generate a surface equation using 3 points
    :param p: a list of points
    :return: a vector shows the surface equation Ax+By+Cz+D=0
    A = surfaceequation[0];B = surfaceequation[1]L;C = surfaceequation[2];D = surfaceequation[3]
    '''

    a = ((p[1][1] - p[0][1]) * (p[2][2] - p[0][2]) - (p[1][2] - p[0][2]) * (p[2][1] - p[0][1]))

    b = ((p[1][2] - p[0][2]) * (p[2][0] - p[0][0]) - (p[1][0] - p[0][0]) * (p[2][2] - p[0][2]))

    c = ((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]))

    d = (0 - (a * p[0][0] + b * p[0][1] + c * p[0][2]))

    surfaceequation = [a,b,c,d]
    return surfaceequation

def getpointfromonesurface(surfaceequation,x,y):
    '''
    get coordinate of a point with given X, Y and surface equation.
    :param surfaceequation:
    :param x:
    :param y:
    :return:
    '''
    import sympy
    z = sympy.Symbol('z')
    # x = sympy.Symbol('x')
    # y = sympy.Symbol('y')
    x=x
    y=y
    expr1=x*surfaceequation[0]+y*surfaceequation[1]+z*surfaceequation[2]+surfaceequation[3]
    result=sympy.solve(expr1, z)[0]
    return [x,y,result]

if __name__=="__main__":
    # a = [[0,0,0], [100,100,0], [100,0,0], [0,100,0]]
#     #
#     # centerPoint(a)
#     a = np.array([[1,0,0],[0,0,1],[0,1,0]])
#     a = np.array([[0.0, 3], [.0, 2.0], [2.0, 2.0],[1.0,0.0],[1.0,1.0],[2.0,1.0]])
    a = np.array([[30.   ,      42.42640615], [8.78679693 ,42.42640615], [15.    ,     21.21320307], [-6.21320307 ,21.21320307]])
    # a = np.array([[0, 1], [0, 0], [1, 1], [1, 0]])
    TSP(a)
    # print(d)
    # isitrighthanded(a)