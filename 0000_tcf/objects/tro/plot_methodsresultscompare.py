import matplotlib.pyplot as plt
import pickle
import math
import numpy as np

objname = ["housing", "planewheel", "tool2", "ttube", "sandpart", "planelowerbody", "planerearstay", "bunnysim"]
type = ["160ray", "160noover", "160over"]
plt.figure(figsize=(5.7, 2.4*len(objname)))

xlist = {}
for objn in objname:
    xlist[objn]=[]
xlabels = {}
for objn in objname:
    xlabels[objn]=[]
for objn in objname:
    for tid in type:
        with open('facets-'+objn+"-"+tid+"-variouscosts"+'.pickle', mode='rb') as f:
            tsall = pickle.load(f)
        xlist[objn].append(tsall)
        xlabels[objn].append(objn+tid)
for objn in objname:
    print(xlist[objn][0])
    print(xlist[objn][1])
    print(xlist[objn][2])

barWidth = 0.2
r1 = np.arange(0,8)
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.figure(figsize=(5.7, 2.4*4))

namelist = ['cost', 'ng', 'ns', 'nc']
for i in range(len(namelist)):
    plt.subplot(4,1,i+1)
    h1 = []
    h2 = []
    h3 = []
    for objn in objname:
        h1.append(xlist[objn][0][i])
        h2.append(xlist[objn][1][i])
        h3.append(xlist[objn][2][i])
    # normalize
    print(h1,h2,h3)
    for objid, objn in enumerate(objname):
        maxv = max([h1[objid], h2[objid], h3[objid]])
        print([h1[objid], h2[objid], h3[objid]], maxv)
        h1[objid] = h1[objid]/maxv
        h2[objid] = h2[objid]/maxv
        h3[objid] = h3[objid]/maxv
    plt.bar(r1, h1, color='#990000', width=barWidth, edgecolor='white')
    plt.bar(r2, h2, color='#009900', width=barWidth, edgecolor='white')
    plt.bar(r3, h3, color='#000099', width=barWidth, edgecolor='white')
    plt.xlabel(namelist[i])
    plt.xticks([r + barWidth for r in range(len(h1))], ['hsg', 'pw', 'tl', 'tb', 'mp', 'plb', 'pt', 'bny'])
plt.show()