import matplotlib.pyplot as plt
import pickle
import math

objname = ["housing", "planewheel", "tool2", "ttube", "sandpart", "planelowerbody", "planerearstay", "bunnysim"]
type = ["160over", "160noover"]
plt.figure(figsize=(5.7, 2.4*len(objname)))

xlist = []
xlabels = []
for objn in objname:
    for tid in type:
        with open('facets-'+objn+"-"+tid+'.pickle', mode='rb') as f:
            tsall = pickle.load(f)
            print(tsall)
        xlist.append(tsall)
        xlabels.append(objn+tid)
for i, tsall in enumerate(xlist):
    plt.subplot(len(xlist)/2, 1, math.floor(i/2)+1)
    plt.plot(range(len(tsall)), sorted(tsall))
plt.show()