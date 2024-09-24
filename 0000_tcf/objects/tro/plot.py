import matplotlib.pyplot as plt
import pickle

plt.figure(figsize=(7, 2.4))
# objname = "bunny"
# objns = [1072, 2147, 4294, 6589, 9178, 12356, 15293, 17432]
objname = "housing"
objns = [94, 189, 379, 561, 724]
# objname = "planerearstay"
# objns = [1794, 2948, 4806, 6638, 9182, 12482, 15924]
stn = objns[-1]

xlist = []
for objn in objns:
    with open('diff-'+objname+str(objn)+"-"+str(stn)+'.pickle', mode='rb') as f:
        difflistall = pickle.load(f)
    diffall = [i-15 for difflist in difflistall for i in difflist]
    xlist.append(diffall)
plt.violinplot(xlist,
                   showmeans=False,
                   showmedians=True)
plt.gca().set_xticklabels([str(i) for i in objns])
plt.show()