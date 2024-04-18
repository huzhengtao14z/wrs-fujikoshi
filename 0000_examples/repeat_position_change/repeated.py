import pickle

# with open("release/2024-03-24 20_45_07.880432.pkl", "wb") as f:
import numpy as np
import os

def get_center(data):
    a = np.asarray(data[5][:3, 3])
    b = np.asarray(data[2][:3, 3])
    c = np.asarray(data[6][:3, 3])
    d = np.asarray(data[7][:3, 3])
    e = np.asarray([a,b,c,d])
    f = np.average(e, axis=0)
    return f

c_path = os.getcwd()
file_names = os.listdir(os.getcwd()+'/data')


confine = []
release = []
for item in file_names:
    confine_temp = []
    release_temp = []
    path_confine = c_path + '/data/' + item + '/confine'
    path_release = c_path + '/data/' + item + '/release'
    release_names = os.listdir(path_confine)
    confine_names = os.listdir(path_release)
    for name in release_names:
        with open(path_confine + '/' + name, 'rb') as f:
            data = pickle.load(f)
        confine_temp.append(get_center(data))
    confine_temp_np = np.average(np.asarray(confine_temp), axis = 0)
    for name in confine_names:
        with open(path_release + '/' + name, 'rb') as f:
            data = pickle.load(f)
        release_temp.append(get_center(data))
    release_temp_np = np.average(np.asarray(release_temp), axis = 0)

    confine.append(confine_temp_np)
    release.append(release_temp_np)

delta_list = []
for i, item in enumerate(confine):
    print(release[i]-item)
    delta_list.append(release[i]-item)
# print(delta_list)