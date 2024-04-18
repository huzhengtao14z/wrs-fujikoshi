import pickle

# with open("release/2024-03-24 20_45_07.880432.pkl", "wb") as f:
import numpy as np
import os

file_names = os.listdir(folder_path)


with open("release/2024-03-24 20_45_07.880432.pkl", 'rb') as f:
    data = pickle.load(f)
with open("release/2024-03-24 20_45_08.581336.pkl", 'rb') as f:
    data1 = pickle.load(f)
with open("release/2024-03-24 20_45_09.248062.pkl", 'rb') as f:
    data2 = pickle.load(f)
with open("release/2024-03-24 20_45_09.982294.pkl", 'rb') as f:
    data3 = pickle.load(f)
with open("release/2024-03-24 20_45_10.781327.pkl", 'rb') as f:
    data4 = pickle.load(f)
with open("release/2024-03-24 20_45_11.581975.pkl", 'rb') as f:
    data5 = pickle.load(f)
with open("release/2024-03-24 20_45_12.448524.pkl", 'rb') as f:
    data6 = pickle.load(f)
with open("release/2024-03-24 20_45_13.181648.pkl", 'rb') as f:
    data7 = pickle.load(f)
with open("release/2024-03-24 20_45_13.882163.pkl", 'rb') as f:
    data8 = pickle.load(f)
with open("release/2024-03-24 20_45_14.583171.pkl", 'rb') as f:
    data9 = pickle.load(f)

def get_center(data):
    a = np.asarray(data[5][:3, 3])
    b = np.asarray(data[2][:3, 3])
    c = np.asarray(data[6][:3, 3])
    d = np.asarray(data[7][:3, 3])
    e = np.asarray([a,b,c,d])
    f = np.average(e, axis=0)
    return f
# print(data)
c0 = get_center(data)
c1 = get_center(data1)
c2 = get_center(data2)
c3 = get_center(data3)
c4 = get_center(data4)
c5 = get_center(data5)
c6 = get_center(data6)
c7 = get_center(data7)
c8 = get_center(data8)
c9 = get_center(data9)

print(c0)
print(c1)
print(c2)
print(c3)
print(c4)
print(c5)
print(c6)
print(c7)
print(c8)
print(c9)



# print(data1)
# print(data2)