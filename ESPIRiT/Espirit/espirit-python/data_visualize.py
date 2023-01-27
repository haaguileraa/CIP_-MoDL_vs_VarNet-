import h5py

#a = '/train/file_brain_AXT1POST_200_6002196.h5'
a_h5 = h5py.File('train/file_brain_AXT1POST_200_6002196.h5', 'r')
k = a_h5.keys()
print(a_h5["kspace"])
