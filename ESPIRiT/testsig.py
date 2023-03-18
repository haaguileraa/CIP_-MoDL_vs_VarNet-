from sigpy.mri.app import EspiritCalib as Espirit
import sigpy as sp
import numpy as np
import h5py as h5

def saveCSM(csm: np.ndarray, name: str, mode = 'train'):
    with h5.File(name, 'w') as target: # creating a new file containing the necesary data to use it with MoDL
            # order: <KeysViewHDF5 ['atb', 'csm', 'mask', 'org']>
            if mode == 'train':
                #target.create_dataset('atb', data = atb_data)    # is the aliased/noisy image # It is not necessary for Brain_data
                target.create_dataset('trnCsm', data = csm)    # saves the coil sensitivity maps
            elif mode == 'test':
                #target.create_dataset('atb', data = atb_data)    # is the aliased/noisy image # It is not necessary for Brain_data
                target.create_dataset('tstCsm', data = csm)    # saves the coil sensitivity maps

orgFilename = 'file_brain_AXFLAIR_201_6002902.h5'

filename = 'csm_' + orgFilename


f = h5.File(orgFilename, 'r')
ksp = f['kspace'] # slices, coils, h, w
#ksp = np.transpose(ksp, (0,2,3,1)) 
ksp = np.swapaxes(ksp, 0,1) # -> coils, slices, h, w

#csm = Espirit(ksp, device=sp.Device(0)).run() # CuPy needed
scm = []
for s in range(ksp.shape[1]):
    k1 = ksp[:, s, ...]

    c1 = Espirit(k1, device=sp.Device(0)).run() # CuPy needed

    scm.append(c1)

scm = np.array(scm)

print(scm.shape)
saveCSM(scm, filename)