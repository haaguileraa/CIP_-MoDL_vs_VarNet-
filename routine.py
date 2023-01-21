import os 
import h5py as h5 
from fastmri.data.subsample import EquispacedMaskFractionFunc
# from fastmri.data.transforms import VarNetDataTransform, apply_mask

import fastmri
from tqdm.notebook import tqdm
from fastmri.data import transforms as T

import numpy as np
# org is the original ground truth
# atb is the aliased/noisy image
# csm saves the coil sensitivity maps
# mask is the undersampling mask 



mask_type="equispaced_fraction"  # VarNet uses equispaced mask
center_fractions = [0.08 ]
accelerations = [4] 
# mask = create_mask_for_mask_type(mask_type, center_fractions, accelerations)
# train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
# val_transform = VarNetDataTransform(mask_func=mask)
# test_transform = VarNetDataTransform()
path = "/train/" # /home/vault/iwbi/shared/cip22_varnet_modl/brain/train/ # train/ # val/
# for filename in os.listdir(path):
#     f = os.path.join(path, filename):

f = "/home/hpc/iwbi/iwbi009h/CIP_-MoDL_vs_VarNet-/train/file_brain_AXT1POST_200_6002124.h5"


def apply_mask(data, 
               mask_type="equispaced_fraction",  # VarNet uses equispaced mask
               center_fractions = [0.08 ],
               accelerations = [4] ):
    
    mask_func = EquispacedMaskFractionFunc(center_fractions, accelerations)
    volume_kspace = data['kspace'][()] # swap channels ??
    volume_kspace = np.swapaxes(volume_kspace, 0,1) # uncomment if needed
    atbs = []
    masks = []
    for slice_kspace in volume_kspace:
        slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
        masked_kspace, mask, _ = T.apply_mask(slice_kspace2, mask_func)   # Apply the mask to k-space
        sampled_image = fastmri.ifft2c(masked_kspace)           # Apply Inverse Fourier Transform to get the complex image
        sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
        sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)
        sampled_image_rss = np.abs(sampled_image_rss.numpy())
        atbs.append(sampled_image_rss)
        masks.append(mask.numpy())
    return np.asarray(atbs), np.asarray(masks)


with h5.File(f, 'r') as data:
    atb_data, mask_data = apply_mask(data)
    with h5.File('test.h5', 'w') as target: # creating a new file containing the necesary data to use it with MoDL
        # order: <KeysViewHDF5 ['atb', 'csm', 'mask', 'org']>
        target.create_dataset('atb', data = atb_data)    # is the aliased/noisy image
        target.create_dataset('csm', data = f['reconstruction_rss'])    # saves the coil sensitivity maps
        target.create_dataset('mask', data = mask_data)   # is the undersampling mask 
        target.create_dataset('org', data = f['reconstruction_rss'])    # this is the original ground truth



