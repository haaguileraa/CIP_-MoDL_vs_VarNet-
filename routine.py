import os 
import h5py as h5 


# org is the original ground truth
# atb is the aliased/noisy image
# csm saves the coil sensitivity maps
# mask is the undersampling mask 


path = "/train/" # /home/vault/iwbi/shared/cip22_varnet_modl/brain/train/ # train/ # val/
# for filename in os.listdir(path):
#     f = os.path.join(path, filename):

f = "train/file_brain_AXT1POST_200_6002124.h5"

with h5.File(f, 'r') as data:
    with h5.File('test.h5', 'w') as target: # creating a new file containing the necesary data to use it with MoDL
        # order: <KeysViewHDF5 ['atb', 'csm', 'mask', 'org']>
        target.create_dataset('atb', data = f['reconstruction_rss'])    # is the aliased/noisy image
        target.create_dataset('csm', data = f['reconstruction_rss'])    # saves the coil sensitivity maps
        target.create_dataset('mask', data = f['reconstruction_rss'])   # is the undersampling mask 
        target.create_dataset('org', data = f['reconstruction_rss'])    # this is the original ground truth

