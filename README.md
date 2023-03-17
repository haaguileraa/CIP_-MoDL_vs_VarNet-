# CIP_-MoDL_vs_VarNet-
Computational Imaging Project CIP

## References: 

* MoDL source code: https://github.com/hkaggarwal/modl 

* VarNet source code: https://github.com/facebookresearch/fastMRI 

# "Blind" tutorial using HPC

## 1. login

* ``` ssh <user>@tinyx.nhr.fau.de ```

## 2. Write in the terminal

* ``` module load python/3.8-anaconda ```

* ``` conda init bash ```

* ``` source ~/.bashrc ```

* ``` nano ~/.profile ```

* ```
    if [ -n "$BASH_VERSION" ]; then
    #  include .bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
    . "$HOME/.bashrc"
    fi
    fi 

* ``` conda config # create ~/.condarc```


* ``` nano  ~/.condarc```

* ``` 
  pkgs_dirs:
  - ${WORK}/software/privat/conda/pkgs
  envs_dirs:
  - ${WORK}/software/privat/conda/envs
  

* ``` conda create --name myenv ```

* ``` conda activate myenv ```

* ``` pip install tensorflow matplotlib numpy fastmri h5py tqdm```

* **NOTE:** If some of the packages can't be pip instal try using ```conda install <package>```

## 3. USE your HPC account using TinyGPU on https://jupyterhub.rrze.uni-erlangen.de/
## 4. Select the kernel named *Python [conda env:conda-myenv]*


## 5. Link the folder to training data (you have to be in CIP_-MoDL_vs_VarNet/):
* ``` ln -s /home/vault/iwbi/shared/cip22_varnet_modl/brain/train```
* ``` ln -s /home/vault/iwbi/shared/cip22_varnet_modl/brain/test```
* ``` ln -s /home/vault/iwbi/shared/cip22_varnet_modl/brain/val```


## 6. Usage Compat: 

* ``` tf_upgrade_v2 --infile <module_to_convert> --outfile <name_of_the_target_file> ```

* See [Upgrading your code to Tensorflow 2.0](https://blog.tensorflow.org/2019/02/upgrading-your-code-to-tensorflow-2-0.html?m=1 "Tensorflow's Blog ")

## 7. Copying files from HPC to local 

* ``` scp -r <user>@tinyx.nhr.fau.de:/home/hpc/iwbi/<user>/origin_dir home/user/objective_directory ```