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
