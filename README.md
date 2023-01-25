# Sequential-text-classification-using-deep-sequence-modelling

# Troubleshooting
  
  ## LibCublas Issue
  ![nvidia-cublas](https://user-images.githubusercontent.com/90950629/201479338-6f5f39f0-54cb-4c4d-aa40-ba72e493ef25.gif)

![image](https://user-images.githubusercontent.com/90950629/201478338-b652984c-369c-42f2-8b97-d50e6aec4f94.png)


` curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg `

` echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list`

` sudo apt-get update -y`

` sudo apt-get install -y nvhpc-22-9`

  ### For Tensorflow 2.x

` python3 `

` import tensorflow as tf `

` config = tf.compat.v1.ConfigProto() `

` config.gpu_options.allow_growth = True `

` session = tf.compat.v1.Session(config=config) `

  ### For Tensorflow 1.x

` config = tf.ConfigProto() `

` config.gpu_options.allow_growth = True`

` session = tf.Session(config=config....) `

 ## Cuda CUL-INT Secure Boot Issue
 
 ![image](https://user-images.githubusercontent.com/90950629/201479842-f52b3f25-8af6-423f-9944-0f343cffb590.png)
 
 Install the appropriate BIOS version\
 Disable Secure Boot in BIOS settings
 
 ` import tensorflow as tf `
 
 ` tf.config.list_physical_devices("GPU") `
 
 ` [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')] `

 ## Pytorch Cuda Visible device Issue
 
`CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start` 

`sudo rmmod nvidia_uvm`

`sudo modprobe nvidia_uvm`
 
 ## Nvidia-NUMA node connection Issue
  
Non-uniform memory access (NUMA) systems are server platforms with more than one system bus. These platforms can utilize multiple processors on a single motherboard, and all processors can access all the memory on the board. When a processor accesses memory that does not lie within its own node (remote memory), data must be transferred over the NUMA connection at a rate that is slower than it would be when accessing local memory. Thus, memory access times are not uniform and depend on the location (proximity) of the memory and the node from which it is accessed.

  ### Check Numa Node Connection
  
  `cat /sys/bus/pci/devices/0000\:01\:00.0/numa_node`\
  -1\
  -1 means no connection, 0 means connected.
  
  ### Reattach Numa node connection
  ` sudo echo 0 | sudo tee -a /sys/bus/pci/devices/0000\:01\:00.0/numa_node `\
  0
 
 ## Nvidia Cuda Libraries Missing
 
 Install the approprita Cudatoolkit and CudaDNN from [Nvidia Official Site](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04) or using conda as below \
 `conda install -c conda-forge cudnn=8.1.0 cudatoolkit=11.7` 
 
 Export the libraries into your current enviornment\
 `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/`\
 or
 Automate this process everytime so that it is included everytime a conda env is invoked\
`mkdir -p $CONDA_PREFIX/etc/conda/activate.d`
`echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`

 
  






