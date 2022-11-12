# Sequential-text-classification-using-deep-sequence-modelling

# Troubleshooting
  
  ## LibCublas Issue
  ![nvidia-cublas](https://user-images.githubusercontent.com/90950629/201479338-6f5f39f0-54cb-4c4d-aa40-ba72e493ef25.gif)

![image](https://user-images.githubusercontent.com/90950629/201478338-b652984c-369c-42f2-8b97-d50e6aec4f94.png)


` curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg `

` echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | sudo tee /etc/apt/sources.list.d/nvhpc.list`

` sudo apt-get update -y`

` sudo apt-get install -y nvhpc-22-9`

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
 `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/`
 
  






